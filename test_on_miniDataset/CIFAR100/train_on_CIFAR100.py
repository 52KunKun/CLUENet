import timm.scheduler
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import sys
import time # 计算FPS
# 开启混合精度训练
from torch.cuda.amp import autocast
import os
import argparse
#里面替换为自己项目目录下的文件路径
sys.path.insert(0, '/root/shared-nvme/PaperProject/')  # 这句话一定要有

from models.CLUA import CLUA_mini_CIFAR
from models import CLUA
from models.SquenceParadigm import pvt_v2_cifar,vit,gvt_cifar,swin
from models.ClusteringParadigm import coc,fec,cluster2
from models.ConvParadigm import ResNet_cifar,ConvMixer_cifar,ShuffleNet2_cifar,MobileNet2_cifar

model_dict = {
    "CLUA_mini":CLUA_mini_CIFAR,
    "pvt_v2_b0":pvt_v2_cifar.pvt_v2_b0,
    "coc_mini_cifar":coc.coc_mini_cifar,
    "fec_mini_cifar":fec.fec_mini_cifar,
    "cluster":cluster2.cluster_mini_cifar,
    "vit_mini_cifar":vit.ViT_mini_cifar,
    "CPVT_mini_cifar":gvt_cifar.pcpvt_b0,
    "swin_mini_cifar":swin.swin_tiny_cifar,
    "resnet_cifar":ResNet_cifar.ResNet_cifar,
    "convmixer_cifar":ConvMixer_cifar.convmixer_cifar,
    "shufflenet_v2_cifar":ShuffleNet2_cifar.shufflenetv2,
    "mobilenet_v2_cifar":MobileNet2_cifar.mobilenetv2,
}

def get_flops_ptflops_method(model,input_size):
    from ptflops import get_model_complexity_info
    print('==> Building model..',flush=True)
    # 通过修改这个调整使用模型
    model = model.cuda()
    flops, params = get_model_complexity_info(model, input_size,as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params,flush=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# 使用数据增强
transforms_fn_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean,std)
])
transforms_fn = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])




def main(args):
    # ImageNet-->迁移到CIFAR100效果太低，主要由于尺寸的原因我们基于尺寸进行微调
    # 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.
    # 训练集
    train_data = torchvision.datasets.CIFAR100(args.data_path, train=True, transform=transforms_fn_train, download=True)
    # 测试集
    test_data = torchvision.datasets.CIFAR100(args.data_path, train=False, transform=transforms_fn, download=True)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练数据集的长度为{}".format(train_data_size),flush=True)
    print("测试数据集的长度为{}".format(test_data_size),flush=True)
    # 利用dataloader来加载数据集
    train = DataLoader(train_data, batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=8) # 在下面的源码解释中可以看出，如果sampler不为默认的None的时候，不用设置shuffle属性了
    test = DataLoader(test_data, batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=8)


    # 测试窗口形式
    model = model_dict[args.model](num_classes=args.num_classes)


    if args.resume:
        model_name = args.model
        folder_path = './'
        files = os.listdir(folder_path)
        # 筛选出以模型名称为前缀的文件
        model_files = [f for f in files if f.startswith(model_name)]

    # 如果有符合条件的文件，则加载最晚的一个模型参数
    if model_files:
        print(f"找到以下保存的模型文件:",flush=True)
        # 按照文件名中的 epoch 信息，找到最新的文件
        model_files.sort(key=lambda f: int(f.split('epoch')[-1].split('.')[0]))  # 假设文件名格式为 'model_name_epochX.pth'
        latest_model_file = model_files[-1]  # 指定最后一个
        print(f"加载最新的模型文件: {latest_model_file}",flush=True)
        # 加载模型参数
        checkpoint = torch.load(os.path.join(folder_path, latest_model_file))
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        # 获取最新的 epoch
        begin_epoch = int(latest_model_file.split('epoch')[-1].split('.')[0])+1 # 假设保存的文件包含 'epoch'
        train_acc_list = checkpoint["train_acc_list"]
        train_loss_list = checkpoint["train_loss_list"]
        test_acc_list = checkpoint["test_acc_list"]
        test_loss_list = checkpoint["test_loss_list"]
        print(f"恢复到第 {begin_epoch} 轮结束的地方",flush=True)
    else:
        print(f"没有找到以 {model_name} 为开头的模型文件，开始新的训练。",flush=True)
        begin_epoch = 1  # 如果没有找到保存的文件，从 epoch 0 开始训练
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []

    model.to(device)
    # 打印模型参数和计算量(CIFAR100图像尺寸的大小）
    get_flops_ptflops_method(model,input_size=(3,32,32))
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()  # 对于cross_entropy来说，他首先会对input进行log_softmax操作，然后再将log_softmax(input)的结果送入nll_loss；而nll_loss的input就是input。
    # 在多分类问题中，如果使用nn.CrossEntropyLoss()，则预测模型的输出层无需添加softmax层！！！
    # 如果是F.nll_loss，则需要添加softmax层!!!
    loss_fn.to(device)

    # learning_rate = 5e-4 # lr=0.05爆炸

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=5e-2)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer,t_initial=args.epochs,
                                                 lr_min=args.lrf,
                                                 warmup_t=args.warmup,
                                                 warmup_lr_init=args.warmup_lr)
    clip_norm = None

    if model_files:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.scheduled:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler.step(begin_epoch-1)

    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(begin_epoch,args.epochs+1):
        print("第{}轮训练开始:lr={:.6f}".format(epoch,optimizer.param_groups[0]["lr"]),flush=True)
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0, 0, 0, 0

        # 训练步骤开始
        model.train()
        with autocast(enabled=use_amp):
            for batch_idx, (data, target) in enumerate(train):
                data, target = data.to(device), target.to(device)
                  # 要将梯度清零，因为如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加
                # output = model(data)
                output = model(data)
                # loss = loss_fn(output, target)
                loss = loss_fn(output, target)
                scaler.scale(loss).backward()
                if clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # 对所有参数应用梯度裁
                scaler.step(optimizer)  # 更新所有的参数
                scaler.update()
                optimizer.zero_grad()
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)  # 选择最大的（概率）值所在的列数就是他所对应的类别数，
                # train_cor += (predicted == target).sum().item()  # 正确分类个数
                train_cor += (predicted == target).sum().item()  # 正确分类个数
                train_sum += target.size(0)  # train_sum+=predicted.shape[0]

            scheduler.step(epoch)
        # 测试步骤开始（跳过预热步骤后再进行测试）
        # 计算模型的前向传播
        model.eval()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch_idx1, (data, target) in enumerate(test):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_cor += (predicted == target).sum().item()
                test_sum += target.size(0)
            end_time = time.perf_counter()
            total_time = end_time-start_time
            fps = test_sum/total_time
            print("Train loss:{}   Train accuracy:{}%   Test loss:{}   Test accuracy:{}%  FPS:{:.2f}".format(train_loss / batch_idx,
                                                                                                 100 * train_cor / train_sum,
                                                                                                 test_loss / batch_idx1,
                                                                                                 100 * test_cor / test_sum,
                                                                                                         fps),flush=True)
            train_loss_list.append(train_loss / batch_idx)
            train_acc_list.append(100 * train_cor / train_sum)
            test_acc_list.append(100 * test_cor / test_sum)
            test_loss_list.append(test_loss / batch_idx1)

        if epoch%10==0 or epoch==args.epochs:
            # 如果epoch为25一个周期或者epoch是最后一个周期保存一次参数
            checkpoint = {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器的state_dict
                'scheduler_state_dict': scheduler.state_dict(),  # 学习率调度器的state_dic
                'train_loss_list':train_loss_list,
                'train_acc_list':train_acc_list,
                'test_acc_list':test_acc_list,
                'test_loss_list':test_loss_list
            }
            torch.save(
                checkpoint,"{}_CIFAR100_epoch{}.pth".format(args.model,epoch))
            print(f"在{epoch}下保存参数"f"完毕",flush=True)

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # fig = plt.figure()
    # plt.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    # plt.plot(range(len(test_loss_list)), test_loss_list, 'red')
    # plt.legend(['训练损失', '测试损失'], fontsize=14, loc='best')
    # plt.xlabel('训练轮数', fontsize=14)
    # plt.ylabel('损失值', fontsize=14)
    # plt.grid()
    # plt.show()
    # # plt.savefig('CIFAR100_figLOSS_6')
    #
    # fig = plt.figure()
    # plt.plot(range(len(train_acc_list)), train_acc_list, 'blue')
    # plt.plot(range(len(test_acc_list)), test_acc_list, 'red')
    # plt.legend(['训练准确率', '测试准确率'], fontsize=14, loc='best')
    # plt.xlabel('训练轮数', fontsize=14)
    # plt.ylabel('准确率(%)', fontsize=14)
    # plt.grid()
    # plt.show()
    # # plt.savefig('CIFAR100_figAccuracy_6')

    # 只要最后的！
    print(f"最终训练准确率：{train_acc_list[-1]}", flush=True)
    print(f"最终测试准确率：acc_top1: {test_acc_list[-1]}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100) # 包括热身阶
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--batch-size', type=int, default=256) # 为了依次可以运行三个程序我们设置为56
    parser.add_argument('--lr', type=float, default=1.5e-2) # PVTv2 3e-3  CLUA 1.5e-2
    parser.add_argument('--lrf', type=float, default=1e-5), # 1e-5
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--model',type=str,default="CLUA_mini")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--resume',type=bool,default=True) # 是否从上一个训练状态去载入
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default='/root/cifar-100/train')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--clip-norm',type=float,default=None)
    parser.add_argument('--scheduled',type=bool,default=True)  # 是否装载学习器，如果不装在学习率打乱重新以epoch更新为基准

    args = parser.parse_args()
    # 当bs越大学习率同步调整越大(以128为基准）
    args.epochs += args.warmup
    main(args)
