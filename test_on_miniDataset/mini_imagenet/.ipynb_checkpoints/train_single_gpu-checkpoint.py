import os
import argparse

import sys
#里面替换为自己项目目录下的文件路径
sys.path.insert(0, '/root/shared-nvme/PaperProject/')  # 这句话一定要有

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import timm.scheduler
from my_dataset import MyDataSet
# 两种奥
from multi_train_utils import train_one_epoch, evaluate
from models.SquenceParadigm import pvt_v2_b0,efficientformer_v2,efficientvit
from models.ClusteringParadigm import coc_tiny,coc_small
from models.ClusteringParadigm import fec_small,fec_base
from models.ClusteringParadigm import cluster_mini
from models.ClusteringParadigm import cluster2
from models import CLUA_past
from models.ConvParadigm import ResNet18,shufflenetv2_mini,shufflenetv2_tiny,convnextv2

model_dict = {
    "ClusterFormer":cluster_mini,
    "ClusterFormer_test": cluster2.cluster_mini,
    "COC":coc_tiny,
    "COC_small":coc_small,
    "FEC":fec_small,
    "FEC_base":fec_base,
    "pvt_v2_b0":pvt_v2_b0,
    "CLUA_tiny_compare":CLUA.CLUA_tiny,
    "CLUA_mini":CLUA.CLUA_mini,
    "CLUA_small":CLUA.CLUA_small,
    "efficientvit_m2":efficientvit.EfficientViT_M2,
    "efficientvit_m3":efficientvit.EfficientViT_M3,
    "efficientformer_v2_s0":efficientformer_v2.efficientformerv2_s0,
    "efficientformer_v2_s1":efficientformer_v2.efficientformerv2_s1,
    "resnet18":ResNet18,
    "convnextv2_s1":convnextv2.convnextv2_atto,
    "convnextv2_s2":convnextv2.convnextv2_femto,
    "shufflenetv2_mini":shufflenetv2_mini,
    "shufflenetv2_tiny":shufflenetv2_tiny,
}

def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)

# 不加载buffer变量
import torch

def load_state_dict_without_buffers(model, state_dict):
    """
    加载模型参数，同时可选择忽略注册的 buffer（如使用 register_buffer 注册的变量）

    参数:
        model: 你的模型
        path: .pth 文件路径
        ignore_buffers: 如果为 True，将删除 state_dict 中所有 buffer 项

    返回:
        加载成功的模型
    """
    # 删除所有在 state_dict 中，但模型没有注册的 buffer
    # model_buffers = dict(model.named_buffers())
    # clean_dict = {k: v for k, v in state_dict.items() if k not in model_buffers}
    missing, \
    unexpected = model.load_state_dict(state_dict, strict=True)
    # print("Loaded with missing:", missing)
    # print("Loaded with unexpected:", unexpected)
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    # 数据尺寸大小为224*224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # 颜色抖动
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                     transforms.ToTensor(),
                                     # 随机擦除
                                     transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    data_root = args.data_path
    json_path = "./classes_name.csv"
    # 实例化训练数据集
    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])

    # check num_classes
    if args.num_classes != len(train_dataset.labels):
        raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),
                                                                        args.num_classes))

    # 实例化验证数据集
    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="shuffled_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)

    # 打印模型的使用情况
    get_flops_ptflops_method(model,(3,224,224))

    # 加载固定权重，一般用于纯测试
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))


    # 如果存在预训练权重则载入
    # 注意以后的model_name都为args。model我们通过字符串定义它在运行文件中的名字
    if args.resume:
        model_name = args.model
        folder_path = './'
        files = os.listdir(folder_path)
        # 筛选出以模型名称为前缀的文件
        model_files = [f for f in files if f.startswith(model_name)]
    # 如果有符合条件的文件，则加载最晚的一个模型参数
    if model_files:
        print(f"找到以下保存的模型文件:")
        # 按照文件名中的 epoch 信息，找到最新的文件
        model_files.sort(key=lambda f: int(f.split('epoch')[-1].split('.')[0]))  # 假设文件名格式为 'model_name_epochX.pth'
        latest_model_file = model_files[-1]  # 指定最后一个
        print(f"加载最新的模型文件: {latest_model_file}")
        # 加载模型参数
        checkpoint = torch.load(os.path.join(folder_path, latest_model_file))
        # 加载模型参数时不加载buffer变量
        # model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        model.load_state_dict(checkpoint["model_state_dict"])
        # 获取最新的 epoch
        # 由于是该epoch结束保存的文件，因此该epoch已经跑完下一个才是我们要开始训练的epoch
        begin_epoch = int(latest_model_file.split('epoch')[-1].split('.')[0])+1  # 假设保存的文件包含 'epoch'
        train_acc_list = checkpoint["train_acc_list"]
        train_loss_list = checkpoint["train_loss_list"]
        test_acc_list = checkpoint["test_acc_list"]
        print(f"恢复到第 {begin_epoch} 轮开始训练的的地方", flush=True)
    else:
        print(f"没有找到以 {model_name} 为开头的模型文件，开始新的训练。从第一个epoch开始训练。", flush=True)
        begin_epoch = 1  # 如果没有找到保存的文件，从 epoch 0 开始训练
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=args.epochs,
                                                 lr_min=args.lrf,
                                                 warmup_t=args.warmup,
                                                 warmup_lr_init=args.warmup_lr)
    loss_function = torch.nn.CrossEntropyLoss() # loss_function放到外面! 不要放到里面，每个模型用的损失函数不一定完全一致
    # 自己自定义一个损失类
    # loss_function = CLUA_res.CLUAClassificationWithSpatialLoss()

    # 不记录学习调度器的状态，这样就直接改变了学习率
    if model_files:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 主动调整学习率调度器状态
        if args.scheduled: # 如果命令行允许装载学习调度器状态
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 恢复学习率调度器的状态
        else:
            scheduler.step(begin_epoch-1)  # 否则学习率调度器通过epoch回到原始状态

    for epoch in range(begin_epoch,args.epochs+1):  # epoch从1开始
        # train
        mean_loss,train_acc = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,clip_norm=args.clip_norm,loss_function=loss_function)

        scheduler.step(epoch) # 更新学习率

        # validate
        acc_top1,acc_top3 = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy_top1: {}%  accuracy_top3: {}%".format(epoch, round(acc_top1*100, 2),round(acc_top3*100,2)))
        tags = ["loss", "accuracy_top1", "learning_rate"]
        # 写到tb
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc_top1, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        train_loss_list.append(mean_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append((acc_top1,acc_top3))  # 双元组

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        # 需要频繁保存
        if (epoch % 5 == 0 or epoch==args.epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器的state_dict
                'scheduler_state_dict': scheduler.state_dict(),  # 保存学习率调度器的状态
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
            }
            torch.save(
                checkpoint, "{}_mini_epoch{}.pth".format(args.model, epoch))
            print(f"在{epoch}下保存参数完毕", flush=True)
    # 打印最后的结果！！
    print(f"最终训练准确率：{train_acc_list[-1]}", flush=True)
    print(f"最终测试准确率：acc_top1: {test_acc_list[-1][0]}, acc_top3: {test_acc_list[-1][1]}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100) # 包括热身阶
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--batch-size', type=int, default=128) # 为了依次可以运行三个程序我们设置为56
    parser.add_argument('--lr', type=float, default=2e-3) # PVTv2 5e-4
    parser.add_argument('--lrf', type=float, default=1e-5), # 1e-5
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--model',type=str,default="CLUA_small")
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--resume',type=bool,default=True) # 是否从上一个训练状态去载入
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default='/root/mini-imagenet')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--clip-norm',type=float,default=None)
    parser.add_argument('--scheduled',type=bool,default=True)  # 是否装载学习器，如果不装在学习率打乱重新以epoch更新为基准
    args = parser.parse_args()
    # 当bs越大学习率同步调整越大(以128为基准）
    args.epochs += args.warmup
    main(args)
