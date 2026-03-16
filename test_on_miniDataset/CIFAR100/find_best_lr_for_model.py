import argparse

import sys
#里面替换为自己项目目录下的文件路径
sys.path.insert(0, '/root/autodl-tmp/myProject/')  # 这句话一定要有

import torch
from torchvision import transforms

from torch_lr_finder import LRFinder

from train_on_CIFAR100 import model_dict,transforms_fn_train,transforms_fn
# 抽取子集比例
from torch.utils.data import Subset
import torchvision

import random
def get_subset(dataset, ratio=0.1, seed=42):
    """从数据集中按比例随机抽取子集"""
    random.seed(seed)
    total_len = len(dataset)
    subset_len = int(total_len * ratio)
    indices = random.sample(range(total_len), subset_len)
    return Subset(dataset, indices)

def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    data_root = args.data_path
    # 实例化训练数据集
    train_dataset = torchvision.datasets.CIFAR100(args.data_path, train=True, transform=transforms_fn_train, download=True)

    train_subset = get_subset(train_dataset,ratio=args.ratio)  # 抽取一部分比例即可

    # 实例化验证数据集
    val_dataset = torchvision.datasets.CIFAR100(args.data_path, train=False, transform=transforms_fn, download=True)

    val_subset = get_subset(val_dataset,ratio=args.ratio)

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,)

    val_loader = torch.utils.data.DataLoader(val_subset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True)

    # create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)

    # 打印模型的使用情况
    get_flops_ptflops_method(model,(3,32,32))

    pg = [p for p in model.parameters() if p.requires_grad]
    # 注意关闭正则化项！！
    optimizer = torch.optim.AdamW(pg, lr=args.lrmin, weight_decay=5e-2)  # 从min开始
    criterion = torch.nn.CrossEntropyLoss()
    step_mode = "linear" if args.linear_test else "exp"

    # num_iters---其实就是epoch
    if not args.amp:
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    else:
        amp_config = {
            'device_type': 'cuda',
            'dtype': torch.float16,
        }
        grad_scaler = torch.cuda.amp.GradScaler()
        lr_finder = LRFinder(
            model, optimizer, criterion, device='cuda',
            amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
        )
    # 不要加入验证集
    lr_finder.range_test(train_loader, end_lr=args.lrmax, num_iter=args.num_iter, step_mode=step_mode)
    lr_finder.plot(log_lr=False)
    # lr_finder.plot(log_lr=True)
    lr_finder.reset()
    print(lr_finder.history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """

    """
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_iter', type=int, default=400)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=128) # 测试会提高学习率使用小型的batch
    # [在线性模式下不要轻易提高最大学习率，它会导致推荐的整体学习率上移动]
    # 正确做法：先在指数空间搜索大致范围，然后在这个范围内使用线性，线性学习率需要保证最低学习率是可以使用的
    # 在4~5个量级以内都可以使用线性学习率，但是超过5个量级以外建议使用指数确定范围（可以连续测试两次确定一下）
    parser.add_argument('--lrmax', type=float, default=1e-2)
    parser.add_argument('--lrmin', type=float, default=1e-4) # 测试的最高与最低学习率
    parser.add_argument('--linear_test', type=bool, default=False)  # 线性学习率测试or指数学习率测试  建议先大范围然后再小范围
    parser.add_argument('--model',type=str,default="mobilenet_v2_cifar")
    parser.add_argument('--amp', type=bool, default=False)  # 混合精度可以减少显存占用
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=r'C:\Users\13779\Desktop\dataset\CIFAR100')
    args = parser.parse_args()
    main(args)

