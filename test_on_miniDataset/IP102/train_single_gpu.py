import os
import argparse

import sys

# Replace with the file path under your own project directory
sys.path.insert(0, '/root/shared-nvme/PaperProject/')
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import timm.scheduler
import data.data as data
from multi_train_utils import train_one_epoch, evaluate
from models.SquenceParadigm import pvt_v2_b0, pvt_v2_b1, efficientformer_v2, efficientvit
from models.ClusteringParadigm import coc_tiny, coc_small, coc_medium
from models.ClusteringParadigm import fec_small, fec_base, fec_large
from models.ClusteringParadigm import cluster_mini, cluster_adj, cluster_tiny
from models import CLUENet
from models.ConvParadigm import ResNet18, shufflenetv2_mini, shufflenetv2_tiny, convnextv2
import random
import numpy as np
import torch


# -----------------------------
# 1. Set global random seed
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 2. Worker initialization function
# -----------------------------
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


model_dict = {
    "ClusterFormer": cluster_mini,
    "ClusterFormer_adjust": cluster_adj,
    "ClusterFormer_tiny": cluster_tiny,
    "COC": coc_tiny,
    "COC_small": coc_small,
    "COC_medium": coc_medium,
    "FEC": fec_small,
    "FEC_base": fec_base,
    "FEC_large": fec_large,
    "pvt_v2_b0": pvt_v2_b0,
    "pvt_v2_b1": pvt_v2_b1,
    "CLUE_tiny": CLUENet.CLUE_tiny,
    "CLUA_mini": CLUENet.CLUE_mini,
    "CLUA_small": CLUENet.CLUE_small,
    "CLUA_base": CLUENet.CLUE_base,

    "efficientvit_m2": efficientvit.EfficientViT_M2,
    "efficientvit_m3": efficientvit.EfficientViT_M3,
    "efficientvit_m5": efficientvit.EfficientViT_M5,
    "efficientformer_v2_s0": efficientformer_v2.efficientformerv2_s0,
    "efficientformer_v2_s1": efficientformer_v2.efficientformerv2_s1,
    "efficientformer_v2_s2": efficientformer_v2.efficientformerv2_s2,
    "resnet18": ResNet18,
    "convnextv2_s1": convnextv2.convnextv2_atto,
    "convnextv2_s2": convnextv2.convnextv2_femto,
    "convnextv2_s3": convnextv2.convnextv2_nano,
    "shufflenetv2_mini": shufflenetv2_mini,
    "shufflenetv2_tiny": shufflenetv2_tiny,
}


def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    # Data size is 224*224
    data_root = args.data_path
    batch_size = args.batch_size
    g = torch.Generator()
    g.manual_seed(42)

    trainset = data.MyDataset('./data/train.txt', data_root=data_root, transform=transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         # Color jitter
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
         transforms.ToTensor(),
         # Random erasing
         transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g)

    valset = data.MyDataset('./data/val.txt', data_root=data_root, transform=transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]))
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=8, worker_init_fn=seed_worker, generator=g)

    testset = data.MyDataset('./data/test.txt', data_root=data_root, transform=transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8, worker_init_fn=seed_worker, generator=g)

    # create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)

    # Print model complexity and usage
    get_flops_ptflops_method(model, (3, 224, 224))

    # Load fixed weights, generally used for pure testing
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # Load pre-trained weights if they exist
    # Note: model_name will be args.model, defined by string in the execution file
    model_name = args.model
    model_files = None
    if args.resume:
        folder_path = './'
        files = os.listdir(folder_path)
        # Filter files starting with the model name as prefix
        model_files = [f for f in files if f.startswith(model_name + "_IP102_epoch")]
    # If matching files exist, load the latest model parameters
    if args.resume and model_files:
        print(f"Found the following saved model files:")
        # Find the latest file based on the epoch information in the filename
        model_files.sort(
            key=lambda f: int(f.split('epoch')[-1].split('.')[0]))  # Assuming format 'model_name_epochX.pth'
        latest_model_file = model_files[-1]  # Select the last one
        print(f"Loading the latest model file: {latest_model_file}")
        # Load model parameters
        checkpoint = torch.load(os.path.join(folder_path, latest_model_file))
        # Load model parameters without loading buffer variables
        model.load_state_dict(checkpoint["model_state_dict"])
        # Get the latest epoch
        # Since the file was saved at the end of that epoch, the next one is where we start training
        begin_epoch = int(latest_model_file.split('epoch')[-1].split('.')[0]) + 1
        train_acc_list = checkpoint["train_acc_list"]
        train_loss_list = checkpoint["train_loss_list"]
        test_acc_list = checkpoint["test_acc_list"]
        best_acc = checkpoint['best_acc']
        print(f"Resuming training from epoch {begin_epoch}", flush=True)
    else:
        print(f"No model file found starting with {model_name}, starting fresh training from epoch 1.", flush=True)
        begin_epoch = 1  # Start from epoch 1 if no saved file is found
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        best_acc = 0.0  # Best accuracy model
    best_model_path = f"{args.model}_best_model.pth"

    # Training from scratch or resume
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    if model_files:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if args.warmup > 0:
        scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=args.epochs,
                                                     lr_min=args.lrf,
                                                     warmup_t=args.warmup,
                                                     warmup_lr_init=args.warmup_lr)
    # If resuming, handle scheduler state
    if model_files:
        # Manually adjust learning rate scheduler state
        if args.scheduled:  # If command line allows loading scheduler state
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler.step(begin_epoch - 1)  # Otherwise, fast-forward scheduler to the current epoch
    loss_function = torch.nn.CrossEntropyLoss()  # loss_function defined outside!

    for epoch in range(begin_epoch, args.epochs + 1):  # Epoch starts from 1
        # train
        mean_loss, train_acc = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=trainloader,
                                               device=device, use_amp=True,
                                               epoch=epoch, clip_norm=args.clip_norm, loss_function=loss_function)

        scheduler.step(epoch)  # Update learning rate

        # validate
        acc_top1, acc_top3 = evaluate(model=model,
                                      data_loader=valloader,
                                      device=device)

        print("[epoch {}] accuracy_top1: {}%  accuracy_top3: {}%".format(epoch, round(acc_top1 * 100, 2),
                                                                         round(acc_top3 * 100, 2)))
        tags = ["loss", "accuracy_top1", "learning_rate"]

        if acc_top1 > best_acc:
            best_acc = acc_top1
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
                'best_acc': best_acc,
            }
            torch.save(best_checkpoint, best_model_path)
            print(f"🌟 New best model saved! Epoch {epoch} | Val Top1 = {best_acc * 100:.2f}%", flush=True)

        # Write to Tensorboard
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc_top1, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        train_loss_list.append(mean_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append((acc_top1, acc_top3))  # Tuple pair

        # Periodic saving
        if (epoch % 10 == 0 or epoch == args.epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
                'best_acc': best_acc,
            }
            torch.save(
                checkpoint, "{}_IP102_epoch{}.pth".format(args.model, epoch))
            print(f"Parameters saved successfully for epoch {epoch}", flush=True)

    # -----------------------------
    # Final testing using the best model on validation set
    # -----------------------------
    print(f"\nTraining complete! Loading best model for final testing...", flush=True)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc_top1, test_acc_top3 = evaluate(model=model, data_loader=testloader, device=device)
    print(f"✅ Best Model Val Top1: {checkpoint['best_acc'] * 100:.2f}%")
    print(f"✅ Final Test Results: Top1 = {test_acc_top1 * 100:.2f}% | Top3 = {test_acc_top3 * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--epochs', type=int, default=100)  # Includes warmup stage
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1.5e-2)
    parser.add_argument('--lrf', type=float, default=1e-5),
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--model', type=str, default="CLUA_backbone_noDDPWarning_posCNN")
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--resume', type=bool, default=False)  # Whether to resume from previous state
    # Root directory for dataset
    parser.add_argument('--data-path', type=str, default='/root/shared-nvme/ip102_v1.1/images')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--scheduled', type=bool, default=True)  # Whether to load scheduler state
    args = parser.parse_args()
    args.epochs += args.warmup
    main(args)