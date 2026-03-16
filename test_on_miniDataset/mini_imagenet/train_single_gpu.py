import os
import argparse
import sys

# Ensure project root is in path
sys.path.insert(0, '/root/shared-nvme/PaperProject/')

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import timm.scheduler
from my_dataset import MyDataSet
from multi_train_utils import train_one_epoch, evaluate
from models.SquenceParadigm import pvt_v2_b0, pvt_v2_b1, efficientformer_v2, efficientvit
from models.ClusteringParadigm import coc_tiny, coc_small, coc_medium
from models.ClusteringParadigm import fec_small, fec_base, fec_large
from models.ClusteringParadigm import cluster_adj, cluster_mini, cluster_tiny
from models import CLUENet
from models.ConvParadigm import ResNet18, shufflenetv2_mini, shufflenetv2_tiny, convnextv2
import torch
import random
import numpy as np


# -----------------------------
# Sets the global random seed for reproducibility.
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    "CLUE_mini": CLUENet.CLUE_mini,
    "CLUE_tiny": CLUENet.CLUE_tiny,
    "CLUE_small": CLUENet.CLUE_small,
    "CLUE_base": CLUENet.CLUE_base,
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
    print('FLOPs: ', flops, 'Params: ', params)


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    # Data transformation for 224x224 input size
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # Color Jittering
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                     transforms.ToTensor(),
                                     # Random Erasing
                                     transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    data_root = args.data_path
    json_path = "./classes_name.json"

    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])

    # Check num_classes consistency
    if args.num_classes != len(train_dataset.labels):
        raise ValueError("Dataset has {} classes, but input num_classes is {}".format(len(train_dataset.labels),
                                                                                      args.num_classes))

    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="shuffled_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=1)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=1)

    # Create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)

    # Print model complexity statistics
    get_flops_ptflops_method(model, (3, 224, 224))

    # Load fixed weights (generally used for inference/testing only)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("Weights file not found: {}".format(args.weights))

    # Resume training logic
    model_files = None
    model_name = args.model

    if args.resume:
        folder_path = './'
        files = os.listdir(folder_path)
        # Filter files starting with the model name prefix
        model_files = [f for f in files if f.startswith(model_name)]

    # If checkpoint files exist, load the latest one
    if model_files:
        print(f"Found existing model checkpoints:")
        # Sort files by epoch number (assumes format 'model_name_epochX.pth')
        model_files.sort(key=lambda f: int(f.split('epoch')[-1].split('.')[0]))
        latest_model_file = model_files[-1]
        print(f"Loading the latest checkpoint: {latest_model_file}")

        checkpoint = torch.load(os.path.join(folder_path, latest_model_file))
        model.load_state_dict(checkpoint["model_state_dict"])

        # Determine starting epoch (checkpoint is saved at the end of an epoch, so start at the next)
        begin_epoch = int(latest_model_file.split('epoch')[-1].split('.')[0]) + 1
        train_acc_list = checkpoint["train_acc_list"]
        train_loss_list = checkpoint["train_loss_list"]
        test_acc_list = checkpoint["test_acc_list"]
        print(f"Resuming training from epoch {begin_epoch}", flush=True)
    else:
        print(f"No checkpoint found starting with {model_name}. Starting new training from epoch 1.", flush=True)
        begin_epoch = 1
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=0.05)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=args.epochs,
                                                 lr_min=args.lrf,
                                                 warmup_t=args.warmup,
                                                 warmup_lr_init=args.warmup_lr)

    # Loss function defined outside the loop
    loss_function = torch.nn.CrossEntropyLoss()

    if model_files:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Handle learning rate scheduler state
        if args.scheduled:  # If loading scheduler state is allowed via CLI
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Sync scheduler to the current epoch without loading full state
            scheduler.step(begin_epoch - 1)

    for epoch in range(begin_epoch, args.epochs + 1):
        # Training phase
        mean_loss, train_acc = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch,
                                               clip_norm=args.clip_norm,
                                               loss_function=loss_function)

        scheduler.step(epoch)  # Update learning rate

        # Validation phase
        acc_top1, acc_top3 = evaluate(model=model,
                                      data_loader=val_loader,
                                      device=device)

        print("[epoch {}] accuracy_top1: {}%  accuracy_top3: {}%".format(epoch, round(acc_top1 * 100, 2),
                                                                         round(acc_top3 * 100, 2)))

        # Logging to Tensorboard
        tags = ["loss", "accuracy_top1", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc_top1, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        train_loss_list.append(mean_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append((acc_top1, acc_top3))

        # Save checkpoint periodically or at the final epoch
        if (epoch % 20 == 0 or epoch == args.epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_list': train_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
            }
            torch.save(checkpoint, "{}_mini_epoch{}.pth".format(args.model, epoch))
            print(f"Checkpoint saved for epoch {epoch}", flush=True)

    # Final summary results
    print(f"Final Training Accuracy: {train_acc_list[-1]}", flush=True)
    print(f"Final Test Accuracy: acc_top1: {test_acc_list[-1][0]}, acc_top3: {test_acc_list[-1][1]}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)  # Total epochs including warmup
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1.5e-2)
    parser.add_argument('--lrf', type=float, default=1e-5)
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: 1e-6)')
    # Model name identifier
    parser.add_argument('--model', type=str, default="CLUA_test_tiny")
    # Resume training from last checkpoint
    parser.add_argument('--resume', type=bool, default=False)
    # Root directory for dataset
    parser.add_argument('--data-path', type=str, default='D:\\datasets\\dataset\\MINI_ImageNet\\mini-imagenet')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--scheduled', type=bool, default=True)  # Load scheduler state or recalculate based on epoch
    args = parser.parse_args()

    # Adjust total epochs by adding warmup
    args.epochs += args.warmup
    main(args)