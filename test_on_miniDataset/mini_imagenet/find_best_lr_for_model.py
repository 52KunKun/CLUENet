import argparse
import sys

# CRITICAL: Replace with your specific project directory path
sys.path.insert(0, '/root/autodl-tmp/myProject/')  # Mandatory for local imports

import torch
from torchvision import transforms
from my_dataset import MyDataSet
from torch_lr_finder import LRFinder

from train_single_gpu import model_dict
# Subset sampling ratio
from torch.utils.data import Subset
import random


def get_subset(dataset, ratio=0.1, seed=42):
    """Randomly samples a subset of the dataset based on the specified ratio."""
    random.seed(seed)
    total_len = len(dataset)
    subset_len = int(total_len * ratio)
    indices = random.sample(range(total_len), subset_len)
    return Subset(dataset, indices)


def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print('FLOPs: ', flops, 'Params: ', params)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    # Input transformation for 224x224 resolution
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # Color Jittering
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                     transforms.ToTensor(),
                                     # Random Erasing
                                     transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    json_path = "./classes_name.json"

    # Instantiate the training dataset
    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])

    train_subset = get_subset(train_dataset, ratio=args.ratio)  # Sample a fraction of the dataset

    # Check num_classes consistency
    if args.num_classes != len(train_dataset.labels):
        raise ValueError("Dataset has {} classes, but input num_classes is {}".format(len(train_dataset.labels),
                                                                                      args.num_classes))

    # Instantiate the validation dataset
    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])

    val_subset = get_subset(val_dataset, ratio=args.ratio)
    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_subset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # Create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)

    # Print model usage and complexity statistics
    get_flops_ptflops_method(model, (3, 224, 224))

    pg = [p for p in model.parameters() if p.requires_grad]

    # WARNING: Ensure weight decay/regularization is turned off or minimized for pure LR testing
    optimizer = torch.optim.AdamW(pg, lr=args.lrmin, weight_decay=5e-2)  # Starting from lrmin
    criterion = torch.nn.CrossEntropyLoss()
    step_mode = "linear" if args.linear_test else "exp"

    # num_iters represents the number of batches/iterations for the test
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

    # Validation set is omitted to maintain the speed of range_test
    lr_finder.range_test(train_loader, end_lr=args.lrmax, num_iter=args.num_iter, step_mode=step_mode)
    lr_finder.plot(log_lr=False)
    # lr_finder.plot(log_lr=True)
    lr_finder.reset()
    print("Learning Rate History: ", lr_finder.history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=1.0)
    # Use a small batch size to increase the sensitivity of the learning rate test
    parser.add_argument('--batch-size', type=int, default=16)

    # [Operational Note]: Do not set lrmax too high in linear mode,
    # as it may cause the recommended learning rate to drift upwards incorrectly.
    # Recommended approach: Perform a broad search in exponential (exp) space first to find the range,
    # then use linear mode within that range. Ensure lrmin is high enough to be usable.
    # Linear mode is suitable for ranges within 4-5 orders of magnitude;
    # use exponential mode for larger ranges.
    parser.add_argument('--lrmax', type=float, default=1e-2)
    parser.add_argument('--lrmin', type=float, default=1e-3)  # Range boundaries for testing

    # linear_test: Toggle between linear and exponential LR step modes
    parser.add_argument('--linear_test', type=bool, default=True)
    parser.add_argument('--model', type=str, default="CLUA3_base")
    # Automatic Mixed Precision (AMP) can reduce VRAM consumption
    parser.add_argument('--amp', type=bool, default=False)

    # Dataset root directory
    parser.add_argument('--data-path', type=str, default=r'C:\Users\13779\Desktop\dataset\MINI_ImageNet\mini-imagenet')
    args = parser.parse_args()
    main(args)