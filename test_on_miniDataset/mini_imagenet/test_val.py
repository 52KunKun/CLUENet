import os
import argparse

import sys
# Replace with the file path under your own project directory
sys.path.insert(0, '/root/shared-nvme/PaperProject/')  # This line is mandatory


import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from multi_train_utils import evaluate_test
import numpy as np
from train_single_gpu import model_dict

def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False)
    print('flops: ', flops, 'params: ', params)



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    # Data size is 224*224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # Color jitter
                                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                                     transforms.ToTensor(),
                                     # Random erasing
                                     transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    json_path = "./classes_name.json"
    # Instantiate training dataset
    train_dataset = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=data_transform["train"])

    # check num_classes
    if args.num_classes != len(train_dataset.labels):
        raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels),
                                                                        args.num_classes))

    # Instantiate validation dataset
    val_dataset = MyDataSet(root_dir=data_root,
                            csv_name="shuffled_val.csv",
                            json_path=json_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=8,
                                             )

    # create model
    model = model_dict[args.model](num_classes=args.num_classes).to(device)


    # Print model complexity and usage
    get_flops_ptflops_method(model,(3,224,224))

    # Load fixed weights, generally used for testing only
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)["model_state_dict"]
            # Dictionary that can be loaded into the model parameters
            print(model.load_state_dict(weights_dict))
        else:
            # Automatic file search logic
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    else:
        # raise RuntimeError("the weight path is not given")
        print("!!! Warning: no model .pth weight path provided !!")

    # validate
    test_acc_list = []
    FPS_list = []
    for epoch in range(1,args.test_nums+1):
        acc_top1,acc_top3,fps = evaluate_test(model=model,
                       data_loader=val_loader,
                       device=device)
        # Without training, the results remain consistent across multiple tests
        print("[result {}] accuracy_top1: {}%  accuracy_top3: {}%".format(epoch, round(acc_top1*100, 2),round(acc_top3*100,2)))
        # Write to Tensorboard
        test_acc_list.append((acc_top1,acc_top3))  # Tuple pair
        if epoch > 2:
            FPS_list.append(fps)
    fps_mean = np.mean(FPS_list)
    fps_std = np.std(FPS_list)
    print(f"Average FPS: {fps_mean:.2f} ± {fps_std:.2f}")

# Pure testing mode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_nums',type=int,default=7) # Number of repeated tests
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128) # Set to 128 (e.g., set to 56 to run three programs sequentially)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument('--model',type=str,default="CLUA_small")
    parser.add_argument('--data-path', type=str, default=r'/root/shared-nvme/dataset/mini-imagenet')
    parser.add_argument('--weights', type=str, default=r'',
                        help='initial weights path')

    args = parser.parse_args()
    main(args)