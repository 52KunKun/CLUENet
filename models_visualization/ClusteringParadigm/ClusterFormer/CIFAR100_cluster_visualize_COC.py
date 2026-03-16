# --------------------------------------------------------
# Context Cluster Visualization
# A script to visualize the clustering results of CoC for a given stage, block, head.
# Different layers/heads will present different clustering patterns.
# Licensed under the Apache-2.0 license [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)

# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --stage {stage} --block {block} --head {head}
# --------------------------------------------------------

import models
import timm
import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from einops import rearrange
import random
from timm.models import load_checkpoint
from torchvision.utils import draw_segmentation_masks


parser = argparse.ArgumentParser(description='Context Cluster visualization')
parser.add_argument('--data_path', type=str,
                    default=r'C:\Users\13779\Desktop\dataset\CIFAR100',
                    help='path to imageFolder')
parser.add_argument('--image', type=int, default=1, help='path to image under the data_path')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
# 重塑图像大小
parser.add_argument('--scale', type=int, default=8, help='Resize img to feature-map size')
parser.add_argument('--model', default='coc_tiny', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--stage', default=0, type=int, help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=0, type=int, help='Index of visualized stage, -1 is the last block ,2,3,4,1')
parser.add_argument('--head', default=0, type=int,  help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
parser.add_argument('--checkpoint', type=str, default="./coc_mini_cifar_CIFAR100_epoch105.pth", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', default=1.0, type=float, help='Transparent, 0-1')
args = parser.parse_args()


import torchvision
# 建议类与映射标签
dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True)
class_names = dataset.classes  # ['apple', 'aquarium_fish', ..., 'worm']

# 构建 object_categories 映射（下标为 index，值为类名）
object_categories = ["" for _ in range(100)]
for idx, name in enumerate(class_names):
    object_categories[idx] = name   # 索引---类名映射
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# Preprocessing
def _preprocess(img_index):
    # 该方法可以预处理图像的性质
    # 调整image_path
    row_image = dataset[img_index][0]  # 第一个位置是图像，第二个位置是标签
    # 这个增强容易可视化，但是可能会出现预测不准确的情况，因为这与常规的验证集增强模式不同
    image = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
    ])(row_image)  # 与测试阶段使用相同的正则化模式
    return image, row_image


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,M,D]
    :param x2: [B,N,D]
    :return: similarity matrix [B,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.permute(0, 2, 1))
    return sim


# forward hook function
def get_attention_score(self, input, output):
    x = input[0]  # input tensor in a tuple
    value = self.v(x)
    x = self.f(x)
    x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
    value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
    if self.fold_w > 1 and self.fold_h > 1:
        b0, c0, w0, h0 = x.shape
        assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
            f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
        x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                      f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
        value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
    b, c, w, h = x.shape
    centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
    value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
    b, c, ww, hh = centers.shape
    sim = torch.sigmoid(self.sim_beta +
                        self.sim_alpha * pairwise_cos_sim(
                            centers.reshape(b, c, -1).permute(0, 2, 1),
                            x.reshape(b, c, -1).permute(0, 2,1)
                        )
                    )  # [B,M,N]
    # sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)  # binary #[B,M,N]
    # changed, for plotting mask.
    mask = mask.reshape(mask.shape[0], mask.shape[1], w, h)  # [(head*fold*fold),m, w,h]
    mask = rearrange(mask, "(h0 f1 f2) m w h -> h0 (f1 f2) m w h",
                     h0=self.heads, f1=self.fold_w, f2=self.fold_h)  # [head, (fold*fold),m, w,h]
    mask_list = []
    for i in range(self.fold_w):
        for j in range(self.fold_h):
            for k in range(mask.shape[2]):
                temp = torch.zeros(self.heads, w * self.fold_w, h * self.fold_h)
                temp[:, i * w:(i + 1) * w, j * h:(j + 1) * h] = mask[:, i * self.fold_w + j, k, :, :]
                mask_list.append(temp.unsqueeze(dim=0))  # [1, heads, w, h]

    mask2 = torch.concat(mask_list, dim=0)  # [ n, heads, w, h]
    global attention
    attention = mask2.detach()


def main():
    global attention
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    from models.ClusteringParadigm import coc_mini_cifar
    model = coc_mini_cifar(num_classes=args.num_classes)

    if args.checkpoint:  # 导入模型
        # timm
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError

    model.network[args.stage * 2][args.block].token_mixer.register_forward_hook(get_attention_score)

    model.eval()

    out = model(image)
    if type(out) is tuple:
        out = out[0]

    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'top1 ==> Prediction is: {object_categories[index]} possibility: {possibility * 100:.3f}%')
    possibility_top3, index_top3 = torch.topk(torch.softmax(out, dim=1).squeeze(), k=3)
    # 遍历每个样本
    print(f"top3 ==> Top 3 Predictions:")
    for i in range(3):  # 输出前三个预测
        # 获取预测的类别名称
        predicted_class = object_categories[index_top3[i].item()]
        # 获取对应类别的概率
        probability = possibility_top3[i].item() * 100  # 转换为百分比
        print(f"    {i + 1}. {predicted_class} - {probability:.3f}%")
    print(f"this image true class is {object_categories[dataset[args.image][1]]}")



    image_name = f"trainset[{args.image}]"
    os.makedirs(f"images_CIFAR100/{args.model}/{image_name}", exist_ok=True)

    from PIL import Image
    # 扩大图像
    h, w = raw_image.size
    h = h * args.scale
    w = w * args.scale
    img = raw_image.resize((h, w), Image.BILINEAR)  # 默认使用双线性插值

    from torchvision.transforms.functional import pil_to_tensor
    img_tensor = pil_to_tensor(img)  # 保持为 uint8，不归一化

    attention = attention[:, args.head, :, :]
    mask = attention.unsqueeze(dim=0)
    mask = F.interpolate(mask, (h, w))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    # randomly selected some good colors.
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue
    if mask.shape[0] == 4:
        colors = colors[0:4]
    if mask.shape[0] > 4:
        colors = colors * (mask.shape[0] // 16)
        random.seed(123)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img_tensor, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = f"images_CIFAR100/{args.model}/{image_name}/Stage{args.stage}_Block{args.block}_Head{args.head}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")


if __name__ == '__main__':
    main()
