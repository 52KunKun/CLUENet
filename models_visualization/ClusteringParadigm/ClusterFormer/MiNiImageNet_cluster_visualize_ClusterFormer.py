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
import json

with open(r"C:\Users\13779\Desktop\CV\project\myProject\test_on_miniDataset\mini_imagenet\classes_name.json","r") as f:
    name_to_idx = json.load(f)
object_categories = ["" for i in range(1000)]
for k,v in name_to_idx.items():
    object_categories[int(v[0])] = v[1]  # k是字符串索引序号，v是一个列表，第二个元素放置的是字符串含义


parser = argparse.ArgumentParser(description='Context Cluster visualization')
parser.add_argument('--data_path', type=str,
                    default=r'C:\Users\13779\Desktop\dataset\MINI_ImageNet\mini-imagenet\images',
                    help='path to imageFolder')
parser.add_argument('--image', type=str, default="n0153282900000016.jpg", help='path to image under the data_path')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
parser.add_argument('--model', default='cluster_mini', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--stage', default=2, type=int, help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=0, type=int, help='Index of visualized stage, -1 is the last block ,2,3,4,1')
parser.add_argument('--head', default=0, type=int,  help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--checkpoint', type=str, default="./ClusterFormer_mini_epoch105.pth", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', default=1.0, type=float, help='Transparent, 0-1')
args = parser.parse_args()


from PIL import Image
# Preprocessing
def _preprocess(image_path):
    # 该方法可以预处理图像的性质
    # 调整image_path
    row_image = Image.open(image_path)
    image = transforms.Compose(
        [
            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(row_image)  # 与测试阶段使用相同的正则化模式
    return image,row_image
def find_classes_image(img_path):
    file_name = os.path.basename(img_path)  # e.g., "n0153282900000122.jpg"
    # 提取类别编号（前9位）
    class_id = file_name[:9]  # e.g., "n01532829"
    # 查找类别名称
    class_name = name_to_idx.get(class_id, "Unknown")
    return class_name[1]


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
from flash_attn import flash_attn_func
def get_attention_score(self, input, output):
    x = input[0]
    value = self.conv_v(x)
    feature = self.conv_f(x)
    x = self.conv1(x)

    # multi-head
    b, c, w, h = x.shape
    x = x.reshape(b * self.heads, int(c / self.heads), w, h)
    value = value.reshape(b * self.heads, int(c / self.heads), w, h)
    feature = feature.reshape(b * self.heads, int(c / self.heads), w, h)

    # window token
    if self.window_w > 1 and self.window_h > 1:
        b, c, w, h = x.shape
        x = x.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w), int(h / self.window_h))
        value = value.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w), int(h / self.window_h))
        feature = feature.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w), int(h / self.window_h))

    b, c, w, h = x.shape
    value = value.reshape(b, w * h, c)

    # centers
    centers = self.centers_proposal(x)
    b, c, c_w, c_h = centers.shape
    centers_feature = self.centers_proposal(feature).reshape(b, c_w * c_h, c)

    feature = feature.reshape(b, w * h, c)

    # processing before flash attention
    centers = centers.reshape(int(b / self.heads), c_w * c_h, self.heads, c).type(torch.half)
    value = value.reshape(int(b / self.heads), w * h, self.heads, c).type(torch.half)
    feature = feature.reshape(int(b / self.heads), w * h, self.heads, c).type(torch.half)

    for _ in range(self.num_clustering):  # iterative clustering and updating centers
        centers = flash_attn_func(centers, value, feature)

    # processing after flash attention
    centers = centers.reshape(b, c, c_w, c_h).type(torch.float)

    # similarity
    similarity = torch.sigmoid(
        self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b, c, -1).permute(0, 2, 1),
                                                          x.reshape(b, c, -1).permute(0, 2, 1)))

    # assign each point to one center
    _, max_idx = similarity.max(dim=1, keepdim=True)
    mask = torch.zeros_like(similarity)
    mask.scatter_(1, max_idx, 1.)


    # changed, for plotting mask.
    mask = mask.reshape(mask.shape[0], mask.shape[1], w, h)  # [(head*fold*fold),m, w,h]
    mask = rearrange(mask, "(h0 f1 f2) m w h -> h0 (f1 f2) m w h",
                     h0=self.heads, f1=self.window_w, f2=self.window_h)  # [head, (fold*fold),m, w,h]
    mask_list = []
    for i in range(self.window_w):
        for j in range(self.window_h):
            for k in range(mask.shape[2]):
                temp = torch.zeros(self.heads, w * self.window_w, h * self.window_h)
                temp[:, i * w:(i + 1) * w, j * h:(j + 1) * h] = mask[:, i * self.window_w + j, k, :, :]
                mask_list.append(temp.unsqueeze(dim=0))  # [1, heads, w, h]

    mask2 = torch.concat(mask_list, dim=0)  # [ n, heads, w, h]
    global attention
    attention = mask2.detach().cpu()


def main():
    global attention
    # 预处理图像
    args.image = os.path.join(args.data_path, args.image)
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    from models.ClusteringParadigm import cluster_mini,cluster_mini_vis
    model = cluster_mini_vis(num_classes=args.num_classes)

    if args.checkpoint:  # 导入模型
        # timm
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError

    model.network[args.stage * 2][args.block].token_mixer.register_forward_hook(get_attention_score)

    model.cuda()
    model.eval()
    device = next(model.parameters()).device
    out = model(image.to(device))
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
    print(f"this image true class is {find_classes_image(args.image)}")


    from torchvision.io import read_image
    img = read_image(args.image)

    # 取出即可
    attention = attention[:, args.head, :, :]
    mask = attention.unsqueeze(dim=0)
    mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    # randomly selected some good colors.
    # randomly selected some good colors.
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue

    if mask.shape[0] == 4:
        colors = colors[0:4]
    if mask.shape[0] > 4:
        colors = (colors * (mask.shape[0] // 16 + 1))[:mask.shape[0]]
        random.seed(123)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    image_name = os.path.basename(args.image).split(".")[0]
    os.makedirs(f"images/{args.model}/{image_name}", exist_ok=True)
    save_path = f"images/{args.model}/{image_name}/Stage{args.stage}_Block{args.block}_Head{args.head}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")
if __name__ == '__main__':
    main()
