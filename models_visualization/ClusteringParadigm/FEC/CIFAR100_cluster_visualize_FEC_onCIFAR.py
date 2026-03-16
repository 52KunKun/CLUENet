# --------------------------------------------------------
# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize_CLUAmodel.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --num_clu {number of clusters}
# --------------------------------------------------------

import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
import random
from torchvision.utils import draw_segmentation_masks
from torch_scatter import scatter_sum
from einops import rearrange
from sklearn.cluster import KMeans
import json
from models import CLUA_past
from PIL import Image
from models.ClusteringParadigm import fec
import shutil
# 忽略警告信息
# 获取ImageNet类别

# 输入参数Arugument允许在执行python文件指定参数
parser = argparse.ArgumentParser(description='mycluster model visualization')
# --image代表输入图像路径
parser.add_argument('--data_path', type=str,
                    default=r'C:\Users\13779\Desktop\dataset\CIFAR100',
                    help='path to imageFolder')
parser.add_argument('--image', type=int, default=1, help='path to image under the data_path')

# --shape表示图像的形状
parser.add_argument('--shape', type=int, default=224, help='image size')
# 指定模型
parser.add_argument('--model', default='FEC_small', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
# 重塑图像大小
parser.add_argument('--scale', type=int, default=8, help='Resize img to feature-map size')
# --checkpoint显然指定的预训练模型文件
parser.add_argument('--checkpoint', type=str, default="./fec_mini_cifar_CIFAR100_epoch105.pth", metavar='PATH',
                    help='path to pretrained checkpoint')
# --alpha
parser.add_argument('--alpha', default=1.0, type=float, help='Transparent, 0-1')
# Note that FEC only results in 49 clusters in the final stage so that we have to adopt KMeans for easier inspection (see the second paragraph in Sec. 5.2).
# FEC may be sensitive when num_clu is a relatively small value, due to the KMeans algorithms.
# 这里说明FEC最后有49个簇类，如果要指定更小规模的聚类需要进一步使用K-Means  建议使用3、4、10（至少为3）
parser.add_argument('--num_clu', type=int, default=3,help='number of clusters')
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

# 捕获输入和输出
def fwd_hook_fec(self, input, output):
    x = input[0]  # input tensor in a tuple
    value = self.conv_v(x)
    x = self.conv_f(x)
    assert self.fold_w == 1 and self.fold_h == 1

    b, c, w, h = x.shape
    centers = F.adaptive_avg_pool2d(x, (w // self.stride, h // self.stride))
    value_centers = rearrange(F.adaptive_avg_pool2d(value, (w // self.stride, h // self.stride)), 'b c w h -> b (w h) c')
    b, c, ww, hh = centers.shape
    sim = pairwise_cos_sim( centers.reshape(b, c, -1).permute(0, 2, 1), x.reshape(b, c, -1).permute(0, 2, 1) )  # [B,M,N]
    # we use mask to sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    global mask_layers
    if sim_max_idx.shape[0] == 1:
        mask_layers.append([sim_max_idx[0, 0, :].detach().cpu().numpy().tolist(), rearrange(centers, 'b c w h -> b (w h) c')[0].detach().cpu().numpy()])
    else:
        mask_layers.append([sim_max_idx[0, 0, :].detach().cpu().numpy().tolist(), None])

def aggregate_masks(mask_layers, num_k=None):
    # mask12装载了每个经过4*4折叠的像素块得到的感受野索引原始图像拉平为1D的位置 这里都将2D拉平为1D来看
    # 这代表了第一层块嵌入4*4对应的感受野区域
    mask23 = {i: [] for i in range(16 * 16)}
    for i, j in enumerate(mask_layers[0][0]):
        mask23[j].append(i)

    mask34 = {i: [] for i in range(8 * 8)}
    for i, j in enumerate(mask_layers[1][0]):
        mask34[j].extend(mask23[i])

    mask45 = {i: [] for i in range(4 * 4)}
    for i, j in enumerate(mask_layers[2][0]):
        mask45[j].extend(mask34[i])

    non_empty_clusters, idx_map = [], {}
    for i in range(4 * 4):
        if len(mask45[i]) > 0:
            non_empty_clusters.append(i)
            idx_map[len(idx_map)] = i

    if num_k is None or num_k >= len(non_empty_clusters):
        final_mask = torch.zeros((1, len(non_empty_clusters), 32 * 32))
        for idx1, idx2 in enumerate(non_empty_clusters):
            final_mask[0, idx1, mask45[idx2]] = 1
        final_mask = final_mask.reshape(1, len(non_empty_clusters), 32, 32)
    else:
        mask56 = {i: [] for i in range(num_k)}
        feats = mask_layers[2][1][non_empty_clusters]
        kmeans = KMeans(n_clusters=num_k, random_state=0, n_init="auto").fit(feats)
        for idx1, idx2 in enumerate(kmeans.labels_):
            mask56[idx2].extend(mask45[idx_map[idx1]])
        # _check(mask56)
        final_mask = torch.zeros((1, num_k, 32 * 32))
        for idx1, idx2 in enumerate(range(num_k)):
            final_mask[0, idx1, mask56[idx2]] = 1
        final_mask = final_mask.reshape(1, num_k, 32, 32)

    return final_mask

@torch.no_grad()
def infer(model, args):
    model.cuda()
    model.eval()    # 测试阶段
    device = next(model.parameters()).device
    global mask_layers  # forward hook，可以获取中间层的信息
    mask_layers = []
    # 预处理图像，并保存原始图像用于可视化
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    # 转为cuda形式
    out = model(image.to(device))  # 前向传播
    if type(out) is tuple: out = out[0]  # 最后logits分数
    # 打印输出概率最合适的类别
    possibility = torch.softmax(out, dim=1).max()
    # 得到最后最合适的类别
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

    # 扩大图像
    h, w = raw_image.size
    h = h * args.scale
    w = w * args.scale
    img = raw_image.resize((h, w), Image.BILINEAR)  # 默认使用双线性插值

    from torchvision.transforms.functional import pil_to_tensor
    img_tensor = pil_to_tensor(img)  # 保持为 uint8，不归一化

    # mask_layers捕获了网络层中的每个像素点指定的簇分配索引
    mask = aggregate_masks(mask_layers, args.num_clu)
    mask = F.interpolate(mask, (h, w))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5

    # randomly selected some good colors.
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue
    if mask.shape[0] <= len(colors):
        colors = colors[:mask.shape[0]]
    else:
        colors = (colors * (mask.shape[0] // 16 + 1))[:mask.shape[0]]

    img_with_masks = draw_segmentation_masks(img_tensor, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    # 并保存原始图像
    save_path = f"images_CIFAR100/{args.model}/{image_name}/all_stages.png"
    save_path2 = f"images_CIFAR100/{args.model}/{image_name}/original_image.png"
    cv2.imwrite(save_path, img_with_masks)  # 语义图像
    # 复制该图片
    img.save(save_path2, format="PNG")
    print(f"==> Generated image is saved to: {save_path}")

def main():
    # 使用timm方法创建model
    # model = timm.create_model(model_name=args.model, pretrained=True)
    model = fec.fec_mini_cifar(num_classes=args.num_classes)
    if args.checkpoint: # 导入模型
        # timm
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError
    # 指定模型层标记
    # 记录的下表位置正好是使用聚类池化的地方，
    # 第一次下采样使用块嵌入，之后三次使用的都是聚类嵌入
    model.network[1].register_forward_hook(fwd_hook_fec)
    model.network[3].register_forward_hook(fwd_hook_fec)
    model.network[5].register_forward_hook(fwd_hook_fec)

    infer(model, args)


if __name__ == '__main__':
    main()
