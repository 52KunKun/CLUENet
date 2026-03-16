# --------------------------------------------------------
# Context Cluster Visualization
# A script to visualize the clustering results of CoC for a given stage, block, head.
# Different layers/heads will present different clustering patterns.
# Licensed under the Apache-2.0 license [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)

# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --stage {stage} --block {block} --head {head}
# --------------------------------------------------------

import os
# 这行代码必须在 import sklearn 或 import numpy 之前运行
os.environ["OMP_NUM_THREADS"] = "1"

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
from torch_scatter import scatter_sum
from sklearn.cluster import KMeans
import json

# 获取ImageNet类别
# 建立idx-->catacaroy_str的映射表（列表即可）
with open(r"C:\Users\13779\Desktop\CV\project\remote_control_autodl\PaperProject\test_on_miniDataset\mini_imagenet\classes_name.json","r") as f:
    name_to_idx = json.load(f)
object_categories = ["" for i in range(1000)]
for k,v in name_to_idx.items():
    object_categories[int(v[0])] = v[1]  # k是字符串索引序号，v是一个列表，第二个元素放置的是字符串含义

parser = argparse.ArgumentParser(description='Context Cluster visualization')
parser.add_argument('--data_path', type=str, default=r'D:\datasets\dataset\MINI_ImageNet\mini-imagenet\images', help='path to imageFolder')
parser.add_argument('--image', type=str, default="n0153282900000016.jpg", help='path to image under the data_path')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
# 重塑图像大小
parser.add_argument('--model', default='FEC_small', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--stage', default=0, type=int, help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=0, type=int, help='Index of visualized stage, -1 is the last block ,2,3,4,1')
parser.add_argument('--head', default=0, type=int,  help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--checkpoint', type=str, default="./FEC_mini_epoch105.pth", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', default=1.0, type=float, help='Transparent, 0-1')
parser.add_argument('--num-clu', type=int, default=5,help='number of clusters')
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
    return class_name[0],class_name[1]

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
    x = input[0]
    value = self.v(x)
    x = self.f(x)
    x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
    value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
    if self.fold_w > 1 and self.fold_h > 1:
        # split the big feature maps to small local regions to reduce computations.
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
    sim = torch.sigmoid(
        self.sim_beta +
        self.sim_alpha * pairwise_cos_sim(
            centers.reshape(b, c, -1).permute(0, 2, 1),
            x.reshape(b, c, -1).permute(0, 2, 1)
        )
    )  # [B,M,N]
    # we use mask to sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)
    sim = sim * mask
    value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
    # aggregate step, out shape [B,M,D]
    M, N = value_centers.shape[1], value2.shape[1]
    value2 = rearrange(value2, 'b n c -> (b n) c')

    mask_idx = sim_max_idx.squeeze(1)  # [head,N]

    sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')

    idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
    sim_max_idx = sim_max_idx + idx_offset
    out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b * M), '(b m) c -> b m c', b=b,
                    m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
    out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

    global attention,input_shape,output_shape
    attention = [mask_idx.detach().cpu().numpy().tolist(),out.detach().cpu().numpy()]
    input_shape = (w, h)
    output_shape = (ww, hh)


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


# 目前分簇个数基本都小于20不需要进一步使用KMeans
def aggregate_masks_to_stage(cluster_mask,stage,mask_layers,num_k=None,feats=None):
    """
    Args:
        cluster_mask: [N]
        stage: int 0~3

    Returns:mask[M,W,H]  W=224,H=224
    """
    # 无论如何总要进入块嵌入层
    # mask12装载了每个经过4*4折叠的像素块得到的感受野索引原始图像拉平为1D的位置 这里都将2D拉平为1D来看
    # 这代表了第一层块嵌入4*4对应的感受野区域
    mask12 = {i: [] for i in range(56 * 56)}
    idx = 0
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            # the clustering is based on 4x4 pixel patch (after standard conv)
            # 这里包含了4*4图像块的2D拉平后的所有索引位置
            mask12[idx] = [i * 224 + j, i * 224 + j + 1, i * 224 + j + 2, i * 224 + j + 3,
                           (i + 1) * 224 + j, (i + 1) * 224 + j + 1, (i + 1) * 224 + j + 2, (i + 1) * 224 + j + 3,
                           (i + 2) * 224 + j, (i + 2) * 224 + j + 1, (i + 2) * 224 + j + 2, (i + 2) * 224 + j + 3,
                           (i + 3) * 224 + j, (i + 3) * 224 + j + 1, (i + 3) * 224 + j + 2, (i + 3) * 224 + j + 3,
                           ]
            idx += 1

    # 如果stage=0 则没有进入第0个簇池化，仅进入固定块嵌入
    # 先建立好输入到该阶段基础像素点的感受野
    mask23 = {i: [] for i in range(28 * 28)}
    for i, j in enumerate(mask_layers[0][0]):
        mask23[j].extend(mask12[i])

    mask34 = {i: [] for i in range(14 * 14)}
    for i, j in enumerate(mask_layers[1][0]):
        mask34[j].extend(mask23[i])

    mask45 = {i: [] for i in range(7 * 7)}
    for i, j in enumerate(mask_layers[2][0]):
        mask45[j].extend(mask34[i])

    # 建立好全部阶段的池化感受野
    global input_shape,output_shape
    H,W = input_shape
    CH,CW = output_shape
    M = CH*CW

    input_mask = [mask12,mask23,mask34,mask45][stage]  # 处于哪个阶段  input_mask[i]表示 将2D拉平为1D后位于位置i的像素点在原始图像上的感受野索引列表（也是2D拉平为1D）
    output_mask = {i:[] for i in range(M)}   # 建立簇中心个数的像素掩码张量

    for i,j in enumerate(cluster_mask):  # cluster_mask[i]标记了拉平为1D形式下每个像素点被分派到的簇索引（每个像素点有且仅会指向一个簇）
        output_mask[j].extend(input_mask[i])
    # 将空簇删除
    non_empty_clusters, idx_map = [], {}
    for i in range(M):
        if len(output_mask[i]) > 0:
            non_empty_clusters.append(i)
            idx_map[len(idx_map)] = i

    # print(f"the true num of clusters is {len(non_empty_clusters)}")
    # final_mask = torch.zeros((1, len(non_empty_clusters), 224 * 224))
    # for idx1, idx2 in enumerate(non_empty_clusters):
    #     final_mask[0, idx1, output_mask[idx2]] = 1
    # final_mask = final_mask.reshape(1, len(non_empty_clusters), 224, 224)

    if num_k is None or num_k >= len(non_empty_clusters):
        final_mask = torch.zeros((1, len(non_empty_clusters), 224 * 224))
        for idx1, idx2 in enumerate(non_empty_clusters):
            final_mask[0, idx1, output_mask[idx2]] = 1
        final_mask = final_mask.reshape(1, len(non_empty_clusters), 224, 224)
    else:
        mask56 = {i: [] for i in range(num_k)}
        feats = feats[non_empty_clusters]  # feats要取空
        kmeans = KMeans(n_clusters=num_k, random_state=0, n_init="auto").fit(feats)
        for idx1, idx2 in enumerate(kmeans.labels_):
            mask56[idx2].extend(output_mask[idx_map[idx1]])
        # _check(mask56)
        final_mask = torch.zeros((1, num_k, 224 * 224))
        for idx1, idx2 in enumerate(range(num_k)):
            final_mask[0, idx1, mask56[idx2]] = 1
        final_mask = final_mask.reshape(1, num_k, 224, 224)

    return final_mask,non_empty_clusters

from compute_cluster_importance import *

import cv2
def soften_binary_mask(mask_tensor, kernel_size=11):
    """
    对硬二值mask做邻域均值平滑，模拟soft mask。
    用于 mask 后处理前的“去碎片化”模拟。
    """
    softened = []
    for i in range(mask_tensor.shape[0]):
        mask = mask_tensor[i].cpu().numpy().astype(np.float32)
        blurred = cv2.blur(mask, (kernel_size, kernel_size))  # 均值滤波
        softened.append(blurred)
    softened = np.stack(softened)
    return torch.from_numpy(softened).to(mask_tensor.device)

@torch.no_grad()
def main():
    global attention,mask_layers,input_shape,output_shape
    mask_layers = []
    args.image = os.path.join(args.data_path, args.image)
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    from models.ClusteringParadigm import fec_small
    model = fec_small(num_classes=args.num_classes)

    if args.checkpoint:  # 导入模型
        # timm
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError

    model.network[args.stage * 2][args.block].token_mixer.register_forward_hook(get_attention_score)
    model.network[1].register_forward_hook(fwd_hook_fec)  # 目的是为了
    model.network[3].register_forward_hook(fwd_hook_fec)
    model.network[5].register_forward_hook(fwd_hook_fec)
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
    image_class_idx, image_class_name = find_classes_image(args.image)
    print(f"this image true class is {image_class_name}")
    isModelTrue = True if index == image_class_idx else False

    from torchvision.io import read_image
    img = read_image(args.image)

    mask_idx = attention[0][args.head]
    feats = attention[1][args.head,:]
    # 需要调和mask，我们需要沿着池化层标记然后再映射到mask上
    # [1,M,224,224]
    mask,non_empty_clusters_idx = aggregate_masks_to_stage(mask_idx,args.stage,mask_layers,args.num_clu,feats)  # 指定mask[M,W,H]，以及第几个阶段
    mask = compute_cluster_contribution_combined_parallel(mask_MHW=mask, input_image_224=image, model=model,
                                                          true_class=image_class_idx)

    # 美化
    mask = mask.float()  # 转为float类型，使双线性插值变得自然
    mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    mask = mask.squeeze(dim=0)
    # 平滑处理
    # Step 2: 多级美化处理（增强 mask 连续性 + 对比度）
    # 2.1 Soft mask gamma增强（放大强响应，使边界更明显）
    mask = mask ** 1.5  # gamma 调整，可调为 1.2~2.0
    # 2.2 大核模糊平滑（扩大区域连贯性）
    mask = soften_binary_mask(mask, kernel_size=13)  # 可尝试 11~15
    # 2.3 全局归一化（让每个通道值域回归 [0, 1]，保持一致性）
    mask_min = mask.amin(dim=(1, 2), keepdim=True)
    mask_max = mask.amax(dim=(1, 2), keepdim=True)
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-6)
    # Step 3: 独占分配（每像素只归属于一个簇）
    argmax_mask = mask.argmax(dim=0)  # [H, W]
    mask = torch.stack([(argmax_mask == k) for k in range(mask.shape[0])]).to(mask.device)  # [K, H, W] Bool tensor

    # randomly selected some good colors.
    colors = [
        (255, 191, 0),  # 琥珀色/金黄色
        (42, 42, 165),  # 深蓝色
        (0, 128, 0),  # 绿色
        (255, 0, 0),  # 红色
        (0, 100, 0),  # 深绿色
        (139, 139, 0),  # 橄榄色/暗黄色
        (80, 127, 255),  # 淡蓝色/矢车菊蓝
        (255, 248, 240),  # 米白色/花卉白
        (255, 255, 255),  # 白色
        (0, 0, 0),  # 黑色
        (220, 245, 245),  # 淡青色/蔚蓝色
        (0, 0, 255),  # 蓝色
        (71, 99, 255),  # 亮蓝色/皇家蓝
        (50, 205, 154),  # 青绿色/绿松石色
        (238, 130, 238),  # 紫罗兰色
        (113, 179, 60)  # 草绿色
    ]

    if mask.shape[0] <=len(colors):
        colors = colors[0:mask.shape[0]]
    if mask.shape[0] > len(colors):
        colors = colors * (mask.shape[0] // len(colors)+1)

    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    img_with_masks = cv2.cvtColor(img_with_masks, cv2.COLOR_RGB2BGR)

    image_name = os.path.basename(args.image).split(".")[0]
    os.makedirs(f"images/{args.model}/{image_name}", exist_ok=True)
    save_path = f"images/{args.model}/{image_name}/Stage{args.stage}_Block{args.block}_Head{args.head}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")

if __name__ == '__main__':
    layers = [3, 4, 5, 2]
    heads = [4, 4, 8, 8]
    image_path_list = [
        "n0153282900000016",
        "n0153282900000042",
        "n0153282900000470",
        "n0174993900000687",
        "n0184338300000156",
        "n0207436700000961",
        "n0924646400000548",
        "n0216545600001051",
    ]
    image_path_list = [
        "n0924646400000548",
    ]
    args.model = "FEC_small"
    args.shape = 224
    args.num_classes = 100
    args.checkpoint = "./FEC_mini_epoch105.pth"
    args.alpha = 1.0
    args.num_clu = 5
    for image in image_path_list:
        if not image.endswith('.jpg'):
            image_path = image + '.jpg'
        else:
            image_path = image
        args.image = image_path
        for stage in range(len(layers)):
            for layer in range(layers[stage]):
                for head in range(heads[stage]):
                    args.stage = stage
                    args.block = layer
                    args.head= head
                    main()

