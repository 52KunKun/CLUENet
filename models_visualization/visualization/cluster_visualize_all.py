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
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from einops import rearrange
import random
from torchvision.utils import draw_segmentation_masks
from torch_scatter import scatter_sum
import json
from PIL import Image
from models import CLUA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from models.utils import get_avgpool_mask_4D
import shutil

# 获取ImageNet类别
# 建立idx-->catacaroy_str的映射表（列表即可）
with open(r"C:\Users\13779\Desktop\CV\project\remote_control_autodl\PaperProject\test_on_miniDataset\mini_imagenet\classes_name.json","r") as f:
    name_to_idx = json.load(f)
object_categories = ["" for i in range(1000)]
for k,v in name_to_idx.items():
    object_categories[int(v[0])] = v[1]  # k是字符串索引序号，v是一个列表，第二个元素放置的是字符串含义

model_dict = {
    "CLUA_mini":CLUA.CLUA_mini,
    "CLUA_tiny_compare":CLUA.CLUA_tiny,
    "CLUA_small":CLUA.CLUA_small,
}

parser = argparse.ArgumentParser(description='mycluster stage visualization')
parser.add_argument('--data_path', type=str, default=r'C:\Users\13779\Desktop\dataset\MINI_ImageNet\mini-imagenet\images', help='path to imageFolder')
parser.add_argument('--image', type=str, default="n0153282900000016.jpg", help='path to image under the data_path')

parser.add_argument('--shape', type=int, default=224, help='image size')
parser.add_argument('--model', default='CLUA_tiny_compare', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
# 指定哪个阶段
# 由于我们阶段内部共享分簇结果，不需要指定块,但要指定注意力头
# parser.add_argument('--stage',  type=int,default=1, help='Index of visualized stage, 0-3')
# parser.add_argument('--head',  type=int, default=0, help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
parser.add_argument('--checkpoint', type=str, default="./CLUA_tiny_mini_epoch105.pth", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', type=float,default=1.0, help='Transparent, 0-1')  # 透明度，可以看到原始图片
parser.add_argument('--num-clu', type=int, default=5,help='number of clusters')
parser.add_argument('--model-num-clu', type=int, default=3,help='number of clusters')

args = parser.parse_args()

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

from flash_attn import flash_attn_func
# forward hook function
def get_attention_score(self, input, output):
    x = input[0]  # input tensor in a tuple
    _,_,H,W = x.shape  # 保存一些关键变量
    sim = output[1]  # 输出值中第二个位置就是sim
    M = sim.shape[1]
    mask = sim>1e-6  # [H,M,N]
    mask_idx = sim.argmax(dim=1)  # [H,N]
    # 重新做一遍，找到簇中心
    b, c, h, w = x.shape  # 可能改变的形状
    # 再映射，这样Q、k都是固定的形状
    K = self.k(x)
    K_centers = self.pool(K)  # 是否
    feature = self.v(x)  # 捕捉特征
    centers_feature = self.pool(feature)

    # 一步实现 2D转为序列
    K = rearrange(K, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
    K_centers = rearrange(K_centers, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
    feature = rearrange(feature, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads).type(torch.half)
    scale = 1 / self.attn_alpha.exp()
    K_norm = F.normalize(K, dim=-1)  # 后面还要用
    K_centers = (F.normalize(K_centers, dim=-1) * scale).type(torch.half)
    # 下面的方法会导致学习效率低下的问题
    # 可以加入偏置
    cluster_centers = flash_attn_func(K_centers, K_norm.type(torch.half), feature, softmax_scale=1).type(
        torch.float)  # 不进行任何归一化

    # 不引入局部特征
    centers_feature = rearrange(centers_feature, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
    # 否则就使用来自上一个块的connect
    # 特征上的中心残差---->可调控的可残差，对残差实现gate调控
    gate = self.gate(cluster_centers, centers_feature)  # 基于两者内容生成gate门
    cluster_centers = (1 - gate) * cluster_centers + gate * centers_feature  # 特征残差传播梯度

    # 特征分派---硬分配模式（可以改为窗口）
    cluster_centers = rearrange(cluster_centers, 'b n hd d -> (b hd) n d')

    # 同时把聚类中心也要加入进来
    global attention,input_shape,output_shape
    # 最后两个维度拉平（报错所有阶段的注意力因子与输入形状与输出形状——以便对任意形状进行可视化）
    attention.append([mask.reshape(mask.shape[0],mask.shape[1],h,w).detach().cpu(),mask_idx.detach().cpu().numpy().tolist(),cluster_centers.detach().cpu().numpy()])  # 拿到这个阶段所有聚类中心的向量
    input_shape.append((H,W))
    output_shape.append((self.ch,self.cw))

# 捕获输入和输出
def fwd_hook_fec(self, input, output):
    # 直接在input上捕捉信息即可
    x = input[0]
    value = self.conv_f(x)  # 不对x进行任何映射，直接去捕捉x的分簇结构
    x = value  # 共享
    # 使用自回归，共享
    b, c, w, h = x.shape
    centers = F.adaptive_avg_pool2d(x, (w // self.stride, h // self.stride))
    value_centers = rearrange(F.adaptive_avg_pool2d(value, (w // self.stride, h // self.stride)),
                              'b c w h -> b (w h) c')
    b, c, ww, hh = centers.shape
    sim = pairwise_cos_sim(centers.reshape(b, c, -1).permute(0, 2, 1),
                           x.reshape(b, c, -1).permute(0, 2, 1))  # [B,M,N]
    # we use mask to sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)
    value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
    # aggregate step, out shape [B,M,D]
    M, N = value_centers.shape[1], value2.shape[1]
    value2 = rearrange(value2, 'b n c -> (b n) c')
    sim_max_idx = rearrange(sim_max_idx.squeeze(1), 'b n -> (b n)')
    idx_offset = (torch.arange(b, device=sim_max_idx.device) * M).unsqueeze(-1).expand(-1, N).flatten()
    sim_max_idx = sim_max_idx + idx_offset
    out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b * M), '(b m) c -> b m c', b=b,
                    m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
    out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)
    # 加入到全局变量
    global mask_layers
    # [N]--[M,D]  M=N//stride**2 （保存所有池化层的表示）
    mask_layers.append((sim_max_idx.cpu().numpy().tolist(),out[0].cpu().numpy()))


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
    H,W = input_shape[stage]
    CH,CW = output_shape[stage]
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
        # 归一化后求取欧式距离 与相似度可以等价?
        # feats = feats/np.linalg.norm(feats,axis=-1,keepdims=True)
        kmeans = KMeans(n_clusters=num_k, random_state=0,n_init="auto").fit(feats)
        for idx1, idx2 in enumerate(kmeans.labels_):
            mask56[idx2].extend(output_mask[idx_map[idx1]])
        # _check(mask56)
        final_mask = torch.zeros((1, num_k, 224 * 224))
        for idx1, idx2 in enumerate(range(num_k)):
            final_mask[0, idx1, mask56[idx2]] = 1
        final_mask = final_mask.reshape(1, num_k, 224, 224)

    return final_mask,non_empty_clusters



def aggregate_masks(mask_layers, num_k=None):
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
    # 创建56*56的索引，注意这是进一步的感受野关注区域，也就是第一次使用聚类池化的感受野区域
    mask23 = {i: [] for i in range(28 * 28)}
    for i, j in enumerate(mask_layers[0][0]):
        mask23[j].extend(mask12[i])

    mask34 = {i: [] for i in range(14 * 14)}
    for i, j in enumerate(mask_layers[1][0]):
        mask34[j].extend(mask23[i])

    mask45 = {i: [] for i in range(7 * 7)}
    for i, j in enumerate(mask_layers[2][0]):
        mask45[j].extend(mask34[i])

    non_empty_clusters, idx_map = [], {}
    for i in range(7 * 7):
        if len(mask45[i]) > 0:
            non_empty_clusters.append(i)
            idx_map[len(idx_map)] = i

    if num_k is None or num_k >= len(non_empty_clusters):
        final_mask = torch.zeros((1, len(non_empty_clusters), 224 * 224))
        for idx1, idx2 in enumerate(non_empty_clusters):
            final_mask[0, idx1, mask45[idx2]] = 1
        final_mask = final_mask.reshape(1, len(non_empty_clusters), 224, 224)
    else:
        mask56 = {i: [] for i in range(num_k)}
        feats = mask_layers[2][1][non_empty_clusters]
        kmeans = KMeans(n_clusters=num_k, random_state=0, n_init="auto").fit(feats)
        for idx1, idx2 in enumerate(kmeans.labels_):
            mask56[idx2].extend(mask45[idx_map[idx1]])
        # _check(mask56)
        final_mask = torch.zeros((1, num_k, 224 * 224))
        for idx1, idx2 in enumerate(range(num_k)):
            final_mask[0, idx1, mask56[idx2]] = 1
        final_mask = final_mask.reshape(1, num_k, 224, 224)

    return final_mask


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
    global attention,mask_layers,input_shape,output_shape   # attention为阶段内部的mask，mask_layers是所有池化层的
    # 初始化全局变量
    attention = []
    mask_layers = []
    input_shape = []
    output_shape = []
    heads_dict = {
        "CLUA_mini":[1, 2, 5, 8],
        "CLUA_tiny_compare":[4, 4, 8, 8],
        "CLUA_small":[4, 4, 8, 8],
    }
    heads = heads_dict[args.model]
    # 预处理图像
    args.image = os.path.join(args.data_path, args.image)
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    # model = timm.create_model(model_name=args.model, pretrained=True)
    model = model_dict[args.model](num_classes=args.num_classes)

    if args.checkpoint:  # 导入模型
        # timm
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # 假设保存的文件包含 'model_state_dict'
        print(f"\n\n==> Loaded checkpoint {args.checkpoint}")
    else:
        raise ValueError

    # 但是，但是，但是 同时也要给所有池化层打入标记
    for stage_idx in range(4):
        model.network[stage_idx * 2].blocks[0].token_mixer.register_forward_hook(get_attention_score)
        if 2*stage_idx+1<len(model.network):
            model.network[2*stage_idx+1].register_forward_hook(fwd_hook_fec)

    model.cuda()
    model.eval()  # 测试阶段
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

    isModelTrue = True if object_categories[index] == find_classes_image(args.image) else False

    from torchvision.io import read_image
    img = read_image(args.image)
    masks = []



    for i in range(4):
        attention_stage = attention[i]
        for head in range(heads[i]):
            mask_idx = attention_stage[1][head]
            feats = attention_stage[2][head,:]
            # 需要调和mask，我们需要沿着池化层标记然后再映射到mask上
            # [1,M,224,224]
            # 这里会筛选非空簇
            mask,non_empty_clusters_idx = aggregate_masks_to_stage(mask_idx,i,mask_layers,args.num_clu,feats)  # 指定mask[M,W,H]，以及第几个阶段
            masks.append(mask)  # 存储阶段可视化结果
    mask = aggregate_masks(mask_layers,args.model_num_clu)
    masks.append(mask)

    # 无美化
    # mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    # mask = mask.squeeze(dim=0)
    # mask = mask > 0.5
    assert len(masks)==sum(heads)+1  # 应该是完全一致的

    max_cluster_num = 0
    for i in range(len(masks)):
        mask = masks[i]  # 循环处理
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
        max_cluster_num = max(max_cluster_num, mask.shape[0])
        masks[i] = mask


    # randomly selected some good colors.（把显著区分的五个颜色放在一起以免混淆）
    colors_pre = [
        "brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]
    # colors_pre = [
    #     "blue", "brown", "green", "deepskyblue", "darkgreen", "darkcyan", "coral", "aliceblue",
    #           "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]
    if max_cluster_num<= len(colors_pre):
        colors = colors_pre[:max_cluster_num]
    else:
        colors = (colors_pre * (max_cluster_num // 16 + 1))[:max_cluster_num]
    # 打乱颜色
    # random.shuffle(colors)
    # colors_RGB = [tuple(int(255 * c) for c in mcolors.to_rgb(name)) for name in colors]
    # 调整颜色亮度（转为RGB+调整颜色亮度）

    image_name = os.path.basename(args.image).split(".")[0]
    if isModelTrue:
        image_name = image_name + "_right"
    else:
        image_name = image_name + "_wrong"
    os.makedirs(f"images/{args.model}/{image_name}", exist_ok=True)

    index = 0
    for stage in range(4):
        for head in range(heads[stage]):
            mask = masks[index]
            img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
            img_with_masks = img_with_masks.detach()
            img_with_masks = TransF.to_pil_image(img_with_masks)
            img_with_masks = np.asarray(img_with_masks)
            save_path = f"images/{args.model}/{image_name}/stage{stage}_head{head}_{mask.shape[0]}CLU.png"
            cv2.imwrite(save_path, img_with_masks)
            print(f"==> Generated image is saved to: {save_path}")
            index += 1 # 索取下一个元素

    mask = masks[index] # 最后一个元素
    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = f"images/{args.model}/{image_name}/all_stages.png"
    save_path2 = f"images/{args.model}/{image_name}/original_image.png"
    cv2.imwrite(save_path, img_with_masks)
    # 复制该图片
    shutil.copy(args.image, save_path2)
    print(f"==> Generated image is saved to: {save_path}")

if __name__ == '__main__':

    image_path_list = [
        "n0153282900000016"
    ]
    args.model = "CLUA_tiny_compare"
    args.shape = 224
    args.num_classes = 100
    args.checkpoint = "./CLUA_tiny_mini_epoch105.pth"
    args.alpha = 1.0
    args.num_clu = 5
    for image in image_path_list:
        if not image.endswith('.jpg'):
            image_path = image + '.jpg'
        else:
            image_path = image
        main()

