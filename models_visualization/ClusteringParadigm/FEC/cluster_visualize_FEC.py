# --------------------------------------------------------
# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize_CLUAmodel.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --num_clu {number of clusters}
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
import random
from torchvision.utils import draw_segmentation_masks
from torch_scatter import scatter_sum
from einops import rearrange
from sklearn.cluster import KMeans
import json
from models import CLUA_past
from PIL import Image
import shutil
# 忽略警告信息
# 获取ImageNet类别
from models.ClusteringParadigm import fec_small
# 建立idx-->catacaroy_str的映射表（列表即可）
with open(r"C:\Users\13779\Desktop\CV\project\remote_control_autodl\PaperProject\test_on_miniDataset\mini_imagenet\classes_name.json","r") as f:
    name_to_idx = json.load(f)
object_categories = ["" for i in range(1000)]
for k,v in name_to_idx.items():
    object_categories[int(v[0])] = v[1]  # k是字符串索引序号，v是一个列表，第二个元素放置的是字符串含义
# 输入参数Arugument允许在执行python文件指定参数
parser = argparse.ArgumentParser(description='mycluster model visualization')
# --image代表输入图像路径
parser.add_argument('--data_path', type=str, default=r'D:\datasets\dataset\MINI_ImageNet\mini-imagenet\images', help='path to imageFolder')
parser.add_argument('--image', type=str, default="n0153282900000016.jpg", help='path to image under the data_path')
# --shape表示图像的形状
parser.add_argument('--shape', type=int, default=224, help='image size')
# 指定模型
parser.add_argument('--model', default='FEC_small', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--num-classes', type=int, default=100, help='the number of images classes')
# 重塑图像大小
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
# --checkpoint显然指定的预训练模型文件
parser.add_argument('--checkpoint', type=str, default="./FEC_mini_epoch105.pth", metavar='PATH', help='path to pretrained checkpoint')
# --alpha
parser.add_argument('--alpha', default=1.0, type=float, help='Transparent, 0-1')
# Note that FEC only results in 49 clusters in the final stage so that we have to adopt KMeans for easier inspection (see the second paragraph in Sec. 5.2).
# FEC may be sensitive when num_clu is a relatively small value, due to the KMeans algorithms.
# 这里说明FEC最后有49个簇类，如果要指定更小规模的聚类需要进一步使用K-Means  建议使用3、4、10（至少为3）
parser.add_argument('--model_num_clu',
                    type=int,
                    nargs='+',    # 关键修改：'+' 表示接受 1 个或多个值，并存为列表
                    default=[3],  # 默认值也建议改成列表格式，保持类型统一
                    help='number of clusters')
args = parser.parse_args()

# Preprocessing
def _preprocess(image_path):
    # 该方法可以预处理图像的性质
    # 调整image_path
    row_image = Image.open(image_path)
    # 这个增强容易可视化，但是可能会出现预测不准确的情况，因为这与常规的验证集增强模式不同
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
    def _check(mask, k=224 * 224):
        total_lst = []
        for v in mask.values():
            total_lst.extend(v)
        total_set = set(total_lst)
        assert len(total_set) == k

    mask12 = {i: [] for i in range(56 * 56)}
    idx = 0
    # 索引是56*56
    for i in range(0, 224, 4):
        for j in range(0, 224, 4):
            # the clustering is based on 4x4 pixel patch (after standard conv)
            mask12[idx] = [i * 224 + j, i * 224 + j + 1, i * 224 + j + 2, i * 224 + j + 3,
                           (i + 1) * 224 + j, (i + 1) * 224 + j + 1, (i + 1) * 224 + j + 2, (i + 1) * 224 + j + 3,
                           (i + 2) * 224 + j, (i + 2) * 224 + j + 1, (i + 2) * 224 + j + 2, (i + 2) * 224 + j + 3,
                           (i + 3) * 224 + j, (i + 3) * 224 + j + 1, (i + 3) * 224 + j + 2, (i + 3) * 224 + j + 3,
                           ]
            idx += 1

    mask23 = {i: [] for i in range(28 * 28)}
    for i, j in enumerate(mask_layers[0][0]):
        mask23[j].extend(mask12[i])

    mask34 = {i: [] for i in range(14 * 14)}
    for i, j in enumerate(mask_layers[1][0]):
        mask34[j].extend(mask23[i])

    mask45 = {i: [] for i in range(7 * 7)}
    for i, j in enumerate(mask_layers[2][0]):
        mask45[j].extend(mask34[i])

    # _check(mask45)
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

from compute_cluster_importance import *
@torch.no_grad()
def infer(model, img_path, num_k=None):
    model.cuda()
    model.eval()    # 测试阶段
    device = next(model.parameters()).device
    global mask_layers  # forward hook，可以获取中间层的信息
    mask_layers = []
    # 预处理图像，并保存原始图像用于可视化
    image, raw_image = _preprocess(img_path)
    image = image.unsqueeze(dim=0)
    # 转为cuda形式
    out = model(image.to(device))  # 前向传播
    if type(out) is tuple: out = out[0]  # 最后logits分数
    # 打印输出概率最合适的类别
    possibility = torch.softmax(out, dim=1).max()
    # 得到最后最合适的类别
    value, index = torch.max(out, dim=1)
    print(f'top1 ==> Prediction is: {object_categories[index]} possibility: {possibility * 100:.3f}%')
    possibility_top3,index_top3 = torch.topk(torch.softmax(out,dim=1).squeeze(),k=3)
    # 遍历每个样本
    print(f"top3 ==> Top 3 Predictions:")
    for i in range(3):  # 输出前三个预测
        # 获取预测的类别名称
        predicted_class = object_categories[index_top3[i].item()]
        # 获取对应类别的概率
        probability = possibility_top3[i].item() * 100  # 转换为百分比
        print(f"    {i + 1}. {predicted_class} - {probability:.3f}%")
    image_class_idx,image_class_name = find_classes_image(args.image)
    print(f"this image true class is {image_class_name}")
    isModelTrue = True if index == image_class_idx else False
    string_t = "right" if isModelTrue else "wrong"
    image_name = os.path.basename(img_path).split(".")[0]
    os.makedirs(f"images/cluster_vis/{args.model}/{image_name}_{string_t}", exist_ok=True)

    from torchvision.io import read_image
    img = read_image(img_path)
    masks = []

    # mask_layers捕获了网络层中的每个像素点指定的簇分配索引
    if args.model_num_clu:
        for k in args.model_num_clu:
            # 这里的 k 就是具体的聚类数（例如 3, 5, ...）
            mask = aggregate_masks(mask_layers, k)
            masks.append(mask)
    # mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    # mask = mask.squeeze(dim=0)
    # mask = mask > 0.5
    max_cluster_num = 0
    # 先进行排序
    for i in range(len(masks)):
        mask = masks[i]
        mask = compute_cluster_contribution_combined_parallel(mask_MHW=mask, input_image_224=image.to(device), model=model,
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
        max_cluster_num = max(max_cluster_num, mask.shape[0])
        masks[i] = mask

    # randomly selected some good colors.（把显著区分的五个颜色放在一起以免混淆）
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
    if max_cluster_num <= len(colors):
        colors = colors[:max_cluster_num]
    else:
        colors = (colors * (max_cluster_num // 16 + 1))[:max_cluster_num]
    index = 0
    for num_clu in args.model_num_clu:
        mask = masks[index]  # 最后一个元素
        index += 1
        img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
        img_with_masks = img_with_masks.detach()
        img_with_masks = TransF.to_pil_image(img_with_masks)
        img_with_masks = np.asarray(img_with_masks)
        img_with_masks = cv2.cvtColor(img_with_masks, cv2.COLOR_RGB2BGR)
        # 并保存原始图像
        save_path = f"images/cluster_vis/{args.model}/{image_name}_{string_t}/all_stages_{num_clu}CLU.png"
        cv2.imwrite(save_path, img_with_masks)
        print(f"==> Generated image is saved to: {save_path}")

    # 复制该图片
    save_path2 = f"images/cluster_vis/{args.model}/{image_name}_{string_t}/original_image.png"
    shutil.copy(img_path, save_path2)


def main():
    # 使用timm方法创建model
    # model = timm.create_model(model_name=args.model, pretrained=True)
    model = fec_small(num_classes=args.num_classes)
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

    args.image = os.path.join(args.data_path, args.image)
    infer(model, args.image, args.model_num_clu)


if __name__ == '__main__':
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
        "n0153282900000040",
        "n0153282900000179",
        "n0153282900001086",
        "n0174993900000260",
        "n0211034100000702",
    ]
    args.model = "FEC_small"
    args.shape = 224
    args.num_classes = 100
    args.checkpoint = "./FEC_mini_epoch105.pth"
    args.alpha = 1.0
    args.model_num_clu = [3, 5, 8, 10, 14, 18, 24]
    for image in image_path_list:
        if not image.endswith('.jpg'):
            image_path = image + '.jpg'
        else:
            image_path = image
        args.image = image_path
        main()

