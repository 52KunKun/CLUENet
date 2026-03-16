import torch

from models.SquenceParadigm import vit
import torchvision
import os
from einops import rearrange, repeat
from torchvision.utils import draw_segmentation_masks

# 建议类与映射标签
dataset = torchvision.datasets.CIFAR100(root=r'C:\Users\13779\Desktop\dataset\CIFAR100', train=True, download=True)
class_names = dataset.classes  # ['apple', 'aquarium_fish', ..., 'worm']

# 构建 object_categories 映射（下标为 index，值为类名）
object_categories = ["" for _ in range(100)]
for idx, name in enumerate(class_names):
    object_categories[idx] = name   # 索引---类名映射
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
attention_maps = []

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


# hook函数
def get_attention_hook(self, x, output):
    global attention_maps
    x = x[0]
    x = self.norm(x)
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attn = self.attend(dots)  # [b h n n]
    b,h,n,n = attn.shape
    attn = attn[:,:,0,1:].reshape(b,h,32//4,32//4)  # 只要第一个位置
    attention_maps.append(attn.detach().cpu())

def infer(layer_idx,head):
    # 注册到某一层，比如第0层的 Attention 模块
    model = vit.ViT_mini_cifar(num_classes=100)
    checkpoint = torch.load("./vit_mini_cifar_CIFAR100_epoch105.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    attn_module = model.transformer.layers[layer_idx][0]  # 注意力模块
    # 加入钩子函数
    attn_module.register_forward_hook(get_attention_hook)
    model.eval()

    # 预处理图像，并保存原始图像用于可视化
    image, raw_image = _preprocess(1)
    image = image.unsqueeze(dim=0)  # 加入批次维度
    out = model(image)  # 前向传播
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
    print(f"this image true class is {object_categories[dataset[1][1]]}")

    image_name = f"trainset[{1}]"
    os.makedirs(f"images_CIFAR100/vit_cifar/{image_name}", exist_ok=True)

    # 扩大图像
    from PIL import Image
    h,w = raw_image.size
    h = h*8
    w = w*8
    img = raw_image.resize((h,w), Image.BILINEAR)  # 默认使用双线性插值

    from torchvision.transforms.functional import pil_to_tensor
    img_tensor = pil_to_tensor(img)  # 保持为 uint8，不归一化

    from torch.nn import functional as F
    import random
    import cv2
    import torchvision.transforms.functional as TransF
    import numpy as np
    global attention_maps

    mask = attention_maps[0][:,head,:].unsqueeze(0)  # 只要分类标记的 [1,h,w]
    mask = (mask > mask.mean()).float() # 大于均值的注意力
    mask = F.interpolate(mask, (h, w))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    # randomly selected some good colors.
    colors = ["green", "brown", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]  # deepskyblue
    if mask.shape[0] <= len(colors):
        colors = colors[:mask.shape[0]]
    else:
        colors = (colors * (mask.shape[0] // 16 + 1))[:mask.shape[0]]
        random.seed(123)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img_tensor, masks=mask, alpha=1.0, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    # 并保存原始图像
    save_path = f"images_CIFAR100/vit_cifar/{image_name}/block{layer_idx}_head{head}.png"
    save_path2 = f"images_CIFAR100/vit_cifar/{image_name}/original_image.png"
    cv2.imwrite(save_path, img_with_masks)  # 语义图像
    # 复制该图片
    img.save(save_path2, format="PNG")
    print(f"==> Generated image is saved to: {save_path}")

if __name__ == "__main__":
    for i in range(8):
        for j in range(16):
            infer(layer_idx=i,head=j)
