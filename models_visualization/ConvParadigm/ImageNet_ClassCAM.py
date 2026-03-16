import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.ConvParadigm import ResNet18
from PIL import Image
from torchvision import transforms
import os

# ==========================================
# 配置区域
# ==========================================
PIXEL_SIZE = 16  # 像素块大小 (数值越小块越密集，建议 8-16)
COLOR_STYLE = 'VIRIDIS'  # 风格: 'VIRIDIS' (蓝紫->黄) 或 'JET' (蓝->红)
# ==========================================

# 加载模型
model = ResNet18(num_classes=100, channels=3)
checkpoint = torch.load("./resnet18_mini_epoch105.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 注册多个中间层hook
feature_maps = {}
gradients = {}


def save_forward(name):
    def forward_hook(module, input, output):
        feature_maps[name] = output

    return forward_hook


def save_backward(name):
    def backward_hook(module, grad_in, grad_out):
        gradients[name] = grad_out[0]

    return backward_hook


for stage in ["layer1", "layer2", "layer3", "layer4"]:
    layer = getattr(model, stage)
    layer.register_forward_hook(save_forward(stage))
    layer.register_backward_hook(save_backward(stage))

# 预处理图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
data_path = r"D:\datasets\dataset\MINI_ImageNet\mini-imagenet\images"
image_name = "n0153282900000016.jpg"
image_name = "n0924646400000548.jpg"
img_path = os.path.join(data_path, image_name)

if not os.path.exists(img_path):
    print(f"Error: Image not found at {img_path}")
    exit()

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)
input_tensor.requires_grad_()  # 关键：允许求导

# 前向传播
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
model.zero_grad()
output[0, pred_class].backward()  # 指定类别反向传播


# =========================================================================
# 修改后的可视化函数：像素块风格 + Viridis 色彩
# =========================================================================
def generate_pixelated_cam(stage_name, imgPIL, pixel_size=16):
    fmap = feature_maps[stage_name]  # [1, C, H, W]
    grad = gradients[stage_name]  # [1, C, H, W]

    # 1. 计算 Grad-CAM 权重 (GAP)
    weights = grad.mean(dim=(2, 3), keepdim=True)

    # 2. 加权求和并 ReLU
    cam = F.relu((weights * fmap).sum(dim=1)).squeeze()
    cam = cam.detach().cpu().numpy()

    # 3. 归一化 [0, 1]
    cam = (cam - cam.min()) / (cam.max() + 1e-6)

    # -----------------------------------------------------------
    # 核心修改：两步法生成像素块
    # -----------------------------------------------------------
    # Step A: 缩放到目标网格大小 (例如 224 // 16 = 14x14)
    # 使用 Cubic 插值保证激活值平滑过渡，避免生硬
    target_h = imgPIL.height // pixel_size
    target_w = imgPIL.width // pixel_size
    cam_dense = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    cam_dense = np.clip(cam_dense, 0, 1)  # 修正插值过冲

    # Step B: 放大回原图大小，使用【最近邻插值】产生方块效果
    cam_pixelated = cv2.resize(cam_dense, (imgPIL.width, imgPIL.height), interpolation=cv2.INTER_NEAREST)

    # -----------------------------------------------------------
    # 色彩映射与混合
    # -----------------------------------------------------------
    # 转为 uint8 以应用 colormap
    cam_uint8 = (cam_pixelated * 255).astype(np.uint8)

    # 应用色彩风格
    if COLOR_STYLE == 'VIRIDIS':
        colored_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_VIRIDIS)
    elif COLOR_STYLE == 'JET':
        colored_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    else:
        colored_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_VIRIDIS)

    # 准备背景 (暗调处理，模拟黑背景辉光效果)
    img_np = np.array(imgPIL)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    dark_bg = (img_bgr * 0.4).astype(np.uint8)  # 背景亮度压低

    # 计算 Alpha 通道 (强度决定透明度)
    alpha = cam_pixelated[:, :, np.newaxis]
    alpha = alpha ** 1.2  # Gamma 矫正，让低激活区域更透明

    # 混合: 背景(暗) * (1-alpha) + 前景(亮色块) * alpha
    bg_part = dark_bg.astype(np.float32) * (1 - alpha)
    fg_part = colored_cam.astype(np.float32) * alpha

    overlay = (bg_part + fg_part).astype(np.uint8)

    return overlay


# ==========================================
# 保存结果
# ==========================================
save_dir = f"grad_cam_pixel_vis_{image_name.split('.')[0]}"
os.makedirs(save_dir, exist_ok=True)

print(f"Processing image: {image_name}")
print(f"Style: {COLOR_STYLE}, Pixel Size: {PIXEL_SIZE}")

for stage in ["layer1", "layer2", "layer3", "layer4"]:
    # 生成图
    vis_img = generate_pixelated_cam(stage, img, pixel_size=PIXEL_SIZE)

    # 保存
    save_path = os.path.join(save_dir, f"{stage}_pixelated_cam.png")
    cv2.imwrite(save_path, vis_img)
    print(f"[✔] Saved Pixelated Grad-CAM for {stage} → {save_path}")

print("All done.")