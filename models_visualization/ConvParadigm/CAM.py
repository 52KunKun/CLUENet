import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from models.ConvParadigm import ResNet_cifar  # 按你的路径导入
import torchvision

# --- Step 1: GradCAM 类 ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()
    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1  # 类别标记
        output.backward(gradient=one_hot)   # 反向传播

        # Global Average Pooling on Gradients（可视化梯度
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False)

        cam -= cam.min()
        cam /= cam.max()
        return cam.squeeze().cpu().numpy()

# --- Step 2: 预处理 CIFAR-100 图像 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
])
root=r'C:\Users\13779\Desktop\dataset\CIFAR100'
dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
image_tensor, label = dataset[1]
input_tensor = image_tensor.unsqueeze(0)

# --- Step 3: 构建模型并载入权重 ---
model = ResNet_cifar.ResNet_cifar(num_classes=100)
model.load_state_dict(torch.load("./resnet_cifar_CIFAR100_epoch105.pth")["model_state_dict"])  # 替换为你的权重路径
model.eval()

# --- Step 4: 初始化 GradCAM（以最后一个 Bottleneck 为例）---
# --- Step 4-6: 遍历四个 layer 的最后一个 Bottleneck，逐一生成 GradCAM 热力图 ---
target_layers = {
    "layer1": model.layer1[-1],
    "layer2": model.layer2[-1],
    "layer3": model.layer3[-1],
    "layer4": model.layer4[-1],
}

for name, layer in target_layers.items():
    cam_extractor = GradCAM(model, layer)
    cam = cam_extractor.generate_cam(input_tensor)
    cam_resized = cv2.resize(cam, (256, 256))

    # 保存纯热力图（无坐标轴、无温度条）
    plt.figure(figsize=(4, 4))
    plt.imshow(cam_resized, cmap='jet')
    plt.axis('off')
    plt.savefig(f"cam_heatmap_{name}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# --- Step 5: 生成并显示可视化 ---
cam = cam_extractor.generate_cam(input_tensor)
raw_image = dataset.data[1]  # 这是 numpy 格式
raw_image = Image.fromarray(raw_image).resize((256, 256))
cam_resized = cv2.resize(cam, (256, 256))

# 热力图叠加
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = heatmap + np.float32(raw_image) / 255
overlay = overlay / np.max(overlay)

# --- Step 6: 保存或展示结果 ---
# --- Step 6: 保存或展示结果 ---
# --- Step 6: 仅保存热力图并显示温度计和坐标轴 ---
# --- Step 6: 仅保存 GradCAM 热力图（无坐标轴、无颜色条） ---
plt.figure(figsize=(4, 4))
plt.imshow(cam_resized, cmap='jet')
plt.axis('off')  # 去除坐标轴
plt.savefig("cam_heatmap_only.png", bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()

