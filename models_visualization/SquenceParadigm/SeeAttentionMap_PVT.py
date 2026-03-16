import os

# 解决 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# --- 请替换为你的模型路径 ---
from models.SquenceParadigm.pvt_v2 import pvt_v2_b0

# -------------------------------------------------------------------------
# 配置区域
# -------------------------------------------------------------------------
# 1. 控制像素块密度：越小越密集 (建议 4 到 8)
PIXEL_SIZE = 16

# 2. 颜色风格选择
# 'VIRIDIS': 暗紫/蓝 -> 绿 -> 黄 (最符合你的描述：蓝黑背景，高亮黄)
# 'JET':     蓝 -> 绿 -> 黄 -> 红 (经典热力图)
COLOR_STYLE = 'VIRIDIS'
COLOR_STYLE = 'JET'

# -------------------------------------------------------------------------
# 全局变量与 Hook
# -------------------------------------------------------------------------
attention_maps = {}
feat_shapes = {}


def compute_attention(stage_name):
    def hook(self, input, output):
        x_input, H, W = input
        feat_shapes[stage_name] = (H, W)

        # PVT Attention 计算逻辑
        B, N, C = x_input.shape
        q = self.q(x_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x_input.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x_input).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x_input.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k = kv[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attention_maps[stage_name] = attn.detach().cpu()

    return hook


# -------------------------------------------------------------------------
# 主程序
# -------------------------------------------------------------------------
if __name__ == '__main__':
    img_path = r"D:\datasets\dataset\MINI_ImageNet\mini-imagenet\images\n0153282900000016.jpg"
    ckpt_path = "pvt_v2_b0_mini_epoch105.pth"

    # 根据风格命名文件夹
    save_dir = f"pvt_{COLOR_STYLE}_pixel_vis_{os.path.basename(img_path).split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载模型
    model = pvt_v2_b0(num_classes=100)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    # 2. 注册 Hook
    for i in range(1, 5):
        try:
            getattr(model, f'block{i}')[-1].attn.register_forward_hook(compute_attention(f"stage{i}"))
        except AttributeError:
            pass

    # 3. 推理
    raw_img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0)
    model(input_tensor)

    print(f">> 开始生成：风格={COLOR_STYLE}, 像素块大小={PIXEL_SIZE}")

    # 4. 可视化循环
    for stage, attn in attention_maps.items():
        attn_avg = attn[:,0,:] # 多头合并
        H_q, W_q = feat_shapes[stage]
        center_idx = (H_q // 2) * W_q + (W_q // 2)
        attn_vector = attn_avg[0, center_idx, :]

        # 恢复形状
        N_k = attn_vector.shape[0]
        side_k = int(N_k ** 0.5)
        heatmap = attn_vector.reshape(side_k, side_k).numpy()

        # A. 归一化 [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # B. 【两步法缩放】保持密集像素块风格
        target_h = raw_img.height // PIXEL_SIZE
        target_w = raw_img.width // PIXEL_SIZE

        # 第一步：Cubic 插值增加密度（平滑过渡）
        heatmap_dense = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        heatmap_dense = np.clip(heatmap_dense, 0, 1)  # 修正过冲

        # 第二步：Nearest 插值变回方块
        heatmap_resized = cv2.resize(heatmap_dense, (raw_img.width, raw_img.height), interpolation=cv2.INTER_NEAREST)

        # C. 【关键】应用色彩映射 (Colormap)
        # 将 0-1 的热力图转为 0-255 的 uint8
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        if COLOR_STYLE == 'VIRIDIS':
            # Viridis: 紫黑 -> 蓝 -> 绿 -> 黄
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        elif COLOR_STYLE == 'JET':
            # Jet: 蓝 -> 绿 -> 黄 -> 红
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # D. 准备背景 (暗调)
        img_np = np.array(raw_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        dark_bg = (img_bgr * 0.4).astype(np.uint8)  # 背景变暗

        # E. 智能混合 (Smart Blending)
        # 我们使用 heatmap 的强度作为 alpha 通道
        # 低注意力区域 -> alpha 接近 0 -> 显示暗背景 (看起来像黑/深蓝)
        # 高注意力区域 -> alpha 接近 1 -> 显示 Colormap 的颜色 (绿/黄)

        alpha = heatmap_resized[:, :, np.newaxis]

        # 对 Alpha 做 Gamma 矫正，让低值更透，高值更实
        # 这样可以去除背景噪音，让高亮块“浮”在空中
        alpha = alpha ** 1.2

        # 混合公式
        # 背景部分 + 前景部分
        bg_part = dark_bg.astype(np.float32) * (1 - alpha)
        fg_part = colored_heatmap.astype(np.float32) * alpha

        overlay = (bg_part + fg_part).astype(np.uint8)

        # F. 保存
        save_path = os.path.join(save_dir, f"{stage}_colormap_{COLOR_STYLE}.png")
        cv2.imwrite(save_path, overlay)
        print(f"[Success] Saved: {save_path}")

    print("全部完成。")