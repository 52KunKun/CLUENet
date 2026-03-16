# 簇分数：模糊簇遮住的部分，计算损失得分
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
# 最好的表现
@torch.no_grad()
def compute_cluster_contribution_combined_parallel(mask_MHW, input_image_224, model, true_class, alpha=0.5,
                                                   blur_kernel=11, blur_sigma=5):
    """
    向量化版本：保留原逻辑
    - loss_drop_score: 模糊替代簇后的类别概率下降
    - keep_score: 仅保留簇后类别概率
    """
    device = input_image_224.device
    mask_MHW = mask_MHW.float().to(device)         # [1, M, H, W]
    _, M, H, W = mask_MHW.shape

    # -------- 模糊整图 --------
    def apply_blur(img):
        return TF.gaussian_blur(img, kernel_size=blur_kernel, sigma=blur_sigma)
    blurred_full = apply_blur(input_image_224)     # [1,3,H,W]

    # -------- 原始模型输出概率 --------
    full_logits = model(input_image_224)
    if isinstance(full_logits, tuple):
        full_logits = full_logits[0]
    full_prob = full_logits.softmax(dim=-1)[0, true_class]  # scalar

    # -------- 构造向量化输入 --------
    # mask: [1,M,H,W] -> [M,1,H,W] -> expand 3通道 -> [M,3,H,W]
    mask_expand = mask_MHW.permute(1,0,2,3).expand(-1, 3, -1, -1)  # [M,3,H,W]
    # input: [1,3,H,W] -> [M,3,H,W]
    img_expand = input_image_224.expand(M, 3, H, W)
    blurred_expand = blurred_full.expand(M, 3, H, W)

    # -------- loss_drop_score: 模糊替代簇 --------
    # 全图平均颜色 [3]
    global_mean = input_image_224.mean(dim=(0, 2, 3), keepdim=True)  # [1,3,1,1]
    global_mean_expand = global_mean.expand(M, 3, H, W)  # [M,3,H,W]

    # 掩蔽区域强破坏
    masked_drop = img_expand * (1 - mask_expand) + global_mean_expand * mask_expand  # [M,3,H,W]
    logits_drop = model(masked_drop)
    if isinstance(logits_drop, tuple):
        logits_drop = logits_drop[0]
    prob_drop = logits_drop.softmax(dim=-1)[:, true_class]  # [M]
    loss_drop_scores = full_prob - prob_drop               # [M]

    # -------- keep_score: 仅保留簇 --------
    masked_keep = img_expand * mask_expand + blurred_expand * (1 - mask_expand)
    logits_keep = model(masked_keep)
    if isinstance(logits_keep, tuple):
        logits_keep = logits_keep[0]
    keep_scores = logits_keep.softmax(dim=-1)[:, true_class]  # [M]

    # -------- 综合评分 --------
    combined_score = alpha * loss_drop_scores + (1 - alpha) * keep_scores

    # -------- 排序 --------
    sorted_indices = torch.argsort(combined_score, descending=True)
    mask_sorted = mask_MHW[:, sorted_indices]

    return mask_sorted

