# Our model is build upon "Image as Set of Points", ICLR23. https://github.com/ma-xu/Context-Cluster/blob/main/models/context_cluster.py
# Thanks the authors for their impressive work!
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple
from einops import rearrange
from torch_scatter import scatter_sum
import torch.nn.functional as F
import math
import numpy as np

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    # print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    # print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}

class LayerNorm(nn.LayerNorm):
    def __init__(self,num_channels,**kwargs):
        super(LayerNorm, self).__init__(num_channels,**kwargs)  
    def forward(self, input):
        # 变换形状后，然后进入LayerNorm再变换形状
        B,C,H,W = input.shape
        input = rearrange(input,"B C H W -> B (H W) C")
        input = super().forward(input)
        return rearrange(input,"B (H W) C -> B C H W",H=H,W=W)
    def cls_init_weights(self, m):  # 对于分类任务的初始化
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.GroupNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4,in_chans=3, embed_dim=768,padding=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        if padding is None:
            padding = (patch_size[0] // 2, patch_size[1] // 2)
        if isinstance(padding, int):
            padding = (padding,padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=padding)
        self.norm = LayerNorm(embed_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.GroupNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class ClusterPool(nn.Module):  
    """
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self,patch_size=3,stride=4, in_chans=5, embed_dim=64,norm_layer=LayerNorm,norm2=False):
        super().__init__()
        self.norm1 = norm_layer(in_chans)
        self.norm2 = norm_layer(embed_dim) if norm2 else nn.Identity() 
        self.stride = stride
        # CRITICAL: Feature sharing is mandatory here; otherwise, the Structural Awareness Layer 
        # will fail to converge/learn. 
        # We implement this by transforming the operation into a channel-wise MLP layer.
        self.conv_f = nn.Conv2d(in_chans, in_chans, kernel_size=1)  # for value
        self.conv_v = nn.Sequential(nn.Conv2d(in_chans,in_chans,kernel_size=1),
                                    nn.GELU(),
                                    nn.Conv2d(in_chans,embed_dim,kernel_size=1)) # for value

        self.conv_skip = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=(patch_size-1)//2, stride=stride)  # for skip connection

    def forward(self, x):
        identity = self.conv_skip(x)
        value = self.conv_f(x)  
        x = value  # sharing
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
        out = rearrange(scatter_sum(value2, sim_max_idx, dim=0, dim_size=b * M), '(b m) c -> b m c', b=b,m=M)  # Different from CoC's implementation "(value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2)", we use scatter_sum to avoid OOM.
        out = (out + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)
        out = rearrange(out, 'b (w h) c -> b c w h', w=ww, h=hh)
        out = identity + self.norm2(self.conv_v(self.norm1(out)))
        return out


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor,x1_norm=False,x2_norm=False):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,M,D]
    :param x2: [B,N,D]
    :return: similarity matrix [B,M,N]
    """
    if not x1_norm:
        x1 = F.normalize(x1, dim=-1)
    if not x2_norm:
        x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

# Grouped Linear Layer
class GroupLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.in_per_group = in_features
        self.out_per_group = out_features
        # Weight shape: [Groups, Out_per_group, In_per_group]
        self.weight = nn.Parameter(torch.Tensor(num_groups, self.out_per_group, self.in_per_group))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_groups, self.out_per_group))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Weight: [G, Cout_g, Cin_g]
        # x: [B, N, G, Cin_g]
        # → Output: [B, N, G, Cout_g]
        
        # Batch matrix multiplication via einsum for grouped processing
        out = torch.einsum("bngc,goc->bngo", x, self.weight)
        
        if self.bias is not None:
            # Broadcase bias: [1, 1, G, Cout_g]
            out = out + self.bias[None, None, :, :]  
            
        return out


class MultiHeadGate(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim // 2

        self.fc1 = GroupLinear(2 * dim, self.hidden_dim, num_heads)
        self.act = nn.GELU()
        self.fc2 = GroupLinear(self.hidden_dim, 1, num_heads)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cluster_centers, centers_feature):
        """
        cluster_centers: [B, N, H, D]
        centers_feature: [B, N, H, D]
        """
        fused = torch.cat([cluster_centers, centers_feature], dim=-1)  # [B, N, H, 2D]
        # fused: [B, N, H, 2D] →  Each head is processed independently.
        x = self.fc1(fused)  # [B, N, H, hidden_dim]
        x = self.act(x)
        x = self.fc2(x)  # [B, N, H, 1]
        gate = self.sigmoid(x)  # [B, N, H, 1]
        return gate

from flash_attn import flash_attn_func

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, cluster_w=7, cluster_h=7, heads=4, head_dim=24, sim_share=False, min_alpha=0.01):
        """
        :param dim: input channel number
        :param out_dim: output channel number
        :param cluster_h: height of the cluster grid (sqrt of proposals)
        :param cluster_w: width of the cluster grid (sqrt of proposals)
        :param heads: number of attention heads
        :param head_dim: dimension per head
        :param min_alpha: lower bound for the scale factor to ensure numerical stability
        """
        super().__init__()
        self.num_heads = heads
        self.head_dim = head_dim
        self.cw = cluster_w
        self.ch = cluster_h
        self.sim_share = sim_share

        # Default: bias is required for linear projections
        # MLP mapping from Value space -> Original space -> Query space
        self.q = nn.Linear(heads*head_dim, heads*head_dim) if not self.sim_share else None
        self.k = nn.Conv2d(dim, heads*head_dim, 1, 1, 0)
        self.v = nn.Conv2d(dim, heads*head_dim, 1, 1, 0)
        self.proj = nn.Conv2d(heads*head_dim, out_dim, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d((cluster_h, cluster_w))
        
        # Adjustable fusion mechanism: Content-aware gate generation
        self.gate = MultiHeadGate(head_dim, heads)
        
        # Trainable sigmoid gate parameter
        # Initialized to log(2.13) to align Sigmoid behavior with Softmax when input is within [-1, 1]
        self.attn_alpha = nn.Parameter(torch.tensor(np.log(2.13)), requires_grad=True) 
        self.min_alpha = min_alpha

        # Scalable similarity parameters
        self.sim_alpha = nn.Parameter(torch.ones(1), requires_grad=True) if not self.sim_share else None
        self.sim_beta = nn.Parameter(torch.zeros(1), requires_grad=True) if not self.sim_share else None

    def forward(self, x, sim=None):  # x shape: [b, c, h, w]
        b, c, h, w = x.shape  # Input resolution might vary
        
        # Project input to Key space
        K = self.k(x)
        K_centers = self.pool(K)  # Generate cluster centers via pooling
        feature = self.v(x)  # Capture Value features
        centers_feature = self.pool(feature)

        # Reshape 2D feature maps into sequences for attention computation
        K = rearrange(K, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
        K_centers = rearrange(K_centers, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
        feature = rearrange(feature, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads).type(torch.half).contiguous()

        # Numerical Stability Note: 
        # For small models (<10M), min_alpha can be set lower (e.g., 0.001).
        # For larger models (>10M), keep it around 0.01 to ensure training safety.
        min_alpha = torch.log(torch.tensor(self.min_alpha, device=self.attn_alpha.device, dtype=self.attn_alpha.dtype))
        alpha = torch.clamp(self.attn_alpha, min=min_alpha)
        scale = 1 / alpha.exp()

        K_norm = F.normalize(K, dim=-1).contiguous()
        K_centers = (F.normalize(K_centers, dim=-1) * scale).type(torch.half).contiguous()
        
        # Perform cluster center update via Flash Attention
        # Note: Standard Softmax-based K-means may lead to low learning efficiency
        cluster_centers = flash_attn_func(K_centers, K_norm.type(torch.half), feature, softmax_scale=1).type(torch.float)
        
        centers_feature = rearrange(centers_feature, 'b (hd c) h w -> b (h w) hd c', hd=self.num_heads)
        
        # Residual connection on cluster features: Use a gated mechanism to control information flow
        gate = self.gate(cluster_centers, centers_feature)
        cluster_centers = (1 - gate) * cluster_centers + gate * centers_feature  # Gradients propagate through residual
        
        # Prepare for feature dispatching (Hard assignment mode)
        cluster_centers = rearrange(cluster_centers, 'b n hd d -> (b hd) n d')

        if sim is None or not self.sim_share:
            # Map back to Query space for similarity computation
            Q_centers = rearrange(self.q(rearrange(cluster_centers, '(b hd) n d -> b n (hd d)', hd=self.num_heads)),
                                  'b n (hd d) -> (b hd) n d', hd=self.num_heads)
            K_norm = rearrange(K_norm, 'b n hd d -> (b hd) n d')
            
            # Compute Hard Assignment similarity matrix
            sim = torch.sigmoid(
                self.sim_beta +
                self.sim_alpha * pairwise_cos_sim(
                    Q_centers, K_norm, x1_norm=False, x2_norm=True
                )
            )  # Resulting shape: [B*hd, M, N]
            
            # Apply Hard Mask: Only the center with the maximum similarity is activated
            sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
            mask = torch.zeros_like(sim)  # Binary mask
            mask.scatter_(1, sim_max_idx, 1.)
            sim = sim * mask  # sim can be used for visualization or passed to next block
            
        # Feature Dispatching: Aggregate cluster centers back to pixel locations
        out = sim.transpose(-1, -2) @ cluster_centers
        
        # Restore multi-head output to original spatial resolution
        out = rearrange(out, '(b hd) (h w) d -> b (hd d) h w', hd=self.num_heads, h=h, w=w)
        out = self.proj(out)
        
        return out, sim  # Similarity matrix can be bypassed in certain configurations

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)  # B D H W -> B H W D
        x = x.permute(0,2,3,1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))  # B C H W  B H W C
        x = self.dwconv(x.permute(0,3,1,2))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 heads=4, head_dim=24,cluster_w=7,cluster_h=7,sim_share=False,min_alpha=0.01):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim,
                                   heads=heads,head_dim=head_dim,
                                   cluster_h=cluster_h,cluster_w=cluster_w,sim_share=sim_share,min_alpha=min_alpha)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale: 
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x,sim=None):
        if self.use_layer_scale:
            skip_connect = x
            x,sim = self.token_mixer(self.norm1(x),sim)
            x = skip_connect+self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)*x)
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            skip_connect = x
            x, sim = self.token_mixer(self.norm1(x), sim)
            x = skip_connect + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,sim  # used to share for module behind this

# 一个阶段内，块基本不变的
class basic_blocks(nn.Module):
    def __init__(self,dim, index, layers,cluster_w=7,cluster_h=7,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=LayerNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                heads=4, head_dim=24,sim_share=False,min_alpha=0.01):
        super(basic_blocks, self).__init__()
        blocks = []
        for block_idx in range(layers[index]):
            block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
            blocks.append(ClusterBlock(
                dim, cluster_w=cluster_w,cluster_h=cluster_h,mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                heads=heads, head_dim=head_dim,sim_share=sim_share if block_idx >0 else False,min_alpha=min_alpha)) 
                # The first block must compute its own similarity matrix to initialize the cluster assignments;
        self.blocks = nn.Sequential(*blocks)

    def forward(self,x):
        sim = None
        for block in self.blocks:
            x,sim = block(x,sim)   # 循环迭代更新sim
        return x

class PosCNN(nn.Module):
    def __init__(self,embed_dim=768, s=1):
        assert s==1,'stride must be 1'
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x):
        x = self.proj(x) + x
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class CLUENet(nn.Module):
    """
    CLUster attEntion Network (CLUENet), the main class of our model.
    --layers: List [x,x,x,x], specifying the number of blocks for the 4 stages.
    --embed_dims, --mlp_ratios: Embedding dimensions and MLP expansion ratios.
    --downsamples: Flags to indicate whether to apply downsampling between stages.
    --norm_layer, --act_layer: Definitions for normalization and activation types.
    --num_classes: Number of target classes for image classification.
    --in_patch_size, --in_stride, --in_pad: Parameters for the initial patch embedding.
    --down_patch_size, --down_stride, --down_pad: Parameters for downsampling (patch embedding).
    --fork_feat: Whether to output features from all 4 stages (for dense prediction tasks).
    --init_cfg, --pretrained: Configurations for loading pretrained weights (mmdet/mmseg compatible).
    """

    def __init__(self, layers, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], downsamples=[True, True, True, True],
                 norm_layer=LayerNorm, pool_norm2=False,
                 act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0, # Non-overlapping patch embedding
                 down_patch_size=3, down_stride=2,
                 drop_rate=0., drop_path_rate=0.,
                 # Layer scale is not applied by default when using LayerNorm
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 # img_size is required for positional information; it does not affect model parameters
                 fork_feat=False,  # If True, returns features from four stages instead of classification logits
                 init_cfg=None,
                 pretrained=None,
                 heads=[1, 2, 5, 8], head_dim=[16, 16, 32, 32],
                 cluster_w=[7, 7, 7, 7], cluster_h=[7, 7, 7, 7], img_size=(224, 224), 
                 sim_share=True, min_alpha=0.00001, **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        
        # Initial patch embedding with coordinate bias integration; 4x4 downsampling
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])
        
        # Learnable positional embedding via depth-wise convolution
        self.pos_embed = PosCNN(embed_dim=embed_dims[0], s=1)
        
        if norm_layer == LayerNorm:
            use_layer_scale = False # Layer scale redundant with LayerNorm
            
        img_size = (img_size[0] // in_stride, img_size[1] // in_stride)

        # Set the main backbone architecture
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 heads=heads[i], head_dim=head_dim[i],
                                 cluster_h=cluster_h[i], cluster_w=cluster_w[i], 
                                 sim_share=sim_share, min_alpha=min_alpha)
            network.append(stage)
            
            if i >= len(layers) - 1:  # Only 3 downsampling stages needed for a 4-stage loop
                break
                
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(ClusterPool(patch_size=down_patch_size,
                                         stride=down_stride, in_chans=embed_dims[i], 
                                         embed_dim=embed_dims[i+1], norm_layer=norm_layer, 
                                         norm2=pool_norm2))
                img_size = (img_size[0] // down_stride, img_size[1] // down_stride)  # Reduce spatial dimensions
        
        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # Add a normalization layer for each stage output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # For RetinaNet, start_level=1; the first norm layer is bypassed
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classification head: includes a final normalization layer before the linear projection
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        
        # Load pre-trained model weights
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):  # Weight initialization for classification tasks
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Kaiming (He) initialization
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training from scratch.')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'init_cfg must specify `checkpoint` for {self.__class__.__name__}.'
            ckpt_path = self.init_cfg['checkpoint'] if self.init_cfg is not None else pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            _state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

            self.load_state_dict(_state_dict, False)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        # Optimized embedding with coordinate information integration
        _, c, img_w, img_h = x.shape
        # Normalize coordinates to range [-0.5, 0.5] with mean 0
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device) - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        
        # Concatenate spatial coordinates with RGB input followed by non-overlapping patch embedding
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        x = self.pos_embed(x)  # Sum with positional encoding
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        
        if self.fork_feat:
            # Return multi-stage features for downstream tasks (e.g., detection/segmentation)
            return outs
        # Return final stage tokens for classification
        return x

    def forward(self, x):
        # 1. Input embedding with spatial coordinate bias
        x = self.forward_embeddings(x)
        # 2. Forward pass through the backbone network
        x = self.forward_tokens(x)
        
        if self.fork_feat:
            return x
            
        # 3. Final classification head: Normalization followed by Global Average Pooling
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out


def CLUE_mini_CIFAR(pretrained=False, **kwargs):
    layers = [2, 2, 2, 2]
    norm_layer = LayerNorm
    embed_dims = [32, 64, 160, 256]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    # 消融实验对象
    cluster_h = [4, 4, 4, 4]  
    cluster_w = [4, 4, 4, 4]  
    # 降低显存占用的方法
    heads = [1, 2, 5, 8]
    head_dim = [32,32,32,32]
    down_patch_size = 3
    down_stride = 2
    in_patch_size = 3 
    in_stride = 1
    in_pad=1
    drop_path_rate = 0.1
    img_size=(32,32)
    min_alpha = 0.0001
    model = CLUENet(
        layers,img_size=img_size,embed_dims=embed_dims, norm_layer=norm_layer,pool_norm2=True,  
        mlp_ratios=mlp_ratios, downsamples=downsamples, in_patch_size=in_patch_size,in_pad=in_pad,
        in_stride=in_stride,
        down_patch_size=down_patch_size,
        down_stride=down_stride,
        cluster_h=cluster_h, cluster_w=cluster_w, drop_path_rate=drop_path_rate,
        heads=heads, head_dim=head_dim,min_alpha=min_alpha,**kwargs)
    # model.default_cfg = default_cfgs['model_tiny']
    return model

def CLUE_mini(pretrained=False, **kwargs):
    layers = [2, 2, 2, 2]
    norm_layer = LayerNorm
    embed_dims = [32, 64, 160, 256]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    cluster_h = [7, 7, 7, 7] 
    cluster_w = [7, 7, 7, 7]  
    heads = [1, 2, 5, 8]
    head_dim = [32,32,32,32]
    down_patch_size = 3
    drop_path_rate = 0.1
    down_stride = 2
    in_patch_size = 4 
    in_stride = 4
    min_alpha = 0.0001
    model = CLUENet(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,pool_norm2=True,
        mlp_ratios=mlp_ratios, downsamples=downsamples, in_patch_size=in_patch_size,
        in_stride=in_stride,
        down_patch_size=down_patch_size,
        down_stride=down_stride,
        cluster_h=cluster_h, cluster_w=cluster_w, drop_path_rate=drop_path_rate,min_alpha=min_alpha,
        heads=heads, head_dim=head_dim,**kwargs)
    # model.default_cfg = default_cfgs['model_tiny']
    return model

def CLUE_tiny(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = LayerNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    cluster_w = [7, 7, 7, 7]
    cluster_h = [7, 7, 7, 7]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    drop_path_rate = 0.1
    min_alpha = 0.0001
    model = CLUENet(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        cluster_w=cluster_w, cluster_h=cluster_h,
        heads=heads, head_dim=head_dim,drop_path_rate=drop_path_rate,min_alpha=min_alpha,
        **kwargs)
    return model

def CLUE_small(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer = LayerNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    cluster_w = [7, 7, 7, 7]
    cluster_h = [7, 7, 7, 7]
    heads = [4, 4, 8, 8]
    head_dim = [32, 32, 32, 32]
    drop_path_rate = 0.3
    down_patch_size = 3
    down_pad = 1
    min_alpha = 0.0001 
    model = CLUENet(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        cluster_w=cluster_w, cluster_h=cluster_h,
        heads=heads, head_dim=head_dim,drop_path_rate=drop_path_rate,min_alpha=min_alpha,**kwargs)
    return model

def CLUE_base(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = LayerNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    cluster_w = [7, 7, 7, 7]
    cluster_h = [7, 7, 7, 7]
    heads = [6, 6, 12, 12]
    head_dim = [32, 32, 32, 32]
    drop_path_rate = 0.3
    down_patch_size = 3
    down_pad = 1
    min_alpha = 0.0001 
    model = CLUENet(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        cluster_w=cluster_w, cluster_h=cluster_h,
        heads=heads, head_dim=head_dim,drop_path_rate=drop_path_rate,min_alpha=min_alpha,**kwargs)
    return model

def get_flops_ptflops_method(model, input_size):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_size,as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)

if __name__ == '__main__':
    # model = CLUA_mini_CIFAR(num_classes=100)
    model = CLUE_base(num_classes=200)
    model = model.cuda()
    get_flops_ptflops_method(model, (3, 224,224))
    # input = torch.rand(1,3,224,224,device='cuda')
    # output = model(input)
