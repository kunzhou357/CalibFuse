# Adapted from Restormer: https://github.com/swz30/Restormer
# Copyright (c) 2022 Syed Waqas Zamir and contributors.
# MIT license; see ../licenses/Restormer-MIT.txt.

"""Restormer 基础构件（自上游 Restormer 适配，MIT 许可）。

本文件提供 CalibFuse 网络使用的 Transformer 基础模块，全部改编自
Restormer（Zamir et al., CVPR 2022）：

- :class:`LayerNorm`（含 :class:`BiasFree_LayerNorm` / :class:`WithBias_LayerNorm`
  两种变体）——作用于通道维的层归一化；
- :class:`FeedForward` —— 门控深度卷积前馈网络（GDFN）；
- :class:`Attention` —— 多头转置自注意力（MDTA），在通道维而非空间维
  做注意力，复杂度与空间分辨率线性相关；
- :class:`TransformerBlock` —— "归一化→注意力→残差 + 归一化→FFN→残差"
  的标准块，是本项目统一的编码/精炼单元；
- :class:`OverlapPatchEmbed` —— 3x3 重叠 patch 嵌入（stem 用）；
- :class:`Downsample` / :class:`Upsample` —— 像素重排式分辨率变换。

这些构件在 ``nets/fusion.py`` 中通过 ``transformer_block()`` 工厂统一
创建（FFN 扩张 2.0、无偏置、WithBias 归一化）。整个网络共 12 个
Transformer 块。本文件与上游保持功能等价，仅添加了注释。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


def to_3d(x):
    """(B, C, H, W) → (B, H·W, C)。

    Restormer 的注意力/归一化作用于**通道维**，需要先把空间维拍平、
    把通道维放到最后一维以适配 ``nn`` 风格的归一化与注意力计算。
    """
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """(B, H·W, C) → (B, C, H, W)，:func:`to_3d` 的逆变换。"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """无偏置通道 LayerNorm：仅对最后一维（通道）标准化并做缩放。

    与标准 LayerNorm 的差别：不减均值、不加偏置，只除以标准差后乘
    可学习缩放。对图像任务，减均值会改变各通道间的相对响应，
    Restormer 的实验表明无偏置变体在复原任务中更稳。
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 输入约定为 (B, N, C)，沿最后一维（通道）计算方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置通道 LayerNorm：标准化 + 可学习缩放与平移。

    即标准 LayerNorm 施加在通道维上。本项目统一使用该变体
    （``transformer_block`` 传 "WithBias"）。
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 输入约定为 (B, N, C)：沿通道维减均值、除标准差，再仿射
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """通道维 LayerNorm 的对外封装（自动做 4D↔3D 重排）。

    参数:
        dim: 通道数 C。
        LayerNorm_type: ``'BiasFree'`` 或 ``'WithBias'``（默认后者）。
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # 4D → 3D → 归一化 → 4D，重排本身无计算开销
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    """门控深度卷积前馈网络（GDFN）。

    结构：1x1 升维（×2 份）→ 3x3 深度卷积 → 拆成两半做门控
    ``gelu(x1) * x2`` → 1x1 降维。门控机制让前馈分支按内容选择性
    传递信息，比普通 MLP 更适合复原任务。
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 一次卷积同时产出门控的两路特征（hidden*2 通道）
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # 拆成两半：一半过 GELU 作为"内容"，另一半作为"门"
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """多头转置自注意力（MDTA）。

    与标准 ViT 注意力的区别：Q、K 沿**通道维** L2 归一化后计算
    ``(B, head, C, C)`` 的通道相关性矩阵，而不是 ``(HW, HW)`` 的空间
    相关性。因此注意力复杂度为 O(C²·HW)，对空间分辨率线性——
    高分辨率图像复原的关键设计。V 仍按空间维聚合。
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 每头一个可学习温度，控制通道相关性的锐度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重排为 (B, head, head_dim, HW)：通道被拆到 head×head_dim
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Q/K 沿通道维 L2 归一化：相关性退化为余弦相似度，
        # 对特征幅度不敏感，训练更稳
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 转置注意力：(head_dim × HW) @ (HW × head_dim) → (C, C) 通道图
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 用通道注意力聚合 V：(C, C) @ (C, HW) → (C, HW)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Restormer Transformer 块：通道 LN → MDTA → 残差；通道 LN → GDFN → 残差。

    项目内统一经由 ``nets/fusion.transformer_block(channels, heads)``
    构造（扩张 2.0、无偏置、WithBias 归一化）。
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 标准预归一化（pre-norm）残差结构
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    """3x3 重叠 patch 嵌入（即一个带 padding 的 3x3 卷积）。

    相比 ViT 的非重叠 16x16 切块，重叠嵌入在复原任务中保留更多
    高频细节；本项目用作两个模态的 stem（可见光 3 通道、红外 1 通道）。
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



## Resizing modules
class Downsample(nn.Module):
    """2× 下采样：3x3 卷积减半通道 + PixelUnshuffle(2) 空间减半。

    通道数守恒（C → C/2 × 4），信息以重排方式保留，无插值损失。
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """2× 上采样：3x3 卷积加倍通道 + PixelShuffle(2) 空间加倍。

    :class:`Downsample` 的对称逆操作（本项目解码器实际用的是
    双线性 ``upsample`` + 卷积，此类保留自上游）。
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
