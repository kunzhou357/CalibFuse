# Adapted from Restormer: https://github.com/swz30/Restormer
# Copyright (c) 2022 Syed Waqas Zamir and contributors.
# MIT license; see ../licenses/Restormer-MIT.txt.

"""Restormer 基础模块（改自上游 Restormer，MIT 许可证）。

提供 CalibFuse 网络复用的基础组件：
- 通道维 LayerNorm 的两种变体（无偏置 / 带偏置）；
- 门控深度卷积前馈网络（GDFN, Gated-Dconv Feed-Forward Network）；
- 多头转置自注意力（MDTA, Multi-DConv Head Transposed Attention）；
- 标准 TransformerBlock（Pre-Norm 结构）；
- 3x3 重叠 patch 嵌入（OverlapPatchEmbed）；
- PixelShuffle / PixelUnshuffle 的 2 倍上/下采样模块。

与上游实现的主要差异（本项目适配）：无。本文件仅添加中文注释，
结构保持与上游一致，便于对照原仓库。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


def to_3d(x):
    """形状转换：(B, C, H, W) -> (B, H*W, C)。"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """形状转换：(B, H*W, C) -> (B, C, H, W)，需给定 h、w。"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """通道维 LayerNorm（无均值中心化、无偏置），只做方差缩放。

    对最后一维（通道）计算方差，用 1/sqrt(var+eps) 缩放输入，
    仅有可学习的逐通道缩放 weight。去掉了均值项，参数更少，
    在特征已被归一化的场景（如 CalibFuse 的 RMS 特征）中足够。
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
        # 通道维方差（有偏估计），加 eps 防除零
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """通道维 LayerNorm（带可学习 scale 与 shift）。

    标准公式：(x - mean) / sqrt(var + eps) * weight + bias。
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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """通道维 LayerNorm 封装：负责 4D <-> 3D 的形状重排。

    统一入口：按 ``LayerNorm_type`` 选择 BiasFree 或 WithBias 变体；
    内部把 (B, C, H, W) 重排为 (B, H*W, C) 计算统计量，再还原形状。
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## 门控深度卷积前馈网络（GDFN）
class FeedForward(nn.Module):
    """门控 Dconv 前馈网络（GDFN）。

    结构：1x1 卷积升维到 2*hidden -> 3x3 深度卷积 ->
    通道一分为二（x1, x2），GELU(x1) * x2 做门控 -> 1x1 卷积降回 dim。
    门控机制让前馈网络具备一定的空间选择性。
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # 深度卷积后按通道均分为两路，一路过 GELU 做门控
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## 多头转置自注意力（MDTA）
class Attention(nn.Module):
    """多头 DConv 转置自注意力（MDTA）。

    核心思想：注意力在"通道"维度上计算（生成 (C, C) 的注意力矩阵，
    而非 (HW, HW)），复杂度对空间分辨率是线性的，
    这是 Restormer 能处理高分辨率图像的关键。

    流程：1x1 投影出 QKV -> 3x3 深度卷积（引入局部上下文）->
    按头重排 -> 通道维 L2 归一化（余弦注意力）-> 温度缩放 softmax ->
    加权求和 -> 重排回 4D -> 1x1 输出投影。
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 逐头可学习温度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # (B, C, H, W) -> (B, head, C/head, HW)：通道维成为注意力维度
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 通道维 L2 归一化使注意力成为余弦相似度矩阵，
        # 对特征幅度不敏感（与 CalibFuse 的 RMS 归一化理念一致）
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 块：LN -> MDTA -> 残差；LN -> GDFN -> 残差。"""

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 先归一化再进子层（Pre-Norm），残差直连保持梯度流畅
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


## 3x3 重叠 patch 嵌入
class OverlapPatchEmbed(nn.Module):
    """3x3 卷积的重叠 patch 嵌入（stride=1，保持分辨率）。"""

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


## 分辨率变换模块
class Downsample(nn.Module):
    """2 倍下采样：3x3 卷积把通道减半 + PixelUnshuffle(2)。

    通道变化：C -> C/2，再经 PixelUnshuffle 空间换通道：
    最终 (B, C, H, W) -> (B, 2C, H/2, W/2)。
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """2 倍上采样：3x3 卷积把通道翻倍 + PixelShuffle(2)。

    通道变化：C -> 2C，再经 PixelShuffle 通道换空间：
    最终 (B, C, H, W) -> (B, C/2, 2H, 2W)。
    （CalibFuse 主网络解码用双线性插值，本模块保留自上游以备复用。）
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
