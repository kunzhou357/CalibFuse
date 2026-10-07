"""图像 I/O 与常用图像变换。

- :func:`save_tensor`：把张量保存为 8-bit 图像；
- :func:`luminance`：BT.601 加权亮度；
- :func:`spatial_gradient` / :func:`gradient_magnitude`：Sobel 梯度；
- :func:`local_contrast`：基于均值池化的局部对比度；
- :func:`normalize_map`：稳健的"均值 + 3 倍标准差"热图归一化。
"""


from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor


def save_tensor(path: str | Path, image: Tensor) -> None:
    """把 [0,1] 的 (C, H, W) 或 (1, C, H, W) 张量保存为 8-bit 图像。

    流程：detach -> clamp(0,1) -> 转回 CPU -> (H, W, C) ->
    乘 255 后四舍五入取整为 uint8 -> 单通道时去掉通道维 -> 保存。
    自动创建父目录。

    注意：输出为 8-bit PNG（四舍五入量化）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).cpu()
    if array.ndim == 4:
        array = array[0]  # 去掉 batch 维
    # 四舍五入（而非截断）以减小量化偏差
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]  # 灰度图去掉通道维
    Image.fromarray(array).save(path)

def luminance(rgb: Tensor) -> Tensor:
    """BT.601 加权亮度：Y = 0.299 R + 0.587 G + 0.114 B。

    单通道输入直接原样返回（红外路径复用同一函数）。
    返回形状保持 (B, 1, H, W) / (1, H, W)。
    """
    if rgb.shape[1] == 1:
        return rgb
    weights = rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)

def spatial_gradient(x: Tensor) -> tuple[Tensor, Tensor]:
    """3x3 Sobel 梯度 ``(gx, gy)``，reflect 填充避免边界伪影。

    归一化 Sobel 核（除以 8，使单位斜坡的梯度为 1）：
    kx 检测水平方向变化，ky 为其转置检测竖直方向变化。
    输入先经 :func:`luminance` 转灰度。
    """
    gray = luminance(x)
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kx), F.conv2d(padded, ky)

def gradient_magnitude(x: Tensor) -> Tensor:
    """梯度幅值 ``sqrt(gx² + gy²)``（加 1e-8 防止开方下溢）。"""
    gx, gy = spatial_gradient(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def local_contrast(x: Tensor, kernel_size: int = 15) -> Tensor:
    """局部对比度 ``|gray - 均值池化(gray)|``。

    kernel_size（默认 15）的滑动窗口均值作为局部背景，
    差的绝对值即为局部对比度；reflect 填充处理边界。
    """
    gray = luminance(x)
    radius = kernel_size // 2
    local_mean = F.avg_pool2d(F.pad(gray, (radius, radius, radius, radius), mode="reflect"),
                              kernel_size, stride=1)
    return (gray - local_mean).abs()

def normalize_map(x: Tensor) -> Tensor:
    """按样本用"均值 + 3 倍标准差"把热图稳健归一化到 [0, 1]。

    以 mean + 3*std 作为白点（而非 max），避免离群热区把
    整张图压暗；加 1e-6 防除零。
    """
    flat = x.flatten(1)
    scale = flat.mean(1) + 3.0 * flat.std(1, unbiased=False)
    return (x / (scale.view(-1, 1, 1, 1) + 1e-6)).clamp(0.0, 1.0)
