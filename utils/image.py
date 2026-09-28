"""图像 I/O 与常用图像变换工具。

提供入口脚本与评估器共用的基础图像操作：

- :func:`save_tensor`：把网络输出张量存为 8-bit PNG/JPG
  （注意：**量化到 8-bit 是指标协议的一部分**——评估器在
  保存后的 PNG 上计算指标，而非在浮点输出上）；
- :func:`luminance`：RGB → ITU-R BT.601 加权亮度；
- :func:`spatial_gradient` / :func:`gradient_magnitude`：Sobel 梯度；
- :func:`local_contrast`：局部对比度（减去均值池化）；
- :func:`normalize_map`：按"均值+3σ"稳健归一化热力图。
"""


from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor


def save_tensor(path: str | Path, image: Tensor) -> None:
    """把 (C,H,W) 或 (1,C,H,W) 张量保存为 8-bit 图像。

    自动创建父目录；值先 clamp 到 [0,1] 再乘 255 四舍五入量化。
    单通道图会去掉尾部 1 维存为灰度。

    参数:
        path: 输出路径（扩展名决定格式）。
        image: 取值约 [0,1] 的张量；4 维时取第 0 个样本。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).cpu()
    if array.ndim == 4:
        array = array[0]
    # (C,H,W) → (H,W,C)，量化到 uint8；这一步的量化误差会进入
    # calibfuse-metrics-v1 指标协议（指标在存盘后的 PNG 上计算）
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)

def luminance(rgb: Tensor) -> Tensor:
    """计算亮度（灰度）。

    对 (B,3,H,W) 的 RGB 用 BT.601 权重 (0.299, 0.587, 0.114) 加权；
    单通道输入原样返回。

    参数:
        rgb: (B,C,H,W) 张量，C 为 1 或 3。

    返回:
        (B,1,H,W) 亮度图。
    """
    if rgb.shape[1] == 1:
        return rgb
    weights = rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)

def spatial_gradient(x: Tensor) -> tuple[Tensor, Tensor]:
    """用 3x3 Sobel 核计算 (gx, gy) 空间梯度。

    核已除以 8（Sobel 核权重和），输出是"平均差分"意义的梯度；
    边界用 reflect 填充，避免引入假边缘。

    参数:
        x: (B,C,H,W) 图像（先转亮度再求梯度）。

    返回:
        二元组 ``(gx, gy)``，各为 (B,1,H,W)。
    """
    gray = luminance(x)
    # Sobel x 核（含 1/8 归一化），y 核为其转置
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kx), F.conv2d(padded, ky)

def gradient_magnitude(x: Tensor) -> Tensor:
    """梯度幅值 ``sqrt(gx^2 + gy^2)``（加小量防零）。

    参数:
        x: (B,C,H,W) 图像。

    返回:
        (B,1,H,W) 梯度幅值图。
    """
    gx, gy = spatial_gradient(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def local_contrast(x: Tensor, kernel_size: int = 15) -> Tensor:
    """局部对比度：|gray - 均值池化(gray)|。

    值大表示该处纹理/边缘丰富，值小表示平坦区域；用于引导裁剪
    与诊断分析。

    参数:
        x: (B,C,H,W) 图像。
        kernel_size: 均值池化窗口，默认 15。

    返回:
        (B,1,H,W) 局部对比度图。
    """
    gray = luminance(x)
    radius = kernel_size // 2
    local_mean = F.avg_pool2d(F.pad(gray, (radius, radius, radius, radius), mode="reflect"),
                              kernel_size, stride=1)
    return (gray - local_mean).abs()

def normalize_map(x: Tensor) -> Tensor:
    """把热力图稳健归一化到 [0,1]。

    缩放尺度取逐样本 ``mean + 3σ``（对离群值比 max 归一化稳健），
    超过尺度的值截断为 1。

    参数:
        x: (B,C,H,W) 任意非负热力图。

    返回:
        归一化后的同形张量。
    """
    flat = x.flatten(1)
    scale = flat.mean(1) + 3.0 * flat.std(1, unbiased=False)
    return (x / (scale.view(-1, 1, 1, 1) + 1e-6)).clamp(0.0, 1.0)
