"""Image I/O and common image transforms.

- :func:`save_tensor`: save a tensor as an 8-bit image (the 8-bit
  quantization is part of the metric protocol);
- :func:`luminance`: BT.601 weighted luminance;
- :func:`spatial_gradient` / :func:`gradient_magnitude`: Sobel gradients;
- :func:`local_contrast`: local contrast via mean pooling;
- :func:`normalize_map`: robust mean + 3*std heatmap normalization.
"""


from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor


def save_tensor(path: str | Path, image: Tensor) -> None:
    """Save a (C, H, W) or (1, C, H, W) tensor in [0, 1] as an 8-bit image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).cpu()
    if array.ndim == 4:
        array = array[0]
    # metrics run on the saved 8-bit PNGs, so this quantization is part of
    # the calibfuse-metrics-v1 protocol
    array = (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)

def luminance(rgb: Tensor) -> Tensor:
    """BT.601 luminance for RGB input; single-channel input passes through."""
    if rgb.shape[1] == 1:
        return rgb
    weights = rgb.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
    return (rgb * weights).sum(1, keepdim=True)

def spatial_gradient(x: Tensor) -> tuple[Tensor, Tensor]:
    """3x3 Sobel gradients ``(gx, gy)`` with reflect padding."""
    gray = luminance(x)
    kx = gray.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kx), F.conv2d(padded, ky)

def gradient_magnitude(x: Tensor) -> Tensor:
    """Gradient magnitude ``sqrt(gx^2 + gy^2)``."""
    gx, gy = spatial_gradient(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def local_contrast(x: Tensor, kernel_size: int = 15) -> Tensor:
    """Local contrast ``|gray - mean-pooled gray|``."""
    gray = luminance(x)
    radius = kernel_size // 2
    local_mean = F.avg_pool2d(F.pad(gray, (radius, radius, radius, radius), mode="reflect"),
                              kernel_size, stride=1)
    return (gray - local_mean).abs()

def normalize_map(x: Tensor) -> Tensor:
    """Robustly normalize a heatmap to [0, 1] using per-sample mean + 3*std."""
    flat = x.flatten(1)
    scale = flat.mean(1) + 3.0 * flat.std(1, unbiased=False)
    return (x / (scale.view(-1, 1, 1, 1) + 1e-6)).clamp(0.0, 1.0)
