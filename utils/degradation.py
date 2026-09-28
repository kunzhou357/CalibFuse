"""物理启发的成对退化（噪声）模型。

模拟低质量采集条件下的两个模态退化，为数据集提供可复现的合成噪声：

- :class:`VisibleNoise`（可见光）：**泊松散粒噪声 + 高斯读出噪声**。
  散粒噪声用 ``torch.poisson(clean * photons) / photons`` 模拟有限
  光子数下的采集（photons 越小噪声越大），读出噪声是加性高斯。
- :class:`InfraredNoise`（红外）：**乘性条带 + 加性条带 + 高斯 +
  坏点（stuck pixels）**。条带 = 平滑随机列剖面 + 行剖面，外加一个
  固定周期的正弦列条纹（模拟读出电路的列/行 FPN 与条纹伪影）；
  坏点以小概率把像素钉在 0 或 1。

两个类都是无参数的可调用对象，噪声幅度参数在每次调用时从传入的
``torch.Generator`` 中抽取（训练管道用 ``seed + epoch*1_000_003 +
index`` 播种，保证确定性复现）。

severity 档位：``"train"``（默认，训练分布）与 ``"hard"``（更重的
退化，区间与 train 不重叠）。参数一律在**对数或线性域均匀采样**后
再还原（photons 用对数均匀，保证暗端也有覆盖）。

注意：噪声模型属于指标/训练协议的一部分，改动会影响可复现性，
需按新协议版本处理。
"""


from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    """用指定生成器在 [low, high) 均匀采样一个标量。

    参数:
        generator: 随机数生成器（决定可复现性）。
        low/high: 采样区间端点。

    返回:
        Python float。
    """
    return float(torch.rand((), generator=generator) * (high - low) + low)

def _smooth_1d(profile: Tensor, kernel: int = 9) -> Tensor:
    """对一维剖面做 9 点移动平均，得到平滑的随机曲线。

    用于生成条带噪声的列/行剖面：白噪声经平滑后变成低频漂移，
    更接近真实传感器的固定模式噪声（FPN）。

    参数:
        profile: (N,) 张量。
        kernel: 平滑窗口，默认 9。

    返回:
        (N,) 平滑后的张量（边界用 padding 保持长度）。
    """
    return F.avg_pool1d(profile[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]


class VisibleNoise:
    """可见光退化：泊松散粒噪声 + 读出噪声。

    观测模型::

        shot = Poisson(clean * photons) / photons   # 散粒（信号相关）
        read ~ N(0, read_sigma^2)                   # 读出（信号无关）
        observed = clamp(shot + read, 0, 1)
    """

    def __init__(self, severity: str = "train") -> None:
        """初始化退化器。

        参数:
            severity: ``"train"`` 或 ``"hard"``，决定参数采样区间
                （photons：train 为 12~80 对数均匀，hard 为 6~12；
                read_sigma：train 为 0.002~0.025，hard 为 0.025~0.045）。
        """
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """对干净可见光图加噪。

        参数:
            clean: 干净图 (3, H, W)，取值 [0,1]。
            generator: 驱动所有随机性的生成器。

        返回:
            二元组 ``(observed, meta)``：
            - ``observed``：加噪观测图 (3, H, W)，截断到 [0,1]；
            - ``meta`` 字典：``damage`` = |obs-clean| 沿通道均值
              (1,H,W)（监督用掩码）；``photons``、``read_sigma``
              为本次采样的退化参数标量。
        """
        if self.severity == "train":
            # 光子数在对数域均匀采样：暗端（噪声重）也有足够覆盖
            photons = math.exp(_uniform(generator, math.log(12.0), math.log(80.0)))
            read_sigma = _uniform(generator, 0.002, 0.025)
        else:
            photons = math.exp(_uniform(generator, math.log(6.0), math.log(12.0)))
            read_sigma = _uniform(generator, 0.025, 0.045)

        # 散粒噪声：期望为 clean，方差随光子数减少而增大；
        # 除回 photons 保持期望仍是 clean
        shot = torch.poisson(clean * photons, generator=generator) / photons

        # 读出噪声：与信号无关的加性高斯
        read = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * read_sigma
        observed = (shot + read).clamp(0.0, 1.0)
        residual = observed - clean

        return observed, {
            # 逐像素损伤图（通道平均），训练/诊断用
            "damage": residual.abs().mean(0, keepdim=True),
            "photons": clean.new_tensor(photons),
            "read_sigma": clean.new_tensor(read_sigma),
        }


class InfraredNoise:
    """红外退化：条带（乘性+加性） + 高斯 + 坏点。

    观测模型::

        column ~ 平滑随机 + 周期正弦，归一化到 std=column_scale
        row    ~ 平滑随机，归一化到 std=row_scale
        stripe = column(行广播) + row(列广播)
        observed = (1 + 0.35·stripe)·clean + stripe + gaussian
                 再把 bad_probability 比例的像素替换为 0 或 1
    """

    def __init__(self, severity: str = "train") -> None:
        """初始化退化器。

        参数:
            severity: ``"train"`` 或 ``"hard"``。train 档参数区间：
                gaussian_sigma 0.004~0.100、column_scale 0.015~0.090、
                row_scale 0.000~0.035、bad_probability 0.0002~0.006；
                hard 档整体加重且区间与 train 不重叠。
        """
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """对干净红外图加噪。

        参数:
            clean: 干净红外图 (1, H, W)，取值 [0,1]。
            generator: 驱动所有随机性的生成器。

        返回:
            二元组 ``(observed, meta)``：
            - ``observed``：加噪观测图 (1, H, W)；
            - ``meta``：``damage`` = |obs-clean| (1,H,W)；
              ``stripe`` = 归一化到 [0,1] 的条带强度图；
              ``bad_pixel`` = 坏点 0/1 掩码；
              ``gaussian_sigma`` = 本次高斯噪声标准差标量。
        """
        _, height, width = clean.shape

        if self.severity == "train":
            gaussian_sigma = _uniform(generator, 0.004, 0.100)
            column_scale = _uniform(generator, 0.015, 0.090)
            row_scale = _uniform(generator, 0.000, 0.035)
            bad_probability = _uniform(generator, 0.0002, 0.006)
        else:
            gaussian_sigma = _uniform(generator, 0.100, 0.140)
            column_scale = _uniform(generator, 0.090, 0.150)
            row_scale = _uniform(generator, 0.035, 0.070)
            bad_probability = _uniform(generator, 0.006, 0.015)

        # 列/行剖面：白噪声平滑成低频漂移
        column = _smooth_1d(torch.randn(width, generator=generator, dtype=clean.dtype))
        row = _smooth_1d(torch.randn(height, generator=generator, dtype=clean.dtype))

        # 给列剖面叠加固定周期正弦：模拟读出电路的周期性条纹；
        # 周期在 [6, min(width,40)) 内随机，相位随机
        coordinate = torch.arange(width, dtype=clean.dtype)
        period = int(torch.randint(6, max(7, min(width, 40)), (), generator=generator))
        column += 0.5 * torch.sin(2.0 * math.pi * coordinate / period + _uniform(generator, 0.0, 2.0 * math.pi))

        # 把剖面归一化到目标标准差，噪声幅度可控
        column = column / column.std().clamp_min(1e-6) * column_scale
        row = row / row.std().clamp_min(1e-6) * row_scale

        # 广播成二维条带图：列剖面向行广播、行剖面向列广播
        stripe = column[None, None, :] + row[None, :, None]
        gaussian = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * gaussian_sigma
        # 条带同时做乘性（增益不均，0.35 系数）与加性（偏置）扰动
        observed = (1.0 + 0.35 * stripe) * clean + stripe + gaussian
        # 坏点：以 bad_probability 概率把像素钉在 0 或 1（各 50%）
        bad_mask = torch.rand(clean.shape, generator=generator) < bad_probability
        bad_values = (torch.rand(clean.shape, generator=generator) > 0.5).to(clean.dtype)
        observed = torch.where(bad_mask, bad_values, observed).clamp(0.0, 1.0)
        residual = observed - clean
        # 条带强度图归一化到 [0,1]（供损失/诊断加权）
        stripe_map = stripe.abs().expand_as(clean)
        return observed, {
            "damage": residual.abs(),
            "stripe": (stripe_map / stripe_map.amax().clamp_min(1e-6)).clamp(0.0, 1.0),
            "bad_pixel": bad_mask.to(clean.dtype),
            "gaussian_sigma": clean.new_tensor(gaussian_sigma),
        }
