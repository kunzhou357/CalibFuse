"""物理启发的成对退化（噪声）模型。

训练时在线合成退化观测（paired degradation），免于收集真实
退化数据的困难，并保证每个干净样本都有逐像素对应的"观测-干净"
监督对。包含两个退化模型：

- :class:`VisibleNoise`（可见光）：泊松散粒噪声 + 高斯读出噪声；
- :class:`InfraredNoise`（红外）：乘性与加性条带 + 高斯噪声 + 坏点。

两个类都是无参数的可调用对象；每次调用时从传入的
``torch.Generator`` 抽取噪声强度参数——该生成器由训练管线以
确定性的种子播种（``seed + epoch * 1_000_003 + index``），因此
退化过程完全可复现，且跨 epoch 不重复。

严重度档位：``"train"``（默认，训练用）与 ``"hard"``（更重的退化，
参数区间与 train 档不重叠，可用于评估鲁棒性上界）。

注意：这两个模型是训练/评测协议的一部分；修改它们的参数区间
或退化公式属于协议变更，会使新旧训练结果不可比。
"""


from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    """从生成器中抽取 [low, high) 区间的均匀浮点标量。

    使用独立的 torch.Generator 而非全局随机态，保证每个样本的
    噪声只依赖其自身种子，与数据加载顺序/worker 数无关。
    """
    return float(torch.rand((), generator=generator) * (high - low) + low)

def _smooth_1d(profile: Tensor, kernel: int = 9) -> Tensor:
    """对 1D 剖面做滑动平均平滑（avg_pool1d 实现）。

    白噪声经 9 点滑动平均后变为低频漂移曲线，
    用于模拟真实传感器固有的低频行/列不均匀性。
    """
    return F.avg_pool1d(profile[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]


class VisibleNoise:
    """可见光退化：泊松散粒噪声 + 高斯读出噪声。

    物理模型::

        shot = Poisson(clean * photons) / photons   # 信号相关的散粒噪声
        read ~ N(0, read_sigma^2)                   # 信号无关的读出噪声
        observed = clamp(shot + read, 0, 1)

    - 泊松噪声的方差与信号强度成正比（光子计数的统计涨落）：
      光子数 photons 越少（弱光），噪声相对越大；
      除以 photons 把期望拉回 clean；
    - 读出噪声模拟传感器电子学的高斯噪声，与信号无关。

    参数区间（对数均匀/均匀采样）:
    - train: photons ∈ [12, 80]（log 均匀，覆盖暗端），read_sigma ∈ [0.002, 0.025]
    - hard:  photons ∈ [6, 12]（弱光更狠），read_sigma ∈ [0.025, 0.045]
    """

    def __init__(self, severity: str = "train") -> None:
        """以 ``"train"`` 或 ``"hard"`` 参数档初始化。"""
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """退化一张干净可见光图像；返回 ``(observed, meta)``。

        参数:
            clean: (3, H, W) ∈ [0,1] 干净 RGB 图像。
            generator: 已播种的 torch.Generator。

        返回:
            observed: 退化后的图像（截断到 [0,1]）。
            meta: 元信息字典:
            - ``damage``: (1, H, W) 逐像素平均绝对残差（|observed - clean|），
              训练中用于判断该模态是否"干净"（全零即干净）；
            - ``photons`` / ``read_sigma``: 本次采样的参数标量。
        """
        if self.severity == "train":
            # 光子数取对数均匀分布：在对数尺度上均匀覆盖弱光（噪声大）到较强光
            photons = math.exp(_uniform(generator, math.log(12.0), math.log(80.0)))
            read_sigma = _uniform(generator, 0.002, 0.025)
        else:
            photons = math.exp(_uniform(generator, math.log(6.0), math.log(12.0)))
            read_sigma = _uniform(generator, 0.025, 0.045)

        # 期望值除回 photons 后仍等于 clean：散粒噪声均值为零（相对信号）
        shot = torch.poisson(clean * photons, generator=generator) / photons
        read = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * read_sigma
        observed = (shot + read).clamp(0.0, 1.0)
        residual = observed - clean

        return observed, {
            "damage": residual.abs().mean(0, keepdim=True),
            "photons": clean.new_tensor(photons),
            "read_sigma": clean.new_tensor(read_sigma),
        }


class InfraredNoise:
    """红外退化：条带（乘性 + 加性）+ 高斯噪声 + 坏点。

    物理模型::

        column ~ 平滑随机 + 固定周期正弦，归一化到 std=column_scale
        row    ~ 平滑随机，归一化到 std=row_scale
        stripe = column（沿行广播） + row（沿列广播）
        observed = (1 + 0.35 * stripe) * clean + stripe + gaussian
                   并有 bad_probability 比例的像素被钉在 0 或 1

    - 列条带：读出电路逐列放大的增益/偏置差异；平滑随机分量模拟
      低频不均匀性，固定周期的正弦分量模拟读出时序引入的周期条纹；
    - 行条带：行扫描器件的行间差异（幅度通常更小）；
    - 条带同时以 0.35*stripe 的比例扰动增益（乘性）并以 stripe 加性
      扰动偏置，覆盖两种真实条带形态；
    - 坏点（stuck pixels）：读出失败的像素，恒为 0 或 1。

    参数区间:
    - train: gaussian_sigma ∈ [0.004, 0.100]，column_scale ∈ [0.015, 0.090]，
      row_scale ∈ [0, 0.035]，bad_probability ∈ [0.0002, 0.006]
    - hard:  各区间整体上移（更重的退化）。
    """

    def __init__(self, severity: str = "train") -> None:
        """以 ``"train"`` 或 ``"hard"`` 参数档初始化。"""
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """退化一张干净红外图像；返回 ``(observed, meta)``。

        参数:
            clean: (1, H, W) ∈ [0,1] 干净灰度图像。
            generator: 已播种的 torch.Generator。

        返回:
            observed: 退化后的图像（截断到 [0,1]）。
            meta: 元信息字典:
            - ``damage``: (1, H, W) 逐像素绝对残差；
            - ``stripe``: 归一化到 [0,1] 的条带强度图（诊断/分析用）；
            - ``bad_pixel``: 坏点掩码（0/1）；
            - ``gaussian_sigma``: 本次采样的高斯噪声标准差。
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

        # 白噪声经平滑成为低频漂移的列/行剖面
        column = _smooth_1d(torch.randn(width, generator=generator, dtype=clean.dtype))
        row = _smooth_1d(torch.randn(height, generator=generator, dtype=clean.dtype))

        # 列剖面叠加固定周期的正弦分量，模拟读出时序的周期性条纹；
        # 周期在 [6, min(width, 40)) 内随机，相位随机
        coordinate = torch.arange(width, dtype=clean.dtype)
        period = int(torch.randint(6, max(7, min(width, 40)), (), generator=generator))
        column += 0.5 * torch.sin(2.0 * math.pi * coordinate / period + _uniform(generator, 0.0, 2.0 * math.pi))

        # 将两个剖面的标准差归一化到目标幅度（column_scale / row_scale）
        column = column / column.std().clamp_min(1e-6) * column_scale
        row = row / row.std().clamp_min(1e-6) * row_scale

        # 条带场：列剖面沿行广播 + 行剖面沿列广播
        stripe = column[None, None, :] + row[None, :, None]
        gaussian = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * gaussian_sigma
        # 条带同时扰动增益（乘性项 0.35*stripe）与偏置（加性项 stripe）
        observed = (1.0 + 0.35 * stripe) * clean + stripe + gaussian
        # 坏点：以 bad_probability 概率将像素钉在 0 或 1（等概率二选一）
        bad_mask = torch.rand(clean.shape, generator=generator) < bad_probability
        bad_values = (torch.rand(clean.shape, generator=generator) > 0.5).to(clean.dtype)
        observed = torch.where(bad_mask, bad_values, observed).clamp(0.0, 1.0)
        residual = observed - clean
        # 条带强度图归一化到 [0,1] 便于可视化/分析
        stripe_map = stripe.abs().expand_as(clean)
        return observed, {
            "damage": residual.abs(),
            "stripe": (stripe_map / stripe_map.amax().clamp_min(1e-6)).clamp(0.0, 1.0),
            "bad_pixel": bad_mask.to(clean.dtype),
            "gaussian_sigma": clean.new_tensor(gaussian_sigma),
        }
