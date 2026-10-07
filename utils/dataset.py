"""成对图像数据集与质量均衡批采样器。

本模块包含四个核心组件：

- :func:`paired_paths`：从 ``vis/``（或 ``visible``/``vi``）与 ``ir/``
  （或 ``infrared``/``inf``）目录收集同名（stem 相同）图像对；
  任一侧缺失的 stem 会被**静默跳过**——务必核对配对数量再信任结果；
- :func:`load_image`：把图像读成 [0, 1] 的 float32 张量 (C, H, W)；
- :class:`PairedFusionDataset`：让每个样本按 epoch 轮换经历四种质量
  状态（全干净 / 可见光带噪 / 红外带噪 / 双侧带噪），同时返回干净对
  与观测对；噪声种子为 ``seed + epoch * 1_000_003 + index``，
  epoch 内可复现、跨 epoch 不重复；
- :class:`QualityBalancedBatchSampler`：每个 batch 内四种状态的样本
  数严格相等（各占 batch_size // 4），因此 ``--batch-size`` 必须能被
  4 整除。

可复现性依赖：每个 epoch 前调用 ``set_epoch``，并且每个 epoch 重建
DataLoader（见 ``train.py``）；不要改用 persistent_workers——持久化
worker 会停留在第一个 epoch 的状态，导致逐样本噪声种子重复。
"""


from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Iterator
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from .degradation import InfraredNoise, VisibleNoise

# 支持的图像扩展名（大小写不敏感）
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def paired_paths(root: str | Path) -> list[tuple[Path, Path]]:
    """收集 ``root`` 下的 (可见光, 红外) 图像对，按 stem 排序。

    依次尝试 ``vis``/``visible``/``vi`` 与 ``ir``/``infrared``/``inf``
    目录名。只配对两侧都存在的 stem，其余文件被静默丢弃
    （调用方应自行核对配对数量）。

    参数:
        root: 包含两个模态子目录的数据根目录。

    返回:
        [(可见光路径, 红外路径), ...]，按 stem 字典序排列，
        保证跨机器的确定性顺序。
    """
    root = Path(root)
    # 在候选目录名中找到第一个实际存在的目录
    vis_dir = next((root / name for name in ("vis", "visible", "vi") if (root / name).is_dir()), None)
    ir_dir = next((root / name for name in ("ir", "infrared", "inf") if (root / name).is_dir()), None)
    if vis_dir is None or ir_dir is None:
        raise FileNotFoundError(f"Expected paired vis/ir folders below {root}")
    # 以 stem 为键建索引；后缀大小写不敏感
    vis = {p.stem: p for p in vis_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    ir = {p.stem: p for p in ir_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}

    # 取 stem 交集并排序（确定性顺序，跨机器一致）
    keys = sorted(vis.keys() & ir.keys())
    if not keys:
        raise RuntimeError(f"No same-stem image pairs found in {vis_dir} and {ir_dir}")
    return [(vis[key], ir[key]) for key in keys]


def load_image(path: Path, mode: str) -> Tensor:
    """把图像读成 [0, 1] 的 float32 张量 (C, H, W)；灰度图为 (1, H, W)。

    参数:
        path: 图像路径。
        mode: PIL 转换模式，可见光用 "RGB"，红外用 "L"。
    """
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]  # 灰度 (H, W) -> (H, W, 1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

def _resize_minimum(x: Tensor, crop: int) -> Tensor:
    """只放不缩：确保图像至少能容纳一个 crop 大小的窗口。

    若短边小于 crop，则按需放大（双线性）；
    已经足够大的图像保持原分辨率（避免无谓的信息损失）。
    """
    height, width = x.shape[-2:]
    scale = max(crop / height, crop / width, 1.0)
    if scale > 1.0:
        x = F.interpolate(x[None], size=(math.ceil(height * scale), math.ceil(width * scale)), mode="bilinear", align_corners=False)[0]
    return x

def _guided_crop(visible: Tensor, infrared: Tensor, size: int, training: bool) -> tuple[Tensor, Tensor]:
    """对图像对做同位置裁剪；训练时偏向信息量大的区域。

    策略:
    - 评估：确定性的中心裁剪（结果可复现）；
    - 训练：一半概率完全随机裁剪（覆盖全局）；
      另一半概率做"对比度引导"裁剪——在红外图局部对比度最高的
      前 10% 位置中随机选一个锚点，并让窗口偏向它
      （保证纹理丰富区域被充分采样）。

    两张图使用完全相同的窗口，像素级配对关系保持不变。
    """
    visible, infrared = _resize_minimum(visible, size), _resize_minimum(infrared, size)
    height, width = infrared.shape[-2:]
    if not training:
        # 中心裁剪：偶数尺寸差异时左/上取较小偏移
        top, left = (height - size) // 2, (width - size) // 2
    elif random.random() < 0.5:
        # 完全随机裁剪
        top, left = random.randint(0, height - size), random.randint(0, width - size)
    else:
        # 对比度引导：|像素 - 15x15 邻域均值| 作为局部对比度，
        # 在前 10% 分位的位置中随机选锚点，窗口向锚点偏置
        contrast = (infrared - F.avg_pool2d(infrared[None], 15, stride=1, padding=7)[0]).abs()[0]
        candidates = (contrast >= torch.quantile(contrast.flatten(), 0.90)).nonzero()
        y, x = candidates[random.randrange(len(candidates))].tolist()
        top = min(max(y - random.randrange(size), 0), height - size)
        left = min(max(x - random.randrange(size), 0), width - size)
    # 两个模态使用同一窗口，保证像素对齐
    return visible[..., top : top + size, left : left + size], infrared[..., top : top + size, left : left + size]


class PairedFusionDataset(Dataset):
    """按四种质量状态轮换的成对数据集。

    样本 ``index`` 在 epoch ``epoch`` 的状态为 ``(index + epoch) % 4``，
    即每个 epoch 样本轮换一种状态，四个 epoch 一个完整周期；
    配合质量均衡采样器，每个 batch 内四种状态的样本数严格相等。

    ``__getitem__`` 同时返回干净对（``visible``/``infrared``）与
    观测对（``visible_observed``/``infrared_observed``）；
    在全干净状态下两者相同。

    噪声种子 ``seed + epoch * 1_000_003 + index``：
    1_000_003 为质数，使 (epoch, index) 的组合几乎不发生种子碰撞。
    """

    # 四种质量状态名称（诊断/日志用）
    STATE_NAMES = ("both_clean", "visible_noisy", "infrared_noisy", "both_noisy")
    def __init__(self, pairs: list[tuple[Path, Path]], crop_size: int = 192, training: bool = True,
                 corruption: bool = True, severity: str = "train", seed: int = 3407) -> None:
        """初始化数据集。

        参数:
            pairs: :func:`paired_paths` 返回的路径对列表。
            crop_size: 裁剪窗口大小（须为 4 的倍数，训练入口已校验）。
            training: True 用随机/对比度引导裁剪，False 用中心裁剪。
            corruption: False 时强制全干净状态（关掉在线退化）。
            severity: 退化档位，"train" 或 "hard"。
            seed: 噪声种子基值。
        """
        self.pairs, self.crop_size, self.training = pairs, crop_size, training
        self.corruption, self.seed, self.epoch = corruption, seed, 0

        # 两个退化模型在初始化时按档位构建（本身无参数）
        self.visible_noise, self.infrared_noise = VisibleNoise(severity), InfraredNoise(severity)

    def __len__(self) -> int:
        """图像对数量。"""
        return len(self.pairs)

    def state_for_index(self, index: int) -> int:
        """当前 epoch 中样本 ``index`` 的质量状态（0-3）。

        0=全干净，1=可见光带噪，2=红外带噪，3=双侧带噪。
        corruption=False 时恒为 0。
        """
        return (index + self.epoch) % 4 if self.corruption else 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch；每次新建 DataLoader 迭代前必须调用。

        由于 worker 每个 epoch 重建（见 train.py），新 worker 会
        拿到更新后的 self.epoch，从而产生不重复的噪声种子。
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """加载一对图像，做同位置裁剪，并施加该状态对应的退化。

        返回:
            字典，包含:
            - 干净对: ``visible`` (3,H,W) / ``infrared`` (1,H,W)；
            - 观测对: ``visible_observed`` / ``infrared_observed``；
            - 退化掩码: ``visible_damage`` (1,H,W)、``infrared_damage``、
              ``infrared_stripe``、``infrared_bad_pixel``；
            - 退化参数标量: ``visible_photons``、``visible_read_sigma``、
              ``infrared_gaussian_sigma``（干净时为无噪声的占位值）；
            - ``quality_state``: 状态编号标量；
            - ``name``: 文件名 stem。
        """
        vis_path, ir_path = self.pairs[index]
        visible, infrared = load_image(vis_path, "RGB"), load_image(ir_path, "L")
        # 红外分辨率与可见光不一致时，双线性对齐到可见光尺寸
        # （与 test.py 的推理路径保持一致）
        if infrared.shape[-2:] != visible.shape[-2:]:
            infrared = F.interpolate(infrared[None], size=visible.shape[-2:], mode="bilinear", align_corners=False)[0]
        visible, infrared = _guided_crop(visible, infrared, self.crop_size, self.training)
        state = self.state_for_index(index)
        # 确定性的逐样本、逐 epoch 噪声种子：
        # 同一 (epoch, index) 总是产生相同噪声；不同 epoch 不重复
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + index)
        visible_obs, infrared_obs = visible, infrared
        # 干净状态下的占位 meta（damage 全零 => 损失判为"干净"）
        visible_meta = {"damage": torch.zeros(1, *visible.shape[-2:]),
                        "photons": torch.tensor(1e6), "read_sigma": torch.tensor(0.0)}
        infrared_meta = {"damage": torch.zeros_like(infrared), "stripe": torch.zeros_like(infrared), "bad_pixel": torch.zeros_like(infrared),
                         "gaussian_sigma": torch.tensor(0.0)}

        # 状态 1/3：可见光带噪；状态 2/3：红外带噪
        # 两个模态共用同一个 generator：种子流确定且不重复
        if state in (1, 3):
            visible_obs, visible_meta = self.visible_noise(visible, generator)
        if state in (2, 3):
            infrared_obs, infrared_meta = self.infrared_noise(infrared, generator)
        return {
            "visible": visible, "infrared": infrared,
            "visible_observed": visible_obs, "infrared_observed": infrared_obs,
            "visible_damage": visible_meta["damage"], "infrared_damage": infrared_meta["damage"],
            "infrared_stripe": infrared_meta["stripe"], "infrared_bad_pixel": infrared_meta["bad_pixel"],
            "visible_photons": visible_meta["photons"], "visible_read_sigma": visible_meta["read_sigma"],
            "infrared_gaussian_sigma": infrared_meta["gaussian_sigma"],
            "quality_state": torch.tensor(state), "name": vis_path.stem,
        }


class QualityBalancedBatchSampler(Sampler[list[int]]):
    """每个 batch 四种质量状态严格各占 ``batch_size // 4`` 的采样器。

    做法：按当前状态把样本分为四组 -> 组内洗牌 -> 轮流（round-robin）
    发牌到各个 batch；样本较少的组取模回绕（wrap around），
    因此各状态的配额始终精确。``batch_size`` 必须能被 4 整除。

    目的：每个 batch 同时包含四种退化情形，使融合损失、恢复损失、
    校准损失的批量统计无偏（不会出现某个 batch 全是干净样本、
    另一个 batch 全是带噪样本的震荡）。
    """

    def __init__(self, dataset: PairedFusionDataset, batch_size: int, seed: int = 3407) -> None:
        """初始化采样器；洗牌种子随 epoch 推进。

        参数:
            dataset: 必须是 :class:`PairedFusionDataset`
                （依赖其 ``state_for_index``）。
            batch_size: 批大小，必须能被 4 整除。
            seed: 洗牌种子基值（实际种子 = seed + epoch）。
        """
        if batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch；需与数据集的 set_epoch 保持同步。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """为当前 epoch 产出批次索引列表。"""
        # 按四种状态分组（状态由 dataset.epoch 决定）
        groups = [[i for i in range(len(self.dataset)) if self.dataset.state_for_index(i) == state] for state in range(4)]
        # 独立的 random.Random 实例：洗牌只依赖 seed+epoch，可复现
        rng = random.Random(self.seed + self.epoch)
        for group in groups:
            rng.shuffle(group)
        per_state = self.batch_size // 4
        # batch 总数由最大组决定；小组取模回绕，保证配额精确
        batches = max(math.ceil(len(group) / per_state) for group in groups)
        for batch_id in range(batches):
            batch = []
            for group in groups:
                start = batch_id * per_state
                batch.extend(group[(start + offset) % len(group)] for offset in range(per_state))
            # batch 内再洗牌，避免四种状态固定占据固定槽位
            # （配合 BN 类统计/Loss 平均更均衡）
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """每 epoch 的 batch 数（由最大的状态组决定）。"""
        per_state = self.batch_size // 4
        counts = [sum(self.dataset.state_for_index(i) == state for i in range(len(self.dataset))) for state in range(4)]
        return max(math.ceil(count / per_state) for count in counts)
