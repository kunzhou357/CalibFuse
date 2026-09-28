"""成对图像数据集与质量均衡采样器。

本文件实现训练/评估共用的数据管道：

- :func:`paired_paths`：把 ``vis/``（或 ``visible``/``vi``）与 ``ir/``
  （或 ``infrared``/``inf``）目录下**同文件名主干**的图像配对；
  任一侧没有对应文件的样本会被**静默跳过**——排查数据问题时先核对
  配对数量是否等于预期。
- :func:`load_image`：读图并转到 [0,1] 的 (C,H,W) 张量。
- :class:`PairedFusionDataset`：核心数据集。把每个样本轮流置入四种
  质量状态（干净 / 可见光噪声 / 红外噪声 / 双噪声），同时返回干净
  图与观测图，供"干净教师"监督使用；噪声由
  :mod:`utils.degradation` 按确定性种子生成。
- :class:`QualityBalancedBatchSampler`：保证每个 batch 内四种状态
  样本数相同的批采样器（因此 ``--batch-size`` 必须能被 4 整除）。

可复现性设计：第 ``index`` 个样本在第 ``epoch`` 轮的噪声由
``Generator(seed + epoch * 1_000_003 + index)`` 驱动——同一样本
同一轮的噪声完全可复现，不同轮之间不重复。这套机制依赖
"每个 epoch 调用 ``set_epoch`` 后重建 DataLoader"的策略（见
``train.py`` 与 ``tests/test_worker_epoch.py``），请勿改成
persistent workers。
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
    """收集 root 下的 (可见光, 红外) 图像对，按主干名排序。

    目录名依次尝试 ``vis``/``visible``/``vi`` 与 ``ir``/``infrared``/
    ``inf``，取第一个存在的。只有**两侧都存在**的主干名才会成对；
    缺失一侧的样本被静默丢弃（注意核对数量）。

    参数:
        root: 数据根目录，其下应包含 vis/ 与 ir/ 两个子目录。

    返回:
        ``(vis_path, ir_path)`` 列表，按主干名排序，保证顺序确定。

    异常:
        FileNotFoundError: 找不到任一模态目录。
        RuntimeError: 没有任何同主干名图像对。
    """
    root = Path(root)
    vis_dir = next((root / name for name in ("vis", "visible", "vi") if (root / name).is_dir()), None)
    ir_dir = next((root / name for name in ("ir", "infrared", "inf") if (root / name).is_dir()), None)
    if vis_dir is None or ir_dir is None:
        raise FileNotFoundError(f"Expected paired vis/ir folders below {root}")
    # 以主干名为键建立索引，过滤非图像扩展名
    vis = {p.stem: p for p in vis_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    ir = {p.stem: p for p in ir_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}

    # 集合交集 = 双侧都存在的主干；排序保证跨机器顺序一致
    keys = sorted(vis.keys() & ir.keys())
    if not keys:
        raise RuntimeError(f"No same-stem image pairs found in {vis_dir} and {ir_dir}")
    return [(vis[key], ir[key]) for key in keys]


def load_image(path: Path, mode: str) -> Tensor:
    """读取图像为 [0,1] float32 张量。

    参数:
        path: 图像路径。
        mode: PIL 转换模式，可见光用 ``"RGB"``（→3 通道），红外用
            ``"L"``（→单通道）。

    返回:
        (C, H, W) 张量；灰度图为 (1, H, W)。
    """
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        # 灰度图补一个通道维，统一为 (H, W, C)
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

def _resize_minimum(x: Tensor, crop: int) -> Tensor:
    """若图像任一边小于 crop，则等比例放大到至少能裁出 crop。

    只放大不缩小（scale 下限 1.0），避免无谓地损失分辨率；
    目标尺寸向上取整。

    参数:
        x: (C, H, W) 张量。
        crop: 目标裁剪边长。

    返回:
        放大后的 (C, H', W') 张量（或原张量）。
    """
    height, width = x.shape[-2:]
    scale = max(crop / height, crop / width, 1.0)
    if scale > 1.0:
        x = F.interpolate(x[None], size=(math.ceil(height * scale), math.ceil(width * scale)), mode="bilinear", align_corners=False)[0]
    return x

def _guided_crop(visible: Tensor, infrared: Tensor, size: int, training: bool) -> tuple[Tensor, Tensor]:
    """对齐裁剪一对图像，训练时偏向信息丰富的区域。

    先把两图放大到至少 size，再按同一位置裁剪（保证像素对齐）：
    - 推理/评估：取中心裁剪，位置确定可复现；
    - 训练：50% 概率随机裁剪；另 50% 做"引导裁剪"——以红外局部
      对比度最高的前 10% 像素中随机选一点为参考，把裁窗随机偏向
      该点。这样训练更多看到纹理/目标丰富的区域，而不是大片平
      滑背景。

    参数:
        visible: (3, H, W) 可见光。
        infrared: (1, H, W) 红外。
        size: 裁剪边长。
        training: 是否训练模式。

    返回:
        ``(visible_crop, infrared_crop)``，均为 (C, size, size)。
    """
    # 先分别放大到不小于 size（两图原始尺寸一致，放大后也一致）
    visible, infrared = _resize_minimum(visible, size), _resize_minimum(infrared, size)
    height, width = infrared.shape[-2:]
    if not training:
        # 评估：中心裁剪
        top, left = (height - size) // 2, (width - size) // 2
    elif random.random() < 0.5:
        # 训练：完全随机位置
        top, left = random.randint(0, height - size), random.randint(0, width - size)
    else:
        # 训练：对比度引导。15x15 均值池化近似局部均值，
        # |ir - 均值| 即局部对比度（纹理/边缘强度）
        contrast = (infrared - F.avg_pool2d(infrared[None], 15, stride=1, padding=7)[0]).abs()[0]
        # 取对比度前 10% 的像素作为候选参考点
        candidates = (contrast >= torch.quantile(contrast.flatten(), 0.90)).nonzero()
        y, x = candidates[random.randrange(len(candidates))].tolist()
        # 裁窗左上角在包含参考点的范围内随机取，并夹紧到图像内
        top = min(max(y - random.randrange(size), 0), height - size)
        left = min(max(x - random.randrange(size), 0), width - size)
    # 两图用同一 (top, left) 裁剪，保持像素级对齐
    return visible[..., top : top + size, left : left + size], infrared[..., top : top + size, left : left + size]


class PairedFusionDataset(Dataset):
    """成对融合数据集：每个样本轮流经历四种质量状态。

    状态由 ``state_for_index`` 决定：``(index + epoch) % 4``——同一
    样本在不同轮次轮换状态，长训练下每个样本在四种状态上大致均衡。
    __getitem__ 同时返回**干净图**（``visible``/``infrared``，监督用）
    与**观测图**（``visible_observed``/``infrared_observed``，网络
    输入用），干净态时二者相同。

    噪声种子 ``seed + epoch * 1_000_003 + index`` 保证可复现且轮间
    不重复（1_000_003 为素数，避免与样本数取模撞车）。
    """

    # 四种质量状态的名称，索引即状态编号
    STATE_NAMES = ("both_clean", "visible_noisy", "infrared_noisy", "both_noisy")
    def __init__(self, pairs: list[tuple[Path, Path]], crop_size: int = 192, training: bool = True,
                 corruption: bool = True, severity: str = "train", seed: int = 3407) -> None:
        """初始化数据集。

        参数:
            pairs: :func:`paired_paths` 的输出。
            crop_size: 裁剪边长（须能被 4 整除，见网络下采样）。
            training: True 用随机/引导裁剪，False 用中心裁剪。
            corruption: False 时永远输出干净状态（状态恒为 0），
                用于纯干净评估。
            severity: 噪声强度档位 ``"train"`` 或 ``"hard"``。
            seed: 噪声随机种子基准值。
        """
        self.pairs, self.crop_size, self.training = pairs, crop_size, training
        self.corruption, self.seed, self.epoch = corruption, seed, 0

        # 两个模态各自的退化器（无参数对象，只封装参数范围）
        self.visible_noise, self.infrared_noise = VisibleNoise(severity), InfraredNoise(severity)

    def __len__(self) -> int:
        """样本对数量。"""
        return len(self.pairs)

    def state_for_index(self, index: int) -> int:
        """样本 index 在当前 epoch 的质量状态编号（0~3）。

        ``(index + epoch) % 4``：epoch 每轮 +1，使同一样本轮换状态。
        corruption=False 时恒为 0（干净）。
        """
        return (index + self.epoch) % 4 if self.corruption else 0

    def set_epoch(self, epoch: int) -> None:
        """更新当前轮数（train.py 每轮重建 DataLoader 前调用）。

        影响状态轮换与噪声种子；worker 每轮重建的策略保证了
        worker 内不会残留旧 epoch（见 tests/test_worker_epoch.py）。
        """
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """取出一个样本：读图 → 对齐裁剪 → 按状态加噪。

        参数:
            index: 样本下标。

        返回:
            字典，键含：
            - 干净图 ``visible`` (3,S,S)、``infrared`` (1,S,S)；
            - 观测图 ``visible_observed``/``infrared_observed``；
            - 退化掩码：``visible_damage``/``infrared_damage``
              （逐像素 |obs-clean|，可见光先做了通道平均）、
              ``infrared_stripe``（归一化条带强度图）、
              ``infrared_bad_pixel``（坏点 0/1 掩码）；
            - 退化参数标量：``visible_photons``、``visible_read_sigma``、
              ``infrared_gaussian_sigma``（干净态取中性值）；
            - ``quality_state``：状态编号张量；
            - ``name``：样本主干名（便于结果对账）。
        """
        vis_path, ir_path = self.pairs[index]
        visible, infrared = load_image(vis_path, "RGB"), load_image(ir_path, "L")
        # 尺寸不一致时以可见光为准双线性重采样红外，保证像素对齐
        if infrared.shape[-2:] != visible.shape[-2:]:
            infrared = F.interpolate(infrared[None], size=visible.shape[-2:], mode="bilinear", align_corners=False)[0]
        visible, infrared = _guided_crop(visible, infrared, self.crop_size, self.training)
        state = self.state_for_index(index)
        # 确定性噪声生成器：同一样本同一轮结果固定，轮间不同
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + index)
        # 先假设干净：观测=干净，掩码全零，参数取中性值
        visible_obs, infrared_obs = visible, infrared
        visible_meta = {"damage": torch.zeros(1, *visible.shape[-2:]),
                        "photons": torch.tensor(1e6), "read_sigma": torch.tensor(0.0)}
        infrared_meta = {"damage": torch.zeros_like(infrared), "stripe": torch.zeros_like(infrared), "bad_pixel": torch.zeros_like(infrared),
                         "gaussian_sigma": torch.tensor(0.0)}

        # 状态 1/3：可见光加噪（泊松散粒 + 读出噪声）
        if state in (1, 3):
            visible_obs, visible_meta = self.visible_noise(visible, generator)
        # 状态 2/3：红外加噪（高斯 + 条带 + 坏点）
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
    """质量均衡批采样器：每个 batch 四种状态各占 batch_size/4。

    实现方式：把样本按当前状态分成四组，各自洗牌后按轮转（取模）
    方式切出 ``batch_size // 4`` 个索引拼成一个 batch，再打乱批内
    顺序。较小的组会环绕重复采样（wrap），因此四种状态的批内配额
    始终精确相等。**batch_size 必须能被 4 整除**，否则构造时抛错。
    """

    def __init__(self, dataset: PairedFusionDataset, batch_size: int, seed: int = 3407) -> None:
        """初始化采样器。

        参数:
            dataset: 关联的数据集（读取其 state_for_index）。
            batch_size: 批大小，须被 4 整除。
            seed: 洗牌种子基准（实际种子 = seed + epoch）。
        """
        if batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        """更新轮数（与数据集的 set_epoch 同步调用，改变洗牌种子）。"""

        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """逐批产出索引列表。

        每轮迭代重新按当前 epoch 分组并洗牌，所以同一 epoch 内
        重复调用 __iter__（DataLoader 每轮重建时会发生）得到一致
        而非重复的划分；跨 epoch 则不同。
        """
        # 按当前状态把所有样本分为四组
        groups = [[i for i in range(len(self.dataset)) if self.dataset.state_for_index(i) == state] for state in range(4)]
        rng = random.Random(self.seed + self.epoch)
        for group in groups:
            rng.shuffle(group)
        per_state = self.batch_size // 4
        # 批数由最大的组决定；小组靠环绕补齐
        batches = max(math.ceil(len(group) / per_state) for group in groups)
        for batch_id in range(batches):
            batch = []
            for group in groups:
                start = batch_id * per_state

                # 环绕取模：小组的索引循环使用，保证每批配额精确
                batch.extend(group[(start + offset) % len(group)] for offset in range(per_state))
            # 打乱批内顺序，避免同状态样本总聚在前面的固定位置
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """每轮的批数（按最大状态组计算，与其他迭代逻辑一致）。"""
        per_state = self.batch_size // 4
        counts = [sum(self.dataset.state_for_index(i) == state for i in range(len(self.dataset))) for state in range(4)]
        return max(math.ceil(count / per_state) for count in counts)
