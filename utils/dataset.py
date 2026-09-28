"""Paired image dataset and quality-balanced batch sampler.

- :func:`paired_paths`: pair same-stem images from ``vis/`` (or
  ``visible``/``vi``) and ``ir/`` (or ``infrared``/``inf``) folders; stems
  missing on either side are silently skipped, so verify pair counts;
- :func:`load_image`: load an image as a [0, 1] tensor;
- :class:`PairedFusionDataset`: cycles each sample through four quality
  states (clean / visible noisy / infrared noisy / both noisy) and returns
  both clean and observed pairs; noise is seeded deterministically as
  ``seed + epoch * 1_000_003 + index``;
- :class:`QualityBalancedBatchSampler`: equal per-state quotas in every
  batch, so ``--batch-size`` must be divisible by four.

Reproducibility relies on calling ``set_epoch`` and recreating the
DataLoader each epoch (see ``train.py``); do not switch to persistent
workers.
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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def paired_paths(root: str | Path) -> list[tuple[Path, Path]]:
    """Collect (visible, infrared) pairs below ``root``, sorted by stem.

    Tries the ``vis``/``visible``/``vi`` and ``ir``/``infrared``/``inf``
    folder names. Only stems present on both sides are paired; the rest
    are dropped silently.
    """
    root = Path(root)
    vis_dir = next((root / name for name in ("vis", "visible", "vi") if (root / name).is_dir()), None)
    ir_dir = next((root / name for name in ("ir", "infrared", "inf") if (root / name).is_dir()), None)
    if vis_dir is None or ir_dir is None:
        raise FileNotFoundError(f"Expected paired vis/ir folders below {root}")
    vis = {p.stem: p for p in vis_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    ir = {p.stem: p for p in ir_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}

    # intersection of stems; sorted for deterministic order across machines
    keys = sorted(vis.keys() & ir.keys())
    if not keys:
        raise RuntimeError(f"No same-stem image pairs found in {vis_dir} and {ir_dir}")
    return [(vis[key], ir[key]) for key in keys]


def load_image(path: Path, mode: str) -> Tensor:
    """Load an image as a [0, 1] float32 tensor (C, H, W); grayscale is (1, H, W)."""
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

def _resize_minimum(x: Tensor, crop: int) -> Tensor:
    """Upscale (never downscale) so a ``crop``-sized window fits."""
    height, width = x.shape[-2:]
    scale = max(crop / height, crop / width, 1.0)
    if scale > 1.0:
        x = F.interpolate(x[None], size=(math.ceil(height * scale), math.ceil(width * scale)), mode="bilinear", align_corners=False)[0]
    return x

def _guided_crop(visible: Tensor, infrared: Tensor, size: int, training: bool) -> tuple[Tensor, Tensor]:
    """Crop a pair at the same position; training prefers informative regions.

    Evaluation takes the deterministic center crop; training takes a fully
    random crop half of the time and, the other half, biases the window
    toward the highest-local-contrast infrared locations.
    """
    visible, infrared = _resize_minimum(visible, size), _resize_minimum(infrared, size)
    height, width = infrared.shape[-2:]
    if not training:
        top, left = (height - size) // 2, (width - size) // 2
    elif random.random() < 0.5:
        top, left = random.randint(0, height - size), random.randint(0, width - size)
    else:
        # contrast-guided: pick a random pixel among the top 10% by local
        # contrast and bias the window toward it
        contrast = (infrared - F.avg_pool2d(infrared[None], 15, stride=1, padding=7)[0]).abs()[0]
        candidates = (contrast >= torch.quantile(contrast.flatten(), 0.90)).nonzero()
        y, x = candidates[random.randrange(len(candidates))].tolist()
        top = min(max(y - random.randrange(size), 0), height - size)
        left = min(max(x - random.randrange(size), 0), width - size)
    # both images use the same window so the pixels stay aligned
    return visible[..., top : top + size, left : left + size], infrared[..., top : top + size, left : left + size]


class PairedFusionDataset(Dataset):
    """Paired dataset cycling each sample through four quality states.

    The state of sample ``index`` at epoch ``epoch`` is
    ``(index + epoch) % 4``. ``__getitem__`` returns both the clean pair
    (``visible``/``infrared``) and the observed pair
    (``visible_observed``/``infrared_observed``); they are identical in the
    clean state. The noise seed ``seed + epoch * 1_000_003 + index`` keeps
    noise reproducible within an epoch and non-repeating across epochs.
    """

    STATE_NAMES = ("both_clean", "visible_noisy", "infrared_noisy", "both_noisy")
    def __init__(self, pairs: list[tuple[Path, Path]], crop_size: int = 192, training: bool = True,
                 corruption: bool = True, severity: str = "train", seed: int = 3407) -> None:
        """Initialize the dataset.

        ``corruption=False`` forces the clean state; ``severity`` selects
        the ``"train"`` or ``"hard"`` degradation ranges.
        """
        self.pairs, self.crop_size, self.training = pairs, crop_size, training
        self.corruption, self.seed, self.epoch = corruption, seed, 0

        self.visible_noise, self.infrared_noise = VisibleNoise(severity), InfraredNoise(severity)

    def __len__(self) -> int:
        """Number of pairs."""
        return len(self.pairs)

    def state_for_index(self, index: int) -> int:
        """Quality state (0-3) of sample ``index`` in the current epoch."""
        return (index + self.epoch) % 4 if self.corruption else 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch; call before each fresh DataLoader iteration."""
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """Load a pair, crop it aligned, and apply the state's degradations.

        Returns the clean pair, the observed pair, degradation masks
        (``visible_damage``/``infrared_damage``, ``infrared_stripe``,
        ``infrared_bad_pixel``), degradation parameter scalars
        (``visible_photons``, ``visible_read_sigma``,
        ``infrared_gaussian_sigma``), ``quality_state``, and ``name``.
        """
        vis_path, ir_path = self.pairs[index]
        visible, infrared = load_image(vis_path, "RGB"), load_image(ir_path, "L")
        if infrared.shape[-2:] != visible.shape[-2:]:
            infrared = F.interpolate(infrared[None], size=visible.shape[-2:], mode="bilinear", align_corners=False)[0]
        visible, infrared = _guided_crop(visible, infrared, self.crop_size, self.training)
        state = self.state_for_index(index)
        # deterministic per-sample, per-epoch noise
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + index)
        visible_obs, infrared_obs = visible, infrared
        visible_meta = {"damage": torch.zeros(1, *visible.shape[-2:]),
                        "photons": torch.tensor(1e6), "read_sigma": torch.tensor(0.0)}
        infrared_meta = {"damage": torch.zeros_like(infrared), "stripe": torch.zeros_like(infrared), "bad_pixel": torch.zeros_like(infrared),
                         "gaussian_sigma": torch.tensor(0.0)}

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
    """Batch sampler with exactly ``batch_size // 4`` samples per quality state.

    Samples are grouped by their current state, shuffled per group, and
    dealt round-robin into batches; smaller groups wrap around so the
    per-state quotas stay exact. ``batch_size`` must be divisible by four.
    """

    def __init__(self, dataset: PairedFusionDataset, batch_size: int, seed: int = 3407) -> None:
        """Initialize the sampler; the shuffle seed advances with each epoch."""
        if batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch; keep in sync with the dataset."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices for the current epoch."""
        groups = [[i for i in range(len(self.dataset)) if self.dataset.state_for_index(i) == state] for state in range(4)]
        rng = random.Random(self.seed + self.epoch)
        for group in groups:
            rng.shuffle(group)
        per_state = self.batch_size // 4
        # batch count follows the largest group; smaller ones wrap around
        batches = max(math.ceil(len(group) / per_state) for group in groups)
        for batch_id in range(batches):
            batch = []
            for group in groups:
                start = batch_id * per_state
                batch.extend(group[(start + offset) % len(group)] for offset in range(per_state))
            # shuffle within the batch so states do not sit in fixed slots
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Batches per epoch (from the largest state group)."""
        per_state = self.batch_size // 4
        counts = [sum(self.dataset.state_for_index(i) == state for i in range(len(self.dataset))) for state in range(4)]
        return max(math.ceil(count / per_state) for count in counts)
