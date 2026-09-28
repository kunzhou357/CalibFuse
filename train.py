"""CalibFuse training entry point.

Default configuration: 100 epochs, crop 256, batch size 4, seed 3407,
AdamW (lr 2e-4 cosine-annealed to 1e-6 with 5 warmup epochs, weight decay
1e-4), CUDA BF16.

Usage::

    python train.py --data datasets/train --output checkpoints/train
    python train.py --data datasets/train --output checkpoints/train \\
        --resume checkpoints/train/latest.pth
    python train.py --output checkpoints/smoke --device cpu \\
        --crop-size 32 --max-batches 1        # smoke test

Outputs (under --output): ``latest.pth`` overwritten every epoch,
``epoch_NNN.pth`` every 10 epochs, ``previews/epoch_NNN.png`` (EMA-teacher
fusion previews), and ``log.txt`` (per-epoch statistics).

Invariants enforced here:
1. Workers are recreated each epoch after ``set_epoch`` so per-image noise
   seeds never repeat; checkpoints without this policy are rejected.
2. Resume rejects fine-tuning runs and requires the stored hyperparameters
   to match the command line.
3. An existing ``latest.pth`` without --resume is an error.
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.fusion import CalibFuse
from utils.dataset import PairedFusionDataset, QualityBalancedBatchSampler, paired_paths
from utils.image import save_tensor
from utils.loss import CalibFuseLoss
from utils.checkpoint import CHECKPOINT_FORMAT, DATA_EPOCH_POLICY


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments; defaults are the standard configuration."""
    parser = argparse.ArgumentParser(description="Train the CalibFuse fusion model")
    parser.add_argument("--data", type=Path, default=Path("datasets/train"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/train"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-batches", type=int,
                        help="Limit each epoch for smoke tests; omit for full training")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CleanEMA:
    """Exponential moving average of the student weights (clean teacher).

    The teacher never trains; it follows the student with
    ``decay = min(0.99, (1 + updates) / (10 + updates))`` — fast early,
    smooth later — and provides clean reference features during training.
    Inference defaults to the EMA weights.
    """

    def __init__(self, student: nn.Module, decay: float = 0.99) -> None:
        """Deep-copy the student as the initial teacher."""
        self.model = deepcopy(student).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """teacher <- decay * teacher + (1 - decay) * student; buffers are copied.

        Under fp16 AMP the caller skips this whenever GradScaler skipped the
        optimizer step, so the teacher never absorbs a skipped update.
        """
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for teacher, source in zip(self.model.parameters(), student.parameters()):
            teacher.lerp_(source.detach(), 1.0 - decay)
        for teacher, source in zip(self.model.buffers(), student.buffers()):
            teacher.copy_(source)
        self.model.eval()

    def state_dict(self) -> dict:
        """Serialize teacher weights and EMA state (stored under ``ema``)."""
        return {"model": self.model.state_dict(), "decay": self.decay, "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        """Restore teacher weights and counters."""
        self.model.load_state_dict(state["model"], strict=True)
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])


def save_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, epoch: int, stats: dict[str, float], args: argparse.Namespace) -> None:
    """Save a full training checkpoint (weights, states, and an args snapshot)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT, "epoch": epoch, "stats": stats,
        "data_epoch_policy": DATA_EPOCH_POLICY,
        "training_epoch_offset": getattr(model, "training_epoch_offset", 0),
        "model_config": model.config, "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        # paths are stored as strings so the payload stays portable
        "train_args": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
    }, path)


def load_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, device: torch.device) -> int:
    """Restore the full training state; returns the start epoch (stored + 1)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"incompatible checkpoint format: {payload.get('format')!r}")
    if payload.get("data_epoch_policy") != DATA_EPOCH_POLICY:
        raise ValueError("This checkpoint predates the worker epoch fix. Start a new training "
                         "run in a new output directory.")
    model.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload["epoch"]) + 1


def cosine_multiplier(epoch: int, epochs: int, minimum_ratio: float, warmup: int = 5) -> float:
    """LR multiplier: linear warmup, then cosine decay to ``minimum_ratio``."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(max((epoch - warmup) / max(epochs - warmup - 1, 1), 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to the device; other fields pass through."""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_train_loader(dataset, sampler, workers: int, device: torch.device) -> DataLoader:
    """Build the training DataLoader with persistent workers disabled.

    Workers must be recreated each epoch so they observe the current dataset
    epoch; persistent workers would retain the first epoch and repeat
    per-image noise seeds.
    """
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=False)


def train_epoch(model: CalibFuse, ema: CleanEMA, loader: DataLoader,
                criterion: CalibFuseLoss, optimizer, scaler, device: torch.device,
                amp_dtype: torch.dtype | None, epoch: int, epochs: int,
                max_batches: int | None = None) -> dict[str, float]:
    """Train one epoch; returns sample-weighted mean loss statistics."""
    model.train()
    # the budget ramp uses total progress epochs so warm starts stay continuous
    model.set_training_progress(epoch + getattr(model, "training_epoch_offset", 0))
    totals: dict[str, float] = {}
    samples = 0
    progress = tqdm(loader, desc=f"train {epoch + 1:03d}/{epochs}", ncols=100)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None and device.type == "cuda"):
            # the network sees the degraded observations; the clean images and
            # the EMA teacher drive the auxiliary supervision
            output = model(batch["visible_observed"], batch["infrared_observed"],
                           clean_visible=batch["visible"], clean_infrared=batch["infrared"],
                           return_auxiliary=True, clean_teacher=ema.model)
            losses = criterion(output, batch)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
        if scaler is None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
        else:
            # fp16: skip the EMA update when the optimizer step was skipped
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
        count = batch["visible"].shape[0]
        samples += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
        progress.set_postfix(loss=f"{losses['total'].item():.4f}")
    return {name: value / max(samples, 1) for name, value in totals.items()}


def main() -> None:
    """Validate arguments, build the pipeline, optionally resume, then train."""
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for smoke tests")
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by four")
    if args.crop_size < 16 or args.crop_size % 4:
        raise ValueError("--crop-size must be at least 16 and divisible by four")
    if args.epochs <= 0 or args.workers < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        raise ValueError("Invalid epochs, workers, or learning rate")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; use --amp fp16")
    # GradScaler is only needed for fp16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    pairs = paired_paths(args.data)
    if len(pairs) < 4:
        raise ValueError("Balanced four-state training requires at least four image pairs")
    dataset = PairedFusionDataset(pairs, args.crop_size, training=True, corruption=True,
                                  severity="train", seed=args.seed)
    sampler = QualityBalancedBatchSampler(dataset, args.batch_size, seed=args.seed)
    loader = make_train_loader(dataset, sampler, args.workers, device)
    model = CalibFuse().to(device)
    if args.resume:
        # reject fine-tuning runs and require the stored hyperparameters to match
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = resumed.get("train_args", {})
        if (resume_args.get("finetune") or resumed.get("training_epoch_offset", 0) != 0
                or resume_args.get("structure_weight", 1.0) != 1.0):
            raise ValueError("The trainer cannot resume a fine-tuning run")
        for key in ("epochs", "batch_size", "crop_size", "lr", "min_lr", "weight_decay", "seed", "amp"):
            if key in resume_args and getattr(args, key) != resume_args[key]:
                raise ValueError(f"Resume must preserve --{key.replace('_', '-')}: {resume_args[key]}")
    ema = CleanEMA(model)
    criterion = CalibFuseLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: cosine_multiplier(epoch, args.epochs, args.min_lr / args.lr))
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, ema, optimizer, scheduler, scaler, device)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output / "latest.pth").exists():
        raise FileExistsError(f"{args.output}/latest.pth already exists; choose a new --output "
                              "or explicitly resume a checkpoint created after the epoch fix")
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    print(f"device={device} amp={args.amp} pairs={len(pairs)} batches={len(loader)} "
          f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs):
        # advance the epoch before iterating so freshly created workers always
        # see the current epoch
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        stats = train_epoch(model, ema, loader, criterion, optimizer, scaler,
                            device, amp_dtype, epoch, args.epochs, args.max_batches)
        scheduler.step()
        save_checkpoint(args.output / "latest.pth", model, ema, optimizer, scheduler,
                        scaler, epoch, stats, args)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output / f"epoch_{epoch + 1:03d}.pth", model, ema, optimizer,
                            scheduler, scaler, epoch, stats, args)
        preview_batch = move_batch(next(iter(loader)), device)
        ema.model.eval()
        with torch.inference_mode():
            preview = ema.model(preview_batch["visible_observed"],
                                preview_batch["infrared_observed"])["fused"]
        save_tensor(preview_dir / f"epoch_{epoch + 1:03d}.png", preview)
        line = (f"epoch={epoch + 1:03d} total={stats['total']:.5f} fusion={stats['fusion']:.5f} "
                f"recovery={stats['recovery']:.5f} anchor={stats['anchor']:.5f} "
                f"calib={stats['adoption'] + stats['error_calibration'] + stats['interaction']:.5f} "
                f"harmful={stats['harmful_correction_rate']:.4f}")
        print(line)
        with open(args.output / "log.txt", "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
