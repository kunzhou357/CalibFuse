"""CalibFuse 训练入口。

默认配置：100 个 epoch，裁剪 256，batch size 4，种子 3407，
AdamW（lr 2e-4，余弦退火至 1e-6，5 个 epoch 线性预热，权重衰减 1e-4），
CUDA BF16。

用法::

    python train.py --data datasets/train --output checkpoints/train
    python train.py --data datasets/train --output checkpoints/train \\
        --resume checkpoints/train/latest.pth
    python train.py --output checkpoints/smoke --device cpu \\
        --crop-size 32 --max-batches 1        # 冒烟测试

输出（均在 --output 下）：``latest.pth``（每个 epoch 覆盖）、
``epoch_NNN.pth``（每 10 个 epoch）、``previews/epoch_NNN.png``
（EMA 教师的融合预览）、``log.txt``（逐 epoch 统计）。

本文件强制的不变量：
1. 每个 epoch 在 ``set_epoch`` 之后重建 worker，使逐图像的噪声种子
   永不重复；缺少该策略的 checkpoint 会被拒绝加载。
2. Resume 要求存储的超参与命令行完全一致，并恢复保存的 RNG 状态，
   使训练可精确续跑。
3. 目标目录已存在 ``latest.pth`` 且未指定 --resume 时直接报错
   （拒绝意外覆盖）。
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
    """解析命令行参数；默认值即标准训练配置。

    关键约束（main 中校验）：
    - --batch-size 必须为正且能被 4 整除（四状态均衡采样）；
    - --crop-size 至少 16 且能被 4 整除（三尺度下采样对齐）；
    - --max-batches 仅用于冒烟测试，省略则完整训练。
    """
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
    """播种所有随机源（python / numpy / torch / cuda），保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CleanEMA:
    """学生权重的指数滑动平均（干净教师）。

    教师从不直接训练；它以
    ``decay = min(0.99, (1 + updates) / (10 + updates))``
    跟随学生——前期快、后期平滑——并在训练中提供干净参考特征。
    推理默认使用 EMA 权重（``--weights ema``）。

    动态衰减的含义：初始 updates 小时 decay ≈ updates/(updates+10)
    较小（教师快速跟上学生），随更新次数增长 decay 趋近 0.99
    （教师越来越平滑）。
    """

    def __init__(self, student: nn.Module, decay: float = 0.99) -> None:
        """深拷贝学生作为初始教师，设为 eval 且不参与求导。"""
        self.model = deepcopy(student).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """teacher <- decay * teacher + (1 - decay) * student；buffer 直接拷贝。

        fp16 AMP 下，若 GradScaler 跳过了优化器步进，
        调用方会跳过本次 EMA 更新——教师绝不吸收被跳过的更新
        （保持学生与教师状态的一致性）。
        """
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for teacher, source in zip(self.model.parameters(), student.parameters()):
            # lerp_：以 (1 - decay) 的权重把学生插值进教师
            teacher.lerp_(source.detach(), 1.0 - decay)
        for teacher, source in zip(self.model.buffers(), student.buffers()):
            teacher.copy_(source)
        self.model.eval()

    def state_dict(self) -> dict:
        """序列化教师权重与 EMA 状态（存于 checkpoint 的 ``ema`` 字段）。"""
        return {"model": self.model.state_dict(), "decay": self.decay, "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        """恢复教师权重与计数器。"""
        self.model.load_state_dict(state["model"], strict=True)
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])


def snapshot_rng() -> dict:
    """捕获所有全局 RNG 状态，使续跑的训练与原运行逐位一致。

    覆盖 Python random、NumPy、torch CPU 以及 CUDA 的全部 RNG 状态。
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    """恢复 :func:`snapshot_rng` 捕获的 RNG 状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, epoch: int, stats: dict[str, float], args: argparse.Namespace) -> None:
    """保存完整训练 checkpoint（权重、各状态、RNG 与命令行参数）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT, "epoch": epoch, "stats": stats,
        "data_epoch_policy": DATA_EPOCH_POLICY,
        "model_config": model.config, "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": snapshot_rng(),
        # 路径转成字符串，保持载荷可移植（跨目录/机器加载）
        "train_args": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
    }, path)


def load_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, device: torch.device) -> int:
    """恢复完整训练状态；返回起始 epoch（存储值 + 1）。

    在 CPU 上加载（优化器/调度器状态会自动重铸到各自参数所在
    设备），并在存在时恢复保存的 RNG 状态。

    兼容性校验：format 与 data_epoch_policy 不符时直接报错——
    worker 修复之前的旧 checkpoint 一律拒绝。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    if payload.get("rng") is not None:
        restore_rng(payload["rng"])
    return int(payload["epoch"]) + 1


def cosine_multiplier(epoch: int, epochs: int, minimum_ratio: float, warmup: int = 5) -> float:
    """学习率乘子：线性预热，随后余弦退火到 ``minimum_ratio``。

    - epoch < warmup：乘子 = (epoch + 1) / warmup，从 1/warmup 线性升到 1；
    - 之后：标准余弦从 1 退火到 minimum_ratio。
    """
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(max((epoch - warmup) / max(epochs - warmup - 1, 1), 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def move_batch(batch: dict, device: torch.device) -> dict:
    """把 batch 中的张量搬到目标设备；其余字段（如文件名）原样透传。"""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_train_loader(dataset, sampler, workers: int, device: torch.device) -> DataLoader:
    """构建训练 DataLoader（禁用持久化 worker）。

    worker 必须每个 epoch 重建：持久化 worker 会停留在创建时的
    第一个 epoch，导致逐图像的噪声种子重复（退化分布塌缩）。
    cuda 时开启 pin_memory 加速 H2D 拷贝。
    """
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=False)


def train_epoch(model: CalibFuse, ema: CleanEMA, loader: DataLoader,
                criterion: CalibFuseLoss, optimizer, scaler, device: torch.device,
                amp_dtype: torch.dtype | None, epoch: int, epochs: int,
                max_batches: int | None = None) -> dict[str, float]:
    """训练一个 epoch；返回按样本数加权的平均损失统计。"""
    model.train()
    # 学生与教师的跨模态预算同步推进，保证预览与训练行为一致
    model.set_training_progress(epoch)
    ema.model.set_training_progress(epoch)
    totals: dict[str, float] = {}
    samples = 0
    progress = tqdm(loader, desc=f"train {epoch + 1:03d}/{epochs}", ncols=100)
    for batch_index, batch in enumerate(progress):
        # 冒烟测试的批数上限
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None and device.type == "cuda"):
            # 网络看到的是退化观测；干净图像与 EMA 教师驱动辅助监督
            # （字典恢复损失、采纳/交互校准、误差校准等）
            output = model(batch["visible_observed"], batch["infrared_observed"],
                           clean_visible=batch["visible"], clean_infrared=batch["infrared"],
                           return_auxiliary=True, clean_teacher=ema.model)
            losses = criterion(output, batch)
        # 非有限损失立即终止（防 NaN 扩散到权重）
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
        if scaler is None:
            # bf16 / 无 AMP：直接反向、梯度裁剪、步进、EMA 更新
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
        else:
            # fp16：优化器步进被 GradScaler 跳过时，同步跳过 EMA 更新
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # scale 未缩小 <=> 本步未被跳过
            if scaler.get_scale() >= old_scale:
                ema.update(model)
        count = batch["visible"].shape[0]
        samples += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
        progress.set_postfix(loss=f"{losses['total'].item():.4f}")
    # 按样本数加权平均（最后一个 batch 可能不满）
    return {name: value / max(samples, 1) for name, value in totals.items()}


def main() -> None:
    """校验参数、构建训练管线、可选续跑，然后开始训练。"""
    args = parse_args()
    # CUDA 是默认设备；缺失时明确报错，避免静默落到 CPU
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for smoke tests")
    # 四状态均衡采样要求 batch 能被 4 整除
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by four")
    # 三尺度（两次 2x 下采样）要求 crop 为 4 的倍数且不太小
    if args.crop_size < 16 or args.crop_size % 4:
        raise ValueError("--crop-size must be at least 16 and divisible by four")
    if args.epochs <= 0 or args.workers < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        raise ValueError("Invalid epochs, workers, or learning rate")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # cuDNN 自动调优 + TF32 矩阵运算：Ampere+ 上的吞吐优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; use --amp fp16")
    # GradScaler 只在 fp16 时需要；bf16 / 无 AMP 直接反向传播
    scaler = (torch.amp.GradScaler("cuda")
              if device.type == "cuda" and amp_dtype == torch.float16 else None)

    pairs = paired_paths(args.data)
    # 四状态均衡训练至少需要 4 对（每组状态 1 个样本）
    if len(pairs) < 4:
        raise ValueError("Balanced four-state training requires at least four image pairs")
    dataset = PairedFusionDataset(pairs, args.crop_size, training=True, corruption=True,
                                  severity="train", seed=args.seed)
    sampler = QualityBalancedBatchSampler(dataset, args.batch_size, seed=args.seed)
    loader = make_train_loader(dataset, sampler, args.workers, device)
    model = CalibFuse().to(device)
    if args.resume:
        # resume 必须保持训练配置一致，否则拒绝（防止静默改变行为）
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = resumed.get("train_args", {})
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
    # 已有 latest.pth 且未指定 --resume：报错而非覆盖（防误删训练）
    if not args.resume and (args.output / "latest.pth").exists():
        raise FileExistsError(f"{args.output}/latest.pth already exists; choose a new --output "
                              "or explicitly resume a checkpoint created after the epoch fix")
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    # 单进程预览 loader：与训练相同的数据集与采样器（同一 epoch 状态），
    # 但 num_workers=0，不参与 worker 重建
    preview_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    print(f"device={device} amp={args.amp} pairs={len(pairs)} batches={len(loader)} "
          f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    for epoch in range(start_epoch, args.epochs):
        # 先推进 epoch 再迭代：新建的 worker 一定看到当前 epoch
        # （这是 DATA_EPOCH_POLICY 契约的核心动作）
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        stats = train_epoch(model, ema, loader, criterion, optimizer, scaler,
                            device, amp_dtype, epoch, args.epochs, args.max_batches)
        scheduler.step()
        preview_batch = move_batch(next(iter(preview_loader)), device)
        ema.model.eval()
        with torch.inference_mode():
            # 预览用 EMA 教师生成（教师即推理权重的来源）
            preview = ema.model(preview_batch["visible_observed"],
                                preview_batch["infrared_observed"])["fused"]
        save_tensor(preview_dir / f"epoch_{epoch + 1:03d}.png", preview)
        # 在预览之后保存 checkpoint：存储的 RNG 状态覆盖完整 epoch，
        # 使 resume 后的预览序列也与原运行一致
        save_checkpoint(args.output / "latest.pth", model, ema, optimizer, scheduler,
                        scaler, epoch, stats, args)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output / f"epoch_{epoch + 1:03d}.pth", model, ema, optimizer,
                            scheduler, scaler, epoch, stats, args)
        # 逐 epoch 日志：calib 为三项校准正则之和
        line = (f"epoch={epoch + 1:03d} total={stats['total']:.5f} fusion={stats['fusion']:.5f} "
                f"recovery={stats['recovery']:.5f} anchor={stats['anchor']:.5f} "
                f"calib={stats['adoption'] + stats['error_calibration'] + stats['interaction']:.5f} "
                f"harmful={stats['harmful_correction_rate']:.4f}")
        print(line)
        with open(args.output / "log.txt", "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
