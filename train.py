"""CalibFuse 训练入口。

默认训练配置：100 epochs、crop 256、batch 4、seed 3407、
AdamW（lr 2e-4 → min 1e-6 余弦退火 + 5 轮 warmup、weight decay 1e-4）、
CUDA BF16。

用法::

    python train.py --data datasets/train --output checkpoints/train
    python train.py --data datasets/train --output checkpoints/train \\
        --resume checkpoints/train/latest.pth
    python train.py --output checkpoints/smoke --device cpu \\
        --crop-size 32 --max-batches 1        # 冒烟测试

输出（全部在 --output 下）：每轮覆写 ``latest.pth``；每 10 轮另存
``epoch_NNN.pth``；``previews/epoch_NNN.png``（EMA 教师的融合预览）；
``log.txt``（逐轮指标行）。

三条硬性纪律（改动前必读）：
1. **worker 纪律**：每轮先 ``dataset.set_epoch``/``sampler.set_epoch``
   再重新创建 DataLoader 迭代器（persistent_workers=False），保证
   worker 不持有过期 epoch、噪声种子不重复。检查点带
   ``data_epoch_policy`` 标记，旧策略检查点会被拒载；
2. **恢复守卫**：拒绝恢复微调实验（finetune/epoch_offset/
   structure_weight 标记），且 8 个关键超参必须与存档一致；
3. **不覆盖**：目标目录已有 ``latest.pth`` 且未指定 --resume 时
   直接报错，防止误覆盖训练结果。
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
    """解析命令行参数（默认值即标准训练配置）。

    返回:
        argparse.Namespace，主要字段见 ``--help``。
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
    """固定所有随机源（random/numpy/torch/CUDA），保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CleanEMA:
    """干净教师：学生权重的指数滑动平均（EMA）副本。

    教师不训练、不回传梯度（``requires_grad_(False)`` + eval），
    仅随学生更新缓慢跟随。训练中通过
    ``model(..., clean_teacher=ema.model)`` 提供干净的参考特征，
    推理默认使用 EMA 权重（更稳）。

    衰减策略：``decay = min(0.99, (1+updates)/(10+updates))``
    ——早期衰减小（跟随快，热身），后期收敛到 0.99（平滑强）。
    """

    def __init__(self, student: nn.Module, decay: float = 0.99) -> None:
        """深拷贝学生作为教师初值。"""
        self.model = deepcopy(student).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """每次优化器步进后调用：教师 ← decay·教师 + (1-decay)·学生。

        buffers（如无参数统计）直接拷贝学生值。
        注意：fp16 AMP 下若 GradScaler 本步跳过了优化器步进
        （出现溢出），调用方也不会调用本方法——教师不会吃进
        被跳过的更新。
        """
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for teacher, source in zip(self.model.parameters(), student.parameters()):
            teacher.lerp_(source.detach(), 1.0 - decay)
        for teacher, source in zip(self.model.buffers(), student.buffers()):
            teacher.copy_(source)
        # 保持 eval（防 dropout/BN 状态意外漂移）
        self.model.eval()

    def state_dict(self) -> dict:
        """序列化教师权重与 EMA 状态（存进 checkpoint 的 ``ema`` 键）。"""
        return {"model": self.model.state_dict(), "decay": self.decay, "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        """恢复教师权重与计数。"""
        self.model.load_state_dict(state["model"], strict=True)
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])


def save_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, epoch: int, stats: dict[str, float], args: argparse.Namespace) -> None:
    """保存一个完整训练检查点。

    载荷字段：格式标识、epoch、本轮统计、数据 epoch 策略、
    training_epoch_offset（热启动时保持预算 ramp 连续）、模型配置、
    学生/教师权重、优化器/调度器/scaler 状态、命令行参数快照
    （供恢复时的一致性校验）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CHECKPOINT_FORMAT, "epoch": epoch, "stats": stats,
        "data_epoch_policy": DATA_EPOCH_POLICY,
        "training_epoch_offset": getattr(model, "training_epoch_offset", 0),
        "model_config": model.config, "model": model.state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        # Path 转字符串，保证载荷可移植
        "train_args": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
    }, path)


def load_checkpoint(path: Path, model: CalibFuse, ema: CleanEMA, optimizer,
                    scheduler, scaler, device: torch.device) -> int:
    """恢复完整训练状态，返回起始 epoch（存档 epoch + 1）。

    与 utils/checkpoint.load_model 一样做格式与数据策略校验；
    不满足数据策略（worker 每轮重建）的检查点会被拒绝并提示
    从头开始。
    """
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
    """学习率乘数：5 轮线性 warmup + 余弦退火到 minimum_ratio。

    warmup 阶段 (0, 1/w, 2/w, …, 1)；之后从 1 余弦衰减到
    ``minimum_ratio``（= min_lr / lr）。供 LambdaLR 使用。

    参数:
        epoch: 当前轮（0 基）。
        epochs: 总轮数。
        minimum_ratio: 退火终点的比率下限。
        warmup: 线性热身轮数，默认 5。
    """
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(max((epoch - warmup) / max(epochs - warmup - 1, 1), 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def move_batch(batch: dict, device: torch.device) -> dict:
    """把批次里的张量搬到设备（非张量字段原样保留）。"""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_train_loader(dataset, sampler, workers: int, device: torch.device) -> DataLoader:
    """构建训练 DataLoader（**禁用 persistent_workers**）。

    Workers receive the dataset epoch when each iterator is created. Persistent
    worker copies would retain the first epoch and repeat per-image noise seeds.

    每个 epoch 开始时 train.py 先 set_epoch 再重新迭代（重新创建
    loader），worker 拿到的都是最新 epoch。若改为 persistent workers，
    worker 内的数据集副本会保留第一个 epoch——噪声种子会整轮重复。
    详见 tests/test_worker_epoch.py。
    """
    # Workers receive the dataset epoch when each iterator is created. Persistent
    # worker copies would retain the first epoch and repeat per-image noise seeds.
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=False)


def train_epoch(model: CalibFuse, ema: CleanEMA, loader: DataLoader,
                criterion: CalibFuseLoss, optimizer, scaler, device: torch.device,
                amp_dtype: torch.dtype | None, epoch: int, epochs: int,
                max_batches: int | None = None) -> dict[str, float]:
    """训练一个 epoch，返回按样本数加权平均的损失统计。

    流程（每个 batch）：
    1. 观测图（带噪）进网络、干净图给 EMA 教师提取参考特征；
    2. autocast（bf16/fp16，仅 CUDA）下前向 + 计算损失；
    3. 非有限损失直接抛错（早发现数值问题）；
    4. 反传 → 梯度裁剪（范数 1.0）→ 优化器步进 → EMA 更新；
       fp16 时经 GradScaler，且**只有 scaler 未降尺度（即本步
       未被跳过）才更新 EMA**；
    5. 损失按 batch 大小加权累计，最后除以总样本数。

    参数:
        model/ema/criterion/optimizer/scaler: 训练组件。
        loader: 训练 DataLoader（batch_sampler 模式）。
        device: 计算设备。
        amp_dtype: autocast 的 dtype；None 表示禁用 AMP。
        epoch/epochs: 当轮/总轮数（进度条与预算 ramp 用）。
        max_batches: 每轮最多跑多少批（冒烟测试用）。

    返回:
        ``{损失名: 平均值}`` 字典。
    """
    model.train()
    # 预算 ramp 用的是"总进度 epoch"：热启动时加上偏移保持连续
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
            # 网络吃"观测图"（带噪），干净图与 EMA 教师用于辅助监督
            output = model(batch["visible_observed"], batch["infrared_observed"],
                           clean_visible=batch["visible"], clean_infrared=batch["infrared"],
                           return_auxiliary=True, clean_teacher=ema.model)
            losses = criterion(output, batch)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
        if scaler is None:
            # bf16 / 无 AMP：直接反传
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
        else:
            # fp16：缩放反传 → 反缩放 → 裁剪 → 步进；
            # 若 scale 变小（说明发生溢出、步进被跳过），跳过 EMA
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
        # 按样本数加权累计各项损失（含两个诊断量）
        count = batch["visible"].shape[0]
        samples += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
        progress.set_postfix(loss=f"{losses['total'].item():.4f}")
    return {name: value / max(samples, 1) for name, value in totals.items()}


def main() -> None:
    """训练主流程：校验 → 构建 → （可选）恢复 → 逐轮训练保存。"""
    args = parse_args()
    # —— 参数合法性校验（提前失败，避免跑到一半才报错）——
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for smoke tests")
    # batch 须被 4 整除：QualityBalancedBatchSampler 每状态配额 batch/4
    if args.batch_size <= 0 or args.batch_size % 4:
        raise ValueError("--batch-size must be positive and divisible by four")
    # crop 须被 4 整除（网络两次 2x 下采样）且不太小
    if args.crop_size < 16 or args.crop_size % 4:
        raise ValueError("--crop-size must be at least 16 and divisible by four")
    if args.epochs <= 0 or args.workers < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr:
        raise ValueError("Invalid epochs, workers, or learning rate")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # 固定输入尺寸下 cudnn 自动调优 + 允许 TF32 提速
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    if device.type == "cuda" and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bf16; use --amp fp16")
    # fp16 才需要 GradScaler（bf16 不需要缩放）
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    # —— 数据集与加载器 ——
    pairs = paired_paths(args.data)
    if len(pairs) < 4:
        raise ValueError("Balanced four-state training requires at least four image pairs")
    dataset = PairedFusionDataset(pairs, args.crop_size, training=True, corruption=True,
                                  severity="train", seed=args.seed)
    sampler = QualityBalancedBatchSampler(dataset, args.batch_size, seed=args.seed)
    loader = make_train_loader(dataset, sampler, args.workers, device)
    # —— 模型 / 损失 / 优化器 / 调度器 ——
    model = CalibFuse().to(device)
    if args.resume:
        # 恢复守卫第一步：检查存档是否来自微调实验（拒绝），
        # 并要求关键超参与命令行完全一致
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
    # 余弦退火乘数调度器：min_lr/lr 为终点比率
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: cosine_multiplier(epoch, args.epochs, args.min_lr / args.lr))
    start_epoch = 0
    if args.resume:
        # 恢复守卫第二步：完整恢复（权重/优化器/调度器/scaler）
        start_epoch = load_checkpoint(args.resume, model, ema, optimizer, scheduler, scaler, device)
    args.output.mkdir(parents=True, exist_ok=True)
    # 防覆盖：非恢复模式下目标目录已有 latest.pth 即报错
    if not args.resume and (args.output / "latest.pth").exists():
        raise FileExistsError(f"{args.output}/latest.pth already exists; choose a new --output "
                              "or explicitly resume a checkpoint created after the epoch fix")
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    print(f"device={device} amp={args.amp} pairs={len(pairs)} batches={len(loader)} "
          f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    # —— 主训练循环 ——
    for epoch in range(start_epoch, args.epochs):
        # 关键顺序：先推进 epoch，再迭代（loader 每轮新建 worker，
        # worker 中的数据集副本因而总是最新 epoch）
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        stats = train_epoch(model, ema, loader, criterion, optimizer, scaler,
                            device, amp_dtype, epoch, args.epochs, args.max_batches)
        scheduler.step()
        # 每轮覆写 latest.pth
        save_checkpoint(args.output / "latest.pth", model, ema, optimizer, scheduler,
                        scaler, epoch, stats, args)
        # 每 10 轮另存一份编号检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(args.output / f"epoch_{epoch + 1:03d}.pth", model, ema, optimizer,
                            scheduler, scaler, epoch, stats, args)
        # 预览：取本 epoch 第一个 batch，用 EMA 教师推理存图
        preview_batch = move_batch(next(iter(loader)), device)
        ema.model.eval()
        with torch.inference_mode():
            preview = ema.model(preview_batch["visible_observed"],
                                preview_batch["infrared_observed"])["fused"]
        save_tensor(preview_dir / f"epoch_{epoch + 1:03d}.png", preview)
        # 日志行：同时打印到控制台与 log.txt
        line = (f"epoch={epoch + 1:03d} total={stats['total']:.5f} fusion={stats['fusion']:.5f} "
                f"recovery={stats['recovery']:.5f} anchor={stats['anchor']:.5f} "
                f"calib={stats['adoption'] + stats['error_calibration'] + stats['interaction']:.5f} "
                f"harmful={stats['harmful_correction_rate']:.4f}")
        print(line)
        with open(args.output / "log.txt", "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
