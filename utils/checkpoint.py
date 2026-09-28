"""Checkpoint compatibility and provenance shared by inference entry points.

checkpoint 兼容性与来源校验（被所有推理入口共享）。

本文件是三个入口脚本（``infer.py``/``test.py``/``diagnose.py``）与
``train.py`` 共用的检查点工具，集中定义了三个协议常量：

- :data:`CHECKPOINT_FORMAT`：检查点文件格式标识；
- :data:`DATA_EPOCH_POLICY`：数据加载器的 epoch 策略标识
  （worker 每轮重建，拒绝修复前的旧检查点）；
- :data:`METRIC_PROTOCOL`：指标协议标识（calibfuse-metrics-v1）。

修改任一常量都会使旧检查点/旧指标不可比，属于协议变更，需要新版本号。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from nets.fusion import CalibFuse

# 检查点格式标识：载荷中必须携带相同值才允许加载
CHECKPOINT_FORMAT = "calibfuse-benefit-calibrated-v1"
# 数据 epoch 策略：每轮重建 DataLoader worker（修复前旧检查点会被拒载）
DATA_EPOCH_POLICY = "recreate-workers-each-epoch-v1"
# 指标协议：指标实现的约定集合，详见 utils/evaluator.py 模块说明
METRIC_PROTOCOL = "calibfuse-metrics-v1"


def sha256_file(path: str | Path) -> str:
    """计算文件的 SHA-256 十六进制摘要（流式读取，适合大文件）。

    用于把检查点哈希写进 ``protocol.json``/``diagnostics.json``，
    保证报告可与特定权重文件一一对应。

    参数:
        path: 文件路径。

    返回:
        64 个十六进制字符的摘要字符串。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        # 每次读 1 MiB，避免一次性载入整个文件
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device, weights: str = "ema") -> tuple[CalibFuse, dict]:
    """按严格校验加载检查点并重建模型。

    校验流程（任何一步失败即抛异常）：
    1. ``weights`` 只能是 ``"ema"``（默认，训练时维护的指数滑动平均
       权重，推理更稳）或 ``"model"``（最终学生权重）；
    2. 请求 CUDA 但不可用时提示改用 ``--device cpu``；
    3. 载荷的 ``format`` 必须等于 :data:`CHECKPOINT_FORMAT`；
    4. 载荷的 ``data_epoch_policy`` 必须等于 :data:`DATA_EPOCH_POLICY`
       ——不满足该策略的旧检查点（worker 持有过期 epoch、噪声种子
       重复）会被直接拒绝；
    5. 模型结构由载荷中的 ``model_config`` 重建（不依赖代码默认值），
       state-dict 以 ``strict=True`` 加载，键名必须完全匹配。

    参数:
        path: 检查点文件路径。
        device: 目标设备。
        weights: ``"ema"`` 或 ``"model"``。

    返回:
        二元组 ``(已 eval 的模型, 完整载荷 dict)``；载荷可继续读取
        epoch、args 等来源信息。

    注意:
        使用 ``torch.load(..., weights_only=False)`` 载入完整训练
        检查点（含优化器状态），**只允许加载可信来源的文件**。
    """
    if weights not in ("ema", "model"):
        raise ValueError("weights must be 'ema' or 'model'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for local checks")
    # Full training checkpoints contain optimizer state: load only trusted files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"incompatible checkpoint format: {payload.get('format')!r}")
    if payload.get("data_epoch_policy") != DATA_EPOCH_POLICY:
        raise ValueError("Expected a checkpoint produced after the worker epoch fix")
    # 用载荷里的 model_config 重建结构，再严格加载权重
    model = CalibFuse(**payload["model_config"]).to(device)
    state = payload["ema"]["model"] if weights == "ema" else payload["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
