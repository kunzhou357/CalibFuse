"""Checkpoint 兼容性常量与各入口共享的模型加载工具。

- :data:`CHECKPOINT_FORMAT`：checkpoint 载荷格式标签；
- :data:`DATA_EPOCH_POLICY`：数据加载 epoch 策略标签
  （worker 每 epoch 重建；旧策略的 checkpoint 在加载时被拒绝）。

修改任何一个常量都会破坏与既有 checkpoint / 评测报告的兼容性，
必须同步打新的版本标签（这也是加载端校验的依据）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from nets.fusion import CalibFuse

# 载荷标签；checkpoint 必须携带相同值才能加载
CHECKPOINT_FORMAT = "calibfuse-benefit-calibrated-v1"
# 数据加载策略：worker 每个 epoch 重建
DATA_EPOCH_POLICY = "recreate-workers-each-epoch-v1"


def sha256_file(path: str | Path) -> str:
    """流式计算文件的 SHA-256 十六进制摘要（1 MiB 分块）。

    用于把权重文件的"指纹"写进 protocol.json，
    保证评测结果可追溯到确切的权重版本。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device, weights: str = "ema") -> tuple[CalibFuse, dict]:
    """加载 checkpoint 并带严格校验地重建模型。

    参数:
        path: checkpoint 路径。
        device: 目标设备（cuda 请求但不可用时直接报错，
            提示改用 --device cpu）。
        weights: ``"ema"``（默认，更稳定）或 ``"model"``（学生权重）。

    返回:
        (评估模式的模型, 完整载荷字典)。

    校验项:
    - ``format`` 必须等于 CHECKPOINT_FORMAT；
    - ``data_epoch_policy`` 必须等于 DATA_EPOCH_POLICY
      （拒绝 worker 修复之前的旧 checkpoint）；
    - 架构按载荷中的 ``model_config`` 重建，state_dict 严格加载。

    安全提示：训练 checkpoint 包含优化器状态，加载使用
    ``torch.load(..., weights_only=False)``——只加载可信文件。
    """
    if weights not in ("ema", "model"):
        raise ValueError("weights must be 'ema' or 'model'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for local checks")
    # 训练 checkpoint 含优化器状态：只加载可信文件
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"incompatible checkpoint format: {payload.get('format')!r}")
    if payload.get("data_epoch_policy") != DATA_EPOCH_POLICY:
        raise ValueError("Expected a checkpoint produced after the worker epoch fix")
    # 按存储的 model_config 重建网络（不依赖代码默认值）
    model = CalibFuse(**payload["model_config"]).to(device)
    # 权重选择：ema 取教师权重，model 取学生权重
    state = payload["ema"]["model"] if weights == "ema" else payload["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
