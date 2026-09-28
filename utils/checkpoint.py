"""Checkpoint compatibility constants and model loading shared by entry points.

- :data:`CHECKPOINT_FORMAT`: checkpoint payload format tag;
- :data:`DATA_EPOCH_POLICY`: data-loader epoch policy tag (workers are
  recreated each epoch; older checkpoints are rejected);
- :data:`METRIC_PROTOCOL`: metric protocol tag (calibfuse-metrics-v1).

Changing any constant breaks comparability with existing checkpoints or
metric reports and requires a new version tag.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from nets.fusion import CalibFuse

# Payload tag; checkpoints must carry the same value to load.
CHECKPOINT_FORMAT = "calibfuse-benefit-calibrated-v1"
# Data-loader policy: workers are recreated each epoch.
DATA_EPOCH_POLICY = "recreate-workers-each-epoch-v1"
# Metric protocol identifier; see utils/evaluator.py for the conventions.
METRIC_PROTOCOL = "calibfuse-metrics-v1"


def sha256_file(path: str | Path) -> str:
    """Streamed SHA-256 hex digest of a file (1 MiB blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path, device: torch.device, weights: str = "ema") -> tuple[CalibFuse, dict]:
    """Load a checkpoint with strict validation and rebuild the model.

    ``weights`` selects ``"ema"`` (default) or ``"model"``. The format and
    data-epoch policy tags must match, and the architecture is rebuilt from
    the payload's ``model_config`` with a strict state-dict load. Returns
    the evaluated model and the full payload.

    Note: full training checkpoints contain optimizer state and are loaded
    with ``torch.load(..., weights_only=False)`` — load trusted files only.
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
    model = CalibFuse(**payload["model_config"]).to(device)
    state = payload["ema"]["model"] if weights == "ema" else payload["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload
