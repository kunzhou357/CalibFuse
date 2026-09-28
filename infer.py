"""Fuse visible/infrared pairs without clean reference images.

Usage::

    python infer.py --data datasets/test_noise --output results/inference

Outputs (under --output):
- ``rgb/<stem>.png``: fused RGB image;
- ``gray/<stem>.png``: luminance of the fused image;
- ``protocol.json``: run provenance (checkpoint path and SHA-256, epoch,
  weight selection, device, sample list).

Defaults to the EMA weights of ``checkpoints/calibfuse.pth``; pass
``--checkpoint checkpoints/train/latest.pth`` to evaluate a fresh run.
Use ``--max-images`` for quick local checks (a full-resolution image takes
~40 s on CPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.checkpoint import load_model, sha256_file
from utils.dataset import load_image, paired_paths
from utils.image import luminance, save_tensor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fuse paired images with CalibFuse")
    parser.add_argument("--data", type=Path, required=True, help="Folder containing vis/ and ir/")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/calibfuse.pth"))
    parser.add_argument("--output", type=Path, default=Path("results/inference"))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    """Fuse each pair, save RGB/gray images, and write the protocol file."""
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(args.device)
    model, payload = load_model(args.checkpoint, device, args.weights)
    pairs = paired_paths(args.data)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    with torch.inference_mode():
        for visible_path, infrared_path in tqdm(pairs, desc="fusing", ncols=100):
            visible = load_image(visible_path, "RGB")[None].to(device)
            infrared = load_image(infrared_path, "L")[None].to(device)
            # align infrared to the visible size, matching the training pipeline
            if infrared.shape[-2:] != visible.shape[-2:]:
                infrared = torch.nn.functional.interpolate(
                    infrared, visible.shape[-2:], mode="bilinear", align_corners=False)
            fused = model(visible, infrared)["fused"]
            save_tensor(args.output / "rgb" / f"{visible_path.stem}.png", fused)
            save_tensor(args.output / "gray" / f"{visible_path.stem}.png", luminance(fused))
    protocol = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": payload["epoch"], "weights": args.weights,
        "data": str(args.data), "images": len(pairs), "device": str(device),
        "precision": "float32", "torch_version": str(torch.__version__),
        "sample_names": [visible.stem for visible, _ in pairs],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
