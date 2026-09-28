"""Evaluate CalibFuse fusion quality on degraded inputs against clean references.

Usage::

    python test.py --data datasets/test_noise --reference datasets/test \\
        --output results/test_noise

Each observed pair is fused and saved, then the metrics in :data:`METRICS`
are computed on the saved 8-bit grayscale PNGs following the
``calibfuse-metrics-v1`` protocol, and summarized into ``metrics.csv`` and
``protocol.json``.

Result reuse: a sample whose ``<output>/gray/<stem>.png`` already exists is
treated as fused — the model is only loaded when at least one sample still
needs fusing, and existing images are never overwritten. Delete the output
folder (or individual gray PNGs) to force re-fusion. Metrics always run on
the saved PNGs, so reused and freshly fused samples follow the same
protocol.

Every observed sample must have a clean pair with the same stem under
--reference; missing references are an error. Metric values follow fixed
protocol conventions (see ``utils/evaluator.py``) and must not be compared
across codebases.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.checkpoint import METRIC_PROTOCOL, load_model, sha256_file
from utils.dataset import load_image, paired_paths
from utils.evaluator import evaluate, read_image
from utils.image import luminance, save_tensor


# Reported metrics; the final csv row is the "mean" summary.
METRICS = ("EN", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the CalibFuse fusion model")
    parser.add_argument("--data", type=Path, default=Path("datasets/test"),
                        help="Observed vis/ir inputs")
    parser.add_argument("--reference", type=Path, default=Path("datasets/test"),
                        help="Clean vis/ir references with matching stems")
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpt/model.pth"))
    parser.add_argument("--output", type=Path, default=Path("results/test"))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--max-images", type=int,
                        help="Limit inference for smoke tests; omit for full evaluation")
    return parser.parse_args()


def main() -> None:
    """Fuse missing samples, evaluate all under the protocol, write reports."""
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(args.device)

    observed_pairs = paired_paths(args.data)
    if args.max_images is not None:
        observed_pairs = observed_pairs[:args.max_images]
    reference_pairs = {visible.stem: (visible, infrared)
                       for visible, infrared in paired_paths(args.reference)}
    missing = [visible.stem for visible, _ in observed_pairs if visible.stem not in reference_pairs]
    if missing:
        raise RuntimeError(f"{len(missing)} observed samples have no clean reference; first={missing[0]}")
    gray_dir, rgb_dir = args.output / "gray", args.output / "rgb"

    # split samples into already-fused (gray PNG exists) and pending
    pending_pairs = [pair for pair in observed_pairs
                     if not (gray_dir / f"{pair[0].stem}.png").exists()]
    reused = len(observed_pairs) - len(pending_pairs)

    # the model is loaded only when at least one sample still needs fusing
    model, payload = (load_model(args.checkpoint, device, args.weights) if pending_pairs
                      else (None, None))
    current_sha = sha256_file(args.checkpoint) if args.checkpoint.exists() else None
    if reused and current_sha is not None:
        previous = args.output / "protocol.json"
        if previous.exists():
            old_sha = json.loads(previous.read_text(encoding="utf-8")).get("checkpoint_sha256")
            if old_sha is not None and old_sha != current_sha:
                print(f"warning: reusing {reused} fusion images produced by a different "
                      f"checkpoint (saved {str(old_sha)[:12]}..., current {str(current_sha)[:12]}...)")

    if model is not None:
        gray_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for visible_path, infrared_path in tqdm(pending_pairs, desc="fusing", ncols=100):
                visible = load_image(visible_path, "RGB")[None].to(device)
                infrared = load_image(infrared_path, "L")[None].to(device)
                # align infrared to the visible size, matching the training pipeline
                if infrared.shape[-2:] != visible.shape[-2:]:
                    infrared = torch.nn.functional.interpolate(infrared, size=visible.shape[-2:],
                                                               mode="bilinear", align_corners=False)
                fused_tensor = model(visible, infrared)["fused"]
                name = visible_path.stem
                save_tensor(rgb_dir / f"{name}.png", fused_tensor)
                save_tensor(gray_dir / f"{name}.png", luminance(fused_tensor))
    print(f"fused {len(pending_pairs)} new result(s), reused {reused} existing result(s)")

    rows = []
    for visible_path, _ in tqdm(observed_pairs, desc="evaluating", ncols=100):
        name = visible_path.stem
        # metrics run on the saved 8-bit PNGs: quantization error is part
        # of the protocol, so reused and fresh results are equivalent here
        fused = read_image(gray_dir / f"{name}.png")
        clean_visible, clean_infrared = reference_pairs[name]
        values = evaluate(read_image(clean_visible, size=fused.shape[:2]),
                          read_image(clean_infrared, size=fused.shape[:2]), fused, METRICS)
        rows.append((name, values))
    means = {metric: float(np.mean([values[metric] for _, values in rows])) for metric in METRICS}
    with open(args.output / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", *METRICS])
        for name, values in rows:
            writer.writerow([name, *[f"{values[metric]:.6f}" for metric in METRICS]])
        writer.writerow(["mean", *[f"{means[metric]:.6f}" for metric in METRICS]])
    protocol = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload["epoch"] if payload is not None else None,
        "weights": args.weights, "observed_data": str(args.data),
        "clean_reference_data": str(args.reference), "images": len(rows), "metrics_mean": means,
        "fused_samples": len(pending_pairs), "reused_fusion_results": reused,
        "checkpoint_sha256": current_sha, "metric_protocol": METRIC_PROTOCOL,
        "device": str(device), "precision": "float32", "torch_version": str(torch.__version__),
        "sample_names": [name for name, _ in rows],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print("  ".join(f"{name}={value:.4f}" for name, value in means.items()))


if __name__ == "__main__":
    main()
