"""评估入口：在带噪输入 + 干净参考上量化 CalibFuse 的融合质量。

用法::

    python test.py --data datasets/test_noise --reference datasets/test \\
        --output results/test_noise

流程：对每个观测对做融合并存图 → **在保存后的 8-bit 灰度 PNG 上**
按 calibfuse-metrics-v1 协议计算 7 个指标 → 汇总写 ``metrics.csv`` 与
``protocol.json``。

- 指标集合（METRICS）：EN、SF、MI、SCD、VIF、Qabf、SSIM；
- **每个观测样本在 --reference 下必须有同主干名的干净对**，
  缺失即报错（不做静默跳过）；
- 注意指标协议的固定约定（MI 用自然对数、SSIM/VIF 对两源求和
  等，详见 utils/evaluator.py 模块说明），不要与其他代码库的
  数值直接比较；
- 输出还包括 rgb/ 与 gray/ 图像目录；csv 最后一行是 ``mean``
  汇总行。
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


# 报表输出的 7 个指标（评估在灰度图上进行）
METRICS = ("EN", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --data 为观测（带噪）输入目录，--reference 为同主干名的干净
    参考目录；--max-images 只限制融合/评估的样本数（冒烟测试用）。
    """
    parser = argparse.ArgumentParser(description="Evaluate the CalibFuse fusion model")
    parser.add_argument("--data", type=Path, default=Path("datasets/test_noise"),
                        help="Observed vis/ir inputs")
    parser.add_argument("--reference", type=Path, default=Path("datasets/test"),
                        help="Clean vis/ir references with matching stems")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/calibfuse.pth"))
    parser.add_argument("--output", type=Path, default=Path("results/test_noise"))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--max-images", type=int,
                        help="Limit inference for smoke tests; omit for full evaluation")
    return parser.parse_args()


def main() -> None:
    """主流程：融合 → 存图 → 按协议评估 → 写 csv 与协议文件。"""
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(args.device)
    model, payload = load_model(args.checkpoint, device, args.weights)

    observed_pairs = paired_paths(args.data)
    if args.max_images is not None:
        observed_pairs = observed_pairs[:args.max_images]
    # 参考集建索引：主干名 → (干净 vis, 干净 ir)
    reference_pairs = {visible.stem: (visible, infrared)
                       for visible, infrared in paired_paths(args.reference)}
    # 硬性要求：每个观测样本都要有干净参考，缺失即失败
    missing = [visible.stem for visible, _ in observed_pairs if visible.stem not in reference_pairs]
    if missing:
        raise RuntimeError(f"{len(missing)} observed samples have no clean reference; first={missing[0]}")
    gray_dir, rgb_dir = args.output / "gray", args.output / "rgb"
    gray_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for visible_path, infrared_path in tqdm(observed_pairs, desc="fusing", ncols=100):
            visible = load_image(visible_path, "RGB")[None].to(device)
            infrared = load_image(infrared_path, "L")[None].to(device)
            # 红外尺寸对齐可见光（与训练/推理管道一致）
            if infrared.shape[-2:] != visible.shape[-2:]:
                infrared = torch.nn.functional.interpolate(infrared, size=visible.shape[-2:],
                                                           mode="bilinear", align_corners=False)
            fused_tensor = model(visible, infrared)["fused"]
            name = visible_path.stem
            save_tensor(rgb_dir / f"{name}.png", fused_tensor)
            save_tensor(gray_dir / f"{name}.png", luminance(fused_tensor))
            # 指标在保存后的 8-bit 灰度 PNG 上计算（协议要求：
            # 量化误差属于指标的一部分）
            fused = read_image(gray_dir / f"{name}.png")
            clean_visible, clean_infrared = reference_pairs[name]
            # 干净参考 resize 到融合图尺寸后参与计算
            values = evaluate(read_image(clean_visible, size=fused.shape[:2]),
                              read_image(clean_infrared, size=fused.shape[:2]), fused, METRICS)
            rows.append((name, values))
    # 汇总均值（最后一行 mean）
    means = {metric: float(np.mean([values[metric] for _, values in rows])) for metric in METRICS}
    with open(args.output / "metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", *METRICS])
        for name, values in rows:
            writer.writerow([name, *[f"{values[metric]:.6f}" for metric in METRICS]])
        writer.writerow(["mean", *[f"{means[metric]:.6f}" for metric in METRICS]])
    # 协议文件：附检查点哈希与指标协议标识，保证可溯源、不可误比
    protocol = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": payload["epoch"],
        "weights": args.weights, "observed_data": str(args.data),
        "clean_reference_data": str(args.reference), "images": len(rows), "metrics_mean": means,
        "checkpoint_sha256": sha256_file(args.checkpoint), "metric_protocol": METRIC_PROTOCOL,
        "device": str(device), "precision": "float32", "torch_version": str(torch.__version__),
        "sample_names": [name for name, _ in rows],
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print("  ".join(f"{name}={value:.4f}" for name, value in means.items()))


if __name__ == "__main__":
    main()
