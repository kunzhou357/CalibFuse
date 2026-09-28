"""Fuse visible/infrared pairs without requiring clean reference images.

推理入口：对成对图像做融合，**不需要干净参考图**。

用法::

    python infer.py --data datasets/test_noise --output results/inference

输出（在 --output 下）：
- ``rgb/<stem>.png``：融合 RGB 图；
- ``gray/<stem>.png``：融合图的亮度灰度图；
- ``protocol.json``：运行协议（检查点路径与 SHA-256、epoch、
  权重选择、设备、样本清单等），保证结果可溯源。

默认用 ``checkpoints/calibfuse.pth`` 的 **EMA 权重**；评估新训练结果时
传 ``--checkpoint checkpoints/train/latest.pth``。全分辨率单图在
CPU 上约 40 s，可用 ``--max-images`` 限制数量做本地快速检查。
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
    """解析命令行参数。

    --data 必填（含 vis/ 与 ir/）；--weights 选 ema（默认）或
    model（最终学生权重）。
    """
    parser = argparse.ArgumentParser(description="Fuse paired images with CalibFuse")
    parser.add_argument("--data", type=Path, required=True, help="Folder containing vis/ and ir/")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/calibfuse.pth"))
    parser.add_argument("--output", type=Path, default=Path("results/inference"))
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    """逐对读取 → 模型融合 → 保存 RGB/灰度图 → 写协议文件。"""
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(args.device)
    # 加载模型（内部校验格式/策略/结构），payload 用于协议记录
    model, payload = load_model(args.checkpoint, device, args.weights)
    pairs = paired_paths(args.data)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    with torch.inference_mode():
        for visible_path, infrared_path in tqdm(pairs, desc="fusing", ncols=100):
            # 加批维 (1,C,H,W) 后上设备
            visible = load_image(visible_path, "RGB")[None].to(device)
            infrared = load_image(infrared_path, "L")[None].to(device)
            # 尺寸不一致时对齐到可见光（与训练管道一致）
            if infrared.shape[-2:] != visible.shape[-2:]:
                infrared = torch.nn.functional.interpolate(
                    infrared, visible.shape[-2:], mode="bilinear", align_corners=False)
            # 前向（网络内部处理填充与裁剪）
            fused = model(visible, infrared)["fused"]
            # 同时保存 RGB 与其亮度灰度（灰度供人工检查与对比）
            save_tensor(args.output / "rgb" / f"{visible_path.stem}.png", fused)
            save_tensor(args.output / "gray" / f"{visible_path.stem}.png", luminance(fused))
    # 协议文件：记录可复现实验所需的一切来源信息
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
