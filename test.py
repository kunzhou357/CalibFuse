"""融合可见光/红外图像对并保存结果（无需干净参考图，不计算指标）。

用法::

    python test.py                 # 零参数：使用下方配置块的默认值
    python test.py --data datasets/test_M3FD --max-images 4   # 一次性覆盖

输出（在 --output 下）:
- ``rgb/<stem>.png``：融合后的 RGB 图像；
- ``gray/<stem>.png``：融合结果的亮度（灰度）图；
- ``protocol.json``：运行溯源（checkpoint 路径与 SHA-256、epoch、
  权重选择、设备、样本列表）。

所有默认值都放在 import 下方的配置块里——请直接编辑文件，
而不是敲一长串命令行参数。需要覆盖时命令行 flag 仍然生效
（CPU 上全分辨率推理一张图约需 40 秒）。
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

# ===================== 默认值：直接改这里，无需 CLI =====================
# 数据目录：需要 vis/（或 visible/vi）与 ir/（或 infrared/inf）子目录；
# 图像对按文件名 stem 匹配。
DATA = Path("datasets/test")
# 权重文件：随仓库发布的 checkpoint；
# 自己训练的输出位于 checkpoints/train/latest.pth。
CHECKPOINT = Path("ckpt/model.pth")
# 输出目录：结果写入 <OUTPUT>/rgb 与 <OUTPUT>/gray。
OUTPUT = Path("results/test")
# 设备："cuda"（默认）或 "cpu"。
DEVICE = "cuda"
# 权重选择："ema"（默认，更稳定）或 "model"（最终学生权重）。
WEIGHTS = "ema"
# 只融合前 N 张图做快速检查；None = 全部图像。
MAX_IMAGES = None
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析 CLI 覆盖项；每个 flag 的默认值都来自上方配置块。"""
    parser = argparse.ArgumentParser(description="Fuse paired images with CalibFuse")
    parser.add_argument("--data", type=Path, default=DATA, help="Folder containing vis/ and ir/")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default=DEVICE)
    parser.add_argument("--weights", choices=("ema", "model"), default=WEIGHTS)
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    return parser.parse_args()


def main() -> None:
    """逐对融合，保存 RGB/灰度图像，并写出 protocol 文件。"""
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    device = torch.device(args.device)
    # 加载模型（含格式/策略校验）与完整载荷（取 epoch 信息）
    model, payload = load_model(args.checkpoint, device, args.weights)
    pairs = paired_paths(args.data)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    with torch.inference_mode():
        for visible_path, infrared_path in tqdm(pairs, desc="fusing", ncols=100):
            visible = load_image(visible_path, "RGB")[None].to(device)
            infrared = load_image(infrared_path, "L")[None].to(device)
            # 红外对齐到可见光尺寸（与训练管线一致）
            if infrared.shape[-2:] != visible.shape[-2:]:
                infrared = torch.nn.functional.interpolate(
                    infrared, visible.shape[-2:], mode="bilinear", align_corners=False)
            fused = model(visible, infrared)["fused"]
            # 网络内部自动 pad 到 4 的倍数并裁回原尺寸，任意分辨率可用
            save_tensor(args.output / "rgb" / f"{visible_path.stem}.png", fused)
            save_tensor(args.output / "gray" / f"{visible_path.stem}.png", luminance(fused))
    # 溯源信息：权重指纹保证结果可复现、可审计
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
