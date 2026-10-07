# CalibFuse

Degradation-robust visible/infrared image fusion.
The three-scale PyTorch model combines modality-specific dictionary recovery,
benefit-calibrated adoption, and reliability-weighted cross-modal interaction.
It has **1,625,131 parameters** and uses **12 Transformer blocks**.

## Installation

Use Python 3.10+ in a fresh environment. Install a CUDA-enabled PyTorch build using
the [official selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA is the default for all entry points; training defaults to BF16.
CPU checks require `--device cpu`.

## Data and checkpoint

Run commands from this directory. Download the datasets from Baidu Netdisk
(`datasets.zip`) and unzip the archive under the repository root:

- Link: https://pan.baidu.com/s/1c94wb9mraY-qzFpo9ifUGw?pwd=abc1
- Extraction code: `abc1`

Arrange the extracted splits and any weights as:

```text
datasets/
  train/{vis,ir}/
  test/{vis,ir}/
  test_noise/{vis,ir}/
ckpt/model.pth
```

Pairs use matching filename stems. Visible images are RGB; infrared images are
grayscale. Test-noise inputs are evaluated against corresponding clean test
sources. All existing training pairs are used; no validation split is created.
Keep training and test sets separate. See [data setup](datasets/README.md).

A trained checkpoint is included at `ckpt/model.pth` (epoch 99; EMA or student
weights selectable via `--weights`), and every entry point defaults to it.
Run a new training (outputs go to `checkpoints/train/`) for your own weights.

## Training

```bash
python train.py --data datasets/train --output checkpoints/train
python train.py --data datasets/train --output checkpoints/train --resume checkpoints/train/latest.pth
```

Defaults: 100 epochs, crop 256, batch 4, seed 3407, AdamW LR `2e-4` / minimum
`1e-6`, weight decay `1e-4`, CUDA BF16. Fresh workers receive the updated epoch
each iteration. Checkpoints, logs, and previews stay under the selected output
directory. Resume must preserve training hyperparameters.

## Inference

```bash
python test.py --data datasets/test_noise --output results/test_noise
```

`test.py` defaults to `ckpt/model.pth` and EMA weights. For a new
training run, add `--checkpoint checkpoints/train/latest.pth`. It needs
no references, computes no metrics, and writes RGB/grayscale PNGs plus a
protocol with checkpoint SHA-256 and sample names. Defaults live in a
config block at the top of `test.py` — edit the file instead of typing
long CLI args (flags still override).

## Fusion results

Outputs of the released `ckpt/model.pth` (EMA weights) on MSRS, M3FD, and
LLVIP test pairs. The degraded examples use the fixed `test_noise` inputs:
Poisson shot/read noise on the visible channel; striping, Gaussian noise,
and stuck pixels on infrared.

### MSRS

| Visible | Infrared | Fused |
|:---:|:---:|:---:|
| <img src="assets/results/msrs_00004N_vis.png" width="260" alt="MSRS 00004N visible"> | <img src="assets/results/msrs_00004N_ir.png" width="260" alt="MSRS 00004N infrared"> | <img src="assets/results/msrs_00004N_fused.png" width="260" alt="MSRS 00004N fused"> |
| <img src="assets/results/msrs_00024N_vis.png" width="260" alt="MSRS 00024N visible"> | <img src="assets/results/msrs_00024N_ir.png" width="260" alt="MSRS 00024N infrared"> | <img src="assets/results/msrs_00024N_fused.png" width="260" alt="MSRS 00024N fused"> |
| <img src="assets/results/msrs_00123D_vis.png" width="260" alt="MSRS 00123D visible"> | <img src="assets/results/msrs_00123D_ir.png" width="260" alt="MSRS 00123D infrared"> | <img src="assets/results/msrs_00123D_fused.png" width="260" alt="MSRS 00123D fused"> |
| <img src="assets/results/msrs_00634D_vis.png" width="260" alt="MSRS 00634D visible"> | <img src="assets/results/msrs_00634D_ir.png" width="260" alt="MSRS 00634D infrared"> | <img src="assets/results/msrs_00634D_fused.png" width="260" alt="MSRS 00634D fused"> |

### Degraded inputs (MSRS with synthetic sensor noise)

| Visible (noisy) | Infrared (noisy) | Fused |
|:---:|:---:|:---:|
| <img src="assets/results/noise_00004N_vis.png" width="260" alt="Noisy MSRS 00004N visible"> | <img src="assets/results/noise_00004N_ir.png" width="260" alt="Noisy MSRS 00004N infrared"> | <img src="assets/results/noise_00004N_fused.png" width="260" alt="Fused noisy MSRS 00004N"> |
| <img src="assets/results/noise_00123D_vis.png" width="260" alt="Noisy MSRS 00123D visible"> | <img src="assets/results/noise_00123D_ir.png" width="260" alt="Noisy MSRS 00123D infrared"> | <img src="assets/results/noise_00123D_fused.png" width="260" alt="Fused noisy MSRS 00123D"> |

### M3FD

| Visible | Infrared | Fused |
|:---:|:---:|:---:|
| <img src="assets/results/m3fd_00011_vis.png" width="260" alt="M3FD 00011 visible"> | <img src="assets/results/m3fd_00011_ir.png" width="260" alt="M3FD 00011 infrared"> | <img src="assets/results/m3fd_00011_fused.png" width="260" alt="M3FD 00011 fused"> |

### LLVIP

| Visible | Infrared | Fused |
|:---:|:---:|:---:|
| <img src="assets/results/llvip_260284_vis.jpg" width="260" alt="LLVIP 260284 visible"> | <img src="assets/results/llvip_260284_ir.jpg" width="260" alt="LLVIP 260284 infrared"> | <img src="assets/results/llvip_260284_fused.png" width="260" alt="LLVIP 260284 fused"> |
| <img src="assets/results/llvip_260314_vis.jpg" width="260" alt="LLVIP 260314 visible"> | <img src="assets/results/llvip_260314_ir.jpg" width="260" alt="LLVIP 260314 infrared"> | <img src="assets/results/llvip_260314_fused.png" width="260" alt="LLVIP 260314 fused"> |
| <img src="assets/results/llvip_260494_vis.jpg" width="260" alt="LLVIP 260494 visible"> | <img src="assets/results/llvip_260494_ir.jpg" width="260" alt="LLVIP 260494 infrared"> | <img src="assets/results/llvip_260494_fused.png" width="260" alt="LLVIP 260494 fused"> |

## Development

```bash
python -m compileall -q train.py test.py nets utils
```

This repository ships the source only; there is no bundled test suite. Keep
training and test sets separate and never commit images, weights, or results.
Source is MIT-licensed; the
Restormer-derived modules in `nets/restormer.py` retain the upstream MIT
notice under `licenses/`. Datasets and binary checkpoints are excluded from
the source license.
