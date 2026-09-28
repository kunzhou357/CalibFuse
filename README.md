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

Run commands from this directory. Download datasets separately and arrange pairs as:

```text
datasets/
  train/{vis,ir}/
  test/{vis,ir}/
  test_noise/{vis,ir}/
checkpoints/calibfuse.pth
```

Pairs use matching filename stems. Visible images are RGB; infrared images are
grayscale. Test-noise inputs are evaluated against corresponding clean test
sources. All existing training pairs are used; no validation split is created.
Keep training and test sets separate. See [data setup](datasets/README.md).

No weight files are included in this repository. Run a new training (outputs go
to `checkpoints/train/`) or place your own weights at `checkpoints/calibfuse.pth`
to use the default paths.

## Training

```bash
python train.py --data datasets/train --output checkpoints/train
python train.py --data datasets/train --output checkpoints/train --resume checkpoints/train/latest.pth
```

Defaults: 100 epochs, crop 256, batch 4, seed 3407, AdamW LR `2e-4` / minimum
`1e-6`, weight decay `1e-4`, CUDA BF16. Fresh workers receive the updated epoch
each iteration. Checkpoints, logs, and previews stay under the selected output
directory. Resume must preserve training hyperparameters.

## Inference and evaluation

```bash
python infer.py --data datasets/test_noise --output results/inference
python test.py --data datasets/test_noise --reference datasets/test --output results/test_noise
python diagnose.py --data datasets/test_noise --reference datasets/test --max-images 4 --crop-size 64
```

All three default to `checkpoints/calibfuse.pth` and EMA weights. For a new
training run, add `--checkpoint checkpoints/train/latest.pth`. `infer.py` needs
no references and writes RGB/grayscale PNGs. `test.py` writes seven metrics,
`metrics.csv`, and a protocol with checkpoint SHA-256 and sample names.

Metric caveat: the evaluator follows the fixed `calibfuse-metrics-v1`
conventions — metrics are computed on saved 8-bit PNGs, MI uses natural logs,
SSIM and VIF sum the two source scores, and PSNR/RMSE use a
`sqrt(SSE)/(m·n)` denominator. Do not compare these numbers directly with
other evaluators.

## Development

```bash
python -m compileall -q train.py infer.py test.py nets utils
```

This repository ships the source only; there is no bundled test suite. Keep
training and test sets separate and never commit images, weights, or results.
See [repository rules](AGENTS.md). Source is MIT-licensed; the
Restormer-derived modules in `nets/restormer.py` retain the upstream MIT
notice under `licenses/`. Datasets and binary checkpoints are excluded from
the source license.
