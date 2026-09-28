# AGENTS.md

Guidance for coding agents working in this repository (referenced by `README.md` as "repository rules").

## What this is

CalibFuse — degradation-robust visible/infrared image fusion. Three-scale PyTorch model (`nets/fusion.py`, class `CalibFuse`), 1,625,131 parameters, 12 Restormer-style Transformer blocks.

This checkout is code + data only: no weights, tests, tools, or docs ship with it. `infer.py`/`test.py` default to `checkpoints/calibfuse.pth`, which does **not** exist here — train first or supply `--checkpoint`. Metric values are protocol-bound (see invariants below).

## Environment

- Use the conda env `img_fusion` (Python 3.12, torch 2.5.1+cu124, CUDA available). The base conda `python` is 3.14 with **no torch installed** — bare `python` fails. Run `conda activate img_fusion` or call that interpreter explicitly.
- CUDA is the default device for every entry point; CPU checks need `--device cpu`.
- `torch.load` uses `weights_only=False` in several places — only load trusted checkpoints.
- One full-resolution CPU image takes ~40 s; use `--max-images` on `infer.py`/`test.py` for local checks and drop it for reported tables.

## Commands (run from repo root)

```bash
python -m compileall -q train.py infer.py test.py nets utils      # syntax check (no test suite in this checkout)

python train.py --data datasets/train --output checkpoints/train     # training; refuses to overwrite existing latest.pth
python train.py --output checkpoints/smoke --device cpu --crop-size 32 --max-batches 1   # cheap smoke run

python infer.py --data datasets/test_noise --output results/inference --checkpoint checkpoints/train/latest.pth              # inference, no references needed
python test.py  --data datasets/test_noise --reference datasets/test --output results/test_noise --checkpoint checkpoints/train/latest.pth   # evaluation
```

Three entry points exist (`train.py`, `infer.py`, `test.py`). Some docstrings still mention `diagnose.py` and `tests/` from an earlier layout; those files are gone — treat such references as historical.

## Architecture map

- `nets/fusion.py` — `CalibFuse`: separate vis (3-ch) / ir (1-ch) stems at scales 32/64/128 (4/4/8 heads); a `CalibratedFusionBlock` per scale; fused features cascade downward as `prior_fused`; `CompactDecoder` + `fusion_head` (sigmoid, RGB in [0,1]). Inputs padded to multiple of 4, so odd sizes work.
- `nets/dictionary.py` — `BenefitCalibratedDictionary`: soft-assignment retrieval, candidate correction gated by learned per-pixel `adoption ∈ [0,1]`, predicted log-error → `reliability = exp(-predicted_error)`. IR uses `axial=True`. The dictionary is **detached from the noisy retrieval path** (`clean_anchor_only=True`) — `dictionary.grad` flows only through the clean-anchor loss.
- `nets/restormer.py` — inherited Transformer blocks; attribution comments + MIT license in `licenses/Restormer-MIT.txt`.
- `utils/loss.py` — fusion terms (intensity, SSIM, gradient, YCbCr) + regularizers driven by the EMA teacher's clean features. `optimal_gain` (detached ridge-optimal per-pixel coefficient) is the calibration core. `harmful_correction_rate` is reported, not optimized.
- `utils/dataset.py` — pairs keyed by filename stem across `vis/`+`ir/` (also accepts `visible`/`vi`, `infrared`/`inf`); **unmatched stems are silently skipped — check pair counts**. Noise (`utils/degradation.py`) is reproducible from `seed + epoch*1_000_003 + index`.
- `utils/checkpoint.py` — `CHECKPOINT_FORMAT`, `DATA_EPOCH_POLICY`, `METRIC_PROTOCOL` shared by all entry points; model config is read from the `model_config` payload.
- `utils/evaluator.py` — metric protocol `calibfuse-metrics-v1` (see invariants below).

## Non-obvious invariants — do not "fix" these

- **Worker epoch policy** (`recreate-workers-each-epoch-v1`): `persistent_workers=False`; `train.py` calls `dataset.set_epoch`/`sampler.set_epoch` before iterating a freshly created loader. Checkpoints without this policy are rejected. Do not optimize into persistent workers.
- **Class name is `CalibFuse`**; state-dict keys come from module attribute names, so keep submodule/attribute names stable when refactoring.
- **Resume guards**: `train.py` rejects checkpoints from fine-tuning runs and requires `--epochs/--batch-size/--crop-size/--lr/--min-lr/--weight-decay/--seed/--amp` to match stored training args.
- **`--batch-size` and `--crop-size` must be divisible by 4** (batch sampler fills 4 quality states; network downsamples by 4).
- **Metric protocol `calibfuse-metrics-v1`**: metrics are computed on the saved 8-bit PNGs; MI uses natural logs; SSIM and VIF **sum** the two source scores (SSIM can exceed 1); the PSNR/RMSE denominator convention is `sqrt(SSE)/(m·n)`; Qabf keeps an equal-gradient branch. Never substitute another implementation or compare these numbers across codebases — a corrected implementation is a new protocol.
- **EMA**: `CleanEMA` decay `min(0.99, (1+updates)/(10+updates))`; under fp16 AMP the EMA step is skipped when `GradScaler` skipped the optimizer step.
- **Cross-modal budget ramp**: `budget_scale` ramps 0→1 over epochs 5–20; `training_epoch_offset` keeps it continuous across warm starts.

## Versioning rules

Any change to architecture, losses, noise, or metrics requires a **new protocol version**:

- Do not silently revise published metric values.
- Do not "fix" the metric evaluator in place — a corrected implementation is a new protocol.
- New training output goes to `checkpoints/train/`; never overwrite a released weights file in place.

## Data

Layout: `datasets/<split>/{vis,ir}/` with `train/` (1,083 pairs), `test/` and `test_noise/` (361 pairs each). `test_noise` is a **fixed benchmark** — never regenerate it; `test/` holds its clean sources under identical stems. This machine also has `datasets/test_LLVIP/` (347 pairs) and `datasets/test_M3FD/` (300 pairs) — clean internal test sets **outside the released protocol**; report results on them as separate experiments. No dataset images or weights are distributed with the source.

## Style and commits

Four-space indent, `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants; no enforced formatter. If (re)adding tests, put them under `tests/test_<module>.py` and import entry points by module (as the earlier suite did via `from train import ...`). Concise imperative commit subjects (e.g., `Fix checkpoint validation`). Never commit images, weights, results, secrets, or machine-specific paths. Document tensor shapes and ranges when interfaces change.

## Read before touching sensitive areas

- `datasets/README.md` — data layout, pairing rules, and provenance caveats.
- The module docstrings in `nets/` and `utils/` are the in-repo architecture reference (detailed, in Chinese); `AGENTS.md` summarizes them.
