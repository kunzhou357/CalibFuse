"""Physically inspired paired degradation (noise) models.

- :class:`VisibleNoise`: Poisson shot noise + Gaussian read noise;
- :class:`InfraredNoise`: multiplicative and additive striping + Gaussian
  noise + stuck pixels.

Both classes are parameter-free callables; the noise magnitudes are drawn
per call from the provided ``torch.Generator`` (seeded deterministically
by the training pipeline). Severity levels: ``"train"`` (default) and
``"hard"`` (heavier, non-overlapping ranges).

These models are part of the training/metric protocol; changing them is a
protocol change.
"""


from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    """Sample a uniform float in [low, high) from the generator."""
    return float(torch.rand((), generator=generator) * (high - low) + low)

def _smooth_1d(profile: Tensor, kernel: int = 9) -> Tensor:
    """Moving-average smoothing of a 1D profile."""
    return F.avg_pool1d(profile[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]


class VisibleNoise:
    """Poisson shot noise + Gaussian read noise.

    ::

        shot = Poisson(clean * photons) / photons   # signal-dependent
        read ~ N(0, read_sigma^2)                   # signal-independent
        observed = clamp(shot + read, 0, 1)
    """

    def __init__(self, severity: str = "train") -> None:
        """Initialize with a ``"train"`` or ``"hard"`` parameter range."""
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """Degrade a clean visible image; returns ``(observed, meta)``.

        ``meta`` carries the per-pixel ``damage`` mask and the sampled
        ``photons`` and ``read_sigma`` parameters.
        """
        if self.severity == "train":
            # log-uniform photon counts keep the dark (noisy) end covered
            photons = math.exp(_uniform(generator, math.log(12.0), math.log(80.0)))
            read_sigma = _uniform(generator, 0.002, 0.025)
        else:
            photons = math.exp(_uniform(generator, math.log(6.0), math.log(12.0)))
            read_sigma = _uniform(generator, 0.025, 0.045)

        # dividing back by photons keeps the expectation at clean
        shot = torch.poisson(clean * photons, generator=generator) / photons
        read = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * read_sigma
        observed = (shot + read).clamp(0.0, 1.0)
        residual = observed - clean

        return observed, {
            "damage": residual.abs().mean(0, keepdim=True),
            "photons": clean.new_tensor(photons),
            "read_sigma": clean.new_tensor(read_sigma),
        }


class InfraredNoise:
    """Striping (multiplicative + additive) + Gaussian noise + stuck pixels.

    ::

        column ~ smooth random + periodic sine, normalized to std=column_scale
        row    ~ smooth random, normalized to std=row_scale
        stripe = column (broadcast over rows) + row (broadcast over columns)
        observed = (1 + 0.35 * stripe) * clean + stripe + gaussian
                   with a bad_probability fraction of pixels pinned to 0 or 1
    """

    def __init__(self, severity: str = "train") -> None:
        """Initialize with a ``"train"`` or ``"hard"`` parameter range."""
        if severity not in {"train", "hard"}:
            raise ValueError("severity must be 'train' or 'hard'")
        self.severity = severity

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        """Degrade a clean infrared image; returns ``(observed, meta)``.

        ``meta`` carries ``damage``, the normalized ``stripe`` map, the
        ``bad_pixel`` mask, and the sampled ``gaussian_sigma``.
        """
        _, height, width = clean.shape

        if self.severity == "train":
            gaussian_sigma = _uniform(generator, 0.004, 0.100)
            column_scale = _uniform(generator, 0.015, 0.090)
            row_scale = _uniform(generator, 0.000, 0.035)
            bad_probability = _uniform(generator, 0.0002, 0.006)
        else:
            gaussian_sigma = _uniform(generator, 0.100, 0.140)
            column_scale = _uniform(generator, 0.090, 0.150)
            row_scale = _uniform(generator, 0.035, 0.070)
            bad_probability = _uniform(generator, 0.006, 0.015)

        # white noise smoothed into low-frequency drift
        column = _smooth_1d(torch.randn(width, generator=generator, dtype=clean.dtype))
        row = _smooth_1d(torch.randn(height, generator=generator, dtype=clean.dtype))

        # fixed-period sine on the column profile emulates readout striping
        coordinate = torch.arange(width, dtype=clean.dtype)
        period = int(torch.randint(6, max(7, min(width, 40)), (), generator=generator))
        column += 0.5 * torch.sin(2.0 * math.pi * coordinate / period + _uniform(generator, 0.0, 2.0 * math.pi))

        # normalize the profiles to the target standard deviations
        column = column / column.std().clamp_min(1e-6) * column_scale
        row = row / row.std().clamp_min(1e-6) * row_scale

        stripe = column[None, None, :] + row[None, :, None]
        gaussian = torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * gaussian_sigma
        # striping perturbs both gain and bias
        observed = (1.0 + 0.35 * stripe) * clean + stripe + gaussian
        # stuck pixels pinned to 0 or 1
        bad_mask = torch.rand(clean.shape, generator=generator) < bad_probability
        bad_values = (torch.rand(clean.shape, generator=generator) > 0.5).to(clean.dtype)
        observed = torch.where(bad_mask, bad_values, observed).clamp(0.0, 1.0)
        residual = observed - clean
        stripe_map = stripe.abs().expand_as(clean)
        return observed, {
            "damage": residual.abs(),
            "stripe": (stripe_map / stripe_map.amax().clamp_min(1e-6)).clamp(0.0, 1.0),
            "bad_pixel": bad_mask.to(clean.dtype),
            "gaussian_sigma": clean.new_tensor(gaussian_sigma),
        }
