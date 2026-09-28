"""Benefit-calibrated dictionary recovery.

Models degradation removal in three steps:

1. **retrieve**: soft-assign the (possibly degraded) feature against a
   learnable atom dictionary to obtain a clean ``reference``;
2. **candidate**: propose a bounded correction from the evidence
   ``(z, reference, z - reference)``;
3. **adopt**: predict a per-pixel ``adoption`` in [0, 1] and form
   ``recovered = z + adoption * candidate`` — the network learns *where*
   and *how much* to correct, not how.

The module also predicts a log error whose negative exponential
``reliability = exp(-predicted_error)`` feeds the downstream cross-modal
interaction as a confidence signal.

Gradient isolation: with ``clean_anchor_only=True`` (default) the noisy
retrieval detaches the dictionary and its key/value projections, so
dictionary gradients come exclusively from the clean-anchor loss path
(see ``CalibFuse.forward``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-sample RMS normalization over (C, H, W).

    Removes the energy scale only — spatial contrast structure is
    preserved. Used network-wide to align visible/infrared feature
    magnitudes.
    """
    scale = x.float().square().mean((1, 2, 3), keepdim=True).add(eps).sqrt()
    return (x.float() / scale).to(x.dtype)


class LocalResidual(nn.Module):
    """Lightweight local residual block (1x1 -> 5x5 depthwise -> 1x1).

    Near-identity initialization keeps the block close to the identity at
    the start of training.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """Initialize the residual block."""
        super().__init__()
        hidden = channels * expansion
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.project_out = nn.Conv2d(hidden, channels, 1)
        # near-identity init: the initial output is almost exactly x
        nn.init.normal_(self.project_out.weight, std=1e-3)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x + residual`` with the input RMS-normalized."""
        return x + self.project_out(F.gelu(self.spatial(self.project_in(rms_normalize(x)))))


class BenefitCalibratedDictionary(nn.Module):
    """Dictionary retrieval that proposes a correction and predicts its adoption.

    Also outputs a reliability estimate; see the module docstring for the
    three-step design and the gradient-isolation contract.
    """

    def __init__(self, channels: int, atoms: int = 64, query_channels: int | None = None,
                 axial: bool = False, clean_anchor_only: bool = True) -> None:
        """Initialize the dictionary module.

        ``axial`` enables row/column factorized convolutions (used by the
        infrared dictionary to capture stripe-like degradations).
        ``clean_anchor_only`` detaches the dictionary in the noisy
        retrieval path.
        """
        super().__init__()
        query_channels = query_channels or max(16, channels // 4)
        self.atoms = atoms
        self.clean_anchor_only = clean_anchor_only
        # dictionary atoms (K, C); updated through the clean-anchor loss only
        self.dictionary = nn.Parameter(torch.randn(atoms, channels) * 0.02)
        self.query_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.query = nn.Conv2d(channels, query_channels, 1)
        self.key = nn.Linear(channels, query_channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        # temperature = 1 + 29 * sigmoid(raw): starts near 27 (sharp assignment)
        self.temperature_raw = nn.Parameter(torch.tensor(2.2))

        # candidate correction network: 3C evidence -> hidden -> C, tanh-bounded
        hidden = max(16, channels // 2)
        self.candidate_in = nn.Conv2d(3 * channels, hidden, 1)
        self.candidate_spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.candidate_out = nn.Conv2d(hidden, channels, 1)
        # near-identity init: the initial candidate is almost zero
        nn.init.normal_(self.candidate_out.weight, std=1e-3)
        nn.init.zeros_(self.candidate_out.bias)

        self.axial = axial
        if axial:
            # row (1x9) / column (9x1) factorized convolutions capture long
            # stripe structures at a fraction of the 9x9 cost
            self.row = nn.Conv2d(hidden, hidden, (1, 9), padding=(0, 4), groups=hidden)
            self.column = nn.Conv2d(hidden, hidden, (9, 1), padding=(4, 0), groups=hidden)
            self.axis_mix = nn.Conv2d(2 * hidden, hidden, 1)

        # adoption head: evidence = (z, reference, candidate, entropy, top-2 margin)
        evidence_channels = 3 * channels + 2
        self.adoption = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # bias -2 => initial adoption ≈ sigmoid(-2) ≈ 0.12
        nn.init.normal_(self.adoption[-1].weight, std=1e-3)
        nn.init.constant_(self.adoption[-1].bias, -2.0)

        # log-error head: evidence = (z, recovered, entropy, top-2 margin)
        self.log_error = nn.Sequential(
            nn.Conv2d(2 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # bias -2 => initial reliability ≈ exp(-softplus(-2)) ≈ 0.88
        nn.init.normal_(self.log_error[-1].weight, std=1e-3)
        nn.init.constant_(self.log_error[-1].bias, -2.0)

    @property
    def temperature(self) -> Tensor:
        """Learnable soft-assignment temperature in (1, 30)."""
        return 1.0 + 29.0 * torch.sigmoid(self.temperature_raw.float())

    def _dictionary_projection(self, detach_dictionary: bool) -> tuple[Tensor, Tensor]:
        """Project dictionary atoms to (keys, values), optionally detached."""
        dictionary = self.dictionary.detach() if detach_dictionary else self.dictionary
        if detach_dictionary:
            # detach the projections as well so this path has no dictionary gradient
            keys = F.linear(dictionary.float(), self.key.weight.detach().float())
            values = F.linear(dictionary.float(), self.value.weight.detach().float())
        else:
            keys = self.key(dictionary.float())
            values = self.value(dictionary.float())
        return keys, values

    def retrieve(self, feature: Tensor, *, detach_dictionary: bool = False) -> tuple[Tensor, Tensor]:
        """Soft-retrieve a clean reference; returns ``(reference, assignment)``.

        Per-pixel cosine similarity between the query and every atom,
        scaled by the temperature, produces the soft assignment
        ``assignment (B, K, H, W)``; the reference is the assignment-
        weighted sum of atom values.
        """
        normalized = rms_normalize(feature)
        query = self.query(normalized + self.query_context(normalized)).float()
        query = F.normalize(query, dim=1)
        keys, values = self._dictionary_projection(detach_dictionary)
        keys = F.normalize(keys, dim=1)
        logits = torch.einsum("bchw,kc->bkhw", query, keys) * self.temperature
        assignment = logits.softmax(dim=1)
        reference = torch.einsum("bkhw,kc->bchw", assignment, values)
        return reference.to(feature.dtype), assignment

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        """Run the full recovery.

        Returns ``input`` (RMS-normalized z), ``reference``, ``assignment``,
        ``candidate``, ``adoption``, ``recovered``,
        ``predicted_log_error``, and ``reliability``.
        """
        z = rms_normalize(feature)
        # noisy-path retrieval: the dictionary is detached by default
        reference, assignment = self.retrieve(z, detach_dictionary=self.clean_anchor_only)
        candidate_input = torch.cat((z, reference, z - reference), 1)
        hidden = F.gelu(self.candidate_spatial(self.candidate_in(candidate_input)))
        if self.axial:
            hidden = hidden + self.axis_mix(torch.cat((self.row(hidden), self.column(hidden)), 1))
        candidate = torch.tanh(self.candidate_out(hidden).float()).to(z.dtype)

        # two per-pixel retrieval-quality signals: normalized entropy and
        # the top-2 probability margin
        entropy = -(assignment.float().clamp_min(1e-8) * assignment.float().clamp_min(1e-8).log()).sum(1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(self.atoms), device=entropy.device))
        top2 = assignment.float().topk(min(2, self.atoms), dim=1).values
        margin = top2[:, :1] - top2[:, 1:2] if self.atoms > 1 else top2[:, :1]
        evidence = torch.cat((z, reference, candidate, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        adoption = torch.sigmoid(self.adoption(evidence).float()).to(z.dtype)
        # core update: recovered = input + adoption * candidate
        recovered = z + adoption * candidate

        error_evidence = torch.cat((z, recovered, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        predicted_log_error = F.softplus(self.log_error(error_evidence).float())
        predicted_error = torch.expm1(predicted_log_error.clamp(max=10.0))
        reliability = torch.exp(-predicted_error).clamp(0.0, 1.0).to(z.dtype)
        return {
            "input": z, "reference": reference, "assignment": assignment,
            "candidate": candidate, "adoption": adoption, "recovered": recovered,
            "predicted_log_error": predicted_log_error,
            "reliability": reliability,
        }
