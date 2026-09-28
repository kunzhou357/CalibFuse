"""CalibFuse training supervision: fusion losses + clean-teacher calibration regularizers.

Fusion losses (final output only): ``intensity`` (max-luminance target),
``structure`` (SSIM), ``gradient`` (per-pixel stronger-source gradient),
``color`` (YCbCr chroma from the visible source).

Calibration regularizers (driven by the EMA teacher's clean features):
- ``recovery``/``candidate``/``anchor``: pull recovered features,
  corrections, and dictionary anchors toward the teacher's clean features;
  the anchor term is the dictionary's only gradient path;
- ``adoption``/``interaction``: regress the learned gates onto
  :func:`optimal_gain`, the ridge-optimal per-pixel coefficient along the
  proposed direction; clean samples force zero gain, and interaction gates
  are capped by ``maximum_transfer * source_reliability``;
- ``error_calibration``: regress the predicted log error onto the true
  error so ``reliability = exp(-error)`` is meaningful;
- ``retrieval``: KL divergence between noisy and clean dictionary
  assignments.

``harmful_correction_rate`` and ``input_teacher_mse`` are reported for
diagnostics and receive no gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .image import luminance, spatial_gradient


def fusion_targets(visible: Tensor, infrared: Tensor) -> dict[str, Tensor]:
    """Build fusion targets: max-luminance and per-pixel stronger-source gradients."""
    visible_y = luminance(visible)
    visible_x, visible_y_gradient = spatial_gradient(visible_y)
    infrared_x, infrared_y = spatial_gradient(infrared)
    # pick the source with the stronger gradient energy at each pixel
    select_infrared = infrared_x.square() + infrared_y.square() > (
        visible_x.square() + visible_y_gradient.square()
    )
    return {
        "luminance": torch.maximum(visible_y, infrared),
        "gradient_x": torch.where(select_infrared, infrared_x, visible_x),
        "gradient_y": torch.where(select_infrared, infrared_y, visible_y_gradient),
    }


def _rgb_to_ycbcr(x: Tensor) -> Tensor:
    """RGB -> YCbCr (float, chroma centered at 0.5); channel order (Y, Cb, Cr)."""
    y = luminance(x)
    cb = 0.5 + (x[:, 2:3] - y) * 0.564
    cr = 0.5 + (x[:, 0:1] - y) * 0.713
    return torch.cat((y, cb, cr), 1)


def _ssim_loss(x: Tensor, y: Tensor) -> Tensor:
    """11x11 window SSIM loss: half of ``1 - SSIM``, clamped to [0, 1]."""
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x.square(), 11, stride=1, padding=5) - mu_x.square()
    sigma_y = F.avg_pool2d(y.square(), 11, stride=1, padding=5) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return ((1.0 - numerator / denominator.clamp_min(1e-8)) * 0.5).clamp(0.0, 1.0).mean()


def retrieval_divergence(student: Tensor, teacher: Tensor) -> Tensor:
    """Per-position KL(student || teacher) over dictionary atoms; teacher detached."""
    target = teacher.detach().float().clamp_min(1e-6)
    prediction = student.float().clamp_min(1e-6)
    return (target * (target.log() - prediction.log())).sum(1).mean((1, 2))


def _feature_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Feature-level SmoothL1 (beta=0.1) against a detached target."""
    return F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=0.1)


def optimal_gain(origin: Tensor, direction: Tensor, target: Tensor,
                 upper: Tensor | float = 1.0, eps: float = 1e-4) -> Tensor:
    """Detached ridge-optimal per-pixel coefficient along a proposed direction.

    Solves ``origin + g * direction ~= target`` per pixel:
    ``g* = <target - origin, direction> / <direction, direction>``, clamped
    to [0, upper]; positions with negligible direction energy get zero.
    Computed under no_grad — this is a supervision target, not part of the
    graph.
    """
    with torch.no_grad():
        numerator = ((target.float() - origin.float()) * direction.float()).sum(1, keepdim=True)
        denominator = direction.float().square().sum(1, keepdim=True).add(eps)
        gain = (numerator / denominator).clamp_min(0.0)
        if torch.is_tensor(upper):
            gain = torch.minimum(gain, upper.detach().float())
        else:
            gain = gain.clamp_max(float(upper))
        # a near-zero direction makes the gain meaningless
        valid = direction.float().square().mean(1, keepdim=True) > eps
        return torch.where(valid, gain, torch.zeros_like(gain))


class CalibFuseLoss(nn.Module):
    """Total CalibFuse loss; see the module docstring for the terms.

    ``total = fusion + recovery_weight * (recovery + 0.25 * candidate)
    + anchor_weight * anchor + retrieval_weight * retrieval
    + calibration_weight * (adoption + error + interaction)``.
    """

    DEFAULTS = {
        "intensity": 2.0,
        "structure": 1.0,
        "gradient": 2.0,
        "color": 1.5,
        "recovery": 0.10,
        "anchor": 0.10,
        "retrieval": 0.05,
        "calibration": 0.05,
    }

    def __init__(self, loss_weights: dict[str, float] | None = None,
                 maximum_transfer: float = 0.15) -> None:
        """Initialize; ``loss_weights`` must contain exactly the DEFAULTS keys.

        ``maximum_transfer`` must match the network's cross-attention
        setting and is used to normalize the interaction gate regression.
        """
        super().__init__()
        weights = dict(self.DEFAULTS if loss_weights is None else loss_weights)
        if set(weights) != set(self.DEFAULTS):
            raise ValueError(f"loss_weights must contain exactly {list(self.DEFAULTS)}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in weights.values()):
            raise ValueError("All loss weights must be finite and nonnegative")
        self.loss_weights = weights
        self.maximum_transfer = maximum_transfer

    def forward(self, output: dict, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute all loss terms.

        ``output`` is the ``CalibFuse(return_auxiliary=True)`` result;
        ``batch`` supplies the clean pairs and degradation masks used to
        detect clean samples. Returns ``total`` and all components; the
        trailing two entries are gradient-free diagnostics.
        """
        # fusion terms: final output against the clean sources
        fused = output["fused"].float()
        visible, infrared = batch["visible"].float(), batch["infrared"].float()
        target = fusion_targets(visible, infrared)
        yf = luminance(fused)
        intensity = F.smooth_l1_loss(yf, target["luminance"], beta=0.02)
        structure = _ssim_loss(yf, target["luminance"])
        gf_x, gf_y = spatial_gradient(fused)
        gradient = 0.5 * (
            F.smooth_l1_loss(gf_x, target["gradient_x"], beta=0.01)
            + F.smooth_l1_loss(gf_y, target["gradient_y"], beta=0.01)
        )
        # chroma only: luminance is handled by the other terms
        color = F.smooth_l1_loss(_rgb_to_ycbcr(fused)[:, 1:],
                                 _rgb_to_ycbcr(visible)[:, 1:], beta=0.02)

        # regularizers over the three scales
        recovery_terms, candidate_terms, anchor_terms = [], [], []
        adoption_terms, error_terms, retrieval_terms, interaction_terms = [], [], [], []
        harmful_before, harmful_after = [], []
        for index, state in enumerate(output["states"]):
            for modality in ("visible", "infrared"):
                branch = state[modality]
                # teacher features are a supervision target, not a graph path
                teacher = output["teacher"][f"{modality}_features"][index].detach()
                # zero damage mask => the sample is clean in this modality
                clean = batch[f"{modality}_damage"].flatten(1).amax(1) == 0
                corrupted = ~clean
                recovery_terms.append(_feature_loss(branch["recovered"], teacher))
                # the anchor term is the dictionary's only gradient path
                anchor = output["clean_anchor_references"][modality][index]
                anchor_terms.append(_feature_loss(anchor, teacher))

                # adopt exactly the ridge-optimal correction; clean samples
                # are forced to zero gain
                alpha_target = optimal_gain(branch["input"], branch["candidate"], teacher)
                if clean.any():
                    alpha_target[clean] = 0.0
                adoption_terms.append(F.smooth_l1_loss(branch["adoption"].float(), alpha_target,
                                                       beta=0.1))
                # candidates and retrieval only make sense for degraded samples
                if corrupted.any():
                    candidate_terms.append(_feature_loss(
                        branch["input"][corrupted] + branch["candidate"][corrupted],
                        teacher[corrupted]))
                    retrieval_terms.append(retrieval_divergence(
                        branch["assignment"][corrupted],
                        output["teacher"][f"{modality}_assignments"][index][corrupted]).mean())

                # error calibration: log1p of the per-pixel MSE to the teacher
                error_target = torch.log1p(
                    (branch["recovered"].float() - teacher.float()).square().mean(1, keepdim=True)
                ).detach()
                error_terms.append(F.smooth_l1_loss(branch["predicted_log_error"], error_target,
                                                     beta=0.05))
                # diagnostics: error against the teacher before/after recovery
                with torch.no_grad():
                    before = (branch["input"].float() - teacher.float()).square().mean(1)
                    after = (branch["recovered"].float() - teacher.float()).square().mean(1)
                    harmful_before.append(before.mean())
                    harmful_after.append((after > before + 1e-4).float().mean())

            # interaction-gate supervision for both message directions
            for message_key, target_modality in (("i_to_v", "visible"), ("v_to_i", "infrared")):
                message = state[message_key]
                receiver = state[target_modality]["recovered"]
                teacher = output["teacher"][f"{target_modality}_features"][index].detach()
                # cap = maximum_transfer x reliability of the *source* modality
                upper = self.maximum_transfer * (
                    state["infrared" if target_modality == "visible" else "visible"]["reliability"]
                )
                gate_target = optimal_gain(receiver, message["direction"], teacher, upper=upper)
                # normalize both sides so gates with different caps stay comparable
                interaction_terms.append(F.smooth_l1_loss(
                    message["gate"].float() / self.maximum_transfer,
                    gate_target / self.maximum_transfer, beta=0.1))

        # average across scales and modalities
        recovery = torch.stack(recovery_terms).mean()
        # all-clean batches have no candidate/retrieval terms; keep the keys
        candidate = (torch.stack(candidate_terms).mean() if candidate_terms
                     else fused.sum() * 0.0)
        anchor = torch.stack(anchor_terms).mean()
        retrieval = (torch.stack(retrieval_terms).mean() if retrieval_terms
                     else fused.sum() * 0.0)
        adoption = torch.stack(adoption_terms).mean()
        error_calibration = torch.stack(error_terms).mean()
        interaction = torch.stack(interaction_terms).mean()
        calibration = adoption + error_calibration + interaction
        fusion = (self.loss_weights["intensity"] * intensity
                  + self.loss_weights["structure"] * structure
                  + self.loss_weights["gradient"] * gradient
                  + self.loss_weights["color"] * color)
        total = (fusion
                 + self.loss_weights["recovery"] * (recovery + 0.25 * candidate)
                 + self.loss_weights["anchor"] * anchor
                 + self.loss_weights["retrieval"] * retrieval
                 + self.loss_weights["calibration"] * calibration)
        return {
            "total": total, "fusion": fusion,
            "intensity": intensity, "structure": structure, "gradient": gradient, "color": color,
            "recovery": recovery, "candidate": candidate, "anchor": anchor,
            "retrieval": retrieval, "adoption": adoption,
            "error_calibration": error_calibration, "interaction": interaction,
            # diagnostics; not part of total
            "harmful_correction_rate": torch.stack(harmful_after).mean(),
            "input_teacher_mse": torch.stack(harmful_before).mean(),
        }
