"""CalibFuse fusion network.

Three-scale symmetric U-shaped model for degradation-robust visible/infrared
fusion:

- separate vis (3-ch) / ir (1-ch) stems and per-scale Transformer encoders;
- a :class:`CalibratedFusionBlock` per scale: dictionary recovery,
  bidirectional reliability-weighted cross-modal messages, and fusion;
- fused features cascade downward as the ``prior_fused`` prior;
- :class:`CompactDecoder` refines bottom-up with residual skip connections;
- ``fusion_head`` (TransformerBlock + conv + sigmoid) outputs RGB in [0, 1].

``cross_budget_scale`` ramps 0 -> 1 over epochs 5-20, so early training
focuses on per-modality recovery before any cross-modal transfer. Inputs
are padded to a multiple of 4 and cropped back afterwards, so arbitrary
sizes work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dictionary import BenefitCalibratedDictionary, LocalResidual, rms_normalize
from .restormer import Downsample, OverlapPatchEmbed, TransformerBlock


def transformer_block(channels: int, heads: int) -> TransformerBlock:
    """Create a Restormer-style block with the project-wide settings.

    Every TransformerBlock in the network is built through this factory:
    FFN expansion 2.0, no conv bias, ``WithBias`` LayerNorm.
    """
    return TransformerBlock(channels, heads, 2.0, False, "WithBias")


def upsample(x: Tensor, size: tuple[int, int]) -> Tensor:
    """Bilinear upsample to the target spatial size."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def pad_to_multiple(x: Tensor, multiple: int = 4) -> tuple[Tensor, tuple[int, int]]:
    """Pad spatial dims to a multiple of ``multiple``.

    Returns ``(padded tensor, original (H, W))``; padding is applied on the
    right and bottom only.
    """
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        # reflect padding requires the pad amount to stay below the side length
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (height, width)


class ReliabilityWeightedCrossAttention(nn.Module):
    """Reliability-weighted cross-modal message passing.

    The message from source to target is::

        message  = gate * direction
        gate     = maximum_transfer * budget_scale * source_reliability * benefit

    Information flows only when the source is reliable and the predicted
    benefit is high. The attention itself is a reliability-weighted cosine
    similarity between RMS-normalized queries and keys (magnitude
    insensitive), not a plain QK^T.
    """

    def __init__(self, channels: int, heads: int, maximum_transfer: float = 0.15) -> None:
        """Initialize the cross-modal attention module."""
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.maximum_transfer = maximum_transfer
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.target_value = nn.Conv2d(channels, channels, 1)
        self.source_value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.source_scale_raw = nn.Parameter(torch.tensor(-2.0))
        hidden = max(12, channels // 2)
        self.benefit = nn.Sequential(
            nn.Conv2d(3 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # near-identity init: benefit starts at sigmoid(-2) ≈ 0.12, so
        # cross-modal transfer is nearly closed at the start of training
        nn.init.normal_(self.benefit[-1].weight, std=1e-3)
        nn.init.constant_(self.benefit[-1].bias, -2.0)

    def message(self, target: Tensor, source: Tensor, source_reliability: Tensor,
                target_reliability: Tensor, budget_scale: float) -> dict[str, Tensor]:
        """Compute one source -> target message.

        Returns ``direction`` (amplitude-aligned to the receiver),
        ``gate``, ``message = gate * direction``, and ``attention``
        (diagnostics).
        """
        # RMS-normalize both sides so the attention ignores energy-scale
        # differences between the two modalities
        target_n, source_n = rms_normalize(target), rms_normalize(source)
        batch, _, height, width = target.shape
        shape = (batch, self.heads, self.head_dim, height * width)
        query = self.q(target_n).float().reshape(shape)
        key = self.k(source_n).float().reshape(shape)
        weight = source_reliability.float().reshape(batch, 1, 1, height * width)

        # reliability-weighted cosine correlation in place of QK^T
        numerator = torch.matmul(query * weight, key.transpose(-2, -1))
        query_norm = (query.square() * weight).sum(-1, keepdim=True)
        key_norm = (key.square() * weight).sum(-1, keepdim=True)
        denominator = (query_norm * key_norm.transpose(-2, -1)).add(1e-6).sqrt()
        correlation = numerator / denominator
        attention = (correlation * self.temperature.float().clamp(0.05, 10.0)).softmax(-1)

        target_value = self.target_value(target_n).float().reshape(shape)
        # unreliable source positions contribute less
        source_value = self.source_value(source_n).float().reshape(shape) * weight
        source_scale = 0.25 * torch.sigmoid(self.source_scale_raw.float())
        raw = torch.matmul(attention, target_value) + source_scale * torch.matmul(attention, source_value)
        raw = self.out(raw.reshape(batch, self.channels, height, width).to(target.dtype))

        # amplitude calibration: direction says where to move, the gate
        # decides by how much
        raw_rms = raw.float().square().mean(1, keepdim=True).add(1e-6).sqrt()
        receiver_rms = target.float().square().mean(1, keepdim=True).add(1e-6).sqrt().detach()
        direction = (raw.float() / raw_rms * receiver_rms).to(target.dtype)
        evidence = torch.cat((target_n, source_n, (target_n - source_n).abs(),
                              source_reliability, target_reliability), 1)
        benefit = torch.sigmoid(self.benefit(evidence).float()).to(target.dtype)
        # gate = magnitude cap x curriculum x source reliability (detached) x benefit
        budget = self.maximum_transfer * float(budget_scale) * source_reliability.detach()
        gate = budget * benefit
        return {"direction": direction, "gate": gate, "message": gate * direction,
                "attention": attention}


class CalibratedFusionBlock(nn.Module):
    """Per-scale recovery -> interaction -> fusion.

    Each modality runs a :class:`BenefitCalibratedDictionary` for recovery;
    bidirectional cross-modal messages enhance both branches; the enhanced
    features and the downsampled ``prior_fused`` from the previous scale
    are projected and refined into this scale's fused features.
    """

    def __init__(self, channels: int, heads: int, atoms: int, query_channels: int,
                 use_transformer_fusion: bool) -> None:
        """Initialize the fusion block.

        ``use_transformer_fusion=False`` switches the refinement to a
        lightweight :class:`LocalResidual` (used at the highest-resolution
        scale where global attention is unnecessary).
        """
        super().__init__()
        # standard convolutions for visible; axial factorization for infrared
        self.visible_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels)
        self.infrared_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels, axial=True)
        # both message directions share these weights
        self.interaction = ReliabilityWeightedCrossAttention(channels, heads)
        fusion_input = 3 * channels
        self.fuse_project = nn.Sequential(nn.Conv2d(fusion_input, channels, 1), nn.GELU())
        self.fuse_refine = (transformer_block(channels, heads) if use_transformer_fusion
                            else LocalResidual(channels))

    def forward(self, visible: Tensor, infrared: Tensor, prior_fused: Tensor | None,
                budget_scale: float) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Run recovery-interaction-fusion for this scale.

        Returns ``(visible, infrared, fused, state)`` where ``state`` holds
        both dictionary states and both messages for the training losses.
        """
        visible_state = self.visible_recovery(visible)
        infrared_state = self.infrared_recovery(infrared)
        v, i = visible_state["recovered"], infrared_state["recovered"]
        # bidirectional messages through the shared interaction module
        i_to_v = self.interaction.message(v, i, infrared_state["reliability"],
                                          visible_state["reliability"], budget_scale)
        v_to_i = self.interaction.message(i, v, visible_state["reliability"],
                                          infrared_state["reliability"], budget_scale)
        visible_out = v + i_to_v["message"]
        infrared_out = i + v_to_i["message"]
        if prior_fused is None:
            # highest-resolution scale: zero placeholder keeps the channel count
            prior_fused = torch.zeros_like(v)
        elif prior_fused.shape[-2:] != v.shape[-2:]:
            prior_fused = upsample(prior_fused, v.shape[-2:])
        fused = self.fuse_refine(self.fuse_project(torch.cat((visible_out, infrared_out, prior_fused), 1)))
        state = {
            "visible": visible_state, "infrared": infrared_state,
            "i_to_v": i_to_v, "v_to_i": v_to_i,
        }
        return v, i, fused, state


class CompactDecoder(nn.Module):
    """Bottom-up decoder: refine, upsample, and combine with skip connections.

    Deep fused features are refined, upsampled level by level, and merged
    with each scale's fused features through concat-refine-residual blocks.
    """

    def __init__(self, channels: tuple[int, int, int], heads: tuple[int, int, int]) -> None:
        """Initialize the decoder with per-scale channels and heads."""
        super().__init__()
        self.deep = transformer_block(channels[2], heads[2])
        self.up_projects = nn.ModuleList([
            nn.Conv2d(channels[index + 1], channels[index], 3, padding=1) for index in range(2)
        ])
        self.refines = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * channels[index], channels[index], 1), nn.GELU(),
                          transformer_block(channels[index], heads[index])) for index in range(2)
        ])

    def forward(self, features: list[Tensor]) -> Tensor:
        """Decode the three-scale fused features back to full resolution."""
        decoded = self.deep(features[2])
        for index in (1, 0):
            up = self.up_projects[index](upsample(decoded, features[index].shape[-2:]))
            # concat-refine-residual skip connection
            decoded = features[index] + self.refines[index](torch.cat((features[index], up), 1))
        return decoded


class CalibFuse(nn.Module):
    """CalibFuse: benefit-calibrated recovery + reliability-weighted interaction.

    Default configuration: channels (32, 64, 128), heads (4, 4, 8),
    atoms (64, 64, 64), query_channels (16, 24, 32) — 1,625,131 parameters
    across 12 Transformer blocks.
    """

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128),
                 heads: tuple[int, int, int] = (4, 4, 8),
                 atoms: tuple[int, int, int] = (64, 64, 64),
                 query_channels: tuple[int, int, int] = (16, 24, 32)) -> None:
        """Build the network; each argument must contain three values."""
        super().__init__()
        if not (len(channels) == len(heads) == len(atoms) == len(query_channels) == 3):
            raise ValueError("channels, heads, atoms, and query_channels must each contain three values")
        # stored in checkpoints as ``model_config`` so loading never depends
        # on code defaults
        self.config = {"channels": channels, "heads": heads, "atoms": atoms,
                       "query_channels": query_channels}
        # cross-modal budget: updated by set_training_progress during
        # training, fixed at 1.0 for inference
        self.cross_budget_scale = 1.0
        self.visible_stem = nn.Sequential(OverlapPatchEmbed(3, channels[0]), nn.GELU())
        self.infrared_stem = nn.Sequential(OverlapPatchEmbed(1, channels[0]), nn.GELU())
        self.visible_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        self.infrared_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        # the highest-resolution scale uses LocalResidual refinement
        self.blocks = nn.ModuleList([
            CalibratedFusionBlock(c, h, k, q, use_transformer_fusion=index > 0)
            for index, (c, h, k, q) in enumerate(zip(channels, heads, atoms, query_channels))
        ])
        self.visible_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.infrared_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.fused_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.decoder = CompactDecoder(channels, heads)
        # output head; forward applies the final sigmoid
        self.fusion_head = nn.Sequential(transformer_block(channels[0], heads[0]),
                                         nn.Conv2d(channels[0], 3, 3, padding=1))

    def set_training_progress(self, epoch: int) -> None:
        """Update the cross-modal budget: 0 before epoch 5, ramping to 1 by epoch 20."""
        self.cross_budget_scale = min(max((epoch - 5) / 15.0, 0.0), 1.0)

    def _encode(self, visible: Tensor, infrared: Tensor, return_auxiliary: bool) -> dict:
        """Multi-scale encode -> fuse -> decode; returns fused image and per-scale states."""
        v, i = self.visible_stem(visible), self.infrared_stem(infrared)
        scales, states, prior = [], [], None
        for index, block in enumerate(self.blocks):
            # re-normalize before each fusion block to keep feature energy
            # stable, matching the dictionary/attention assumptions
            v = rms_normalize(self.visible_encoder[index](v))
            i = rms_normalize(self.infrared_encoder[index](i))
            v, i, fused, state = block(v, i, prior, self.cross_budget_scale)
            scales.append(fused)
            if return_auxiliary:
                states.append(state)
            if index < 2:
                v = self.visible_down[index](v)
                i = self.infrared_down[index](i)
                prior = self.fused_down[index](fused)
        return {"fused": torch.sigmoid(self.fusion_head(self.decoder(scales))),
                "states": states}

    @torch.no_grad()
    def clean_reference_features(self, visible: Tensor, infrared: Tensor) -> dict:
        """Extract clean per-scale features and dictionary assignments.

        This is the clean-teacher path: encoding plus dictionary recovery
        only — no cross-modal interaction and no decoding. Downsampling
        uses the recovered features so the clean path mirrors the noisy one.
        """
        v, i = self.visible_stem(visible), self.infrared_stem(infrared)
        result = {"visible_features": [], "infrared_features": [],
                  "visible_assignments": [], "infrared_assignments": []}
        for index, block in enumerate(self.blocks):
            v = rms_normalize(self.visible_encoder[index](v))
            i = rms_normalize(self.infrared_encoder[index](i))
            result["visible_features"].append(v)
            result["infrared_features"].append(i)
            v_state = block.visible_recovery(v)
            i_state = block.infrared_recovery(i)
            result["visible_assignments"].append(v_state["assignment"])
            result["infrared_assignments"].append(i_state["assignment"])
            if index < 2:
                v = self.visible_down[index](v_state["recovered"])
                i = self.infrared_down[index](i_state["recovered"])
        return result

    def forward(self, visible: Tensor, infrared: Tensor, *, clean_visible: Tensor | None = None,
                clean_infrared: Tensor | None = None, return_auxiliary: bool = False,
                clean_teacher: nn.Module | None = None) -> dict:
        """Run the network.

        Inference needs only the two inputs and returns
        ``{"fused": (B, 3, H, W)}``. Training additionally passes clean
        references and the EMA teacher and receives ``states``,
        ``teacher``, and ``clean_anchor_references`` — the dictionary
        parameters receive gradients only through that clean anchor path.
        """
        if infrared.shape[1] != 1:
            infrared = infrared.mean(1, keepdim=True)
        # pad to a multiple of 4, then crop the output back
        visible, original_size = pad_to_multiple(visible)
        infrared, _ = pad_to_multiple(infrared)
        output = self._encode(visible, infrared, return_auxiliary)
        output["fused"] = output["fused"][..., :original_size[0], :original_size[1]]
        if not return_auxiliary:
            output.pop("states")
            return output
        # training-only: clean references and dictionary anchors
        if clean_visible is None or clean_infrared is None:
            raise ValueError("clean_visible and clean_infrared are required for auxiliary training output")
        if clean_infrared.shape[1] != 1:
            clean_infrared = clean_infrared.mean(1, keepdim=True)
        clean_visible, _ = pad_to_multiple(clean_visible)
        clean_infrared, _ = pad_to_multiple(clean_infrared)
        # the teacher defaults to self; training passes the EMA model
        teacher = self if clean_teacher is None else clean_teacher
        with torch.no_grad():
            clean = teacher.clean_reference_features(clean_visible, clean_infrared)
        output["teacher"] = clean
        # re-retrieve on clean features without detaching the dictionary:
        # the dictionary's only gradient path is this clean-anchor loss
        output["clean_anchor_references"] = {"visible": [], "infrared": []}
        for index, block in enumerate(self.blocks):
            for modality, recovery in (("visible", block.visible_recovery),
                                       ("infrared", block.infrared_recovery)):
                reference, _ = recovery.retrieve(clean[f"{modality}_features"][index],
                                                 detach_dictionary=False)
                output["clean_anchor_references"][modality].append(reference)
        return output
