"""CalibFuse 训练监督：融合损失 + 干净教师校准正则。

损失分两大类（符号约定：fused 为网络输出，Y_v/Y_i 为干净源，
teacher 为 EMA 教师在干净图像上的特征）：

**融合损失（只作用于最终输出）**
- ``intensity`` 强度损失：目标为 max(Y_v 亮度, Y_i) 的 max 亮度融合规则；
- ``structure`` 结构损失：SSIM（11x11 高斯窗）；
- ``gradient`` 梯度损失：逐像素取梯度能量更强的源梯度；
- ``color`` 颜色损失：可见光源的 YCbCr 色度（Cb/Cr），仅约束色度，
  亮度由前三项处理。

**校准正则（以 EMA 教师的干净特征为监督）**
- ``recovery`` / ``candidate`` / ``anchor``：把恢复后特征、修正结果、
  字典锚点拉向教师的干净特征；其中 anchor 项是字典参数唯一的梯度路径；
- ``adoption`` / ``interaction``：把学习到的门控回归到
  :func:`optimal_gain`——沿提议方向的岭回归最优逐像素系数；
  干净样本的增益被强制为 0；交互门控以
  ``maximum_transfer * source_reliability`` 为上限做归一化回归；
- ``error_calibration``：把预测的对数误差回归到真实误差，
  使 ``reliability = exp(-error)`` 有明确语义；
- ``retrieval``：带噪样本与干净教师的字典分配的 KL 散度。

``harmful_correction_rate``（恢复后误差反而变大的像素比例）与
``input_teacher_mse``（恢复前对教师的 MSE）仅作诊断报告，
不参与梯度。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .image import luminance, spatial_gradient


def fusion_targets(visible: Tensor, infrared: Tensor) -> dict[str, Tensor]:
    """构造融合目标：max 亮度目标与逐像素更强源的梯度目标。

    经典融合规则的可微形式：
    - 亮度目标 = max(可见光亮度, 红外)——保留两源中最亮的结构；
    - 梯度目标 = 在每个像素上，取 (gx² + gy²) 能量更大的那个源的梯度
      ——保留两源中最锐利的边缘。

    参数:
        visible: (B, 3, H, W) 干净可见光。
        infrared: (B, 1, H, W) 干净红外。

    返回:
        {"luminance": (B,1,H,W), "gradient_x": (B,1,H,W), "gradient_y": (B,1,H,W)}
    """
    visible_y = luminance(visible)
    visible_x, visible_y_gradient = spatial_gradient(visible_y)
    infrared_x, infrared_y = spatial_gradient(infrared)
    # 逐像素比较两个源的梯度能量，决定该位置的梯度来自哪个源
    select_infrared = infrared_x.square() + infrared_y.square() > (
        visible_x.square() + visible_y_gradient.square()
    )
    return {
        "luminance": torch.maximum(visible_y, infrared),
        "gradient_x": torch.where(select_infrared, infrared_x, visible_x),
        "gradient_y": torch.where(select_infrared, infrared_y, visible_y_gradient),
    }


def _rgb_to_ycbcr(x: Tensor) -> Tensor:
    """RGB -> YCbCr（浮点，色度以 0.5 为中心）；通道顺序 (Y, Cb, Cr)。

    使用 BT.601 的 Y 定义，Cb/Cr 为常见简化形式（未做完整 16-235
    量化，因为网络工作在 [0,1] 浮点域）。
    """
    y = luminance(x)
    cb = 0.5 + (x[:, 2:3] - y) * 0.564
    cr = 0.5 + (x[:, 0:1] - y) * 0.713
    return torch.cat((y, cb, cr), 1)


def _ssim_loss(x: Tensor, y: Tensor) -> Tensor:
    """11x11 均值窗 SSIM 损失：``mean(0.5 * (1 - SSIM))``，截断到 [0,1]。

    统计量用 avg_pool2d（11x11 box 窗）近似高斯加权，
    C1=(0.01)²、C2=(0.03)² 为标准稳定项。
    """
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
    """字典分配的 KL(student || teacher)，逐位置计算后对 (H, W) 取均值。

    监督目标：带噪样本的原子分配应接近教师在干净图像上的分配
    ——即"退化不应改变特征在字典流形上的位置"。
    教师分配 detach；两侧加 clamp_min(1e-6) 防 log(0)。
    """
    target = teacher.detach().float().clamp_min(1e-6)
    prediction = student.float().clamp_min(1e-6)
    return (target * (target.log() - prediction.log())).sum(1).mean((1, 2))


def _feature_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """特征级 SmoothL1（beta=0.1），目标端 detach。

    beta=0.1 在小误差区域退化为 L2、大误差区域为 L1，
    对特征残差的离群值鲁棒。
    """
    return F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=0.1)


def optimal_gain(origin: Tensor, direction: Tensor, target: Tensor,
                 upper: Tensor | float = 1.0, eps: float = 1e-4) -> Tensor:
    """沿提议方向的"岭回归最优"逐像素系数（detach 的监督目标）。

    对每个像素求解 ``origin + g * direction ≈ target`` 的最小二乘解::

        g* = <target - origin, direction> / <direction, direction>

    然后截断到 [0, upper]；方向能量可忽略（近零）的位置增益置 0。
    整体在 no_grad 下计算——它是监督目标，不参与反向图。

    用途：
    - 采纳门控：origin=z、direction=candidate、target=教师干净特征
      ——"最优的修正幅度是多少"；
    - 交互门控：origin=接收方特征、direction=消息方向、
      upper=maximum_transfer x 源可靠性——"最优的消息注入量是多少"。

    参数:
        origin: (B, C, H, W) 起点（当前特征）。
        direction: (B, C, H, W) 提议方向（候选修正或消息）。
        target: (B, C, H, W) 监督目标（教师干净特征）。
        upper: 增益上限；浮点或与 origin 同形的张量（逐像素上限）。
        eps: 分母稳定项与"方向近零"的判定阈值。

    返回:
        (B, 1, H, W) 的最优增益图。
    """
    with torch.no_grad():
        # 通道维内积：分子 = Σ_c (target-origin)·direction
        numerator = ((target.float() - origin.float()) * direction.float()).sum(1, keepdim=True)
        denominator = direction.float().square().sum(1, keepdim=True).add(eps)
        gain = (numerator / denominator).clamp_min(0.0)
        if torch.is_tensor(upper):
            # 逐像素上限（detach 后再截断，避免上限回传梯度）
            gain = torch.minimum(gain, upper.detach().float())
        else:
            gain = gain.clamp_max(float(upper))
        # 方向能量近零时增益无意义（除法不稳定），置 0
        valid = direction.float().square().mean(1, keepdim=True) > eps
        return torch.where(valid, gain, torch.zeros_like(gain))


class CalibFuseLoss(nn.Module):
    """CalibFuse 总损失；各损失项见模块级文档。

    组合式::

        total = fusion
              + w_recovery * (recovery + 0.25 * candidate)
              + w_anchor   * anchor
              + w_retrieval * retrieval
              + w_calibration * (adoption + error + interaction)

    其中 fusion = 2.0*intensity + 1.0*structure + 2.0*gradient + 1.5*color。
    """

    # 默认权重（如需覆盖必须提供完全相同的键集合）
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
                 maximum_transfer: float = 0.15,
                 term_scales: dict[str, float] | None = None) -> None:
        """初始化损失模块。

        参数:
            loss_weights: 覆盖默认权重；键必须与 DEFAULTS 完全一致，
                且全部为有限非负数。
            maximum_transfer: 必须与网络跨模态注意力的同名参数一致，
                用于交互门控回归的归一化（除以它，使不同上限的
                门控损失量级可比）。
            term_scales: 消融用的逐项缩放（默认全部 1.0，即无行为变化）。
                允许的键：``recovery`` / ``candidate`` / ``anchor`` /
                ``retrieval`` / ``adoption`` / ``error_calibration`` /
                ``interaction``；值为 0 表示完全去掉该项。
                与 loss_weights 的区别：loss_weights 作用于损失家族
                （calibration 家族捆绑 adoption+error+interaction），
                term_scales 可拆到单项，供消融实验使用。
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
        allowed = ("recovery", "candidate", "anchor", "retrieval",
                   "adoption", "error_calibration", "interaction")
        scales = dict(term_scales) if term_scales else {}
        if not set(scales) <= set(allowed):
            raise ValueError(f"term_scales keys must be a subset of {list(allowed)}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in scales.values()):
            raise ValueError("All term scales must be finite and nonnegative")
        self.term_scales = scales

    def term_scale(self, name: str) -> float:
        """取单项损失的消融缩放；未指定的项返回 1.0。"""
        return self.term_scales.get(name, 1.0)

    def forward(self, output: dict, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """计算全部损失项。

        参数:
            output: ``CalibFuse(return_auxiliary=True)`` 的返回值，
                含 fused / states / teacher / clean_anchor_references。
            batch: 数据批次，提供干净对与退化掩码
                （掩码全零 <=> 该模态样本干净）。

        返回:
            字典：``total`` 与全部分量；末尾两项
            （``harmful_correction_rate`` / ``input_teacher_mse``）
            为无梯度诊断量。
        """
        # ---- 融合损失：最终输出 vs 干净源构造的目标 ----
        fused = output["fused"].float()
        visible, infrared = batch["visible"].float(), batch["infrared"].float()
        target = fusion_targets(visible, infrared)
        yf = luminance(fused)
        # 强度：SmoothL1(beta=0.02) 对 max 亮度目标
        intensity = F.smooth_l1_loss(yf, target["luminance"], beta=0.02)
        # 结构：SSIM 损失
        structure = _ssim_loss(yf, target["luminance"])
        # 梯度：x/y 两个方向的 SmoothL1(beta=0.01) 取平均
        gf_x, gf_y = spatial_gradient(fused)
        gradient = 0.5 * (
            F.smooth_l1_loss(gf_x, target["gradient_x"], beta=0.01)
            + F.smooth_l1_loss(gf_y, target["gradient_y"], beta=0.01)
        )
        # 颜色：只约束 Cb/Cr 色度（可见光的色度是唯一的颜色来源；
        # 红外为灰度不提供色度）
        color = F.smooth_l1_loss(_rgb_to_ycbcr(fused)[:, 1:],
                                 _rgb_to_ycbcr(visible)[:, 1:], beta=0.02)

        # ---- 校准正则：逐尺度 x 逐模态累计 ----
        recovery_terms, candidate_terms, anchor_terms = [], [], []
        adoption_terms, error_terms, retrieval_terms, interaction_terms = [], [], [], []
        harmful_before, harmful_after = [], []
        for index, state in enumerate(output["states"]):
            for modality in ("visible", "infrared"):
                branch = state[modality]
                # 教师干净特征是监督目标（detach，不构成梯度通路）
                teacher = output["teacher"][f"{modality}_features"][index].detach()
                # damage 掩码全零 => 该模态此样本为干净
                clean = batch[f"{modality}_damage"].flatten(1).amax(1) == 0
                corrupted = ~clean
                # 恢复损失：恢复后特征 -> 教师干净特征
                recovery_terms.append(_feature_loss(branch["recovered"], teacher))
                # 锚点损失：干净特征上可导的字典检索结果 -> 教师特征。
                # 这是字典参数唯一的梯度路径
                anchor = output["clean_anchor_references"][modality][index]
                anchor_terms.append(_feature_loss(anchor, teacher))

                # 采纳监督：回归到岭回归最优增益；干净样本增益强制为 0
                # （干净 => 不需要修正）
                alpha_target = optimal_gain(branch["input"], branch["candidate"], teacher)
                if clean.any():
                    alpha_target[clean] = 0.0
                adoption_terms.append(F.smooth_l1_loss(branch["adoption"].float(), alpha_target,
                                                       beta=0.1))
                # 候选修正与检索 KL 只对退化样本有意义
                # （干净样本没有可修正的内容）
                if corrupted.any():
                    candidate_terms.append(_feature_loss(
                        branch["input"][corrupted] + branch["candidate"][corrupted],
                        teacher[corrupted]))
                    retrieval_terms.append(retrieval_divergence(
                        branch["assignment"][corrupted],
                        output["teacher"][f"{modality}_assignments"][index][corrupted]).mean())

                # 误差校准：真实误差取 log1p(逐像素对教师的 MSE)，
                # 与网络预测的 softplus 有界对数误差对齐
                error_target = torch.log1p(
                    (branch["recovered"].float() - teacher.float()).square().mean(1, keepdim=True)
                ).detach()
                error_terms.append(F.smooth_l1_loss(branch["predicted_log_error"], error_target,
                                                     beta=0.05))
                # 诊断量（无梯度）：恢复前 MSE，以及"恢复后误差反而增大
                # 超过 1e-4"的像素占比（有害修正率）
                with torch.no_grad():
                    before = (branch["input"].float() - teacher.float()).square().mean(1)
                    after = (branch["recovered"].float() - teacher.float()).square().mean(1)
                    harmful_before.append(before.mean())
                    harmful_after.append((after > before + 1e-4).float().mean())

            # 两个消息方向的交互门控监督
            for message_key, target_modality in (("i_to_v", "visible"), ("v_to_i", "infrared")):
                message = state[message_key]
                receiver = state[target_modality]["recovered"]
                teacher = output["teacher"][f"{target_modality}_features"][index].detach()
                # 上限 = maximum_transfer x 发送方（源）模态的可靠性：
                # 源越不可靠，允许注入的消息越少
                upper = self.maximum_transfer * (
                    state["infrared" if target_modality == "visible" else "visible"]["reliability"]
                )
                # 目标：受上限约束的岭回归最优注入量
                gate_target = optimal_gain(receiver, message["direction"], teacher, upper=upper)
                # 双方同除 maximum_transfer 归一化，
                # 使不同上限的门控损失量级一致
                interaction_terms.append(F.smooth_l1_loss(
                    message["gate"].float() / self.maximum_transfer,
                    gate_target / self.maximum_transfer, beta=0.1))

        # 跨尺度、跨模态取平均；term_scale 为该项的消融缩放（默认 1.0）
        recovery = torch.stack(recovery_terms).mean() * self.term_scale("recovery")
        # 全干净 batch 没有候选/检索项；用 0*x 保持计算图连通与键完整
        candidate = (torch.stack(candidate_terms).mean() if candidate_terms
                     else fused.sum() * 0.0) * self.term_scale("candidate")
        anchor = torch.stack(anchor_terms).mean() * self.term_scale("anchor")
        retrieval = (torch.stack(retrieval_terms).mean() if retrieval_terms
                     else fused.sum() * 0.0) * self.term_scale("retrieval")
        adoption = torch.stack(adoption_terms).mean() * self.term_scale("adoption")
        error_calibration = torch.stack(error_terms).mean() * self.term_scale("error_calibration")
        interaction = torch.stack(interaction_terms).mean() * self.term_scale("interaction")
        # 校准正则族 = 采纳 + 误差校准 + 交互门控
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
            # 诊断量；不计入 total
            "harmful_correction_rate": torch.stack(harmful_after).mean(),
            "input_teacher_mse": torch.stack(harmful_before).mean(),
        }
