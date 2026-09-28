"""CalibFuse 训练监督：融合损失 + 干净教师驱动的校准正则。

损失分两大类（由 :class:`CalibFuseLoss` 汇总）：

**一、融合损失**（只看最终输出，经典图像融合监督）：
- ``intensity``：融合亮度逼近 ``max(vis, ir)`` 亮度——保留两模态
  中更亮（信息更强）的响应；
- ``structure``：SSIM 结构损失；
- ``gradient``：融合梯度逼近"逐像素更强梯度模态"的梯度——纹理
  从两个源中择优继承；
- ``color``：YCbCr 色度（Cb/Cr）回归可见光色度——颜色只来自可见光。

**二、校准正则**（本项目的特色，全部由 EMA 教师的**干净特征**驱动，
``output["teacher"]`` 在 no_grad 下产生）：
- ``recovery``/``candidate``/``anchor``：把恢复特征、候选修正、
  字典锚点拉向教师的干净编码特征；其中 ``anchor`` 项是字典参数
  接收梯度的唯一路径（见 nets/dictionary.py 的梯度隔离说明）；
- ``adoption``/``interaction``：把学到的门（采纳率、交互门）回归到
  :func:`optimal_gain` ——**沿提议方向、以教师特征为目标的岭最优
  逐像素系数**。干净样本的最优增益强制为 0（干净就别改）；
  交互门的最优增益还被"幅度上限 × 源可靠度"截断（不可靠的源
  不许多传）。这是"收益校准"的监督落地：网络学**何时改、改多少**；
- ``error_calibration``：回归预测 log 误差到真实 log 误差，使
  ``reliability = exp(-误差)`` 有意义；
- ``retrieval``：噪声样本的字典软分配与教师干净分配的 KL 散度，
  让噪声下的检索行为向干净检索对齐。

``harmful_correction_rate`` 与 ``input_teacher_mse`` 只统计不上梯度：
前者是恢复后误差反而变大的像素比例，后者是恢复前输入对教师特征
的 MSE（作为 baseline 参照）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .image import luminance, spatial_gradient


def fusion_targets(visible: Tensor, infrared: Tensor) -> dict[str, Tensor]:
    """由干净源图构造融合的目标（经典融合监督的目标构造法）。

    - 亮度目标：``max(可见光亮度, 红外)``——两者取更亮者，保证热目标
      与可见光高光都被保留；
    - 梯度目标：逐像素比较两模态的梯度平方和，谁强取谁的
      (gx, gy)——纹理按位置择优继承。

    参数:
        visible: 干净可见光 (B,3,H,W)。
        infrared: 干净红外 (B,1,H,W)。

    返回:
        字典：``luminance`` (B,1,H,W)、``gradient_x``/``gradient_y``
        (B,1,H,W)。
    """
    visible_y = luminance(visible)
    visible_x, visible_y_gradient = spatial_gradient(visible_y)
    infrared_x, infrared_y = spatial_gradient(infrared)
    # 逐像素判定红外梯度能量是否强于可见光
    select_infrared = infrared_x.square() + infrared_y.square() > (
        visible_x.square() + visible_y_gradient.square()
    )
    return {
        "luminance": torch.maximum(visible_y, infrared),
        "gradient_x": torch.where(select_infrared, infrared_x, visible_x),
        "gradient_y": torch.where(select_infrared, infrared_y, visible_y_gradient),
    }


def _rgb_to_ycbcr(x: Tensor) -> Tensor:
    """RGB → YCbCr（浮点版，色度中心 0.5）。

    只用于色度损失：Cb/Cr 与 Y 分开，允许融合图自带亮度结构，
    而色度只需对齐可见光。

    参数:
        x: (B,3,H,W) RGB。

    返回:
        (B,3,H,W)，通道顺序 (Y, Cb, Cr)。
    """
    y = luminance(x)
    cb = 0.5 + (x[:, 2:3] - y) * 0.564
    cr = 0.5 + (x[:, 0:1] - y) * 0.713
    return torch.cat((y, cb, cr), 1)


def _ssim_loss(x: Tensor, y: Tensor) -> Tensor:
    """11x11 均值窗口的 SSIM 损失（1-SSIM 的一半，截断到 [0,1]）。

    用均值池化近似局部统计（比高斯窗快），C1/C2 取标准值
    (0.01, 0.03)²。乘 0.5 使数值范围与其它损失项更匹配。

    参数:
        x, y: (B,1,H,W) 图（通常为亮度通道）。

    返回:
        标量损失。
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
    """字典软分配的 KL(student‖teacher)（按原子求和、按位置平均）。

    教师分配 detach，只约束学生。让噪声输入下的检索分布向干净
    输入下的检索分布靠拢——噪声不应显著改变"该用哪些字典原子"。

    参数:
        student: 学生分配 (B, K, H, W)。
        teacher: 教师分配 (B, K, H, W)。

    返回:
        (B, H, W) 的逐位置散度（调用方负责进一步平均）。
    """
    target = teacher.detach().float().clamp_min(1e-6)
    prediction = student.float().clamp_min(1e-6)
    return (target * (target.log() - prediction.log())).sum(1).mean((1, 2))


def _feature_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """特征级 SmoothL1（beta=0.1），目标侧 detach。

    用于恢复/锚点/候选等特征回归项：SmoothL1 对离群值比 MSE 温和，
    小 beta 使其在小误差区接近 L1。
    """
    return F.smooth_l1_loss(prediction.float(), target.detach().float(), beta=0.1)


def optimal_gain(origin: Tensor, direction: Tensor, target: Tensor,
                 upper: Tensor | float = 1.0, eps: float = 1e-4) -> Tensor:
    """Detached ridge-optimal per-pixel coefficient along a proposed direction.

    沿提议方向的最优逐像素增益（岭回归闭式解，已 detach）。

    给定起点 ``origin``、方向 ``direction``、目标 ``target``，求标量
    ``g`` 使 ``origin + g·direction`` 最接近 target::

        g* = <target - origin, direction> / <direction, direction>

    约束：g* ≥ 0（不许反向修正）；≤ upper（幅度上限，交互门用
    "maximum_transfer × 源可靠度" 作为上限）；方向能量过小的位置
    置 0（避免除小数放大噪声）。全程 no_grad——它是**监督目标**，
    不是计算图的一部分。

    参数:
        origin: 起点特征 (B,C,H,W)，如字典输入或恢复后特征。
        direction: 提议方向 (B,C,H,W)，如候选修正或消息方向。
        target: 目标特征 (B,C,H,W)（通常为教师干净特征）。
        upper: 增益上限，标量或 (B,1,H,W) 张量。
        eps: 数值稳定小量。

    返回:
        (B,1,H,W) 最优增益（无梯度）。
    """
    with torch.no_grad():
        numerator = ((target.float() - origin.float()) * direction.float()).sum(1, keepdim=True)
        denominator = direction.float().square().sum(1, keepdim=True).add(eps)
        gain = (numerator / denominator).clamp_min(0.0)
        if torch.is_tensor(upper):
            gain = torch.minimum(gain, upper.detach().float())
        else:
            gain = gain.clamp_max(float(upper))
        # 方向几乎为零的位置：增益无意义，置 0
        valid = direction.float().square().mean(1, keepdim=True) > eps
        return torch.where(valid, gain, torch.zeros_like(gain))


class CalibFuseLoss(nn.Module):
    """CalibFuse 总损失。

    权重默认值（DEFAULTS）：
    - 融合项：intensity 2.0、structure 1.0、gradient 2.0、color 1.5；
    - 正则项：recovery 0.10、anchor 0.10、retrieval 0.05、
      calibration 0.05（calibration = adoption + error + interaction）。

    total = fusion
          + recovery 权重 × (recovery + 0.25·candidate)
          + anchor 权重 × anchor
          + retrieval 权重 × retrieval
          + calibration 权重 × calibration
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
        """初始化并校验损失权重。

        参数:
            loss_weights: 覆盖默认权重；必须**恰好**包含 DEFAULTS
                的八个键（多/少/错名都抛错），值为非负有限数。
            maximum_transfer: 交互门幅度上限，需与网络
                ReliabilityWeightedCrossAttention 的设置一致
                （默认 0.15）；用于把门归一化后回归。
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
        """计算全部损失项。

        参数:
            output: ``CalibFuse(return_auxiliary=True)`` 的输出，需含
                ``fused``、``states``（各尺度字典/消息状态）、
                ``teacher``（EMA 教师干净特征与分配）、
                ``clean_anchor_references``（不 detach 字典的干净检索）。
            batch: 数据集批次，需含干净图 ``visible``/``infrared`` 与
                退化掩码 ``visible_damage``/``infrared_damage``
                （用于判定样本是否干净）。

        返回:
            字典：``total`` 及各分项（intensity/structure/gradient/
            color、recovery/candidate/anchor/retrieval/adoption/
            error_calibration/interaction、harmful_correction_rate、
            input_teacher_mse）。后两项为诊断量，不参与反传。
        """
        # —— 融合项：只用最终输出与干净源图 ——
        fused = output["fused"].float()
        visible, infrared = batch["visible"].float(), batch["infrared"].float()
        target = fusion_targets(visible, infrared)
        yf = luminance(fused)
        # 亮度：小 beta SmoothL1，贴近 L1 但零点平滑
        intensity = F.smooth_l1_loss(yf, target["luminance"], beta=0.02)
        structure = _ssim_loss(yf, target["luminance"])
        # 梯度：对融合图全彩求梯度后逐向回归目标
        gf_x, gf_y = spatial_gradient(fused)
        gradient = 0.5 * (
            F.smooth_l1_loss(gf_x, target["gradient_x"], beta=0.01)
            + F.smooth_l1_loss(gf_y, target["gradient_y"], beta=0.01)
        )
        # 颜色：只回归 Cb/Cr（[:, 1:]），Y 由其它项管
        color = F.smooth_l1_loss(_rgb_to_ycbcr(fused)[:, 1:],
                                 _rgb_to_ycbcr(visible)[:, 1:], beta=0.02)

        # —— 正则项：遍历三个尺度的状态 ——
        recovery_terms, candidate_terms, anchor_terms = [], [], []
        adoption_terms, error_terms, retrieval_terms, interaction_terms = [], [], [], []
        harmful_before, harmful_after = [], []
        for index, state in enumerate(output["states"]):
            for modality in ("visible", "infrared"):
                branch = state[modality]
                # 教师干净特征：detach，监督信号而非计算路径
                teacher = output["teacher"][f"{modality}_features"][index].detach()
                # 用退化掩码判定该样本此模态是否干净
                # （damage 全零 ⇒ 干净）
                clean = batch[f"{modality}_damage"].flatten(1).amax(1) == 0
                corrupted = ~clean
                # 恢复项：恢复特征 → 教师特征
                recovery_terms.append(_feature_loss(branch["recovered"], teacher))
                # 锚点项：干净特征经字典（不 detach）检索的参考 → 教师。
                # 这是字典参数获得梯度的唯一通道
                anchor = output["clean_anchor_references"][modality][index]
                anchor_terms.append(_feature_loss(anchor, teacher))

                # 采纳率监督：最优增益 = argmin ||z + g·candidate - teacher||；
                # 干净样本强制 g*=0（"干净就别动"）
                alpha_target = optimal_gain(branch["input"], branch["candidate"], teacher)
                if clean.any():
                    alpha_target[clean] = 0.0
                adoption_terms.append(F.smooth_l1_loss(branch["adoption"].float(), alpha_target,
                                                       beta=0.1))
                # 候选项与检索项只统计**受污染**样本（干净样本无退化可恢复）
                if corrupted.any():
                    candidate_terms.append(_feature_loss(
                        branch["input"][corrupted] + branch["candidate"][corrupted],
                        teacher[corrupted]))
                    # 检索分布对齐：噪声分配 vs 教师干净分配的 KL
                    retrieval_terms.append(retrieval_divergence(
                        branch["assignment"][corrupted],
                        output["teacher"][f"{modality}_assignments"][index][corrupted]).mean())

                # 误差校准：真实 log 误差 = log1p(逐像素 MSE(recovered, teacher))
                error_target = torch.log1p(
                    (branch["recovered"].float() - teacher.float()).square().mean(1, keepdim=True)
                ).detach()
                error_terms.append(F.smooth_l1_loss(branch["predicted_log_error"], error_target,
                                                     beta=0.05))
                # 诊断量（无梯度）：恢复前/后的教师距离，
                # after > before 的比例即"帮倒忙"率
                with torch.no_grad():
                    before = (branch["input"].float() - teacher.float()).square().mean(1)
                    after = (branch["recovered"].float() - teacher.float()).square().mean(1)
                    harmful_before.append(before.mean())
                    harmful_after.append((after > before + 1e-4).float().mean())

            # —— 交互门监督（双向消息）——
            for message_key, target_modality in (("i_to_v", "visible"), ("v_to_i", "infrared")):
                message = state[message_key]
                receiver = state[target_modality]["recovered"]
                teacher = output["teacher"][f"{target_modality}_features"][index].detach()
                # 上限 = maximum_transfer × **源模态**可靠度：
                # 源越不可靠，允许传输的增益越小
                upper = self.maximum_transfer * (
                    state["infrared" if target_modality == "visible" else "visible"]["reliability"]
                )
                # 门的最优增益目标（岭最优 + 上限截断）
                gate_target = optimal_gain(receiver, message["direction"], teacher, upper=upper)
                # 双方同除 maximum_transfer 归一化后再回归，
                # 使不同上限下的门可比
                interaction_terms.append(F.smooth_l1_loss(
                    message["gate"].float() / self.maximum_transfer,
                    gate_target / self.maximum_transfer, beta=0.1))

        # —— 汇总：各项在尺度/模态间取平均 ——
        recovery = torch.stack(recovery_terms).mean()
        # batch 全干净时无候选项/检索项，用 0 占位保持键存在
        candidate = (torch.stack(candidate_terms).mean() if candidate_terms
                     else fused.sum() * 0.0)
        anchor = torch.stack(anchor_terms).mean()
        retrieval = (torch.stack(retrieval_terms).mean() if retrieval_terms
                     else fused.sum() * 0.0)
        adoption = torch.stack(adoption_terms).mean()
        error_calibration = torch.stack(error_terms).mean()
        interaction = torch.stack(interaction_terms).mean()
        # 校准 = 采纳率 + 误差校准 + 交互门
        calibration = adoption + error_calibration + interaction
        fusion = (self.loss_weights["intensity"] * intensity
                  + self.loss_weights["structure"] * structure
                  + self.loss_weights["gradient"] * gradient
                  + self.loss_weights["color"] * color)
        # 总损失：融合 + 三个正则族（candidate 以 0.25 折入 recovery）
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
            # 以下两项为诊断量，不在 total 中
            "harmful_correction_rate": torch.stack(harmful_after).mean(),
            "input_teacher_mse": torch.stack(harmful_before).mean(),
        }
