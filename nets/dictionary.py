"""收益校准字典（Benefit-Calibrated Dictionary）模块。

本文件实现 CalibFuse 的核心恢复组件 :class:`BenefitCalibratedDictionary`。
它把"去退化"建模为三步：

1. **检索（retrieve）**：以当前（可能退化的）特征为查询，在可学习的
   原子字典中做软分配检索，得到一个"干净参考" ``reference``；
2. **候选校正（candidate）**：由 ``(z, reference, z - reference)`` 三路
   证据经卷积网络生成一个候选修正量，``tanh`` 限幅；
3. **采纳（adoption）**：逐像素预测采纳系数 ``adoption ∈ [0,1]``，
   最终 ``recovered = z + adoption * candidate``。

即网络学习的不是"怎么改"，而是**"哪里该改、改多少"**——这是整个项目
"benefit-calibrated"（收益校准）理念的来源。

此外模块还预测一个对数误差 ``predicted_log_error``，其负指数
``reliability = exp(-predicted_error)`` 作为该位置恢复结果的可信度，
供下游跨模态交互（:class:`~nets.fusion.ReliabilityWeightedCrossAttention`）
作为置信信号使用。

梯度隔离设计（重要）：
- 构造参数 ``clean_anchor_only=True``（默认）时，噪声路径的检索会把
  **字典本身 detach**（连同 key/value 投影），因此 ``self.dictionary``
  的梯度只来自训练循环中"干净锚点损失"那条路径（见
  ``CalibFuse.forward`` 里 ``detach_dictionary=False`` 的干净重检索）。
  ``tests/test_smoke.py`` 专门测试了这一梯度隔离行为，勿破坏。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """逐样本 RMS 归一化：除以全体特征元素平方的均值平方根。

    与 LayerNorm/InstanceNorm 不同，这里把 (C,H,W) 一起平均，等价于
    对每个样本估计一个全局幅度尺度后除掉——**只消除能量差异，不改变
    空间对比结构**。可见光与红外特征的幅值分布差异很大，本函数是全
    网络统一使用的"能量对齐"手段（stem 后、字典/交互前都会调用）。

    参数:
        x: 输入张量 (B, C, H, W)，任意 dtype。
        eps: 防除零小量。

    返回:
        归一化后的张量，形状与 dtype 同输入。
    """
    scale = x.float().square().mean((1, 2, 3), keepdim=True).add(eps).sqrt()
    return (x.float() / scale).to(x.dtype)


class LocalResidual(nn.Module):
    """Small local residual block used where global attention is unnecessary.

    轻量局部残差块：1x1 升维 → 5x5 深度卷积 → 1x1 降维，外加残差连接。
    用于最高分辨率尺度的融合精炼（全局注意力在该层开销大且非必要）。

    采用**近恒等初始化**：输出投影权重 std=1e-3、偏置为 0，因此训练
    初始时该块几乎等于恒等映射（输出 ≈ 输入），网络随后从"不改"学起，
    避免随机初始化的精炼块在训练早期破坏已对齐的特征。
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """初始化局部残差块。

        参数:
            channels: 输入/输出通道数。
            expansion: 隐藏层通道扩张倍数，默认 2。
        """
        super().__init__()
        hidden = channels * expansion
        self.project_in = nn.Conv2d(channels, hidden, 1)
        # 5x5 深度（分组）卷积：感受野大而参数量小
        self.spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.project_out = nn.Conv2d(hidden, channels, 1)
        # 近恒等初始化：输出投影几乎为零 ⇒ 初始输出 ≈ x
        nn.init.normal_(self.project_out.weight, std=1e-3)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """前向：先 RMS 归一化再做残差卷积。

        参数:
            x: 输入特征 (B, C, H, W)。

        返回:
            ``x + out``，形状不变。
        """
        return x + self.project_out(F.gelu(self.spatial(self.project_in(rms_normalize(x)))))


class BenefitCalibratedDictionary(nn.Module):
    """Dictionary retrieval that proposes a correction and predicts its adoption.

    收益校准字典：检索干净参考 → 生成候选修正 → 预测逐像素采纳率，
    同时输出可靠度估计。详见模块文档字符串。

    主要子模块：
    - ``dictionary``：(atoms, C) 的可学习原子矩阵；
    - ``query_context``/``query``/``key``/``value``：检索用的查询增强与
      键值投影；
    - ``candidate_*``：候选修正生成网络；
    - ``adoption``：采纳率预测头（输入 3C+2 通道证据）；
    - ``log_error``：对数误差预测头（输入 2C+2 通道证据）；
    - ``axial`` 分支（可选）：行/列分解卷积，仅红外字典启用。
    """

    def __init__(self, channels: int, atoms: int = 64, query_channels: int | None = None,
                 axial: bool = False, clean_anchor_only: bool = True) -> None:
        """初始化字典模块。

        参数:
            channels: 特征通道数 C。
            atoms: 字典原子数 K。
            query_channels: 查询/键投影的目标通道数；None 时取
                ``max(16, C//4)``。
            axial: 是否启用行/列分解卷积分支。红外图常呈水平/垂直
                条带退化，行、列方向分解的卷积更适合捕获这类结构，
                因此红外恢复字典置 True。
            clean_anchor_only: True（默认）时，forward 内的噪声检索
                对字典 detach，字典梯度只走干净锚点路径（见模块说明）。
        """
        super().__init__()
        query_channels = query_channels or max(16, channels // 4)
        self.atoms = atoms
        self.clean_anchor_only = clean_anchor_only
        # 字典原子 (K, C)，小方差随机初始化；训练中经干净锚点损失更新
        self.dictionary = nn.Parameter(torch.randn(atoms, channels) * 0.02)
        # 查询上下文：3x3 深度卷积，让查询感知局部邻域而非单像素
        self.query_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.query = nn.Conv2d(channels, query_channels, 1)
        self.key = nn.Linear(channels, query_channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        # 检索温度的原值：temperature = 1 + 29·sigmoid(raw)，
        # 初始 sigmoid(2.2)≈0.90 ⇒ 温度 ≈ 27，分配较尖锐（接近硬选择）
        self.temperature_raw = nn.Parameter(torch.tensor(2.2))

        # —— 候选修正网络：3C(拼接证据) → 隐藏 → C，tanh 限幅输出 ——
        hidden = max(16, channels // 2)
        self.candidate_in = nn.Conv2d(3 * channels, hidden, 1)
        self.candidate_spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.candidate_out = nn.Conv2d(hidden, channels, 1)
        # 近恒等初始化：初始时候选修正几乎为 0（tanh(≈0)≈0），
        # recovered ≈ z，即"先不改，学会了再改"
        nn.init.normal_(self.candidate_out.weight, std=1e-3)
        nn.init.zeros_(self.candidate_out.bias)

        self.axial = axial
        if axial:
            # 行方向 (1x9) 与列方向 (9x1) 分解卷积 + 1x1 混合，
            # 以远小于全 9x9 卷积的参数量捕获长条带结构
            self.row = nn.Conv2d(hidden, hidden, (1, 9), padding=(0, 4), groups=hidden)
            self.column = nn.Conv2d(hidden, hidden, (9, 1), padding=(4, 0), groups=hidden)
            self.axis_mix = nn.Conv2d(2 * hidden, hidden, 1)

        # —— 采纳率预测头：证据 = (z, reference, candidate, 熵, top2 边际) ——
        # 3C + 2 个通道（熵与边际各占 1 通道）
        evidence_channels = 3 * channels + 2
        self.adoption = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 近恒等初始化 + 偏置 -2 ⇒ 初始采纳率 ≈ sigmoid(-2) ≈ 0.12，
        # 初始阶段只轻微采纳候选修正
        nn.init.normal_(self.adoption[-1].weight, std=1e-3)
        nn.init.constant_(self.adoption[-1].bias, -2.0)

        # —— 对数误差预测头：证据 = (z, recovered, 熵, top2 边际) ——
        # 2C + 2 通道；softplus 保证输出 > 0
        self.log_error = nn.Sequential(
            nn.Conv2d(2 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 偏置 -2 ⇒ 初始预测 log 误差 ≈ softplus(-2) ≈ 0.13，
        # 初始可靠度 ≈ exp(-0.13) ≈ 0.88（先假设自己大体可靠）
        nn.init.normal_(self.log_error[-1].weight, std=1e-3)
        nn.init.constant_(self.log_error[-1].bias, -2.0)

    @property
    def temperature(self) -> Tensor:
        """检索温度，标量张量，范围 (1, 30)。

        ``temperature = 1 + 29 * sigmoid(temperature_raw)``：用 sigmoid
        重参数化保证温度恒正且可学习。温度越大，字典软分配越接近
        硬选择（one-hot）；越小越接近均匀平均。
        """
        return 1.0 + 29.0 * torch.sigmoid(self.temperature_raw.float())

    def _dictionary_projection(self, detach_dictionary: bool) -> tuple[Tensor, Tensor]:
        """把字典原子投影为键与值。

        参数:
            detach_dictionary: True 时把字典参数与 key/value 投影权重
                全部 detach——用于噪声路径的检索，保证该路径对字典
                **无梯度**；False 时保留梯度（干净锚点路径专用）。

        返回:
            二元组 ``(keys, values)``：形状分别为 (K, query_channels)、
            (K, channels)，均为 float32。
        """
        dictionary = self.dictionary.detach() if detach_dictionary else self.dictionary
        if detach_dictionary:
            # detach 模式下投影权重也一并 detach，彻底切断梯度
            keys = F.linear(dictionary.float(), self.key.weight.detach().float())
            values = F.linear(dictionary.float(), self.value.weight.detach().float())
        else:
            keys = self.key(dictionary.float())
            values = self.value(dictionary.float())
        return keys, values

    def retrieve(self, feature: Tensor, *, detach_dictionary: bool = False) -> tuple[Tensor, Tensor]:
        """在字典中软检索，得到干净参考特征。

        流程：
        1. RMS 归一化 → 查询 = query(z + query_context(z))，
           再沿通道 L2 归一化（与键的可比性）；
        2. 字典原子投影为键/值，键做 L2 归一化；
        3. 逐像素查询-键余弦相似度 × 温度 → softmax 得软分配
           ``assignment (B, K, H, W)``；
        4. ``reference = Σ_k assignment_k · value_k`` (B, C, H, W)。

        参数:
            feature: 输入特征 (B, C, H, W)（通常已 RMS 归一化）。
            detach_dictionary: 见 :meth:`_dictionary_projection`。

        返回:
            二元组 ``(reference, assignment)``，dtype 与输入一致。
        """
        normalized = rms_normalize(feature)
        # 查询 = 逐点投影 + 局部上下文增强，二者相加后投影到低维
        query = self.query(normalized + self.query_context(normalized)).float()
        query = F.normalize(query, dim=1)
        keys, values = self._dictionary_projection(detach_dictionary)
        keys = F.normalize(keys, dim=1)
        # einsum：逐像素与每个原子做点积 → (B, K, H, W) logits
        logits = torch.einsum("bchw,kc->bkhw", query, keys) * self.temperature
        assignment = logits.softmax(dim=1)
        # 软分配对原子值加权求和 → (B, C, H, W) 参考特征
        reference = torch.einsum("bkhw,kc->bchw", assignment, values)
        return reference.to(feature.dtype), assignment

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        """完整的恢复前向。

        参数:
            feature: 输入特征 (B, C, H, W)。

        返回:
            字典，包含（全部为 (B,C,H,W) 或 (B,1,H,W)/(B,K,H,W)）：
            - ``input``: RMS 归一化后的输入 z；
            - ``reference``: 字典检索得到的干净参考；
            - ``assignment``: 字典软分配 (B, K, H, W)；
            - ``candidate``: 候选修正量（tanh 限幅）；
            - ``adoption``: 逐像素采纳率 (B,1,H,W)，值域 (0,1)；
            - ``recovered``: ``z + adoption * candidate``，恢复结果；
            - ``predicted_log_error``: 预测的 log 误差（softplus，>0）；
            - ``reliability``: ``exp(-predicted_error)`` 截断到 [0,1]，
              下游交互模块的置信信号。
        """
        z = rms_normalize(feature)
        # 噪声路径检索：字典默认被 detach（clean_anchor_only=True），
        # 梯度不经过字典参数
        reference, assignment = self.retrieve(z, detach_dictionary=self.clean_anchor_only)
        # 候选修正的证据：z、参考、二者之差（"缺什么补什么"）
        candidate_input = torch.cat((z, reference, z - reference), 1)
        hidden = F.gelu(self.candidate_spatial(self.candidate_in(candidate_input)))
        if self.axial:
            # 红外专用：行/列分解卷积捕获条带状退化结构
            hidden = hidden + self.axis_mix(torch.cat((self.row(hidden), self.column(hidden)), 1))
        candidate = torch.tanh(self.candidate_out(hidden).float()).to(z.dtype)

        # —— 检索质量的两个标量证据（逐像素，广播为 1 通道）——
        # 熵：分配越尖锐（越接近硬选择）熵越低。除以 ln(K) 归一化到 [0,1]
        entropy = -(assignment.float().clamp_min(1e-8) * assignment.float().clamp_min(1e-8).log()).sum(1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(self.atoms), device=entropy.device))
        # top-2 边际：最大与次大分配概率之差，越大表示选择越果断
        top2 = assignment.float().topk(min(2, self.atoms), dim=1).values
        margin = top2[:, :1] - top2[:, 1:2] if self.atoms > 1 else top2[:, :1]
        # 采纳率：由 (z, reference, candidate, 熵, 边际) 证据预测
        evidence = torch.cat((z, reference, candidate, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        adoption = torch.sigmoid(self.adoption(evidence).float()).to(z.dtype)
        # 核心：恢复 = 原特征 + 采纳率 × 候选修正
        recovered = z + adoption * candidate

        # 误差预测：证据为 (z, recovered, 熵, 边际)
        error_evidence = torch.cat((z, recovered, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        # softplus 保证 log 误差为正；clamp(max=10) 防 expm1 数值溢出
        predicted_log_error = F.softplus(self.log_error(error_evidence).float())
        predicted_error = torch.expm1(predicted_log_error.clamp(max=10.0))
        # 可靠度 = exp(-误差)，误差越大越不可靠；截断到 [0,1]
        reliability = torch.exp(-predicted_error).clamp(0.0, 1.0).to(z.dtype)
        return {
            "input": z, "reference": reference, "assignment": assignment,
            "candidate": candidate, "adoption": adoption, "recovered": recovered,
            "predicted_log_error": predicted_log_error,
            "reliability": reliability,
        }
