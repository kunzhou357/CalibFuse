"""受益校准的字典恢复（Benefit-Calibrated Dictionary Recovery）。

本模块将"退化去除"建模为三个步骤（ retrieve -> candidate -> adopt ）：

1. **检索（retrieve）**：将（可能已退化的）特征 z 与一组可学习的
   字典原子（atoms）做软分配（soft assignment），通过分配系数对原子
   值加权求和，得到一个"干净参考"（reference）。
2. **候选修正（candidate）**：以证据 ``(z, reference, z - reference)``
   为输入，网络提出一个有界（tanh 限制在 [-1, 1]）的修正向量。
3. **采纳（adopt）**：预测一个逐像素的采纳系数 adoption ∈ [0, 1]，
   并按 ``recovered = z + adoption * candidate`` 融合。
   网络只学习"在哪些位置修正、修正多少"（where & how much），
   而不直接学习"如何修正"（how）——修正方向来自字典检索，
   修正幅度受采纳门控约束，从结构上抑制过度修正。

模块同时预测一个对数误差 log-error，其负指数
``reliability = exp(-predicted_error)`` 作为置信度信号，供下游的
跨模态交互（ReliabilityWeightedCrossAttention）使用：源模态可靠性越低，
其向目标模态传递的信息越少。

梯度隔离（gradient isolation）：默认 ``clean_anchor_only=True`` 时，
"带噪检索"路径会 detach 字典参数及其 key/value 投影，因此字典的
梯度完全来自"干净锚点损失"（clean-anchor loss，见
``CalibFuse.forward`` 中的 ``clean_anchor_references`` 分支）。
这一设计保证字典学到的原子对应干净特征空间，而不会被训练期
的退化样本污染。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    """按样本（沿 C、H、W 三个维度）做 RMS 归一化。

    只去除特征的整体能量尺度，保留空间上的对比度结构。
    全网络统一使用该归一化，使可见光与红外特征的幅度对齐：
    - 可见光与红外的原始能量量纲差异很大，归一化后注意力与字典
      余弦相似度才具备可比性；
    - 在融合块之前对各尺度编码特征再次归一化，保持特征能量稳定，
      与字典/注意力模块的假设一致。

    参数:
        x: 形状 (B, C, H, W) 的特征。
        eps: 数值稳定项，防止除零。

    返回:
        与输入同形状、同 dtype 的 RMS 归一化特征。
    """
    # 用 float32 计算统计量，避免半精度下平方求和溢出/下溢
    scale = x.float().square().mean((1, 2, 3), keepdim=True).add(eps).sqrt()
    return (x.float() / scale).to(x.dtype)


class LocalResidual(nn.Module):
    """轻量级局部残差块（1x1 投影 -> 5x5 深度卷积 -> 1x1 投影）。

    用于最高分辨率尺度（stride=1、全局注意力开销过大的场景），
    代替完整 TransformerBlock 做特征精修。
    采用"近恒等初始化"（project_out 的权重 std=1e-3、偏置为 0），
    训练开始时该块几乎等价于恒等映射，避免破坏预归一化特征。

    参数:
        channels: 输入/输出通道数。
        expansion: 隐藏层通道扩张倍数（默认 2）。
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """初始化残差块的三个卷积层。"""
        super().__init__()
        hidden = channels * expansion
        # 1x1 卷积：升维到 hidden
        self.project_in = nn.Conv2d(channels, hidden, 1)
        # 5x5 深度卷积：空间信息交互，分组数等于通道数（逐通道卷积）
        self.spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        # 1x1 卷积：降维回 channels
        self.project_out = nn.Conv2d(hidden, channels, 1)
        # 近恒等初始化：初始输出 residual ≈ 0，因此整体输出 ≈ x
        nn.init.normal_(self.project_out.weight, std=1e-3)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """返回 ``x + residual``；残差分支的输入先做 RMS 归一化。

        归一化只作用于残差分支，主干直连通路保持原始 x，
        从而保证残差学习的稳定性。
        """
        return x + self.project_out(F.gelu(self.spatial(self.project_in(rms_normalize(x)))))


class BenefitCalibratedDictionary(nn.Module):
    """字典检索 + 候选修正提议 + 采纳预测的复合模块。

    除输出恢复后的特征外，还输出可靠性估计（reliability），
    作为跨模态交互的置信度信号。三步设计与梯度隔离契约见模块级
    文档字符串。

    设计要点:
    - 字典原子是可学习参数 (K, C)，但只通过干净锚点损失更新；
    - 候选修正网络与采纳头都采用近恒等初始化，训练初期
      "几乎不修正"，由损失逐步放开修正幅度，训练稳定；
    - ``axial=True`` 时启用行/列分解卷积（供红外字典使用，
      用于捕获条带状退化）。
    """

    def __init__(self, channels: int, atoms: int = 64, query_channels: int | None = None,
                 axial: bool = False, clean_anchor_only: bool = True) -> None:
        """初始化字典模块。

        参数:
            channels: 输入特征通道数 C。
            atoms: 字典原子数量 K（软分配的类别数）。
            query_channels: 查询/键的投影维度 d；为 None 时取
                ``max(16, channels // 4)``。
            axial: 是否启用行(1x9)/列(9x1)分解卷积。红外字典设为
                True，因为条带退化沿行/列方向延伸。
            clean_anchor_only: 为 True（默认）时，带噪检索路径
                detach 字典与 key/value 投影，字典梯度只来自
                干净锚点损失路径。
        """
        super().__init__()
        query_channels = query_channels or max(16, channels // 4)
        self.atoms = atoms
        self.clean_anchor_only = clean_anchor_only
        # 字典原子 (K, C)；0.02 标准差的随机初始化。
        # 注意：该参数只经干净锚点损失获得梯度（见 retrieve 的 detach 逻辑）
        self.dictionary = nn.Parameter(torch.randn(atoms, channels) * 0.02)
        # 查询分支：深度卷积提供局部上下文，再与原特征相加后投影到查询空间
        self.query_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.query = nn.Conv2d(channels, query_channels, 1)
        # 键/值分支：对字典原子做线性投影（K 个原子 -> K 个键和 K 个值）
        self.key = nn.Linear(channels, query_channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        # 可学习温度 = 1 + 29 * sigmoid(raw)：初始 sigmoid(2.2)≈0.9，
        # 温度约 27，软分配接近 hard argmax（检索锐利），
        # 但保留 (1, 30) 的可调范围
        self.temperature_raw = nn.Parameter(torch.tensor(2.2))

        # 候选修正网络：证据为 3C 通道 (z, reference, z-reference)
        # -> 1x1 升维 -> 5x5 深度卷积 -> 1x1 降维，tanh 限幅到 [-1, 1]
        hidden = max(16, channels // 2)
        self.candidate_in = nn.Conv2d(3 * channels, hidden, 1)
        self.candidate_spatial = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.candidate_out = nn.Conv2d(hidden, channels, 1)
        # 近恒等初始化：初始候选修正几乎为零，训练初期 "不乱改"
        nn.init.normal_(self.candidate_out.weight, std=1e-3)
        nn.init.zeros_(self.candidate_out.bias)

        self.axial = axial
        if axial:
            # 行(1x9)/列(9x1)分解卷积：以远低于 9x9 的代价捕获长条带结构。
            # 两条支路拼接后经 1x1 卷积混合，以残差方式加回主特征
            self.row = nn.Conv2d(hidden, hidden, (1, 9), padding=(0, 4), groups=hidden)
            self.column = nn.Conv2d(hidden, hidden, (9, 1), padding=(4, 0), groups=hidden)
            self.axis_mix = nn.Conv2d(2 * hidden, hidden, 1)

        # 采纳头：证据 = (z, reference, candidate, 分配熵, top-2 概率差)
        # 共 3C + 2 通道 -> hidden -> 1
        evidence_channels = 3 * channels + 2
        self.adoption = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 偏置 -2 => 初始 adoption ≈ sigmoid(-2) ≈ 0.12：
        # 初期只采纳约 12% 的候选修正，避免训练早期的大幅扰动
        nn.init.normal_(self.adoption[-1].weight, std=1e-3)
        nn.init.constant_(self.adoption[-1].bias, -2.0)

        # 对数误差头：证据 = (z, recovered, 分配熵, top-2 概率差)
        # 共 2C + 2 通道 -> hidden -> 1
        self.log_error = nn.Sequential(
            nn.Conv2d(2 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 偏置 -2 => softplus(-2)≈0.13，初始可靠性 ≈ exp(-0.13) ≈ 0.88：
        # 初期各位置默认"较可靠"，让跨模态消息可以流动，
        # 再由误差校准损失逐步学到真实的可靠性分布
        nn.init.normal_(self.log_error[-1].weight, std=1e-3)
        nn.init.constant_(self.log_error[-1].bias, -2.0)

    @property
    def temperature(self) -> Tensor:
        """可学习的软分配温度，取值范围 (1, 30)。

        温度越高，softmax 分配越尖锐（越接近硬分配）。
        """
        return 1.0 + 29.0 * torch.sigmoid(self.temperature_raw.float())

    def _dictionary_projection(self, detach_dictionary: bool) -> tuple[Tensor, Tensor]:
        """将字典原子投影为 (keys, values)，可选拟影（detach）。

        参数:
            detach_dictionary: 为 True 时，字典参数与 key/value 投影
                权重都走 detach 版本，使该路径完全没有字典梯度。

        返回:
            (keys (K, d), values (K, C))，均为 float32。
        """
        dictionary = self.dictionary.detach() if detach_dictionary else self.dictionary
        if detach_dictionary:
            # 同时 detach 投影权重，保证该路径不产生任何字典相关梯度
            keys = F.linear(dictionary.float(), self.key.weight.detach().float())
            values = F.linear(dictionary.float(), self.value.weight.detach().float())
        else:
            keys = self.key(dictionary.float())
            values = self.value(dictionary.float())
        return keys, values

    def retrieve(self, feature: Tensor, *, detach_dictionary: bool = False) -> tuple[Tensor, Tensor]:
        """软检索一个"干净参考"特征；返回 ``(reference, assignment)``。

        流程:
        1. 对特征做 RMS 归一化，并经"特征 + 局部上下文"后投影到查询
           空间 d，再做 L2 归一化（余弦相似度的分子）；
        2. 字典原子投影为键并 L2 归一化；
        3. 逐像素查询-原子余弦相似度乘以可学习温度，得到
           ``assignment (B, K, H, W)``（K 维 softmax）；
        4. 参考特征 = 分配系数对原子值的加权和。

        参数:
            feature: (B, C, H, W) 特征。
            detach_dictionary: 见 :meth:`_dictionary_projection`。

        返回:
            reference: (B, C, H, W) 干净参考特征（dtype 与输入一致）。
            assignment: (B, K, H, W) 软分配概率。
        """
        normalized = rms_normalize(feature)
        # 查询 = 投影(归一化特征 + 深度卷积上下文)；再 L2 归一化
        query = self.query(normalized + self.query_context(normalized)).float()
        query = F.normalize(query, dim=1)
        keys, values = self._dictionary_projection(detach_dictionary)
        keys = F.normalize(keys, dim=1)
        # logits[b,k,h,w] = <query[b,:,h,w], key[k]> * temperature
        logits = torch.einsum("bchw,kc->bkhw", query, keys) * self.temperature
        assignment = logits.softmax(dim=1)
        # 参考特征 = 分配系数对原子值的加权求和
        reference = torch.einsum("bkhw,kc->bchw", assignment, values)
        return reference.to(feature.dtype), assignment

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        """执行完整的三步恢复流程。

        参数:
            feature: (B, C, H, W) 输入特征（可能是退化特征）。

        返回:
            包含以下键的字典:
            - ``input``: RMS 归一化后的输入 z；
            - ``reference``: 字典检索得到的干净参考；
            - ``assignment``: (B, K, H, W) 软分配；
            - ``candidate``: tanh 有界的候选修正；
            - ``adoption``: (B, 1, H, W) ∈ [0,1] 采纳系数；
            - ``recovered``: z + adoption * candidate；
            - ``predicted_log_error``: 预测的对数恢复误差（softplus 有界）；
            - ``reliability``: exp(-predicted_error) ∈ [0,1]。
        """
        z = rms_normalize(feature)
        # 带噪路径检索：默认 detach 字典（clean_anchor_only=True），
        # 使字典梯度只来自干净锚点损失
        reference, assignment = self.retrieve(z, detach_dictionary=self.clean_anchor_only)
        # 候选修正分支：证据为 (z, reference, z - reference)
        candidate_input = torch.cat((z, reference, z - reference), 1)
        hidden = F.gelu(self.candidate_spatial(self.candidate_in(candidate_input)))
        if self.axial:
            # 轴向分解卷积（红外专用）：行/列长程条带上下文以残差方式注入
            hidden = hidden + self.axis_mix(torch.cat((self.row(hidden), self.column(hidden)), 1))
        # tanh 将候选修正限制在 [-1, 1]，提供数值有界性
        candidate = torch.tanh(self.candidate_out(hidden).float()).to(z.dtype)

        # 两个逐像素的"检索质量"信号：
        # 1) 归一化分配熵：熵大 => 分配不确定 => 检索不可靠；
        # 2) top-2 概率差（margin）：差距小 => 两个原子难分 => 不可靠。
        # 熵按 log(K) 归一化到 [0, 1]
        entropy = -(assignment.float().clamp_min(1e-8) * assignment.float().clamp_min(1e-8).log()).sum(1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(self.atoms), device=entropy.device))
        top2 = assignment.float().topk(min(2, self.atoms), dim=1).values
        margin = top2[:, :1] - top2[:, 1:2] if self.atoms > 1 else top2[:, :1]
        # 采纳头的证据：状态 + 修正 + 检索质量信号
        evidence = torch.cat((z, reference, candidate, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        adoption = torch.sigmoid(self.adoption(evidence).float()).to(z.dtype)
        # 核心更新式：recovered = input + adoption * candidate
        # 采纳系数决定"在何处、以多大程度"采纳修正
        recovered = z + adoption * candidate

        # 误差校准分支：以 (z, recovered, 熵, margin) 预测恢复误差的对数
        error_evidence = torch.cat((z, recovered, entropy.to(z.dtype), margin.to(z.dtype)), 1)
        # softplus 保证 log_error 非负；clamp(max=10) 防止 expm1 溢出
        predicted_log_error = F.softplus(self.log_error(error_evidence).float())
        predicted_error = torch.expm1(predicted_log_error.clamp(max=10.0))
        # 可靠性 = exp(-误差)，截断到 [0, 1]：
        # 误差为 0 => 可靠性 1；误差增大 => 可靠性指数衰减
        reliability = torch.exp(-predicted_error).clamp(0.0, 1.0).to(z.dtype)
        return {
            "input": z, "reference": reference, "assignment": assignment,
            "candidate": candidate, "adoption": adoption, "recovered": recovered,
            "predicted_log_error": predicted_log_error,
            "reliability": reliability,
        }
