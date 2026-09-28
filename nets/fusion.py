"""CalibFuse 融合网络主体。

本文件实现退化鲁棒可见光/红外图像融合的三尺度 PyTorch 模型
:class:`CalibFuse`。

整体数据流（多尺度对称 U 形结构）::

    可见光 RGB (B,3,H,W) ─┐
                          ├─> 各自 stem 卷积嵌入 ─> 尺度1(32ch, 全分辨率)
    红外灰度 (B,1,H,W) ──┘         │
                                   │  编码 TransformerBlock + RMS 归一化
                                   ▼
                        CalibratedFusionBlock
                        （字典恢复 + 双向跨模态消息 + 融合投影）
                                   │
                    ┌──────────────┴──────────────┐
              fused_down 下采样              vis/ir 各自下采样
              （作为下一尺度 prior_fused）        │
                    ▼                             ▼
              尺度2(64ch, 1/2 分辨率) …… 同上 …… 尺度3(128ch, 1/4 分辨率)
                                   │
                        CompactDecoder 自底向上解码
                        （深层 refine + 上采样 + 跳跃拼接）
                                   ▼
                fusion_head（TransformerBlock + 3x3 卷积 + sigmoid）
                                   ▼
                      融合 RGB 图像 (B,3,H,W)，取值 [0,1]

关键设计：
- 每个尺度上，``BenefitCalibratedDictionary`` 分别对两个模态做"恢复"，
  并输出 ``reliability``（可靠度）作为跨模态交互的置信信号；
- ``ReliabilityWeightedCrossAttention`` 计算双向消息（红外→可见、可见→红外），
  消息强度受“预算 × 源可靠度 × 收益预测”三门控调制；
- ``cross_budget_scale`` 在第 5~20 个 epoch 之间从 0 线性升到 1（训练课程），
  保证训练早期不做任何跨模态迁移，先学好各模态自身的恢复；
- 输入被填充到 4 的倍数（见 :func:`pad_to_multiple`），输出再裁回原始尺寸，
  因此任意奇偶尺寸均可推理。

"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dictionary import BenefitCalibratedDictionary, LocalResidual, rms_normalize
from .restormer import Downsample, OverlapPatchEmbed, TransformerBlock


def transformer_block(channels: int, heads: int) -> TransformerBlock:
    """按本项目的统一超参构造一个 Restormer 风格 Transformer 块。

    统一配置为：FFN 扩张倍数 2.0、不使用卷积偏置（bias=False）、
    LayerNorm 采用带偏置的 ``WithBias`` 变体。整个网络中所有
    TransformerBlock 都经由本工厂函数创建，保证超参一致。

    参数:
        channels: 该块的特征通道数（即 Restormer 中的 dim）。
        heads: 多头转置自注意力的头数，必须能整除 channels。

    返回:
        构造好的 :class:`~nets.restormer.TransformerBlock`。
    """
    return TransformerBlock(channels, heads, 2.0, False, "WithBias")


def upsample(x: Tensor, size: tuple[int, int]) -> Tensor:
    """双线性上采样到指定空间尺寸。

    使用 ``align_corners=False``（半像素对齐），与 PyTorch 默认推荐一致，
    避免上采样时出现网格伪影。用于解码器逐级放大以及
    ``prior_fused`` 与当前尺度分辨率不一致时的对齐。

    参数:
        x: 输入特征图 (B, C, H, W)。
        size: 目标 (高, 宽)。

    返回:
        上采样后的特征图 (B, C, size[0], size[1])。
    """
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def pad_to_multiple(x: Tensor, multiple: int = 4) -> tuple[Tensor, tuple[int, int]]:
    """把空间尺寸填充到 multiple 的整数倍，并记录原始尺寸以便裁剪还原。

    网络总共进行两次 2× 下采样（三个尺度），因此要求输入边长能被 4 整除。
    填充优先使用 reflect 模式（边缘镜像，不引入常量偏置）；
    只有当图太小（填充量超过原图尺寸）时才退化为 replicate 模式，
    因为 reflect 填充要求填充量小于对应边长。

    参数:
        x: 输入张量 (B, C, H, W)。
        multiple: 目标倍数，默认 4（对应两次 2× 下采样）。

    返回:
        二元组 ``(填充后的张量, 原始 (H, W))``。填充只发生在右、下两侧。
    """
    height, width = x.shape[-2:]
    # (-h) % m 给出"补到下一个倍数"所需的像素数；h 本身是倍数时为 0
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        # reflect 填充要求填充量小于边长，否则只能用 replicate
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (height, width)


class ReliabilityWeightedCrossAttention(nn.Module):
    """收益校准的可靠度加权跨模态注意力。

    计算从一个模态（source，消息的发出方）到另一个模态（target，消息的
    接收方）的传输消息。核心思想：**只有当源模态可靠、且证据表明传输
    有收益时，才允许跨模态信息流过**。

    消息的最终形式为::

        message = gate * direction
        gate    = maximum_transfer * budget_scale * source_reliability * benefit
        direction = 校准 RMS 后的注意力聚合值

    其中：
    - ``maximum_transfer``（0.15）：单次消息幅度的硬上限，防止跨模态
      信息淹没接收方自身的特征；
    - ``budget_scale``：由 :meth:`CalibFuse.set_training_progress` 设置的
      全局训练课程系数（epoch 5→20 从 0 线性升至 1），训练初期完全关闭
      跨模态传输；
    - ``source_reliability``：源模态字典恢复头预测的可靠度（detach，
      不让门控梯度回流去污染可靠度估计）；
    - ``benefit``：由一个小卷积网络根据双向证据预测的逐像素收益值。

    注意力本身是"可靠度加权的相关性"而非标准 QK^T：查询与键先做 RMS
    归一化，再计算加权余弦相似度，使注意力分数对特征的能量尺度不敏感。
    """

    def __init__(self, channels: int, heads: int, maximum_transfer: float = 0.15) -> None:
        """初始化跨模态注意力模块。

        参数:
            channels: 输入特征通道数 C。
            heads: 注意力头数，必须整除 channels。
            maximum_transfer: 消息幅度上限，默认 0.15。
        """
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.maximum_transfer = maximum_transfer
        # 四个 1x1 卷积分别产生查询、键、接收方数值、源数值；
        # 注意 q/k 作用于 RMS 归一化后的特征，value 分支分开投影
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.target_value = nn.Conv2d(channels, channels, 1)
        self.source_value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        # 每头一个可学习的注意力温度（缩放系数），初始化为 1
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        # 源数值的混合比例原值：sigmoid(-2)≈0.12，再乘 0.25，
        # 即初始时源数值只占聚合值约 3%，让消息以接收方数值为主
        self.source_scale_raw = nn.Parameter(torch.tensor(-2.0))
        # 收益预测头：输入拼接的证据（3C+2 通道），输出单通道收益 logits
        hidden = max(12, channels // 2)
        self.benefit = nn.Sequential(
            nn.Conv2d(3 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 近恒等初始化：权重极小 + 偏置 -2 ⇒ 初始 sigmoid(-2)≈0.12，
        # 训练开始时跨模态收益门接近关闭，模型先从“不迁移”学起
        nn.init.normal_(self.benefit[-1].weight, std=1e-3)
        nn.init.constant_(self.benefit[-1].bias, -2.0)

    def message(self, target: Tensor, source: Tensor, source_reliability: Tensor,
                target_reliability: Tensor, budget_scale: float) -> dict[str, Tensor]:
        """计算 source → target 方向的一条跨模态消息。

        参数:
            target: 接收方特征 (B, C, H, W)。
            source: 发出方特征 (B, C, H, W)。
            source_reliability: 源模态可靠度 (B, 1, H, W)，取值 [0,1]。
            target_reliability: 接收方可靠度 (B, 1, H, W)，仅作为收益
                预测的证据输入，不参与门控公式。
            budget_scale: 全局训练课程系数（0~1）。

        返回:
            字典，包含：
            - ``direction``：幅度已对齐接收方 RMS 的消息方向 (B,C,H,W)；
            - ``gate``：逐像素门控 (B,1,H,W)，值域 [0, maximum_transfer×…]；
            - ``message``：``gate * direction``，即最终叠加到接收方的增量；
            - ``attention``：(B, heads, HW, HW) 注意力矩阵，供诊断。
        """
        # 先对双方特征做 RMS 归一化，消除能量尺度差异（可见光/红外
        # 特征幅值分布差别很大，直接点积会让注意力偏向高能量模态）
        target_n, source_n = rms_normalize(target), rms_normalize(source)
        batch, _, height, width = target.shape
        # 统一重排为 (B, heads, head_dim, HW)，便于沿空间维做加权相关
        shape = (batch, self.heads, self.head_dim, height * width)
        query = self.q(target_n).float().reshape(shape)
        key = self.k(source_n).float().reshape(shape)
        # 源可靠度作为逐空间位置的权重，广播到 (B,1,1,HW)
        weight = source_reliability.float().reshape(batch, 1, 1, height * width)

        # —— 可靠度加权余弦相关性（替代标准 QK^T）——
        # 分子：Σ w · q·k ；分母：sqrt(Σ w·q² · Σ w·k²)
        # 等价于按 w 加权的余弦相似度，值域 [-1,1]，天然对幅度不敏感
        numerator = torch.matmul(query * weight, key.transpose(-2, -1))
        query_norm = (query.square() * weight).sum(-1, keepdim=True)
        key_norm = (key.square() * weight).sum(-1, keepdim=True)
        denominator = (query_norm * key_norm.transpose(-2, -1)).add(1e-6).sqrt()
        correlation = numerator / denominator
        # 可学习温度（逐头）做锐化/平滑后按源维 softmax
        attention = (correlation * self.temperature.float().clamp(0.05, 10.0)).softmax(-1)

        # —— 聚合接收方与源数值 ——
        target_value = self.target_value(target_n).float().reshape(shape)
        # 源数值先乘可靠度权重：不可靠的位置贡献被压低
        source_value = self.source_value(source_n).float().reshape(shape) * weight
        source_scale = 0.25 * torch.sigmoid(self.source_scale_raw.float())
        raw = torch.matmul(attention, target_value) + source_scale * torch.matmul(attention, source_value)
        raw = self.out(raw.reshape(batch, self.channels, height, width).to(target.dtype))

        # —— 幅度校准：把消息 RMS 拉到与接收方特征 RMS 一致 ——
        # 这样 direction 表示“往哪个方向改”，幅度由后面的 gate 决定
        raw_rms = raw.float().square().mean(1, keepdim=True).add(1e-6).sqrt()
        receiver_rms = target.float().square().mean(1, keepdim=True).add(1e-6).sqrt().detach()
        direction = (raw.float() / raw_rms * receiver_rms).to(target.dtype)
        # —— 收益预测的证据：双方归一化特征、差异幅度、双方可靠度 ——
        evidence = torch.cat((target_n, source_n, (target_n - source_n).abs(),
                              source_reliability, target_reliability), 1)
        benefit = torch.sigmoid(self.benefit(evidence).float()).to(target.dtype)
        # 门控 = 幅度上限 × 课程系数 × 源可靠度（detach） × 预测收益
        budget = self.maximum_transfer * float(budget_scale) * source_reliability.detach()
        gate = budget * benefit
        return {"direction": direction, "gate": gate, "message": gate * direction,
                "attention": attention}


class CalibratedFusionBlock(nn.Module):
    """单个尺度上的校准融合块。

    每个尺度执行三步：
    1. **恢复**：可见光与红外各走一个 :class:`BenefitCalibratedDictionary`，
       得到去退化后的特征 ``recovered`` 及其 ``reliability``；
    2. **交互**：``ReliabilityWeightedCrossAttention`` 计算双向消息
       （红外→可见、可见→红外），并叠加到各自恢复特征上；
    3. **融合**：把（增强后的可见光、增强后的红外、上一尺度传来的
       ``prior_fused`` 先验）三路拼接后投影、精炼，得到本尺度融合特征。

    ``prior_fused`` 是更深尺度看到更粗 fusion 上下文的通道：浅层融合
    结果下采样后作为深层融合的先验输入，形成跨尺度的级联。
    """

    def __init__(self, channels: int, heads: int, atoms: int, query_channels: int,
                 use_transformer_fusion: bool) -> None:
        """初始化融合块。

        参数:
            channels: 本尺度通道数。
            heads: 交互注意力头数。
            atoms: 字典原子数（传给两个恢复字典）。
            query_channels: 字典查询投影的通道数。
            use_transformer_fusion: True 时融合精炼用全局 TransformerBlock，
                False 时用轻量 LocalResidual（最高分辨率尺度用后者，
                省参数且全局注意力在浅层并非必要）。
        """
        super().__init__()
        # 可见光恢复字典（标准卷积）；红外恢复字典（axial=True，
        # 行/列分解卷积，更适合灰度红外图的条带结构）
        self.visible_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels)
        self.infrared_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels, axial=True)
        # 双向消息共享同一组交互权重（两次调用 message 方法）
        self.interaction = ReliabilityWeightedCrossAttention(channels, heads)
        fusion_input = 3 * channels
        # 3C → C 的 1x1 投影 + GELU
        self.fuse_project = nn.Sequential(nn.Conv2d(fusion_input, channels, 1), nn.GELU())
        self.fuse_refine = (transformer_block(channels, heads) if use_transformer_fusion
                            else LocalResidual(channels))

    def forward(self, visible: Tensor, infrared: Tensor, prior_fused: Tensor | None,
                budget_scale: float) -> tuple[Tensor, Tensor, Tensor, dict]:
        """执行本尺度的恢复-交互-融合流程。

        参数:
            visible: 可见光编码特征 (B, C, H, W)。
            infrared: 红外编码特征 (B, C, H, W)。
            prior_fused: 上一（更浅）尺度融合特征下采样后的先验；
                尺度不匹配时会被上采样对齐；最高分辨率尺度传 None。
            budget_scale: 跨模态传输预算的课程系数。

        返回:
            四元组 ``(v, i, fused, state)``：
            - ``v``/``i``：两个模态恢复后的特征（下采样前返回，
              供编码器继续向更深尺度传递）；
            - ``fused``：本尺度融合特征；
            - ``state``：包含两个模态字典状态和双向消息的字典，
              供训练损失（恢复/锚点/采纳率/交互校准）使用。
        """
        visible_state = self.visible_recovery(visible)
        infrared_state = self.infrared_recovery(infrared)
        v, i = visible_state["recovered"], infrared_state["recovered"]
        # 双向消息：红外→可见 与 可见→红外（同一个 interaction 模块）
        i_to_v = self.interaction.message(v, i, infrared_state["reliability"],
                                          visible_state["reliability"], budget_scale)
        v_to_i = self.interaction.message(i, v, visible_state["reliability"],
                                          infrared_state["reliability"], budget_scale)
        # 把消息叠加到各自恢复特征上，得到增强后的两模态特征
        visible_out = v + i_to_v["message"]
        infrared_out = i + v_to_i["message"]
        if prior_fused is None:
            # 最高分辨率尺度：无先验，用零张量占位（保持拼接通道数一致）
            prior_fused = torch.zeros_like(v)
        elif prior_fused.shape[-2:] != v.shape[-2:]:
            # 尺度不一致时上采样对齐（正常路径下 fused_down 已保证 1/2 尺寸）
            prior_fused = upsample(prior_fused, v.shape[-2:])
        fused = self.fuse_refine(self.fuse_project(torch.cat((visible_out, infrared_out, prior_fused), 1)))
        state = {
            "visible": visible_state, "infrared": infrared_state,
            "i_to_v": i_to_v, "v_to_i": v_to_i,
        }
        return v, i, fused, state


class CompactDecoder(nn.Module):
    """自底向上的紧凑解码器。

    从最深层（1/4 分辨率、128 通道）出发：
    - 先用一个 TransformerBlock 精炼深层融合特征；
    - 逐级上采样到上一尺度分辨率，经 3x3 卷积投影到该尺度通道数；
    - 与该尺度的融合特征做**残差跳跃连接**（不是直接相加，
      而是拼接后 1x1 卷积 + Transformer 精炼，再以残差形式加回）。

    这种“拼接-精炼-残差”结构既保留了浅层细节，又允许网络学习
    如何把深层语义与浅层纹理组合。
    """

    def __init__(self, channels: tuple[int, int, int], heads: tuple[int, int, int]) -> None:
        """初始化解码器。

        参数:
            channels: 三个尺度的通道数，如 (32, 64, 128)。
            heads: 三个尺度的注意力头数，如 (4, 4, 8)。
        """
        super().__init__()
        # 最深层（index=2）的精炼块
        self.deep = transformer_block(channels[2], heads[2])
        # 两级上采样投影：channels[i+1] → channels[i]
        self.up_projects = nn.ModuleList([
            nn.Conv2d(channels[index + 1], channels[index], 3, padding=1) for index in range(2)
        ])
        # 两级跳跃精炼：拼接(浅层特征, 上采样特征) 2C → 1x1 投影 C → Transformer
        self.refines = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * channels[index], channels[index], 1), nn.GELU(),
                          transformer_block(channels[index], heads[index])) for index in range(2)
        ])

    def forward(self, features: list[Tensor]) -> Tensor:
        """解码三尺度融合特征。

        参数:
            features: 三个尺度的融合特征 [尺度1(全分辨率), 尺度2(1/2), 尺度3(1/4)]。

        返回:
            全分辨率的解码特征 (B, channels[0], H, W)。
        """
        # 从最深层开始精炼
        decoded = self.deep(features[2])
        # 依次处理尺度 2（1/2 分辨率）和尺度 1（全分辨率）
        for index in (1, 0):
            # 上采样对齐到当前尺度分辨率并投影通道
            up = self.up_projects[index](upsample(decoded, features[index].shape[-2:]))
            # 跳跃连接：拼接浅层特征与上采样特征，精炼后以残差加回
            decoded = features[index] + self.refines[index](torch.cat((features[index], up), 1))
        return decoded


class CalibFuse(nn.Module):
    """CalibFuse 主模型：收益校准恢复 + 可靠度加权交互。

    三尺度双分支编码 + 收益校准恢复 + 可靠度加权交互。

    配置由四个三元组决定（对应三个尺度）：
    - ``channels``：各尺度通道数，默认 (32, 64, 128)；
    - ``heads``：各尺度注意力头数，默认 (4, 4, 8)；
    - ``atoms``：各尺度字典原子数，默认 (64, 64, 64)；
    - ``query_channels``：各尺度字典查询通道数，默认 (16, 24, 32)。

    默认配置共 1,625,131 参数、12 个 Transformer 块。
    """

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128),
                 heads: tuple[int, int, int] = (4, 4, 8),
                 atoms: tuple[int, int, int] = (64, 64, 64),
                 query_channels: tuple[int, int, int] = (16, 24, 32)) -> None:
        """按配置构建整个网络。

        参数:
            channels: 三个尺度的通道数。
            heads: 三个尺度的注意力头数。
            atoms: 三个尺度每个恢复字典的原子数。
            query_channels: 三个尺度字典查询投影的通道数。

        异常:
            ValueError: 任一参数不是长度 3 的序列时抛出。
        """
        super().__init__()
        if not (len(channels) == len(heads) == len(atoms) == len(query_channels) == 3):
            raise ValueError("channels, heads, atoms, and query_channels must each contain three values")
        # 把构造配置存进 self.config：保存 checkpoint 时会写入 payload 的
        # model_config，加载方据此重建结构，而不依赖代码中的默认值
        self.config = {"channels": channels, "heads": heads, "atoms": atoms,
                       "query_channels": query_channels}
        # 跨模态传输预算的课程系数，训练时由 set_training_progress 更新；
        # 推理时保持 1.0（即训练完成后的满预算状态）
        self.cross_budget_scale = 1.0
        # 两个模态各自的 stem：可见光 3 通道输入，红外 1 通道输入，
        # 都用 3x3 重叠 patch 嵌入 + GELU 映射到 channels[0]
        self.visible_stem = nn.Sequential(OverlapPatchEmbed(3, channels[0]), nn.GELU())
        self.infrared_stem = nn.Sequential(OverlapPatchEmbed(1, channels[0]), nn.GELU())
        # 每个尺度、每个模态一个 Transformer 编码块（共 3×2=6 个）
        self.visible_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        self.infrared_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        # 每个尺度一个校准融合块；最高分辨率尺度（index=0）用 LocalResidual
        # 融合精炼，其余尺度用 TransformerBlock（共 3 个融合精炼块）
        self.blocks = nn.ModuleList([
            CalibratedFusionBlock(c, h, k, q, use_transformer_fusion=index > 0)
            for index, (c, h, k, q) in enumerate(zip(channels, heads, atoms, query_channels))
        ])
        # 两条模态分支的下采样（各 2 级）+ 融合特征下采样（供 prior_fused）
        self.visible_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.infrared_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.fused_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.decoder = CompactDecoder(channels, heads)
        # 输出头：TransformerBlock + 3x3 卷积到 3 通道；forward 中再过
        # sigmoid 压到 [0,1]
        self.fusion_head = nn.Sequential(transformer_block(channels[0], heads[0]),
                                         nn.Conv2d(channels[0], 3, 3, padding=1))

    def set_training_progress(self, epoch: int) -> None:
        """按当前 epoch 更新跨模态传输预算的课程系数。

        ``budget_scale = clamp((epoch - 5) / 15, 0, 1)``：
        epoch ≤ 5 时为 0（完全关闭跨模态迁移，让两个模态先各自学会
        恢复），epoch 5→20 线性升至 1（逐步放开跨模态交互），
        epoch ≥ 20 后保持 1。训练循环每个 epoch 调用一次。

        参数:
            epoch: 当前训练轮数（从 0 或 1 计数均可，公式连续）。
        """
        self.cross_budget_scale = min(max((epoch - 5) / 15.0, 0.0), 1.0)

    def _encode(self, visible: Tensor, infrared: Tensor, return_auxiliary: bool) -> dict:
        """多尺度编码主干：stem → 逐尺度（编码 → 融合 → 下采样）→ 解码。

        参数:
            visible: 已填充的可见光图 (B,3,H,W)。
            infrared: 已填充的红外灰度图 (B,1,H,W)。
            return_auxiliary: True 时收集每个尺度的中间状态（字典状态、
                双向消息），供训练损失使用；推理时传 False 省内存。

        返回:
            字典：``fused`` 为 sigmoid 后的融合图 (B,3,H,W)；
            ``states`` 为各尺度状态列表（return_auxiliary=False 时为空表）。
        """
        v, i = self.visible_stem(visible), self.infrared_stem(infrared)
        scales, states, prior = [], [], None
        for index, block in enumerate(self.blocks):
            # 编码块 + RMS 归一化：每个尺度进入融合块前都重新归一化，
            # 保持特征能量稳定，也与字典/交互模块的归一化假设一致
            v = rms_normalize(self.visible_encoder[index](v))
            i = rms_normalize(self.infrared_encoder[index](i))
            v, i, fused, state = block(v, i, prior, self.cross_budget_scale)
            scales.append(fused)
            if return_auxiliary:
                states.append(state)
            if index < 2:
                # 尚未到最深层：两模态与融合特征各自下采样，
                # fused 下采样结果作为下一尺度的 prior_fused
                v = self.visible_down[index](v)
                i = self.infrared_down[index](i)
                prior = self.fused_down[index](fused)
        # 解码 + 输出头 + sigmoid：fused 图取值 [0,1]
        return {"fused": torch.sigmoid(self.fusion_head(self.decoder(scales))),
                "states": states}

    @torch.no_grad()
    def clean_reference_features(self, visible: Tensor, infrared: Tensor) -> dict:
        """用干净（无退化）输入提取各尺度的参考特征与字典分配。

        这是**干净教师路径**：训练时由 EMA 教师（``clean_teacher=ema.model``）
        调用本方法，为正则化损失提供干净特征目标（恢复/锚点损失）与
        干净字典分配（用于最优增益计算）。全程 ``no_grad``，教师的
        参数不会收到梯度，学生也不通过该路径回传。

        与 :meth:`_encode` 的区别：这里不做跨模态交互、不做解码，只走
        编码 + 字典恢复；且下采样用的是**恢复后**的特征
        （``v_state["recovered"]``），与噪声路径保持一致的结构假设。

        参数:
            visible: 干净可见光图 (B,3,H,W)。
            infrared: 干净红外灰度图 (B,1,H,W)。

        返回:
            字典，键为：
            - ``visible_features``/``infrared_features``：各尺度归一化编码
              特征列表（长度 3）；
            - ``visible_assignments``/``infrared_assignments``：各尺度字典
              软分配 (B, atoms, H, W) 列表。
        """
        v, i = self.visible_stem(visible), self.infrared_stem(infrared)
        result = {"visible_features": [], "infrared_features": [],
                  "visible_assignments": [], "infrared_assignments": []}
        for index, block in enumerate(self.blocks):
            v = rms_normalize(self.visible_encoder[index](v))
            i = rms_normalize(self.infrared_encoder[index](i))
            result["visible_features"].append(v)
            result["infrared_features"].append(i)
            # 只跑各自的字典恢复头（不需要交互与融合）
            v_state = block.visible_recovery(v)
            i_state = block.infrared_recovery(i)
            result["visible_assignments"].append(v_state["assignment"])
            result["infrared_assignments"].append(i_state["assignment"])
            if index < 2:
                # 注意：用恢复后的特征下采样，保证干净/噪声两条路径
                # 在更深尺度看到的是同构的特征
                v = self.visible_down[index](v_state["recovered"])
                i = self.infrared_down[index](i_state["recovered"])
        return result

    def forward(self, visible: Tensor, infrared: Tensor, *, clean_visible: Tensor | None = None,
                clean_infrared: Tensor | None = None, return_auxiliary: bool = False,
                clean_teacher: nn.Module | None = None) -> dict:
        """前向传播。

        推理用法（仅需两个输入）::

            output = model(vis, ir)           # -> {"fused": (B,3,H,W)}

        训练用法（需要干净参考与 EMA 教师）::

            output = model(vis, ir, clean_visible=vis_clean,
                           clean_infrared=ir_clean, return_auxiliary=True,
                           clean_teacher=ema.model)
            # 额外得到 states / teacher / clean_anchor_references

        参数:
            visible: 可见光图 (B,3,H,W)，取值任意（内部不裁剪）。
            infrared: 红外图 (B,C,H,W)；若 C>1 会被按通道均值转灰度。
            clean_visible: 干净可见光参考，return_auxiliary=True 时必填。
            clean_infrared: 干净红外参考，同上。
            return_auxiliary: 是否返回训练所需的中间状态。
            clean_teacher: 提供干净参考特征的教师模型（通常传 EMA 模型）；
                None 时用自身（等效于教师=学生，仅用于测试）。

        返回:
            字典，至少含 ``fused`` (B,3,H,W)；return_auxiliary=True 时
            额外含：
            - ``states``：各尺度融合块状态（字典状态 + 双向消息）；
            - ``teacher``：干净教师的参考特征与分配；
            - ``clean_anchor_references``：用干净特征、**不 detach 字典**
              重新检索得到的字典锚点参考——字典参数的梯度只经由这条
              干净锚点路径获得（clean_anchor_only 设计），噪声检索路径
              对字典完全 detach。
        """
        # 红外统一转单通道灰度
        if infrared.shape[1] != 1:
            infrared = infrared.mean(1, keepdim=True)
        # 填充到 4 的倍数后编码，最后裁回原始尺寸
        visible, original_size = pad_to_multiple(visible)
        infrared, _ = pad_to_multiple(infrared)
        output = self._encode(visible, infrared, return_auxiliary)
        output["fused"] = output["fused"][..., :original_size[0], :original_size[1]]
        if not return_auxiliary:
            # 纯推理路径：丢掉空 states 列表直接返回
            output.pop("states")
            return output
        # —— 以下为训练专用：准备干净参考与字典锚点 ——
        if clean_visible is None or clean_infrared is None:
            raise ValueError("clean_visible and clean_infrared are required for auxiliary training output")
        if clean_infrared.shape[1] != 1:
            clean_infrared = clean_infrared.mean(1, keepdim=True)
        clean_visible, _ = pad_to_multiple(clean_visible)
        clean_infrared, _ = pad_to_multiple(clean_infrared)
        # 教师默认为自身；训练时传入 EMA 模型以获得更稳定的干净参考
        teacher = self if clean_teacher is None else clean_teacher
        with torch.no_grad():
            clean = teacher.clean_reference_features(clean_visible, clean_infrared)
        output["teacher"] = clean
        # 用干净特征重新检索字典参考，且不 detach 字典：
        # 这是字典参数接收梯度的唯一路径（干净锚点损失）
        output["clean_anchor_references"] = {"visible": [], "infrared": []}
        for index, block in enumerate(self.blocks):
            for modality, recovery in (("visible", block.visible_recovery),
                                       ("infrared", block.infrared_recovery)):
                reference, _ = recovery.retrieve(clean[f"{modality}_features"][index],
                                                 detach_dictionary=False)
                output["clean_anchor_references"][modality].append(reference)
        return output
