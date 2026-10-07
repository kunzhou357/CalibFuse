"""CalibFuse 融合网络（主模型）。

面向退化鲁棒的可见光/红外图像融合的三尺度对称 U 形网络：

- 可见光（3 通道）/红外（1 通道）各自的 stem 与逐尺度 Transformer 编码器；
- 每个尺度一个 :class:`CalibratedFusionBlock`：字典恢复 ->
  可靠性加权的双向跨模态消息传递 -> 融合；
- 上层融合特征经下采样后作为 ``prior_fused`` 先验级联到下一尺度；
- :class:`CompactDecoder` 自底向上解码，结合各尺度的跳跃连接精修；
- ``fusion_head``（TransformerBlock + 卷积 + sigmoid）输出 [0, 1] 的 RGB 图像。

跨模态预算（cross_budget_scale）在第 5~20 个 epoch 之间从 0 线性爬升到 1：
早期训练专注各模态自身的退化恢复，之后再逐步放开跨模态信息交互，
防止早期不可靠的消息污染特征（课程式训练策略）。

输入在内部填充（pad）到 4 的倍数、输出后裁回原尺寸，因此任意
分辨率图像均可处理（三尺度下采样两次 + 网络 4 倍对齐要求）。

默认配置：channels (32, 64, 128)，heads (4, 4, 8)，atoms (64, 64, 64)，
query_channels (16, 24, 32) —— 共 1,625,131 个参数、12 个 Transformer 块。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dictionary import BenefitCalibratedDictionary, LocalResidual, rms_normalize
from .restormer import Downsample, OverlapPatchEmbed, TransformerBlock


def transformer_block(channels: int, heads: int) -> TransformerBlock:
    """按项目统一设置创建 Restormer 风格的 TransformerBlock。

    网络中所有 TransformerBlock 都经由该工厂构建，保证超参数一致：
    FFN 扩张倍数 2.0、卷积无偏置、WithBias 版 LayerNorm
    （即带可学习 scale/shift 的通道维 LayerNorm）。

    参数:
        channels: 特征通道数。
        heads: MDTA 注意力的头数。
    """
    return TransformerBlock(channels, heads, 2.0, False, "WithBias")


def upsample(x: Tensor, size: tuple[int, int]) -> Tensor:
    """双线性插值上采样到目标空间尺寸 (H, W)，align_corners=False。"""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def pad_to_multiple(x: Tensor, multiple: int = 4) -> tuple[Tensor, tuple[int, int]]:
    """将空间维度填充到 ``multiple`` 的整数倍。

    网络含两次 2 倍下采样，特征需要 4 倍空间对齐；推理时任意
    分辨率输入先在此填充。

    参数:
        x: (B, C, H, W) 输入。
        multiple: 对齐倍数（默认 4）。

    返回:
        (填充后的张量, 原始 (H, W))。填充只作用于右侧与下侧。
    """
    height, width = x.shape[-2:]
    pad_h = (-height) % multiple  # 等价于 (multiple - H % multiple) % multiple
    pad_w = (-width) % multiple
    if pad_h or pad_w:
        # reflect 填充要求填充量小于对应边长，否则退化为 replicate 填充
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (height, width)


class ReliabilityWeightedCrossAttention(nn.Module):
    """可靠性加权的跨模态消息传递（跨模态交互模块）。

    从源模态到目标模态的消息为::

        message  = gate * direction
        gate     = maximum_transfer * budget_scale * source_reliability * benefit

    即：只有当"源模态可靠"且"预测的融合收益（benefit）高"时，
    信息才被允许流动。门控的四个因子各司其职：
    - ``maximum_transfer``（固定 0.15）：单次传递的幅度上限，
      从结构上限制跨模态注入的量级；
    - ``budget_scale``：训练课程系数（epoch 5~20 从 0 爬升到 1）；
    - ``source_reliability``：源模态字典恢复模块估计的可靠性（detach，
      不回传梯度）；
    - ``benefit``：由证据网络预测的"接收该消息对融合有益程度"。

    注意力本身不是普通的 QK^T 点积，而是可靠性加权的余弦相似度
    （查询/键均 RMS 归一化，对幅度不敏感），消除了两个模态间
    能量尺度差异的影响。
    """

    def __init__(self, channels: int, heads: int, maximum_transfer: float = 0.15) -> None:
        """初始化跨模态注意力。

        参数:
            channels: 特征通道数（须能被 heads 整除）。
            heads: 注意力头数。
            maximum_transfer: 跨模态传递幅度上限（与损失模块中的
                同名参数保持一致，用于交互门控的回归监督归一化）。
        """
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.maximum_transfer = maximum_transfer
        # 1x1 卷积生成查询/键；目标/源各自独立的值投影；
        # out 为输出混合卷积
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.target_value = nn.Conv2d(channels, channels, 1)
        self.source_value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        # 可学习注意力温度（逐头），推理时截断到 [0.05, 10]
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        # 源值缩放的原始参数：0.25 * sigmoid(-2) ≈ 0.027，
        # 初期源模态的值贡献很小
        self.source_scale_raw = nn.Parameter(torch.tensor(-2.0))
        # 收益（benefit）预测网络：证据 = 两模态特征 + 差值 + 两个可靠性图
        # (3C + 2 通道) -> hidden（深度卷积） -> 1
        hidden = max(12, channels // 2)
        self.benefit = nn.Sequential(
            nn.Conv2d(3 * channels + 2, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 近恒等初始化：初始 benefit = sigmoid(-2) ≈ 0.12，
        # 训练开始时跨模态传递几乎关闭（配合 budget_scale 课程）
        nn.init.normal_(self.benefit[-1].weight, std=1e-3)
        nn.init.constant_(self.benefit[-1].bias, -2.0)

    def message(self, target: Tensor, source: Tensor, source_reliability: Tensor,
                target_reliability: Tensor, budget_scale: float) -> dict[str, Tensor]:
        """计算一条 source -> target 的消息。

        参数:
            target: (B, C, H, W) 接收方（恢复后）特征。
            source: (B, C, H, W) 发送方（恢复后）特征。
            source_reliability: (B, 1, H, W) 源模态可靠性图。
            target_reliability: (B, 1, H, W) 目标模态可靠性图。
            budget_scale: 跨模态预算课程系数 ∈ [0, 1]。

        返回:
            字典，包含:
            - ``direction``: 消息方向（幅度已对齐到接收方）；
            - ``gate``: 门控图（各因子逐像素相乘）；
            - ``message``: gate * direction，加到目标特征上；
            - ``attention``: 注意力图（诊断用）。
        """
        # 双方先做 RMS 归一化：注意力只看方向（余弦），忽略两模态的能量尺度差
        target_n, source_n = rms_normalize(target), rms_normalize(source)
        batch, _, height, width = target.shape
        shape = (batch, self.heads, self.head_dim, height * width)
        # 注意：查询来自"目标"特征、键来自"源"特征——
        # 接收方决定"我需要什么"，发送方提供"我能给什么"
        query = self.q(target_n).float().reshape(shape)
        key = self.k(source_n).float().reshape(shape)
        # 逐位置可靠性权重（源侧），参与相似度与分母的计算
        weight = source_reliability.float().reshape(batch, 1, 1, height * width)

        # 可靠性加权余弦相关（代替标准 QK^T）：
        # 分子 = Σ w * q·k；分母 = sqrt(Σ w q² · Σ w k²)
        numerator = torch.matmul(query * weight, key.transpose(-2, -1))
        query_norm = (query.square() * weight).sum(-1, keepdim=True)
        key_norm = (key.square() * weight).sum(-1, keepdim=True)
        denominator = (query_norm * key_norm.transpose(-2, -1)).add(1e-6).sqrt()
        correlation = numerator / denominator
        # 温度截断到 [0.05, 10] 保证数值稳定，再沿源位置做 softmax
        attention = (correlation * self.temperature.float().clamp(0.05, 10.0)).softmax(-1)

        # 目标自身特征的值投影（自信息）
        target_value = self.target_value(target_n).float().reshape(shape)
        # 源特征值投影乘以可靠性：不可靠的源位置贡献更小
        source_value = self.source_value(source_n).float().reshape(shape) * weight
        # 源值的整体缩放：0.25 * sigmoid(raw)，初期很小
        source_scale = 0.25 * torch.sigmoid(self.source_scale_raw.float())
        # 聚合：以目标值为主、源值为辅
        raw = torch.matmul(attention, target_value) + source_scale * torch.matmul(attention, source_value)
        raw = self.out(raw.reshape(batch, self.channels, height, width).to(target.dtype))

        # 幅度校准（amplitude calibration）：
        # direction 只表达"往哪个方向移动"，把 raw 的 RMS 缩放到接收方
        # 的 RMS（接收方 RMS detach，不回传梯度），使 message 的量级
        # 与接收方特征天然可比，门控因此可以在 [0, 上限] 内解释为比例
        raw_rms = raw.float().square().mean(1, keepdim=True).add(1e-6).sqrt()
        receiver_rms = target.float().square().mean(1, keepdim=True).add(1e-6).sqrt().detach()
        direction = (raw.float() / raw_rms * receiver_rms).to(target.dtype)
        # 收益网络的证据：双方归一化特征、差值绝对值、两个可靠性图
        evidence = torch.cat((target_n, source_n, (target_n - source_n).abs(),
                              source_reliability, target_reliability), 1)
        benefit = torch.sigmoid(self.benefit(evidence).float()).to(target.dtype)
        # 门控 = 幅度上限 x 课程系数 x 源可靠性(detach) x 收益
        # source_reliability detach：可靠性是"信号"，不承担"优化器压力"
        budget = self.maximum_transfer * float(budget_scale) * source_reliability.detach()
        gate = budget * benefit
        return {"direction": direction, "gate": gate, "message": gate * direction,
                "attention": attention}


class CalibratedFusionBlock(nn.Module):
    """单尺度的"恢复 -> 交互 -> 融合"复合块。

    每个模态先经各自的 :class:`BenefitCalibratedDictionary` 做退化恢复；
    随后经共享权重的跨模态注意力模块进行双向消息传递，增强两个分支；
    最后将增强后的两个特征与上一尺度下采样传来的 ``prior_fused``
    先验拼接、投影并精修为本尺度的融合特征。

    参数:
        channels: 该尺度通道数。
        heads: 该尺度注意力头数。
        atoms: 该尺度字典原子数。
        query_channels: 该尺度字典查询维度。
        use_transformer_fusion: 精修用 TransformerBlock（True）或轻量
            :class:`LocalResidual`（False）。最高分辨率尺度（index=0）
            使用 LocalResidual，因为全局注意力在该尺度代价过高且收益小。
    """

    def __init__(self, channels: int, heads: int, atoms: int, query_channels: int,
                 use_transformer_fusion: bool) -> None:
        """初始化融合块。"""
        super().__init__()
        # 可见光字典用标准卷积；红外字典用轴向分解卷积（捕获条带退化）
        self.visible_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels)
        self.infrared_recovery = BenefitCalibratedDictionary(channels, atoms, query_channels, axial=True)
        # 双向消息共享同一套交互权重（参数效率 + 对称性正则）
        self.interaction = ReliabilityWeightedCrossAttention(channels, heads)
        # 融合输入：增强后的可见光 + 红外 + 上一尺度先验 = 3C 通道
        fusion_input = 3 * channels
        self.fuse_project = nn.Sequential(nn.Conv2d(fusion_input, channels, 1), nn.GELU())
        self.fuse_refine = (transformer_block(channels, heads) if use_transformer_fusion
                            else LocalResidual(channels))

    def forward(self, visible: Tensor, infrared: Tensor, prior_fused: Tensor | None,
                budget_scale: float) -> tuple[Tensor, Tensor, Tensor, dict]:
        """执行该尺度的"恢复-交互-融合"。

        参数:
            visible: (B, C, H, W) 归一化后的可见光编码特征。
            infrared: (B, C, H, W) 归一化后的红外编码特征。
            prior_fused: 上一尺度融合特征（最高分辨率尺度为 None）。
            budget_scale: 跨模态预算课程系数。

        返回:
            (visible, infrared, fused, state)：前两项为字典恢复后的
            特征（供编码器级联），fused 为本尺度融合特征；state 携带
            两个字典状态与两个方向的消息，供训练损失使用。
        """
        # 各模态独立的字典恢复（含可靠性估计）
        visible_state = self.visible_recovery(visible)
        infrared_state = self.infrared_recovery(infrared)
        v, i = visible_state["recovered"], infrared_state["recovered"]
        # 双向消息：同一交互模块、不同方向。
        # i_to_v：红外 -> 可见光（源=红外，接收=可见光）；
        # v_to_i：可见光 -> 红外。
        i_to_v = self.interaction.message(v, i, infrared_state["reliability"],
                                          visible_state["reliability"], budget_scale)
        v_to_i = self.interaction.message(i, v, visible_state["reliability"],
                                          infrared_state["reliability"], budget_scale)
        # 门控消息以残差方式注入：跨模态信息只能"增强"，不能"替换"
        visible_out = v + i_to_v["message"]
        infrared_out = i + v_to_i["message"]
        if prior_fused is None:
            # 最高分辨率尺度没有先验：用零占位保持通道数一致
            prior_fused = torch.zeros_like(v)
        elif prior_fused.shape[-2:] != v.shape[-2:]:
            # 先验来自上一尺度（分辨率减半），上采样回本尺度
            prior_fused = upsample(prior_fused, v.shape[-2:])
        # 拼接 -> 1x1 投影 + GELU -> 精修（Transformer 或 LocalResidual）
        fused = self.fuse_refine(self.fuse_project(torch.cat((visible_out, infrared_out, prior_fused), 1)))
        # 打包各中间量，供 CalibFuseLoss 监督
        state = {
            "visible": visible_state, "infrared": infrared_state,
            "i_to_v": i_to_v, "v_to_i": v_to_i,
        }
        return v, i, fused, state


class CompactDecoder(nn.Module):
    """自底向上的紧凑解码器：精修 -> 上采样 -> 与跳跃连接合并。

    最深层融合特征先经一个 TransformerBlock 精修，随后逐级上采样，
    与各尺度融合特征通过"拼接-精修-残差"块合并，
    最终输出全分辨率融合特征。

    参数:
        channels: 三个尺度各自的通道数。
        heads: 三个尺度各自的注意力头数。
    """

    def __init__(self, channels: tuple[int, int, int], heads: tuple[int, int, int]) -> None:
        """按逐尺度配置初始化解码器。"""
        super().__init__()
        # 最深层（1/4 分辨率、通道最多）的精修块
        self.deep = transformer_block(channels[2], heads[2])
        # 上采样通道压缩：channels[index+1] -> channels[index]（3x3 卷积）
        self.up_projects = nn.ModuleList([
            nn.Conv2d(channels[index + 1], channels[index], 3, padding=1) for index in range(2)
        ])
        # 跳跃连接精修：2C 拼接 -> 1x1 压缩 + GELU -> Transformer
        self.refines = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * channels[index], channels[index], 1), nn.GELU(),
                          transformer_block(channels[index], heads[index])) for index in range(2)
        ])

    def forward(self, features: list[Tensor]) -> Tensor:
        """将三个尺度的融合特征解码回全分辨率。

        参数:
            features: [尺度0 (全分辨率), 尺度1 (1/2), 尺度2 (1/4)] 的
                融合特征列表。

        返回:
            全分辨率的融合特征 (B, channels[0], H, W)。
        """
        decoded = self.deep(features[2])
        # 从深层到浅层逐级解码（index 1 -> 0）
        for index in (1, 0):
            up = self.up_projects[index](upsample(decoded, features[index].shape[-2:]))
            # 拼接-精修-残差跳跃连接：浅层细节 + 深层语义
            decoded = features[index] + self.refines[index](torch.cat((features[index], up), 1))
        return decoded


class CalibFuse(nn.Module):
    """CalibFuse：受益校准的恢复 + 可靠性加权的交互。

    默认配置：channels (32, 64, 128)，heads (4, 4, 8)，
    atoms (64, 64, 64)，query_channels (16, 24, 32)——
    共 1,625,131 个参数、12 个 Transformer 块。

    使用方式:
    - 推理：``model(visible, infrared)["fused"]``；
    - 训练：额外传入干净参考与 EMA 教师，开启
      ``return_auxiliary=True``，返回字典状态、教师特征与干净锚点
      参考，供 :class:`~utils.loss.CalibFuseLoss` 使用。
    """

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128),
                 heads: tuple[int, int, int] = (4, 4, 8),
                 atoms: tuple[int, int, int] = (64, 64, 64),
                 query_channels: tuple[int, int, int] = (16, 24, 32)) -> None:
        """构建网络；每个元组参数必须恰好包含三个值。"""
        super().__init__()
        if not (len(channels) == len(heads) == len(atoms) == len(query_channels) == 3):
            raise ValueError("channels, heads, atoms, and query_channels must each contain three values")
        # 配置存入 checkpoint 的 model_config 字段：加载时按存储配置
        # 重建网络，不依赖代码里的默认值（向后兼容）
        self.config = {"channels": channels, "heads": heads, "atoms": atoms,
                       "query_channels": query_channels}
        # 跨模态预算：训练中由 set_training_progress 按 epoch 更新；
        # 推理固定为 1.0（完全放开跨模态交互）
        self.cross_budget_scale = 1.0
        # 可见光/红外各自的 stem：3x3 重叠 patch 嵌入 + GELU
        self.visible_stem = nn.Sequential(OverlapPatchEmbed(3, channels[0]), nn.GELU())
        self.infrared_stem = nn.Sequential(OverlapPatchEmbed(1, channels[0]), nn.GELU())
        # 逐尺度编码器：每个尺度一个 TransformerBlock
        self.visible_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        self.infrared_encoder = nn.ModuleList([transformer_block(c, h) for c, h in zip(channels, heads)])
        # 三个尺度的融合块；最高分辨率尺度（index=0）用 LocalResidual 精修
        self.blocks = nn.ModuleList([
            CalibratedFusionBlock(c, h, k, q, use_transformer_fusion=index > 0)
            for index, (c, h, k, q) in enumerate(zip(channels, heads, atoms, query_channels))
        ])
        # 两条下采样通路（前两个尺度）：模态特征与融合先验各自下采样
        self.visible_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.infrared_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.fused_down = nn.ModuleList([Downsample(c) for c in channels[:2]])
        self.decoder = CompactDecoder(channels, heads)
        # 输出头：Transformer 精修 + 3x3 卷积输出 3 通道；
        # sigmoid 在 forward 中施加，输出 [0,1] RGB
        self.fusion_head = nn.Sequential(transformer_block(channels[0], heads[0]),
                                         nn.Conv2d(channels[0], 3, 3, padding=1))

    def set_training_progress(self, epoch: int) -> None:
        """更新跨模态预算：epoch < 5 时为 0，epoch 5~20 线性爬升到 1。

        课程式策略：先让各模态字典恢复专注处理自身退化，
        再逐步放开跨模态交互，避免早期不可靠的消息互相污染。
        学生与 EMA 教师需同步调用（见 train_epoch），保证预览一致。
        """
        self.cross_budget_scale = min(max((epoch - 5) / 15.0, 0.0), 1.0)

    def _encode(self, visible: Tensor, infrared: Tensor, return_auxiliary: bool) -> dict:
        """多尺度编码 -> 融合 -> 解码；返回融合图像与逐尺度状态。

        参数:
            visible: (B, 3, H, W) 可见光输入。
            infrared: (B, 1, H, W) 红外输入。
            return_auxiliary: 是否收集逐尺度的字典/消息状态（训练用）。

        返回:
            {"fused": (B, 3, H, W) ∈ [0,1], "states": [逐尺度状态字典]}
        """
        v, i = self.visible_stem(visible), self.infrared_stem(infrared)
        scales, states, prior = [], [], None
        for index, block in enumerate(self.blocks):
            # 每个融合块之前对编码特征重新 RMS 归一化：
            # 保持特征能量稳定，与字典/注意力模块的余弦假设一致
            v = rms_normalize(self.visible_encoder[index](v))
            i = rms_normalize(self.infrared_encoder[index](i))
            v, i, fused, state = block(v, i, prior, self.cross_budget_scale)
            scales.append(fused)
            if return_auxiliary:
                states.append(state)
            if index < 2:
                # 模态特征与融合先验分别下采样后进入下一尺度
                v = self.visible_down[index](v)
                i = self.infrared_down[index](i)
                prior = self.fused_down[index](fused)
        return {"fused": torch.sigmoid(self.fusion_head(self.decoder(scales))),
                "states": states}

    @torch.no_grad()
    def clean_reference_features(self, visible: Tensor, infrared: Tensor) -> dict:
        """提取干净图像的逐尺度特征与字典分配（干净教师路径）。

        只做"编码 + 字典恢复"，不做跨模态交互，也不做解码。
        注意下采样作用在"恢复后"的特征上，使干净路径与带噪路径
        的信息流结构一致（对照公平）。

        该方法在 no_grad 下运行（教师是监督目标，不是梯度通路）。

        参数:
            visible: (B, 3, H, W) 干净可见光。
            infrared: (B, 1, H, W) 干净红外。

        返回:
            {"visible_features": [3 个尺度], "infrared_features": [3 个],
             "visible_assignments": [3 个], "infrared_assignments": [3 个]}
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
                # 与带噪路径一致：下采样作用于恢复后特征
                v = self.visible_down[index](v_state["recovered"])
                i = self.infrared_down[index](i_state["recovered"])
        return result

    def forward(self, visible: Tensor, infrared: Tensor, *, clean_visible: Tensor | None = None,
                clean_infrared: Tensor | None = None, return_auxiliary: bool = False,
                clean_teacher: nn.Module | None = None) -> dict:
        """前向传播。

        参数:
            visible: (B, 3, H, W) ∈ [0,1] 可见光输入（可为退化图像）。
            infrared: (B, 1, H, W) 红外输入（多通道会被自动转灰度）。
            clean_visible / clean_infrared: 干净参考对（训练专用），
                与 clean_teacher 一起驱动字典锚点损失。
            return_auxiliary: 为 True 时返回训练所需的辅助量。
            clean_teacher: 提供"干净特征"的教师模型（训练传 EMA 模型；
                缺省时用自身）。

        返回:
            推理：``{"fused": (B, 3, H, W)}``。
            训练（return_auxiliary=True）：额外包含 ``states``
            （逐尺度字典/消息状态）、``teacher``（教师干净特征与分配）、
            ``clean_anchor_references``（干净特征上可导的字典检索结果
            ——字典参数唯一的梯度来源）。
        """
        if infrared.shape[1] != 1:
            # 红外允许多通道输入，自动加权求和为单通道
            infrared = infrared.mean(1, keepdim=True)
        # 填充到 4 的倍数；输出后再裁回原尺寸
        visible, original_size = pad_to_multiple(visible)
        infrared, _ = pad_to_multiple(infrared)
        output = self._encode(visible, infrared, return_auxiliary)
        output["fused"] = output["fused"][..., :original_size[0], :original_size[1]]
        if not return_auxiliary:
            output.pop("states")
            return output
        # 以下为训练专用：干净参考与字典锚点
        if clean_visible is None or clean_infrared is None:
            raise ValueError("clean_visible and clean_infrared are required for auxiliary training output")
        if clean_infrared.shape[1] != 1:
            clean_infrared = clean_infrared.mean(1, keepdim=True)
        clean_visible, _ = pad_to_multiple(clean_visible)
        clean_infrared, _ = pad_to_multiple(clean_infrared)
        # 教师缺省为自身；训练时传入 EMA 模型（更稳定的干净特征）
        teacher = self if clean_teacher is None else clean_teacher
        with torch.no_grad():
            clean = teacher.clean_reference_features(clean_visible, clean_infrared)
        output["teacher"] = clean
        # 在干净特征上"重新检索"，且不 detach 字典：
        # 该路径产生的锚点参考是字典参数唯一的梯度来源——
        # 保证字典原子始终对齐"干净"特征空间，不被训练期的
        # 退化样本拉偏
        output["clean_anchor_references"] = {"visible": [], "infrared": []}
        for index, block in enumerate(self.blocks):
            for modality, recovery in (("visible", block.visible_recovery),
                                       ("infrared", block.infrared_recovery)):
                reference, _ = recovery.retrieve(clean[f"{modality}_features"][index],
                                                 detach_dictionary=False)
                output["clean_anchor_references"][modality].append(reference)
        return output
