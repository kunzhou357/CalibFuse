# CalibFuse 方法说明（论文方法部分参考）

> 本文档面向论文写作，系统整理 CalibFuse 的方法设计与全部公式。
> 所有细节均以仓库源码为准：网络见 `nets/fusion.py`、`nets/dictionary.py`，
> 损失见 `utils/loss.py`，退化模型见 `utils/degradation.py`，
> 数据与采样见 `utils/dataset.py`。
> 记号约定：$V\in[0,1]^{3\times H\times W}$ 为可见光（RGB），$I\in[0,1]^{1\times H\times W}$ 为红外（灰度），
> $\hat F\in[0,1]^{3\times H\times W}$ 为融合输出。

---

## 1 问题定义与总体思路

**任务**：给定一对可见光/红外图像（两者可能各自携带传感器退化），
生成一幅既保留结构细节、又对退化鲁棒的融合图像。

**核心观点**：融合前必须先回答三个问题——
1. 这个模态自身退化了吗、恢复得如何（恢复与可靠性估计）；
2. 对方模态的信息对我有收益吗、收益多大（收益校准）；
3. 允许多少对方信息进入（有界门控交互）。

CalibFuse 将三者显式建模为网络结构：
**受益校准的字典恢复**（§4）、**可靠性加权的跨模态交互**（§5）、
以及**课程式跨模态预算**（§8.3），在统一的三尺度 U 形框架（§3）中联合训练。

网络规模：**1,625,131 参数、12 个 Transformer 块**，
通道配置 $(32, 64, 128)$、注意力头 $(4, 4, 8)$、字典原子每尺度 64、
查询维度 $(16, 24, 32)$。

---

## 2 预备模块

### 2.1 RMS 归一化

全网统一使用按样本的 RMS 归一化（`rms_normalize`）：

$$
\mathrm{RMS}(x) = x \,\Big/\, \sqrt{\tfrac{1}{CHW}\textstyle\sum_{c,h,w} x_{c,h,w}^2 + \varepsilon}
$$

只去除整体能量尺度、保留空间对比度结构。可见光与红外特征的
原始能量量纲差异大，归一化后二者的余弦相似度、字典检索才有可比性。
每个尺度的编码特征在进入融合块前都会重新归一化，保持能量稳定。

### 2.2 Restormer 基础块

编码器、精修块与输出头均采用 Restormer 的
TransformerBlock（`nets/restormer.py`，MIT 许可，见 `licenses/`）：

- **MDTA 转置注意力**：注意力在通道维计算（$(C, C)$ 矩阵而非 $(HW, HW)$），
  复杂度对空间分辨率线性；查询/键做通道维 L2 归一化，形成余弦注意力。
- **GDFN 门控前馈**：$1{\times}1$ 升维 → $3{\times}3$ 深度卷积 →
  通道对半门控 $\mathrm{GELU}(x_1)\odot x_2$ → $1{\times}1$ 降维。
- 统一设置：FFN 扩张 2.0、卷积无偏置、带偏置 LayerNorm；Pre-Norm 残差结构。

下采样为 $3{\times}3$ 卷积（通道减半）+ PixelUnshuffle(2)。

---

## 3 网络总体结构

```
可见光 V ─ stem(3x3嵌入) ┐
                          ├─ 逐尺度: Transformer编码 → RMS归一化 ─┐
红外   I ─ stem(3x3嵌入) ┘                                        │
                                                                  ▼
        ┌────────────── 3 个尺度的 CalibratedFusionBlock ───────────────┐
        │  字典恢复(可见光) ┐                                            │
        │  字典恢复(红外)   ├→ 双向可靠性加权消息 → 融合 → prior 级联 │
        │  (共享)跨模态交互 ┘    (§5)              (§6)     ↓下采样     │
        └──────────────────────────────────────────────────────────────┘
                          │  三尺度融合特征
                          ▼
        CompactDecoder（自底向上：精修→上采样→拼接-精修-残差跳跃）
                          ▼
        fusion_head（Transformer + 3x3 conv + sigmoid）→ F̂ ∈ [0,1]
```

**数据流要点**：
- 每个尺度：模态特征归一化 → `CalibratedFusionBlock`（恢复+交互+融合）；
- 模态特征与该尺度融合特征分别经下采样进入下一尺度；
  融合特征作为先验 `prior_fused` 上采样后拼入下一尺度的融合输入，
  实现**由粗到细的先验级联**；
- 最高分辨率尺度（$32$ 通道）的融合精修用轻量 `LocalResidual`
  （$1{\times}1$ → $5{\times}5$ 深度卷积 → $1{\times}1$，近恒等初始化），
  其余尺度用 TransformerBlock。

输入在入口处填充至 4 的倍数（三尺度两次 2× 下采样 + 4 倍对齐要求），
输出裁回原尺寸，故**任意分辨率可处理**；红外若为多通道输入，
自动转为单通道灰度。

---

## 4 受益校准的字典恢复（Benefit-Calibrated Dictionary Recovery）

每个尺度、每个模态各有一个字典恢复模块（红外侧带轴向分解卷积），
把"退化去除"分解为三步（`nets/dictionary.py`）。

### 4.1 第一步：字典检索（retrieve）

可学习字典 $D \in \mathbb{R}^{K\times C}$（$K=64$ 个原子），
经值投影得 $V_D \in \mathbb{R}^{K\times C}$、经键投影并 L2 归一化得 $K_D$。
对（可能退化的）特征 $z$：

$$
q = \mathrm{L2norm}\big(W_q\,(\bar z + \mathrm{DWConv}(\bar z))\big),\qquad \bar z = \mathrm{RMS}(z)
$$

$$
A_{k,h,w} = \mathrm{softmax}_k\!\Big(\tau \cdot \frac{\langle q_{h,w},\, k_k\rangle}{\|q\|\,\|k_k\|}\Big),\qquad
r = \sum_{k=1}^{K} A_k\, V_D[k]
$$

其中可学习温度 $\tau = 1 + 29\sigma(\theta_\tau) \in (1, 30)$（初始约 27，接近硬分配），
$A \in \mathbb{R}^{K\times H\times W}$ 为软分配，$r$ 为"干净参考"特征。

### 4.2 第二步：候选修正（candidate）

以证据 $(z,\, r,\, z-r)$ 为输入，网络提出有界修正：

$$
\Delta = \tanh\big(W_o\,\mathrm{GELU}(\mathrm{DWConv}_{5\times5}(W_i [z; r; z-r]))\big)
$$

红外侧额外叠加行 $(1{\times}9)$ / 列 $(9{\times}1)$ 分解深度卷积的轴向上下文
（以远低于 $9{\times}9$ 的代价捕获长条带结构）。$\tanh$ 保证修正有界。

### 4.3 第三步：受益校准采纳（adopt）

预测逐像素采纳系数 $\alpha \in [0,1]$ 并执行**核心更新式**：

$$
\tilde z = z + \alpha \odot \Delta
$$

$\alpha$ 由采纳头从证据 $(z, r, \Delta, \mathcal{H}, \mathcal{M})$ 预测，
其中 $\mathcal H$ 为归一化分配熵（$\mathcal H = -\sum_k A_k \log A_k / \log K$）、
$\mathcal M$ 为 top-2 分配概率差——二者刻画检索质量：
熵大或 margin 小意味着检索不可靠。
网络学习的是"**在何处、采纳多少**"（where & how much），
而非直接回归修正内容（how）。

### 4.4 可靠性估计

误差校准头从证据 $(z, \tilde z, \mathcal{H}, \mathcal{M})$ 预测对数恢复误差
$e = \mathrm{softplus}(\cdot)$，并给出可靠性：

$$
\mathrm{rel} = \exp(-\,\mathrm{expm1}(\min(e, 10))) \in [0, 1]
$$

可靠性作为置信度信号进入 §5 的跨模态门控：源模态越不可靠，
其消息被注入得越少。误差校准由 §7.4 的监督保证语义正确。

### 4.5 梯度隔离（干净锚点契约）

**关键设计**：带噪路径的检索把字典 $D$ 与键/值投影全部 detach，
因此字典参数**只**从"干净锚点损失"（§7.3）获得梯度。
这保证字典原子始终对齐干净特征空间，不被训练期的退化样本拉偏。

### 4.6 近恒等初始化

候选修正输出层（权重 $\mathcal N(0, 10^{-3})$、偏置 0）与
采纳头末层（偏置 $-2$，初始 $\alpha \approx \sigma(-2) \approx 0.12$）
均为近恒等初始化：训练从"几乎不修正"起步，由损失逐步放开，
避免早期大幅扰动。

---

## 5 可靠性加权的跨模态交互（Reliability-Weighted Cross-Modal Interaction）

同一交互模块（共享权重）在每个尺度上计算**双向**消息
$i{\to}v$ 与 $v{\to}i$（`ReliabilityWeightedCrossAttention.message`）。

### 5.1 可靠性加权余弦注意力

查询来自**接收方**（"我需要什么"）、键来自**发送方**（"我能给什么"），
双方先 RMS 归一化以消除能量尺度差：

$$
\mathrm{corr}(q, k) = \frac{\sum_p w_p\, q_p k_p}{\sqrt{\big(\textstyle\sum_p w_p q_p^2\big)\big(\textstyle\sum_p w_p k_p^2\big)} + \varepsilon}
$$

其中 $w_p$ 为源侧逐位置可靠性。注意力
$a = \mathrm{softmax}_p(\mathrm{corr}\cdot \tau_a)$，
$\tau_a$ 为逐头可学习温度（推理截断到 $[0.05, 10]$）。
聚合以目标自身为主、源为辅：

$$
o = \mathrm{Conv}_{1\times1}\Big(\mathrm{Attn}(V_t) + s\cdot \mathrm{Attn}(w \odot V_s)\Big),\quad s = 0.25\,\sigma(\theta_s)\approx 0.027
$$

### 5.2 幅度校准（direction）

把聚合输出 $o$ 的 RMS 缩放到接收方特征的 RMS，得到**方向**：

$$
\mathrm{dir} = \frac{o}{\mathrm{RMS}(o)} \cdot \mathrm{RMS}(t)\big|_{\text{detach}}
$$

方向只表达"往哪里移动"，量级天然与接收方可比，
使门控可以解释为比例而非绝对幅度。

### 5.3 受益门控（gate）——方法的核心公式

$$
\boxed{\; m = g \odot \mathrm{dir},\qquad
g = \underbrace{g_{\max}}_{0.15}\;\cdot\;\underbrace{\beta}_{\text{课程}}\;\cdot\;\underbrace{\mathrm{rel}_{\text{源}}}_{\text{detach}}\;\cdot\;\underbrace{b}_{\in[0,1]}\;}
$$

- $g_{\max} = 0.15$：单次传递的**幅度上限**，从结构上限制跨模态注入量级；
- $\beta \in [0,1]$：跨模态预算课程系数（§8.3）；
- $\mathrm{rel}_{\text{源}}$：发送方字典恢复估计的可靠性（**detach**，
  只作信号、不承担优化器压力）；
- $b$：受益预测头从证据
  $[\mathrm{RMS}(t); \mathrm{RMS}(s); |\mathrm{RMS}(t)-\mathrm{RMS}(s)|; \mathrm{rel}_{\text{源}}; \mathrm{rel}_{\text{目}}]$
  预测的"接收该消息的融合收益"（近恒等初始化，初始 $\approx 0.12$）。

**只有当源可靠且预测收益高时，信息才流动**；
消息以残差方式注入（$\tilde t = t + m$）：
跨模态信息只能"增强"，不能"替换"。

---

## 6 校准融合块与紧凑解码器

### 6.1 CalibratedFusionBlock（每尺度）

1. 双模态字典恢复：$v, i$ 为恢复后特征，附带可靠性图；
2. 双向消息：$v^{+} = v + m_{i\to v}$，$i^{+} = i + m_{v\to i}$；
3. 融合：拼接 $[v^{+}; i^{+}; P]$（$P$ 为上一尺度先验上采样；
   最高分辨率尺度 $P=0$）→ $1{\times}1$ 投影 + GELU → 精修。

### 6.2 CompactDecoder 与输出头

三尺度融合特征 $\{F_0, F_1, F_2\}$（$F_2$ 最深）自底向上解码：

$$
d_2 = \mathrm{TB}(F_2),\qquad
d_j = F_j + \mathrm{Refine}_j\big([\,F_j;\ \mathrm{Up}(\mathrm{Conv}(d_{j+1}))\,]\big),\ j = 1, 0
$$

$\mathrm{Up}$ 为双线性上采样，Refine = $1{\times}1$ 拼接压缩 + GELU + Transformer。
最后 $\hat F = \sigma(\mathrm{Conv}_{3\times3}(\mathrm{TB}(d_0)))$。

---

## 7 训练监督（utils/loss.py）

总损失（默认权重见括号）：

$$
\mathcal{L} = \mathcal{L}_{\text{fus}} \;+\; 0.10\,(\mathcal{L}_{\text{rec}} + 0.25\,\mathcal{L}_{\text{cand}})
\;+\; 0.10\,\mathcal{L}_{\text{anc}} \;+\; 0.05\,\mathcal{L}_{\text{KL}} \;+\; 0.05\,\mathcal{L}_{\text{cal}}
$$

其中 $\mathcal{L}_{\text{cal}} = \mathcal{L}_{\alpha} + \mathcal{L}_{e} + \mathcal{L}_{g}$
（采纳 + 误差校准 + 交互门控）。所有正则对 3 个尺度、2 个模态取平均。

### 7.1 融合损失（只作用于最终输出 $\hat F$）

以干净源构造目标（可微的经典融合规则，`fusion_targets`）：

- 亮度目标 $Y^{*} = \max(Y_V, I)$（max 规则，保留两源最亮结构）；
- 梯度目标：逐像素取 $(g_x^2 + g_y^2)$ 能量更大的源的 Sobel 梯度。

$$
\mathcal{L}_{\text{fus}} = 2.0\,\mathrm{SmoothL1}_{0.02}(Y_{\hat F}, Y^{*})
+ 1.0\,\mathcal{L}_{\text{SSIM}}(Y_{\hat F}, Y^{*})
+ 2.0\,\mathrm{SmoothL1}_{0.01}(\nabla_{\hat F}, \nabla^{*})
+ 1.5\,\mathrm{SmoothL1}_{0.02}(C_{\hat F}, C_{V})
$$

颜色项只约束 YCbCr 的色度（Cb/Cr）——色度唯一来自可见光；
亮度由前三项处理。SSIM 项为 $\mathrm{mean}(0.5(1-\mathrm{SSIM}))$（11×11 窗）。

### 7.2 干净教师（EMA）

教师 = 学生的指数滑动平均，参数按
$\theta_T \leftarrow \min(0.99, \tfrac{1+u}{10+u})\,\theta_T + (1 - \cdot)\,\theta_S$
更新（buffer 直接拷贝）；fp16 下优化器步进被跳过时同步跳过 EMA。
教师在**干净图像**上编码（`clean_reference_features`，no_grad），
提供逐尺度干净特征 $\{T^{(s)}\}$ 与干净字典分配 $\{A_T^{(s)}\}$。
推理默认使用 EMA 权重。

### 7.3 恢复 / 候选 / 锚点（特征级，目标均 detach）

对每尺度 $s$、每模态：

$$
\mathcal{L}_{\text{rec}} = \mathrm{SL1}_{0.1}\big(\tilde z^{(s)},\, T^{(s)}\big),\qquad
\mathcal{L}_{\text{cand}} = \mathrm{SL1}_{0.1}\big(z^{(s)} + \Delta^{(s)},\, T^{(s)}\big)\ \text{(仅退化样本)}
$$

$$
\mathcal{L}_{\text{anc}} = \mathrm{SL1}_{0.1}\big(\mathrm{retrieve}_{\text{可导}}(T^{(s)}),\, T^{(s)}\big)
$$

**锚点项是字典参数唯一的梯度路径**（§4.5）：在干净特征上重新检索且不 detach，
把原子拉向"干净特征的可重建表示"。
候选项同样只对退化样本计算（干净样本无可修正内容）。

### 7.4 采纳门控与交互门控：岭回归最优增益监督

**最优增益**（`optimal_gain`，no_grad）：对每个像素求解
$\arg\min_g \|o + g\,d - t\|^2$：

$$
g^{*} = \mathrm{clip}\Big(\frac{\langle t - o,\, d\rangle}{\langle d, d\rangle + \varepsilon},\ 0,\ u\Big),\qquad
\langle d, d\rangle \le \varepsilon \Rightarrow g^{*} = 0
$$

- **采纳监督** $\mathcal L_\alpha$：$o = z$，$d = \Delta$，$t = T$；
  **干净样本的增益目标强制为 0**（干净 ⇒ 无需修正）。
- **交互监督** $\mathcal L_g$：$o$ = 接收方恢复特征，$d$ = 消息方向，$t$ = $T$，
  上限 $u = g_{\max} \cdot \mathrm{rel}_{\text{源}}$；
  门控与目标同除 $g_{\max}$ 归一化后做 SmoothL1。

这等价于把学习的门控回归到"如果只允许沿该方向走 $g$ 步，
走多少步最优"——门控因此具有明确的几何语义。

### 7.5 误差校准与检索一致性

$$
\mathcal{L}_{e} = \mathrm{SL1}_{0.05}\big(e^{(s)},\ \log(1 + \mathrm{MSE}_{ch}(\tilde z^{(s)}, T^{(s)}))\big),\qquad
\mathcal{L}_{\text{KL}} = \mathbb{E}_{\text{退化样本}}\big[\mathrm{KL}(A \,\|\, A_T)\big]
$$

误差校准让 $\mathrm{rel} = e^{-e}$ 具有真实语义（§4.4）；
检索 KL 约束"退化不改变特征在字典流形上的位置"。

### 7.6 诊断量（无梯度）

- `harmful_correction_rate`：恢复后误差反而增大（$>10^{-4}$）的像素占比；
- `input_teacher_mse`：恢复前对教师的 MSE。
  训练日志逐 epoch 报告（`log.txt`）。

---

## 8 训练策略

### 8.1 在线成对退化（utils/degradation.py）

训练时在线合成观测，"观测-干净"逐像素对齐，天然免费获得监督：

- **可见光**：泊松散粒 + 高斯读出
  $\mathrm{obs} = \mathrm{clip}(\mathrm{Poisson}(Y\varphi)/\varphi + \epsilon_r,\ 0, 1)$，
  $\varphi \sim \mathrm{LogU}[12, 80]$，$\epsilon_r \sim \mathcal N(0, \sigma_r^2)$，
  $\sigma_r \sim U[0.002, 0.025]$；
- **红外**：条带（列 = 平滑随机 + 固定周期正弦、行 = 平滑随机，
  归一化到目标 std）同时扰动增益与偏置
  $\mathrm{obs} = (1 + 0.35\,S)\odot I + S + \epsilon_g$，
  外加随机坏点（钉在 0/1）；
  $\sigma_g \sim U[0.004, 0.100]$，列幅度 $U[0.015, 0.090]$，
  行幅度 $U[0, 0.035]$，坏点率 $U[2{\times}10^{-4}, 0.006]$。
  另有参数区间不重叠的 `hard` 档用于鲁棒性上限评估。

噪声参数每次从独立播种的 `torch.Generator` 抽取——
种子 = $seed + epoch \times 1{,}000{,}003 + index$，逐样本逐 epoch 可复现且不重复。

### 8.2 四状态轮换与质量均衡采样（utils/dataset.py）

样本 $i$ 在 epoch $e$ 的质量状态为 $(i + e) \bmod 4$：
{全干净, 可见光带噪, 红外带噪, 双侧带噪}。
`QualityBalancedBatchSampler` 保证**每个 batch 四种状态严格各占
batch/4**（组内洗牌 + round-robin 发牌 + 小组回绕），
使各损失项的批量统计无偏。因此 batch size 必须被 4 整除。

worker 每个 epoch 重建（`DATA_EPOCH_POLICY` 契约）：
持久化 worker 会停留在第一个 epoch、使噪声种子重复。

### 8.3 课程式跨模态预算

$\beta$ 随 epoch 线性爬升：epoch < 5 为 0（跨模态完全关闭，
先学好单模态恢复），epoch 5→20 线性升到 1（逐步放开交互），
之后恒为 1。学生与教师同步更新以保证预览一致。
推理固定 $\beta = 1$。

### 8.4 优化设置（train.py 默认值）

| 项 | 值 |
|---|---|
| 优化器 | AdamW，lr $2\times10^{-4}$，weight decay $10^{-4}$ |
| 调度 | 5 epoch 线性预热 + 余弦退火至 $10^{-6}$ |
| epoch / crop / batch / seed | 100 / 256 / 4 / 3407 |
| AMP | CUDA BF16（fp16 备选，带 GradScaler；跳步时同步跳过 EMA） |
| 梯度裁剪 | 全局范数 1.0 |
| 数值保护 | 非有限损失立即终止训练 |
| CUDA 优化 | cuDNN benchmark、TF32、float32 matmul high |

裁剪策略：评估取确定性中心裁剪；训练 50% 随机裁剪、
50% 红外局部对比度引导（前 10% 高对比位置为锚点），
两模态同窗保证像素对齐。

---

## 9 推理（test.py）

1. 红外尺寸与可见光不一致时双线性对齐到可见光尺寸；
2. 网络内部 pad 到 4 的倍数、输出裁回原尺寸（任意分辨率可用）；
3. 默认 EMA 权重（`--weights model` 可选学生权重）；
   默认权重为随仓库发布的 `ckpt/model.pth`（epoch 99）；
4. 输出 RGB PNG 与其 BT.601 亮度 PNG，并写 `protocol.json`
   （权重 SHA-256、epoch、设备、样本列表）溯源。

---

## 附：符号与实现对照表

| 论文符号 | 实现 | 位置 |
|---|---|---|
| $z \to r$ 字典检索 | `BenefitCalibratedDictionary.retrieve` | nets/dictionary.py |
| $\tilde z = z + \alpha\Delta$ | `forward` 中 `recovered` | nets/dictionary.py |
| $\mathrm{rel}$ | `reliability = exp(-expm1(log_err))` | nets/dictionary.py |
| $m = g\odot\mathrm{dir}$ | `ReliabilityWeightedCrossAttention.message` | nets/fusion.py |
| $g_{\max}=0.15$ | `maximum_transfer` | nets/fusion.py / utils/loss.py |
| 课程 $\beta$ | `set_training_progress` | nets/fusion.py |
| 干净锚点 | `clean_anchor_references` | nets/fusion.py |
| $g^{*}$ 最优增益 | `optimal_gain` | utils/loss.py |
| 四状态轮换 | `state_for_index = (i+e) % 4` | utils/dataset.py |
| EMA 教师 | `CleanEMA` | train.py |
