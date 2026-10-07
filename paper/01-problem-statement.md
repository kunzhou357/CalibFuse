# 问题陈述稿 v1(2026-09-28)

> 依据五篇参考论文(Text-IF CVPR24 / Deno-IF NeurIPS25 / Dream-IF AAAI26 / AMG-Fuse / ReCoFuse CVPR26)校准后的定稿候选。**v2 方案中"退化融合是空白"的说法作废**——该方向已拥挤,我们的问题必须且可以更锋利。

## 0. 一句话问题(论文的心脏)

> **退化鲁棒融合一直被框架成"恢复放在哪"的架构问题(先恢复、一体化、互耦合);我们把它重新框架为"信什么、信多少"的决策问题——退化毁掉的不是像素,是信任,而信任分配至今没有任何方法显式定义、监督、检验过。**

配套的"我们提供什么"(装配必然性):

> 问题要求三件事:逐像素可靠性估计、每次信息传递上的门控、以及"收益"的可计算定义。前两者是现成零件(可靠性头、门控注意力);"收益"的定义看似不可得——**这正是信任分配一直隐式化的结构性原因**——我们证明它在训练时可得:干净教师路径下,最优传递增益有闭式解(ridge-optimal gain),门从 trick 变成可回归、可检验的量。

## 1. 英文正文级问题陈述(可直接入 Intro)

### 1.1 常识档(问题存在性与地位,Intro 第 1–2 段素材)

> Infrared–visible fusion is, at its core, a **trust-allocation decision**: at every pixel, the network must decide how much of each modality to believe. Sensor degradation corrupts this decision heterogeneously across space and modality — low-light visible imagery suffers shot and read noise; uncooled infrared sensors add striping, stuck pixels, and Gaussian noise — so a single scene routinely contains regions where one modality is nearly pristine and the other nearly unusable. The community's response has been an increasingly sophisticated debate about **where** degradation should be removed: before fusion (restoration-then-fusion cascades), inside a single network (integrated hard regression), or through mutually reinforcing restoration–fusion modules (reciprocal coupling). Yet across all these paradigms, the trust-allocation decision itself remains implicit: no method estimates per-pixel reliability, no method states what a *beneficial* information transfer is, and no method can tell, after the fact, whether the network trusted the right modality in the right place. We argue that this missing formalization — not the placement of the restoration module — is the load-bearing gap of degradation-robust fusion.

### 1.2 锋利档(机制解释,Intro 第 3 段素材;实验坐实)

> Why does implicit trust allocation fail? Because the objectives that made fusion work on clean data are precisely the ones that **reward degradation**. The gradient term of a standard fusion loss prefers the stronger local gradient — and noise carries the strongest gradients in the image; the max-intensity term prefers the brightest response — and shot-noise speckle is bright. Feeding degraded inputs to a fusion network trained under such losses does not merely "reduce quality": the objective actively teaches the network to preserve and amplify the very degradation it was supposed to overcome. No placement of a restoration module can fully repair this, because the fault lies in the **decision rule**, not in the location of the denoiser.

### 1.3 "为什么被忽略"段(结构性解释,绝不暗示前人傻)

> Three structural reasons kept this question unasked. First, benchmark culture: the field's datasets and leaderboards are built on clean, registered pairs, so degraded inputs fall outside the evaluation loop. Second, the restoration lens: once degradation is seen as a *restoration* problem, the natural questions become architectural — where to put the denoiser, how tightly to couple it — and the decision rule inside fusion stays out of sight. Third, and most deeply, trust allocation looked unformalizable: "how much should this transfer be believed?" seems to require a ground truth for *benefit* that is unavailable at test time. We show it is available at training time: with a clean-path teacher, the optimal transfer gain is computable in closed form, which turns every gate in the network into a quantity that can be defined, regressed, and audited.

## 2. 竞争地形校准(五篇各占了什么,我们拿什么差异句)

| 论文 | 它的问题框架 | 它占住的 | 我们的差异句(unlike 句式) |
|---|---|---|---|
| Text-IF (CVPR24) | 融合对退化"无助"且不可交互 | 语义文本引导 + all-in-one 去退化(215M 参数) | Unlike text-guided all-in-one restoration, we do not add semantics or extra modalities: we formalize the trust decision the text is meant to proxy. |
| Deno-IF (NeurIPS25) | 有噪融合方法依赖配对监督、泛化差 | 无监督(卷积低秩先验)、1.43M 轻量联合去噪融合 | Unlike unsupervised low-rank recovery, our supervision is *calibrative*: clean references are used not to reconstruct pixels but to regress gates onto computable optimal gains. |
| Dream-IF (AAAI26) | 融合与恢复被分开处理,忽视"一模态主导区提示另一模态待增强区" | relative dominance 互增强 + 退化 prompt | Unlike dominance-driven enhancement, our per-pixel weights are not derived from the fusion mapping but supervised against a closed-form target, hence inspectable. |
| AMG-Fuse (2026) | 级联不稳定+误差累积;Pseudo-GT 有模态偏置 | mask 量化各模态贡献 + mask 引导交互 | 同上:贡献 mask 是从融合映射反推的启发式;我们给出收益的定义式与监督源,并额外给出可靠性与有害修正率等可审计量。 |
| ReCoFuse (CVPR26) | 硬回归 vs 解耦两代范式都有极限;"核心问题是恢复与融合的关系" | 互耦合范式(diffusion,首发定义权) | **正面接管其句式**:它说核心问题是"where restoration happens";我们说那是第二代问题,真正核心是"how fusion allocates trust"——正交轴,任何耦合范式内部都缺这一层。 |

**范式地图(Fig.1 结构,借用 ReCoFuse/Text-IF 的 taxonomy 图式)**:级联解耦 → 一体化硬回归 → 互耦合(以上全部:恢复放哪里)∥ **信任分配形式化(我们:恢复结果信多少)**。Our Fig.1 右下角放 gate/可靠性热图:噪声区门自动关闭——一张图证明"决策可见"。

**必须防守的一点**:按 ReCoFuse 的分类法,我们属于"一体化"阵营(端到端、用干净参考)。防守句:信任分配是**正交轴**,与任何耦合范式兼容;我们恰以 1.63M 确定性前馈(无 diffusion 采样)证明它不需要重机械。

## 3. 显而易见性三测自检

1. **外行测试:过**。"该信谁、信多少"是日常语言;退化=信任被腐蚀的比喻自明。
2. **后见之明测试:过,且更强**。读过 ReCoFuse 的人会立刻认出耦合之辩,然后发现正交问题"我早该想到"——这正是接管句式的效果。
3. **结构解释测试:过**。三条结构性原因(基准文化、恢复视角、形式化看似不可能),无人被指傻;且第三条直接引出我们的技术贡献(closed-form gain),问题陈述与方案无缝闭环。

## 4. 对 v2 写作方案的修订(即时生效)

1. §1 问题陈述替换为本文档 §0–§1 版本;"被忽略的空白"措辞全局禁用;
2. 锋利钩子(损失奖励噪声)从"头牌"降为**机制证据**(第 3 段+小分析实验),因为"融合对噪声无助"已被 Text-IF/Deno-IF 占住;我们的增量是**指出奖励机制的定量结构**;
3. 基线清单更新(见 00 方案 §6):主打退化家族 Text-IF、Deno-IF、Dream-IF、ReCoFuse、OmniFuse、BA-Fusion + 经典干净融合 CDDFuse、SwinFusion、TarDAL + 级联两档;参数量对比表直接引用 Deno-IF 的气泡图式(我们 1.63M vs Text-IF 215M / DDFM 552M);
4. Related Work 骨架按本文档 §2 表展开:按"恢复放哪"三代范式组织,末节"trust allocation"归我们;
5. 参考论文库在 `/Users/chou/Desktop/面向退化/`(共 12 篇,含 Shi_Degradation-Robust CVPR26、DRMF、DSPFusion、URFusion、FreeFusion,后续按需提取)。

## 5. 待办

- [ ] 损失奖励噪声的定量小实验(梯度能量:噪声图 vs 干净图;max 亮度命中噪声像素比例)→ 坐实 1.2;
- [ ] Fig.1 草图(范式地图 + gate 热图);
- [ ] Introduction 逐段细纲(下一个交付物)。
