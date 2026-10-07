# CalibFuse 论文写作指导方案(v3,2026-10-07)

> 输入:七篇主流退化融合论文全文解剖(ControlFusion NeurIPS25 / DSPFusion CVPR25 / PIAFusion Inf.Fusion22 / URFusion TIP25 / AMG-Fuse 26 / Deno-IF NeurIPS25 / Dream-IF AAAI26)。
> 本文档与既有文档的关系:**00 = 总纲(排列组合定位论),01 = 问题陈述,02 = 本文(写作指导:章节级怎么写、借哪些句子、venue 决策、闭环检查)**。
> 最重要的新情报在 §2:**URFusion(TIP 2025)已发表"损失保留甚至放大退化"的定性论断,我们的锋利钩子必须降位改造**。

---

## 0. 四个问题的直接回答

### 0.1 "创新点不新颖"怎么办?

七篇解剖给出的答案比安慰更硬:**这个领域的"架构故事"已经被讲尽了,而"监督/定义故事"完全空着。**

- 七篇全部把退化鲁棒框架成**架构/先验问题**:ControlFusion 用语言-视觉提示调制(182.54M)、DSPFusion 用双先验引导(13.99M)、Dream-IF 用 relative dominance 互增强、AMG-Fuse 用贡献 mask 引导、Deno-IF 用低秩先验蒸馏、URFusion 用内容/外观二分。**没有一篇定义"每次跨模态传递该信多少"、没有一篇监督一个门控、没有一篇审计学到的权重**——Dream-IF 的 RD 可视化被推到附录且无定量验证,Deno-IF 的闭式解教的是"干净图像长什么样"(图像级),不是"该传多少"(增益级)。
- 按总纲(00 §0),我们**本来就不卖单点新颖性**,卖的是"问题(信任分配)× 必然装配 × 工作量 × 格式"。七篇解剖证明这个定位选对了:每篇都有我们可攻击的空位(见 §2 表),而我们的空位(闭式增益监督 + gate 审计实验)无人占据。
- 一句话:**新颖性焦虑的解药不是把方法包装得更玄,而是把 claim 收窄到无人占据的那条轴上**——"benefit 的闭式可监督定义"是七篇中零重叠的主张,守住它就守住了论文。

### 0.2 期刊还是会议?——**按期刊准备,首选 Information Fusion**

| 判断依据 | 事实 | 指向 |
|---|---|---|
| 我们的强项 | 统一复评(10 方法×3 数据集×双条件)、审计实验、8+ 行消融、协议可复现 | 期刊奖励彻底性 |
| 我们的弱项 | 无 diffusion/大模型热点;1.63M 非最小(Deno-IF 1.43M、TarDAL 0.3M 更小);装配式方法 | 顶会单轮深审易死于"novelty 不足" |
| 竞争地形 | 顶会已被占坑:ControlFusion/Deno-IF(NeurIPS25)、DSPFusion(CVPR25)、Dream-IF(AAAI26);TIP 已收 URFusion | 顶会通道拥挤 |
| 体量 | 统一复评 + 审计 + 双协议指标放不进 8 页 | 期刊无页限 |
| 社区主场 | PIAFusion(Inf.Fusion 22)等融合主线论文都在此;审稿人池与基线作者重叠 | Information Fusion |

**行动决策:主线 Information Fusion(一区,IF≈14,融合社区主场)。备选:IEEE TMM / Neurocomputing(更快)。会议版本(8 页)保留为快速占坑选项(AAAI/ICME),砍统一复评、保留审计实验与动机图。** 期刊版与会议版共用同一套骨架(00 §3 的五段链),只扩分析与消融。

### 0.3 逻辑能闭环吗?——能,但有三处焊点,每一处都有对应实验

完整链条(带编号,写作时逐环检查):

```
① 真实传感器退化(物理:微测辐射热计/CMOS 噪声,借 Deno-IF 引句)
② 退化使信任异质(空间×模态:同一场景一模态近 pristine、另一模态近不可用)
③ 现有响应全部是"恢复放哪里"的架构之争(三代范式,借 Dream-IF Fig.1)
④ 架构之争不解决决策规则:标准损失奖励噪声
   【W1 焊点:机理分解+定量小实验——URFusion 已定性说过,我们必须更细】
⑤ 所以重新框架:信任分配——"信什么、信多少"从未被定义/监督/检验
⑥ 信任分配三要素:可靠性估计(现成)/门控(现成)/benefit 定义(看似不可得)
⑦ benefit 在训练期可得:干净 EMA 教师 + ridge 闭式最优增益
   【W2 焊点:推导严格性,含适用边界声明】
⑧ 必然装配:字典恢复+可靠性加权交互+课程预算+四态均衡(每个零件大方引用)
⑨ 实验证据:主表领先 + gate-增益相关审计 + 可靠性热图 + harmful-rate
   【W3 焊点:gate-增益相关曲线是核心 claim 的直接证据,相关性弱则论文塌方】
⑩ 回扣审计性:我们能指出"网络在每一处信了谁"——七篇都做不到
```

**三个焊点的处置**:W1 是写作问题(引用+增量限定,§2.2 已给修订段落);W2 是推导问题(§5.5 有写法);W3 是**科学风险,必须最先跑**(§9 行动清单第一项)。另有 W4(干净条件不掉点,守住"校准无代价")、W5(与 URFusion 的增量清晰化)。

### 0.4 能讲好故事吗?——体裁完全成立,且我们的钩子比七篇都锋利

七篇 Intro 的共同体裁就是我们的体裁:"现象 → 分类 → However-gap → we propose 统一 X → contributions"。区别只在 gap 的层级:它们全部停在**架构层**(提示/先验/主导度/mask),我们把 gap 立在**形式化层**(信任量无定义、无监督、无审计)——这是更深的楼层,而且有闭式解这个"硬通货"收尾。PIAFusion 的 "how to fusion vs when to fusion" 证明:换轴重框架在这个社区是被认可的故事模板,我们只是再换一次轴(where to restore → how much to trust)。

---

## 1. 七篇解剖总表

| 论文 | venue/规模 | 问题框架 | 核心机制 | 与我们的关系 | 威胁等级 |
|---|---|---|---|---|---|
| ControlFusion | NeurIPS 2025;182.54M(含冻结 CLIP 102M) | 退化可控性:提示统一建模退化类型+程度 | 语言-视觉退化提示 + FiLM 式调制恢复融合 | 一体化重装 paradigm;Fig.1(III)"性能随退化陡降"是最佳外部动机证据 | 低(无逐像素可靠性、全局调制) |
| DSPFusion | CVPR 2025;13.99M | 双先验:退化判别+扩散恢复语义先验 | SPEN/DPEN+紧凑潜空间扩散(自认 coarse-grained) | "为退化找回干净参照系"同族;**SF/SD 被噪声虚高的自认句**是度量侧黄金证据 | 中(教师思想相邻,粒度/确定性都不同) |
| PIAFusion | Inf.Fusion 2022;未报参数 | 光照失衡(昼夜模态权重) | 图像级光照标量→损失权重 | **max 强度/梯度规则的具名实例(Eq.10/11)**,损失病理分析的靶子;MSRS 出处必引 | 低(标量权重、只进损失) |
| URFusion | TIP 2025;表格数值缺失待核 | 退化二分:内容相关/外观相关,无监督 | 特征级约束+外观统计赋值 | **最高警告:已发表"损失误保留甚至放大退化"论断**;我们的钩子必须降位改造(§2.2) | 高(动机撞车,机制不撞) |
| AMG-Fuse | arXiv 2026;59.74M | 伪真值有模态偏置;mask 量化贡献 | 线性混合反解闭式贡献 mask M=(F−IR)/(VI−IR+F) | **"闭式+逐像素贡献"先例**,必须划界(post-hoc 分解 vs 最优增益定义) | 中高(表面相似,实质不同) |
| Deno-IF | NeurIPS 2025;1.43M | 有监督噪声融合泛化差 | 卷积低秩闭式优化→蒸馏进 I2Former | **闭式解+教师+轻量三重近邻**;其闭式解教图像不教增益;"implicit teacher"话语须划界;**度量虚高引句**;轻量排名的对照点 | 高 |
| Dream-IF | AAAI 2026;参数量未报告 | 主导区指示另一模态待增强区 | RD=σ(Conv(F)) 自生成逐像素权重,无监督 | **per-pixel 门控的"无定义/无监督/无审计"反面教材**;其 Fig.1 三分法=我们三代范式分类的图示来源,必引 | 中高 |

**七篇的共同空白(=我们的三张独牌)**:① gate 的闭式可监督定义;② gate vs 最优增益的审计实验;③ 参数-性能-鲁棒三轴效率审计(Dream-IF 连参数量都没报)。

---

## 2. 竞争地形更新与故事修订(对 01 的修订指示)

### 2.1 增补四行差异句(并入 01 §2 表)

| 论文 | 我们的差异句(unlike 句式,可直接用) |
|---|---|
| ControlFusion | Unlike prompt-modulated restoration-fusion, degradation prompts can at best declare *where* degradation lies; the per-pixel decision of how much each transfer deserves to believe cannot be delegated to a global prompt — it must be calibrated, supervised, and audited. |
| DSPFusion | Unlike diffusion-restored semantic priors — coarse-grained by their own description, stochastic, and free of optimality guarantees — our teacher targets are closed-form, deterministic, and defined per transfer. |
| PIAFusion | Unlike illumination-aware weighting, which modulates a scalar between two modalities at training time, trust under degradation is spatially heterogeneous and must be resolved per pixel, per transfer, at inference time. |
| URFusion | Unlike feature-level escapes from image-level losses — which still leave every gate unsupervised — we do not move the constraints to another level; we give the decision rule itself a computable optimum. |

### 2.2 【关键修订】锋利钩子降位:URFusion 撞车处理

**事实**:URFusion(TIP 2025)Intro 已发表:(a) "Despite continuous evolution of network architecture, the loss functions (critical factor guiding network optimization) are all image-level constraints."(b) "the degradations will be mistakenly identified as critical information and retained in fused images."(c) "the degradations are also preserved or even enhanced during fusion."

**结论**:01 §1.2 的锋利档("the objective actively teaches the network to preserve and amplify the very degradation")**不能再作为我们的独家发现陈述**。修订为三层增量结构:

1. **承认已知的定性观察**(引 URFusion):损失会把退化误当关键信息保留甚至放大——已被指出;
2. **我们的增量一(机理分解)**:把它从定性观察推成可推导的逐项结构——梯度项取"更强局部梯度者胜出"而噪声携带全频最强梯度;max 强度项取"更亮者胜出"而散斑噪声是亮的;并配定量小实验(梯度能量对比:噪声图 vs 干净图;max 命中噪声像素比例)。**度量侧证据链现成**:DSPFusion "when images are affected by noise or rain, both SF and SD values may be artificially inflated" + Deno-IF "statistics-based metrics may produce misleading results as noise may artificially inflate these values"——指标端已被社区承认,以这些统计量为代理的损失端必然继承病理(此推理链是新的);
3. **我们的增量二(响应批评)**:已有响应都绕开了决策规则——ControlFusion/DSPFusion/AMG-Fuse **默默把 max 项换成干净源图计算**(隐性默认,无分析);URFusion **逃到特征级**但门仍无定义;Deno-IF 换成低秩分解目标但**仍用 max(Ly_v, Lr) 结构**。"换参考/换层级/换先验"都是回避,没有一家重新定义"该信多少"。

**修订后的锋利档段落(替换 01 §1.2,可直接用)**:

> Why do these paradigms all struggle? Recent work has observed that image-level fusion constraints mistake degradation for critical information and retain — or even amplify — it [URFusion]. We sharpen this observation into its mechanism: the gradient term of a standard fusion loss prefers the stronger local gradient, and noise carries the strongest gradients in the image; the max-intensity term prefers the brightest response, and shot-noise speckle is bright. The pathology is measurable: on degraded pairs, X% of max-intensity selections and Y× of gradient energy are attributable to noise rather than structure (Sec. 3.2). Moreover, the metric side of this pathology is already acknowledged — noise "artificially inflates" SF and SD [DSPFusion] and renders statistics-based metrics "misleading" [Deno-IF] — and losses built on the same statistics inherit it. Existing responses all sidestep the decision rule itself: silently recomputing max-terms on clean sources [ControlFusion, DSPFusion, AMG-Fuse], escaping to feature level [URFusion], or swapping in low-rank targets [Deno-IF] leave unchanged the question that matters at inference time: *how much of each cross-modal transfer should be believed here, now?*

(数字 X/Y 待 W1 实验填入;这正是"claim 无证据不写"军规的执行。)

### 2.3 划界弹药库(Related Work / rebuttal 直接用)

- **vs AMG-Fuse(审稿人最可能问:"它已有闭式贡献 mask")**:Theirs is a post-hoc decomposition of a teacher's output between two modalities, computed on clean sources — their own analysis shows it is numerically unstable on degraded inputs; ours defines, on degraded inputs, the optimal gain of each cross-modal transfer against a clean-path teacher, regressed onto every gate and auditable after training.
- **vs Deno-IF(闭式优化指导 + 教师双重相邻)**:Their closed-form updates supervise *what the clean image looks like* (image-level targets, strictly intra-modal, 30 patch-wise iterations per step); our closed-form solution supervises *how much to transfer* (gain-level targets, cross-modal, computed once per training step). Their "implicit teacher" teaches appearance; ours teaches allocation.
- **vs Dream-IF(RD 逐像素权重)**:Relative dominance is self-declared — σ(Conv(F)) of the very features whose degradation it is supposed to detect — constrained only to sum to one, with validation deferred to qualitative appendix visualizations. We define the target (closed-form gain), supervise it (regression), and audit it (gate–gain correlation, reliability maps).
- **vs Deno-IF 无监督侧翼(它批评有监督配对泛化差)**:We do not learn a degradation→clean regression mapping; the four-state balanced sampler controls the exposure distribution to calibrate allocation, and the teacher targets are self-generated by the network's own clean path — no external noisy/clean pairing of a restorer. We additionally cover stripe/stuck-point infrared degradations beyond Gaussian/speckle.
- **参数量措辞红线**:绝不写 smallest(TarDAL 0.3M、CDDFuse 1.19M、Deno-IF 1.43M 都更小)。标准句:with 1.63M parameters — comparable to clean-scenario lightweights while adding the degradation robustness they lack — and a single deterministic forward pass, vs 13.99M+10-step sampling [DSPFusion] and 182.54M with a frozen CLIP [ControlFusion]。

---

## 3. 论文骨架:标题、摘要、Introduction 逐段细纲

### 3.1 标题候选

1. **CalibFuse: Benefit-Calibrated Trust Allocation for Degradation-Robust Infrared-Visible Image Fusion**(安全,期刊风)
2. **Trust, Don't Just Restore: Closed-Form Supervision of Cross-Modal Transfers under Sensor Degradation**(故事强,稍险)
3. **How Much to Trust? Calibrating Every Cross-Modal Transfer in Degraded Image Fusion**(问句钩子)

期刊版建议 1(检索友好),副标题可留 story 词。关键词:trust allocation / benefit calibration / degradation-robust fusion / closed-form supervision。

### 3.2 Abstract 模板(八句式,仿七篇共同结构)

1. 背景一句:Infrared–visible image fusion assumes trustworthy inputs; real sensors violate this assumption(借 Deno-IF 传感器句)。
2. 问题一句:degradation corrupts trust heterogeneously across space and modality。
3. 现有+gap 一句:existing methods debate where restoration should happen (cascade / integrated / coupled), yet the decision rule — how much each cross-modal transfer deserves to be believed — remains implicit everywhere。
4. 机制观察一句(带 URFusion 让步):although image-level losses have been observed to retain degradation, the per-term mechanism and, more fundamentally, the missing definition of "beneficial transfer" have not been addressed。
5. We propose 一句:CalibFuse formalizes trust allocation: a clean-path EMA teacher yields a closed-form ridge-optimal gain for every cross-modal transfer, turning each gate from a trick into a regressed, auditable quantity。
6. 装备一句:per-modality dictionary recovery with reliability estimation, reliability-weighted gated interaction, curriculum-scheduled budgets, and four-state balanced training。
7. 结果一句(数字待填):consistently outperforms X methods across MSRS/M3FD/LLVIP under clean and four degraded states, while gate–gain correlation reaches r=…(审计数字)。
8. 卖点一句:1.63M parameters, single forward pass, full protocol and code released。

### 3.3 Introduction 逐段细纲(期刊版七段;每段给论点句+素材+借句)

**P1 现象与物理动机**(常识档,过外行测试)
- 论点句:IVIF is a trust-allocation decision at heart; sensors degrade trust heterogeneously。
- 素材:借 Deno-IF "prone to significant noise, stemming from degraded signal acquisition (especially in low-cost microbolometers and CMOS sensors)";一场景内一模态 pristine/另一模态 unusable(01 §1.1 原句)。
- 功能:不引任何争论,只立常识。

**P2 三代范式地图(立架构之靶)**
- 论点句:the community's response is a debate about where restoration should happen。
- 素材:cascade(借 URFusion "Due to the lack of coupling between restoration and fusion methods…",AMG-Fuse "error accumulation" 双指控)→ integrated hard regression(借 Dream-IF "directly merge the two tasks without exploring…";DSPFusion/Text-IF)→ mutual coupling(借 Dream-IF Fig.1 与 AMG-Fuse "paradigm gap" 措辞)。
- 分类句式借 PIAFusion:"According to where degradation is handled, existing methods can be roughly divided into three generations, i.e., …"。
- **必须引 Dream-IF 的 Fig.1 三分法作为分类法出处**(它已画了 (a) Joint Cascade (b) Direct Integration (c) Mutual Enhancement),我们的话术:三代之争共享同一个未问出的问题。

**P3 机制证据:决策规则失效(修订后的锋利档,§2.2 段落落位于此)**

**P4 重新框架:信任分配(01 §1.1 常识档主体搬运)**
- 论点句:no method estimates per-pixel reliability, states what a beneficial transfer is, or can tell after the fact whether the network trusted the right modality in the right place。
- 借句升级:PIAFusion 的换轴句 "the existing techniques focus more on how to fusion images/features but ignore when to fusion" → 我们:existing techniques focus on where restoration happens but ignore how much each transfer deserves trust。

**P5 为什么被忽略(三条结构性原因,01 §1.3 原文)**
- 基准文化 / 恢复视角 / 形式化看似不可得——第三条直接引出闭式增益。

**P6 我们提供什么(装配一口气,借 DSPFusion 的 Stage 目标句式)**
- 目标句(仿 DSPFusion "Stage I aims to…"):Our goal is to make every gate a quantity that can be defined, regressed, and audited。
- 三层校准 + 闭式增益 + 四态 + 课程,一段走完;强调"零件都已存在,缺的是按问题要求的装配"(00 §2.2 话术)。

**P7 贡献四条 + 结果一句**(00 §4 的 C1–C4,C1 头牌=问题+基准,C2=闭式增益监督,注意 DSPFusion 已用 "first to comprehensively address various degradations"、Deno-IF/URFusion 已用 "for the first time"——我们的 first 声明严格限定为:**the first to define and supervise the optimal gain of cross-modal transfers in degraded fusion**)

### 3.4 Related Work 组织(期刊三小节)

- **RW1 三代范式**(按 P2 顺序展开,每代 3-5 篇;ControlFusion/DSPFusion/Dream-IF/Text-IF/Deno-IF/AMG-Fuse 全部归位;收尾句借 AMG-Fuse "mainly focus on feature restoration and do not sufficiently explore the paradigm gap" 反转为 "…do not touch the decision rule inside fusion")。
- **RW2 损失与监督的已有响应**(PIAFusion max 规则公式 → ControlFusion/DSPFusion/AMG-Fuse 换干净参考 → URFusion 特征级逃逸 → Deno-IF 低秩教师;每家一句话划界,弹药见 §2.3;收尾:all sidestep the decision rule)。
- **RW3 门控/权重作为信任的雏形**(PIAFusion 标量光照权重 → DSPFusion 通道级 PGFM → Dream-IF 逐像素 RD → AMG-Fuse 贡献 mask;收尾句:in all of them the gate is a trick — employed, never defined;we make it a supervised, auditable quantity)。

### 3.5 Method 叙事规则

- 开场**先形式化后总览图**(借 Deno-IF/URFusion 的公理化开场,不用 Dream-IF 的图先行):3.1 Problem Formulation 放退化观测模型 + 信任分配形式化(哪个量是"信任");3.2 损失病理分析(§2.2 的机理分解+定量小实验,期刊版给足版面——这是 C1 的证据主场)。
- 每模块统一微叙事:**设计问题 → 机制 → 为什么替代装配不行**(AMG-Fuse 3.1 的"闭式解+失败模式分析"是模板:先给朴素式,再分析数值不稳定,再给修补式——我们写 ridge 增益时照此办理:为什么需要 ε 正则、为什么 clip 到 [0, u]、单步贪心的边界)。
- 每加一项监督配**反事实句**(借 URFusion:"As a single similarity loss will lead to trivial solutions, …")——样例:Without a regressable target, a gate degenerates into an unmoored trick; the closed-form gain turns it into an auditable quantity。
- 模块顺序按问题链(00 §3):模态内校准 → 模态间校准 → benefit 凭什么可信(教师+闭式推导,W2 焊点在此)→ 训练协议。

### 3.6 Experiments 组织(每张表/图先说验证哪条 claim)

| 小节 | 内容 | 验证 |
|---|---|---|
| 5.1 协议 | 数据出处(MSRS/M3FD/LLVIP 正式引用+test_noise 生成设置文档化)、四态退化协议、统一复评声明、指标双协议(社区标准主表+内部协议附录交叉验证) | C1/C3 可信度 |
| 5.2 干净主表 | 10 方法对比 | W4 焊点(不掉点) |
| 5.3 退化主表+分状态表+severity 扫描 | 四态×方法;借 Deno-IF 分噪声等级表格式 | C1/C3 |
| 5.4 **审计实验(差异化核心)** | gate vs 最优增益散点/相关;可靠性热图 vs 噪声图;harmful-correction-rate;噪声区门自动关断统计 | C2(W3 焊点) |
| 5.5 消融 | ≥8 行:三层机制各去/换档+teacher/curriculum/四态 | C2 |
| 5.6 效率 | 参数/FLOPs/fps 表+气泡图(借 Deno-IF Fig.1 图式:1.63M 小气泡对 13.99M/59.74M/182.54M) | C4 |
| 5.7 下游+真实退化(加分) | YOLO 检测表(借 Dream-IF/AMG-Fuse);真实退化样本回应合成质疑 | 泛化性 |

结果句式模板(借 AMG-Fuse/Dream-IF 的"数字→机制归因"):Notably, the largest gains concentrate in the dual-degradation state (Tab. X, +Y% over sub-optimal), attributed to gates that close where reliability collapses — a behavior no baseline exhibits。注意 AG/SF 高≠干净(配噪声侧指标交叉,防"指标虚高"反噬——这正是我们批评别人的,自己必须免疫)。

---

## 4. 句子库(按用途;原句出处 → 化用方向)

**A. 立靶/gap**
- "a crucial issue exists, i.e., the illumination imbalance has never been investigated."(PIAFusion)→ …i.e., how much each cross-modal transfer should be trusted has never been explicitly defined。
- "Despite continuous evolution of network architecture, the loss functions (critical factor guiding network optimization) are all image-level constraints."(URFusion)→ 引用后推进:the supervision itself, not the architecture, is where robustness is lost。
- "these methods purely aim to preserve the original information in source images without distinguishing the information quality."(Deno-IF)→ 接:but "quality" was never given a supervisable definition。
- "existing methods lack degradation level modeling, causing a sharp decline in performance as degradation intensifies."(ControlFusion)→ 改:what collapses with severity is not capacity but calibration。

**B. 重新框架/换轴**
- "the existing techniques focus more on how to fusion images/features but ignore when to fusion images/features."(PIAFusion)→ focus on where restoration happens but ignore how much to trust each transfer。
- "We assume that various degradations can be categorized into two types…"(URFusion)→ 假设句模板:We posit that degradation does not destroy pixels so much as the trust owed to them, and that this loss of trust is heterogeneous across space and modality。
- "fits static distributions of pseudo targets rather than learning the dynamic mechanisms of IVIF."(AMG-Fuse)→ hard-regressing a static reference teaches appearance, not allocation。

**C. unlike/划界**(§2.1、§2.3 已给五条完整弹药)

**D. 贡献/首创**(注意 first 措辞红线:只限 closed-form gain supervision)
- "To our knowledge, it is the first model that comprehensively addresses various degradations in image fusion."(DSPFusion)→ 对照式引用:degradation coverage has been claimed; what remains unclaimed is the definability of benefit。

**E. 结果归因**
- "improves 3.67%, 3.86% and 3.56% over the sub-optimal method in snow, rain, and haze scenes."(AMG-Fuse)→ 分状态百分比句式。
- "our method shows smaller fluctuations, showing its robustness."(Deno-IF)→ 跨状态方差句式,配四态均衡采样。

**F. 防守/让步/诚实装置**
- "In real-world scenarios, directly subtracting infrared images from visible images can lead to misleading or unstable behaviour."(AMG-Fuse)→ 闭式解失败模式分析模板(W2 用)。
- "It is worth noting that our model does not take any prior knowledge of the degradation type."(Dream-IF)→ 盲设定声明:our network receives no degradation label, state flag, or prompt; the four-state sampler supervises allocation, it does not inform the model。
- "While standalone fusion models … appear more lightweight, they necessitate a separate, computationally intensive restoration stage… When the overhead of this two-stage process is factored in…"(ControlFusion)→ 总账效率辩护,反用:级联总账 vs 我们单前馈。

---

## 5. 图表预算(期刊版:8 图 6 表;会议版砍半)

| 编号 | 内容 | 备注 |
|---|---|---|
| Fig.1 | 动机图:三代范式地图(引 Dream-IF Fig.1 式)+ 右下 gate/可靠性热图(噪声区门自动关断) | 一图三职:问题存在+现有失效+方向有效(00 §1.3) |
| Fig.2 | 架构总览(三尺度+三层校准) | 通栏 |
| Fig.3/4 | 定性对比:干净/退化(Dream-IF 版式:红绿双框+zoom+每数据集一行) | 三件套规范(00 §7) |
| Fig.5 | **审计图组**:gate–gain 散点+可靠性热图 vs 噪声真值+harmful-rate 曲线 | 差异化核心,放正文(对 Dream-IF 把 RD 推附录的超越声明) |
| Fig.6 | severity 扫描曲线 | 回应 ControlFusion Fig.1(III) |
| Fig.7 | 参数-性能气泡图(Deno-IF Fig.1 式) | 1.63M 小气泡 |
| Fig.8 | 失败案例(1-2 个) | 自信信号 |
| Tab.1/2 | 干净/退化主表(10 方法×7 指标) | 最优加粗次优下划线 |
| Tab.3 | 分退化状态表 | |
| Tab.4 | 消融 ≥8 行 | |
| Tab.5 | 效率表(参数/FLOPs/fps) | |
| Tab.6 | (可选)下游检测 | |

---

## 6. 红队问答(投稿前逐条自答,P5 阶段模拟审稿)

1. **"损失奖励噪声 URFusion 已经说过"** → §2.2 三层增量:定性已发表(引)、机理分解+定量是我们的、且指出所有已有响应(换参考/逃特征级/换先验)绕开决策规则。
2. **"AMG-Fuse 已闭式量化贡献 mask"** → §2.3 弹药:post-hoc 干净域分解 vs 退化输入上的最优增益定义;其自身 3.1 节承认退化域数值不稳。
3. **"Deno-IF 已用闭式优化解指导融合网络"** → 图像级 vs 增益级;intra-modal vs 跨模态;逐 patch 30 迭代 vs 每步一次。
4. **"Dream-IF 已做逐像素跨模态权重"** → 自生成无监督(σ(Conv(F)) + 求和归一)vs 闭式目标回归+审计;其验证只有附录定性图。
5. **"合成退化的泛化性"** → 真实退化样本(加分项)+ 物理噪声模型出处 + Deno-IF 无监督侧翼的防守(§2.3)。
6. **"1.63M 不是最小"** → 效率-鲁棒位置措辞红线(§2.3 末条);URFusion Table X 数值投稿前核对(提取文本缺失)。
7. **"为什么需要干净参考(非无监督)"** → 我们不学恢复映射,教师由网络自身干净路径 EMA 自生成,无外部恢复器配对;无监督与有监督的折中点是 Limitations 承认项。
8. **"ridge 增益是单步贪心,不是全局最优"** → W2:明确声明定义域(给定方向的一维最优)+ g_max 结构上限的保守性;Limitations 收录。
9. **"指标协议可比性"** → 双协议声明:主表社区标准统一复算,内部协议附录交叉;绝不与文献自报数字同表(Dream-IF 的 SSIM>1/表头错置是反面教材,引用其数字只引相对结论)。

---

## 7. 行动清单(优先级排序;并入 00 §6/§8 的更新)

1. **【P0,科学风险最高】跑 W3 审计实验**:现有 ckpt 上计算 gate vs 最优增益相关(推理期可算:optimal_gain 需要 teacher 特征,可在测试集干净对上离线复现)。r 显著→核心 claim 立;不显著→立即复盘,这决定论文命运,先于一切写作;
2. **【P0】W1 定量小实验**:梯度能量(噪声/干净)+ max 命中噪声像素比例,填 §2.2 的 X/Y;
3. **【P0-P1】基线统一复跑**(清单见 00 §6;新增注意:URFusion 参数量需从原 PDF Table X 核对;DSPFusion 需核实正式 venue 后引用);
4. **【P1】按 §2.2 修订 01 的问题陈述,按 §3 写 Intro 初稿+摘要 v1**;
5. 【P1】数据出处文档(MSRS/M3FD/LLVIP + test_noise 生成设置);
6. 【P2】效率数据(GPU fps;CPU 40s/张不进论文)+ 气泡图;
7. 【P2-3】真实退化样本、下游检测表(加分,期刊版建议都做)。

venue 决策(§0.2)执行影响:按 Information Fusion 格式准备(无页限、图表可全展开);会议 8 页版本作为可拆卸子集保留(Fig.1/2/5 + Tab.2/4 是不可砍核心,统一复评与 severity 可移附录)。
