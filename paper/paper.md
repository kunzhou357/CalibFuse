# CalibFuse: Benefit-Calibrated Trust Allocation for Degradation-Robust Infrared–Visible Image Fusion

> **草稿 v0.1(2026-10-07)**:摘要 + Introduction,供参考与修改,非定稿。
> 结构依据 `paper/02-writing-guide.md` §3;段落标注为【功能|素材/借句来源】。
> 待填数字一律用 `[..]` 标出,集中在文末"待填清单"。术语全文统一:
> trust allocation / cross-modal transfer / benefit / reliability / gate /
> clean-path teacher / closed-form ridge-optimal gain / four-state balanced training。

---

## Abstract

Infrared–visible image fusion (IVIF) silently presumes that both sensors deserve trust. On real hardware the presumption fails: low-light CMOS frames carry shot and read noise, uncooled infrared arrays add striping, stuck pixels, and Gaussian fluctuations, so a single scene routinely contains regions where one modality is nearly pristine and the other nearly unusable. Existing degradation-robust fusion methods are organized around *where* restoration should happen — before fusion, inside a monolithic network, or through mutually coupled modules — yet across all three generations the underlying decision rule remains implicit: no method defines per-pixel reliability against a reference, states what a *beneficial* cross-modal transfer is, or can verify, after the fact, whether the network trusted the right modality in the right place. We argue the failure runs deeper than architecture: the standard fusion objective actively rewards degradation — its gradient term prefers the strongest local gradients, which noise carries, and its max-intensity term prefers the brightest response, which speckle provides — so every response that merely relocates restoration leaves the rewarding of noise intact. CalibFuse instead formalizes trust allocation. A clean-path EMA teacher turns the benefit of every cross-modal transfer into a ridge regression whose optimal gain is available in closed form, making each gate in the network defined, regressed, and auditable; per-modality dictionary recovery supplies reliability estimates, reliability-weighted gated interaction spends them, and curriculum-scheduled budgets with four-state balanced training stabilize optimization. Across MSRS, M3FD, and LLVIP under clean and four degraded states, CalibFuse outperforms [N] baselines by [..] while remaining competitive on clean data, and its learned gates track their closed-form optima (r = [..]) — with 1.63M parameters in a single deterministic forward pass. Code, protocols, and evaluation suites are released.

> 【摘要|八句式:背景→gap→机制→方案→装配→结果→轻量→开源。机制句已按 02 §2.2 做 URFusion 让步("actively rewards"是我们的机理增量,未声称首次发现"保留退化")。】

---

## 1. Introduction

**[P1|现象与物理动机|借句:Deno-IF 传感器噪声;01 §1.1 素材;过外行测试]**

Infrared–visible image fusion (IVIF) combines the complementary strengths of two sensors — thermal radiation, which penetrates darkness and glare, and reflected light, which carries texture and color — into a single image for detection, tracking, and scene understanding. The task is, at its core, a trust-allocation decision: at every pixel, the network must decide how much of each modality to believe. Real sensors corrupt this decision heterogeneously across space and modality. Low-cost microbolometer and CMOS arrays, together with the embedded processing pipelines behind them, inject significant noise into both channels; low-light visible frames suffer shot and read noise, while uncooled infrared sensors add striping, stuck pixels, and Gaussian fluctuations. Degradation is therefore not a global property of a scene but a local property of a sensor: a single frame routinely contains regions where one modality is nearly pristine and the other nearly unusable.

**[P2|三代范式立靶|分类句借 PIAFusion 换轴;级联批评借 URFusion/AMG-Fuse;必引 Dream-IF Fig.1]**

The community's response has been an increasingly sophisticated debate about *where* degradation should be removed. Restoration-then-fusion cascades delegate cleanliness to an upstream denoiser; yet a restorer optimized for intra-modal fidelity knows nothing of cross-modal complementarity, and its residual artifacts propagate into fusion — an error accumulation that a downstream network cannot repair. Single-network methods fold restoration and fusion into one end-to-end regression; they remove the seam, at the price of heavy models and an even denser black box. The newest generation couples the two tasks through mutually reinforcing modules, letting one modality's strength indicate where the other needs enhancement. Across all three generations, however, one thing never changes: the decision rule that governs how much cross-modal information is believed. It is implicit everywhere — buried in attention weights, gating values, or fusion coefficients that no method defines, supervises, or examines.

**[P3|机制证据:决策规则失效|修订后锋利档(02 §2.2);URFusion 让步+机理分解+度量侧证据链+已有响应绕开;X/Y 待 W1 实验]**

Why does an implicit rule fail under degradation? Because the objectives that made fusion work on clean data are precisely the ones that reward it. Recent work has observed that image-level fusion constraints mistake degradation for critical information and retain — or even amplify — it. We sharpen this observation into its mechanism. The gradient term of a standard fusion loss prefers the stronger local gradient, and noise carries the strongest gradients in the image; the max-intensity term prefers the brightest response, and shot-noise speckle is bright. The pathology is measurable rather than rhetorical: on degraded pairs, [X]% of max-intensity selections and [Y]× of gradient energy are attributable to noise rather than structure (Sec. 3.2). Its reflection on the metric side is already acknowledged — noise "artificially inflates" SF and SD, and statistics-based metrics become "misleading" under noise — and losses built on the same statistics inherit the bias. Existing responses sidestep the rule itself: silently recomputing max-terms on clean sources, escaping to feature-level constraints, or substituting low-rank targets all relocate the supervision without ever asking, at each pixel and each transfer, *how much should be believed*.

**[P4|重新框架:信任分配|01 §1.1 收尾+换轴句(PIAFusion 式);段尾是全文枢纽:benefit 看似不可得→训练期可得]**

We reframe the problem. Degradation-robust fusion is not an architecture question but a trust-allocation question: degradation destroys not pixels but the trust owed to them, and that loss of trust varies independently across space and modality. Where prior efforts ask *where restoration should happen*, we ask *how much each cross-modal transfer deserves to be believed*. The distinction is consequential. A well-placed restorer still cannot state whether the cleaned signal it hands over deserves more confidence than the rival modality's raw one; a tightly coupled system still cannot reveal, after the fact, which modality it trusted at the pixels that matter. Making the allocation explicit requires three things the field has not supplied together: per-pixel reliability estimates, a gate on every information transfer, and — crucially — a computable definition of what a *beneficial* transfer is. The first two are standard parts. The third has seemed unformalizable: quantifying benefit appears to require a reference that does not exist at test time. It is unavailable at test time; we show it is available at training time.

**[P5|为什么被忽略:三条结构性原因|01 §1.3 原文;第三条直通闭式贡献;绝不暗示前人傻]**

Three structural reasons kept this question unasked. First, benchmark culture: the field's datasets and leaderboards are built on clean, registered pairs, so degraded inputs fall outside the evaluation loop entirely. Second, the restoration lens: once degradation is seen as a *restoration* problem, the natural questions become architectural — where to place the denoiser, how tightly to couple it — and the decision rule inside fusion stays out of sight. Third, and most deeply, trust allocation looked unformalizable: stating how much a transfer should be believed seems to demand a ground truth for benefit that is unavailable at test time. This is precisely the gap our method closes.

**[P6|我们提供什么:装配一口气|借 DSPFusion 目标句式;所有零件大方引用;闭式增益是收束句]**

CalibFuse makes every gate a quantity that is defined, regressed, and audited. The key is a clean-path teacher: an exponential-moving-average copy of the network that encodes the clean counterparts of each training pair. Along any message direction the network proposes, the gain that best moves a degraded feature toward the teacher's clean feature solves a one-dimensional ridge problem in closed form — so the optimal gain of every cross-modal transfer becomes a per-pixel regression target, and gates stop being tricks. Around this supervision we assemble the parts the problem demands, each borrowed openly: per-modality dictionary recovery with reliability estimation calibrates what each modality contributes; reliability-weighted gated interaction lets information flow only when the source is reliable and the predicted benefit is high; curriculum-scheduled cross-modal budgets let recovery mature before interaction opens; and four-state balanced training — clean, either side degraded, both degraded — keeps the loss statistics of all quality regimes unbiased.

**[P7|贡献四条+结果句|00 §4 C1–C4;first 措辞红线:只限 closed-form gain supervision;数字待填]**

Our contributions are fourfold. **(i)** We identify and empirically ground the missing decision rule: standard fusion losses reward degradation — a mechanism we decompose term by term and quantify — and we release a physics-grounded degradation protocol with four-state balanced training. **(ii)** We give benefit a closed-form, supervisable definition: the ridge-optimal gain of every cross-modal transfer under a clean-path EMA teacher — to our knowledge, the first supervision of fusion gates against computable optima — together with the calibrated assembly this demands. **(iii)** We re-evaluate [10] representative methods across MSRS, M3FD, and LLVIP under clean and degraded conditions in a unified protocol, adding mechanism-level ablations and gate-behavior audits: gate–optimal-gain correlation, reliability maps versus injected degradation, and harmful-correction rates. **(iv)** CalibFuse sets the state of the art under degradation while remaining competitive on clean inputs, improving [metric] by [..]% over the sub-optimal method — with 1.63M parameters in a single deterministic forward pass, orders of magnitude lighter than prompt- or diffusion-guided restoration-fusion pipelines. Code, protocols, and evaluation suites are publicly available.

---

## 附:待填清单与讨论点

**待实验数字(W1/W3/基线复跑,见 02 §7):**
- [ ] P3 的 `[X]%`、`[Y]×`:损失病理定量小实验;
- [ ] 摘要/P7 的 `[N]`、`[..]%`、`r = [..]`:基线数、主表提升、gate–增益相关(W3,最高科学风险,最先跑);
- [ ] P7 `[metric]`:主指标选择(需社区标准协议定稿后回填)。

**写作留给你定夺的口子(下次讨论):**
1. P2 的三代范式是否点名引文(Text-IF/ControlFusion/DSPFusion/Dream-IF)还是留到 Related Work——期刊版建议 Intro 只泛指+少量点名;
2. P3 的让步句是否要在这里就点名 URFusion,还是同 P2 一样留到 Related Work(目前的写法是"Recent work has observed…"不点名,引用挂句尾);
3. P6 "gates stop being tricks" 口语化程度——是保留锋利还是学术化("gates cease to be heuristic devices");
4. 标题沿用候选 1,还是换问句式候选 3;
5. 摘要最后一句 "Code… released" 依赖真实开源计划,若不开源删。
