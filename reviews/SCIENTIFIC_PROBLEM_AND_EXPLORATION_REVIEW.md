# 科学问题—科学探索审查：从受控反例到可用的 MLIP 选择方法

> 审查对象：仓库 `C-Xiyuan/testing`，提交 `799277df71fd0abe4e619251b32db1da86af85f9`（包括在审查期间新增的 window-sweep 分析）
>
> 审查原则：科学问题不是一个题目、公式或已有现象，而是学界尚未解除、持续阻碍可靠推断的困境；科学探索不是把代码和图表堆到一起，而是针对该困境建立完整的解决链，并对所有足以改变结论的环节设置独立、可重复、可证伪的实验。
>
> 本审查只保留影响科学结论的事项。一般代码风格、仓库命名和排版问题不在范围内。

## 1. 执行结论

当前工作可靠建立了一个窄而有价值的结论：

> 在一个 Lennard–Jones 液体状态、一个预选 pair-count observable 和一个 pair-radial Gaussian basis 中，可以构造 held-out force RMSE 匹配、但该 observable 的直接采样偏移显著不同的 null/aligned error fields。

这是一项**受控存在性证明**。它说明 force RMSE 不是该特定 observable 误差的充分统计量。它没有证明以下三个更强命题：

1. 训练产生的真实 MLIP error fields 会稳定、普遍地占据这种方向；
2. `force error is a coarse filter, not a selector` 是现实模型选择中的一般规律；
3. response vector 在无 oracle、有限参考预算的应用中能比 force RMSE 更可靠地选择模型。

因此，当前项目的证据等级应当是 **controlled mechanism proof-of-concept**，而不是 validated selector 或通用 MLIP validation method。

最关键的修改不是再写一轮讨论，而是重构科学问题：

> **在无法获得可遍历真值势、参考计算预算有限、候选模型可能共享系统误差且 reweighting overlap 有限的条件下，能否以低于逐模型端到端模拟的成本，对“体系 × 状态点 × observable”给出经过校准的误差预测，并实际降低模型选择的错误率与 regret？该方法的失效边界能否在使用前被识别？**

这是学界仍被困住的部分。单纯说明 force RMSE 与下游物理可能不一致，已经不是未解问题：相关 benchmark 和 validation literature 已直接展示这种脱钩，并建议按物性验证模型。[Fu et al. 2023](https://arxiv.org/abs/2210.07237)、[Morrow et al. 2023](https://pubmed.ncbi.nlm.nih.gov/37003727/)、[Liu et al. 2023](https://arxiv.org/abs/2306.11639/)。

## 2. 科学问题地图

| 要素 | 审查结论 |
|---|---|
| 研究类型 | 当前是探索/机制型研究；稿件却部分使用了方法验证和选模建议的语言。 |
| 作者陈述的困境 | held-out force RMSE 被用于选模，但它度量 `∇δU` 的范数，而静态 observable 的一阶偏移取决于 `Cov(A, δU)`；因此小 force error 不保证可靠物理。见 `paper/main.md:40–69`。 |
| 已经被文献解决的部分 | “force error 与 simulation outcome 并非总一致”及“应做 property-level validation”已有充分先例；标准 reweighting/covariance identity 也不是新理论。稿件已在 `paper/main.md:71–83, 867–929` 部分承认这一点。 |
| 真正未解决的困境 | 在无 oracle、有限预算、共享系统偏差、有限 overlap 的现实条件下，如何给出 observable-specific、可校准、可迁移且能改善决策的廉价诊断。 |
| 为什么重要 | MLIP 正被当作 first-principles 的替代品运行有限温结构与动力学模拟。Dyna-Mat 表明较低 force error 在平均意义上有用，但个别体系仍会在低 force error 下发生定性结构失败；真正的问题是识别这些例外，而不是宣布 RMSE 普遍无用。[Dyna-Mat 2026](https://arxiv.org/abs/2607.03433) |
| 为什么困难 | `δU` 在真实应用中不可遍历；observable 依赖模型诱导的分布或传播子；committee 会漏掉 shared bias；reweighting 会受 overlap、相关样本 ESS 和非线性限制；效应还依赖体系、状态、尺寸、ensemble 和 observable。 |
| 解决标准 | 新方法必须在预注册的真实选模任务上，相对 force-RMSE baseline 降低 false acceptance、failure rate 或 regret，并给出经过验证的 coverage 与失效报警；相关系数本身不是解决标准。 |

### 2.1 需要纠正的理论表述

`paper/main.md:182–192` 将

```text
beta * sigma(A) * sigma(deltaU) * |rho(A, deltaU)|
```

称为 Cauchy–Schwarz bound。在线性响应层面，含 `rho` 的表达式只是 `Cov = rho * sigma * sigma` 的恒等改写；真正的 Cauchy–Schwarz 上界不含 `rho`。若该式用于完整 observable shift，还必须显式保留高阶余项。这不是排版问题，而是理论链的边界条件，必须修正。

同样，`paper/main.md:565–583` 附近关于“任何 scalar 都不可能总结模型质量”的结论过强。本研究只否定了未指定任务时 force RMSE 对特定 observable 的充分性；一个带有明确效用函数的 minimax risk、failure probability 或多目标加权标量并没有被排除。可守的结论是：**不存在一个不依赖任务定义、却保证所有 observables 同序的通用 scalar。**

## 3. 作者当前的解决链，以及每一环必须通过的实验

当前推理链可以重建为：

```text
force RMSE 与 observable response 测量不同对象
    -> 可按 covariance 方向设计 null/aligned error fields
    -> 在相同 force RMSE 下产生不同直接 observable shift
    -> response estimator 能预测这种 shift
    -> 真实 fitted models 也表现相同
    -> response vector 可用于现实 model selection
```

前两步得到了一定支持；后三步没有闭合。以下环节中，任何一个失败都会从根本上改变论文结论。

| 关键环节 | 必须成立的事实 | 当前证据 | 仍存替代解释 | 判定 |
|---|---|---|---|---|
| 数学机制 | 小扰动下 direct shift 与 response expansion 一致 | 标准理论；LJ 设计场和 fitted arm 有表面上约 `1 sigma` 的 RMS residual | 共同 reference-chain 偏移、遗漏 prediction uncertainty、有限 overlap | 理论成立；校准未完成 |
| 构造性反例 | null/aligned 差异能跨 construction chain、direct chain、basis 和 target 稳定复现 | exp07 单一 construction/evaluation split；两个 force levels 实为同方向缩放 | 偶然的 construction trajectory、basis/target 特异性、共同随机流 | 单设置内支持；稳健性不足 |
| estimator 校准 | 预测不仅相关，而且无系统偏差、区间 coverage 正确 | 稿件报告 designed `1.01 sigma`、fitted `1.06 sigma` | `results/revision_statistics.json` 显示 designed arm 8/8 residual 同号，common offset `-0.861 sigma`；另有 35%/3.7σ direct–reweight discrepancy | 未通过 |
| warning light | 在未知样本上能识别 response 何时不可用 | 二阶项 heuristic；部分幅度 sweep | `results/exp06_response_validation/summary.json` 明确 `flagged_cases_all_agree=false`；两个 null 点二阶修正均令 agreement 变差；最大 breakdown arm 没有 diagnostic | 未验证，已有 false-trust case |
| selector 规律 | 在真实且相近的 fitted candidates 中，force RMSE 选模失败而 response 更好 | 34 个 designed surrogates 的 post-hoc band | field-family cluster、共同 reference/random stream、noise-floor denominator、看数据选 band | 不支持现实 selector 结论 |
| fitted-error 转移 | 机制在训练产生的 error manifold 中稳定存在 | 同一 LJ、同一 observable 的 10 个小模型；所有 ranking CI 跨零 | 数据集/架构/seed 特异性；多数 truth 在 noise floor | feasibility only |
| 无-oracle 实用性 | committee/cheap reference 能近似 oracle score 且能报警 shared bias | 仅提出 homogeneous/heterogeneous committee 设想 | committee mean 抵消共同偏差；相关工作已有更成熟 reweighting/experimental calibration | 未实验 |

## 4. 对根本结论最重要的稳健性问题

### P0-1：存在两个未解决的端到端一致性失败；在此之前不得称 estimator “validated”

`paper/main.md:757–803` 报告 exact reweighting 与 direct sampling 相差 35%（3.7σ）。用于排除 error-bar 问题的六链实验测的是另一个 bin，因此不能证明 headline bin 的偏差来源已被排除；“incomplete relaxation 必然只向零偏”也不是充分论证。

`paper/main.md:839–863` 的 dilute check 在一个有限密度点与 hand-derived result 相差 RMS 4.7σ、最差 6.5σ。这里比较的是允许 `O(rho)` 误差的低密度近似，而非 exact identity；因此它的证据等级低于前述 35% discrepancy。它说明当前 dilute benchmark **尚未验证**，并不单独证明 estimator 失效。若要用 `O(rho)` 解释差异，必须做 density series 或精确的 `N=2` benchmark。

前一个冲突是必须解决的端到端 release blocker：exact identity、direct ensemble 和误差条的一致性是整个 response predictor 的 measurement gate。后一个问题是未通过的 benchmark，不应被写成验证证据，也不应与 exact-identity 冲突合并推断。

**最低修复实验：**

- 同时测 4.25 Å 和 5.25 Å bins；至少 8 条独立 reference chains 与 8 条独立 surrogate chains；
- 用 HMC 和 single-particle Metropolis 两个不共享推进机制的 sampler；
- 比较 forward FEP、reverse FEP、BAR/MBAR 和 direct difference；
- 报告 autocorrelation-aware ESS、最大权重、prediction/direct covariance；
- 预注册通过标准：先定义 direct–reweighting difference 的 estimand 和具有物理意义的等价界；使用传播 shared-reference covariance 的 paired interval，并在预设 family-wise error 下完成等价性检验。chain 数由 pilot variance 和目标区间宽度反推，而不是以“95% CI 看起来重叠”为通过标准；否则撤回 “validated cheap predictor”。

### P0-2：`1.01 sigma` 不是 calibration pass，最新补算反而提示共同系统偏移

`scripts/compute_revision_statistics.py:11–18` 已承认同一 arm 中所有 residual 共享 reference-chain error。`results/revision_statistics.json` 进一步显示 exp07 designed arm 的 8 个 residual 全为负，报告的共同 offset 为 `-0.861 sigma`；预算 40 的 fitted arm同样 8/8 为负。由于这些 residual 共享 reference realization 且 fields 也不是独立抽样，8/8 同号只能作为共同偏移的警报，不能按 8 次独立 Bernoulli trials 进行 sign-test 推断。

因此，RMS residual 接近 1 可能由共同偏移加较小散布组成，而不是 prediction intervals 校准良好。当前 denominator 也主要使用 measurement SE，没有完整传播 prediction uncertainty 及其与共同 reference 的 covariance。

**所需分析：**以独立 chain 为实验单位，用 joint paired/block bootstrap 或层级模型估计 intercept、slope、coverage 和 common-offset distribution。不能继续把同一 reference realization 下的多个 fields 当作独立 calibration cases。

### P0-3：exp07 支持“反例存在”，但尚不支持“反例稳定”

当前 point estimate 很大：在 `4e-3 eV/A` 组中 aligned–null 为 10.111 pairs；按现有表格误差朴素合并约 11.43σ。但两条 direct runs 使用共同随机流，其 covariance 未估计；两个 force levels 又是相同 field direction 的缩放，不是独立 field replicates。null-space generalisation 使用同一 construction chain 的 nested prefixes 和同一 evaluation half，也不是 construction-level replication。

**最低稳健性实验：**

- 至少 8 条独立 construction trajectories；
- 每条 construction 对应完全独立的 evaluation trajectory；
- 每个 field 至少 4 条独立 direct chains；
- 至少 3 个预注册 targets、2 种 basis、2 个 system sizes；
- 独立 construction/reference cluster 是实验单位；共同随机数可作为预注册 paired design，但必须估计 covariance 并对 aligned–null paired contrast 推断；
- 先定义 aligned–null contrast、force-RMSE equivalence margin 与最小有意义 observable effect；以 pilot cluster variance 做功效/区间精度设计，在 multiplicity-adjusted equivalence test 通过后再检验 paired observable contrast。不能用任意的“≥80%”或未经依据的 `1%` 当作稳定性证明。

如果只在当前 cell 通过，结论必须保持为 “single-state constructive example”。

### P0-4：34-member zoo 的 bootstrap 不能支撑模型总体的置信区间

`experiments/exp05_proxy_correlation/run.py:177–183` 直接声明 members independent，并对 rows 做 ordinary bootstrap。但 34 个成员是若干人为参数族：18 个 shell points 来自同一个 factorial family，多个 amplitude points 只是相同形状缩放；所有 members 共享 reference data、baseline 和随机流。它们可以被当作一个**固定有限 zoo 的描述量**，不能被当作来自现实 MLIP 总体的 34 个 IID draws。

此外：

- band 在看数据后选择；
- 15 个点中 6 个 force RMSE 完全相同；
- `30x` 的分母 `null_f4e-03` 没有离开噪声地板；
- 最新补算给出 raw spread 12.05×，去掉两个 designed fields 后为 6.51×；
- member bootstrap 没有传播每个 measured norm 的 Monte Carlo uncertainty，noise subtraction 后的 flooring 会进一步扰动低端排序。

因此，`force error is a coarse filter, not a selector` 目前是一个后验 hypothesis，而不是结论。应将其改写为：

> 在当前人工 zoo 的一个后验 force-error window 中，force RMSE 未能解析 observable-error ordering；是否适用于训练得到的相近模型，需要预注册的 selection experiment。

提交 `799277d` 新增了对多个 multiplicative windows 的 sweep，并诚实承认原 headline window 位于同宽窗口的第 16 百分位。这降低了只报告单一有利窗口的呈现偏差，但**没有修复推断问题**：所有窗口仍反复使用同一组 34 个 clustered designed fields、同一 reference realization 和同一 noise-subtracted/floored truth；高度重叠的窗口也不是新的独立重复。该 sweep 可以作为固定 zoo 上的敏感性描述，不能把一个后验、相关数据集转换成现实 fitted-model selection 的外部证据。因此最新提交仍不足以支持保留 `coarse filter, not a selector` 为一般结论。

### P0-5：二阶 warning light 有明确假阴性，必须从“结果”降为待验证 heuristic

`results/exp06_response_validation/summary.json` 中两个 cases 被标为 trustworthy，但第一个并不与 direct sampling 一致，所以 `flagged_cases_all_agree=false`。最新 revision statistics 还显示两个 null fields 加入二阶项后 agreement 均变差。最大 response breakdown arm 又没有保存 diagnostic。

正确验证方式不是展示几条曲线，而是在 development set 上冻结 threshold，再在 held-out shapes、states 和 fitted models 上测：

- sensitivity/specificity；
- false-trust rate；
- prediction-interval coverage；
- 不同 overlap/ESS 区间的 calibration curve。

若 false-trust rate 的上置信界不能低于预设容忍值（建议 5–10%），该项不能作为安全报警器。

### P1-1：fitted-model arm 不是 external validation，更不支持 selector claim

`paper/main.md:654–677` 的 10 个模型共享 LJ system、observable、train pool、test/reference chain 和代码路径；神经网络每种架构只有两个 seeds。所有 ranking intervals 跨零，多数 direct truth 在 noise floor。该实验最多说明 response calculation 可以接触 training-induced errors；不能称 external validity pass。

EGNN 两 seed 的 27× anecdote 是 `n=2`，应作为生成新假设的观察，而不是现实选模规律。

### P1-2：仓库把“实现能力”误写成“证据范围”

README 列出 LJ/SW/EAM、RDF/phonon/diffusion/elastic/melting，以及 exp01–exp08；当前论文却明确承认 SW/EAM 没有 observable-error result、没有 dynamics、没有 finite-size，也没有 published MLIP/DFT。空的 `atomlab/training/__init__.py` 和缺失的 exp01/02/04/08 进一步说明 roadmap 被写成了完成清单。

这会直接扭曲科学问题的完成度判断。README 必须区分 `implemented`, `validated`, `used in claim-bearing experiment`, `planned` 四种状态。

## 5. 扩大解决问题范围：三个连续 gate

扩大范围不能等同于“多跑几个体系”。每跨一层都必须更换实验单位，并设置失败后收缩结论的规则。

### Gate A — 机制边界：从单一构造到跨状态/误差结构的稳定机制

**A1. Pair-radial boundary matrix（优先级最高）**

- LJ states：稀疏 `(rho*, T*)=(0.50,1.20)`、当前 `(0.79,1.00)`、稠密 `(0.90,1.20)`；当前点做 `N={108,256,500}`；
- observables：预注册 primary pair count，同时测全 RDF norm、coordination、`U/N`、pressure；
- fields：null/aligned/random/high-frequency；主机制 arm 严格在预设等价界内匹配 force RMSE，另设固定 `beta*sigma(deltaU)` 的 calibration arm。两者回答不同问题，不得混为同一 matched-RMSE 证据；
- replication：6 条 construction × 独立 evaluation；每 field 6 条 direct chains；
- statistics：独立 construction/reference seed 为 cluster，层级估计 state/size/field-family；共同随机数只作为显式 paired design，并传播 covariance；
- gate：在看 confirmatory data 前定义 primary estimand、force 等价界、最小有意义 observable effect、允许的 calibration error 和 multiplicity family；根据 pilot cluster variance 决定每 cell 的 cluster 数，使等价界和 effect interval 达到预设精度。通过与否由这些 simultaneous intervals/等价性检验决定，而不是任意的成功 cell 比例。

**A2. Angular/many-body matrix**

- SW-Si：300/900 K；pair、angular、mixed error bases；ADF、tetrahedral order、RDF、pressure；
- EAM-Cu：300/1000 K；pair、density、embedding、mixed bases；RDF、Q6、pressure；
- 每个 basis family 独立 construction/direct replication；
- 若 angular/embedding arms 不通过，标题和结论必须明确限定为 pair-radial errors。

**A3. Dynamics 作为另一个科学问题**

static response formula 不覆盖 propagator。若测试动力学，应单列：LJ diffusion/VACF，SW-Si VDOS/phonon；用 Einstein 与 Green–Kubo、time-step/trajectory convergence、8 MD seeds/field。只有预注册的 `rho_force,dynamic - rho_force,static` 在多个体系复制，才可保留 static/dynamic reversal conjecture。

### Gate B — 真实误差流形：从人工 field 到 fitted/DFT MLIP

**B1. Analytic-oracle bridge**

- 在 SW-Si 与 EAM-Cu 上分别训练 pair/spline、linear ACE、BPNN/GAP-like、equivariant GNN；
- budgets `{100,1000}`，每 family/budget 至少 8 个 initialization/data-order seeds；
- train/validation/test/reference/direct chains 全分离；
- 预先定义 comparable-force stratum；按 family/data split 做 hierarchical inference；
- primary gate：预先定义 calibration intercept/slope、prediction error 与 false-trust rate 的 estimands；用开发任务确定具有物理意义的等价/非劣界，并在 held-out systems/families 上用 multiplicity-adjusted intervals 检验。模型/cluster 数应由 pilot variance 与目标区间宽度说明，而不是机械使用 `[0.8,1.2]` 或未给依据的 coverage 范围。

若 designed fields 成功而 fitted errors 失败，项目应保留“可构造反例”，撤回现实误差流形的解释。

**B2. DFT-level audit**

不要浅尝很多材料。建议先深做 Si：diamond/hot solid/liquid，冻结一个 DFT protocol；每 state 至少 6 条 AIMD chains，并用 ESS stopping rule，而不是 raw frame count。比较 ACE/GAP、equivariant GNN、foundation-model fine-tune，在预注册 band 中做 response calibration 与 direct observable validation。off-the-shelf models 若训练在不同 DFT labels 上，只能作 secondary arm，不能混成一个 oracle error field。

### Gate C — 实用选择决策：证明方法真的帮人选得更好

相关系数不是 selection utility。必须先冻结 policies：

1. minimum force RMSE；
2. 预定义 energy+force scalar baseline；
3. response policy：在 observable tolerance vector 下最小化 `max_j |predicted error_j|/tau_j`；
4. random 与 oracle best 仅作为下/上界。

no-oracle policy 必须先冻结它在现实中可访问的信息源，例如固定预算的稀疏 DFT labels、heterogeneous committee deviations 和 reference-ensemble frames；不能暗中使用 oracle `deltaU`。同时冻结 shared-bias stress test、overlap/ESS reject rule，以及信息不足时 abstain 的规则。force baseline 与 response policy 必须在相同总预算下比较，成本包括 reference labels、trajectory generation、committee training/evaluation 和 direct validation。

建议以 `2 systems × 3 states × 4 training-data splits` 作为初始设计骨架，但最终 task 数必须由 development-task variance、目标 regret/failure-rate precision 和 power analysis 决定。每个 task 的 candidate pool 保持多个 families、budgets 和 seeds。policy 只能查看上述冻结的信息与 test metrics；truth 来自独立 direct simulation。

Primary loss 应为预注册 minimax failure：

```text
L = max_j |observable_error_j| / tolerance_j
failure = 1[L > 1]
regret = L_selected - min_candidate(L)
```

预注册时应先由领域容忍度定义 loss，再把 development tasks 用于确定最小有意义的 regret/failure-rate 改善和非劣界；confirmatory tasks 上以 paired hierarchical estimand、simultaneous interval 和预先规定的 multiplicity control 判定。任务数由 pilot variance 与目标 interval precision/power 证明。若 calibration 良好但不能在预算匹配下改善 regret，结论必须是 “diagnostic, not a selector”。

## 6. 建议的执行与止损顺序

1. **Release blocker：**解决 exact reweighting/direct sampling 与 dilute benchmark 冲突；
2. **保住最窄结论：**多 construction/direct seeds 复现 exp07；
3. **清理推断：**撤回 IID bootstrap、post-hoc selector 和已验证 warning-light 的措辞；
4. **Gate A：**跨 state/size 后再投入 angular/many-body；
5. **Gate B1：**解析 oracle 下的真实 fitted errors；
6. **Gate B2：**只有 B1 通过才投入昂贵 DFT；
7. **Gate C：**只有通过盲测决策效用后，才把 `selector` 写进标题、摘要或结论。

所有 claim-bearing experiments 应在 clean tagged commit 上运行，保存 environment lock、config、seed、raw trajectories、model weights/checkpoints 与 hashes。当前主要 manifests 均为 `git_dirty: true` 且指向多个旧 commits；在没有 dirty diff 的情况下，现有结果无法被精确重建。

## 7. 写作和问题—证据闭环

当前标题、摘要和结论把三个层级混在一起：

- 已支持：force norm 不充分的受控存在性反例；
- 部分支持：oracle response 在单一 toy setting 中有预测信息；
- 未支持：现实 model-selection rule 与 no-oracle practical predictor。

若暂不补实验，建议将论文收缩为 Gate A 的 mechanism paper，并将结论改成：

> We provide a controlled existence proof that matched force RMSE does not determine the response of one static observable in a pair-potential oracle system. Whether this construction describes fitted MLIP error manifolds or improves model selection remains untested.

`coarse filter, not a selector`、`external-validity check passes`、`validated here`、`second-order term gives advance warning` 均应撤回或改成待检验假设。

文献定位也必须更新。当前 bibliography 被旧 SPEC 清单锁定，遗漏了与 practical payoff 直接重叠的最新工作：[Errors that matter / PET-EXP 2026](https://arxiv.org/abs/2604.24607) 已用 shallow ensembles 和 statistical reweighting 以接近单模型成本给出 observable-level uncertainty，并校准到实验；[Dyna-Mat 2026](https://arxiv.org/abs/2607.03433) 已在 15 个 foundation MLIPs 和有限温 trajectories 上比较 single-point errors 与 structural/dynamical observables。这两项工作会直接改变本稿“何处仍未解决、何处可能新”的答案。

## 8. 要求 Claude 逐条回应

请 Claude 不要仅回复“同意并将在未来工作中处理”。需要在当前工作分支 `claude/ai-science-project-plan-p0obrk` 中提交一份 `reviews/CLAUDE_RESPONSE.md`，逐条使用以下模板：

| Review item | Agree / disagree / partially agree | Evidence or counter-evidence | Concrete action | Commit / experiment ID | Claim after action |
|---|---|---|---|---|---|
| P0-1 end-to-end consistency |  |  |  |  |  |
| P0-2 calibration/common offset |  |  |  |  |  |
| P0-3 counterexample replication |  |  |  |  |  |
| P0-4 clustered zoo/post-hoc band |  |  |  |  |  |
| P0-5 warning-light false trust |  |  |  |  |  |
| P1-1 fitted-model inference |  |  |  |  |  |
| P1-2 README evidence status |  |  |  |  |  |
| Gate A scope |  |  |  |  |  |
| Gate B scope |  |  |  |  |  |
| Gate C decision test |  |  |  |  |  |

回复还必须明确回答以下问题：

1. 用一句话写出当前数据能支持的**最窄结论**；
2. 接受还是拒绝本审查重构后的科学问题？若拒绝，给出原始文献证据说明真正未解困境是什么；
3. 哪些 headline claims 将立即降级，哪些需要新实验后保留；
4. 对 P0-1 至 P0-5，给出 estimand、等价/非劣界、独立 chain/seed 数及其基于 pilot variance 的依据、实验单位、对照、readout、multiplicity 处理和预注册 decision rule；
5. 从 Gate A/B/C 中选择明确的 primary systems、states、observables 和 model families；不得只写 “more systems and models”；
6. 给出资源预算和止损点：特别说明 no-oracle response 的可访问输入、shared-bias/overlap/abstention 规则，以及与 force baseline 的总成本匹配；每个 gate 失败时，标题、摘要、结论分别如何收缩；
7. 说明如何在 clean commit 上重跑并封存 provenance；
8. 对 Eq. 5、`validated`, `external validity`, `warning light`, `selector` 等措辞逐项给出修订文本。

在 Claude 的 response 和相应实验/文字修改进入仓库前，本审查建议阻止将当前稿件标记为 submission-ready。

## 9. 核心外部证据

- Fu et al. **Forces are not Enough**：已展示 force accuracy 与 simulation metrics 不对齐。[arXiv](https://arxiv.org/abs/2210.07237)
- Morrow et al. **How to validate machine-learned interatomic potentials**：已明确主张 numerical metrics 与 physically guided validation 并用。[PubMed](https://pubmed.ncbi.nlm.nih.gov/37003727/)
- Imbalzano et al. **Uncertainty estimation for molecular dynamics and sampling**：已将 finite-data model uncertainty 传播至 thermodynamic averages，并提出 on-the-fly reweighting。[PubMed](https://pubmed.ncbi.nlm.nih.gov/33607885/)
- Liu et al. **Discrepancies and Error Evaluation Metrics for MLIPs**：已指出低平均误差不足，并提出与具体性质相关的指标。[arXiv](https://arxiv.org/abs/2306.11639/)
- Kellner et al. **Errors that matter**：以 ensemble、实验校准和 reweighting 提供接近单模型成本的 observable uncertainty，直接逼近本稿未完成的 practical layer。[arXiv](https://arxiv.org/abs/2604.24607)
- Gawkowski et al. **Dyna-Mat**：在 15 个 foundation MLIPs 的有限温结构/动力学 benchmark 中发现“平均相关、个体定性失败”，说明问题应被表述为条件化信任与例外检测。[arXiv](https://arxiv.org/abs/2607.03433)
