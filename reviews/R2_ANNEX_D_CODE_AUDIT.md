# R2 附件 D：核心物理代码审计与测试套件

> 第二轮审计证据附录之四。范围：claim-bearing 链上的核心模块
> （neighbors/cell、lennard_jones、perturbations、sampling、observables_lib/rdf、models 的 δU 路径、equilibrate），
> 以及 tests/ 覆盖面盘点；response.py/fep.py/statistics.py/seeds.py 的审计在附件 B。
> 测试套件在审计机上完整运行（结果见 D.4）。

## D.0 总判定

**核心链在数学与实现上是健全的**：δU 进入 Cov、进入预测、进入 SumPotential 采样的是同一个函数（同一 half list、同一 7.0 Å cutoff）；force RMSE 在 designed-field 匹配、Gram 构造与 exp03 模型评估三处共用同一 per-component (1/3N) 口径；HMC 的 Metropolis 校正、力缓存回滚、burn-in 内自适应（旧 bug 已修）经核实正确。一条 P1 与三条 P2 需要回应。

## D.1 P1 —— aligned 场的符号是未固定的 LAPACK 工件，exp11 的单侧判定依赖它

`atomlab/potentials/perturbations.py:1417`（`_solve_weights`, mode="aligned"）取 `y = vt[0]`（对 1×K′ 白化协方差的 SVD）。**SVD 行向量的符号是任意的、依赖数据与 LAPACK 实现**（经验证：`vt[0] = ±m/|m|` 随输入翻转）。代码中没有任何一行固定 Cov(A, δU_aligned) 的符号，因此 aligned 场既可能抬高也可能压低目标 pair count，取决于 LAPACK 的内部选择。

后果分层：exp07 的预测-测量一致性安全（两者一起翻）；但 **exp11 的预注册规则是带符号的单侧判定**（`run.py:283`：aligned−null 均值契约的 95% 下限 > +1.0 pairs；`run.py:289` 还硬编码 `+10.111` 参照）。同一次运行内六个相似 cluster 的符号大概率一致（对固定协方差向量 ±5% 扰动下符号局部稳定——但这是 LAPACK 的性质，不是代码的保证）；**换一个 BLAS/LAPACK 构建、或一个接近翻转面的协方差抽样，整个契约翻号，「replicates」在物理完全相同的情况下变成「claim is withdrawn」**。这同时威胁跨平台重跑（clean re-run 计划！）与任何带符号 shift 的跨运行比较。

修法（一行）：按 `sign(Σ y·c) > 0` 之类的约定固定方向（使 Cov(A, δU_aligned) 恒为负/shift 恒为正），并加回归测试钉住符号。**在 clean re-run 之前必须先修此项**，否则 re-run 结果可能与现档不可比。

## D.2 P2

1. **exp11 场间 CRN**（`run.py:223, 242–247`）：与附件 B.1-1 相互印证——注释断言独立为假、`contrast_error` 高估簇内噪声、between/within 诊断失真；契约均值无偏、主 CI（between-cluster）不受影响。
2. **exp06 的直接链从不做平稳性检查**（`run.py:92–101, 226`；exp11 显式 `check=False`）：参考系有 guard，扰动系直接链只有固定 burn_in=300 + 接受率检查。残余弛豫把 direct 拉向 U₀ 值——**恰好发生在决定 linear/reweighted/direct 失效排序的大幅度端**（最大 0.032 eV ≈ 3 kT/pair）。exp07 检查了，exp06 没有；exp10 的教训（start-bias）在 exp06 未被回补。
3. **w^{3/2} 反驳实验的两个最大宽度违反自身前提**（`exp06/run.py:77–81` + `perturbations.py:334` 默认 `r_on = 0.85·cutoff = 5.95 Å`）：r0=4.2 时宽度 0.65 与 1.0 违反 `r0 + 3w < r_on`（6.15、7.2 > 5.95），高斯 bump 被 quintic switch 截断——**拟合出的 0.56 指数在两个最大宽度处混入了截断效应**，而 `summarise_widths` 用了全部六点。§5.1 的头条负结果（0.56 对 1.5）不会翻转（去掉饱和点已知给 0.72），但「测得指数」的表述需要加上截断 caveat，或以 `r0 + 3w < r_on` 的子集重新拟合。

## D.3 P3（卫生级，节选）

- `sampling.py:224–234` 首个自适应窗口以 10 为分母除 ~11 次接受（率可 >1）；`temperature_measured` 声明未填充；`energy_drift_per_step` 对全部 proposal 而非文档所称 accepted 轨迹平均。
- `observables_lib.py:94` `to_g_of_r` 用 ρ=N/V（无 N−1 修正），与 rdf.py 口径差 −0.9%@N=108——仅图示层，claim-bearing 原始计数不受影响；`PairBinObservable` 保留 i==j 自映像对（本几何下测度零）；直方图右边界约定与 `_rdf_kernel` 不一致（测度零）。
- `exp07/run.py:337–339`：`measured_max` 取 8 个含噪 bin 的最大值——对 null 场**上偏**；且 `measured_max_error` 取的是**预测**峰位 bin 的误差，未必是实测最大 bin。§4.3 的「null 移动最大非目标 bin 11.36 ± 0.95」带 max-of-noisy-bins 偏置与错位误差条，caption 应注明或改报固定 bin。
- exp07 全部直接链共享 `seed=ctx.seed+2`（已知，exp11 部分改良）；exp11 六 cluster 共享单一平衡起始构型（与附件 B.1-2 一致）。

## D.4 测试套件运行

在审计机（4 核，Python 3.11.15 / NumPy 2.4.6 / SciPy 1.17.1 / Torch 2.13.0+cpu）完整运行 `pytest -q`：结果见本文件末尾追记。

## D.5 核实为健全的部分（正面清单）

- **neighbors/cell**：镜像枚举上界、ghost 剪枝、bin 宽 ≥ r_search 的 3×3×3 stencil、r ≤ cutoff 含边界、相对 caller 未包裹坐标的 shift 记账（对 HMC 漂移正确）、half-list 平局规则（含自映像）、Verlet skin/2 判据、`min_cell_width` 用垂直宽度；对 O(N²) 参考与 ASE 交叉验证。cutoff 余量：2×7.0 = 14 < L = 17.54。
- **LJ shifted-force**：φ = u − u(rc) − (r−rc)u′(rc)，边界处恰为零；numba/numpy 双路径一致；力符号约定跨 LJ/perturbations/Gram 一致。
- **HMC/Metropolis**：总 H 上的 Metropolis、每 proposal 重抽 Maxwell 动量（MVV2E 单位正确）、velocity-Verlet 次序、拒绝时力缓存回滚、自适应严格限于 burn-in（旧 bug 在代码中已修——**但无回归测试**）、每步重建邻居表（skin=0）、remove_com 对平移不变势合法；对 Einstein 晶体闭式解与独立 Metropolis 采样器双重验证。
- **δU/观测量口径一致性**（本审计最重要的正面结论）：`PairBinObservable`、`build_shell_design`、`PairPerturbation.compute` 同 half list、同 cutoff；进入协方差、预测与采样的 δU 是同一函数。force RMSE 三处同口径。
- **whitening/aligned 数学**：Gram 特征分解带 rcond 地板（防零力方向藏进 null space）；aligned = G⁻¹c 方向是正确的最大化解（除 D.1 的符号）。
- **exp03 δU 路径**：`model.energy − potential.energy` 逐帧；模型力误差不能偏置测得 shift（HMC 接受只看能量）；直接链 burn_in=1000 + drift-check-and-exclude。
- **rdf.py 归一化**正确且不在 claim 路径上；**equilibrate.py** 的硬失败 guard 设计合理（Q6 σ 反保守——对拒绝式检查是安全方向）。

## D.6 三个最该补的测试

1. **HMC 自适应冻结回归测试**：断言 `burn_in=0, adapt=True` 时 `dt` 恒定、burn-in 后恒定——已知历史 bug 无回归测试。
2. **`observables_lib.py` 完全无测试**：claim-bearing 观测量 `PairBinObservable` 的 half-list 因子、边界包含、cutoff ≥ edges[-1] guard、自映像处理全部未测。
3. **aligned 符号钉住 + 端到端小测试**：固定 `AlignedPerturbation.design_covariance` 符号并断言（修 D.1）；把「−β·Cov₀ 预测 = reweighting = 直接采样（held-out 帧）」的小规模端到端检查纳入测试套件（目前只存在于实验脚本中）。
