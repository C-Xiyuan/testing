# R2 附件 B：统计机械审计（exp09/exp10/exp11 与 standalone 脚本）

> 第二轮审计证据附录之二。方法：逐行审读实验与脚本代码、`git log -p` 追踪修复、
> 从 deposit 重新计算全部头条统计量。严重度：P0 = 使某陈述结论失效；P1 = 实质推断缺陷，
> 结论存活但某个数字/方向错误；P2 = 真实缺陷、影响有限；P3 = 卫生/可追溯性。
> 先说公道话：**预注册判定规则真实存在于代码中并被遵守（含两次对作者不利的判定）；
> 两个自报 bug 的修复真实、正确、有回归测试；重算的所有头条数字均能从 deposit 复现。**
> 以下缺陷不推翻这一总评，但每一条都需要回应。

## B.1 种子与独立性

1. **[P1] exp11 在场间重复了它要消灭的 CRN 缺陷。**`exp11/run.py:224` 直接链种子 `base + 10 + 37*d` 无场依赖：同一 cluster 内 null/aligned/random 的第 d 条直接链共享同一 PCG64 流与同一起始构型——正是该文件 docstring（7–9 行）谴责 exp07 的模式。且 `CLAUDE_RESPONSE.md` P0-3 承诺的「common random numbers permitted only as an explicit paired design with the covariance estimated」**未实现**：`run.py:242–247` 注释断言「两场直接链独立」（对流而言为假），`contrast_error` 按独立性加方差。缓解（本审计从 clusters.json 实测）：aligned–null 配对相关 −0.09（与 0 无异，配对 SEM 0.512 对独立假设 0.490），故 CI [10.105, 11.238] 实质无恙——但承诺未兑现、注释失实，且 1.10 的 within 分母依赖这一未验证假设。
2. **[P2] exp11「clusters 之间只共享势/目标/基」为假**：六个 cluster 的 construction、evaluation 与全部直接链都从**同一次** `equilibrated_configuration(seed=ctx.seed)` 的单一构型出发（`run.py:153–158`）。该构型的任何残余非平衡是跨 cluster 共模、between-cluster scatter 不可见。直接链未做 drift check（`check=False`，run.py:225），从参考平衡构型在扰动势下出发（burn_in=300）——正是 exp10 所研究的 start-bias 结构。方向性：瞬态会把 aligned shift 拉向零（对存在性结论保守）；数据无大瞬态迹象。
3. **[P2] exp03/05/06/07 彼此共享直接链流**：四个实验的直接链全部用同一整数种子（`ctx.seed+2 = 2`，各 run.py 行号见附录）与同一起始构型；exp05 全部 34 成员共享一条参考链、一条直接流。**`exp05/run.py` 的 `spearman_with_ci` docstring 至今断言「members are independent … ordinary bootstrap is the right one here」**——被自己的种子代码直接反驳，两轮修订未改。
4. **[P2] 参考链与 anneal 链撞流**：`equilibrate.py` 用 `ctx.seed`/`ctx.seed+1` 做 melt/anneal，exp03/05/06/07 的参考链再用 `seed=ctx.seed+1`——参考链重放 anneal 链的随机流。混沌发散使实际影响可忽略，但属真实的流复用。exp09/10/11 经枚举无此类冲突。
5. **[P3] exp10 primary 臂起点聚簇**：8 条 from_surrogate 链全部取自**一条** presoak 链（seed 7777）间隔 ~143 帧的帧——8 链 SEM 按独立处理。τ_int≈1.7 帧下影响很小。
6. **[P3] exp09「shared component 0.134 pairs」不可从 deposit 复现**：records.json 只存 per-field SEM 与合并测量值，无 4×8 per-chain 直接链均值。声称「measured from the deposited chain means」的量，deposit 里没有它的原料。

## B.2 exp11 推断

1. **[OK] CI 算术精确复核**：6 契约的 t 区间（mean 10.6714、sd 0.5395、t₅=2.5706 → ±0.5688）与 summary.json 一致；±2% force 等价界真实在代码与 config 中、逐 cluster 出样核查、worst 0.86%；in-sample force RMSE 六 cluster 精确等于 4.0e-3（Gram-whitening 构造有效）；per-cluster 数据完整入档。
2. **[P2] between/within 比 1.10 不是方差分解**：`between`（0.540）是 cluster 契约的**总**散布、已含 within 成分（0.490）；真 between 分量 ≈ √(0.540²−0.490²) ≈ 0.23 pairs，在 5 dof 下与 0 相容。定性读法可存，比值本身不该被引用为分解。
3. **[P1] 「median 26×」与 deposit 字段名实不符、且分母全部与零相容。**代码计算并入档的 `null_suppression` = |random cov_out|/max(|null cov_out|,1e-30) → [7.0, 5.7, 1.6, 3.6, 153.3, 3.9]，**median 4.8×**——这些数字没有出现在任何文档；文档引用的 [21, 19, 16, 30, 467, 44]（median 26×）是 aligned/null 之比，只能从 clusters.json 手工重算，实验代码从未计算或入档。更实质：六个 cluster 的 null 场出样协方差全部与零无异（0.05–1.7σ），每个分母都是噪声抽样；`max(.,1e-30)` 是溢出保护不是噪声地板。**26× 应读作噪声受限的 suppression 下界**，"the defensible figure is a median of 26×" 需改写；正确的表述是「null 场出样一阶效应在全部六个 cluster 与零相容；suppression 只能给下界」。方向对结论保守，但数字的地位必须改。
4. **[P3] 最小效应 1.0 pairs 的辩护引用 exp10「已确立的分辨能力」**——而 exp10 判定是 underpowered，恰未确立该分辨能力。推断无恙（CI 下界 10.1 ≫ 1.0），辩护文字与 exp10 结果不一致。

## B.3 exp10

1. **[OK] 预注册规则在代码中**：±0.5 界、CI-containment 等价逻辑、underpowered 判定、判定优先级——全部实现并如实产出「all 8 comparisons underpowered」。
2. **[P1] 协方差修正符号写反。**实测 ρ 是 corr(参考链均值 R̄, 逐链预测 P) = −0.116/−0.086；比较量 diff = (S̄−R̄) − P 中该协方差以 **+2Cov(R̄,P)** 进入：ρ<0 时 Var(diff) **小于** quadrature，quadrature 是**保守**的。代码 `err_corr = sqrt(e_d²+e_t²−2ρ·e_d·e_t)` 在 ρ<0 时反而给出**更宽**的「修正」区间（0.6830 对 0.6744），而回应/commit 文本得出相反方向的结论（「quadrature very slightly anti-conservative」）。公式还错用了整段 direct 误差而非 R̄ 分量；同一 ρ 又被套到 MBAR 行——那里起支配作用的是 direct 与 MBAR **共享 surrogate 样本**的强正协方差，量根本不对。数值影响 ~1%、判定不变，但入档的全部「covariance-corrected」数字与方向性文字是错的。
3. **[P2] Bonferroni 数错**：预注册家族 8 项，代码用 α/4 的 z=2.50 而非 α/8 的 2.734——区间偏窄 ~9%；且全程用 z 而非 ~7–14 dof 的 t（对等价性判定是反保守方向）。因结局是 underpowered，未反转任何判定。
4. **[P2] 「Incomplete relaxation is excluded by measurement」措辞过强**：闸门只在弱方向实现（不能证明两臂分离），最深 discard 处入档 gap 为 [0.40±0.85, **−1.28±1.08**]——5.25 Å bin 的点估计超界。可辩护的说法：低 discard 处 bound ≈0.3 pairs，预注册比较点处无分辨。RESULTS.md §5.7 披露了 75–90% 变宽，但「excluded」一词超出 deposit。
5. **[P2] underpowered 的主要自伤原因未被点名**：direct 估计取 90% discard（每链 1500 帧只用 150），**包括无瞬态可弃的 primary from_surrogate 臂**——弃掉 ~10× 数据（0.67–0.78 的误差本可 ≈0.15–0.24）。预注册如此（非 p-hacking），但「pilot 乐观 2.5×」的归因与两份互相矛盾的预注册文本（response 里 pilot sd 0.354→±0.35「inside the bound」；模块 docstring pilot sd ~0.8→±0.8「wider than the bound … expected」）并存，且没有一处指出 90%-discard 规则是欠功效的主因。**补齐方案因此比「8× 链数」更便宜：primary 臂按低 discard 重分析（数据已在档）+ 适度加链即可达界。**
6. **[P3] 2.7σ 归因口误**：(−1.948−(−3.778))/0.821（exp06 自己的误差）= 2.23σ；2.7σ 是除以 exp10 的 ±0.67 得出的。RESULTS.md 与 CLAUDE_RESPONSE 的「on its own error bar」措辞与算术不符。
7. **[OK] step-size bug 修复正确**（tune_step_size 抛弃链 + 固定 kernel；MH 修正下任意固定步长精确；两臂共享同一调优值故无臂间不对称；run 重启）；reverse FEP 符号处理正确；Metropolis 4+4 处处如实披露。

## B.4 `reweight()` bug 与 FEP/MBAR

1. **[已验证] bug 与修复属实**——在 `atomlab/analysis/response.py:410–450`（**非 fep.py**；`CLAUDE_RESPONSE.md` 的证据列指向 fep.py 系笔误），commit `9adfe3c`：原代码把 shift 的误差赋为 reweighted mean 单独的 bootstrap 误差；权重趋 0/1 硬排除时该误差真趋零而 shift 仍携带参考均值全部不确定度，Kish/max-weight 全程无警。修复以「差作为单一统计量」重采样，自动处理两项相关。回归测试断言单调上升与硬排除极限（tests/test_response.py:313–352）；修复后行为在 two_particle_exact.json 复核（误差随幅度增长而非塌缩 4 个量级）。被作废的是误差条（各中心值一直正确）：exp06 reweighted 列、exp10 FEP 臂、§5.2 讨论。
2. **[OK] BAR/MBAR 实现正确**：Bennett 自洽式含 log(n_f/n_r) 项、二分法稳健、方差下界假设已注明且 claim-bearing 调用方一律 bootstrap；两态 MBAR 以 BAR 解播种即精确（两态时 BAR 就是 MBAR 自洽解）；误差用 moving-block bootstrap 且 dF 不确定度随每次重采样传播。[P3] 块长 √n 启发式、块可跨链边界——τ≈1.7 下无害。

## B.5 warning-light 校准

1. **[结论：14/73（19%）/AUC 0.556 是 code+deposit 复现的版本**；`CLAUDE_RESPONSE.md` §2 的 25 例/4/22/AUC 0.59 表是 exp09 之前的旧快照（89−64=25），同一文档内两表并存未调和。]
2. **[P1] 伪重复主导样本**：exp09 贡献的 64「例」是 8 场 × 8 参考链——每场的直接测量（同 4 条链、且场间共享流）进入**八次**，只在被差分的参考链上不同。73 个 trusted 中的大头与 14 个 false-trust 中的 10 个来自这些格。二项区间、Mann-Whitney p 与点估计都建立在远非可交换的单元上。docstring 的「in places a reference chain」严重轻描。
3. **[P2] 真值标签自身被后续证据推翻**：exp06 行用单链 blocking 误差；0.0005 幅度的那个「false trust」恰是 exp10 证明为直接测量单链漂移的案例（direct −1.948 对 exp10 的 −3.78 与预测一致）——若干「disagreement」是**标签错**而非诊断错，把 false-trust 率推高。因主张已撤回，方向只是强化否定结论；但 19%/0.556 不应再被引用为该诊断的性质，正确表述是「在本仓库的相关样本上无判别力；干净的错误率需要独立单元重测」。
4. **[OK] Clopper–Pearson、Mann-Whitney AUC、单侧方向、0.25 阈值来源标注（非 held-out）均正确。**

## B.6 zoo 不确定度传播与 blocking

1. **[OK] `zoo_uncertainty_propagation.py` 实现与声称一致**（redraw-before-floor、jitter 扫描、45% floor 率复现、1.41× 下三组区间复现）；docstring 未声称修复 clustering（诚实）。
2. **[P2] 但 clustering 从未修复且 exp05 文内断言仍错**（见 B.1-3）：exp05/band_correlations/band_robustness/本脚本的成员-iid bootstrap 区间一律偏窄，只能作 fixed-zoo 描述——回应的降格正确，区间的「描述性」地位应写在每个引用处。
3. **[P2] blocking 0.53 对 1.30 的方向矛盾有代码级解释**：(a) max-over-levels 规则（statistics.py:239–253）系统性高估 ~20–30%（8 层中 ~6 层是 10–22% 相对噪声的平台估计取最大）；(b) 两次 between-chain scatter 都因**所有链共享起始构型**而被压低；(c) 6/8 链下 sd-of-sd 各 ~27–32%，两个比值各自离 1 只有 ~1.5–2σ。**方向翻转本身未被确立**——「blocking 可靠性可变」的结论成立，但应补上「两次测量都带 ~30% 噪声与共享起点偏置」。
4. **[OK] autocorrelation（Wiener–Khinchin、Sokal 窗）、moving-block bootstrap（联合行重采样）、jackknife、Kish 加权统计——全部标准且正确。**

## B.7 两粒子精确基准

**[OK] 几何与归一化重推导无误**（r² 壳在内切球半径以下精确、Z = L³ + 4π∫r²(e^{−βu}−1)dr、cutoff 5.0 < L/2 = 6.0）；参考 pair 能量直接取自采样器同一势对象（无重复实现漂移——对比 `validate_low_density_limit.py` 手工重实现 shifted-force LJ 的风险）；采样器 0.66σ、linear 失效点 βσ=0.21、reweighting 全程 0.67–0.81σ 与误差校准（修复后）均从 deposit 复核。诚实限制（N=2 的 βσ 不可迁移）已在脚本与 commit 声明。顺带一个对 warning-light 的小注脚：amp 2e-3 处 ratio 0.151 < 0.25 会「信任」一个 5.3%（2.1σ）偏差的预测——阈值的边缘失效在精确体系里也可见。

## B.8 已验证的修复清单（对回应声称的逐项确认）

reweight() 修复（真实、正确、有回归测试）；exp10 step-size 修复（正确、重启）；exp10/exp09/exp11 全部头条数字从 deposit 复现；zoo 传播数字复现；manifests「committed tree」在有意义的口径上成立（dirty_paths 仅自身输出、quick_mode false、代码摘要机制有效）；warning-light 撤回在脚本/JSON/论文三处一致（14/73/0.556）；Metropolis 4+4 处处如实。

## B.9 与回应文本不符或内部矛盾的声称清单

1. 「quadrature slightly anti-conservative」——方向错（B.3-2）。
2. exp11「nothing shared between clusters」与「两场直接链独立」——均不成立（B.1-1/2）。
3. 26× 与 deposit 的 `null_suppression`（4.8×）名实错位；六分母与零相容（B.2-3）。
4. 同一回应文档并存 89-case 与 25-case 两套 warning-light 表（B.5-1）。
5. 两份互相矛盾的 exp10 预注册功效文本；90%-discard 主因未点名（B.3-5）。
6. 「Incomplete relaxation excluded by measurement」超出 deposit（B.3-4）。
7. 「shared component 0.134 pairs measured from deposited chain means」——原料未入档（B.1-6）。
8. exp05 docstring 的成员独立断言未改（B.1-3）。
9. 「2.7σ on its own error bar」实为 2.23σ（B.3-6）。
