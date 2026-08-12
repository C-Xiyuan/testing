# R2 附件 C：沉积数据 ↔ 文档声明逐数核对

> 第二轮审计证据附录之三。方法：加载 `results/` 全部 JSON，与 `docs/RESULTS.md`、`paper/main.md`、
> `reviews/CLAUDE_RESPONSE.md`、`paper/RESPONSE.md` 的每个定量声明比对；核对全部九份 manifest。
> **一句话总评：deposit 本身内部一致、几乎支撑 `docs/RESULTS.md` 的每个数字（少数舍入与两条无 backing 的旁支除外）；
> 失配集中在交付层（paper/、SPEC、README、两份 response 文档）。**

## C.0 Manifest 全表

| 实验 | commit | dirty | dirty_paths | quick | wall (s) | 完成时刻 (UTC) |
|---|---|---|---|---|---|---|
| exp03_model_zoo | 8a2802c | **是** | 未记录（旧 schema） | 否 | 4796 | 08-12 01:45 |
| exp03_model_zoo_n40 | 400b2cb | **是** | 未记录 | 否 | 3722 | 08-12 02:46 |
| exp03_model_zoo_n400 | 2dd024a | **是** | 未记录 | **是（--quick）** | 292 | 08-12 00:10 |
| exp05_proxy_correlation | 964dbe8 | **是** | 未记录 | 否 | 3474 | 08-12 00:55 |
| exp06_response_validation | 590a1a1 | **是** | 未记录 | 否 | 2105 | 08-11 21:03 |
| exp07_designed_counterexamples | 590a1a1 | **是** | 未记录 | 否 | 2645 | 08-11 21:11 |
| exp09_calibration_replication | 9ffab92 | 是 | 仅自身输出 | 否 | 4813 | 08-12 06:12 |
| exp10_endtoend_consistency | 475670d | 是 | 仅自身输出 | 否 | 4188 | 08-12 07:01 |
| exp11_counterexample_replication | 9adfe3c | 是 | 仅自身输出 | 否 | 5304 | 08-12 06:45 |

九份 manifest、八个不同 commit、合计 31,339 s ≈ 8.7 h；库版本（Python 3.11.15 / NumPy 2.4.6 / SciPy 1.17.1 / Torch 2.13.0+cpu）与论文 §3.8 一致。承诺的 exp03/05/06/07 clean re-run **未执行**。

## C.1 CRITICAL（与附件 A 交叉确认）

1. **论文 §4.2 + Fig 2 caption + §7(iii) 的 exp09 数字无 deposit backing 且与被引文件相反**：−0.34±0.46 / 36-of-64-negative / −2.47…+1.64σ / rms 1.71σ / "quick mode" 对 deposit 的 +0.781 / 22 negative / −2.54…+1.83 / 1.985 / `quick_mode: false`。quick run 的 manifest 未入档——论文引用的那次运行在仓库里**不存在**。
2. **「二阶项使残差变差 1.71→2.12σ」在 main.md/main.tex/paper-RESPONSE 三处存活**，deposit 为 1.985→**1.717**（改善）；`CLAUDE_RESPONSE.md` 的「All five places are corrected」不实。
3. **论文 §5.3/Table VI/贡献 5 断言 discrepancy「unresolved」且决定性测试「was not run」**；deposit（exp10 summary.json）六 estimator 表逐位核实、direct−MBAR = +0.047/+0.160。docs 与 paper 对同一「release blocker」断言相反状态。

## C.2 MAJOR

1. **Manifest 记录的是 finish 时刻而非 launch 时刻的 commit（`experiments/common.py:286`），「run on committed trees」名不副实且有实际后果**：exp09 启动 ≈04:52（时 HEAD `afd3c73`）而记录 `9ffab92`（06:11，启动后 79 分钟）；exp10 启动 ≈05:52（HEAD `6d9df4d`）而记录 `475670d`；exp11 启动 ≈05:16（HEAD `f795c01`）而记录 `9adfe3c`。commit-then-launch 的 2–4 秒间隔支持「started after committing」的实质，但记录的不是启动 commit。**有实际后果的一处：`reweight()` 修复（`9adfe3c`，06:30）落在 exp10（05:52 启动、已 import 模块）与 exp11 运行中途**——exp10 的 FEP 臂误差由「import 时刻状态无法从 manifest 确定」的代码算出（记录的摘要是修复后的源码）。Kish 0.83 下数值影响可忽略、无 FEP 误差条被引用，但 provenance 声明在字面上不成立。修法：manifest 在 launch 时刻快照 commit/digest（finish 时再记一次以侦测中途漂移）。
2. **README 状态列五行滞后**：exp10/exp11/两粒子基准 "running"（皆已完成且 claim-bearing）；`analysis/fep` "pending exp10"；exp06 "one unresolved disagreement"。
3. **`CLAUDE_RESPONSE.md` 两张 warning-light 表并存互斥**；deposit 支持 89 例版（14/73、19.2%、上限 28.3%、AUC 0.5564、MWU p 0.2373——逐项复核）；25 例表是 exp09 前旧快照。
4. **图 `headline_regimes` (b) 印 30×**，与 revision_statistics.json 的 `resolved_from_zero: false` 及正文 strike 相抵（SPEC §6/§7 仍强制该图不变——SPEC 层级的锁死）。
5. **`paper/RESPONSE.md:395` 仍断言 360× out-of-sample suppression**；deposit（exp11 clusters.json）给 [21.1, 18.8, 16.0, 30.3, 467.2, 44.0]，中位 25.7。
6. **论文 §3.8 计量全部过期**：「seven runs / six commits / 18,148 s ≈5.0 h / 最大 4,796 s」对 deposit 的九 run / 八 commit / 31,339 s ≈8.7 h / 最大 5,304 s（exp11）。18,148 隐含一个 ≈1,114 s 的第七 run——即未入档的 quick exp09：论文的计量把一个仓库里不存在的 run 算了进去。

## C.3 MINOR（docs↔paper↔deposit 三方小失配清单）

| # | 位置 | 声明 | deposit | 判定 |
|---|---|---|---|---|
| C1 | RESULTS.md:30 | 低力级「shifts scaled down by four」 | 实测 [−0.547, −1.450, +0.835, +1.835] ≠ 上级/4；paper 已明说 not-divided-by-four | docs 松、paper 对 |
| C2 | RESULTS.md:129 | 30 帧构造给 1.2–1.6 pairs | 无任何 backing（generalisation.json 从 250 帧起）；paper Table II 已拒引 | docs 保留 SPEC 强制的无据数 |
| C3 | RESULTS.md:124 | 「roughly 400×」 | 10.206/0.028261 = **361×** | 舍入过度（docs/§5.6 与 paper 用 360×） |
| C4 | RESULTS.md:120 | 0.038 ± 0.078 | 0.03747 ± 0.07749 | docs 舍入错（paper 对） |
| C5 | RESULTS.md:217 | prediction >0.8 于「100%」窗口 | 285/286 = **99.65%** | 夸张 0.35% |
| C6 | RESULTS.md:373 | EGNN 力误差差「36%」 | 35.2% | paper 已用 35% |
| C7 | 摘要 | force match 「0.5%」 | 0.52%（正文/Table I 正确） | 摘要与正文不一 |
| C8 | RESULTS.md:477 | 复现 exp06 至「0.4%」 | 0.29% | 低估了自己的一致性 |
| C9 | RESULTS.md:478 | 「2.7σ on its own error bar」 | 用 exp06 自身误差为 2.23σ；2.7σ 出自 exp10 误差 | 归因错 |
| C10 | RESULTS.md:541 | linear 失效「crosses 10% at βσ≈0.21」 | 网格 0.052→5.3%、0.209→28.6%；对数内插交叉 ≈ **0.09** | 0.21 是首个超 10% 的网格点，非交叉点 |
| C11 | RESULTS.md:474 | 「the question is closed」 | 预注册判定 `any_underpowered: true` | 措辞超判定 |
| C12 | RESULTS.md:88–93 | null 场二阶「points the same way」 | `second_order_improves_agreement: **false**`（两力级）；二阶估计 (+0.021, +0.332) 与测量残差 (−0.547, −0.220) **反号** | docs 的正面 spin 与 deposit 旗标相反；paper §4.2 对同一 deposit 读法又相反——两个 spin 必须去掉一个 |
| C13 | RESULTS.md:646 | ⟨A⟩=209.66/211.38 | 绝对参考链均值未入档（相对结构可复核） | 补入档或删数 |
| C14 | RESULTS.md:697 | CRN 分量 0.134 pairs「from the deposited chain means」 | per-chain 直接链均值未入档 | 原料缺失 |
| C15 | RESULTS.md:437 | 「±0.832 the sweep had quoted」 | exp06 deposit 为 0.8206/0.792；0.832 只在 between_chain_scatter.txt | 两个入档工件第三位相抵 |
| C16 | RESULTS.md:269 | top-k「0.33–0.67」 | raw top-3 为 **0.00–0.67**；噪声扣除版未入档 | 「depending on correction」不可验 |
| C17 | 摘要:23–24 | 把 0.57σ/−0.86σ 分解接到 1.06σ fitted arm | fitted-400 臂分解为 1.00σ/−0.49σ | 错误归属（附件 A.2-6 同） |
| C18 | exp11 summary.json | 字段名 `null_suppression` = [7.0, 5.7, 1.6, 3.6, 153.3, 3.9] | 文档的 26× 系 aligned/null，另一统计量 | 名实错位（附件 B.2-3 同） |
| C19 | README:141 | 「Every manifest records … dirty_paths, SHA-256」 | 仅 exp09/10/11 有新 schema | 以偏概全 |
| C20 | README:91–94 | exp03→§5.3、exp05→§5.1、exp07→§4 | 实为 §3b、§3a、§1–3 | 指针错行 |
| C21 | RESULTS.md:390 | 宽度扫描「factor of five, 2.1–10.0」 | dynamic_range **4.66**、max 9.9456（paper 已用 4.66/9.9） | docs 舍入过度 |
| C22 | RESULTS.md:560 | 「max weight … never exceeded 0.0003」 | 非 two_particle_exact.json 字段（只在 commit message） | 无 deposit backing |
| C23 | CLAUDE_RESPONSE P0-3 行 | 「the stability objection is answered」 | exp11 config：6 clusters × **1 target × 1 basis × 1 size × 1 state**、每场 3 条直接链 | 只答了 construction 轴 |

## C.4 逐位核实为正确的部分（抽查通过清单）

exp10 全表与全部诊断、between-chain SEM 与相关系数、relaxation 零-discard gap、underpowered 判定；exp11 契约/CI/力匹配比/三场逐 cluster 移位/exp07 单 cluster 10.111/出入样协方差；exp09 全分解表、row effects、2.58→0.94、0.591/0.456=1.30、80 分钟、variance budget 1.19/1.434；exp07 头条表、11.36/11.18、generalisation 各行、361×/72×；exp05 两口径 ρ 与 CI、360.7/2.593/30.23/12.05/6.51、窗口扫描全部数字（16th percentile、286 窗、43%）、floor 率 59/45/39/34%；exp03 Table IV/V 全行、被剔除的 EGNN 对（−95.46±1.66 / −112.84±2.81，`direct_equilibrated: false`）、n400 缺三字段与论文声明一致、30.4σ；validation：low-density 4.68/6.53σ、两粒子六行 σ 列、采样器 0.66σ、between-chain 0.589/0.312/0.53、fresh-chain 1.725±0.101/1.757；revision_statistics 三臂分解、宽度指数 0.559 [0.090, 1.338]、1.4%、0.72。

**结论**：数据层可信，交付层失真。这决定了本轮审计的修复优先级排序——先同步文档，再谈新实验。
