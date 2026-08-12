# 科学问题—科学探索审计（第二轮）：证据升级了，论文没有跟上

> 审查对象：仓库 `C-Xiyuan/testing`，项目状态截至提交 `4cbda50`（含 exp09/exp10/exp11 与全部第一轮回应性工作；其后的提交为本审计自身的文件）。
>
> 审查原则与第一轮相同：科学问题是学界尚未解除、持续阻碍可靠推断的困境；科学探索是针对该困境的完整解决链，其中每一个足以改变结论的环节都必须有独立、可重复、可证伪的实验支撑。本轮在此基础上增加一条：**作为最终呈现的论文必须与沉积证据同步**——一个以 auditability 为论点的工作，交付物与 deposit 脱节本身就是科学问题。
>
> 本轮方法：全文阅读 `docs/`、`paper/`、`reviews/`；逐一目验 `figures/` 全部图像；对 `results/*.json` 与文档声明做数值级核对（附件 C）；对 exp09/exp10/exp11 与全部 standalone 脚本做逐行统计方法审计并从 deposit 重算头条统计量（附件 B）；对 `paper/main.tex` 做逐行审计、实测字数、解析全部引文与交叉引用、用 `tex2md.py` 重生成比对（附件 A）；对 `atomlab/` claim-bearing 模块做正确性审读并运行测试套件（附件 D）；对外部标准（AIP/JCP 投稿细则、NeurIPS checklist）做检索核实；对第一轮引入的两篇 2026 文献做真实性核实。
>
> 附件：`R2_ANNEX_A_MANUSCRIPT_AUDIT.md`（论文）、`R2_ANNEX_B_STATISTICS_AUDIT.md`（统计机械）、`R2_ANNEX_C_RESULTS_VERIFICATION.md`（数据核对）、`R2_ANNEX_D_CODE_AUDIT.md`（核心代码与测试）。

---

## 1. 执行结论

第一轮审查后的工作是实质性的，不是修辞性的。exp09/exp10/exp11 三个实验在预注册规则下运行，四处对作者自己不利的结果被记录并写回文档（360× 不复现、二阶项 quick-run 错误、P0-1 反驳论证方向搞反、`reweight()` 中发现真实 bug）。**最窄结论被 exp11 实质性升级了**：aligned−null 对比在六条独立 construction cluster 上复现（+10.671，95% CI [+10.105, +11.238]，between/within 比 1.10），这是本项目目前最强的一块证据。

但本轮审计的核心结论是：**仓库现在存在三层相互矛盾的叙述，而作为最终呈现的论文停在最旧的一层。**

| 层 | 载体 | 状态 |
|---|---|---|
| 当前证据 | `docs/RESULTS.md`、`results/`、`reviews/CLAUDE_RESPONSE.md` | 含 exp10、两粒子精确基准、`reweight()` bug 修复，最新 |
| 论文 | `paper/main.tex` / `main.md` | 停在 exp11（提交 `6dc85ab`）；§5.3 仍称 35% discrepancy「open/unresolved、下一步测试 neither was run」，而 exp10 恰恰运行了该测试并推翻了 discrepancy；§5.6 的稀释气体疑云已被两粒子基准解决而论文未反映；§7 限制条款仍引用 quick-mode exp09 并称「all seven runs came from dirty working trees」 |
| 规范与回应 | `paper/SPEC.md`（自称 binding）、`paper/RESPONSE.md` | 停在第一轮审查之前；SPEC 强制要求的多个说法（1.01σ 校准、400× suppression、30× spread、摘要必须写 "coarse filter, not a selector"）恰恰是其后被撤回的；RESPONSE.md 保留 quick-mode exp09 数字与「二阶项使残差变差 1.71→2.12σ」——生产版结论方向相反 |

在此状态下，**任何一次「按 SPEC 合规」的写作都会把已撤回的说法写回论文**；任何一次按 RESPONSE.md 提交的审稿回应都会向审稿人陈述与 deposit 矛盾的数字。这不是排版问题，是证据治理问题，与第一轮的 P1-2（把实现能力写成证据范围）同源：**这一次是把旧证据状态写成当前证据状态。**

第二个核心结论：**exp10 的措辞纪律需要收紧。**预注册规则的判定是 "underpowered, not consistent"，而 `docs/RESULTS.md` §5.2 写「the question is closed」、README 的语气也接近关闭。可辩护的表述是三段式：(a) exp06 的 1.9-pair gap 在八链测量下不再出现，旧 direct 值处于新测量的尾部——但注意「2.7σ on its own error bar」的算术实为 **2.23σ**（2.7σ 来自误用 exp10 的误差条，附件 C.3-C9），**特定的 discrepancy 被更高功效的测量取代**；(b) estimator 一致性在 ±0.5 pairs 等价界下**未建立**（区间 ±1.3，宽于界 2.6×）；(c) 补齐比声称的「8× 链数」还便宜——欠功效的主因是预注册的 90%-discard 规则把无瞬态可弃的 primary 臂也只留每链 10% 帧（附件 B.3-5）：对已入档数据按低 discard 预注册一次修订分析 + 适度加链即可达界。在补齐之前，"closed" 应写成 "superseded by a higher-powered measurement; equivalence at the pre-registered bound not yet established"；「incomplete relaxation is excluded by measurement」也应降为「bounded at ≈0.3 pairs at low discard; unresolved at the deepest rung」（deposit 最深 discard 处 −1.28±1.08，附件 B.3-4）。

第三个核心结论：**P0-3 只在 construction-chain 这一根轴上被回答。**第一轮为反例稳定性列出的最低矩阵是 construction × targets(3) × bases(2) × sizes(2)；exp11 交付了 6 clusters × 1 target × 1 basis × 1 size × 1 state。`CLAUDE_RESPONSE.md` P0-3 行「the stability objection is answered」超出了交付物；正确的表述是「answered along the construction-chain axis; target/basis/size/state axes remain Gate A1」。

---

## 2. 第一轮 P0/P1 逐项落实核查

判定含义：**已解决** = 实验完成且经本轮独立核对；**部分** = 有实质进展但留有本轮指出的缺口；**未动** = 仅有设计。

| 项 | GPT 的行动 | 本轮核查 | 判定 |
|---|---|---|---|
| P0-1 端到端一致性 | exp10：8 参考链 + 16 surrogate 链、双 bin、双 sampler、六 estimator；bracketing 排除弛豫；发现并修复自身 bracketing 臂 step-size bug | 六 estimator 表与全部诊断经逐位核实（附件 C.4）；预注册判定为 **underpowered**；"closed" 措辞超出判定（§1）；「relaxation excluded by measurement」超出 deposit（最深 discard 处 −1.28±1.08，附件 B.3-4）；协方差修正符号反（B.3-2）；欠功效主因是 90%-discard 规则（B.3-5）——补齐比预想便宜 | **部分** |
| P0-2 校准/共同偏移 | exp09 生产版 64-cell 分解：offset 归因参考链实现；二阶项移除 grand mean 与场结构；发现预测自身误差缺失（interaction 的 1.19×） | 分解表逐位核实（附件 C.4）、分解逻辑成立；但 (a) exp09 自认的 CRN 缺陷方向性被误标为 conservative——对「二阶修正后无场结构」这一**零假设式头条结论**它是 anti-conservative 的：用作者自己报告的共享分量（0.134/0.401）修正列效应噪声地板，0.435σ → ≈0.41σ，方差比 0.94 → ≈1.06（一阶 2.58 → ≈2.9），结论方向不翻转但「exactly the truncation」余量更薄，需在 Gate A1 用逐场独立流复测；(b) 0.134 的原料（per-chain 直接链均值）未入档（B.1-6）；(c) null 场二阶案例的 deposit 旗标是 improves=false，docs 的「points the same way」spin 不成立（C12） | **部分** |
| P0-3 反例复现 | exp11：6 独立 construction clusters，预注册 estimand/等价界/最小效应，全部通过；360× 自我纠正为 median 26× | 契约/CI/力匹配逐位核实（附件 C.4）、cluster 间种子独立性经枚举确认；但 (a) 仅覆盖 construction 轴（1 target × 1 basis × 1 size）；(b) 场间直接链共享随机流、承诺的 paired-covariance 设计未实现（附件 B.1-1）；(c) 六 cluster 共享同一平衡起始构型且直接链未做 drift check（B.1-2）；(d) 26× 与入档字段名实错位、分母与零相容（B.2-3） | **部分（construction 轴已闭合，带四项保留）** |
| P0-4 clustered zoo | 撤回推断，限定为 fixed-zoo 描述；`zoo_uncertainty_propagation.py` 补测量误差传播并汇报 floor 率 | 文本处理到位（RESULTS.md §3a 的限定段落是模范写法）；**但图 `headline_regimes.png` 面板 (b) 仍印着 "physics spans 30×"**，而 30× 在正文已被 struck——图文相抵，见 §7 | **文本已解决，图未跟上** |
| P0-5 warning light | 89-case screening：false-trust 19%、AUC 0.556；claim 撤回；同时确立「作为 correction 有效、作为 gate 无效」的区分 | 撤回到位且区分清晰；deposit 支持 89 例版（逐项复核，附件 C.2-3），`CLAUDE_RESPONSE.md` §2 的 25 例表为旧快照须修正；**但 89 例中 64 例是 8 场 × 8 参考链的伪重复、且部分真值标签已被 exp10 推翻**（附件 B.5-2/3）——19%/0.556 只能作描述统计，不应再作为该诊断的性质被引用 | **撤回成立；引用的数字需降格** |
| P1-1 fitted arm 推断 | §3b 重写为 feasibility；EGNN n=2 标注为 hypothesis-generating | 核查通过 | **已解决** |
| P1-2 README 证据状态 | 四态表重写 | 表本身合格；但五行滞后（exp10/exp11/两粒子基准 "running"、`analysis/fep` "pending exp10"、exp06 "one unresolved disagreement"），三处章节指针错行（exp03→§5.3、exp05→§5.1、exp07→§4，附件 C.3-C20），「Every manifest records … SHA-256」以偏概全（仅新三个 run 有新 schema，C19） | **已解决（新增滞后与指针瑕疵）** |
| Gate A/B/C | 设计具体化（primary cell、预算、止损表） | 设计合格；执行为零；见 §5 | **未动** |

另有一项第一轮承诺需要追认：`CLAUDE_RESPONSE.md` §3.7 承诺 exp03/05/06/07「will be re-run on a tagged commit **before any of their numbers are used in a submitted document**」。这四个实验支撑论文 Table I–VI 的大多数数字，re-run 未执行（附件 C.0），而其墙钟成本合计仅 ≈4.7 h。承诺成立则当前论文不可提交；承诺若要修改，需要显式说明并给出替代的可复现性论证。同时，§3.7 的另一半——「manifests record the commit and the digest」——被本轮发现在字面上不成立（manifest 记录 finish 时刻的 commit，见 R2-P0-1a）。

---

## 3. 本轮新发现的问题

### R2-P0-1 论文与证据脱节（submission blocker）

全文逐行审计见 **附件 A（`reviews/R2_ANNEX_A_MANUSCRIPT_AUDIT.md`）**。要点（行号以 `main.tex` 计）：

0. **最重的一条（附件 A.2-1，CRITICAL）：§4.2（691–705, 714–721）与 Fig. 2 caption（1706–1710）保留 exp09 smoke-run 的全套数字**——grand mean「−0.34 ± 0.46」对 deposit 的 **+0.781σ（符号相反）**、「36/64 negative」对 22/64、「rms 1.71σ」对 1.985σ，以及**「adding the second-order term raises the rms from 1.71σ to 2.12σ」对 deposit 与同一论文 §2.3(iii)/§4.7 的 1.985→1.717σ（方向相反）**。论文目前同时断言一个已撤回的命题（§4.2）和它的否定（§4.7）。`CLAUDE_RESPONSE.md` 声称的「All five places are corrected」漏掉了这两处；Fig. 2 caption 还把过期数字标注为出自 `results/exp09_calibration_replication`——该目录里是不同的值，这是 RULE 1 级别的违规。
1. **§5.3 标题与正文过期。**「An open 35% discrepancy」「we record the item unresolved」「…neither was run」（1158–1214）。exp10 正是这个实验，已运行、已入档（`summary.json` 含 `underpowered: true` 判定字段）、已在 `docs/RESULTS.md` §5.7 记入相反结论。论文对自己最重的未解项陈述了一个已被推翻的状态，并且**断言一个已完成的实验未曾运行**——这恰是 SPEC §11.16 在相反方向上禁止的行为模式。
2. **§5.6 稀释气体节过期。**两粒子精确基准（`475670d`）已把 4.7σ 归因于基准自身的 O(ρ) 近似、并确立 reweighting 在 200 倍扰动幅度范围内 1.5–1.8% 精度；论文仍停在「most likely … not confirmed; quarter-density test not run」。
3. **`reweight()` bug 未披露。**该 bug 使 shift 的误差条在权重塌缩时虚假归零、且两个标准诊断（Kish、max weight）全程健康——直接触及论文中所有 reweighting 误差条的可信叙述；修复与回归测试均已存在（`atomlab/analysis/response.py:410–450`，commit `9adfe3c`；tests/test_response.py:313），论文只字未提。按论文自己的披露标准（equilibration failure 写满一小节），这是必须披露的同级事件。
4. **已撤回措辞残留两处**：§6 的 "a mechanism, **validated** at 1.01σ (n = 8) and 1.06σ (n = 10)"（1344，违反 §3.8 撤回承诺）；§4.4 段首主题句仍是一般化的 selector 命题「force RMSE cannot choose among models of comparable force error — the only situation a practitioner is ever in」（809–810），修正后的 within-zoo 表述迟至 844–850 才出现——审稿人读的是主题句。
5. **§7 限制条款与 §3.8 过期且互相矛盾**：limitation (iii) 称 deposit 的 exp09 为「quick mode」而 manifest 记录 `quick_mode: false`；「seven runs … six commits … every one dirty」实为 8 runs、7 commits，且 exp09/exp11 的 dirty 仅为自身输出目录（对这两个 run「不可从命名版本重建」不成立）；Data Availability 声明只点名 Secs 4.4–4.5 不可重建，与 §3.8 的「every run」互斥；「Untested here」清单含两项已测项。
6. **摘要缺最强证据且有一处错误归属。**exp11 六 cluster 复现与 exp10 判定完全缺席；摘要与结论把 designed-arm 的分解（0.57σ/−0.86σ）同时套到 1.06σ 的 fitted arm 上，而 fitted arm 自己的 deposit 分解是 −0.49σ/1.00σ（附件 A.2-6）。
7. **结论节**以 1.01σ/1.06σ 为 estimator 段落主数字，未提 exp10 六 estimator 收敛——后者才是「the estimators agree」目前最有力的呈现；「a factor of fifty」还是结论里首次出现的数字（违 SPEC §4.8）。
8. **正文超长 + 字数注释造假**：实测 11,633 词对 8,000–9,200 上限（Results +71%、What did not work +40%、Conclusion +45%）；文件头注释声称 9,198 且「all within ±20%」（附件 A.2-3）。
9. **main.md 与 main.tex 双向失同步**（附件 A.3）：用当前 `tex2md.py` 重新生成与已提交 md 差 414 行；md 摘要为 trim 前旧稿、公式编号错位、`[Gawkowski2026]` 裸键孤引回归；`tex2md.py` 自身缺新节/新引文映射；md 头注「generated by tex2md.py」当前为假。

### R2-P0-1a 附：provenance 声明的字面失效（manifest 记录 finish 而非 launch 的 commit）

附件 C.2-1。`experiments/common.py:286` 在实验**结束**时快照 HEAD：exp09 启动于 ≈04:52（时 HEAD `afd3c73`）而 manifest 记录 79 分钟后的 `9ffab92`；exp10、exp11 同理。commit-then-launch 的秒级间隔支持「started after committing」的实质，但「their manifests record the commit」记录的不是启动 commit。**有实际后果的一处：`reweight()` 修复 commit `9adfe3c`（06:30）落在 exp10（05:52 启动、已 import 旧代码）与 exp11 运行中途**——exp10 的 forward/reverse FEP 误差由 manifest 无法确定其 import 时刻状态的代码计算（记录的源码摘要是修复后的）。数值影响可忽略（Kish 0.83、无 FEP 误差条被引用），但一个以 provenance 为卖点的仓库，其 provenance 机制在这个案例上恰好没有覆盖到自己最重要的一次 bug 修复。修法：launch 时快照 commit+digest，finish 时再记一次以侦测中途漂移；两处都写进 manifest。

### R2-P0-2 SPEC.md 自称 binding、内容已错

`paper/SPEC.md` 声明「binding … this document wins」。但其中：

- §3（摘要必须含）第 4 条强制写入 1.01σ/1.06σ 校准、第 5 条强制写入 "Force error is a coarse filter, not a selector"——后者已被撤回为 fixed-zoo 描述；
- §10 claim 阶梯第 2 条（1.01σ「agrees at exactly the level the error bars claim」）、第 4 条（~400× suppression——deposit 实为 361×，且 exp11 已改立中位 26× 的下界读法）、第 5 条（30× spread、coarse filter）均为已撤回或已被 exp11/exp09 修正的表述；SPEC §4.2 还强制引用 30 帧构造「1.2–1.6 pairs」——该数**无任何 deposit backing**（generalisation.json 从 250 帧起），paper Table II 已依 RULE 1 拒引，而 `docs/RESULTS.md:129` 因 SPEC 强制仍保留它（附件 C.3-C2）：同一条规则体系内 RULE 1 与 §4.2 互相矛盾；
- §6 冻结六张图、禁止新图，而 exp09/10/11 的结果按 SPEC 自己的「figures for every claim-bearing result」逻辑需要呈现；§9 冻结引文清单（RULE 2），而 Kellner 2026 与 Gawkowski 2026 已按第一轮审查要求加入——**当前论文严格意义上已违反自己的 RULE 2**；
- RULE 1 的可追溯文件清单不含 `results/exp09|exp10|exp11/*`、`results/validation/*`、`revision_statistics.json`——按字面执行，论文引用这些新结果反而违规。

后果：SPEC 已从质量装置退化为错误放大器——任何「按规合规」的修订都会把撤回的说法写回去。**必须重签 SPEC v2**（更新 claim 阶梯至 `CLAUDE_RESPONSE.md` §3.3 的降级表、扩充 RULE 1 文件清单、解冻图与引文清单并重新冻结），或显式声明 SPEC 已被 review 链取代。二者必居其一，且需在仓库留痕。

### R2-P0-3 paper/RESPONSE.md 保留已被推翻的数字

该文件以「Response to the referee」名义存在，将随投稿寄出。其中：quick-mode exp09 的全套数字（grand mean −0.34±0.46、36/64 negative、rms 1.71σ）；「二阶项使残差**变差** 1.71σ→2.12σ」（R3 节）——生产版方向相反（1.985→1.717σ，见 `docs/RESULTS.md` §5.5）；Part 3 仍断言 360× out-of-sample suppression 与 "coarse filter and not a selector"。`RESULTS.md` §2.1 声称五处 quick-run 错误「All five places are corrected」——本文件是未被清点的第六处。作为将来要提交的文书，它当前会向审稿人陈述与 deposit 相反的结论。修法：追加显式 postscript（注明哪些段落被 exp09 生产版/exp10/exp11 取代），或重写并注明版本。

### R2-P0-4 图与正文相抵、图内统计残缺（详见 §7 图表逐一审计）

最严重的一条：`headline_regimes` 面板 (b) 注释 "physics spans 30×"，正文已 struck 30×、改立 12.1×/6.5×。其余：裸 ρ 无区间违反 SPEC §5 规则 5/15；`headline_prediction` 仍以 "rms residual 1.0σ" 为唯一注释；四个 flooring-to-zero 的成员被画成实心点/实心柱（上限被画成测量值）；Fig 3 的 null 目标 bin 根本不在绘图网格上（图的核心主张不可见）；Fig 6 面板 (b) 的 y 轴标签叠入面板 (a) 绘图区。

### R2-P0-5 新统计机械的缺陷（代码级）

逐行审计见 **附件 B（`reviews/R2_ANNEX_B_STATISTICS_AUDIT.md`）**。先说公道话：预注册判定规则真实存在于代码中并被遵守（含两次对作者不利的判定），两个自报 bug 的修复真实、正确、有回归测试，重算的全部头条数字都能从 deposit 复现。以下六条是仍需回应的实质缺陷：

1. **exp11 在场间重复了它要消灭的 CRN 缺陷，且承诺的 paired 设计未实现**（附件 B.1-1）。同 cluster 内三个场的第 d 条直接链共享同一随机流与起始构型；`run.py:242` 注释错误断言独立；`CLAUDE_RESPONSE.md` P0-3 承诺的「covariance estimated」不存在。实测配对相关 −0.09，CI 实质无恙——但这是**运气**（扰动势足够不同使 CRN 轨迹退相关），不是设计。
2. **exp10 的协方差修正符号写反**（附件 B.3-2）：ρ(R̄,P)<0 时 quadrature 是保守的，代码与文档却得出「anti-conservative」并把「修正」区间算得更宽；同一 ρ 还被错误地套到 direct−MBAR 行（那里起支配作用的是共享 surrogate 样本的强正协方差）。判定不受影响，入档的全部 covariance-corrected 数字与方向性文字需要更正。
3. **exp11 的「median 26×」与 deposit 名实错位、六个分母全部与零相容**（附件 B.2-3）：入档字段 `null_suppression` 实为 random/null 之比（median **4.8×**）、其数字未出现在任何文档；文档引用的 26× 是 aligned/null 之比、实验代码从未计算或入档；且六个 cluster 的 null 出样协方差全部与零无异（0.05–1.7σ），26× 只能作**噪声受限的下界**。正确表述：「null 场的出样一阶效应在全部六个 cluster 与零相容；suppression 因此只能给下界」——这比 26× 更强也更诚实。
4. **warning-light 的 89 例中 64 例是伪重复**（附件 B.5-2/3）：8 场 × 8 参考链，每场的同一直接测量进入八次；且若干「disagreement」的真值标签本身已被 exp10 推翻（0.0005 幅度的 false-trust 正是被证明为单链漂移的那个案例）。撤回结论不变（方向只会更否定），但 19%/AUC 0.556 不应再被引用为该诊断的性质；需要独立单元的干净重测才配得上一个数字。
5. **exp10「incomplete relaxation excluded by measurement」超出 deposit**（附件 B.3-4）：闸门只实现了弱方向（不能证明分离），最深 discard 处 5.25 Å bin 的 gap 点估计 −1.28±1.08 超界。可辩护表述：低 discard 处 bound ≈0.3 pairs、预注册比较点处无分辨。
6. **exp10 欠功效的主因是 90%-discard 规则而非 pilot 乐观**（附件 B.3-5）：primary from_surrogate 臂本无瞬态可弃，却也只用每链 10% 的帧；直接后果是 0.67–0.78 的误差本可 ≈0.15–0.24。两份互相矛盾的预注册功效文本（0.354 与 ~0.8 的 pilot sd）并存。**这使补齐比想象的便宜：对已入档的 primary 臂按低 discard 重分析（预注册一个修订分析计划）+ 适度加链即可达 ±0.5 界。**

7. **null 场的二阶项案例被向相反方向 spin 了两次**（附件 C.3-C12）：deposit（`revision_statistics.json.second_order_null_fields`）记录两个力级均 `second_order_improves_agreement: false`——二阶估计 (+0.021, +0.332) 与测量残差 (−0.547, −0.220) **反号**、加入后残差变差（−0.107→−0.439）。`docs/RESULTS.md` §2.1 把它写成「points the same way」（正面），`paper/main.tex` §4.2 把它写成支持「二阶项使残差变差」的一般结论（负面）。两个 spin 与同一 deposit 各自相抵：正确表述是「null 场的二阶案例不支持修正方向（n=2、量级在误差条内），exp09 的 64-cell 结果才是二阶项作为 correction 的证据」。

另有一批 P2/P3 级：exp03/05/06/07 彼此共享直接链流与起始构型、exp05 docstring 的成员独立断言两轮未改、blocking 0.53-对-1.30 矛盾的三成因（max-over-levels 高估、共享起点压低 between-chain scatter、双方 ~30% 噪声——方向翻转未被确立）、`reweight()` bug 位置笔误（response.py 非 fep.py）、「2.7σ on its own error bar」实为 2.23σ（附件 C.3-C9）、exp09「0.134 pairs」与 ⟨A⟩ 绝对均值的原料未入档（C13/C14）、两粒子「10% 失效点 βσ≈0.21」实为首个超限网格点而对数内插交叉 ≈0.09（C10）——全部见附件 B、C。

### R2-P0-6 核心代码与测试

[AGENT-CODE-FINDINGS]

### R2-P1（重要但不改变结论）

1. README 五行状态滞后 + 三处章节指针错行 + 一处 manifest 描述以偏概全（§2 P1-2 行；附件 C.2-2、C.3-C19/C20）。
2. `figures/review_response.(png|pdf)` 为孤儿图：由 `scripts/make_review_figures.py` 生成、含五个面板（exp09 方差分量、二阶修正、exp10 relaxation ladder、**exp10 六 estimator forest（带 ±0.5 界阴影）**、exp11 六 cluster 对比）——恰是论文缺失的三个实验的图、脚本还内置 quick_mode 拒读保护，但论文与 docs 均未引用。「缺的图其实已经画好了」使 R2-P0-1 的修复更没有借口。
3. `reviews/CLAUDE_RESPONSE.md` §1 P0-5 行与 §2 warning-light 表数字互斥；deposit 逐项支持 89 例版（附件 C.2-3），§2 的 25 例表是 exp09 前旧快照，应标注或修正。
4. 摘要与 RESULTS.md 的区间不一致（ρ 0.83 [0.67,0.93] 对 [0.68,0.92]；smoothness −0.02 对 −0.01）经核实为 noise-subtracted/raw 两套口径皆真实入档且各自有标注（附件 A.2-21 注）——但摘要与 docs 各用一套，SPEC §5 规则 7 要求的「单一口径 + 脚注给另一套」应当执行到摘要与 docs 层。
5. Raw trajectories 仍不保存（仅 per-frame observables）；`docs/methods.md` §8.1 自己估算存储成本「roughly 200 MB per claim-bearing experiment, which is affordable and was simply not done」——既然 affordable，对 exp11 的六 cluster 与未来 clean re-run 应当保存，否则「re-analyse the original frames」永远不可能。
6. `docs/methods.md:71` 仍写 equilibration 失败时能量错「30–40 %」；按其自身端点（−0.036 至 −0.043 对 −0.0311）应为 16–38%。paper/RESPONSE.md R14 已在论文中改正此数，docs 未同步——与 R2-P0-1 同类的反向脱节（这次是 docs 落后于 paper）。
7. `docs/theory.md:515–519` 保留 SPEC §9 明令必须替换或删除的无引文句「benchmark studies comparing MLIPs on downstream simulation tasks have reported exactly this dissociation」——SPEC 点名了这一句，两轮修订后它仍在。

---

## 4. 头条结论的证据现状（本轮口径）

| 结论 | 当前证据 | 本轮评级 | 距下一级还差什么 |
|---|---|---|---|
| 反例存在且**跨构造链稳定** | exp07 + exp11（6 clusters，CI [+10.105, +11.238]，force match ≤0.86%，算术逐位核实） | **established（单 cell 内）** | target/basis/size/state 轴（Gate A1）；跨轴失败则退回 single-cell。附带条件：null「suppression 中位 26×」应改立为**噪声受限下界**（六个分母均与零相容，且 deposit 字段 `null_suppression` 是另一统计量——附件 B.2-3）；正确且更强的表述是「null 场出样一阶效应在全部六 cluster 与零相容」 |
| 一阶 response 预测 tracks 直接测量 | exp07/03 三臂 + exp09 分解 + exp10 六 estimator 收敛 | **established as tracking**（×50 幅度范围） | interval coverage 未建立（exp10 underpowered；exp09 的 1.4× 误差条低估已归因但未修复进管线） |
| 「误差条错了，estimator 没错」 | exp09 分解 + variance budget（interaction=1.19× 缺失项） | **supported** | 把 prediction 误差纳入所有在册误差条并重跑受影响表格；CRN 修正后场结构比值 ≈1.06 需 Gate A1 复测 |
| 二阶项作为 correction | exp09 生产版（grand mean +0.781→+0.089σ；场结构 2.58→0.94[本轮修正≈1.06]） | **supported, margin thinner than stated** | 独立流复测；跨幅度/跨观测量泛化未测 |
| 二阶项作为 gate（warning light） | 89-case screening，AUC 0.556 | **withdrawn（正确）** | 若要复活：held-out 冻结阈值 + 预设 false-trust 上限，按 CLAUDE_RESPONSE §3.4 P0-5 设计 |
| 两 regime（across decades 强、band 内弱） | exp05 + band_robustness（窗口无关表述）+ uncertainty propagation | **fixed-zoo descriptive（正确限定）** | fitted models 上的预注册 selection experiment（Gate C）；在此之前不得回写 selector 语言 |
| estimator 机械正确性 | 两粒子精确基准（reweighting 1.5–1.8% across 200×；linear 10% 失效点按对数内插为 **βσ≈0.09**，文档的 0.21 是首个超限网格点——附件 C.3-C10）+ `reweight()` bug 修复（真实、正确、有回归测试） | **established（N=2）** | N=108 的传递性说明已给（βσ 分布形状不同）；建议 N=3 quadrature 桥接（见 §5）；失效点表述需改 |
| 35% discrepancy | exp10：+0.047/+0.160，旧 direct 在新测量的 2.23σ 尾部（文档的 2.7σ 系误差条归因错误，附件 C.3-C9） | **superseded, equivalence not yet established** | 低-discard 重分析 + 适度加链（附件 B.3-5）达 ±0.5 界，或在所有文档统一「underpowered」措辞 |
| force RMSE 与 observable error 的机制解释 | theory.md + 全部实验 | 一致 | 角度/多体误差（A2）、动力学（A3）未触及——结论 scope 已正确限定 |

---

## 5. 扩大解决问题的范围

第一轮的 Gate A/B/C 框架被接受但执行为零。本轮不重复该框架，只做三件事：按作者自己的成本口径排出**性价比序**；指出两个第一轮未覆盖、现在变得便宜的扩展；重申止损。

成本口径（来自 `CLAUDE_RESPONSE.md` §3.6，4 核机）：HMC 链 ≈115 s；equilibration ≈56 s；BPNN fit ≈13 s；EGNN fit ≈40 s。

| 优先级 | 实验 | 回答什么 | 估算成本 | 失败时的收缩 |
|---|---|---|---|---|
| 1 | **exp10 补齐**：先对已入档 primary 臂预注册低-discard 重分析（成本≈0，附件 B.3-5），再按重分析后的方差决定加链数 | 把 "underpowered" 变成等价性判定 | ≈0 – 8 CPU·h | 区间仍超界→撤回 estimator agreement 的定量表述，保留定性 tracking |
| 2 | **exp03/05/06/07 clean re-run（tagged commit）** | 兑现 §3.7 承诺；Table I–VI 可复现 | ≈4.7 h 墙钟（六份旧 manifest wall-time 合计 17,034 s——附件 C.0；含 quick 的 n400 臂按生产设置会略增） | 数字漂移超过误差条→逐表修订并披露 |
| 3 | **exp11 扩展：+2 targets、+1 basis、N=256** | P0-3 的其余轴；Gate A1 的一半 | 每新 cell ≈ 6 clusters ×（construction+evaluation+6 direct）链 ≈ 3–6 CPU·h/cell，全矩阵 ≈40–60 CPU·h | 某轴失败→标题与结论限定到通过的轴 |
| 4 | **K-observable response 向量的联合校准** | 论文的最终建议（report a vector）目前完全未测：8 bin 的预测/测量对与 direct chains 已在档，只差把 per-bin 预测区间的**联合覆盖率**算出来 | ≈0（纯分析，复用 deposit） | 联合覆盖差→「vector of response scores」从 recommendation 降为 proposal |
| 5 | **committee 方向测试（Gate C 的最小前置）** | no-oracle 可行性的第一道闸：用已有 10 fitted models 构造 committee mean 作 δU 替代，测 (a) committee-δU 与 true-δU 的 response 分量相关；(b) shared-bias stress（common biased subsample 重训小委员会）下该相关是否塌缩 | ≈10–20 CPU·h（重训 + 已有链复用） | 塌缩且不可检测→Gate C 的 response policy 需要 abstention 规则先行，论文 §6 committee 段措辞再降一档 |
| 6 | **A2：SW-Si 角度误差臂** | 机制是否为 pair-radial 特有 | ≈55 CPU·h（作者自己的 Gate A1 口径外推） | 失败→标题加 "for pair-radial errors" |
| 7 | **N=3 quadrature 桥接** | 精确基准从 N=2 向多体过渡（3 体相对坐标 6 维，重要性采样量级可行；或 N=3 直接 MC 到 10⁻⁴ 精度） | ≈数 CPU·h | 不可行→保留 N=2 并明示传递性论证的边界 |

第 4、5 两项是本轮新增的扩展方向，第一轮未覆盖：**第 4 项把论文的最终建议从口号变成可检验声明**（几乎零成本，最高性价比）；**第 5 项在不进入完整 Gate C 的前提下，先回答 no-oracle 路线是否有生存空间**——这是重构后科学问题（无 oracle、有限预算、共享偏差下的可校准误差预测）真正的第一块实验证据。

止损重申：Gate C 通过前，selector 语言不回标题/摘要/结论（作者已承诺）；A2 失败则机制结论限定 pair-radial；第 5 项失败则 no-oracle 实用性主张整体降为 open problem，且论文 §6 的「complementary」定位需要改写为「Kellner et al. 路线目前是唯一有实验支撑的 practical layer」。

---

## 6. 论文对标审计（JCP/AIP 细则 + NeurIPS 严谨性清单）

外部标准核实结果：AIP 官方作者须知确认——摘要一段、≤250 词、无公式/引文/脚注/表；节次序 Title→…→conclusion→supplementary→acknowledgments→author declarations→data availability→appendixes→references；图阿拉伯数字、表罗马数字；图内标注 ≥8 pt；双栏图 ≤6.69 in；**所有图必须有 alt text**；data availability 必须用 AIP 模板。SPEC §1.2 的约束表与官方要求一致，可继续作为格式基准。NeurIPS checklist（作为「同类会议」严谨性对标）中与本稿相关的六条：claims 与证据匹配、limitations、error bars 与统计显著性、可复现性（代码/数据/指令）、实验设置完整披露、compute 报告。

对标结果：

| 维度 | 现状 | 判定 |
|---|---|---|
| Claims ↔ 证据 | §5.3/§5.6/§7 与 deposit 相抵（R2-P0-1） | **fail，直到同步** |
| Limitations | §7 存在且直白，但三处过期 | 结构 pass，内容需更新 |
| Error bars / 显著性 | 文本纪律极好（±定义、σ-距离、无 p 值滥用）；图残缺（§7 审计） | 文本 pass，图 fail |
| 可复现性 | manifests + 内容摘要机制是超出常规的好实践；但 claim-bearing 四实验 dirty-tree、raw trajectories 不存 | **有条件 pass**（承诺的 clean re-run 完成后） |
| 实验设置披露 | methods 完整；exp09–11 的设置只在 docs/reviews 中，论文缺 | 需随同步补入 |
| Compute 报告 | §3.8 有硬件与 wall-time，但计量全部过期：「seven runs / six commits / 18,148 s ≈5.0 h」对 deposit 的九 run / 八 commit / 31,339 s ≈8.7 h，且把一个未入档的 quick exp09 算了进去（附件 C.2-6） | 结构 pass，数字 fail |
| 字数/结构 | 实测 11,633 词（正文 11,347）对 8,000–9,200 上限；Results +71%、What did not work +40%、Conclusion +45% 超节配额；**文件头注释声称 9,198 且逐节合规——注释为假**（附件 A.2-3）；排版估计 19–21 页，按 A.4-3 删减后 ~15–16 页 | **fail** |
| 引文 | 50 条全被引、无孤条、经典条目抽查全对（含 Wu2024 二作已改 Wenjiang Zhou）；Kellner/Gawkowski 两条 2026 引文经检索确认真实（arXiv:2604.24607、2607.03433）；**但两条不在 SPEC §9 白名单而 refs.bib 头注虚假声明全部合规**（附件 A.2-5）；四条条目缺卷页；md 中 `[Kellner2026]`/`[Gawkowski2026]` 为未解析裸键 | 部分 |

**叙述与简洁性（对最终稿的具体要求）：**

1. **摘要重写**（保持 ≤250 词）：以 exp11 复现后的反例为第一定量结果（10.7 pairs，6 clusters）；estimator 段以 exp10 六 estimator 收敛 + tracking-across-×50 呈现，1.01σ/1.06σ 降入正文；保留两 regime 与三个自我反驳；scope 句保留。
2. **§4.2/§4.7 合并**：现在的结构是「先陈述 1.01σ、再在 4.7 拆穿它」——作为 RULE 3 的叙事这成立，但 4.2 与 4.7 相隔两千词，读者在 4.2–4.6 之间会带着一个后文将撤回的校准印象读完全部主结果。修法：4.2 陈述 tracking + 前向指针改为同节内小字段落（"decomposed in §4.7"→直接给 offset/scatter 两个数），或把 4.7 提前为 4.3。
3. **草稿史元叙述收敛**：全文 "an earlier draft/version …"、"our notes …" 句式至少六处（main.tex:476, 852, 892–893, 906, 1194, 1310–1311，附件 A.4-4）。RULE 3 要求负结果是交付物——负结果保留在 §5；但**修订史**（哪一稿写错了什么）应收敛为 §5 末尾或 SI 的单一 "Corrections during review" 清单，正文每处一句话点到。一处此类披露是 integrity 信号，六处读起来是 change-log；JCP 读者要在物理论证与修订考古之间来回切换。
4. **§5.3 重写**为「A discrepancy that did not survive replication」：exp06 单链值 → 排除清单 → exp10 表 → underpowered 判定 → 8× 链数计划（或已补齐的结果）。
5. **§5.6 重写**：两粒子基准作为主证据（附 σ 表），O(ρ) 归因从 "likely" 升为 "measured"，`reweight()` bug 作为 boxed disclosure。
6. **术语统一**：second-order 统计量四个名字（warning light / correction-not-gate / self-diagnostic / the diagnostic，全文 21 处）固定为 "second-order diagnostic"，首次出现括注别名；"zoo" 一词三个指称（34 designed / 10 fitted / P1 注册的 model zoo）在 designed-vs-fitted 为论证主轴的章节里必须拆开命名（附件 A.4-5）；exp11 表行的裸 "null"（main.tex:1077）与 §3.5 的 scalar-null/curve-null 区分不一致。
7. **新图两张**（需 SPEC v2 解冻）：exp11 六 cluster forest plot（对比 + force-match 双轴）；exp10 六 estimator 收敛图（两 bin 并排、等价界阴影）。这两张图各自承载一个 P0 的关闭，比现有任何一张图的信息密度都高。

---

## 7. 图表逐一审计（全部图像经目验）

| 图 | 问题 | 等级 |
|---|---|---|
| `headline_regimes` | (b) 注释 **"physics spans 30×"** 与正文相抵（正文已 struck，立 12.1×/6.5×）；两面板标题裸 ρ 无区间无 n（违 SPEC §5.15）；(a) 散点无误差条，四个 floors-to-zero 成员（`null_f4e-03` 45% 等）画成普通点——上限被画成测量；(b) 柱状图无误差条、6 根 "4.0" 刻度无法区分成员 | **P0：必须重生成** |
| `headline_prediction` | 注释 "rms residual 1.0σ" 是已被降级的数字，应换为 offset/scatter 分解或直接删；两 force level 的同名场共用色/形状无区分；identity 线无带宽（可加 ±1 SE 带） | P1 |
| `headline_mechanism` | (b) 标题 "5× the damage"（实测 4.7×、曲线非单调越峰回落——标题与数据形状不符）；first-order 序列无误差条且 caption 需声明 deposit 无 per-point 不确定度；(a) 连线暗示函数关系，建议改为预测线（−βσσρ）+ 测量散点 | P1 |
| `exp07_counterexamples` | **null 场的目标 bin（3.4–3.9 Å）不在绘图网格上**——图的核心主张（null 对目标无作用）在图中不可见，仅靠 caption 补救；修法：加目标 bin 阴影竖带并绘出该 bin 的测量点；标题过长成句 | **P0** |
| `exp06_response_validation` | (b) y 轴标签竖排叠入 (a) 绘图区（布局缺陷）；(a) reweighted 曲线在大幅度处的饱和实为权重塌缩，需注 ESS 或截断显示；无任何面板体现 `reweight()` bug 修复后的误差条行为 | P1 |
| `exp05_proxy_correlation` | (a) 线性 y 轴 + 对数 x（SPEC 指定 log–log）；band 成员与 designed extremes 未标记；(b) 单一极端点（150）支配全图，其余 33 点挤在左下——log–log 或 inset；(c) 合格（含区间的 forest plot 是全套图中最好的一张） | P1 |
| `review_response` | 孤儿图（无任何文档引用）；若保留，(a) 的 "1.6x" 双注释未说明分别对应 row/column；建议并入 exp09 正式图或删除 | P1 |
| 缺失 | exp09 分解表、exp10 estimator 收敛、exp11 cluster forest、两粒子 σ 表——四个 review 后新结果**均无图**，而它们是当前证据等级最高的内容 | **P0（与 R2-P0-2 联动）** |

通用：全部图需按 SPEC §5.15 在标题/注释中带区间与 n；色板（蓝/橙/绿）色盲安全，保留；PDF 版本需与 PNG 同步重生成。

---

## 8. 要求 GPT 逐条回应

与第一轮相同的纪律：不接受「同意并将处理」。请在本分支（`claude/research-project-audit-n1ecg6`）或工作分支提交 `reviews/GPT_RESPONSE_R2.md`，使用以下模板逐条回应，并在每条给出可核查的 commit/文件证据：

| Item | Agree / disagree / partially | Evidence or counter-evidence | Concrete action | Commit / file | Claim after action |
|---|---|---|---|---|---|
| R2-P0-1 论文与证据脱节（子项 0–9 逐条） | | | | | |
| R2-P0-1a manifest launch-commit 修法 | | | | | |
| R2-P0-2 SPEC v2 或显式废止 | | | | | |
| R2-P0-3 RESPONSE.md postscript/重写 | | | | | |
| R2-P0-4 图重生成清单（§7 全表） | | | | | |
| R2-P0-5 统计机械缺陷（逐条） | | | | | |
| R2-P0-6 代码/测试缺陷（逐条） | | | | | |
| exp10 措辞（closed→superseded/underpowered）与 8× 补齐决定 | | | | | |
| exp09 CRN 方向性修正（0.94→≈1.06）入文 | | | | | |
| P0-3 剩余轴（targets/bases/sizes）时间表 | | | | | |
| clean re-run 承诺的执行或显式修改 | | | | | |
| §5 扩展表 1–7 的取舍与预算 | | | | | |
| §6 叙述修订 1–7 的执行 | | | | | |
| R2-P1 1–7 | | | | | |

并回答以下问题：

1. 用一句话写出**当前**数据支持的最窄结论（应当已比第一轮的版本强——包含 exp11 的复现轴——请精确措辞）。
2. exp10 的判定是 "underpowered"：你选择补齐 8× 链数，还是在全部文档统一改写措辞？两者选一，不允许「closed」与「underpowered」并存于仓库。
3. SPEC v2 还是废止 SPEC？若 v2：新的 claim 阶梯、RULE 1 文件清单、图清单、引文清单各是什么？
4. 论文同步 commit 计划：§5.3/§5.6/§7/摘要/结论/`validated` 残留/`reweight()` 披露，各自的修订文本或 commit。
5. exp03/05/06/07 clean re-run：执行时间表，或对「before any of their numbers are used in a submitted document」承诺的显式修改与理由。
6. 对 §5 扩展表第 4 项（response 向量联合校准，成本≈0）与第 5 项（committee 方向测试）：接受哪些、何时运行、预注册什么判定规则？
7. 图表：§7 表中每一项的处理（重生成/caption 修补/删除），以及 exp09–11 新图的设计。
8. 对本轮四处方法学修正是否有异议？若有，请给出定量反驳：(a) exp09 CRN 方向性（0.94→≈1.06，对零假设式结论 anti-conservative）；(b) exp10 "closed" 措辞与 relaxation "excluded" 措辞；(c) exp11 26× 改立为噪声受限下界（分母与零相容 + deposit 字段名实错位）；(d) warning-light 19%/AUC 0.556 因伪重复与被推翻的真值标签而降为描述统计。

在 `GPT_RESPONSE_R2.md` 与相应修改进入仓库之前，本审计维持第一轮的建议：**当前稿件不得标记为 submission-ready**——且本轮给出的理由与第一轮不同：不是证据不足，而是**论文尚未呈现已经取得的证据**。

---

## 9. 外部核实记录

- Kellner *et al.*, "Errors that matter: Uncertainty-aware universal machine-learning potentials calibrated on experiments", arXiv:2604.24607 —— 检索确认真实（Kellner, Hansen, Bligaard, Jacobsen, Ceriotti, 2026-04）。
- Gawkowski *et al.*, "Dyna-Mat: End-to-end benchmarking of foundation machine learning interatomic potentials in finite-temperature ensembles", arXiv:2607.03433 —— 检索确认真实（15 foundation MLIPs，四 tier，2026-07）；其 "on average lower force error → lower observable error" 结论与本仓库 across-decades regime 一致，引用定位恰当。
- AIP/JCP 作者须知（publishing.aip.org）—— 摘要 250 词一段无公式、节次序、图表编号、8 pt 标注、6.69 in 双栏宽、alt text 强制、data availability 模板：与 SPEC §1.2 一致。
- NeurIPS Paper Checklist —— claims/limitations/error bars/reproducibility/experimental detail/compute 六维，用作 §6 对标框架。

*Sources: [arXiv:2604.24607](https://arxiv.org/abs/2604.24607), [arXiv:2607.03433](https://arxiv.org/abs/2607.03433), [AIP author instructions](https://publishing.aip.org/resources/researchers/author-instructions/), [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)*
