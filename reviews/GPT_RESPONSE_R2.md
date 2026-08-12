# Response to the Fable R2 audit

**Response date:** 2026-08-13  
**R2 audit baseline:** `4cbda50bcaa3844e601e01fe751d3b2489557b7b`  
**Corrective comparison point:** `f0b1f6e5766ad3cb14222e079a9009eaa18bb282`

## Executive disposition

The R2 audit correctly found that the old manuscript lagged behind its own
evidence. Most of its highest-level manuscript, specification, response-letter
and provenance findings were already repaired by `f0b1f6e`. This pass accepts
the remaining truth-surface findings, corrects them, and records two defects
that became visible only after the repaired pipelines were inspected:

1. the arbitrary sign of an SVD vector could reverse the one-sided aligned-field
   estimand; exp09 and exp11 v2 were therefore superseded before production by
   sign-stable v3 protocols;
2. the old six-width exponent mixed genuine width dependence with truncation by
   the switching region and could not refute the proposed exponent.

No corrected production experiment has run. P0-1--P0-3 remain partial and Gates
A--C remain unanswered.

## Finding-by-finding response

| R2 finding | Current disposition | Evidence and action |
|---|---|---|
| P0-1: exp10 paper said the decisive test was not run | **Already fixed at f0; obsolete against current text** | The manuscript and `docs/RESULTS.md` now report point non-recurrence, underpowered intervals and a failed corrected relaxation gate. No equivalence or causal explanation is claimed. |
| P0-2: exp09 smoke values and reversed second-order story remained in paper | **Already fixed at f0; obsolete** | Smoke values were withdrawn; legacy exp09 is described as crossed, non-IID and provenance-invalid. Its row/column tests and mechanism attribution are withdrawn. |
| P0-3: exp11 covered only the construction axis | **Still valid** | The current ceiling is one state, size, target and basis. Six legacy cluster point contrasts do not complete Gate A. |
| P0-3: field-level shared streams and invalid paired UQ | **Still valid for legacy; repaired but untested for v3** | Exp11 v3 uses declared paired seed/start blocks and does not claim proposal-level exact CRN. Production is unrun. |
| P0-3: aligned SVD sign could reverse the one-sided endpoint | **New, accepted, release-blocking** | The sign convention is now fixed in observable space with regression coverage. Exp09/exp11 v2 are superseded-before-production; v3 must be run from a clean commit. |
| P0-4: general selector claim from clustered designed zoo | **Already narrowed at f0; figure lag was still valid** | Text is fixed-panel/post-hoc only. The figure pipeline was separately regenerated; no fitted-model population or utility inference is licensed. |
| P0-5: warning-light rows treated as independent performance cases | **Still valid; corrected here** | The 89-row table is now historical fixed-panel description. Binomial intervals, sensitivity/error-rate and Mann--Whitney inference are withdrawn because 64 rows are crossed exp09 cells and truth is partly reused/superseded. |
| P1-1: fitted arm overgeneralisation and EGNN n=2 | **Already fixed at f0** | It remains same-system feasibility/hypothesis generation, not ranking or external validation. |
| P1-2: status pointers and manuscript variants | **Mostly fixed at f0; kept partial until final release audit** | README status and TeX-to-Markdown direction were repaired. This pass synchronises the manuscript, results narrative and ledger; future numerical/figure changes still require a contradiction scan. |
| Gates A/B/C | **Unanswered** | Designs exist; no gate has production evidence. |
| R2-P0-1a: manifests sampled finish rather than launch | **Code fixed, legacy irreparable** | The corrected harness records launch and finish source states and fails closed on drift/dirty production. No legacy result is retrospectively upgraded. |
| R2-P0-2: binding SPEC amplified obsolete claims | **Already fixed at f0** | `paper/SPEC.md` is a short corrective specification rather than the historical binding brief. |
| R2-P0-3: `paper/RESPONSE.md` retained contradicted numbers | **Already fixed at f0** | It redirects to the audited response and explicitly retires the old answer. |
| R2-P0-4: figures contradicted narrowed claims or omitted uncertainty | **Still valid at f0; handled by the parallel figure remediation** | Captions here no longer defend graphical defects. Generated figures must remain legacy/fixed-panel and non-release evidence. |
| R2-P0-5: exp10 covariance sign/dependence and chain concatenation | **Fixed in code, scientifically unanswered** | Exp10 v2 uses joint complete-chain resampling and estimator-specific dependence. Legacy corrected intervals are withdrawn; production remains unrun. |
| R2-P0-5: exp11 suppression ratios with near-zero denominator | **Already narrowed at f0** | Ratios are descriptive/non-release; aligned-minus-null cluster contrast is the legacy readout. |
| R2-P0-5: width exponent | **R2 concern strengthened; corrected here** | Only widths 0.10, 0.15, 0.25 and 0.40 A satisfy `r0+3w<r_on`; their descriptive slopes are 0.787 direct and 0.724 first order. Shared trajectories and missing first-order pointwise UQ preclude an interval. The experiment is inconclusive about 1.5. |
| R2-P0-6: HMC adaptation freeze and observable tests | **Fixed here** | The adaptation window off-by-one is corrected; regression tests pin complete burn-in windows and production freeze. `PairBinObservable` now has direct edge/PBC/count/trajectory tests and its finite-$N$ $g(r)$ normalisation uses $(N-1)/V$. |
| R2-P0-6: exp07 maximum-bin statistic | **Still valid; corrected here** | The 11.36 +/- 0.95 maximum of eight noisy bins and unadjusted pointwise SE are withdrawn in body and caption. The deposited curve remains descriptive. |
| Clean rerun promise for exp03/05/06/07 | **Still valid, release-blocking** | The ledger now requires a frozen signed/tagged rerun or removal of every dependent number/figure before submission. |
| Exp06 direct-chain stationarity and shared starts | **Still valid; fail-closed here rather than cosmetically patched** | The legacy module now refuses non-quick execution. A release run requires a new frozen exp06 v2 with independent upstream units, surrogate-specific warmup, energy/target/$\delta U$ stationarity gates and raw chain deposition. No existing exp06 result is upgraded. |

## Annex A: manuscript and submission surface

| Annex A item | Disposition |
|---|---|
| Old exp09 values, exp10 omission, designed/fitted decomposition mix-up, `validated` wording | **Already fixed at f0**. |
| `reweight()` error-bar bug absent from manuscript | **Fixed here**: the manuscript now states the omitted reference-mean component, the joint-difference repair and the need for clean reruns. |
| Word-count header false / manuscript over target | **Fixed here**: `paper/wordcount.py` reports abstract 246 and main text 9,078, within the 8,000--9,200 target; the header records the same values. |
| Run/commit/time counts stale | **Fixed here**: nine runs, eight commits, 31,339 s (8.7 h), largest 5,304 s; dirty-tree implications are stated without claiming every number is impossible to approximate. |
| Inline unnumbered tables | **Fixed here**: the cluster and warning-light blocks are numbered REVTeX tables with inferential caveats. |
| `main.md` not generated from `main.tex` | **Fixed at f0 and regenerated after this response**. `main.tex` remains the source of record. |
| Citation metadata defects | **Still requires the release citation audit**; no bibliographic claim is marked resolved by this response. |

## Annex C: numerical truth-surface corrections

The following claims were corrected in `docs/RESULTS.md`:

- 30-frame 1.2--1.6-pair value: removed because no artefact was retained;
- 400x: corrected to the 360x fixed-chain point ratio and kept non-release;
- 100%: corrected to 285/286 (99.65%);
- EGNN force difference: corrected to 35.2%;
- width range: corrected to 4.66 (2.13--9.95 pairs);
- top-three overlap: corrected to the deposited raw 0.00--0.67, with no
  deposited noise-subtracted series;
- maximum weight below 0.0003: withdrawn because the series was not deposited;
- absolute reference means 209.66/211.38: withdrawn because source means were
  not deposited;
- exp09 shared component 0.134 pairs / 11%: withdrawn because the required
  per-chain direct means were not deposited;
- exp06 reproduction: corrected from 0.4% and 2.7 sigma to 0.29% and 2.23 sigma;
- equilibration energy discrepancy: corrected from 30--40% to 16--38%.

Items already corrected before this pass include low-force scaling wording,
exp09 production values, the null-field second-order spin, exp10's
underpowered/non-evaluable disposition, N=2 threshold overreach, README status
links, fitted-arm attribution and top-$k$ wording in the manuscript table.

## Additional fail-safe repairs

The review also revealed code paths capable of regenerating claims already
withdrawn in prose. The legacy exp06 summary now reports only the four
unswitched descriptive width slopes and never presents the norm of per-bin
errors as the standard error of a curve norm. The legacy exp07 summary stores a
post-selected curve maximum only under an explicitly descriptive key, removes
the aligned/null maximum ratio and records that selection-adjusted uncertainty
is unavailable. Regression tests prevent both inferential schemas from
reappearing. In addition, exp06 now fails closed for non-quick execution because
its shared-start/shared-stream design and missing stationarity evidence cannot
be repaired by relabelling its output. These changes do not rehabilitate either
legacy experiment.

## Direct answers to the requested response template

The table below is intentionally explicit about what was changed and what was
not converted into evidence. “Repaired code” never means “resolved scientific
item.” The commit column is left as **this corrective commit** until the whole
code/protocol/text set is committed atomically; no SHA is invented in advance.

### R2-P0-1, subitems 0--9

| Subitem | Agree / disagree | Concrete action and auditable file | Claim after action |
|---|---|---|---|
| 0, stale exp09 smoke values | Agree | Removed the smoke-run calibration story from `paper/main.tex` and `docs/RESULTS.md`; regenerated `paper/main.md`. | Legacy exp09 is a crossed, non-IID exploratory decomposition, not calibration. |
| 1, exp10 absent/“not run” | Agree | Rewrote the status in `paper/main.tex`, `docs/RESULTS.md` and the claim ledger. | Its point discrepancy did not recur, but the formal comparison is underpowered and non-evaluable. |
| 2, two-particle interpretation | Partly disagree with Fable's causal upgrade | The N=2 calculation is now called an unmatched implementation sanity check. Its temperature, cutoff, bins, perturbation and shared reference chain do not identify the old residual as $O(\rho)$. | Reweighting machinery passed one N=2 check; the many-particle/dilute attribution remains open. |
| 3, `reweight()` bug | Agree | Added the omitted-reference-mean/bootstrap disclosure to `paper/main.tex`; the joint-difference repair remains covered by `tests/test_response.py`. | Old reweighting error bars are not upgraded; affected claim-bearing runs require regeneration. |
| 4, residual `validated`/selector wording | Agree | Removed general selector and calibrated-predictor language from the manuscript, README and ledger. | Fixed-zoo and fitted-arm results are descriptive feasibility only. |
| 5, provenance/compute inconsistencies | Agree | Corrected the run/commit/runtime counts; `experiments/common.py` now captures launch and finish state and rejects source/protocol drift. | Legacy manifests remain invalid; only future clean runs can pass provenance. |
| 6, abstract attribution | Agree on the error; disagree with promoting legacy exp11 to proof | Rewrote the abstract under the ledger ceiling and separated designed from fitted residuals. | The abstract reports a provisional legacy fixed-cell observation, not established replication. |
| 7, conclusion hierarchy | Agree | Rewrote the conclusion; removed first-use factor claims and estimator “agreement.” | Tracking is provisional; equivalence and coverage are not established. |
| 8, false word count | Agree | Reduced main text to 9,078 words and abstract to 246; `paper/wordcount.py` and the header agree. | AIP length requirement passes; scientific release gates do not. |
| 9, TeX/Markdown divergence | Agree | Made `paper/main.tex` authoritative, repaired `paper/tex2md.py`, regenerated twice, and added `tests/test_tex2md.py`. | The generated Markdown is byte-stable; it is not an independent source. |

### Remaining P0/P1 template items

| Item | Agree / disagree / partially | Concrete action / evidence | Claim after action |
|---|---|---|---|
| R2-P0-1a launch provenance | Agree | `experiments/common.py` records start/end commit, source bytes/modes, protocol digest and artifacts; tests include hidden index-flag and mid-run drift cases. | Applies only to future runs; no legacy manifest is repaired. |
| R2-P0-2 SPEC | Agree; chose explicit replacement | `paper/SPEC.md` is the active corrective spec, the claim ladder is `reviews/CURRENT_CLAIM_LEDGER.md`, figure status is `figures/README.md`, and static citation allow-listing was abolished in favour of source verification. | Old binding requirements are retired and cannot resurrect withdrawn claims. |
| R2-P0-3 response letter | Agree | `paper/RESPONSE.md` is explicitly historical and redirects to the audited response. | It is not a submission response and supplies no evidence. |
| R2-P0-4 figures | Agree | Regenerated Figures 1--6 and their PNG/PDF/SVG variants; scripts and alt text now carry the same evidence caveats. `review_response` is labelled orphaned historical material. | All current figures are legacy descriptive panels; no v3 figure exists. |
| R2-P0-5.1 exp11 stream/UQ | Agree for legacy | v3 declares paired seed/start blocks rather than exact proposal-level CRN, uses cluster units and paired analysis, and is frozen in `protocols/exp11_v3.json`. | Repaired but unrun; legacy UQ is not rehabilitated. |
| R2-P0-5.2 exp10 covariance | Agree | v2 uses estimator-specific complete-chain joint resampling; misleading legacy covariance-corrected numbers were withdrawn. | Repaired but unrun; no equivalence claim. |
| R2-P0-5.3 exp11 26x | Agree on invalidity; do not promote it to a numeric lower bound | Removed the ratio as a primary result and recorded the schema/name mismatch. | Null predictions were noise-limited; no auditable suppression factor is claimed. |
| R2-P0-5.4 warning-light pseudo-replication | Agree | Removed binomial/AUC inference and labelled the 89 rows as reused/crossed historical descriptions. | The diagnostic is withdrawn as a gate. |
| R2-P0-5.5 relaxation excluded | Agree | All truth surfaces say relaxation remains live and the corrected gate fails. | Point non-recurrence only. |
| R2-P0-5.6 90% discard/power | Agree | Chose uniform underpowered wording rather than a post-hoc low-discard rescue. A new v2 run must follow its frozen primary analysis. | No “closed” statement and no retroactive equivalence test. |
| R2-P0-5.7 null second order | Agree | Reported the two null rows as mixed/failed directional evidence rather than a general correction result. | No validation of the second-order diagnostic. |
| R2-P0-6.1 aligned SVD sign | Agree | Fixed the scalar observable-space direction and added LAPACK sign-flip regression tests; superseded exp09/11 v2 before production with v3. | Scalar v3 direction is deterministic; degenerate vector targets require a new protocol. |
| R2-P0-6.2 width exponent | Agree | Refit only the four switch-compliant widths (direct 0.787; first order 0.724) and labelled all uncertainty redraws non-inferential. | The experiment is inconclusive about 1.5. |
| R2-P0-6.3 exp06 stationarity | Agree | Legacy exp06 now fails closed for non-quick execution. | A new v2 with independent starts, surrogate warmup, $U/A/\delta U$ stationarity and raw chains is required. |
| R2-P0-6.4 max-of-eight | Agree | Removed the 11.36 $\pm$ 0.95 inference and the aligned/null maximum ratio from code, text and caption. | The deposited curve is descriptive only. |
| R2-P0-6.5 tests | Agree | Added HMC adaptation-freeze, `PairBinObservable`, aligned-direction, legacy-claim schema and TeX-generation tests. | These protect implementation semantics, not scientific generalisation. |
| R2-P1.1 README/status | Agree | Corrected experiment states, pointers and manifest scope. | Evidence status is explicit. |
| R2-P1.2 orphan review figure | Agree | `figures/README.md` labels it historical, unreferenced and non-v3. | It cannot be cited as production evidence. |
| R2-P1.3 warning-light tables | Agree | `reviews/CLAUDE_RESPONSE.md` now uses the cumulative withdrawn disposition; obsolete snapshots are not a performance estimate. | No inferential error rate. |
| R2-P1.4 raw/noise rank conventions | Agree | Captions/tables name raw versus noise-subtracted conventions and give the alternative only as a labelled sensitivity. | No silent mixing of estimands. |
| R2-P1.5 raw trajectories | Agree | Corrected exp09/10/11 pipelines deposit raw coordinates and chain metadata; legacy runs did not. | Future artifacts are re-analysable; legacy ones are not. |
| R2-P1.6 equilibration percentage | Agree | Corrected `docs/methods.md` to 16--38%. | The old 30--40% figure is retired. |
| R2-P1.7 uncited theory claim | Agree | `docs/theory.md` is labelled historical and now deletes “exactly this dissociation” and the unsupported novelty/predictor assertion. | Literature novelty follows the manuscript audit, not the notebook. |

### Answers to the eight direct questions

1. **Narrowest current conclusion.** In one LJ $N=108$ state, for one target
   pair count and one radial basis, provisional legacy artifacts from six
   construction seeds report a large aligned--null point contrast, but invalid
   launch provenance, shared-stream UQ and unrun v3 production prevent calling
   it a confirmatory proof.

2. **Exp10 choice.** We chose uniform **underpowered / equivalence not
   established** wording, not a retroactive “closed” analysis. No 8x completion
   is claimed. Clean exp10 v2 production is a release gate.

3. **SPEC choice.** We replaced the old binding brief with the active corrective
   `paper/SPEC.md`. The current claim ladder lives in the ledger; RULE 1 is a
   manifest-and-artifact rule rather than a stale allow-list; the current figure
   inventory is `figures/README.md`; citations require primary-source checking
   rather than a frozen legacy list.

4. **Manuscript synchronization.** The abstract, conclusion, discrepancy and
   dilute-limit sections, limitations, `validated` remnants, compute report and
   reweight disclosure were revised in `paper/main.tex`; `paper/main.md` is
   regenerated and guarded by `tests/test_tex2md.py`. There is no pending hidden
   manuscript variant.

5. **Exp03/05/06/07 commitment.** No unsupported calendar promise is made.
   Before submission, each retained result must come from a new versioned,
   signed/tagged, stationarity-gated run, or every dependent number and figure
   is removed. Legacy exp06 now enforces this by refusing non-quick execution.

6. **Extension items 4 and 5.** We accept both as useful research questions but
   reject running them on invalid legacy aggregates. Item 4 must use independent
   construction panels and a joint chain-level max-statistic interval over the
   eight bins; the protocol must freeze simultaneous coverage and a practical
   error margin before data. Item 5 belongs after Gate B and must compare true
   versus committee response on held-out tasks, include homogeneous/shared-bias
   stress, and freeze a false-negative/abstention endpoint. Neither protocol is
   authored or run, so both remain unanswered rather than being listed as
   completed “zero-cost analyses.”

7. **Figure disposition.** `headline_regimes`, `headline_prediction`,
   `headline_mechanism`, exp05, exp06 and exp07 were regenerated with the
   corrected estimands/caveats; exp07 now shows the actual target measurement;
   exp05 ranges are fixed-zoo row sensitivity, not CIs; `review_response` is
   retained only as a labelled historical orphan. No exp09--11 evidence figure
   is created until corrected production exists.

8. **Five methodological corrections.** We accept all directional criticisms,
   with two stricter dispositions: the unretained exp09 CRN component cannot
   justify publishing the approximate 1.06 correction, so the inferential claim
   is withdrawn; and the exp11 26x ratio is removed rather than promoted as a
   numeric lower bound. Exp10 remains underpowered with relaxation unresolved;
   warning-light rates are descriptive only; aligned sign is pinned before
   production; and the width result is restricted to the switch-compliant
   subset and remains inconclusive.

## Scientific ceiling and required next action

The study is a provisional controlled fixed-cell mechanism observation, not a
validated predictor or selection method. Before any scientific upgrade:

1. commit code, protocols, tests and authoritative text as one frozen revision;
2. run exp09/exp11 v3 and exp10 v2 from clean production starts and deposit raw
   independent/paired units;
3. implement and run frozen replacements for legacy exp03/05/06/07, or remove
   their submission claims;
4. run Gate A before expanding beyond the fixed-cell construction axis;
5. reserve fitted-model prevalence and decision utility for Gates B and C.
