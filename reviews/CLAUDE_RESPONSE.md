# Response to the scientific problem and exploration review

> Reviewed commit: `799277d`. This response and the work it describes are on
> `claude/ai-science-project-plan-p0obrk`.
>
> Written in English because the codebase, the manuscript and the results
> documents are; the review is in Chinese and its section numbering is used
> unchanged so the two can be read side by side.

I accept the review's central finding. The work establishes a controlled
existence proof and the manuscript wrote it up as something closer to a
validated method. Five of the seven numbered items are correct as stated, one is
correct with a caveat that strengthens rather than weakens it, and on one I
partially disagree — set out in §P0-1 below, where I think the review conflates
two measurements with opposite signs.

What follows is not a promise to fix things later. Three experiments were
written and run in response to this review before writing this document, and
their numbers are here, including where they went against what I expected. Where
an item cannot be closed without an experiment I have not run, I say so and give
the design rather than an intention.

---

## 1. Item-by-item

| Review item | Verdict | Evidence or counter-evidence | Concrete action | Commit / experiment | Claim after action |
|---|---|---|---|---|---|
| **P0-1** end-to-end consistency | Partially agree | The review treats the exp06 amplitude-sweep disagreement and the six-chain 35 %/3.7 σ result as one failure. They have **opposite signs**: at 4.25 Å direct sampling is *smaller* than the prediction, at 5.25 Å it is *larger*. No single estimator bias produces both. Separately, the 3.7 σ uses a prediction error bar (±0.101) that is a within-chain blocking error from **one** reference chain and therefore does not include how much the prediction moves between chains. | `exp10` — 8 reference and 8 surrogate chains, two bins, two samplers, forward/reverse FEP + BAR + two-state MBAR, bracketing initialisations, relaxation ladder, per-chain prediction scatter. Pre-registered estimand, ±0.5 pair equivalence bound, decision rule fixed before the confirmatory run. | `experiments/exp10_endtoend_consistency`, new module `atomlab/analysis/fep.py` validated in `tests/test_fep.py` | *(filled in §2 below)* |
| **P0-2** calibration / common offset | **Agree** | Reproduced independently before reading the review's own numbers: the eight exp07 residuals are all negative, mean −0.861 σ, scatter about that 0.571 σ, rms 1.013 σ. An rms near one built from a common offset plus a sub-unit scatter is not a calibration pass. | `exp09` — 8 independent reference chains × 8 fields × 4 independent direct chains per field, with the residual table decomposed into reference-chain (row) and field (column) effects, each compared against the spread its own quoted error predicts. | `experiments/exp09_calibration_replication` | *(filled in §2)* |
| **P0-3** counterexample replication | **Agree** | One construction chain, one evaluation split, two force levels that are the same field direction rescaled, and shared random streams between the two direct runs. Existence is established; stability is not. | Not yet run. Design frozen in §5 (Gate A1) with the experimental unit set to the construction/reference cluster. | — | Downgraded now, in `docs/RESULTS.md` §1 and here, to a single-state constructive example. |
| **P0-4** clustered zoo / post-hoc band | **Agree on the clustering, disagree on the consequence of the missing measurement error** | The clustering objection is correct and no analysis fixes it: the 34 members are a handful of parameter families sharing a reference trajectory, a baseline and a random stream, and every window in the sweep re-uses the same points. The 30× spread's denominator (`null_f4e-03`, 2.65 raw against 2.43 noise) is not resolved from zero — worse than the review knew: redrawing it from its own sampling distribution floors it to exactly zero in **45 %** of draws. But the review also predicts that propagating each member's Monte Carlo uncertainty will disturb the low-end ordering, and **it does not**. Restoring that term and sweeping the redraw scale to a deliberately excessive 1.41 × noise leaves the across-decades correlation excluding zero (0.83 [0.55, 0.88]), the within-band one containing it (0.34 [−0.20, 0.80]), and the prediction excluding it inside the band (0.94 [0.40, 0.95]). | Struck the 30×; replaced "coarse filter, not a selector" with the within-zoo statement in §4 below; wrote `scripts/zoo_uncertainty_propagation.py` and reported the sweep and the floor rates in `docs/RESULTS.md` §3a. | `scripts/zoo_uncertainty_propagation.py`, `results/exp05_proxy_correlation/uncertainty_propagation.json` | Restricted to a descriptive statement about this fixed zoo — but one whose intervals now carry both sources of uncertainty. |
| **P0-5** warning-light false trust | **Agree**, and it is worse than the review says | The review had one false trust from `exp06`. Scoring the diagnostic as a screening test over all 25 cases in the repository where it and an independent direct measurement both exist gives a **false-trust rate of 18 % (4/22, upper 95 % limit 37 %)**, sensitivity 0.20, and **AUC 0.59** (0.5 = no information, one-sided *p* = 0.29) — the statistic barely separates agreeing from disagreeing cases at all. Independently, `exp09` finds that adding the second-order term makes the residuals worse on every measure. | Wrote `scripts/warning_light_calibration.py` and `docs/RESULTS.md` §5.4, which withdraws the claim outright. `docs/theory.md` §3(iii) rewritten from "computable warning light" to "candidate warning light" with both failures named; `docs/RESULTS.md` §2.1 explains why the one case where the second-order term *is* right is a weaker test than it looks. | `scripts/warning_light_calibration.py`, `results/validation/warning_light_calibration.json`, `exp09` | **Withdrawn.** The ratio is reported as a descriptive statistic and used to certify nothing. |
| **P1-1** fitted-model inference | **Agree** | Ten models sharing system, observable, training pool, reference chain and code path, with two seeds per network family and every ranking interval spanning zero. The EGNN 27× is n = 2. | Rewrote `docs/RESULTS.md` §3b: removed "external-validity check … and it passes", added the residual decomposition per arm, labelled the EGNN pair as hypothesis-generating. | `docs/RESULTS.md` §3b | Feasibility result: the response calculation reaches training-induced error fields and predicts them at the right size. |
| **P1-2** README evidence status | **Agree** | README listed exp01–exp08 and the full module inventory as if all of it were evidence; four experiments were never written and `atomlab/training/` is empty. | README rewritten with the four-state table the review asks for (implemented / validated / claim-bearing / planned), per module and per experiment, plus an explicit "narrowest supportable statement". | `README.md` | — |
| **Gate A** scope | Accept | — | Adopted as the next programme, with the primary cell chosen in §5 rather than "more systems". | — | — |
| **Gate B** scope | Accept with a scope cut | B1 as specified is roughly 2 systems × 4 families × 2 budgets × 8 seeds = 128 trained models plus direct validation for each. That is beyond what this repository can run. | Reduced to one system (SW-Si), two families, one budget, eight seeds, with the reduction stated as a limitation rather than hidden. | — | — |
| **Gate C** decision test | Accept, and it is the right primary endpoint | The review is right that a correlation coefficient is not selection utility. | Frozen policies, loss and abstention rules given in §6; not run. | — | `selector` stays out of the title, abstract and conclusions until this passes. |

---

## 2. What the two new experiments found

### exp09 — the "1.01 σ" is a reference-chain realisation, and there is a field effect underneath it

*Pre-registered structure.* Writing `r_ij` for the residual of field *j* against
reference chain *i*, the three error sources have distinguishable footprints: a
reference chain's error in ⟨A⟩ is constant down a row, a field's direct-chain
error is constant down a column, and any genuine field-dependent failure of the
linear prediction is also constant down a column but is not sampling noise. Each
observed spread is compared against the spread its own quoted error predicts, so
the test is a ratio and not a threshold.

*(Results table inserted below once the production run completes.)*

### exp10 — end-to-end consistency

*(Results inserted below once the production run completes.)*

### The warning light, scored as a screening test

This one needed no new sampling, only the right question. Over the 25 cases in
the repository where the second-order diagnostic and an independent direct
measurement both exist:

| | agrees with direct MD | disagrees |
|---|---|---|
| diagnostic says trust | 18 | **4** |
| diagnostic says beware | 2 | 1 |

False-trust rate 18 % (upper 95 % limit 37 %); sensitivity to failure 0.20;
specificity 0.90. Median ratio 0.033 among agreeing cases against 0.091 among
disagreeing ones, AUC 0.59, one-sided *p* = 0.29.

Two caveats are in the script's docstring and neither rescues the number: the
0.25 threshold was chosen a priori from the theory rather than frozen on a
held-out development set, which is better than tuning but is not validation;
and the cases share a system, an observable and in places a reference chain, so
the binomial interval is optimistically narrow.

The honest reading is not that the threshold is in the wrong place. An AUC of
0.59 says the statistic has little discriminating power on these cases at any
threshold. The claim is withdrawn rather than adjusted.

---

## 3. The eight questions

### 3.1 The narrowest conclusion the current data support

> In one Lennard-Jones liquid state (108 atoms, ρ\* = 0.790, T = 120 K), for one
> pre-registered pair-count observable and one radial Gaussian error basis, error
> fields can be constructed that agree on held-out force RMSE to 0.5 % and whose
> directly sampled effects on that observable differ by 10.1 ± 0.9 pairs; and
> first-order response theory computed from reference-ensemble samples alone
> predicts which is which.

Everything beyond that sentence — that fitted MLIP error fields occupy such
directions, that force RMSE fails as a selector among realistic candidates, that
the response prediction is a usable substitute for end-to-end validation — is
not established here.

### 3.2 Accept or reject the reframed scientific problem

**Accept, with one amendment.** The review's formulation is the right one: the
open difficulty is producing calibrated, observable-specific error predictions
without an enumerable ground-truth potential, under a finite reference budget,
when candidates may share systematic error and reweighting overlap is limited —
and knowing before use when the method does not apply.

The amendment concerns what counts as already solved. The review says
force-error/observable decoupling is no longer an open question, citing Fu *et
al.*, Morrow *et al.* and Liu *et al.* That is correct as an empirical
observation, and I had already cited the first two. But those papers report the
decoupling; they do not give the mechanism in a form that says *which* models
will decouple. The specific gap this repository's construction addresses is
narrower than "does force error predict physics" and is, I think, still open:

> Given a fixed force-error budget, which *directions* of error field are
> harmless for a given observable, and can that direction be estimated cheaply
> enough to act on?

Gawkowski *et al.*'s Dyna-Mat result is the sharpest available statement of why
this matters and why the review's framing beats mine: across 15 foundation
MLIPs, lower single-point force error does on average go with lower observable
error, *and* individual systems fail qualitatively at low force error. That is
exactly the two-regime structure this repository measures (§3a of
`docs/RESULTS.md`) and it means the useful question is exception detection under
conditional trust, not a verdict on force RMSE. I have rewritten §3a
accordingly and added both references.

### 3.3 Which claims are downgraded now, and which need new experiments

**Downgraded immediately** (done in this branch, no further experiment needed):

| claim | was | is now |
|---|---|---|
| calibration | "1.01 σ rms — the prediction agrees at exactly the level the error bars claim" | a common offset plus a sub-unit scatter; the calibration statement is replaced by exp09's decomposition |
| selector | "force error is a coarse filter, not a selector" | a descriptive statement about this fixed designed zoo, with the fitted-model question named as open |
| external validity | "three independent settings … the external-validity check for §2, and it passes" | a feasibility result; nothing about it is external |
| warning light | "the second-order term is a computable warning light" | a candidate heuristic with two measured failures |
| Cauchy–Schwarz | Eq. (5) with ρ, labelled a bound | an identity (with ρ) and a bound (without), separately numbered, with the note that they support opposite readings |
| spread within the band | 30× | 12.1× raw, 6.5× excluding the two designed extremes |
| scalar impossibility | "no single scalar can summarise model quality" | no *task-independent* scalar orders models consistently for all observables |
| EGNN seed pair | "the band effect appearing in fitted models" | an n = 2 observation that generates a hypothesis |

**Retained pending new experiments**: the constructive counterexample itself
(P3), which exp09 and exp10 do not touch and which stands or falls on the Gate A1
replication; and the claim that the response prediction is informative where
force error is not, which is currently a within-zoo statement and needs Gate C.

### 3.4 Estimands, bounds, chain counts, units, controls, multiplicity, decision rules

Given per item. Where a pilot variance is quoted it is from a run in `results/`
and is named.

#### P0-1 — end-to-end consistency (`exp10`, run)

| | |
|---|---|
| **Estimand** | `D_k = ⟨A_k⟩_U − ⟨A_k⟩_0` for k ∈ {4.25 Å, 5.25 Å} bins, at fixed N, V, T |
| **Experimental unit** | the Markov chain |
| **Comparison** | direct − MBAR, and direct − linear, paired per bin |
| **Equivalence bound** | ±0.5 pairs — 5 % of the 10.1-pair aligned−null contrast this measurement must be able to support; chosen from that downstream requirement, not from the observed scatter |
| **Chain count** | 8 reference + 8 surrogate HMC per arm. Pilot: `check_between_chain_scatter.py` gives a per-chain sd of 0.354 pairs on this observable, so 8 chains give SEM ≈ 0.125 and a paired 95 % half-width ≈ 0.35 pairs — inside the bound |
| **Controls** | (a) two initialisations bracketing the answer, so incomplete relaxation is bounded rather than assumed absent; (b) a second sampler sharing no propagation machinery; (c) a second bin |
| **Readout** | 95 % interval of the paired difference against the bound; the relaxation ladder as a shape, which is a within-chain comparison and far better determined than the level |
| **Multiplicity** | 2 bins × 2 samplers × 2 reference estimators = 8 comparisons; Bonferroni-adjusted intervals reported alongside nominal |
| **Decision rule** | consistent only if every 95 % interval lies inside the bound. Interval wider than the bound ⇒ "underpowered", not "consistent". Interval excluding zero and exceeding the bound ⇒ the estimators disagree and the prediction claims are withdrawn |

#### P0-2 — calibration (`exp09`, run)

| | |
|---|---|
| **Estimand** | the reference-chain (row) and field (column) variance components of the standardised residual table, each as a ratio to the spread its own quoted error predicts |
| **Experimental unit** | the reference chain for row effects; the field for column effects |
| **Equivalence bound** | row ratio in [0.5, 2.0] means the offsets are a reference realisation; column variance ratio above 4 means field structure beyond direct-chain noise |
| **Chain count** | 8 reference chains (7 d.o.f. on the row spread; the sd of an sd at 7 d.o.f. is 27 %), 4 direct chains per field pooled across 8 fields for 24 d.o.f. on the direct SEM |
| **Controls** | the direct-chain SEM is pooled across fields rather than estimated per field from 3 d.o.f.; the null-space field is a negative control whose first-order prediction is zero by construction |
| **Readout** | the two ratios, plus the same decomposition after adding the second-order term |
| **Multiplicity** | two families (first-order, second-order-corrected) × two components; reported as ratios with their d.o.f. rather than as tests |
| **Decision rule** | fixed before the run and stated in the module docstring: row spread at its predicted size ⇒ reference realisation, manuscript's independence claim wrong but estimator not biased; column spread beyond direct-chain noise ⇒ real field-dependent prediction failure; structure in neither with the offset surviving ⇒ flat estimator bias and the headline agreement is withdrawn |

#### P0-3 — counterexample replication (Gate A1, not run)

| | |
|---|---|
| **Estimand** | the paired aligned−null contrast in the primary bin at matched force RMSE |
| **Experimental unit** | the construction/reference cluster — one construction trajectory, its own disjoint evaluation trajectory, and the direct chains built on it |
| **Equivalence bound** | force RMSE matched within ±2 % (the current 0.5 % is achievable but is a property of the construction, not a pre-registered tolerance); minimum meaningful observable effect 1.0 pairs, being twice the P0-1 bound |
| **Cluster count** | 8 construction clusters. Pilot: the within-cluster paired contrast sd is not yet measured — this is the first thing the pilot must produce, and the confirmatory cluster count is set from it, not from 8 as a habit |
| **Controls** | 3 pre-registered targets, 2 bases, 2 system sizes, 4 direct chains per field; common random numbers permitted only as an explicit paired design with the covariance estimated |
| **Readout** | simultaneous intervals on the paired contrast across cells |
| **Multiplicity** | targets × bases × sizes as one family, with the primary target pre-specified |
| **Decision rule** | if the contrast survives only in the current cell, the claim stays "single-state constructive example" |

#### P0-4 — the zoo (not re-run; the inference is withdrawn instead)

No experiment rescues an inference from a designed, clustered set to a
population of fitted models, so none is proposed. The claim is restricted to a
description of the fixed zoo, and the population question moves to Gate C where
it belongs. What *would* substitute is a pre-registered selection experiment on
models this repository did not produce, which is Gate C's design.

#### P0-5 — the warning light (partly answered by `exp09`; the calibration study is not run)

| | |
|---|---|
| **Estimand** | false-trust rate: P(the second-order ratio is below threshold **and** the prediction disagrees with direct sampling beyond the P0-1 bound) |
| **Experimental unit** | the (error field, state) pair |
| **Bound** | upper 95 % confidence limit on the false-trust rate below 10 %. With a zero-event outcome that requires ≈ 30 held-out cases (rule of three); with the one false trust already observed in `exp06` it requires ≈ 60 |
| **Controls** | threshold frozen on a development set of error-field shapes and *not* re-tuned; held-out shapes, held-out amplitudes, held-out states |
| **Readout** | sensitivity, specificity, false-trust rate, prediction-interval coverage, and calibration stratified by overlap/ESS |
| **Decision rule** | if the upper limit on the false-trust rate exceeds 10 %, the diagnostic may be reported but never used to certify a case |

### 3.5 Primary systems, states, observables, model families

Not "more systems and models". The choices, with the reason each was chosen over
the alternative:

| | choice | why this and not the alternative |
|---|---|---|
| **Gate A1 primary cell** | LJ, (ρ\*, T\*) = (0.79, 1.00), N = 108 | it is the cell every current result is in, so it is the one where a replication failure is most informative; the (0.50, 1.20) and (0.90, 1.20) states and N = 256, 500 are secondary and answer a different question (does the mechanism move with state) |
| **Gate A1 primary observable** | pairs in the 4.0–4.5 Å bin | pre-registered in the current work and the one the counterexample was built against; the full g(r) norm, coordination number, U/N and pressure are secondary readouts recorded in the same run |
| **Gate A2** | SW-Si at 300 K and 1000 K, with angular error bases | silicon because the angular term is the whole point of SW, so a pair-radial-only mechanism fails visibly there; EAM-Cu is deferred, since embedding-density errors are a third mechanism and doing two badly is worse than one properly |
| **Gate B1 primary** | SW-Si, pair-spline and BPNN, budget 1000, 8 seeds each | two families that differ in *what they can represent* rather than in size; 8 seeds because the EGNN observation that motivated this is n = 2 and the first job is to find out whether seed-to-seed observable variance is real |
| **Gate B1 secondary** | E(3)-equivariant network, 4 seeds | it produced the largest apparent effect and the worst equilibration failures, so it is the most interesting and the least trustworthy |
| **Gate B2** | not attempted | it requires DFT reference data this repository has no access to, and the review is right that it should wait for B1 |

### 3.6 Budget, stop-loss, and the no-oracle rules

**Budget.** In units of this machine (four cores), the measured costs are: an
HMC chain of 1500 frames on 108 LJ atoms ≈ 115 s; a Metropolis chain of 700
sweeps ≈ 270 s; equilibration ≈ 56 s; a BPNN fit at budget 400 ≈ 13 s; an
E(3)-equivariant fit ≈ 40 s. Gate A1 as specified (3 states × 3 sizes × 4 field
families × 8 clusters × 6 direct chains) is ≈ 1700 chains ≈ 55 CPU-hours, which
is feasible. Gate B1 at 2 families × 8 seeds × direct validation is ≈ 20
CPU-hours. Gate C at 24 tasks × 10 candidates × direct truth is ≈ 400
CPU-hours, which is not, and is the point at which this stops being a
single-machine project.

**Stop-loss, per gate, stated as what the title, abstract and conclusion become:**

| gate fails | title | abstract | conclusion |
|---|---|---|---|
| A1 (mechanism does not replicate across construction clusters) | drop "error fields"; becomes a note on response estimation in a Lennard-Jones liquid | the counterexample is withdrawn | the study reports a negative replication |
| A2 (angular/embedding arms fail) | add "for pair-radial errors" | scope stated in the first sentence | the mechanism is a property of radial error bases |
| B1 (fitted errors do not show it) | unchanged | "constructible, not observed in fitted models" | the constructive result stands; the error-manifold interpretation is withdrawn |
| C (no regret improvement at matched budget) | "selector" never appears | "diagnostic, not a selector" | the method predicts, and does not yet help choose |

**No-oracle inputs.** A policy at Gate C may see, and nothing else: (a) held-out
energies and forces on a fixed label budget; (b) reference-ensemble frames from
the *reference* potential where one exists, or from a designated cheap
surrogate where it does not, with the frame count charged to the same budget;
(c) committee deviations among the candidate models themselves. It may **not**
see δU, the direct-simulation truth, or any quantity derived from them.

**Shared bias.** A committee mean cancels error common to its members, which is
the failure mode this construction is built to expose, so the stress test is
mandatory rather than optional: candidates are deliberately trained on a common
biased subsample and the policy must either detect the shared component or
abstain. A policy that neither detects nor abstains fails Gate C regardless of
its regret on the unbiased tasks.

**Overlap and abstention.** A response prediction is used only if the Kish
effective sample fraction exceeds 0.2, the largest single weight is below 0.05,
and the second-order ratio is below the frozen threshold. If any fails, the
policy abstains and pays the cost of direct validation for that candidate.
Abstention is scored: a policy that abstains on everything has zero failures and
maximal cost, and the budget matching is what makes that unattractive.

**Cost matching.** The force-RMSE baseline and the response policy are compared
at equal total cost, counting reference labels, trajectory generation, committee
training and evaluation, and any direct validation the policy triggers.

### 3.7 Clean re-runs and sealed provenance

The review is right that `git_dirty: true` on every manifest makes the results
unreconstructable. Two changes, both in this branch:

1. **Manifests now record what was dirty, not just that something was.** Every
   manifest carries `code.dirty_paths` — the porcelain listing — and
   `code.sha256`, a digest over the content of every tracked Python file under
   `atomlab/`, `experiments/` and `scripts/`, sorted by path. The digest changes
   if and only if code that can affect a number changed, so a dirty tree
   containing only an edited README is distinguishable from one containing an
   edited sampler. (`experiments/common.py`.)
2. **The confirmatory runs were launched from a committed tree.** `exp09` and
   `exp10` were both started after committing, and their manifests record the
   commit and the digest.

What remains outstanding: `exp03`, `exp05`, `exp06` and `exp07` were run on
dirty trees and their manifests cannot be repaired retroactively. They will be
re-run on a tagged commit before any of their numbers are used in a submitted
document, and until then `docs/RESULTS.md` marks them as such. Raw trajectories
are not stored — only per-frame observable values and energies — which is a real
limitation for exact reconstruction and is now stated in `docs/methods.md`
rather than left to be discovered.

### 3.8 Verbatim revised wording

**Eq. (5)** (`paper/main.tex`, `\label{eq:cs}`; `docs/theory.md` §3). Replaced by
two numbered statements:

> Factoring the covariance into scales and a correlation is a definition, not an
> estimate:
> $$\Delta\langle A\rangle = -\beta\,\sigma_0(A)\,\sigma_0(\delta U)\,\rho_0(A,\delta U) + O(\delta U^2)$$
> with $\rho_0$ the Pearson correlation under the reference ensemble. This is an
> *equality* at first order, and that is why it indicts the reported metric: the
> magnitude $\sigma_0(\delta U)$ is one factor of three, and $\rho_0 \in [-1,1]$
> is free.
>
> Bounding rather than factoring gives the Cauchy–Schwarz inequality, a
> different and weaker statement:
> $$|\Delta\langle A\rangle| \le \beta\,\sigma_0(A)\,\sigma_0(\delta U) + O(\delta U^2)$$
> obtained by setting $|\rho_0| \le 1$, and attained only when the error field is
> perfectly correlated with the observable — the designed worst case, not the
> generic one.

**"validated"** — every use in connection with the response estimator is
replaced. Old: *"the response estimator is validated against direct sampling"*.
New:

> The response estimator agrees with direct sampling across a factor of fifty in
> shift magnitude. Its calibration at the level of the quoted error bars is
> assessed in exp09 and exp10 and is reported there; the word *validated* is not
> used of it.

The word is retained only where it means what it says: closed-form checks in
`tests/` and `docs/validation.md` (SW cohesive energy exact, LJ lattice sum,
harmonic-oscillator response, BAR on a displaced harmonic pair).

**"external validity"** — removed. Old: *"Three independent settings, all landing
at one standard error. That is the external-validity check for §2, and it
passes."* New:

> The fitted-model arm shares the system, the observable, the reference chain,
> the training pool and the code path with the designed-field arm, so it is not
> an external check. What it establishes is that the response calculation
> reaches training-induced error fields and predicts them at the right size — a
> feasibility result, and the arm's residuals carry the same common offset the
> designed arm does.

**"warning light"** — kept as a name, demoted as a claim. Old: *"(iii) The
second-order term is a computable warning light."* New:

> **(iii) The second-order term is computable, and is a candidate warning
> light.** That it ought to work is not evidence that it does, and the
> measurements here say it does not: exp06 flags two cases trustworthy of which
> one disagrees with direct sampling, and exp09 finds that adding the
> second-order term makes the residuals worse on every measure. Until a
> threshold is frozen on one set of error fields and its false-trust rate
> measured on another, the ratio is a heuristic under test. It is used to flag
> cases for attention and never to certify one.

**"selector"** — removed from every summarising position. Old: *"force error is a
coarse filter, not a selector."* New:

> Within this fixed zoo of designed error fields, and at every window width of
> 3× or below, force RMSE fails to resolve the ordering of observable error
> (median ρ 0.32–0.34) while the response prediction resolves it (median ρ
> 0.90–0.94). Whether this holds among *fitted* models of comparable force error
> is a separate question and requires a pre-registered selection experiment.

The word "selector" does not appear in the title, abstract or conclusions and
will not until Gate C passes.

---

## 4. What I do not accept, and why

One item, and it is a matter of reading two measurements rather than of
principle.

The review groups the exp06 disagreement and the six-chain 35 %/3.7 σ result as
a single "exact reweighting vs direct sampling" failure and calls it the release
blocker. They are not the same measurement and they do not have the same sign.
At 4.25 Å (`exp06`, one chain each side) direct sampling gives a *smaller*
magnitude than the prediction, at every one of seven amplitudes. At 5.25 Å
(`check_between_chain_scatter.py`, six chains each side) direct sampling gives a
*larger* magnitude, by 35 %. No single estimator bias produces both signs; a
chain that has not relaxed produces the first and not the second, and an
overestimating predictor produces the second and not the first.

I agree entirely that this must be resolved before anything is called validated
— that is why `exp10` exists and why it measures both bins. The disagreement is
only about whether the manuscript had one unexplained result or two, and it
matters because the two point at different fixes.

I also note, without disputing the item, that the 3.7 σ figure uses a
within-chain blocking error on the prediction from a single reference chain. If
the prediction's between-chain scatter is comparable to that blocking error, the
significance is arithmetic. `exp10` measures that scatter rather than assuming
either way.

---

## 5. Gate A, made concrete

*(Full design; the primary cell is the one every current result is in, so that a
replication failure is maximally informative.)*

**A1 — pair-radial boundary matrix.** LJ at (ρ\*, T\*) ∈ {(0.50, 1.20), (0.79,
1.00), (0.90, 1.20)}, with N ∈ {108, 256, 500} at the middle state. Fields:
null, aligned, random, high-frequency. Two arms that answer different questions
and are never pooled: a matched-force-RMSE arm (within ±2 %) and a fixed
βσ(δU) calibration arm. Eight construction clusters, each with its own disjoint
evaluation trajectory and six direct chains per field. The cluster is the
experimental unit; common random numbers are permitted only as a declared paired
design with the covariance estimated.

**A2 — angular and many-body.** SW-Si at 300 and 1000 K with pair, angular and
mixed error bases, reading out ADF, tetrahedral order, RDF and pressure. If the
angular arm fails, the title and conclusions are restricted to pair-radial
errors.

**A3 — dynamics, as a separate problem.** The static response formula does not
cover a propagator, so dynamical observables are not a bigger version of this
experiment; they are a different one. Not attempted here, and the
static/dynamic reversal conjecture in the manuscript is marked as a conjecture
that nothing in this repository tests.

---

## 6. Gate C, made concrete

Frozen before any candidate is scored:

**Policies.** (1) minimum force RMSE; (2) a fixed energy+force scalar baseline;
(3) the response policy, minimising `max_j |predicted error_j| / τ_j` over the
observable tolerance vector; (4) random and oracle-best as lower and upper
bounds.

**Loss.** `L = max_j |observable_error_j| / τ_j`; `failure = 1[L > 1]`;
`regret = L_selected − min_candidate L`. The tolerances τ_j are set from what
the observable is used for — for g(r) in a liquid, the width at which a
structural conclusion would change — and are fixed before candidates exist.

**Design skeleton.** 2 systems × 3 states × 4 training-data splits, with the
final task count set by development-task variance and the target regret
precision rather than by the skeleton. Each task's candidate pool spans
families, budgets and seeds.

**Judgement.** Paired hierarchical estimand across tasks, simultaneous
intervals, multiplicity fixed in advance. If calibration is good but regret does
not improve at matched budget, the conclusion is *diagnostic, not a selector*,
and the paper says so in the abstract.

---

## 7. Literature

Both works the review names are real, were not in the bibliography, and change
where the open question sits. Both are now cited:

- Kellner *et al.*, *Errors that matter* (arXiv:2604.24607) — uncertainty-aware
  universal potentials calibrated against experiment, with PET-EXP giving
  observable-level uncertainty at near single-model cost. This is the practical
  layer this repository does not have, and it is closer to done than the
  manuscript implied the field was.
- Gawkowski *et al.*, *Dyna-Mat* (arXiv:2607.03433) — end-to-end
  finite-temperature benchmarking of 15 foundation MLIPs, finding that lower
  force error correlates with lower observable error on average while individual
  systems fail qualitatively. This is the two-regime structure of §3a measured
  on real models, and it is the strongest argument for the review's framing over
  mine.

---

## 8. Status

This response is not a claim that the work is now sound. It is a record of what
was wrong, what has been corrected, what two new experiments found, and what
remains untested. The manuscript should not be treated as submission-ready, and
the review's recommendation to that effect stands.
