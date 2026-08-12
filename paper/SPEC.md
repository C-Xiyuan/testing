# SPEC.md — binding manuscript specification

**Status:** binding. The writing agent must follow this document exactly. Where this
document and any other document in the repository disagree, this document wins, with
one exception: the three RULES below outrank everything, including this specification.

**RULE 1 — No invented numbers.** Every quantity in the manuscript must be traceable to
`results/*/summary.json`, `results/*/records.json`, `results/*/amplitude_records.json`,
`results/*/width_records.json`, `results/*/generalisation.json`, or a figure produced from
them. If a number does not exist, compute it from the JSON with a deposited script, or
leave the claim out. Recomputation is allowed; invention is not.

**RULE 2 — No invented citations.** Cite only the works enumerated in §9 of this
specification. Adding a reference not on that list is a violation, however plausible it
sounds.

**RULE 3 — The negative results are the deliverable.** Three of this study's own
predictions were refuted: the `width^{3/2}` scaling, the proposed smoothness-ratio
metric, and prediction P1 itself. A manuscript that hides, softens, or relegates any of
them to supplementary material is a failed deliverable.

---

## 1. Venue and article type

**Venue: *The Journal of Chemical Physics* (AIP Publishing).**
**Article type: Regular Article.** Not a Communication, not a Note.

### 1.1 Justification

The study is a statistical-mechanics argument — a cumulant expansion of the
free-energy-perturbation identity — validated by controlled sampling of a Lennard-Jones
fluid with analytically constructed perturbations. Its evidence is a designed-perturbation
programme on a classical reference potential plus a ten-member zoo of small models fitted
to that potential. It is not a new architecture, not a dataset, and not a benchmark of
published machine-learned potentials. JCP is the venue whose norms match that object
exactly: it publishes perturbation-theory and sampling methodology validated on
Lennard-Jones systems as a matter of routine, so the deliberate choice of an analytic
reference (which is the methodological crux — `U₀` is known everywhere, not on a test set)
reads there as a design decision rather than as a missing DFT calculation. The venue also
already owns this conversation and most of the machinery: Morrow, Gardner and Deringer's
"How to validate machine-learned interatomic potentials" is JCP 158, 121501 (2023);
Imbalzano *et al.*'s uncertainty-to-observable reweighting is JCP 154, 074102 (2021);
Wu *et al.*'s force-error correction for thermal conductivity is JCP 161, 014103 (2024);
Flyvbjerg and Petersen's blocking estimator — the error bar behind nearly every number
here — is JCP 91, 461 (1989). Regular Articles carry no length limit, which RULE 3 makes
decisive.

**Against the alternatives.** *Phys. Rev. Materials* would ask what material was studied,
and the honest answer is none: liquid argon at one state point was chosen precisely
because it is uninteresting as physics and exact as a reference. *Machine Learning:
Science and Technology* fits the motivation but not the object — the paper advances no
machine learning, and its ten fitted models are deliberately small surrogates of a
classical potential, not the published potentials an MLST or *npj Computational Materials*
referee would expect to see evaluated. *Nature Machine Intelligence* is excluded outright:
its 3500-word Article limit would force the refutations into supplementary material, which
RULE 3 forbids, and its broad-impact bar is not met by a two-seed observation on toy
models. *J. Chem. Theory Comput.* is the correct second choice if JCP declines — same
audience, same tolerance for length, marginally more pressure on the absence of a
chemically realistic system.

### 1.2 Venue-imposed constraints the manuscript must satisfy

| Constraint | Requirement |
|---|---|
| Section order | Title, authors, affiliations, abstract, main text, conclusion, supplementary-material section, acknowledgments, author declarations, data availability statement, appendices, references — in that order |
| Heading numbering | `1.`, `1.1.`, `1.1.1.` (AIP style) |
| Abstract | one paragraph, ≤250 words, no displayed equations, no references, no footnotes, no tables |
| Figures | numbered with **arabic** numerals (Fig. 1, Fig. 2, …) |
| Tables | numbered with **Roman** numerals (Table I, Table II, …) |
| Figure width | single column ≤3.37 in; double column ≤6.69 in; minimum label size 8 pt |
| References | numeric, in order of first citation, cited as superscripts |
| Class file | REVTeX 4.2, `\documentclass[aip,jcp,reprint,amsmath,amssymb,floatfix]{revtex4-2}`; the `abstract` environment must precede `\maketitle`. Verify against AIP's current LaTeX template before submission. |
| Data availability | mandatory; AIP prescribes the wording — see §8 |

Alt text is required for every figure. Provide it as a one-sentence plain-language
description per figure in a file `paper/alt_text.md`.

---

## 2. Title, running title, front matter

**Title (fixed):**

> Error fields, not error norms: how the error in a fitted interatomic potential reaches a physical observable

Do not substitute a different title. Do not use "MLIP" in the title.

**Authors and affiliations:** insert a placeholder block
`[AUTHOR NAME], [AFFILIATION], [EMAIL]`. **Do not invent author names, affiliations,
ORCID identifiers, or funding sources.**

**Acknowledgments:** one paragraph. It must acknowledge nothing that cannot be verified.
If there is no funding to declare, write `[FUNDING STATEMENT TO BE COMPLETED]`. No
dedications.

**Author declarations:** include the three required subsections — Conflict of Interest,
Author Contributions (CRediT), Ethics Approval — with placeholders where the information
is not available. Do not fabricate CRediT assignments; write
`[AUTHOR CONTRIBUTIONS TO BE COMPLETED]`.

---

## 3. Abstract

- **One paragraph, 200–250 words.** Count it and state the count in a comment.
- **No displayed equations.** The central relation must be given inline and in words:
  "the shift in a static observable is minus β times the covariance, under the reference
  ensemble, of the observable with the potential's error field." Inline `Δ⟨A⟩ = −β
  Cov₀(A, δU)` is permitted; a displayed equation is not.
- **No numbered references, no footnotes, no tables.**
- Every nonstandard symbol defined (`β`, `δU`, `A`, `ρ`).
- Must contain, in this order:
  1. the practice being audited (force RMSE reported, physics inferred);
  2. the mechanism (covariance is a projection, force RMSE is a norm of a gradient);
  3. the constructive demonstration — four error fields at force RMSE matched to 0.5%,
     observable shifts from 0.3σ to 16.2σ;
  4. the quantitative validation — 1.01σ rms over eight designed fields, 1.06σ over ten
     fitted models, from reference-ensemble samples with no surrogate simulation;
  5. **the corrected P1, stated plainly in the abstract**: force RMSE ranks well across
     three decades (ρ = 0.83) and is not resolved from zero within a factor-2.6 band
     (ρ = 0.34), where the response prediction retains ρ = 0.94. Force error is a coarse
     filter, not a selector;
  6. **at least one refutation named in the abstract** — the smoothness-ratio metric
     proposed in this work carries no information (ρ = −0.01), and the `width^{3/2}`
     scaling it predicted is wrong (measured exponent 0.56).
- The abstract must not claim generality beyond a single Lennard-Jones state point and a
  pair-radial observable. One clause of scope is required.

---

## 4. Section structure, contents and word targets

Total main-text target: **8,000–9,200 words**, excluding abstract, references, table
contents and figure captions. Section targets are binding to ±20%.

### 1. Introduction — 900 words
- The inferential chain being audited: fit, report held-out force RMSE, run dynamics,
  extract an observable, believe it.
- The structural mismatch in one sentence: `Δ⟨A⟩` is a linear functional of `δU`;
  `RMSE_F²` is a quadratic functional of `∇δU`.
- Why an analytic reference potential rather than DFT: `U₀` known everywhere, `δU` exactly
  computable, reference observables convergeable to arbitrary precision, and perturbations
  designable — which converts an observational claim into a constructive one.
- Prior work in one compressed paragraph, with citations (Fu 2023, Stocker 2022, Morrow
  2023) establishing that the empirical dissociation is known, and (Imbalzano 2021,
  Thaler 2021, Zwanzig 1954) establishing that the machinery is standard. **The novelty
  claim must be stated defensively here and again in §6**: what is new is the reading of a
  textbook identity as an indictment of the reported evaluation metric, the analytic-oracle
  construction, the constructive falsification at matched force RMSE, and the
  across-decades/within-band decomposition.
- A contributions list of 4–6 bullets, each of which is checkable against §10 of this spec.
- A pointer to §5 ("What did not work") in the introduction itself. The reader must learn
  in the first page that three predictions of this study were refuted.

### 2. Theory — 1,300 words
- **2.1 Setup.** `U = U₀ + δU`; `δU` is a function on configuration space, not a number.
  `δF = −∇δU`. Define `RMSE_F² = (1/3N)⟨‖∇δU‖²⟩`.
- **2.2 The exact relation.** `⟨A⟩_U = ⟨A e^{−βδU}⟩₀ / ⟨e^{−βδU}⟩₀`, exact for any `δU`.
  State the consequence: the surrogate ensemble is determined by the joint distribution of
  `(A, δU)` under the reference ensemble and by nothing else.
- **2.3 Cumulant expansion.** Result 1 (full cumulant series) and Result 1a
  (`Δ⟨A⟩ = −β Cov₀(A, δU) + O(δU²)`). Three consequences, each one short paragraph:
  (i) it is a covariance, not a norm — give the Cauchy–Schwarz bound
  `|Δ⟨A⟩| ≤ β σ₀(A) σ₀(δU) |ρ₀(A, δU)|` and note that force RMSE reports neither of the
  three factors; (ii) it needs only reference-ensemble samples, hence one energy evaluation
  per stored frame per model; (iii) the second-order term is a computable warning light.
- **2.4 Vector observables.** The norm is taken *after* the projection — the opposite order
  from force RMSE. Free energy contrasts: Zwanzig's relation is controlled at leading order
  by `⟨δU⟩₀`, so a model can have good free energy and bad structure.
- **2.5 The frequency argument.** `‖δF‖² ~ Σ_k k²|c_k|²` against
  `Cov₀(A, δU) = Σ_k c_k Cov₀(A, φ_k)`, with `Cov₀(A, φ_k)` decaying in `k` because
  physical observables are smooth. Name both failure modes: false alarm (high-frequency
  interpolation wiggle inflates force RMSE, harms nothing) and false confidence
  (low-frequency systematic error is nearly free in force RMSE and moves observables).
- **2.6 A heuristic that failed, retracted in advance.** State the Gaussian-shell scaling
  estimate `RMSE_F ~ a/√w`, `|Cov₀(g, δU)| ~ a·w`, hence `w^{3/2}` at fixed force error —
  **and immediately retract it as a heuristic**, giving both analytic reasons (the
  pair-count-∝-`w` estimate fails once `w` approaches the scale on which `g(r)` varies;
  the coupling saturates and then partially cancels once `w` exceeds the observable's bin
  width). The retraction was made on analytic grounds *before* the measurement; say so.
  Forward-reference §5.1 where the measurement confirms it.
- **2.7 What the theory does not cover.** Dynamical observables: the propagator changes and
  Result 1a does not apply. State the conjecture (force error should be a *better* proxy
  for dynamical than for static observables) explicitly as **pre-registered and untested
  in this work**, cite Wu 2024 as independent literature evidence for the asymmetry in
  transport, and say plainly that the experiment designed to test it was not run.

### 3. Methods — 1,400 words
Every subsection below is required. Contents are fixed by `docs/methods.md`; do not add
procedures that were not performed.
- **3.1 System.** LJ argon, N = 108, cubic PBC, ρ = 0.0200 Å⁻³, T = 120 K; in reduced units
  ρ\* = 0.790, T\* = 1.004. ε = 0.0103 eV, σ = 3.405 Å, cutoff 7.0 Å, **shifted-force**.
  Justify the cutoff mode with Table VII (Appendix A) and state why it matters: a shifted
  (not shifted-force) potential delivers a force impulse at the cutoff, which would be
  indistinguishable from the systematic effect under study.
- **3.2 Sampling.** Hamiltonian Monte Carlo, not molecular dynamics. State the argument in
  full: velocity Verlet is symplectic and time-reversible, so the proposal is symmetric and
  the stationary distribution is exactly `exp(−βU)` regardless of integration error, which
  appears only as reduced acceptance. This forecloses the objection that the thermostat is
  the systematic error. 8 leapfrog steps per proposal, step size adapted in burn-in to
  acceptance 0.75, ≈30 fs after adaptation. Independent single-particle Metropolis sampler
  as a cross-check. Equipartition validation (Table VIII) — report **both** moments, since a
  wrong temperature scale reproduces the mean and fails the variance.
- **3.3 Equilibration, and the failure that made it necessary.** This subsection is
  required and must not be compressed. The fcc-scaled starting configuration is a crystal;
  direct sampling produced a slowly melting crystal: Q₆ falling monotonically 0.5745 → 0.43
  over 600 proposals and still falling, energy climbing and still climbing, ensemble energy
  wrong by 30–40% (−0.036 to −0.043 eV/atom against the equilibrated −0.0311). No sampler
  diagnostic caught it, because acceptance and integration error measure whether the chain
  is simulated correctly, not whether it has reached stationarity. State the protocol
  (melt at 5× target T, anneal, refuse Q₆ > 0.20) and the post-protocol values
  (Q₆ = 0.074, U/atom = −0.0315 eV). State that every production trajectory is split in
  half and a drift above 3 combined standard errors **raises** rather than warns.
- **3.4 Observables.** Pair counts in radial bins, `A_k(x) = |{(i,j): r_k ≤ r_ij < r_{k+1}}|`.
  Explain the choice: response theory applies to a plain function of configuration, and a
  bin count is exactly that, with no normalisation between predicted and measured quantity.
  The designed-perturbation arm additionally targets a **scalar** — the pair count in a
  single bin across the first peak — which makes the covariance a K-vector rather than a
  K × J matrix. **State this as a limitation here as well as in §7**: the headline
  demonstration is about a single-bin observable.
- **3.5 Designed error fields.** Pair error fields expanded in Gaussian shell bumps, making
  the covariance linear in the coefficients, so *null* (first-order effect zero by
  construction), *aligned* (maximal effect per unit force error) and *random* (control) are
  linear algebra rather than optimisation. All scaled to identical force RMSE on the
  reference ensemble. Analytic forces and virial, agreeing with finite differences to ~10⁻¹⁰
  on cubic and sheared triclinic cells. **Out-of-sample discipline is mandatory to state**:
  a null field is orthogonal to the observable on its construction frames by definition, so
  fields are built on one half of the reference trajectory and every reported quantity —
  force RMSE, predicted shift, covariance — is computed on the other half.
- **3.6 Fitted models.** Ten models fitted to 400 configurations of LJ argon, with training
  and test data drawn from separate Markov chains: pair splines (`spline_k8`, `spline_k24`),
  linear ACE-style bases (`linear_l2`, `linear_l4`), Behler–Parrinello networks
  (`bpnn_16x2` and `bpnn_64x2`, two initialisation seeds each), and E(3)-equivariant
  networks (`egnn_c8`, two initialisation seeds). **The GAP-style kernel model was
  implemented but is not included in the reported zoo; say so.** Do not repeat the stale
  claim in `docs/RESULTS.md` §6 that the linear and Behler–Parrinello implementations did
  not complete — the records in `results/exp03_model_zoo/records.json` show that they did.
- **3.7 Uncertainty estimation.** See §5 of this spec. Must name the estimators and state
  that `1/√M` is never used.
- **3.8 Reproducibility and compute.** Per-experiment manifests record the git commit, the
  dirty flag, the full configuration, per-stage wall times, library versions, and seeds
  derived from a single experiment seed through named `SeedSequence` sub-streams. Add one
  sentence of compute reporting (hardware class and total wall time) taken from the
  manifests; if a manifest field is absent, say so rather than estimating.

### 4. Results — 2,400 words
- **4.1 Four error fields at matched force RMSE (Table I, Fig. 1a).** Report both force
  levels from `results/exp07_designed_counterexamples/records.json`. Do **not** write "the
  same pattern holds at 1.0 × 10⁻³ with all shifts scaled down by four" — that holds for
  the predictions (2.552 vs 10.206) and not for the measurements (1.835 vs 9.891). Give the
  actual numbers at both levels. State the two conclusions: the reported metric cannot
  distinguish these four models (force RMSE agrees to 0.5%, verified out of sample:
  1.0005e-3/0.9954e-3 and 4.002e-3/3.981e-3), and the quantity that does distinguish them
  is `ρ(A, δU)`, monotonically (Fig. 1a).
- **4.2 The prediction works (Fig. 2, Table II).** 1.01σ rms residual between predicted and
  measured across all eight designed fields — "agrees at exactly the level the error bars
  claim, neither better nor worse". Emphasise that no simulation with the surrogate was
  involved in the prediction. Then the second-order check: for a null field the first-order
  term vanishes by construction, so any residual is second order, and the measured residual
  agrees with the second-order estimate at both levels (−0.540 and −0.439 pairs, both within
  the error bar). Then the construction-frame budget (Table II): out-of-sample predicted
  shift 0.041 ± 0.067, 0.142 ± 0.071, 0.037 ± 0.077, 0.028 ± 0.078 at 250/500/1000/2000
  frames, against the aligned field's 10.206 at the same force error — a suppression of
  roughly 400×, out of sample. Include the failure at 30 construction frames (1.2–1.6
  pairs): the null space of a covariance estimated from 30 samples is the null space of
  the noise.
- **4.3 Orthogonality is specific to the observable it was built for (Fig. 3).** The null
  field leaves its target bin alone (−0.220 ± 0.640) and moves the largest non-target bin by
  11.36 ± 0.95 pairs, against the aligned field's 11.18 ± 1.21. **This must be presented as
  a limit on the claim as well as a result**: the demonstration shows that force RMSE
  fails to predict a *particular* observable, not that any model is globally harmless.
  Conclude that no single scalar can summarise model quality — not force RMSE and not any
  replacement — and that what is computable is a *vector* of response scores, one per
  observable of interest.
- **4.4 Across decades and within a band (Table III, Fig. 4, Fig. 5).** This subsection
  carries the refutation of P1 and must be written as such, with the original prediction
  quoted from the pre-registration before the result is given. Across 34 designed
  surrogates spanning 5.0 × 10⁻⁴ to 1.80 × 10⁻¹ eV/Å (a factor of 361), the Spearman rank
  correlation between force RMSE and observable error is ρ = 0.83 [0.68, 0.92], n = 34.
  **P1 as stated is not supported, and the corrected claim is stronger**: within the 15
  members whose force RMSE spans a factor of 2.59, observable error still spans a factor
  of 30, force RMSE gives ρ = 0.34 with a 95% interval that includes zero, and the response
  prediction gives ρ = 0.94 with an interval that does not. Force error is a coarse filter,
  not a selector. Then the two secondary findings: the **smoothness ratio proposed in this
  work is refuted** (ρ = −0.01 [−0.37, 0.37], n = 34 — "proposed and refuted in the same
  study"), and per-atom energy spread (ρ = 0.92 [0.81, 0.97]) outranks every force metric,
  which the theory does not predict. Give the per-atom-energy-spread result one sentence of
  physical discussion — it plausibly tracks `σ₀(δU)`, the second factor in the
  Cauchy–Schwarz bound of §2.3 — and label that reading explicitly as a conjecture, not a
  result. Report the noise-subtraction convention (§5.6). **Do not report top-k overlap
  statistics** (they moved between 0.33 and 0.67 depending on the noise correction and are
  too noisy to carry a claim); say in one sentence that they were computed and are withheld
  for that reason.
- **4.5 Fitted models (Table IV).** The response formula holds on models that were actually
  fitted: 1.06σ rms over ten, matching the 1.01σ on designed fields. This is the
  external-validity check for §4.2 and it passes. Then, immediately and without softening:
  **the rank correlations from this arm are not reportable.** Force RMSE spans
  4.0 × 10⁻⁴ to 6.1 × 10⁻³ eV/Å and buys almost no measurable difference; nearly every
  architecture at this data budget on this reference is good enough that its observable
  error is indistinguishable from zero; every confidence interval spans zero (force RMSE
  ρ = 0.43 [−0.35, 0.96], n = 10). Nothing about ranking is concluded from this arm.
  Then the seed pair: `egnn_c8_s0` and `egnn_c8_s1` differ only in initialisation seed;
  force RMSE 0.0045 vs 0.0061 (36%); measured observable error −0.15 ± 0.57 (0.3σ) vs
  −4.10 ± 0.69 (6.0σ), a factor of 27; both predicted from the reference trajectory alone
  (−0.98 and −3.03). **Label this n = 2 explicitly in the running text and in the table
  caption**, cite Bouthillier 2021 on seed variance, and state that repeating it at n ≥ 8
  is the highest-value follow-up. Note that both EGNN records carry
  `linear_trustworthy = false` (second-order ratios 0.41 and 0.14) — the self-diagnostic of
  §2.3(iii) fired on exactly the two models where it should, and the prediction is
  correspondingly quoted as an ordering rather than a precise value.
- **4.6 Damage varies fivefold at fixed force error (Fig. 1b).** Holding force RMSE constant
  (spread 3.3 × 10⁻¹⁵ across the sweep — i.e. exactly) and varying only the width of the
  error field in `r`, the measured observable error spans a factor of 4.7, from 2.13 to
  9.95 pairs; the first-order prediction tracks it across the range. Report the measured
  spread from `width_records.json` rather than the rounded "factor of five".

### 5. What did not work — 1,200 words
This section is required, must carry a heading of its own, and must appear in the main
text before the discussion. It must not be renamed to anything softer.
- **5.1 The `width^{3/2}` scaling is wrong.** Predicted exponent 1.5; measured 0.56
  (direct) and 0.45 (first-order prediction) — off by a factor of three in the exponent.
  The retraction was made analytically before the measurement (§2.6); the measurement
  confirms it. State what survives: force error and observable error depend differently on
  the shape of the error field, so their ratio is not a constant — §4.6 measures that ratio
  varying by 4.7.
- **5.2 The proposed smoothness-ratio metric is refuted.** `std(δU)/RMSE_F`, proposed in
  this work as an empirical stand-in for the inverse spectral weighting, has
  ρ = −0.01 [−0.37, 0.37] against observable error across the zoo, n = 34. It carries no
  information here. (Cross-reference §4.4; do not report it only once.)
- **5.3 An open 35% discrepancy, with three explanations excluded by measurement
  (Table VI).** In the amplitude sweep at the smallest perturbation the three estimates
  disagreed: first order −3.835 ± 0.170, exact reweighting −3.884 (ESS fraction 0.83),
  direct sampling −1.948 ± 0.821. The two sharing reference samples agree to 1.3%; the one
  requiring an independent chain disagrees with both by a factor of two. The obvious
  diagnosis — that the direct estimate's within-chain error bar misses the between-chain
  offset — **was tested and was wrong**: six independent seed pairs give within-chain
  blocking error 0.589, between-chain scatter 0.312, ratio 0.53, so blocking is conservative
  by about a factor of two. The six measured shifts are
  [2.553, 2.773, 2.349, 2.611, 1.788, 2.188], mean +2.377 ± 0.144. On a fresh 3000-frame
  reference chain the same-sample estimators give +1.725 ± 0.101 (first order) and +1.757
  (exact reweighting) — a 35% systematic difference from the direct +2.377. Three
  explanations are excluded: not the error bars (between-chain scatter is half the quoted
  error); not second-order truncation (first-order and the all-orders reweighting identity
  agree to 2%); not incomplete relaxation (that would bias the measured shift toward the
  reference, making it smaller, and it is larger). **State the remaining candidate and that
  it was not tested**: the Kish effective sample size assumes independence and the true
  independent count is smaller by the autocorrelation time; recomputing the ESS as
  `M/(2τ_int)` with the τ the codebase already estimates is the obvious next test and was
  not run. Record the item as unresolved rather than attributing it to the nearest
  plausible cause. Note that the §4.1 measurements used a different observable and twice as
  many frames and are calibrated at 1.01σ, so this is not a general failure of either
  estimator.
- **5.4 The first-order prediction breaks down at large `δU`, as it must (Table V).** A
  four-model arm at a larger data/error regime
  (`results/exp03_model_zoo_n400/summary.json`) gives a prediction residual of **30.4σ**,
  with force RMSE up to 4.34 × 10⁻² eV/Å and observable errors up to 126 pairs; the worst
  case has predicted 0.026 ± 0.251 against measured −5.173 ± 1.863. Report this as the
  demonstrated boundary of validity of linear response, not as a hidden failure. State
  honestly that these records do not carry the `second_order_ratio` /
  `linear_trustworthy` fields, so the self-diagnostic of §2.3(iii) **cannot be checked
  against this arm** — which is a gap in the evidence for the warning-light claim, and must
  be said in those words.
- **5.5 The dilute-gas closed form disagrees by 5.4σ.** At ρ\* = 0.079 the estimator and the
  exact reweighting identity agree to under 2% in every bin; both disagree with a
  hand-derived closed form by 5.4σ rms. Record the correction the hand calculation needed
  and the estimator did not: pair number is conserved, so a perturbation that depletes one
  shell must enrich others, and the pair-separation distribution is normalised — dropping
  that term gives a prediction wrong by a factor of ten in the outer bins and wrong in sign
  in the tail, while a covariance is mean-subtracted and the subtraction *is* the
  normalisation. State that the residual is *most likely* the `O(ρ)` correction to
  `g = e^{−βu}`, that **this is not confirmed**, and that the confirming test — repeating at
  a quarter of the density — was not run.

### 6. Relation to prior work — 900 words
- Fu 2023 and Stocker 2022 established the empirical dissociation. State precisely what
  this paper adds and nothing more: (i) their comparisons are confounded — models differ
  simultaneously in force error, architecture, training data and *magnitude* of error,
  whereas this study decouples magnitude from shape by construction at force RMSE matched
  to 0.5%; (ii) they offer no mechanism, and this paper supplies one and validates it at
  1.01σ / 1.06σ; (iii) their failure mode is dynamical instability — large `δU`,
  out-of-distribution, non-perturbative — while this one is in-distribution, perturbative
  and silent: nothing blows up, the trajectory is fine, and the answer is wrong by 16σ.
  These are different diseases and the paper must say so explicitly.
- **The covariance formula is not claimed as new.** Say so in these terms: it is standard
  thermodynamic perturbation theory (Zwanzig 1954, Bennett 1976), used as a parameter
  gradient in differentiable trajectory reweighting (Thaler 2021) and for propagating
  committee uncertainty to thermodynamic averages (Imbalzano 2021). Frederiksen 2004 is the
  deep precedent for the conceptual claim that potential error is observable-dependent.
  What is new is the four items listed in §1 and nothing else.
- Wu 2024 derives a *norm-based* correction for a transport observable, where `δF` enters
  the equations of motion directly. Contrast it explicitly with the projection mechanism
  for static observables, and connect it to the untested conjecture of §2.7.
- Morrow 2023 is the validation-practice statement this work supports quantitatively.
- Kovács 2021 ("beyond RMSE"), Deng 2025 (systematic softening) and Póta 2024 may be cited
  once each as evidence that practitioners already distrust force RMSE.
- One paragraph on the practical consequence: a vector of response scores, one per
  observable, at one energy evaluation per stored frame per model; and the honest caveat
  that the practitioner has no `U₀`, so the usable substitute is a committee mean
  (Imbalzano 2021) whose failure mode — homogeneous committees share systematic error,
  which cancels in the committee mean — is predictable and **was not tested here**.

### 7. Scope and limitations — 500 words
A single labelled section listing, without euphemism, every limitation in §11 of this
spec. It must include: one reference potential, one state point, one observable class,
N = 108 with no finite-size check, pair-radial error fields only (no angular or many-body
error, which is where real architectures differ), no DFT and no chemically realistic
system, no real published potentials, no dynamical observables, the untested conjecture,
the untested committee predictor, the post hoc band, and n = 2 on the seed pair. State
which of these were foreclosed by design and which by resource limits, honestly and
separately.

### 8. Conclusion — 350 words
JCP requires a conclusion section. Lead it with the refutation: **force error is a coarse
filter, not a selector**, which is the corrected form of this study's own prediction and
not the one it set out to demonstrate. Then the mechanism, then the estimator, then the
scope. Do not introduce a number that has not appeared earlier.

### 9. Supplementary material — 60 words
A short section naming the contents of the deposit: the full 34-member designed-surrogate
record listing, the seven-row amplitude sweep, the six-row width sweep, the four-model
large-`δU` arm, the per-experiment manifests, and the analysis scripts.

### Appendices
- **Appendix A: Simulation and estimator validation.** Table VII (cutoff modes) and
  Table VIII (external and closed-form validation targets). ~250 words.
- **Appendix B: Statistical estimators and their validation.** Blocking, moving-block
  bootstrap, jackknife, and the AR(1) validation. ~250 words.

---

## 5. Uncertainty, significance and sample size — binding rules

1. **No `1/√M`.** Every uncertainty comes from correlated-sample statistics. State this
   once in §3.7 and once in the caption of Table I.
2. **Means:** Flyvbjerg–Petersen blocking, plateau taken over levels retaining at least
   eight blocks. Cite Flyvbjerg 1989.
3. **Covariances, correlations and fitted slopes:** moving-block bootstrap, block length =
   four integrated autocorrelation times of the **product** series that carries the error
   (not the observable's own τ, which underestimates whenever `δU` varies slowly — say
   why). Cite Künsch 1989 and Politis–Romano 1994. State the number of bootstrap resamples
   (2000 for the values in `results/exp05_proxy_correlation/summary.json`) and the seed for
   any newly computed interval.
4. **Define `±` once, explicitly, as a standard error** (not a standard deviation), in
   §3.7 and in the caption of Table I. Ambiguous `±` is a violation.
5. **Every correlation is reported as `ρ = value [low, high], n = N`.** No bare ρ anywhere
   in the manuscript, including in figure titles quoted in the text, including the
   within-band number. `n` must be stated: n = 34 (zoo), n = 15 (band), n = 10 (fitted
   models), n = 8 (designed fields), n = 6 (seed pairs), n = 4 (large-`δU` arm), n = 2
   (EGNN seed pair).
6. **The within-band correlations must be recomputed with intervals.**
   `results/exp05_proxy_correlation/summary.json` contains no band statistics, and
   `scripts/make_band_figure.py` reports the band ρ as a bare point estimate. Deposit a
   script that computes, on the same series the figure uses, the band ρ for force RMSE and
   for the response prediction with a 95% percentile bootstrap interval over the 15 band
   members, with a fixed seed, and report both. **If the force-RMSE band interval includes
   zero, the manuscript must say so and must phrase the finding as "not resolved from
   zero", never as "demonstrably zero" or "force error fails".** The contrast that survives
   is that the response prediction's interval excludes zero while force RMSE's does not.
7. **Two noise conventions exist and must be reconciled.** The correlations in
   `summary.json` are computed on the raw observable-error norm; the regime figure
   subtracts the sampling noise in quadrature (`|v_true|² = |v_meas|² − E|noise|²`). The
   two differ by under 0.01 in ρ. Pick the noise-subtracted convention for all reported
   correlations, state the convention in one sentence in §3.7, and report the raw-series
   values from `summary.json` in a footnote to Table III as a robustness check. Explain in
   one sentence why the subtraction is necessary at the low-force-error end, where signal
   and noise are comparable.
8. **Significance as σ-distances.** Quote as `value ± error (X.Xσ)`, computed as
   |value|/error, to one decimal place. Do not report p-values.
9. **Digits.** Uncertainty to two significant figures; the central value truncated to the
   same decimal place. Do not quote more significant figures than the uncertainty supports.
   Exception: exactness checks (e.g. −4.3366000 eV/atom against exactly −2ε) may carry
   their full digits, and must be labelled as exactness checks.
10. **Error bars on every point of every figure** that plots a sampled quantity. Where a
    published figure lacks them, say so in the caption rather than implying they are absent
    from the data.
11. **The `linear_trustworthy` / `second_order_ratio` self-diagnostic must be reported
    wherever it exists** (Table IV: 8 of 10 fitted models trustworthy; both EGNN records
    not), and its absence must be reported where it does not exist (§5.4, the large-`δU`
    arm).
12. **Withheld statistics must be named.** Top-k overlap was computed and is withheld;
    say so and say why (0.33–0.67 depending on the noise correction).

---

## 6. Figures

Six figures, all in the main text, in this order. Source files are in
`/home/user/testing/figures/`. **Submit the `.pdf` (vector) versions**; the `.png` versions
are for drafts only. Do not create new figures; do not modify the plotted data. Captions
must be self-contained, must define every symbol and every panel, and must state the
uncertainty convention where error bars appear.

| # | File | Section | Must show |
|---|---|---|---|
| **1** | `headline_mechanism.pdf` | 4.1, 4.6 | Two panels. (a) measured shift in the target pair count against `ρ(A, δU)` for the designed fields at matched force RMSE — the monotone, near-linear relation the covariance formula predicts. (b) `‖Δ⟨A⟩‖` against the width of the error field at fixed force RMSE, spanning a factor of 4.7. Caption must state that force RMSE is held fixed in both panels and give the residual spread (3.3 × 10⁻¹⁵) in panel (b). |
| **2** | `headline_prediction.pdf` | 4.2 | Single panel: measured shift (direct sampling) against predicted shift (`−β Cov₀(A, δU)`, no surrogate simulation) for all eight designed fields, with the identity line. Caption must state the 1.01σ rms residual, n = 8, and that the prediction used only reference-ensemble samples. |
| **3** | `exp07_designed_counterexamples_counterexamples.pdf` | 4.3 | Shift in pair count against `r` for null, aligned and random fields, one panel per force level (1 × 10⁻³ and 4 × 10⁻³ eV/Å). Caption must state that the null field's target bin is the one it was constructed against, and that its largest non-target excursion (11.36 ± 0.95) is comparable to the aligned field's (11.18 ± 1.21). |
| **4** | `headline_regimes.pdf` | 4.4 | Two panels. (a) observable error against force RMSE across all 34 designed surrogates on log–log axes, with the 1.5–6.0 × 10⁻³ eV/Å band shaded, ρ across decades in the panel title. (b) observable error for the 15 band members ordered by ascending force RMSE, showing 30× variation over a 2.6× force-RMSE range. Caption must give both ρ values **with intervals and n**, and must state that the band is analyst-chosen and post hoc. |
| **5** | `exp05_proxy_correlation_proxy_correlation.pdf` | 4.4 | Three panels: (a) observable error against force RMSE with error bars; (b) measured against predicted observable error; (c) Spearman ρ with 95% intervals for every proxy metric. Panel (c) is where the smoothness-ratio refutation and the per-atom-energy-spread result are visible; the caption must name both. Caption must cross-reference Fig. 4(a) rather than re-reporting its number. |
| **6** | `exp06_response_validation_response_validation.pdf` | 5.1, 5.3 | Two panels. (a) first-order, reweighted and direct estimates against perturbation amplitude — where linear response holds and where it stops. (b) `‖shift‖` at fixed force RMSE against the width of the error field, from which the exponent 0.56 (direct) / 0.45 (first-order) is fitted against the predicted 1.5. Caption must state the predicted and measured exponents and that the prediction is refuted. Cross-reference Fig. 1(b) for the same width sweep rather than re-reporting the dynamic range. |

**Not used:** none. All six available figures are used. No figure in
`/home/user/testing/figures/` may be omitted, and no new figure may be introduced.

Where panels overlap (Fig. 1b with Fig. 6b; Fig. 4a with Fig. 5a), the caption must
cross-reference and the running text must report the number once only.

---

## 7. Tables

Roman numerals, AIP style, each with a caption understandable without the text, each cited
in the running text in order.

| # | Section | Contents | Source |
|---|---|---|---|
| **I** | 4.1 | Designed fields at matched force RMSE. Rows: null, random (seed 1), random (seed 0), aligned, at both force levels. Columns: field, out-of-sample force RMSE, `ρ(A, δU)`, predicted shift, measured shift ± SE, σ-distance. Caption must define `±` as a blocking standard error and state the matching tolerance (0.5%). | `results/exp07_designed_counterexamples/records.json` |
| **II** | 4.2 | Null-space construction budget: construction frames (250, 500, 1000, 2000), in-sample predicted shift, out-of-sample predicted shift ± SE. Caption must give the aligned field's 10.206 at the same force error for scale. | `results/exp07_designed_counterexamples/summary.json` (`null_space_generalisation`) and `generalisation.json` |
| **III** | 4.4 | Regimes. Columns: across the zoo (n = 34) and within the band (n = 15). Rows: force-RMSE spread (361× / 2.59×), observable-error spread (30× in band), ρ(force RMSE, truth) with interval, ρ(response prediction, truth) with interval, ρ(smoothness ratio, truth) with interval, ρ(per-atom energy spread, truth) with interval. Footnote giving the raw-series values from `summary.json`. Caption must state that the band is post hoc. | `results/exp05_proxy_correlation/summary.json`, `records.json`, plus the newly deposited band-interval script |
| **IV** | 4.5 | The ten fitted models. Columns: name, architecture, force RMSE (eV/Å), predicted shift, measured shift ± SE, σ-distance, second-order ratio, linear-trustworthy flag. Caption must state that the rank correlations from this arm are not reportable, quote force RMSE ρ = 0.43 [−0.35, 0.96], n = 10, and label the EGNN pair as n = 2. | `results/exp03_model_zoo/records.json`, `summary.json` |
| **V** | 5.4 | The large-`δU` arm: four models, force RMSE, predicted, measured ± SE, and the 30.4σ aggregate residual. Caption must state that these records lack the second-order self-diagnostic. | `results/exp03_model_zoo_n400/summary.json`, `records.json` |
| **VI** | 5.3 | The open discrepancy. Rows: first order (−3.835 ± 0.170), exact reweighting (−3.884, ESS fraction 0.83), direct sampling (−1.948 ± 0.821); then within-chain blocking error 0.589, between-chain scatter 0.312, ratio 0.53, six-seed mean +2.377 ± 0.144, fresh-chain first order +1.725 ± 0.101 and reweighting +1.757. Caption must state that three explanations are excluded by measurement and the item is unresolved. | `results/exp06_response_validation/amplitude_records.json`, `docs/RESULTS.md` §5.2 (values traceable to `scripts/check_between_chain_scatter.py`) |
| **VII** | Appendix A | Cutoff modes: truncated, shifted, shifted-force, switched; `|Δu|` and `|ΔF|` measured at `r_c ∓ 10⁻⁷` Å. | `docs/methods.md` §1 |
| **VIII** | Appendix A | Validation against external and closed-form targets: SW Si cohesive energy, LJ fcc lattice sum, EAM Cu lattice constant / cohesive energy / bulk modulus, fcc/bcc/sc Q₆, SOAP and ACSF rotational invariance, HMC ⟨U⟩ and var(U) against equipartition, phonon dispersion. Caption must note the EAM observation — the potential reproduces its three fitted targets exactly and still gives a maximum phonon frequency 30% below copper's — as the same statement this paper makes about force error, arriving from a different direction. | `docs/RESULTS.md` §7, `docs/validation.md` |

Any value in Table VI or VIII that cannot be located in a `results/*.json` file must be
recomputed from the deposited scripts before it is printed, or removed. If a value in
`docs/RESULTS.md` cannot be traced to either, delete the row and say in the caption that
the check was performed but its artefact was not retained.

---

## 8. Data and code availability statement

Required by AIP, placed **after the acknowledgments and author declarations and before the
appendices**. Use AIP's repository-with-DOI template verbatim, extended with the code
sentence. Reproduce exactly this text, filling the bracketed fields before submission:

> **DATA AVAILABILITY**
>
> The data that support the findings of this study are openly available in Zenodo at
> http://doi.org/[DOI], reference number [DEPOSIT ID]. The deposit contains the complete
> `atomlab` source code, the experiment configurations, and the raw outputs of every
> experiment reported here — `summary.json`, `records.json`, and the per-experiment
> manifests, each of which records the git commit that produced it, whether the working
> tree was dirty at the time, the full configuration, per-stage wall times, library
> versions, and the random seed. No external dataset was used, downloaded, or required:
> every configuration analysed here was generated by the deposited code from the stated
> seeds. All analysis scripts that produce the figures and tables in this article are
> included in the same deposit.

Constraints on this statement:
- The DOI and deposit identifier are the **only** placeholders permitted. Do not invent
  either.
- Do not substitute a bare GitHub URL. A DOI-minting deposit is required.
- Do not weaken "openly available" to "available on reasonable request". Nothing here is
  restricted, and a paper whose argument is about auditability cannot use the weaker
  template.
- Cite the deposit as a numbered reference as well as naming it here.

---

## 9. References

**Style.** Numeric, in order of first citation, cited as superscript numerals (AIP/JCP
style). Format: author initials + surname, journal, volume, page, year, DOI. Use BibTeX
with the REVTeX 4.2 AIP style; paste the compiled `.bbl` inline at final submission. Do
not hand-number references.

**Target count: 40–55.** Expect roughly 45.

**The permitted bibliography is exactly the list below.** Every entry was verified by the
literature investigation. Adding anything not on this list violates RULE 2. Omitting an
entry is permitted; adding is not.

*Evaluation-gap literature (all must be cited):*
Fu, Wu, Wang, Xie, Keten, Gómez-Bombarelli, Jaakkola, *Transactions on Machine Learning
Research* (2023), arXiv:2210.07237 · Stocker, Gasteiger, Becker, Günnemann, Margraf,
*Mach. Learn.: Sci. Technol.* **3**, 045010 (2022) · Morrow, Gardner, Deringer,
*J. Chem. Phys.* **158**, 121501 (2023) · Kovács, van der Oord, Kucera, Allen, Cole,
Ortner, Csányi, *J. Chem. Theory Comput.* **17**, 7696 (2021) · Wu, Zhou, Dong, Ying,
Wang, Song, Fan, Xiong, *J. Chem. Phys.* **161**, 014103 (2024) · Póta, Ahlawat, Csányi,
Simoncelli, arXiv:2408.00755 (2024) · Deng, Choi, Zhong, Riebesell, Anand, Li, Jun,
Persson, Ceder, *npj Comput. Mater.* **11** (2025).

*Uncertainty propagation to observables (Imbalzano and Thaler must both be cited):*
Imbalzano, Zhuang, Kapil, Rossi, Engel, Grasselli, Ceriotti, *J. Chem. Phys.* **154**,
074102 (2021) · Thaler, Zavadlav, *Nat. Commun.* **12**, 6884 (2021) · Frederiksen,
Jacobsen, Brown, Sethna, *Phys. Rev. Lett.* **93**, 165501 (2004) · Perez, Subramanyam,
Maliyov, Swinburne, *npj Comput. Mater.* **11**, 263 (2025).

*Architectures and descriptors (cite those actually relevant to the fitted zoo and to the
framing; do not cite architectures the study did not implement):*
Behler, Parrinello, *Phys. Rev. Lett.* **98**, 146401 (2007) · Behler, *J. Chem. Phys.*
**134**, 074106 (2011) · Bartók, Payne, Kondor, Csányi, *Phys. Rev. Lett.* **104**, 136403
(2010) · Bartók, Kondor, Csányi, *Phys. Rev. B* **87**, 184115 (2013) · Drautz, *Phys. Rev.
B* **99**, 014104 (2019) · Thompson, Swiler, Trott, Foiles, Tucker, *J. Comput. Phys.*
**285**, 316 (2015) · Batzner *et al.*, *Nat. Commun.* **13**, 2453 (2022) · Batatia,
Kovács, Simm, Ortner, Csányi, NeurIPS 35 (2022) · Schütt, Sauceda, Kindermans,
Tkatchenko, Müller, *J. Chem. Phys.* **148**, 241722 (2018) · Gilmer, Schoenholz, Riley,
Vinyals, Dahl, ICML (2017).

*Reference potentials:*
Jones, *Proc. R. Soc. Lond. A* **106**, 463 (1924) · Verlet, *Phys. Rev.* **159**, 98
(1967) · Rahman, *Phys. Rev.* **136**, A405 (1964) · Stillinger, Weber, *Phys. Rev. B*
**31**, 5262 (1985) · Daw, Baskes, *Phys. Rev. B* **29**, 6443 (1984) · Finnis, Sinclair,
*Philos. Mag. A* **50**, 45 (1984) · Cleri, Rosato, *Phys. Rev. B* **48**, 22 (1993).
**Finnis–Sinclair and Cleri–Rosato are mandatory if the EAM implementation is described**:
its embedding function is the second-moment tight-binding form, not the Daw–Baskes
tabulated form, and citing only Daw–Baskes would be a misattribution.

*Statistical machinery:*
Zwanzig, *J. Chem. Phys.* **22**, 1420 (1954) · Bennett, *J. Comput. Phys.* **22**, 245
(1976) · Shirts, Chodera, *J. Chem. Phys.* **129**, 124105 (2008) · Duane, Kennedy,
Pendleton, Roweth, *Phys. Lett. B* **195**, 216 (1987) · Neal, in *Handbook of Markov
Chain Monte Carlo* (2011) · Metropolis, Rosenbluth, Rosenbluth, Teller, Teller,
*J. Chem. Phys.* **21**, 1087 (1953) · Flyvbjerg, Petersen, *J. Chem. Phys.* **91**, 461
(1989) · Künsch, *Ann. Stat.* **17**, 1217 (1989) · Politis, Romano, *J. Am. Stat. Assoc.*
**89**, 1303 (1994) · Kish, *Survey Sampling* (Wiley, 1965) · Steinhardt, Nelson,
Ronchetti, *Phys. Rev. B* **28**, 784 (1983) · Lechner, Dellago, *J. Chem. Phys.* **129**,
114707 (2008) · Swope, Andersen, Berens, Wilson, *J. Chem. Phys.* **76**, 637 (1982) ·
Lebowitz, Percus, Verlet, *Phys. Rev.* **153**, 250 (1967) · Frenkel, Smit, *Understanding
Molecular Simulation*, 2nd ed. (Academic Press, 2002) · Allen, Tildesley, *Computer
Simulation of Liquids*, 2nd ed. (OUP, 2017) · Hansen, McDonald, *Theory of Simple Liquids*,
4th ed. (Academic Press, 2013).

*Machine-learning reporting norms (cite Bouthillier at the seed pair; the others are
optional):*
Bouthillier *et al.*, MLSys (2021), arXiv:2103.03098 · Pineau *et al.*, *J. Mach. Learn.
Res.* **22**, 1 (2021) · Kapoor *et al.*, *Sci. Adv.* (2024), doi:10.1126/sciadv.adk3452 ·
Maxson, Soyemi, Chen, Szilvási, *J. Phys. Chem. C* (2024),
doi:10.1021/acs.jpcc.4c00028 · Soares *et al.*, *J. Chem. Inf. Model.* **63**, 3227 (2023).

**Explicitly forbidden:**
- Grossfield *et al.*, *Living J. Comput. Mol. Sci.* — the DOI was not verified. Do not
  cite it.
- Any sentence of the form "benchmark studies have reported this dissociation" without a
  named citation. `docs/theory.md` §9 currently contains one; it must be replaced by Fu
  2023 and Stocker 2022 or deleted.
- Kish 1965 must be cited only for the effective-sample-size construction, which is where
  it is used.
- Do not cite MACE, SchNet, GemNet, DimeNet or SNAP as if they were evaluated here. They
  were not. Cite them only as background for what the field's architectures are.

---

## 10. What this paper claims

In descending order of evidential strength. The manuscript must not present a weaker
claim as though it were a stronger one, and the abstract must not lead with anything below
item 5.

1. **The response identity.** For a static observable, `Δ⟨A⟩ = −β Cov₀(A, δU) + O(δU²)`,
   with the full cumulant series available. This is a derivation, and it is standard
   thermodynamic perturbation theory — claimed as *correctly applied*, never as *new*.
2. **The identity is quantitatively accurate on this system.** 1.01σ rms residual between
   prediction and direct measurement over eight designed error fields (n = 8); 1.06σ over
   ten fitted models (n = 10). The prediction requires only reference-ensemble samples: one
   energy evaluation per stored frame per model, no simulation with the surrogate.
3. **Where the first-order term vanishes by construction, the second-order term of the same
   expansion accounts for the residual** — measured residuals −0.540 and −0.439 pairs at
   the two force levels, both within the error bar of the second-order estimate.
4. **At force RMSE matched to 0.5%, the observable error can be moved from 0.3σ to 16.2σ by
   changing only the correlation `ρ(A, δU)`** — null −0.220 ± 0.640 against aligned
   +9.891 ± 0.611 at 4.0 × 10⁻³ eV/Å. The reported metric cannot distinguish these models;
   the correlation can, monotonically. The null-space construction survives out of sample,
   with a ~400× suppression relative to the aligned field at 2000 construction frames.
5. **Force error is a coarse filter, not a selector.** Across three decades of force RMSE,
   ρ = 0.83 [0.68, 0.92], n = 34 — a strongly ranked quantity. Within a factor-2.6 band,
   observable error still spans 30×, force RMSE gives ρ = 0.34 with an interval that
   includes zero, and the response prediction gives ρ = 0.94 with an interval that does
   not. **This refutes prediction P1 as originally stated and replaces it with something
   more useful.** It must be presented as a refutation of this study's own pre-registered
   claim, in those words.
6. **Harmlessness is a relation between an error and a question, not a property of a
   model.** The null field leaves its target bin untouched and moves the largest non-target
   bin by 11.36 ± 0.95 pairs, comparable to the aligned field's 11.18 ± 1.21. No scalar can
   summarise model quality; a vector of response scores can be computed cheaply.
7. **At fixed force RMSE, varying only the width of the error field varies the damage by a
   factor of 4.7** (2.13 to 9.95 pairs), tracked by the first-order prediction.
8. **The smoothness-ratio metric proposed in this work is refuted**
   (ρ = −0.01 [−0.37, 0.37], n = 34) and **the `width^{3/2}` scaling it predicted is
   refuted** (measured exponent 0.56 direct, 0.45 predicted, against 1.5). Both were this
   study's own predictions. These are claims — negative ones — and they must be reported as
   results, not as caveats.
9. **The linear prediction has a demonstrated boundary of validity**: a 30.4σ prediction
   residual in the large-`δU` arm (n = 4), reported as the expected breakdown of linear
   response, with the honest admission that the second-order self-diagnostic was not
   recorded for those runs and so cannot be checked against them.
10. **Illustration, not statistic:** two E(3)-equivariant networks differing only in
    initialisation seed have force RMSE differing by 36% and observable error differing by
    a factor of 27 (−0.15 ± 0.57 against −4.10 ± 0.69), both predicted from the reference
    trajectory alone. **n = 2.** This is the paper's most quotable observation and its
    weakest evidence, and both facts must appear in the same paragraph.

---

## 11. What this paper must not claim

Each item below is a specific overreach the results do not support. A manuscript
containing any of them fails compliance.

1. **Not "force error is useless."** It ranks models well across three decades,
   ρ = 0.83 [0.68, 0.92]. Any phrasing implying force RMSE carries no information —
   including in the title, abstract, conclusion or a figure caption — is a violation. The
   permitted phrasing is *coarse filter, not selector*.
2. **Not "force error fails within the band."** The within-band interval includes zero.
   The claim is that force RMSE is **not resolved from zero** there while the response
   prediction is resolved and high. Do not upgrade an unresolved correlation into a
   demonstrated absence.
3. **Not that the band is anything but post hoc.** The window 1.5–6.0 × 10⁻³ eV/Å is
   analyst-chosen and hard-coded in `scripts/make_band_figure.py`. It was not
   pre-registered. Say so where the result is first reported, in the caption of Fig. 4, and
   in §7.
4. **Not that the fitted-model arm demonstrated anything about ranking.** Every confidence
   interval from that arm spans zero (force RMSE ρ = 0.43 [−0.35, 0.96], force MAE
   ρ = 0.35 [−0.49, 0.92], energy RMSE ρ = 0.05 [−0.77, 0.76], smoothness ratio
   ρ = −0.53 [−0.86, 0.06], response prediction ρ = 0.18 [−0.70, 0.83], n = 10). The arm
   establishes exactly one thing: the response formula's residual is 1.06σ on models that
   were actually fitted. Nothing about ranking is concluded, and the manuscript must say
   "nothing about ranking can be concluded from it, and nothing is."
5. **Not that the seed-pair result generalises.** n = 2. Do not write "seeds differ by an
   order of magnitude in observable error" as a general statement, do not extrapolate to
   other architectures or systems, and do not place it in the abstract as evidence.
6. **Not that the covariance formula is novel.** It is Zwanzig's free-energy-perturbation
   identity expanded in cumulants, used as a parameter gradient by Thaler and Zavadlav
   (2021) and for committee-uncertainty propagation by Imbalzano *et al.* (2021). Claim
   novelty only for the evaluation-protocol reading, the analytic-oracle construction, the
   constructive falsification at matched force RMSE, and the across-decades/within-band
   decomposition.
7. **Not that this restates or merely confirms Fu *et al.* (2023).** Their failure mode is
   dynamical instability at large, out-of-distribution `δU`. This one is in-distribution,
   perturbative and silent. State the distinction; do not claim to have reproduced their
   benchmark, because no published potential and no dynamical metric was evaluated here.
8. **Not that the results transfer** to other reference potentials, other state points,
   other observable classes, DFT-quality references, real published machine-learned
   potentials, or angular and many-body error structures. Every reported number is
   Lennard-Jones argon at N = 108, ρ\* = 0.790, T\* = 1.004, with a pair-radial error field
   and a pair-count observable. Stillinger–Weber and EAM implementations exist and are
   validated, and **no observable-error result was produced on either** — say that, do not
   imply otherwise by describing them in the methods without qualification.
9. **Not a finite-size claim.** N = 108 only. No size study was run. Do not assert the
   covariance mechanism is size-independent; state that the check is outstanding, and
   note that pair-count fluctuations in a closed NVT cell carry an ensemble dependence
   (Lebowitz, Percus and Verlet 1967).
10. **Not that the dynamical conjecture was tested.** `experiments/exp05` as designed —
    rank correlations separated into static and dynamical observables — was never run. The
    conjecture of §2.7 is pre-registered and open. Do not present the static results as
    bearing on it.
11. **Not that the committee predictor works.** The practical, oracle-free estimator of
    `docs/theory.md` §7.4 was designed and not tested; `experiments/exp08` did not run. Its
    predicted failure mode — homogeneous committees share systematic error, which cancels
    in the committee mean — is a hypothesis, not a finding.
12. **Not that the 35% discrepancy of §5.3 is explained.** Three explanations are excluded
    by measurement; the remaining candidate (correlated-sample behaviour of the reweighting
    estimator, with the Kish ESS assuming independence) was not tested. Do not attribute
    the discrepancy to it. Do not omit the discrepancy.
13. **Not that the 5.4σ dilute-gas residual is understood.** The `O(ρ)` reading is stated as
    likely and explicitly unconfirmed; the confirming test was not run.
14. **Not that the null-space field is a harmless model.** It is harmless for one bin and
    damaging elsewhere. Do not describe it as "an error field with no physical
    consequence".
15. **Not that top-k overlap supports anything.** It was computed, it is unstable, it is
    withheld.
16. **Not that the study is incomplete or in progress.** `README.md` says "Under active
    construction" and `docs/RESULTS.md` §6 lists unfinished arms; that framing is correct
    internally and wrong in a manuscript. Present every gap as **scope**, in §7, not as
    work in progress. Correspondingly, do **not** repeat `docs/RESULTS.md` §6's stale
    statement that the linear/ACE and Behler–Parrinello implementations did not complete —
    the records show they did, and that concession would gratuitously concede the external
    validity that §4.5 actually supplies.
17. **Not any claim about melting points, elastic constants, phonon spectra, diffusivities
    or vibrational spectra of the surrogate models.** None were measured.

---

## 12. Compliance checklist

The compliance reviewer will score the manuscript against these numbered items. Each is
pass/fail.

**Venue and format**
1. Manuscript is a JCP Regular Article, REVTeX 4.2 with the `aip,jcp` class options, and
   the `abstract` environment precedes `\maketitle`.
2. Section order is: title, authors, affiliations, abstract, main text, conclusion,
   supplementary material, acknowledgments, author declarations, data availability
   statement, appendices, references.
3. Headings are numbered `1.`, `1.1.`; figures use arabic numerals; tables use Roman
   numerals.
4. Every figure and table is cited in the running text in order of first appearance.
5. Main text is 8,000–9,200 words excluding abstract, references, captions and table
   contents; the count is stated in a source comment.
6. Alt text is supplied for all six figures in `paper/alt_text.md`.

**Abstract**
7. One paragraph, 200–250 words, no displayed equations, no references, no footnotes, no
   tables; every nonstandard symbol defined.
8. The abstract states the corrected P1 (ρ = 0.83 across decades, ρ = 0.34 within the
   band) and names at least one refuted prediction of this study.
9. The abstract carries an explicit scope clause naming the single system and observable
   class.

**Numbers and traceability**
10. Every numeric claim in the manuscript is traceable to a named file under `results/`, to
    a figure produced from those files, or to a deposited script that recomputes it from
    them. No exceptions.
11. No number appears that is not in those sources — in particular no rounded restatement
    that changes a value (e.g. "scaled down by four" for the 1 × 10⁻³ measurements, which
    is false).
12. Where `docs/RESULTS.md` and a `results/*.json` file disagree, the JSON wins and the
    discrepancy is either resolved or stated.

**Uncertainty and statistics**
13. `±` is defined once, explicitly, as a standard error, in the methods and in the caption
    of Table I.
14. The methods state that uncertainties come from Flyvbjerg–Petersen blocking (means) and
    a moving-block bootstrap at four integrated autocorrelation times of the product series
    (covariances, correlations, slopes), with a jackknife cross-check, and that `1/√M` is
    never used.
15. Every correlation in the manuscript is reported as `ρ = value [low, high], n = N`. No
    bare ρ anywhere, including figure captions and panel titles quoted in the text.
16. The within-band correlations are reported **with bootstrap intervals**, computed by a
    deposited script with a stated seed and resample count.
17. If the within-band force-RMSE interval includes zero, the manuscript says "not resolved
    from zero" and does not claim demonstrated failure.
18. The noise-subtraction convention is stated once, applied consistently, and the raw-series
    values from `summary.json` appear in a footnote to Table III.
19. σ-distances are quoted as |value|/error to one decimal place; no p-values appear.
20. Uncertainties carry two significant figures and central values are truncated to match;
    exactness checks are labelled as such.
21. Sampled quantities plotted in figures carry error bars, or the caption states why they
    do not.
22. The second-order self-diagnostic is reported where recorded (Table IV) and its absence
    is reported where it is not (Table V, §5.4).
23. Top-k overlap is named as computed-and-withheld, with the reason, and is not used as
    evidence.

**Negative results (RULE 3)**
24. §5 "What did not work" exists in the main text, with its own heading, before the
    discussion, and is not renamed to anything softer.
25. The `width^{3/2}` refutation is reported with both measured exponents (0.56, 0.45)
    against the predicted 1.5, and the theory section retracts the heuristic in advance.
26. The smoothness-ratio refutation is reported with ρ = −0.01 [−0.37, 0.37], n = 34, and
    is identified as this study's own proposal.
27. The refutation of P1 is stated as a refutation of this study's own pre-registered
    prediction, quoting the prediction first.
28. The open 35% discrepancy is reported with the three excluded explanations, the
    between-chain measurement that overturned the obvious diagnosis, and the untested
    remaining candidate, and is labelled unresolved.
29. The 30.4σ large-`δU` breakdown is reported in the main text, with the admission that
    the self-diagnostic cannot be checked against those runs.
30. The 5.4σ dilute-gas disagreement is reported and labelled unconfirmed.
31. The equilibration failure (Q₆ 0.5745 → 0.43, energy wrong by 30–40%, no sampler
    diagnostic caught it) is reported in the methods in full.

**Claims discipline**
32. Every item in §10 that appears in the manuscript is stated at the strength given there
    and no higher.
33. No item in §11 appears in any form, in any section, including captions and the
    conclusion. In particular: force error is never called useless; the fitted-model arm is
    never used to support a ranking claim; the seed pair is labelled n = 2 wherever it
    appears; the band is labelled post hoc wherever it appears.
34. §7 "Scope and limitations" exists, lists every limitation in §11, and separates
    design choices from resource limits.
35. The novelty claim is stated defensively in both the introduction and §6, naming
    Zwanzig, Thaler–Zavadlav and Imbalzano *et al.* as prior users of the same identity.

**References and availability**
36. Every reference is on the permitted list in §9; the count is 40–55; none of the
    forbidden items appears.
37. No sentence of the form "benchmark studies have reported…" appears without a named
    citation.
38. Finnis–Sinclair and Cleri–Rosato are cited if the EAM embedding function is described.
39. Bouthillier *et al.* is cited at the seed-pair result.
40. The data availability statement appears in the required position, uses the AIP
    repository-with-DOI wording given in §8 verbatim, mentions the code and the manifests,
    contains no placeholder except the DOI and deposit identifier, and is accompanied by a
    numbered reference to the deposit.

**Integrity**
41. No author name, affiliation, funding source, ORCID, CRediT assignment or
    acknowledgment is invented; unavailable items are bracketed placeholders.
42. No claim is made about an experiment that was not run: exp05's static/dynamical split,
    exp08's committee predictor, the Stillinger–Weber and EAM observable arms, the
    finite-size check, and the quarter-density dilute-gas check are all named as not run.
