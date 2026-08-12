# Results

This document contains both historical results and their current audit status.
Methods are in `docs/methods.md`, the argument being tested in
`docs/theory.md`, and outputs in `results/`.  The exp09/exp10/exp11 production
artefacts are **legacy exploratory evidence**: their manifests captured
repository state at run completion rather than start, so they do not prove
which source was executed.  They must not be used as release evidence.  The
authoritative dispositions and claim ceilings are in
`reviews/CURRENT_CLAIM_LEDGER.md`.

Negative results and things that did not work are in §5 and §6, not buried.

---

## 1. Legacy fixed-cell construction

Liquid argon, 108 atoms, T = 120 K, ρ\* = 0.790. Four error fields were added to
the reference potential, each scaled so that its **force RMSE on the reference
ensemble is identical to within 0.5 %**. The observable is the number of pairs
in a single radial bin across the first peak. The fields differ only in how
their error correlates with that observable.

At force RMSE = 4.0 × 10⁻³ eV/Å:

| error field | ρ(A, δU) | predicted shift | measured shift | significance |
|---|---|---|---|---|
| null-space | +0.009 | −0.113 | **−0.220 ± 0.640** | 0.3 σ from zero |
| random (seed 1) | +0.110 | −1.306 | −1.482 ± 0.579 | 2.6 σ |
| random (seed 0) | −0.430 | +5.778 | +5.320 ± 0.589 | 9.0 σ |
| aligned | −0.582 | +10.206 | **+9.891 ± 0.611** | 16.2 σ |

At 1.0 × 10⁻³ eV/Å the **first-order predictions** scale down by four; the
noisier measured shifts do not.  The legacy table above is retained as an
observation, subject to the provenance and replication limits below.

Two bounded observations follow:

**A small force-RMSE separation coexists with a large target-point contrast.**
The extreme fields differ by 0.52% in force RMSE and 10.11 pairs in their legacy
target-bin point estimates. No practical RMSE-equivalence tolerance was frozen,
and paired contrast uncertainty was not retained, so this is a provisional
fixed-cell observation rather than exact equivalence, formal sufficiency, fitted-
model prevalence, or selection-policy evidence.

**Their ordering is associated with the constructed correlation.** The measured
shift is a monotone, near-linear function of ρ(A, δU) at fixed force error —
`figures/headline_mechanism.png`, panel (a) — which is what
`Δ⟨A⟩ = −β σ_A σ_δU ρ` says it should be.

## 2. Legacy prediction points and unresolved calibration

The "predicted" column above is `−β Cov₀(A, δU)` computed from the reference
trajectory alone. **No molecular dynamics or Monte Carlo with the surrogate was
involved in producing it.** The measured column comes from explicit sampling of
each perturbed potential.

Across all eight fields, the residual between prediction and measurement is
**1.01 σ rms** (`figures/headline_prediction.png`).

**That number was presented as a calibration pass and it is not one.** An
external review (`reviews/`) pointed out what decomposing the eight residuals
shows: all eight are negative, the common offset is −0.861 σ, and the scatter
about that offset is 0.571 σ. An rms near one is therefore a systematic
displacement plus a spread *smaller* than the quoted error bars — not eight
independent predictions landing where they should. The eight are also not
independent: every measured shift is (surrogate chain mean − reference chain
mean) and all eight share one reference chain, which induces a common component
that the rms summary does not model.

Legacy `exp09` explores the size of that effect with eight reference chains and
four direct-chain indices per field.  The direct random streams were reused
across fields, so the 64 cells are not independent.  Its decomposition is
descriptive evidence only; §5.5 states the remaining design and provenance
limits.

The computational-cost hypothesis remains worth testing: evaluating a model's
effect on a stored observable costs one energy evaluation per stored frame
instead of a full simulation.  No decision-utility or cost-matched selection
experiment (Gate C) has yet shown a practical payoff.

### 2.1 The second-order term is an exploratory correction and fails as a gate

These are different claims about the same quantity and the difference is the
whole of §5.4 and §5.5. Stated together so neither is read as the other:

**As an exploratory correction in legacy exp09, the second-order term changes
the decomposition.** Across its 64 non-independent cells it takes the grand
mean from +0.781 σ to +0.089 σ and the apparent field-effect spread from 0.698
to 0.422 against a nominal direct-chain noise floor of 0.435.  This is
consistent with truncation contributing, but the non-IID streams and incomplete
joint uncertainty propagation prevent a mechanism or calibration claim.

For a null-space field the first-order term vanishes by construction, so a
resolved physical residual would begin at higher order.  The two legacy
measurements are not precise enough to validate that residual or its coverage;
agreement “within an error bar” is a non-rejection, not confirmation.

**As a gate, thresholding on the ratio of second to first order, it does not
work at all.** Over 89 recorded cases with a direct comparator (not 89
independent experimental units) its
false-trust rate is 19 % and its AUC is 0.556 (§5.4). A correction that improves
the average is not the same thing as a statistic whose size tells you which
individual case will disagree.  Whether it is a reliable correction requires
the v2 independent-unit rerun; it is already unsuitable as a gate.

> **Historical correction, not a validation.** An earlier version said the
> opposite because it promoted an `exp09 --quick` result into prose.  The later
> legacy production artefact gives the decomposition above, but its shared
> streams and invalid start-of-run provenance prevent either run from settling
> calibration.  The v2 implementation repairs those design/provenance defects;
> production has not been rerun.

### 2.2 How much data the construction needs

A null-space field is orthogonal to the observable *on the frames it was built
from* by definition, so the whole construction is worthless unless it survives
on fresh frames. Constructing on one half of the reference trajectory and
evaluating on the other:

| frames used to construct | out-of-sample predicted shift |
|---|---|
| 250 | 0.041 ± 0.067 |
| 500 | 0.142 ± 0.071 |
| 1000 | 0.038 ± 0.078 |
| 2000 | 0.028 ± 0.078 |

Against the aligned field's 10.2 pairs at the same force error, that is a
suppression of roughly 400× **on this construction chain**. Across six legacy
fixed-cell construction clusters the same unstable ratio has a median of 26×
and a range of 16–467× (§5.6).  Because its denominator is near zero and the
run lacks valid start-of-run provenance, neither 400× nor 26× is a release
estimate; the cluster-level aligned-minus-null contrast is the safer readout.

An earlier attempt at 30 construction frames gave 1.2–1.6 pairs — the null space
of a covariance estimated from 30 samples is strongly affected by sampling
noise.  The tested counts above show no monotone degradation, but they do not
identify a general sample-size threshold.

## 3. Orthogonality is specific to the observable it was built for

The null-space field leaves its target bin alone and moves the *rest* of the
`g(r)` curve by +11.4 pairs, which is as much as the aligned field moves it
(+11.2). See `figures/exp07_designed_counterexamples_counterexamples.png`.

This is the point rather than a caveat. There is no such thing as an error field
that is harmless in general; harmlessness is a relation between an error and a
particular question.

It follows that no *task-independent* scalar can order models the same way for
every observable — not force RMSE, and not any replacement computed without
reference to what the model is for. It does **not** follow that no scalar is
useful, which an earlier version of this section claimed. Once a set of
observables and a tolerance for each are fixed, a scalar loss over that set is
well defined; `max_j |Δ⟨A_j⟩| / τ_j` is one, and nothing here argues against it.
What cannot exist is the thing force RMSE is currently used as: a single number,
computed before anyone says what the model is for, that ranks models correctly
for whatever comes next. What can be computed is a *vector* of response scores,
one per observable anyone cares about, and a scalar built from it afterwards.

## 3a. A legacy two-regime pattern in one fixed, clustered zoo

This is the result that refuted the project's own prediction, and it is more
useful than the prediction was.

**P1 as originally stated — that force RMSE would rank models weakly against
downstream physics — is not supported.** Over a zoo of 34 surrogates spanning
three decades of force RMSE (5 × 10⁻⁴ to 1.8 × 10⁻¹ eV/Å), the rank correlation
between force RMSE and measured observable error is **ρ = 0.83 [0.68, 0.92]**.
Within this fixed zoo that is a strong descriptive association: widely separated
members tend to have the same ordering under the two quantities.

Within this constructed zoo, force RMSE resolves less among members of
comparable force error.  Restricting descriptively to the 15 members whose force
RMSE differs by a factor of 2.6:

| | across the zoo | within the band |
|---|---|---|
| force RMSE spread | 360× | 2.6× |
| observable error spread, raw | — | 12.1× |
| observable error spread, noise-subtracted | — | ~~30×~~ (see below) |
| observable error spread, excluding the two designed extremes | — | **6.5×** |
| ρ (force RMSE, truth) | **+0.835** | **+0.343** |
| ρ (response prediction, truth) | +0.982 | **+0.943** |

**The 30× figure should not be used and is struck above.** It is a ratio of
noise-subtracted norms whose denominator is the member `null_f4e-03`, whose
observable error is 2.65 raw against a noise level of 2.43 — that is, the
denominator is not resolved from zero, so the ratio is a statement about the
noise floor rather than about the physics. Two defensible numbers replace it:
the raw spread is **12.1×**, and excluding the two members that were
*constructed* to sit at the extremes (the null-space and aligned fields, which
are the answer rather than evidence for it) the spread among the remaining 13 is
**6.5×**. Sixfold is the number to quote.

Inside this selected band the observable-error point estimates vary by roughly a
factor of six; the corresponding rank correlations are 0.343 for force RMSE and
0.943 for the response prediction.  These are fixed-zoo descriptions, not
population-level explained variance or model-selection performance.  See
`figures/headline_regimes.png`.

**That window was chosen after looking at the data, so it does not stand on its
own.** `scripts/band_robustness.py` reduces dependence on that single window by
sweeping every multiplicative window of every width across the same force-RMSE
range.  It does not remove clustering or create independent model draws:

| window width | placements | median ρ (force) | median ρ (prediction) |
|---|---|---|---|
| 2.0× | 16 | 0.32 | 0.90 |
| 2.6× | 24 | 0.33 | 0.94 |
| 3.0× | 29 | 0.34 | 0.94 |
| 4.0× | 55 | **0.58** | 0.94 |
| 6.0× | 70 | 0.59 | 0.95 |
| 10× | 92 | 0.62 | 0.97 |

Two things follow, and one of them is a correction to the paragraph above.

The correction: the reported window is 4.0× wide, and the *median* 4.0× window
gives ρ = 0.58, not 0.34. The reported value sits at the **16th percentile** of
same-width placements. It was a favourable window, and quoting it alone
overstated the effect.

The descriptive pattern that survives the window sweep in this fixed zoo: at
widths of 3× or below, the median ρ(force) over all placements is **0.32–0.34**
and never exceeds 0.90, while the median ρ(prediction) is **0.90–0.94**. Across
all 286 overlapping windows of the same clustered points, the prediction is
above 0.8 in **100 %** of them; force error is below 0.5 in 43 %.  Those repeated
windows are sensitivity calculations, not 286 independent replications and not
a test of candidate-selection utility.

This check was run before any external review, and it found against the way the
result had been presented.  The full window-sweep description, with its
clustered-unit caveat, is the one to quote.

**What this does and does not license.** "Force error is a coarse filter, not a
selector" is how this was written, and as a general statement about model
selection it is not supported by this zoo, for a reason that survives the
window sweep: the 34 members are not 34 draws from any population of models. They
are a handful of parameter families — 18 shell points from one factorial grid,
several amplitude points that are the same shape rescaled — sharing a reference
trajectory, a baseline and a random stream. Sweeping windows over that set
removes the presentation bias of quoting one favourable interval; it does not
turn a designed, clustered set into evidence about fitted models. Every window
re-uses the same points.

The supported form, which is narrower and is what §3a should be read as
claiming:

> Within this fixed zoo of designed error fields, and at every window width of
> 3× or below, the median rank correlation with observable error is 0.32–0.34
> for force RMSE and 0.90–0.94 for the response prediction.  These overlapping
> windows reuse a clustered, designed set.  Whether the contrast recurs among
> *fitted* models of comparable force error is a separate question requiring a
> pre-registered selection experiment on models this study did not produce.

Independent evidence bears on the wider claim in both directions and should be
read alongside it. Gawkowski *et al.*'s finite-temperature benchmark of 15
foundation MLIPs finds that lower single-point force error does on average go
with lower observable error — consistent with the across-decades regime here —
*and* that individual systems show qualitative structural failures at low force
error, which is the within-band regime. The useful framing is therefore
conditional trust and exception detection, not a verdict on force RMSE.

Two secondary observations from the same run:

- **The smoothness ratio does not work.** `std(δU)/force_RMSE`, proposed in the
  theory document as an empirical stand-in for the inverse spectral weighting,
  has ρ = **−0.01 [−0.37, 0.37]** against observable error. It carries no
  information at all here. Proposed and refuted in the same study.
- Per-atom energy spread (`ρ = 0.92`) outranks every force metric across the
  zoo, which is not something the theory predicts and is worth no more than the
  observation.

One statistical caution applies to this section. The observable error is a norm,
so noise inflates it: `E|v_measured|² = |v_true|² + E|noise|²`. All numbers above
subtract the noise in quadrature. Doing so changes the correlations by under
0.01 but is necessary at the low-force-error end, where signal and noise are
comparable. Top-k overlap statistics were also computed and are *not* reported
as evidence: with 34 members they moved between 0.33 and 0.67 depending on that
correction, which means they are too noisy to carry a claim.

**The intervals above bootstrap the membership of the zoo and not the precision
of each member**, which the review is right to flag: every entry in the
observable-error column is a Monte Carlo estimate with its own noise, and the
quadrature subtraction is then floored at zero, which shuffles exactly the low
end where the interesting comparisons sit.
`scripts/zoo_uncertainty_propagation.py` restores the missing term by redrawing
each member's measured norm from its own sampling distribution inside every
bootstrap replicate, and sweeps the redraw scale rather than trusting one guess
at it:

| redraw σ | whole zoo, ρ(force) | band, ρ(force) | band, ρ(prediction) |
|---|---|---|---|
| 0 (as published) | 0.83 [0.67, 0.93] | 0.34 [−0.25, 0.79] | 0.94 [0.71, 1.00] |
| 0.35 × noise | 0.83 [0.67, 0.92] | 0.34 [−0.24, 0.80] | 0.94 [0.70, 0.99] |
| 0.71 × noise | 0.83 [0.64, 0.91] | 0.34 [−0.23, 0.80] | 0.94 [0.62, 0.97] |
| 1.41 × noise | 0.83 [0.55, 0.88] | 0.34 [−0.20, 0.80] | 0.94 [0.40, 0.95] |

**The two-regime description survives this uncertainty sensitivity analysis
within the fixed zoo.** Even at a deliberately excessive 1.41 × noise redraw — well above the
`noise/√8` the norm of an eight-component vector actually carries — the
across-decades correlation still excludes zero, the within-band one still
contains it, and the prediction still excludes it inside the band.  This does
not repair the clustered sampling frame or support fitted-model prevalence.

The same calculation says something sharper about which points in
`figures/headline_regimes.png` are measurements. Redrawing each member's norm at
0.71 × noise, these members land at exactly zero on a large fraction of draws:

| member | in band | raw | noise | published | floors to zero |
|---|---|---|---|---|---|
| `random_s3_f5e-04` | no | 2.12 | 2.45 | 0.00 | 59 % |
| `null_f4e-03` | **yes** | 2.65 | 2.43 | 1.05 | **45 %** |
| `random_s2_f5e-04` | no | 3.40 | 2.83 | 1.88 | 39 % |
| `null_f1e-03` | no | 3.18 | 2.47 | 2.01 | 34 % |

These are upper limits drawn as points. `null_f4e-03` is the one that matters:
it was the denominator of the 30× spread struck above, and its observable error
is unresolved from zero on nearly half of all redraws. Striking that ratio
follows from this sensitivity calculation; it is not an independent
experimental replication.

## 3b. A same-system fitted-model feasibility arm

Ten models — a pair spline, a linear ACE-style basis, Behler-Parrinello networks
and E(3)-equivariant networks — fitted to 400 configurations of Lennard-Jones
argon, with training and test data drawn from separate Markov chains.

**The response formula reaches training-induced error fields.** Predicted against
measured observable shift, in rms units of the measurement's own uncertainty,
with the same decomposition §2 now applies to itself:

| setting | models | rms | common offset | scatter about it | all same sign |
|---|---|---|---|---|---|
| designed error fields (exp07) | 8 | 1.01 σ | −0.86 σ | 0.57 σ | yes (8/8) |
| fitted models, 400 configurations | 10 | 1.06 σ | −0.49 σ | 1.00 σ | no (7/10) |
| fitted models, 40 configurations | 8 | 1.00 σ | −0.85 σ | 0.57 σ | yes (8/8) |

**This was written as "three independent settings, all landing at one standard
error — the external-validity check for §2, and it passes". It is neither
independent nor a pass.** The three settings share the Lennard-Jones system, the
observable, the reference chain, the training pool and the code path; the two
neural-network families have two seeds each. And the decomposition on the right
of the table shows the same signature in two of the three arms as in §2: a
common negative offset with a scatter smaller than the quoted errors. The
400-configuration arm, whose residuals change sign, has a scatter of about one
quoted standard error and no resolved common offset.  It is closest to the
nominal pattern, but has the fewest usable models and does not by itself validate
calibration.

The retained observation is narrower: **the response calculation can be
applied to these training-induced error fields, not only to designed ones.** It
is a same-system feasibility result.  The residual decomposition above does not
validate calibration, and nothing about this arm is external.

The low-budget arm also produced the guard's best moment. Two E(3)-equivariant
networks fitted to 40 configurations gave apparent observable shifts of **−95.5
and −112.8 pairs** — twenty times any real effect in this study — and the
stationarity check identified both chains as unequilibrated and excluded them.
Without it, the largest numbers in this paper would have been transients on the
way to the models' own equilibria rather than differences between ensembles.
That is the second time the same check has caught a result that would have been
reported (§4 of `docs/methods.md` is the first).

Most of the arm is uninformative for ranking: its legacy observable-error
estimates are largely unresolved from zero at the available precision. Force
RMSE spans 0.0004–0.0061 eV/Å, but **the rank correlations from this arm are not
reportable** — with ten models whose observable errors are mostly at the noise
floor, every confidence interval spans zero (force RMSE: ρ = 0.43 [−0.35,
0.96]). Nothing about ranking can be concluded from it.

One pair in the zoo is a hypothesis-generating observation:

| model | difference | force RMSE | predicted | measured |
|---|---|---|---|---|
| `egnn_c8_s0` | — | 0.0045 | −0.98 | −0.15 ± 0.57 (0.3 σ) |
| `egnn_c8_s1` | **initialisation seed only** | 0.0061 | −3.03 | **−4.10 ± 0.69 (6.0 σ)** |

Same architecture, same training data, same hyperparameters. Their force errors
differ by 36 %; their legacy observable-error point estimates differ by a factor
of 27, with one unresolved from zero and the other reported at six standard
errors.  With only two initialisation seeds this is not a selection experiment
and does not estimate a seed-to-seed distribution.  The response calculation's
larger predicted magnitude coincided with the larger measured magnitude in this
pair.

This is compatible with the band pattern of §3a appearing in fitted models, but
**it is n = 2 and should be read as an observation that generates a hypothesis,
not as a regularity.** Two seeds cannot establish that seed-to-seed observable
variance generally exceeds what force error anticipates, nor validate the
response calculation as a selector.  Whether the pattern recurs is what Gate B
would test.

## 4. A legacy fixed-zoo width sweep gives a fivefold range

Holding force RMSE constant to 0.0 % and varying only the **width** of the error
field in `r` (`figures/headline_mechanism.png`, panel b), the measured
observable error spans a factor of five, from 2.1 to 10.0 pairs.

The first-order point prediction follows the ordering across the tested width
range; this remains a descriptive result for the fixed constructed fields.

## 5. What did not work

### 5.1 The width^{3/2} scaling is wrong

`docs/theory.md` originally offered a scaling estimate predicting that at fixed
force error the observable damage grows as `width^{3/2}`.  In the legacy
fixed-zoo output, the fitted exponent is **0.56** (direct) and **0.45**
(first-order prediction). The estimate misses that legacy pattern by about a
factor of three in the exponent.

The retraction was made in the theory document before this measurement, on
analytic grounds: both inputs to the estimate have regimes of validity that a
real system leaves quickly. The legacy pattern is consistent with that analytic
retraction.
What survives is the weaker and still sufficient statement that force error and
observable error depend differently on the shape of the error field, so their
ratio is not a constant — §4 measures that ratio varying by five.

### 5.2 The exp06 discrepancy was not reproduced, but remains unresolved

In the legacy exp06 amplitude sweep, at the smallest perturbation, first order
(−3.835), exact reweighting (−3.884) and direct sampling (−1.948 ± 0.821) did
not coincide.  A later six-seed check at a **different bin** gave a direct mean
of +2.377 ± 0.144 against same-sample estimates +1.725 ± 0.101 and +1.757.  That
different-bin check cannot identify the cause of the original discrepancy or
validate its uncertainty model.

Legacy exp10 then evaluated both bins with multiple reference and surrogate
chains.  Its point estimates did not reproduce the old gaps: direct minus MBAR
was +0.047 pairs at 4.25 Å and +0.160 at 5.25 Å.  This is evidence of **point
non-recurrence**, not evidence of equivalence and not a causal diagnosis of the
old observations.

The prospectively recorded legacy equivalence result is inconclusive: every comparison was
reported `underpowered`, with intervals wider than the ±0.5-pair equivalence
margin.  More seriously, the legacy relaxation gate subtracted uncertainty
from the observed arm gap instead of adding it.  Under the corrected direction
the gate fails, so incomplete relaxation is **not excluded** and the primary
end-to-end endpoint is not evaluable from that run.  The primary direct estimate
also used the final 10 % of each chain, only 150 frames, and the reweighting/MBAR
bootstrap concatenated chains instead of preserving chain-level units.

No result identifies the exp06 values as single-chain excursions, proves an
estimator bias, or proves the absence of one.  The v2 implementation corrects
the gate, chain-level uncertainty propagation, fixed-discard analysis and
start-of-run provenance. A quick end-to-end smoke completed and was correctly
marked `smoke_only`; its failed gates are not scientific results. The production
experiment has **not** been rerun;
until then P0-1 remains partially resolved and release-blocking.

### 5.3 The dilute-gas residual and an unmatched N=2 sanity check

The legacy low-density calculation at ρ\* = 0.079 gives a 4.7 σ rms / 6.5 σ
worst disagreement between the sampled estimators and a hand-derived
`g(r) = exp(−βu) + O(ρ)` approximation
(`results/validation/low_density_limit.txt`).  That disagreement remains
unexplained.  Finite-density corrections are one live explanation, not an
established cause.

`scripts/validate_two_particle_exact.py` supplies a useful exact-quadrature
sanity check, but it is **not a matched test of the low-density run**.  It uses
N=2 and different temperature, cutoff, bin-range and perturbation settings.
The amplitude sweep also reuses one reference chain, so its six rows are
correlated transformations rather than six independent coverage tests.

| βσ(δU) | 2nd/1st | ESS | \|exact shift\| | linear error | reweighting error | reweighting, in σ |
|---|---|---|---|---|---|---|
| 0.013 | 0.038 | 1.000 | 0.00299 | **+0.2 %** | −1.5 % | 0.86 |
| 0.052 | 0.151 | 0.998 | 0.01137 | **+5.3 %** | −1.5 % | 0.82 |
| 0.209 | 0.604 | 0.974 | 0.03727 | **+28.6 %** | −1.6 % | 0.87 |
| 0.522 | 1.510 | 0.918 | 0.06429 | **+86.4 %** | −1.7 % | 0.85 |
| 1.306 | 3.775 | 0.857 | 0.07978 | **+275 %** | −1.8 % | 0.97 |
| 2.612 | 7.549 | 0.830 | 0.08138 | **+636 %** | −1.8 % | 0.82 |

The N=2 sampler lies 0.66 σ rms from its exact quadrature over four bins and the
reweighted point estimates lie 1.5–1.8 % from it in this setup.  This is a V1
implementation sanity check only.  It cannot attribute the legacy 4.7 σ
residual to `O(ρ)`, validate many-body uncertainty, or transfer a threshold to
N=108.  The grid shows less than 10 % linear bias at βσ=0.052 and more than 10 %
at βσ=0.209; it therefore **brackets** a crossing and does not locate a
pre-registered “breakdown point” at 0.21.

#### The benchmark found a bug in a claim-bearing module

The first run of this sweep reported reweighting residuals of 0.02, 0.07, 0.38,
1.49, 18.04 and 445.20 σ — apparently a catastrophic failure at large
perturbation. The absolute errors told a different story: they were 1.5–1.8 % at
every amplitude, exactly as above. **What was collapsing was the error bar**,
from 0.00152 to 0.0000005, while the Kish effective sample size stayed between
0.83 and 1.00 and the maximum weight fraction never exceeded 0.0003. Both
standard diagnostics reported a healthy estimate the whole way down.

`reweight()` computed `shift = reweighted mean − reference mean` and then handed
the shift the bootstrap error of the *reweighted mean alone*. As βδU grows the
weights approach a hard 0/1 exclusion; the reweighted mean of a depleted bin
becomes the same number in every resample and its error genuinely does go to
zero. The shift does not — it still carries all of the reference mean's
uncertainty, and none of it was being reported.  The implementation was changed
to resample the difference as one statistic, which handles the within-chain
correlation between its two terms.  The bug fix and its unit tests are valuable,
but claim-bearing legacy experiments that used old code or invalid provenance
require clean reruns; prose cannot retroactively repair their uncertainty.

### 5.4 The second-order warning light does not work

This was written up as a result and it is not one. The claim was that the ratio
of the second-order cumulant term to the first is a self-diagnostic: below a
threshold the linear prediction can be trusted, above it not.

That is a screening test, and a screening test is judged by its error rates on
cases where the truth is known some other way. Those were computed over 89
recorded cases with a direct comparator — **not 89 independent units** — across
`exp06`, two `exp03` arms and legacy `exp09`
(`scripts/warning_light_calibration.py`, output in
`results/validation/warning_light_calibration.json`):

| | agrees with direct MD | disagrees |
|---|---|---|
| diagnostic says trust | 59 | **14** |
| diagnostic says beware | 13 | 3 |

- **False-trust rate 19 %** (14 of 73), upper 95 % limit **28 %**. A diagnostic
  that certifies fourteen wrong answers in seventy-three is worse than no
  diagnostic, because it turns an unknown into a confident error.
- **Sensitivity 0.18**: it flags three of the seventeen genuine disagreements.
- **The statistic barely separates the two groups at all.** Median ratio 0.096
  for agreeing cases and 0.098 for disagreeing ones — indistinguishable — with
  AUC 0.556 against 0.5 for no information, one-sided *p* = 0.24. This is not a
  threshold that needs moving; it is a statistic with almost no discriminating
  power here.

Two caveats, neither of which rescues it. The 0.25 threshold was chosen a
priori from the theory rather than fitted to these data — better than tuning,
but not the frozen-development-set protocol a real validation needs. And the 89
cases share a system, an observable and in places a reference chain, so the
binomial interval is optimistically narrow.

**The claim is withdrawn.** The ratio is reported alongside every prediction as
a descriptive statistic and is not used to certify anything.

This is *not* a finding that the second-order term is useless.  Legacy exp09
suggests that adding it changes the average residual decomposition, but the
non-IID field streams and invalid provenance prevent a confirmatory claim.  In
any event, the ratio's size has not been shown to identify which individual
prediction will disagree and is not a certification gate.

### 5.5 What legacy exp09 suggests about the "1.01 σ"

Legacy `exp09` contains 8 reference chains × 8 designed error fields × 4 direct
chain indices = 64 residual cells (`results/exp09_calibration_replication/`).
Those cells are **not IID**: each chain index reuses its direct random stream
across all fields.  Its manifest also captured repository state at run end and
cannot prove which source was executed.  The numbers below are therefore an
exploratory decomposition, not a calibration result.

The residual table decomposes because the three error sources have different
footprints: a reference chain's error in ⟨A⟩ is constant down a row, a field's
direct-chain error is constant down a column, and a genuine field-dependent
failure of the prediction is also constant down a column but is not noise. Each
observed spread is set against the spread its own quoted error predicts, so the
test is a ratio rather than a threshold.

| | first order | + second-order term | predicted by the quoted errors |
|---|---|---|---|
| residual rms | 1.985 σ | 1.717 σ | 1 |
| **grand mean** | **+0.781 σ** | **+0.089 σ** | 0 |
| row (reference-chain) spread | 1.477 σ | 1.161 σ | 0.898 |
| **column (field) spread** | **0.698 σ** | **0.422 σ** | 0.435 |
| interaction rms | 0.997 σ | 1.267 σ | — |

Four patterns appear in this legacy table; none identifies a unique mechanism.

**The offset is compatible with a reference-draw contribution.** The eight row effects are
[+1.69, +1.83, −2.54, −0.28, +0.47, +0.59, −1.34, −0.41] σ and they track their
own chains' means: the two lowest reference chains (⟨A⟩ = 209.66) give the two
most positive rows, the highest (211.38) gives the most negative. exp07's eight
residuals were all *negative*; exp09's are 42/64 *positive*. A quantity whose
sign changes across reference draws, which is evidence against treating the
original all-negative pattern as eight independent calibration successes.  It
does not by itself prove an unbiased estimator or identify the whole offset.

**Adding the second-order term changes the legacy decomposition.** It takes the
grand mean from +0.781 σ to +0.089 σ
and the field-effect spread from 0.698 to 0.422 against a noise floor of 0.435 —
a variance ratio of 2.58 falling to 0.94. The largest first-order field effect
is the `aligned` field's +1.36 σ, and `aligned` is the field with the largest
|ρ(A, δU)|, which is exactly where a second-order correction should be largest.
This pattern is consistent with truncation contributing, but shared streams and
the missing joint chain-level uncertainty prevent a confirmatory attribution.

**The prediction's own error was missing from the denominator and is of the
same order as the interaction.** The residuals were normalised by
`sqrt(direct_sem² + reference_error²)`, which describes the *measurement* and
omits the prediction's Monte Carlo error — a per-cell quantity, the only one with
an interaction footprint. `scripts/residual_variance_budget.py` finds the
observed interaction (0.997 σ) is 1.19 × the omitted term (0.840 σ), and
restoring it takes the rms from 1.985 σ to 1.434 σ.  This post-hoc variance
budget does not replace joint chain-level propagation in a confirmatory run.

**The reference-chain error model is also unstable.** The row spread is
1.477 σ against 0.898 predicted, a factor 1.64; and directly, the scatter of the
eight reference means (0.591 pairs) exceeds their mean blocking error (0.456) by
1.30.  This legacy cell suggests blocking underestimates scatter across these
reference chains.  It contradicts
`scripts/check_between_chain_scatter.py`, which measured the ratio at **0.53**
— blocking conservative by a factor of two — on a different bin of the same
system (§5.2). Two measurements of the same kind of quantity, giving opposite
answers, mean the blocking error's reliability is itself variable and should not
be assumed in either direction.

**Current ceiling.** The table suggests that reference-draw variation,
truncation and omitted prediction uncertainty can all matter.  It cannot show
that there is no systematic bias, that direct-chain noise explains all field
structure, or that “only the error bar” was wrong.  The original 1.01 σ was not
a calibration statistic, and P0-2 remains only partially resolved.

#### A design flaw in exp09, found in its own output

exp09 seeds its direct chains by chain index alone, `seed + 5000 + 17·d`, with
no dependence on the field. All eight fields therefore share four random
streams — the same common-random-number entanglement the review criticised in
exp07, reproduced in the experiment written to fix a different problem.

The component shared across fields at fixed chain index is 0.134 pairs against
a per-field chain-to-chain sd of 0.401, about 11 % of the legacy direct-chain
variance.  This invalidates an IID reading of the 64 cells and confounds field
comparisons with chain-index effects.  The direction and magnitude of the
resulting error must be propagated from a declared paired design; aggregate
post-hoc arguments do not establish that it is necessarily conservative.

The legacy artefacts are retained unchanged as an audit record.  The
`exp09_calibration_replication_v2` code is repaired to use field-specific
streams and start-of-run provenance; production has not been rerun.

### 5.6 A fixed-cell legacy counterexample replication

`exp11`, review item P0-3. exp07 established the aligned-minus-null contrast
once, from one construction trajectory, with two force levels that are the same
field rescaled and two direct runs sharing a random stream. Legacy exp11 repeats
the construction in six fixed-cell clusters. Construction and evaluation
trajectories differ by cluster, but direct random streams were reused across
fields inside each cluster and the manifest has invalid start-of-run
provenance. The cluster is the only defensible top-level unit.

| cluster | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| aligned − null (pairs) | 10.88 | 10.69 | 11.01 | 11.28 | 10.44 | 9.73 |
| force-RMSE match, out of sample | 0.991 | 1.004 | 0.998 | 0.995 | 1.001 | 0.999 |

**Mean +10.671 pairs, 95 % CI [+10.105, +11.238]** with the cluster as the unit,
against a pre-registered minimum meaningful effect of 1.0 pairs. All six
clusters passed the 2 % force-matching tolerance out of sample, worst deviation
0.86 %.  Thus the **fixed-cell cluster-level aligned-minus-null contrast is
stable in this legacy run**.  It does not complete Gate A, which also requires
predeclared targets, bases, sizes and states.

The published within-cluster uncertainty (0.490) and between/within ratio (1.10)
treated field means as independent despite their shared streams and are
withdrawn.  The predeclared random control was expected to lie between null and
aligned, but clusters 4 and 5 did not; describing those controls as having
“behaved as designed” was a post-hoc reinterpretation.  These defects do not
erase the top-level six-cluster contrast, but they block stronger mechanism,
control and within/between-variance claims.

The `exp11_counterexample_replication_v2` code repairs field seed/start pairing,
paired uncertainty, intention-to-test force gates, control reporting and
provenance and adds a held-out covariance-null manipulation gate plus tri-state
effect interpretation. A quick smoke completed, failed closed and is not
evidence. Production has not been rerun.

#### The suppression ratio is unstable and not release evidence

This is the number that does not survive, and it is one of ours.
Earlier prose quoted a **≈360×** suppression of
the null field's predicted shift relative to the aligned field's, measured on
one construction chain. Across the six legacy fixed-cell clusters that ratio is:

| cluster | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| aligned / null predicted shift | 21× | 19× | 16× | 30× | 467× | 44× |

**Median 26×, range 16–467×.** exp07's 360× sits in the upper tail. The
distribution is heavy-tailed for an understandable reason — the denominator is a
near-cancelling quantity, so one cluster in six lands near zero and produces an
enormous ratio — which is exactly why a single draw of it should never have been
quoted as a property of the method.

The algebraic in-sample orthogonality is exact:
the null field's covariance with the target is ~10⁻¹⁶ on its construction frames
and 10⁻³ on held-out frames, a degradation of thirteen orders of magnitude. That
is not a defect — it is the statement that a null space estimated from finite
samples is only null on those samples, which §2.2 already says.  Because this
near-zero-denominator ratio is unstable and the legacy run lacks valid
provenance, neither 360× nor median 26× is a release estimate.  The cluster-level
aligned-minus-null contrast above is the defensible legacy result.

### 5.7 Legacy exp10: point non-recurrence, formal question unresolved

`exp10`, review item P0-1 and the one the review calls a release blocker.
`exp06`'s smallest amplitude had two reference-based point estimates within
1.3 % of each other — linear −3.835, reweighted −3.884 pairs — and direct
sampling saying −1.948 ± 0.821. Either the prediction was wrong, in which case
the central claim of this repository fails, or the direct measurement was, in
which case every "measured" column is suspect.

Eight reference chains, sixteen surrogate chains in two bracketing
arms, two bins of opposite discrepancy sign, two samplers sharing no propagation
machinery, and six estimators of the same two numbers:

| estimator | 4.25 Å | 5.25 Å |
|---|---|---|
| linear response | −3.82 ± 0.07 | +1.81 ± 0.06 |
| forward FEP | −3.78 | +1.80 |
| reverse FEP | −4.17 | +1.89 |
| MBAR, both directions | −3.83 ± 0.08 | +1.81 ± 0.07 |
| direct, HMC (8+8 chains) | −3.78 ± 0.67 | +1.97 ± 0.78 |
| direct, Metropolis (4+4 chains) | −3.01 ± 1.48 | +1.68 ± 2.10 |

**The old discrepancy does not recur in these point estimates.** Direct minus MBAR is **+0.047** pairs at
4.25 Å and **+0.160** at 5.25 Å, against exp06's apparent 1.9-pair gap. The
linear prediction of −3.82 reproduces exp06's own −3.835 to 0.4 %; what does not
reproduce is exp06's direct measurement of −1.948, which sits 2.7 σ from this
one on its own quoted error.  These observations do not identify why either
legacy run differed.

**Incomplete relaxation is not excluded.** The legacy gate used
`gap − 1.96×SE ≤ bound`, so greater uncertainty made passing easier; the correct
upper-confidence gate adds uncertainty.  With that direction corrected, the two
bin upper bounds are about 2.07 and 3.39 pairs against a 0.5-pair bound, so the
gate fails.  The primary 90 % discard also retains only 150 frames per chain.

**The estimator comparison is underpowered and, because the prerequisite
relaxation gate fails, not evaluable as a confirmatory endpoint.** Every interval contains zero, but the direct arm's error is
0.67–0.78 pairs from eight chain pairs, so the 95 % intervals are ±1.3, wider
than the ±0.5 equivalence bound fixed in advance. Point agreement at 0.05 and
0.16 pairs is not the same as having established equivalence to 0.5, and the
decision rule does not let the first be reported as the second. Reaching the
bound would need substantially more independent chains after a valid
stationarity/relaxation design.

Weight-overlap diagnostics were numerically benign, but they do not repair the
failed relaxation gate, chain-concatenated bootstrap, invalid provenance or the
incorrect covariance correction.  “All diagnostics healthy” is therefore not a
defensible summary.

#### What the legacy run can and cannot say

The opposite signs at the two bins argue against collapsing all legacy
differences into one simple directional bias, but they do not prove
single-realisation causation.  The published covariance correction also used
the wrong sign/scale for the shared reference term and applied one correlation
to estimators with different dependence structures; those corrected intervals
are withdrawn.  The v2 repair implementation addresses these analyses and
provenance capture but remains pending final code review and a clean commit;
production has not been rerun.  P0-1 therefore remains a release blocker.

## 6. What is not here

- **exp01, exp02, exp04, exp08 were never written.** The README used to list
  them as if they had been. Reference physics, dataset generation, the full
  observable matrix and the committee predictor are all absent, and
  `atomlab/training/` is an empty package.
- **Fitted models are a feasibility arm, not an external check.** §3b uses ten
  models that share the system, observable, reference chain, training pool and
  code path with the designed-field arms. It shows the response calculation
  reaches training-induced error fields. It does not show that fitted models
  occupy the regime the designed fields explore, and the difference matters for
  every conclusion phrased as advice.
- **No legacy manifest proves the executed source.** Several older runs record
  dirty trees; exp09/exp10/exp11 additionally read commit/digest state at run
  completion.  Start-of-run and end-of-run provenance, source-change detection
  and artefact hashes are implemented on the repair branch, but become evidence
  only after clean v2 production reruns.
- **Raw trajectories are not stored** for the legacy runs, only selected
  aggregates or per-frame values.  Deterministic seeds do not guarantee
  reproduction without the executed source and environment.
- **Dynamical observables.** Diffusion and vibrational spectra need real time
  evolution, which Monte Carlo cannot provide. The molecular dynamics layer
  exists and is tested; the experiments using it were not reached.
- **A melting-point determination**, deliberately: two-phase coexistence is not
  affordable at the number of models compared here, and the design document
  says so rather than reporting a badly converged number.

## 7. Supporting validation

From `docs/validation.md`, checks run outside the test suite against targets
from outside the codebase:

| check | target | obtained |
|---|---|---|
| Stillinger-Weber Si cohesive energy | exactly −2ε = −4.33660 eV/atom | −4.3366000 |
| Lennard-Jones fcc lattice sum | −8.6108 ε/atom | −8.603 (cutoff-extrapolated) |
| EAM Cu lattice constant / cohesive energy / bulk modulus | 3.615 Å / −3.54 eV / 140 GPa | exact on all three |
| fcc / bcc / sc Steinhardt Q₆ | 0.5745 / 0.5106 / 0.3536 | 0.5745 / 0.5107 / 0.3536 |
| SOAP, ACSF rotational invariance | 0 | 8 × 10⁻¹⁵, 2 × 10⁻¹⁵ |
| HMC ⟨U⟩ and var(U) vs equipartition | exact closed forms | 0.5 σ, within 3 % |
| phonon dispersion vs closed form | 2√(k/m)\|sin(qa/2)\| | 10⁻⁵ relative |

And one that is worth stating even though it is not a defect: the analytic EAM
reproduces its three fitted targets exactly and still gives a maximum phonon
frequency 30 % below copper's. Fitting a quantity does not constrain the
quantities you did not fit, even closely related ones — which is the same
statement this study makes about force error, arriving from a different
direction.
