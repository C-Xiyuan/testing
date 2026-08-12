# Results

Every number here was produced by code in this repository on an equilibrated
reference ensemble whose stationarity was verified. Methods are in
`docs/methods.md`, the argument being tested in `docs/theory.md`, and the raw
outputs in `results/` with a manifest recording the git commit that produced
them.

Negative results and things that did not work are in §5 and §6, not buried.

---

## 1. The headline

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

The same pattern holds at 1.0 × 10⁻³ eV/Å with all shifts scaled down by four.

Two things follow, and neither is an interpretation:

**The reported metric cannot distinguish these models.** Their force RMSE agrees
to 0.5 %. Any procedure that selects a model by force error would rank them as
equivalent. One of them leaves the observable untouched and another moves it by
sixteen standard errors.

**The quantity that does distinguish them is the correlation.** The measured
shift is a monotone, near-linear function of ρ(A, δU) at fixed force error —
`figures/headline_mechanism.png`, panel (a) — which is what
`Δ⟨A⟩ = −β σ_A σ_δU ρ` says it should be.

## 2. The prediction works

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
mean) and all eight share one reference chain, so a single reference realisation
displaces all of them together.

`exp09` measures the size of that effect directly, with eight independent
reference chains and four independent direct chains per field, decomposing the
residual table into a reference-chain contribution, a field contribution, and
the rest. The corrected statement is §5.5, and it is
not the one this section made.

The practical payoff is unaffected by that distinction, and is the reason the
prediction is worth calibrating: evaluating a model's effect on an observable
costs one energy evaluation per stored frame instead of a full simulation.

### 2.1 The second-order term works as a correction and fails as a gate

These are different claims about the same quantity and the difference is the
whole of §5.4 and §5.5. Stated together so neither is read as the other:

**As a correction, added to the prediction, the second-order term does its job.**
Across `exp09`'s 64 cells it takes the systematic offset from **+0.781 σ to
+0.089 σ** and removes the field-dependent structure entirely — the spread of
field effects falls from 0.698 to 0.422 against a direct-chain noise floor of
0.435, so what looked like a real per-field bias at first order is exactly the
truncation. The largest first-order field effect belongs to the `aligned` field
(+1.36 σ), which is the field with the largest |ρ(A, δU)| and therefore the one
where a second-order correction should bite hardest. That is the mechanism
predicting where it is needed and then supplying it.

For a null-space field the first-order term vanishes by construction, so any
residual effect must be second order; at both force levels the measured residual
agrees with the second-order estimate (−0.540 and −0.439 pairs, both within the
error bar). That case is a weaker test than it looks — there is nothing for the
correction to be added to — but it points the same way.

**As a gate, thresholding on the ratio of second to first order, it does not
work at all.** Over 89 cases with an independent direct measurement its
false-trust rate is 19 % and its AUC is 0.556 (§5.4). A correction that improves
the average is not the same thing as a statistic whose size tells you which
individual case will disagree, and the second-order term is the first without
being the second.

> **A correction to this document.** An earlier version of this section said the
> opposite — that adding the second-order term makes the residuals worse on
> every measure. That came from `exp09` run under `--quick`, at 5 % of
> production chain length, where the second-order estimate is noise. The
> manifest recorded `quick_mode: true` and the figure script refuses to read
> such a run, but the number reached prose anyway. The production run is above.

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
suppression of roughly 400× **on this construction chain**. Across six
independent construction chains the same ratio has a median of 26× and a range
of 16–467× (§5.6), so this chain was a favourable draw and the figure to carry
forward is the median, not this one.

An earlier attempt at 30 construction frames gave 1.2–1.6 pairs — the null space
of a covariance estimated from 30 samples is the null space of the noise. The
construction is only as good as the covariance estimate, and the table above is
where that stops mattering.

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

## 3a. Force error works across decades and fails within a band

This is the result that refuted the project's own prediction, and it is more
useful than the prediction was.

**P1 as originally stated — that force RMSE would rank models weakly against
downstream physics — is not supported.** Over a zoo of 34 surrogates spanning
three decades of force RMSE (5 × 10⁻⁴ to 1.8 × 10⁻¹ eV/Å), the rank correlation
between force RMSE and measured observable error is **ρ = 0.83 [0.68, 0.92]**.
That is strong. A model a hundred times worse in force error really is worse,
and no theory was needed to say so.

What force RMSE cannot do is choose among models of *comparable* force error —
which is the only situation a practitioner is ever in. Restricting to the 15
members whose force RMSE differs by a factor of 2.6:

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

Inside the band the physics still varies by roughly a factor of six, force error
explains almost none of it, and the response prediction explains nearly all of
it. See `figures/headline_regimes.png`.

**That window was chosen after looking at the data, so it does not stand on its
own.** `scripts/band_robustness.py` removes the choice by sweeping every
multiplicative window of every width across the whole force-RMSE range:

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

The result that survives without any window selection: at widths of 3× or below,
the median ρ(force) over all placements is **0.32–0.34** and never exceeds 0.90,
while the median ρ(prediction) is **0.90–0.94**. Across all 286 windows of all
widths, the prediction is above 0.8 in **100 %** of them; force error is below
0.5 in 43 %. The narrower the comparison — that is, the closer to the situation
of choosing between comparable models — the less force error resolves and the
more the prediction does.

This check was run before any external review, and it found against the way the
result had been presented. The window-free statement is the one to quote.

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
> 3× or below, force RMSE fails to resolve the ordering of observable error
> (median ρ 0.32–0.34) while the response prediction resolves it (median ρ
> 0.90–0.94). Whether this holds among *fitted* models of comparable force error
> is a separate question, and requires a pre-registered selection experiment on
> models this study did not produce.

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

**The two-regime description survives, and this one goes in the study's
favour.** Even at a deliberately excessive 1.41 × noise redraw — well above the
`noise/√8` the norm of an eight-component vector actually carries — the
across-decades correlation still excludes zero, the within-band one still
contains it, and the prediction still excludes it inside the band. The omitted
term was real and it was not load-bearing.

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
is unresolved from zero on nearly half of all redraws. Striking that ratio was
not a concession to the review's argument; it is what this calculation says
independently.

## 3b. The same thing happens to models that were actually fitted

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
400-configuration arm, which is the one whose residuals do change sign, has a
scatter of a full standard error and no significant offset — so it is the only
one of the three that looks like a calibration result, and it is the one with
the fewest usable models.

What this arm does establish is narrower and still worth having: **the response
calculation can be applied to training-induced error fields, not only to
designed ones, and produces predictions of the right size.** It is a feasibility
result. It is not an external-validity check, because nothing about it is
external.

The low-budget arm also produced the guard's best moment. Two E(3)-equivariant
networks fitted to 40 configurations gave apparent observable shifts of **−95.5
and −112.8 pairs** — twenty times any real effect in this study — and the
stationarity check identified both chains as unequilibrated and excluded them.
Without it, the largest numbers in this paper would have been transients on the
way to the models' own equilibria rather than differences between ensembles.
That is the second time the same check has caught a result that would have been
reported (§4 of `docs/methods.md` is the first).

Most of the zoo is uninformative for a different reason than expected: on this
reference potential, at this data budget, nearly every architecture is good
enough that its observable error is indistinguishable from zero. Force RMSE
spans 0.0004–0.0061 eV/Å and buys almost no measurable difference. **The rank
correlations from this arm are therefore not reportable** — with ten models whose
observable errors are mostly at the noise floor, every confidence interval spans
zero (force RMSE: ρ = 0.43 [−0.35, 0.96]). Nothing about ranking can be
concluded from it, and nothing is.

But one pair in the zoo is worth the whole arm:

| model | difference | force RMSE | predicted | measured |
|---|---|---|---|---|
| `egnn_c8_s0` | — | 0.0045 | −0.98 | −0.15 ± 0.57 (0.3 σ) |
| `egnn_c8_s1` | **initialisation seed only** | 0.0061 | −3.03 | **−4.10 ± 0.69 (6.0 σ)** |

Same architecture, same training data, same hyperparameters. Their force errors
differ by 36 %; their observable errors differ by a factor of 27, one
indistinguishable from zero and the other significant at six standard errors.
Anyone selecting between these two on force RMSE would see a 36 % difference and
reasonably call them equivalent.

The response formula predicted both, from the reference trajectory alone.

This looks like the band effect of §3a appearing in fitted models rather than
designed ones, between two runs of the *same* model — as narrow a band as it is
possible to construct. **It is n = 2, and should be read as an observation that
generates a hypothesis rather than as a regularity.** Two seeds cannot establish
that seed-to-seed observable variance generally exceeds what force error
anticipates; they can establish that it happened once, here, and that the
response calculation saw it coming. Whether it is typical is exactly what the
larger seed sweeps in the Gate B plan (`reviews/CLAUDE_RESPONSE.md`) are for.

## 4. At fixed force error, the damage still varies fivefold

Holding force RMSE constant to 0.0 % and varying only the **width** of the error
field in `r` (`figures/headline_mechanism.png`, panel b), the measured
observable error spans a factor of five, from 2.1 to 10.0 pairs.

The first-order prediction tracks this across the whole range.

## 5. What did not work

### 5.1 The width^{3/2} scaling is wrong

`docs/theory.md` originally offered a scaling estimate predicting that at fixed
force error the observable damage grows as `width^{3/2}`. Measured on a verified
ensemble, the exponent is **0.56** (direct) and **0.45** (first-order
prediction). The estimate is off by a factor of three in the exponent.

The retraction was made in the theory document before this measurement, on
analytic grounds: both inputs to the estimate have regimes of validity that a
real system leaves quickly. The measurement confirms it was right to retract.
What survives is the weaker and still sufficient statement that force error and
observable error depend differently on the shape of the error field, so their
ratio is not a constant — §4 measures that ratio varying by five.

### 5.2 An open discrepancy, with three explanations ruled out

In the amplitude sweep, at the smallest perturbation, the three estimates
disagreed: first order −3.835, exact reweighting −3.884, direct sampling
−1.948 ± 0.821. The two sharing the reference samples agreed to 1.3 %; the one
requiring an independent chain disagreed with both by a factor of two.

The obvious diagnosis was that the direct estimate's error bar was too small:
blocking measures error *within* a chain, and a difference between two chains
also carries the offset between their slow modes, which no within-chain
estimator can see. `scripts/check_between_chain_scatter.py` measured that offset
by sampling both potentials from six independent seeds each.

**The diagnosis was wrong.** Measured on the same perturbation and a single bin:

| quantity | value |
|---|---|
| within-chain blocking error | 0.589 pairs |
| between-chain scatter | 0.312 pairs |
| ratio | **0.53** |

Blocking is *conservative* by about a factor of two, not optimistic. The
measured shift across the six independent chain pairs is
`[2.553, 2.773, 2.349, 2.611, 1.788, 2.188]` — mean **+2.377 ± 0.144**, single
pair spread 0.354, against the ±0.832 the sweep had quoted.

So the direct measurement is reliable and the discrepancy is real. On a fresh
3000-frame reference chain the same-sample estimators give **+1.725 ± 0.101**
(first order) and **+1.757** (exact reweighting, ESS fraction 0.83) — a 35 %
systematic difference from the +2.377 that direct sampling measures.

Three explanations are excluded by measurement:

- **Not the error bars.** Between-chain scatter is half the quoted error.
- **Not the second-order truncation.** First-order and the all-orders
  reweighting identity agree with each other to 2 %.
- **Not incomplete relaxation of the perturbed chain.** That would bias the
  measured shift *toward* the reference and make it smaller in magnitude; it is
  larger.

Two things about that third bullet, both of which the external review pressed on
and both of which turned out to matter.

**The exclusion of incomplete relaxation was an argument, not a measurement.**
It is also incomplete on its own terms: it explains why bin 5.25 Å cannot be a
relaxation artefact, and says nothing about bin 4.25 Å, where the same
perturbation gives a direct shift *smaller* in magnitude than the prediction at
every one of exp06's seven amplitudes. Those are opposite signs in two bins of
one experiment. `exp10` measures both, with surrogate chains started from
configurations equilibrated under the reference *and* under the surrogate, so
relaxation is bracketed rather than argued about.

**The prediction's error bar was the wrong one.** The ±0.101 quoted on the
+1.725 is a within-chain blocking error from a *single* reference chain. It
describes how well that chain determines its own covariance and says nothing
about how much the covariance moves between chains — which is exactly what a
comparison against an independently sampled ensemble needs. If the between-chain
scatter of the prediction is comparable to 0.1 pairs, the 3.7 σ is arithmetic.
`exp10` computes the prediction separately from each of eight reference chains
and reports that scatter.

The open question is therefore narrower than this section originally said, and
the results are in §5.6.

### 5.3 The dilute-gas closed form

The estimator was checked against the exact `g(r) = exp(−βu)` limit at ρ\* =
0.079. The estimator and the exact reweighting identity agree to under 2 % in
every bin. Both disagree with a hand-derived closed form by **4.7 σ rms, 6.5 σ
at worst** over seven bins (`results/validation/low_density_limit.txt`). An
earlier version of this section quoted 5.4 σ, which had no deposited
computation behind it; the numbers here are from a re-run whose log is in the
repository.

The hand calculation needed one correction along the way that is worth recording
because the estimator gets it right automatically: the number of pairs is
conserved, so a perturbation that depletes one shell must enrich others, and the
pair-separation distribution is *normalised*. Dropping the normalisation term
gives a prediction wrong by a factor of ten in the outer bins and wrong in sign
in the tail. A covariance is mean-subtracted, and that subtraction *is* the
normalisation, so the estimator never had the problem.

The residual 4.7 σ is most likely the `O(ρ)` correction to `g = exp(−βu)`, which
at ρ\* = 0.079 is not negligible. **This is not confirmed**, and the external
review is right that until it is, this is a benchmark the estimator has *not
passed* rather than evidence that it works. The test that would confirm it —
a density series, or an exact `N = 2` calculation where `g = exp(−βu)` holds
identically — has not been run. Note also that what is being compared here is an
approximation valid to `O(ρ)` against an estimator that makes no such
approximation, so this disagreement carries less weight than one between two
exact quantities.

### 5.4 The second-order warning light does not work

This was written up as a result and it is not one. The claim was that the ratio
of the second-order cumulant term to the first is a self-diagnostic: below a
threshold the linear prediction can be trusted, above it not.

That is a screening test, and a screening test is judged by its error rates on
cases where the truth is known some other way. Those are computable from every
case in this repository where both the diagnostic and an independent direct
measurement exist — 89 of them, across `exp06`, two `exp03` arms and `exp09`
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

This is *not* a finding that the second-order term is useless. Added to the
prediction it removes a real systematic offset and a real field-dependent bias
(§2.1, §5.5). What it cannot do is tell you in advance which case will be wrong,
and those are different jobs: an AUC of 0.556 says the ratio's size carries
almost no information about whether that particular prediction will disagree,
however useful the term itself is when added.

### 5.5 What the "1.01 σ" actually was

`exp09`, 8 independent reference chains × 8 designed error fields × 4
independent direct chains per field = 64 residuals, 80 minutes,
`results/exp09_calibration_replication/`. Its manifest records
`dirty_paths: ['results/exp09_calibration_replication/']` — the only difference
between the working tree and commit `9ffab92` was the experiment's own output.

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

Four things follow, and the first is the answer to the review's question.

**The offset is a property of the reference draw.** The eight row effects are
[+1.69, +1.83, −2.54, −0.28, +0.47, +0.59, −1.34, −0.41] σ and they track their
own chains' means: the two lowest reference chains (⟨A⟩ = 209.66) give the two
most positive rows, the highest (211.38) gives the most negative. exp07's eight
residuals were all *negative*; exp09's are 42/64 *positive*. A quantity whose
sign flips with the reference draw is a realisation, not a bias. The review's
P0-2 is confirmed and its mechanism identified.

**Most of the remaining offset is the truncation, and the second-order term
removes it.** Adding that term takes the grand mean from +0.781 σ to +0.089 σ
and the field-effect spread from 0.698 to 0.422 against a noise floor of 0.435 —
a variance ratio of 2.58 falling to 0.94. The largest first-order field effect
is the `aligned` field's +1.36 σ, and `aligned` is the field with the largest
|ρ(A, δU)|, which is exactly where a second-order correction should be largest.
This was pre-registered in the experiment's docstring as the test of whether the
column structure is the truncation showing itself, and it is.

**The prediction's own error was missing from the denominator, and it is the
whole of the interaction.** The residuals were normalised by
`sqrt(direct_sem² + reference_error²)`, which describes the *measurement* and
omits the prediction's Monte Carlo error — a per-cell quantity, the only one with
an interaction footprint. `scripts/residual_variance_budget.py` finds the
observed interaction (0.997 σ) is 1.19 × the omitted term (0.840 σ), and
restoring it takes the rms from 1.985 σ to 1.434 σ. The review's complaint that
prediction uncertainty was never propagated is correct and this is its size.

**What is left over is the error bar on the reference chain.** The row spread is
1.477 σ against 0.898 predicted, a factor 1.64; and directly, the scatter of the
eight reference means (0.591 pairs) exceeds their mean blocking error (0.456) by
1.30. Both say the same thing: blocking underestimates how much ⟨A⟩ moves
between independent chains here. Note that this contradicts
`scripts/check_between_chain_scatter.py`, which measured the ratio at **0.53**
— blocking conservative by a factor of two — on a different bin of the same
system (§5.2). Two measurements of the same kind of quantity, giving opposite
answers, mean the blocking error's reliability is itself variable and should not
be assumed in either direction.

**So the corrected claim.** Predicted and measured observable shifts agree with
no detectable systematic bias once the second-order term is included
(+0.089 σ over 64 cells) and no field-dependent structure beyond direct-chain
noise. The residuals are wider than the quoted errors by roughly 1.4 ×, and that
excess is accounted for: the prediction's own error was omitted from the
denominator, and the reference chain's blocking error understates its
between-chain scatter. Neither is a defect in the response formula. **What was
wrong was the error bar, not the estimator** — and the "1.01 σ rms" was never
evidence either way.

#### A design flaw in exp09, found in its own output

exp09 seeds its direct chains by chain index alone, `seed + 5000 + 17·d`, with
no dependence on the field. All eight fields therefore share four random
streams — the same common-random-number entanglement the review criticised in
exp07, reproduced in the experiment written to fix a different problem.

Its size is measurable from the deposited chain means. The component shared
across fields at fixed chain index is 0.134 pairs against a per-field
chain-to-chain sd of 0.401, so about 11 % of the direct-chain variance. The
consequence is one-directional: shared streams make the per-field direct errors
more alike, which *lowers* the column-effect noise floor and so makes the test
for field structure conservative rather than permissive. It does confound the
grand mean with a common direct-chain draw, which is why the grand mean is
reported alongside the row and column structure rather than on its own.

The seeds are left as they were run. Editing them now would break the match
between the deposited numbers and the source digest the manifest records, which
is the thing this repository has just finished building.

### 5.6 The counterexample replicates; its headline suppression does not

`exp11`, review item P0-3. exp07 established the aligned-minus-null contrast
once, from one construction trajectory, with two force levels that are the same
field rescaled and two direct runs sharing a random stream. This repeats it with
the **construction cluster as the experimental unit**: six clusters, each with
its own construction trajectory, its own disjoint evaluation trajectory and its
own direct chains, sharing nothing but the potential, the target and the basis
geometry. Estimand, matching tolerance, minimum meaningful effect and decision
rule were fixed in the module docstring before the run.

| cluster | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| aligned − null (pairs) | 10.88 | 10.69 | 11.01 | 11.28 | 10.44 | 9.73 |
| force-RMSE match, out of sample | 0.991 | 1.004 | 0.998 | 0.995 | 1.001 | 0.999 |

**Mean +10.671 pairs, 95 % CI [+10.105, +11.238]** with the cluster as the unit,
against a pre-registered minimum meaningful effect of 1.0 pairs. All six
clusters passed the 2 % force-matching tolerance out of sample, worst deviation
0.86 %. **The contrast replicates.**

Two things make this stronger than a repeated measurement. The between-cluster
spread is 0.540 pairs against a within-cluster 0.490, a ratio of **1.10** — so
choosing a different construction trajectory moves the answer barely more than
re-running the direct chains does. The effect is a property of the construction
*method*, not of a lucky chain. And exp07's single-cluster 10.111 sits near the
bottom of the six, so that number was if anything conservative.

The null and random controls behave as designed. Null-field shifts across the
six are [−0.57, +0.60, +0.29, −0.21, −0.34, +0.23] pairs, straddling zero;
random fields at the same force RMSE give [+2.24, +4.46, +1.28, +1.04, −3.92,
+0.03], scattered as an arbitrary direction should be; aligned fields give
[+10.31, +11.29, +11.30, +11.07, +10.10, +9.96].

#### The 360× suppression was a favourable construction chain

This is the number that does not survive, and it is one of ours.
`docs/RESULTS.md` §2.2 and the manuscript both quote a **≈360×** suppression of
the null field's predicted shift relative to the aligned field's, measured on
one construction chain. Across six independent clusters that ratio is:

| cluster | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| aligned / null predicted shift | 21× | 19× | 16× | 30× | 467× | 44× |

**Median 26×, range 16–467×.** exp07's 360× sits in the extreme upper tail. The
distribution is heavy-tailed for an understandable reason — the denominator is a
near-cancelling quantity, so one cluster in six lands near zero and produces an
enormous ratio — which is exactly why a single draw of it should never have been
quoted as a property of the method.

The underlying construction is sound and the in-sample orthogonality is exact:
the null field's covariance with the target is ~10⁻¹⁶ on its construction frames
and 10⁻³ on held-out frames, a degradation of thirteen orders of magnitude. That
is not a defect — it is the statement that a null space estimated from finite
samples is only null on those samples, which §2.2 already says. What is wrong is
the size of the surviving suppression. **The defensible figure is a median of
26× with a factor-of-thirty spread across construction chains, not 360×.**

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
- **Nothing was run on a clean commit.** `exp03`, `exp05`, `exp06` and `exp07`
  all have `git_dirty: true` in their manifests and cannot be reconstructed
  exactly. Manifests now also record which paths were dirty and a content digest
  over all source files (`experiments/common.py`), so future runs are traceable;
  the existing ones are not, and must be re-run on a tagged commit before their
  numbers appear in a submitted document.
- **Raw trajectories are not stored**, only per-frame observable values and
  energies. Re-running reproduces the numbers from the same seed; it does not
  let a reader re-analyse the original configurations.
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
