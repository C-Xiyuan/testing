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
**1.01 σ rms** — the prediction agrees with the measurement at exactly the level
the error bars claim, neither better nor worse
(`figures/headline_prediction.png`).

This is the practical payoff. Evaluating a model's effect on an observable
costs one energy evaluation per stored frame instead of a full simulation.

### 2.1 The second-order term is also right

For a null-space field the first-order term vanishes by construction, so any
residual effect must be second order — and the same expansion predicts that too.
At both force levels the measured residual agrees with the second-order estimate
(−0.540 and −0.439 pairs, both within the error bar). The formula is not merely
right at leading order; its own correction term is right as well.

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
suppression of roughly 400×, and it holds out of sample.

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
particular question. It follows that no single scalar can summarise model
quality — not force RMSE, and not any replacement for it either. What can be
computed is a *vector* of response scores, one per observable anyone cares about.

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
| observable error spread | — | **30×** |
| ρ (force RMSE, truth) | **+0.835** | **+0.343** |
| ρ (response prediction, truth) | +0.982 | **+0.943** |

Inside the band the physics still varies by a factor of thirty, force error
explains almost none of it, and the response prediction explains nearly all of
it. See `figures/headline_regimes.png`.

So the corrected claim, which the data does support: **force error is a coarse
filter, not a selector.** It will tell you that a badly fitted model is bad. It
will not tell you which of your good models to use, and the designed
counterexamples of §1 show why — within a band, the ordering is set by the
projection, which force error does not measure.

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

## 3b. The same thing happens to models that were actually fitted

Ten models — a pair spline, a linear ACE-style basis, Behler-Parrinello networks
and E(3)-equivariant networks — fitted to 400 configurations of Lennard-Jones
argon, with training and test data drawn from separate Markov chains.

**The response formula works on fitted models too: 1.06 σ rms** between predicted
and measured observable shift across the ten, matching the 1.01 σ obtained on
designed error fields. That is the external-validity check for §2, and it passes.

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

This is the band effect of §3a appearing in fitted models rather than designed
ones — and it appears between two runs of the *same* model, which is as narrow a
band as it is possible to construct.

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

What remains is a genuine open question about either the reweighting
estimator's finite-sample behaviour under correlated samples — the Kish
effective sample size assumes independence, and the true independent count here
is smaller by the autocorrelation time — or something not yet identified. It is
recorded here unresolved rather than attributed to the nearest plausible cause.

Note that the exp07 measurements in §1 used a different observable and twice as
many frames and are well calibrated at 1.01 σ rms against prediction, so
whatever this is, it is not a general failure of either estimator.

### 5.3 The dilute-gas closed form

The estimator was checked against the exact `g(r) = exp(−βu)` limit at ρ\* =
0.079. The estimator and the exact reweighting identity agree to under 2 % in
every bin. Both disagree with a hand-derived closed form by 5.4 σ rms.

The hand calculation needed one correction along the way that is worth recording
because the estimator gets it right automatically: the number of pairs is
conserved, so a perturbation that depletes one shell must enrich others, and the
pair-separation distribution is *normalised*. Dropping the normalisation term
gives a prediction wrong by a factor of ten in the outer bins and wrong in sign
in the tail. A covariance is mean-subtracted, and that subtraction *is* the
normalisation, so the estimator never had the problem.

The residual 5.4 σ is most likely the `O(ρ)` correction to `g = exp(−βu)`, which
at ρ\* = 0.079 is not negligible. **This is not confirmed.** The test that would
confirm it — repeating at a quarter of the density and checking the deviation
falls — has not been run.

## 6. What is not here

- **exp03/exp04, the trained model zoo.** Three of the model implementations
  (linear/ACE, GAP kernel, Behler-Parrinello) did not complete: the agents
  writing them hit a session limit. The E(3)-equivariant network, the
  descriptors and the pair spline did land and are tested. The consequence is
  that every result above uses *designed* error fields rather than fitted
  models. That is the better test of the mechanism, for the reason given in
  `experiments/exp05_proxy_correlation/run.py` — it decouples the size of the
  error from its shape — but it is a weaker claim about external validity, and
  the study does not currently show that real fitted models occupy the regime
  the designed fields explore.
- **exp05 has not been run**, only written. It needs roughly an hour of compute.
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
