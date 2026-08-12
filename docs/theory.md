# Error fields, not error norms

*A linear-response account of how machine-learned potential error reaches physical observables.*

---

## 0. The question

A machine-learned interatomic potential is fitted, its force RMSE on a held-out
set is reported, and it is then used to run molecular dynamics from which some
physical quantity — a radial distribution function, a phonon spectrum, a
diffusion coefficient, an elastic constant — is extracted and compared to
experiment or to high-level theory.

The implicit inferential chain is: *small force error ⇒ good physics*. This
document works out what actually controls the second quantity given the first,
and finds that the chain is broken in a specific, quantifiable way. Force error
is a **norm** of the error field. What observables respond to is a
**projection** of the error field. Norms and projections are related only by an
inequality, and that inequality is loose in exactly the regime MLIPs live in.

Everything below is standard statistical mechanics assembled toward an unusual
question. The contribution is not the machinery; it is what the machinery says
about how these models should be evaluated, and the controlled numerical
programme in `experiments/` that tests whether it says it correctly.

---

## 1. Setup and notation

Fix `N` atoms in a volume `V` at temperature `T`, with `β = 1/k_BT`. Let
`x ∈ ℝ^{3N}` denote a configuration. Two potentials are in play:

- `U₀(x)` — the **reference**, the physical truth of the study. In this
  repository it is an analytic classical potential (Lennard-Jones,
  Stillinger–Weber, EAM), which is the methodological crux: `U₀` is then known
  *everywhere*, not merely on a test set.
- `U(x) = U₀(x) + δU(x)` — the **surrogate**, a fitted model. The **error
  field** `δU` is a function on configuration space, not a number.

Canonical averages in the two ensembles are

```
⟨A⟩₀ = (1/Z₀) ∫ A(x) e^{−βU₀(x)} dx ,     ⟨A⟩_U = (1/Z) ∫ A(x) e^{−βU(x)} dx
```

The object of interest is the **observable error**

```
Δ⟨A⟩ ≡ ⟨A⟩_U − ⟨A⟩₀
```

and the question is what functional of `δU` controls it.

The quantity practitioners actually measure is the force error. Since forces are
`F = −∇U`, the error in the forces is `δF = −∇δU`: the **gradient** of the error
field. The usual scalar summary is

```
RMSE_F² = (1/3N) ⟨ ‖∇δU‖² ⟩_test
```

Note already the structural mismatch: `Δ⟨A⟩` is a linear functional of `δU`,
while `RMSE_F` is a quadratic functional of `∇δU`. There is no reason for one to
determine the other, and §4 shows how badly it fails to.

---

## 2. The exact relation

No approximation is needed to begin. Multiplying and dividing by `e^{−βU₀}`,

```
⟨A⟩_U = ⟨A e^{−βδU}⟩₀ / ⟨e^{−βδU}⟩₀                                    (2.1)
```

This is exact for any `δU`. It is the free-energy-perturbation identity, and it
says something already worth pausing on: **the surrogate ensemble is completely
determined by the joint distribution of `(A, δU)` under the reference
ensemble.** Nothing else about the model matters. Two models with identical
`(A, δU)` joint statistics give identical physics for `A`, however different
their parameters, architectures, or force errors.

Equation (2.1) is exact but numerically treacherous — the exponential average is
dominated by rare configurations where `δU` is very negative — so we also need
its expansion, which is where the physics becomes legible.

---

## 3. Cumulant expansion, and the first-order result

Expand both sides of (2.1) in powers of `βδU`. Writing `δŨ ≡ δU − ⟨δU⟩₀` for the
centred error field and `Ã ≡ A − ⟨A⟩₀`:

```
⟨A e^{−βδU}⟩₀ = ⟨A⟩₀ − β⟨A δU⟩₀ + (β²/2)⟨A δU²⟩₀ − …
⟨e^{−βδU}⟩₀   = 1 − β⟨δU⟩₀ + (β²/2)⟨δU²⟩₀ − …
```

Dividing and collecting orders gives the central result of this document:

> **Result 1 (response expansion).**
> ```
> Δ⟨A⟩ = −β Cov₀(A, δU)  +  (β²/2) ⟨Ã δŨ²⟩₀  −  …
>       = Σ_{n≥1}  ((−β)^n / n!) · κ₀(A; δU, …, δU)          (n copies of δU)
> ```
> where `κ₀(·)` is the joint cumulant under the reference ensemble.

The `n = 1` term is the whole story whenever the error is small:

> **Result 1a (linear response).**
> ```
> Δ⟨A⟩ = −β Cov₀(A, δU) + O(δU²)
> ```

Three things about this deserve emphasis, because they are the entire argument
of this repository.

**(i) It is a covariance, not a norm.** `Cov₀(A, δU)` can vanish identically for
an error field of arbitrarily large magnitude, and can be large for an error
field of tiny magnitude. Factoring the covariance into scales and a correlation
is a definition, not an estimate:

```
Δ⟨A⟩ = −β · σ₀(A) · σ₀(δU) · ρ₀(A, δU)  +  O(δU²)                      (3.1)
```

with `ρ₀` the Pearson correlation under the reference ensemble. Equation (3.1)
is an *equality* at first order — every factor on the right is defined so that
it is — and that is exactly why it indicts the reported metric. The magnitude
`σ₀(δU)` is one factor of three, and `ρ₀ ∈ [−1, 1]` is free: nothing in the
magnitude constrains it.

Bounding rather than factoring gives the Cauchy–Schwarz inequality, which is a
different and weaker statement:

```
|Δ⟨A⟩| ≤ β · σ₀(A) · σ₀(δU)  +  O(δU²)                                (3.1a)
```

obtained by setting `|ρ₀| ≤ 1`. This is the only sense in which a magnitude
bounds an observable error, and it is attained only when the error field is
perfectly correlated with the observable — the designed worst case of
`exp07`, not the generic one. Reporting `σ₀(δU)`, let alone a norm of its
gradient, therefore reports a ceiling that is generically far from reached. The
correlation `ρ₀(A, δU)` is what distinguishes a harmless model from a harmful
one, and it is nowhere in standard MLIP practice.

An earlier draft of this document wrote (3.1) with a `≤` and called it the
Cauchy–Schwarz bound, which is neither: with `ρ₀` retained it is an identity,
and the inequality only appears once `ρ₀` is discarded. The distinction matters
because the two support opposite readings — (3.1a) says a small error field
cannot do much damage, while (3.1) says how much damage it does is set by a
factor the field's magnitude does not determine.

**(ii) It requires only reference-ensemble samples.** Every expectation in
Result 1a is taken under `⟨·⟩₀`. Given a reference trajectory and the ability to
evaluate both `U₀` and `U` on its frames, `Δ⟨A⟩` is predictable **without ever
running molecular dynamics with the surrogate.** For a study that must evaluate
dozens of models against a dozen observables this is a large practical saving;
more importantly, it is a falsifiable prediction, and `experiments/exp06` tests
it against direct MD.

**(iii) The second-order term is computable, and is a candidate warning light.**
The `n = 2` term `(β²/2)⟨ÃδŨ²⟩₀` is estimable from the very same samples, and
its size relative to the first-order term is the obvious self-diagnostic: a small
ratio ought to mean the linear prediction is trustworthy, and a large one that
the error field is not a perturbation at all.

That it *ought* to work is not evidence that it does, and the measurements in
this repository so far say it does not. `exp06` flags two cases as trustworthy
of which only one agrees with direct sampling. `exp09` finds that adding the
second-order term to the prediction makes the residuals worse on every measure —
larger rms, larger field-to-field spread, larger interaction term — rather than
better. Both are recorded in `docs/RESULTS.md`. Until a threshold is frozen on
one set of error fields and its false-trust rate measured on another, the ratio
is a heuristic under test, not a validated gate. It is used in this repository
to *flag* cases for attention and never to certify one.

### 3.1 Vector observables

`g(r)` is not a scalar; it is a vector of bin occupancies `A_k(x)`. Result 1a
applies componentwise, giving a predicted **difference curve**

```
Δ⟨g(r_k)⟩ = −β Cov₀(g_k, δU)
```

The covariance is now a vector, and the natural summary of "how wrong is the
structure" is a norm of that vector — but note the norm is taken *after* the
projection, which is the opposite order from what force RMSE does.

### 3.2 An exactly solvable limit

In the low-density limit of a pair fluid, `g(r) = e^{−βu(r)}` exactly. Perturbing
the pair potential `u → u + δu` gives, exactly,

```
δg(r) = −β g(r) δu(r) + O(δu²)
```

which Result 1a must reproduce. It does, and this is the sharpest available unit
test of the whole response module: a closed-form answer against which the
sampled covariance estimator can be validated to statistical precision.
`tests/test_response.py` checks it.

### 3.3 Free energy behaves differently, and the contrast is instructive

The same expansion applied to the free energy gives Zwanzig's relation

```
ΔF = −(1/β) ln⟨e^{−βδU}⟩₀ = ⟨δU⟩₀ − (β/2) Var₀(δU) + …                (3.2)
```

So the free energy error is controlled at leading order by the **mean** of the
error field, while observable errors are controlled by its **covariances**. A
model can therefore have an excellent free energy and poor structure, or the
reverse. This is already enough to rule out a *task-independent* scalar: no
number computed without reference to which quantity is wanted can order models
the same way for every quantity, because the orderings genuinely differ. It is a
useful sanity check on intuition — "the model is 3 meV/atom off" is a statement
about `⟨δU⟩₀` and says nothing about `g(r)`.

It does **not** rule out a useful scalar. Once a task is specified — a set of
observables and a tolerance for each — a scalar loss over that set is perfectly
well defined, and minimising a worst-case ratio `max_j |Δ⟨A_j⟩| / τ_j` is a
scalar criterion that this argument leaves entirely intact. An earlier draft
wrote "no single scalar can summarise model quality", which overstates what
follows from the mathematics. The defensible statement is: *there is no scalar
that is independent of the task definition and still orders models consistently
for all observables.*

---

## 4. Why force error is the wrong summary — the frequency argument

Result 1a explains *that* norms are the wrong object. This section explains
*why the particular norm the field has settled on is especially bad*, and turns
that into a quantitative, falsifiable prediction.

Decompose the error field into modes. Take any orthonormal basis `{φ_k}` on the
relevant region of configuration space, labelled so that `k` indexes a spatial
frequency, and write `δU = Σ_k c_k φ_k`. Then

```
Cov₀(A, δU) = Σ_k c_k Cov₀(A, φ_k)                                     (4.1)
‖δF‖²      = ⟨‖∇δU‖²⟩ = Σ_{k,k'} c_k c_{k'} ⟨∇φ_k · ∇φ_{k'}⟩ ~ Σ_k k² |c_k|²   (4.2)
```

The gradient in (4.2) contributes a factor of `k²`: **force error weights the
error field by the square of its spatial frequency.** The covariance in (4.1)
applies no such weight; worse, physical observables `A` are smooth, low-frequency
functions of configuration, so `Cov₀(A, φ_k)` typically *decays* with `k`.

The two functionals therefore weight the error spectrum in opposite directions.
This yields two failure modes, and they are not symmetric in their consequences:

- **False alarm.** High-frequency error — wiggly, rapidly-oscillating
  interpolation error of the kind neural networks produce between training
  points — inflates force RMSE by `k²` while contributing almost nothing to any
  observable. Such a model is *reported* as bad and *is* fine.
- **False confidence.** Low-frequency, systematic error — a slightly wrong
  long-range tail, a slightly wrong well depth, a smooth bias from an
  under-covered region of the training distribution — contributes almost nothing
  to force RMSE and can shift observables substantially. Such a model is
  *reported* as good and *is not*. This is the dangerous direction, and it is
  precisely the error structure that regularised fits and smooth kernels prefer
  to produce.

### 4.1 A concrete scaling — and the limits of the estimate

Make this sharp with a one-parameter family. Perturb a pair potential by a
Gaussian shell bump of amplitude `a`, centre `r₀`, width `w`:

```
δu(r) = a · exp[ −(r − r₀)² / 2w² ]
```

Its contribution to the force error scales as `|δu′| ~ a/w`, and the fraction of
pairs falling inside the shell scales as `w`, so

```
RMSE_F ~ a / √w                                                        (4.3)
```

Its coupling to the radial distribution function is an integral over the shell,

```
|Cov₀(g, δU)| ~ a · w                                                  (4.4)
```

Therefore, **at fixed force error**, the observable error scales as

```
|Δ⟨g⟩| / RMSE_F  ~  w^{3/2}                                            (4.5)
```

which is unbounded. Two error fields with *identical force RMSE* would then
differ in observable error by any factor one likes, simply by differing in
width.

**Treat (4.5) as a heuristic, not as a prediction.** Both of its inputs are
crude, and each has an explicit regime of validity that a real measurement
leaves quickly:

- (4.3) counts pairs in the shell as `∝ w`, which holds only while `w` is small
  compared with the scale over which `g(r)` varies. In liquid argon near the
  first peak that scale is a few tenths of an angstrom, so the estimate is
  already marginal at `w = 0.3 Å`.
- (4.4) treats the observable as integrating the whole bump. Once `w` exceeds
  the resolution of the observable — the bin width of a discretised `g(r)` — the
  coupling saturates, and beyond that the bump pushes pairs into the bin from
  one side while pushing them out on the other, so the two contributions partly
  cancel.

Both objections are analytic and stand on their own; neither needs a
measurement to establish. The exponent in (4.5) is therefore not a
no-free-parameter prediction, and an earlier draft of this document should not
have claimed it was.

*(A preliminary run appeared to confirm the breakdown quantitatively, and those
numbers were briefly quoted here. They have been removed: that run sampled a
reference ensemble which had not equilibrated — see
`experiments/equilibrate.py` — so its numbers were not measurements of the
liquid at all. The measured width dependence is whatever `experiments/exp06`
reports on a verified-stationary ensemble, and nothing else.)*

What survives, and what the experiments actually test, is the qualitative
statement that (4.3) and (4.4) have *different* dependence on the shape of the
error field, so the ratio in (4.5) is not a constant — force error and
observable error cannot both be summaries of the same thing.
`experiments/exp06` measures the width dependence over a decade, using the norm
of the whole predicted `Δg(r)` curve rather than a single bin so that the
saturation artefact above does not confound it, and reports the fitted exponent
whatever it turns out to be. The *sharp* test of the mechanism is not this
scaling at all but the designed null-space and aligned fields of §5, where the
projection is controlled directly rather than through a proxy for it.

---

## 5. Which errors are invisible

It is worth naming the structures in `δU` that Result 1a says are free.

**Constants.** `δU = c` shifts nothing. Trivial, but it is the reason
energy-only metrics must be computed on *differences* or per-atom after
alignment.

**Errors off the typical set.** The average `⟨·⟩₀` is dominated by the typical
set of the reference ensemble. Error supported on configurations that are never
visited at temperature `T` — high-energy, compressed, or defected structures —
contributes nothing to `Δ⟨A⟩` at that temperature, *and* nothing to a test-set
force error computed on samples from the same ensemble. Such error is invisible
to both. It becomes visible only under extrapolation: a different temperature, a
phase transition, a rare event. This is the honest steel-manning of force error —
it is not that it fails to see this error, it is that *nothing* sampled from the
training distribution sees it, which is an argument for out-of-distribution
probes rather than for better in-distribution norms.

**Errors orthogonal to the specific observable.** The most important case, and
the one that makes model quality irreducibly observable-dependent: `Cov₀(A, δU)`
can vanish for one observable and not another. A scalar that does not know which
observable is wanted is being asked for a projection onto every direction at
once, which is why no such number exists; a scalar defined *after* the
observables and their tolerances are fixed is a different object and is not
excluded here. The constructive version of this — deliberately building `δU` orthogonal to
a chosen observable while carrying large force error — is `NullSpacePerturbation`
in `atomlab/potentials/perturbations.py` and is the sharpest falsification test
in the programme (`experiments/exp07`).

---

## 6. Dynamical observables: what the theory does and does not cover

Everything above concerns **static** averages. Transport coefficients are
different in kind, and honesty requires separating what follows from what is
conjecture.

A Green–Kubo transport coefficient is an integral of an equilibrium
time-correlation function, e.g. for self-diffusion

```
D = (1/3) ∫₀^∞ ⟨v(0) · v(t)⟩ dt
```

Perturbing `U₀ → U₀ + δU` changes this in two ways: the **ensemble** the average
is taken over (covered by Result 1a) and the **dynamics** generating `v(t)`
(not covered — the propagator itself changes). The second effect has no
equally clean expression, and the trajectory-level divergence it produces grows
exponentially with the Lyapunov exponent, so a trajectory-wise perturbative
treatment is hopeless past a few picoseconds.

This suggests, but does not prove, a testable asymmetry:

> **Conjecture.** Force error should be a *better* proxy for dynamical
> observables than for static ones, because `δF` enters the equations of motion
> directly, whereas static averages see only the projection `Cov₀(A, δU)`.

`experiments/exp05` measures rank correlations separately for static and
dynamical observables and can confirm or refute this. It is stated here in
advance so that the result counts as a test rather than as a story told
afterwards. A refutation would be at least as interesting as a confirmation.

---

## 7. Estimation

### 7.1 The oracle estimator

Given `M` frames `{x_m}` from a reference MD trajectory, evaluate `A_m = A(x_m)`
and `δU_m = U(x_m) − U₀(x_m)`, and form the sample covariance

```
Δ̂⟨A⟩ = −β · (1/(M−1)) Σ_m (A_m − Ā)(δU_m − δŪ)                        (7.1)
```

MD frames are correlated, so the error bar must come from **block bootstrap**
with a block length of several correlation times, never from the naive
`1/√M`. The correlation time is estimated from the integrated autocorrelation of
the product series `(A_m − Ā)(δU_m − δŪ)`, which is the series that actually
enters — using `A`'s own correlation time underestimates it when `δU` is
slowly varying.

### 7.2 The non-perturbative check

Equation (2.1) evaluated on the same samples gives the reweighted estimate

```
⟨A⟩_U ≈ Σ_m A_m w_m / Σ_m w_m ,     w_m = e^{−βδU_m}                   (7.2)
```

exact to all orders but with variance that grows with `Var(βδU)`. Its
trustworthiness is quantified by the effective sample size
`ESS = (Σw)² / Σw²`. Comparing three numbers on identical samples —
first-order (7.1), reweighted (7.2), and direct MD with the surrogate —
is the cleanest available validation: they must agree in the small-`δU` limit
and must diverge in an understood order as `δU` grows. `experiments/exp06` runs
exactly this three-way comparison as a function of perturbation amplitude.

### 7.3 Which ensemble to sample

Result 1a expands around `U₀`, so its expectations are under `⟨·⟩₀`. Expanding
instead around `U` gives the mirror statement

```
Δ⟨A⟩ = +β Cov_U(A, δU) + O(δU²)                                        (7.3)
```

The two estimates use different samples and agree only to the extent that linear
response holds. **Their disagreement is a free, assumption-light diagnostic for
the breakdown of linear response** — the same logic that underlies Bennett
acceptance-ratio methods. Both directions are implemented, and the experiments
report both.

### 7.4 The practical estimator, without an oracle

The oracle estimator needs `δU = U − U₀`, and a practitioner has no `U₀`. The
usable substitute replaces the unknown truth with a **committee mean** over `M`
independently trained models:

```
δU_i ≈ U_i − (1/M) Σ_j U_j                                             (7.4)
```

Then `−β Cov(A, δU_i)` estimates how far model `i`'s prediction of `⟨A⟩` sits
from the committee consensus, and the spread across `i` is an **uncertainty
estimate for the observable itself** — obtained from single-point energy
evaluations on an existing trajectory, with no MD per model.

This is the practical payoff and simultaneously the claim most worth being
sceptical of. Committee spread estimates *disagreement*, not *error*; models
sharing an architecture, a training set and an inductive bias share their
systematic error, and shared error cancels exactly in (7.4). The failure mode is
therefore predictable and one-directional: the committee predictor should track
the oracle for architecture-diverse committees and under-report for homogeneous
ones. `experiments/exp08` is built to detect this, with deliberately homogeneous
and deliberately heterogeneous committees as the two arms, and it is designed so
that a negative result is reportable rather than absorbed.

---

## 8. What would falsify this

The argument is only worth making if it can fail. It fails if:

1. Force RMSE turns out to rank-correlate strongly (Spearman `ρ > 0.9`) with
   every observable error across an architecture-diverse model zoo. The
   frequency argument of §4 would then be describing a subspace real models do
   not occupy.
2. The first-order prediction (7.1) fails to match direct MD in the regime where
   the second-order diagnostic says it should hold. That would indicate an error
   in the derivation or in the estimator, not in the field's practice.
3. The designed null-space perturbations of §5 fail to invert the ranking —
   i.e. constructing `δU` with large `‖δF‖` and zero `Cov₀(A, δU)` still damages
   `A`. That would mean the linear term is not the dominant one at realistic
   error magnitudes, which would be a more interesting result than the one
   expected here.

Each is checked explicitly, and the pre-registration of these criteria in
advance of running the experiments is deliberate.

---

## 9. Relation to existing work

The observation that force error does not predict simulation quality is not new
as an *empirical* finding: benchmark studies comparing MLIPs on downstream
simulation tasks have reported exactly this dissociation, and the practical
advice to validate on simulation observables rather than on held-out forces is
increasingly standard.

What this repository adds is (a) a *mechanism* — the covariance formula and its
frequency-weighting consequence — rather than an observation; (b) an
**analytic-oracle** methodology in which `δU` is known exactly at every point of
configuration space, so the mechanism can be tested rather than inferred;
(c) **designed** perturbations that make the prediction falsifiable
constructively rather than only observationally; and (d) a cheap predictor with
a stated failure mode and an experiment built to expose it.

The statistical mechanics is textbook — free-energy perturbation, cumulant
expansions, Zwanzig's relation, Bennett's acceptance ratio. The contribution is
the redirection: these are usually deployed to *compute* free energies, and here
they are deployed to *audit a machine learning evaluation protocol*.
