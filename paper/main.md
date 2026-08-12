# Error fields, not error norms: how the error in a fitted interatomic potential reaches a physical observable

**[AUTHOR NAME]**, [AFFILIATION], [EMAIL]

*Prepared for* The Journal of Chemical Physics *(Regular Article). This is the
Markdown rendering of `paper/main.tex`: same content, same numbers, same section
order. Figures are the `.pdf` files in `../figures/`.*

---

## Abstract

*(247 words.)*

Machine-learned interatomic potentials are selected on held-out force RMSE, then
used to compute physical observables. We audit that inference. For a static
observable *A* at inverse temperature β, with error field δ*U* (fitted minus
reference potential), perturbation theory gives Δ⟨*A*⟩ = −β Cov₀(*A*, δ*U*): the
shift in a static observable is minus β times the covariance, under the reference
ensemble, of the observable with the potential's error field. That is a
projection; force RMSE is a norm of the error field's gradient. On an analytic
Lennard-Jones reference, where δ*U* is known everywhere, four error fields
matched in force RMSE to 0.5 % shift a target pair count from 0.3σ to 16.2σ,
ordered by the Pearson correlation ρ(*A*, δ*U*). Predicted from
reference-ensemble samples alone, with no surrogate simulation, those shifts
match measurement to 1.01σ over eight designed fields and 1.06σ over ten fitted
models. Force error ranks models well across three decades (Spearman ρ = 0.83
[0.67, 0.93], *n* = 34) but is unresolved from zero inside a factor-2.6 band
where the observable error still spans 30× (ρ = 0.34 [−0.26, 0.77]) while the
response prediction retains ρ = 0.94 [0.76, 1.00], both *n* = 15: a coarse
filter, not a selector. That refutes one of our predictions; two more fail: our
smoothness-ratio metric carries no information (ρ = −0.02 [−0.37, 0.35],
*n* = 34), and our *w*^{3/2} width scaling measures exponent 0.56. Scope: one
Lennard-Jones state point, one pair-radial observable class.

---

## 1. Introduction

A machine-learned interatomic potential is fitted to energies and forces, its
root-mean-square force error on a held-out set is reported, the model is used to
run dynamics, a physical observable is extracted from that trajectory, and the
observable is believed. Every step of that chain is routine, and the chain as a
whole encodes an inference: small force error implies good physics. This paper
asks what actually connects the two ends, and finds that the connection is not
the one the reported metric measures.

The mismatch is structural and can be stated in one sentence. Writing the fitted
potential as *U* = *U*₀ + δ*U*, where *U*₀ is the reference and δ*U* is the
**error field** — a function on configuration space, not a number — the shift in
a static ensemble average ⟨*A*⟩ is, at leading order, a linear functional of
δ*U*, while the squared force RMSE is a quadratic functional of ∇δ*U*. A linear
functional of a function and a quadratic functional of its gradient are related
by an inequality and by nothing stronger. The inequality is loose, and Section 4
shows how loose.

Testing that claim requires knowing δ*U*, which in ordinary practice is exactly
what one does not have: the reference is density-functional theory, available on
a finite set of computed configurations. We therefore take an analytic classical
potential as the ground truth. Lennard-Jones argon [1, 24, 25] is uninteresting
as physics and exact as a reference, and that is the point. Because *U*₀ is known
everywhere in configuration space rather than on a test set, δ*U* is exactly
computable at every sampled configuration, reference observables converge to
arbitrary precision so that a discrepancy is a property of the surrogate rather
than of the reference, and — decisively — error fields can be *designed* to have
chosen statistical properties. That converts an observational claim about a
correlation between two reported numbers into a constructive one: we can build
models that a force-error report cannot distinguish and that disagree about the
physics by sixteen standard errors.

The empirical half of this story is already known. Benchmarks that run
simulations with machine-learned potentials rather than only scoring them on
held-out forces have reported that force error and simulation quality come apart
(Fu *et al.* 2023; Stocker *et al.* 2022), and the recommendation to validate on
simulation observables is now explicit in the methodological literature (Morrow
*et al.* 2023). Practitioners have also reported specific failures that a force
RMSE does not anticipate (Kovács *et al.* 2021; Deng *et al.* 2025; Póta *et al.*
2024). The statistical machinery we use is equally standard: the identity below
is Zwanzig's free-energy-perturbation relation (1954) expanded in cumulants, and
the same covariance appears as a parameter gradient in differentiable trajectory
reweighting (Thaler and Zavadlav 2021) and as the vehicle for propagating
committee uncertainty to thermodynamic averages (Imbalzano *et al.* 2021). We
claim no novelty for it, here or anywhere below. What we claim is narrower and we
state it defensively: the reading of that textbook identity as an indictment of
the reported evaluation metric; the analytic-oracle construction that makes δ*U*
exactly known; the constructive falsification at force RMSE matched to 0.5 %; and
the decomposition of the proxy's performance into an across-decades regime and a
within-band regime, which turns a vague "weak proxy" claim into two sharp and
opposite ones.

**Contributions.**

1. A derivation, from the free-energy-perturbation identity, that the error in a
   static observable is a covariance between the observable and the error field,
   with the full cumulant series available and its second-order term usable as a
   self-diagnostic (Section 2).
2. A quantitative validation of that prediction on an analytic reference: 1.01σ
   root-mean-square residual over eight designed error fields (*n* = 8) and 1.06σ
   over ten fitted models (*n* = 10), at a cost of one energy evaluation per
   stored frame per model and no simulation with the surrogate (Sections 4.2 and
   4.5).
3. A constructive counterexample set: four error fields matched in force RMSE to
   0.5 %, whose target-observable shifts run from −0.220 ± 0.640 to
   +9.891 ± 0.611 pairs, ordered by ρ(*A*, δ*U*) (Section 4.1).
4. A decomposition of force RMSE's value as a ranking metric: strong across three
   decades, unresolved within a narrow band where the physics still varies by 30×
   (Section 4.4). This refutes our own pre-registered prediction and replaces it
   with a more useful one.
5. Three refuted predictions of our own, reported as results rather than as
   caveats (Section 5), together with a demonstrated boundary of validity of the
   linear estimator and an open, unresolved 35 % discrepancy between two ways of
   measuring the same shift.
6. A methodological result of independent interest: a reference ensemble that was
   a slowly melting crystal, wrong in energy by 30–40 %, and invisible to every
   sampler diagnostic we had (Section 3.3).

Section 5, "What did not work," is not an appendix and not a limitations
paragraph. Three predictions made in this study were refuted by this study: a
*w*^{3/2} scaling of observable error with the width of the error field, a
smoothness-ratio metric we proposed as a replacement diagnostic, and the
prediction that force RMSE would rank models weakly. The first two are wrong. The
third is wrong in an instructive direction — force RMSE ranks models *well*
across decades and poorly only inside a narrow band — and the corrected statement
is stronger and more useful than the one we set out to demonstrate.

---

## 2. Theory

### 2.1. Setup

Fix *N* atoms in a volume *V* at temperature *T*, with β = 1/*k*<sub>B</sub>*T*,
and let *x* ∈ ℝ^{3N} be a configuration. The reference potential is *U*₀(*x*) and
the surrogate is

    U(x) = U₀(x) + δU(x),

so the error field δ*U* is a function on configuration space. The error in the
forces is its gradient, δ*F* = −∇δ*U*, and the scalar summary in universal use is

    RMSE_F² = (1/3N) ⟨‖∇δU‖²⟩_test.                                   (1)

Canonical averages in the two ensembles are ⟨·⟩₀ and ⟨·⟩_U, and the quantity of
interest is the observable error Δ⟨*A*⟩ ≡ ⟨*A*⟩_U − ⟨*A*⟩₀.

### 2.2. The exact relation

Multiplying and dividing by e^{−β*U*₀} gives, with no approximation,

    ⟨A⟩_U = ⟨A e^{−βδU}⟩₀ / ⟨e^{−βδU}⟩₀.                              (2)

This is the free-energy-perturbation identity (Zwanzig 1954), exact for any δ*U*,
and it already carries the argument of this paper: the surrogate ensemble is
determined by the joint distribution of (*A*, δ*U*) under the reference ensemble,
and by nothing else about the model. Two surrogates with identical (*A*, δ*U*)
joint statistics predict identical physics for *A* whatever their architectures,
parameter counts or force errors. Equation (2) is numerically treacherous,
because the exponential average is dominated by rare configurations where δ*U* is
very negative (Bennett 1976; Shirts and Chodera 2008; Frenkel and Smit 2002), so
we also need its expansion.

### 2.3. Cumulant expansion

Expanding numerator and denominator of Eq. (2) in powers of βδ*U* and collecting
orders gives the central result.

**Result 1 (response expansion).**

    Δ⟨A⟩ = Σ_{n≥1} ((−β)ⁿ / n!) κ₀(A; δU, …, δU)   (n copies of δU),     (3)

where κ₀ is the joint cumulant under the reference ensemble. Written out to
second order, with Ã = *A* − ⟨*A*⟩₀ and δŨ = δ*U* − ⟨δ*U*⟩₀,

    Δ⟨A⟩ = −β Cov₀(A, δU) + (β²/2)⟨Ã δŨ²⟩₀ − …

**Result 1a (linear response).**

    Δ⟨A⟩ = −β Cov₀(A, δU) + O(δU²).                                    (4)

Three consequences carry the rest of the paper.

*(i) It is a covariance, not a norm.* Cov₀(*A*, δ*U*) can vanish identically for
an error field of arbitrarily large magnitude and can be large for an error field
of tiny magnitude. Magnitude enters only through the Cauchy–Schwarz bound

    |Δ⟨A⟩| ≤ β σ₀(A) σ₀(δU) |ρ₀(A, δU)|,                               (5)

with ρ₀ the Pearson correlation under the reference ensemble. Force RMSE reports
none of the three factors on the right: not the observable's own fluctuation
σ₀(*A*), not the error field's spread σ₀(δ*U*), and above all not the
correlation, which is the factor that separates a harmless model from a harmful
one and which appears nowhere in standard practice.

*(ii) It needs only reference-ensemble samples.* Every expectation in Eq. (4) is
taken under ⟨·⟩₀. Given a stored reference trajectory and the ability to evaluate
*U* and *U*₀ on its frames, Δ⟨*A*⟩ is predictable without ever simulating the
surrogate: one energy evaluation per stored frame per model. For screening many
models against many observables this is a large saving, and it is also what makes
the claim falsifiable, since the prediction and the direct measurement are then
separate experiments.

*(iii) The second-order term is a computable warning light.* The *n* = 2 term is
estimable from the same samples. Its ratio to the first-order term is a
self-diagnostic: where the ratio is small the linear prediction is trustworthy,
and where it is not, the error field is not a perturbation and no linear
reasoning about it is safe — including the field's own implicit assumption that a
little more accuracy helps a little. We report this diagnostic wherever it was
recorded, and report its absence where it was not.

### 2.4. Vector observables, and the contrast with free energy

The radial distribution function is not a scalar but a vector of bin occupancies
*A_k*, and Eq. (4) applies componentwise, giving a predicted difference curve
Δ⟨*A_k*⟩ = −β Cov₀(*A_k*, δ*U*). The natural summary of "how wrong is the
structure" is then a norm of that vector — but the norm is taken *after* the
projection, which is the opposite order of operations from force RMSE, where the
norm is taken first and no projection is ever performed.

The same expansion applied to the free energy gives Zwanzig's relation,

    ΔF = −(1/β) ln⟨e^{−βδU}⟩₀ = ⟨δU⟩₀ − (β/2) Var₀(δU) + …,

so the free-energy error is controlled at leading order by the *mean* of the
error field while observable errors are controlled by its *covariances*. A model
can therefore have an excellent free energy and poor structure, or the reverse.
Already at this level no single scalar can summarise model quality.

### 2.5. Why this particular norm

Result 1a says that norms are the wrong kind of object. A second argument says
why the norm the field has settled on is especially poorly matched. Expand the
error field in an orthonormal basis {φ_k} labelled by spatial frequency,
δ*U* = Σ_k c_k φ_k. Then

    Cov₀(A, δU) = Σ_k c_k Cov₀(A, φ_k),                                (6)
    ⟨‖∇δU‖²⟩ ~ Σ_k k² |c_k|².                                          (7)

The gradient in Eq. (7) contributes a factor *k*²: force error weights the error
spectrum by the square of its spatial frequency. The covariance in Eq. (6)
applies no such weight, and because physical observables are smooth functions of
configuration, Cov₀(*A*, φ_k) typically *decays* with *k*. The two functionals
weight the same error spectrum in opposite directions, which produces two failure
modes that are not symmetric in their consequences. A *false alarm* is
high-frequency interpolation wiggle of the kind flexible regressors produce
between training points: inflated by *k*² in the force error, contributing almost
nothing to any smooth observable, reported as bad and in fact harmless. A *false
confidence* is smooth systematic error — a slightly wrong tail, a slightly wrong
well depth, a bias from an under-covered region of the training distribution:
nearly free in force RMSE, and able to move observables. The second is the
dangerous direction, and it is the error structure that regularised fits and
smooth kernels prefer to produce.

### 2.6. A heuristic, retracted in advance

It is tempting to make Section 2.5 quantitative with a one-parameter family.
Perturb a pair potential by a Gaussian shell bump of amplitude *a*, centre *r*₀
and width *w*, δ*u*(*r*) = *a* exp[−(*r* − *r*₀)² / 2*w*²]. Its gradient scales as
*a*/*w* and the fraction of pairs inside the shell scales as *w*, giving
RMSE_F ~ *a*/√*w*; its coupling to the radial distribution function is an
integral over the shell, |Cov₀(*g*, δ*U*)| ~ *a w*. At fixed force error the ratio
would then grow as

    |Δ⟨g⟩| / RMSE_F ~ w^{3/2},                                         (8)

which is unbounded: two error fields with identical force RMSE could differ in
observable error by any factor one likes.

We retract Eq. (8) as a prediction, on analytic grounds and in advance of the
measurement. Both inputs have regimes of validity that a real system leaves
quickly. Counting pairs in the shell as ∝ *w* holds only while *w* is small
compared with the scale on which *g*(*r*) varies, which near the first peak of
liquid argon is a few tenths of an ångström, so the estimate is already marginal
at *w* = 0.3 Å. And treating the observable as integrating the whole bump fails
once *w* exceeds the bin width of a discretised *g*(*r*): the coupling saturates,
and beyond saturation the bump pushes pairs into a bin from one side while
pushing them out on the other, so the two contributions partly cancel. Neither
objection needs a measurement. What survives, and what we do test, is the weaker
statement that force error and observable error depend differently on the *shape*
of the error field, so their ratio is not a constant. Section 4.6 measures that
ratio varying by a factor of 4.7 at fixed force error, and Section 5.1 reports
the measured exponent, 0.56, against the 1.5 of Eq. (8).

### 2.7. What the theory does not cover

Everything above concerns static averages. Transport coefficients are different
in kind: perturbing *U*₀ → *U*₀ + δ*U* changes both the ensemble the average is
taken over, which Result 1a covers, and the propagator generating the dynamics,
which it does not. Trajectory-level divergence grows exponentially with the
Lyapunov exponent, so a trajectory-wise perturbative treatment is hopeless past a
few picoseconds.

This suggests an asymmetry, which we state as a conjecture and label
pre-registered and untested: force error should be a *better* proxy for dynamical
observables than for static ones, because δ*F* enters the equations of motion
directly whereas static averages see only the projection Cov₀(*A*, δ*U*). There
is independent literature support for the asymmetry — Wu *et al.* (2024) derive a
correction to lattice thermal conductivity that depends on a *norm* of the force
error, which is exactly what one expects when δ*F* acts through the dynamics
rather than through a projection. The experiment designed to test our conjecture,
a rank correlation computed separately for static and dynamical observables, was
not run, and nothing in this paper bears on it.

---

## 3. Methods

### 3.1. System

The reference system is Lennard-Jones argon, *N* = 108 atoms in a cubic cell with
periodic boundary conditions at number density ρ = 0.0200 Å⁻³ and temperature
*T* = 120 K; in reduced units ρ* = ρσ³ = 0.790 and
*T*\* = *k*<sub>B</sub>*T*/ε = 1.004, a fluid state point below the freezing
density at that temperature. The parameters are ε = 0.0103 eV, σ = 3.405 Å,
cutoff 7.0 Å, in the **shifted-force** form.

The cutoff mode is not cosmetic. Table VII in Appendix A gives the energy and
force discontinuities of the four available modes measured at *r_c* ∓ 10⁻⁷ Å. A
merely shifted potential has a continuous energy and a force discontinuity of
1.8 × 10⁻⁴ eV/Å, which delivers an impulse every time a pair crosses the cutoff.
This study measures small systematic differences between ensembles; an artefact
of that kind would not be distinguishable from the effect under investigation.

### 3.2. Sampling

Static observables are sampled with Hamiltonian Monte Carlo (Duane *et al.* 1987;
Neal 2011), not molecular dynamics. Each proposal draws momenta from a
Maxwell–Boltzmann distribution, integrates a short velocity Verlet trajectory
(Swope *et al.* 1982), and accepts on the change in the total Hamiltonian.
Because velocity Verlet is symplectic and time-reversible the proposal is
symmetric, so the stationary distribution is exactly e^{−β*U*} *regardless of
integration error*, which appears only as a reduced acceptance rate and never as
a biased distribution.

That property is the reason for the choice, and it forecloses an objection that
would otherwise be fatal: a thermostat sampling something slightly other than the
canonical distribution would introduce exactly the kind of small systematic
difference between two ensembles that this study attributes to model error.
Hamiltonian Monte Carlo removes the possibility by construction. Production
settings are 8 leapfrog steps per proposal with the step size adapted during
burn-in to an acceptance of 0.75, giving ≈ 30 fs after adaptation. An independent
single-particle Metropolis sampler sharing no code beyond the potential is
included as a cross-check.

The sampler was validated against exact equipartition on a 32-atom Einstein
crystal at 100 K, where both moments are known in closed form:
⟨*U*⟩ = 0.41462 ± 0.00196 eV against the exact (3*N*/2)*k*<sub>B</sub>*T* =
0.41363 eV (0.5σ), and Var(*U*) = 0.003459 eV² against the exact
(3*N*/2)(*k*<sub>B</sub>*T*)² = 0.003564 eV² (Table VIII). Both moments matter,
because an incorrect temperature scale reproduces the mean and fails on the
variance.

### 3.3. Equilibration, and the failure that made it necessary

The liquid is built by scaling an fcc lattice to liquid density. That
configuration is a crystal, and sampling it directly produces a slowly melting
crystal rather than a liquid.

On the original protocol the global Steinhardt bond-order parameter
(Steinhardt *et al.* 1983; Lechner and Dellago 2008) *Q*₆ fell monotonically from
0.5745, the perfect fcc value, to 0.43 over 600 proposals and was still falling,
while the potential energy climbed monotonically and was still climbing. The
equilibrated liquid sits at −0.0311 eV/atom; those runs were sampling −0.036 to
−0.043 eV/atom, so the reference ensemble was wrong in energy by 30–40 %.

No sampler diagnostic caught this. Acceptance rate and integration error were
healthy throughout, because they measure whether the Markov chain is being
*simulated* correctly, not whether it has reached its stationary distribution.
Those are different questions and only the second bears on an ensemble average.
We record the failure at length because it is invisible in exactly the way that
matters: every quantity in this paper is a small difference between two
ensembles, and an unrelaxed reference ensemble does not announce itself — it
simply moves the answer.

The protocol now used melts at 5× the target temperature for a few hundred
proposals, anneals to the target temperature, and refuses to return a
configuration with *Q*₆ > 0.20. Melting at the target temperature also works but
is an order of magnitude slower, because the barrier to be crossed is precisely
the one that makes the crystal metastable. A 108-atom liquid gives *Q*₆ ≈ 0.08
and perfect fcc gives 0.5745, so the threshold is far from both. After the
protocol, *Q*₆ = 0.074 and *U*/*N* = −0.0315 eV with no drift between the halves
of the production run. Every production trajectory is then split in half and both
its energy and its order parameter are compared between halves; a drift exceeding
three combined standard errors *raises* rather than warns, because a failure mode
with no outward sign will eventually be ignored if it is only a warning. The
quick-run mode that shrinks every other size parameter for smoke tests
deliberately never shortens equilibration.

### 3.4. Observables

The measured quantities are pair counts in radial bins,
*A_k*(*x*) = |{(*i*,*j*) : *r_k* ≤ *r_ij* < *r*_{k+1}}|, on eight bins spanning
3.0 to 7.0 Å. These differ from the radial distribution function *g*(*r*) in that
bin only by a constant at fixed *N* and *V* (Allen and Tildesley 2017), and the
conversion is provided, but the analysis is done on the counts. The reason is
interpretability rather than convenience: response theory applies to an
observable that is a plain function of configuration, and a bin count is exactly
that, so no normalisation stands between the predicted quantity and the measured
one.

The bins taken together form a vector observable. The designed-perturbation arm
additionally targets a *scalar*: the pair count in a single bin, 3.4 to 3.9 Å,
across the first peak. That matters statistically, since it makes the covariance
a *K*-vector rather than a *K* × *J* matrix and so estimable from far fewer
frames. It is also a limitation, stated here as well as in Section 7: the
headline demonstration of Section 4.1 concerns a single-bin observable, and
Section 4.3 shows directly that the result does not transfer to the other bins of
the same curve.

### 3.5. Designed error fields

Error fields are constructed as full potentials with analytic forces and virial,
so that a surrogate is built by ordinary addition. All derivatives agree with
finite differences to ~10⁻¹⁰ on both cubic and sheared triclinic cells.

The constructions used here expand a pair error field in a basis of Gaussian
shell bumps, which makes the covariance with an observable *linear* in the
expansion coefficients. Choosing coefficients in the null space of that
covariance, or parallel to it, is then linear algebra rather than optimisation,
and yields three families: *null*, whose first-order predicted effect on the
target observable is zero by construction; *aligned*, of maximal effect per unit
force error; and *random*, the control. All are then scaled to the same force
RMSE on the reference ensemble, so the number a practitioner would report is held
fixed across the comparison.

Out-of-sample discipline is essential and we state it explicitly. A null field is
orthogonal to the observable *on the frames it was built from* by definition, so
the construction is worthless unless it survives on fresh frames. The reference
trajectory is split: fields are constructed on one half, and every reported
number — force RMSE, predicted shift, covariance, measured shift — is computed on
the other. How many construction frames are needed before the construction
generalises is itself measured and reported (Table II), because with a badly
estimated covariance the null space found is the null space of the noise.

### 3.6. Fitted models

Ten models were fitted to 400 configurations of Lennard-Jones argon, with
training and test data drawn from separate Markov chains: two pair splines
(`spline_k8`, `spline_k24`); two linear atomic-cluster-expansion-style bases
(`linear_l2`, `linear_l4`); Behler–Parrinello networks at two widths and two
initialisation seeds each (`bpnn_16x2`, `bpnn_64x2`); and an *E*(3)-equivariant
message-passing network at two initialisation seeds (`egnn_c8`). These are
deliberately small surrogates of a classical potential, not the published
potentials whose evaluation motivates the work; the architectures are cited as
background, not as systems evaluated here. A
Gaussian-approximation-potential-style kernel model is implemented in the same
codebase but is not included in the reported zoo.

### 3.7. Uncertainty estimation

Every uncertainty in this paper is a **standard error** obtained from
correlated-sample statistics. The symbol ± always denotes a standard error and
never a standard deviation, and 1/√*M* is never used: Monte Carlo samples are
correlated, and treating *M* frames as *M* independent samples understates the
error by roughly √(2τ_int), a factor of several for a typical liquid observable.

Means use Flyvbjerg–Petersen blocking (1989), with the plateau taken over levels
retaining at least eight blocks. Covariances, correlations and fitted slopes use
a moving-block bootstrap (Künsch 1989; Politis and Romano 1994) with block length
set to four integrated autocorrelation times of the *product* series that carries
the error, (*A_m* − Ā)(δ*U_m* − δŪ); using the observable's own correlation time
underestimates the block length whenever δ*U* varies slowly, which is precisely
the systematic-error case this paper is about. A jackknife with a different bias
structure is available as a cross-check; Appendix B gives the estimators and
their validation in full. Interval estimates from
`results/exp05_proxy_correlation` use 2000 bootstrap resamples; intervals
computed for this manuscript by the deposited script
`scripts/compute_band_correlations.py` use 2000 resamples with seed 20240517.

Every correlation is reported as ρ = value [low, high] with its *n*. Significance
is quoted as a σ-distance, |value|/error, to one decimal place; no *p*-values are
used. Uncertainties are quoted to two significant figures and central values to
the same decimal place, with two deliberate exceptions: shifts in pair counts are
given throughout at the three-decimal precision of the deposited records, so that
every one of them can be checked against the JSON without rounding; and exactness
checks (Appendix A) carry their full digits and are labelled as such. Prediction residuals quoted in σ are the difference between the measured
and predicted shift divided by the standard error of the *measurement*.

Two noise conventions exist and we reconcile them here. The observable error
across the designed-surrogate zoo is a norm of a difference curve, and a norm is
inflated by noise: E|*v*_meas|² = |*v*_true|² + E|*v*_noise|². All correlations
reported in this paper use the noise-subtracted series
|*v*_true| = (|*v*_meas|² − E|*v*_noise|²)^{1/2}, floored at zero. The
subtraction changes every reported ρ by less than 0.01, but it is necessary at
the low-force-error end, where signal and noise are comparable and the raw norm
is almost entirely noise. The raw-series values are given in a footnote to
Table III as a robustness check.

### 3.8. Reproducibility and compute

Each experiment writes a manifest recording the git commit, whether the working
tree was dirty, the full configuration, per-stage wall times, library versions
and the seed. Seeds are derived from a single experiment seed through named
`SeedSequence` sub-streams, so inserting a stage does not perturb every result
after it.

The five experiment runs reported here total 13,311 s of wall time, about 3.7 h,
on a four-core x86-64 Linux machine (Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1,
PyTorch 2.13.0+cpu). The manifests record the core count and the platform string
but not the processor model, so we do not state one. The largest single run is
the ten-model fitted zoo at 4,796 s.

---

## 4. Results

### 4.1. Four error fields at matched force RMSE

Four error fields were built against a single target observable, the pair count
in the bin from 3.4 to 3.9 Å, and each was scaled to a prescribed force RMSE. Two
levels were run, nominally 1.0 × 10⁻³ and 4.0 × 10⁻³ eV/Å. Measured out of sample
— on the half of the reference trajectory not used for the construction — the
four fields agree in force RMSE to 0.5 % at each level — a full spread of 0.52 %
at both — running from 0.99535 to 1.00054 × 10⁻³ eV/Å at the lower level and
3.9814 to 4.0022 × 10⁻³ eV/Å at the upper. Table I gives the full comparison and
Fig. 1(a) plots it.

The measured shifts in the target bin at the upper level run from −0.220 ± 0.640
pairs for the null field, which is 0.3σ from zero, to +9.891 ± 0.611 pairs for
the aligned field, which is 16.2σ from zero. The two random controls sit between
them, at −1.482 ± 0.579 (2.6σ) and +5.320 ± 0.589 (9.0σ). At the lower level the
same four fields give −0.547 ± 0.580 (0.9σ), −1.450 ± 0.576 (2.5σ),
+0.835 ± 0.589 (1.4σ) and +1.835 ± 0.579 (3.2σ). We emphasise that the
lower-level measurements are not the upper-level ones divided by four: the
*predictions* scale almost exactly with the force level (2.552 against 10.206
pairs for the aligned field) because they are linear in δ*U*, but the
measurements at the lower level are noisier relative to their own signal and land
at 1.835 rather than at a quarter of 9.891.

Two conclusions follow, and neither is an interpretation. First, the reported
metric cannot distinguish these four models. Their force RMSE agrees to 0.5 % out
of sample; any procedure that selects a model by force error would rank them as
equivalent; one of them leaves the target observable untouched and another moves
it by sixteen standard errors. Second, the quantity that does distinguish them is
the correlation ρ(*A*, δ*U*). Ordering the four fields by ρ — random seed 1 at
+0.110, null at +0.009, random seed 0 at −0.430, aligned at −0.582 — orders the
measured shifts monotonically at both force levels, and the relation is close to
linear [Fig. 1(a)], which is what Δ⟨*A*⟩ = −β σ₀(*A*) σ₀(δ*U*) ρ₀ requires when
the first two factors are held nearly fixed.

### 4.2. The prediction works

The predicted column of Table I is −β Cov₀(*A*, δ*U*) evaluated on the reference
trajectory alone. No molecular dynamics and no Monte Carlo with any surrogate
entered it; the measured column comes from explicit Hamiltonian Monte Carlo
sampling of each perturbed potential, 3000 frames each. Across all eight designed
fields the root-mean-square residual between prediction and measurement is 1.01σ
(Fig. 2, *n* = 8). The prediction agrees with the measurement at exactly the
level the error bars claim, neither better nor worse. This is the practical
payoff: evaluating a model's effect on an observable costs one energy evaluation
per stored frame rather than a full simulation.

The second-order term of the same expansion is right as well. For a null field
the first-order term vanishes by construction, so any residual effect must be
second order, and Eq. (3) predicts it. The second-order estimates are
+0.021 ± 0.023 pairs at the lower force level and +0.332 ± 0.374 at the upper.
Subtracting the first- and second-order terms from the measurements leaves
−0.540 ± 0.580 and −0.439 ± 0.640 pairs, both consistent with zero. The formula
is not merely right at leading order; its own correction term is right too, which
is what licenses the use of that term as the self-diagnostic of Section 2.3(iii).

How much data does the null-space construction need? Table II gives the answer.
Constructed on 250, 500, 1000 and 2000 frames and evaluated on disjoint frames,
the out-of-sample predicted shift is 0.041 ± 0.067, 0.142 ± 0.071,
0.037 ± 0.077 and 0.028 ± 0.078 pairs, against the in-sample values of order
10⁻¹⁵ that the construction guarantees. Set against the aligned field's 10.206
pairs at the same force error, the suppression is roughly 400×, and it holds out
of sample. An earlier attempt using 30 construction frames gave 1.2 to 1.6 pairs:
the null space of a covariance estimated from 30 samples is the null space of the
noise. The construction is only as good as the covariance estimate, and Table II
is where that stops mattering.

### 4.3. Orthogonality is specific to the observable it was built for

The null field is orthogonal to one observable, and to one observable only.
Figure 3 shows the full difference curve for the null, aligned and random fields
at both force levels. At 4 × 10⁻³ eV/Å the null field leaves its target bin alone
at −0.220 ± 0.640 pairs and moves the largest non-target bin by 11.36 ± 0.95
pairs, which is as much as the aligned field moves its own largest non-target
bin, 11.18 ± 1.21 pairs.

This is a limit on the claim as much as it is a result. What Section 4.1
demonstrates is that force RMSE fails to predict a *particular* observable, not
that any model is globally harmless. The null field is not a benign potential; it
is a potential that has been made blind in one direction and is as damaging as
its aligned counterpart in others. Harmlessness is a relation between an error
and a question, not a property of a model. It follows that no single scalar can
summarise model quality — not force RMSE and not any replacement for it, because
a scalar is a projection onto one direction and there are as many directions as
there are questions. What can be computed cheaply is a *vector* of response
scores, one per observable of interest, at one energy evaluation per stored frame
per model.

### 4.4. Across decades and within a band

This subsection reports the refutation of one of this study's own predictions.
The prediction, registered before the experiments were run, was:

> **P1 (weak-proxy).** Across a model zoo, rank correlation between force RMSE
> and downstream observable error is weak and observable-dependent.

The pre-registered falsification criterion attached to it was that force RMSE
should rank-correlate strongly, with Spearman ρ > 0.9, against observable error
across a diverse zoo.

The measurement. Thirty-four designed surrogates were built spanning
5.0 × 10⁻⁴ to 1.80 × 10⁻¹ eV/Å in force RMSE, a factor of 361: eighteen Gaussian
shell perturbations at three centres, three widths and two amplitudes; four
high-frequency oscillatory fields; eight random fields at two force levels; and
the null and aligned fields at two force levels. The observable is the full
eight-bin difference curve and the observable error is its norm. Across the whole
zoo the Spearman rank correlation between force RMSE and observable error is
ρ = 0.83 [0.67, 0.93], *n* = 34 [Table III, Fig. 4(a)]. That is strong. A model a
hundred times worse in force error really is worse, and no theory was needed to
say so.

**P1 as stated is refuted.** A rank correlation of 0.83 with an interval whose
lower end is 0.67 is not a weak proxy, and the prediction we registered is
wrong. The separate and stricter criterion we also registered — that force RMSE
would have to reach ρ > 0.9 against *every* observable error for the whole
argument to collapse — is a different test, and the interval here includes 0.9,
so that test is not decided by this measurement. Both statements are ours and we
report both.

What replaces P1 is sharper. Force RMSE cannot choose among models of
*comparable* force error, which is the only situation a practitioner is ever in.
Restricting to the 15 zoo members whose force RMSE lies in the window 1.5 to
6.0 × 10⁻³ eV/Å — a window chosen by the analyst after seeing the data,
hard-coded in the figure script, and not pre-registered — force RMSE spans a
factor of 2.59 while the observable error still spans a factor of 30. Inside that
band force RMSE gives ρ = 0.34 [−0.26, 0.77], *n* = 15, an interval that includes
zero, while the response prediction of Eq. (4) gives ρ = 0.94 [0.76, 1.00],
*n* = 15, an interval that does not [Fig. 4(b)]. We are careful about what this
licenses: force RMSE is *not resolved from zero* inside the band. It is not
demonstrated to be zero, and 15 members is not many. The contrast that survives
is between one predictor whose interval excludes zero and another whose does not.

The corrected claim is therefore: *force error is a coarse filter, not a
selector*. It will tell you that a badly fitted model is bad. It will not tell
you which of your good models to use, and the designed counterexamples of
Section 4.1 show why: within a band the ordering is set by the projection, which
force error does not measure.

Two secondary findings come from the same run and are visible in Fig. 5(c).
First, **the smoothness ratio proposed in this work is refuted**. The quantity
σ₀(δ*U*)/RMSE_F was proposed by us as an empirical stand-in for the inverse
spectral weighting of Section 2.5; across the zoo it gives ρ = −0.02
[−0.37, 0.35], *n* = 34, and on the raw series ρ = −0.01 [−0.37, 0.37],
*n* = 34. It carries no information here. Proposed and refuted in the same study;
see Section 5.2. Second, the per-atom energy spread outranks every force metric,
ρ = 0.92 [0.82, 0.97], *n* = 34, against 0.83 for force RMSE. The theory does not
predict this. A plausible reading is that the per-atom energy spread tracks
σ₀(δ*U*), the second factor in the Cauchy–Schwarz bound of Eq. (5), and so
captures the magnitude factor that force RMSE measures only through a gradient —
but we label that a conjecture rather than a result, since nothing here tests it.

Top-*k* overlap statistics between proxy rankings and the truth ranking were also
computed. They are withheld: with 34 members they moved between 0.33 and 0.67
depending on the noise-subtraction convention, which means they are too noisy to
carry a claim.

### 4.5. Models that were actually fitted

The designed fields test the mechanism; the fitted models test whether it
survives contact with objects produced by fitting rather than by construction.
Table IV lists the ten models of Section 3.6, their force RMSE, and their
predicted and measured shifts in the target bin.

The response formula holds. The root-mean-square residual between predicted and
measured shift across the ten is 1.06σ, matching the 1.01σ obtained on the
designed fields. This is the external-validity check for Section 4.2 and it
passes. The second-order self-diagnostic is recorded for every model in this arm
and flags 8 of the 10 as linearly trustworthy.

Immediately, and without softening: **the rank correlations from this arm are not
reportable.** Force RMSE spans only 4.0 × 10⁻⁴ to 6.1 × 10⁻³ eV/Å across the ten
models and buys almost no measurable difference in the observable, whose error
spans 0.049 to 4.10 pairs with most values at the noise floor. On this reference
potential, at this data budget, nearly every architecture is good enough that its
observable error is indistinguishable from zero. Every confidence interval from
this arm spans zero: force RMSE ρ = 0.43 [−0.35, 0.96], force MAE ρ = 0.35
[−0.49, 0.92], energy RMSE ρ = 0.05 [−0.77, 0.76], the smoothness ratio
ρ = −0.53 [−0.86, 0.06], the response prediction ρ = 0.18 [−0.70, 0.83], all
*n* = 10. Nothing about ranking can be concluded from this arm, and nothing is.
It establishes exactly one thing, the 1.06σ residual.

One pair in the zoo is worth stating on its own, with its weakness attached.
`egnn_c8_s0` and `egnn_c8_s1` are the same architecture, the same training data
and the same hyperparameters, differing only in initialisation seed. Their force
RMSE differs by 36 %, 0.0045 against 0.0061 eV/Å. Their measured target-bin
errors are −0.15 ± 0.57 pairs (0.3σ, indistinguishable from zero) and
−4.10 ± 0.69 pairs (6.0σ), a factor of 27. Both were predicted from the reference
trajectory alone, at −0.98 and −3.03 pairs. Anyone selecting between these two on
force RMSE would see a 36 % difference and reasonably call them equivalent.
**This is *n* = 2.** It is the most quotable observation in the paper and its
weakest evidence, and the two facts belong in the same sentence. Seed variance is
a known and under-reported source of spread in machine-learning benchmarks
(Bouthillier *et al.* 2021; Pineau *et al.* 2021; Kapoor *et al.* 2024);
repeating this comparison at *n* ≥ 8 seeds is the highest-value follow-up we can
name.

Both EGNN records carry `linear_trustworthy = false`, with second-order ratios of
0.41 and 0.14 against a median of 0.015 across the other eight models. The
self-diagnostic of Section 2.3(iii) fired on exactly the two models where the
linear prediction is least reliable, and the predictions for those two are
correspondingly quoted here as an ordering — the second seed is predicted to be
worse, and is — rather than as precise values.

### 4.6. Damage varies fivefold at fixed force error

The final designed experiment holds force RMSE constant and varies only the width
*w* of the error field in *r*. Six Gaussian shell fields at *r*₀ = 4.2 Å and *w*
from 0.10 to 1.00 Å were amplitude-scaled to 2.0 × 10⁻³ eV/Å; the residual spread
in force RMSE across the sweep is 3.3 × 10⁻¹⁵ eV/Å, that is, exactly fixed.
Figure 1(b) shows the result. The measured norm of the difference curve spans a
factor of 4.66, from 2.13 ± 2.15 pairs at *w* = 0.10 Å to 9.95 ± 2.47 pairs at
*w* = 0.65 Å. The first-order prediction tracks the same rise across the range,
from 3.22 to 9.68 pairs, with much smaller uncertainty; the direct measurements
at the small-*w* end carry error bars comparable to their own values, so the
ordering across the sweep is carried mainly by the prediction, and the honest
statement of the measurement is that the observed range is 4.7× with the lowest
point consistent with zero.

---

## 5. What did not work

### 5.1. The *w*^{3/2} scaling is wrong

Section 2.6 predicted, and then retracted on analytic grounds, that at fixed
force error the observable damage would grow as *w*^{3/2}. The measurement on the
sweep of Section 4.6, on a verified-stationary ensemble, gives a fitted exponent
of **0.56** from the direct sampling and **0.45** from the first-order prediction
[Fig. 6(b)], against the predicted 1.5. The estimate is wrong by a factor of
three in the exponent.

The retraction was made analytically and before the measurement, for the two
reasons given in Section 2.6, so this is not a prediction quietly withdrawn once
it failed — but it did fail, and a heuristic that survives its own author's
scepticism only to be refuted by measurement is worth reporting as a refutation.
What survives is the weaker and still sufficient statement that force error and
observable error depend differently on the shape of the error field, so their
ratio is not a constant; Section 4.6 measures that ratio varying by 4.7 at
exactly fixed force RMSE, and that is the statement the argument of this paper
actually needs.

### 5.2. The proposed smoothness-ratio metric is refuted

If force RMSE weights the error spectrum by *k*² and observables weight it by
roughly *k*⁰, then a ratio of an error-field magnitude to a force error should
carry information about where in the spectrum a model's error sits. We proposed
σ₀(δ*U*)/RMSE_F on exactly that reasoning. Measured against observable error
across the 34-member zoo it gives ρ = −0.02 [−0.37, 0.35], *n* = 34 on the
noise-subtracted series and ρ = −0.01 [−0.37, 0.37], *n* = 34 on the raw series.
On either convention it carries no information at all here. Proposed and refuted
in the same study.

For completeness, the same quantity restricted to the post hoc band of
Section 4.4 gives ρ = 0.84 [0.49, 0.98], *n* = 15. We do not present that as
support for the metric: the band is analyst-chosen after the fact, the members
inside it are dominated by one family of shell perturbations, and the zoo-level
correlation — the pre-specified comparison — is consistent with zero. The metric
is refuted at the level at which it was proposed.

### 5.3. An open 35 % discrepancy, with three explanations excluded

In the amplitude sweep of Fig. 6(a), at the smallest perturbation
(βσ₀(δ*U*) = 0.427), three estimates of the same shift in the peak bin
disagreed: first order −3.835 ± 0.170 pairs, exact reweighting −3.884 pairs at an
effective-sample-size fraction (Kish 1965) of 0.83, and direct sampling
−1.948 ± 0.821 pairs (Table VI). The two estimators sharing the reference samples
agree with each other to 1.3 %; the one requiring an independent chain disagrees
with both by a factor of two.

The obvious diagnosis is that the direct estimate's error bar is too small.
Blocking measures the error within a chain, and a difference between two chains
also carries the offset between their slow modes, which no within-chain estimator
can see. **We tested that diagnosis and it was wrong.** Sampling both potentials
from six independent seeds each gives a within-chain blocking error of 0.589
pairs against a between-chain scatter of 0.312 pairs, a ratio of 0.53: blocking
is *conservative* here by about a factor of two, not optimistic. The six
independent seed pairs give measured shifts of
[2.553, 2.773, 2.349, 2.611, 1.788, 2.188] pairs, mean +2.377 ± 0.144 (*n* = 6).
This check was run on the same perturbation but on a different bin — the one
centred at 5.25 Å rather than the peak bin at 4.25 Å — so the sign and size
differ from the row above; what transfers is the size of the between-chain offset
and the reliability of the direct estimate.

So the direct measurement is reliable and the discrepancy is real. On a fresh
3000-frame reference chain the two same-sample estimators give +1.725 ± 0.101
pairs (first order) and +1.757 pairs (exact reweighting), against the direct
+2.377 ± 0.144: a difference of 0.652 pairs, 35 % of the reweighted value and
38 % of the first-order value, and 3.7σ on the combined uncertainty.

Three explanations are excluded by measurement. *Not the error bars*: the
between-chain scatter is half the quoted within-chain error, not larger. *Not
second-order truncation*: the first-order estimate and the all-orders reweighting
identity agree with each other to 2 %, so the truncation cannot be the
difference. *Not incomplete relaxation of the perturbed chain*: that would bias
the measured shift *toward* the reference and make it smaller in magnitude, and
it is larger.

What remains is a candidate we did not test. The Kish effective sample size used
to certify the reweighting estimate assumes independent samples, and the true
independent count in a correlated chain is smaller by the integrated
autocorrelation time; recomputing the effective size as *M*/(2τ_int), with the
τ_int the codebase already estimates, is the obvious next test and was not run.
We record the item as unresolved rather than attributing it to the nearest
plausible cause. It is not a general failure of either estimator: the
measurements of Section 4.1 used a different observable and twice as many frames
and are calibrated at 1.01σ against the same prediction.

### 5.4. The first-order prediction breaks down at large δ*U*, as it must

Linear response is a first-order truncation and must fail somewhere. A separate
four-model arm, fitted at a training budget of 20 configurations and run in the
reduced quick-run mode, puts the failure on the record (Table V). Force RMSE
there reaches 4.34 × 10⁻² eV/Å and observable errors reach 126 pairs; the
aggregate prediction residual is **30.4σ**, *n* = 4, against the 1.01σ and 1.06σ
of the perturbative arms. The largest single residual is 60.5σ, for the
*E*(3)-equivariant model at 4.34 × 10⁻² eV/Å, whose predicted shift of
−14.2 ± 7.3 pairs faces a measured −126.16 ± 1.85. The most instructive row is
the mildest: `spline_k12`, at force RMSE 1.55 × 10⁻³ eV/Å, has a predicted shift
of 0.026 ± 0.251 pairs, consistent with zero, against a measured −5.173 ± 1.863
pairs. A small force error and a near-zero predicted response are together not a
guarantee.

This is the demonstrated boundary of validity of Eq. (4), reported as a result
and not as a hidden failure. It carries a gap in our evidence that we state in
those words: **the records from this arm do not contain the `second_order_ratio`
or `linear_trustworthy` fields, so the self-diagnostic of Section 2.3(iii) cannot
be checked against this arm.** The claim that the second-order term gives advance
warning of exactly this breakdown is supported by the amplitude sweep of
Fig. 6(a), where the diagnostic flags the two smallest amplitudes as trustworthy
and the five larger ones as not, and by the fitted zoo of Section 4.5, where it
flags the two EGNN seeds; it is not supported by the arm where the breakdown is
largest, because the field was not recorded there.

The amplitude sweep shows the two estimators failing in different directions. As
the shell amplitude rises from 5 × 10⁻⁴ to 3.2 × 10⁻², the first-order estimate
grows linearly to −245.5 ± 10.9 pairs while the direct measurement saturates at
−144.1 ± 0.6, and the reweighting estimate collapses to a constant −43.9 as its
effective-sample-size fraction falls from 0.83 to 5 × 10⁻⁴ — the standard failure
of an exponential average dominated by one frame. The largest βσ₀(δ*U*) at which
all three agree is 0.854.

### 5.5. The dilute-gas closed form disagrees

The sharpest available end-to-end test of the estimator is the dilute limit of a
pair fluid, where *g*(*r*) = e^{−β*u*(*r*)} + *O*(ρ) exactly, so that a
perturbation δ*u* of the pair potential shifts the bin counts by a quantity
available in closed form. Run at ρ* = 0.079 and *T*\* = 2.510 on 64 atoms with
6000 reference frames, the sampled covariance estimator and the exact reweighting
identity agree with each other to within 0.003 pairs in every bin, at most 1.5 %
of the shift in any bin where the shift is resolved. Both disagree with our
hand-derived closed form, by 4.7σ root-mean-square and 6.5σ at worst over seven
bins.

One correction the hand calculation needed, and the estimator did not, is worth
recording. Pair number is conserved, so a perturbation that depletes one shell
must enrich others, and the pair-separation distribution is normalised. Dropping
that normalisation term gives a closed form wrong by a factor of ten in the outer
bins and wrong in sign in the tail. A covariance is mean-subtracted, and the
subtraction *is* the normalisation, so the estimator never had the problem — a
small illustration of why an estimator that computes the right object is safer
than a derivation that reconstructs it.

The residual disagreement is most likely the *O*(ρ) correction to
*g* = e^{−β*u*}, which at ρ* = 0.079 is not negligible. **This is not
confirmed.** The confirming test — repeating at a quarter of the density and
checking that the deviation falls — was not run.

---

## 6. Relation to prior work

**The empirical dissociation.** That force error and simulation quality come
apart is an established empirical finding. Fu *et al.* (2023) benchmarked
machine-learned force fields on simulation tasks rather than on held-out forces
and found the ranking by force error to be a poor guide to the ranking by
simulation outcome; Stocker *et al.* (2022) found graph-network potentials with
excellent test errors to be unreliable in long, hot molecular dynamics. This
paper does not claim to have reproduced either benchmark: no published potential
and no dynamical metric was evaluated here. What it adds is threefold.
*(i) Confound removal.* In those comparisons models differ simultaneously in
force error, architecture, training data and — decisively — the *magnitude* of
their error, so shape and size cannot be separated. Here the magnitude is held
fixed by construction, to 0.5 % in force RMSE, and only the shape varies.
*(ii) A mechanism.* Those studies offer an observation; Eq. (4) supplies a
mechanism and we validate it at 1.01σ (*n* = 8) and 1.06σ (*n* = 10).
*(iii) A different disease.* Their failure mode is dynamical instability: large
δ*U*, out of distribution, non-perturbative, and visible — simulations that melt,
blow up or drift. The failure mode here is in-distribution, perturbative and
silent. Nothing blows up, the trajectory is fine, the diagnostics are healthy, and
the answer is wrong by 16σ. These are different diseases with different remedies,
and conflating them would understate both.

**The identity is not new.** We say this plainly because the temptation to
overclaim is real. Equation (4) is standard thermodynamic perturbation theory:
Zwanzig's free-energy-perturbation identity (1954) expanded in cumulants, with
Bennett's acceptance ratio (1976) and its multistate successor (Shirts and
Chodera 2008) as the practical machinery around it. The same covariance is used
as a parameter gradient in differentiable trajectory reweighting, where Thaler
and Zavadlav (2021) differentiate reweighted ensemble averages with respect to
potential parameters to train against experimental observables; and it is used by
Imbalzano *et al.* (2021) to propagate committee uncertainty in the potential
through to uncertainties in thermodynamic averages. Frederiksen *et al.* (2004)
are the deep precedent for the conceptual claim, showing by Bayesian ensemble
sampling of potential parameters that the uncertainty a potential induces is a
property of the quantity being predicted and not a single number attached to the
model. What is new here is the four items listed in Section 1: the reading of the
identity as an audit of the reported evaluation metric, the analytic-oracle
construction, the constructive falsification at matched force RMSE, and the
across-decades/within-band decomposition. Nothing else.

**Norm-based corrections and the dynamical case.** Wu *et al.* (2024) derive a
correction to lattice thermal conductivity computed with machine-learned
potentials in which the force error enters through a norm, and correct a
systematic underestimation with it. That is the expected structure for a
transport coefficient: δ*F* enters the equations of motion directly, so the
magnitude of the force error is the right variable, and no projection onto an
observable is involved. The contrast with the projection mechanism for static
observables is exactly the asymmetry conjectured in Section 2.7, and that work is
independent evidence for it — not evidence produced by this work, which tested no
dynamical observable. Related work on transport with foundation models (Póta
*et al.* 2024) points the same way.

**Validation practice.** Morrow, Gardner and Deringer (2023) argue that
machine-learned potentials should be validated on the physical properties they
will be used to compute rather than on held-out errors alone. This paper supports
that recommendation quantitatively and explains why it is necessary rather than
merely prudent: the held-out error is a norm and the property is a projection,
and Section 4.4 measures how much of the ordering the norm fails to capture where
it matters. Reports that practitioners already distrust force RMSE (Kovács
*et al.* 2021), and observations of systematic softening in universal potentials
with good aggregate errors (Deng *et al.* 2025), are consistent with the same
reading.

**The practical consequence, and its untested part.** The estimator recommended
by this work is a *vector* of response scores — one −β Cov₀(*A_k*, δ*U*) per
observable of interest — computed at one energy evaluation per stored frame per
model on an existing reference trajectory. The honest caveat is that a
practitioner has no *U*₀, so δ*U* is not available. The usable substitute
replaces the unknown truth with a committee mean over independently trained
models (Imbalzano *et al.* 2021), giving a per-model deviation from consensus
whose spread across the committee is an uncertainty on the observable itself. Its
failure mode is predictable and one-directional: homogeneous committees share
systematic error, and shared error cancels exactly in a committee mean, so such a
committee should under-report. **This was not tested.** The experiment designed to
test it, with deliberately homogeneous and deliberately heterogeneous committee
arms, was not run, and the committee predictor is a hypothesis in this paper and
not a finding.

---

## 7. Scope and limitations

We list the limitations without euphemism, separating those foreclosed by design
from those foreclosed by resources.

*By design.* One reference potential, Lennard-Jones argon, and one state point,
*N* = 108, ρ* = 0.790, *T*\* = 1.004. The choice of an analytic reference is the
methodological crux of the paper — it is what makes δ*U* exactly known and error
fields designable — and it is simultaneously the reason no result here is a claim
about a chemically realistic system. No density-functional-theory reference was
used and none was needed. No published machine-learned potential was evaluated;
the ten fitted models are deliberately small surrogates of a classical potential.
Stillinger–Weber silicon and an embedded-atom-method copper with a second-moment
tight-binding embedding function (Finnis and Sinclair 1984; Cleri and Rosato
1993) are implemented and validated in the same codebase (Appendix A), and *no
observable-error result was produced on either*. One observable class throughout:
pair counts in radial bins, and for the headline demonstration a single bin.
Error fields are pair-radial only — no angular and no many-body error structure,
which is precisely where real architectures differ from one another. No dynamical
observable of any kind: no diffusivity, no vibrational spectrum, no melting
point, no elastic constant of any surrogate model was measured, and no claim
about any of them is made.

*By resources.* No finite-size study: every number is *N* = 108, and we do not
assert that the covariance mechanism is size-independent. The check is
outstanding, and it is not obviously trivial, since pair-count fluctuations in a
closed *NVT* cell carry a known ensemble dependence (Lebowitz, Percus and Verlet
1967) that the covariance in Eq. (4) inherits. The seed-pair observation of
Section 4.5 is *n* = 2 and does not generalise as stated; it is an illustration,
not a statistic. The band of Section 4.4 is post hoc: the window
1.5–6.0 × 10⁻³ eV/Å was chosen by the analyst after seeing the data and is
hard-coded in the figure script, and the 15 members inside it were not
pre-registered. The dynamical conjecture of Section 2.7 is untested here. The
committee predictor of Section 6 is untested here. The remaining candidate
explanation for the discrepancy of Section 5.3 is untested. The quarter-density
repeat that would confirm the reading of Section 5.5 was not run. The large-δ*U*
arm of Section 5.4 lacks the second-order diagnostic field, so the warning-light
claim cannot be checked where the breakdown is largest.

Finally, a limitation of the framing rather than of the data. Nothing here shows
that force error is useless; Section 4.4 shows the opposite across three decades.
The claim is about resolution, not about information content.

---

## 8. Conclusion

Force error is a coarse filter, not a selector. That is the corrected form of a
prediction this study made and refuted: we expected force RMSE to rank models
weakly against downstream physics, and instead it ranks them well across three
decades, ρ = 0.83 [0.67, 0.93], *n* = 34, and is not resolved from zero inside a
factor-2.6 band where the physics still varies by 30×, ρ = 0.34 [−0.26, 0.77],
*n* = 15. The response prediction is resolved there, ρ = 0.94 [0.76, 1.00],
*n* = 15. Two further predictions of ours failed outright: a smoothness-ratio
metric we proposed carries no information, ρ = −0.02 [−0.37, 0.35], *n* = 34, and
a *w*^{3/2} scaling we derived has a measured exponent of 0.56.

The mechanism behind the corrected claim is that the error in a static observable
is a covariance between the observable and the potential's error field, while
force RMSE is a norm of that field's gradient. A covariance is a projection; a
norm is not. At force RMSE matched to 0.5 % we moved a target pair count from
−0.220 ± 0.640 to +9.891 ± 0.611 pairs by changing only the correlation between
the error field and the observable, and the null-space construction that achieves
the first of those survives out of sample with a 400× suppression relative to its
aligned counterpart. The same field moves a non-target bin by 11.36 ± 0.95 pairs,
so harmlessness is a relation between an error and a question, not a property of
a model.

The estimator that follows is cheap and was validated here: predicted from
reference-ensemble samples alone, the shift in an observable agrees with direct
measurement at 1.01σ over eight designed error fields and 1.06σ over ten fitted
models, at one energy evaluation per stored frame per model, and its own
second-order term gives advance warning where it fails. What we recommend
reporting is therefore not a better scalar but a vector: one response score per
observable anyone intends to compute.

The scope is one Lennard-Jones state point, *N* = 108, pair-radial error fields
and a pair-count observable, with no dynamical observable and no chemically
realistic system. Whether the same decomposition holds for angular and many-body
error structures, and whether the conjectured reversal for transport observables
is real, are open.

---

## 9. Supplementary material

The supplementary deposit contains the complete 34-member designed-surrogate
record listing behind Section 4.4, the seven-row amplitude sweep and six-row
width sweep behind Figs. 1(b) and 6, the four-model large-δ*U* arm of
Section 5.4, the per-experiment manifests, and the analysis scripts that produce
every figure and table in this article.

---

## Acknowledgments

[FUNDING STATEMENT TO BE COMPLETED]

## Author declarations

**Conflict of Interest.** The authors have no conflicts to disclose.

**Author Contributions.** [AUTHOR CONTRIBUTIONS TO BE COMPLETED]

**Ethics Approval.** Ethics approval was not required for this work, which
involves no human or animal subjects.

## Data availability

The data that support the findings of this study are openly available in Zenodo
at http://doi.org/[DOI], reference number [DEPOSIT ID] (Ref. 48). The deposit
contains the complete `atomlab` source code, the experiment configurations, and
the raw outputs of every experiment reported here — `summary.json`,
`records.json`, and the per-experiment manifests, each of which records the git
commit that produced it, whether the working tree was dirty at the time, the full
configuration, per-stage wall times, library versions, and the random seed. No
external dataset was used, downloaded, or required: every configuration analysed
here was generated by the deposited code from the stated seeds. All analysis
scripts that produce the figures and tables in this article are included in the
same deposit.

---

## Appendix A: Simulation and estimator validation

Table VII gives the energy and force discontinuities of the four cutoff modes,
measured at *r_c* ∓ 10⁻⁷ Å. Only `shifted_force` and `switched` remove the force
discontinuity; `shifted`, the most common choice, leaves a force jump identical
to that of the plain truncation. Since every result in this paper is a small
systematic difference between two ensembles, an impulse delivered whenever a pair
crosses the cutoff would be indistinguishable from the effect under study, and
`shifted_force` is used throughout.

Table VIII lists the checks run against targets from outside the codebase. Two
entries deserve comment. The Stillinger–Weber silicon cohesive energy is an
*exactness check*: at the diamond lattice constant every bond sits at the
two-body minimum, every angle is exactly tetrahedral so the three-body term
vanishes, and the second-neighbour shell lies outside the cutoff, so the answer
is −2ε identically and the code returns −4.3366000 eV/atom against
−4.33660 eV/atom. And the embedded-atom copper reproduces its three fitted
targets — lattice constant, cohesive energy, bulk modulus — exactly, while giving
a maximum phonon frequency of 5.14 THz against roughly 7.5 THz for real copper,
about 30 % soft at the zone boundary. That is not a defect, since the reference
potentials define this study's ground truth and are not claims about experiment.
It is the thesis of this paper arriving from another direction: fitting a
quantity exactly, even a second-derivative quantity such as the bulk modulus,
does not constrain the quantities that were not fitted.

## Appendix B: Statistical estimators and their validation

Means use Flyvbjerg–Petersen blocking: the sample series is repeatedly halved by
pairwise averaging, the apparent standard error is computed at each level, and
the plateau is read off over levels retaining at least eight blocks. Statistics
that are not means — covariances, rank correlations, fitted slopes — use a
moving-block bootstrap with block length four integrated autocorrelation times of
the product series (*A_m* − Ā)(δ*U_m* − δŪ), resampled with replacement;
percentile intervals are quoted. A delete-one jackknife with a different bias
structure is available as a cross-check.

The estimators were validated on an AR(1) process of known correlation time: the
blocking estimate recovers the naive standard error inflated by √(2τ_int) to
within 30 %, and the measured integrated autocorrelation time matches the
analytic 1/2 + φ/(1 − φ) to within 15 %. The response estimators were validated
against closed forms on a harmonic reference perturbed by a linear field, where
the cumulant expansion terminates and each order can be isolated: ⟨*x*⟩ shifts by
exactly −ε/*k* (pure first order), ⟨*x*²⟩ by exactly ε²/*k*² (pure second order),
and the Zwanzig free-energy shift is −ε²/2*k*; all three are recovered within
statistical error. The reweighting estimator additionally reports the Kish
effective sample size and the largest single-frame weight fraction, since an
exponential average can be precise-looking and meaningless.

---

## Figures

**Figure 1** — `../figures/headline_mechanism.pdf`.
The mechanism, at force RMSE held fixed in both panels. (a) Measured shift in the
target pair count (the bin from 3.4 to 3.9 Å), in pairs, against the Pearson
correlation ρ(*A*, δ*U*) between that observable and the error field, for the
four designed fields of Table I at out-of-sample force RMSE matched to 0.5 %. The
relation is monotone and close to linear, as
Δ⟨*A*⟩ = −β σ₀(*A*) σ₀(δ*U*) ρ₀(*A*, δ*U*) requires. (b) Norm of the measured
shift in the eight-bin difference curve, in pairs, against the width *w* of a
Gaussian shell error field at *r*₀ = 4.2 Å, with the amplitude rescaled at each
width so that the force RMSE is 2.0 × 10⁻³ eV/Å throughout; the residual spread
in force RMSE across the sweep is 3.3 × 10⁻¹⁵ eV/Å. The measured range is a
factor of 4.66. Error bars are standard errors from Flyvbjerg–Petersen blocking;
no 1/√*M* estimate is used anywhere. The same width sweep appears in Fig. 6(b) on
logarithmic axes, where the exponent is fitted.

**Figure 2** — `../figures/headline_prediction.pdf`.
Measured against predicted shift in the target pair count, in pairs, for all
eight designed error fields (*n* = 8; four fields at each of two force levels).
The prediction is −β Cov₀(*A*, δ*U*) evaluated on reference-ensemble samples
alone — no molecular dynamics or Monte Carlo with any surrogate entered it — and
the measurement comes from explicit Hamiltonian Monte Carlo sampling of each
perturbed potential. The line is the identity. The root-mean-square residual,
measured minus predicted in units of the standard error of the measurement, is
1.01σ. Error bars are standard errors.

**Figure 3** — `../figures/exp07_designed_counterexamples_counterexamples.pdf`.
Shift in pair count against pair separation *r* for the null, aligned and random
error fields, one panel per force level (1 × 10⁻³ and 4 × 10⁻³ eV/Å). The null
field's target bin — the one it was constructed to be orthogonal to, from 3.4 to
3.9 Å — is left alone (−0.220 ± 0.640 pairs at the upper level), but its largest
non-target excursion, 11.36 ± 0.95 pairs, is comparable to the aligned field's
11.18 ± 1.21 pairs. Orthogonality is specific to the observable it was built for.
Error bars are standard errors from blocking.

**Figure 4** — `../figures/headline_regimes.pdf`.
Two regimes for force RMSE as a ranking metric, over the 34 designed surrogates.
(a) Observable error — the noise-subtracted norm of the eight-bin difference
curve, in pairs — against force RMSE on log–log axes, with the band
1.5–6.0 × 10⁻³ eV/Å shaded. Across the zoo the Spearman rank correlation is
ρ = 0.83 [0.67, 0.93], *n* = 34. (b) Observable error for the 15 band members,
ordered by ascending force RMSE: force RMSE spans a factor of 2.59 while the
observable error spans a factor of 30, giving ρ = 0.34 [−0.26, 0.77], *n* = 15,
an interval that includes zero, against ρ = 0.94 [0.76, 1.00], *n* = 15 for the
response prediction. **The band is analyst-chosen and post hoc**: it was fixed
after inspecting the data and was not pre-registered. Intervals are 95 %
percentile bootstrap over zoo members. The panel titles in the published figure
quote the point estimates without intervals; the intervals are as given here.

**Figure 5** — `../figures/exp05_proxy_correlation_proxy_correlation.pdf`.
Proxy metrics against measured observable error across the 34 designed
surrogates. (a) Observable error against force RMSE, with standard-error bars;
the rank correlation for this panel is given in Fig. 4(a) and is not restated
here. (b) Measured against predicted observable error, the prediction being the
norm of −β Cov₀(*A_k*, δ*U*) from reference-ensemble samples alone. (c) Spearman
ρ with 95 % bootstrap intervals for every proxy metric. Panel (c) is where two
results of Section 4.4 are visible: the smoothness ratio σ₀(δ*U*)/RMSE_F,
proposed in this work, gives ρ = −0.02 [−0.37, 0.35], *n* = 34 and is refuted;
the per-atom energy spread gives ρ = 0.92 [0.82, 0.97], *n* = 34 and outranks
every force metric, which the theory does not predict.

**Figure 6** — `../figures/exp06_response_validation_response_validation.pdf`.
Validation and breakdown of the response estimators. (a) First-order, exactly
reweighted and directly sampled shifts in the peak bin, in pairs, against the
amplitude of a Gaussian shell perturbation at *r*₀ = 4.2 Å, *w* = 0.35 Å. All
three agree at amplitude 1 × 10⁻³ (βσ₀(δ*U*) = 0.854, the largest value at which
they do); above it the first-order estimate continues to grow linearly while the
direct measurement saturates and the reweighting estimate collapses as its
effective-sample-size fraction falls from 0.83 to 5 × 10⁻⁴. The disagreement at
the *smallest* amplitude is the unresolved discrepancy of Section 5.3. (b) Norm
of the shift at fixed force RMSE against the width *w* of the error field, on
log–log axes, from which the exponents are fitted: 0.56 from the direct sampling
and 0.45 from the first-order prediction, **against the 1.5 predicted in
Section 2.6, which is thereby refuted**. This is the same width sweep as
Fig. 1(b), where its dynamic range is reported. Error bars are standard errors;
where a plotted quantity is a fitted exponent rather than a sampled mean, no
error bar is drawn.

---

## Tables

### Table I. Four designed error fields at two matched force-RMSE levels.

*Source: `results/exp07_designed_counterexamples/records.json`.*

ρ(*A*, δ*U*) is the Pearson correlation between the target observable (the pair
count between 3.4 and 3.9 Å) and the error field under the reference ensemble;
the predicted shift is −β Cov₀(*A*, δ*U*) from reference-ensemble samples alone;
the measured shift is from explicit Hamiltonian Monte Carlo sampling of the
perturbed potential. Force RMSE is quoted out of sample, on the half of the
reference trajectory not used to construct the fields; within each level the four
values agree to **0.5 %**. **Every ± in this table, and everywhere in this
article, is a standard error** — from Flyvbjerg–Petersen blocking for means and a
moving-block bootstrap for covariances — never a standard deviation, and 1/√*M*
is never used. The σ-distance is |measured| / |its standard error|.

| Level | Field | RMSE_F out of sample (eV/Å) | ρ(*A*, δ*U*) | predicted (pairs) | measured (pairs) | σ |
|---|---|---|---|---|---|---|
| 1.0 × 10⁻³ | null | 1.00054 × 10⁻³ | +0.009 | −0.028 ± 0.083 | −0.547 ± 0.580 | 0.9 |
| | random (seed 1) | 0.99896 × 10⁻³ | +0.110 | −0.327 ± 0.083 | −1.450 ± 0.576 | 2.5 |
| | random (seed 0) | 0.99775 × 10⁻³ | −0.430 | +1.445 ± 0.104 | +0.835 ± 0.589 | 1.4 |
| | aligned | 0.99535 × 10⁻³ | −0.582 | +2.552 ± 0.140 | +1.835 ± 0.579 | 3.2 |
| 4.0 × 10⁻³ | null | 4.00217 × 10⁻³ | +0.009 | −0.113 ± 0.333 | −0.220 ± 0.640 | 0.3 |
| | random (seed 1) | 3.99586 × 10⁻³ | +0.110 | −1.306 ± 0.331 | −1.482 ± 0.579 | 2.6 |
| | random (seed 0) | 3.99100 × 10⁻³ | −0.430 | +5.778 ± 0.418 | +5.320 ± 0.589 | 9.0 |
| | aligned | 3.98141 × 10⁻³ | −0.582 | +10.206 ± 0.558 | +9.891 ± 0.611 | 16.2 |

### Table II. How many frames the null-space construction needs.

*Source: `results/exp07_designed_counterexamples/summary.json`
(`null_space_generalisation`) and `generalisation.json`.*

The field is built on *n*_construct frames of the reference trajectory and
evaluated on disjoint frames. In sample the predicted shift is zero to machine
precision by construction; out of sample it is not, and the table shows that it
remains small from 250 frames upward. For scale, the aligned field at the same
force error (4 × 10⁻³ eV/Å) has a predicted shift of 10.206 pairs, so the
suppression out of sample is roughly 400×. An earlier construction using 30
frames gave 1.2–1.6 pairs. ± is a standard error.

| *n*_construct | in sample (pairs) | out of sample (pairs) |
|---|---|---|
| 250 | 9.1 × 10⁻¹⁵ | 0.041 ± 0.067 |
| 500 | 1.2 × 10⁻¹⁵ | 0.142 ± 0.071 |
| 1000 | 5.0 × 10⁻¹⁵ | 0.037 ± 0.077 |
| 2000 | 8.4 × 10⁻¹⁵ | 0.028 ± 0.078 |

### Table III. Two regimes, across the zoo and within the band.

*Source: `results/exp05_proxy_correlation/{summary,records}.json` and
`scripts/compute_band_correlations.py` (seed 20240517, 2000 resamples).*

All correlations are Spearman rank correlations against the measured observable
error, computed on the noise-subtracted series (Section 3.7), with 95 %
percentile bootstrap intervals over zoo members from 2000 resamples, seed
20240517. **The band — force RMSE in 1.5–6.0 × 10⁻³ eV/Å — is analyst-chosen and
post hoc**, fixed after inspecting the data and not pre-registered. The response
prediction is the norm of −β Cov₀(*A_k*, δ*U*).

| | across the zoo (*n* = 34) | within the band (*n* = 15) |
|---|---|---|
| force-RMSE spread | 361× | 2.59× |
| observable-error spread | — | 30× |
| ρ(force RMSE) | 0.83 [0.67, 0.93] | 0.34 [−0.26, 0.77] |
| ρ(response prediction) | 0.98 [0.95, 0.99] | 0.94 [0.76, 1.00] |
| ρ(smoothness ratio) | −0.02 [−0.37, 0.35] | 0.84 [0.49, 0.98] |
| ρ(per-atom energy spread) | 0.92 [0.82, 0.97] | 0.91 [0.71, 0.99] |

*Footnote.* On the raw (non-noise-subtracted) series the zoo values are force
RMSE ρ = 0.83 [0.68, 0.92], force MAE ρ = 0.85 [0.70, 0.94], per-atom energy
spread ρ = 0.92 [0.81, 0.97], smoothness ratio ρ = −0.01 [−0.37, 0.37], response
prediction ρ = 0.98 [0.95, 0.99], all *n* = 34 with 2000 resamples; every value
moves by less than 0.01.

### Table IV. The ten fitted models.

*Source: `results/exp03_model_zoo/records.json`, `summary.json`.*

All trained on 400 configurations of Lennard-Jones argon with training and test
data drawn from separate Markov chains. The predicted shift is
−β Cov₀(*A*, δ*U*) from reference-ensemble samples alone; the measured shift is
from direct sampling of each fitted model. The root-mean-square prediction
residual over the ten is 1.06σ. **The rank correlations from this arm are not
reportable**: every interval spans zero (force RMSE ρ = 0.43 [−0.35, 0.96],
*n* = 10), because at this data budget on this reference nearly every
architecture is good enough that its observable error is indistinguishable from
zero. Nothing about ranking is concluded from this arm. The two `egnn_c8` rows
differ only in initialisation seed and are an illustration at ***n* = 2**, not a
statistic. The second-order ratio is the self-diagnostic of Section 2.3(iii); it
flags 8 of 10 models as linearly trustworthy, and the two it does not flag are
the EGNN pair. ± is a standard error.

| Model | Architecture | RMSE_F (eV/Å) | predicted (pairs) | measured (pairs) | σ | 2nd-order ratio | trustworthy |
|---|---|---|---|---|---|---|---|
| `spline_k8` | pair spline, 8 knots | 0.00472 | +1.667 ± 0.107 | +0.982 ± 0.564 | 1.7 | 0.017 | yes |
| `spline_k24` | pair spline, 24 knots | 0.00088 | +0.245 ± 0.067 | +0.049 ± 0.603 | 0.1 | 0.090 | yes |
| `linear_l2` | linear ACE-style, *l* ≤ 2 | 0.00070 | −0.044 ± 0.035 | −0.920 ± 0.549 | 1.7 | 0.131 | yes |
| `linear_l4` | linear ACE-style, *l* ≤ 4 | 0.00065 | −0.064 ± 0.022 | −0.478 ± 0.611 | 0.8 | 0.005 | yes |
| `bpnn_16x2_s0` | Behler–Parrinello, 16×2 | 0.00042 | +0.250 ± 0.043 | −0.065 ± 0.631 | 0.1 | 0.009 | yes |
| `bpnn_16x2_s1` | Behler–Parrinello, 16×2 | 0.00041 | +0.257 ± 0.042 | −0.467 ± 0.580 | 0.8 | 0.010 | yes |
| `bpnn_64x2_s0` | Behler–Parrinello, 64×2 | 0.00044 | +0.236 ± 0.041 | +0.589 ± 0.620 | 1.0 | 0.014 | yes |
| `bpnn_64x2_s1` | Behler–Parrinello, 64×2 | 0.00040 | +0.235 ± 0.041 | +0.365 ± 0.564 | 0.6 | 0.018 | yes |
| `egnn_c8_s0` | *E*(3)-equivariant, seed 0 | 0.00450 | −0.979 ± 0.206 | −0.152 ± 0.567 | 0.3 | 0.412 | no |
| `egnn_c8_s1` | *E*(3)-equivariant, seed 1 | 0.00609 | −3.028 ± 0.290 | −4.101 ± 0.686 | 6.0 | 0.142 | no |

### Table V. The large-δ*U* arm.

*Source: `results/exp03_model_zoo_n400/records.json`, `summary.json`.*

Four models fitted at a training budget of 20 configurations and run in the
reduced quick-run mode, which puts them well outside the perturbative regime. The
aggregate root-mean-square prediction residual is 30.4σ, *n* = 4, against 1.01σ
and 1.06σ in the perturbative arms. **These records do not carry the
`second_order_ratio` or `linear_trustworthy` fields, so the second-order
self-diagnostic cannot be checked against this arm**; that is a gap in the
evidence for the warning-light claim of Section 2.3(iii). ± is a standard error.

| Model | RMSE_F (eV/Å) | predicted (pairs) | measured (pairs) | σ |
|---|---|---|---|---|
| `spline_k12` | 1.55 × 10⁻³ | +0.026 ± 0.251 | −5.173 ± 1.863 | 2.8 |
| `linear_l3` | 5.08 × 10⁻⁴ | +0.180 ± 0.092 | −6.120 ± 2.438 | 2.6 |
| `bpnn_16x2_s0` | 1.53 × 10⁻² | −4.938 ± 2.297 | −14.507 ± 1.820 | 5.3 |
| `egnn_c8_s0` | 4.34 × 10⁻² | −14.180 ± 7.337 | −126.160 ± 1.849 | 60.5 |
| **aggregate rms residual** | | | | **30.4** |

### Table VI. The open discrepancy of Section 5.3.

*Source: `results/exp06_response_validation/amplitude_records.json` (upper block)
and the recorded output of `scripts/check_between_chain_scatter.py`, deposited as
`results/validation/between_chain_scatter.txt` (lower block).*

Upper block: three estimates of the same shift at the smallest perturbation
amplitude of the sweep, in the peak bin at 4.25 Å. The two estimators sharing the
reference samples agree to 1.3 %; the direct estimate, which needs an independent
chain, differs from both by a factor of two. Lower block: the between-chain
diagnostic, run on the same perturbation but on the bin at 5.25 Å, which
overturns the obvious explanation — blocking is conservative here, not
optimistic. **Three explanations are excluded by measurement (error bars,
second-order truncation, incomplete relaxation) and the item is unresolved**; the
remaining candidate, the correlated-sample behaviour of the reweighting
estimator's effective sample size, was not tested. The lower block is not written
to a `results/*.json` file. ± is a standard error.

| Quantity | Value (pairs) |
|---|---|
| *Peak bin, 4.25 Å, amplitude 5 × 10⁻⁴* | |
| first order | −3.835 ± 0.170 |
| exact reweighting (ESS fraction 0.83) | −3.884 |
| direct sampling | −1.948 ± 0.821 |
| *Between-chain diagnostic, bin at 5.25 Å, n = 6 seed pairs* | |
| within-chain blocking error | 0.589 |
| between-chain scatter | 0.312 |
| ratio (between/within) | 0.53 |
| six-seed mean shift | +2.377 ± 0.144 |
| fresh-chain first order | +1.725 ± 0.101 |
| fresh-chain exact reweighting | +1.757 |

### Table VII. Lennard-Jones cutoff modes.

*Source: `docs/methods.md` §1, `docs/validation.md` §2.*

Energy and force discontinuities measured at *r_c* ∓ 10⁻⁷ Å with *r_c* = 7.0 Å.
The `shifted` mode, which is the common choice, removes the energy jump and
leaves the force jump untouched. `shifted_force` is used throughout this work.

| Mode | \|Δ*u*\| (eV) | \|Δ*F*\| (eV/Å) |
|---|---|---|
| `truncated` | 2.4 × 10⁻⁴ | 1.8 × 10⁻⁴ |
| `shifted` | 1.8 × 10⁻¹¹ | 1.8 × 10⁻⁴ |
| `shifted_force` | 8.7 × 10⁻¹⁹ | 1.6 × 10⁻¹¹ |
| `switched` | 2.2 × 10⁻¹⁹ | 1.4 × 10⁻¹⁶ |

### Table VIII. Validation against targets from outside the codebase.

*Source: `docs/validation.md`.*

The Stillinger–Weber entry is an *exactness check* and carries its full digits
for that reason: at the diamond lattice constant the answer is −2ε identically.
The embedded-atom copper entry is the one worth dwelling on — the potential
reproduces its three fitted targets exactly and still gives a maximum phonon
frequency about 30 % below copper's, which is the statement this paper makes
about force error arriving from a different direction. ± is a standard error.

| Check | Target | Obtained |
|---|---|---|
| SW Si cohesive energy | exactly −2ε = −4.33660 eV/atom | −4.3366000 |
| LJ fcc lattice sum | −8.6108 ε/atom | −8.603 (cutoff-extrapolated) |
| EAM Cu lattice constant | 3.615 Å | 3.6150 Å |
| EAM Cu cohesive energy | −3.54 eV/atom | −3.5400 eV/atom |
| EAM Cu bulk modulus | 140 GPa | 140.0 GPa |
| EAM Cu max. phonon freq. | ≈ 7.5 THz (real Cu) | 5.14 THz |
| fcc/bcc/sc *Q*₆ | 0.5745 / 0.5106 / 0.3536 | 0.5745 / 0.5107 / 0.3536 |
| SOAP rotational invariance | 0 | 8 × 10⁻¹⁵ |
| ACSF rotational invariance | 0 | 2 × 10⁻¹⁵ |
| HMC ⟨*U*⟩ (Einstein crystal) | 0.41363 eV | 0.41462 ± 0.00196 (0.5σ) |
| HMC Var(*U*) (Einstein crystal) | 0.003564 eV² | 0.003459 eV² |
| Phonon dispersion | 2√(*k*/*m*) \|sin(*qa*/2)\| | 10⁻⁵ relative |

---

## References

Numbered in order of first citation in `main.tex`; the bibliographic records are
in `paper/refs.bib`, from which the numbering is generated at compile time. The
works cited are:

Fu, Wu, Wang, Xie, Keten, Gómez-Bombarelli and Jaakkola, *Transactions on Machine
Learning Research* (2023), arXiv:2210.07237 · Stocker, Gasteiger, Becker,
Günnemann and Margraf, *Mach. Learn.: Sci. Technol.* **3**, 045010 (2022) ·
Morrow, Gardner and Deringer, *J. Chem. Phys.* **158**, 121501 (2023) · Kovács,
van der Oord, Kucera, Allen, Cole, Ortner and Csányi, *J. Chem. Theory Comput.*
**17**, 7696 (2021) · Wu, Zhou, Dong, Ying, Wang, Song, Fan and Xiong,
*J. Chem. Phys.* **161**, 014103 (2024) · Póta, Ahlawat, Csányi and Simoncelli,
arXiv:2408.00755 (2024) · Deng, Choi, Zhong, Riebesell, Anand, Li, Jun, Persson
and Ceder, *npj Comput. Mater.* **11** (2025) · Imbalzano, Zhuang, Kapil, Rossi,
Engel, Grasselli and Ceriotti, *J. Chem. Phys.* **154**, 074102 (2021) · Thaler
and Zavadlav, *Nat. Commun.* **12**, 6884 (2021) · Frederiksen, Jacobsen, Brown
and Sethna, *Phys. Rev. Lett.* **93**, 165501 (2004) · Behler and Parrinello,
*Phys. Rev. Lett.* **98**, 146401 (2007) · Behler, *J. Chem. Phys.* **134**,
074106 (2011) · Bartók, Payne, Kondor and Csányi, *Phys. Rev. Lett.* **104**,
136403 (2010) · Bartók, Kondor and Csányi, *Phys. Rev. B* **87**, 184115 (2013) ·
Drautz, *Phys. Rev. B* **99**, 014104 (2019) · Thompson, Swiler, Trott, Foiles
and Tucker, *J. Comput. Phys.* **285**, 316 (2015) · Batzner *et al.*,
*Nat. Commun.* **13**, 2453 (2022) · Batatia, Kovács, Simm, Ortner and Csányi,
NeurIPS **35** (2022) · Schütt, Sauceda, Kindermans, Tkatchenko and Müller,
*J. Chem. Phys.* **148**, 241722 (2018) · Gilmer, Schoenholz, Riley, Vinyals and
Dahl, ICML (2017) · Jones, *Proc. R. Soc. Lond. A* **106**, 463 (1924) · Verlet,
*Phys. Rev.* **159**, 98 (1967) · Rahman, *Phys. Rev.* **136**, A405 (1964) ·
Stillinger and Weber, *Phys. Rev. B* **31**, 5262 (1985) · Daw and Baskes,
*Phys. Rev. B* **29**, 6443 (1984) · Finnis and Sinclair, *Philos. Mag. A* **50**,
45 (1984) · Cleri and Rosato, *Phys. Rev. B* **48**, 22 (1993) · Zwanzig,
*J. Chem. Phys.* **22**, 1420 (1954) · Bennett, *J. Comput. Phys.* **22**, 245
(1976) · Shirts and Chodera, *J. Chem. Phys.* **129**, 124105 (2008) · Duane,
Kennedy, Pendleton and Roweth, *Phys. Lett. B* **195**, 216 (1987) · Neal, in
*Handbook of Markov Chain Monte Carlo* (2011) · Metropolis, Rosenbluth,
Rosenbluth, Teller and Teller, *J. Chem. Phys.* **21**, 1087 (1953) · Flyvbjerg
and Petersen, *J. Chem. Phys.* **91**, 461 (1989) · Künsch, *Ann. Stat.* **17**,
1217 (1989) · Politis and Romano, *J. Am. Stat. Assoc.* **89**, 1303 (1994) ·
Kish, *Survey Sampling* (Wiley, 1965) · Steinhardt, Nelson and Ronchetti,
*Phys. Rev. B* **28**, 784 (1983) · Lechner and Dellago, *J. Chem. Phys.* **129**,
114707 (2008) · Swope, Andersen, Berens and Wilson, *J. Chem. Phys.* **76**, 637
(1982) · Lebowitz, Percus and Verlet, *Phys. Rev.* **153**, 250 (1967) · Frenkel
and Smit, *Understanding Molecular Simulation*, 2nd ed. (Academic Press, 2002) ·
Allen and Tildesley, *Computer Simulation of Liquids*, 2nd ed. (OUP, 2017) ·
Hansen and McDonald, *Theory of Simple Liquids*, 4th ed. (Academic Press, 2013) ·
Bouthillier *et al.*, MLSys (2021), arXiv:2103.03098 · Pineau *et al.*,
*J. Mach. Learn. Res.* **22**, 1 (2021) · Kapoor *et al.*, *Sci. Adv.* (2024),
doi:10.1126/sciadv.adk3452 · and the Zenodo data and code deposit.
