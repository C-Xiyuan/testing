# Methods

What was actually done, in enough detail to be repeated or disbelieved.
Numerical results live in `docs/RESULTS.md`; the argument they test is in
`docs/theory.md`; the module contract is `docs/design.md`.

---

## 1. The system

Lennard-Jones argon, 108 atoms, in a cubic periodic cell at number density
ρ = 0.0200 Å⁻³ and temperature T = 120 K. In reduced units that is
ρ\* = ρσ³ = 0.790 and T\* = k_BT/ε = 1.004, which places the state point in the
fluid region, below the freezing density at that temperature.

Potential parameters ε = 0.0103 eV, σ = 3.405 Å, cutoff 7.0 Å, in the
**shifted-force** form. The cutoff treatment is not cosmetic. Measured at
r_c ∓ 10⁻⁷ Å, the four available modes give:

| mode | \|Δu\| (eV) | \|ΔF\| (eV/Å) |
|---|---|---|
| truncated | 2.4 × 10⁻⁴ | 1.8 × 10⁻⁴ |
| shifted | 1.8 × 10⁻¹¹ | 1.8 × 10⁻⁴ |
| shifted_force | 8.7 × 10⁻¹⁹ | 1.6 × 10⁻¹¹ |
| switched | 2.2 × 10⁻¹⁹ | 1.4 × 10⁻¹⁶ |

A merely *shifted* potential has a continuous energy and a discontinuous force,
which delivers an impulse whenever a pair crosses the cutoff. Since the study
measures small systematic differences, an artefact of that kind would not be
distinguishable from the effect under investigation.

## 2. Sampling

**Not molecular dynamics.** Static observables are measured from Hamiltonian
Monte Carlo, implemented in `atomlab/sampling.py`. Each proposal draws momenta
from Maxwell-Boltzmann, integrates a short velocity-Verlet trajectory, and
accepts on the change in the total Hamiltonian. Because velocity Verlet is
symplectic and time-reversible the proposal is symmetric, so the stationary
distribution is exactly exp(−βU) *regardless of integration error* — which
appears only as a reduced acceptance rate, never as a biased distribution.

That property is the reason for the choice. A thermostat that samples something
slightly other than the canonical distribution would introduce exactly the kind
of small systematic difference this study attributes to model error. HMC removes
that possibility by construction.

Validated against exact equipartition on a 32-atom Einstein crystal at 100 K,
where both moments are known in closed form:

| quantity | exact | measured |
|---|---|---|
| ⟨U⟩ | (3N/2)k_BT = 0.41363 eV | 0.41462 ± 0.00196 (0.5 σ) |
| var(U) | (3N/2)(k_BT)² = 0.003564 eV² | 0.003459 |

Both moments matter: an incorrect temperature scale reproduces the mean and
fails on the variance. A single-particle Metropolis sampler sharing no code
beyond the potential is included as an independent cross-check.

Typical production settings: 8 leapfrog steps per proposal, step size adapted
during burn-in to an acceptance of 0.75, giving ≈ 30 fs after adaptation.

## 3. Equilibration, and the failure that made it necessary

The liquid is built by scaling an fcc lattice to liquid density. **That
configuration is a crystal**, and sampling it directly produces a slowly melting
crystal rather than a liquid. Measured on the original protocol: the global
bond-order parameter Q₆ fell monotonically from 0.5745 (perfect fcc) to 0.43
over 600 proposals and was still falling, while the potential energy climbed
monotonically and was still climbing. The equilibrated liquid sits at
−0.0311 eV/atom; those runs were sampling −0.036 to −0.043, so the reference
ensemble was wrong by 30–40 % in energy.

No sampler diagnostic caught this. Acceptance rate and integration error were
healthy throughout, because they measure whether the Markov chain is being
*simulated* correctly, not whether it has reached its stationary distribution.
Those are different questions and only the second one bears on an ensemble
average.

The protocol now used (`experiments/equilibrate.py`):

1. **Melt** at 5 × the target temperature for a few hundred proposals. Melting
   at the target temperature works but is an order of magnitude slower, because
   the barrier to cross is precisely the one that makes the crystal metastable.
2. **Anneal** to the target temperature.
3. **Refuse** to return a configuration with Q₆ above 0.20. A 108-atom liquid
   gives ≈ 0.08; perfect fcc gives 0.5745.

After the protocol: Q₆ = 0.074, U/atom = −0.0315 eV, with no drift between the
first and second halves of the production run.

Every production trajectory is then split in half and both its energy and its
order parameter are compared between halves. A drift exceeding 3 combined
standard errors **raises**, rather than warning. The failure mode gives no
outward sign, so anything softer than an exception would eventually be ignored.

## 4. Observables

Measured quantities are pair counts in radial bins,
`A_k(x) = |{(i,j) : r_k ≤ r_ij < r_{k+1}}|`. These differ from `g(r)` in that
bin only by a constant at fixed N and V, and the conversion is provided, but the
analysis is done on the counts.

The reason is interpretability rather than convenience: response theory applies
to an observable that is a plain function of the configuration, and a bin count
is exactly that, so the correspondence between the predicted quantity and the
measured one stays explicit with no normalisation standing between them.

The bins taken together form a vector observable. The designed-perturbation
experiments additionally target a **scalar**: the pair count in a single bin
across the first peak. That matters statistically — it makes the covariance a
K-vector rather than a K × J matrix, estimable from far fewer frames — and
rhetorically, since the claim becomes about one number rather than a curve.

Supporting estimators were validated against exact lattice values: fcc Q₆ =
0.5745 and Q₄ = 0.1909, simple cubic Q₆ = 0.3536 and Q₄ = 0.7638, fcc
coordination 12 at a/√2, all reproduced exactly.

## 5. Error bars

Every reported uncertainty comes from correlated-sample statistics
(`atomlab/analysis/statistics.py`), never from `1/√M`. Monte Carlo samples are
correlated, and treating M frames as M independent samples understates the error
by roughly √(2τ) — for a typical liquid observable, a factor of several.

- Means use Flyvbjerg-Petersen blocking, with the plateau taken over levels
  retaining at least eight blocks.
- Arbitrary statistics (covariances, fitted slopes, rank correlations) use a
  moving-block bootstrap with block length set to four integrated
  autocorrelation times of the *product* series that carries the error — using
  the observable's own correlation time underestimates it whenever δU varies
  slowly, which is the systematic-error case.
- A jackknife with different bias structure is available as a cross-check.

Validated on an AR(1) process with known τ: the blocking estimate recovers the
naive standard error inflated by √(2τ) to within 30 %, and the measured
integrated autocorrelation time matches 1/2 + φ/(1−φ) to within 15 %.

## 6. Response estimators

Three quantities are computed on identical samples wherever a comparison is
made (`atomlab/analysis/response.py`):

- **first order**: −β Cov₀(A, δU), with a block-bootstrap interval;
- **second order**: (β²/2)⟨Ã δŨ²⟩₀, reported but not added, as a
  self-diagnostic on the truncation;
- **reweighted**: ⟨A e^{−βδU}⟩₀ / ⟨e^{−βδU}⟩₀, exact to all orders, with the
  Kish effective sample size and the largest single-frame weight fraction, since
  an exponential average can be precise-looking and meaningless.

Validated against closed forms on a harmonic reference perturbed by a linear
field, where the expansion terminates and each order can be isolated:
`⟨x⟩` shifts by exactly −ε/k (pure first order), `⟨x²⟩` by exactly ε²/k² (pure
second order), and the Zwanzig free-energy shift is −ε²/2k. All three are
recovered to within statistical error.

## 7. Designed error fields

`atomlab/potentials/perturbations.py` constructs error fields with chosen
properties, each a full potential with analytic forces and virial so that a
surrogate is built by ordinary addition. All derivatives agree with finite
differences to ~10⁻¹⁰ on both cubic and sheared triclinic cells.

The constructions used here expand a pair error field in a basis of Gaussian
shell bumps, which makes the covariance with an observable *linear* in the
coefficients. Choosing coefficients in the null space of that covariance, or
parallel to it, is then linear algebra rather than optimisation:

- **null** — first-order predicted effect zero by construction;
- **aligned** — maximal effect per unit force error;
- **random** — the control.

All three are scaled to the same force RMSE on the reference ensemble, so the
number a practitioner would report is held fixed across the comparison.

**Out-of-sample discipline.** A null-space field is orthogonal to the observable
on the frames it was built from *by definition*. The reference trajectory is
therefore split: fields are constructed on one half and every reported number —
force RMSE, predicted shift, covariance — is computed on the other. How many
frames the construction needs before it generalises is itself measured and
reported, because with a badly estimated covariance the null space found is the
null space of the noise.

## 8. Reproducibility

Each experiment writes a manifest recording the git commit, whether the tree was
dirty, the full configuration, per-stage wall times, library versions and the
seed. Seeds are derived from a single experiment seed through named
`SeedSequence` sub-streams, so inserting a stage does not perturb every result
after it. Stages are cached, so an interrupted campaign resumes rather than
restarts, and the manifest records which stages were reused.

`--quick` shrinks every size parameter for a smoke test — except equilibration,
which is deliberately never shortened. An unequilibrated reference does not test
the pipeline faster; it produces numbers from the wrong distribution, which is
the failure the guard exists for.
