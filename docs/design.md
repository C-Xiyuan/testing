# `atomlab` — design contract

This document is the **interface contract**. Every module is implemented against
it, so that independently written pieces compose without negotiation. If you are
implementing a module listed here, treat the signatures below as fixed: other
modules are being written against them right now.

---

## 1. What the project is trying to establish

Machine-learned interatomic potentials (MLIPs) are almost universally selected
and reported on by **force error** — force RMSE or MAE on a held-out test set.
The premise of this repository is that this number is close to uninformative
about the thing practitioners actually want, which is whether a molecular
dynamics simulation driven by the model reproduces the *physics*: radial
distribution functions, phonon spectra, diffusivities, equations of state,
elastic response, melting behaviour.

The claim is not merely empirical. For a static observable `A` in the canonical
ensemble, writing the surrogate potential as `U = U₀ + δU`, first-order
perturbation theory gives

```
⟨A⟩_U − ⟨A⟩_{U₀} = −β · Cov_{U₀}(A, δU) + O(δU²)
```

The error in an observable is controlled by the **covariance of the observable
with the error field**, not by any norm of the error. A model with large `‖δU‖`
that happens to be uncorrelated with `A` is harmless; a model with small `‖δU‖`
aligned with `A` is not. Force RMSE is a norm. Hence the failure.

This yields three testable predictions, which the experiments in `experiments/`
are built to check:

- **P1 (weak-proxy).** Across a model zoo, rank correlation between force RMSE
  and downstream observable error is weak and observable-dependent.
- **P2 (theory).** The covariance formula predicts `Δ⟨A⟩` quantitatively, to
  within the second-order remainder, using **only reference-ensemble samples** —
  no MD with the surrogate is required.
- **P3 (constructive).** One can *design* perturbations `δU` with large force
  error but (by construction) near-zero covariance with a target observable, and
  vice versa, producing pairs of models that invert the usual ranking. This is
  the sharpest possible falsification test of the "force error is enough" view.

The ground truth is an **analytic classical potential** (Lennard-Jones,
Stillinger-Weber, EAM). This is the essential methodological move: it makes
`U₀` exactly known everywhere in configuration space, makes `δU` exactly
computable rather than estimated, and makes reference observables convergeable
to arbitrary precision. No external dataset is needed or wanted.

---

## 2. Units

Metal units, exactly as LAMMPS defines them. See `atomlab/units.py`.

| Quantity | Unit |
|---|---|
| length | Å |
| energy | eV |
| mass | amu |
| time | ps |
| force | eV/Å |
| velocity | Å/ps |
| temperature | K |
| pressure (API) | bar or GPa, always named in the signature |
| stress (internal) | eV/Å³ |

Newton's law carries a conversion factor: `a [Å/ps²] = F [eV/Å] / (m [amu] · MVV2E)`
with `MVV2E = 1.0364269e-4`. Kinetic energy is `0.5 · MVV2E · Σ m v²`.
**Never** hard-code these numbers; import from `atomlab.units`.

## 3. Core types

Defined in `atomlab/types.py`. Do not redefine or subclass them.

- `Configuration` — positions `(N,3)`, `cell` `(3,3)` with **rows** as lattice
  vectors (`r = s @ cell`), `pbc` `(3,)` bool, `species` `(N,)` int32 type
  indices, `masses` `(N,)` amu, optional labels `energy` / `forces` / `virial`.
- `Result` — what a potential returns: `energy`, `forces`, `virial`, optional
  per-atom `energies`, free-form `extra`.
- `Trajectory` — `(T,N,3)` **unwrapped** positions, `cells`, `times`, optional
  `velocities`, a `template` configuration, and a `scalars` dict of `(T,)` logs.
- `Dataset` — a list of labelled configurations with `split()` helpers.

### Virial sign convention (get this right)

`virial` is `W_ab = −∂U/∂ε_ab` at zero strain, where strain acts as
`r → (1+ε)r` on positions **and** lattice vectors. Units eV. For a pair
potential this is `Σ_{i<j} f_ij ⊗ r_ij` with `f_ij` the force on `i` from `j`
and `r_ij = r_i − r_j`. The potential part of stress is `W/V`; the kinetic part
is added by the MD observables layer. `Potential.numerical_virial` is the
arbiter — if your analytic virial disagrees with it, your analytic virial is
wrong.

## 4. The `Potential` interface

Defined in `atomlab/potentials/base.py`. **Implement `compute()` and nothing
else is required.**

```python
class Potential(abc.ABC):
    cutoff: float          # Å; neighbour machinery relies on this
    name: str              # short id used in tables and figures

    @abc.abstractmethod
    def compute(self, configuration: Configuration, *,
                forces: bool = True, virial: bool = True) -> Result: ...
```

Free for all subclasses: `energy`, `energy_per_atom`, `forces`, `virial`,
`stress`, `pressure`, `label`, `numerical_forces`, `numerical_virial`, and the
algebra `p1 + p2`, `2.0 * p`, `p1 - p2` (via `SumPotential` / `ScaledPotential`).
That algebra is load-bearing: the perturbation experiments literally construct
`U₀ + δU` this way.

**Contract details.**
- `compute` must not mutate the input configuration.
- Forces are `−∂U/∂r`, in eV/Å.
- A potential with a finite `cutoff` must be **exactly** zero beyond it,
  including in its derivatives — i.e. use a smooth cutoff function, never a bare
  truncation, unless the truncation is the intended physics (see
  `lennard_jones.py`, which offers both and documents the difference).
- Everything must be correct under the minimum image convention for cells whose
  smallest width exceeds `2 * cutoff`, and must **raise** if that is violated.

## 5. Module map

Each row is owned by exactly one implementation task. Do not create files
outside your own row; if you need something from another row, import it and
assume it works as specified.

### 5.1 Geometry and neighbours

| File | Responsibility |
|---|---|
| `atomlab/cell.py` | Minimum-image displacement, wrapping, cell metrics, `min_cell_width(cell, pbc)`, lattice-parameter ↔ matrix conversion, strain helpers. |
| `atomlab/neighbors.py` | Neighbour lists. Cell-linked-list construction, half and full lists, Verlet skin with rebuild triggers. Must handle small cells where an atom sees the same neighbour through multiple images. |
| `atomlab/build.py` | Lattice builders: `fcc`, `bcc`, `sc`, `diamond`, `hcp`, plus `random_gas`, `rattle`, `vacancy`, `interstitial`, `liquid_from_melt` helpers. All return `Configuration`. |

**Neighbour list API** (fixed):

```python
@dataclass
class NeighborList:
    i: np.ndarray        # (P,) int32 central atom index
    j: np.ndarray        # (P,) int32 neighbour atom index
    shift: np.ndarray    # (P,3) int32 periodic image of j, in lattice units
    n_atoms: int
    cutoff: float
    half: bool           # True => each pair appears once (i<j ordering by image)

def build_neighbor_list(configuration, cutoff, *, half=False, skin=0.0) -> NeighborList: ...
def pair_vectors(configuration, nl) -> tuple[np.ndarray, np.ndarray]:
    """Return (D, r) with D[p] = r_j + shift @ cell − r_i and r = |D|."""
```

Note the displacement sign: `D` points **from i to j**. Every potential in this
repository assumes that.

### 5.2 Reference potentials

| File | Responsibility |
|---|---|
| `atomlab/potentials/lennard_jones.py` | LJ 12-6, both shifted-force and smoothly-switched variants; `LennardJones.argon()` preset. |
| `atomlab/potentials/stillinger_weber.py` | Stillinger–Weber silicon (original 1985 parameters), two- and three-body terms. |
| `atomlab/potentials/eam.py` | Analytic embedded-atom model for an fcc metal, with an analytic (not tabulated) embedding function so derivatives are exact. |
| `atomlab/potentials/morse.py` | Morse pair potential, used as a second two-body reference. |
| `atomlab/potentials/harmonic.py` | Einstein crystal / harmonic tether, needed as a free-energy reference state. |
| `atomlab/potentials/perturbations.py` | **The experimental instrument.** Constructs `δU` perturbations with controlled properties — see §6. |

Validation targets that must be hit (these are physics, not style):
- LJ fcc at `a₀ = 1.5496 σ` (cutoff-free limit) gives lattice energy
  `−8.610 ε/atom` from the Lennard-Jones lattice sums.
- Stillinger–Weber silicon: diamond structure at `a₀ = 5.431 Å` has cohesive
  energy exactly **`−2ε = −4.33660 eV/atom`** for `ε = 2.1683 eV`. This is exact,
  not fitted-approximate: the diamond nearest-neighbour distance `a₀√3/4 =
  2.3517 Å` coincides with `2^{1/6}σ`, where the two-body term attains its
  minimum value of exactly `−ε`; every bond angle is exactly tetrahedral so the
  three-body term vanishes identically; and the second-neighbour shell at
  `3.8403 Å` lies outside the `aσ = 3.77118 Å` cutoff. *(An earlier draft of this
  document said `−4.3363`, which corresponds to `ε = 2.16815 eV` — the 50
  kcal/mol rounding of the same parameter. The value above is the one consistent
  with the `ε` specified here.)*
- Every potential passes `check_forces < 1e-6` and `check_virial < 1e-6`.
- Cross-check against ASE's `LennardJones` and `StillingerWeber` calculators in
  the test suite (dev dependency only — never import `ase` from `atomlab/`).

### 5.3 Molecular dynamics

| File | Responsibility |
|---|---|
| `atomlab/md/integrators.py` | `VelocityVerlet`, `Langevin` (BAOAB splitting), `NoseHooverChain` (Suzuki–Yoshida), `MTKBarostat` for NPT. Each exposes `step(state, potential)`. |
| `atomlab/md/state.py` | `MDState` dataclass: positions (unwrapped), velocities, cell, forces, cached `Result`, plus thermostat internal variables. |
| `atomlab/md/simulate.py` | The driver: `run_md(...) -> Trajectory`, logging, frame stride, progress, restarts, centre-of-mass removal. |
| `atomlab/md/velocities.py` | Maxwell–Boltzmann initialisation, COM removal, rescaling, degrees-of-freedom bookkeeping. |

Non-negotiable properties, each with a test:
- NVE energy drift over 50 ps at 1 fs with LJ argon: `< 1e-4 eV/atom`, and drift
  must scale as `dt²` when the timestep is halved.
- Langevin and NHC both reproduce the target temperature to within statistical
  error, and reproduce the **kinetic energy distribution**, not just its mean.
- The configurational temperature `⟨|F|²⟩ / ⟨∇·F⟩` agrees with the thermostat
  temperature — this catches subtle integrator bugs that mean-KE checks miss.

### 5.4 Observables

Each observable is a function of a `Trajectory` (or of a `Configuration` +
`Potential` for static ones) returning a dataclass with the value, a bootstrap
error bar, and enough metadata to plot it.

| File | Provides |
|---|---|
| `atomlab/observables/rdf.py` | `radial_distribution(traj, ...)`, partial RDFs, coordination number, structure factor `S(q)` via Fourier transform of `g(r)`. |
| `atomlab/observables/adf.py` | Bond-angle distribution, tetrahedral order parameter `q`, Steinhardt `Q4`/`Q6`. |
| `atomlab/observables/dynamics.py` | MSD (with proper unwrapping), self-diffusion coefficient from both Einstein and Green–Kubo routes, VACF. |
| `atomlab/observables/vdos.py` | Vibrational density of states from the VACF power spectrum, with windowing. |
| `atomlab/observables/phonons.py` | Finite-displacement force constants, dynamical matrix, phonon band structure and DOS at arbitrary q. |
| `atomlab/observables/thermo.py` | Equation of state (Birch–Murnaghan fit), bulk modulus, thermal expansion, heat capacity, elastic constants `C11/C12/C44` by strain–stress fitting. |
| `atomlab/observables/melting.py` | Lindemann index and solid/liquid discrimination via `Q6`. **Scoped down:** a two-phase coexistence melting-point determination needs multi-nanosecond runs on thousands of atoms, which four CPU cores cannot deliver at the number of models this study compares. Melting is therefore represented by the cheap order-parameter diagnostics only, and no melting temperature is reported. |

### 5.3a Sampling: why static observables do not use molecular dynamics

`atomlab/sampling.py` provides Hamiltonian Monte Carlo and single-particle
Metropolis, and **every static observable in this study is measured from HMC
samples, not from MD**. The reason is specific to what is being studied.

The whole argument concerns small, systematic differences between ensembles.
A thermostat that samples something slightly other than the canonical
distribution — and most of them do, to some degree, at finite timestep — would
introduce exactly the kind of small systematic difference the study is trying to
attribute to model error. HMC removes that risk by construction: its Metropolis
test makes the stationary distribution exactly `exp(-βU)` regardless of
integration error, which appears only as a reduced acceptance rate. The
reference ensemble is then unimpeachable, and the response theory of
`docs/theory.md` — which is a statement about canonical averages and nothing
else — applies without an asterisk.

MD retains one irreplaceable role: **dynamical** observables. Diffusion
coefficients, velocity autocorrelations and vibrational spectra from the VACF
require real time evolution, which Monte Carlo cannot provide at any price.
`Trajectory.info["dynamical"]` records which sampler produced a trajectory, and
the dynamical estimators refuse a non-dynamical one rather than silently
returning a meaningless number.

Every estimator must return **error bars** computed by block averaging or
bootstrap over correlated samples. A comparison between two models without
error bars is not evidence, and this project's entire claim is a comparison.

### 5.5 Machine-learned models

All subclass `Potential`, so MD does not know or care that they are learned.

| File | Provides |
|---|---|
| `atomlab/models/descriptors/acsf.py` | Behler–Parrinello symmetry functions G2/G4/G5 with analytic derivatives. |
| `atomlab/models/descriptors/soap.py` | SOAP power spectrum via spherical harmonics + radial basis, analytic derivatives. |
| `atomlab/models/descriptors/bispectrum.py` | Polynomial/ACE-style invariant basis (radial × angular products) for the linear model. |
| `atomlab/models/linear.py` | Linear/ridge fit on an invariant basis — the "ACE-lite" baseline. Fits energies **and** forces jointly by stacking the design matrix. |
| `atomlab/models/bpnn.py` | Behler–Parrinello neural network (per-species MLP on ACSF), PyTorch, with autograd forces. |
| `atomlab/models/gap.py` | Sparse/kernel ridge regression on SOAP (GAP-like), with predictive variance for uncertainty. |
| `atomlab/models/egnn.py` | E(3)-equivariant message-passing network written from scratch: real spherical harmonics, Clebsch–Gordan tensor products, gated nonlinearities. `L_max` configurable. |
| `atomlab/models/pair_spline.py` | Learned **pair-only** potential (cubic spline in `r`). Deliberately under-expressive; the point is that it is exactly right for LJ and structurally wrong for SW. |
| `atomlab/models/ensemble.py` | Committee of models: mean prediction, disagreement as an uncertainty estimate, and `delta_u_estimate()` used by the practical predictor of §6.4. |

Model contract additions on top of `Potential`:

```python
class MLModel(Potential):
    def fit(self, train: Dataset, *, val: Dataset | None = None, **kw) -> FitReport: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "MLModel": ...
    @property
    def n_parameters(self) -> int: ...
```

Requirements with tests:
- **Exact invariance/equivariance** under translation, rotation, permutation and
  periodic image choice, to machine precision for descriptor models and to
  `<1e-6` relative for the neural ones.
- Analytic forces match `numerical_forces` to `<1e-6` eV/Å for every model.
- Every model is trainable to a useful accuracy on a few thousand 64-atom
  configurations within minutes on 4 CPU cores. Size the architectures for that;
  this is a scientific study, not a scaling demo.

### 5.6 Training

| File | Provides |
|---|---|
| `atomlab/training/generate.py` | Reference dataset generation: MD sampling at a grid of temperatures/densities, rattled crystals, compressed/expanded cells, defected structures, liquid snapshots. |
| `atomlab/training/trainer.py` | Loss (`w_E · energy + w_F · force + w_V · virial`), optimiser loop, early stopping, LR schedule, deterministic seeding, `FitReport`. |
| `atomlab/training/active_learning.py` | Uncertainty-driven selection loop, used to produce models that differ in *where* their error lives rather than in how big it is. |

### 5.7 Analysis — the scientific core

| File | Provides |
|---|---|
| `atomlab/analysis/metrics.py` | The proxy-metric zoo: energy MAE/RMSE, force RMSE/MAE/cosine, per-atom-energy error, virial error, Hessian/curvature error, force error restricted to high-energy configurations, ensemble disagreement, extrapolation grade. |
| `atomlab/analysis/response.py` | **The linear-response machinery.** See §6. |
| `atomlab/analysis/correlation.py` | Spearman/Kendall rank correlation between proxy metrics and observable errors, with bootstrap confidence intervals and multiple-comparison control. |
| `atomlab/analysis/statistics.py` | Block averaging, autocorrelation time, bootstrap, jackknife, and the statistical-significance tests used in the report. |

## 6. The response module in detail

`atomlab/analysis/response.py` is the piece the whole repository exists to
support. It must provide:

### 6.1 First-order prediction

```python
def predict_shift(A_samples, dU_samples, temperature, *, weights=None) -> ResponsePrediction
```

Given `A` and `δU` evaluated on the **same reference-ensemble samples**,
returns `−β·Cov(A, δU)` with a block-bootstrap error bar and an estimate of the
second-order remainder (`β²/2 · [⟨A δU²⟩ − ⟨A⟩⟨δU²⟩ − 2⟨δU⟩Cov(A,δU)]`), which
tells the caller when to distrust the linear result.

### 6.2 Exponential reweighting (the non-perturbative check)

```python
def reweight(A_samples, dU_samples, temperature) -> ReweightResult
```

The exact identity `⟨A⟩_U = ⟨A e^{−βδU}⟩₀ / ⟨e^{−βδU}⟩₀`. This is exact to all
orders but suffers from the usual exponential-average variance problem, so it
must also return the **effective sample size** `ESS = (Σw)²/Σw²`. Comparing
first-order, reweighted, and direct-MD answers on the same system is the
cleanest possible validation of the theory, because all three should agree in
the small-`δU` limit and diverge in an understood way outside it.

### 6.3 Vector-valued observables

`g(r)` is a vector observable (one component per bin), so the covariance becomes
a vector and the prediction is a whole predicted difference curve `Δg(r)`. The
API must handle `A_samples` of shape `(M,)` or `(M, K)` uniformly.

### 6.4 The practical predictor

The formula above needs `δU = U_model − U_true`, which a practitioner does not
have. The practical version replaces the unknown truth with a committee mean:
`δU_i ≈ U_i − ⟨U⟩_committee`. `response.py` must expose this as

```python
def committee_response_score(models, A_fn, configurations, temperature) -> dict
```

and the experiments must check how well it tracks the oracle version. If it
does, this is a usable model-selection criterion that costs one energy
evaluation per sample and no MD at all. That is the practical payoff, and it is
the claim most worth being sceptical of — design the experiment to be able to
report a negative result.

## 7. Designed perturbations (`potentials/perturbations.py`)

To test **P3** we need `δU` we control. Provide at least:

- `RadialShellPerturbation(r0, width, amplitude)` — a smooth bump in the pair
  energy localised in `r`. Its covariance with `g(r)` is large near `r₀` and
  small elsewhere, so it moves a *chosen* part of the RDF.
- `AngularPerturbation(...)` — three-body term that changes bond angles while
  leaving the pair distribution nearly untouched.
- `NullSpacePerturbation(reference_samples, target_observable, ...)` — the key
  construction: a perturbation with **large force norm but numerically
  orthogonal** to a target observable, built by projecting a random smooth
  perturbation onto the null space of `Cov(A, ·)` over a basis of shell bumps.
- `HighEnergyPerturbation(...)` — supported only on configurations rarely
  visited at the sampling temperature, so it is nearly invisible to both the
  test-set force error and the observables, but explodes under extrapolation.

Each must be a proper `Potential` with analytic forces and virial, so it can be
added to a reference potential and run in MD directly.

## 8. Experiments

`experiments/expNN_name/run.py`, each writing to `results/expNN_name/` and
figures to `figures/`. Each script takes `--config` (a JSON/YAML in the same
directory), is deterministic given a seed, and records the git commit,
wall-clock time, and full parameter set in a `manifest.json`.

| Experiment | Question |
|---|---|
| `exp01_reference_physics` | Establish converged ground-truth observables for LJ argon and SW silicon, with error bars. Everything downstream compares to these. |
| `exp02_datasets` | Build train/val/test sets at several budgets and sampling strategies. Characterise their coverage. |
| `exp03_model_zoo` | Train the full model zoo across budgets/seeds. Report standard metrics. |
| `exp04_observables` | Run MD with every trained model; measure every observable; assemble the error matrix. |
| `exp05_proxy_correlation` | **P1.** Rank-correlate every proxy metric against every observable error. |
| `exp06_response_validation` | **P2.** First-order vs reweighted vs direct MD, as a function of perturbation strength. |
| `exp07_designed_counterexamples` | **P3.** The constructed null-space and aligned perturbations; show the ranking inverts. |
| `exp08_practical_predictor` | §6.4 committee predictor vs oracle, as a model-selection rule. |

## 9. Engineering standards

- **NumPy-first.** Vectorise; use `numba.njit` only where profiling justifies it
  (pair loops, neighbour construction, RDF accumulation). Keep a pure-NumPy
  reference implementation next to every jitted kernel and test them against
  each other — a fast wrong kernel is the most expensive kind of bug here.
- **Determinism.** Every stochastic entry point takes an explicit `seed` or
  `rng`. No module-level `np.random` calls. Ever.
- **Type hints** on all public functions; NumPy-style docstrings that state
  units and array shapes.
- **Tests** live in `tests/test_<module>.py` and must cover: analytic vs
  numerical derivatives, symmetry invariances, known physical constants, and
  agreement between fast and reference implementations. Mark anything over ~10 s
  `@pytest.mark.slow`.
- **No silent fallbacks.** If a cutoff exceeds half the cell width, raise. If a
  fit does not converge, say so in the `FitReport`. Quiet degradation would
  contaminate the very correlations this study is measuring.
- Import only from the standard library, NumPy, SciPy, Numba, PyTorch, and
  Matplotlib inside `atomlab/`. `ase` is a **test-only** cross-check.
