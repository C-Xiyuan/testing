# atomlab — error fields, not error norms

A controlled study of **how the error in a machine-learned interatomic potential
reaches the physics you actually care about**, and why the metric the field
reports — force RMSE — is largely the wrong instrument for predicting it.

---

## The claim

Machine-learned interatomic potentials (MLIPs) are selected and reported on by
force error. They are then used to run molecular dynamics, from which physical
observables are extracted: radial distribution functions, phonon spectra,
diffusivities, elastic constants, melting points. The implicit chain is *small
force error ⇒ good physics*.

For a static observable `A`, writing the surrogate as `U = U₀ + δU`, first-order
perturbation theory in the canonical ensemble gives

$$\Delta\langle A\rangle \;=\; -\beta\,\mathrm{Cov}_0\!\left(A,\;\delta U\right) \;+\; \mathcal{O}(\delta U^2)$$

The observable error is a **covariance between the observable and the error
field** — a projection. Force RMSE is a **norm** of the error field's gradient.
Norms and projections are connected only by an inequality, and gradients weight
the error spectrum by `k²` while observables weight it by roughly `k⁰` or less.
The two functionals therefore rank models by nearly opposite criteria.

`docs/theory.md` derives this, works out the frequency-weighting argument, and
states in advance what would falsify it.

## Why an analytic reference potential

The study uses Lennard-Jones, Stillinger–Weber and EAM as the *ground truth*
rather than DFT data. This looks like a limitation and is actually the point:

- `U₀` is known **everywhere in configuration space**, not on a test set, so
  `δU` is exactly computable rather than estimated.
- Reference observables converge to arbitrary precision, so a disagreement
  between model and truth is a property of the model, not of the reference.
- Perturbations `δU` can be **designed** with chosen properties, which turns an
  observational claim into a constructive, falsifiable one.

No external dataset is used, downloaded, or needed.

## What is here

```
atomlab/
  units.py, types.py         metal units; Configuration / Result / Trajectory / Dataset
  cell.py, neighbors.py      triclinic minimum image, cell-linked neighbour lists
  build.py                   lattice and structure builders
  potentials/                Lennard-Jones, Morse, Stillinger-Weber, EAM, harmonic
    perturbations.py         designed δU with controlled covariance structure
  md/                        velocity Verlet, BAOAB Langevin, Nose-Hoover chains, MTK NPT
  observables/               g(r), S(q), ADF, Steinhardt, MSD/VACF/D, VDOS, phonons,
                             equation of state, elastic constants, melting diagnostics
  models/                    ACSF/SOAP/bispectrum descriptors, linear (ACE-like), BPNN,
                             GAP-like kernel model, E(3)-equivariant MPNN, pair spline
  training/                  dataset generation, trainer, active learning
  analysis/                  proxy-metric zoo, response theory, rank correlation, statistics
experiments/                 exp01..exp08, each reproducible from a config + seed
docs/theory.md               the derivation
docs/design.md               the interface contract
```

## The experiments

| | Question |
|---|---|
| `exp01_reference_physics` | Converged ground-truth observables with error bars. |
| `exp02_datasets` | Training sets at several budgets and sampling strategies. |
| `exp03_model_zoo` | Train every architecture across budgets and seeds. |
| `exp04_observables` | Run MD with every model; assemble the observable-error matrix. |
| `exp05_proxy_correlation` | **P1** — how well does any proxy metric rank models? |
| `exp06_response_validation` | **P2** — first-order vs reweighted vs direct MD. |
| `exp07_designed_counterexamples` | **P3** — invert the ranking by construction. |
| `exp08_practical_predictor` | A committee predictor that needs no ground truth, and its failure mode. |

## Install and run

```bash
pip install -e ".[dev]"
pytest -q                      # fast correctness suite
pytest -q -m slow              # physics validation (minutes)
python experiments/exp01_reference_physics/run.py --config config.json
```

Requires NumPy, SciPy, Numba, PyTorch (CPU is sufficient — everything here is
sized for four cores), Matplotlib. `ase` is a test-only cross-check dependency
and is never imported from `atomlab/`.

## Status

Under active construction. `docs/design.md` is the contract; results and figures
land in `results/` and `figures/` as each experiment completes.
