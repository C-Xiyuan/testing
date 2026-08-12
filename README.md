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

## What is here, and what it is evidence for

An earlier version of this section listed everything the codebase can do, which
read as a list of everything the study had shown. Those are different claims and
an external review was right to say so. Four states are distinguished below, and
only the last one supports a sentence in the results:

| state | meaning |
|---|---|
| **implemented** | code exists and runs |
| **validated** | checked against an independent closed-form or literature value, in `tests/` or `docs/validation.md` |
| **claim-bearing** | a number produced by this module appears in `docs/RESULTS.md` and is load-bearing for a conclusion |
| **planned** | not written |

| module | implemented | validated | claim-bearing |
|---|:--:|:--:|:--:|
| `units`, `types`, `cell`, `neighbors`, `build` | ✔ | ✔ | ✔ |
| `potentials/lennard_jones` | ✔ | ✔ (fcc lattice sum) | ✔ |
| `potentials/stillinger_weber` | ✔ | ✔ (E_coh = −2ε exactly) | ✘ |
| `potentials/eam` | ✔ | ✔ (a₀, E_coh, B₀) | ✘ |
| `potentials/morse`, `harmonic` | ✔ | ✔ | ✘ |
| `potentials/perturbations` | ✔ | ✔ (finite-difference forces) | ✔ |
| `sampling` (HMC, Metropolis) | ✔ | ✔ (equipartition, two samplers agree) | ✔ |
| `md/` (Verlet, Langevin, Nosé–Hoover, MTK) | ✔ | ✔ (energy conservation, T/P control) | ✘ |
| `observables/rdf`, `adf`, `thermo` | ✔ | ✔ | partly — `rdf` only |
| `observables/phonons` | ✔ | ✔ (1e-5 vs analytic chain) | ✘ |
| `observables/dynamics`, `vdos` | ✔ | ✔ | ✘ |
| `models/descriptors` (ACSF, SOAP, bispectrum) | ✔ | ✔ (rotational invariance to 1e-15) | ✔ |
| `models/linear`, `bpnn`, `egnn`, `pair_spline` | ✔ | ✔ (fit/predict round-trip) | ✔ |
| `analysis/response` | ✔ | ✔ (harmonic oscillator, exact) | ✔ |
| `analysis/fep` (BAR, MBAR) | ✔ | ✔ (displaced harmonic, exact) | pending exp10 |
| `analysis/statistics` | ✔ | ✔ (blocking vs known τ) | ✔ |
| `analysis/metrics`, `correlation` | ✔ | ✔ | ✔ |
| `training/` | ✘ | — | — | 

`training/` is empty: the model-fitting used in `exp03` lives in the model
classes themselves, and the dataset-generation and active-learning module the
directory was created for was never written.

## The experiments

Numbering is historical; four of the eight originally planned were never
written, and saying so is cheaper than pretending otherwise.

| | Question | State |
|---|---|---|
| `exp03_model_zoo` | Fitted models across architectures and budgets. | run; claim-bearing (§5.3) |
| `exp05_proxy_correlation` | **P1** — how well does any proxy metric rank models? | run; claim-bearing (§3, §5.1) |
| `exp06_response_validation` | **P2** — first-order vs reweighted vs direct MD. | run; **one unresolved disagreement**, see exp10 |
| `exp07_designed_counterexamples` | **P3** — invert the ranking by construction. | run; claim-bearing (§4) |
| `exp09_calibration_replication` | Are the exp07 residuals calibrated, or one shared offset? | run; claim-bearing (§5.5) |
| `exp10_endtoend_consistency` | Reconcile reference-based estimators with direct sampling. | written; running |
| `exp11_counterexample_replication` | Does the counterexample survive a change of construction chain? | written; running |
| `exp01`, `exp02`, `exp04`, `exp08` | reference physics, datasets, observable matrix, committee predictor. | planned; not written |

Standalone analyses that need no new sampling, in `scripts/`:

| | Question | State |
|---|---|---|
| `warning_light_calibration.py` | What are the second-order gate's error rates? | run; claim-bearing (§5.4) |
| `zoo_uncertainty_propagation.py` | Does the band survive each member's own measurement error? | run; claim-bearing (§3a) |
| `residual_variance_budget.py` | Where does the exp09 interaction come from? | run; claim-bearing (§5.5) |
| `validate_two_particle_exact.py` | An exact benchmark with no density expansion. | running |
| `band_robustness.py` | Is the reported window typical of its width? | run; claim-bearing (§3a) |
| `check_between_chain_scatter.py` | Is the blocking error optimistic or conservative? | run; claim-bearing (§5.2, §5.5) |

## Install and run

```bash
pip install -e ".[dev]"
pytest -q                      # fast correctness suite
pytest -q -m slow              # physics validation (minutes)
python experiments/exp07_designed_counterexamples/run.py --quick
```

Requires NumPy, SciPy, Numba, PyTorch (CPU is sufficient — everything here is
sized for four cores), Matplotlib. `ase` is a test-only cross-check dependency
and is never imported from `atomlab/`.

## Status of the scientific claims

The narrowest statement the data support:

> In one Lennard-Jones liquid state, for one pre-registered pair-count
> observable and one radial Gaussian basis, error fields can be constructed that
> match on held-out force RMSE and differ significantly in the directly sampled
> value of that observable.

That is a controlled existence proof. It does **not** establish that fitted
MLIP error fields occupy such directions in practice, that force RMSE fails as a
selector among realistic candidates, or that the response estimator is a usable
substitute for end-to-end validation. `reviews/` holds an external critique and
`reviews/CLAUDE_RESPONSE.md` the item-by-item reply, including which claims were
withdrawn and which experiments are outstanding.

`docs/design.md` is the interface contract; results and figures land in
`results/` and `figures/` as each experiment completes. Every manifest records
the commit, the working-tree dirt, and a SHA-256 over all source files, so a
number can be traced to the code that produced it.
