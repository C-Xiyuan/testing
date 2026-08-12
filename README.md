# atomlab — error fields, not error norms

A controlled study of **how an error field can reach a static observable**.  The
current evidence is a bounded Lennard-Jones construction, not a validated model
selection method and not evidence that force RMSE generally fails on fitted
potentials.  The authoritative claim status is
[`reviews/CURRENT_CLAIM_LEDGER.md`](reviews/CURRENT_CLAIM_LEDGER.md).

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
Relating these objects requires an ensemble- and system-dependent inequality
(for example a Poincaré constant), a fixed energy gauge, and the specified
observable. There is no task-agnostic ordering, so the two functionals can rank
constructed error fields differently.
How often this occurs for fitted models, and whether using the response score
improves selection decisions, remain unanswered (Gates B and C).

`docs/theory.md` is the historical derivation/design notebook; its standard
identities remain useful, but its empirical promises and trust language are
superseded by the current claim ledger.

## Why an analytic reference potential

The claim-bearing study uses Lennard-Jones as the analytic *ground truth*
rather than DFT data. Stillinger–Weber and EAM implementations exist and have
unit-level checks, but no observable-error result from either is part of the
evidence. The analytic oracle is useful because:

- `U₀` is known **everywhere in configuration space**, not on a test set, so
  `δU` is exactly computable rather than estimated.
- Reference observables can be converged independently without label noise, so
  their sampling uncertainty can be separated from surrogate error.
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

Here, “claim-bearing” records use in the **legacy narrative**.  It does not make
the associated result release-ready; the claim ceiling and provenance status in
`reviews/CURRENT_CLAIM_LEDGER.md` still apply.

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
| `analysis/fep` (BAR, MBAR) | ✔ | ✔ (displaced harmonic, exact) | legacy exp10 only; confirmatory v2 pending |
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
| `exp03_model_zoo` | Fitted models across architectures and budgets. | legacy same-system feasibility evidence only (§3b); no ranking inference |
| `exp05_proxy_correlation` | **P1** — how well does any proxy metric rank models? | legacy fixed-zoo description only (§3a, §5.1); no population or selection inference |
| `exp06_response_validation` | **P2** — first-order vs reweighted vs direct sampling. | legacy run; discrepancy not reproduced in exp10 point estimates, but not resolved |
| `exp07_designed_counterexamples` | **P3** — invert the ranking by construction. | legacy fixed-cell construction (§1, §4); provisional until clean regeneration |
| `exp09_calibration_replication` | Are the exp07 residuals calibrated, or one shared offset? | legacy exploratory run; non-IID direct streams and invalid provenance; v2 repaired and smoke-tested fail-closed, production not rerun |
| `exp10_endtoend_consistency` | Reconcile reference-based estimators with direct sampling. | legacy point non-recurrence only; formal result underpowered/not evaluable; v2 repaired and smoke-tested fail-closed, production not rerun |
| `exp11_counterexample_replication` | Does the counterexample survive a change of construction chain? | legacy fixed-cell partial replication; field-level UQ defect and invalid provenance; v2 repaired with held-out manipulation gate and smoke-tested fail-closed, production not rerun |
| `exp01`, `exp02`, `exp04`, `exp08` | reference physics, datasets, observable matrix, committee predictor. | planned; not written |

Standalone analyses that need no new sampling, in `scripts/`:

| | Question | State |
|---|---|---|
| `warning_light_calibration.py` | What are the second-order gate's error rates? | legacy diagnostic; the gate is withdrawn (§5.4) |
| `zoo_uncertainty_propagation.py` | Does the band survive each member's own measurement error? | legacy fixed-zoo sensitivity analysis; no new independent units (§3a) |
| `residual_variance_budget.py` | Where does the exp09 interaction come from? | legacy post-hoc decomposition; non-IID and not joint UQ (§5.5) |
| `validate_two_particle_exact.py` | An exact N=2 quadrature/sampler sanity check. | run; parameters do not match exp06, so it does not explain the dilute-gas residual |
| `band_robustness.py` | Is the reported window typical of its width? | legacy window sensitivity analysis over the same clustered zoo (§3a) |
| `check_between_chain_scatter.py` | Is the blocking error optimistic or conservative? | legacy diagnostic contradicted by exp09's different-bin estimate (§5.2, §5.5) |

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

The narrowest statement suggested by the legacy artefacts:

> In one Lennard-Jones liquid state, for one pre-registered pair-count
> observable and one radial Gaussian basis, legacy fixed-cell construction
> clusters report a large directly sampled aligned-minus-null contrast while
> differing by 0.52% in held-out force RMSE. The paired contrast uncertainty
> was not retained.

That is a provisional, fixed-cell legacy observation pending clean v2
replication. It does **not** establish that fitted
MLIP error fields occupy such directions in practice, that force RMSE fails as a
selector among realistic candidates, or that the response estimator is a usable
substitute for end-to-end validation.  The supporting exp07/exp09/exp10/exp11
production artefacts also predate a valid start-of-run provenance record; they
must be regenerated before release.  `reviews/` holds the external critique,
the historical Claude response, and the current claim ledger.

`docs/design.md` records the historical architecture/experiment plan, including
unimplemented and refuted items. `paper/SPEC.md` is now the short audited
corrective specification. Neither overrides the current claim ledger. Results
and figures land in `results/` and `figures/`.
Legacy manifests read
repository state at run completion and therefore cannot prove which source was
executed.  The repair branch captures start and end state separately, rejects
dirty production starts and hashes artefacts, but those safeguards become
evidence only after new v2 production runs complete.
