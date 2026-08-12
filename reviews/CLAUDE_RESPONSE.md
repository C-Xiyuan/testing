# Corrected response to the scientific problem and exploration review

> Original review baseline: `c7003a471659b269ef035e5a88ec71a1d9c05b91`
> Legacy response/result checkpoint: `4cbda50bcaa3844e601e01fe751d3b2489557b7b`
> Corrected after an adversarial audit on 2026-08-12.

The earlier version of this response accepted many of the review's objections
but then promoted exploratory legacy runs into answers they did not support.  In
particular it called P0-1 closed while its own formal rule said underpowered,
treated non-IID cells as independent, inferred single-chain causes without a
causal test and claimed provenance that the manifests did not record.  Those
statements are withdrawn.

[`CURRENT_CLAIM_LEDGER.md`](CURRENT_CLAIM_LEDGER.md) is authoritative.  This
document records the corrected item-by-item response and the work still needed.

## Corrected item-by-item response

| Review item | Current disposition | Evidence retained | Claim withdrawn / work required |
|---|---|---|---|
| **P0-1** end-to-end consistency | **Partially resolved; release-blocking** | Legacy exp10 point estimates did not reproduce exp06's old gaps: direct−MBAR +0.047 and +0.160 pairs. | The equivalence comparisons were underpowered; the corrected relaxation gate fails; chain-concatenated resampling, covariance UQ and provenance are defective.  No equivalence, relaxation exclusion, estimator correctness or single-chain cause is established.  Clean v2 production is required. |
| **P0-2** calibration/common offset | **Partially resolved** | Legacy exp09 leaves reference-draw variation, truncation and omitted prediction uncertainty as plausible contributors; the original 1.01σ is not a calibration statistic. | Field-level direct streams were reused, so the 64 cells are non-IID; the post-hoc variance budget is not joint UQ; provenance is invalid.  “No detectable bias” and “only the error bar was wrong” are withdrawn. |
| **P0-3** counterexample replication | **Partially resolved** | In one fixed cell, six construction clusters gave aligned−null contrasts 9.73–11.28 pairs; cluster mean +10.671, 95% CI [+10.105,+11.238]. | Field streams were shared.  The within-cluster 0.490 and ratio 1.10 are invalid.  Random controls failed their prospective ordering in clusters 4 and 5.  The result is not the target×basis×size/state Gate A matrix. |
| **P0-4** clustered zoo/post-hoc band | **Partially resolved by narrowing** | A two-regime pattern can be described for this fixed, designed, clustered zoo. | No inference to a population of fitted models and no “coarse filter, not selector” policy claim. |
| **P0-5** warning-light gate | **Resolved by withdrawal** | The ratio may be reported descriptively. | It certifies nothing and cannot be reused as a Gate C abstention rule without new held-out calibration. |
| **P1-1** fitted-model inference | **Partially resolved by narrowing** | Same-system feasibility: the calculation can be applied to training-induced error fields. | Not external validation; no ranking inference; EGNN seed comparison remains n=2. |
| **P1-2** evidence status | **Resolved for this corrective branch** | README, results narrative, source-of-record manuscript, generated Markdown, response and specification now separate legacy evidence, code repairs, smoke checks and unrun production. | Preserve one-way TeX→Markdown generation and repeat the citation/result audit for every future evidence change. |
| **Gate A** | **Unanswered** | Legacy exp11 is one fixed-cell partial observation. | Run the predeclared boundary matrix; do not count the current six clusters as the complete gate. |
| **Gate B** | **Unanswered** | None. | Run the multi-seed fitted-model prevalence study before making fitted-MLIP claims. |
| **Gate C** | **Unanswered** | None. | Run cost-matched selection policies with frozen tolerances, regret/failure endpoints and scored abstention before making selection-utility claims. |

## P0-1: what exp10 actually says

The legacy table is retained because its point estimates are useful:

| estimator | 4.25 Å | 5.25 Å |
|---|---:|---:|
| linear response | −3.82 ± 0.07 | +1.81 ± 0.06 |
| MBAR | −3.83 ± 0.08 | +1.81 ± 0.07 |
| direct HMC | −3.78 ± 0.67 | +1.97 ± 0.78 |
| direct Metropolis | −3.01 ± 1.48 | +1.68 ± 2.10 |

These points do not reproduce the old exp06 discrepancy.  That is the ceiling.

The earlier response was internally contradictory: it said “the estimators
agree” while reporting that every predeclared interval was wider than the
±0.5-pair equivalence bound.  Failure to reject a difference is not established
equivalence.  The correct status is underpowered.

The situation is stricter after code audit.  The arm gate used
`gap − 1.96×SE ≤ bound`; a correct upper-confidence gate adds uncertainty.  The
legacy gaps and errors then give upper bounds about 2.07 and 3.39 pairs, both
above 0.5.  Because the prerequisite gate fails, incomplete relaxation remains
live and the confirmatory comparison is not evaluable.  The legacy primary
analysis also discarded 90% of each chain, retained only 150 frames, concatenated
chains for reweighting/MBAR resampling and used an invalid covariance correction.

The opposite directions of the two old discrepancies are evidence against one
simple directional-bias story.  They do not establish that both were
single-realisation excursions, nor identify which legacy measurement was
“wrong”.

The exp10 v2 code repairs the gate, distinguishes equivalence from meaningful
difference and inconclusive results, preserves raw configurations and chain
units, uses joint complete-chain uncertainty, isolates the HMC--MBAR primary
family, and records launch/end provenance. A quick smoke completed and failed
closed; it is explicitly `smoke_only`. **No v2 production run has completed.**

## P0-2: what exp09 actually says

The legacy exp09 table changes sign across reference chains and its decomposition
changes when the second-order term and omitted prediction uncertainty are added.
That is useful exploratory evidence that the original all-negative residuals
were not eight independent calibration successes.

It is not a clean factorial replication.  Direct seeds depended on chain index
but not field, so every field reused the same four direct streams.  Treating 64
cells as independent is pseudo-replication; the grand mean and field component
are entangled with chain-index effects.  Post-hoc claims that this necessarily
makes a test conservative do not replace propagation of the paired covariance.
The manifest also records end-state source, not launch source.

Accordingly, the following previous conclusions are withdrawn:

- “the offset is a reference realisation, not a bias” as a complete diagnosis;
- “the second-order term removes the field structure” as confirmation;
- “the estimator shows no detectable bias”;
- “what was wrong was the error bar, not the estimator”.

The retained statement is that reference draw, truncation and missing prediction
uncertainty are all plausible contributors and require a properly independent,
jointly analysed rerun.  `exp09_calibration_replication_v2` repairs field streams,
unit-of-analysis handling and provenance; production has not been rerun.

## P0-3: what exp11 actually says

With construction cluster as the top-level unit, the six legacy contrasts are
10.88, 10.69, 11.01, 11.28, 10.44 and 9.73 pairs.  Their t interval, [10.105,
11.238], lies above the specified one-pair effect.  This supports a narrow
fixed-cell stability observation.

The earlier response went further than the design permits.  Null, aligned and
random fields reused the same direct seed at a given chain index, but their
standard errors were added as if independent.  Therefore the stated
within-cluster SD 0.490, between/within ratio 1.10 and claim that changing the
construction barely matters more than rerunning direct chains are withdrawn.
The random control was prospectively expected to lie between null and aligned;
it failed in clusters 4 and 5, so “controls behaved as designed” is also
withdrawn.  The aligned/null suppression ratio has a near-zero denominator and
spans 16–467×; neither a single 360× nor median 26× is release evidence.

The exp11 v2 code makes paired seed/start blocks explicit, computes paired-chain
uncertainty, reports control-order failures, gates on force matching and held-out
covariance-nullness, uses tri-state effect interpretation, and records raw
configurations plus launch/end provenance. Its quick smoke failed both
manipulation gates and remained `smoke_only`. Production has not been rerun.
Even a successful v2 rerun of this same cell remains only part of Gate A.

## Exact N=2 calculation

The exact quadrature caught a genuine error-bar bug in `reweight()`, and that
unit-level finding is retained.  Its scientific interpretation was overstated.
The N=2 run differs from exp06/low-density work in temperature, cutoff, bins and
perturbation shape, and all amplitudes reuse one reference chain.  It is an
unmatched V1 sanity check, not evidence that the legacy 4.7σ residual is caused
by `O(ρ)`.  Nor does the grid locate a breakdown at βσ≈0.21: it only brackets a
10% bias crossing between 0.052 and 0.209 in that N=2 setup.

## Provenance correction

The previous response claimed launch-time commit and digest provenance for the
confirmatory experiments.  That is false.
Legacy manifests read commit/digest state when the run finished.  Exp10, for
example, began before the recorded commit existed and may have imported code
that was changed during the run.  End-state cleanliness cannot repair this.

The common v2 harness captures source state before creating the output directory,
writes an in-progress launch record, captures end state separately, detects
source changes, rejects dirty production starts and hashes artifacts.  These are
code safeguards, not retroactive validation.  Only new production artefacts can
carry valid provenance.

## Current bounded conclusion

> In one Lennard-Jones liquid state, for one pair-count target and one radial
> basis, legacy artefacts report a large cluster-level
> aligned-minus-null point contrast while force RMSE differs by 0.52%. The paired
> contrast uncertainty is unavailable. The result is provisional
> until a clean v2 rerun, remains fixed-cell even then, and does not establish
> prevalence in fitted models,
> transfer across targets/bases/sizes/states, calibrated end-to-end prediction
> or selection utility.

The manuscript is not submission-ready.  P0-1 remains release-blocking; Gate A,
Gate B and Gate C remain unanswered.  Corrected code must not be described as a
completed experiment.
