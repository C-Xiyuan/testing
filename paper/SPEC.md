# Audited manuscript specification

**Status:** active corrective specification, 2026-08-12.

This file replaces the legacy authoring brief. The legacy brief required claims
that later audit showed were unsupported; it remains available in Git history.
When documents conflict, the order of authority is:

1. `reviews/CURRENT_CLAIM_LEDGER.md`;
2. `reviews/SCIENTIFIC_PROBLEM_AND_EXPLORATION_REVIEW.md`;
3. `reviews/CORRECTIVE_RESEARCH_ROADMAP.md`;
4. this specification;
5. manuscript prose and historical result summaries.

## Scientific problem

The unresolved practical problem is not whether force RMSE can fail in
principle. It is whether, under finite reference data and without an oracle, an
observable-specific diagnostic can make a calibrated, cheaper, and better
model-selection decision than force RMSE for a declared system, state point,
and utility function.

The current repository has not solved that problem. Its strongest numerical
artifact is a provisional legacy observation in one LJ liquid cell: within a
fixed pair-radial basis and one target pair-count bin, fields separated by 0.52%
in force RMSE have different reported direct observable shifts. Its paired
uncertainty is unavailable and clean v3 replication is pending. This does not establish the
frequency of such fields among fitted MLIPs, a general selector, or transfer to
other systems, observables, sizes, phases, or dynamics.

## Evidence rules

- Every number must resolve to a tracked result artifact plus a valid manifest,
  or be labelled legacy/descriptive and non-release.
- A production manifest is valid only when source and protocol are tracked,
  clean at launch, unchanged through completion, and artifacts belong to that
  run. Quick/smoke output is never scientific evidence.
- The independent experimental unit must be named. Frames, amplitude scalings,
  fields sharing one construction trajectory, and rows sharing one reference
  chain are not independent replicates.
- Shared-data covariance must be propagated jointly. Independent quadrature is
  permitted only for genuinely independent units.
- Post-hoc windows and fixed designed zoos may be described conditionally but
  cannot support population confidence intervals or selector claims.
- Failure to reject a difference is not equivalence. Equivalence requires the
  complete declared interval inside a predeclared practical margin; meaningful
  difference requires it wholly outside that margin; overlap is inconclusive.
- A diagnostic threshold is not a trust certificate until sensitivity,
  specificity, false-trust rate, and coverage are calibrated on held-out units.
- Null, negative, and underpowered outcomes must remain distinct.

## Current permitted claims

1. The first-order static response identity is
   `-beta Cov_0(A, delta U)`; it is standard theory, not a novelty claim.
2. Force RMSE and observable response interrogate different geometry. Any
   Poincare-style relationship is ensemble- and system-dependent; the manuscript
   must state gauge and overlap conditions rather than claim there is only a
   generic loose inequality.
3. The deposited fixed-cell construction is a provisional legacy observation,
   not a formal sufficiency, exact-equivalence, prevalence, or model-selection
   result. It may be described only with its provenance and paired-UQ caveats.
4. The fixed designed zoo supplies descriptive sensitivity only. Its old 30x
   spread is floor-sensitive and its member bootstrap has no population
   estimand.
5. The fitted-model arm is same-system feasibility. With ten models, shared
   data, and ranking intervals crossing zero, it is not external validation.
6. The second-order ratio and weight-concentration checks are unvalidated
   screens. They may fail closed but may never certify a result.

## Corrective experiments

- **exp09 v3:** fixed-panel conditional calibration with independent starts,
  field-specific streams, complete-chain crossed bootstrap, stationarity checks, raw
  coordinates, and no field-population claim.
- **exp10 v2:** HMC-direct minus MBAR at two endpoints is the sole primary
  family; Metropolis/linear comparisons are sensitivity analyses. Complete-chain
  percentile bootstrap, relaxation, stationarity, and overlap gates fail closed.
- **exp11 v3:** construction cluster is the unit; force matching and held-out
  covariance-nullness are manipulation gates; direct chains use declared
  paired seed/start blocks; the conclusion is limited to the fixed cell.

Until clean production runs exist, these are corrected protocols and code, not
new scientific results.

## Expansion gates

- **Gate A:** mechanism stability across predeclared states, sizes, targets,
  bases, and angular/many-body error families.
- **Gate B:** transfer to training-induced errors and at least one consistent
  DFT-reference/modern-MLIP setting.
- **Gate C:** blinded no-oracle decision utility, evaluated by failure rate and
  regret at fixed reference cost against force and energy/force baselines.

Claims must contract when a gate fails. Only Gate C can license language that a
response diagnostic improves real model selection.

## Manuscript and bibliography

`paper/main.tex` is authoritative; `paper/main.md` must be regenerated with
`paper/tex2md.py` and show no diff on a second generation. Citations are not
limited to a historical allow-list. Every added citation must be checked against
the primary source, and qualitative compatibility must not be written as an
identical empirical regime.
