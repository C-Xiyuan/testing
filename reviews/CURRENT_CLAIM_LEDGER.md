# Current claim ledger

**Status date:** 2026-08-13

**Audit baseline:** `c7003a471659b269ef035e5a88ec71a1d9c05b91`

**Legacy-result checkpoint:** `4cbda50bcaa3844e601e01fe751d3b2489557b7b`

This is the repository's authoritative scientific-status document.  If the
README, results narrative, manuscript or either response letter makes a stronger
claim, this ledger wins.  The legacy numerical artefacts are retained for audit,
not promoted to confirmatory evidence.

## Status vocabulary

- **Resolved:** the requested evidential issue has a defensible disposition.
- **Partially resolved:** useful evidence exists, but the decisive design or
  uncertainty requirement is unmet.
- **Unanswered:** the required experiment has not been run.
- **Regressed:** a later change weakened the evidence or reintroduced a defect.
- **Contradicted:** the available evidence directly conflicts with the claim.

“Code repaired” below means only that the identified implementation change is
present on the current worktree.  It is not a scientific result and does not
assert that a production experiment passed. No corrected production run has yet
supplied claim-bearing data. Exp09 and exp11 v2 were superseded before production
when the arbitrary SVD sign of the aligned construction was identified; their
frozen successors are v3. Exp10 remains v2.

## Cumulative dispositions

| Item | Disposition | Strongest warranted statement | What remains |
|---|---|---|---|
| **P0-1** end-to-end consistency | **Partially resolved; release-blocking** | Legacy exp10 point estimates did not reproduce the old exp06 gaps: direct−MBAR was +0.047 and +0.160 pairs in the two bins. | The formal comparisons were underpowered, the corrected relaxation gate fails, chain units were lost in resampling, covariance propagation was wrong and provenance is invalid.  No equivalence, relaxation exclusion or single-chain cause is established. |
| **P0-2** calibration/common offset | **Partially resolved** | Legacy exp09 leaves reference-draw variation, truncation and omitted prediction uncertainty as plausible contributors.  The original 1.01σ is not a calibration statistic. | Direct streams were reused across fields, the cells are non-IID, joint chain-level UQ is absent and provenance is invalid.  No “unbiased estimator” or “only the error bar was wrong” claim is licensed. |
| **P0-3** counterexample replication | **Partially resolved** | In one fixed LJ cell, six construction clusters gave aligned−null contrasts 9.73–11.28 pairs; cluster-level mean 10.671, 95% CI [10.105, 11.238]. | Field-level streams were shared, the published within/between ratio is invalid, two random controls violated the prospective ordering and provenance is invalid.  This is not Gate A. |
| **P0-4** clustered zoo/post-hoc band | **Partially resolved by claim restriction** | The two-regime pattern may be described only for the fixed, designed, clustered zoo, with uncertainty and post-hoc window selection disclosed. | No population inference about fitted models or model-selection utility follows. |
| **P0-5** warning-light gate | **Resolved by withdrawal** | The second/first-order ratio may be descriptive. | It must not certify predictions or be reused as an abstention gate without a new held-out calibration study. |
| **P1-1** fitted-model inference | **Partially resolved by claim restriction** | The fitted-model arm is a same-system feasibility observation: the calculation can be applied to training-induced error fields. | It is not external validation, supports no ranking claim and the EGNN seed pair remains n=2. |
| **P1-2** evidence-status consistency | **Partially resolved; release-blocking truth-surface audit remains** | The corrective texts distinguish legacy evidence, repaired code, unretained smoke checks and unrun production. | Regenerated figures, manuscript word count, citations and all numerical statements must pass one final contradiction/provenance audit; code repair alone cannot close this item. |
| **Gate A** boundary/replication matrix | **Unanswered** | Legacy exp11 supplies only one fixed-cell pilot-like cluster result. | Required target × basis × size/state matrix and angular/many-body boundary tests have not run. |
| **Gate B** fitted-error prevalence | **Unanswered** | None. | Predeclared multi-seed, representation-diverse fitted-model campaign has not run. |
| **Gate C** selection utility | **Unanswered** | None. | Frozen policies, cost-matched candidate selection, regret/failure endpoints and honest abstention have not run. |

## Legacy artefacts and corrected-pipeline state

| Experiment | Legacy artefact ceiling | Repair status | Production status |
|---|---|---|---|
| `exp09_calibration_replication` | Exploratory non-IID variance decomposition only. | V3 uses field-specific streams/starts, a deterministic observable-space aligned sign, fixed-panel complete-chain bootstrap, raw configurations, stationarity and launch/end provenance. V2 was superseded before production; no smoke artifact is retained as evidence. | **V3 production not run.** No tracked `results/exp09_calibration_replication_v3/` production evidence. |
| `exp10_endtoend_consistency` | Point non-recurrence only; formal endpoint not evaluable. | v2 primary family is HMC-direct minus MBAR at two endpoints; complete-chain percentile UQ, relaxation/stationarity/overlap gates, independent upstream replicas, raw configurations and launch/end provenance are implemented. Any earlier smoke output was non-evidence and is not retained. | **Production not rerun.** No tracked `results/exp10_endtoend_consistency_v2/` production evidence. |
| `exp11_counterexample_replication` | Fixed-cell cluster-level contrast only. | V3 uses construction cluster as unit, deterministic aligned sign, paired seed/start blocks (not proposal-level exact CRN), force-match and held-out covariance-null gates, tri-state interpretation, raw configurations and launch/end provenance. V2 was superseded before production; no smoke artifact is retained as evidence. | **V3 production not run.** No tracked `results/exp11_counterexample_replication_v3/` production evidence. |

The common provenance harness now records launch and finish source states,
detects source changes, rejects dirty production starts and inventories output
hashes.  Legacy manifests cannot be repaired retroactively because they sampled
repository state only at the end of a run.

## Manuscript synchronization direction

`paper/main.tex` is the submission source of record.  `paper/main.md` is a
generated reading copy and must be regenerated with `paper/tex2md.py`, not
edited as an independent manuscript variant.  `paper/SPEC.md` is now labelled a
superseded legacy authoring brief; its older mandatory wording must not mark any
P0 item or Gate A/B/C complete merely because corrected code exists.
`paper/RESPONSE.md` now redirects to the audited response, and `paper/SPEC.md`
has been replaced by a short active corrective specification; their historical
overclaims remain available only in Git history. Citation metadata must still be
checked whenever references change.

## N=2 exact-quadrature check

`results/validation/two_particle_exact.json` is an implementation sanity check,
not a causal resolution of the low-density residual.  Its N, temperature,
cutoff, bin range and perturbation shape do not match exp06 or the low-density
calculation, and all amplitudes reuse one reference chain.  It therefore does
not prove that the 4.7σ residual is an `O(ρ)` effect, establish many-body
coverage, or locate a breakdown threshold at βσ≈0.21.  The sampled grid merely
brackets a 10% linear-bias crossing between βσ=0.052 and 0.209 in its own setup.

## Current scientific conclusion

The strongest current numerical artifact is a **provisional, fixed-cell legacy
observation**: in one LJ state, for one pair-count target and one radial basis,
legacy artefacts report aligned and null fields separated by 0.52% in force RMSE
and 10.11 pairs in target point estimates; paired uncertainty is unavailable.
The work does not establish prevalence
among fitted MLIPs, generalisation across states/bases/sizes, calibrated
end-to-end prediction or selection utility.

## Release gates

Do not describe the repository as submission-ready until all of the following
are true:

1. clean exp09/exp11 v3 and exp10 v2 production runs pass start/end provenance and deposit raw
   chain-level analysis data;
2. P0-1 has a valid relaxation/stationarity gate and a defensible equivalence or
   meaningfully-different disposition;
3. P0-2 and P0-3 are recomputed with their true independent/paired units;
4. Gate A has a defensible boundary disposition;
5. new versioned replacements for legacy exp03/05/06/07 are run from a frozen
   signed/tagged revision, or all numerical claims and figures depending on
   them are removed from submission;
6. every manuscript truth surface is regenerated from this claim ceiling and
   passes a contradiction scan.

Gates B and C are required for claims about fitted-model prevalence and
selection utility respectively; if they remain unrun, those claims must remain
absent rather than be described as future validation of an already established
method.
