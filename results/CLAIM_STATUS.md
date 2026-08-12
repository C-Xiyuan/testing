# Result artefact status

Result directories in this tree are not automatically claim-bearing.
[`../reviews/CURRENT_CLAIM_LEDGER.md`](../reviews/CURRENT_CLAIM_LEDGER.md) is the
authoritative disposition.

- `exp09_calibration_replication/`, `exp10_endtoend_consistency/` and
  `exp11_counterexample_replication/` are legacy exploratory artefacts.  Their
  manifests do not prove start-of-run source, and each has design or uncertainty
  defects described in the ledger.
- v2 code repairs exist for `exp09_calibration_replication_v2`,
  `exp10_endtoend_consistency_v2` and
  `exp11_counterexample_replication_v2`, but **no v2 production result is
  present or claim-bearing**.
- `validation/two_particle_exact.json` is an unmatched N=2 implementation sanity
  check, not a causal explanation of the low-density residual and not a
  many-body calibration.

Do not copy a legacy number into a manuscript without its claim ceiling and
independent-unit/provenance qualification.
