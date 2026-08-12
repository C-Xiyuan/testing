# Figure evidence status

All current panels in this directory are derived from legacy exploratory
deposits. No v3 production result or v3 figure exists. A polished graphic is
not evidence by itself. In particular, the following regenerated panels remain
**descriptive legacy artifacts**:

- `headline_regimes`: a fixed, clustered 34-field zoo with shared trajectories;
  its highlighted 15-field window was selected after inspection. Whiskers are
  deposited Monte Carlo noise scales, and the six noise-dominated rows are
  flagged. Row-resampling intervals are sensitivity summaries, not population
  confidence intervals.
- `headline_mechanism` and `headline_prediction`: eight designed-field rows from
  one construction/reference setup. Prediction and measurement whiskers are
  marginal standard errors; shared-offset and paired covariance uncertainty was
  not deposited. The old rms-residual-in-sigma calibration headline is removed.
- `exp05_proxy_correlation_proxy_correlation`: correlations across the same
  fixed, clustered 34-member zoo. Every bracket and forest-plot whisker is the
  2.5--97.5% percentile range from resampling zoo rows. It measures sensitivity
  to membership of this deposited panel; it is not a confidence interval for a
  model population and cannot support prevalence or selector claims.
- `exp06_response_validation_response_validation`: one amplitude and width
  sweep. Reweighting has no deposited interval. The width panel has no norm-SE
  whiskers because `direct_norm_error` is the norm of per-bin standard errors,
  not the standard error of the plotted norm. Only widths 0.10--0.40 Å satisfy
  `r0 + 3w < r_on`; 0.65 and 1.00 Å are marked switch-affected.
- `exp07_designed_counterexamples_counterexamples`: one legacy construction
  cluster with invalid launch provenance and shared streams. The shaded
  3.4--3.9 Å target and its separately deposited measurement are now visible;
  the curve's nearby 3.5--4.0 Å bin is not substituted for that target.

Each regenerated figure is exported as PNG, PDF and editable-text SVG. Scientific
claims must follow the current claim ledger and future corrected production
results, not these legacy graphics.

`review_response.{png,pdf}` is an orphaned historical review graphic. It is not
referenced by the manuscript, was not regenerated as v3 evidence, and must not
be used to imply that exp09--11 production has run.
