# Corrective research roadmap

**Status:** prospective plan, not evidence.  This document converts the audit in
`SCIENTIFIC_PROBLEM_AND_EXPLORATION_REVIEW.md` into executable gates.  A failed
gate narrows the claim; it is not permission to change the threshold or select a
different window after seeing the data.

## The problem the current study has not solved

The established empirical observation is not merely that force RMSE and a
downstream observable can disagree.  The open practical problem is:

> With no everywhere-evaluable oracle, a finite reference budget, correlated
> trajectories and potentially shared model bias, can a cheap diagnostic give
> calibrated, observable-specific risk and improve a prospective model-selection
> decision relative to force RMSE?

The present work answers only a necessary subquestion: one fixed Lennard--Jones
cell admits a constructed pair-radial null/aligned contrast at matched force
RMSE.  It has not established cross-condition stability, prevalence among fitted
errors, no-oracle calibration or selection utility.

## Release blockers before new scientific claims

The corrected `exp09`, `exp10` and `exp11` v2 pipelines must be run from one
frozen, clean commit with master seed `20260812`.  Smoke runs, legacy seed 0,
missing raw arrays, changed source during execution, failed stationarity, failed
overlap or a missing manifest make an output non-evidential.

### P0-1: end-to-end identity

- Independent reference and surrogate chains are the outer resampling units;
  blocks never cross chain boundaries.
- HMC from both initial ensembles, a secondary independent Metropolis arm,
  forward and reverse FEP, BAR/MBAR and direct differences are retained.
- The frozen primary family contains only HMC direct minus MBAR at the two bins;
  Metropolis and linear-response comparisons form a separate sensitivity family
  and cannot change the primary disposition.
- Primary endpoint: direct minus MBAR in the 4.25 and 5.25 Å bins.
- Practical equivalence region: ±0.5 pairs.  Equivalence requires the complete
  family-wise TOST interval inside the region; meaningful difference requires a
  simultaneous two-sided interval wholly outside it; otherwise the result is
  inconclusive.
- Stationarity and chain-aware overlap gates are evaluated first.  Failure makes
  the estimator comparison `not_evaluable`.
- The old 35% discrepancy is closed only if every primary comparison is
  evaluable and equivalent.  A small point difference with a wide interval does
  not close it.

### P0-2: conditional calibration

- The fixed eight-field panel is not a population of fields.  Reference chains
  and field-specific direct chains are resampled at their actual levels while
  reference mean, first order and second order are recomputed on the same block
  draws.
- Primary outputs are per-field residual intervals, intercept/slope intervals
  and coverage against a prospectively justified tolerance in pairs.
- With one construction panel and no justified tolerance, v2 must report
  `exploratory_conditional`, never `calibrated`.

### P0-3: fixed-cell construction stability

- Eight independent construction clusters, four paired seed/start blocks for
  aligned/null per cluster, independent upstream equilibration and a
  block-bootstrap force-match interval wholly inside the tolerance.
- A held-out covariance-null manipulation interval must also pass; a direct
  contrast alone cannot certify that the intended null/aligned mechanism
  transported out of sample.
- The cluster is the inferential unit. Equal HMC seeds do not guarantee exact
  common-random-number synchrony because acceptance branching can consume the
  stream differently; the declared seed/start blocks are analysed as paired
  differences without claiming proposal-level CRN.
- Passing this run supports only “not unique to one construction seed in this
  fixed cell.”  It does not pass Gate A.

## Gate A — mechanism boundary, not just another plot

### A1: pair-radial boundary matrix

- Cells: three predeclared Lennard--Jones state points; at the current state,
  `N={108,256}`.
- Targets: primary pair count, full RDF-vector norm and pressure.
- Bases: the current Gaussian shell basis and one independently specified radial
  basis.
- Replication: at least eight independent construction/evaluation clusters per
  cell and four independent direct chains per field.
- Primary matched-pair criterion: the complete force-RMSE-ratio interval lies
  within ±1%; at least 80% of predeclared cells have aligned-minus-null in the
  expected direction and above a predeclared practical margin, with a
  hierarchical interval excluding zero.
- Failure disposition: restrict the paper to the cells that pass; no universal
  projection diagnostic.

### A2: angular and many-body error geometry

- SW silicon at two temperatures: pair, angular and mixed error bases; RDF,
  angle distribution, tetrahedral order and pressure.
- EAM copper at two temperatures: pair, density, embedding and mixed bases;
  RDF, bond order and pressure.
- Each basis family receives independent construction clusters and random
  directions.  A successful pair arm cannot stand in for a failed angular or
  embedding arm.
- Failure disposition: title and conclusions must say “pair-radial constructed
  errors only.”

Dynamic observables form a separate scientific question because a static
covariance identity does not describe the perturbed propagator.  Diffusion,
VACF, VDOS or phonons must therefore be preregistered and analysed separately,
not appended as if Gate A automatically covered them.

## Gate B — bridge from designed fields to real fitted errors

### B1: analytic-oracle fitted bridge

- SW silicon and EAM copper retain an exact oracle while replacing hand-designed
  fields with training-induced errors.
- At minimum: four model families, two data budgets and eight independent
  training/data-order seeds per family.  Training, test, reference and direct
  chains are disjoint.
- Models selected for expensive direct validation are fixed before direct
  observable results are read.
- Primary calibration endpoints: held-out intercept/slope, residual coverage and
  false-accept rate at predeclared observable tolerances, clustered by training
  dataset/seed.
- Failure disposition: the fixed-cell legacy construction remains a provisional
  observation that does not explain the real fitted-error manifold.

### B2: one deep DFT system

- Use one frozen electronic-structure protocol and at least solid and liquid
  states of the same chemistry.  Reference chains stop on effective-sample-size
  targets, not raw frame counts.
- Compare controlled retraining of a local linear model, a local neural model
  and an equivariant model on the same labels; published checkpoints trained on
  incompatible reference levels are a secondary arm only.
- All observables, tolerances, force-error bands and model-selection rules are
  frozen before direct model trajectories are inspected.
- Failure disposition: any success is state/phase/reference-level conditional.

## Gate C — prospective decision utility

Correlation is not selection utility.  Freeze and blind-test at least these
policies:

1. minimum held-out force RMSE;
2. a fixed energy-plus-force baseline;
3. observable-response risk, minimising the maximum normalised predicted error;
4. random choice and oracle best as lower/upper references.

The experimental unit is an independent selection task
(`system × state × training-data split`), not a model seed inside one pool.
For each selected model, truth comes from independent direct chains.  Primary
loss is the maximum endpoint error divided by its predeclared tolerance;
secondary outputs are failure rate, regret, abstention/calibration and a
reference-budget cost curve.

Gate C passes only if the response policy reduces both regret and failure rate
relative to force RMSE with task-level paired intervals excluding zero, without
materially worsening any primary system/observable.  If prediction calibration
improves but decision regret does not, the method is an auditing diagnostic, not
a selector.

## Stop/go order

1. Run corrected P0-1/P0-2/P0-3 pipelines and publish raw evidence.
2. Run A1.  Stop expansion if the fixed-cell mechanism is not stable across its
   nearest state/size/target perturbations.
3. Run A2 and B1.  Stop any practical-selector language if fitted errors fail.
4. Commit DFT resources only after B1 passes.
5. Run Gate C only after an observable predictor has held-out calibration and a
   fail-closed overlap/stationarity policy.

At every stage, the configuration, estimand, hierarchy, decision rule and
failure wording must be committed before production data are generated.
