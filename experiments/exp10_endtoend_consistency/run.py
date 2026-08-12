#!/usr/bin/env python3
"""exp10 -- the release blocker: do the reference-based estimators and direct MD agree?

Review item P0-1. `results/exp06_response_validation/amplitude_records.json`
contains a disagreement the manuscript did not resolve. At the smallest
amplitude, where the second-order ratio is 0.006 and linear response should be
exact for practical purposes, the two reference-based estimators agree with each
other to 1% -- linear -3.835, reweighted -3.884 pairs -- and direct sampling
says -1.948 +/- 0.821. Two estimators built from the reference ensemble agree,
and the ensemble they claim to describe does not.

Either the reference-based prediction is wrong, in which case the central claim
of this repository fails, or the direct measurement is wrong, in which case
every "measured" column in the manuscript is suspect. Both possibilities are
worse than the missing experiment, so this is the experiment.

Two candidate explanations survive a look at the raw curves, and they point in
opposite directions, which is why this needs an experiment rather than an
argument.

*Incomplete relaxation.* In the reported bin the direct shift is smaller in
magnitude than the linear prediction at every one of the seven amplitudes --
-1.95 against -3.84, -6.61 against -7.67, -12.64 against -15.34, and so on. A
random measurement error does not have a sign. A direct chain that has not
finished relaxing into the perturbed ensemble does: it still partly reports the
reference ensemble it started from, biasing every measured shift toward zero.
exp06 started its surrogate chains from a reference-equilibrated configuration,
discarded 300 moves, and ran no stationarity check on them at all.

*A single-chain realisation.* Against that, the discrepancy does not have a
consistent sign across bins. `scripts/check_between_chain_scatter.py` measured
the same perturbation at 5.25 A with six independent chains on each side and
found the direct shift *larger* than the prediction, +2.377 +/- 0.144 against
+1.725 +/- 0.101. Under-relaxation cannot make one bin under-shift and another
over-shift. And exp06's smallest-amplitude row is one chain against one chain,
with several bins the perturbation barely touches also displaced by two to four
standard errors -- the signature of one chain's slow modes, not of an estimator.

There is a third possibility neither the manuscript nor the review priced in:
the +1.725 prediction carries a within-chain blocking error from a *single*
reference chain, so its quoted +/-0.101 says nothing about how much the
prediction itself moves between reference realisations. If that scatter is
comparable to the discrepancy, the 3.7 sigma is arithmetic rather than physics.
This experiment measures it directly by predicting from each reference chain
separately.

Distinguishing these requires measurement, not argument:

  1. **Bracket the answer from both sides.** Half the surrogate chains start
     from configurations equilibrated under the *reference*, and half from
     configurations equilibrated under the *surrogate*. Incomplete relaxation
     biases the first toward zero shift and the second toward the full shift, so
     the truth is between them. When the two arms meet, relaxation is complete;
     where they have not met, the gap is the size of the artefact.
  2. **Resolve it in time.** Both arms are measured over a ladder of discarded
     prefixes from the same chains, which costs nothing extra and turns "is it
     equilibrated" into a curve rather than a verdict.
  3. **Use an estimator that can see the other ensemble.** Forward reweighting
     from the reference is variance-limited by a tail the reference barely
     samples. With surrogate chains in hand, BAR and two-state MBAR
     (`atomlab.analysis.fep`) use both directions, and their overlap
     diagnostics say whether the answer deserves belief.
  4. **Change the sampler.** Everything in exp06 came from one HMC
     implementation. A single-particle Metropolis arm shares no propagation
     machinery, so agreement between them is evidence and disagreement localises
     the fault.
  5. **Change the observable.** Two bins, 4.25 A and 5.25 A -- the bin exp06
     reported and the bin the six-chain follow-up used -- since the two gave
     discrepancies of opposite sign and a single experiment should cover both.
  6. **Propagate the prediction's own chain-to-chain uncertainty.** The linear
     prediction is computed separately from each reference chain, so its
     between-chain scatter is measured rather than assumed to be the blocking
     error of one chain.

Prospectively specified for the corrected v2 run
-------------------------------------------------
*Estimand.* For each bin k, D_k = <A_k>_U - <A_k>_0, the shift in the mean
occupancy of that pair-distance bin between the reference and surrogate
canonical ensembles at fixed N, V, T.

*Primary comparison.* The shared-data difference between the HMC direct estimate
at the frozen 50% discard and the MBAR estimate, recomputed in one joint
complete-chain bootstrap, per bin.

*Equivalence bound.* +/- 0.5 pairs. The contrast this measurement has to be able
to support is the aligned-null difference of 10.1 pairs in exp07; a systematic
error of 0.5 pairs is 5% of that, below the point where any conclusion in the
manuscript would change. Chosen from that downstream requirement, not from the
observed scatter.

*Decision rule.* The primary family is HMC direct sampling minus MBAR at the two
bins. Consistency is declared only if both family-wise equivalence intervals are
contained in the bound. Metropolis and linear-response comparisons are separate
sensitivity analyses: they can flag a problem but cannot change the primary
disposition. If an interval overlaps the bound the result is inconclusive, not
consistent and not discrepant; meaningful difference requires the complete
simultaneous interval to lie outside the bound.

*Chain count.* Eight reference and eight surrogate HMC chains per arm. The exp06
pilot gives a per-chain standard deviation of about 0.8 pairs on this
observable, so eight chains give a standard error near 0.28 and a paired
interval near +/- 0.8 pairs -- wider than the equivalence bound. Thus the run
may honestly remain inconclusive. The relaxation ladder is a diagnostic and
gate, not a substitute for an adequately precise primary interval.

*Multiplicity.* The primary family contains two endpoints (HMC--MBAR). The six
remaining sampler/estimator/end-point combinations form a secondary sensitivity
family. Each family has its own Bonferroni-adjusted intervals alongside nominal
95% intervals.

*What was decided after seeing data, and what it was.* A smoke-scale run
(100 frames per chain, far below production) was used to debug the plumbing, and
two design decisions were taken after seeing it, so they are recorded here
rather than presented as prior:

  - the primary direct arm is the one started from surrogate-equilibrated
    configurations, on the principle that a chain beginning in the ensemble it
    is measuring carries no relaxation transient; the other arm is reported
    beside it and the difference between them is the artefact;
  - a convergence gate was added: if the two initialisations have not met to
    within the equivalence bound at the frozen primary discard, no comparison against
    the direct estimate is reported as estimator disagreement, because an
    unrelaxed chain is not a measurement of the ensemble in question.

Both make the test harder to pass, not easier, and neither depends on which way
the answer comes out. The smoke run's numbers are not used anywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.fep import (
    bennett_acceptance_ratio,
    mbar_two_state_average,
    overlap_diagnostics,
)
from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis, integrated_autocorrelation_time
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo, metropolis_nvt
from atomlab.units import beta as inverse_temperature
from experiments.common import (
    ExperimentContext,
    main,
    require_fresh_production_seed,
    require_frozen_protocol,
)
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

EXPERIMENT_NAME = "exp10_endtoend_consistency_v2"
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "protocols" / "exp10_v2.json"

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    # exp06's bin edges, so bin 2 is 4.0-4.5 A (centre 4.25) and bin 4 is
    # 5.0-5.5 A (centre 5.25) -- the two the review asks for.
    "observable": {"r_min": 3.0, "r_max": 7.0, "n_bins": 8,
                   "report_bins": [2, 4]},
    # The disputed point: exp06's smallest amplitude, where linear response is
    # least in doubt and the disagreement is therefore most damaging.
    "field": {"r0": 4.2, "width": 0.35, "amplitude": 0.0005},
    # Chain lengths are set by the wall-clock available rather than by what
    # would be ideal, and the consequence is visible in the output: the
    # decision rule reports "underpowered" when the interval is wider than the
    # bound, which is the honest outcome of a short chain rather than a defect
    # to be hidden. Sixteen surrogate chains at 1500 frames is the budget.
    "sampling": {"n_reference": 1500, "n_surrogate": 1500, "n_presoak": 2000,
                 "n_leapfrog": 8, "step_size": 2e-3, "burn_in": 300,
                 "n_melt": 400, "n_anneal": 1000},
    "replication": {"n_reference_chains": 8, "n_surrogate_chains": 8},
    # A Metropolis sweep costs a full energy evaluation per attempted move and
    # is about thirteen times an HMC move here, so this arm is sized as a
    # cross-check rather than a second primary measurement: enough to say
    # whether the HMC answer is a property of the ensemble or of the sampler,
    # not enough to resolve the equivalence bound. Its interval is reported and
    # will read "underpowered" if it is wider than the bound, which is the
    # correct outcome rather than a defect.
    "metropolis": {"n_chains": 4, "n_sweeps": 350, "burn_in": 150,
                   "max_displacement": 0.12},
    # Fractions of each surrogate chain discarded before averaging.
    "relaxation": {"discard_fractions": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]},
    "analysis": {
        "equivalence_bound_pairs": 0.5,
        "familywise_alpha": 0.05,
        # Frozen for the v2 rerun before new chains are generated. The legacy
        # run showed that 90% discard leaves only 150 frames/chain; 50% retains
        # enough data for chain-level inference while remaining a prospective
        # choice for the independent rerun. It must not be used to relabel the
        # legacy data confirmatory.
        "primary_discard_fraction": 0.50,
        # Family-wise percentile tails at alpha/(2*8) need substantially more
        # than the legacy 400 draws (which left roughly one draw per tail).
        "n_resamples": 5000,
        # Operational fail-closed overlap guard added after the first production
        # run exposed that diagnostics were reported but never used.  These are
        # future-run safety thresholds, not retrospectively preregistered tests.
        "min_work_overlap": 0.10,
        "min_kish_fraction": 0.10,
        "max_weight_fraction": 0.10,
        "min_autocorrelation_adjusted_ess_fraction": 0.02,
        "max_chain_weight_fraction": 0.35,
        "min_central_work_overlap": 0.10,
        "min_chain_pair_overlap_fraction": 0.50,
    },
}


def simultaneous_z(alpha: float, n_comparisons: int) -> float:
    """Two-sided Bonferroni critical value for a family of comparisons."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be positive")
    return float(norm.ppf(1.0 - alpha / (2.0 * n_comparisons)))


def equivalence_disposition(
    estimate: float,
    standard_error: float,
    bound: float,
    *,
    z: float,
) -> dict:
    """Classify an estimate against a symmetric practical-equivalence region.

    ``equivalent`` requires the complete interval to lie inside ``[-bound,
    bound]``. ``meaningfully_different`` requires the complete interval to lie
    outside that region. Everything else is inconclusive. This prevents the
    common but serious error of treating failure to establish equivalence as
    proof of disagreement (or a noisy estimate as proof of convergence).
    """
    estimate = float(estimate)
    standard_error = float(standard_error)
    bound = float(bound)
    if standard_error < 0.0 or bound <= 0.0 or z <= 0.0:
        raise ValueError("standard_error must be non-negative; bound and z positive")
    half_width = z * standard_error
    ci = (estimate - half_width, estimate + half_width)
    upper_abs = abs(estimate) + half_width
    lower_abs = max(0.0, abs(estimate) - half_width)
    if upper_abs <= bound:
        status = "equivalent"
    elif lower_abs > bound:
        status = "meaningfully_different"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "ci": [float(ci[0]), float(ci[1])],
        "half_width": float(half_width),
        "upper_abs": float(upper_abs),
        "lower_abs": float(lower_abs),
        "precision_insufficient_for_equivalence": bool(half_width > bound),
    }


def interval_disposition(ci, bound: float) -> dict:
    """Classify a directly estimated interval against ``[-bound, bound]``."""
    lo, hi = map(float, ci)
    bound = float(bound)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi and bound > 0.0):
        raise ValueError("ci must be finite and ordered; bound must be positive")
    if lo >= -bound and hi <= bound:
        status = "equivalent"
    elif lo > bound or hi < -bound:
        status = "meaningfully_different"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "ci": [lo, hi],
        "precision_insufficient_for_equivalence": bool((hi - lo) / 2.0 > bound),
    }


def percentile_interval(samples, lower_probability, upper_probability):
    samples = np.asarray(samples, dtype=float)
    if not 0.0 < lower_probability < upper_probability < 1.0:
        raise ValueError("bootstrap probabilities must be ordered inside (0,1)")
    return np.quantile(samples, [lower_probability, upper_probability], axis=0)


def independent_chain_difference_bootstrap(first, second, *, n_resamples, seed):
    """Difference of two independent arm means, with chain as the unit."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("arm arrays must be (chain, endpoint) with matching endpoints")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_resamples)):
        i = rng.integers(0, len(first), size=len(first))
        j = rng.integers(0, len(second), size=len(second))
        out.append(second[j].mean(axis=0) - first[i].mean(axis=0))
    return np.asarray(out, dtype=float)


def overall_disposition(*, arms_converged: bool, stationarity_passed: bool,
                        overlap_passed: bool, all_equivalent: bool,
                        any_meaningful_difference: bool) -> str:
    """Fail closed before interpreting any estimator comparison."""
    if not (arms_converged and stationarity_passed and overlap_passed):
        return "not_evaluable"
    if all_equivalent:
        return "equivalent"
    if any_meaningful_difference:
        return "meaningfully_different"
    return "inconclusive"


def comparison_family(sampler: str, estimator: str) -> str:
    """The frozen primary family is HMC direct sampling versus MBAR only."""
    return "primary" if sampler == "hmc" and estimator == "mbar" else "secondary"


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def tune_step_size(ctx, cfg, potential, seed, *, n_steps=300):
    """Adapt the HMC step size on a throwaway chain, and return it.

    The bracketing arms below must record their chains from the first move, so
    they run with ``burn_in=0`` -- and step-size adaptation only happens during
    burn-in, because an adapting proposal is not a valid MCMC kernel. Without
    this helper those chains would keep the configured 2 fs while every other
    chain in the experiment adapts to roughly 30 fs, sampling the surrogate
    ensemble more than an order of magnitude more slowly than the reference
    ensemble it is compared against. That is not a small inefficiency: the whole
    question is whether the surrogate chains reach stationarity, and answering
    it with a deliberately crippled kernel would answer a different question.

    The tuning chain starts from the same configuration and is discarded; only
    its step size is kept, so the recorded chain is a fixed kernel started at
    the intended point.
    """
    s = ctx.config["sampling"]
    _, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"], n_samples=1,
        n_leapfrog=s["n_leapfrog"], step_size=s["step_size"],
        burn_in=int(n_steps), seed=seed)
    return float(report.final_step_size)


def hmc(ctx, cfg, potential, n_samples, seed, *, burn_in=None, label="", check=True,
        step_size=None):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"] if step_size is None else float(step_size),
        burn_in=s["burn_in"] if burn_in is None else int(burn_in), seed=seed,
        adapt=step_size is None)
    if report.acceptance < 0.2:
        raise RuntimeError(f"{label}: acceptance {report.acceptance:.2f} too low")
    if check:
        check_equilibrated(
            traj, label=label, check_order=True,
            n_sigma=20.0 if ctx.quick else 3.0,
        )
    return traj, report


def chain_values(traj, observable, bins):
    """Per-frame values of the reported bins, shape (n_frames, len(bins))."""
    return observable.evaluate_trajectory(traj)[:, bins]


MIN_FRAMES_FOR_MEAN = 20


def usable_fractions(n_frames, fractions):
    """Which ladder points leave enough frames to average over.

    Under ``--quick`` the chains are short enough that the deepest discards
    would average a handful of frames, so they are dropped rather than reported
    with a meaningless error bar.  Returned explicitly so the caller uses the
    same list everywhere instead of assuming the deepest requested point exists.
    """
    keep = [f for f in fractions
            if n_frames - int(round(f * n_frames)) >= MIN_FRAMES_FOR_MEAN]
    if not keep:
        raise RuntimeError(f"no ladder point leaves {MIN_FRAMES_FOR_MEAN} frames "
                           f"out of {n_frames}")
    return keep


def ladder(values, fractions):
    """Mean of each bin after discarding a leading fraction of the chain."""
    n = values.shape[0]
    return {f: values[int(round(f * n)):].mean(axis=0)
            for f in usable_fractions(n, fractions)}


def _split_series_diagnostic(values, *, n_sigma: float = 3.0) -> dict:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return {"passed": False, "reason": "non-finite values"}
    if values.ndim == 1:
        values = values[:, None]
    half = values.shape[0] // 2
    if half < 8:
        return {"passed": False, "reason": "fewer than 16 frames"}
    z_scores = []
    for index in range(values.shape[1]):
        first = blocking_analysis(values[:half, index])
        second = blocking_analysis(values[half:, index])
        denominator = float(np.hypot(first.error, second.error))
        difference = abs(float(first.value) - float(second.value))
        z_scores.append(difference / denominator if denominator > 0.0
                        else (0.0 if difference == 0.0 else float("inf")))
    return {
        "passed": bool(np.all(np.asarray(z_scores) <= n_sigma)),
        "n_sigma": n_sigma,
        "half_difference_z": z_scores,
    }


def tail_stationarity(trajectory, a, du, discard: float, *, label: str) -> dict:
    """Evaluate energy, endpoint and error-field drift on an analysed tail."""
    start = int(round(discard * trajectory.n_frames))
    tail = trajectory[start:]
    try:
        energy = check_equilibrated(
            tail, label=f"{label} tail after {discard:.0%} discard", check_order=True
        )
    except (RuntimeError, ValueError) as exc:
        return {"passed": False, "error": str(exc), "n_frames": tail.n_frames}
    a_check = _split_series_diagnostic(np.asarray(a)[start:])
    du_check = _split_series_diagnostic(np.asarray(du)[start:])
    return {
        "passed": bool(a_check["passed"] and du_check["passed"]),
        "n_frames": tail.n_frames,
        "energy": energy,
        "observable": a_check,
        "delta_u": du_check,
    }


def _stack_chain_field(chains, key: str) -> np.ndarray:
    arrays = [np.asarray(chain[key]) for chain in chains]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"cannot store ragged raw chain field {key!r}")
    return np.stack(arrays, axis=0)


def _linear_point(a, du, temperature):
    a = np.asarray(a, dtype=float)
    du = np.asarray(du, dtype=float).ravel()
    if a.ndim == 1:
        a = a[:, None]
    return -inverse_temperature(temperature) * np.array([
        np.cov(a[:, index], du, ddof=1)[0, 1]
        for index in range(a.shape[1])
    ])


def _reweight_point(a, du, temperature):
    a = np.asarray(a, dtype=float)
    du = np.asarray(du, dtype=float).ravel()
    if a.ndim == 1:
        a = a[:, None]
    log_w = -inverse_temperature(temperature) * du
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()
    return weights @ a - a.mean(axis=0)


def chain_weight_diagnostics(chains, temperature, *, sign, discard=0.0):
    """Weight concentration with autocorrelation and chain identity retained."""
    beta = inverse_temperature(temperature)
    log_weights, taus, n_frames = [], [], []
    for chain in chains:
        start = int(round(discard * len(chain["du"])))
        du = np.asarray(chain["du"][start:], dtype=float)
        if du.size < 4:
            raise ValueError("each overlap chain needs at least four analysed frames")
        log_w = float(sign) * beta * du
        log_weights.append(log_w)
        n_frames.append(du.size)
    global_max = max(float(values.max()) for values in log_weights)
    weights = [np.exp(values - global_max) for values in log_weights]
    total = float(sum(values.sum() for values in weights))
    chain_shares = [float(values.sum() / total) for values in weights]
    autocorrelation_adjusted_weight_square_sum = 0.0
    for values in weights:
        tau = float(integrated_autocorrelation_time(values))
        taus.append(tau)
        autocorrelation_adjusted_weight_square_sum += (
            max(1.0, 2.0 * tau) * float(np.sum(values**2))
        )
    adjusted_ess = total**2 / autocorrelation_adjusted_weight_square_sum
    return {
        "chain_weight_fractions": chain_shares,
        "max_chain_weight_fraction": max(chain_shares),
        "autocorrelation_times": taus,
        "autocorrelation_adjusted_ess": adjusted_ess,
        "autocorrelation_adjusted_ess_fraction": (
            adjusted_ess / float(sum(n_frames))
        ),
    }


def robust_work_overlap(reference_chains, surrogate_chains, *, discard=0.0,
                        lower_quantile=0.05, upper_quantile=0.95):
    """Overlap of central work ranges, without letting one outlier pass the gate.

    The legacy range-based diagnostic is retained for continuity, but a single
    reciprocal outlier can make two otherwise disjoint distributions appear to
    overlap.  This guard asks both whether frames occupy the other ensemble's
    central interval and whether central intervals overlap for a substantial
    fraction of independent chain pairs.
    """
    if not 0.0 <= discard < 1.0:
        raise ValueError("discard must lie in [0, 1)")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("central quantiles must be ordered inside (0, 1)")

    reference = [np.asarray(chain["du"], dtype=float).ravel()
                 for chain in reference_chains]
    surrogate = []
    for chain in surrogate_chains:
        values = np.asarray(chain["du"], dtype=float).ravel()
        start = int(round(discard * values.size))
        surrogate.append(values[start:])
    if not reference or not surrogate or any(values.size < 4 for values in
                                              reference + surrogate):
        raise ValueError("each arm needs non-empty chains with at least four frames")

    def central_interval(values):
        return np.quantile(values, [lower_quantile, upper_quantile])

    ref_all = np.concatenate(reference)
    sur_all = np.concatenate(surrogate)
    ref_interval = central_interval(ref_all)
    sur_interval = central_interval(sur_all)
    ref_inside_sur = np.mean(
        (ref_all >= sur_interval[0]) & (ref_all <= sur_interval[1])
    )
    sur_inside_ref = np.mean(
        (sur_all >= ref_interval[0]) & (sur_all <= ref_interval[1])
    )
    pair_overlap = []
    for ref_values in reference:
        ref_bounds = central_interval(ref_values)
        for sur_values in surrogate:
            sur_bounds = central_interval(sur_values)
            pair_overlap.append(
                max(ref_bounds[0], sur_bounds[0])
                <= min(ref_bounds[1], sur_bounds[1])
            )
    return {
        "quantiles": [float(lower_quantile), float(upper_quantile)],
        "reference_central_interval": ref_interval.tolist(),
        "surrogate_central_interval": sur_interval.tolist(),
        "central_mutual_frame_fraction": float(min(ref_inside_sur, sur_inside_ref)),
        "chain_pair_overlap_fraction": float(np.mean(pair_overlap)),
    }


def joint_chain_bootstrap(refs, surrogate_chains, temperature, discard, *,
                          n_resamples, seed):
    """Bootstrap complete independent chains and preserve shared-data covariance.

    The HMC direct estimate, response prediction and MBAR estimate all reuse the
    reference chains, while direct and MBAR also reuse the surrogate chains. A
    separate error bar for each number followed by quadrature cannot represent
    that dependence. Resampling the two sets of independent chain units and
    recomputing every estimator on each draw does so by construction and never
    joins blocks across chain boundaries.
    """
    ref_a = [np.asarray(chain["a"], dtype=float) for chain in refs]
    ref_du = [np.asarray(chain["du"], dtype=float) for chain in refs]
    sur_a = []
    sur_du = []
    for chain in surrogate_chains:
        start = int(round(discard * chain["a"].shape[0]))
        sur_a.append(np.asarray(chain["a"][start:], dtype=float))
        sur_du.append(np.asarray(chain["du"][start:], dtype=float))

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in (
        "direct", "linear", "forward_fep", "reverse_fep", "mbar",
        "direct_minus_linear", "direct_minus_mbar",
    )}
    for _ in range(int(n_resamples)):
        ref_index = rng.integers(0, len(ref_a), size=len(ref_a))
        sur_index = rng.integers(0, len(sur_a), size=len(sur_a))
        a_ref = np.concatenate([ref_a[index] for index in ref_index], axis=0)
        du_ref = np.concatenate([ref_du[index] for index in ref_index], axis=0)
        a_sur = np.concatenate([sur_a[index] for index in sur_index], axis=0)
        du_sur = np.concatenate([sur_du[index] for index in sur_index], axis=0)

        direct = np.mean(
            [sur_a[index].mean(axis=0) for index in sur_index], axis=0
        ) - np.mean(
            [ref_a[index].mean(axis=0) for index in ref_index], axis=0
        )
        linear = _linear_point(a_ref, du_ref, temperature)
        forward = _reweight_point(a_ref, du_ref, temperature)
        reverse = -_reweight_point(a_sur, -du_sur, temperature)
        mbar = np.asarray(mbar_two_state_average(
            a_ref, du_ref, a_sur, du_sur, temperature,
            n_resamples=0,
        )["shift"].value, dtype=float)

        for key, value in (
            ("direct", direct), ("linear", linear),
            ("forward_fep", forward), ("reverse_fep", reverse),
            ("mbar", mbar), ("direct_minus_linear", direct - linear),
            ("direct_minus_mbar", direct - mbar),
        ):
            samples[key].append(np.asarray(value, dtype=float))

    arrays = {key: np.asarray(value) for key, value in samples.items()}
    return {
        "unit": "independent_chain",
        "n_reference_chains": len(ref_a),
        "n_surrogate_chains": len(sur_a),
        "n_resamples": int(n_resamples),
        "samples": {key: value.tolist() for key, value in arrays.items()},
        "sem": {key: value.std(axis=0, ddof=1).tolist()
                for key, value in arrays.items()},
        "correlation": {
            "direct_linear": [float(np.corrcoef(
                arrays["direct"][:, index], arrays["linear"][:, index]
            )[0, 1]) for index in range(arrays["direct"].shape[1])],
            "direct_mbar": [float(np.corrcoef(
                arrays["direct"][:, index], arrays["mbar"][:, index]
            )[0, 1]) for index in range(arrays["direct"].shape[1])],
        },
    }


def run(ctx: ExperimentContext) -> dict:
    require_fresh_production_seed(ctx, expected=20260812)
    require_frozen_protocol(
        ctx, protocol_path=PROTOCOL_PATH,
        expected_protocol=EXPERIMENT_NAME, expected_version=2,
    )
    cfg, potential = build_system(ctx)
    initial_cfg = cfg.copy()
    temperature = ctx.config["system"]["temperature"]
    o, s_cfg = ctx.config["observable"], ctx.config["sampling"]
    rep, an = ctx.config["replication"], ctx.config["analysis"]
    n_reference_chains = 2 if ctx.quick else int(rep["n_reference_chains"])
    n_surrogate_chains = 2 if ctx.quick else int(rep["n_surrogate_chains"])
    bins = list(o["report_bins"])
    edges = np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    observable = PairBinObservable(edges, cutoff=ctx.config["potential"]["cutoff"])
    f = ctx.config["field"]
    field = RadialShellPerturbation(r0=f["r0"], width=f["width"],
                                    amplitude=f["amplitude"],
                                    cutoff=ctx.config["potential"]["cutoff"])
    surrogate = potential + field
    fractions = list(ctx.config["relaxation"]["discard_fractions"])
    print(f"  system: {cfg.n_atoms} atoms, T = {temperature} K")
    print(f"  reporting bins {bins} at r = {np.round(centres[bins], 2)} A")
    print(f"  field: Gaussian shell r0={f['r0']} w={f['width']} a={f['amplitude']:.1e} eV")

    with ctx.timed("equilibration"):
        cfg, eq = equilibrated_configuration(
            cfg, potential, temperature, n_melt=int(ctx.scaled("sampling.n_melt")),
            n_anneal=int(ctx.scaled("sampling.n_anneal")), n_leapfrog=s_cfg["n_leapfrog"],
            step_size=s_cfg["step_size"], seed=ctx.seed,
            q6_target=1.0 if ctx.quick else 0.20)
    print(f"    {eq}")

    # -- reference ensemble --------------------------------------------------
    with ctx.timed("reference_chains"):
        refs = []
        for i in range(n_reference_chains):
            reference_start, reference_equilibration = equilibrated_configuration(
                initial_cfg, potential, temperature,
                n_melt=int(ctx.scaled("sampling.n_melt")),
                n_anneal=int(ctx.scaled("sampling.n_anneal")),
                n_leapfrog=s_cfg["n_leapfrog"], step_size=s_cfg["step_size"],
                seed=ctx.seed + 2000 + 101 * i,
                q6_target=1.0 if ctx.quick else 0.20,
            )
            traj, report = hmc(
                ctx, reference_start, potential, ctx.scaled("sampling.n_reference"),
                seed=ctx.seed + 3000 + 41 * i, label=f"reference {i}"
            )
            a = chain_values(traj, observable, bins)
            du = np.array([field.energy(traj.frame(j)) for j in range(traj.n_frames)])
            refs.append({"index": i, "a": a, "du": du, "traj": traj,
                         "seed": ctx.seed + 3000 + 41 * i,
                         "equilibration_seed": ctx.seed + 2000 + 101 * i,
                         "equilibration": dict(reference_equilibration.__dict__),
                         "energy": np.asarray(traj.scalars["potential_energy"]),
                         "acceptance": float(report.acceptance),
                         "mean": a.mean(axis=0)})
            print(f"      reference {i}: <A> = {np.round(refs[-1]['mean'], 2)}")
    ref_means = np.array([r["mean"] for r in refs])
    ref_grand = ref_means.mean(axis=0)
    ref_sem = ref_means.std(axis=0, ddof=1) / np.sqrt(len(refs))
    print(f"    reference grand mean {np.round(ref_grand, 3)} "
          f"+/- {np.round(ref_sem, 3)} (over {len(refs)} chains)")

    # Independently equilibrated configurations under the surrogate.  A former
    # version selected all eight starts from one presoak trajectory, making that
    # upstream trajectory a shared experimental unit.  Each arm replicate now
    # has its own presoak seed and starts from a different reference chain.
    with ctx.timed("surrogate_presoak"):
        presoaks = []
        for i in range(n_surrogate_chains):
            # Generate the surrogate upstream replicate independently from the
            # reference replicas. A different RNG attached to one shared liquid
            # configuration would not remove the ancestral dependence while
            # testing incomplete mixing.
            start, presoak_equilibration = equilibrated_configuration(
                initial_cfg, surrogate, temperature,
                n_melt=int(ctx.scaled("sampling.n_melt")),
                n_anneal=int(ctx.scaled("sampling.n_anneal")),
                n_leapfrog=s_cfg["n_leapfrog"], step_size=s_cfg["step_size"],
                seed=ctx.seed + 7000 + 103 * i,
                q6_target=1.0 if ctx.quick else 0.20,
            )
            presoak, report = hmc(
                ctx, start, surrogate, ctx.scaled("sampling.n_presoak"),
                seed=ctx.seed + 7777 + 97 * i,
                label=f"surrogate presoak {i}",
            )
            presoaks.append({
                "traj": presoak,
                "seed": ctx.seed + 7777 + 97 * i,
                "equilibration_seed": ctx.seed + 7000 + 103 * i,
                "equilibration": dict(presoak_equilibration.__dict__),
                "acceptance": float(report.acceptance),
                "mean": chain_values(presoak, observable, bins).mean(axis=0),
            })
        soak_frames = [entry["traj"].frame(-1) for entry in presoaks]
    print(f"    {len(presoaks)} independent presoaks: "
          f"<A> = {np.round(np.mean([p['mean'] for p in presoaks], axis=0), 2)}")

    # -- surrogate ensemble, bracketed ---------------------------------------
    # burn_in=0 on purpose: the relaxation is the measurement, so nothing may be
    # silently discarded inside the sampler.
    with ctx.timed("surrogate_chains"):
        arms = {}
        starts = {
            "from_reference": [refs[i % len(refs)]["traj"].frame(
                refs[i % len(refs)]["traj"].n_frames - 1 - 7 * i)
                for i in range(n_surrogate_chains)],
            "from_surrogate": soak_frames,
        }
        tuned = tune_step_size(ctx, cfg, surrogate, seed=ctx.seed + 6006)
        print(f"      step size tuned on a throwaway chain: {tuned * 1e3:.2f} fs "
              f"(configured {s_cfg['step_size'] * 1e3:.2f} fs)")
        deepest = None
        for arm, start_configs in starts.items():
            chains = []
            for i, start in enumerate(start_configs):
                traj, report = hmc(
                    ctx, start, surrogate, ctx.scaled("sampling.n_surrogate"),
                    seed=ctx.seed + 4000 + 53 * i
                    + (0 if arm == "from_reference" else 1),
                    burn_in=0, label=f"{arm} {i}", check=False,
                    step_size=tuned,
                )
                a = chain_values(traj, observable, bins)
                du = np.array([field.energy(traj.frame(j)) for j in range(traj.n_frames)])
                chains.append({
                    "a": a,
                    "du": du,
                    "energy": np.asarray(traj.scalars["potential_energy"]),
                    "traj": traj,
                    "acceptance": float(report.acceptance),
                    "seed": ctx.seed + 4000 + 53 * i
                    + (0 if arm == "from_reference" else 1),
                    "ladder": ladder(a, fractions),
                })
                deepest = max(chains[-1]["ladder"]) if deepest is None \
                    else min(deepest, max(chains[-1]["ladder"]))
            arms[arm] = chains
            final = np.array([c["ladder"][deepest] for c in chains])
            print(f"      {arm}: <A> after discarding {deepest:.0%} = "
                  f"{np.round(final.mean(axis=0), 2)}")
    fractions = [f for f in fractions if f <= deepest]

    # -- the relaxation curve ------------------------------------------------
    relaxation = []
    for frac in fractions:
        row = {"discard_fraction": frac}
        for arm, chains in arms.items():
            vals = np.array([c["ladder"][frac] for c in chains if frac in c["ladder"]])
            if vals.size == 0:
                continue
            row[arm] = (vals.mean(axis=0) - ref_grand).tolist()
            row[f"{arm}_sem"] = (vals.std(axis=0, ddof=1) / np.sqrt(len(vals))).tolist()
            row[f"{arm}_chain_means"] = vals.tolist()
        if "from_reference" in row and "from_surrogate" in row:
            # The two arms use independent RNG streams and independent starts;
            # their equal indices are not a pair. Propagate the two chain-mean
            # variances independently rather than manufacturing a paired design.
            row["arm_gap"] = (np.array(row["from_surrogate"])
                              - np.array(row["from_reference"])).tolist()
            row["arm_gap_sem"] = np.hypot(
                np.asarray(row["from_surrogate_sem"], dtype=float),
                np.asarray(row["from_reference_sem"], dtype=float),
            ).tolist()
        relaxation.append(row)

    print("\n    relaxation ladder (shift relative to the reference grand mean):")
    print(f"      {'discard':>8} {'from_reference':>22} {'from_surrogate':>22} {'gap':>16}")
    for row in relaxation:
        if "arm_gap" not in row:
            continue
        print(f"      {row['discard_fraction']:8.0%} "
              f"{str(np.round(row['from_reference'], 2)):>22} "
              f"{str(np.round(row['from_surrogate'], 2)):>22} "
              f"{str(np.round(row['arm_gap'], 2)):>16}")

    primary_discard = float(an["primary_discard_fraction"])
    if primary_discard not in fractions:
        raise ValueError(
            f"primary discard {primary_discard} is not an available ladder point "
            f"{fractions}"
        )
    stationarity = {
        arm: [
            tail_stationarity(chain["traj"], chain["a"], chain["du"],
                              primary_discard,
                              label=f"{arm} {index}")
            for index, chain in enumerate(chains)
        ]
        for arm, chains in arms.items()
    }
    stationarity["reference"] = [
        tail_stationarity(chain["traj"], chain["a"], chain["du"], 0.0,
                          label=f"reference {index}")
        for index, chain in enumerate(refs)
    ]

    # Preserve the chain boundary and every per-frame quantity needed to redo
    # the response, direct, relaxation and covariance analyses. Aggregates alone
    # cannot support a later correction of the uncertainty model.
    ctx.save_npz(
        "chains_raw",
        reference_a=_stack_chain_field(refs, "a"),
        reference_du=_stack_chain_field(refs, "du"),
        reference_energy=_stack_chain_field(refs, "energy"),
        reference_positions=np.stack([chain["traj"].positions for chain in refs]),
        reference_cells=np.stack([chain["traj"].cells for chain in refs]),
        reference_seeds=np.asarray([chain["seed"] for chain in refs], dtype=np.int64),
        reference_equilibration_seeds=np.asarray([
            chain["equilibration_seed"] for chain in refs
        ], dtype=np.int64),
        surrogate_from_reference_a=_stack_chain_field(arms["from_reference"], "a"),
        surrogate_from_reference_du=_stack_chain_field(arms["from_reference"], "du"),
        surrogate_from_reference_energy=_stack_chain_field(
            arms["from_reference"], "energy"
        ),
        surrogate_from_reference_positions=np.stack([
            chain["traj"].positions for chain in arms["from_reference"]
        ]),
        surrogate_from_reference_cells=np.stack([
            chain["traj"].cells for chain in arms["from_reference"]
        ]),
        surrogate_from_reference_seeds=np.asarray([
            chain["seed"] for chain in arms["from_reference"]
        ], dtype=np.int64),
        surrogate_from_surrogate_a=_stack_chain_field(arms["from_surrogate"], "a"),
        surrogate_from_surrogate_du=_stack_chain_field(arms["from_surrogate"], "du"),
        surrogate_from_surrogate_energy=_stack_chain_field(
            arms["from_surrogate"], "energy"
        ),
        surrogate_from_surrogate_positions=np.stack([
            chain["traj"].positions for chain in arms["from_surrogate"]
        ]),
        surrogate_from_surrogate_cells=np.stack([
            chain["traj"].cells for chain in arms["from_surrogate"]
        ]),
        surrogate_from_surrogate_seeds=np.asarray([
            chain["seed"] for chain in arms["from_surrogate"]
        ], dtype=np.int64),
        presoak_a=np.stack([
            chain_values(entry["traj"], observable, bins) for entry in presoaks
        ], axis=0),
        presoak_energy=np.stack([
            np.asarray(entry["traj"].scalars["potential_energy"], dtype=float)
            for entry in presoaks
        ], axis=0),
        presoak_positions=np.stack([entry["traj"].positions for entry in presoaks]),
        presoak_cells=np.stack([entry["traj"].cells for entry in presoaks]),
        presoak_seeds=np.asarray([entry["seed"] for entry in presoaks], dtype=np.int64),
        presoak_equilibration_seeds=np.asarray([
            entry["equilibration_seed"] for entry in presoaks
        ], dtype=np.int64),
        species=np.asarray(refs[0]["traj"].template.species),
        pbc=np.asarray(refs[0]["traj"].template.pbc),
    )

    # -- estimators ----------------------------------------------------------
    analysis_resamples = 200 if ctx.quick else int(an["n_resamples"])
    with ctx.timed("estimators"):
        estimates = compare_estimators(
            refs, arms["from_surrogate"], arms["from_reference"], temperature,
            primary_discard, n_resamples=analysis_resamples, seed=ctx.seed)

    # -- independent sampler -------------------------------------------------
    with ctx.timed("metropolis_cross_check"):
        metro = metropolis_arm(ctx, cfg, potential, surrogate, observable, bins,
                               field, temperature, n_resamples=analysis_resamples)

    summary = assemble(estimates, metro, relaxation, ref_grand, ref_sem,
                       centres[bins], float(an["equivalence_bound_pairs"]),
                       primary_discard, analysis=an, stationarity=stationarity)
    summary["tuned_step_size_fs"] = tuned * 1e3
    ctx.save_json("relaxation", relaxation)
    ctx.save_json("estimates", estimates)
    ctx.save_json("metropolis", metro)
    ctx.save_json("summary", summary)
    report_summary(summary)
    return summary


def compare_estimators(refs, surrogate_chains, other_chains, temperature, discard, *,
                       n_resamples, seed):
    """Every estimator of the same shift, on the same data, side by side."""
    a_ref = np.concatenate([r["a"] for r in refs], axis=0)
    du_ref = np.concatenate([r["du"] for r in refs])
    n_keep = [int(round(discard * c["a"].shape[0])) for c in surrogate_chains]
    a_sur = np.concatenate([c["a"][k:] for c, k in zip(surrogate_chains, n_keep)], axis=0)
    du_sur = np.concatenate([c["du"][k:] for c, k in zip(surrogate_chains, n_keep)])

    out = {"n_reference_samples": int(a_ref.shape[0]),
           "n_surrogate_samples": int(a_sur.shape[0])}

    joint = joint_chain_bootstrap(
        refs, surrogate_chains, temperature, discard,
        n_resamples=n_resamples, seed=seed + 100,
    )
    out["joint_chain_bootstrap"] = joint

    linear = predict_shift(a_ref, du_ref, temperature, n_resamples=n_resamples, seed=seed)
    # Per-chain predictions as well as the pooled one. The manuscript quoted a
    # within-chain blocking error for the prediction, which says how well one
    # chain determines its own covariance and nothing about how much the
    # covariance moves between chains -- the quantity a difference against an
    # independently sampled ensemble actually needs.
    per_chain = np.array([np.asarray(predict_shift(r["a"], r["du"], temperature,
                                                   n_resamples=max(100, n_resamples // 4),
                                                   seed=seed + 10 + r["index"]).value)
                          for r in refs])
    out["linear"] = {"value": np.asarray(linear.value).tolist(),
                     "within_pooled_chain_error": np.asarray(linear.error).tolist(),
                     "error": joint["sem"]["linear"],
                     "second_order": np.asarray(linear.second_order.value).tolist(),
                     "second_order_ratio": np.asarray(linear.second_order_ratio).tolist(),
                     "per_chain": per_chain.tolist(),
                     "per_chain_reference_mean": [r["mean"].tolist() for r in refs],
                     "between_chain_scatter": per_chain.std(axis=0, ddof=1).tolist(),
                     "between_chain_sem": (per_chain.std(axis=0, ddof=1)
                                           / np.sqrt(len(per_chain))).tolist()}

    fwd = reweight(a_ref, du_ref, temperature, n_resamples=n_resamples, seed=seed)
    out["forward_fep"] = {"value": np.asarray(fwd.shift.value).tolist(),
                          "within_pooled_chain_error": np.asarray(fwd.shift.error).tolist(),
                          "error": joint["sem"]["forward_fep"],
                          "ess_fraction": fwd.ess_fraction,
                          "max_weight_fraction": fwd.max_weight_fraction}

    # Reverse FEP: reweight the surrogate ensemble back onto the reference, with
    # the sign of the energy difference flipped because the roles swap. Its
    # `shift` is then <A>_0 - <A>_U, so the sign is flipped back to keep every
    # row of the table pointing the same way.
    rev = reweight(a_sur, -du_sur, temperature, n_resamples=n_resamples, seed=seed + 1)
    out["reverse_fep"] = {"value": (-np.asarray(rev.shift.value)).tolist(),
                          "within_pooled_chain_error": np.asarray(rev.shift.error).tolist(),
                          "error": joint["sem"]["reverse_fep"],
                          "ess_fraction": rev.ess_fraction,
                          "max_weight_fraction": rev.max_weight_fraction}

    bar = bennett_acceptance_ratio(du_ref, du_sur, temperature)
    out["bar_delta_f"] = {"value": bar.value, "error": bar.error}

    # Point estimate only. Its uncertainty comes from the joint chain bootstrap
    # above; concatenating the independent chains and block-bootstrapping across
    # their boundaries would use the wrong experimental unit.
    mbar = mbar_two_state_average(a_ref, du_ref, a_sur, du_sur, temperature,
                                  n_resamples=0, seed=seed + 2)
    out["mbar"] = {"value": np.asarray(mbar["shift"].value).tolist(),
                   "error": joint["sem"]["mbar"],
                   "kish_reference": mbar["kish_reference"],
                   "kish_surrogate": mbar["kish_surrogate"],
                   "max_weight_reference": mbar["max_weight_reference"],
                   "max_weight_surrogate": mbar["max_weight_surrogate"],
                   "delta_f": mbar["delta_f"]}

    # Direct: chain means are the experimental unit, so the error is their
    # scatter, not a within-chain blocking estimate. The primary arm is the one
    # started from configurations already equilibrated under the surrogate --
    # not because it gives the answer one wants, but because a chain that begins
    # in the ensemble it is measuring has no relaxation transient to contaminate
    # it. The other arm is reported beside it, and the difference between them
    # *is* the relaxation artefact rather than an error bar on it.
    ref_chain = np.array([r["a"].mean(axis=0) for r in refs])
    for arm_name, chains in (("direct", surrogate_chains),
                             ("direct_from_reference", other_chains)):
        keep = [int(round(discard * c["a"].shape[0])) for c in chains]
        sur_chain = np.array([c["a"][k:].mean(axis=0) for c, k in zip(chains, keep)])
        value = sur_chain.mean(axis=0) - ref_chain.mean(axis=0)
        err = (np.asarray(joint["sem"]["direct"])
               if arm_name == "direct" else
               np.sqrt(sur_chain.var(axis=0, ddof=1) / len(sur_chain)
                       + ref_chain.var(axis=0, ddof=1) / len(ref_chain)))
        out[arm_name] = {"value": value.tolist(), "error": err.tolist(),
                         "n_reference_chains": len(ref_chain),
                         "n_surrogate_chains": len(sur_chain),
                         "reference_chain_means": ref_chain.tolist(),
                         "surrogate_chain_means": sur_chain.tolist()}

    out["direct_estimator_correlation"] = joint["correlation"]

    out["overlap"] = overlap_diagnostics(du_ref, du_sur, temperature)
    out["overlap"]["forward_chain_diagnostics"] = chain_weight_diagnostics(
        refs, temperature, sign=-1.0, discard=0.0
    )
    out["overlap"]["reverse_chain_diagnostics"] = chain_weight_diagnostics(
        surrogate_chains, temperature, sign=+1.0, discard=discard
    )
    out["overlap"]["robust_work_overlap"] = robust_work_overlap(
        refs, surrogate_chains, discard=discard
    )
    out["tau_reference_per_chain"] = [
        float(integrated_autocorrelation_time(chain["a"])) for chain in refs
    ]
    out["tau_surrogate_per_chain"] = [
        float(integrated_autocorrelation_time(chain["a"][int(round(
            discard * chain["a"].shape[0]
        )):])) for chain in surrogate_chains
    ]
    out["blocking_reference_per_chain"] = [
        np.asarray(blocking_analysis(chain["a"]).error).tolist() for chain in refs
    ]
    return out


def metropolis_arm(ctx, cfg, potential, surrogate, observable, bins, field, temperature,
                   *, n_resamples):
    """The same shift from a sampler that shares no propagation machinery.

    Single-particle Metropolis costs a full energy evaluation per attempted move
    and is roughly thirteen times more expensive per sweep than an HMC move
    here, so this arm is deliberately smaller.  It is a check on whether the
    HMC answer is a property of the ensemble or of the sampler, and for that a
    wider error bar is acceptable; it is not a second primary measurement.
    """
    m = ctx.config["metropolis"]
    n_chains = 2 if ctx.quick else int(m["n_chains"])
    out = {"n_chains": n_chains, "n_sweeps": int(ctx.scaled("metropolis.n_sweeps"))}
    raw = {}
    for label, pot in (("reference", potential), ("surrogate", surrogate)):
        means, accs, values, energies, delta_u, diagnostics, trajectories = (
            [], [], [], [], [], [], []
        )
        for i in range(n_chains):
            traj, report = metropolis_nvt(
                cfg, pot, temperature, n_sweeps=int(ctx.scaled("metropolis.n_sweeps")),
                burn_in=int(m["burn_in"]), max_displacement=m["max_displacement"],
                seed=ctx.seed + 8000 + 61 * i + (0 if label == "reference" else 1))
            a = observable.evaluate_trajectory(traj)[:, bins]
            du = np.array([
                field.energy(traj.frame(frame)) for frame in range(traj.n_frames)
            ])
            values.append(a)
            energies.append(np.asarray(traj.scalars["potential_energy"], dtype=float))
            delta_u.append(du)
            means.append(a.mean(axis=0))
            accs.append(report.acceptance)
            diagnostics.append(tail_stationarity(
                traj, a, du, 0.0, label=f"metropolis {label} {i}"
            ))
            trajectories.append(traj)
        means = np.array(means)
        out[label] = {"chain_means": means.tolist(),
                      "mean": means.mean(axis=0).tolist(),
                      "sem": (means.std(axis=0, ddof=1) / np.sqrt(len(means))).tolist(),
                      "acceptance": float(np.mean(accs)),
                      "stationarity": diagnostics}
        raw[f"{label}_a"] = np.stack(values, axis=0)
        raw[f"{label}_energy"] = np.stack(energies, axis=0)
        raw[f"{label}_du"] = np.stack(delta_u, axis=0)
        raw[f"{label}_positions"] = np.stack(
            [trajectory.positions for trajectory in trajectories], axis=0
        )
        raw[f"{label}_cells"] = np.stack(
            [trajectory.cells for trajectory in trajectories], axis=0
        )
        raw[f"{label}_seeds"] = np.asarray([
            ctx.seed + 8000 + 61 * i + (0 if label == "reference" else 1)
            for i in range(n_chains)
        ], dtype=np.int64)
        print(f"      metropolis {label}: {np.round(means.mean(axis=0), 2)} "
              f"+/- {np.round(means.std(axis=0, ddof=1) / np.sqrt(len(means)), 2)} "
              f"(acceptance {np.mean(accs):.2f})")
    raw["species"] = np.asarray(trajectories[0].template.species)
    raw["pbc"] = np.asarray(trajectories[0].template.pbc)
    shift = np.array(out["surrogate"]["mean"]) - np.array(out["reference"]["mean"])
    err = np.sqrt(np.array(out["surrogate"]["sem"])**2 + np.array(out["reference"]["sem"])**2)
    out["shift"] = shift.tolist()
    out["shift_error"] = err.tolist()
    rng = np.random.default_rng(ctx.seed + 9900)
    ref_means = np.asarray(out["reference"]["chain_means"], dtype=float)
    sur_means = np.asarray(out["surrogate"]["chain_means"], dtype=float)
    bootstrap = []
    for _ in range(int(n_resamples)):
        ref_index = rng.integers(0, len(ref_means), size=len(ref_means))
        sur_index = rng.integers(0, len(sur_means), size=len(sur_means))
        bootstrap.append(
            sur_means[sur_index].mean(axis=0) - ref_means[ref_index].mean(axis=0)
        )
    bootstrap = np.asarray(bootstrap)
    out["chain_bootstrap"] = {
        "unit": "independent_chain",
        "n_resamples": int(len(bootstrap)),
        "samples": bootstrap.tolist(),
        "sem": bootstrap.std(axis=0, ddof=1).tolist(),
    }
    ctx.save_npz("metropolis_raw", **raw)
    return out


def assemble(est, metro, relaxation, ref_grand, ref_sem, radii, bound, discard, *,
             analysis, stationarity):
    """Apply the prospectively frozen v2 rule, per bin and declared family."""
    # Two reference-based estimators are compared against direct sampling. MBAR
    # is the primary one because it uses both ensembles; the linear prediction
    # is included because it is the estimator the manuscript's claims rest on,
    # and its uncertainty is taken as the scatter of the per-chain predictions,
    # not the blocking error of the pooled samples.
    targets = {
        "mbar": (np.array(est["mbar"]["value"]), np.array(est["mbar"]["error"])),
        "linear": (np.array(est["linear"]["value"]),
                   np.array(est["linear"]["error"])),
    }
    alpha = float(analysis.get("familywise_alpha", 0.05))
    n_primary = len(radii)
    n_secondary = (2 * len(targets) * len(radii)) - n_primary
    joint = est["joint_chain_bootstrap"]
    comparisons = []
    for sampler, value, error in (
            ("hmc", np.array(est["direct"]["value"]), np.array(est["direct"]["error"])),
            ("metropolis", np.array(metro["shift"]), np.array(metro["shift_error"]))):
        for name, (target, target_err) in targets.items():
            family = comparison_family(sampler, name)
            family_size = n_primary if family == "primary" else n_secondary
            diff = value - target
            if sampler == "hmc":
                difference_samples = np.asarray(
                    joint["samples"][f"direct_minus_{name}"], dtype=float
                )
                rho = np.asarray(joint["correlation"][f"direct_{name}"], dtype=float)
                uncertainty_method = "joint independent-chain percentile bootstrap"
            else:
                # The Metropolis chains share neither reference nor surrogate
                # samples with the HMC/reference-based estimators. Combine their
                # independently generated chain-bootstrap draws, rather than a
                # Gaussian quadrature approximation.
                metro_samples = np.asarray(
                    metro["chain_bootstrap"]["samples"], dtype=float
                )
                target_samples = np.asarray(joint["samples"][name], dtype=float)
                if metro_samples.shape != target_samples.shape:
                    raise RuntimeError("bootstrap sample counts/shapes do not match")
                difference_samples = metro_samples - target_samples
                rho = np.zeros(difference_samples.shape[1], dtype=float)
                uncertainty_method = "independent chain percentile bootstrap"
            err = difference_samples.std(axis=0, ddof=1)
            for j, r in enumerate(radii):
                nominal_ci = percentile_interval(
                    difference_samples[:, j], 0.025, 0.975
                )
                # Equivalence uses two one-sided tests at alpha/m, hence an
                # interval with alpha/m in each tail. Meaningful difference is
                # assessed with a simultaneous two-sided interval, alpha/(2m)
                # in each tail. With only 8 HMC and 4 Metropolis chains, normal-z
                # intervals would overstate precision; the complete-chain
                # bootstrap distribution is used directly.
                familywise_tost_ci = percentile_interval(
                    difference_samples[:, j],
                    alpha / family_size,
                    1.0 - alpha / family_size,
                )
                simultaneous_ci = percentile_interval(
                    difference_samples[:, j],
                    alpha / (2.0 * family_size),
                    1.0 - alpha / (2.0 * family_size),
                )
                nominal = interval_disposition(nominal_ci, bound)
                familywise_tost = interval_disposition(familywise_tost_ci, bound)
                familywise_difference = interval_disposition(simultaneous_ci, bound)
                comparisons.append({
                    "sampler": sampler, "against": name, "bin_radius": float(r),
                    "family": family,
                    "direct": float(value[j]), "direct_error": float(error[j]),
                    "reference_based": float(target[j]),
                    "reference_based_error": float(target_err[j]),
                    "difference": float(diff[j]), "difference_error": float(err[j]),
                    "uncertainty_method": uncertainty_method,
                    "direct_reference_based_correlation": float(rho[j]),
                    "ci95": nominal["ci"],
                    "ci_familywise_tost": familywise_tost["ci"],
                    "ci_bonferroni_two_sided": familywise_difference["ci"],
                    "nominal_disposition": nominal["status"],
                    "familywise_tost_disposition": familywise_tost["status"],
                    "familywise_difference_disposition": (
                        familywise_difference["status"]
                    ),
                    "within_bound": nominal["status"] == "equivalent",
                    "meaningfully_different": (
                        nominal["status"] == "meaningfully_different"
                    ),
                    "underpowered": familywise_tost[
                        "precision_insufficient_for_equivalence"
                    ] or familywise_tost["status"] == "inconclusive",
                })
    first = relaxation[0]
    selected = next(
        (row for row in relaxation
         if np.isclose(float(row["discard_fraction"]), float(discard))),
        None,
    )
    if selected is None:
        raise ValueError(f"discard fraction {discard} is absent from relaxation ladder")
    # Whether the two initialisations have met. This gates everything else: if
    # they have not, the direct estimate still carries a relaxation transient
    # and no comparison against it means anything, whatever its error bar says.
    gap = np.asarray(selected["arm_gap"], dtype=float) \
        if selected.get("arm_gap") is not None else None
    gap_error = np.asarray(selected["arm_gap_sem"], dtype=float) \
        if selected.get("arm_gap_sem") is not None else None
    relaxation_samples = None
    relaxation_dispositions = []
    if (selected.get("from_reference_chain_means") is not None
            and selected.get("from_surrogate_chain_means") is not None):
        relaxation_samples = independent_chain_difference_bootstrap(
            np.asarray(selected["from_reference_chain_means"], dtype=float),
            np.asarray(selected["from_surrogate_chain_means"], dtype=float),
            n_resamples=int(joint["n_resamples"]),
            seed=2026081201,
        )
        for endpoint in range(len(radii)):
            ci = percentile_interval(
                relaxation_samples[:, endpoint],
                alpha / max(1, len(radii)),
                1.0 - alpha / max(1, len(radii)),
            )
            relaxation_dispositions.append(interval_disposition(ci, bound))
    arms_converged = bool(
        relaxation_dispositions
        and all(item["status"] == "equivalent" for item in relaxation_dispositions)
    )
    overlap = est["overlap"]
    overlap_gate = {
        "min_work_overlap": float(analysis.get("min_work_overlap", 0.10)),
        "min_kish_fraction": float(analysis.get("min_kish_fraction", 0.10)),
        "max_weight_fraction": float(analysis.get("max_weight_fraction", 0.10)),
        "min_autocorrelation_adjusted_ess_fraction": float(analysis.get(
            "min_autocorrelation_adjusted_ess_fraction", 0.02
        )),
        "max_chain_weight_fraction": float(analysis.get(
            "max_chain_weight_fraction", 0.35
        )),
        "min_central_work_overlap": float(analysis.get(
            "min_central_work_overlap", 0.10
        )),
        "min_chain_pair_overlap_fraction": float(analysis.get(
            "min_chain_pair_overlap_fraction", 0.50
        )),
    }
    chain_overlap = [
        overlap["forward_chain_diagnostics"],
        overlap["reverse_chain_diagnostics"],
    ]
    overlap_passed = bool(
        overlap["work_overlap"] >= overlap_gate["min_work_overlap"]
        and min(overlap["kish_forward"], overlap["kish_reverse"])
        >= overlap_gate["min_kish_fraction"]
        and max(overlap["max_weight_forward"], overlap["max_weight_reverse"])
        <= overlap_gate["max_weight_fraction"]
        and min(item["autocorrelation_adjusted_ess_fraction"]
                for item in chain_overlap)
        >= overlap_gate["min_autocorrelation_adjusted_ess_fraction"]
        and max(item["max_chain_weight_fraction"] for item in chain_overlap)
        <= overlap_gate["max_chain_weight_fraction"]
        and overlap["robust_work_overlap"]["central_mutual_frame_fraction"]
        >= overlap_gate["min_central_work_overlap"]
        and overlap["robust_work_overlap"]["chain_pair_overlap_fraction"]
        >= overlap_gate["min_chain_pair_overlap_fraction"]
    )
    hmc_stationarity_passed = bool(
        stationarity
        and all(item["passed"] for arm in stationarity.values() for item in arm)
    )
    metropolis_stationarity_passed = bool(
        all(
            item["passed"]
            for label in ("reference", "surrogate")
            for item in metro[label]["stationarity"]
        )
    )
    # The smaller Metropolis arm is a sensitivity analysis. Its failure is
    # reported but cannot veto the frozen HMC-minus-MBAR primary estimand.
    stationarity_passed = hmc_stationarity_passed
    primary_comparisons = [item for item in comparisons
                           if item["family"] == "primary"]
    secondary_comparisons = [item for item in comparisons
                             if item["family"] == "secondary"]
    all_familywise_equivalent = bool(primary_comparisons) and all(
        item["familywise_tost_disposition"] == "equivalent"
        for item in primary_comparisons
    )
    any_familywise_difference = any(
        item["familywise_difference_disposition"] == "meaningfully_different"
        for item in primary_comparisons
    )
    disposition = overall_disposition(
        arms_converged=arms_converged,
        stationarity_passed=stationarity_passed,
        overlap_passed=overlap_passed,
        all_equivalent=all_familywise_equivalent,
        any_meaningful_difference=any_familywise_difference,
    )
    return {
        "equivalence_bound_pairs": bound,
        "familywise_alpha": alpha,
        "comparison_families": {
            "primary": {"definition": "HMC direct minus MBAR", "size": n_primary},
            "secondary": {"definition": "all other sensitivity comparisons",
                          "size": n_secondary},
        },
        "bootstrap_tail_probabilities": {
            "nominal_95_two_sided": [0.025, 0.975],
            "primary_bonferroni_two_sided": [
                alpha / (2.0 * n_primary), 1.0 - alpha / (2.0 * n_primary)
            ],
            "primary_bonferroni_tost": [
                alpha / n_primary, 1.0 - alpha / n_primary
            ],
            "secondary_bonferroni_two_sided": [
                alpha / (2.0 * n_secondary), 1.0 - alpha / (2.0 * n_secondary)
            ],
            "secondary_bonferroni_tost": [
                alpha / n_secondary, 1.0 - alpha / n_secondary
            ],
            "relaxation_tost": [
                alpha / max(1, len(radii)),
                1.0 - alpha / max(1, len(radii)),
            ],
        },
        "arms_converged": arms_converged,
        "relaxation_dispositions": relaxation_dispositions,
        "final_gap": gap.tolist() if gap is not None else None,
        "final_gap_error": gap_error.tolist() if gap_error is not None else None,
        "relaxation_chain_bootstrap": (
            relaxation_samples.tolist() if relaxation_samples is not None else None
        ),
        "stationarity": stationarity,
        "metropolis_stationarity": {
            label: metro[label]["stationarity"]
            for label in ("reference", "surrogate")
        },
        "hmc_stationarity_passed": hmc_stationarity_passed,
        "metropolis_stationarity_passed": metropolis_stationarity_passed,
        "primary_stationarity_passed": stationarity_passed,
        "stationarity_passed": stationarity_passed,
        "overlap_gate": overlap_gate,
        "overlap_passed": overlap_passed,
        "analysis_gate_passed": bool(
            arms_converged and stationarity_passed and overlap_passed
        ),
        "overall_disposition": disposition,
        "discard_fraction_used": discard,
        "bin_radii": np.asarray(radii).tolist(),
        "reference_grand_mean": np.asarray(ref_grand).tolist(),
        "reference_sem": np.asarray(ref_sem).tolist(),
        "estimators": {
            "linear": est["linear"]["value"],
            "linear_error": est["linear"]["error"],
            "linear_between_chain_sem": est["linear"]["between_chain_sem"],
            "forward_fep": est["forward_fep"]["value"],
            "reverse_fep": est["reverse_fep"]["value"],
            "mbar": est["mbar"]["value"],
            "mbar_error": est["mbar"]["error"],
            "direct_hmc": est["direct"]["value"],
            "direct_hmc_error": est["direct"]["error"],
            "direct_metropolis": metro["shift"],
            "direct_metropolis_error": metro["shift_error"],
        },
        "overlap": overlap,
        "mbar_kish": [est["mbar"]["kish_reference"], est["mbar"]["kish_surrogate"]],
        "relaxation_gap_start": first.get("arm_gap"),
        "relaxation_gap_at_primary_discard": selected.get("arm_gap"),
        "comparisons": comparisons,
        "all_within_bound": all(c["within_bound"] for c in primary_comparisons),
        "all_familywise_equivalent": all_familywise_equivalent,
        "any_underpowered": any(c["underpowered"] for c in primary_comparisons),
        "any_disagreement": any_familywise_difference,
        "secondary_alerts": [
            c for c in secondary_comparisons
            if c["familywise_difference_disposition"] == "meaningfully_different"
            or c["familywise_tost_disposition"] != "equivalent"
        ],
    }


def report_summary(s):
    print("\n  --- P0-1: end-to-end consistency ---")
    e = s["estimators"]
    print(f"  {'estimator':<24}" + "".join(f"{r:>12.2f} A" for r in s["bin_radii"]))
    for label, key, ekey in (("linear response", "linear", "linear_error"),
                             ("forward FEP", "forward_fep", None),
                             ("reverse FEP", "reverse_fep", None),
                             ("MBAR (both directions)", "mbar", "mbar_error"),
                             ("direct, HMC", "direct_hmc", "direct_hmc_error"),
                             ("direct, Metropolis", "direct_metropolis",
                              "direct_metropolis_error")):
        vals = np.atleast_1d(e[key])
        errs = np.atleast_1d(e[ekey]) if ekey else None
        cells = "".join(
            f"{v:>9.2f}" + (f" +/-{errs[i]:>4.2f}" if errs is not None else "     ")
            for i, v in enumerate(vals))
        print(f"  {label:<24}{cells}")

    print(f"  {'linear, between-chain SEM':<24}"
          + "".join(f"{v:>14.2f}" for v in np.atleast_1d(e["linear_between_chain_sem"])))
    print("    (the manuscript quoted the within-chain blocking error instead; "
          "compare the two)")

    o = s["overlap"]
    print(f"\n  overlap: work_overlap {o['work_overlap']:.2f}, "
          f"Kish forward {o['kish_forward']:.3f}, reverse {o['kish_reverse']:.3f}, "
          f"max weight {max(o['max_weight_forward'], o['max_weight_reverse']):.3f}")
    if s["relaxation_gap_start"] is not None:
        print(f"  bracketing gap (from_surrogate - from_reference): "
              f"{np.round(s['relaxation_gap_start'], 2)} at no discard -> "
              f"{np.round(s['relaxation_gap_at_primary_discard'], 2)} at "
              f"{s['discard_fraction_used']:.0%}")

    print(f"\n  prospectively frozen v2 bound +/-{s['equivalence_bound_pairs']} pairs; "
          f"direct minus reference-based:")
    for c in s["comparisons"]:
        verdict = c["nominal_disposition"]
        print(f"    {c['sampler']:<11} vs {c['against']:<7} {c['bin_radius']:.2f} A  "
              f"{c['difference']:+7.3f}  95% [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"{verdict}; familywise TOST={c['familywise_tost_disposition']}")
    if not s["arms_converged"]:
        print(f"\n  GATE FAILED: the two initialisations are still "
              f"{np.round(s['final_gap'], 2)} pairs apart (error "
              f"{np.round(s['final_gap_error'], 2)}) at the frozen primary discard, "
              f"against a bound of {s['equivalence_bound_pairs']}.")
        print("  The surrogate chains have not relaxed, so the direct estimate is "
              "not a measurement of the surrogate ensemble and none of the "
              "comparisons above can be read as estimator disagreement. Longer "
              "chains are needed; the size of the gap is the size of the artefact.")
        return
    if not s["stationarity_passed"]:
        print("\n  GATE FAILED: at least one analysed chain tail failed the "
              "pre-comparison stationarity diagnostic. Comparisons are not evaluable.")
        return
    if not s["overlap_passed"]:
        print("\n  GATE FAILED: the predeclared work/weight overlap thresholds were "
              "not met. Reweighting and MBAR comparisons are not evaluable.")
        return
    if s["any_disagreement"]:
        print("  VERDICT: at least one family-wise interval lies wholly outside "
              "the practical-equivalence region.")
    elif s["all_familywise_equivalent"]:
        print("  VERDICT: every comparison passes family-wise equivalence.")
    else:
        print("  VERDICT: neither family-wise equivalence nor meaningful difference "
              "is established -- inconclusive, not consistent.")


if __name__ == "__main__":
    main(
        run, default_config=DEFAULTS, name=EXPERIMENT_NAME,
        description="Reconcile reference-based estimators with direct sampling",
        protocol_path=PROTOCOL_PATH, protocol_name=EXPERIMENT_NAME,
        protocol_version=2,
    )
