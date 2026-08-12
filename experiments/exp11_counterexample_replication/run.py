#!/usr/bin/env python3
"""exp11 -- does the counterexample survive a change of construction chain?

Review item P0-3, which is correct. exp07 establishes that a null-space and an
aligned error field of identical force RMSE separate a target pair count by
10.11 pairs. It establishes it once. One construction trajectory built both
fields, one evaluation half measured both, the two force levels are the same
field direction rescaled rather than independent replicates, and the two direct
runs share a random stream whose covariance was never estimated. The result is
an existence proof and the manuscript should not have implied more.

This experiment attempts to replicate it with the construction cluster as the experimental
unit. A cluster is one construction trajectory, its own disjoint evaluation
trajectory, and its own direct chains -- nothing is shared between clusters
except the reference potential, the target definition and the basis geometry.
The contrast is then a per-cluster quantity with a distribution, and the claim
becomes a statement about that distribution rather than about one draw.

It also tests something exp07 could not, and which matters more than the
replication. The null-space construction makes the covariance vanish *on the
frames it was built from*. Whether the field is still quiet on frames it has
never seen is the whole question -- an in-sample-only null is a fitting
artefact, not a counterexample. Every number here is computed on the evaluation
trajectory, and the in-sample and out-of-sample covariances are reported side by
side so the suppression can be read as a factor rather than asserted. The same
applies to the force RMSE match: matching on the construction frames and
reporting the match is circular, so the match is verified out of sample too, and
a cluster whose out-of-sample match falls outside the prospectively specified
tolerance fails the primary cell. It is retained rather than selected away.

Prospectively specified for the corrected v3 run
-------------------------------------------------
*Estimand.* Per cluster c, the paired contrast
D_c = (measured shift of the aligned field) - (measured shift of the null
field) in the target bin, both measured against that cluster's own reference
mean. The primary quantity is the mean of D_c over clusters, with the cluster
as the unit of replication.

*Matching tolerance.* The two fields' force RMSE, measured on the evaluation
trajectory, must agree within 2%. exp07 achieved 0.52%, but that was a property
of its construction rather than a declared tolerance, and out of sample the
match is looser than in sample by construction.

*Minimum meaningful effect.* 1.0 pairs, fixed as a practical margin before the
v3 rerun. The legacy exp10 run did not establish resolution at this margin, so
this threshold defines scientific relevance but does not by itself establish
measurement power; an interval wider than it remains underpowered.

*Cluster count.* Eight. exp07 gives a single-cluster
contrast of 10.11 pairs with a within-cluster error near 0.9; the
between-cluster component is unmeasured -- it is what this experiment is for --
so eight was chosen to give seven degrees of freedom on it. If the observed
interval is wider than the minimum meaningful effect, the result is reported as
underpowered rather than as a replication.

*Aligned direction.* For this scalar target, orient the leading singular
direction so Cov(A, delta_U_aligned) <= 0 and its first-order shift is
non-negative. This prevents a BLAS/LAPACK sign convention from changing the
one-sided decision. A vector target with a degenerate leading singular subspace
would require a new orientation protocol.

*Decision rule.* The counterexample replicates if the 95% interval on the mean
of D_c, computed from the between-cluster scatter with t_(C-1), excludes the
minimum meaningful effect from below -- that is, if the lower limit exceeds 1.0
pairs. Practical absence requires the complete interval to lie inside +/-1 pair;
directional contradiction requires its upper limit to be below zero. Every
overlapping case is inconclusive. The legacy magnitude is called overstated only
if the new interval's upper limit is below 10.111 pairs.

*Controls.* The random field at the same force RMSE is descriptive. An arbitrary
direction is not mathematically constrained to land between null and aligned;
the legacy run incorrectly called this a positive control and then reinterpreted
two ordering failures post hoc. This version records but does not gate on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.response import predict_shift
from atomlab.analysis.statistics import (
    block_bootstrap,
    blocking_analysis,
    integrated_autocorrelation_time,
)
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import (
    AlignedPerturbation,
    NullSpacePerturbation,
    RandomShellPerturbation,
    ShellBasis,
    build_shell_design,
)
from atomlab.sampling import hybrid_monte_carlo
from experiments.common import (
    ExperimentContext,
    main,
    require_fresh_production_seed,
    require_frozen_protocol,
)
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

EXPERIMENT_NAME = "exp11_counterexample_replication_v3"
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "protocols" / "exp11_v3.json"

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    "observable": {"target_bin": [3.4, 3.9]},
    "sampling": {"n_construction": 1500, "n_evaluation": 1500, "n_direct": 900,
                 "n_leapfrog": 8, "step_size": 2e-3, "burn_in": 300,
                 "n_melt": 400, "n_anneal": 1000},
    "replication": {"n_clusters": 8, "n_direct_chains": 4},
    "perturbations": {"basis_r_min": 3.0, "basis_r_max": 6.6, "n_basis": 14,
                      "force_rms": 4.0e-3},
    "analysis": {"match_tolerance": 0.02, "minimum_effect_pairs": 1.0,
                 "max_null_to_aligned_covariance_ratio": 0.20},
}


def direct_seed(base: int, field_name: str, chain: int) -> tuple[int, str | None]:
    """Paired seed/start blocks; not an exactly synchronised CRN stream.

    HMC may consume acceptance uniforms differently under different potentials,
    so equal seeds do not imply proposal-by-proposal synchrony. They still
    identify a prospectively matched start/seed block for paired analysis.
    """
    if field_name in {"null", "aligned"}:
        return base + 10 + 37 * chain, f"base-{base}-chain-{chain}"
    return base + 500 + 37 * chain, None


def paired_contrast_sem(aligned, null) -> float:
    aligned = np.asarray(aligned, dtype=float)
    null = np.asarray(null, dtype=float)
    if aligned.shape != null.shape or aligned.ndim != 1 or aligned.size < 2:
        raise ValueError("aligned and null must be matching vectors with >=2 chains")
    differences = aligned - null
    return float(differences.std(ddof=1) / np.sqrt(differences.size))


def series_stationarity(values, *, label: str, n_sigma: float = 3.0) -> dict:
    """Fail closed when a claim-bearing scalar series drifts between halves."""
    values = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label}: series contains non-finite values")
    half = values.size // 2
    if half < 8:
        raise ValueError(f"{label}: need at least 16 frames, got {values.size}")
    first = blocking_analysis(values[:half])
    second = blocking_analysis(values[half:])
    error = float(np.hypot(first.error, second.error))
    difference = abs(float(first.value) - float(second.value))
    z = (difference / error if error > 0.0
         else (0.0 if difference == 0.0 else float("inf")))
    if z > n_sigma:
        raise RuntimeError(f"{label}: half-chain drift is {z:.2f} sigma")
    return {
        "first_half": float(first.value),
        "second_half": float(second.value),
        "half_difference_z": z,
        "n_sigma": n_sigma,
    }


def force_mse_series(field, frames) -> np.ndarray:
    """Per-frame mean squared force, preserving the paired frame unit."""
    values = []
    for frame in frames:
        forces = field.compute(frame, forces=True, virial=False).forces
        values.append(float(np.mean(np.asarray(forces, dtype=float) ** 2)))
    return np.asarray(values, dtype=float)


def delta_u_series(field, trajectory) -> np.ndarray:
    """Evaluate the perturbation coordinate on every retained direct frame."""
    return np.asarray(
        [field.energy(trajectory.frame(index))
         for index in range(trajectory.n_frames)],
        dtype=float,
    )


def paired_force_match(aligned_mse, null_mse, tolerance, *, seed, n_resamples=500):
    """Block-bootstrap the aligned/null force-RMSE ratio on shared frames."""
    aligned_mse = np.asarray(aligned_mse, dtype=float)
    null_mse = np.asarray(null_mse, dtype=float)
    if aligned_mse.shape != null_mse.shape:
        raise ValueError("aligned/null force series must share the same frames")
    joint = np.column_stack([aligned_mse, null_mse])
    means = joint.mean(axis=0)
    if np.any(means <= 0.0):
        raise ValueError("force-MSE means must be positive")
    # The ratio can carry a slow shared mode invisible in either marginal. Its
    # first-order influence is x/mu_x - y/mu_y (the square root only rescales it),
    # so include that coordinate when choosing the moving-block length.
    ratio_influence = joint[:, 0] / means[0] - joint[:, 1] / means[1]
    block_coordinates = np.column_stack([joint, ratio_influence])
    tau = integrated_autocorrelation_time(block_coordinates)
    block_length = int(np.clip(
        np.ceil(4.0 * tau), 1, max(1, len(joint) // 4)
    ))

    def statistic(sample):
        return np.sqrt(sample[:, 0].mean() / sample[:, 1].mean())

    estimate = block_bootstrap(
        joint, statistic, n_resamples=n_resamples, seed=seed,
        block_length=block_length,
    )
    samples = np.asarray(estimate.extra["samples"], dtype=float)
    ci = np.quantile(samples, [0.025, 0.975])
    passed = bool(ci[0] >= 1.0 - tolerance and ci[1] <= 1.0 + tolerance)
    return {
        "ratio": float(estimate.value),
        "error": float(estimate.error),
        "ci95": ci.tolist(),
        "block_length": int(estimate.extra["block_length"]),
        "maximum_coordinate_tau": float(tau),
        "passed": passed,
    }


def heldout_null_alignment_check(a, null_du, aligned_du, maximum_ratio, *, seed,
                                 n_resamples=500):
    """Manipulation check for covariance-nullness on held-out frames.

    The absolute null/aligned covariance ratio is recomputed on identical block
    bootstrap samples.  A point ratio alone cannot establish that the null
    construction transported out of sample; the complete upper interval must
    lie below the frozen margin.
    """
    joint = np.column_stack([
        np.asarray(a, dtype=float),
        np.asarray(null_du, dtype=float),
        np.asarray(aligned_du, dtype=float),
    ])
    centered = joint - joint.mean(axis=0, keepdims=True)
    # The statistic is a ratio of covariances, so its slow coordinates are the
    # centered products A*dU, not necessarily any marginal series. Multiplying
    # a slow mode by a rapidly changing observable can hide its autocorrelation
    # completely in both marginals while leaving the covariance product slow.
    block_coordinates = np.column_stack([
        joint,
        centered[:, 0] * centered[:, 1],
        centered[:, 0] * centered[:, 2],
    ])
    tau = integrated_autocorrelation_time(block_coordinates)
    block_length = int(np.clip(np.ceil(4.0 * tau), 1, max(1, len(joint) // 4)))

    def statistic(sample):
        null_cov = np.cov(sample[:, 0], sample[:, 1], ddof=1)[0, 1]
        aligned_cov = np.cov(sample[:, 0], sample[:, 2], ddof=1)[0, 1]
        if abs(aligned_cov) < 1e-30:
            return float("inf")
        return abs(null_cov / aligned_cov)

    estimate = block_bootstrap(
        joint, statistic, n_resamples=n_resamples, seed=seed,
        block_length=block_length,
    )
    samples = np.asarray(estimate.extra["samples"], dtype=float)
    ci = np.quantile(samples, [0.025, 0.975])
    return {
        "ratio": float(estimate.value),
        "error": float(estimate.error),
        "ci95": ci.tolist(),
        "maximum_ratio": float(maximum_ratio),
        "block_length": int(estimate.extra["block_length"]),
        "maximum_coordinate_tau": float(tau),
        "passed": bool(np.all(np.isfinite(ci)) and ci[1] <= maximum_ratio),
    }


def effect_disposition(ci, minimum_effect, *, legacy_effect=10.111):
    """Tri-state scientific interpretation without accepting the null by default."""
    lo, hi = map(float, ci)
    minimum_effect = float(minimum_effect)
    if lo > minimum_effect:
        status = "replicated"
    elif hi < 0.0:
        status = "directionally_contradicted"
    elif lo >= -minimum_effect and hi <= minimum_effect:
        status = "practically_absent"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "legacy_magnitude_overstated": bool(hi < float(legacy_effect)),
    }


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def sample(ctx, cfg, potential, n_samples, seed, *, label):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"], burn_in=s["burn_in"], seed=seed)
    if report.acceptance < 0.2:
        raise RuntimeError(f"{label}: acceptance {report.acceptance:.2f} too low")
    stationarity = check_equilibrated(
        traj, label=label, check_order=True, n_sigma=20.0 if ctx.quick else 3.0
    )
    return traj, report, stationarity


def run(ctx: ExperimentContext) -> dict:
    require_fresh_production_seed(ctx, expected=20260812)
    require_frozen_protocol(
        ctx, protocol_path=PROTOCOL_PATH,
        expected_protocol=EXPERIMENT_NAME, expected_version=3,
    )
    initial_cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    cutoff = ctx.config["potential"]["cutoff"]
    target = PairBinObservable(np.asarray(ctx.config["observable"]["target_bin"], float),
                               cutoff=cutoff)
    rep, pcfg = ctx.config["replication"], ctx.config["perturbations"]
    an = ctx.config["analysis"]
    centres = np.linspace(pcfg["basis_r_min"], pcfg["basis_r_max"], pcfg["n_basis"])
    basis = ShellBasis(centres, np.full(pcfg["n_basis"], float(centres[1] - centres[0])),
                       cutoff, r_on=cutoff - 1.0)

    n_clusters = 2 if ctx.quick else int(rep["n_clusters"])
    print(f"  {n_clusters} clusters, "
          f"{2 if ctx.quick else rep['n_direct_chains']} direct chains "
          f"per field, force RMSE {pcfg['force_rms']:.1e} eV/A")

    clusters = []
    for c in range(n_clusters):
        with ctx.timed(f"cluster_{c}"):
            clusters.append(one_cluster(ctx, c, initial_cfg, potential, target, basis,
                                        temperature, cutoff))
    summary = analyse(clusters, float(an["match_tolerance"]),
                      float(an["minimum_effect_pairs"]))
    ctx.save_json("clusters", [{k: v for k, v in c.items() if k != "raw"}
                               for c in clusters])
    ctx.save_npz(
        "raw",
        construction_a=np.stack([c["raw"]["construction_a"] for c in clusters]),
        construction_energy=np.stack([
            c["raw"]["construction_energy"] for c in clusters
        ]),
        construction_positions=np.stack([
            c["raw"]["construction_positions"] for c in clusters
        ]),
        construction_cells=np.stack([c["raw"]["construction_cells"] for c in clusters]),
        evaluation_a=np.stack([c["raw"]["evaluation_a"] for c in clusters]),
        evaluation_energy=np.stack([
            c["raw"]["evaluation_energy"] for c in clusters
        ]),
        evaluation_positions=np.stack([
            c["raw"]["evaluation_positions"] for c in clusters
        ]),
        evaluation_cells=np.stack([c["raw"]["evaluation_cells"] for c in clusters]),
        evaluation_du=np.stack([c["raw"]["evaluation_du"] for c in clusters]),
        evaluation_force_mse=np.stack([
            c["raw"]["evaluation_force_mse"] for c in clusters
        ]),
        direct_a=np.stack([c["raw"]["direct_a"] for c in clusters]),
        direct_energy=np.stack([c["raw"]["direct_energy"] for c in clusters]),
        direct_du=np.stack([c["raw"]["direct_du"] for c in clusters]),
        direct_positions=np.stack([c["raw"]["direct_positions"] for c in clusters]),
        direct_cells=np.stack([c["raw"]["direct_cells"] for c in clusters]),
        field_weights=np.stack([c["raw"]["field_weights"] for c in clusters]),
        field_names=np.asarray(["null", "aligned", "random"]),
        species=np.asarray(clusters[0]["raw"]["species"]),
        pbc=np.asarray(clusters[0]["raw"]["pbc"]),
    )
    ctx.save_json("summary", summary)
    report_summary(summary)
    return summary


def one_cluster(ctx, index, cfg, potential, target, basis, temperature, cutoff):
    """One construction trajectory, its own evaluation trajectory, its own chains.

    Nothing crosses cluster boundaries. The seeds are offset by a large stride
    per cluster so that no chain in one cluster shares a stream with any chain
    in another; sharing streams and then treating clusters as replicates is
    exactly the error this experiment exists to avoid repeating.
    """
    pcfg = ctx.config["perturbations"]
    rep = ctx.config["replication"]
    base = ctx.seed + 100000 * (index + 1)

    sampling = ctx.config["sampling"]
    # Cluster independence starts upstream: each cluster gets its own melt and
    # anneal seed, not merely a new production RNG from one shared configuration.
    cluster_cfg, equilibration = equilibrated_configuration(
        cfg, potential, temperature,
        n_melt=int(ctx.scaled("sampling.n_melt")),
        n_anneal=int(ctx.scaled("sampling.n_anneal")),
        n_leapfrog=sampling["n_leapfrog"],
        step_size=sampling["step_size"], seed=base,
        q6_target=1.0 if ctx.quick else 0.20,
    )
    construction, construction_report, construction_stationarity = sample(
        ctx, cluster_cfg, potential, ctx.scaled("sampling.n_construction"),
        seed=base + 1, label=f"cluster {index} construction"
    )
    cons_frames = [construction.frame(i) for i in range(construction.n_frames)]
    construction_a = target.evaluate_trajectory(construction).ravel()
    diagnostic_sigma = 20.0 if ctx.quick else 3.0
    construction_endpoint_stationarity = series_stationarity(
        construction_a, label=f"cluster {index} construction observable",
        n_sigma=diagnostic_sigma,
    )
    design = build_shell_design(cons_frames, basis)
    common = dict(basis=basis, design=design, cutoff=cutoff,
                  target_force_rms=pcfg["force_rms"], seed=base + 2)
    fields = {
        "null": NullSpacePerturbation(cons_frames, target, temperature, **common),
        "aligned": AlignedPerturbation(cons_frames, target, temperature, **common),
        "random": RandomShellPerturbation(cons_frames, target, temperature,
                                          **dict(common, seed=base + 3)),
    }

    # Evaluation trajectory: disjoint from the construction one, and everything
    # reported below is computed on it.
    evaluation, evaluation_report, evaluation_stationarity = sample(
        ctx, cluster_cfg, potential, ctx.scaled("sampling.n_evaluation"),
        seed=base + 4, label=f"cluster {index} evaluation"
    )
    eval_frames = [evaluation.frame(i) for i in range(evaluation.n_frames)]
    a_eval = target.evaluate_trajectory(evaluation).ravel()
    evaluation_endpoint_stationarity = series_stationarity(
        a_eval, label=f"cluster {index} evaluation observable",
        n_sigma=diagnostic_sigma,
    )
    ref_mean = float(a_eval.mean())
    ref_error = float(blocking_analysis(a_eval).error)

    n_direct = 2 if ctx.quick else int(rep["n_direct_chains"])
    start_indices = np.linspace(
        evaluation.n_frames // 2, evaluation.n_frames - 1, n_direct
    ).astype(int)
    out = {
        "cluster": index,
        "reference_mean": ref_mean,
        "reference_error": ref_error,
        "equilibration": dict(equilibration.__dict__),
        "construction": {
            "seed": base + 1,
            "acceptance": float(construction_report.acceptance),
            "final_step_size": float(construction_report.final_step_size),
            "stationarity": construction_stationarity,
            "observable_stationarity": construction_endpoint_stationarity,
        },
        "evaluation": {
            "seed": base + 4,
            "acceptance": float(evaluation_report.acceptance),
            "final_step_size": float(evaluation_report.final_step_size),
            "stationarity": evaluation_stationarity,
            "observable_stationarity": evaluation_endpoint_stationarity,
        },
        "fields": {},
    }
    raw_direct_a, raw_direct_energy, raw_direct_du = [], [], []
    raw_eval_du, raw_weights = [], []
    raw_direct_positions, raw_direct_cells = [], []
    raw_force_mse = []
    for name, field in fields.items():
        du_in = np.array([field.energy(c) for c in cons_frames])
        du_out = np.array([field.energy(c) for c in eval_frames])
        du_stationarity = series_stationarity(
            du_out, label=f"cluster {index} {name} evaluation delta-U",
            n_sigma=diagnostic_sigma,
        )
        force_mse = force_mse_series(field, eval_frames)
        force_mse_stationarity = series_stationarity(
            force_mse, label=f"cluster {index} {name} evaluation force MSE",
            n_sigma=diagnostic_sigma,
        )
        a_cons = target.evaluate_trajectory(construction).ravel()
        cov_in = float(np.cov(a_cons, du_in, ddof=1)[0, 1])
        cov_out = float(np.cov(a_eval, du_out, ddof=1)[0, 1])
        pred = predict_shift(a_eval, du_out, temperature, n_resamples=300, seed=base + 5)

        means, direct_a, direct_energy, direct_du = [], [], [], []
        direct_diagnostics = []
        direct_positions, direct_cells = [], []
        for d, start_index in enumerate(start_indices):
            # Aligned and null share a declared start/seed block and are analysed
            # through within-block differences. HMC acceptance branching can
            # desynchronise the streams, so this is not claimed as exact CRN.
            seed, pair_id = direct_seed(base, name, d)
            start = evaluation.frame(int(start_index))
            traj, report, stationarity = sample(
                ctx, start, potential + field, ctx.scaled("sampling.n_direct"),
                seed=seed, label=f"cluster {index} {name} {d}"
            )
            values = target.evaluate_trajectory(traj).ravel()
            observable_stationarity = series_stationarity(
                values, label=f"cluster {index} {name} {d} observable",
                n_sigma=diagnostic_sigma,
            )
            delta_u = delta_u_series(field, traj)
            direct_delta_u_stationarity = series_stationarity(
                delta_u, label=f"cluster {index} {name} {d} direct delta-U",
                n_sigma=diagnostic_sigma,
            )
            means.append(float(values.mean()))
            direct_a.append(values)
            direct_energy.append(np.asarray(traj.scalars["potential_energy"], dtype=float))
            direct_du.append(delta_u)
            direct_positions.append(np.asarray(traj.positions, dtype=float))
            direct_cells.append(np.asarray(traj.cells, dtype=float))
            direct_diagnostics.append({
                "chain": d,
                "seed": seed,
                "start_index": int(start_index),
                "paired_seed_start_id": pair_id,
                "acceptance": float(report.acceptance),
                "final_step_size": float(report.final_step_size),
                "stationarity": stationarity,
                "observable_stationarity": observable_stationarity,
                "delta_u_stationarity": direct_delta_u_stationarity,
            })
        means = np.array(means)
        out["fields"][name] = {
            "force_rmse_in_sample": field.force_rms(cons_frames),
            "force_rmse_out_of_sample": field.force_rms(eval_frames),
            "covariance_in_sample": cov_in,
            "covariance_out_of_sample": cov_out,
            "predicted": float(np.asarray(pred.value)),
            "predicted_error": float(np.asarray(pred.error)),
            "delta_u_stationarity": du_stationarity,
            "force_mse_stationarity": force_mse_stationarity,
            "direct_means": means.tolist(),
            "direct_diagnostics": direct_diagnostics,
            "measured": float(means.mean() - ref_mean),
            "measured_error": float(np.sqrt(means.var(ddof=1) / len(means)
                                            + ref_error**2)),
        }
        raw_direct_a.append(np.stack(direct_a))
        raw_direct_energy.append(np.stack(direct_energy))
        raw_direct_du.append(np.stack(direct_du))
        raw_direct_positions.append(np.stack(direct_positions))
        raw_direct_cells.append(np.stack(direct_cells))
        raw_eval_du.append(du_out)
        raw_weights.append(np.asarray(field.weights, dtype=float))
        raw_force_mse.append(force_mse)
    a, n = out["fields"]["aligned"], out["fields"]["null"]
    out["contrast"] = a["measured"] - n["measured"]
    paired_differences = np.asarray(a["direct_means"]) - np.asarray(n["direct_means"])
    out["paired_chain_contrasts"] = paired_differences.tolist()
    out["contrast_error"] = paired_contrast_sem(
        a["direct_means"], n["direct_means"]
    )
    tolerance = float(ctx.config["analysis"]["match_tolerance"])
    force_match = paired_force_match(
        raw_force_mse[1], raw_force_mse[0], tolerance,
        seed=base + 900,
    )
    out["force_match"] = force_match
    out["match_ratio"] = force_match["ratio"]
    out["force_match_passed"] = force_match["passed"]
    alignment_check = heldout_null_alignment_check(
        a_eval, raw_eval_du[0], raw_eval_du[1],
        float(ctx.config["analysis"]["max_null_to_aligned_covariance_ratio"]),
        seed=base + 901,
    )
    out["heldout_null_alignment_check"] = alignment_check
    out["heldout_null_alignment_passed"] = alignment_check["passed"]
    out["random_null_covariance_ratio"] = (
        abs(out["fields"]["random"]["covariance_out_of_sample"])
        / max(abs(n["covariance_out_of_sample"]), 1e-30)
    )
    out["aligned_null_predicted_ratio"] = (
        abs(a["predicted"]) / max(abs(n["predicted"]), 1e-30)
    )
    lo, hi = sorted((n["measured"], a["measured"]))
    out["random_between_null_and_aligned"] = bool(
        lo <= out["fields"]["random"]["measured"] <= hi
    )
    out["raw"] = {
        "construction_a": construction_a,
        "construction_energy": np.asarray(
            construction.scalars["potential_energy"], dtype=float
        ),
        "construction_positions": np.asarray(construction.positions, dtype=float),
        "construction_cells": np.asarray(construction.cells, dtype=float),
        "evaluation_a": a_eval,
        "evaluation_energy": np.asarray(
            evaluation.scalars["potential_energy"], dtype=float
        ),
        "evaluation_positions": np.asarray(evaluation.positions, dtype=float),
        "evaluation_cells": np.asarray(evaluation.cells, dtype=float),
        "evaluation_du": np.stack(raw_eval_du),
        "direct_a": np.stack(raw_direct_a),
        "direct_energy": np.stack(raw_direct_energy),
        "direct_du": np.stack(raw_direct_du),
        "direct_positions": np.stack(raw_direct_positions),
        "direct_cells": np.stack(raw_direct_cells),
        "field_weights": np.stack(raw_weights),
        "evaluation_force_mse": np.stack(raw_force_mse),
        "species": np.asarray(construction.template.species),
        "pbc": np.asarray(construction.template.pbc),
    }
    print(f"      cluster {index}: contrast {out['contrast']:+7.3f} "
          f"+/- {out['contrast_error']:.3f}   match {out['match_ratio']:.4f}   "
          f"aligned/null prediction ratio {out['aligned_null_predicted_ratio']:.0f}x")
    return out


def analyse(clusters, tolerance, minimum_effect):
    # Intention-to-test: matching failures fail the cell rather than selecting
    # inconvenient clusters out of the sampling distribution.
    failed_match = [c["cluster"] for c in clusters if not c["force_match_passed"]]
    failed_alignment = [c["cluster"] for c in clusters
                        if not c.get("heldout_null_alignment_passed", False)]
    contrasts = np.array([c["contrast"] for c in clusters])
    n = len(contrasts)
    mean = float(contrasts.mean())
    between = float(contrasts.std(ddof=1)) if n > 1 else float("nan")
    sem = between / np.sqrt(n) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else float("nan")
    ci = [mean - half, mean + half]
    primary_gate_passed = not failed_match and not failed_alignment
    scientific = effect_disposition(ci, minimum_effect)
    if not primary_gate_passed:
        scientific["status"] = "not_evaluable"
    return {
        "n_clusters_run": len(clusters),
        "n_clusters_analysed": n,
        "force_match_failures": failed_match,
        "heldout_null_alignment_failures": failed_alignment,
        "primary_gate_passed": primary_gate_passed,
        "match_tolerance": tolerance,
        "minimum_effect_pairs": minimum_effect,
        "contrasts": contrasts.tolist(),
        "contrast_mean": mean,
        "between_cluster_sd": between,
        "paired_chain_sem_per_cluster": [c["contrast_error"] for c in clusters],
        "sem": sem,
        "ci95": ci,
        "effect_disposition": scientific["status"],
        "replicates": scientific["status"] == "replicated",
        "legacy_magnitude_overstated": scientific["legacy_magnitude_overstated"],
        "random_null_covariance_ratio": [
            c["random_null_covariance_ratio"] for c in clusters
        ],
        "aligned_null_predicted_ratio": [
            c["aligned_null_predicted_ratio"] for c in clusters
        ],
        "random_between_null_and_aligned": [
            c["random_between_null_and_aligned"] for c in clusters
        ],
        "random_ordering_failures": [
            c["cluster"] for c in clusters
            if not c["random_between_null_and_aligned"]
        ],
        "null_measured": [c["fields"]["null"]["measured"] for c in clusters],
        "aligned_measured": [c["fields"]["aligned"]["measured"] for c in clusters],
        "random_measured": [c["fields"]["random"]["measured"] for c in clusters],
        "exp07_single_cluster_contrast": 10.111,
    }


def report_summary(s):
    print("\n  --- P0-3: does the counterexample replicate across clusters? ---")
    print(f"  {s['n_clusters_analysed']} of {s['n_clusters_run']} clusters analysed "
          f"(intention-to-test; force-RMSE tolerance {s['match_tolerance']:.0%})")
    if s["force_match_failures"]:
        print(f"  GATE FAILED: clusters {s['force_match_failures']} missed the "
              "force-RMSE tolerance; no confirmatory verdict is permitted")
    if s["heldout_null_alignment_failures"]:
        print(f"  GATE FAILED: clusters {s['heldout_null_alignment_failures']} failed "
              "the held-out covariance-null manipulation check")
    print(f"  per-cluster contrast (aligned - null): "
          f"{np.round(s['contrasts'], 2)}")
    print(f"  mean {s['contrast_mean']:+.3f} pairs, 95% CI "
          f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    print(f"  between-cluster sd {s['between_cluster_sd']:.3f}; paired-chain SEMs "
          f"{np.round(s['paired_chain_sem_per_cluster'], 3)}")
    print(f"  exp07 reported {s['exp07_single_cluster_contrast']:.2f} from one cluster")
    print(f"  null field shifts   : {np.round(s['null_measured'], 2)}")
    print(f"  random field shifts : {np.round(s['random_measured'], 2)}")
    print(f"  aligned field shifts: {np.round(s['aligned_measured'], 2)}")
    print(f"  descriptive aligned/null prediction ratios: "
          f"{np.round(s['aligned_null_predicted_ratio'], 0)}")
    print(f"  random ordering failures (descriptive, not a gate): "
          f"{s['random_ordering_failures']}")
    if not s["primary_gate_passed"]:
        verdict = "not evaluable: at least one force-match gate failed"
    elif s["effect_disposition"] == "replicated":
        verdict = (f"the contrast replicates -- the lower 95% limit "
                   f"{s['ci95'][0]:.2f} exceeds the prospectively specified minimum "
                   f"effect of {s['minimum_effect_pairs']:.1f} pairs")
    elif s["effect_disposition"] == "practically_absent":
        verdict = ("the complete interval lies inside the +/-minimum-effect "
                   "region; an effect of practical size is absent in this fixed cell")
    elif s["effect_disposition"] == "directionally_contradicted":
        verdict = ("the complete interval lies below zero; the prespecified "
                   "aligned-minus-null direction is contradicted")
    else:
        verdict = ("inconclusive: the interval permits more than one scientific "
                   "disposition, so neither replication nor absence is established")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main(
        run, default_config=DEFAULTS, name=EXPERIMENT_NAME,
        description="Replicate the designed counterexample across construction clusters",
        protocol_path=PROTOCOL_PATH, protocol_name=EXPERIMENT_NAME,
        protocol_version=3,
    )
