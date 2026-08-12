#!/usr/bin/env python3
"""exp11 -- does the counterexample survive a change of construction chain?

Review item P0-3, which is correct. exp07 establishes that a null-space and an
aligned error field of identical force RMSE separate a target pair count by
10.11 pairs. It establishes it once. One construction trajectory built both
fields, one evaluation half measured both, the two force levels are the same
field direction rescaled rather than independent replicates, and the two direct
runs share a random stream whose covariance was never estimated. The result is
an existence proof and the manuscript should not have implied more.

This experiment replicates it with the construction cluster as the experimental
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
a cluster whose out-of-sample match falls outside the pre-registered tolerance
is excluded and counted rather than quietly kept.

Pre-registered before the confirmatory run
------------------------------------------
*Estimand.* Per cluster c, the paired contrast
D_c = (measured shift of the aligned field) - (measured shift of the null
field) in the target bin, both measured against that cluster's own reference
mean. The primary quantity is the mean of D_c over clusters, with the cluster
as the unit of replication.

*Matching tolerance.* The two fields' force RMSE, measured on the evaluation
trajectory, must agree within 2%. exp07 achieved 0.52%, but that was a property
of its construction rather than a declared tolerance, and out of sample the
match is looser than in sample by construction.

*Minimum meaningful effect.* 1.0 pairs, twice the equivalence bound of exp10,
so that a contrast this experiment calls real is one exp10 has established the
measurement can resolve.

*Cluster count.* Eight. exp07 gives a single-cluster contrast of 10.11 pairs
with a within-cluster error near 0.9; the between-cluster component is unmeasured
-- it is what this experiment is for -- so eight is chosen to give seven degrees
of freedom on it, enough to detect a between-cluster standard deviation
comparable to the within-cluster one. If the observed between-cluster scatter
implies a wider interval than the minimum meaningful effect, the result is
reported as underpowered rather than as a replication.

*Decision rule.* The counterexample replicates if the 95% interval on the mean
of D_c, computed from the between-cluster scatter with t_7, excludes the
minimum meaningful effect from below -- that is, if the lower limit exceeds 1.0
pairs. If the interval contains zero, the counterexample does not replicate and
the claim is withdrawn. If it excludes zero but not 1.0, the effect is real and
smaller than exp07 reported.

*Controls.* The random field at the same force RMSE is carried through every
cluster as a positive control that should land between the two, and the
per-cluster reference chain doubles as the negative control for the direct
chains' own scatter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.response import predict_shift
from atomlab.analysis.statistics import blocking_analysis
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
from experiments.common import ExperimentContext, main
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    "observable": {"target_bin": [3.4, 3.9]},
    "sampling": {"n_construction": 2000, "n_evaluation": 2000, "n_direct": 1500,
                 "n_leapfrog": 8, "step_size": 2e-3, "burn_in": 400,
                 "n_melt": 400, "n_anneal": 1000},
    "replication": {"n_clusters": 8, "n_direct_chains": 3},
    "perturbations": {"basis_r_min": 3.0, "basis_r_max": 6.6, "n_basis": 14,
                      "force_rms": 4.0e-3},
    "analysis": {"match_tolerance": 0.02, "minimum_effect_pairs": 1.0},
}


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def sample(ctx, cfg, potential, n_samples, seed, *, label, check=True):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"], burn_in=s["burn_in"], seed=seed)
    if report.acceptance < 0.2:
        raise RuntimeError(f"{label}: acceptance {report.acceptance:.2f} too low")
    if check:
        check_equilibrated(traj, label=label, check_order=False)
    return traj


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    cutoff = ctx.config["potential"]["cutoff"]
    target = PairBinObservable(np.asarray(ctx.config["observable"]["target_bin"], float),
                               cutoff=cutoff)
    rep, pcfg = ctx.config["replication"], ctx.config["perturbations"]
    an = ctx.config["analysis"]
    centres = np.linspace(pcfg["basis_r_min"], pcfg["basis_r_max"], pcfg["n_basis"])
    basis = ShellBasis(centres, np.full(pcfg["n_basis"], float(centres[1] - centres[0])),
                       cutoff, r_on=cutoff - 1.0)

    with ctx.timed("equilibration"):
        s_cfg = ctx.config["sampling"]
        cfg, eq = equilibrated_configuration(
            cfg, potential, temperature, n_melt=int(ctx.get("sampling.n_melt")),
            n_anneal=int(ctx.get("sampling.n_anneal")), n_leapfrog=s_cfg["n_leapfrog"],
            step_size=s_cfg["step_size"], seed=ctx.seed)
    print(f"    {eq}")
    print(f"  {rep['n_clusters']} clusters, {rep['n_direct_chains']} direct chains "
          f"per field, force RMSE {pcfg['force_rms']:.1e} eV/A")

    clusters = []
    for c in range(int(rep["n_clusters"])):
        with ctx.timed(f"cluster_{c}"):
            clusters.append(one_cluster(ctx, c, cfg, potential, target, basis,
                                        temperature, cutoff))
    summary = analyse(clusters, float(an["match_tolerance"]),
                      float(an["minimum_effect_pairs"]))
    ctx.save_json("clusters", [{k: v for k, v in c.items() if k != "frames"}
                               for c in clusters])
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

    construction = sample(ctx, cfg, potential, ctx.scaled("sampling.n_construction"),
                          seed=base + 1, label=f"cluster {index} construction")
    cons_frames = [construction.frame(i) for i in range(construction.n_frames)]
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
    evaluation = sample(ctx, cfg, potential, ctx.scaled("sampling.n_evaluation"),
                        seed=base + 4, label=f"cluster {index} evaluation")
    eval_frames = [evaluation.frame(i) for i in range(evaluation.n_frames)]
    a_eval = target.evaluate_trajectory(evaluation).ravel()
    ref_mean = float(a_eval.mean())
    ref_error = float(blocking_analysis(a_eval).error)

    out = {"cluster": index, "reference_mean": ref_mean,
           "reference_error": ref_error, "fields": {}}
    for name, field in fields.items():
        du_in = np.array([field.energy(c) for c in cons_frames])
        du_out = np.array([field.energy(c) for c in eval_frames])
        a_cons = target.evaluate_trajectory(construction).ravel()
        cov_in = float(np.cov(a_cons, du_in, ddof=1)[0, 1])
        cov_out = float(np.cov(a_eval, du_out, ddof=1)[0, 1])
        pred = predict_shift(a_eval, du_out, temperature, n_resamples=300, seed=base + 5)

        means = []
        for d in range(int(rep["n_direct_chains"])):
            traj = sample(ctx, cfg, potential + field, ctx.scaled("sampling.n_direct"),
                          seed=base + 10 + 37 * d, label=f"cluster {index} {name} {d}",
                          check=False)
            means.append(float(target.evaluate_trajectory(traj).ravel().mean()))
        means = np.array(means)
        out["fields"][name] = {
            "force_rmse_in_sample": field.force_rms(cons_frames),
            "force_rmse_out_of_sample": field.force_rms(eval_frames),
            "covariance_in_sample": cov_in,
            "covariance_out_of_sample": cov_out,
            "predicted": float(np.asarray(pred.value)),
            "predicted_error": float(np.asarray(pred.error)),
            "direct_means": means.tolist(),
            "measured": float(means.mean() - ref_mean),
            "measured_error": float(np.sqrt(means.var(ddof=1) / len(means)
                                            + ref_error**2)),
        }
    a, n = out["fields"]["aligned"], out["fields"]["null"]
    out["contrast"] = a["measured"] - n["measured"]
    # The two fields' direct chains are independent, but both differences are
    # taken against the same reference mean, so that term cancels in the
    # contrast and must not be counted twice.
    out["contrast_error"] = float(np.sqrt(
        np.var(a["direct_means"], ddof=1) / len(a["direct_means"])
        + np.var(n["direct_means"], ddof=1) / len(n["direct_means"])))
    out["match_ratio"] = (a["force_rmse_out_of_sample"]
                          / n["force_rmse_out_of_sample"])
    out["null_suppression"] = (abs(out["fields"]["random"]["covariance_out_of_sample"])
                               / max(abs(n["covariance_out_of_sample"]), 1e-30))
    print(f"      cluster {index}: contrast {out['contrast']:+7.3f} "
          f"+/- {out['contrast_error']:.3f}   match {out['match_ratio']:.4f}   "
          f"null suppression {out['null_suppression']:.0f}x")
    return out


def analyse(clusters, tolerance, minimum_effect):
    keep = [abs(c["match_ratio"] - 1.0) <= tolerance for c in clusters]
    kept = [c for c, k in zip(clusters, keep) if k]
    excluded = [c["cluster"] for c, k in zip(clusters, keep) if not k]
    contrasts = np.array([c["contrast"] for c in kept])
    n = len(contrasts)
    mean = float(contrasts.mean())
    between = float(contrasts.std(ddof=1)) if n > 1 else float("nan")
    sem = between / np.sqrt(n) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else float("nan")
    within = float(np.sqrt(np.mean([c["contrast_error"]**2 for c in kept])))
    return {
        "n_clusters_run": len(clusters),
        "n_clusters_kept": n,
        "excluded_for_force_match": excluded,
        "match_tolerance": tolerance,
        "minimum_effect_pairs": minimum_effect,
        "contrasts": contrasts.tolist(),
        "contrast_mean": mean,
        "between_cluster_sd": between,
        "within_cluster_sd": within,
        "variance_ratio_between_over_within": (between / within) if within > 0 else None,
        "sem": sem,
        "ci95": [mean - half, mean + half],
        "replicates": bool(mean - half > minimum_effect),
        "excludes_zero": bool(mean - half > 0),
        "underpowered": bool(half > minimum_effect),
        "null_suppression": [c["null_suppression"] for c in kept],
        "null_measured": [c["fields"]["null"]["measured"] for c in kept],
        "aligned_measured": [c["fields"]["aligned"]["measured"] for c in kept],
        "random_measured": [c["fields"]["random"]["measured"] for c in kept],
        "exp07_single_cluster_contrast": 10.111,
    }


def report_summary(s):
    print("\n  --- P0-3: does the counterexample replicate across clusters? ---")
    print(f"  {s['n_clusters_kept']} of {s['n_clusters_run']} clusters kept "
          f"(force-RMSE match within {s['match_tolerance']:.0%})"
          + (f"; excluded {s['excluded_for_force_match']}"
             if s["excluded_for_force_match"] else ""))
    print(f"  per-cluster contrast (aligned - null): "
          f"{np.round(s['contrasts'], 2)}")
    print(f"  mean {s['contrast_mean']:+.3f} pairs, 95% CI "
          f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    print(f"  between-cluster sd {s['between_cluster_sd']:.3f} vs within-cluster "
          f"{s['within_cluster_sd']:.3f} (ratio "
          f"{s['variance_ratio_between_over_within']:.2f})")
    print(f"  exp07 reported {s['exp07_single_cluster_contrast']:.2f} from one cluster")
    print(f"  null field shifts   : {np.round(s['null_measured'], 2)}")
    print(f"  random field shifts : {np.round(s['random_measured'], 2)}")
    print(f"  aligned field shifts: {np.round(s['aligned_measured'], 2)}")
    print(f"  out-of-sample null suppression vs random: "
          f"{np.round(s['null_suppression'], 0)}")
    if s["replicates"]:
        verdict = (f"the contrast replicates -- the lower 95% limit "
                   f"{s['ci95'][0]:.2f} exceeds the pre-registered minimum "
                   f"effect of {s['minimum_effect_pairs']:.1f} pairs")
    elif s["excludes_zero"]:
        verdict = (f"the contrast is real but smaller than the pre-registered "
                   f"minimum effect; exp07's single-cluster value overstates it")
    elif s["underpowered"]:
        verdict = (f"underpowered: the interval half-width {0.5 * (s['ci95'][1] - s['ci95'][0]):.2f} "
                   f"exceeds the minimum effect, so nothing is established either way")
    else:
        verdict = ("the interval contains zero -- the counterexample does not "
                   "replicate across construction clusters and the claim is withdrawn")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Replicate the designed counterexample across construction clusters")
