#!/usr/bin/env python3
"""exp09 -- is the "1.01 sigma" agreement a calibration pass, or one shared offset?

Written in response to review item P0-2, which is correct.

The manuscript reported that predicted and measured observable shifts agree to
1.01 sigma rms over eight designed error fields, and presented that as evidence
the prediction is calibrated. Decomposing those eight residuals shows otherwise:
all eight are negative, with a common offset of -0.861 sigma and a scatter about
that offset of only 0.571 sigma. An rms near one is therefore a systematic bias
plus a scatter *smaller* than the quoted error bars -- not eight independent
calibration cases landing where they should.

There is an obvious candidate cause, and it is testable. Every field's measured
shift is (direct chain mean - reference chain mean), and all eight share one
reference chain. If that chain's own realisation of <A> is high by epsilon, every
measured shift is low by epsilon and every residual acquires the same negative
offset. Under that explanation the offset is a property of the reference draw,
not of the estimator, and repeating with independent reference chains should
give offsets that scatter around zero.

This experiment does that. R independent reference chains, the same designed
fields throughout, and D independent direct chains per field, so that:

  - the offset can be computed separately per reference chain and its
    distribution over chains inspected;
  - direct-chain noise is separated from reference-chain noise;
  - the experimental unit for calibration becomes the reference chain, which is
    what the review asks for.

Two outcomes are informative and neither is assumed. If the per-chain offsets
straddle zero with the spread the reference error predicts, the estimator is
unbiased and the manuscript's error propagation was correct but its independence
claim was not. If they are consistently negative across independent reference
draws, the estimator has a real systematic bias and the headline agreement must
be withdrawn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    "sampling": {"n_construction": 2000, "n_reference": 1500, "n_direct": 1500,
                 "n_leapfrog": 8, "step_size": 2e-3, "burn_in": 400,
                 "n_melt": 400, "n_anneal": 1000},
    "replication": {"n_reference_chains": 6, "n_direct_chains": 3},
    "perturbations": {"basis_r_min": 3.0, "basis_r_max": 6.6, "n_basis": 14,
                      "force_rms": 4.0e-3, "n_random": 2},
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
    return traj, report


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    cutoff = ctx.config["potential"]["cutoff"]
    target = PairBinObservable(np.asarray(ctx.config["observable"]["target_bin"], float),
                               cutoff=cutoff)
    rep, pcfg = ctx.config["replication"], ctx.config["perturbations"]

    with ctx.timed("equilibration"):
        s_cfg = ctx.config["sampling"]
        cfg, eq = equilibrated_configuration(
            cfg, potential, temperature,
            n_melt=int(ctx.get("sampling.n_melt")), n_anneal=int(ctx.get("sampling.n_anneal")),
            n_leapfrog=s_cfg["n_leapfrog"], step_size=s_cfg["step_size"], seed=ctx.seed)
    print(f"    {eq}")

    # One construction chain, disjoint from every reference chain used below.
    with ctx.timed("construction"):
        cons, _ = sample(ctx, cfg, potential, ctx.scaled("sampling.n_construction"),
                         seed=ctx.seed + 900, label="construction")
        cons_frames = [cons.frame(i) for i in range(cons.n_frames)]
        centres = np.linspace(pcfg["basis_r_min"], pcfg["basis_r_max"], pcfg["n_basis"])
        basis = ShellBasis(centres, np.full(pcfg["n_basis"], float(centres[1] - centres[0])),
                           cutoff, r_on=cutoff - 1.0)
        design = build_shell_design(cons_frames, basis)
        common = dict(basis=basis, design=design, cutoff=cutoff,
                      target_force_rms=pcfg["force_rms"], seed=ctx.seed)
        fields = [
            ("null", NullSpacePerturbation(cons_frames, target, temperature, **common)),
            ("aligned", AlignedPerturbation(cons_frames, target, temperature, **common)),
        ]
        for k in range(int(pcfg["n_random"])):
            fields.append((f"random{k}", RandomShellPerturbation(
                cons_frames, target, temperature, **dict(common, seed=ctx.seed + 100 + k))))
    print(f"    {len(fields)} fields at force RMSE {pcfg['force_rms']:.1e} eV/A")

    # Independent direct chains per field, and independent reference chains.
    with ctx.timed("direct_chains"):
        direct_means = {}
        for name, field in fields:
            vals = []
            for d in range(int(rep["n_direct_chains"])):
                traj, _ = sample(ctx, cfg, potential + field,
                                 ctx.scaled("sampling.n_direct"),
                                 seed=ctx.seed + 5000 + 17 * d, label=f"{name} direct {d}",
                                 check=False)
                vals.append(float(target.evaluate_trajectory(traj).ravel().mean()))
            direct_means[name] = np.array(vals)
            print(f"      {name:9s} direct chain means: {np.round(vals, 3)}")

    with ctx.timed("reference_chains"):
        refs = []
        for r in range(int(rep["n_reference_chains"])):
            traj, _ = sample(ctx, cfg, potential, ctx.scaled("sampling.n_reference"),
                             seed=ctx.seed + 2000 + 31 * r, label=f"reference {r}")
            a = target.evaluate_trajectory(traj).ravel()
            frames = [traj.frame(i) for i in range(traj.n_frames)]
            refs.append({"index": r, "mean": float(a.mean()),
                         "error": float(blocking_analysis(a).error),
                         "a": a, "frames": frames})
            print(f"      reference {r}: <A> = {refs[-1]['mean']:.3f} "
                  f"+/- {refs[-1]['error']:.3f}")

    ref_means = np.array([r["mean"] for r in refs])
    print(f"    reference chain means: mean {ref_means.mean():.3f}, "
          f"scatter {ref_means.std(ddof=1):.3f}, "
          f"mean quoted error {np.mean([r['error'] for r in refs]):.3f}")

    # Residual, per (reference chain, field). The offset per reference chain is
    # the quantity the review asks about.
    records, offsets = [], []
    for r in refs:
        row = []
        for name, field in fields:
            du = np.array([field.energy(c) for c in r["frames"]])
            pred = predict_shift(r["a"], du, temperature, n_resamples=300, seed=ctx.seed)
            predicted = float(np.asarray(pred.value))
            pred_err = float(np.asarray(pred.error))
            measured = float(direct_means[name].mean() - r["mean"])
            direct_sem = float(direct_means[name].std(ddof=1)
                               / np.sqrt(len(direct_means[name])))
            err = float(np.sqrt(direct_sem**2 + r["error"]**2))
            residual = (measured - predicted) / err
            row.append(residual)
            records.append({
                "reference_chain": r["index"], "field": name,
                "predicted": predicted, "predicted_error": pred_err,
                "measured": measured, "measurement_error": err,
                "direct_chain_sem": direct_sem, "reference_error": r["error"],
                "residual_sigma": residual,
            })
        offsets.append(float(np.mean(row)))
        print(f"      reference {r['index']}: per-field residuals "
              f"{np.round(row, 2)}  offset {offsets[-1]:+.3f} sigma")

    offsets = np.array(offsets)
    all_res = np.array([x["residual_sigma"] for x in records])
    summary = {
        "n_reference_chains": len(refs),
        "n_direct_chains_per_field": int(rep["n_direct_chains"]),
        "n_fields": len(fields),
        "reference_mean_scatter": float(ref_means.std(ddof=1)),
        "reference_quoted_error_mean": float(np.mean([r["error"] for r in refs])),
        "per_chain_offset_sigma": offsets.tolist(),
        "offset_mean": float(offsets.mean()),
        "offset_scatter": float(offsets.std(ddof=1)),
        "offset_sem": float(offsets.std(ddof=1) / np.sqrt(len(offsets))),
        "n_offsets_negative": int((offsets < 0).sum()),
        "all_residuals_rms": float(np.sqrt((all_res**2).mean())),
        "scatter_within_chain": float(np.mean([
            np.std([x["residual_sigma"] for x in records if x["reference_chain"] == r["index"]],
                   ddof=1) for r in refs])),
    }
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    report_summary(summary)
    return summary


def report_summary(s):
    print("\n  --- P0-2: is the offset a property of the reference draw? ---")
    print(f"  {s['n_reference_chains']} independent reference chains, "
          f"{s['n_direct_chains_per_field']} direct chains per field, "
          f"{s['n_fields']} fields")
    print(f"  reference <A> scatter across chains : {s['reference_mean_scatter']:.3f} pairs")
    print(f"  mean quoted reference error         : {s['reference_quoted_error_mean']:.3f} pairs")
    print(f"  per-chain offsets (sigma)           : "
          f"{np.round(s['per_chain_offset_sigma'], 3)}")
    print(f"  offset mean {s['offset_mean']:+.3f} +/- {s['offset_sem']:.3f} sigma "
          f"({s['n_offsets_negative']}/{s['n_reference_chains']} negative)")
    print(f"  scatter of offsets across chains    : {s['offset_scatter']:.3f} sigma")
    print(f"  residual scatter within a chain     : {s['scatter_within_chain']:.3f} sigma")
    verdict = ("offsets straddle zero -- the single-chain offset was a reference "
               "realisation, not an estimator bias"
               if abs(s["offset_mean"]) < 2.0 * s["offset_sem"] else
               "offsets are systematically non-zero across independent reference "
               "chains -- the estimator carries a real bias and the headline "
               "agreement must be withdrawn")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Separate reference-chain realisation from estimator bias")
