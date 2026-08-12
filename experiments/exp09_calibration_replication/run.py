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
fields throughout, and D independent direct chains per field, giving a crossed
table of residuals in which the three error sources have different footprints:
a reference chain's error is constant down a row, a field's direct-chain error
is constant down a column, and any genuine field-dependent failure of the linear
prediction is *also* constant down a column but is not sampling noise. A two-way
decomposition therefore separates them, and each observed spread can be compared
against the spread its own quoted error predicts.

Three outcomes are possible and none is assumed. Row spread at the size the
reference error predicts means the single-chain offset was a reference
realisation: the manuscript's error propagation was right and its independence
claim was wrong. Column spread beyond what the direct-chain scatter allows means
the prediction fails in a field-dependent way, which is a real result and not a
noise artefact. Structure in neither, with the offset surviving, would mean a
flat estimator bias and the headline agreement would have to be withdrawn.

The field-dependent case has a candidate mechanism that this experiment also
tests. Linear response truncates a cumulant series, and the next term is
estimable from the same samples. If the column effects are the truncation
showing itself, adding that term should shrink them; if they persist, the
explanation is something else. Both residual tables are reported.
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
    # Eight fields to match exp07's table, so the column effects are estimated
    # on the same footing as the residuals under dispute; eight reference chains
    # so the row spread has seven degrees of freedom rather than the one the
    # original design had.
    "replication": {"n_reference_chains": 8, "n_direct_chains": 4},
    "perturbations": {"basis_r_min": 3.0, "basis_r_max": 6.6, "n_basis": 14,
                      "force_rms": 4.0e-3, "n_random": 6},
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

    # Chain-to-chain scatter of the direct means, pooled across fields. With
    # only D chains per field the per-field sd carries D-1 degrees of freedom
    # and is far too noisy to normalise a residual by; pooling over the fields
    # buys n_fields*(D-1) and costs only the assumption that a perturbation of
    # fixed force RMSE does not change how fast the chain mixes -- which the
    # per-field values printed above let a reader check.
    n_direct = int(rep["n_direct_chains"])
    pooled_sd = float(np.sqrt(np.mean([direct_means[n].var(ddof=1) for n, _ in fields])))
    direct_sem = pooled_sd / np.sqrt(n_direct)
    per_field_sem = {n: float(direct_means[n].std(ddof=1) / np.sqrt(n_direct))
                     for n, _ in fields}
    print(f"    direct chain SEM: pooled {direct_sem:.3f} pairs "
          f"({n_direct} chains x {len(fields)} fields), per field "
          f"{np.round(list(per_field_sem.values()), 3)}")

    # Residual, per (reference chain, field).
    #
    # The crude question -- "does the mean offset straddle zero?" -- cannot be
    # answered from the offsets alone, because every reference chain is compared
    # against the same direct chains, so the six offsets share a common
    # direct-chain error and are not independent draws. What the crossed design
    # does support is a decomposition. Writing r_ij for reference chain i and
    # field j, the three error sources enter with different footprints:
    #
    #   r_ij ~ b_j  +  d_j  -  e_i  +  noise
    #
    # where e_i is the reference chain's own error in <A> (constant down a row),
    # d_j the direct chains' error for that field (constant down a column), and
    # b_j any real, field-dependent failure of the linear prediction (also
    # constant down a column, and the only term that is not sampling noise).
    # Row effects therefore measure the reference realisation; column effects
    # measure d_j + b_j, and d_j has a known scale -- the direct-chain SEM --
    # so column structure larger than that scale is evidence for b_j.
    records = []
    for r in refs:
        row = []
        for name, field in fields:
            du = np.array([field.energy(c) for c in r["frames"]])
            pred = predict_shift(r["a"], du, temperature, n_resamples=300, seed=ctx.seed)
            predicted = float(np.asarray(pred.value))
            pred_err = float(np.asarray(pred.error))
            second = float(np.asarray(pred.second_order.value))
            measured = float(direct_means[name].mean() - r["mean"])
            err = float(np.sqrt(direct_sem**2 + r["error"]**2))
            residual = (measured - predicted) / err
            row.append(residual)
            records.append({
                "reference_chain": r["index"], "field": name,
                "predicted": predicted, "predicted_error": pred_err,
                "second_order": second, "second_order_ratio": float(pred.second_order_ratio),
                "predicted_2nd": predicted + second,
                "measured": measured, "measurement_error": err,
                "direct_chain_sem": direct_sem,
                "direct_chain_sem_this_field": per_field_sem[name],
                "reference_error": r["error"],
                "residual_sigma": residual,
                "residual_sigma_2nd": (measured - predicted - second) / err,
            })
        print(f"      reference {r['index']}: per-field residuals "
              f"{np.round(row, 2)}  row mean {np.mean(row):+.3f} sigma")

    field_names = [n for n, _ in fields]
    summary = {
        "n_reference_chains": len(refs),
        "n_direct_chains_per_field": int(rep["n_direct_chains"]),
        "n_fields": len(fields),
        "fields": field_names,
        "reference_mean_scatter": float(ref_means.std(ddof=1)),
        "reference_quoted_error_mean": float(np.mean([r["error"] for r in refs])),
        "direct_chain_sem_pooled": direct_sem,
        "direct_chain_sem_per_field": per_field_sem,
        "first_order": decompose(records, field_names, refs, "residual_sigma"),
        "second_order_corrected": decompose(records, field_names, refs,
                                            "residual_sigma_2nd"),
    }
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    report_summary(summary)
    return summary


def decompose(records, field_names, refs, key):
    """Two-way decomposition of the residual table into row and column effects.

    Returns the grand mean, the row (reference-chain) and column (field)
    effects, and for each the observed spread next to the spread that sampling
    noise alone predicts.  The comparison is the whole point: a row spread near
    its predicted value says the reference draw explains the offsets, and a
    column spread far above its predicted value says something field-dependent
    and real is left over.
    """
    table = {(x["reference_chain"], x["field"]): x[key] for x in records}
    r = np.array([[table[(ref["index"], name)] for name in field_names] for ref in refs])
    grand = float(r.mean())
    row_effects = r.mean(axis=1) - grand
    col_effects = r.mean(axis=0) - grand
    interaction = r - grand - row_effects[:, None] - col_effects[None, :]

    # Predicted spreads. Residuals are already in units of
    # err = sqrt(direct_sem^2 + ref_err^2), so a reference-chain error of size
    # ref_err contributes ref_err/err to every entry in its row, and likewise
    # for the direct chains down a column.
    ref_err = np.array([x["reference_error"] for x in records])
    direct_sem = np.array([x["direct_chain_sem"] for x in records])
    err = np.array([x["measurement_error"] for x in records])
    row_predicted = float(np.mean(ref_err / err))
    col_predicted = float(np.mean(direct_sem / err))

    n_rows, n_cols = r.shape
    col_excess = float(np.sqrt(max(col_effects.var(ddof=1) - col_predicted**2, 0.0)))
    return {
        "grand_mean": grand,
        "row_effects": row_effects.tolist(),
        "col_effects": col_effects.tolist(),
        "row_spread_observed": float(row_effects.std(ddof=1)),
        "row_spread_predicted": row_predicted,
        "col_spread_observed": float(col_effects.std(ddof=1)),
        "col_spread_predicted": col_predicted,
        "col_spread_excess": col_excess,
        "interaction_rms": float(np.sqrt((interaction**2).mean())),
        "residual_rms": float(np.sqrt((r**2).mean())),
        "n_negative": int((r < 0).sum()),
        "n_total": int(r.size),
        # Under the null "column structure is direct-chain noise only", the ratio
        # of observed to predicted column variance is F with (n_cols-1) and
        # n_cols*(n_direct-1) degrees of freedom; we report the ratio and let the
        # write-up carry the caveat rather than printing a p-value from an
        # approximation.
        "col_variance_ratio": float(col_effects.var(ddof=1) / max(col_predicted**2, 1e-30)),
        "shape": [n_rows, n_cols],
    }


def report_summary(s):
    print("\n  --- P0-2: what is the structure of the residuals? ---")
    print(f"  {s['n_reference_chains']} independent reference chains, "
          f"{s['n_direct_chains_per_field']} direct chains per field, "
          f"{s['n_fields']} fields")
    print(f"  reference <A> scatter across chains : {s['reference_mean_scatter']:.3f} pairs")
    print(f"  mean quoted reference error         : {s['reference_quoted_error_mean']:.3f} pairs")

    for label, key in (("linear prediction", "first_order"),
                       ("+ second-order term", "second_order_corrected")):
        d = s[key]
        print(f"\n  {label}: rms {d['residual_rms']:.2f} sigma, "
              f"{d['n_negative']}/{d['n_total']} negative")
        print(f"    grand mean                     {d['grand_mean']:+.3f} sigma")
        print(f"    row (reference-chain) effects  {np.round(d['row_effects'], 2)}")
        print(f"      spread {d['row_spread_observed']:.3f} observed vs "
              f"{d['row_spread_predicted']:.3f} predicted by the reference error")
        print(f"    column (field) effects         {np.round(d['col_effects'], 2)}  "
              f"({', '.join(s['fields'])})")
        print(f"      spread {d['col_spread_observed']:.3f} observed vs "
              f"{d['col_spread_predicted']:.3f} predicted by direct-chain noise "
              f"(variance ratio {d['col_variance_ratio']:.1f})")
        print(f"    interaction rms                {d['interaction_rms']:.3f} sigma")

    fo, so = s["first_order"], s["second_order_corrected"]
    print("\n  reading:")
    print(f"    - row spread {fo['row_spread_observed']:.2f} vs "
          f"{fo['row_spread_predicted']:.2f} predicted: the per-chain offset "
          f"{'is' if fo['row_spread_observed'] < 2 * fo['row_spread_predicted'] else 'is NOT'}"
          f" consistent with a reference-chain realisation")
    print(f"    - column spread {fo['col_spread_observed']:.2f} vs "
          f"{fo['col_spread_predicted']:.2f} predicted: field-dependent structure "
          f"{'beyond' if fo['col_variance_ratio'] > 4 else 'within'} direct-chain noise"
          + (f" (excess {fo['col_spread_excess']:.2f} sigma)"
             if fo["col_variance_ratio"] > 4 else ""))
    print(f"    - adding the second-order term takes the rms from "
          f"{fo['residual_rms']:.2f} to {so['residual_rms']:.2f} sigma and the "
          f"column spread from {fo['col_spread_observed']:.2f} to "
          f"{so['col_spread_observed']:.2f}")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Decompose the exp07 residuals into reference-chain, "
                     "field, and truncation contributions")
