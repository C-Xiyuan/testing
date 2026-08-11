#!/usr/bin/env python3
"""exp07 -- designed counterexamples: prediction P3.

The sharpest test in the programme.  Three error fields are built with
**identical force RMSE** on the same reference ensemble:

* ``null`` -- coefficients in the null space of the covariance with the target
  observable, so first-order theory predicts *no* effect;
* ``aligned`` -- coefficients parallel to that covariance, so theory predicts
  the maximum effect obtainable at that force error;
* ``random`` -- an arbitrary field of the same force error, the control.

If force RMSE were an adequate summary of model quality the three would damage
the observable equally.  The theory says one will do essentially nothing and
another a great deal.  Direct simulation with each perturbed potential decides.

Two disciplines make this a test rather than a demonstration:

**Out of sample.**  The null-space construction makes the covariance vanish on
the frames it was built from, which is in-sample by definition.  The reference
trajectory is therefore split: the perturbations are constructed on one half and
every number reported -- force RMSE, predicted shift, covariance -- is computed
on the other.  A construction that only worked in-sample would show up here as a
predicted shift that is small on the construction half and not on the evaluation
half.

**Direct measurement.**  The predicted shifts are first-order theory.  They are
checked against explicit Monte Carlo sampling of each perturbed potential, with
block-bootstrap error bars, so the claim rests on a measurement rather than on
the formula being tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis, block_bootstrap
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import (
    aligned_perturbation,
    basis_energy_matrix,
    build_shell_basis,
    force_rms,
    null_space_perturbation,
    random_perturbation,
)
from atomlab.sampling import hybrid_monte_carlo
from experiments.common import ExperimentContext, main
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {
        "lattice_constant": 5.4,
        "reps": [3, 3, 3],
        "density": 0.0200,          # atoms / A^3, liquid argon
        "temperature": 120.0,       # K
    },
    "potential": {
        "epsilon": 0.0103,          # eV
        "sigma": 3.405,             # A
        "cutoff": 7.0,              # A
        "mode": "shifted_force",
    },
    "observable": {
        "r_min": 3.0,
        "r_max": 7.0,
        "n_bins": 8,
    },
    "sampling": {
        "n_reference": 1200,
        "n_direct": 1200,
        "n_leapfrog": 8,
        "step_size": 2e-3,
        "burn_in": 300,
    },
    "perturbations": {
        "basis_r_min": 3.0,
        "basis_r_max": 6.6,
        "n_basis": 14,
        "force_rms_levels": [1.0e-3, 2.0e-3, 4.0e-3],   # eV/A
        "n_random_seeds": 3,
    },
}


def build_system(ctx: ExperimentContext):
    s = ctx.config["system"]
    p = ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    potential = LennardJones(
        epsilon=p["epsilon"], sigma=p["sigma"], cutoff=p["cutoff"], mode=p["mode"]
    )
    return cfg, potential


def sample(ctx, cfg, potential, n_samples, seed):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg,
        potential,
        ctx.config["system"]["temperature"],
        n_samples=int(n_samples),
        n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"],
        burn_in=s["burn_in"],
        seed=seed,
    )
    if report.acceptance < 0.2:
        raise RuntimeError(
            f"sampler acceptance {report.acceptance:.2f} is too low to trust "
            f"({report.notes})"
        )
    return traj, report


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o = ctx.config["observable"]
    observable = PairBinObservable(
        np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1),
        cutoff=ctx.config["potential"]["cutoff"],
    )

    print(f"  system: {cfg.n_atoms} atoms, L = {cfg.cell[0, 0]:.2f} A, "
          f"rho = {cfg.density:.4f} /A^3, T = {temperature} K")

    # ---- reference ensemble, split into construction and evaluation halves --
    with ctx.timed("reference_sampling"):
        n_ref = int(ctx.scaled("sampling.n_reference"))
        ref, ref_report = sample(ctx, cfg, potential, n_ref, seed=ctx.seed + 1)
    print(f"    {ref_report}")

    frames = [ref.frame(i) for i in range(ref.n_frames)]
    half = ref.n_frames // 2
    construct_frames, evaluate_frames = frames[:half], frames[half:]

    a_all = observable.evaluate_trajectory(ref)
    a_construct, a_evaluate = a_all[:half], a_all[half:]
    a_reference = blocking_analysis(a_evaluate)
    print(f"    observable: {observable.n_bins} radial bins, "
          f"<A> in bin 0 = {a_reference.value[0]:.1f} +/- {a_reference.error[0]:.1f} pairs")

    # ---- build the designed fields on the construction half only ------------
    pcfg = ctx.config["perturbations"]
    basis = build_shell_basis(
        pcfg["basis_r_min"], pcfg["basis_r_max"], pcfg["n_basis"],
        cutoff=ctx.config["potential"]["cutoff"],
    )
    with ctx.timed("basis_energies"):
        energies_construct = basis_energy_matrix(basis, construct_frames)

    records = []
    for level in pcfg["force_rms_levels"]:
        built = []
        null, null_diag = null_space_perturbation(
            basis, a_construct, construct_frames,
            target_force_rms=level, seed=ctx.seed, energies=energies_construct,
        )
        built.append(("null", null, null_diag))

        aligned, aligned_diag = aligned_perturbation(
            basis, a_construct, construct_frames,
            target_force_rms=level, energies=energies_construct,
        )
        built.append(("aligned", aligned, aligned_diag))

        for k in range(int(pcfg["n_random_seeds"])):
            built.append((
                f"random{k}",
                random_perturbation(basis, construct_frames,
                                    target_force_rms=level, seed=ctx.seed + 100 + k),
                {},
            ))

        for name, perturbation, diagnostics in built:
            label = f"{name}@{level:.1e}"
            with ctx.timed(label):
                records.append(
                    evaluate_one(ctx, cfg, potential, perturbation, observable,
                                 evaluate_frames, a_evaluate, temperature,
                                 name=name, level=level, diagnostics=diagnostics)
                )
            r = records[-1]
            print(f"    {label:20s} F_rms(out) = {r['force_rms_out_of_sample']:.3e}  "
                  f"predicted = {r['predicted_max']:7.3f}  "
                  f"measured = {r['measured_max']:7.3f} +/- {r['measured_max_error']:.3f}")

    summary = summarise(records)
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    make_figure(ctx, records, observable)
    report(summary)
    return summary


def evaluate_one(ctx, cfg, potential, perturbation, observable, evaluate_frames,
                 a_evaluate, temperature, *, name, level, diagnostics) -> dict:
    """Predict, then measure, the observable shift caused by one perturbation."""
    f_rms_out = force_rms(perturbation, evaluate_frames)
    du = np.array([perturbation.energy(c) for c in evaluate_frames])

    prediction = predict_shift(a_evaluate, du, temperature, n_resamples=400, seed=ctx.seed)
    rw = reweight(a_evaluate, du, temperature, n_resamples=200, seed=ctx.seed)

    surrogate = potential + perturbation
    direct, direct_report = sample(
        ctx, cfg, surrogate, int(ctx.scaled("sampling.n_direct")), seed=ctx.seed + 2
    )
    a_direct = observable.evaluate_trajectory(direct)

    measured = a_direct.mean(axis=0) - a_evaluate.mean(axis=0)
    # The two ensembles are sampled independently, so their errors add in
    # quadrature; a paired estimator is not available here because the frames
    # do not correspond.
    err = np.hypot(blocking_analysis(a_direct).error, blocking_analysis(a_evaluate).error)

    predicted = np.asarray(prediction.value, dtype=float)
    peak = int(np.argmax(np.abs(predicted))) if np.abs(predicted).max() > 0 else int(np.argmax(np.abs(measured)))

    return {
        "name": name,
        "force_rms_level": level,
        "force_rms_out_of_sample": f_rms_out,
        "beta_sigma_dU": prediction.beta_sigma_dU,
        "second_order_ratio": np.asarray(prediction.second_order_ratio).tolist(),
        "max_correlation": float(np.abs(np.asarray(prediction.correlation)).max()),
        "predicted": predicted.tolist(),
        "predicted_error": np.asarray(prediction.error, dtype=float).tolist(),
        "predicted_max": float(np.abs(predicted).max()),
        "reweighted": np.asarray(rw.shift.value, dtype=float).tolist(),
        "reweighted_ess_fraction": rw.ess_fraction,
        "measured": measured.tolist(),
        "measured_error": err.tolist(),
        "measured_max": float(np.abs(measured).max()),
        "measured_max_error": float(err[peak]),
        "measured_at_predicted_peak": float(measured[peak]),
        "peak_bin": peak,
        "direct_acceptance": direct_report.acceptance,
        "construction_diagnostics": {k: float(v) if isinstance(v, (int, float)) else v
                                     for k, v in diagnostics.items()},
    }


def summarise(records) -> dict:
    """Reduce to the comparison the experiment exists to make."""
    by_level = {}
    for r in records:
        by_level.setdefault(f"{r['force_rms_level']:.1e}", []).append(r)

    out = {"levels": {}}
    for level, group in by_level.items():
        null = [r for r in group if r["name"] == "null"]
        aligned = [r for r in group if r["name"] == "aligned"]
        randoms = [r for r in group if r["name"].startswith("random")]
        if not (null and aligned):
            continue
        n, a = null[0], aligned[0]
        random_mean = float(np.mean([r["measured_max"] for r in randoms])) if randoms else float("nan")
        out["levels"][level] = {
            "force_rms_out_of_sample": {
                "null": n["force_rms_out_of_sample"],
                "aligned": a["force_rms_out_of_sample"],
            },
            "measured_max": {
                "null": n["measured_max"],
                "null_error": n["measured_max_error"],
                "aligned": a["measured_max"],
                "aligned_error": a["measured_max_error"],
                "random_mean": random_mean,
            },
            "predicted_max": {"null": n["predicted_max"], "aligned": a["predicted_max"]},
            "aligned_over_null_measured": (
                a["measured_max"] / n["measured_max"] if n["measured_max"] > 0 else float("inf")
            ),
            "null_measured_within_error": abs(n["measured_max"]) < 2.0 * n["measured_max_error"],
        }
    return out


def make_figure(ctx, records, observable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from atomlab.utils import plotting as P

    P.use_style()
    levels = sorted({r["force_rms_level"] for r in records})
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 2.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, level in zip(axes, levels):
        group = [r for r in records if r["force_rms_level"] == level]
        for idx, name in enumerate(["aligned", "random0", "null"]):
            match = [r for r in group if r["name"] == name]
            if not match:
                continue
            r = match[0]
            P.series(ax, observable.centres, r["measured"], r["measured_error"],
                     index=idx, label=name, fill=False)
            ax.plot(observable.centres, r["predicted"], color=P.series_color(idx),
                    linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axhline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
        ax.set_title(f"force RMSE = {level:.0e} eV/Å")
        ax.set_xlabel("r (Å)")
    axes[0].set_ylabel("shift in pair count")
    axes[0].legend(ncol=1)
    fig.suptitle("Identical force error, opposite consequences "
                 "(points: measured; dotted: first-order prediction)", y=1.04)
    P.save_figure(fig, ctx.figure_path("counterexamples"))
    plt.close(fig)


def report(summary):
    print("\n  --- P3: designed counterexamples ---")
    for level, s in summary["levels"].items():
        null_flag = "consistent with zero" if s["null_measured_within_error"] else "NOT zero"
        print(f"  at force RMSE {level} eV/A:")
        print(f"    null-space field : {s['measured_max']['null']:+.3f} "
              f"+/- {s['measured_max']['null_error']:.3f} pairs  ({null_flag})")
        print(f"    aligned field    : {s['measured_max']['aligned']:+.3f} "
              f"+/- {s['measured_max']['aligned_error']:.3f} pairs")
        print(f"    random control   : {s['measured_max']['random_mean']:+.3f} pairs (mean)")
        print(f"    aligned / null   : {s['aligned_over_null_measured']:.1f}x "
              f"at identical reported force error")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Designed error fields with matched force RMSE and opposite consequences")
