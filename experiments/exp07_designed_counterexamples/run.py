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
    AlignedPerturbation,
    NullSpacePerturbation,
    RandomShellPerturbation,
    ShellBasis,
    build_shell_design,
)
from atomlab.potentials.perturbations import observable_matrix
from atomlab.sampling import hybrid_monte_carlo
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
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
        # The designed fields target a SCALAR: the pair count in one bin across
        # the first peak. A scalar target makes the covariance a K-vector rather
        # than a K x J matrix, which is estimable from far fewer frames, and it
        # makes the claim sharper -- one number is left alone while another
        # error field of identical force RMSE moves it.
        "target_bin": [3.4, 3.9],
    },
    "sampling": {
        "n_reference": 4000,
        "n_direct": 3000,
        "n_leapfrog": 8,
        "step_size": 2e-3,
        "burn_in": 300,
        "n_melt": 400,
        "n_anneal": 1000,
    },
    "perturbations": {
        "basis_r_min": 3.0,
        "basis_r_max": 6.6,
        "n_basis": 14,
        "force_rms_levels": [1.0e-3, 4.0e-3],   # eV/A
        "n_random_seeds": 2,
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


def sample(ctx, cfg, potential, n_samples, seed, *, label="trajectory"):
    """Sample the canonical ensemble, and verify the result is stationary.

    ``cfg`` must already be an equilibrated liquid (see
    :func:`experiments.equilibrate.equilibrated_configuration`); the drift check
    afterwards is what catches the case where it is not.
    """
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"], burn_in=s["burn_in"], seed=seed,
    )
    if report.acceptance < 0.2:
        raise RuntimeError(
            f"sampler acceptance {report.acceptance:.2f} is too low to trust "
            f"({report.notes})"
        )
    check_equilibrated(traj, label=label, check_order=False)
    return traj, report


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o = ctx.config["observable"]
    observable = PairBinObservable(
        np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1),
        cutoff=ctx.config["potential"]["cutoff"],
    )
    target = PairBinObservable(np.asarray(o["target_bin"], dtype=float),
                               cutoff=ctx.config["potential"]["cutoff"])

    print(f"  system: {cfg.n_atoms} atoms, L = {cfg.cell[0, 0]:.2f} A, "
          f"rho = {cfg.density:.4f} /A^3, T = {temperature} K")

    # ---- reference ensemble, split into construction and evaluation halves --
    with ctx.timed("equilibration"):
        s_cfg = ctx.config["sampling"]
        cfg, eq_report = equilibrated_configuration(
            cfg, potential, temperature,
            n_melt=int(ctx.scaled("sampling.n_melt")),
            n_anneal=int(ctx.scaled("sampling.n_anneal")),
            n_leapfrog=s_cfg["n_leapfrog"], step_size=s_cfg["step_size"], seed=ctx.seed,
        )
    print(f"    {eq_report}")

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
    cutoff = ctx.config["potential"]["cutoff"]
    centres = np.linspace(pcfg["basis_r_min"], pcfg["basis_r_max"], pcfg["n_basis"])
    widths = np.full(pcfg["n_basis"], float(centres[1] - centres[0]))
    basis = ShellBasis(centres, widths, cutoff, r_on=cutoff - 1.0)
    with ctx.timed("basis_design"):
        design = build_shell_design(construct_frames, basis)

    # ---- how many frames does the construction need to generalise? ---------
    # A null-space field is orthogonal to the observable *on the frames it was
    # built from* by construction. Whether it stays orthogonal on fresh frames
    # depends entirely on how well the covariance was estimated, and with a
    # (n_basis x n_bins) covariance from too few samples the null space found is
    # the null space of the noise. This sweep measures where that turns around,
    # and costs nothing -- no sampling of the perturbed potential is involved.
    with ctx.timed("null_space_generalisation"):
        generalisation = null_space_generalisation(
            ctx, basis, construct_frames, target, evaluate_frames,
            target.evaluate_trajectory(evaluate_frames), temperature, cutoff,
        )
    print("\n    null-space generalisation (predicted |shift|, pairs):")
    print(f"      {'n_construct':>12} {'in-sample':>10} {'out-of-sample':>14}")
    for g in generalisation:
        print(f"      {g['n_construct']:12d} {g['in_sample']:10.4f} {g['out_of_sample']:14.4f}")

    records = []
    for level in pcfg["force_rms_levels"]:
        common = dict(basis=basis, design=design, cutoff=cutoff,
                      target_force_rms=level, seed=ctx.seed)
        built = [
            ("null", NullSpacePerturbation(construct_frames, target, temperature, **common), {}),
            ("aligned", AlignedPerturbation(construct_frames, target, temperature, **common), {}),
        ]
        for k in range(int(pcfg["n_random_seeds"])):
            kw = dict(common, seed=ctx.seed + 100 + k)
            built.append((f"random{k}",
                          RandomShellPerturbation(construct_frames, target,
                                                  temperature, **kw), {}))

        for name, perturbation, diagnostics in built:
            label = f"{name}@{level:.1e}"
            with ctx.timed(label):
                records.append(
                    evaluate_one(ctx, cfg, potential, perturbation, observable, target,
                                 evaluate_frames, a_evaluate, temperature,
                                 name=name, level=level, diagnostics=diagnostics)
                )
            r = records[-1]
            print(f"    {label:20s} F_rms(out) = {r['force_rms_out_of_sample']:.3e}  "
                  f"predicted = {r['predicted_max']:7.3f}  "
                  f"measured = {r['measured_max']:7.3f} +/- {r['measured_max_error']:.3f}")

    summary = summarise(records)
    summary["null_space_generalisation"] = generalisation
    ctx.save_json("generalisation", generalisation)
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    make_figure(ctx, records, observable)
    report(summary)
    return summary


def null_space_generalisation(ctx, basis, construct_frames, observable,
                              evaluate_frames, a_evaluate, temperature, cutoff) -> list:
    """Predicted shift of a null-space field, in sample and out, versus sample count.

    In-sample the answer is zero by construction and carries no information.
    Out-of-sample it is the quantity that decides whether the counterexample is
    real, and it can only fall to zero once the covariance matrix is estimated
    from enough frames to be something other than noise.
    """
    counts, out = [], []
    m = len(construct_frames)
    for fraction in (0.125, 0.25, 0.5, 1.0):
        n = max(basis.centers.size + 4, int(m * fraction))
        if n > m or n in counts:
            continue
        counts.append(n)

    a_construct_full = observable_matrix(observable, construct_frames)
    for n in counts:
        perturbation = NullSpacePerturbation(
            construct_frames[:n], observable, temperature,
            cutoff=cutoff, basis=basis, target_force_rms=1.0e-3, seed=ctx.seed,
        )
        du_in = np.array([perturbation.energy(c) for c in construct_frames[:n]])
        du_out = np.array([perturbation.energy(c) for c in evaluate_frames])
        in_sample = predict_shift(a_construct_full[:n], du_in, temperature,
                                  n_resamples=100, seed=ctx.seed)
        out_sample = predict_shift(a_evaluate, du_out, temperature, n_resamples=100,
                                   seed=ctx.seed)
        out.append({
            "n_construct": n,
            "in_sample": float(np.abs(np.asarray(in_sample.value)).max()),
            "out_of_sample": float(np.abs(np.asarray(out_sample.value)).max()),
            "out_of_sample_error": float(np.abs(np.asarray(out_sample.error)).max()),
        })
    return out


def evaluate_one(ctx, cfg, potential, perturbation, observable, target, evaluate_frames,
                 a_evaluate, temperature, *, name, level, diagnostics) -> dict:
    """Predict, then measure, the observable shift caused by one perturbation."""
    f_rms_out = perturbation.force_rms(evaluate_frames)
    du = np.array([perturbation.energy(c) for c in evaluate_frames])

    prediction = predict_shift(a_evaluate, du, temperature, n_resamples=400, seed=ctx.seed)
    rw = reweight(a_evaluate, du, temperature, n_resamples=200, seed=ctx.seed)

    surrogate = potential + perturbation
    direct, direct_report = sample(
        ctx, cfg, surrogate, int(ctx.scaled("sampling.n_direct")), seed=ctx.seed + 2
    )
    a_direct = observable.evaluate_trajectory(direct)

    # The headline number is the SCALAR target the fields were designed against.
    t_direct = target.evaluate_trajectory(direct).ravel()
    t_reference = target.evaluate_trajectory(evaluate_frames).ravel()
    du_target = predict_shift(t_reference, du, temperature, n_resamples=400, seed=ctx.seed)
    target_measured = float(t_direct.mean() - t_reference.mean())
    target_error = float(np.hypot(blocking_analysis(t_direct).error,
                                  blocking_analysis(t_reference).error))

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
        "target_predicted": float(np.asarray(du_target.value)),
        "target_predicted_error": float(np.asarray(du_target.error)),
        # For a null-space field the first-order term vanishes by construction,
        # so whatever residual effect survives must be second order. The theory
        # predicts that residual too, and reporting it turns "the null field did
        # not do exactly nothing" from an embarrassment into a second, sharper
        # test of the same expansion.
        "target_second_order": float(np.asarray(du_target.second_order.value)),
        "target_second_order_error": float(np.asarray(du_target.second_order.error)),
        "target_measured": target_measured,
        "target_error": target_error,
        "target_correlation": float(np.asarray(du_target.correlation)),
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
            "target": {
                "null_predicted": n["target_predicted"],
                "null_measured": n["target_measured"],
                "null_error": n["target_error"],
                "aligned_predicted": a["target_predicted"],
                "aligned_measured": a["target_measured"],
                "aligned_error": a["target_error"],
                "random_measured_mean": float(np.mean([r["target_measured"] for r in randoms]))
                if randoms else float("nan"),
                "null_second_order": n["target_second_order"],
                "null_measured_minus_second_order":
                    n["target_measured"] - n["target_predicted"] - n["target_second_order"],
                "null_consistent_with_zero":
                    abs(n["target_measured"]) < 2.0 * n["target_error"],
                "null_consistent_with_second_order":
                    abs(n["target_measured"] - n["target_predicted"] - n["target_second_order"])
                    < 2.0 * n["target_error"],
                "aligned_significant":
                    abs(a["target_measured"]) > 2.0 * a["target_error"],
            },
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
    gen = summary.get("null_space_generalisation") or []
    if gen:
        print("\n  --- how many frames the null-space construction needs ---")
        for g in gen:
            print(f"    n_construct = {g['n_construct']:5d}: out-of-sample predicted "
                  f"|shift| = {g['out_of_sample']:.4f} +/- {g['out_of_sample_error']:.4f} pairs")
    print("\n  --- P3: designed counterexamples ---")
    for level, s in summary["levels"].items():
        null_flag = "consistent with zero" if s["null_measured_within_error"] else "NOT zero"
        print(f"  at force RMSE {level} eV/A:")
        t = s["target"]
        print(f"    TARGET observable (the scalar the fields were designed against):")
        print(f"      null-space  predicted {t['null_predicted']:+7.3f}  "
              f"measured {t['null_measured']:+7.3f} +/- {t['null_error']:.3f}  "
              f"{'consistent with zero' if t['null_consistent_with_zero'] else 'NOT zero'}")
        print(f"      aligned     predicted {t['aligned_predicted']:+7.3f}  "
              f"measured {t['aligned_measured']:+7.3f} +/- {t['aligned_error']:.3f}  "
              f"{'significant' if t['aligned_significant'] else 'not significant'}")
        print(f"      random      measured {t['random_measured_mean']:+7.3f} (mean)")
        print(f"      null-space residual vs second-order prediction: "
              f"{t['null_measured_minus_second_order']:+7.3f} "
              f"({'consistent' if t['null_consistent_with_second_order'] else 'NOT consistent'})")
        print(f"    full curve: null {s['measured_max']['null']:+.3f} "
              f"+/- {s['measured_max']['null_error']:.3f}, aligned "
              f"{s['measured_max']['aligned']:+.3f} +/- {s['measured_max']['aligned_error']:.3f} "
              f"({null_flag})")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Designed error fields with matched force RMSE and opposite consequences")
