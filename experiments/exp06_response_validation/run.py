#!/usr/bin/env python3
"""exp06 -- validating the response theory: prediction P2.

Two questions, both answered by measurement rather than by assertion.

**Does the first-order formula predict the observable shift?**  For a fixed
error field scaled through a range of amplitudes, three numbers are computed on
the same system: the first-order prediction ``-beta Cov(A, dU)``, the
exponentially reweighted value (exact to all orders but variance-limited), and
the shift measured by explicit sampling of the perturbed potential.  All three
must agree while the perturbation is small, and must part company in an
understood order as it grows -- first-order first, reweighting second.  If they
do, the formula can be used to predict a model's downstream error from a
reference trajectory alone, with no simulation of the model at all.  That is the
practical payoff and it is worth establishing carefully.

**How much does the shape of the error matter, at fixed force error?**  Section
4.1 of ``docs/theory.md`` gives a heuristic estimate that error fields of
different width in ``r``, held at *identical* force RMSE, produce observable
errors scaling as ``width^{3/2}``.  That estimate has a limited regime of
validity and is not a prediction to be defended -- see the correction in that
section.  What the second part of this experiment measures is the thing the
estimate was reaching for: how far the observable error can vary while the
reported force error does not vary at all.  The fitted exponent is reported for
what it is worth, using the norm of the whole predicted difference curve rather
than a single bin so that the saturation artefact discussed in the theory does
not confound it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.common import ExperimentContext, main
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {
        "lattice_constant": 5.4,
        "reps": [3, 3, 3],
        "density": 0.0200,
        "temperature": 120.0,
    },
    "potential": {
        "epsilon": 0.0103,
        "sigma": 3.405,
        "cutoff": 7.0,
        "mode": "shifted_force",
    },
    "observable": {"r_min": 3.0, "r_max": 7.0, "n_bins": 8},
    "sampling": {
        "n_reference": 2000,
        "n_direct": 1500,
        "n_leapfrog": 8,
        "step_size": 2e-3,
        "burn_in": 300,
        "n_melt": 400,
        "n_anneal": 1000,
    },
    "amplitude_sweep": {
        "r0": 4.2,
        "width": 0.35,
        "amplitudes": [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032],
    },
    "width_sweep": {
        "r0": 4.2,
        "widths": [0.10, 0.15, 0.25, 0.40, 0.65, 1.00],
        "force_rms": 2.0e-3,
    },
}


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def sample(ctx, cfg, potential, n_samples, seed):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"], burn_in=s["burn_in"], seed=seed,
    )
    if report.acceptance < 0.2:
        raise RuntimeError(f"acceptance {report.acceptance:.2f} too low to trust")
    return traj, report


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o = ctx.config["observable"]
    observable = PairBinObservable(
        np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1),
        cutoff=ctx.config["potential"]["cutoff"],
    )
    print(f"  system: {cfg.n_atoms} atoms, L = {cfg.cell[0, 0]:.2f} A, T = {temperature} K")

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
        ref, ref_report = sample(ctx, cfg, potential,
                                 ctx.scaled("sampling.n_reference"), seed=ctx.seed + 1)
    print(f"    {ref_report}")
    frames = [ref.frame(i) for i in range(ref.n_frames)]
    a_ref = observable.evaluate_trajectory(ref)
    a_ref_mean = a_ref.mean(axis=0)
    a_ref_err = blocking_analysis(a_ref).error

    amplitude_records = amplitude_sweep(
        ctx, cfg, potential, observable, ref, frames, a_ref, a_ref_mean, a_ref_err, temperature
    )
    width_records = width_sweep(
        ctx, cfg, potential, observable, frames, a_ref, a_ref_mean, a_ref_err, temperature
    )

    summary = {
        "amplitude_sweep": summarise_amplitudes(amplitude_records),
        "width_sweep": summarise_widths(width_records),
    }
    ctx.save_json("amplitude_records", amplitude_records)
    ctx.save_json("width_records", width_records)
    ctx.save_json("summary", summary)
    make_figures(ctx, amplitude_records, width_records)
    report(summary)
    return summary


def amplitude_sweep(ctx, cfg, potential, observable, ref, frames,
                    a_ref, a_ref_mean, a_ref_err, temperature) -> list:
    """First-order vs reweighted vs direct, as the perturbation grows."""
    spec = ctx.config["amplitude_sweep"]
    records = []
    print("\n  amplitude sweep (first-order vs reweighted vs direct):")
    print(f"    {'amp(eV)':>9} {'F_rms':>10} {'B*sd(dU)':>9} {'linear':>9} "
          f"{'reweight':>9} {'ESS':>6} {'direct':>16} {'2nd/1st':>8}")

    for amplitude in spec["amplitudes"]:
        perturbation = RadialShellPerturbation(
            spec["r0"], spec["width"], amplitude, ctx.config["potential"]["cutoff"]
        )
        du = np.array([perturbation.energy(c) for c in frames])
        prediction = predict_shift(a_ref, du, temperature, n_resamples=400, seed=ctx.seed)
        rw = reweight(a_ref, du, temperature, n_resamples=200, seed=ctx.seed)

        with ctx.timed(f"direct_amp_{amplitude:g}"):
            direct, direct_report = sample(ctx, cfg, potential + perturbation,
                                           ctx.scaled("sampling.n_direct"), seed=ctx.seed + 2)
        a_direct = observable.evaluate_trajectory(direct)
        measured = a_direct.mean(axis=0) - a_ref_mean
        measured_err = np.hypot(blocking_analysis(a_direct).error, a_ref_err)

        predicted = np.asarray(prediction.value, dtype=float)
        peak = int(np.argmax(np.abs(predicted)))
        records.append({
            "amplitude": amplitude,
            "force_rms": perturbation.force_rms(frames),
            "beta_sigma_dU": prediction.beta_sigma_dU,
            "peak_bin": peak,
            "linear": float(predicted[peak]),
            "linear_error": float(np.asarray(prediction.error)[peak]),
            "reweighted": float(np.asarray(rw.shift.value)[peak]),
            "reweighted_ess_fraction": rw.ess_fraction,
            "direct": float(measured[peak]),
            "direct_error": float(measured_err[peak]),
            "second_order_ratio": float(np.asarray(prediction.second_order_ratio)[peak]),
            "linear_trustworthy": bool(prediction.is_trustworthy),
            "direct_acceptance": direct_report.acceptance,
            "linear_curve": predicted.tolist(),
            "direct_curve": measured.tolist(),
            "direct_curve_error": measured_err.tolist(),
        })
        r = records[-1]
        print(f"    {amplitude:9.4f} {r['force_rms']:10.2e} {r['beta_sigma_dU']:9.3f} "
              f"{r['linear']:9.3f} {r['reweighted']:9.3f} {r['reweighted_ess_fraction']:6.2f} "
              f"{r['direct']:8.3f}+/-{r['direct_error']:<6.3f} {r['second_order_ratio']:8.2f}")
    return records


def width_sweep(ctx, cfg, potential, observable, frames,
                a_ref, a_ref_mean, a_ref_err, temperature) -> list:
    """Observable damage at fixed force error, as a function of error width."""
    spec = ctx.config["width_sweep"]
    records = []
    print(f"\n  width sweep at fixed force RMSE = {spec['force_rms']:.1e} eV/A:")
    print(f"    {'width(A)':>9} {'amp(eV)':>10} {'F_rms':>10} {'linear':>9} {'direct':>16}")

    for width in spec["widths"]:
        perturbation = RadialShellPerturbation.matched_force_error(
            width, spec["force_rms"], frames,
            r0=spec["r0"], cutoff=ctx.config["potential"]["cutoff"],
        )
        du = np.array([perturbation.energy(c) for c in frames])
        prediction = predict_shift(a_ref, du, temperature, n_resamples=400, seed=ctx.seed)

        with ctx.timed(f"direct_width_{width:g}"):
            direct, _ = sample(ctx, cfg, potential + perturbation,
                               ctx.scaled("sampling.n_direct"), seed=ctx.seed + 3)
        a_direct = observable.evaluate_trajectory(direct)
        measured = a_direct.mean(axis=0) - a_ref_mean
        measured_err = np.hypot(blocking_analysis(a_direct).error, a_ref_err)

        predicted = np.asarray(prediction.value, dtype=float)
        records.append({
            "width": width,
            "amplitude": perturbation.amplitude,
            "force_rms": perturbation.force_rms(frames),
            "linear_norm": float(np.linalg.norm(predicted)),
            "direct_norm": float(np.linalg.norm(measured)),
            "direct_norm_error": float(np.linalg.norm(measured_err)),
            "linear_curve": predicted.tolist(),
            "direct_curve": measured.tolist(),
        })
        r = records[-1]
        print(f"    {width:9.2f} {r['amplitude']:10.2e} {r['force_rms']:10.2e} "
              f"{r['linear_norm']:9.3f} {r['direct_norm']:8.3f}+/-{r['direct_norm_error']:<6.3f}")
    return records


def summarise_amplitudes(records) -> dict:
    """Where does linear response stop working, and does its own flag catch it?"""
    linear = np.array([r["linear"] for r in records])
    direct = np.array([r["direct"] for r in records])
    error = np.array([r["direct_error"] for r in records])
    reweighted = np.array([r["reweighted"] for r in records])
    trustworthy = np.array([r["linear_trustworthy"] for r in records])

    within = np.abs(linear - direct) < 2.0 * error
    return {
        "n_amplitudes": len(records),
        "linear_agrees_with_direct": within.tolist(),
        "reweighted_agrees_with_direct": (np.abs(reweighted - direct) < 2.0 * error).tolist(),
        "self_flagged_trustworthy": trustworthy.tolist(),
        # The claim worth making is not "the formula always works" but "it works
        # where it says it does": every case it flags trustworthy should agree.
        "flagged_cases_all_agree": bool(np.all(within[trustworthy])) if trustworthy.any() else None,
        "n_flagged": int(trustworthy.sum()),
        "largest_agreeing_beta_sigma": float(
            max([r["beta_sigma_dU"] for r, ok in zip(records, within) if ok], default=float("nan"))
        ),
    }


def summarise_widths(records) -> dict:
    """Fit the exponent the frequency argument predicts to be 3/2."""
    widths = np.array([r["width"] for r in records])
    direct = np.array([r["direct_norm"] for r in records])
    linear = np.array([r["linear_norm"] for r in records])

    def fit(y):
        good = (y > 0) & np.isfinite(y)
        if good.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log(widths[good]), np.log(y[good]), 1)[0])

    return {
        "predicted_exponent": 1.5,
        "measured_exponent_direct": fit(direct),
        "measured_exponent_linear": fit(linear),
        "dynamic_range": float(direct.max() / direct.min()) if direct.min() > 0 else float("inf"),
        "force_rms_spread": float(
            np.ptp([r["force_rms"] for r in records]) / np.mean([r["force_rms"] for r in records])
        ),
    }


def make_figures(ctx, amplitude_records, width_records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from atomlab.utils import plotting as P

    P.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    amp = np.array([r["amplitude"] for r in amplitude_records])
    for idx, (key, err_key, label) in enumerate([
        ("direct", "direct_error", "direct sampling"),
        ("linear", "linear_error", "first order"),
        ("reweighted", None, "reweighted"),
    ]):
        y = np.array([r[key] for r in amplitude_records])
        e = (np.array([r[err_key] for r in amplitude_records]) if err_key
             else np.zeros_like(y))
        P.series(ax1, amp, y, e, index=idx, label=label, fill=False)
    ax1.set_xscale("log")
    ax1.set_xlabel("perturbation amplitude (eV)")
    ax1.set_ylabel("shift in pair count")
    ax1.legend()
    ax1.set_title("Where linear response holds")

    widths = np.array([r["width"] for r in width_records])
    direct = np.array([r["direct_norm"] for r in width_records])
    errs = np.array([r["direct_norm_error"] for r in width_records])
    P.series(ax2, widths, direct, errs, index=0, label="measured", fill=False)
    good = direct > 0
    if good.sum() >= 2:
        reference = direct[good][0] * (widths[good] / widths[good][0]) ** 1.5
        ax2.plot(widths[good], reference, color=P.TEXT_SECONDARY, linestyle="--",
                 linewidth=1.0, label="width$^{3/2}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("width of the error field (Å)")
    ax2.set_ylabel("|shift| at fixed force RMSE")
    ax2.legend()
    ax2.set_title("Same force error, different damage")

    P.add_panel_labels([ax1, ax2])
    P.save_figure(fig, ctx.figure_path("response_validation"))
    plt.close(fig)


def report(summary):
    a, w = summary["amplitude_sweep"], summary["width_sweep"]
    print("\n  --- P2: does the first-order formula predict the shift? ---")
    print(f"  cases the formula flagged as trustworthy: {a['n_flagged']}/{a['n_amplitudes']}")
    print(f"  every flagged case agrees with direct sampling: {a['flagged_cases_all_agree']}")
    print(f"  largest beta*sd(dU) at which it still agrees:   {a['largest_agreeing_beta_sigma']:.3f}")
    print("\n  --- the frequency argument ---")
    print(f"  measured exponent (direct):     {w['measured_exponent_direct']:.2f}")
    print(f"  measured exponent (first order):{w['measured_exponent_linear']:.2f}")
    print(f"  predicted by theory:            {w['predicted_exponent']:.2f}")
    print(f"  observable error spans {w['dynamic_range']:.0f}x across widths, "
          f"while force RMSE varies by {100 * w['force_rms_spread']:.1f}%")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="First-order response vs reweighting vs direct sampling")
