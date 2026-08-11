#!/usr/bin/env python3
"""exp05 -- how well does any cheap proxy rank surrogates? Prediction P1.

A zoo of surrogate potentials is built, each an analytic reference plus a
designed error field.  For every member we compute the metrics a practitioner
would report -- force RMSE first among them -- and, separately, the thing they
are supposed to stand in for: the error in a physical observable, measured by
explicit sampling of that surrogate.  Then we ask how well each proxy ranks the
zoo against the truth.

**Why a designed zoo rather than trained models.**  Trained models would be more
externally valid and are the subject of exp03/exp04.  A designed zoo is better
for testing the *mechanism*, because it decouples the two things that are
confounded in any set of real models: how large the error is, and what shape it
has.  Here force RMSE spans two orders of magnitude by construction while the
shape of the error varies independently, so a weak rank correlation cannot be
dismissed as a range artefact and a strong one cannot be attributed to the zoo
being too narrow.  The members are also cheap enough that every one of them gets
a real measurement rather than a prediction.

The decision-relevant statistic is not the correlation coefficient.  A
practitioner does not use the whole ranking; they pick the best model.  So
top-k overlap between the proxy ranking and the truth ranking is reported
alongside, and it is the number to look at.
"""

from __future__ import annotations

import itertools
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
    HighFrequencyPerturbation,
    NullSpacePerturbation,
    RadialShellPerturbation,
    RandomShellPerturbation,
    ShellBasis,
    build_shell_design,
)
from atomlab.sampling import hybrid_monte_carlo
from experiments.common import ExperimentContext, main
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    "observable": {"r_min": 3.0, "r_max": 7.0, "n_bins": 8},
    "sampling": {"n_reference": 2000, "n_direct": 1200, "n_leapfrog": 8,
                 "step_size": 2e-3, "burn_in": 300},
    "zoo": {
        "shell_r0": [3.6, 4.2, 5.0],
        "shell_width": [0.12, 0.30, 0.70],
        "shell_amplitude": [0.001, 0.004],
        "highfreq_wavelength": [0.5, 0.9],
        "highfreq_amplitude": [0.002, 0.008],
        "random_seeds": [0, 1, 2, 3],
        "random_force_rms": [5.0e-4, 4.0e-3],
        "designed_force_rms": [1.0e-3, 4.0e-3],
        "n_basis": 14,
    },
    "bootstrap": {"n_resamples": 2000},
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
        step_size=s["step_size"], burn_in=s["burn_in"], seed=seed)
    if report.acceptance < 0.2:
        raise RuntimeError(f"acceptance {report.acceptance:.2f} too low to trust")
    return traj, report


def build_zoo(ctx, frames, observable, temperature):
    """Surrogate error fields spanning two decades of force error."""
    z = ctx.config["zoo"]
    cutoff = ctx.config["potential"]["cutoff"]
    members = []

    for r0, width, amplitude in itertools.product(
        z["shell_r0"], z["shell_width"], z["shell_amplitude"]
    ):
        members.append((f"shell_r{r0}_w{width}_a{amplitude}",
                        RadialShellPerturbation(r0=r0, width=width,
                                                amplitude=amplitude, cutoff=cutoff)))

    for wavelength, amplitude in itertools.product(
        z["highfreq_wavelength"], z["highfreq_amplitude"]
    ):
        members.append((f"highfreq_l{wavelength}_a{amplitude}",
                        HighFrequencyPerturbation(wavelength=wavelength,
                                                  amplitude=amplitude, cutoff=cutoff)))

    centres = np.linspace(3.0, cutoff - 0.4, z["n_basis"])
    widths = np.full(z["n_basis"], float(centres[1] - centres[0]))
    basis = ShellBasis(centres, widths, cutoff, r_on=cutoff - 1.0)
    design = build_shell_design(frames, basis)
    common = dict(basis=basis, design=design, cutoff=cutoff)

    for seed, level in itertools.product(z["random_seeds"], z["random_force_rms"]):
        members.append((f"random_s{seed}_f{level:.0e}",
                        RandomShellPerturbation(frames, observable, temperature,
                                                target_force_rms=level, seed=seed, **common)))
    for level in z["designed_force_rms"]:
        members.append((f"null_f{level:.0e}",
                        NullSpacePerturbation(frames, observable, temperature,
                                              target_force_rms=level, seed=0, **common)))
        members.append((f"aligned_f{level:.0e}",
                        AlignedPerturbation(frames, observable, temperature,
                                            target_force_rms=level, seed=0, **common)))
    return members


def proxy_metrics(perturbation, frames, temperature) -> dict:
    """Every cheap metric a practitioner might compute, on the reference frames.

    All are computed from the error field alone -- no simulation of the
    surrogate -- which is exactly the situation a practitioner is in when
    choosing between fitted models.
    """
    all_errors, du = [], []
    for cfg in frames:
        all_errors.append(perturbation.forces(cfg).ravel())
        du.append(perturbation.energy(cfg))
    errors = np.concatenate(all_errors)
    du = np.asarray(du)
    n_atoms = frames[0].n_atoms

    force_rmse = float(np.sqrt((errors**2).mean()))
    return {
        "force_rmse": force_rmse,
        "force_mae": float(np.abs(errors).mean()),
        "force_max": float(np.abs(errors).max()),
        "force_p95": float(np.percentile(np.abs(errors), 95)),
        "force_p99": float(np.percentile(np.abs(errors), 99)),
        # A constant energy offset is unobservable, so the informative energy
        # metric is the spread of the error field, not its mean.
        "energy_std_per_atom": float(du.std(ddof=1) / n_atoms),
        "energy_range_per_atom": float(np.ptp(du) / n_atoms),
        # theory.md section 4: the empirical stand-in for the inverse spectral
        # weighting. High values mean smooth, systematic error; low values mean
        # high-frequency wiggle that inflates force error harmlessly.
        "smoothness_ratio": float(du.std(ddof=1) / force_rmse) if force_rmse > 0 else np.inf,
    }


def spearman_with_ci(x, y, n_resamples=2000, seed=0):
    """Spearman rho with a percentile bootstrap interval over zoo members.

    The members are independent of each other, so an ordinary (non-block)
    bootstrap is the right one here.  The interval is not optional: with a few
    dozen members the sampling distribution of a rank correlation is wide, and
    quoting rho alone would be exactly the overclaiming this project is about.
    """
    from scipy.stats import spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, x.size, size=x.size)
        if np.unique(x[idx]).size < 3 or np.unique(y[idx]).size < 3:
            draws[b] = np.nan
            continue
        draws[b] = spearmanr(x[idx], y[idx]).statistic
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"rho": point, "ci_low": float(lo), "ci_high": float(hi),
            "n": int(x.size), "n_bootstrap": int(draws.size)}


def top_k_overlap(proxy, truth, k):
    """Fraction of the true best-k that a proxy's best-k also contains.

    Lower is better for both arrays (they are errors), so 'best' is smallest.
    """
    proxy, truth = np.asarray(proxy), np.asarray(truth)
    best_proxy = set(np.argsort(proxy)[:k].tolist())
    best_truth = set(np.argsort(truth)[:k].tolist())
    return len(best_proxy & best_truth) / k


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o = ctx.config["observable"]
    observable = PairBinObservable(
        np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1),
        cutoff=ctx.config["potential"]["cutoff"])
    print(f"  system: {cfg.n_atoms} atoms, T = {temperature} K")

    with ctx.timed("reference_sampling"):
        ref, ref_report = sample(ctx, cfg, potential,
                                 ctx.scaled("sampling.n_reference"), seed=ctx.seed + 1)
    print(f"    {ref_report}")
    frames = [ref.frame(i) for i in range(ref.n_frames)]
    a_ref = observable.evaluate_trajectory(ref)
    a_ref_mean = a_ref.mean(axis=0)
    a_ref_err = blocking_analysis(a_ref).error

    # Metric evaluation only needs a subset; sampling every frame for 30 zoo
    # members would dominate the runtime without improving the metrics.
    metric_frames = frames[:: max(1, len(frames) // 300)]

    with ctx.timed("build_zoo"):
        zoo = build_zoo(ctx, metric_frames, observable, temperature)
    print(f"    zoo: {len(zoo)} surrogates")

    records = []
    for name, perturbation in zoo:
        with ctx.timed(f"member_{name}"):
            metrics = proxy_metrics(perturbation, metric_frames, temperature)
            du = np.array([perturbation.energy(c) for c in frames])
            prediction = predict_shift(a_ref, du, temperature, n_resamples=200, seed=ctx.seed)

            direct, direct_report = sample(ctx, cfg, potential + perturbation,
                                           ctx.scaled("sampling.n_direct"), seed=ctx.seed + 2)
            a_direct = observable.evaluate_trajectory(direct)
            measured = a_direct.mean(axis=0) - a_ref_mean
            measured_err = np.hypot(blocking_analysis(a_direct).error, a_ref_err)

        records.append({
            "name": name,
            **metrics,
            "predicted_norm": float(np.linalg.norm(np.asarray(prediction.value))),
            "observable_error": float(np.linalg.norm(measured)),
            "observable_error_noise": float(np.linalg.norm(measured_err)),
            "measured_curve": measured.tolist(),
            "direct_acceptance": direct_report.acceptance,
        })
        r = records[-1]
        print(f"    {name:28s} F_rmse={r['force_rmse']:.2e}  "
              f"|dA|={r['observable_error']:7.3f} +/- {r['observable_error_noise']:.3f}  "
              f"predicted={r['predicted_norm']:7.3f}")

    summary = analyse(ctx, records)
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    make_figure(ctx, records, summary)
    report(summary)
    return summary


def analyse(ctx, records) -> dict:
    truth = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    metric_names = ["force_rmse", "force_mae", "force_max", "force_p95", "force_p99",
                    "energy_std_per_atom", "energy_range_per_atom", "smoothness_ratio",
                    "predicted_norm"]
    n_boot = ctx.config["bootstrap"]["n_resamples"]

    correlations = {}
    for metric in metric_names:
        values = np.array([r[metric] for r in records])
        stats = spearman_with_ci(values, truth, n_resamples=n_boot, seed=ctx.seed)
        stats["top3_overlap"] = top_k_overlap(values, truth, min(3, len(records)))
        stats["top5_overlap"] = top_k_overlap(values, truth, min(5, len(records)))
        correlations[metric] = stats

    resolvable = truth > 2.0 * noise
    return {
        "n_members": len(records),
        "force_rmse_range": [float(min(r["force_rmse"] for r in records)),
                             float(max(r["force_rmse"] for r in records))],
        "observable_error_range": [float(truth.min()), float(truth.max())],
        "n_resolvable": int(resolvable.sum()),
        "correlations": correlations,
        "headline": {
            "force_rmse": correlations["force_rmse"],
            "predicted_norm": correlations["predicted_norm"],
        },
    }


def make_figure(ctx, records, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from atomlab.utils import plotting as P

    P.use_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 2.9))

    force = np.array([r["force_rmse"] for r in records])
    truth = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    predicted = np.array([r["predicted_norm"] for r in records])

    ax1.errorbar(force, truth, yerr=noise, fmt="o", markersize=4,
                 color=P.SERIES[0], ecolor=P.GRID, elinewidth=0.8, capsize=1.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("force RMSE (eV/Å)")
    ax1.set_ylabel("|observable error|")
    rho = summary["correlations"]["force_rmse"]
    ax1.set_title(f"ρ = {rho['rho']:.2f}  [{rho['ci_low']:.2f}, {rho['ci_high']:.2f}]")

    ax2.errorbar(predicted, truth, yerr=noise, fmt="o", markersize=4,
                 color=P.SERIES[1], ecolor=P.GRID, elinewidth=0.8, capsize=1.5)
    lim = [0, max(predicted.max(), truth.max()) * 1.05]
    ax2.plot(lim, lim, color=P.TEXT_SECONDARY, linestyle="--", linewidth=0.8)
    ax2.set_xlabel("predicted |shift| (response theory)")
    ax2.set_ylabel("measured |observable error|")
    rho2 = summary["correlations"]["predicted_norm"]
    ax2.set_title(f"ρ = {rho2['rho']:.2f}  [{rho2['ci_low']:.2f}, {rho2['ci_high']:.2f}]")

    names = list(summary["correlations"])
    rhos = [summary["correlations"][n]["rho"] for n in names]
    los = [summary["correlations"][n]["ci_low"] for n in names]
    his = [summary["correlations"][n]["ci_high"] for n in names]
    order = np.argsort(rhos)
    y = np.arange(len(names))
    ax3.errorbar([rhos[i] for i in order], y,
                 xerr=[[rhos[i] - los[i] for i in order], [his[i] - rhos[i] for i in order]],
                 fmt="o", markersize=4, color=P.SERIES[2], ecolor=P.GRID,
                 elinewidth=0.9, capsize=1.5)
    ax3.set_yticks(y, [names[i].replace("_", " ") for i in order])
    ax3.axvline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
    ax3.set_xlabel("Spearman ρ vs observable error")
    ax3.set_title("with 95% intervals")

    P.add_panel_labels([ax1, ax2, ax3])
    P.save_figure(fig, ctx.figure_path("proxy_correlation"))
    plt.close(fig)


def report(summary):
    print("\n  --- P1: how well does each proxy rank the zoo? ---")
    print(f"  {summary['n_members']} surrogates, force RMSE spanning "
          f"{summary['force_rmse_range'][0]:.1e} to {summary['force_rmse_range'][1]:.1e} eV/A")
    print(f"  {summary['n_resolvable']} of them have an observable error resolvable "
          f"above the sampling noise")
    print(f"  {'metric':24s} {'rho':>6} {'95% interval':>18} {'top-3':>7} {'top-5':>7}")
    for name, s in sorted(summary["correlations"].items(), key=lambda kv: -kv[1]["rho"]):
        print(f"  {name:24s} {s['rho']:6.2f}  [{s['ci_low']:6.2f}, {s['ci_high']:6.2f}] "
              f"{s['top3_overlap']:7.2f} {s['top5_overlap']:7.2f}")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Rank correlation between cheap proxy metrics and measured observable error")
