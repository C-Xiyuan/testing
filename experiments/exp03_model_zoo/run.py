#!/usr/bin/env python3
"""exp03 -- the same question, asked of models that were actually fitted.

exp05 and exp07 use *designed* error fields, which is the better test of the
mechanism because it decouples how large an error is from what shape it has.
The obvious objection is that real fitted models might not occupy the regime
those designed fields explore -- that their errors might all have much the same
shape, so that force RMSE would rank them adequately after all.

This experiment answers that objection with fitted models: four architectures
(a pair spline, a linear ACE-style basis, a Behler-Parrinello network, an
E(3)-equivariant message-passing network) trained on Lennard-Jones argon at
several data budgets and seeds.  For each we compute what a practitioner would
report -- force RMSE on a held-out test set -- and, separately, the thing it is
supposed to stand for: the error in a physical observable, both predicted by
response theory and measured by explicit sampling.

The pair spline is included because it is *exactly* the right functional form
for a pair potential, so it should be the one model whose error is small and
structureless.  Whether force RMSE ranks the rest is the question.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.response import predict_shift
from atomlab.analysis.statistics import blocking_analysis
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.sampling import hybrid_monte_carlo
from atomlab.types import Dataset
from experiments.common import ExperimentContext, main
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    "observable": {"r_min": 3.0, "r_max": 7.0, "n_bins": 8,
                   "target_bin": [3.4, 3.9]},
    "sampling": {"n_reference": 3000, "n_direct": 1500, "n_leapfrog": 8,
                 "step_size": 2e-3, "burn_in": 300,
                 "n_melt": 400, "n_anneal": 1000},
    "data": {
        "n_train_pool": 400,
        "n_test": 120,
        # A model's cutoff is shorter than the reference's, as it always is in
        # practice. That truncation is itself a source of error, and a realistic
        # one, so it is not corrected for.
        "model_cutoff": 6.0,
        "budgets": [40, 400],
    },
    "zoo": {
        "pair_spline_knots": [8, 24],
        "linear_l_max": [2, 4],
        "bpnn_hidden": [[16, 16], [64, 64]],
        "bpnn_epochs": 250,
        "egnn_channels": [8],
        "egnn_epochs": 60,
        "seeds": [0, 1],
    },
}


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def sample(ctx, cfg, potential, n_samples, seed, *, label="trajectory", check=True):
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


def build_models(ctx):
    """Instantiate the zoo. Each entry is (name, factory) built lazily."""
    from atomlab.models.bpnn import BPNN
    from atomlab.models.descriptors.acsf import ACSF
    from atomlab.models.descriptors.bispectrum import Bispectrum
    from atomlab.models.egnn import EGNN
    from atomlab.models.linear import LinearPotential
    from atomlab.models.pair_spline import PairSpline

    z, cutoff = ctx.config["zoo"], ctx.config["data"]["model_cutoff"]
    entries = []

    for knots in z["pair_spline_knots"]:
        entries.append((f"spline_k{knots}",
                        lambda k=knots: PairSpline(cutoff=cutoff, r_min=2.2, n_knots=k)))

    for l_max in z["linear_l_max"]:
        entries.append((f"linear_l{l_max}",
                        lambda l=l_max: LinearPotential(
                            Bispectrum(r_cut=cutoff, n_radial_2b=8, n_radial_3b=4, l_max=l))))

    for hidden in z["bpnn_hidden"]:
        for seed in z["seeds"]:
            entries.append((f"bpnn_{hidden[0]}x{len(hidden)}_s{seed}",
                            lambda h=hidden, s=seed: BPNN(
                                ACSF.default((0,), cutoff), hidden=tuple(h), seed=s)))

    for channels in z["egnn_channels"]:
        for seed in z["seeds"]:
            entries.append((f"egnn_c{channels}_s{seed}",
                            lambda c=channels, s=seed: EGNN(
                                cutoff=cutoff, channels=c, l_max=1, n_layers=2, seed=s)))
    return entries


def fit_one(name, factory, train, val, ctx):
    """Fit one model, returning it and a record of how the fit went."""
    z = ctx.config["zoo"]
    model = factory()
    kwargs = {}
    if name.startswith("bpnn"):
        kwargs = {"epochs": z["bpnn_epochs"]}
    elif name.startswith("egnn"):
        kwargs = {"epochs": z["egnn_epochs"]}

    start = time.time()
    try:
        report = model.fit(train, val=val, **kwargs)
    except TypeError:
        # Linear and spline fits take no epoch budget.
        report = model.fit(train, val=val)
    return model, report, time.time() - start


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o, d = ctx.config["observable"], ctx.config["data"]
    observable = PairBinObservable(np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1),
                                   cutoff=ctx.config["potential"]["cutoff"])
    target = PairBinObservable(np.asarray(o["target_bin"], dtype=float),
                               cutoff=ctx.config["potential"]["cutoff"])
    print(f"  system: {cfg.n_atoms} atoms, T = {temperature} K")

    with ctx.timed("equilibration"):
        s_cfg = ctx.config["sampling"]
        cfg, eq = equilibrated_configuration(
            cfg, potential, temperature,
            n_melt=int(ctx.get("sampling.n_melt")), n_anneal=int(ctx.get("sampling.n_anneal")),
            n_leapfrog=s_cfg["n_leapfrog"], step_size=s_cfg["step_size"], seed=ctx.seed)
    print(f"    {eq}")

    with ctx.timed("reference_sampling"):
        ref, ref_report = sample(ctx, cfg, potential, ctx.scaled("sampling.n_reference"),
                                 seed=ctx.seed + 1, label="reference")
    print(f"    {ref_report}")
    frames = [ref.frame(i) for i in range(ref.n_frames)]
    a_ref = observable.evaluate_trajectory(ref)
    t_ref = target.evaluate_trajectory(ref).ravel()
    t_ref_mean, t_ref_err = t_ref.mean(), blocking_analysis(t_ref).error

    # Training and test data come from *separate* chains, so a model cannot be
    # tested on configurations correlated with the ones it was fitted to.
    with ctx.timed("training_data"):
        pool_traj, _ = sample(ctx, cfg, potential, ctx.scaled("data.n_train_pool"),
                              seed=ctx.seed + 11, label="train pool")
        # No drift check on the test set. The reference trajectory needs one
        # because its averages are ensemble averages and a drifting chain would
        # bias them; a test set is a bag of labelled configurations whose only
        # job is to be disjoint from the training data, and it is deliberately
        # small.
        test_traj, _ = sample(ctx, cfg, potential, ctx.scaled("data.n_test"),
                              seed=ctx.seed + 12, label="test", check=False)
        pool = potential.label([pool_traj.frame(i) for i in range(pool_traj.n_frames)])
        test = Dataset(potential.label([test_traj.frame(i) for i in range(test_traj.n_frames)]))
    print(f"    {len(pool)} training configurations, {len(test)} test")

    entries = build_models(ctx)
    budgets = [b for b in d["budgets"] if b <= len(pool)] or [len(pool)]
    print(f"    zoo: {len(entries)} architectures x {len(budgets)} budgets")

    records = []
    for budget in budgets:
        train = Dataset(pool[:budget])
        val = Dataset(pool[budget:budget + max(8, budget // 5)]) if budget < len(pool) else None
        for name, factory in entries:
            label = f"{name}@n{budget}"
            with ctx.timed(label):
                try:
                    record = evaluate_model(ctx, name, factory, train, val, test, cfg,
                                            potential, observable, target, frames, a_ref,
                                            t_ref, t_ref_mean, t_ref_err, temperature, budget)
                except Exception as exc:                     # noqa: BLE001
                    print(f"    {label}: FAILED -- {type(exc).__name__}: {exc}")
                    records.append({"name": name, "budget": budget, "failed": True,
                                    "error": f"{type(exc).__name__}: {exc}"})
                    continue
            records.append(record)
            r = record
            print(f"    {label:24s} F_rmse={r['force_rmse']:.4f}  "
                  f"predicted={r['target_predicted']:+7.3f}  "
                  f"measured={r['target_measured']:+7.3f}+/-{r['target_error']:.3f}  "
                  f"({r['fit_seconds']:.0f}s fit)")

    summary = analyse(ctx, [r for r in records if not r.get("failed")])
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    report_summary(summary, records)
    return summary


def evaluate_model(ctx, name, factory, train, val, test, cfg, potential,
                   observable, target, frames, a_ref, t_ref, t_ref_mean, t_ref_err,
                   temperature, budget):
    model, fit_report, fit_seconds = fit_one(name, factory, train, val, ctx)
    metrics = model.evaluate(test)

    du = np.array([model.energy(c) - potential.energy(c) for c in frames])
    prediction = predict_shift(t_ref, du, temperature, n_resamples=400, seed=ctx.seed)

    direct, direct_report = sample(ctx, cfg, model, ctx.scaled("sampling.n_direct"),
                                   seed=ctx.seed + 2, label=f"{name} direct", check=False)
    t_direct = target.evaluate_trajectory(direct).ravel()
    measured = float(t_direct.mean() - t_ref_mean)
    error = float(np.hypot(blocking_analysis(t_direct).error, t_ref_err))

    return {
        "name": name,
        "budget": budget,
        "n_parameters": int(getattr(model, "n_parameters", -1)),
        "fit_seconds": fit_seconds,
        "fit_converged": bool(getattr(fit_report, "converged", True)),
        "force_rmse": metrics["force_rmse"],
        "force_mae": metrics["force_mae"],
        "force_cosine": metrics["force_cosine"],
        "energy_rmse": metrics["energy_rmse"],
        "beta_sigma_dU": prediction.beta_sigma_dU,
        "du_std_per_atom": float(du.std(ddof=1) / cfg.n_atoms),
        "smoothness_ratio": float(du.std(ddof=1) / metrics["force_rmse"]),
        "target_correlation": float(np.asarray(prediction.correlation)),
        "target_predicted": float(np.asarray(prediction.value)),
        "target_predicted_error": float(np.asarray(prediction.error)),
        "target_measured": measured,
        "target_error": error,
        "direct_acceptance": direct_report.acceptance,
    }


def analyse(ctx, records) -> dict:
    if len(records) < 4:
        return {"n_models": len(records), "insufficient": True}
    from scipy.stats import spearmanr

    truth = np.array([abs(r["target_measured"]) for r in records])
    out = {"n_models": len(records)}
    for metric in ("force_rmse", "force_mae", "energy_rmse", "smoothness_ratio",
                   "du_std_per_atom", "target_predicted"):
        values = np.array([abs(r[metric]) for r in records])
        rho = float(spearmanr(values, truth).statistic)
        rng = np.random.default_rng(ctx.seed)
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(values), len(values))
            if np.unique(values[idx]).size > 2:
                draws.append(spearmanr(values[idx], truth[idx]).statistic)
        draws = np.array([d for d in draws if np.isfinite(d)])
        out[metric] = {"rho": rho,
                       "ci": [float(np.percentile(draws, 2.5)),
                              float(np.percentile(draws, 97.5))]}
    chi = np.array([(r["target_measured"] - r["target_predicted"]) / r["target_error"]
                    for r in records])
    out["prediction_residual_sigma"] = float(np.sqrt((chi**2).mean()))
    out["force_rmse_range"] = [float(min(r["force_rmse"] for r in records)),
                               float(max(r["force_rmse"] for r in records))]
    out["observable_error_range"] = [float(truth.min()), float(truth.max())]
    return out


def report_summary(summary, records):
    failed = [r for r in records if r.get("failed")]
    print(f"\n  --- fitted-model zoo: {summary.get('n_models', 0)} models"
          f"{f', {len(failed)} failed' if failed else ''} ---")
    if summary.get("insufficient"):
        print("  too few models fitted to correlate anything")
        return
    print(f"  force RMSE spans {summary['force_rmse_range'][0]:.4f} to "
          f"{summary['force_rmse_range'][1]:.4f} eV/A")
    print(f"  observable error spans {summary['observable_error_range'][0]:.3f} to "
          f"{summary['observable_error_range'][1]:.3f} pairs")
    print(f"  prediction vs measurement: {summary['prediction_residual_sigma']:.2f} sigma rms")
    print(f"\n  {'metric':22s} {'rho vs observable error':>24}")
    for metric in ("force_rmse", "force_mae", "energy_rmse", "du_std_per_atom",
                   "smoothness_ratio", "target_predicted"):
        s = summary[metric]
        print(f"  {metric:22s} {s['rho']:+.2f}  [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Do force errors rank actually-fitted models by their physics?")
