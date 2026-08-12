#!/usr/bin/env python3
"""exp09 v2 -- conditional calibration on one frozen designed-field panel.

The legacy exp09 crossed eight reference chains with one set of direct chains
for eight fields, but then described the resulting 64 cells as though they were
independent.  It also reused the same four random streams across all fields,
pooled visibly heterogeneous direct-chain variances, omitted the covariance
between the reference mean and response prediction, and skipped stationarity
checks on the surrogate chains.  Those choices make the legacy numerical output
exploratory and prevent a calibration verdict.

This version makes the actual units explicit:

* a reference Markov chain is one independent reference unit;
* a direct Markov chain is one unit nested within its named field;
* the eight fields are a *fixed panel* built from one construction trajectory,
  not eight draws from a population of possible error fields;
* each reference chain's joint ``[<A>, first_1, ..., second_J]`` statistic is
  computed once, so chain-level resampling retains all shared covariance;
* complete direct chains are resampled separately inside each field; and
* the crossed chain levels are combined without a second inner bootstrap that
  would count finite-chain noise twice. No cell count is
  presented as an inferential sample size and no direct-chain variance is pooled
  across fields.

The output is deliberately fail-closed.  A completed run can support only a
conditional description of this frozen panel.  It cannot pass the confirmatory
calibration gate until independent construction panels and a physically justified
equivalence bound are supplied prospectively.  All production trajectories must
pass energy, target-observable, and (where applicable) delta-U stationarity checks,
and the raw arrays required to recompute every reported number are deposited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.statistics import blocking_analysis
from atomlab.units import beta as inverse_temperature
from experiments.common import ExperimentContext, main, require_frozen_protocol


EXPERIMENT_NAME = "exp09_calibration_replication_v2"
FIELD_PANEL_ID = "single-construction-panel-v2"
PRODUCTION_MASTER_SEED = 20260812
ACTUAL_CONSTRUCTION_PANELS = 1
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "protocols" / "exp09_v2.json"

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
    "observable": {"target_bin": [3.4, 3.9]},
    "sampling": {
        "n_construction": 2000,
        "n_reference": 1500,
        "n_direct": 1500,
        "n_leapfrog": 8,
        "step_size": 2e-3,
        "burn_in": 400,
        # This adaptation/discard occurs under each surrogate potential.  It is
        # not borrowed from the reference sampler.
        "surrogate_burn_in": 800,
        "n_melt": 400,
        "n_anneal": 1000,
    },
    "replication": {
        "n_reference_chains": 8,
        "n_direct_chains": 4,
    },
    "perturbations": {
        "basis_r_min": 3.0,
        "basis_r_max": 6.6,
        "n_basis": 14,
        "force_rms": 4.0e-3,
        "n_random": 6,
    },
    "analysis": {
        "n_bootstrap": 2000,
        "confidence_level": 0.95,
        # About 150 correlated-series checks are gated in one production run;
        # four sigma keeps the family-wise false-abort rate modest without
        # turning drift into a warning.
        "stationarity_n_sigma": 4.0,
        # A confirmatory population-level calibration claim requires multiple
        # independently constructed panels.  This experiment deliberately has
        # one and therefore reports conditional evidence only.
        "minimum_construction_panels_confirmatory": 3,
        # No arbitrary value is supplied.  A scientific tolerance must come from
        # the downstream observable requirement before confirmatory sampling.
        "equivalence_bound_pairs": None,
    },
}


def field_specific_direct_seed(master_seed: int, field_index: int, chain_index: int) -> int:
    """Return a unique, reproducible stream for one (field, direct-chain) unit."""
    if field_index < 0 or chain_index < 0:
        raise ValueError("field_index and chain_index must be non-negative")
    return int(master_seed + 1_000_000 + 10_000 * field_index + 97 * chain_index)


def field_specific_start_seed(master_seed: int, field_index: int, chain_index: int) -> int:
    """Independent surrogate melt/anneal stream for one direct-chain start."""
    return field_specific_direct_seed(master_seed, field_index, chain_index) + 50_000_000


def validate_master_seed(seed: int, *, quick_mode: bool) -> None:
    """Prevent the v2 production run from masquerading as a legacy-seed rerun.

    The legacy outputs used master seed 0.  The corrected design changes both the
    units and randomisation schedule, so its confirmatory candidate run is
    prospectively frozen at a new seed.  Seed 0 remains available only for cheap
    smoke testing, whose manifest is already labelled ``smoke_only``.
    """
    if quick_mode:
        return
    if int(seed) != PRODUCTION_MASTER_SEED:
        raise RuntimeError(
            "exp09 v2 production is frozen at master seed "
            f"{PRODUCTION_MASTER_SEED}; received {seed}. Legacy seed 0 is not an "
            "independent rerun (use --quick for smoke testing only)."
        )


def reference_seed(master_seed: int, chain_index: int) -> int:
    if chain_index < 0:
        raise ValueError("chain_index must be non-negative")
    return int(master_seed + 100_000 + 101 * chain_index)


def reference_start_seed(master_seed: int, chain_index: int) -> int:
    """Independent melt/anneal stream for a reference production chain."""
    return reference_seed(master_seed, chain_index) + 50_000_000


def series_stationarity(values, *, label: str, n_sigma: float = 3.0) -> dict:
    """Check a claim-bearing scalar series and return auditable half diagnostics."""
    values = np.asarray(values, dtype=float).ravel()
    half = values.size // 2
    if half < 8:
        raise ValueError(f"{label}: need at least 16 frames, got {values.size}")
    first = blocking_analysis(values[:half])
    second = blocking_analysis(values[half:])
    gap = abs(float(first.value) - float(second.value))
    error = float(np.hypot(first.error, second.error))
    if error > 0.0:
        z = gap / error
    else:
        z = 0.0 if gap <= 1e-14 else float("inf")
    result = {
        "first_half": float(first.value),
        "second_half": float(second.value),
        "difference": gap,
        "difference_z": z,
        "n_sigma": float(n_sigma),
        "passed": bool(z <= n_sigma),
    }
    if not result["passed"]:
        raise RuntimeError(f"{label}: half-chain drift is {z:.2f} sigma")
    return result


def _response_statistics(a, du, temperature: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute reference mean and first/second response terms on one joint sample."""
    a = np.asarray(a, dtype=float).ravel()
    du = np.asarray(du, dtype=float)
    if du.ndim != 2 or du.shape[1] != a.size:
        raise ValueError("du must have shape (n_fields, n_frames) aligned with a")
    if a.size < 4:
        raise ValueError("need at least four reference frames")
    b = inverse_temperature(float(temperature))
    ac = a - a.mean()
    dc = du - du.mean(axis=1, keepdims=True)
    first = -b * np.sum(dc * ac[None, :], axis=1) / (a.size - 1)
    second = 0.5 * b**2 * np.mean(dc**2 * ac[None, :], axis=1)
    return float(a.mean()), first, second


def _ols_intercept_slope(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    denom = float(np.sum(xc**2))
    if denom <= 1e-30:
        return float("nan"), float("nan")
    slope = float(np.sum(xc * (y - y.mean())) / denom)
    return float(y.mean() - slope * x.mean()), slope


def multiway_calibration_bootstrap(
    reference_a,
    reference_du,
    direct_a,
    temperature: float,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Complete-chain bootstrap for the crossed fixed-field design.

    Parameters
    ----------
    reference_a : (R, T_r)
        Target observable on independent reference chains.
    reference_du : (R, J, T_r)
        All J fields evaluated on the same reference frames.  Fields remain a
        fixed panel; the bootstrap never pretends to draw new fields.
    direct_a : (J, D, T_d)
        Target observable on D independent surrogate chains for each field.

    Returns
    -------
    dict
        Point estimates and B bootstrap draws. Reference chains are resampled
        as complete row units; direct chains are resampled independently within
        each field. Each chain-level statistic is computed once from its full
        retained trajectory. Between-chain scatter therefore already includes
        finite-trajectory noise and is not inflated by a second inner bootstrap.
    """
    reference_a = np.asarray(reference_a, dtype=float)
    reference_du = np.asarray(reference_du, dtype=float)
    direct_a = np.asarray(direct_a, dtype=float)
    if reference_a.ndim != 2:
        raise ValueError("reference_a must have shape (reference_chain, frame)")
    if reference_du.ndim != 3:
        raise ValueError("reference_du must have shape (reference_chain, field, frame)")
    if direct_a.ndim != 3:
        raise ValueError("direct_a must have shape (field, direct_chain, frame)")
    n_reference, n_reference_frames = reference_a.shape
    r2, n_fields, du_frames = reference_du.shape
    j2, n_direct, _ = direct_a.shape
    if (r2, du_frames) != (n_reference, n_reference_frames) or j2 != n_fields:
        raise ValueError("reference/direct arrays do not share reference and field axes")
    if n_reference < 2 or n_direct < 2:
        raise ValueError("need at least two reference and two direct chains per field")
    if n_resamples < 20:
        raise ValueError("need at least 20 bootstrap resamples")

    # Compute the complete-chain vector jointly so reference mean, all fields and
    # both response orders retain their covariance when a chain ID is resampled.
    reference_point = []
    for r in range(n_reference):
        ref_mean, first, second = _response_statistics(
            reference_a[r], reference_du[r], temperature,
        )
        reference_point.append(np.concatenate([[ref_mean], first, second]))
    reference_point = np.stack(reference_point)

    # Each field keeps its own direct-chain distribution; fields are fixed and
    # heteroskedasticity is never erased by a pooled error estimate.
    direct_point = direct_a.mean(axis=2)

    rng = np.random.default_rng(seed + 900_000)
    reference_ids = rng.integers(0, n_reference, size=(n_resamples, n_reference))
    reference_draw = reference_point[reference_ids].mean(axis=1)

    direct_draw = np.empty((n_resamples, n_fields), dtype=float)
    for j in range(n_fields):
        direct_ids = rng.integers(0, n_direct, size=(n_resamples, n_direct))
        direct_draw[:, j] = direct_point[j, direct_ids].mean(axis=1)

    ref_mean_point = float(reference_point[:, 0].mean())
    predicted_first_point = reference_point[:, 1:1 + n_fields].mean(axis=0)
    predicted_second_point = reference_point[:, 1 + n_fields:].mean(axis=0)
    direct_mean_point = direct_point.mean(axis=1)
    measured_point = direct_mean_point - ref_mean_point
    residual_first_point = measured_point - predicted_first_point
    residual_second_point = residual_first_point - predicted_second_point

    ref_mean_draw = reference_draw[:, 0]
    predicted_first_draw = reference_draw[:, 1:1 + n_fields]
    predicted_second_draw = reference_draw[:, 1 + n_fields:]
    measured_draw = direct_draw - ref_mean_draw[:, None]
    residual_first_draw = measured_draw - predicted_first_draw
    residual_second_draw = residual_first_draw - predicted_second_draw

    regression_first = np.array([
        _ols_intercept_slope(predicted_first_draw[b], measured_draw[b])
        for b in range(n_resamples)
    ])
    regression_second = np.array([
        _ols_intercept_slope(
            predicted_first_draw[b] + predicted_second_draw[b], measured_draw[b]
        )
        for b in range(n_resamples)
    ])
    point_regression_first = _ols_intercept_slope(predicted_first_point, measured_point)
    point_regression_second = _ols_intercept_slope(
        predicted_first_point + predicted_second_point, measured_point
    )

    return {
        "point": {
            "reference_mean": ref_mean_point,
            "direct_mean": direct_mean_point,
            "measured_shift": measured_point,
            "predicted_first": predicted_first_point,
            "predicted_second": predicted_second_point,
            "residual_first": residual_first_point,
            "residual_second": residual_second_point,
            "regression_first": np.asarray(point_regression_first),
            "regression_second": np.asarray(point_regression_second),
            "direct_chain_means": direct_point,
        },
        "draws": {
            "reference_mean": ref_mean_draw,
            "direct_mean": direct_draw,
            "measured_shift": measured_draw,
            "predicted_first": predicted_first_draw,
            "predicted_second": predicted_second_draw,
            "residual_first": residual_first_draw,
            "residual_second": residual_second_draw,
            "regression_first": regression_first,
            "regression_second": regression_second,
        },
        "units": {
            "n_reference_chains": n_reference,
            "n_direct_chains_per_field": n_direct,
            "n_fixed_fields": n_fields,
            "cell_count_not_an_n": int(n_reference * n_fields),
        },
    }


def _interval(draws, level: float) -> list:
    alpha = 0.5 * (1.0 - float(level))
    return np.quantile(np.asarray(draws, dtype=float), [alpha, 1.0 - alpha], axis=0).tolist()


def summarise_bootstrap(result, field_names, *, level: float) -> tuple[list, dict]:
    """Turn auditable bootstrap draws into per-field and fixed-panel summaries."""
    point = result["point"]
    draws = result["draws"]
    n_fields = len(field_names)
    if np.asarray(point["residual_first"]).shape != (n_fields,):
        raise ValueError("field_names does not match bootstrap output")

    first_ci = np.asarray(_interval(draws["residual_first"], level), dtype=float)
    second_ci = np.asarray(_interval(draws["residual_second"], level), dtype=float)
    records = []
    for j, name in enumerate(field_names):
        records.append({
            "field": name,
            "direct_chain_means": np.asarray(point["direct_chain_means"])[j].tolist(),
            "direct_mean": float(np.asarray(point["direct_mean"])[j]),
            "measured_shift": float(np.asarray(point["measured_shift"])[j]),
            "predicted_first": float(np.asarray(point["predicted_first"])[j]),
            "predicted_second": float(np.asarray(point["predicted_second"])[j]),
            "residual_first": float(np.asarray(point["residual_first"])[j]),
            "residual_first_ci": first_ci[:, j].tolist(),
            "residual_second": float(np.asarray(point["residual_second"])[j]),
            "residual_second_ci": second_ci[:, j].tolist(),
        })

    panel = {}
    for order in ("first", "second"):
        residual_point = np.asarray(point[f"residual_{order}"], dtype=float)
        residual_draw = np.asarray(draws[f"residual_{order}"], dtype=float)
        regression_point = np.asarray(point[f"regression_{order}"], dtype=float)
        regression_draw = np.asarray(draws[f"regression_{order}"], dtype=float)
        panel[order] = {
            "mean_residual": float(residual_point.mean()),
            "mean_residual_ci": _interval(residual_draw.mean(axis=1), level),
            "field_spread_sd": float(residual_point.std(ddof=1)),
            "field_spread_sd_ci": _interval(residual_draw.std(axis=1, ddof=1), level),
            "ols_intercept": float(regression_point[0]),
            "ols_intercept_ci": _interval(regression_draw[:, 0], level),
            "ols_slope": float(regression_point[1]),
            "ols_slope_ci": _interval(regression_draw[:, 1], level),
            "fields_with_zero_in_residual_ci": int(np.sum(
                (first_ci[0] <= 0.0) & (first_ci[1] >= 0.0)
                if order == "first" else
                (second_ci[0] <= 0.0) & (second_ci[1] >= 0.0)
            )),
        }
    return records, panel


def evidence_gate(
    *,
    n_reference: int,
    n_direct: int,
    n_construction_panels: int,
    minimum_construction_panels: int,
    equivalence_bound,
    raw_saved: bool,
    stationarity_passed: bool,
    quick_mode: bool,
    fixed_field_panel: bool = True,
) -> dict:
    """Fail closed: separate an auditable exploratory run from confirmation."""
    exploratory_reasons = []
    if n_reference < 2:
        exploratory_reasons.append("fewer_than_two_reference_chain_units")
    if n_direct < 2:
        exploratory_reasons.append("fewer_than_two_direct_chain_units_per_field")
    if not raw_saved:
        exploratory_reasons.append("raw_evidence_missing")
    if not stationarity_passed:
        exploratory_reasons.append("stationarity_not_verified")
    if quick_mode:
        exploratory_reasons.append("quick_mode_is_smoke_only")
    exploratory_passed = not exploratory_reasons

    confirmatory_reasons = list(exploratory_reasons)
    if fixed_field_panel:
        confirmatory_reasons.append("fixed_field_panel_allows_conditional_inference_only")
    if n_construction_panels < minimum_construction_panels:
        confirmatory_reasons.append("single_fixed_field_panel_no_population_inference")
    if equivalence_bound is None:
        confirmatory_reasons.append("no_prospectively_justified_equivalence_bound")
    return {
        "exploratory_conditional": {
            "passed": exploratory_passed,
            "reasons": exploratory_reasons,
        },
        "confirmatory_calibration": {
            "passed": not confirmatory_reasons,
            "reasons": confirmatory_reasons,
        },
        "evidence_status": (
            "confirmatory_conditional" if not confirmatory_reasons else
            "exploratory_conditional" if exploratory_passed else
            "not_evaluable"
        ),
    }


def build_system(ctx):
    # Keep the optional simulation stack out of module import so the statistical
    # design helpers can be audited in a lightweight environment.
    from atomlab.build import fcc, scale_to_density
    from atomlab.potentials.lennard_jones import LennardJones

    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(
        fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"]
    )
    return cfg, LennardJones(
        epsilon=p["epsilon"], sigma=p["sigma"], cutoff=p["cutoff"], mode=p["mode"]
    )


def sample(ctx, cfg, potential, n_samples, seed, *, label, burn_in=None):
    """Sample one chain and fail immediately on energy non-stationarity."""
    from atomlab.sampling import hybrid_monte_carlo
    from experiments.equilibrate import check_equilibrated

    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"],
        burn_in=int(s["burn_in"] if burn_in is None else burn_in), seed=int(seed),
    )
    if report.acceptance < 0.2:
        raise RuntimeError(f"{label}: acceptance {report.acceptance:.2f} too low")
    stationarity = check_equilibrated(
        traj, label=label, check_order=True,
        n_sigma=(max(20.0, float(ctx.config["analysis"]["stationarity_n_sigma"]))
                 if ctx.quick else
                 float(ctx.config["analysis"]["stationarity_n_sigma"])),
    )
    return traj, report, stationarity


def run(ctx: ExperimentContext) -> dict:
    from atomlab.potentials.perturbations import (
        AlignedPerturbation,
        NullSpacePerturbation,
        RandomShellPerturbation,
        ShellBasis,
        build_shell_design,
    )
    from experiments.equilibrate import equilibrated_configuration
    from experiments.observables_lib import PairBinObservable

    validate_master_seed(ctx.seed, quick_mode=ctx.quick)
    require_frozen_protocol(
        ctx, protocol_path=PROTOCOL_PATH,
        expected_protocol=EXPERIMENT_NAME, expected_version=2,
    )
    initial_cfg, potential = build_system(ctx)
    temperature = float(ctx.config["system"]["temperature"])
    cutoff = float(ctx.config["potential"]["cutoff"])
    target = PairBinObservable(
        np.asarray(ctx.config["observable"]["target_bin"], dtype=float), cutoff=cutoff
    )
    rep = ctx.config["replication"]
    pcfg = ctx.config["perturbations"]
    analysis = ctx.config["analysis"]
    stationarity_n_sigma = float(analysis["stationarity_n_sigma"])
    if ctx.quick:
        stationarity_n_sigma = max(20.0, stationarity_n_sigma)

    # One equilibrated seed configuration is only a launch point.  Every
    # reference chain receives its own RNG and production burn-in; every direct
    # chain receives an independent field-specific melt, anneal and surrogate
    # burn-in below.
    with ctx.timed("base_equilibration"):
        s = ctx.config["sampling"]
        base_cfg, equilibration = equilibrated_configuration(
            initial_cfg, potential, temperature,
            n_melt=int(ctx.scaled("sampling.n_melt")),
            n_anneal=int(ctx.scaled("sampling.n_anneal")),
            n_leapfrog=s["n_leapfrog"], step_size=s["step_size"], seed=ctx.seed,
            q6_target=1.0 if ctx.quick else 0.20,
        )

    with ctx.timed("construction"):
        construction_seed = int(ctx.seed + 10_000)
        construction, construction_report, construction_energy_stationarity = sample(
            ctx, base_cfg, potential, ctx.scaled("sampling.n_construction"),
            construction_seed, label="construction",
        )
        construction_a = target.evaluate_trajectory(construction).ravel()
        construction_a_stationarity = series_stationarity(
            construction_a, label="construction target observable",
            n_sigma=stationarity_n_sigma,
        )
        construction_frames = [construction.frame(i) for i in range(construction.n_frames)]
        centres = np.linspace(
            pcfg["basis_r_min"], pcfg["basis_r_max"], int(pcfg["n_basis"])
        )
        basis = ShellBasis(
            centres, np.full(centres.size, float(centres[1] - centres[0])),
            cutoff, r_on=cutoff - 1.0,
        )
        design = build_shell_design(construction_frames, basis)
        common = dict(
            basis=basis, design=design, cutoff=cutoff,
            target_force_rms=pcfg["force_rms"], seed=ctx.seed + 20_000,
        )
        fields = [
            ("null", NullSpacePerturbation(
                construction_frames, target, temperature, **common
            )),
            ("aligned", AlignedPerturbation(
                construction_frames, target, temperature, **common
            )),
        ]
        n_random = max(1, int(pcfg["n_random"] * 0.05)) if ctx.quick \
            else int(pcfg["n_random"])
        for k in range(n_random):
            fields.append((
                f"random{k}",
                RandomShellPerturbation(
                    construction_frames, target, temperature,
                    **dict(common, seed=ctx.seed + 21_000 + k),
                ),
            ))
    field_names = [name for name, _ in fields]
    n_fields = len(fields)

    references = []
    with ctx.timed("reference_chains"):
        n_reference = 2 if ctx.quick else int(rep["n_reference_chains"])
        for r in range(n_reference):
            seed = reference_seed(ctx.seed, r)
            start_seed = reference_start_seed(ctx.seed, r)
            reference_start, reference_equilibration = equilibrated_configuration(
                initial_cfg, potential, temperature,
                n_melt=int(ctx.scaled("sampling.n_melt")),
                n_anneal=int(ctx.scaled("sampling.n_anneal")),
                n_leapfrog=ctx.config["sampling"]["n_leapfrog"],
                step_size=ctx.config["sampling"]["step_size"],
                seed=start_seed,
                q6_target=1.0 if ctx.quick else 0.20,
            )
            traj, report, energy_stationarity = sample(
                ctx, reference_start, potential, ctx.scaled("sampling.n_reference"),
                seed, label=f"reference {r}",
            )
            a = target.evaluate_trajectory(traj).ravel()
            a_stationarity = series_stationarity(
                a, label=f"reference {r} target observable",
                n_sigma=stationarity_n_sigma,
            )
            frames = [traj.frame(i) for i in range(traj.n_frames)]
            du = np.stack([
                np.asarray([field.energy(frame) for frame in frames], dtype=float)
                for _, field in fields
            ])
            du_stationarity = [
                series_stationarity(
                    du[j], label=f"reference {r} {field_names[j]} delta-U",
                    n_sigma=stationarity_n_sigma,
                )
                for j in range(n_fields)
            ]
            references.append({
                "index": r,
                "seed": seed,
                "equilibration_seed": start_seed,
                "equilibration": dict(reference_equilibration.__dict__),
                "trajectory": traj,
                "a": a,
                "du": du,
                "report": report,
                "energy_stationarity": energy_stationarity,
                "a_stationarity": a_stationarity,
                "du_stationarity": du_stationarity,
            })

    n_direct = 2 if ctx.quick else int(rep["n_direct_chains"])
    direct_a = []
    direct_u = []
    direct_du = []
    direct_positions = []
    direct_cells = []
    direct_seeds = np.empty((n_fields, n_direct), dtype=np.int64)
    direct_start_seeds = np.empty((n_fields, n_direct), dtype=np.int64)
    direct_start_positions = []
    direct_start_cells = []
    direct_diagnostics = []
    with ctx.timed("direct_chains"):
        for j, (name, field) in enumerate(fields):
            field_a, field_u, field_du, field_diagnostics = [], [], [], []
            field_positions, field_cells = [], []
            field_start_positions, field_start_cells = [], []
            for d in range(n_direct):
                seed = field_specific_direct_seed(ctx.seed, j, d)
                start_seed = field_specific_start_seed(ctx.seed, j, d)
                # Every field/chain pair gets an independent melt and anneal
                # under its own surrogate potential.  This removes both the
                # legacy shared stream and the shared-start dependence.
                start, start_equilibration = equilibrated_configuration(
                    initial_cfg, potential + field, temperature,
                    n_melt=int(ctx.scaled("sampling.n_melt")),
                    n_anneal=int(ctx.scaled("sampling.n_anneal")),
                    n_leapfrog=ctx.config["sampling"]["n_leapfrog"],
                    step_size=ctx.config["sampling"]["step_size"], seed=start_seed,
                    q6_target=1.0 if ctx.quick else 0.20,
                )
                traj, report, energy_stationarity = sample(
                    ctx, start, potential + field, ctx.scaled("sampling.n_direct"),
                    seed, label=f"{name} direct {d}",
                    burn_in=int(ctx.scaled("sampling.surrogate_burn_in")),
                )
                values = target.evaluate_trajectory(traj).ravel()
                frames = [traj.frame(i) for i in range(traj.n_frames)]
                delta_u = np.asarray([field.energy(frame) for frame in frames], dtype=float)
                a_stationarity = series_stationarity(
                    values, label=f"{name} direct {d} target observable",
                    n_sigma=stationarity_n_sigma,
                )
                du_stationarity = series_stationarity(
                    delta_u, label=f"{name} direct {d} delta-U",
                    n_sigma=stationarity_n_sigma,
                )
                field_a.append(values)
                field_u.append(np.asarray(traj.scalars["potential_energy"], dtype=float))
                field_du.append(delta_u)
                field_positions.append(np.asarray(traj.positions, dtype=float))
                field_cells.append(np.asarray(traj.cells, dtype=float))
                field_start_positions.append(np.asarray(start.positions, dtype=float))
                field_start_cells.append(np.asarray(start.cell, dtype=float))
                direct_seeds[j, d] = seed
                direct_start_seeds[j, d] = start_seed
                field_diagnostics.append({
                    "field": name,
                    "chain": d,
                    "seed": seed,
                    "start_seed": start_seed,
                    "start_id": f"field-{j}-chain-{d}-seed-{start_seed}",
                    "start_equilibration": dict(start_equilibration.__dict__),
                    "surrogate_burn_in": int(ctx.scaled("sampling.surrogate_burn_in")),
                    "acceptance": float(report.acceptance),
                    "final_step_size": float(report.final_step_size),
                    "energy_stationarity": energy_stationarity,
                    "target_stationarity": a_stationarity,
                    "delta_u_stationarity": du_stationarity,
                })
            direct_a.append(np.stack(field_a))
            direct_u.append(np.stack(field_u))
            direct_du.append(np.stack(field_du))
            direct_positions.append(np.stack(field_positions))
            direct_cells.append(np.stack(field_cells))
            direct_start_positions.append(np.stack(field_start_positions))
            direct_start_cells.append(np.stack(field_start_cells))
            direct_diagnostics.append(field_diagnostics)

    reference_a = np.stack([record["a"] for record in references])
    reference_u = np.stack([
        np.asarray(record["trajectory"].scalars["potential_energy"], dtype=float)
        for record in references
    ])
    reference_du = np.stack([record["du"] for record in references])
    direct_a_array = np.stack(direct_a)
    direct_u_array = np.stack(direct_u)
    direct_du_array = np.stack(direct_du)
    direct_positions_array = np.stack(direct_positions)
    direct_cells_array = np.stack(direct_cells)
    direct_start_positions_array = np.stack(direct_start_positions)
    direct_start_cells_array = np.stack(direct_start_cells)
    construction_du = np.stack([
        np.asarray([field.energy(frame) for frame in construction_frames], dtype=float)
        for _, field in fields
    ])
    construction_du_stationarity = [
        series_stationarity(
            construction_du[j], label=f"construction {field_names[j]} delta-U",
            n_sigma=stationarity_n_sigma,
        )
        for j in range(n_fields)
    ]
    field_weights = np.stack([
        np.asarray(field.weights, dtype=float) for _, field in fields
    ])

    n_bootstrap = int(analysis["n_bootstrap"])
    if ctx.quick:
        n_bootstrap = max(100, n_bootstrap // 10)
    bootstrap = multiway_calibration_bootstrap(
        reference_a, reference_du, direct_a_array, temperature,
        n_resamples=n_bootstrap, seed=ctx.seed + 2_000_000,
    )
    records, panel = summarise_bootstrap(
        bootstrap, field_names, level=float(analysis["confidence_level"])
    )

    # Save the evidence before claiming even exploratory evaluability.  Bootstrap
    # draws are included so every interval can be reproduced without RNG replay.
    raw_path = ctx.save_npz(
        "raw",
        field_names=np.asarray(field_names),
        construction_a=construction_a,
        construction_u=np.asarray(construction.scalars["potential_energy"], dtype=float),
        construction_positions=np.asarray(construction.positions, dtype=float),
        construction_cells=np.asarray(construction.cells, dtype=float),
        construction_du=construction_du,
        reference_a=reference_a,
        reference_u=reference_u,
        reference_du=reference_du,
        reference_positions=np.stack([
            record["trajectory"].positions for record in references
        ]),
        reference_cells=np.stack([
            record["trajectory"].cells for record in references
        ]),
        direct_a=direct_a_array,
        direct_u=direct_u_array,
        direct_du=direct_du_array,
        direct_positions=direct_positions_array,
        direct_cells=direct_cells_array,
        direct_start_positions=direct_start_positions_array,
        direct_start_cells=direct_start_cells_array,
        field_weights=field_weights,
        construction_seed=np.asarray(construction_seed, dtype=np.int64),
        reference_seeds=np.asarray([r["seed"] for r in references], dtype=np.int64),
        reference_equilibration_seeds=np.asarray([
            r["equilibration_seed"] for r in references
        ], dtype=np.int64),
        direct_seeds=direct_seeds,
        direct_start_seeds=direct_start_seeds,
        species=np.asarray(construction.template.species),
        pbc=np.asarray(construction.template.pbc),
        bootstrap_reference_mean=bootstrap["draws"]["reference_mean"],
        bootstrap_direct_mean=bootstrap["draws"]["direct_mean"],
        bootstrap_measured_shift=bootstrap["draws"]["measured_shift"],
        bootstrap_predicted_first=bootstrap["draws"]["predicted_first"],
        bootstrap_predicted_second=bootstrap["draws"]["predicted_second"],
        bootstrap_residual_first=bootstrap["draws"]["residual_first"],
        bootstrap_residual_second=bootstrap["draws"]["residual_second"],
        bootstrap_regression_first=bootstrap["draws"]["regression_first"],
        bootstrap_regression_second=bootstrap["draws"]["regression_second"],
    )

    diagnostics = {
        "field_panel_id": FIELD_PANEL_ID,
        "field_panel_scope": "fixed; no field-population inference",
        "base_equilibration": dict(equilibration.__dict__),
        "construction": {
            "seed": construction_seed,
            "acceptance": float(construction_report.acceptance),
            "final_step_size": float(construction_report.final_step_size),
            "energy_stationarity": construction_energy_stationarity,
            "target_stationarity": construction_a_stationarity,
            "delta_u_stationarity": dict(zip(
                field_names, construction_du_stationarity
            )),
        },
        "references": [{
            "chain": r["index"],
            "seed": r["seed"],
            "equilibration_seed": r["equilibration_seed"],
            "equilibration": r["equilibration"],
            "acceptance": float(r["report"].acceptance),
            "final_step_size": float(r["report"].final_step_size),
            "energy_stationarity": r["energy_stationarity"],
            "target_stationarity": r["a_stationarity"],
            "delta_u_stationarity": dict(zip(field_names, r["du_stationarity"])),
        } for r in references],
        "direct": dict(zip(field_names, direct_diagnostics)),
        "raw_schema": {
            "reference_a": "[reference_chain, frame] target pair count",
            "reference_u": "[reference_chain, frame] reference total potential energy",
            "reference_du": "[reference_chain, field, frame] field energy",
            "reference_positions": "[reference_chain, frame, atom, xyz] raw coordinates",
            "reference_cells": "[reference_chain, frame, 3, 3] simulation cells",
            "direct_a": "[field, direct_chain, frame] target pair count",
            "direct_u": "[field, direct_chain, frame] surrogate total potential energy",
            "direct_du": "[field, direct_chain, frame] field energy",
            "direct_positions": "[field, direct_chain, frame, atom, xyz] raw coordinates",
            "direct_cells": "[field, direct_chain, frame, 3, 3] simulation cells",
            "direct_start_positions": "[field, direct_chain, atom, xyz] independent starts",
            "direct_start_cells": "[field, direct_chain, 3, 3] start cells",
            "field_weights": "[field, shell_basis] designed coefficients",
            "bootstrap_*": "multiway bootstrap draws used for every interval",
        },
    }
    ctx.save_json("diagnostics", diagnostics)

    gate = evidence_gate(
        n_reference=len(references), n_direct=n_direct,
        n_construction_panels=ACTUAL_CONSTRUCTION_PANELS,
        minimum_construction_panels=int(
            analysis["minimum_construction_panels_confirmatory"]
        ),
        equivalence_bound=analysis["equivalence_bound_pairs"],
        raw_saved=raw_path.exists(), stationarity_passed=True,
        quick_mode=ctx.quick, fixed_field_panel=True,
    )
    summary = {
        "experiment_version": 2,
        "field_panel_id": FIELD_PANEL_ID,
        "scope": "conditional on one frozen constructed-field panel",
        "independent_units": {
            "reference": "Markov chain",
            "direct": "Markov chain nested within field",
            "field": "fixed panel member, not an independent population draw",
        },
        "n_reference_chains": len(references),
        "n_direct_chains_per_field": n_direct,
        "n_fixed_fields": n_fields,
        "n_construction_panels": ACTUAL_CONSTRUCTION_PANELS,
        "crossed_cells_descriptive_only": len(references) * n_fields,
        "n_bootstrap": n_bootstrap,
        "confidence_level": float(analysis["confidence_level"]),
        "panel": panel,
        "gate": gate,
        "claim": (
            "conditional exploratory decomposition only; no estimator calibration "
            "or field-population conclusion"
        ),
    }
    ctx.save_json("records", records)
    ctx.save_json("summary", summary)
    report_summary(summary, records)
    return summary


def report_summary(summary, records):
    print("\n  --- P0-2 exp09 v2: fixed-panel conditional analysis ---")
    print(
        f"  units: {summary['n_reference_chains']} reference chains; "
        f"{summary['n_direct_chains_per_field']} direct chains within each of "
        f"{summary['n_fixed_fields']} fixed fields"
    )
    print(
        "  crossed cells are descriptive only: "
        f"{summary['crossed_cells_descriptive_only']} is not an inferential n"
    )
    for order, result in summary["panel"].items():
        print(
            f"  {order:6s}: fixed-panel mean residual {result['mean_residual']:+.3f} "
            f"CI {np.round(result['mean_residual_ci'], 3)}; "
            f"slope {result['ols_slope']:.3f} "
            f"CI {np.round(result['ols_slope_ci'], 3)}"
        )
    print(f"  evidence status: {summary['gate']['evidence_status']}")
    reasons = summary["gate"]["confirmatory_calibration"]["reasons"]
    print(f"  confirmatory gate: FAIL CLOSED ({'; '.join(reasons)})")
    print("  per-field residual CIs:")
    for record in records:
        print(
            f"    {record['field']:9s} first {record['residual_first']:+.3f} "
            f"{np.round(record['residual_first_ci'], 3)}; "
            f"second {record['residual_second']:+.3f} "
            f"{np.round(record['residual_second_ci'], 3)}"
        )


if __name__ == "__main__":
    main(
        run,
        default_config=DEFAULTS,
        description=(
            "Conditional calibration decomposition with reference-chain and "
            "field-specific direct-chain units"
        ),
        name=EXPERIMENT_NAME,
        protocol_path=PROTOCOL_PATH,
        protocol_name=EXPERIMENT_NAME,
        protocol_version=2,
    )
