#!/usr/bin/env python3
"""Recompute, from the deposited records alone, every quantity added to the
manuscript in revision.

Nothing here is a new experiment. Every number is a function of files already in
``results/``; this script exists so that each of them is reproducible by one
command rather than asserted in prose.

Six blocks:

1. **Prediction-residual decomposition.** The manuscript originally quoted a
   single root-mean-square prediction residual per arm. Because every field in an
   arm is measured against the *same* reference-chain mean
   (``t_reference`` in ``experiments/exp07_designed_counterexamples/run.py`` and
   ``t_ref_mean`` in ``experiments/exp03_model_zoo/run.py``), the residuals share
   one realisation of the reference-chain error and are not independent. The rms
   is therefore decomposed into a common offset and the scatter about it, and the
   sign pattern is reported.

2. **Null-versus-aligned difference.** In the difference between two fields of
   the same arm the shared reference mean cancels identically. Only the legacy
   point contrast is auditable: the aligned/null direct chains reused random
   streams and their covariance was not deposited, so quadrature of marginal
   errors is neither a valid contrast error nor demonstrably conservative.

3. **Band robustness.** Spearman rho, with 95% percentile bootstrap intervals
   over members, inside the analyst-chosen band on three nested subsets: all 15
   members, the 13 with the two designed fields removed, and the 9 shell-family
   members, whose force RMSE values are all distinct. The estimator is the one
   in ``scripts/compute_band_correlations.py`` (2000 resamples, seed 20240517,
   resamples with fewer than three distinct values on an axis discarded), but
   each subset is given its own generator, so the intervals do not depend on
   evaluation order; the full band as deposited in ``band_correlations.json``
   is copied in alongside for comparison. Also the observable-error spread on
   the noise-subtracted and raw series, and with the designed fields removed.

4. **Width exponent.** Descriptive log-log slopes plus a legacy
   independent-marginal sensitivity range. There is no valid interval because
   cross-width covariance and raw chain units were not deposited, and the
   stored norm of per-bin errors is not the sampling SE of a curve norm. The
   primary description uses only widths satisfying ``r0 + 3w < r_on``; the
   six-point switch-contaminated fit is retained only as legacy description.

5. **Outer-bin share.** The fraction of the squared norm of the measured
   difference curve carried by the two outermost bins, which sit at the potential
   cutoff, averaged over the 34-member zoo.

6. **Second-order check.** Residual after the first-order term and after the
   first- plus second-order terms, for the two null fields.

Usage:  python3 scripts/compute_revision_statistics.py
Writes: results/revision_statistics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BAND = (1.5e-3, 6.0e-3)
N_BOOT = 20000
SEED = 20240517


def load(rel):
    return json.loads((ROOT / rel).read_text())


def legacy_component_se_norm(record):
    """Read the non-inferential width diagnostic across result schemas.

    New exp06 outputs name the quantity honestly.  The fallback exists only so
    the deposited legacy records remain readable; it must not make new outputs
    depend on the withdrawn ``direct_norm_error`` key.
    """
    if "legacy_component_se_norm_noninferential" in record:
        return float(record["legacy_component_se_norm_noninferential"])
    if "direct_norm_error" in record:
        return float(record["direct_norm_error"])
    raise KeyError(
        "width record lacks legacy_component_se_norm_noninferential "
        "(or the legacy direct_norm_error fallback)"
    )


def residual_decomposition(records, label):
    """Signed residuals (measured - predicted)/SE, and their decomposition."""
    r = np.array(
        [
            (x["target_measured"] - x["target_predicted"]) / x["target_error"]
            for x in records
        ]
    )
    n = len(r)
    mean = float(r.mean())
    scatter = float(r.std(ddof=1))
    return {
        "arm": label,
        "n": n,
        "residuals_sigma": [float(v) for v in r],
        "rms_sigma": float(np.sqrt((r**2).mean())),
        "common_offset_sigma": mean,
        "scatter_about_offset_sigma": scatter,
        "n_negative": int((r < 0).sum()),
        "t_statistic_on_offset": float(mean / (scatter / np.sqrt(n))),
        "sign_test_two_sided_p": float(2.0 * 0.5**n) if (r < 0).all() or (r > 0).all() else None,
    }


def main():
    out = {}

    # ---- 1. Prediction-residual decomposition -------------------------------
    exp07 = load("results/exp07_designed_counterexamples/records.json")
    zoo400 = load("results/exp03_model_zoo/records.json")
    zoo40 = load("results/exp03_model_zoo_n40/records.json")
    zoo40_kept = [x for x in zoo40 if x.get("direct_equilibrated", True)]
    zoo40_dropped = [x for x in zoo40 if not x.get("direct_equilibrated", True)]

    out["residual_decomposition"] = [
        residual_decomposition(exp07, "exp07 designed fields"),
        residual_decomposition(zoo400, "exp03 fitted zoo, budget 400"),
        residual_decomposition(zoo40_kept, "exp03 fitted zoo, budget 40 (equilibrated only)"),
    ]
    out["n40_arm"] = {
        "n_kept": len(zoo40_kept),
        "n_dropped_not_equilibrated": len(zoo40_dropped),
        "dropped": [
            {
                "name": x["name"],
                "force_rmse": x["force_rmse"],
                "target_measured": x["target_measured"],
                "target_error": x["target_error"],
            }
            for x in zoo40_dropped
        ],
    }

    # ---- 2. Null-versus-aligned difference ---------------------------------
    diffs = []
    for level in sorted({x["force_rms_level"] for x in exp07}):
        rec = {x["name"]: x for x in exp07 if x["force_rms_level"] == level}
        d = rec["aligned"]["target_measured"] - rec["null"]["target_measured"]
        e = float(np.hypot(rec["aligned"]["target_error"], rec["null"]["target_error"]))
        diffs.append(
            {
                "force_rms_level": level,
                "aligned_minus_null_pairs": float(d),
                "legacy_marginal_quadrature_error_pairs": e,
                "legacy_quadrature_signal_ratio": float(abs(d) / e),
                "contrast_uncertainty_auditable": False,
                "reason": (
                    "aligned/null direct-chain covariance was not deposited; "
                    "marginal quadrature is not known to be conservative"
                ),
            }
        )
    out["null_vs_aligned_difference"] = diffs

    # ---- 3. Band robustness -------------------------------------------------
    z = load("results/exp05_proxy_correlation/records.json")
    names = [x["name"] for x in z]
    force = np.array([x["force_rmse"] for x in z])
    raw = np.array([x["observable_error"] for x in z])
    noise = np.array([x["observable_error_noise"] for x in z])
    pred = np.array([x["predicted_norm"] for x in z])
    truth = np.sqrt(np.maximum(raw**2 - noise**2, 0.0))

    band = (force >= BAND[0]) & (force <= BAND[1])
    designed = np.array([n.startswith(("null", "aligned")) for n in names])
    shells = np.array([n.startswith("shell") for n in names])

    subsets = {
        "band_all": band,
        "band_minus_designed": band & ~designed,
        "band_shell_family": band & shells,
    }
    # Same estimator as scripts/compute_band_correlations.py: 95% percentile
    # bootstrap over members, 2000 resamples, resamples with fewer than three
    # distinct values on either axis discarded. Each call is given its own
    # generator seeded with SEED, so a subset's interval does not depend on
    # which other subsets were computed before it.
    def rho_ci(x, y, n_boot=2000):
        point = float(spearmanr(x, y).statistic)
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
                continue
            d = spearmanr(x[idx], y[idx]).statistic
            if np.isfinite(d):
                draws.append(d)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return point, float(lo), float(hi)

    out["band_robustness"] = {}
    for k, m in subsets.items():
        pf, lf, hf = rho_ci(force[m], truth[m])
        pp, lp, hp = rho_ci(pred[m], truth[m])
        out["band_robustness"][k] = {
            "n": int(m.sum()),
            "rho_force_rmse": pf,
            "rho_force_rmse_ci95": [lf, hf],
            "rho_response_prediction": pp,
            "rho_response_prediction_ci95": [lp, hp],
            "observable_error_spread_noise_subtracted": float(truth[m].max() / truth[m].min()),
        }
    out["band_robustness"]["band_all"]["observable_error_spread_raw"] = float(
        raw[band].max() / raw[band].min()
    )

    # The manuscript quotes the full band from the deposited
    # band_correlations.json, whose bootstrap draws all five proxies from one
    # generator stream; the interval below is the same estimator on an
    # independent stream, and the two are recorded together so the difference
    # between them is visible rather than silent.
    dep = load("results/exp05_proxy_correlation/band_correlations.json")
    out["band_robustness"]["band_all"]["as_deposited_band_correlations"] = {
        "rho_force_rmse": dep["band"]["force_rmse"]["rho"],
        "rho_force_rmse_ci95": [
            dep["band"]["force_rmse"]["ci_low"],
            dep["band"]["force_rmse"]["ci_high"],
        ],
        "rho_response_prediction": dep["band"]["predicted_norm"]["rho"],
        "rho_response_prediction_ci95": [
            dep["band"]["predicted_norm"]["ci_low"],
            dep["band"]["predicted_norm"]["ci_high"],
        ],
    }
    out["band_robustness"]["ties"] = {
        "n_band": int(band.sum()),
        "n_sharing_the_modal_force_rmse": int(
            (np.round(force[band], 12) == np.round(np.median(force[band]), 12)).sum()
        ),
        "modal_force_rmse": float(np.median(force[band])),
    }
    i_min = np.where(band)[0][np.argmin(truth[band])]
    out["band_robustness"]["spread_denominator"] = {
        "name": names[i_min],
        "observable_error_raw": float(raw[i_min]),
        "observable_error_noise": float(noise[i_min]),
        "observable_error_noise_subtracted": float(truth[i_min]),
        "resolved_from_zero": bool(raw[i_min] > 2.0 * noise[i_min]),
    }

    # The exp05 "null" field is built against the whole eight-bin curve, not the
    # single target bin of exp07. Record its curve so the two are not confused.
    j = names.index("null_f4e-03")
    out["curve_null_field"] = {
        "name": names[j],
        "measured_curve_pairs": [float(v) for v in z[j]["measured_curve"]],
        "largest_component_pairs": float(np.abs(z[j]["measured_curve"]).max()),
        "norm_pairs": float(raw[j]),
        "noise_pairs": float(noise[j]),
    }

    # ---- 4. Width exponent --------------------------------------------------
    w = load("results/exp06_response_validation/width_records.json")
    width = np.array([x["width"] for x in w])
    direct = np.array([x["direct_norm"] for x in w])
    derr = np.array([legacy_component_se_norm(x) for x in w])
    linear = np.array([x["linear_norm"] for x in w])

    def slope(y, x=width):
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    manifest = load("results/exp06_response_validation/manifest.json")
    width_spec = manifest["config"]["width_sweep"]
    cutoff = float(manifest["config"]["potential"]["cutoff"])
    r0 = float(width_spec["r0"])
    r_on = 0.85 * cutoff  # PairPerturbation's deposited default
    compliant = r0 + 3.0 * width < r_on

    def legacy_independent_marginal_sensitivity(y, yerr, x):
        """Exploratory perturbation of deposited marginal norm errors.

        This is deliberately *not* an inferential bootstrap: widths share a
        reference trajectory and random streams, their cross-width covariance
        was not deposited, and the stored component-SE norm is not the sampling
        SE of the curve norm.  The draws only show
        how a historically used independent-Gaussian approximation behaves.
        """
        rng = np.random.default_rng(SEED)
        samples = []
        for _ in range(N_BOOT):
            draw = y + rng.normal(0.0, yerr)
            if (draw <= 0).any():
                continue
            samples.append(slope(draw, x))
        return np.asarray(samples)

    draws_all = legacy_independent_marginal_sensitivity(direct, derr, width)
    draws_compliant = legacy_independent_marginal_sensitivity(
        direct[compliant], derr[compliant], width[compliant]
    )
    lo_all, hi_all = np.percentile(draws_all, [2.5, 97.5])
    lo_compliant, hi_compliant = np.percentile(draws_compliant, [2.5, 97.5])
    out["width_exponent"] = {
        "primary_interpretation": "inconclusive on the unswitched compliant subset",
        "unswitched_condition": "r0 + 3*w < r_on",
        "r0_angstrom": r0,
        "r_on_angstrom": r_on,
        "compliant_widths_angstrom": [float(v) for v in width[compliant]],
        "switch_truncated_widths_angstrom": [float(v) for v in width[~compliant]],
        "compliant_direct": slope(direct[compliant], width[compliant]),
        "compliant_legacy_independent_marginal_sensitivity_95pct_range": [
            float(lo_compliant), float(hi_compliant)
        ],
        "compliant_legacy_sensitivity_n_valid_draws": int(draws_compliant.size),
        "compliant_legacy_sensitivity_fraction_reaching_1.5": float(
            (draws_compliant >= 1.5).mean()
        ),
        "compliant_first_order": slope(linear[compliant], width[compliant]),
        "legacy_all_six_direct": slope(direct),
        "legacy_all_six_independent_marginal_sensitivity_95pct_range": [
            float(lo_all), float(hi_all)
        ],
        "legacy_all_six_sensitivity_n_valid_draws": int(draws_all.size),
        "legacy_all_six_sensitivity_fraction_reaching_1.5": float(
            (draws_all >= 1.5).mean()
        ),
        "legacy_all_six_first_order": slope(linear),
        "first_order_has_no_deposited_per_point_error": True,
        "sensitivity_is_not_an_inferential_interval": True,
        "sensitivity_limitations": (
            "shared reference/random streams and missing cross-width covariance; "
            "the stored component-SE norm is not the sampling SE of the curve norm"
        ),
        "direct_norms_pairs": [float(v) for v in direct],
        "monotone": bool(np.all(np.diff(direct) > 0)),
        "argmax_width": float(width[int(np.argmax(direct))]),
    }

    # ---- 5. Outer-bin share -------------------------------------------------
    curves = np.array([x["measured_curve"] for x in z])
    sq = curves**2
    share = sq / sq.sum(axis=1, keepdims=True)
    out["difference_curve_bin_shares"] = {
        "bin_centres_angstrom": [3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75],
        "mean_share_of_squared_norm": [float(v) for v in share.mean(axis=0)],
        "mean_share_outer_two_bins": float(share[:, -2:].sum(axis=1).mean()),
        "mean_share_peak_bin_4.25": float(share[:, 2].mean()),
    }

    # ---- 6. Second-order check ---------------------------------------------
    so = []
    for x in exp07:
        if x["name"] != "null":
            continue
        after1 = x["target_measured"] - x["target_predicted"]
        after2 = after1 - x["target_second_order"]
        so.append(
            {
                "force_rms_level": x["force_rms_level"],
                "measured": x["target_measured"],
                "measurement_error": x["target_error"],
                "second_order": x["target_second_order"],
                "second_order_error": x["target_second_order_error"],
                "residual_after_first_order": float(after1),
                "residual_after_first_and_second": float(after2),
                "second_order_improves_agreement": bool(abs(after2) < abs(after1)),
            }
        )
    out["second_order_null_fields"] = so

    print(json.dumps(out, indent=1))
    (ROOT / "results/revision_statistics.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
