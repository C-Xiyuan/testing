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
   the same arm the shared reference mean cancels identically. The quoted
   uncertainty is the quadrature sum of the two tabulated errors, which
   double-counts the shared term and is therefore an upper bound.

3. **Band robustness.** Spearman rho inside the analyst-chosen band on three
   nested subsets: all 15 members, the 13 with the two designed fields removed,
   and the 9 shell-family members, whose force RMSE values are all distinct.
   Also the observable-error spread on the noise-subtracted and raw series, and
   with the designed fields removed.

4. **Width exponent.** A bootstrap interval on the fitted log-log exponent of the
   direct-sampling width sweep, propagating the per-point blocking errors, plus
   the exponent with the largest (saturating) width dropped.

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
                "upper_bound_error_pairs": e,
                "lower_bound_sigma": float(abs(d) / e),
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
    out["band_robustness"] = {
        k: {
            "n": int(m.sum()),
            "rho_force_rmse": float(spearmanr(force[m], truth[m]).statistic),
            "rho_response_prediction": float(spearmanr(pred[m], truth[m]).statistic),
            "observable_error_spread_noise_subtracted": float(truth[m].max() / truth[m].min()),
        }
        for k, m in subsets.items()
    }
    out["band_robustness"]["band_all"]["observable_error_spread_raw"] = float(
        raw[band].max() / raw[band].min()
    )
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
    derr = np.array([x["direct_norm_error"] for x in w])
    linear = np.array([x["linear_norm"] for x in w])

    def slope(y, x=width):
        return float(np.polyfit(np.log(x), np.log(y), 1)[0])

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        y = direct + rng.normal(0.0, derr)
        if (y <= 0).any():
            continue
        draws.append(slope(y))
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out["width_exponent"] = {
        "direct": slope(direct),
        "direct_ci95": [float(lo), float(hi)],
        "direct_ci_n_valid_resamples": int(draws.size),
        "fraction_of_resamples_reaching_1.5": float((draws >= 1.5).mean()),
        "first_order": slope(linear),
        "first_order_has_no_deposited_per_point_error": True,
        "direct_dropping_largest_width": slope(direct[:-1], width[:-1]),
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
