#!/usr/bin/env python3
"""Is the band result an artefact of where the window was drawn?

Section 4.4 of the manuscript reports that within a factor-2.6 window of force
RMSE the rank correlation with observable error collapses to 0.34. The window
was chosen by the analyst after seeing the data, which the limitations section
concedes. A post-hoc window is a real objection: with 34 points one can usually
find *some* interval where a correlation vanishes.

This script removes the objection by refusing to choose. It sweeps every
multiplicative window of every width over the whole force-RMSE range and reports
the correlation inside each, so the claim becomes a property of the data rather
than of one interval. Two things are then checkable:

  - whether the reported window is typical or cherry-picked, and
  - whether the response prediction stays informative in the same windows where
    force error stops being informative, which is the actual claim.

Nothing here is fitted. It is a sweep of an existing measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_MEMBERS = 8          # below this a rank correlation is meaningless
WIDTHS = [2.0, 2.6, 3.0, 4.0, 6.0, 10.0]
REPORTED = (1.5e-3, 6.0e-3)


def load():
    records = json.loads((ROOT / "results/exp05_proxy_correlation/records.json").read_text())
    force = np.array([r["force_rmse"] for r in records])
    raw = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    predicted = np.array([r["predicted_norm"] for r in records])
    # |dA| is a norm, so noise inflates it; subtract in quadrature.
    truth = np.sqrt(np.maximum(raw**2 - noise**2, 0.0))
    return force, truth, predicted


def sweep(force, truth, predicted, width, n_centres=120):
    """Slide a multiplicative window of given width across the force range."""
    lo, hi = force.min(), force.max()
    centres = np.exp(np.linspace(np.log(lo * np.sqrt(width)),
                                 np.log(hi / np.sqrt(width)), n_centres))
    out = []
    for c in centres:
        a, b = c / np.sqrt(width), c * np.sqrt(width)
        m = (force >= a) & (force <= b)
        if m.sum() < MIN_MEMBERS:
            continue
        rf = spearmanr(force[m], truth[m]).statistic
        rp = spearmanr(predicted[m], truth[m]).statistic
        spread = truth[m].max() / max(truth[m].min(), 1e-12)
        out.append({"centre": float(c), "lo": float(a), "hi": float(b),
                    "n": int(m.sum()), "rho_force": float(rf),
                    "rho_predicted": float(rp), "observable_spread": float(spread)})
    return out


def main():
    force, truth, predicted = load()
    print(f"{len(force)} surrogates, force RMSE {force.min():.2e} to {force.max():.2e} eV/A")
    print(f"reported window {REPORTED[0]:.1e}-{REPORTED[1]:.1e} "
          f"(width {REPORTED[1]/REPORTED[0]:.1f}x)\n")

    print(f"{'width':>6} {'windows':>8} {'rho_force':>22} {'rho_predicted':>22} "
          f"{'obs spread':>11}")
    print(f"{'':>6} {'':>8} {'median  [min, max]':>22} {'median  [min, max]':>22} {'median':>11}")

    summary = {}
    for width in WIDTHS:
        rows = sweep(force, truth, predicted, width)
        if not rows:
            print(f"{width:6.1f} {0:8d}   (no window holds {MIN_MEMBERS}+ members)")
            continue
        rf = np.array([r["rho_force"] for r in rows])
        rp = np.array([r["rho_predicted"] for r in rows])
        sp = np.array([r["observable_spread"] for r in rows])
        print(f"{width:6.1f} {len(rows):8d} "
              f"{np.median(rf):8.2f}  [{rf.min():5.2f}, {rf.max():5.2f}] "
              f"{np.median(rp):8.2f}  [{rp.min():5.2f}, {rp.max():5.2f}] "
              f"{np.median(sp):11.0f}x")
        summary[f"{width:g}x"] = {
            "n_windows": len(rows),
            "rho_force_median": float(np.median(rf)),
            "rho_force_min": float(rf.min()), "rho_force_max": float(rf.max()),
            "rho_predicted_median": float(np.median(rp)),
            "rho_predicted_min": float(rp.min()), "rho_predicted_max": float(rp.max()),
            "observable_spread_median": float(np.median(sp)),
            "fraction_force_below_0.5": float((rf < 0.5).mean()),
            "fraction_predicted_above_0.8": float((rp > 0.8).mean()),
        }

    # Where does the reported window sit in the distribution of same-width windows?
    rows = sweep(force, truth, predicted, REPORTED[1] / REPORTED[0])
    rf = np.array([r["rho_force"] for r in rows])
    reported_rho = spearmanr(
        force[(force >= REPORTED[0]) & (force <= REPORTED[1])],
        truth[(force >= REPORTED[0]) & (force <= REPORTED[1])]).statistic
    percentile = float((rf < reported_rho).mean() * 100)
    print(f"\nthe reported window's rho_force = {reported_rho:.2f} sits at the "
          f"{percentile:.0f}th percentile")
    print(f"of the {len(rows)} same-width windows -- "
          f"{'typical, not cherry-picked' if 15 < percentile < 85 else 'ATYPICAL: check for selection'}")

    all_widths = {w: sweep(force, truth, predicted, w) for w in WIDTHS}
    flat_f = np.array([r["rho_force"] for rows in all_widths.values() for r in rows])
    flat_p = np.array([r["rho_predicted"] for rows in all_widths.values() for r in rows])
    print(f"\nover all {len(flat_f)} windows of all widths:")
    print(f"  rho(force)     median {np.median(flat_f):+.2f}, "
          f"{100*(flat_f < 0.5).mean():.0f}% below 0.5")
    print(f"  rho(predicted) median {np.median(flat_p):+.2f}, "
          f"{100*(flat_p > 0.8).mean():.0f}% above 0.8")

    summary["reported_window"] = {
        "lo": REPORTED[0], "hi": REPORTED[1],
        "rho_force": float(reported_rho),
        "percentile_among_same_width_windows": percentile,
    }
    summary["all_windows"] = {
        "n": int(len(flat_f)),
        "rho_force_median": float(np.median(flat_f)),
        "rho_force_fraction_below_0.5": float((flat_f < 0.5).mean()),
        "rho_predicted_median": float(np.median(flat_p)),
        "rho_predicted_fraction_above_0.8": float((flat_p > 0.8).mean()),
    }
    out = ROOT / "results/exp05_proxy_correlation/band_robustness.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
