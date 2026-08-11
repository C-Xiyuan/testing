#!/usr/bin/env python3
"""Build the two headline figures from the completed experiment results.

Figure 1 -- the claim.  At a force RMSE held fixed to within 0.5 %, the measured
observable error is a monotone function of the correlation between the error
field and the observable, spanning zero to ten pairs.  Force error cannot see
that axis at all.

Figure 2 -- the prediction.  Every measured shift against the one predicted from
the reference trajectory alone, with no simulation of the surrogate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atomlab.utils import plotting as P


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def figure_one(records, width_records):
    """Correlation is the axis that matters; force error is not."""
    P.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    fig.subplots_adjust(wspace=0.42, top=0.80)

    levels = sorted({r["force_rms_level"] for r in records})
    for idx, level in enumerate(levels):
        group = sorted((r for r in records if r["force_rms_level"] == level),
                       key=lambda r: r["target_correlation"])
        # Signed, not absolute. Theory says the shift is proportional to
        # -beta sigma_A sigma_dU rho, so the signed correlation is the axis the
        # relation is monotone in; taking |rho| folds the line in half.
        rho = np.array([r["target_correlation"] for r in group])
        measured = np.array([r["target_measured"] for r in group])
        errors = np.array([r["target_error"] for r in group])
        spread = np.ptp([r["force_rms_out_of_sample"] for r in group]) / level
        P.series(ax1, rho, measured, errors, index=idx, fill=False,
                 label=f"force RMSE {level:.0e} eV/Å  (±{100 * spread:.1f}%)")
        for r, x, y in zip(group, rho, measured):
            if r["name"] in ("null", "aligned") and idx == len(levels) - 1:
                ax1.annotate(r["name"], xy=(x, y), xytext=(0.4, -0.9),
                             textcoords="offset fontsize", fontsize=7,
                             color=P.TEXT_SECONDARY)

    ax1.axhline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
    ax1.axvline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
    ax1.set_xlabel(r"$\rho(A,\,\delta U)$")
    ax1.set_ylabel("measured shift (pairs)")
    ax1.set_title("What controls the damage", pad=8)
    ax1.legend(fontsize=6.5, loc="upper right")

    widths = np.array([r["width"] for r in width_records])
    predicted = np.array([abs(r["linear_norm"]) for r in width_records])
    measured = np.array([abs(r["direct_norm"]) for r in width_records])
    errors = np.array([r["direct_norm_error"] for r in width_records])
    force = np.array([r["force_rms"] for r in width_records])

    P.series(ax2, widths, measured, errors, index=0, label="measured", fill=False)
    P.series(ax2, widths, predicted, np.zeros_like(predicted), index=1,
             label="first-order prediction", fill=False)
    ax2.set_xscale("log")
    ax2.set_xlabel("width of the error field (Å)")
    ax2.set_ylabel(r"$\|\Delta\langle A\rangle\|$ (pairs)")
    ax2.set_title(f"Same force error, {measured.max() / measured.min():.0f}× the damage", pad=8)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.text(0.03, 0.94, f"force RMSE constant to "
             f"{100 * np.ptp(force) / force.mean():.1f}%",
             transform=ax2.transAxes, fontsize=6.5, color=P.TEXT_SECONDARY, va="top")

    P.add_panel_labels([ax1, ax2], offset=(-0.20, 1.10))
    return fig


def figure_two(records):
    """Predicted from the reference trajectory alone, against measured."""
    P.use_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))

    predicted = np.array([r["target_predicted"] for r in records])
    measured = np.array([r["target_measured"] for r in records])
    errors = np.array([r["target_error"] for r in records])
    names = [r["name"] for r in records]

    lim = [min(predicted.min(), measured.min()) - 1.5,
           max(predicted.max(), measured.max()) + 1.5]
    ax.plot(lim, lim, color=P.TEXT_SECONDARY, linestyle="--", linewidth=0.9, zorder=1)

    style = {"null": 0, "aligned": 1, "random0": 2, "random1": 2}
    seen = set()
    for name, x, y, e in zip(names, predicted, measured, errors):
        idx = style.get(name, 2)
        label = ("random" if idx == 2 else name) if idx not in seen else None
        seen.add(idx)
        st = P.series_style(idx)
        ax.errorbar(x, y, yerr=e, marker=st["marker"], color=st["color"],
                    markersize=5, linestyle="none", elinewidth=0.9, capsize=2,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.6,
                    label=label, zorder=3)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted shift (no surrogate simulation)")
    ax.set_ylabel("measured shift (direct sampling)")
    ax.set_title("First-order response vs measurement")
    ax.legend(fontsize=7, loc="upper left")

    residual = measured - predicted
    chi = residual / errors
    ax.text(0.97, 0.06, f"rms residual {np.sqrt((chi**2).mean()):.1f}σ",
            transform=ax.transAxes, ha="right", fontsize=7, color=P.TEXT_SECONDARY)
    return fig


def main():
    records = load("exp07_designed_counterexamples/records.json")
    width_records = load("exp06_response_validation/width_records.json")

    fig1 = figure_one(records, width_records)
    paths = P.save_figure(fig1, ROOT / "figures" / "headline_mechanism")
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig1)

    fig2 = figure_two(records)
    paths = P.save_figure(fig2, ROOT / "figures" / "headline_prediction")
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig2)

    # The numbers the report quotes, printed so they can be checked against it.
    print("\n--- numbers for the report ---")
    for level in sorted({r["force_rms_level"] for r in records}):
        group = [r for r in records if r["force_rms_level"] == level]
        f = np.array([r["force_rms_out_of_sample"] for r in group])
        print(f"force RMSE {level:.0e}: actual spread {100 * np.ptp(f) / f.mean():.2f}%")
        for r in sorted(group, key=lambda r: abs(r["target_correlation"])):
            print(f"  {r['name']:8s} rho={r['target_correlation']:+.3f} "
                  f"predicted={r['target_predicted']:+7.3f} "
                  f"measured={r['target_measured']:+7.3f}+/-{r['target_error']:.3f} "
                  f"({abs(r['target_measured']) / r['target_error']:.1f} sigma from zero)")
    chi = np.array([(r["target_measured"] - r["target_predicted"]) / r["target_error"]
                    for r in records])
    print(f"\nprediction vs measurement: rms residual {np.sqrt((chi**2).mean()):.2f} sigma "
          f"over {len(chi)} fields")


if __name__ == "__main__":
    main()
