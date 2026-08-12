#!/usr/bin/env python3
"""Figures for the three experiments written in response to the external review.

Each panel answers one review item and is drawn so that the answer is legible
whichever way it came out -- no panel is arranged to look like a pass.

(a) exp09. The residual table decomposed into a reference-chain effect and a
    field effect, each plotted against the spread its own quoted error predicts.
    A bar at its predicted height is sampling noise; a bar well above it is
    structure the error bars do not account for.
(b) exp09 again. Residual rms with and without the second-order correction. The
    manuscript treated the second-order term as a warning light; if the
    corrected bar is taller, the correction is making things worse.
(c) exp10. The relaxation ladder: the measured shift from chains started in the
    reference ensemble and from chains started in the surrogate ensemble, as a
    function of how much of each chain is discarded. The two must meet. Where
    they have not, the gap is the size of the artefact.
(d) exp10. Every estimator of the same shift, with its interval, against the
    pre-registered equivalence bound.
(e) exp11. The per-cluster aligned-minus-null contrast, with the mean and its
    interval, against the single-cluster value exp07 reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomlab.utils.plotting import add_panel_labels, save_figure, series_color, use_style

RESULTS = ROOT / "results"


def load(name, stem="summary"):
    """Load a result, refusing anything produced under ``--quick``.

    A smoke run writes to the same directory as a production one, so without
    this check a figure drawn from 5%-size chains is indistinguishable from the
    real thing. The manifest records ``quick_mode``; a figure has no business
    reading a result whose manifest says it is not a result.
    """
    directory = RESULTS / name
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("quick_mode"):
        print(f"  skipping {name}: manifest says quick_mode, not a production run")
        return None
    path = directory / f"{stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def panel_components(ax, s):
    """(a) Observed vs predicted spread, per variance component."""
    d = s["first_order"]
    labels = ["reference chain\n(row)", "field\n(column)"]
    observed = [d["row_spread_observed"], d["col_spread_observed"]]
    predicted = [d["row_spread_predicted"], d["col_spread_predicted"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, observed, 0.34, label="observed", color=series_color(0))
    ax.bar(x + 0.18, predicted, 0.34, label="predicted by the quoted error",
           color=series_color(1), alpha=0.75)
    for i, (o, p) in enumerate(zip(observed, predicted)):
        ax.text(i, max(o, p) * 1.04, f"{o / p:.1f}x", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"spread of the effect ($\sigma$)")
    ax.set_title("(a) where the residual structure lives", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)


def panel_second_order(ax, s):
    """(b) Does adding the second-order term help?

    Each metric is drawn next to the value it should take if the prediction were
    unbiased and the error bars right, because "1" is the target for the rms and
    for nothing else on this axis -- a single reference line across all three
    would be wrong for two of them.
    """
    keys = ["first_order", "second_order_corrected"]
    names = ["linear\nprediction", "+ second-order\nterm"]
    metrics = [("grand_mean", "systematic\noffset", lambda d: 0.0),
               ("residual_rms", "residual\nrms", lambda d: 1.0),
               ("col_spread_observed", "field-effect\nspread",
                lambda d: d["col_spread_predicted"])]
    x = np.arange(len(metrics))
    for j, key in enumerate(keys):
        vals = [s[key][m] for m, _, _ in metrics]
        ax.bar(x + (j - 0.5) * 0.34, vals, 0.32, label=names[j],
               color=series_color(j), alpha=0.9)
    for i, (_, _, target) in enumerate(metrics):
        t = target(s["first_order"])
        ax.plot([i - 0.52, i + 0.52], [t, t], color="0.35", lw=1.1, ls=":",
                zorder=5)
    ax.plot([], [], color="0.35", lw=1.1, ls=":", label="value if unbiased")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _ in metrics], fontsize=7)
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("(b) the second-order correction", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)


def panel_relaxation(ax, relaxation, bound):
    """(c) The two initialisations, converging or not."""
    fracs = [r["discard_fraction"] for r in relaxation if "arm_gap" in r]
    for j, (arm, label) in enumerate((("from_reference", "started in the reference ensemble"),
                                      ("from_surrogate", "started in the surrogate ensemble"))):
        y = np.array([r[arm][0] for r in relaxation if "arm_gap" in r])
        e = np.array([r[f"{arm}_sem"][0] for r in relaxation if "arm_gap" in r])
        ax.errorbar(np.array(fracs) * 100, y, yerr=e, marker="o", ms=4, lw=1.2,
                    color=series_color(j), label=label, capsize=2)
    ax.set_xlabel("fraction of each chain discarded (%)")
    ax.set_ylabel("measured shift (pairs)")
    ax.set_title("(c) has the surrogate chain relaxed?", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)


def panel_estimators(ax, s):
    """(d) Every estimator of one number, against the bound."""
    e = s["estimators"]
    rows = [("linear response", e["linear"][0], e["linear_between_chain_sem"][0]),
            ("forward FEP", e["forward_fep"][0], None),
            ("reverse FEP", e["reverse_fep"][0], None),
            ("MBAR", e["mbar"][0], e["mbar_error"][0]),
            ("direct, HMC", e["direct_hmc"][0], e["direct_hmc_error"][0]),
            ("direct, Metropolis", e["direct_metropolis"][0],
             e["direct_metropolis_error"][0])]
    y = np.arange(len(rows))[::-1]
    centre = e["mbar"][0]
    ax.axvspan(centre - s["equivalence_bound_pairs"], centre + s["equivalence_bound_pairs"],
               color=series_color(1), alpha=0.12,
               label=f"$\\pm${s['equivalence_bound_pairs']} pair bound")
    for yy, (label, value, err) in zip(y, rows):
        ax.errorbar(value, yy, xerr=(1.96 * err if err else None), marker="o", ms=4,
                    color=series_color(0), capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel(f"shift at {s['bin_radii'][0]:.2f} " + r"$\mathrm{\AA}$ (pairs)")
    ax.set_title("(d) one number, six estimators", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def panel_clusters(ax, s):
    """(e) The counterexample, once per construction cluster."""
    c = np.array(s["contrasts"])
    x = np.arange(len(c))
    ax.bar(x, c, 0.6, color=series_color(0), label="per cluster")
    ax.axhline(s["contrast_mean"], color=series_color(1), lw=1.4,
               label=f"mean {s['contrast_mean']:.2f}")
    ax.axhspan(s["ci95"][0], s["ci95"][1], color=series_color(1), alpha=0.15,
               label="95% CI")
    ax.axhline(s["exp07_single_cluster_contrast"], color="0.35", ls="--", lw=1.0,
               label=f"exp07, one cluster ({s['exp07_single_cluster_contrast']:.1f})")
    ax.axhline(s["minimum_effect_pairs"], color="0.6", ls=":", lw=0.9,
               label=f"minimum effect ({s['minimum_effect_pairs']:.1f})")
    ax.set_xticks(x)
    ax.set_xlabel("construction cluster")
    ax.set_ylabel("aligned $-$ null (pairs)")
    ax.set_title("(e) does the counterexample replicate?", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False)


def main():
    use_style()
    nine = load("exp09_calibration_replication")
    ten = load("exp10_endtoend_consistency")
    ten_relax = load("exp10_endtoend_consistency", "relaxation")
    eleven = load("exp11_counterexample_replication")

    panels = []
    if nine:
        panels += [("components", lambda ax: panel_components(ax, nine)),
                   ("second", lambda ax: panel_second_order(ax, nine))]
    if ten and ten_relax:
        panels += [("relax", lambda ax: panel_relaxation(
            ax, ten_relax, ten["equivalence_bound_pairs"])),
            ("estimators", lambda ax: panel_estimators(ax, ten))]
    if eleven:
        panels += [("clusters", lambda ax: panel_clusters(ax, eleven))]
    if not panels:
        print("no results yet; nothing to draw")
        return

    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, draw) in zip(axes, panels):
        draw(ax)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    written = save_figure(fig, ROOT / "figures" / "review_response")
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"panels drawn: {[name for name, _ in panels]}")


if __name__ == "__main__":
    main()
