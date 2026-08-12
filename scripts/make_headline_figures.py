#!/usr/bin/env python3
"""Build descriptive figures from the legacy exp06/exp07 deposits.

These panels do not establish a universal selector, calibration, or population
law. exp07 contains four constructed fields at each of two force levels, all
sharing reference/direct streams. exp06 is one fixed width sweep whose two
widest Gaussian shells enter the cutoff switch and whose deposited norm error
is not a standard error of the norm. The figures expose these design limits
instead of converting point ranges or marginal errors into headline claims.
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
from matplotlib.lines import Line2D

from atomlab.utils import plotting as P

FIGURE_FORMATS = ("png", "pdf", "svg")  # editable .svg + project .png/.pdf


def configure_figure_style():
    """Use explicit publication-safe fonts and editable vector text."""
    P.use_style(fontsize=9.0)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def figure_one(records, width_records, width_config):
    """Show the fixed designed panel and the switch-limited width sweep."""
    configure_figure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    fig.subplots_adjust(wspace=0.40, top=0.84, bottom=0.18)

    levels = sorted({r["force_rms_level"] for r in records})
    for idx, level in enumerate(levels):
        group = sorted(
            (r for r in records if r["force_rms_level"] == level),
            key=lambda r: r["target_correlation"],
        )
        rho = np.array([r["target_correlation"] for r in group])
        measured = np.array([r["target_measured"] for r in group])
        errors = np.array([r["target_error"] for r in group])
        spread = np.ptp([r["force_rms_out_of_sample"] for r in group]) / level
        style = P.series_style(idx)
        ax1.errorbar(
            rho, measured, yerr=errors, fmt=style["marker"], markersize=5.2,
            color=style["color"], ecolor=style["color"], elinewidth=0.9,
            capsize=2, markeredgecolor=P.SURFACE, markeredgewidth=0.6,
            linestyle="none",
            label=(f"force RMSE {level:.0e} eV/Å; n = {len(group)} fields "
                   f"(spread {100 * spread:.1f}%)"),
        )
        for record, x, y in zip(group, rho, measured):
            if record["name"] == "null" and idx == len(levels) - 1:
                ax1.annotate(
                    record["name"], xy=(x, y), xytext=(0.4, -0.9),
                    textcoords="offset fontsize", fontsize=7,
                    color=P.TEXT_SECONDARY,
                )

    ax1.axhline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
    ax1.axvline(0.0, color=P.TEXT_SECONDARY, linewidth=0.6)
    ax1.set_xlabel(r"reference-sample $\rho(A,\,\delta U)$")
    ax1.set_ylabel("measured shift (pairs)")
    ax1.set_title("Legacy fixed designed fields", pad=8)
    ax1.legend(fontsize=6.7, loc="upper right")
    ax1.text(
        0.03, 0.04,
        "marginal ±1 SE\nshared reference/direct streams\n"
        "descriptive—not a selector",
        transform=ax1.transAxes, fontsize=7.0, color=P.TEXT_SECONDARY,
        va="bottom",
        bbox={
            "facecolor": P.SURFACE, "edgecolor": "none", "alpha": 0.82,
            "pad": 1.0,
        },
    )

    widths = np.array([r["width"] for r in width_records])
    predicted = np.array([abs(r["linear_norm"]) for r in width_records])
    measured = np.array([abs(r["direct_norm"]) for r in width_records])
    force = np.array([r["force_rms"] for r in width_records])

    r0 = float(width_config["width_sweep"]["r0"])
    cutoff = float(width_config["potential"]["cutoff"])
    r_on = 0.85 * cutoff
    untruncated = r0 + 3.0 * widths < r_on
    affected = ~untruncated
    switch_width = (r_on - r0) / 3.0
    bridge = np.r_[np.flatnonzero(untruncated)[-1], np.flatnonzero(affected)]

    # Point estimates only. ``direct_norm_error`` is the norm of componentwise
    # SEs, not the SE of this norm, and prediction UQ was not deposited.
    # Rendering either as a confidence interval would invent inferential UQ.
    ax2.plot(
        widths[untruncated], measured[untruncated], marker=P.MARKERS[0],
        color=P.SERIES[0], linestyle="-", label="direct point estimate",
    )
    ax2.plot(
        widths[untruncated], predicted[untruncated], marker=P.MARKERS[1],
        color=P.SERIES[1], linestyle="--", label="first-order point estimate",
    )
    ax2.plot(
        widths[bridge], measured[bridge], marker=P.MARKERS[0],
        markerfacecolor=P.SURFACE, markeredgecolor=P.SERIES[0],
        color=P.SERIES[0], linestyle=":",
    )
    ax2.plot(
        widths[bridge], predicted[bridge], marker=P.MARKERS[1],
        markerfacecolor=P.SURFACE, markeredgecolor=P.SERIES[1],
        color=P.SERIES[1], linestyle=":",
    )
    ax2.axvspan(
        switch_width, widths.max() * 1.08, color=P.NEUTRAL, alpha=0.12,
        label="3σ tail enters cutoff switch",
    )
    ax2.axvline(
        switch_width, color=P.TEXT_SECONDARY, linestyle=":", linewidth=0.8
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("width of the error field (Å)")
    ax2.set_ylabel(r"$\|\Delta\langle A\rangle\|$ (pairs)")
    ax2.set_title("Legacy fixed-force width sweep (n = 6)", pad=8)
    ax2.set_ylim(1.75, 12.0)
    ax2.legend(fontsize=6.7, loc="lower right")

    direct_slope = float(
        np.polyfit(np.log(widths[untruncated]), np.log(measured[untruncated]), 1)[0]
    )
    predicted_slope = float(
        np.polyfit(np.log(widths[untruncated]), np.log(predicted[untruncated]), 1)[0]
    )
    ax2.text(
        0.03, 0.97,
        f"untruncated n = {untruncated.sum()}\n"
        f"slopes {direct_slope:.3f} / {predicted_slope:.3f}",
        transform=ax2.transAxes, fontsize=6.5, color=P.TEXT_SECONDARY,
        va="top",
    )
    ax2.text(
        0.97, 0.97, "descriptive only\nno joint norm UQ",
        transform=ax2.transAxes, fontsize=6.5, color=P.TEXT_SECONDARY,
        ha="right", va="top",
        bbox={"facecolor": P.SURFACE, "edgecolor": "none", "alpha": 0.88, "pad": 0.7},
    )
    ax2.text(
        0.97, 0.42,
        f"w ≥ {widths[affected].min():.2f} Å:\nswitch-affected",
        transform=ax2.transAxes, fontsize=6.8, color=P.TEXT_SECONDARY,
        ha="right", va="top",
    )

    fig.text(
        0.5, 0.035,
        f"Legacy deposits; force matching used one shared reference "
        f"(range {100 * np.ptp(force) / force.mean():.2f}%). "
        "Neither panel supplies confirmatory replication.",
        ha="center", fontsize=7.5, color=P.TEXT_SECONDARY,
    )
    P.add_panel_labels([ax1, ax2], offset=(-0.18, 1.08))
    return fig


def figure_two(records):
    """Show marginal x/y errors without claiming independent calibration."""
    configure_figure_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))

    predicted = np.array([r["target_predicted"] for r in records])
    predicted_errors = np.array([r["target_predicted_error"] for r in records])
    measured = np.array([r["target_measured"] for r in records])
    errors = np.array([r["target_error"] for r in records])
    names = [r["name"] for r in records]
    levels = np.array([r["force_rms_level"] for r in records])

    lim = [
        min((predicted - predicted_errors).min(), (measured - errors).min()) - 1.0,
        max((predicted + predicted_errors).max(), (measured + errors).max()) + 1.0,
    ]
    ax.plot(
        lim, lim, color=P.TEXT_SECONDARY, linestyle="--", linewidth=0.9,
        zorder=1,
    )

    style = {"null": 0, "aligned": 1, "random0": 2, "random1": 3}
    low_level = float(levels.min())
    for name, level, x, xerr, y, yerr in zip(
        names, levels, predicted, predicted_errors, measured, errors
    ):
        idx = style[name]
        point_style = P.series_style(idx)
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, marker=point_style["marker"],
            color=point_style["color"], markersize=5, linestyle="none",
            elinewidth=0.9, capsize=2,
            markerfacecolor=(P.SURFACE if level == low_level
                             else point_style["color"]),
            markeredgecolor=point_style["color"], markeredgewidth=0.9,
            zorder=3,
        )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("first-order point prediction (pairs)")
    ax.set_ylabel("direct-sampling point estimate (pairs)")
    ax.set_title(f"Legacy shared-stream tracking panel (n = {len(records)} fields)")

    field_handles = [
        Line2D(
            [], [], color=P.series_color(idx), marker=P.MARKERS[idx],
            linestyle="none", label=name.replace("random", "random "),
        )
        for name, idx in style.items()
    ]
    field_legend = ax.legend(
        handles=field_handles, fontsize=6.8, loc="upper left",
        title="field", title_fontsize=7.0,
    )
    ax.add_artist(field_legend)
    level_handles = [
        Line2D(
            [], [], color=P.TEXT_SECONDARY, marker="o",
            markerfacecolor=P.SURFACE, linestyle="none",
            label=f"{levels.min():.0e} eV/Å (open)",
        ),
        Line2D(
            [], [], color=P.TEXT_SECONDARY, marker="o",
            markerfacecolor=P.TEXT_SECONDARY, linestyle="none",
            label=f"{levels.max():.0e} eV/Å (filled)",
        ),
    ]
    ax.legend(
        handles=level_handles, fontsize=6.7, loc="lower right",
        title="force RMSE", title_fontsize=7.0,
    )
    ax.text(
        0.97, 0.54,
        "marginal ±1 SE on x and y\n"
        "shared-offset/covariance UQ absent\n"
        "not an independent calibration test",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
        color=P.TEXT_SECONDARY,
        bbox={
            "facecolor": P.SURFACE, "edgecolor": "none", "alpha": 0.84,
            "pad": 1.5,
        },
    )
    return fig


def figure_response_validation(amplitude_records, width_records, width_config):
    """Replace the legacy Fig. 6 rendering without inventing unavailable UQ.

    The amplitude deposit contains marginal errors for direct and first-order
    values but no reweighting interval. The width deposit contains
    ``direct_norm_error = ||per-bin SE||``, which is not the SE of the plotted
    norm, and no prediction-norm interval. Panel (b) therefore plots point
    estimates only and separates the four untruncated widths from the two whose
    Gaussian tails enter the cutoff switch.
    """
    configure_figure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.8))
    fig.subplots_adjust(wspace=0.38, top=0.84, bottom=0.18)

    amplitude = np.array([r["amplitude"] for r in amplitude_records])
    direct = np.array([r["direct"] for r in amplitude_records])
    direct_error = np.array([r["direct_error"] for r in amplitude_records])
    linear = np.array([r["linear"] for r in amplitude_records])
    linear_error = np.array([r["linear_error"] for r in amplitude_records])
    reweighted = np.array([r["reweighted"] for r in amplitude_records])
    ess = np.array([r["reweighted_ess_fraction"] for r in amplitude_records])
    collapsed = ess < 0.05

    ax1.errorbar(
        amplitude, direct, yerr=direct_error, marker=P.MARKERS[0],
        color=P.SERIES[0], linestyle="-", elinewidth=0.9, capsize=2,
        markeredgecolor=P.SURFACE, markeredgewidth=0.6,
        label="direct ± marginal SE",
    )
    ax1.errorbar(
        amplitude, linear, yerr=linear_error, marker=P.MARKERS[1],
        color=P.SERIES[1], linestyle="--", elinewidth=0.9, capsize=2,
        markeredgecolor=P.SURFACE, markeredgewidth=0.6,
        label="first order ± marginal SE",
    )
    ax1.plot(
        amplitude[~collapsed], reweighted[~collapsed], marker=P.MARKERS[2],
        color=P.SERIES[2], linestyle="-", label="reweighted point estimate",
    )
    ax1.plot(
        amplitude[np.r_[np.flatnonzero(~collapsed)[-1], np.flatnonzero(collapsed)]],
        reweighted[np.r_[np.flatnonzero(~collapsed)[-1], np.flatnonzero(collapsed)]],
        marker=P.MARKERS[2], markerfacecolor=P.SURFACE,
        markeredgecolor=P.SERIES[2], color=P.SERIES[2], linestyle=":",
    )
    ax1.axvspan(
        amplitude[collapsed].min(), amplitude.max(), color=P.NEUTRAL,
        alpha=0.10, label="reweighting ESS < 0.05",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("perturbation amplitude (eV)")
    ax1.set_ylabel("shift in target-bin pair count")
    ax1.set_title("Legacy amplitude sweep (n = 7)")
    ax1.legend(fontsize=6.6, loc="lower left")

    widths = np.array([r["width"] for r in width_records])
    width_direct = np.array([abs(r["direct_norm"]) for r in width_records])
    width_linear = np.array([abs(r["linear_norm"]) for r in width_records])
    r0 = float(width_config["width_sweep"]["r0"])
    cutoff = float(width_config["potential"]["cutoff"])
    r_on = 0.85 * cutoff
    untruncated = r0 + 3.0 * widths < r_on
    affected = ~untruncated
    switch_width = (r_on - r0) / 3.0
    bridge = np.r_[np.flatnonzero(untruncated)[-1], np.flatnonzero(affected)]

    ax2.plot(
        widths[untruncated], width_direct[untruncated], marker=P.MARKERS[0],
        color=P.SERIES[0], linestyle="-", label="direct point estimate",
    )
    ax2.plot(
        widths[untruncated], width_linear[untruncated], marker=P.MARKERS[1],
        color=P.SERIES[1], linestyle="--", label="first-order point estimate",
    )
    ax2.plot(
        widths[bridge], width_direct[bridge], marker=P.MARKERS[0],
        markerfacecolor=P.SURFACE, markeredgecolor=P.SERIES[0],
        color=P.SERIES[0], linestyle=":",
    )
    ax2.plot(
        widths[bridge], width_linear[bridge], marker=P.MARKERS[1],
        markerfacecolor=P.SURFACE, markeredgecolor=P.SERIES[1],
        color=P.SERIES[1], linestyle=":",
    )
    ax2.axvspan(
        switch_width, widths.max() * 1.08, color=P.NEUTRAL, alpha=0.12,
        label="3σ tail enters cutoff switch",
    )
    ax2.axvline(
        switch_width, color=P.TEXT_SECONDARY, linestyle=":", linewidth=0.8
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("width of the error field (Å)")
    ax2.set_ylabel(r"$\|\Delta\langle A\rangle\|$ (pairs)")
    ax2.set_title("Legacy fixed-force width sweep (n = 6)")
    ax2.set_ylim(1.75, 11.35)
    ax2.legend(fontsize=6.6, loc="lower right")
    direct_slope = float(
        np.polyfit(
            np.log(widths[untruncated]), np.log(width_direct[untruncated]), 1
        )[0]
    )
    linear_slope = float(
        np.polyfit(
            np.log(widths[untruncated]), np.log(width_linear[untruncated]), 1
        )[0]
    )
    ax2.text(
        0.03, 0.97,
        f"untruncated n = {untruncated.sum()}\n"
        f"slopes {direct_slope:.3f} / {linear_slope:.3f}",
        transform=ax2.transAxes, fontsize=6.5, color=P.TEXT_SECONDARY,
        va="top",
    )
    ax2.text(
        0.97, 0.97, "descriptive only\nno joint norm UQ",
        transform=ax2.transAxes, fontsize=6.5, color=P.TEXT_SECONDARY,
        ha="right", va="top",
        bbox={"facecolor": P.SURFACE, "edgecolor": "none", "alpha": 0.88, "pad": 0.7},
    )

    fig.text(
        0.5, 0.035,
        "Legacy single-reference deposits; reweighting has no deposited interval; "
        "no six-point width fit, norm-SE error bars or confirmatory interpretation.",
        ha="center", fontsize=7.5, color=P.TEXT_SECONDARY,
    )
    P.add_panel_labels([ax1, ax2], offset=(-0.18, 1.08))
    return fig


def main():
    records = load("exp07_designed_counterexamples/records.json")
    amplitude_records = load("exp06_response_validation/amplitude_records.json")
    width_records = load("exp06_response_validation/width_records.json")
    width_manifest = load("exp06_response_validation/manifest.json")

    fig1 = figure_one(records, width_records, width_manifest["config"])
    paths = P.save_figure(
        fig1, ROOT / "figures" / "headline_mechanism",
        formats=FIGURE_FORMATS,
    )
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig1)

    fig2 = figure_two(records)
    paths = P.save_figure(
        fig2, ROOT / "figures" / "headline_prediction",
        formats=FIGURE_FORMATS,
    )
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig2)

    fig6 = figure_response_validation(
        amplitude_records, width_records, width_manifest["config"]
    )
    paths = P.save_figure(
        fig6, ROOT / "figures" / "exp06_response_validation_response_validation",
        formats=FIGURE_FORMATS,
    )
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig6)

    print("\n--- deposited point estimates and marginal errors ---")
    for level in sorted({r["force_rms_level"] for r in records}):
        group = [r for r in records if r["force_rms_level"] == level]
        force_values = np.array([r["force_rms_out_of_sample"] for r in group])
        print(
            f"force RMSE {level:.0e}: actual spread "
            f"{100 * np.ptp(force_values) / force_values.mean():.2f}%"
        )
        for record in sorted(group, key=lambda r: abs(r["target_correlation"])):
            print(
                f"  {record['name']:8s} rho={record['target_correlation']:+.3f} "
                f"predicted={record['target_predicted']:+7.3f}"
                f"+/-{record['target_predicted_error']:.3f} "
                f"measured={record['target_measured']:+7.3f}"
                f"+/-{record['target_error']:.3f}"
            )
    print(
        "\nNo aggregate residual-in-sigma statistic is reported: prediction and "
        "measurement uncertainties are marginal, and rows share "
        "reference/direct streams."
    )


if __name__ == "__main__":
    main()
