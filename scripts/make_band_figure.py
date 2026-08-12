#!/usr/bin/env python3
"""Descriptive regime plot for the legacy, fixed set of designed fields.

The 34 rows are a clustered, deliberately constructed panel that shares a
reference trajectory and random streams. They are not independent draws from a
population of fitted models. The highlighted force-error window was chosen
after inspecting the same rows. Consequently, its row-bootstrap intervals are
sensitivity summaries for this finite panel, never population coverage or
evidence for a general model-selection rule.

The plot uses the deposited *raw* observable-error norm and its Monte Carlo
noise scale. It does not headline the floor-sensitive 30-fold ratio produced by
subtracting noise in quadrature. Rows whose noise-subtracted signal is no
larger than the noise scale are flagged rather than drawn as ordinary resolved
measurements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atomlab.utils import plotting as P

BAND = (1.5e-3, 6.0e-3)
N_BOOTSTRAP = 4000
SEED = 20240517
FIGURE_FORMATS = ("png", "pdf", "svg")  # editable .svg + project .png/.pdf


def row_bootstrap_spearman(x, y, *, seed):
    """Return rho and a descriptive row-resampling sensitivity interval.

    The historical bootstrap ignores field-family clustering and shared
    trajectories. Its interval is therefore not a confidence interval for a
    population of fitted models; the plot labels that limitation explicitly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, x.size, size=x.size)
        if np.unique(x[idx]).size < 3 or np.unique(y[idx]).size < 3:
            continue
        draws.append(float(spearmanr(x[idx], y[idx]).statistic))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(spearmanr(x, y).statistic), float(lo), float(hi)


def short_name(name):
    """Compact but unique label for the 15-member post-hoc window."""
    if name.startswith("shell_"):
        bits = name.split("_")
        return f"shell {bits[1][1:]}/{bits[2][1:]}"
    if name.startswith("random_"):
        return name.split("_f", 1)[0].replace("random_", "random ")
    if name.startswith("null_"):
        return "designed null"
    if name.startswith("aligned_"):
        return "designed aligned"
    return name


def main():
    records = json.loads(
        (ROOT / "results/exp05_proxy_correlation/records.json").read_text()
    )
    names = np.array([r["name"] for r in records])
    force = np.array([r["force_rmse"] for r in records])
    raw = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    predicted = np.array([r["predicted_norm"] for r in records])

    # Quadrature subtraction is used only to flag unresolved rows. Plotting its
    # floored values is what created the withdrawn floor-sensitive 30x ratio.
    signal = np.sqrt(np.maximum(raw**2 - noise**2, 0.0))
    noise_dominated = signal <= noise
    band = (force >= BAND[0]) & (force <= BAND[1])

    rho_all = row_bootstrap_spearman(force, raw, seed=SEED)
    rho_band = row_bootstrap_spearman(force[band], raw[band], seed=SEED + 1)
    rho_pred_band = row_bootstrap_spearman(
        predicted[band], raw[band], seed=SEED + 2
    )

    P.use_style(fontsize=9.0)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.7))
    fig.subplots_adjust(wspace=0.58, top=0.86, bottom=0.16)

    ax1.axvspan(BAND[0], BAND[1], color=P.SERIES[1], alpha=0.10, zorder=0)
    for mask, color, marker, label, zorder in (
        (~band & ~noise_dominated, P.SERIES[0], "o", "outside post-hoc window", 3),
        (band & ~noise_dominated, P.SERIES[1], "s", "inside post-hoc window", 4),
    ):
        ax1.errorbar(
            force[mask], raw[mask], yerr=noise[mask], fmt=marker,
            markersize=5.0, color=color, ecolor=P.GRID, elinewidth=0.8,
            capsize=1.8, markeredgecolor=P.SURFACE, markeredgewidth=0.6,
            linestyle="none", zorder=zorder, label=label,
        )
    ax1.errorbar(
        force[noise_dominated], raw[noise_dominated], yerr=noise[noise_dominated],
        fmt="X", markersize=6.0, color=P.TEXT_SECONDARY,
        ecolor=P.TEXT_SECONDARY, elinewidth=0.8, capsize=1.8,
        markerfacecolor=P.SURFACE, markeredgewidth=0.9, linestyle="none",
        zorder=5, label="noise-dominated (flagged)",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("force RMSE (eV/Å)")
    ax1.set_ylabel(r"raw $\|\Delta\langle A\rangle\|$ (pairs)")
    ax1.set_title(
        f"Fixed designed panel: ρ = {rho_all[0]:.2f} "
        f"[{rho_all[1]:.2f}, {rho_all[2]:.2f}], n = {len(records)}",
        pad=8,
    )
    ax1.legend(fontsize=7.2, loc="upper left")
    ax1.text(
        0.97, 0.03,
        "whiskers: deposited MC noise scale\n"
        f"{noise_dominated.sum()} of {len(records)} rows are noise-dominated",
        transform=ax1.transAxes, fontsize=7.2, va="bottom", ha="right",
        color=P.TEXT_SECONDARY,
        bbox={
            "facecolor": P.SURFACE, "edgecolor": "none", "alpha": 0.82,
            "pad": 1.0,
        },
    )

    order = np.argsort(force[band])
    band_force = force[band][order]
    band_raw = raw[band][order]
    band_noise = noise[band][order]
    band_noise_dominated = noise_dominated[band][order]
    band_names = names[band][order]
    y = np.arange(band.sum())
    ax2.errorbar(
        band_raw[~band_noise_dominated], y[~band_noise_dominated],
        xerr=band_noise[~band_noise_dominated], fmt="s", markersize=5.2,
        color=P.SERIES[1], ecolor=P.GRID, elinewidth=0.9, capsize=1.8,
        markeredgecolor=P.SURFACE, markeredgewidth=0.6, linestyle="none",
        zorder=3,
    )
    ax2.errorbar(
        band_raw[band_noise_dominated], y[band_noise_dominated],
        xerr=band_noise[band_noise_dominated], fmt="X", markersize=6.0,
        color=P.TEXT_SECONDARY, ecolor=P.TEXT_SECONDARY, elinewidth=0.9,
        capsize=1.8, markerfacecolor=P.SURFACE, markeredgewidth=0.9,
        linestyle="none", zorder=4,
    )
    labels = [
        f"{short_name(name)}  ·  {value*1e3:.2f}"
        for name, value in zip(band_names, band_force)
    ]
    ax2.set_yticks(y, labels, fontsize=7.1)
    ax2.invert_yaxis()
    # Reserve two label-heights above the first row for the descriptive
    # sensitivity summary instead of covering the observations.
    ax2.set_ylim(band.sum() - 0.5, -2.6)
    ax2.set_xlabel(r"raw $\|\Delta\langle A\rangle\|$ (pairs)")
    ax2.set_ylabel(r"field · force RMSE ($10^{-3}$ eV/Å)")
    ax2.set_title(
        f"Post-hoc window: ρ = {rho_band[0]:.2f} "
        f"[{rho_band[1]:.2f}, {rho_band[2]:.2f}], n = {band.sum()}",
        pad=8,
    )
    band_names_text = band_names.astype(str)
    not_designed = ~(
        np.char.startswith(band_names_text, "null_")
        | np.char.startswith(band_names_text, "aligned_")
    )
    raw_spread = float(band_raw.max() / band_raw.min())
    raw_spread_without_designed = float(
        band_raw[not_designed].max() / band_raw[not_designed].min()
    )
    ax2.text(
        0.97, 0.98,
        f"response ρ = {rho_pred_band[0]:.2f} "
        f"[{rho_pred_band[1]:.2f}, {rho_pred_band[2]:.2f}]\n"
        f"raw range: {raw_spread:.1f}×; {raw_spread_without_designed:.1f}× "
        "without designed extremes\n"
        "95% intervals: row-resampling sensitivity only",
        transform=ax2.transAxes, fontsize=7.0, va="top", ha="right",
        color=P.TEXT_SECONDARY,
    )

    fig.text(
        0.5, 0.025,
        "Legacy fixed, clustered zoo with shared trajectories; the highlighted "
        "window was analyst-chosen after inspection. No population or "
        "model-selection inference.",
        ha="center", fontsize=7.5, color=P.TEXT_SECONDARY,
    )
    P.add_panel_labels([ax1, ax2], offset=(-0.18, 1.08))
    paths = P.save_figure(
        fig, ROOT / "figures" / "headline_regimes", formats=FIGURE_FORMATS
    )
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig)

    print(f"\nfixed designed panel: {len(records)} fields, force RMSE "
          f"{force.min():.1e} to {force.max():.1e} eV/A")
    print(f"across the whole panel rho(force) = {rho_all[0]:+.3f} "
          f"[{rho_all[1]:+.3f}, {rho_all[2]:+.3f}]")
    print(f"within the post-hoc band rho(force) = {rho_band[0]:+.3f} "
          f"[{rho_band[1]:+.3f}, {rho_band[2]:+.3f}] "
          f"({band.sum()} members, force spread "
          f"{force[band].max()/force[band].min():.1f}x)")
    print(f"within the post-hoc band rho(predicted) = {rho_pred_band[0]:+.3f} "
          f"[{rho_pred_band[1]:+.3f}, {rho_pred_band[2]:+.3f}]")
    print(f"within the post-hoc band raw observable norm spans {raw_spread:.1f}x "
          f"({raw_spread_without_designed:.1f}x without designed extremes)")
    print(f"noise-dominated rows flagged: {noise_dominated.sum()} / {len(records)}")
    print("intervals are descriptive row-resampling sensitivity summaries; "
          "family clustering and shared trajectories are not propagated")


if __name__ == "__main__":
    main()
