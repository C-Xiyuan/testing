#!/usr/bin/env python3
"""The regime figure: where force error informs, and where it does not.

exp05 refuted the naive form of prediction P1. Across a zoo spanning three
decades of force RMSE, force error ranks models *well* -- Spearman 0.83. A model
a hundred times worse in force error really is worse, and no theory was needed to
say so.

What it cannot do is choose among models of comparable force error, which is the
only situation a practitioner is ever in. Restricted to the members whose force
RMSE differs by a factor of 2.6, the observable error still varies by 30x and the
rank correlation collapses to 0.34, while the response prediction holds at 0.94.

This figure shows both facts in one frame, because reporting either alone would
be misleading.
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


def main():
    records = json.loads((ROOT / "results/exp05_proxy_correlation/records.json").read_text())
    force = np.array([r["force_rmse"] for r in records])
    raw = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    predicted = np.array([r["predicted_norm"] for r in records])

    # |dA| is a norm, so noise inflates it: E|v_meas|^2 = |v_true|^2 + E|noise|^2.
    # Subtracting in quadrature removes that bias, which matters only at the low
    # end where signal and noise are comparable.
    truth = np.sqrt(np.maximum(raw**2 - noise**2, 0.0))
    band = (force >= BAND[0]) & (force <= BAND[1])

    rho_all = spearmanr(force, truth).statistic
    rho_band = spearmanr(force[band], truth[band]).statistic
    rho_pred_band = spearmanr(predicted[band], truth[band]).statistic

    P.use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    fig.subplots_adjust(wspace=0.34, top=0.82)

    ax1.axvspan(BAND[0], BAND[1], color=P.SERIES[1], alpha=0.10, zorder=0)
    ax1.scatter(force[~band], truth[~band], s=26, color=P.SERIES[0],
                edgecolor=P.SURFACE, linewidth=0.6, zorder=3, label="rest of the zoo")
    ax1.scatter(force[band], truth[band], s=34, color=P.SERIES[1], marker="s",
                edgecolor=P.SURFACE, linewidth=0.6, zorder=4, label="comparable force error")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("force RMSE (eV/Å)")
    ax1.set_ylabel(r"$\|\Delta\langle A\rangle\|$ (pairs)")
    ax1.set_title(f"Across decades: ρ = {rho_all:.2f}", pad=8)
    ax1.legend(fontsize=6.5, loc="upper left")

    order = np.argsort(force[band])
    x = np.arange(band.sum())
    ax2.bar(x, truth[band][order], color=P.SERIES[1], width=0.72, zorder=3)
    ax2.set_xticks(x, [f"{v*1e3:.1f}" for v in force[band][order]],
                   rotation=60, fontsize=5.6)
    ax2.set_xlabel("force RMSE within the band (10⁻³ eV/Å), ascending")
    ax2.set_ylabel(r"$\|\Delta\langle A\rangle\|$ (pairs)")
    ax2.set_title(f"Inside the band: ρ = {rho_band:.2f}", pad=8)
    ax2.text(0.03, 0.95,
             f"force error spans {force[band].max()/force[band].min():.1f}×\n"
             f"physics spans {truth[band].max()/max(truth[band].min(), 1e-9):.0f}×\n"
             f"response prediction: ρ = {rho_pred_band:.2f}",
             transform=ax2.transAxes, fontsize=6.8, va="top", color=P.TEXT_SECONDARY)

    P.add_panel_labels([ax1, ax2], offset=(-0.20, 1.10))
    paths = P.save_figure(fig, ROOT / "figures" / "headline_regimes")
    print("wrote", [str(p.relative_to(ROOT)) for p in paths])
    plt.close(fig)

    print(f"\nzoo: {len(records)} surrogates, force RMSE "
          f"{force.min():.1e} to {force.max():.1e} eV/A")
    print(f"across the whole zoo   rho(force) = {rho_all:+.3f}")
    print(f"within the band        rho(force) = {rho_band:+.3f}   "
          f"({band.sum()} members, force spread {force[band].max()/force[band].min():.1f}x)")
    print(f"within the band        rho(predicted) = {rho_pred_band:+.3f}")
    print(f"within the band        observable error spans "
          f"{truth[band].max()/max(truth[band].min(), 1e-9):.0f}x")


if __name__ == "__main__":
    main()
