#!/usr/bin/env python3
"""Recompute the exp05 rank correlations with bootstrap intervals, on the
noise-subtracted observable-error series that the regime figure uses.

`results/exp05_proxy_correlation/summary.json` reports correlations computed on
the raw observable-error norm and contains no statistics restricted to the
analyst-chosen band; `scripts/make_band_figure.py` prints the band rho as a bare
point estimate with no interval. This script closes both gaps.

Convention. |Delta<A>| is a norm, so sampling noise inflates it:
E|v_meas|^2 = |v_true|^2 + E|noise|^2. All correlations here are computed on
truth = sqrt(max(|v_meas|^2 - E|noise|^2, 0)). The band is the analyst-chosen,
post hoc window 1.5e-3 <= force RMSE <= 6.0e-3 eV/A hard-coded in
scripts/make_band_figure.py.

Intervals are 95% percentile bootstrap over zoo members (resampling members,
not frames), 2000 resamples, numpy default_rng seed 20240517.

Usage:  python3 scripts/compute_band_correlations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BAND = (1.5e-3, 6.0e-3)
N_BOOT = 2000
SEED = 20240517


def rho_ci(x, y, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    point = spearmanr(x, y).statistic
    n = len(x)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        draws.append(spearmanr(x[idx], y[idx]).statistic)
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, n


def main():
    records = json.loads(
        (ROOT / "results/exp05_proxy_correlation/records.json").read_text()
    )
    force = np.array([r["force_rmse"] for r in records])
    raw = np.array([r["observable_error"] for r in records])
    noise = np.array([r["observable_error_noise"] for r in records])
    truth = np.sqrt(np.maximum(raw**2 - noise**2, 0.0))

    proxies = {
        "force_rmse": force,
        "force_mae": np.array([r["force_mae"] for r in records]),
        "energy_std_per_atom": np.array([r["energy_std_per_atom"] for r in records]),
        "smoothness_ratio": np.array([r["smoothness_ratio"] for r in records]),
        "predicted_norm": np.array([r["predicted_norm"] for r in records]),
    }

    band = (force >= BAND[0]) & (force <= BAND[1])
    rng = np.random.default_rng(SEED)

    out = {
        "convention": "noise-subtracted |Delta<A>|",
        "band_window_eV_per_A": list(BAND),
        "band_is_post_hoc": True,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "zoo": {},
        "band": {},
        "spreads": {
            "force_rmse_spread_zoo": float(force.max() / force.min()),
            "force_rmse_spread_band": float(force[band].max() / force[band].min()),
            "observable_error_spread_band": float(
                truth[band].max() / truth[band].min()
            ),
            "force_rmse_min_zoo": float(force.min()),
            "force_rmse_max_zoo": float(force.max()),
            "n_band": int(band.sum()),
        },
    }

    for name, v in proxies.items():
        p, lo, hi, n = rho_ci(v, truth, rng)
        out["zoo"][name] = {"rho": p, "ci_low": lo, "ci_high": hi, "n": n}
        p, lo, hi, n = rho_ci(v[band], truth[band], rng)
        out["band"][name] = {"rho": p, "ci_low": lo, "ci_high": hi, "n": n}

    print(json.dumps(out, indent=1))
    (ROOT / "results/exp05_proxy_correlation/band_correlations.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
