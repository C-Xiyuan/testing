#!/usr/bin/env python3
"""What survives when the zoo's own measurement error is put into its intervals?

Review item P0-4, second half. `experiments/exp05_proxy_correlation/run.py`
bootstraps rows of the surrogate table to get an interval on a rank correlation,
which propagates the uncertainty in *which members were drawn* and none of the
uncertainty in *what each member's observable error is*. Every entry in that
column is a Monte Carlo estimate with its own noise, and the analysis then
subtracts that noise in quadrature and floors the result at zero -- an operation
that moves the low end of the ranking around, which is exactly where the
interesting comparisons are.

This script redoes the correlations with both sources of uncertainty. Each
resample draws members with replacement *and* redraws every member's observable
error from its own sampling distribution before the noise subtraction and the
floor, so the reported interval covers the two things that can move a rank: the
composition of the zoo and the precision of each measurement.

Three consequences are worth separating in the output.

*The interval widens.* That is arithmetic and expected.

*The floor bites.* A member whose true error is below its noise level lands at
zero on a large fraction of resamples, and ties at zero are ranked arbitrarily.
The fraction of resamples in which a member floors is reported per member: it
says which points in the published figure are measurements and which are upper
limits drawn as points.

*The ordering of the two regimes may or may not survive.* The claim under
dispute is that force RMSE resolves observable error across decades and not
within a band. If the across-decades correlation stays well away from zero and
the within-band one does not, the two-regime description survives its own
measurement error; if the band interval was only excluding zero because the
measurement error was omitted, it does not.

Nothing here is refitted and no new sampling is done. It is the published
analysis with one term restored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_RESAMPLES = 4000
SEED = 20240517
BAND = (1.5e-3, 6.0e-3)
# The two members that were constructed to sit at the extremes of the band. They
# are the answer, not evidence for it, so the band is reported with and without.
DESIGNED = ("null", "aligned")


def load():
    records = json.loads(
        (ROOT / "results/exp05_proxy_correlation/records.json").read_text())
    return {
        "name": np.array([r["name"] for r in records]),
        "force": np.array([r["force_rmse"] for r in records]),
        "raw": np.array([r["observable_error"] for r in records]),
        "noise": np.array([r["observable_error_noise"] for r in records]),
        "predicted": np.array([r["predicted_norm"] for r in records]),
    }


def truth(raw, noise):
    """Noise-subtracted observable error, floored at zero -- as published."""
    return np.sqrt(np.maximum(raw**2 - noise**2, 0.0))


def resample(data, rng, mask, *, jitter):
    """One bootstrap replicate of both correlations under a member mask.

    ``jitter`` is the standard deviation of the redraw as a multiple of the
    member's noise level. The published observable error is the norm of an
    eight-component difference vector, so the sampling error of that norm is
    between ``noise/sqrt(8)`` (well above the noise floor, where the norm
    behaves like one component) and about ``0.37 * noise`` (at the floor, where
    it is chi-distributed). The default 0.71 is above both, deliberately: an
    over-generous redraw that leaves the conclusion standing is worth more than
    a carefully calibrated one that does.
    """
    idx_pool = np.flatnonzero(mask)
    idx = rng.choice(idx_pool, size=idx_pool.size, replace=True)
    raw = data["raw"][idx]
    noise = data["noise"][idx]
    if jitter:
        raw = np.abs(raw + rng.normal(0.0, jitter * noise))
    t = truth(raw, noise)
    if np.ptp(t) == 0:
        return None
    return (spearmanr(data["force"][idx], t).statistic,
            spearmanr(data["predicted"][idx], t).statistic,
            (t == 0).mean())


def interval(data, mask, *, jitter, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = [resample(data, rng, mask, jitter=jitter) for _ in range(N_RESAMPLES)]
    rows = [r for r in rows if r is not None]
    force = np.array([r[0] for r in rows])
    pred = np.array([r[1] for r in rows])
    floored = np.array([r[2] for r in rows])
    point_truth = truth(data["raw"][mask], data["noise"][mask])
    return {
        "n": int(mask.sum()),
        "rho_force": float(spearmanr(data["force"][mask], point_truth).statistic),
        "rho_force_ci": [float(np.percentile(force, 2.5)),
                         float(np.percentile(force, 97.5))],
        "rho_predicted": float(spearmanr(data["predicted"][mask], point_truth).statistic),
        "rho_predicted_ci": [float(np.percentile(pred, 2.5)),
                             float(np.percentile(pred, 97.5))],
        "mean_fraction_floored": float(floored.mean()),
    }


def main():
    data = load()
    force = data["force"]
    band = (force >= BAND[0]) & (force <= BAND[1])
    designed = np.array([any(d in n for d in DESIGNED) for n in data["name"]])
    subsets = {
        "whole zoo": np.ones(force.size, bool),
        "band": band,
        "band, designed removed": band & ~designed,
    }

    print(f"{N_RESAMPLES} resamples, seed {SEED}\n")
    header = (f"{'subset':<26}{'n':>4}  {'rho(force RMSE)':>26}  "
              f"{'rho(response prediction)':>26}")
    out = {}
    # 0.0 reproduces the published analysis; the rest sweep the redraw scale so
    # the conclusion does not rest on one guess at how noisy each norm is.
    for jitter in (0.0, 0.35, 0.71, 1.41):
        label = ("members only (as published)" if jitter == 0.0
                 else f"members + measurement error, redraw {jitter:.2f} x noise")
        print(f"--- bootstrap over {label} ---")
        print(header)
        out[label] = {}
        for name, mask in subsets.items():
            r = interval(data, mask, jitter=jitter)
            out[label][name] = r
            print(f"{name:<26}{r['n']:>4}  "
                  f"{r['rho_force']:>7.2f} [{r['rho_force_ci'][0]:>5.2f},"
                  f"{r['rho_force_ci'][1]:>5.2f}]  "
                  f"{r['rho_predicted']:>7.2f} [{r['rho_predicted_ci'][0]:>5.2f},"
                  f"{r['rho_predicted_ci'][1]:>5.2f}]")
        print()

    # Which members are measurements and which are upper limits?
    t = truth(data["raw"], data["noise"])
    rng = np.random.default_rng(SEED + 1)
    floor_rate = np.zeros(force.size)
    for _ in range(N_RESAMPLES):
        redrawn = np.abs(data["raw"] + rng.normal(0.0, 0.71 * data["noise"]))
        floor_rate += truth(redrawn, data["noise"]) == 0
    floor_rate /= N_RESAMPLES
    at_risk = np.argsort(-floor_rate)[:6]
    print("members most often floored to zero when their own error is redrawn:")
    print(f"  {'member':<30}{'in band':>9}{'raw':>8}{'noise':>8}"
          f"{'published':>11}{'floors':>9}")
    for i in at_risk:
        print(f"  {data['name'][i]:<30}{'yes' if band[i] else 'no':>9}"
              f"{data['raw'][i]:>8.2f}{data['noise'][i]:>8.2f}"
              f"{t[i]:>11.2f}{floor_rate[i]:>8.0%}")
    out["floor_rate"] = {str(n): float(f) for n, f in zip(data["name"], floor_rate)}

    published = out["members only (as published)"]
    restored = out["members + measurement error, redraw 0.71 x noise"]
    print()
    for name in subsets:
        p, q = published[name], restored[name]
        widened = ((q["rho_force_ci"][1] - q["rho_force_ci"][0])
                   / max(p["rho_force_ci"][1] - p["rho_force_ci"][0], 1e-12))
        print(f"{name:<26} force interval widens {widened:4.2f}x; "
              f"{'still excludes' if q['rho_force_ci'][0] > 0 else 'includes'} zero, "
              f"prediction "
              f"{'still excludes' if q['rho_predicted_ci'][0] > 0 else 'includes'} zero")

    dest = ROOT / "results/exp05_proxy_correlation/uncertainty_propagation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
