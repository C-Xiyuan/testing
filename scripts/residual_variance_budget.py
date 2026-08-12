#!/usr/bin/env python3
"""Where does the residual variance in exp09 actually come from?

exp09 decomposes its residual table into a reference-chain effect (constant down
a row), a field effect (constant down a column) and an interaction. The first
two have obvious candidate causes with known scales. The interaction does not,
and it is the interesting one: it is the part of the disagreement that varies
with *both* the reference chain and the field, which neither the reference
error nor the direct-chain error can produce.

There is one term with exactly that footprint, and the manuscript left it out of
the denominator. The prediction is computed from reference chain i for field j,
so its Monte Carlo error is a per-cell quantity. exp09's residuals are
normalised by sqrt(direct_sem^2 + reference_error^2), which is the error on the
*measurement* alone. If the omitted prediction error accounts for the
interaction, the fix is arithmetic -- put it in the denominator -- and the
review's objection that prediction uncertainty was never propagated is both
correct and sufficient. If the interaction is larger than the prediction error
can explain, something else is going on and the estimator is not merely
mis-normalised.

This script decides between those by comparing the observed interaction against
the prediction error the records already carry, and re-runs the whole
decomposition with the prediction error included so the two normalisations can
be read side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def table(records, key, fields, chains):
    lookup = {(r["reference_chain"], r["field"]): r[key] for r in records}
    return np.array([[lookup[(c, f)] for f in fields] for c in chains])


def decompose(r):
    grand = r.mean()
    row = r.mean(axis=1) - grand
    col = r.mean(axis=0) - grand
    return grand, row, col, r - grand - row[:, None] - col[None, :]


def main(path=None):
    directory = Path(path) if path else ROOT / "results/exp09_calibration_replication"
    records = json.loads((directory / "records.json").read_text())
    fields = sorted({r["field"] for r in records})
    chains = sorted({r["reference_chain"] for r in records})
    print(f"{len(chains)} reference chains x {len(fields)} fields "
          f"= {len(records)} cells\n")

    measured = table(records, "measured", fields, chains)
    predicted = table(records, "predicted", fields, chains)
    meas_err = table(records, "measurement_error", fields, chains)
    pred_err = table(records, "predicted_error", fields, chains)

    print(f"{'normalised by':<38} {'rms':>7} {'grand':>8} {'row sd':>8} "
          f"{'col sd':>8} {'inter':>8}")
    out = {}
    for label, err in (("measurement error only (as published)", meas_err),
                       ("measurement + prediction error", np.hypot(meas_err, pred_err))):
        r = (measured - predicted) / err
        grand, row, col, inter = decompose(r)
        print(f"{label:<38} {np.sqrt((r**2).mean()):7.3f} {grand:+8.3f} "
              f"{row.std(ddof=1):8.3f} {col.std(ddof=1):8.3f} "
              f"{np.sqrt((inter**2).mean()):8.3f}")
        out[label] = {"rms": float(np.sqrt((r**2).mean())), "grand_mean": float(grand),
                      "row_sd": float(row.std(ddof=1)), "col_sd": float(col.std(ddof=1)),
                      "interaction_rms": float(np.sqrt((inter**2).mean()))}

    # Is the interaction the size the omitted prediction error predicts?
    r_pub = (measured - predicted) / meas_err
    _, _, _, inter = decompose(r_pub)
    expected = float(np.sqrt(np.mean((pred_err / meas_err) ** 2)))
    observed = float(np.sqrt((inter**2).mean()))
    print(f"\ninteraction rms observed        : {observed:.3f}")
    print(f"prediction error / measurement  : {expected:.3f}  "
          f"(the per-cell term left out of the denominator)")
    print(f"ratio                           : {observed / max(expected, 1e-12):.2f}")
    verdict = ("the interaction is what the omitted prediction error accounts for; "
               "the normalisation was wrong and nothing else is"
               if 0.5 < observed / max(expected, 1e-12) < 2.0 else
               "the interaction is not explained by the omitted prediction error, "
               "so mis-normalisation is not the whole story")
    print(f"VERDICT: {verdict}")

    out["interaction_vs_prediction_error"] = {
        "interaction_rms": observed, "prediction_error_scale": expected,
        "ratio": observed / max(expected, 1e-12), "verdict": verdict,
    }
    dest = directory / "variance_budget.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
