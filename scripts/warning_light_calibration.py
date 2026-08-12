#!/usr/bin/env python3
"""Does the second-order ratio actually warn when linear response fails?

Review item P0-5. The manuscript treats the ratio of the second-order cumulant
term to the first as a self-diagnostic: below a threshold the linear prediction
is to be trusted, above it not. That is a screening test, and a screening test
is characterised by its error rates on cases where the truth is known
independently -- not by a plot of the statistic.

This script computes those rates on every case in the repository where both the
diagnostic and an independent direct measurement exist. The quantity that
matters is the **false-trust rate**: the fraction of cases the diagnostic
declares trustworthy that in fact disagree with direct sampling. A diagnostic
whose false-trust rate is high is worse than no diagnostic, because it converts
an unknown into a wrong confident answer.

Two things must be said before any number is read.

*The threshold was not frozen on a held-out development set.* The 0.25 used
here is the value already in ``ResponsePrediction.is_trustworthy``, chosen a
priori on the grounds that a truncated series needs its next term to be small.
That is better than fitting a threshold to these data -- nothing here was tuned
-- but it is not the protocol P0-5 asks for, which is a threshold frozen on one
set of error-field shapes and evaluated on another. What this script gives is
therefore an audit of a pre-existing threshold on the available cases, not a
validation.

*The cases are not independent.* They come from three experiments sharing a
system, an observable and in places a reference chain. The confidence interval
below treats them as independent Bernoulli trials, which makes it optimistically
narrow; it is reported because an interval that is too narrow still bounds the
diagnostic from below, and the point estimate is already the problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.25          # as used in atomlab.analysis.response
AGREEMENT_SIGMA = 2.0     # |measured - predicted| within this many combined errors


def clopper_pearson_upper(k, n, level=0.95):
    """Upper confidence limit on a binomial rate, exact rather than normal."""
    if n == 0:
        return float("nan")
    if k == n:
        return 1.0
    return float(stats.beta.ppf(level, k + 1, n - k))


def from_exp06():
    path = ROOT / "results/exp06_response_validation/amplitude_records.json"
    if not path.exists():
        return []
    out = []
    for r in json.loads(path.read_text()):
        bin_index = int(r["peak_bin"])
        predicted = float(np.asarray(r["linear_curve"])[bin_index])
        pred_err = float(np.asarray(r["linear_error"]))
        measured = float(np.asarray(r["direct_curve"])[bin_index])
        meas_err = float(np.asarray(r["direct_curve_error"])[bin_index])
        out.append({
            "source": "exp06 amplitude sweep",
            "case": f"amplitude {r['amplitude']:g}",
            "ratio": float(r["second_order_ratio"]),
            "predicted": predicted, "measured": measured,
            "error": float(np.hypot(pred_err, meas_err)),
        })
    return out


def from_exp03(directory):
    path = ROOT / "results" / directory / "records.json"
    manifest = ROOT / "results" / directory / "manifest.json"
    if not path.exists():
        return []
    if manifest.exists() and json.loads(manifest.read_text()).get("quick_mode"):
        print(f"  skipping {directory}: quick-mode run")
        return []
    out = []
    for r in json.loads(path.read_text()):
        if r.get("second_order_ratio") is None or not r.get("direct_equilibrated", True):
            continue
        out.append({
            "source": f"exp03 ({directory})",
            "case": r["name"],
            "ratio": float(r["second_order_ratio"]),
            "predicted": float(r["target_predicted"]),
            "measured": float(r["target_measured"]),
            "error": float(r["target_error"]),
        })
    return out


def from_exp09():
    path = ROOT / "results/exp09_calibration_replication/records.json"
    manifest = ROOT / "results/exp09_calibration_replication/manifest.json"
    if not path.exists():
        return []
    if manifest.exists() and json.loads(manifest.read_text()).get("quick_mode"):
        print("  skipping exp09: quick-mode run")
        return []
    return [{
        "source": "exp09 replication",
        "case": f"{r['field']} vs chain {r['reference_chain']}",
        "ratio": float(r["second_order_ratio"]),
        "predicted": float(r["predicted"]),
        "measured": float(r["measured"]),
        "error": float(np.hypot(r["measurement_error"], r["predicted_error"])),
    } for r in json.loads(path.read_text())]


def main():
    cases = (from_exp06() + from_exp03("exp03_model_zoo")
             + from_exp03("exp03_model_zoo_n40") + from_exp09())
    if not cases:
        print("no cases with both a diagnostic and a direct measurement")
        return
    for c in cases:
        c["agrees"] = abs(c["measured"] - c["predicted"]) <= AGREEMENT_SIGMA * c["error"]
        c["trusted"] = c["ratio"] < THRESHOLD

    n = len(cases)
    trusted = [c for c in cases if c["trusted"]]
    flagged = [c for c in cases if not c["trusted"]]
    false_trust = [c for c in trusted if not c["agrees"]]
    agree = [c for c in cases if c["agrees"]]
    disagree = [c for c in cases if not c["agrees"]]

    print(f"{n} cases from {len(set(c['source'] for c in cases))} experiments; "
          f"threshold {THRESHOLD}, agreement within {AGREEMENT_SIGMA:g} sigma\n")
    print(f"{'':<26}{'agrees':>10}{'disagrees':>12}")
    print(f"{'diagnostic says trust':<26}"
          f"{sum(c['agrees'] for c in trusted):>10}"
          f"{len(false_trust):>12}")
    print(f"{'diagnostic says beware':<26}"
          f"{sum(c['agrees'] for c in flagged):>10}"
          f"{sum(not c['agrees'] for c in flagged):>12}")

    rate = len(false_trust) / len(trusted) if trusted else float("nan")
    upper = clopper_pearson_upper(len(false_trust), len(trusted))
    sens = (sum(not c["agrees"] for c in flagged) / len(disagree)) if disagree else float("nan")
    spec = (sum(c["agrees"] for c in trusted) / len(agree)) if agree else float("nan")
    print(f"\nfalse-trust rate      {rate:.3f}  "
          f"({len(false_trust)}/{len(trusted)}), upper 95% limit {upper:.3f}")
    print(f"sensitivity to failure {sens:.3f}   (disagreements the diagnostic flags)")
    print(f"specificity            {spec:.3f}   (agreements it lets through)")

    # Does the statistic separate the two groups at all, threshold aside?
    ra = np.array([c["ratio"] for c in agree])
    rd = np.array([c["ratio"] for c in disagree])
    if ra.size and rd.size:
        auc = float(np.mean(rd[:, None] > ra[None, :])
                    + 0.5 * np.mean(rd[:, None] == ra[None, :]))
        u = stats.mannwhitneyu(rd, ra, alternative="greater")
        print(f"\nratio, cases that agree    : median {np.median(ra):.4f} "
              f"[{ra.min():.4f}, {ra.max():.4f}], n = {ra.size}")
        print(f"ratio, cases that disagree : median {np.median(rd):.4f} "
              f"[{rd.min():.4f}, {rd.max():.4f}], n = {rd.size}")
        print(f"AUC {auc:.3f} (0.5 = no information), one-sided p = {u.pvalue:.3f}")
    else:
        auc, u = float("nan"), None

    if not np.isnan(upper) and upper < 0.10:
        verdict = ("the upper limit on the false-trust rate is below 10%, so on "
                   "these cases the threshold is safe to use as a gate")
    elif not np.isnan(rate) and rate > 0:
        verdict = (f"{len(false_trust)} of {len(trusted)} trusted cases disagree "
                   f"with direct sampling; the upper limit on the false-trust "
                   f"rate is {upper:.0%}, far above any tolerance one would set, "
                   f"so the diagnostic must not be used to certify a case")
    else:
        verdict = ("no false trust observed, but the upper limit is "
                   f"{upper:.0%} -- too few trusted cases to establish safety")
    print(f"\nVERDICT: {verdict}")

    dest = ROOT / "results/validation/warning_light_calibration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "threshold": THRESHOLD, "agreement_sigma": AGREEMENT_SIGMA,
        "n_cases": n, "n_trusted": len(trusted), "n_false_trust": len(false_trust),
        "false_trust_rate": rate, "false_trust_upper_95": upper,
        "sensitivity": sens, "specificity": spec, "auc": auc,
        "threshold_was_frozen_on_held_out_data": False,
        "cases_are_independent": False,
        "verdict": verdict,
        "cases": cases,
    }, indent=2, default=float))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
