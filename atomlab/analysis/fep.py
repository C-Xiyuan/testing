"""Bidirectional free-energy perturbation: BAR, and two-state MBAR averages.

:mod:`atomlab.analysis.response` estimates an observable shift two ways, both of
them one-sided: a cumulant expansion truncated at first order, and forward
exponential reweighting from the reference ensemble.  Both inherit the same
weakness.  Forward reweighting is dominated by configurations where ``dU`` is
very negative, which are rare under the reference ensemble precisely when they
matter most, so its variance is controlled by a tail the reference chain barely
samples.  When a study reports that reweighting and direct sampling disagree,
the first question a reader should ask is whether the reweighting was ever
converged, and a one-sided estimator cannot answer it.

The bidirectional estimators here can.  Given samples from *both* ensembles they
use each to cover the other's tail:

* :func:`bennett_acceptance_ratio` returns the free-energy difference that makes
  the forward and reverse work distributions consistent.  Bennett showed this is
  the minimum-variance estimator among those built from the two sample sets, and
  its self-consistency condition fails visibly -- rather than silently -- when
  the two ensembles do not overlap.
* :func:`mbar_two_state_average` uses that free energy to weight *all* samples,
  from both ensembles, into an average under either one.  For an observable this
  is what BAR is for a free energy: the estimator that uses both directions.
* :func:`overlap_diagnostics` reports whether the estimate deserves to be
  believed at all -- the Kish effective sample size of each weight set, the
  largest single weight, and the Bhattacharyya-style overlap of the two work
  distributions.  A number without these is not interpretable.

Nothing here is new; BAR is Bennett (1976) and MBAR is Shirts and Chodera
(2008).  It is here because a disagreement between exact reweighting and direct
sampling is a claim about a measurement, and a one-directional measurement is
not enough to make it.

References
----------
C. H. Bennett, *Efficient estimation of free energy differences from Monte Carlo
data*, J. Comput. Phys. **22**, 245 (1976).

M. R. Shirts and J. D. Chodera, *Statistically optimal analysis of samples from
multiple equilibrium states*, J. Chem. Phys. **129**, 124105 (2008).
"""

from __future__ import annotations

import numpy as np

from ..units import beta as inverse_temperature
from .statistics import Estimate, block_bootstrap

__all__ = [
    "bennett_acceptance_ratio",
    "mbar_two_state_average",
    "overlap_diagnostics",
]


def _logistic(x: np.ndarray) -> np.ndarray:
    """``1 / (1 + exp(x))``, without overflowing for large positive ``x``."""
    out = np.empty_like(x, dtype=float)
    pos = x > 0
    out[pos] = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


def bennett_acceptance_ratio(
    du_forward,
    du_reverse,
    temperature: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> Estimate:
    """Free-energy difference ``F_U - F_0`` from work samples in both directions.

    Parameters
    ----------
    du_forward:
        ``U(x) - U0(x)`` evaluated on configurations sampled from the
        **reference** ensemble ``exp(-beta U0)``.
    du_reverse:
        The same energy difference, with the same sign convention, evaluated on
        configurations sampled from the **surrogate** ensemble ``exp(-beta U)``.
        Passing this with the opposite sign is the classic way to get a
        plausible and wrong answer, so the convention is stated twice.
    temperature:
        In kelvin.

    Returns
    -------
    Estimate
        ``value`` is ``dF`` in eV.  ``error`` is Bennett's analytic variance,
        which assumes independent samples and is therefore a lower bound for
        correlated chains -- ``extra["n_forward"]`` and ``extra["n_reverse"]``
        are reported so a reader can rescale by an autocorrelation time, and
        the experiments that use this bootstrap the whole pipeline instead.

    Notes
    -----
    The estimator solves

    ``sum_f 1/(1 + exp(+beta(dU_f - dF) + ln(n_f/n_r)))
       = sum_r 1/(1 + exp(-beta(dU_r - dF) - ln(n_f/n_r)))``

    by bisection on ``dF``.  Bisection rather than Newton because the
    self-consistency function is monotone in ``dF`` and bisection cannot be
    thrown by the numerically flat regions that appear when overlap is poor --
    it just converges to a bracket that :func:`overlap_diagnostics` will then
    show to be untrustworthy.
    """
    f = np.asarray(du_forward, dtype=float).ravel()
    r = np.asarray(du_reverse, dtype=float).ravel()
    if f.size == 0 or r.size == 0:
        raise ValueError("BAR needs samples in both directions")
    b = inverse_temperature(temperature)
    n_f, n_r = f.size, r.size
    m = np.log(n_f / n_r)

    def imbalance(df: float) -> float:
        return float(_logistic(b * (f - df) + m).sum() - _logistic(-b * (r - df) - m).sum())

    # Bracket. The answer lies between the two one-sided FEP estimates whenever
    # they are finite, but poor overlap can push it outside, so widen until the
    # sign changes rather than trusting the bracket.
    lo, hi = min(f.min(), r.min()), max(f.max(), r.max())
    span = max(hi - lo, 1e-6)
    lo, hi = lo - span, hi + span
    for _ in range(60):
        if imbalance(lo) * imbalance(hi) <= 0:
            break
        lo, hi = lo - span, hi + span
        span *= 2
    else:  # pragma: no cover - only reachable with degenerate input
        raise RuntimeError("could not bracket the BAR self-consistency root")

    # The imbalance is monotone in df but its direction depends on nothing we
    # should have to reason about, so take it from the bracket rather than
    # assume it -- getting this backwards converges just as smoothly, to the
    # wrong end of the interval.
    increasing = imbalance(lo) < 0
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        if (imbalance(mid) > 0) == increasing:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    df = 0.5 * (lo + hi)

    # Bennett (1976) eq. 10b: the variance of beta*dF in terms of the Fermi
    # weights at the solution.  Assumes independent samples in each direction,
    # so for a correlated chain it is a lower bound; the experiments that use
    # this bootstrap the whole pipeline rather than relying on it.
    pf = _logistic(b * (f - df) + m)
    pr = _logistic(-b * (r - df) - m)
    var_reduced = ((pf**2).mean() / max(pf.mean() ** 2, 1e-300) - 1.0) / n_f
    var_reduced += ((pr**2).mean() / max(pr.mean() ** 2, 1e-300) - 1.0) / n_r
    return Estimate(
        value=float(df),
        error=float(np.sqrt(max(var_reduced, 0.0)) / b),
        n_effective=float(n_f + n_r),
        method="bennett_acceptance_ratio(Bennett variance, assumes independence)",
        extra={"n_forward": n_f, "n_reverse": n_r,
               "mean_fermi_forward": float(pf.mean()),
               "mean_fermi_reverse": float(pr.mean())},
    )


def mbar_two_state_average(
    a_forward,
    du_forward,
    a_reverse,
    du_reverse,
    temperature: float,
    *,
    delta_f: float | None = None,
    n_resamples: int = 400,
    seed: int = 0,
):
    """Average of ``A`` under both ensembles, using samples from both.

    Every sample contributes to both answers, with the mixture weights that make
    the estimator statistically optimal for two states.  Returns a dictionary
    with the reference average, the surrogate average, the shift between them,
    and the diagnostics that say whether any of it is trustworthy.

    The error on the shift is a moving-block bootstrap over the two chains
    resampled independently, which keeps the correlation structure of each and
    -- unlike an analytic MBAR covariance -- does not assume the samples are
    independent.  The two chains are genuinely independent of each other, so
    resampling them separately and recombining is legitimate.
    """
    af = np.asarray(a_forward, dtype=float)
    ar = np.asarray(a_reverse, dtype=float)
    duf = np.asarray(du_forward, dtype=float).ravel()
    dur = np.asarray(du_reverse, dtype=float).ravel()
    scalar = af.ndim == 1
    if scalar:
        af, ar = af[:, None], ar[:, None]
    if af.shape[0] != duf.size or ar.shape[0] != dur.size:
        raise ValueError("observable and energy-difference samples must line up")

    b = inverse_temperature(temperature)

    def estimate(af_, duf_, ar_, dur_, df=None):
        if df is None:
            df = float(bennett_acceptance_ratio(duf_, dur_, temperature).value)
        n_f, n_r = duf_.size, dur_.size
        # Reduced potentials of every sample in both states. State 0 is the
        # reference, whose reduced potential we may set to zero because only
        # differences matter; state 1 then costs beta*dU.
        u0 = np.concatenate([np.zeros(n_f), np.zeros(n_r)])
        u1 = b * np.concatenate([duf_, dur_])
        a_all = np.concatenate([af_, ar_], axis=0)
        # log denominator of the MBAR mixture: log sum_s N_s exp(f_s - u_s)
        log_terms = np.stack([np.log(n_f) + 0.0 - u0,
                              np.log(n_r) + b * df - u1], axis=0)
        log_denom = np.logaddexp(log_terms[0], log_terms[1])
        out = {}
        for name, u in (("reference", u0), ("surrogate", u1)):
            log_w = -u - log_denom
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()
            out[name] = w @ a_all
            out[f"{name}_kish"] = float(1.0 / np.sum(w**2) / w.size)
            out[f"{name}_max_weight"] = float(w.max())
        out["shift"] = out["surrogate"] - out["reference"]
        out["delta_f"] = df
        return out

    point = estimate(af, duf, ar, dur)

    # Bootstrap: resample each chain in blocks, recombine, redo BAR each time so
    # the free-energy uncertainty propagates into the observable.
    rng = np.random.default_rng(seed)
    joint_f = np.concatenate([af, duf[:, None]], axis=1)
    joint_r = np.concatenate([ar, dur[:, None]], axis=1)
    k = af.shape[1]

    def resample(joint):
        n = joint.shape[0]
        length = max(1, min(n // 4, int(np.sqrt(n))))
        n_blocks = int(np.ceil(n / length))
        starts = rng.integers(0, n - length + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(length)[None, :]).ravel()[:n]
        return joint[idx]

    shifts = []
    for _ in range(int(n_resamples)):
        bf, br = resample(joint_f), resample(joint_r)
        try:
            s = estimate(bf[:, :k], bf[:, k], br[:, :k], br[:, k])["shift"]
        except (RuntimeError, ValueError):  # pragma: no cover - degenerate resample
            continue
        shifts.append(s)
    shifts = np.asarray(shifts)
    error = shifts.std(axis=0, ddof=1) if shifts.shape[0] > 1 else np.full(k, np.nan)

    def unwrap(v):
        arr = np.asarray(v, dtype=float)
        return float(arr.reshape(-1)[0]) if scalar else arr

    return {
        "shift": Estimate(
            unwrap(point["shift"]), unwrap(error), float(shifts.shape[0]),
            f"mbar_two_state(block bootstrap n={shifts.shape[0]})",
        ),
        "reference_average": unwrap(point["reference"]),
        "surrogate_average": unwrap(point["surrogate"]),
        "delta_f": point["delta_f"],
        "kish_reference": point["reference_kish"],
        "kish_surrogate": point["surrogate_kish"],
        "max_weight_reference": point["reference_max_weight"],
        "max_weight_surrogate": point["surrogate_max_weight"],
    }


def overlap_diagnostics(du_forward, du_reverse, temperature: float) -> dict:
    """Whether the two ensembles overlap enough for any of this to mean anything.

    Three numbers, each of which can independently condemn an estimate:

    ``kish_forward`` / ``kish_reverse``
        Effective fraction of samples carrying the one-sided exponential
        average.  Below a few per cent the corresponding FEP estimate is a
        report on a handful of configurations.
    ``max_weight_forward`` / ``max_weight_reverse``
        Largest single normalised weight.  If one frame carries a third of the
        answer, the error bar is fiction whatever its provenance.
    ``work_overlap``
        Fraction of the reverse work distribution lying inside the range of the
        forward one, and vice versa; the smaller of the two.  BAR needs the two
        distributions to intersect, and this says whether they do.
    """
    f = np.asarray(du_forward, dtype=float).ravel()
    r = np.asarray(du_reverse, dtype=float).ravel()
    b = inverse_temperature(temperature)

    def kish(log_w):
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        return float(1.0 / np.sum(w**2) / w.size), float(w.max())

    kish_f, max_f = kish(-b * f)
    kish_r, max_r = kish(+b * r)
    inside_r = float(np.mean((r >= f.min()) & (r <= f.max())))
    inside_f = float(np.mean((f >= r.min()) & (f <= r.max())))
    return {
        "kish_forward": kish_f,
        "kish_reverse": kish_r,
        "max_weight_forward": max_f,
        "max_weight_reverse": max_r,
        "work_overlap": min(inside_f, inside_r),
        "mean_forward": float(f.mean()),
        "mean_reverse": float(r.mean()),
        "std_forward": float(f.std(ddof=1)),
        "std_reverse": float(r.std(ddof=1)),
    }
