"""Statistics for correlated time series.

Every number this project reports comes from a molecular dynamics trajectory,
and MD samples are correlated.  Treating ``M`` frames as ``M`` independent
samples understates error bars by roughly ``sqrt(2 tau)``, which for a typical
liquid-state observable with ``tau ~ 50`` frames is a factor of ten.  Since the
central claim of this repository is a *comparison* between models, error bars
that are too small by a factor of ten would manufacture exactly the kind of
false discrimination the study is trying to expose.  Hence this module.

The three tools here are the standard ones and they answer different questions:

* :func:`integrated_autocorrelation_time` -- how many frames does one
  independent sample cost?  Use it to report the effective sample size and to
  choose a block length.
* :func:`blocking_analysis` -- Flyvbjerg--Petersen: does the error estimate
  plateau as blocks grow?  Use it when you want an error bar on a *mean* and
  want evidence that the trajectory was long enough.
* :func:`block_bootstrap` -- an error bar on an *arbitrary statistic* of the
  series (a covariance, a fitted slope, a rank correlation).  Use it for
  everything the first two cannot handle, which in this project is most things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Estimate",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "block_average",
    "blocking_analysis",
    "block_bootstrap",
    "jackknife",
    "weighted_statistics",
    "confidence_interval",
]


@dataclass
class Estimate:
    """A number with an uncertainty and a record of how it was obtained.

    Attributes
    ----------
    value:
        The point estimate.  Scalar or array.
    error:
        One standard error, same shape as ``value``.
    n_effective:
        Effective number of independent samples behind the estimate.
    method:
        How the error was computed, e.g. ``"block_bootstrap(n=1000, L=64)"``.
        Recorded because an error bar without its provenance is not auditable.
    extra:
        Method-specific diagnostics.
    """

    value: np.ndarray | float
    error: np.ndarray | float
    n_effective: float = float("nan")
    method: str = ""
    extra: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

    @property
    def relative_error(self) -> np.ndarray | float:
        """``|error / value|``, guarding against division by zero."""
        v = np.asarray(self.value, dtype=float)
        e = np.asarray(self.error, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(v) > 0, np.abs(e / v), np.inf)
        return float(out) if out.ndim == 0 else out

    def significantly_differs_from(self, other: "Estimate | float", n_sigma: float = 2.0) -> bool:
        """True if this estimate differs from ``other`` by more than ``n_sigma``.

        Errors are combined in quadrature, which assumes the two estimates are
        independent.  When comparing two models measured on the *same* reference
        trajectory that assumption is wrong and conservative; prefer a paired
        analysis in that case.
        """
        if isinstance(other, Estimate):
            diff = np.asarray(self.value) - np.asarray(other.value)
            sigma = np.hypot(np.asarray(self.error), np.asarray(other.error))
        else:
            diff = np.asarray(self.value) - float(other)
            sigma = np.asarray(self.error)
        with np.errstate(divide="ignore", invalid="ignore"):
            return bool(np.all(np.abs(diff) > n_sigma * sigma))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        v = np.asarray(self.value)
        if v.ndim == 0:
            return f"Estimate({float(v):.6g} +/- {float(np.asarray(self.error)):.3g})"
        return f"Estimate(array{v.shape}, mean={v.mean():.6g})"


def _as_series(x) -> np.ndarray:
    """Coerce to ``(M,)`` or ``(M, K)`` float array, samples along axis 0."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        raise ValueError("expected a series of samples, got a scalar")
    if a.ndim > 2:
        raise ValueError(f"expected shape (M,) or (M, K), got {a.shape}")
    return a


def autocorrelation(x, max_lag: int | None = None, *, normalize: bool = True) -> np.ndarray:
    """Normalised autocorrelation function of a series, via FFT.

    Parameters
    ----------
    x:
        ``(M,)`` or ``(M, K)`` samples; for 2-D input each column is treated as
        an independent series and the result has shape ``(max_lag+1, K)``.
    max_lag:
        Largest lag returned.  Defaults to ``M // 2``; beyond that the estimator
        is dominated by noise because it averages over very few pairs.
    normalize:
        If True, divide by the zero-lag value so ``C(0) = 1``.

    Notes
    -----
    Uses the Wiener--Khinchin theorem with zero padding to the next power of two
    to avoid the circular wraparound that would otherwise contaminate long lags.
    The unbiased ``1/(M-t)`` normalisation is used; it has larger variance at
    long lags than the biased ``1/M`` form but does not systematically pull the
    integrated time down, which matters here because underestimating ``tau`` is
    the failure mode with consequences.
    """
    a = _as_series(x)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    m = a.shape[0]
    if max_lag is None:
        max_lag = m // 2
    max_lag = int(min(max_lag, m - 1))

    centered = a - a.mean(axis=0, keepdims=True)
    n_fft = 1 << int(np.ceil(np.log2(2 * m)))
    f = np.fft.rfft(centered, n=n_fft, axis=0)
    acf = np.fft.irfft(f * np.conjugate(f), n=n_fft, axis=0)[: max_lag + 1]
    counts = (m - np.arange(max_lag + 1))[:, None]
    acf = acf / counts
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            acf = np.where(acf[0] > 0, acf / acf[0], 0.0)
    return acf[:, 0] if squeeze else acf


def integrated_autocorrelation_time(x, *, c: float = 5.0, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time ``tau`` in units of the sample spacing.

    Uses Sokal's automatic windowing: sum the autocorrelation function up to the
    smallest window ``W`` satisfying ``W >= c * tau(W)``.  The window is
    essential -- the naive sum over all lags does not converge, because the
    noise in the tail of the estimated ACF accumulates without bound.

    ``c = 5`` is Sokal's recommendation and trades bias against variance
    sensibly for the exponentially-decaying correlations MD produces.  Returns
    at least ``0.5``, the value for a perfectly uncorrelated series under the
    convention ``tau = 1/2 + sum_{t>=1} C(t)``.

    For 2-D input the maximum over columns is returned, which is the
    conservative choice when the columns will be blocked together.
    """
    a = _as_series(x)
    if a.ndim == 2:
        return max(integrated_autocorrelation_time(a[:, k], c=c, max_lag=max_lag)
                   for k in range(a.shape[1]))
    if a.size < 4 or np.allclose(a, a[0]):
        return 0.5
    acf = autocorrelation(a, max_lag=max_lag)
    cumulative = 0.5 + np.cumsum(acf[1:])
    windows = np.arange(1, len(cumulative) + 1)
    ok = windows >= c * np.abs(cumulative)
    idx = int(np.argmax(ok)) if ok.any() else len(cumulative) - 1
    return float(max(0.5, cumulative[idx]))


def effective_sample_size(x, *, c: float = 5.0) -> float:
    """``M / (2 tau)``: the number of statistically independent samples."""
    a = _as_series(x)
    tau = integrated_autocorrelation_time(a, c=c)
    return float(a.shape[0] / (2.0 * tau))


def block_average(x, n_blocks: int) -> np.ndarray:
    """Split into ``n_blocks`` contiguous blocks and return the block means.

    Any remainder at the end of the series is discarded rather than folded into
    the last block, so that all blocks carry equal weight.
    """
    a = _as_series(x)
    m = a.shape[0]
    if n_blocks < 1 or n_blocks > m:
        raise ValueError(f"n_blocks must be in [1, {m}], got {n_blocks}")
    size = m // n_blocks
    trimmed = a[: size * n_blocks]
    return trimmed.reshape(n_blocks, size, *a.shape[1:]).mean(axis=1)


def blocking_analysis(x, *, min_blocks: int = 8) -> Estimate:
    """Flyvbjerg--Petersen blocking estimate of the error on the mean.

    Repeatedly halve the number of samples by averaging adjacent pairs.  For a
    correlated series the naive standard error grows with each transformation
    and then plateaus once the block length exceeds the correlation time; the
    plateau value is the correct error bar.

    We take the plateau as the largest estimate obtained while at least
    ``min_blocks`` blocks remain -- with fewer blocks than that the error on the
    error is so large that the "plateau" is mostly noise, and picking the
    maximum over a noisy tail would systematically overestimate.

    Returns
    -------
    Estimate
        ``value`` is the sample mean, ``error`` the plateau standard error, and
        ``extra["curve"]`` the full sequence of ``(block_length, stderr)`` pairs
        so a caller can plot it and judge whether a plateau was really reached.
    """
    a = _as_series(x)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    mean = a.mean(axis=0)

    curve = []
    level = a
    length = 1
    while level.shape[0] >= min_blocks:
        stderr = level.std(axis=0, ddof=1) / np.sqrt(level.shape[0])
        curve.append((length, stderr if not squeeze else float(stderr[0])))
        n_pairs = level.shape[0] // 2
        level = 0.5 * (level[: 2 * n_pairs : 2] + level[1 : 2 * n_pairs : 2])
        length *= 2

    if not curve:
        err = a.std(axis=0, ddof=1) / np.sqrt(max(a.shape[0], 1))
    else:
        stack = np.stack([np.atleast_1d(np.asarray(s)) for _, s in curve], axis=0)
        err = stack.max(axis=0)

    tau = integrated_autocorrelation_time(a)
    out_value = float(mean[0]) if squeeze else mean
    out_error = float(np.atleast_1d(err)[0]) if squeeze else np.asarray(err)
    return Estimate(
        value=out_value,
        error=out_error,
        n_effective=a.shape[0] / (2.0 * tau),
        method=f"blocking(levels={len(curve)})",
        extra={"curve": curve, "tau": tau},
    )


def block_bootstrap(
    x,
    statistic: Callable[[np.ndarray], np.ndarray | float],
    *,
    n_resamples: int = 1000,
    block_length: int | None = None,
    seed: int = 0,
    tau_multiplier: float = 4.0,
) -> Estimate:
    """Moving-block bootstrap error bar for an arbitrary statistic.

    The ordinary bootstrap resamples individual points and therefore destroys
    the serial correlation, giving error bars that are too small by exactly the
    factor we are trying to account for.  The moving-block bootstrap resamples
    contiguous blocks instead, preserving correlation within a block, and is
    consistent provided the block length grows with the series but stays a small
    fraction of it.

    Parameters
    ----------
    x:
        ``(M,)`` or ``(M, K)`` series.  Rows are resampled together, so for
        multi-column input the joint structure across columns is preserved --
        this matters when the statistic is a covariance between columns.
    statistic:
        Callable applied to a resampled array of the same shape as ``x``.
    block_length:
        Defaults to ``tau_multiplier * tau`` rounded up, clipped to
        ``[1, M // 4]``.  Four correlation times per block leaves under 2% of
        the correlation unaccounted for at the block boundaries.
    seed:
        Seeds a dedicated ``numpy.random.Generator``; no global state is used.

    Returns
    -------
    Estimate
        ``value`` is the statistic on the full series (not the bootstrap mean,
        which is biased), ``error`` the standard deviation over resamples, and
        ``extra["samples"]`` the full bootstrap distribution so that percentile
        intervals can be formed without re-running.
    """
    a = _as_series(x)
    m = a.shape[0]
    if m < 4:
        raise ValueError(f"need at least 4 samples to bootstrap, got {m}")

    if block_length is None:
        tau = integrated_autocorrelation_time(a)
        block_length = int(np.clip(np.ceil(tau_multiplier * tau), 1, max(1, m // 4)))
    block_length = int(np.clip(block_length, 1, m))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block_length))
    max_start = m - block_length

    point = np.asarray(statistic(a), dtype=float)
    samples = np.empty((n_resamples, *point.shape), dtype=float)
    for b in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:m]
        samples[b] = np.asarray(statistic(a[idx]), dtype=float)

    err = samples.std(axis=0, ddof=1)
    return Estimate(
        value=point if point.ndim else float(point),
        error=err if err.ndim else float(err),
        n_effective=m / block_length,
        method=f"block_bootstrap(n={n_resamples}, L={block_length})",
        extra={"samples": samples, "block_length": block_length},
    )


def jackknife(x, statistic: Callable[[np.ndarray], np.ndarray | float], *, n_blocks: int = 20) -> Estimate:
    """Delete-one-block jackknife, used as an independent check on the bootstrap.

    Cheaper than the bootstrap (``n_blocks`` evaluations rather than a thousand)
    and with a different bias structure, so agreement between the two is
    meaningful evidence that neither is malfunctioning.
    """
    a = _as_series(x)
    m = a.shape[0]
    n_blocks = int(np.clip(n_blocks, 2, m))
    size = m // n_blocks
    point = np.asarray(statistic(a), dtype=float)

    partials = []
    for b in range(n_blocks):
        mask = np.ones(m, dtype=bool)
        mask[b * size : (b + 1) * size] = False
        partials.append(np.asarray(statistic(a[mask]), dtype=float))
    partials = np.stack(partials, axis=0)

    mean = partials.mean(axis=0)
    err = np.sqrt((n_blocks - 1) / n_blocks * ((partials - mean) ** 2).sum(axis=0))
    return Estimate(
        value=point if point.ndim else float(point),
        error=err if err.ndim else float(err),
        n_effective=float(n_blocks),
        method=f"jackknife(n_blocks={n_blocks})",
        extra={"partials": partials},
    )


def weighted_statistics(values, weights) -> dict:
    """Weighted mean, variance and effective sample size for importance sampling.

    Used by the exponential-reweighting estimator in
    :mod:`atomlab.analysis.response`, where the weights ``exp(-beta dU)`` can be
    dominated by a handful of configurations.  The Kish effective sample size
    ``(sum w)^2 / sum w^2`` is the diagnostic that says when that has happened:
    an ESS far below the number of frames means the answer rests on a few
    samples and should not be trusted regardless of how small its nominal error
    bar is.

    Returns a dict with ``mean``, ``variance``, ``ess``, ``ess_fraction`` and
    ``max_weight_fraction`` (the share of total weight carried by the single
    largest sample -- an ESS can look acceptable while one frame still dominates).
    """
    v = _as_series(values)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != v.shape[0]:
        raise ValueError(f"weights must be ({v.shape[0]},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights sum to zero or are not finite")

    wn = w / total
    shape = (-1,) + (1,) * (v.ndim - 1)
    mean = (wn.reshape(shape) * v).sum(axis=0)
    var = (wn.reshape(shape) * (v - mean) ** 2).sum(axis=0)
    ess = float(total**2 / np.sum(w**2))
    return {
        "mean": float(mean) if np.ndim(mean) == 0 else mean,
        "variance": float(var) if np.ndim(var) == 0 else var,
        "ess": ess,
        "ess_fraction": ess / v.shape[0],
        "max_weight_fraction": float(wn.max()),
    }


def confidence_interval(samples, level: float = 0.95, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval from a bootstrap distribution.

    The percentile interval is used rather than the normal approximation because
    several statistics in this project (rank correlations, ratios of small
    numbers) have visibly skewed sampling distributions.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    alpha = 0.5 * (1.0 - level)
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [alpha, 1.0 - alpha], axis=axis)
    return lo, hi
