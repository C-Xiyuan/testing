"""Rank correlation between cheap proxy metrics and downstream observable error.

This module is the statistical instrument for **prediction P1** of
``docs/theory.md``: that force error -- and more generally any norm-like proxy
metric -- ranks models poorly against the physics they are used to produce.
The claim is a statement about *rankings*, so the estimator is a rank
correlation; and the claim is made from a model zoo of a few dozen entries, so
the estimator is useless without an interval.

Three design decisions follow from the argument the module exists to support,
and each is a guard against the kind of overclaiming this project criticises.

1. **The interval is not optional.**  With ``n = 30`` models the 95% bootstrap
   interval on a Spearman ``rho`` is typically ``+-0.35`` wide.  Reporting
   ``rho = 0.4`` from such a sample without its interval is not evidence of
   anything.  :func:`rank_correlation_with_ci` therefore returns a
   :class:`RankCorrelation`, whose ``__repr__`` and :meth:`RankCorrelation.text`
   both print the interval, and whose :attr:`RankCorrelation.brackets_zero`
   flag makes "consistent with no relationship at all" impossible to overlook.
2. **Multiplicity is controlled explicitly.**  A metric zoo of ~15 entries
   against ~10 observables is 150 hypothesis tests; at ``alpha = 0.05`` roughly
   seven of them are expected to look significant under a global null.
   :func:`proxy_quality_matrix` therefore reports Benjamini--Hochberg adjusted
   p-values next to the raw ones, and records in
   :attr:`ProxyQualityMatrix.family` exactly which set of tests the correction
   was taken over.
3. **The decision, not the coefficient, is the point.**  A practitioner does
   not consume a ranking; they pick the best one or two models.
   :func:`top_k_agreement` measures the overlap between the top ``k`` by proxy
   and the top ``k`` by truth, together with the *regret* -- how much worse the
   proxy-selected model actually is -- which is the quantity with units of
   physics rather than of statistics.

Tie conventions
---------------

Both estimators are implemented directly here rather than delegated, so that
the tie handling is explicit and testable; ``tests/test_metrics.py`` checks
them against :mod:`scipy.stats` on data containing deliberate ties.

* :func:`spearman` uses **midranks** (``scipy.stats.rankdata(method="average")``)
  and then the Pearson correlation *of the ranks*.  The familiar
  ``1 - 6 sum d^2 / n(n^2-1)`` shortcut is only valid without ties and is not
  used.  This matches ``scipy.stats.spearmanr``.
* :func:`kendall_tau` computes **tau-b**, ``(C - D) / sqrt((n0-n1)(n0-n2))``,
  where tied pairs count as neither concordant nor discordant and are removed
  from the normalisation separately for each variable.  This matches
  ``scipy.stats.kendalltau``'s default.  Ties matter here in practice: metric
  values are floats and rarely tie, but *bootstrap resamples duplicate models*,
  so every resample of a rank correlation is a tied sample by construction.

All randomness takes an explicit ``seed`` and uses a dedicated
:class:`numpy.random.Generator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.stats import kendalltau as _scipy_kendalltau
from scipy.stats import rankdata
from scipy.stats import spearmanr as _scipy_spearmanr

from .statistics import Estimate

__all__ = [
    "RankCorrelation",
    "BHResult",
    "ProxyQualityMatrix",
    "TopKAgreement",
    "spearman",
    "kendall_tau",
    "rank_correlation",
    "rank_correlation_pvalue",
    "rank_correlation_with_ci",
    "benjamini_hochberg",
    "proxy_quality_matrix",
    "top_k_agreement",
]


# --------------------------------------------------------------------------
# Point estimators
# --------------------------------------------------------------------------


def _clean_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays of equal length, dropping non-finite pairs.

    A model that failed to produce one metric should not silently poison the
    correlation for every other metric, so pairs containing a NaN in either
    slot are dropped and the remaining sample size is what gets reported.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"x and y must have the same length, got {a.shape} and {b.shape}")
    good = np.isfinite(a) & np.isfinite(b)
    return a[good], b[good]


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(B, n)`` arrays -> ``(B,)``.

    Rows with zero variance in either input (which a bootstrap resample can
    easily produce, e.g. by drawing the same model ``n`` times) return NaN
    rather than 0/0; the caller drops them and records how many were lost.
    """
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman(x, y) -> float:
    """Spearman rank correlation coefficient ``rho``.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Paired samples; dimensionless.  Pairs where either value is non-finite
        are dropped.

    Returns
    -------
    float
        ``rho`` in ``[-1, 1]``, or NaN if fewer than three usable pairs remain
        or either variable is constant (a constant has no ranking, so the
        correlation is undefined rather than zero).

    Notes
    -----
    Ties receive the **average** of the ranks they span (midranks), and the
    coefficient is then the Pearson correlation of those midranks.  This is the
    ``scipy.stats.spearmanr`` convention and is the only one that keeps
    ``rho = 1`` for a perfectly monotone relation that happens to contain
    repeated values.
    """
    a, b = _clean_pair(x, y)
    if a.size < 3:
        return float("nan")
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    return float(_pearson_rows(ra[None, :], rb[None, :])[0])


def _tau_b_from_signs(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """tau-b for stacked sign matrices ``(B, n, n)`` -> ``(B,)``.

    ``sx[b, i, j] = sign(x_i - x_j)``.  Concordant minus discordant pairs is
    ``sum_{i<j} sx*sy``, which equals half the full-matrix sum because the
    product is symmetric under ``i <-> j``.  Tied pairs contribute a zero sign
    and so count as neither, and are additionally removed from the
    normalisation via ``n1``/``n2`` -- that removal is what distinguishes
    tau-b from tau-a.
    """
    n = sx.shape[1]
    n0 = 0.5 * n * (n - 1)
    cd = 0.5 * (sx * sy).sum(axis=(1, 2))
    # Off-diagonal zeros of the sign matrix are exactly the tied pairs.
    n1 = 0.5 * ((sx == 0).sum(axis=(1, 2)) - n)
    n2 = 0.5 * ((sy == 0).sum(axis=(1, 2)) - n)
    den = np.sqrt((n0 - n1) * (n0 - n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, cd / den, np.nan)


def kendall_tau(x, y) -> float:
    """Kendall rank correlation ``tau-b``.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Paired samples; dimensionless.  Non-finite pairs are dropped.

    Returns
    -------
    float
        ``tau_b`` in ``[-1, 1]``, or NaN if fewer than three usable pairs
        remain or one variable is entirely tied.

    Notes
    -----
    ``tau_b = (C - D) / sqrt((n0 - n1) (n0 - n2))`` with ``n0 = n(n-1)/2`` the
    number of pairs, ``n1`` the number of pairs tied in ``x`` and ``n2`` the
    number tied in ``y``.  Computed by an explicit ``O(n^2)`` sign product,
    which is affordable for the few-dozen-model zoos this project compares and
    leaves nothing about the tie handling implicit.  Matches
    ``scipy.stats.kendalltau(..., variant="b")``.

    ``tau`` is preferred to ``rho`` when the sample is small: it is a direct
    probability statement about pairs of models ("how often does the proxy get
    the ordering of two models right?"), which is closer to the decision a
    practitioner makes than the rank-variance ratio ``rho`` reports.
    """
    a, b = _clean_pair(x, y)
    if a.size < 3:
        return float("nan")
    sx = np.sign(a[None, :, None] - a[None, None, :])
    sy = np.sign(b[None, :, None] - b[None, None, :])
    return float(_tau_b_from_signs(sx, sy)[0])


def rank_correlation(x, y, method: str = "spearman") -> float:
    """Dispatch to :func:`spearman` or :func:`kendall_tau` by name."""
    method = method.lower()
    if method in ("spearman", "rho"):
        return spearman(x, y)
    if method in ("kendall", "kendalltau", "tau"):
        return kendall_tau(x, y)
    raise ValueError(f"unknown rank correlation method {method!r}")


def rank_correlation_pvalue(
    x,
    y,
    *,
    method: str = "spearman",
    p_method: str = "asymptotic",
    n_permutations: int = 5000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for the null of no monotone association.

    Parameters
    ----------
    x, y : array_like, shape (n,)
    method : {"spearman", "kendall"}
    p_method : {"asymptotic", "permutation"}
        ``"asymptotic"`` delegates to SciPy (a ``t`` approximation for Spearman,
        the normal approximation with tie correction for Kendall).  Both are
        only asymptotically valid, and with ``n ~ 20-40`` models they are
        mildly anticonservative -- which is why ``"permutation"`` exists: it
        shuffles ``y`` against ``x`` and is exact up to Monte Carlo error under
        the exchangeability null, at a cost of ``n_permutations`` correlations.
    n_permutations, seed :
        Only used by the permutation route; the seed makes it reproducible.

    Returns
    -------
    float
        Two-sided p-value, NaN if the correlation itself is undefined.

    Notes
    -----
    The permutation p-value uses the ``(1 + #{|rho*| >= |rho|}) / (1 + B)``
    convention, which never returns exactly zero -- a p-value of zero from a
    finite number of permutations is a statement the data cannot support.
    """
    a, b = _clean_pair(x, y)
    if a.size < 3:
        return float("nan")
    if p_method == "asymptotic":
        if method.lower() in ("spearman", "rho"):
            return float(_scipy_spearmanr(a, b).pvalue)
        return float(_scipy_kendalltau(a, b, variant="b").pvalue)
    if p_method != "permutation":
        raise ValueError(f"unknown p_method {p_method!r}")

    observed = rank_correlation(a, b, method)
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for t in range(n_permutations):
        null[t] = rank_correlation(a, rng.permutation(b), method)
    finite = np.isfinite(null)
    hits = int((np.abs(null[finite]) >= abs(observed) - 1e-15).sum())
    return float((1 + hits) / (1 + int(finite.sum())))


# --------------------------------------------------------------------------
# Interval estimate
# --------------------------------------------------------------------------


@dataclass
class RankCorrelation(Estimate):
    """A rank correlation that cannot be reported without its interval.

    Subclasses :class:`~atomlab.analysis.statistics.Estimate` so that it drops
    into anything expecting one (``value``/``error``/``method``), while adding
    the percentile interval and printing it in every string form.  The point of
    the subclass is rhetorical as much as technical: with a few dozen models
    the interval is the headline, and an object whose ``repr`` is
    ``0.42`` invites exactly the sentence this project exists to argue against.

    Attributes
    ----------
    value : float
        Point estimate of the rank correlation on the full sample.
    error : float
        Standard deviation over bootstrap resamples.  Reported for
        compatibility; prefer the interval, which does not assume symmetry --
        the sampling distribution of a rank correlation near ``|rho| = 1`` is
        strongly skewed.
    ci_low, ci_high : float
        Percentile bootstrap interval at ``level``.
    level : float
        Nominal coverage, e.g. ``0.95``.
    p_value : float
        Two-sided p-value against the no-association null (see
        :func:`rank_correlation_pvalue`).
    n_samples : int
        Number of usable pairs, i.e. models with both numbers finite.
    correlation_method : str
        ``"spearman"`` or ``"kendall"``.
    n_degenerate : int
        Bootstrap resamples discarded because a resampled variable had zero
        rank variance.  A large count means the interval rests on fewer
        resamples than requested and the sample is very small.
    """

    ci_low: float = float("nan")
    ci_high: float = float("nan")
    level: float = 0.95
    p_value: float = float("nan")
    n_samples: int = 0
    correlation_method: str = "spearman"
    n_degenerate: int = 0

    @property
    def brackets_zero(self) -> bool:
        """True if the interval contains zero: consistent with *no* ranking skill."""
        return bool(self.ci_low <= 0.0 <= self.ci_high)

    @property
    def ci_width(self) -> float:
        """Width of the interval; the number that says how little ``n`` models buy."""
        return float(self.ci_high - self.ci_low)

    def text(self) -> str:
        """One-line report: point estimate, interval, sample size, p-value."""
        tag = "" if not self.brackets_zero else "  [CI includes 0]"
        return (
            f"{self.correlation_method} = {float(self.value):+.3f} "
            f"({self.level:.0%} CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"n={self.n_samples}, p={self.p_value:.3g}){tag}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RankCorrelation({self.text()})"

    __str__ = __repr__

    def to_dict(self) -> dict:
        """Flat record for a results table."""
        return {
            "method": self.correlation_method,
            "value": float(self.value),
            "stderr": float(self.error),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
            "brackets_zero": self.brackets_zero,
        }


def _bootstrap_rho(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    n_bootstrap: int,
    seed: int,
    chunk: int = 256,
) -> np.ndarray:
    """Paired (case) bootstrap distribution of a rank correlation.

    Resamples *models* with replacement, keeping each model's (proxy, truth)
    pair together -- the sampling unit is the model, because that is what a
    replication of this study would redraw.  Vectorised in chunks so that the
    coverage validation in the test suite (hundreds of datasets times a
    thousand resamples) is seconds rather than minutes.
    """
    n = a.size
    rng = np.random.default_rng(seed)
    out = np.empty(n_bootstrap)
    done = 0
    while done < n_bootstrap:
        size = min(chunk, n_bootstrap - done)
        idx = rng.integers(0, n, size=(size, n))
        xa, xb = a[idx], b[idx]
        if method == "spearman":
            out[done : done + size] = _pearson_rows(
                rankdata(xa, method="average", axis=1), rankdata(xb, method="average", axis=1)
            )
        else:
            sx = np.sign(xa[:, :, None] - xa[:, None, :])
            sy = np.sign(xb[:, :, None] - xb[:, None, :])
            out[done : done + size] = _tau_b_from_signs(sx, sy)
        done += size
    return out


def rank_correlation_with_ci(
    x,
    y,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    method: str = "spearman",
    level: float = 0.95,
    p_method: str = "asymptotic",
) -> RankCorrelation:
    """Rank correlation with a percentile bootstrap confidence interval.

    Parameters
    ----------
    x : array_like, shape (n,)
        Proxy metric across ``n`` models (dimensionless for the correlation;
        each metric carries its own units, which are irrelevant to a rank
        statistic).
    y : array_like, shape (n,)
        Downstream observable error across the same models, same order.
    n_bootstrap : int
        Number of paired resamples.  2000 is enough for a 95% percentile
        interval; the Monte Carlo error on an interval endpoint is then well
        below the sampling error the interval is measuring.
    seed : int
        Seeds a dedicated generator; results are exactly reproducible.
    method : {"spearman", "kendall"}
    level : float
        Nominal coverage of the interval.
    p_method : {"asymptotic", "permutation"}
        Passed to :func:`rank_correlation_pvalue`.

    Returns
    -------
    RankCorrelation

    Notes
    -----
    The interval is a **percentile** interval, chosen over the normal
    approximation because the sampling distribution of a rank correlation is
    bounded in ``[-1, 1]`` and visibly skewed once ``|rho| > 0.5``; a
    symmetric ``+- 2 sigma`` interval routinely extends past 1.

    Percentile bootstrap intervals for correlation coefficients are known to
    undercover slightly at small ``n`` (they neglect the bias and the
    acceleration that BCa corrects).  ``tests/test_metrics.py`` measures the
    empirical coverage against a known population Spearman ``rho`` and reports
    it, so the shortfall is quantified rather than assumed away.
    """
    method = "spearman" if method.lower() in ("spearman", "rho") else "kendall"
    a, b = _clean_pair(x, y)
    n = a.size
    point = rank_correlation(a, b, method)
    if n < 4 or not np.isfinite(point):
        return RankCorrelation(
            value=point,
            error=float("nan"),
            n_effective=float(n),
            method=f"bootstrap(n={n_bootstrap})",
            ci_low=float("nan"),
            ci_high=float("nan"),
            level=level,
            p_value=float("nan"),
            n_samples=int(n),
            correlation_method=method,
        )

    samples = _bootstrap_rho(a, b, method, n_bootstrap, seed)
    finite = samples[np.isfinite(samples)]
    alpha = 0.5 * (1.0 - level)
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    return RankCorrelation(
        value=float(point),
        error=float(finite.std(ddof=1)),
        n_effective=float(n),
        method=f"percentile_bootstrap(n={n_bootstrap}, level={level})",
        extra={"samples": samples},
        ci_low=float(lo),
        ci_high=float(hi),
        level=float(level),
        p_value=rank_correlation_pvalue(a, b, method=method, p_method=p_method, seed=seed),
        n_samples=int(n),
        correlation_method=method,
        n_degenerate=int(samples.size - finite.size),
    )


# --------------------------------------------------------------------------
# Multiple-comparison control
# --------------------------------------------------------------------------


@dataclass
class BHResult:
    """Outcome of a false-discovery-rate correction.

    Attributes
    ----------
    rejected : ndarray of bool, shape (m,)
        Which hypotheses are rejected at ``alpha``.
    adjusted : ndarray of float, shape (m,)
        Adjusted p-values (q-values), in the original input order.  Rejection is
        equivalent to ``adjusted <= alpha``.
    alpha : float
        Target FDR.
    n_tests : int
        Size of the family actually corrected over -- NaN p-values are excluded
        from the family, since a test that could not be computed is not a test
        that was performed.
    dependence : str
        ``"independent"`` (Benjamini--Hochberg) or ``"arbitrary"``
        (Benjamini--Yekutieli).
    n_rejected : int
    """

    rejected: np.ndarray
    adjusted: np.ndarray
    alpha: float
    n_tests: int
    dependence: str
    n_rejected: int


def benjamini_hochberg(pvalues, alpha: float = 0.05, *, dependence: str = "independent") -> BHResult:
    """Benjamini--Hochberg (or Benjamini--Yekutieli) FDR control.

    Parameters
    ----------
    pvalues : array_like, shape (m,)
        Raw p-values for one family of tests.  NaN entries are passed through
        as NaN and excluded from ``m``.
    alpha : float
        Target false discovery rate: the expected fraction of *rejections* that
        are false, as opposed to the family-wise error rate Bonferroni controls.
        FDR is the right target here -- with 150 correlations, insisting on no
        false positive at all would leave no power to detect the real ones.
    dependence : {"independent", "arbitrary"}
        ``"independent"`` is standard BH, valid under independence or positive
        regression dependence (PRDS).  ``"arbitrary"`` applies the
        Benjamini--Yekutieli ``sum_{i=1}^m 1/i`` inflation, valid under any
        dependence structure at the cost of a factor ``~ln m`` in power.
        Proxy metrics computed on one test set are strongly and *positively*
        dependent (they are all functions of the same error field), which is
        the PRDS case, so ``"independent"`` is the default; the conservative
        option is one keyword away when that assumption is doubted.

    Returns
    -------
    BHResult

    Notes
    -----
    The step-up procedure: sort p-values ascending, form
    ``q_i = p_i * m * c(m) / i``, then enforce monotonicity by taking the
    running minimum **from the largest p-value downwards** (without which a
    small p-value could end up with a larger adjusted value than a bigger one),
    and clip to 1.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    good = np.isfinite(p)
    m = int(good.sum())
    adjusted = np.full(p.shape, np.nan)
    rejected = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return BHResult(rejected, adjusted, float(alpha), 0, dependence, 0)

    if dependence == "independent":
        c_m = 1.0
    elif dependence == "arbitrary":
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    else:
        raise ValueError(f"dependence must be 'independent' or 'arbitrary', got {dependence!r}")

    idx = np.flatnonzero(good)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranks = np.arange(1, m + 1, dtype=float)
    q = p[order] * m * c_m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotone non-decreasing in p
    q = np.clip(q, 0.0, 1.0)
    adjusted[order] = q
    rejected[order] = q <= alpha
    return BHResult(rejected, adjusted, float(alpha), m, dependence, int(rejected.sum()))


# --------------------------------------------------------------------------
# The proxy-quality table
# --------------------------------------------------------------------------


@dataclass
class ProxyQualityMatrix:
    """Rank correlations between every proxy metric and every observable error.

    Every entry is a correlation across the model zoo: rows are proxy metrics
    (force RMSE, energy MAE, ...), columns are observables (RDF error, phonon
    error, ...), and the entry is how well that metric ranks models for that
    observable.

    Attributes
    ----------
    metric_names : list of str, length M
    observable_names : list of str, length O
    rho : ndarray, shape (M, O)
        Point estimates of the rank correlation.
    ci_low, ci_high : ndarray, shape (M, O)
        Percentile bootstrap interval endpoints.
    p_raw : ndarray, shape (M, O)
        Uncorrected two-sided p-values.
    p_adjusted : ndarray, shape (M, O)
        FDR-adjusted p-values over the whole ``M x O`` family.
    significant : ndarray of bool, shape (M, O)
        ``p_adjusted <= alpha``.
    n_models : ndarray of int, shape (M, O)
        Usable models per cell (pairs with a NaN on either side are dropped).
    family : str
        Human-readable statement of the family the FDR correction was taken
        over.  Recorded because "which family?" is the question that makes or
        breaks a multiplicity claim, and it must not live only in someone's
        memory of how the script was called.
    method, level, alpha, dependence : str / float
        Estimator settings, carried so a table can document itself.
    """

    metric_names: list
    observable_names: list
    rho: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    p_raw: np.ndarray
    p_adjusted: np.ndarray
    significant: np.ndarray
    n_models: np.ndarray
    family: str
    method: str = "spearman"
    level: float = 0.95
    alpha: float = 0.05
    dependence: str = "independent"
    model_names: list = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rho.shape

    @property
    def n_significant(self) -> int:
        """Number of cells surviving FDR control."""
        return int(np.nansum(self.significant))

    @property
    def n_brackets_zero(self) -> int:
        """Number of cells whose confidence interval contains zero."""
        with np.errstate(invalid="ignore"):
            return int(np.sum((self.ci_low <= 0.0) & (self.ci_high >= 0.0)))

    def cell(self, metric: str, observable: str) -> dict:
        """Every number for one (metric, observable) pair, as a flat dict."""
        i = self.metric_names.index(metric)
        j = self.observable_names.index(observable)
        return {
            "metric": metric,
            "observable": observable,
            "rho": float(self.rho[i, j]),
            "ci_low": float(self.ci_low[i, j]),
            "ci_high": float(self.ci_high[i, j]),
            "p_raw": float(self.p_raw[i, j]),
            "p_adjusted": float(self.p_adjusted[i, j]),
            "significant": bool(self.significant[i, j]),
            "n_models": int(self.n_models[i, j]),
        }

    def to_records(self) -> list:
        """Long-format list of dicts, one per cell -- ready for pandas or JSON."""
        return [
            self.cell(m, o) for m in self.metric_names for o in self.observable_names
        ]

    def to_dataframe(self):
        """Long-format :class:`pandas.DataFrame` (pandas imported lazily)."""
        import pandas as pd  # local import: pandas is not needed to use this module

        return pd.DataFrame(self.to_records())

    def summary(self) -> str:
        """A short paragraph stating the multiplicity situation honestly."""
        total = int(self.rho.size)
        return (
            f"{total} correlations ({self.shape[0]} metrics x {self.shape[1]} observables), "
            f"method={self.method}, family='{self.family}', FDR alpha={self.alpha} "
            f"({self.dependence} dependence): {self.n_significant} significant after "
            f"adjustment, {self.n_brackets_zero} with a {self.level:.0%} interval "
            f"containing zero."
        )


def _align(metrics_by_model: Mapping, observables_by_model: Mapping) -> tuple[list, list, list]:
    """Return the shared model list plus the metric and observable name lists."""
    models = [m for m in metrics_by_model if m in observables_by_model]
    if len(models) < 3:
        raise ValueError(
            f"need at least 3 models present in both dictionaries, got {len(models)}; "
            "a rank correlation over fewer models is not a measurement"
        )
    metric_names: list = []
    for name in models:
        for key in metrics_by_model[name]:
            if key not in metric_names:
                metric_names.append(key)
    observable_names: list = []
    for name in models:
        for key in observables_by_model[name]:
            if key not in observable_names:
                observable_names.append(key)
    return models, metric_names, observable_names


def proxy_quality_matrix(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    observable_errors_by_model: Mapping[str, Mapping[str, float]],
    *,
    method: str = "spearman",
    n_bootstrap: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    alpha: float = 0.05,
    dependence: str = "independent",
    p_method: str = "asymptotic",
    family: str | None = None,
) -> ProxyQualityMatrix:
    """Rank-correlate every proxy metric against every observable error.

    This is the table that tests P1.  Each cell answers: *if I had ranked these
    models by this cheap metric, how well would that ranking have matched the
    ranking by this physical observable's error?*

    Parameters
    ----------
    metrics_by_model : mapping
        ``{model_name: {metric_name: value}}``.  Typically the output of
        :func:`atomlab.analysis.metrics.compute_all_metrics` per model.
    observable_errors_by_model : mapping
        ``{model_name: {observable_name: error}}``.  Errors, not values: the
        correlation is between "how bad is this metric" and "how wrong is this
        observable", so both must point the same way.  Metrics for which larger
        is better (force cosine) still correlate *negatively* with error and
        that is the correct, interpretable sign -- no sign flipping is applied
        here, because silently reorienting a metric would hide exactly the
        inversions this study is looking for.
    method, n_bootstrap, seed, level, p_method :
        Passed through to :func:`rank_correlation_with_ci`.
    alpha, dependence :
        Passed to :func:`benjamini_hochberg`.
    family : str, optional
        Description of the family being corrected.  Defaults to a statement of
        the full ``M x O`` grid, which is the family actually tested.

    Returns
    -------
    ProxyQualityMatrix

    Notes
    -----
    Only models present in *both* dictionaries are used, in the iteration order
    of ``metrics_by_model``; the model list is recorded in the result so that a
    later reader can see which zoo the correlation was measured over.

    **The family.** The FDR correction is taken over all ``M x O`` cells
    computed in this single call.  If a caller runs several matrices (e.g. one
    per reference system) and wants control over their union, they must pool
    the p-values themselves -- correcting each matrix separately and then
    reporting the union is not FDR control over the union, and this docstring
    is the only place that warning will ever be read.
    """
    models, metric_names, observable_names = _align(metrics_by_model, observable_errors_by_model)
    n_m, n_o = len(metric_names), len(observable_names)

    rho = np.full((n_m, n_o), np.nan)
    ci_low = np.full((n_m, n_o), np.nan)
    ci_high = np.full((n_m, n_o), np.nan)
    p_raw = np.full((n_m, n_o), np.nan)
    counts = np.zeros((n_m, n_o), dtype=int)

    metric_table = np.array(
        [[float(metrics_by_model[m].get(k, np.nan)) for k in metric_names] for m in models]
    )
    obs_table = np.array(
        [[float(observable_errors_by_model[m].get(k, np.nan)) for k in observable_names] for m in models]
    )

    for i in range(n_m):
        for j in range(n_o):
            # Distinct seed per cell so the resamples are independent across the
            # table but the whole table is reproducible from one seed.
            est = rank_correlation_with_ci(
                metric_table[:, i],
                obs_table[:, j],
                n_bootstrap=n_bootstrap,
                seed=seed + i * n_o + j,
                method=method,
                level=level,
                p_method=p_method,
            )
            rho[i, j] = est.value
            ci_low[i, j] = est.ci_low
            ci_high[i, j] = est.ci_high
            p_raw[i, j] = est.p_value
            counts[i, j] = est.n_samples

    bh = benjamini_hochberg(p_raw.ravel(), alpha=alpha, dependence=dependence)
    p_adj = bh.adjusted.reshape(n_m, n_o)
    significant = bh.rejected.reshape(n_m, n_o)

    return ProxyQualityMatrix(
        metric_names=list(metric_names),
        observable_names=list(observable_names),
        rho=rho,
        ci_low=ci_low,
        ci_high=ci_high,
        p_raw=p_raw,
        p_adjusted=p_adj,
        significant=significant,
        n_models=counts,
        family=family
        or (
            f"all {n_m} x {n_o} = {n_m * n_o} (proxy metric, observable) rank correlations "
            f"computed over {len(models)} models in this call"
        ),
        method="spearman" if method.lower() in ("spearman", "rho") else "kendall",
        level=level,
        alpha=alpha,
        dependence=dependence,
        model_names=list(models),
    )


# --------------------------------------------------------------------------
# The decision-relevant statistic
# --------------------------------------------------------------------------


@dataclass
class TopKAgreement:
    """How often selecting by proxy selects the models truth would have selected.

    Attributes
    ----------
    fraction : float
        ``|top-k by proxy  ∩  top-k by truth| / k``.
    n_overlap : int
        The intersection size itself.
    k, n_models : int
    expected_by_chance : float
        ``k / n_models``: the overlap fraction a random selection would achieve
        in expectation (hypergeometric mean).  A proxy is only useful to the
        extent that ``fraction`` exceeds this, and with ``k/n = 0.2`` a "40%
        overlap" is a coin flip dressed up.
    regret : float
        ``truth[best by proxy] - min(truth)``, in the units of the truth
        column.  The practically meaningful number: how much worse is the model
        you would actually have shipped.
    relative_regret : float
        ``regret / min(truth)``, dimensionless, NaN if the best truth value is
        zero.
    selected_by_proxy, selected_by_truth : list of int
        Indices of the two top-k sets, best first.
    ci_low, ci_high : float
        Optional percentile bootstrap interval on ``fraction`` over resampled
        model zoos; NaN unless ``n_bootstrap > 0``.
    """

    fraction: float
    n_overlap: int
    k: int
    n_models: int
    expected_by_chance: float
    regret: float
    relative_regret: float
    selected_by_proxy: list
    selected_by_truth: list
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def text(self) -> str:
        """One-line report including the chance baseline."""
        ci = ""
        if np.isfinite(self.ci_low):
            ci = f", 95% CI [{self.ci_low:.2f}, {self.ci_high:.2f}]"
        return (
            f"top-{self.k} overlap {self.n_overlap}/{self.k} = {self.fraction:.2f} "
            f"(chance {self.expected_by_chance:.2f}{ci}); "
            f"regret {self.regret:.4g}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TopKAgreement({self.text()})"


def _top_k_indices(values: np.ndarray, k: int, lower_is_better: bool) -> np.ndarray:
    """Indices of the ``k`` best entries, best first.

    Ties are broken by index (stable sort), which is deterministic and, for the
    purpose of this statistic, conservative in neither direction -- it simply
    does not let an arbitrary tie-break masquerade as skill or as failure.
    """
    keys = values if lower_is_better else -values
    order = np.argsort(keys, kind="stable")
    return order[:k]


def top_k_agreement(
    metric_values,
    truth_values,
    k: int,
    *,
    metric_lower_is_better: bool = True,
    truth_lower_is_better: bool = True,
    n_bootstrap: int = 0,
    seed: int = 0,
    level: float = 0.95,
) -> TopKAgreement:
    """Overlap between the top ``k`` models by proxy and the top ``k`` by truth.

    A practitioner never uses the whole ranking; they take the best model, or
    shortlist the best few.  The rank correlation answers a question nobody
    asks.  This answers the one they do, and it is the statistic this project
    considers decision-relevant.

    Parameters
    ----------
    metric_values : array_like, shape (n,)
        Proxy metric per model, in the metric's own units.
    truth_values : array_like, shape (n,)
        Downstream observable error per model, same order.
    k : int
        Shortlist size, ``1 <= k < n``.
    metric_lower_is_better, truth_lower_is_better : bool
        Orientation of each column.  Force RMSE and observable errors are
        lower-is-better; force cosine similarity is not, and passing
        ``metric_lower_is_better=False`` for it is the caller's responsibility
        (:data:`atomlab.analysis.metrics.METRIC_INFO` records the direction of
        every metric in the zoo for exactly this purpose).
    n_bootstrap : int
        If positive, resample models with replacement to put a percentile
        interval on the overlap fraction.  With ``n ~ 30`` models that interval
        is wide, which is the honest situation and worth showing.
    seed, level :
        Bootstrap controls.

    Returns
    -------
    TopKAgreement

    Notes
    -----
    Models with a non-finite value in either column are dropped before ranking,
    so ``n_models`` in the result may be smaller than the input length.  The
    reported indices refer to positions in the *cleaned* array.
    """
    a, b = _clean_pair(metric_values, truth_values)
    n = a.size
    k = int(k)
    if not 1 <= k < max(n, 2):
        raise ValueError(f"k must satisfy 1 <= k < n_models, got k={k}, n_models={n}")

    proxy_top = _top_k_indices(a, k, metric_lower_is_better)
    truth_top = _top_k_indices(b, k, truth_lower_is_better)
    overlap = len(set(proxy_top.tolist()) & set(truth_top.tolist()))

    best_by_proxy = int(proxy_top[0])
    best_truth = b.min() if truth_lower_is_better else b.max()
    regret = float(abs(b[best_by_proxy] - best_truth))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = float(regret / abs(best_truth)) if abs(best_truth) > 0 else float("nan")

    ci_low = ci_high = float("nan")
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        fracs = np.empty(n_bootstrap)
        for t in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            pa = _top_k_indices(a[idx], k, metric_lower_is_better)
            pb = _top_k_indices(b[idx], k, truth_lower_is_better)
            fracs[t] = len(set(idx[pa].tolist()) & set(idx[pb].tolist())) / k
        alpha = 0.5 * (1.0 - level)
        ci_low, ci_high = (float(v) for v in np.quantile(fracs, [alpha, 1.0 - alpha]))

    return TopKAgreement(
        fraction=overlap / k,
        n_overlap=int(overlap),
        k=k,
        n_models=int(n),
        expected_by_chance=k / n,
        regret=regret,
        relative_regret=rel,
        selected_by_proxy=proxy_top.tolist(),
        selected_by_truth=truth_top.tolist(),
        ci_low=ci_low,
        ci_high=ci_high,
    )
