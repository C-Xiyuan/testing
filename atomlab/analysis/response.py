"""Linear-response machinery: predicting observable error from the error field.

This module implements the estimators derived in ``docs/theory.md``.  The result
they rest on is that for a static observable ``A`` in the canonical ensemble,
with a surrogate potential ``U = U0 + dU``,

    <A>_U - <A>_0 = -beta * Cov_0(A, dU) + (beta^2/2) * <A~ dU~^2>_0 - ...

so the error in an observable is controlled by the **covariance of the
observable with the error field**, not by any norm of that field.  Three
consequences shape the API:

1. Everything is computed from samples of the *reference* ensemble.  Given a
   reference trajectory and the ability to evaluate both potentials on its
   frames, an observable shift is predictable without running MD with the
   surrogate at all (:func:`predict_shift`).
2. The prediction is a truncated series, so it needs a self-diagnostic.  The
   second-order term is estimable from the same samples and is returned
   alongside the first (:attr:`ResponsePrediction.second_order`), as is the
   exact-but-noisy reweighted answer (:func:`reweight`).  Disagreement between
   the three is informative rather than embarrassing: it localises where linear
   response stops being a valid way to think about model error.
3. A practitioner has no ``U0``.  :func:`committee_response` replaces the
   unknown truth with a committee mean, which is the usable form of the idea --
   and the one whose failure mode (shared systematic error cancels) the
   experiments are built to expose.

All expectations use block-resampled error bars from
:mod:`atomlab.analysis.statistics`, because MD frames are correlated and this
module's entire output is a set of comparisons between numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from ..units import beta as inverse_temperature
from .statistics import (
    Estimate,
    block_bootstrap,
    integrated_autocorrelation_time,
    weighted_statistics,
)

__all__ = [
    "ResponsePrediction",
    "ReweightResult",
    "predict_shift",
    "predict_shift_from_surrogate_samples",
    "reweight",
    "response_diagnostics",
    "delta_u_samples",
    "committee_delta_u",
    "committee_response",
    "spectral_decomposition",
]


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class ResponsePrediction:
    """Predicted shift in an observable, with its trustworthiness diagnostics.

    Attributes
    ----------
    first_order:
        ``-beta Cov_0(A, dU)``.  Shape ``()`` for a scalar observable, ``(K,)``
        for a vector one such as ``g(r)``.  Carries a block-bootstrap error bar.
    second_order:
        ``(beta^2/2) <A~ dU~^2>_0``, the next term in the cumulant expansion.
        Not added to the prediction -- it is reported so that its size relative
        to ``first_order`` can be used as a validity check.
    beta_sigma_dU:
        ``beta * std(dU)``.  The natural dimensionless measure of how large the
        perturbation is.  Linear response is safe when this is well below one;
        by the time it reaches order one the expansion is asymptotic at best.
    correlation:
        Pearson correlation between ``A`` and ``dU`` under the reference
        ensemble.  This is the quantity that distinguishes a harmful error field
        from a harmless one of the same magnitude, and it is the number this
        project argues should be reported in place of force RMSE.
    sigma_A, sigma_dU:
        Reference-ensemble standard deviations, so a caller can reconstruct the
        Cauchy-Schwarz decomposition ``shift = -beta * sigma_A * sigma_dU * rho``.
    n_samples, tau:
        Sample count and integrated autocorrelation time of the product series.
    """

    first_order: Estimate
    second_order: Estimate
    beta_sigma_dU: float
    correlation: np.ndarray | float
    sigma_A: np.ndarray | float
    sigma_dU: float
    n_samples: int
    tau: float
    meta: dict = field(default_factory=dict)

    @property
    def value(self) -> np.ndarray | float:
        """The predicted shift (first order only)."""
        return self.first_order.value

    @property
    def error(self) -> np.ndarray | float:
        """Statistical error on the predicted shift."""
        return self.first_order.error

    @property
    def second_order_ratio(self) -> np.ndarray | float:
        """``|second_order / first_order|``.

        The single most useful number here.  Below ~0.1 the linear prediction is
        quantitative.  Above ~1 the series is not usefully truncated and any
        conclusion phrased as "reducing the error by half would halve the
        observable error" is unfounded.
        """
        first = np.abs(np.asarray(self.first_order.value, dtype=float))
        second = np.abs(np.asarray(self.second_order.value, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(first > 0, second / first, np.inf)
        return float(ratio) if ratio.ndim == 0 else ratio

    @property
    def is_trustworthy(self) -> bool:
        """Heuristic gate combining the two validity checks.

        Requires the second-order ratio below 0.25 (in the median over
        components, so one noisy RDF bin cannot veto an otherwise clean curve)
        and ``beta*std(dU) < 1``.  Deliberately conservative: the cost of
        trusting a broken linear prediction in this study is a wrong scientific
        conclusion, while the cost of falling back to direct MD is compute.
        """
        ratio = np.asarray(self.second_order_ratio, dtype=float)
        median_ratio = float(np.median(ratio[np.isfinite(ratio)])) if ratio.size else np.inf
        return bool(median_ratio < 0.25 and self.beta_sigma_dU < 1.0)

    def summary(self) -> dict:
        """Flat dictionary suitable for a results table."""
        return {
            "predicted_shift": np.asarray(self.first_order.value).tolist(),
            "predicted_error": np.asarray(self.first_order.error).tolist(),
            "second_order_ratio": np.asarray(self.second_order_ratio).tolist(),
            "beta_sigma_dU": self.beta_sigma_dU,
            "correlation": np.asarray(self.correlation).tolist(),
            "n_samples": self.n_samples,
            "tau": self.tau,
            "trustworthy": self.is_trustworthy,
        }


@dataclass
class ReweightResult:
    """Exponentially reweighted estimate of an observable under the surrogate.

    Exact to all orders in ``dU``, and therefore the arbiter when the
    perturbative prediction is in doubt -- but only to the extent that the
    effective sample size says the reweighting is supported by the data.

    Attributes
    ----------
    mean:
        ``<A e^{-beta dU}>_0 / <e^{-beta dU}>_0``, the surrogate-ensemble average.
    shift:
        ``mean - <A>_0``, directly comparable to
        :attr:`ResponsePrediction.first_order`.
    ess, ess_fraction, max_weight_fraction:
        Reweighting diagnostics.  ``ess_fraction`` below ~0.1 means the answer
        rests on a small subset of frames; ``max_weight_fraction`` catches the
        case where a single frame dominates while the ESS still looks tolerable.
    free_energy_shift:
        ``-ln<e^{-beta dU}>_0 / beta``, the Zwanzig free-energy difference.
        Included because it is free once the weights exist, and because the
        contrast with the observable shift is itself instructive: the free energy
        responds to the *mean* of the error field where observables respond to
        its covariances.
    """

    mean: Estimate
    shift: Estimate
    ess: float
    ess_fraction: float
    max_weight_fraction: float
    free_energy_shift: float
    n_samples: int
    meta: dict = field(default_factory=dict)

    @property
    def is_trustworthy(self) -> bool:
        """Whether the reweighting is supported by enough distinct samples."""
        return bool(self.ess_fraction > 0.1 and self.max_weight_fraction < 0.1)


# --------------------------------------------------------------------------
# Core estimators
# --------------------------------------------------------------------------


def _prepare(a_samples, du_samples) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate and align observable and error-field samples."""
    a = np.asarray(a_samples, dtype=float)
    du = np.asarray(du_samples, dtype=float).ravel()
    scalar = a.ndim == 1
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"A_samples must be (M,) or (M, K), got {np.shape(a_samples)}")
    if a.shape[0] != du.shape[0]:
        raise ValueError(
            f"A_samples has {a.shape[0]} samples but dU_samples has {du.shape[0]}; "
            "they must be evaluated on the same frames of the same trajectory"
        )
    if a.shape[0] < 4:
        raise ValueError("need at least 4 samples for a response estimate")
    if not np.all(np.isfinite(du)):
        raise ValueError("dU_samples contains non-finite values")
    return a, du, scalar


def predict_shift(
    a_samples,
    du_samples,
    temperature: float,
    *,
    n_resamples: int = 1000,
    block_length: int | None = None,
    seed: int = 0,
) -> ResponsePrediction:
    """First-order prediction of the observable shift caused by an error field.

    Implements ``<A>_U - <A>_0 = -beta Cov_0(A, dU) + O(dU^2)``.

    Parameters
    ----------
    a_samples:
        ``(M,)`` or ``(M, K)`` observable values on ``M`` frames of a
        **reference-ensemble** trajectory.  A vector observable such as ``g(r)``
        is handled componentwise and returns a predicted difference curve.
    du_samples:
        ``(M,)`` values of ``dU = U_surrogate - U_reference`` in eV on the *same*
        frames.  Total energy differences, not per atom -- the derivation is in
        terms of the extensive potential and using per-atom values would rescale
        the answer by ``N``.
    temperature:
        Temperature of the reference ensemble in kelvin.  This must be the
        temperature the samples were drawn at; using a different one silently
        rescales the prediction.
    n_resamples, block_length, seed:
        Passed to :func:`~atomlab.analysis.statistics.block_bootstrap`.  The
        block length defaults to four correlation times of the product series.

    Returns
    -------
    ResponsePrediction

    Notes
    -----
    The covariance is computed with the ``1/(M-1)`` convention; with correlated
    samples this is still unbiased for the covariance itself, and the error bar
    -- which is what the correlation actually damages -- comes from the block
    bootstrap rather than from a sample-count formula.
    """
    a, du, scalar = _prepare(a_samples, du_samples)
    b = inverse_temperature(temperature)
    m, k = a.shape

    joint = np.concatenate([a, du[:, None]], axis=1)

    def first_order_stat(block: np.ndarray) -> np.ndarray:
        aa, dd = block[:, :k], block[:, k]
        aa_c = aa - aa.mean(axis=0, keepdims=True)
        dd_c = dd - dd.mean()
        return -b * (aa_c * dd_c[:, None]).sum(axis=0) / (block.shape[0] - 1)

    def second_order_stat(block: np.ndarray) -> np.ndarray:
        aa, dd = block[:, :k], block[:, k]
        aa_c = aa - aa.mean(axis=0, keepdims=True)
        dd_c = dd - dd.mean()
        return 0.5 * b**2 * (aa_c * (dd_c**2)[:, None]).mean(axis=0)

    # The product series is what carries the statistical error, so its own
    # correlation time sets the block length -- using A's alone underestimates
    # it whenever dU varies slowly, which is exactly the systematic-error case.
    product = (a - a.mean(axis=0, keepdims=True)) * (du - du.mean())[:, None]
    tau = integrated_autocorrelation_time(product)
    if block_length is None:
        block_length = int(np.clip(np.ceil(4.0 * tau), 1, max(1, m // 4)))

    first = block_bootstrap(
        joint, first_order_stat, n_resamples=n_resamples, block_length=block_length, seed=seed
    )
    second = block_bootstrap(
        joint, second_order_stat, n_resamples=max(200, n_resamples // 4),
        block_length=block_length, seed=seed + 1,
    )

    sigma_a = a.std(axis=0, ddof=1)
    sigma_du = float(du.std(ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(
            (sigma_a > 0) & (sigma_du > 0),
            np.asarray(first.value) / (-b * np.maximum(sigma_a, 1e-300) * max(sigma_du, 1e-300)),
            0.0,
        )

    def _unwrap(v):
        arr = np.asarray(v, dtype=float)
        return float(arr.reshape(-1)[0]) if scalar else arr

    return ResponsePrediction(
        first_order=Estimate(
            _unwrap(first.value), _unwrap(first.error), first.n_effective, first.method,
            {"samples": first.extra.get("samples")},
        ),
        second_order=Estimate(
            _unwrap(second.value), _unwrap(second.error), second.n_effective, second.method
        ),
        beta_sigma_dU=b * sigma_du,
        correlation=_unwrap(corr),
        sigma_A=_unwrap(sigma_a),
        sigma_dU=sigma_du,
        n_samples=m,
        tau=tau,
        meta={"temperature": temperature, "block_length": block_length, "direction": "forward"},
    )


def predict_shift_from_surrogate_samples(
    a_samples,
    du_samples,
    temperature: float,
    **kwargs,
) -> ResponsePrediction:
    """Mirror estimator expanding around the surrogate instead of the reference.

    Expanding the same identity around ``U`` rather than ``U0`` gives
    ``<A>_U - <A>_0 = +beta Cov_U(A, dU) + O(dU^2)``, requiring samples from the
    *surrogate* ensemble.  The two estimates use disjoint data and agree only to
    the extent that linear response holds, so their difference is an
    assumption-light diagnostic for its breakdown -- the same logic that
    underlies Bennett's acceptance-ratio method.

    Parameters are as for :func:`predict_shift`, except that the samples must
    come from MD driven by the surrogate.
    """
    pred = predict_shift(a_samples, du_samples, temperature, **kwargs)
    flip = lambda e: Estimate(  # noqa: E731 - local alias keeps the negation legible
        value=-np.asarray(e.value) if np.ndim(e.value) else -float(e.value),
        error=e.error,
        n_effective=e.n_effective,
        method=e.method,
        extra=e.extra,
    )
    pred.first_order = flip(pred.first_order)
    pred.second_order = flip(pred.second_order)
    pred.correlation = -np.asarray(pred.correlation) if np.ndim(pred.correlation) else -pred.correlation
    pred.meta["direction"] = "reverse"
    return pred


def reweight(
    a_samples,
    du_samples,
    temperature: float,
    *,
    n_resamples: int = 500,
    block_length: int | None = None,
    seed: int = 0,
) -> ReweightResult:
    """Exact reweighted estimate ``<A>_U = <A e^{-beta dU}>_0 / <e^{-beta dU}>_0``.

    This is the free-energy-perturbation identity and it is exact for any ``dU``.
    Its weakness is variance, not bias: the weights are exponential in ``dU`` and
    can concentrate on a handful of frames, at which point the estimate is
    precise-looking and meaningless.  The returned diagnostics exist to catch
    that, and callers should check :attr:`ReweightResult.is_trustworthy` before
    using the number as ground truth.

    The weights are computed as ``exp(-beta (dU - min dU))`` and normalised,
    which is algebraically identical to the definition but avoids overflow when
    ``beta*dU`` is large and negative.
    """
    a, du, scalar = _prepare(a_samples, du_samples)
    b = inverse_temperature(temperature)
    m, k = a.shape

    shifted = -b * (du - du.min())
    weights = np.exp(shifted - shifted.max())

    stats = weighted_statistics(a, weights)
    ref_mean = a.mean(axis=0)
    mean = np.atleast_1d(np.asarray(stats["mean"], dtype=float))
    shift = mean - ref_mean

    joint = np.concatenate([a, du[:, None]], axis=1)

    def reweighted_stat(block: np.ndarray) -> np.ndarray:
        aa, dd = block[:, :k], block[:, k]
        s = -b * (dd - dd.min())
        w = np.exp(s - s.max())
        return (w[:, None] * aa).sum(axis=0) / w.sum()

    def shift_stat(block: np.ndarray) -> np.ndarray:
        """The shift, resampled as one quantity rather than two.

        An earlier version gave ``shift`` the bootstrap error of the reweighted
        mean.  That is wrong in both directions and it fails hardest exactly
        where the estimator matters.  When ``beta*dU`` is large the weights
        approach a hard 0/1 exclusion, the reweighted mean of a depleted bin is
        the same number in every resample, and its bootstrap error collapses to
        zero -- while the shift, which subtracts the reference mean, still
        carries all of that mean's uncertainty.  The exact two-particle
        benchmark shows the quoted error falling four orders of magnitude while
        the actual error holds constant, with the Kish effective sample size
        never dropping below 0.76 and so never warning.

        Resampling the difference directly fixes both problems at once: the
        reference term's error is included, and the correlation between the two
        terms is handled by construction because they are computed on the same
        resample.
        """
        return reweighted_stat(block) - block[:, :k].mean(axis=0)

    boot = block_bootstrap(
        joint, reweighted_stat, n_resamples=n_resamples, block_length=block_length, seed=seed
    )
    boot_shift = block_bootstrap(
        joint, shift_stat, n_resamples=n_resamples, block_length=block_length, seed=seed
    )

    # Zwanzig: dF = -ln<exp(-beta dU)>/beta, computed in the shifted frame so the
    # exponential never overflows, then corrected by the shift that was removed.
    log_mean_w = np.log(np.exp(shifted - shifted.max()).mean()) + shifted.max()
    free_energy_shift = float(-(log_mean_w) / b + du.min())

    def _unwrap(v):
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        return float(arr[0]) if scalar else arr

    return ReweightResult(
        mean=Estimate(_unwrap(mean), _unwrap(boot.error), stats["ess"], boot.method),
        shift=Estimate(_unwrap(shift), _unwrap(boot_shift.error), stats["ess"],
                       boot_shift.method),
        ess=stats["ess"],
        ess_fraction=stats["ess_fraction"],
        max_weight_fraction=stats["max_weight_fraction"],
        free_energy_shift=free_energy_shift,
        n_samples=m,
        meta={"temperature": temperature},
    )


def response_diagnostics(a_samples, du_samples, temperature: float, **kwargs) -> dict:
    """Run both estimators and return a comparison table.

    Convenience wrapper used by the experiments: it produces the first-order
    prediction, the exact reweighted value, their difference, and every validity
    flag, in one dictionary suitable for direct serialisation.
    """
    pred = predict_shift(a_samples, du_samples, temperature, **kwargs)
    rw = reweight(a_samples, du_samples, temperature)
    linear = np.asarray(pred.first_order.value, dtype=float)
    exact = np.asarray(rw.shift.value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(exact) > 0, np.abs(linear - exact) / np.abs(exact), np.inf)
    return {
        "linear": pred.summary(),
        "reweighted_shift": exact.tolist(),
        "reweighted_ess_fraction": rw.ess_fraction,
        "reweighted_max_weight_fraction": rw.max_weight_fraction,
        "reweighted_trustworthy": rw.is_trustworthy,
        "linear_vs_reweighted_relative_gap": rel.tolist() if rel.ndim else float(rel),
        "free_energy_shift": rw.free_energy_shift,
    }


# --------------------------------------------------------------------------
# Building dU samples from potentials
# --------------------------------------------------------------------------


def delta_u_samples(trajectory, surrogate, reference) -> np.ndarray:
    """Evaluate ``dU = U_surrogate - U_reference`` on every frame of a trajectory.

    Parameters
    ----------
    trajectory:
        An :class:`~atomlab.types.Trajectory` from **reference** dynamics.
    surrogate, reference:
        :class:`~atomlab.potentials.base.Potential` instances.

    Returns
    -------
    ndarray
        ``(T,)`` total-energy differences in eV.

    Notes
    -----
    If the reference trajectory already logged its potential energy, that is
    reused rather than recomputed -- it is identical by construction and saves
    half the work.  The check is exact equality of the potential object, not of
    the energies, so a mismatch cannot slip through.
    """
    logged = trajectory.scalars.get("potential_energy")
    same_reference = trajectory.info.get("potential") is reference
    out = np.empty(trajectory.n_frames)
    for t in range(trajectory.n_frames):
        cfg = trajectory.frame(t)
        u_ref = float(logged[t]) if (logged is not None and same_reference) else reference.energy(cfg)
        out[t] = surrogate.energy(cfg) - u_ref
    return out


def committee_delta_u(trajectory, models: Sequence) -> np.ndarray:
    """Committee error-field estimates ``dU_i = U_i - mean_j U_j``.

    This is the practical substitute for the oracle ``dU``, which needs a ground
    truth the practitioner does not have.  Each model's deviation from the
    committee consensus stands in for its deviation from the truth.

    Returns
    -------
    ndarray
        ``(n_models, T)`` array of committee-referenced error fields, in eV.

    Notes
    -----
    The substitution is exact only if the committee mean equals the truth.  It
    fails in one specific and predictable direction: error shared by every member
    -- from a common architecture, training set, or inductive bias -- cancels in
    the mean and is invisible here.  The committee predictor therefore
    *under-reports*, never over-reports, and the size of the gap is what
    ``experiments/exp08`` measures with deliberately homogeneous and
    heterogeneous committees.
    """
    if len(models) < 2:
        raise ValueError("a committee needs at least two models")
    energies = np.empty((len(models), trajectory.n_frames))
    for t in range(trajectory.n_frames):
        cfg = trajectory.frame(t)
        for i, model in enumerate(models):
            energies[i, t] = model.energy(cfg)
    return energies - energies.mean(axis=0, keepdims=True)


def committee_response(
    trajectory,
    models: Sequence,
    observable_fn: Callable,
    temperature: float,
    *,
    seed: int = 0,
) -> dict:
    """Ground-truth-free estimate of observable uncertainty across a committee.

    For each model, predicts how far its ``<A>`` sits from the committee
    consensus using ``-beta Cov(A, U_i - mean_j U_j)``.  The spread of those
    predictions is an uncertainty estimate for the observable itself, obtained
    from single-point energy evaluations on an existing reference trajectory and
    **no molecular dynamics per model**.

    Parameters
    ----------
    trajectory:
        Frames to evaluate on.  In the practical setting these come from MD with
        one committee member, not from an unavailable ground truth; the theory
        then describes the shift relative to that member's ensemble, which is
        the quantity a practitioner can actually act on.
    observable_fn:
        Callable mapping a :class:`~atomlab.types.Configuration` to a scalar or
        a ``(K,)`` vector.  Evaluated once per frame and reused across models.
    temperature:
        Kelvin.

    Returns
    -------
    dict
        ``per_model`` maps model name to its predicted shift and correlation;
        ``spread`` is the standard deviation of predicted shifts across the
        committee, the headline uncertainty number; ``max_pairwise_gap`` is the
        largest disagreement between any two members, which is the quantity
        relevant to "could my conclusion have gone the other way".
    """
    a_samples = np.array([np.atleast_1d(observable_fn(cfg)) for cfg in trajectory])
    if a_samples.shape[1] == 1:
        a_samples = a_samples[:, 0]
    du = committee_delta_u(trajectory, models)

    per_model = {}
    shifts = []
    for i, model in enumerate(models):
        pred = predict_shift(a_samples, du[i], temperature, seed=seed + i)
        name = getattr(model, "name", f"model{i}")
        per_model[name] = {
            "predicted_shift": np.asarray(pred.first_order.value).tolist(),
            "predicted_error": np.asarray(pred.first_order.error).tolist(),
            "correlation": np.asarray(pred.correlation).tolist(),
            "beta_sigma_dU": pred.beta_sigma_dU,
            "second_order_ratio": np.asarray(pred.second_order_ratio).tolist(),
        }
        shifts.append(np.atleast_1d(np.asarray(pred.first_order.value, dtype=float)))

    stack = np.stack(shifts, axis=0)
    spread = stack.std(axis=0, ddof=1)
    pairwise = np.abs(stack[:, None, :] - stack[None, :, :]).max(axis=(0, 1))
    return {
        "per_model": per_model,
        "spread": spread.tolist() if spread.size > 1 else float(spread),
        "max_pairwise_gap": pairwise.tolist() if pairwise.size > 1 else float(pairwise),
        "n_models": len(models),
        "n_frames": trajectory.n_frames,
    }


# --------------------------------------------------------------------------
# The frequency argument, made measurable
# --------------------------------------------------------------------------


def spectral_decomposition(
    du_samples,
    a_samples,
    force_error_rms: float,
    temperature: float,
) -> dict:
    """Decompose the Cauchy-Schwarz bound into its three factors.

    ``docs/theory.md`` argues that observable error factorises as

        |shift| = beta * sigma(A) * sigma(dU) * |rho(A, dU)|

    and that force RMSE informs only ``sigma(dU)``, and that badly -- it weights
    the error spectrum by ``k^2`` while observables weight it by roughly ``k^0``.
    This helper returns the three factors separately, plus the ratio
    ``sigma(dU) / force_error_rms``, which is the empirical stand-in for the
    inverse spectral weighting: a model whose error is dominated by
    high-frequency wiggle has a *small* ratio (much force error per unit energy
    error), and one whose error is smooth and systematic has a *large* one.

    Tracking that ratio across a model zoo is how the frequency argument becomes
    an observation rather than an assertion.
    """
    a = np.asarray(a_samples, dtype=float)
    du = np.asarray(du_samples, dtype=float).ravel()
    b = inverse_temperature(temperature)
    if a.ndim == 1:
        a = a[:, None]

    sigma_a = a.std(axis=0, ddof=1)
    sigma_du = float(du.std(ddof=1))
    du_c = du - du.mean()
    a_c = a - a.mean(axis=0, keepdims=True)
    # ddof=1 here as well, so that the returned factors multiply back exactly to
    # the covariance rather than differing by O(1/M) -- the factorisation is the
    # point of this function, so it has to be exact.
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (a_c * du_c[:, None]).sum(axis=0) / (a.shape[0] - 1)
        rho = cov / np.maximum(sigma_a * sigma_du, 1e-300)

    smoothness = sigma_du / force_error_rms if force_error_rms > 0 else float("inf")
    scalar = lambda v: v.tolist() if v.size > 1 else v.item()  # noqa: E731
    return {
        "beta": b,
        "sigma_A": scalar(sigma_a),
        "sigma_dU": sigma_du,
        "correlation": scalar(rho),
        "bound": scalar(b * sigma_a * sigma_du),
        "smoothness_ratio": smoothness,
        "force_error_rms": force_error_rms,
    }
