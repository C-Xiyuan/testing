"""The scientific core: response theory, proxy metrics, and correlated statistics."""

from .response import (
    ResponsePrediction,
    ReweightResult,
    committee_delta_u,
    committee_response,
    delta_u_samples,
    predict_shift,
    predict_shift_from_surrogate_samples,
    response_diagnostics,
    reweight,
    spectral_decomposition,
)
from .statistics import (
    Estimate,
    autocorrelation,
    block_bootstrap,
    blocking_analysis,
    confidence_interval,
    effective_sample_size,
    integrated_autocorrelation_time,
    jackknife,
    weighted_statistics,
)

__all__ = [
    "predict_shift",
    "predict_shift_from_surrogate_samples",
    "reweight",
    "response_diagnostics",
    "delta_u_samples",
    "committee_delta_u",
    "committee_response",
    "spectral_decomposition",
    "ResponsePrediction",
    "ReweightResult",
    "Estimate",
    "blocking_analysis",
    "block_bootstrap",
    "jackknife",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "weighted_statistics",
    "confidence_interval",
]
