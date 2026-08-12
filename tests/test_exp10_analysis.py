"""Pure statistical gates for the exp10 end-to-end consistency experiment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

import experiments.exp10_endtoend_consistency.run as exp10


def test_critical_values_distinguish_simultaneous_ci_from_tost():
    assert exp10.simultaneous_z(0.05, 8) == pytest.approx(2.73437, rel=1e-4)
    # Bonferroni TOST is one-sided alpha/m, not the simultaneous two-sided CI.
    assert norm.ppf(1 - 0.05 / 8) == pytest.approx(2.49771, rel=1e-4)


def test_equivalence_requires_the_entire_interval_inside_margin():
    equivalent = exp10.equivalence_disposition(0.05, 0.10, 0.5, z=1.96)
    touching = exp10.equivalence_disposition(0.45, 0.10, 0.5, z=1.96)
    different = exp10.equivalence_disposition(0.90, 0.10, 0.5, z=1.96)
    assert equivalent["status"] == "equivalent"
    assert touching["status"] == "inconclusive"
    assert different["status"] == "meaningfully_different"


def test_excluding_zero_but_overlapping_margin_is_inconclusive():
    result = exp10.equivalence_disposition(0.60, 0.20, 0.5, z=1.96)
    assert result["ci"][0] > 0.0
    assert result["lower_abs"] < 0.5 < result["upper_abs"]
    assert result["status"] == "inconclusive"


def test_interval_disposition_uses_asymmetric_bootstrap_limits():
    assert exp10.interval_disposition([-0.20, 0.45], 0.5)["status"] == "equivalent"
    assert exp10.interval_disposition([0.10, 0.70], 0.5)["status"] == "inconclusive"
    assert exp10.interval_disposition([0.55, 1.10], 0.5)["status"] == (
        "meaningfully_different"
    )


def test_independent_arm_bootstrap_does_not_manufacture_index_pairs():
    first = np.array([[0.0], [0.0], [10.0], [10.0]])
    second = np.array([[1.0], [1.0], [11.0], [11.0]])
    samples = exp10.independent_chain_difference_bootstrap(
        first, second, n_resamples=500, seed=9
    )
    # Index pairing would make every draw exactly one. Independent arm
    # resampling retains the between-chain uncertainty.
    assert samples[:, 0].std(ddof=1) > 2.0


def test_legacy_relaxation_numbers_fail_the_correct_gate():
    gap = np.array([0.3966666667, 1.2758333333])
    error = np.array([0.8521951114, 1.0778814741])
    results = [
        exp10.equivalence_disposition(value, se, 0.5, z=1.96)
        for value, se in zip(gap, error)
    ]
    assert all(result["status"] != "equivalent" for result in results)
    assert [result["upper_abs"] for result in results] == pytest.approx(
        [2.067, 3.388], abs=0.002
    )


def test_failed_gate_overrides_surface_agreement():
    assert exp10.overall_disposition(
        arms_converged=False,
        stationarity_passed=True,
        overlap_passed=True,
        all_equivalent=True,
        any_meaningful_difference=False,
    ) == "not_evaluable"


def test_only_hmc_mbar_comparisons_belong_to_primary_family():
    assert exp10.comparison_family("hmc", "mbar") == "primary"
    assert exp10.comparison_family("hmc", "linear") == "secondary"
    assert exp10.comparison_family("metropolis", "mbar") == "secondary"


def _chains(repeat=1):
    x = np.linspace(-1.0, 1.0, 40)
    beta = exp10.inverse_temperature(120.0)
    refs, surrogate = [], []
    for index, level in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
        a = level + x
        # Within each chain, -beta*Cov(A,dU) is approximately ``level``.
        du = -level * x / (beta * np.var(x, ddof=1))
        refs.append({"a": np.repeat(a[:, None], repeat, axis=0),
                     "du": np.repeat(du, repeat)})
        y = np.full_like(x, 0.25)
        surrogate.append({"a": np.repeat(y[:, None], repeat, axis=0),
                          "du": np.zeros(x.size * repeat)})
    return refs, surrogate


def test_joint_bootstrap_keeps_shared_reference_covariance(monkeypatch):
    # MBAR arithmetic is not the target of this test; replace its point estimate
    # so the test isolates the chain resampling and shared-reference covariance.
    def fake_mbar(a_ref, _du_ref, a_sur, _du_sur, _temperature, **_kwargs):
        shift = np.asarray(a_sur).mean(axis=0) - np.asarray(a_ref).mean(axis=0)
        return {"shift": SimpleNamespace(value=shift)}

    monkeypatch.setattr(exp10, "mbar_two_state_average", fake_mbar)
    refs, surrogate = _chains()
    result = exp10.joint_chain_bootstrap(
        refs, surrogate, 120.0, 0.0, n_resamples=200, seed=7
    )
    # Direct contains -reference mean while the prediction follows the reference
    # chain level, so the covariance sign must be negative, not assumed positive.
    assert result["correlation"]["direct_linear"][0] < -0.9


def test_duplicating_frames_does_not_create_new_chain_units(monkeypatch):
    def fake_mbar(a_ref, _du_ref, a_sur, _du_sur, _temperature, **_kwargs):
        shift = np.asarray(a_sur).mean(axis=0) - np.asarray(a_ref).mean(axis=0)
        return {"shift": SimpleNamespace(value=shift)}

    monkeypatch.setattr(exp10, "mbar_two_state_average", fake_mbar)
    refs, surrogate = _chains(repeat=1)
    long_refs, long_surrogate = _chains(repeat=10)
    short = exp10.joint_chain_bootstrap(
        refs, surrogate, 120.0, 0.0, n_resamples=200, seed=11
    )
    long = exp10.joint_chain_bootstrap(
        long_refs, long_surrogate, 120.0, 0.0, n_resamples=200, seed=11
    )
    assert long["sem"]["direct"] == pytest.approx(short["sem"]["direct"])


def test_chain_weight_diagnostic_detects_one_chain_dominance():
    balanced = [{"du": np.zeros(100)} for _ in range(8)]
    dominated = [{"du": np.zeros(100)} for _ in range(7)] + [
        {"du": np.full(100, -1.0)}
    ]
    good = exp10.chain_weight_diagnostics(
        balanced, 120.0, sign=-1.0
    )
    bad = exp10.chain_weight_diagnostics(
        dominated, 120.0, sign=-1.0
    )
    assert good["max_chain_weight_fraction"] == pytest.approx(1 / 8)
    assert bad["max_chain_weight_fraction"] > 0.99
    assert bad["autocorrelation_adjusted_ess_fraction"] < (
        good["autocorrelation_adjusted_ess_fraction"]
    )


def test_robust_overlap_rejects_reciprocal_range_outliers():
    # The pooled min/max ranges overlap because of one reciprocal outlier, but
    # essentially none of the central mass or independent chain pairs do.
    refs = [{"du": np.r_[np.zeros(99), 10.0]} for _ in range(4)]
    surrogate = [{"du": np.r_[np.full(99, 10.0), 0.0]} for _ in range(4)]
    diagnostic = exp10.robust_work_overlap(refs, surrogate)
    assert diagnostic["central_mutual_frame_fraction"] < 0.05
    assert diagnostic["chain_pair_overlap_fraction"] == 0.0


def test_robust_overlap_accepts_matching_central_work_distributions():
    x = np.linspace(-1.0, 1.0, 100)
    refs = [{"du": x + 0.01 * index} for index in range(4)]
    surrogate = [{"du": x - 0.01 * index} for index in range(4)]
    diagnostic = exp10.robust_work_overlap(refs, surrogate)
    assert diagnostic["central_mutual_frame_fraction"] > 0.8
    assert diagnostic["chain_pair_overlap_fraction"] == 1.0


def test_constant_stationary_series_is_not_rejected_for_zero_sem():
    result = exp10._split_series_diagnostic(np.ones(40))
    assert result["passed"]
    assert result["half_difference_z"] == [0.0]


def test_nonfinite_stationarity_series_fails_closed():
    values = np.ones(40)
    values[3] = np.nan
    assert not exp10._split_series_diagnostic(values)["passed"]
