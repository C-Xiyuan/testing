"""Design, uncertainty, and fail-closed regressions for exp09 v3."""

from __future__ import annotations

import numpy as np
import pytest

import experiments.exp09_calibration_replication.run as exp09


def constant_chain_arrays(frame_repeat=1):
    """Small crossed dataset whose only uncertainty is at the chain-unit level."""
    reference_levels = np.array([-1.0, -0.3, 0.4, 1.2])
    direct_levels = np.array([
        [2.0, 2.2, 1.8],
        [5.0, 8.0, 2.0],
    ])
    n_frames = 20 * frame_repeat
    reference_a = np.repeat(reference_levels[:, None], n_frames, axis=1)
    reference_du = np.zeros((4, 2, n_frames))
    direct_a = np.repeat(direct_levels[:, :, None], n_frames, axis=2)
    return reference_a, reference_du, direct_a


def test_v3_uses_a_new_result_name():
    assert exp09.EXPERIMENT_NAME == "exp09_calibration_replication_v3"
    assert exp09.PRODUCTION_MASTER_SEED == 20260812
    assert exp09.ACTUAL_CONSTRUCTION_PANELS == 1


def test_v3_main_entrypoint_records_the_same_protocol_version():
    source = exp09.__loader__.get_source(exp09.__name__)
    assert "protocol_version=3" in source
    assert "protocol_version=2" not in source
    assert '"experiment_version": 3' in source
    assert '"experiment_version": 2' not in source


def test_production_refuses_legacy_or_unregistered_master_seed():
    with pytest.raises(RuntimeError, match="Legacy seed 0 is not an independent rerun"):
        exp09.validate_master_seed(0, quick_mode=False)
    with pytest.raises(RuntimeError, match="frozen at master seed"):
        exp09.validate_master_seed(17, quick_mode=False)
    exp09.validate_master_seed(exp09.PRODUCTION_MASTER_SEED, quick_mode=False)
    exp09.validate_master_seed(0, quick_mode=True)


def test_direct_seed_is_unique_by_field_and_chain():
    seeds = {
        exp09.field_specific_direct_seed(7, field, chain)
        for field in range(8)
        for chain in range(4)
    }
    assert len(seeds) == 32
    for chain in range(4):
        assert exp09.field_specific_direct_seed(7, 0, chain) != (
            exp09.field_specific_direct_seed(7, 1, chain)
        )
    start_seeds = {
        exp09.field_specific_start_seed(7, field, chain)
        for field in range(8)
        for chain in range(4)
    }
    assert len(start_seeds) == 32
    assert seeds.isdisjoint(start_seeds)

    reference_seeds = {exp09.reference_seed(7, chain) for chain in range(8)}
    reference_start_seeds = {
        exp09.reference_start_seed(7, chain) for chain in range(8)
    }
    assert len(reference_seeds) == len(reference_start_seeds) == 8
    all_groups = (seeds, start_seeds, reference_seeds, reference_start_seeds)
    for index, left in enumerate(all_groups):
        for right in all_groups[index + 1:]:
            assert left.isdisjoint(right)


def test_joint_response_statistic_matches_definition():
    a = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    du = np.vstack([a**2, -0.5 * a])
    mean, first, second = exp09._response_statistics(a, du, 120.0)
    beta = exp09.inverse_temperature(120.0)
    ac = a - a.mean()
    dc = du - du.mean(axis=1, keepdims=True)
    expected_first = -beta * (dc * ac).sum(axis=1) / (a.size - 1)
    expected_second = 0.5 * beta**2 * (dc**2 * ac).mean(axis=1)
    assert mean == pytest.approx(a.mean())
    assert first == pytest.approx(expected_first)
    assert second == pytest.approx(expected_second)


def test_multiway_bootstrap_does_not_turn_frames_into_new_chain_units():
    short = constant_chain_arrays(frame_repeat=1)
    long = constant_chain_arrays(frame_repeat=10)
    result_short = exp09.multiway_calibration_bootstrap(
        *short, 120.0, n_resamples=100, seed=13
    )
    result_long = exp09.multiway_calibration_bootstrap(
        *long, 120.0, n_resamples=100, seed=13
    )
    # Duplicating every within-chain frame adds no independent unit.  For these
    # constant series, the entire uncertainty distribution must be unchanged.
    assert result_long["draws"]["residual_first"] == pytest.approx(
        result_short["draws"]["residual_first"]
    )
    assert result_short["units"]["n_reference_chains"] == 4
    assert result_short["units"]["n_direct_chains_per_field"] == 3
    assert result_short["units"]["cell_count_not_an_n"] == 8


def test_complete_chain_bootstrap_does_not_double_count_iid_chain_noise():
    """Average reported SD tracks the chain-mean sampling SD, not sqrt(2) times it."""
    n_datasets, n_chains, n_frames = 24, 8, 300
    estimated = []
    for dataset in range(n_datasets):
        rng = np.random.default_rng(1000 + dataset)
        reference_a = rng.normal(size=(n_chains, n_frames))
        reference_du = np.zeros((n_chains, 1, n_frames))
        direct_a = rng.normal(size=(1, n_chains, n_frames))
        result = exp09.multiway_calibration_bootstrap(
            reference_a, reference_du, direct_a, 120.0,
            n_resamples=250, seed=2000 + dataset,
        )
        estimated.append(np.std(
            result["draws"]["measured_shift"][:, 0], ddof=1
        ))
    theoretical = np.sqrt(2.0 / (n_chains * n_frames))
    ratio = float(np.mean(estimated) / theoretical)
    assert 0.80 < ratio < 1.15


def test_field_specific_direct_uncertainty_is_not_pooled():
    arrays = constant_chain_arrays()
    result = exp09.multiway_calibration_bootstrap(
        *arrays, 120.0, n_resamples=300, seed=17
    )
    direct_sd = result["draws"]["direct_mean"].std(axis=0, ddof=1)
    # Field 0 has narrow chain-to-chain variation; field 1 is intentionally
    # heteroskedastic.  A pooled SEM would erase this difference.
    assert direct_sd[1] > 8.0 * direct_sd[0]


def test_joint_bootstrap_keeps_reference_mean_prediction_covariance():
    x = np.linspace(-1.0, 1.0, 48)
    reference_a = np.stack([x + level for level in (-0.4, -0.1, 0.2, 0.5)])
    # A nonlinear dU makes the moving-block reference mean and covariance term
    # move together.  They must remain on the same bootstrap draw.
    reference_du = np.stack([
        np.stack([(a + 0.3) ** 2, -(a - 0.2) ** 3]) for a in reference_a
    ])
    direct_a = np.stack([
        np.stack([np.full(48, level) for level in (1.0, 1.2, 0.8)]),
        np.stack([np.full(48, level) for level in (-1.0, -0.7, -1.3)]),
    ])
    result = exp09.multiway_calibration_bootstrap(
        reference_a, reference_du, direct_a, 120.0,
        n_resamples=200, seed=23,
    )
    draws = result["draws"]
    assert draws["residual_first"] == pytest.approx(
        draws["measured_shift"] - draws["predicted_first"]
    )
    assert draws["residual_second"] == pytest.approx(
        draws["measured_shift"]
        - draws["predicted_first"]
        - draws["predicted_second"]
    )
    correlation = np.corrcoef(
        draws["measured_shift"][:, 0], draws["predicted_first"][:, 0]
    )[0, 1]
    assert abs(correlation) > 0.1


def test_summary_is_conditional_on_fixed_fields():
    arrays = constant_chain_arrays()
    result = exp09.multiway_calibration_bootstrap(
        *arrays, 120.0, n_resamples=100, seed=29
    )
    records, panel = exp09.summarise_bootstrap(
        result, ["quiet", "noisy"], level=0.95
    )
    assert [record["field"] for record in records] == ["quiet", "noisy"]
    assert set(panel) == {"first", "second"}
    assert "mean_residual_ci" in panel["first"]
    assert "ols_slope_ci" in panel["second"]


def test_confirmatory_gate_fails_closed_for_one_panel_and_no_bound():
    gate = exp09.evidence_gate(
        n_reference=8,
        n_direct=4,
        n_construction_panels=1,
        minimum_construction_panels=3,
        equivalence_bound=None,
        raw_saved=True,
        stationarity_passed=True,
        quick_mode=False,
    )
    assert gate["exploratory_conditional"]["passed"] is True
    assert gate["confirmatory_calibration"]["passed"] is False
    assert gate["evidence_status"] == "exploratory_conditional"
    assert set(gate["confirmatory_calibration"]["reasons"]) == {
        "fixed_field_panel_allows_conditional_inference_only",
        "single_fixed_field_panel_no_population_inference",
        "no_prospectively_justified_equivalence_bound",
    }


def test_stationarity_is_fail_closed():
    stationary = np.tile([0.0, 1.0], 20)
    result = exp09.series_stationarity(stationary, label="stationary", n_sigma=3.0)
    assert result["passed"] is True
    drifting = np.concatenate([np.zeros(20), np.ones(20)])
    with pytest.raises(RuntimeError, match="half-chain drift"):
        exp09.series_stationarity(drifting, label="drifting", n_sigma=3.0)
