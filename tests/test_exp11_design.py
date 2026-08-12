"""Experimental-unit and gate regressions for corrected exp11."""

from __future__ import annotations

import numpy as np
import pytest

import experiments.exp11_counterexample_replication.run as exp11


def test_crn_seed_shared_only_by_declared_aligned_null_pair():
    for chain in range(4):
        null_seed, null_pair = exp11.direct_seed(1000, "null", chain)
        aligned_seed, aligned_pair = exp11.direct_seed(1000, "aligned", chain)
        random_seed, random_pair = exp11.direct_seed(1000, "random", chain)
        assert (null_seed, null_pair) == (aligned_seed, aligned_pair)
        assert random_seed != null_seed
        assert random_pair is None
    assert len({exp11.direct_seed(1000, "null", chain)[0]
                for chain in range(4)}) == 4


def test_paired_sem_uses_chain_differences_not_independent_quadrature():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    aligned = null + np.array([10.0, 10.1, 9.9, 10.0])
    paired = exp11.paired_contrast_sem(aligned, null)
    independent = np.sqrt(
        aligned.var(ddof=1) / aligned.size + null.var(ddof=1) / null.size
    )
    assert paired < 0.1
    assert independent > 0.8


def test_force_match_gate_uses_the_full_paired_interval():
    null = np.ones(200)
    rng = np.random.default_rng(19)
    noisy_mse = rng.lognormal(mean=0.0, sigma=0.8, size=200)
    noisy_mse /= noisy_mse.mean()
    tight = exp11.paired_force_match(
        np.linspace(0.99, 1.01, 200) ** 2,
        null, 0.02, seed=3, n_resamples=200,
    )
    noisy = exp11.paired_force_match(
        noisy_mse,
        null, 0.02, seed=3, n_resamples=200,
    )
    assert tight["passed"]
    assert not noisy["passed"]
    assert noisy["ci95"][1] - noisy["ci95"][0] > (
        tight["ci95"][1] - tight["ci95"][0]
    )


def test_force_match_blocking_follows_ratio_influence_slow_mode():
    rng = np.random.default_rng(44)
    n = 1500
    fast = rng.normal(scale=0.05, size=n)
    slow = np.empty(n)
    slow[0] = rng.normal(scale=0.02)
    for index in range(1, n):
        slow[index] = 0.995 * slow[index - 1] + rng.normal(
            scale=0.02 * np.sqrt(1.0 - 0.995**2)
        )
    # Marginals are dominated by shared fast noise; their ratio cancels it and
    # exposes the small, highly correlated slow coordinate.
    aligned = np.exp(fast + slow)
    null = np.exp(fast - slow)
    marginal_tau = exp11.integrated_autocorrelation_time(
        np.column_stack([aligned, null])
    )
    result = exp11.paired_force_match(
        aligned, null, 0.02, seed=6, n_resamples=200
    )
    assert marginal_tau < 2.0
    assert result["maximum_coordinate_tau"] > 10.0
    assert result["block_length"] >= 40


def cluster(index, contrast, *, match=True, random_between=True):
    return {
        "cluster": index,
        "contrast": contrast,
        "contrast_error": 0.2,
        "force_match_passed": match,
        "heldout_null_alignment_passed": True,
        "random_null_covariance_ratio": 3.0,
        "aligned_null_predicted_ratio": 20.0,
        "random_between_null_and_aligned": random_between,
        "fields": {
            "null": {"measured": 0.0},
            "aligned": {"measured": contrast},
            "random": {"measured": contrast / 2},
        },
    }


def test_force_match_failure_is_retained_and_fails_primary_gate():
    clusters = [cluster(index, 10.0 + index / 10, match=index != 2)
                for index in range(8)]
    result = exp11.analyse(clusters, tolerance=0.02, minimum_effect=1.0)
    assert result["n_clusters_analysed"] == 8
    assert result["contrasts"][2] == pytest.approx(10.2)
    assert result["force_match_failures"] == [2]
    assert result["primary_gate_passed"] is False
    assert result["replicates"] is False


def test_random_ordering_failure_is_reported_not_reinterpreted():
    clusters = [cluster(index, 10.0, random_between=index not in {4, 5})
                for index in range(8)]
    result = exp11.analyse(clusters, tolerance=0.02, minimum_effect=1.0)
    assert result["random_ordering_failures"] == [4, 5]
    # It is descriptive and therefore does not silently fail the primary gate.
    assert result["primary_gate_passed"] is True


def test_cluster_is_the_inferential_unit():
    clusters = [cluster(index, value) for index, value in enumerate(
        [9.8, 10.0, 10.2, 10.1, 9.9, 10.3, 10.0, 9.7]
    )]
    result = exp11.analyse(clusters, tolerance=0.02, minimum_effect=1.0)
    expected_sem = np.std(result["contrasts"], ddof=1) / np.sqrt(8)
    assert result["sem"] == pytest.approx(expected_sem)
    assert result["n_clusters_analysed"] == 8


def test_constant_stationary_series_passes_zero_sem_edge_case():
    result = exp11.series_stationarity(np.ones(40), label="constant")
    assert result["half_difference_z"] == 0.0


def test_nonfinite_stationarity_series_fails_closed():
    values = np.ones(40)
    values[5] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        exp11.series_stationarity(values, label="bad")


def test_direct_delta_u_series_retains_every_frame():
    class Field:
        @staticmethod
        def energy(frame):
            return float(frame)

    class Trajectory:
        n_frames = 4

        @staticmethod
        def frame(index):
            return index + 0.25

    assert exp11.delta_u_series(Field(), Trajectory()) == pytest.approx(
        [0.25, 1.25, 2.25, 3.25]
    )


def test_effect_disposition_does_not_accept_absence_from_overlap_with_zero():
    assert exp11.effect_disposition([-0.3, 1.3], 1.0)["status"] == "inconclusive"
    assert exp11.effect_disposition([-0.8, 0.9], 1.0)["status"] == (
        "practically_absent"
    )
    assert exp11.effect_disposition([-2.0, -0.1], 1.0)["status"] == (
        "directionally_contradicted"
    )
    assert exp11.effect_disposition([1.1, 2.0], 1.0)["status"] == "replicated"


def test_legacy_magnitude_overstatement_requires_upper_interval_below_it():
    assert not exp11.effect_disposition([1.1, 12.0], 1.0)[
        "legacy_magnitude_overstated"
    ]
    assert exp11.effect_disposition([1.1, 9.0], 1.0)[
        "legacy_magnitude_overstated"
    ]


def test_failed_heldout_null_manipulation_check_blocks_mechanistic_claim():
    clusters = [cluster(index, 10.0) for index in range(8)]
    clusters[3]["heldout_null_alignment_passed"] = False
    result = exp11.analyse(clusters, tolerance=0.02, minimum_effect=1.0)
    assert result["heldout_null_alignment_failures"] == [3]
    assert result["effect_disposition"] == "not_evaluable"
    assert not result["replicates"]


def test_heldout_alignment_check_fails_when_null_equals_aligned():
    rng = np.random.default_rng(22)
    a = rng.normal(size=400)
    du = a + 0.1 * rng.normal(size=400)
    result = exp11.heldout_null_alignment_check(
        a, du, du, 0.2, seed=4, n_resamples=300
    )
    assert not result["passed"]
    assert result["ci95"][0] > 0.9


def test_heldout_null_gate_blocks_on_covariance_product_correlation():
    rng = np.random.default_rng(41)
    n = 1500
    a = rng.choice([-1.0, 1.0], size=n)
    slow = np.empty(n)
    slow[0] = rng.normal()
    for index in range(1, n):
        slow[index] = 0.99 * slow[index - 1] + rng.normal(scale=np.sqrt(1 - 0.99**2))
    null_du = a * slow
    aligned_du = a
    marginal_tau = exp11.integrated_autocorrelation_time(
        np.column_stack([a, null_du, aligned_du])
    )
    result = exp11.heldout_null_alignment_check(
        a, null_du, aligned_du, 0.2, seed=5, n_resamples=200
    )
    assert marginal_tau < 2.0
    assert result["maximum_coordinate_tau"] > 10.0
    assert result["block_length"] >= 40
