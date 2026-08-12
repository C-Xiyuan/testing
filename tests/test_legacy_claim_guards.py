"""Regressions that keep withdrawn legacy statistics from reappearing."""

from __future__ import annotations

from types import SimpleNamespace

import experiments.exp06_response_validation.run as exp06
import experiments.exp07_designed_counterexamples.run as exp07
from scripts.compute_revision_statistics import legacy_component_se_norm


def test_width_schema_reader_prefers_new_honest_key_and_supports_legacy_deposit():
    assert legacy_component_se_norm({
        "legacy_component_se_norm_noninferential": 1.25,
    }) == 1.25
    assert legacy_component_se_norm({"direct_norm_error": 2.5}) == 2.5
    assert legacy_component_se_norm({
        "legacy_component_se_norm_noninferential": 1.25,
        "direct_norm_error": 99.0,
    }) == 1.25

    import pytest
    with pytest.raises(KeyError, match="component_se_norm"):
        legacy_component_se_norm({})


def test_legacy_exp06_cannot_masquerade_as_clean_production():
    import pytest

    with pytest.raises(RuntimeError, match="legacy exp06 is non-release"):
        exp06.require_versioned_release_replacement(SimpleNamespace(quick=False))
    exp06.require_versioned_release_replacement(SimpleNamespace(quick=True))


def test_width_summary_uses_only_unswitched_points_and_has_no_fake_norm_se():
    records = [
        {"width": w, "direct_norm": d, "linear_norm": l, "force_rms": 0.002}
        for w, d, l in zip(
            [0.10, 0.15, 0.25, 0.40, 0.65, 1.00],
            [2.1, 5.3, 4.9, 7.7, 9.9, 8.4],
            [2.0, 3.0, 4.0, 5.0, 5.5, 5.8],
        )
    ]
    summary = exp06.summarise_widths(records, r0=4.2, cutoff=7.0)
    assert summary["compliant_widths"] == [0.10, 0.15, 0.25, 0.40]
    assert summary["switch_truncated_widths"] == [0.65, 1.00]
    assert summary["norm_uncertainty_available"] is False
    assert "measured_exponent_direct" not in summary
    assert "predicted_exponent" not in summary


def _counterexample_record(name: str, measured_max: float) -> dict:
    return {
        "name": name,
        "force_rms_level": 0.004,
        "target_predicted": 0.0,
        "target_measured": measured_max,
        "target_error": 0.5,
        "target_second_order": 0.0,
        "force_rms_out_of_sample": 0.004,
        "legacy_postselected_max_descriptive": abs(measured_max),
        "predicted_max": abs(measured_max),
    }


def test_counterexample_summary_does_not_recreate_postselected_inference():
    summary = exp07.summarise([
        _counterexample_record("null", -0.2),
        _counterexample_record("aligned", 9.8),
        _counterexample_record("random0", 1.0),
        _counterexample_record("random1", 2.0),
    ])
    level = summary["levels"]["4.0e-03"]
    assert level["curve_max_descriptive"]["selection_adjusted_uncertainty_available"] is False
    assert "aligned_over_null_measured" not in level
    assert "null_measured_within_error" not in level
    assert "measured_max" not in level
