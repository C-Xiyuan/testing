"""BAR and two-state MBAR against a case with a closed-form answer.

The same displaced-harmonic system used in ``test_response.py``, but now with
samples drawn from *both* ensembles, because that is what the bidirectional
estimators need.  With

    U0(x) = k x^2 / 2 ,   U1(x) = k (x - d)^2 / 2 + c

the partition functions differ only by the constant, so

    dF = F1 - F0 = c            exactly, for any d
    <x>_1 - <x>_0 = d           exactly, for any c

which separates the two things being tested: ``c`` exercises the free-energy
root-finding and ``d`` exercises the reweighting.  Getting one right and the
other wrong is a distinguishable failure, which is the point of choosing a case
where they are independent.

The estimators are also checked where they are supposed to fail.  Pushing ``d``
far apart destroys the overlap, and the diagnostics must say so -- an estimator
that returns a confident wrong number when the ensembles do not overlap is worse
than one that returns nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.analysis.fep import (
    bennett_acceptance_ratio,
    mbar_two_state_average,
    overlap_diagnostics,
)
from atomlab.units import beta as inverse_temperature

TEMPERATURE = 120.0
BETA = inverse_temperature(TEMPERATURE)


def draw(k, d, c, n, seed):
    """Samples from both ensembles, plus ``dU = U1 - U0`` on each set."""
    rng = np.random.default_rng(seed)
    width = 1.0 / np.sqrt(BETA * k)
    x0 = rng.normal(0.0, width, size=n)
    x1 = rng.normal(d, width, size=n)

    def du(x):
        return 0.5 * k * (x - d) ** 2 + c - 0.5 * k * x**2

    return x0, du(x0), x1, du(x1)


@pytest.mark.parametrize("c", [0.0, 0.01, -0.02])
def test_bar_recovers_the_free_energy_offset(c):
    k, d = 4.0, 0.05
    _, duf, _, dur = draw(k, d, c, 20000, seed=11)
    est = bennett_acceptance_ratio(duf, dur, TEMPERATURE)
    assert abs(est.value - c) < 4.0 * est.error + 1e-4


def test_bar_is_insensitive_to_the_displacement():
    """dF = c whatever d is; a d-dependent answer would mean a sign error."""
    k, c = 4.0, 0.01
    values = []
    for d in (0.02, 0.05, 0.09):
        _, duf, _, dur = draw(k, d, c, 20000, seed=23)
        values.append(bennett_acceptance_ratio(duf, dur, TEMPERATURE).value)
    assert np.allclose(values, c, atol=2e-3)


def test_bar_error_bar_matches_the_scatter_it_claims_to_describe():
    """The analytic variance is only worth having if it is calibrated.

    Forty independent draws of the same size give an empirical standard
    deviation of the estimate; the analytic error should reproduce it.  Bennett's
    formula assumes independent samples, which is true by construction here --
    the point of the test is the prefactor, not the correlation handling.
    """
    k, d, c = 4.0, 0.05, 0.01
    values, errors = [], []
    for seed in range(40):
        _, duf, _, dur = draw(k, d, c, 2000, seed=1000 + seed)
        est = bennett_acceptance_ratio(duf, dur, TEMPERATURE)
        values.append(est.value)
        errors.append(est.error)
    empirical = float(np.std(values, ddof=1))
    claimed = float(np.mean(errors))
    # sd of an sd from 40 draws is ~11%, so allow a factor of 1.4 either way.
    assert 1 / 1.4 < claimed / empirical < 1.4, (claimed, empirical)


def test_mbar_recovers_the_observable_shift():
    k, d, c = 4.0, 0.05, 0.01
    x0, duf, x1, dur = draw(k, d, c, 20000, seed=37)
    out = mbar_two_state_average(x0, duf, x1, dur, TEMPERATURE, n_resamples=100, seed=5)
    assert abs(out["shift"].value - d) < 4.0 * out["shift"].error + 1e-3
    assert abs(out["reference_average"]) < 0.01
    assert abs(out["surrogate_average"] - d) < 0.01


def test_mbar_beats_one_sided_reweighting_where_it_should():
    """Forward-only reweighting is variance-limited; using both sides is not.

    With the ensembles pushed apart, forward exponential reweighting from the
    reference has to reach into a tail it barely samples.  MBAR sees the same
    region directly in the reverse samples.  The test asserts the accuracy
    ordering rather than a fixed tolerance, because that ordering is the reason
    the bidirectional estimator is in the codebase at all.
    """
    k, d, c = 4.0, 0.18, 0.0
    x0, duf, x1, dur = draw(k, d, c, 20000, seed=41)
    forward_weights = np.exp(-BETA * (duf - duf.min()))
    forward = float(np.sum(forward_weights * x0) / np.sum(forward_weights) - x0.mean())
    out = mbar_two_state_average(x0, duf, x1, dur, TEMPERATURE, n_resamples=60, seed=7)
    assert abs(out["shift"].value - d) < abs(forward - d)


def test_overlap_diagnostics_flag_a_broken_case():
    k, c = 4.0, 0.0
    _, near_f, _, near_r = draw(k, 0.02, c, 4000, seed=53)
    _, far_f, _, far_r = draw(k, 0.60, c, 4000, seed=53)
    near = overlap_diagnostics(near_f, near_r, TEMPERATURE)
    far = overlap_diagnostics(far_f, far_r, TEMPERATURE)
    assert near["work_overlap"] > 0.9
    assert far["work_overlap"] < near["work_overlap"]
    assert far["kish_forward"] < near["kish_forward"]
    assert far["max_weight_forward"] > near["max_weight_forward"]


def test_sign_convention_is_enforced_by_the_answer():
    """Passing the reverse work with a flipped sign must not silently succeed.

    The tell is not that the flipped number is a bit worse.  It is that the
    flipped estimator stops responding to the quantity it is supposed to be
    measuring: the correct answer tracks ``c`` one-for-one, while the flipped
    one barely moves when ``c`` changes by a factor of three.  A user comparing
    two systems would see a stable, confident, meaningless number.
    """
    k, d = 4.0, 0.05
    correct, flipped, errors = [], [], []
    for c in (0.01, 0.03):
        _, duf, _, dur = draw(k, d, c, 8000, seed=61)
        good = bennett_acceptance_ratio(duf, dur, TEMPERATURE)
        bad = bennett_acceptance_ratio(duf, -dur, TEMPERATURE)
        assert abs(good.value - c) < 4.0 * good.error + 1e-4
        correct.append(good.value)
        flipped.append(bad.value)
        errors.append(good.error)
    assert abs((correct[1] - correct[0]) - 0.02) < 10.0 * max(errors)
    assert abs(flipped[1] - flipped[0]) < 0.1 * 0.02
    assert abs(flipped[1] - 0.03) > 20.0 * max(errors)
