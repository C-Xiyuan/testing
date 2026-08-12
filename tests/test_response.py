"""Validation of the linear-response estimators against exactly solvable cases.

The value of this test file is that the answers are known in closed form, so a
disagreement is unambiguously a bug in the estimator rather than a sampling
artefact.  The workhorse is a one-dimensional harmonic reference perturbed by a
linear field, for which every order of the cumulant expansion can be written
down:

    U0(x) = k x^2 / 2 ,  dU(x) = eps * x
    U(x)  = k (x + eps/k)^2 / 2 - eps^2 / 2k

so the surrogate ensemble is the reference shifted by ``-eps/k`` with the same
width.  Hence, exactly,

    <x>_U  - <x>_0  = -eps/k          (pure first order; higher orders vanish)
    <x^2>_U - <x^2>_0 = eps^2/k^2     (pure second order; first order vanishes)

Those two cases isolate the two terms of the expansion cleanly, which is exactly
what is needed to tell a wrong prefactor from a wrong sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.analysis.response import (
    predict_shift,
    predict_shift_from_surrogate_samples,
    response_diagnostics,
    reweight,
    spectral_decomposition,
)
from atomlab.analysis.statistics import (
    autocorrelation,
    block_bootstrap,
    blocking_analysis,
    confidence_interval,
    effective_sample_size,
    integrated_autocorrelation_time,
    jackknife,
    weighted_statistics,
)
from atomlab.units import KB
from atomlab.units import beta as inverse_temperature

TEMPERATURE = 300.0
BETA = 1.0 / (KB * TEMPERATURE)


def harmonic_samples(k: float, n: int, seed: int) -> np.ndarray:
    """Draw independent samples from the 1-D harmonic reference ensemble."""
    sigma = np.sqrt(1.0 / (BETA * k))
    return np.random.default_rng(seed).normal(0.0, sigma, size=n)


class TestFirstOrderExact:
    """A = x, dU = eps*x: the expansion terminates at first order."""

    def test_predicted_shift_matches_closed_form(self):
        k, eps, n = 1.0, 0.01, 400_000
        x = harmonic_samples(k, n, seed=1)
        pred = predict_shift(x, eps * x, TEMPERATURE, n_resamples=200)

        exact = -eps / k
        assert pred.first_order.value == pytest.approx(exact, rel=0.02)
        # The estimator must also know how confident to be: the exact answer
        # should sit within a few sigma of the prediction.
        assert abs(pred.first_order.value - exact) < 4.0 * pred.first_order.error

    def test_surrogate_ensemble_endpoint_has_same_first_order_sign(self):
        """Backward-endpoint expansion estimates U minus U0, not its negative."""
        k, eps, n = 1.0, 0.01, 400_000
        sigma = np.sqrt(1.0 / (BETA * k))
        x_u = np.random.default_rng(111).normal(-eps / k, sigma, size=n)
        pred = predict_shift_from_surrogate_samples(
            x_u, eps * x_u, TEMPERATURE, n_resamples=200, seed=8
        )
        assert pred.first_order.value == pytest.approx(-eps / k, rel=0.02)
        assert pred.correlation > 0.99

    def test_second_order_term_vanishes_by_symmetry(self):
        k, eps, n = 1.0, 0.01, 400_000
        x = harmonic_samples(k, n, seed=2)
        pred = predict_shift(x, eps * x, TEMPERATURE, n_resamples=200)
        # <x~ dU~^2> involves <x^3> = 0 for a Gaussian, so the second-order
        # correction must be zero up to sampling noise -- and therefore tiny
        # compared with the first-order term.
        assert pred.second_order_ratio < 0.05
        assert pred.passes_unvalidated_linearity_screen

    def test_scales_linearly_in_perturbation_amplitude(self):
        k, n = 1.0, 200_000
        x = harmonic_samples(k, n, seed=3)
        amplitudes = np.array([0.002, 0.005, 0.01, 0.02])
        shifts = [predict_shift(x, a * x, TEMPERATURE, n_resamples=100).value for a in amplitudes]
        ratios = np.array(shifts) / (-amplitudes / k)
        assert np.allclose(ratios, 1.0, rtol=0.03)

    def test_sign_convention(self):
        """A positive dU correlated with A must *decrease* <A>."""
        x = harmonic_samples(1.0, 100_000, seed=4)
        pred = predict_shift(x, 0.05 * x, TEMPERATURE, n_resamples=100)
        assert pred.value < 0
        assert pred.correlation == pytest.approx(1.0, abs=0.01)


class TestSecondOrderExact:
    """A = x^2, dU = eps*x: first order vanishes, second order is the answer."""

    def test_first_order_vanishes(self):
        k, eps, n = 1.0, 0.05, 400_000
        x = harmonic_samples(k, n, seed=5)
        pred = predict_shift(x**2, eps * x, TEMPERATURE, n_resamples=200)
        sigma2 = 1.0 / (BETA * k)
        # Compare against the scale of the observable itself, not to zero.
        assert abs(pred.value) < 0.02 * sigma2

    def test_second_order_matches_closed_form(self):
        k, eps, n = 1.0, 0.05, 400_000
        x = harmonic_samples(k, n, seed=6)
        pred = predict_shift(x**2, eps * x, TEMPERATURE, n_resamples=200)
        # <x^2>_U - <x^2>_0 = eps^2/k^2, and the cumulant expansion says this
        # is (beta^2/2)<x~^2 dU~^2> = beta^2 eps^2 sigma^4 = eps^2/k^2.
        exact = eps**2 / k**2
        assert pred.second_order.value == pytest.approx(exact, rel=0.05)

    def test_reweighting_recovers_the_full_answer(self):
        k, eps, n = 1.0, 0.05, 400_000
        x = harmonic_samples(k, n, seed=7)
        rw = reweight(x**2, eps * x, TEMPERATURE, n_resamples=100)
        assert rw.shift.value == pytest.approx(eps**2 / k**2, rel=0.10)
        assert rw.passes_weight_concentration_screen


class TestReweighting:
    def test_matches_first_order_for_small_perturbations(self):
        x = harmonic_samples(1.0, 200_000, seed=8)
        du = 0.005 * x
        linear = predict_shift(x, du, TEMPERATURE, n_resamples=100).value
        exact = reweight(x, du, TEMPERATURE, n_resamples=100).shift.value
        assert linear == pytest.approx(exact, rel=0.05)

    def test_free_energy_shift_matches_closed_form(self):
        """dF = -eps^2/(2k) for a linear perturbation of a harmonic well."""
        k, eps = 1.0, 0.05
        x = harmonic_samples(k, 400_000, seed=9)
        rw = reweight(x, eps * x, TEMPERATURE, n_resamples=50)
        assert rw.free_energy_shift == pytest.approx(-(eps**2) / (2 * k), rel=0.05, abs=1e-5)

    def test_effective_sample_size_collapses_for_large_perturbations(self):
        """The diagnostic must fire before the estimate becomes nonsense."""
        x = harmonic_samples(1.0, 20_000, seed=10)
        small = reweight(x, 0.002 * x, TEMPERATURE, n_resamples=20)
        huge = reweight(x, 2.0 * x, TEMPERATURE, n_resamples=20)
        assert small.ess_fraction > 0.9
        assert huge.ess_fraction < 0.1
        assert not huge.passes_weight_concentration_screen

    def test_no_overflow_for_large_negative_du(self):
        x = harmonic_samples(1.0, 5_000, seed=11)
        rw = reweight(x, -5.0 * x, TEMPERATURE, n_resamples=10)
        assert np.isfinite(rw.mean.value)
        assert np.isfinite(rw.free_energy_shift)


class TestVectorObservables:
    def test_componentwise_prediction(self):
        """A vector observable is handled componentwise with the right shapes."""
        k, eps, n = 1.0, 0.01, 200_000
        x = harmonic_samples(k, n, seed=12)
        a = np.stack([x, x**2, np.ones_like(x)], axis=1)
        pred = predict_shift(a, eps * x, TEMPERATURE, n_resamples=100)

        assert np.shape(pred.value) == (3,)
        assert pred.value[0] == pytest.approx(-eps / k, rel=0.03)
        assert abs(pred.value[1]) < 0.02 / (BETA * k)   # odd moment, vanishes
        assert pred.value[2] == pytest.approx(0.0, abs=1e-12)  # constant observable

    def test_constant_observable_is_unmoved(self):
        """A perturbation cannot shift an observable that does not vary."""
        x = harmonic_samples(1.0, 10_000, seed=13)
        pred = predict_shift(np.full_like(x, 7.0), 3.0 * x, TEMPERATURE, n_resamples=50)
        assert pred.value == pytest.approx(0.0, abs=1e-12)


class TestOrthogonality:
    """The central claim: magnitude does not determine damage."""

    def test_uncorrelated_error_field_does_no_damage(self):
        """A large dU statistically independent of A shifts <A> not at all."""
        rng = np.random.default_rng(14)
        n = 200_000
        x = harmonic_samples(1.0, n, seed=15)
        independent = rng.normal(0.0, 0.05, size=n)   # 5x the size of the aligned case below

        big_but_orthogonal = predict_shift(x, independent, TEMPERATURE, n_resamples=200)
        small_but_aligned = predict_shift(x, 0.01 * x, TEMPERATURE, n_resamples=200)

        assert big_but_orthogonal.sigma_dU > 3 * small_but_aligned.sigma_dU
        assert abs(big_but_orthogonal.value) < 0.1 * abs(small_but_aligned.value)

    def test_cauchy_schwarz_factorisation(self):
        x = harmonic_samples(1.0, 100_000, seed=16)
        du = 0.01 * x + 0.02 * np.random.default_rng(17).normal(size=x.size)
        pred = predict_shift(x, du, TEMPERATURE, n_resamples=100)
        reconstructed = -BETA * pred.sigma_A * pred.sigma_dU * pred.correlation
        assert pred.value == pytest.approx(reconstructed, rel=1e-6)

    def test_spectral_decomposition_factors_multiply_back(self):
        x = harmonic_samples(1.0, 50_000, seed=18)
        d = spectral_decomposition(0.01 * x, x, force_error_rms=0.01, temperature=TEMPERATURE)
        assert d["bound"] * abs(d["correlation"]) == pytest.approx(abs(-BETA * 0.01 * d["sigma_A"] ** 2), rel=1e-6)
        assert d["smoothness_ratio"] == pytest.approx(d["sigma_dU"] / 0.01, rel=1e-12)


class TestDiagnostics:
    def test_linear_and_reweighted_agree_in_small_analytic_case(self):
        x = harmonic_samples(1.0, 200_000, seed=19)
        diag = response_diagnostics(x, 0.005 * x, TEMPERATURE, n_resamples=100)
        assert diag["linear"]["passes_unvalidated_linearity_screen"]
        assert diag["reweighted_weight_screen_passed"]
        assert diag["linear_vs_reweighted_relative_gap"] < 0.1

    def test_disagreement_is_flagged_for_large_perturbations(self):
        """Where linear response breaks, the diagnostics must say so."""
        x = harmonic_samples(1.0, 100_000, seed=20)
        pred = predict_shift(x**2, 1.5 * x, TEMPERATURE, n_resamples=50)
        assert not pred.passes_unvalidated_linearity_screen

    def test_mismatched_sample_counts_raise(self):
        with pytest.raises(ValueError, match="same frames"):
            predict_shift(np.zeros(10), np.zeros(11), TEMPERATURE)

    def test_nonfinite_du_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            predict_shift(np.zeros(10), np.full(10, np.nan), TEMPERATURE)


class TestStatisticsModule:
    def test_autocorrelation_of_white_noise_decays_immediately(self):
        x = np.random.default_rng(21).normal(size=20_000)
        acf = autocorrelation(x, max_lag=20)
        assert acf[0] == pytest.approx(1.0)
        assert np.all(np.abs(acf[1:]) < 0.05)
        assert integrated_autocorrelation_time(x) == pytest.approx(0.5, abs=0.25)

    def test_autocorrelation_of_ar1_matches_theory(self):
        """AR(1) with coefficient phi has C(t) = phi^t and tau = 1/2 + phi/(1-phi)."""
        phi, n = 0.9, 400_000
        rng = np.random.default_rng(22)
        noise = rng.normal(size=n)
        x = np.empty(n)
        x[0] = noise[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]

        acf = autocorrelation(x, max_lag=30)
        assert acf[1] == pytest.approx(phi, abs=0.02)
        assert acf[10] == pytest.approx(phi**10, abs=0.03)

        tau_exact = 0.5 + phi / (1.0 - phi)
        assert integrated_autocorrelation_time(x) == pytest.approx(tau_exact, rel=0.15)

    def test_blocking_recovers_the_true_error_for_correlated_data(self):
        """The naive standard error is wrong by sqrt(2 tau); blocking must fix it."""
        phi, n = 0.9, 200_000
        rng = np.random.default_rng(23)
        noise = rng.normal(size=n)
        x = np.empty(n)
        x[0] = noise[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]

        naive = x.std(ddof=1) / np.sqrt(n)
        est = blocking_analysis(x)
        tau = 0.5 + phi / (1.0 - phi)
        assert est.error == pytest.approx(naive * np.sqrt(2 * tau), rel=0.3)

    def test_block_bootstrap_agrees_with_blocking_on_a_mean(self):
        x = np.random.default_rng(24).normal(size=50_000)
        boot = block_bootstrap(x, lambda a: a.mean(), n_resamples=400, seed=0)
        assert boot.value == pytest.approx(x.mean(), rel=1e-12)
        assert boot.error == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)), rel=0.15)

    def test_block_bootstrap_preserves_joint_structure_across_columns(self):
        """Rows must be resampled together, or covariances come out wrong."""
        rng = np.random.default_rng(25)
        a = rng.normal(size=20_000)
        b = 0.5 * a + rng.normal(size=20_000)
        joint = np.stack([a, b], axis=1)
        est = block_bootstrap(joint, lambda z: np.cov(z[:, 0], z[:, 1])[0, 1], n_resamples=300, seed=0)
        assert est.value == pytest.approx(0.5, rel=0.1)
        assert est.error < 0.05

    def test_jackknife_agrees_with_bootstrap(self):
        x = np.random.default_rng(26).normal(size=20_000)
        boot = block_bootstrap(x, lambda a: a.mean(), n_resamples=400, seed=0)
        jack = jackknife(x, lambda a: a.mean(), n_blocks=40)
        assert jack.error == pytest.approx(boot.error, rel=0.35)

    def test_effective_sample_size_falls_with_correlation(self):
        rng = np.random.default_rng(27)
        independent = rng.normal(size=20_000)
        smoothed = np.convolve(independent, np.ones(50) / 50, mode="same")
        assert effective_sample_size(independent) > 5_000
        assert effective_sample_size(smoothed) < effective_sample_size(independent) / 5

    def test_weighted_statistics_ess(self):
        w = np.ones(1000)
        assert weighted_statistics(np.arange(1000.0), w)["ess"] == pytest.approx(1000.0)
        spike = np.zeros(1000)
        spike[0] = 1.0
        stats = weighted_statistics(np.arange(1000.0), spike)
        assert stats["ess"] == pytest.approx(1.0)
        assert stats["max_weight_fraction"] == pytest.approx(1.0)

    def test_confidence_interval_brackets_the_truth(self):
        samples = np.random.default_rng(28).normal(5.0, 1.0, size=10_000)
        lo, hi = confidence_interval(samples, level=0.95)
        assert lo < 5.0 < hi
        assert hi - lo == pytest.approx(2 * 1.96, rel=0.1)


def test_reweight_shift_error_does_not_collapse_as_weights_harden():
    """The reweighted shift's error must not vanish when the weights become 0/1.

    Found by the exact two-particle benchmark, where the quoted error on the
    reweighted shift fell four orders of magnitude across an amplitude sweep
    while the actual error held constant, and the Kish effective sample size
    never dropped below 0.76 -- so neither the error bar nor the standard
    diagnostic gave any warning.

    The mechanism is specific and reproducible.  As ``beta*dU`` grows on the
    frames that carry the observable, the reweighted mean of the depleted
    component becomes the same number in every bootstrap resample and its own
    error genuinely does go to zero.  The *shift* subtracts the reference mean,
    so it still carries all of that mean's uncertainty -- and an earlier version
    of :func:`reweight` handed the shift the reweighted mean's error instead of
    its own.

    The limiting behaviour is what makes this checkable without a tolerance
    pulled from the air: with the weights excluding a component entirely, the
    shift is ``0 - <A>_0``, so its error must converge to the error on ``<A>_0``.
    """
    rng = np.random.default_rng(0)
    n = 4000
    hit = rng.random(n) < 0.25
    a = np.stack([hit.astype(float), (~hit).astype(float)], axis=1)
    beta = inverse_temperature(TEMPERATURE)

    errors = []
    for reduced in (0.05, 0.5, 2.0, 8.0):
        du = np.where(hit, reduced / beta, 0.0)
        result = reweight(a, du, TEMPERATURE, n_resamples=200, seed=1)
        errors.append(float(np.asarray(result.shift.error)[0]))
        # The diagnostics stay healthy throughout, which is the point: they
        # cannot be relied on to catch this.
        assert result.ess_fraction > 0.7

    assert errors == sorted(errors), errors
    plain = block_bootstrap(a, lambda x: x.mean(axis=0), n_resamples=200, seed=1)
    plain_error = float(np.asarray(plain.error)[0])
    assert abs(errors[-1] - plain_error) < 0.25 * plain_error, (errors[-1], plain_error)


def test_reweight_blocking_follows_weighted_influence_not_raw_marginals():
    rng = np.random.default_rng(92)
    n = 3000
    a = rng.choice([-1.0, 1.0], size=n)
    slow = np.empty(n)
    slow[0] = rng.normal()
    for index in range(1, n):
        slow[index] = 0.995 * slow[index - 1] + rng.normal(
            scale=np.sqrt(1.0 - 0.995**2)
        )
    du = a * slow * (0.6 / BETA)
    marginal_tau = integrated_autocorrelation_time(np.column_stack([a, du]))
    automatic = reweight(a, du, TEMPERATURE, n_resamples=250, seed=4)
    short = reweight(a, du, TEMPERATURE, n_resamples=250, block_length=3, seed=4)
    assert marginal_tau < 2.0
    assert automatic.meta["maximum_influence_tau"] > 5.0
    assert automatic.meta["block_length"] >= 20
    assert automatic.shift.error > 1.5 * short.shift.error


def test_legacy_trustworthy_aliases_cannot_succeed_silently():
    x = harmonic_samples(1.0, 2000, seed=91)
    prediction = predict_shift(x, 0.01 * x, TEMPERATURE, n_resamples=50, seed=2)
    reweighted = reweight(x, 0.01 * x, TEMPERATURE, n_resamples=50, seed=2)
    with pytest.warns(FutureWarning, match="unvalidated legacy screen"):
        _ = prediction.is_trustworthy
    with pytest.warns(FutureWarning, match="not a trustworthiness certificate"):
        _ = reweighted.is_trustworthy
    assert "is_trustworthy" not in prediction.summary()
