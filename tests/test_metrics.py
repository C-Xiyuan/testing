"""Validation of the proxy-metric zoo and the rank-correlation analysis.

Two very different kinds of claim are checked here, and both need to be right
before the correlation study in ``experiments/exp05`` means anything.

**The metrics must be the quantities they claim to be.**  The load-bearing case
is curvature: :func:`atomlab.analysis.metrics.hessian` differentiates *analytic
forces*, so its correctness is exactly the question of whether that derivative
is taken correctly.  It is checked against systems whose Hessian is known in
closed form -- an Einstein crystal (``H = k I``, exactly) and a harmonic pair
(a non-trivial ``3x3`` block structure with a known radial/transverse split) --
rather than against another numerical scheme, so a disagreement is unambiguous.

**The statistics must not overclaim.**  The rank estimators are checked against
``scipy.stats`` including on tied data (bootstrap resampling *always* produces
ties, so tie handling is not an edge case here but the normal case), the
Benjamini--Hochberg adjustment against a hand-computed example, and the
bootstrap confidence interval against its own nominal coverage on synthetic
datasets with a known population rank correlation.  That last test is the real
one: an interval whose coverage is unknown is not an interval, and this project
criticises other people for reporting numbers without them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.stats import kendalltau as scipy_kendalltau
from scipy.stats import spearmanr as scipy_spearmanr

from atomlab.analysis.correlation import (
    RankCorrelation,
    benjamini_hochberg,
    kendall_tau,
    proxy_quality_matrix,
    rank_correlation_pvalue,
    rank_correlation_with_ci,
    spearman,
    top_k_agreement,
)
from atomlab.analysis.metrics import (
    DIAGNOSTIC,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRIC_INFO,
    compute_all_metrics,
    describe_metrics,
    distributional_metrics,
    evaluate_pair,
    extrapolation_grade,
    hessian,
    hessian_error,
    mahalanobis_distances,
    metric_direction,
    metric_names,
    smoothness_ratio,
    standard_metrics,
)
from atomlab.build import fcc, rattle
from atomlab.cell import minimum_image
from atomlab.models.base import Descriptor, DescriptorOutput
from atomlab.potentials.harmonic import EinsteinCrystal, HarmonicPair
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.types import Configuration, Dataset

# --------------------------------------------------------------------------
# Shared toy system: LJ argon, perturbed by shell bumps of known width
# --------------------------------------------------------------------------

ARGON_A0 = 5.26  # A, fcc lattice parameter near the triple point
LJ_CUTOFF = 5.0  # A; below half the 10.52 A cell width, as the MIC requires


def argon_reference() -> LennardJones:
    """The analytic ground truth used throughout this file."""
    return LennardJones.argon(cutoff=LJ_CUTOFF)


def argon_dataset(n_configs: int = 12, *, sigma: float = 0.12, seed: int = 3) -> Dataset:
    """``n_configs`` rattled 32-atom fcc argon cells, labelled by nothing.

    Rattle amplitudes vary across the set so that the energy and shortest-bond
    deciles are non-degenerate -- a test set of identical configurations would
    make the distributional metrics vacuous.
    """
    base = fcc(ARGON_A0, "Ar", (2, 2, 2))
    rng = np.random.default_rng(seed)
    configs = [
        rattle(base, sigma * (0.5 + rng.random()), seed=int(rng.integers(1, 10_000)))
        for _ in range(n_configs)
    ]
    return Dataset(configs)


def shell_model(reference: LennardJones, amplitude: float, width: float, r0: float = 3.9):
    """``U_0 + delta_U`` with a Gaussian shell bump of known amplitude and width."""
    return reference + RadialShellPerturbation(r0, width, amplitude, LJ_CUTOFF)


class CoordinationDescriptor(Descriptor):
    """A two-component per-atom descriptor: smooth coordination numbers.

    Deliberately trivial.  The extrapolation grade is a statement about the
    *distribution* of descriptors, so what is being tested is the Mahalanobis
    machinery and the plumbing through the ``Descriptor`` interface, not any
    particular featurisation.  Using ACSF or SOAP here would test those modules
    instead, at fifty times the cost.
    """

    def __init__(self, cutoff: float = 5.0):
        self.cutoff = float(cutoff)
        self.name = "coordination"

    @property
    def n_features(self) -> int:
        return 2

    def compute(self, configuration: Configuration, *, derivatives: bool = True) -> DescriptorOutput:
        n = configuration.n_atoms
        d = configuration.positions[:, None, :] - configuration.positions[None, :, :]
        if np.asarray(configuration.pbc).any():
            d = minimum_image(d.reshape(-1, 3), configuration.cell, configuration.pbc).reshape(n, n, 3)
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        inside = r < self.cutoff
        f = np.where(inside, np.exp(-((r / 3.0) ** 2)), 0.0).sum(axis=1)
        g = np.where(inside, 1.0 / np.maximum(r, 1e-12), 0.0).sum(axis=1)
        return DescriptorOutput(features=np.stack([f, g], axis=1))


# --------------------------------------------------------------------------
# Rank estimators against scipy, with and without ties
# --------------------------------------------------------------------------


class TestRankEstimatorsAgainstScipy:
    """Our direct implementations must reproduce scipy exactly, ties included."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_spearman_matches_scipy_continuous(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=30)
        y = 0.6 * x + rng.normal(size=30)
        assert spearman(x, y) == pytest.approx(scipy_spearmanr(x, y).statistic, abs=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_kendall_matches_scipy_continuous(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=25)
        y = -0.4 * x + rng.normal(size=25)
        assert kendall_tau(x, y) == pytest.approx(
            scipy_kendalltau(x, y, variant="b").statistic, abs=1e-12
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_both_match_scipy_with_heavy_ties(self, seed):
        """Integer-valued data with many repeats: the midrank / tau-b conventions.

        This is the case that matters in practice, because every bootstrap
        resample duplicates models and is therefore tied by construction.
        """
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 4, size=40).astype(float)
        y = rng.integers(0, 3, size=40).astype(float)
        assert spearman(x, y) == pytest.approx(scipy_spearmanr(x, y).statistic, abs=1e-12)
        assert kendall_tau(x, y) == pytest.approx(
            scipy_kendalltau(x, y, variant="b").statistic, abs=1e-12
        )

    def test_partial_ties_preserve_perfect_monotonicity(self):
        """A monotone relation with a tie in *both* variables still gives 1."""
        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
        assert spearman(x, y) == pytest.approx(1.0)
        assert kendall_tau(x, y) == pytest.approx(1.0)

    def test_constant_variable_is_undefined_not_zero(self):
        """A constant has no ranking; the correlation is NaN, never 0."""
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        assert np.isnan(spearman(x, y))
        assert np.isnan(kendall_tau(x, y))

    def test_non_finite_pairs_are_dropped(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
        assert spearman(x, y) == pytest.approx(1.0)


class TestPerfectAndNullRelations:
    """The three anchor cases every rank correlation must get right."""

    def test_monotone_relation_gives_plus_one(self):
        x = np.linspace(0.0, 1.0, 20)
        y = np.exp(3.0 * x)  # strictly increasing but strongly non-linear
        est = rank_correlation_with_ci(x, y, n_bootstrap=500, seed=0)
        assert est.value == pytest.approx(1.0)
        assert est.ci_low == pytest.approx(1.0)
        assert est.ci_high == pytest.approx(1.0)
        assert not est.brackets_zero

    def test_reversed_relation_gives_minus_one(self):
        x = np.linspace(0.0, 1.0, 20)
        est = rank_correlation_with_ci(x, -x**3, n_bootstrap=500, seed=0, method="kendall")
        assert est.value == pytest.approx(-1.0)
        assert est.ci_high == pytest.approx(-1.0)

    def test_independent_data_gives_interval_bracketing_zero(self):
        """Over 20 independent draws, ~95% of intervals should contain rho = 0."""
        brackets = 0
        for seed in range(20):
            rng = np.random.default_rng(1000 + seed)
            x, y = rng.normal(size=40), rng.normal(size=40)
            est = rank_correlation_with_ci(x, y, n_bootstrap=800, seed=seed)
            brackets += int(est.brackets_zero)
        assert brackets >= 16, f"only {brackets}/20 intervals bracketed zero"

    def test_interval_is_impossible_to_omit(self):
        """The repr of a rank correlation carries its interval and sample size."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=30)
        est = rank_correlation_with_ci(x, 0.5 * x + rng.normal(size=30), n_bootstrap=400, seed=1)
        text = repr(est)
        assert "CI" in text and "n=30" in text
        assert isinstance(est, RankCorrelation)
        assert est.ci_low < est.value < est.ci_high
        assert est.ci_width > 0.0


class TestPValues:
    def test_asymptotic_matches_scipy(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=25)
        y = 0.8 * x + rng.normal(size=25)
        assert rank_correlation_pvalue(x, y) == pytest.approx(scipy_spearmanr(x, y).pvalue)
        assert rank_correlation_pvalue(x, y, method="kendall") == pytest.approx(
            scipy_kendalltau(x, y, variant="b").pvalue
        )

    def test_permutation_pvalue_is_small_for_a_strong_relation_and_never_zero(self):
        rng = np.random.default_rng(3)
        x = np.arange(20.0)
        y = x + rng.normal(size=20)
        p = rank_correlation_pvalue(x, y, p_method="permutation", n_permutations=500, seed=0)
        assert 0.0 < p <= 1.0 / 501.0 * 2

    def test_permutation_pvalue_is_uniform_ish_under_the_null(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=25), rng.normal(size=25)
        p = rank_correlation_pvalue(x, y, p_method="permutation", n_permutations=400, seed=1)
        assert 0.0 < p <= 1.0

    def test_estimate_interface_is_preserved(self):
        """A RankCorrelation must still behave as a statistics.Estimate."""
        from atomlab.analysis.statistics import Estimate

        rng = np.random.default_rng(5)
        x = rng.normal(size=25)
        est = rank_correlation_with_ci(x, 0.9 * x + 0.2 * rng.normal(size=25), n_bootstrap=300)
        assert isinstance(est, Estimate)
        assert est.significantly_differs_from(0.0, n_sigma=2.0)
        assert np.isfinite(est.relative_error)
        assert set(est.to_dict()) >= {"value", "ci_low", "ci_high", "p_value", "brackets_zero"}


class TestBootstrapCoverage:
    """Does the nominal 95% interval actually cover the population value 95% of the time?

    Bivariate normal data with Pearson correlation ``r`` has population Spearman
    ``rho_s = (6/pi) arcsin(r/2)``, an exact identity, so there is a known
    target.  Percentile bootstrap intervals for correlations are expected to
    undercover slightly at ``n = 40``; the test asserts a band rather than
    exactly 0.95 and the measured value is printed so the shortfall is reported
    rather than assumed.
    """

    @staticmethod
    def _coverage(r: float, n: int, n_trials: int, n_bootstrap: int, seed: int) -> tuple[float, float]:
        rho_true = (6.0 / np.pi) * np.arcsin(r / 2.0)
        cov = np.array([[1.0, r], [r, 1.0]])
        chol = np.linalg.cholesky(cov)
        rng = np.random.default_rng(seed)
        hits = 0
        for t in range(n_trials):
            z = rng.normal(size=(n, 2)) @ chol.T
            est = rank_correlation_with_ci(z[:, 0], z[:, 1], n_bootstrap=n_bootstrap, seed=t)
            hits += int(est.ci_low <= rho_true <= est.ci_high)
        return hits / n_trials, rho_true

    def test_coverage_for_moderate_correlation(self):
        coverage, rho_true = self._coverage(0.7, n=40, n_trials=300, n_bootstrap=600, seed=11)
        print(f"\n[coverage] rho_true={rho_true:.4f}  n=40  empirical 95% coverage = {coverage:.3f}")
        assert 0.85 <= coverage <= 1.0

    def test_coverage_under_the_null(self):
        coverage, rho_true = self._coverage(0.0, n=30, n_trials=300, n_bootstrap=600, seed=23)
        print(f"\n[coverage] rho_true={rho_true:.4f}  n=30  empirical 95% coverage = {coverage:.3f}")
        assert 0.88 <= coverage <= 1.0


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_hand_computed_example(self):
        """m = 3, p = [0.001, 0.5, 0.6].

        Raw adjustment ``p * m / i`` gives ``[0.003, 0.75, 0.6]``; the step-up
        monotonicity fix replaces 0.75 by ``min(0.75, 0.6) = 0.6``.  At
        ``alpha = 0.05`` only the first is rejected.
        """
        res = benjamini_hochberg([0.001, 0.5, 0.6], alpha=0.05)
        assert res.adjusted == pytest.approx([0.003, 0.6, 0.6])
        assert res.rejected.tolist() == [True, False, False]
        assert res.n_tests == 3 and res.n_rejected == 1

    def test_hand_computed_example_all_borderline(self):
        """p_i = i/100 for i = 1..5: every adjusted value is exactly 0.05."""
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        res = benjamini_hochberg(p, alpha=0.05)
        assert res.adjusted == pytest.approx([0.05] * 5)
        assert res.rejected.all()

    def test_order_is_preserved_and_unsorted_input_handled(self):
        res = benjamini_hochberg([0.6, 0.001, 0.5], alpha=0.05)
        assert res.adjusted == pytest.approx([0.6, 0.003, 0.6])
        assert res.rejected.tolist() == [False, True, False]

    def test_adjusted_never_below_raw_and_monotone_in_p(self):
        rng = np.random.default_rng(0)
        p = rng.random(50)
        res = benjamini_hochberg(p, alpha=0.1)
        assert np.all(res.adjusted >= p - 1e-12)
        order = np.argsort(p)
        assert np.all(np.diff(res.adjusted[order]) >= -1e-12)

    def test_nan_pvalues_leave_the_family(self):
        res = benjamini_hochberg([0.001, np.nan, 0.5, 0.6], alpha=0.05)
        assert res.n_tests == 3
        assert np.isnan(res.adjusted[1]) and not res.rejected[1]
        assert res.adjusted[0] == pytest.approx(0.003)

    def test_benjamini_yekutieli_is_more_conservative(self):
        p = [0.001, 0.5, 0.6]
        bh = benjamini_hochberg(p, alpha=0.05)
        by = benjamini_hochberg(p, alpha=0.05, dependence="arbitrary")
        # c(3) = 1 + 1/2 + 1/3 = 11/6
        assert by.adjusted[0] == pytest.approx(0.003 * 11.0 / 6.0)
        assert np.all(by.adjusted >= bh.adjusted - 1e-12)


# --------------------------------------------------------------------------
# The proxy-quality table and the decision statistic
# --------------------------------------------------------------------------


def synthetic_zoo(n_models: int = 24, seed: int = 5):
    """A fake model zoo where one metric is informative and one is noise.

    ``metric_good`` is monotonically related to ``observable_A``; ``metric_junk``
    is independent of everything.  This is the structure P1 predicts for real
    zoos (some metric-observable pairs correlate, most do not), so it exercises
    the table in the regime it will be used in.
    """
    rng = np.random.default_rng(seed)
    truth = rng.lognormal(mean=-2.0, sigma=0.6, size=n_models)
    metrics, observables = {}, {}
    for i in range(n_models):
        name = f"model{i:02d}"
        metrics[name] = {
            "metric_good": float(truth[i] * np.exp(0.15 * rng.normal())),
            "metric_junk": float(rng.random()),
        }
        observables[name] = {
            "observable_A": float(truth[i]),
            "observable_B": float(rng.random()),
        }
    return metrics, observables


class TestProxyQualityMatrix:
    def test_shape_names_and_content(self):
        metrics, observables = synthetic_zoo()
        table = proxy_quality_matrix(metrics, observables, n_bootstrap=400, seed=0)
        assert table.shape == (2, 2)
        assert table.metric_names == ["metric_good", "metric_junk"]
        assert table.observable_names == ["observable_A", "observable_B"]
        assert len(table.model_names) == 24
        # the informative pair is strongly correlated, the junk pair is not
        assert table.cell("metric_good", "observable_A")["rho"] > 0.85
        assert abs(table.cell("metric_junk", "observable_B")["rho"]) < 0.5
        assert table.cell("metric_junk", "observable_B")["ci_low"] <= 0.0

    def test_every_cell_carries_an_interval_and_adjusted_p(self):
        metrics, observables = synthetic_zoo()
        table = proxy_quality_matrix(metrics, observables, n_bootstrap=300, seed=1)
        assert np.all(table.ci_low <= table.rho + 1e-12)
        assert np.all(table.ci_high >= table.rho - 1e-12)
        assert np.all(table.p_adjusted >= table.p_raw - 1e-12)
        assert np.all(table.n_models == 24)
        assert "4 (proxy metric, observable) rank correlations" in table.family
        records = table.to_records()
        assert len(records) == 4 and set(records[0]) >= {"rho", "ci_low", "p_adjusted"}
        assert table.to_dataframe().shape[0] == 4
        assert "significant after adjustment" in table.summary()

    def test_missing_metric_values_shrink_the_sample_not_the_table(self):
        metrics, observables = synthetic_zoo(n_models=20, seed=9)
        metrics["model00"]["metric_good"] = float("nan")
        table = proxy_quality_matrix(metrics, observables, n_bootstrap=200, seed=2)
        assert table.shape == (2, 2)
        assert table.n_models[0, 0] == 19
        assert table.n_models[1, 0] == 20

    def test_kendall_variant_runs_and_agrees_in_sign(self):
        metrics, observables = synthetic_zoo()
        rho_table = proxy_quality_matrix(metrics, observables, n_bootstrap=200, seed=0)
        tau_table = proxy_quality_matrix(
            metrics, observables, n_bootstrap=200, seed=0, method="kendall"
        )
        assert tau_table.method == "kendall"
        assert np.sign(tau_table.rho[0, 0]) == np.sign(rho_table.rho[0, 0])
        assert abs(tau_table.rho[0, 0]) <= abs(rho_table.rho[0, 0]) + 1e-9


class TestTopKAgreement:
    def test_perfect_proxy_selects_the_same_models(self):
        truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = top_k_agreement(truth * 2.0, truth, k=3)
        assert result.fraction == 1.0
        assert result.regret == 0.0
        assert result.expected_by_chance == pytest.approx(0.5)

    def test_reversed_proxy_selects_the_worst_models(self):
        truth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = top_k_agreement(-truth, truth, k=2)
        assert result.fraction == 0.0
        assert result.selected_by_proxy == [5, 4]
        assert result.regret == pytest.approx(0.5)
        assert result.relative_regret == pytest.approx(5.0)

    def test_overlap_is_a_set_but_regret_is_about_the_single_pick(self):
        """Top-2 sets can agree completely while the *chosen* model is not the
        best one: the set overlap is 1.0 and the regret is not zero.  That gap
        is the reason both numbers are reported."""
        proxy = np.array([1.0, 2.0, 3.0, 4.0])
        truth = np.array([2.0, 1.0, 4.0, 3.0])  # same top-2 set, order swapped
        result = top_k_agreement(proxy, truth, k=2)
        assert result.fraction == 1.0
        assert result.n_overlap == 2
        assert result.selected_by_proxy == [0, 1]
        assert result.selected_by_truth == [1, 0]
        assert result.regret == pytest.approx(1.0)  # truth[0] = 2.0 vs best 1.0

    def test_higher_is_better_metric_orientation(self):
        """Force cosine is higher-is-better; passing the flag must invert the pick."""
        cosine = np.array([0.99, 0.95, 0.90, 0.80])
        error = np.array([0.01, 0.02, 0.03, 0.04])
        good = top_k_agreement(cosine, error, k=2, metric_lower_is_better=False)
        bad = top_k_agreement(cosine, error, k=2, metric_lower_is_better=True)
        assert good.fraction == 1.0 and bad.fraction == 0.0

    def test_bootstrap_interval_on_the_overlap(self):
        rng = np.random.default_rng(4)
        truth = rng.random(20)
        proxy = truth + 0.3 * rng.normal(size=20)
        result = top_k_agreement(proxy, truth, k=5, n_bootstrap=400, seed=0)
        assert 0.0 <= result.ci_low <= result.fraction + 1e-12
        assert result.ci_high <= 1.0
        assert "chance" in result.text()

    def test_k_must_be_smaller_than_the_zoo(self):
        with pytest.raises(ValueError):
            top_k_agreement(np.arange(5.0), np.arange(5.0), k=5)


# --------------------------------------------------------------------------
# Curvature: the analytic-derivative test for this module
# --------------------------------------------------------------------------


class TestHessianAgainstClosedForm:
    """The Hessian differentiates analytic forces; check it where the answer is exact."""

    def test_einstein_crystal_hessian_is_exactly_k_times_identity(self):
        rng = np.random.default_rng(0)
        n = 6
        cell = np.eye(3) * 12.0
        sites = rng.random((n, 3)) * 10.0
        k = 0.5 + rng.random(n)  # eV/A^2, one spring per atom
        potential = EinsteinCrystal(sites, k)
        cfg = Configuration(
            positions=sites + 0.05 * rng.normal(size=(n, 3)), cell=cell, pbc=True, symbols=("Ar",)
        )

        numeric = hessian(potential, cfg, delta=1e-4)
        analytic = np.diag(np.repeat(k, 3))
        err = np.max(np.abs(numeric - analytic))
        print(f"\n[hessian] Einstein crystal max|H_fd - H_exact| = {err:.3e} eV/A^2")
        assert err < 1e-8

    def test_harmonic_pair_hessian_matches_the_radial_transverse_split(self):
        """For ``u = k(r-r0)^2/2``:

        ``d2U/dr1 dr1 = u'' n n^T + (u'/r)(I - n n^T)`` and
        ``d2U/dr1 dr2 = -d2U/dr1 dr1``.  The transverse block is proportional to
        ``u'/r``, which vanishes at ``r = r0``; evaluating away from the minimum
        is therefore essential, or the test would pass for a wrong transverse
        term.
        """
        k, r0 = 3.0, 2.0
        r = 2.4  # deliberately not r0
        positions = np.array([[0.0, 0.0, 0.0], [r * 0.6, r * 0.8, 0.0]])
        cfg = Configuration(positions=positions, cell=np.eye(3) * 30.0, pbc=False, symbols=("Ar",))
        potential = HarmonicPair(k, r0, cutoff=5.0)

        d = positions[0] - positions[1]
        nhat = d / np.linalg.norm(d)
        proj = np.outer(nhat, nhat)
        block = k * proj + (k * (r - r0) / r) * (np.eye(3) - proj)
        analytic = np.block([[block, -block], [-block, block]])

        numeric = hessian(potential, cfg, delta=1e-4)
        err = np.max(np.abs(numeric - analytic))
        print(f"[hessian] harmonic pair max|H_fd - H_exact| = {err:.3e} eV/A^2")
        assert err < 1e-7

    def test_hessian_is_symmetric_and_has_three_zero_modes_for_a_pair_potential(self):
        reference = argon_reference()
        cfg = rattle(fcc(ARGON_A0, "Ar", (2, 2, 2)), 0.05, seed=1)
        h = hessian(reference, cfg, delta=1e-4)
        assert np.max(np.abs(h - h.T)) < 1e-10
        evals = np.linalg.eigvalsh(h)
        # translational invariance: three exactly-zero modes, tiny at finite delta
        assert np.max(np.abs(evals[:3])) < 1e-6
        assert evals[3] > 1e-3

    def test_hessian_error_is_zero_for_a_model_equal_to_the_reference(self):
        reference = argon_reference()
        cfg = rattle(fcc(ARGON_A0, "Ar", (2, 2, 2)), 0.05, seed=2)
        out = hessian_error(reference, reference, cfg, n_skip_modes=3, n_low_modes=6)
        assert out["hessian_frobenius_error"] == pytest.approx(0.0, abs=1e-12)
        assert out["hessian_low_eigenvalue_rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_hessian_error_grows_with_perturbation_amplitude(self):
        reference = argon_reference()
        cfg = rattle(fcc(ARGON_A0, "Ar", (2, 2, 2)), 0.05, seed=3)
        small = hessian_error(shell_model(reference, 1e-3, 0.3), reference, cfg)
        large = hessian_error(shell_model(reference, 1e-2, 0.3), reference, cfg)
        assert large["hessian_frobenius_error"] > 5.0 * small["hessian_frobenius_error"]
        assert large["hessian_low_eigenvalue_rmse"] > small["hessian_low_eigenvalue_rmse"]


# --------------------------------------------------------------------------
# The metric zoo on a toy model/reference pair
# --------------------------------------------------------------------------


class TestStandardMetrics:
    def test_identical_model_gives_exactly_zero_error(self):
        reference = argon_reference()
        data = argon_dataset(6)
        evaluation = evaluate_pair(reference, reference, data)
        std = standard_metrics(evaluation)
        for key in ("energy_mae", "energy_rmse", "force_rmse", "force_mae", "virial_rmse", "stress_rmse"):
            assert std[key] == pytest.approx(0.0, abs=1e-12), key
        assert std["force_cosine"] == pytest.approx(1.0, abs=1e-12)

    def test_energy_offset_is_visible_raw_and_removed_by_the_shifted_metric(self):
        """A constant ``dU`` moves no observable (theory sec. 5); the pair of
        energy metrics is what makes that distinction measurable."""

        reference = argon_reference()
        from atomlab.potentials.base import Potential
        from atomlab.types import Result

        class ConstantShift(Potential):
            cutoff = 0.0
            name = "shift"

            def compute(self, configuration, *, forces=True, virial=True):
                return Result(
                    energy=0.05 * configuration.n_atoms,
                    forces=np.zeros((configuration.n_atoms, 3)),
                    virial=np.zeros((3, 3)) if virial else None,
                )

        data = argon_dataset(6)
        evaluation = evaluate_pair(reference + ConstantShift(), reference, data)
        std = standard_metrics(evaluation)
        assert std["energy_mae"] == pytest.approx(0.05, rel=1e-9)
        assert std["energy_rmse_shifted"] == pytest.approx(0.0, abs=1e-12)
        assert std["force_rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_force_vector_rmse_is_exactly_sqrt3_times_the_component_rmse(self):
        reference = argon_reference()
        data = argon_dataset(8)
        evaluation = evaluate_pair(shell_model(reference, 5e-3, 0.4), reference, data)
        std = standard_metrics(evaluation)
        ratio = std["force_vector_rmse"] / std["force_rmse"]
        assert ratio == pytest.approx(np.sqrt(3.0), rel=1e-12)


class TestDistributionalMetrics:
    def test_tail_metrics_order_correctly_and_exceed_the_mean(self):
        reference = argon_reference()
        data = argon_dataset(12)
        evaluation = evaluate_pair(shell_model(reference, 5e-3, 0.35), reference, data)
        std = standard_metrics(evaluation)
        dist = distributional_metrics(evaluation)
        assert std["force_rmse"] < dist["force_error_p95"] < dist["force_error_p99"] <= dist["force_error_max"]

    def test_deciles_select_independently_on_a_constructed_evaluation(self):
        """Energy rank and bond rank are deliberately opposed here.

        In rattled LJ argon the two deciles happen to select the *same*
        configuration -- short bonds are what make a rattled cell expensive --
        so a test on real data cannot distinguish "selected by energy" from
        "selected by bond length".  This constructs a :class:`PairEvaluation`
        by hand with the two orderings reversed, which is the only way to check
        that each metric reads the column it claims to.
        """
        from atomlab.analysis.metrics import PairEvaluation

        n_config = 10
        errors = np.arange(1.0, n_config + 1.0)  # config c has force error c+1
        forces_ref = np.zeros((2 * n_config, 3))
        forces_model = np.repeat(errors, 2)[:, None] * np.ones((1, 3))
        evaluation = PairEvaluation(
            n_atoms=np.full(n_config, 2),
            energy_model=np.zeros(n_config),
            energy_reference=np.arange(n_config, dtype=float) * 2.0,  # hottest = last
            forces_model=forces_model,
            forces_reference=forces_ref,
            config_index=np.repeat(np.arange(n_config), 2),
            volumes=np.full(n_config, 1000.0),
            min_bond=np.arange(1.0, n_config + 1.0),  # shortest bond = first
        )
        dist = distributional_metrics(evaluation, decile=0.1)
        assert dist["force_rmse_short_bond"] == pytest.approx(errors[0])
        assert dist["force_rmse_high_energy"] == pytest.approx(errors[-1])
        assert dist["force_error_max"] == pytest.approx(errors[-1] * np.sqrt(3.0))

    def test_subset_metrics_use_the_intended_configurations(self):
        """The short-bond decile must select the configuration with the smallest
        minimum bond length, and its force RMSE must equal a direct computation
        over exactly those atoms."""
        reference = argon_reference()
        data = argon_dataset(10)
        model = shell_model(reference, 8e-3, 0.3)
        evaluation = evaluate_pair(model, reference, data)
        dist = distributional_metrics(evaluation, decile=0.1)

        worst = int(np.argmin(evaluation.min_bond))
        rows = evaluation.config_index == worst
        direct = np.sqrt((evaluation.delta_forces[rows] ** 2).mean())
        assert dist["force_rmse_short_bond"] == pytest.approx(direct, rel=1e-12)

        hottest = int(np.argmax(evaluation.energy_per_atom_reference))
        rows = evaluation.config_index == hottest
        direct = np.sqrt((evaluation.delta_forces[rows] ** 2).mean())
        assert dist["force_rmse_high_energy"] == pytest.approx(direct, rel=1e-12)


class TestSmoothnessRatio:
    def test_ratio_grows_with_the_width_of_the_error_field(self):
        """Theory sec. 4.1: ``sigma(dU) ~ a w`` and ``RMSE_F ~ a / sqrt(w)``, so
        the ratio increases with the width of the perturbation at fixed
        amplitude.  This is the measurement that turns the frequency argument
        into an observation."""
        reference = argon_reference()
        data = argon_dataset(12)
        ratios = []
        for width in (0.15, 0.3, 0.6):
            evaluation = evaluate_pair(shell_model(reference, 4e-3, width), reference, data)
            std = standard_metrics(evaluation)
            ratios.append(smoothness_ratio(evaluation.delta_u, std["force_rmse"]))
        print(f"\n[smoothness] widths (0.15, 0.30, 0.60) A -> ratios {np.round(ratios, 3).tolist()} A")
        assert ratios[0] < ratios[1] < ratios[2]

    def test_agrees_with_the_response_module(self):
        from atomlab.analysis.response import spectral_decomposition

        reference = argon_reference()
        data = argon_dataset(10)
        evaluation = evaluate_pair(shell_model(reference, 4e-3, 0.3), reference, data)
        std = standard_metrics(evaluation)
        mine = smoothness_ratio(evaluation.delta_u, std["force_rmse"])
        theirs = spectral_decomposition(
            evaluation.delta_u, evaluation.energy_reference, std["force_rmse"], 100.0
        )["smoothness_ratio"]
        assert mine == pytest.approx(theirs, rel=1e-12)


class TestExtrapolationGrade:
    def test_mahalanobis_is_zero_at_the_training_mean_and_grows_outward(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(500, 3))
        mean = train.mean(axis=0)
        d = mahalanobis_distances(np.stack([mean, mean + 5.0]), train)
        assert d[0] < 1e-8
        assert d[1] > 5.0

    def test_singular_directions_do_not_blow_up(self):
        """A descriptor basis with an exactly constant column is normal, not an
        error; the ridge must keep the distance finite."""
        rng = np.random.default_rng(1)
        train = np.column_stack([rng.normal(size=200), np.ones(200)])
        test = np.column_stack([rng.normal(size=5), np.ones(5)])
        d = mahalanobis_distances(test, train)
        assert np.all(np.isfinite(d))

    def test_grade_is_larger_for_a_more_distorted_test_set(self):
        descriptor = CoordinationDescriptor()
        base = fcc(ARGON_A0, "Ar", (2, 2, 2))
        train = [rattle(base, 0.05, seed=s) for s in range(6)]
        near = [rattle(base, 0.06, seed=100 + s) for s in range(3)]
        far = [rattle(base, 0.35, seed=200 + s) for s in range(3)]
        from atomlab.analysis.metrics import descriptor_features

        train_f = descriptor_features(descriptor, train)
        near_g = extrapolation_grade(descriptor_features(descriptor, near), train_f)
        far_g = extrapolation_grade(descriptor_features(descriptor, far), train_f)
        print(f"\n[extrapolation] near={near_g:.2f}  far={far_g:.2f}")
        assert far_g > 2.0 * near_g


# --------------------------------------------------------------------------
# The single entry point
# --------------------------------------------------------------------------


class TestComputeAllMetrics:
    @staticmethod
    def _full_call(n_configs: int = 10):
        reference = argon_reference()
        data = argon_dataset(n_configs)
        model = shell_model(reference, 5e-3, 0.3)
        ensemble = [
            shell_model(reference, 4e-3, 0.30),
            shell_model(reference, 6e-3, 0.32),
            shell_model(reference, 5e-3, 0.28),
        ]
        descriptor = CoordinationDescriptor()
        train = [rattle(fcc(ARGON_A0, "Ar", (2, 2, 2)), 0.08, seed=500 + s) for s in range(5)]
        return compute_all_metrics(
            model,
            reference,
            data,
            ensemble=ensemble,
            variance_fn=lambda cfg: 1e-3 * cfg.n_atoms,
            descriptor=descriptor,
            train_configurations=train,
            max_hessian_atoms=64,
        )

    def test_every_metric_is_finite_and_documented(self):
        start = time.perf_counter()
        metrics = self._full_call()
        elapsed = time.perf_counter() - start
        print(f"\n[compute_all_metrics] {len(metrics)} metrics in {elapsed:.2f} s")
        for name, value in metrics.items():
            print(f"  {name:34s} {value: .6g}  [{METRIC_INFO[name][1]}]")
        assert list(metrics) == metric_names()
        assert set(metrics) == set(METRIC_INFO)
        for name, value in metrics.items():
            assert np.isfinite(value), f"{name} is not finite: {value}"

    def test_catalogue_is_well_formed(self):
        for name, entry in METRIC_INFO.items():
            assert len(entry) == 3, name
            description, units, direction = entry
            assert description.strip() and units.strip()
            assert direction in (LOWER_IS_BETTER, HIGHER_IS_BETTER, DIAGNOSTIC), name
            assert metric_direction(name) == direction
        assert metric_direction("force_cosine") == HIGHER_IS_BETTER
        assert metric_direction("smoothness_ratio") == DIAGNOSTIC
        assert "force_rmse" in describe_metrics()

    def test_missing_optional_inputs_give_nan_not_zero(self):
        """A model with no ensemble and no descriptor must report NaN for those
        metrics -- reporting zero would put a perfectly-confident-looking row in
        the correlation table."""
        reference = argon_reference()
        metrics = compute_all_metrics(
            shell_model(reference, 5e-3, 0.3), reference, argon_dataset(4), max_hessian_atoms=64
        )
        assert set(metrics) == set(METRIC_INFO)
        for name in (
            "ensemble_force_disagreement",
            "ensemble_energy_disagreement",
            "predictive_variance",
            "extrapolation_grade",
        ):
            assert np.isnan(metrics[name]), name
        assert np.isfinite(metrics["force_rmse"])

    def test_curvature_is_nan_when_no_configuration_is_small_enough(self):
        reference = argon_reference()
        metrics = compute_all_metrics(
            shell_model(reference, 5e-3, 0.3), reference, argon_dataset(3), max_hessian_atoms=4
        )
        assert np.isnan(metrics["hessian_frobenius_error"])
        assert np.isfinite(metrics["force_rmse"])

    def test_predictive_variance_uses_a_model_method_when_present(self):
        reference = argon_reference()

        class WithVariance(type(reference)):
            def predictive_variance(self, configuration):
                return 2e-3 * configuration.n_atoms**2

        model = WithVariance(reference.epsilon, reference.sigma, LJ_CUTOFF)
        metrics = compute_all_metrics(model, reference, argon_dataset(3), max_hessian_atoms=4)
        assert metrics["predictive_variance"] == pytest.approx(2e-3)

    def test_predictive_variance_read_from_result_extra(self):
        """The other supported route: a model that reports variance in ``Result.extra``."""
        from atomlab.potentials.base import Potential
        from atomlab.types import Result

        reference = argon_reference()

        class NoisyCopy(Potential):
            cutoff = LJ_CUTOFF
            name = "noisy"

            def compute(self, configuration, *, forces=True, virial=True):
                res = reference.compute(configuration, forces=forces, virial=virial)
                res.extra["variance"] = 4e-3 * configuration.n_atoms**2
                return res

        metrics = compute_all_metrics(NoisyCopy(), reference, argon_dataset(3), max_hessian_atoms=4)
        assert metrics["predictive_variance"] == pytest.approx(4e-3)
        assert metrics["force_rmse"] == pytest.approx(0.0, abs=1e-12)

    def test_ensemble_discovered_from_the_model_when_not_passed(self):
        reference = argon_reference()

        class Committee(type(reference)):
            models: list = []

        model = Committee(reference.epsilon, reference.sigma, LJ_CUTOFF)
        model.models = [shell_model(reference, 3e-3, 0.3), shell_model(reference, 5e-3, 0.3)]
        metrics = compute_all_metrics(model, reference, argon_dataset(3), max_hessian_atoms=4)
        assert np.isfinite(metrics["ensemble_force_disagreement"])
        assert np.isfinite(metrics["ensemble_energy_disagreement"])

    def test_stored_labels_can_be_reused_and_agree_with_recomputation(self):
        reference = argon_reference()
        labelled = Dataset(reference.label(argon_dataset(4)))
        model = shell_model(reference, 5e-3, 0.3)
        from_labels = standard_metrics(evaluate_pair(model, reference, labelled, use_labels=True))
        recomputed = standard_metrics(evaluate_pair(model, reference, labelled, use_labels=False))
        for key, value in recomputed.items():
            assert from_labels[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key

    def test_results_are_deterministic(self):
        reference = argon_reference()
        data = argon_dataset(4)
        model = shell_model(reference, 5e-3, 0.3)
        first = compute_all_metrics(model, reference, data, max_hessian_atoms=64)
        second = compute_all_metrics(model, reference, data, max_hessian_atoms=64)
        assert list(first) == list(second)
        # NaN != NaN, so compare with equal_nan rather than dict equality: the
        # unavailable metrics are legitimately NaN here and must stay NaN.
        assert np.array_equal(
            np.array(list(first.values())), np.array(list(second.values())), equal_nan=True
        )

    def test_zoo_of_models_feeds_the_correlation_table(self):
        """End-to-end: several models -> metrics -> proxy quality matrix.

        This is the shape of ``experiments/exp05``, run small.  It checks the
        two modules compose without special-casing any metric, which is the
        property the uniform catalogue exists to guarantee.
        """
        reference = argon_reference()
        data = argon_dataset(8)
        rng = np.random.default_rng(0)
        metrics_by_model, observables_by_model = {}, {}
        for i in range(6):
            amplitude = 2e-3 * (i + 1)
            width = 0.2 + 0.05 * i
            model = shell_model(reference, amplitude, width)
            name = f"shell{i}"
            metrics_by_model[name] = compute_all_metrics(
                model, reference, data, max_hessian_atoms=4
            )
            # A stand-in observable error: proportional to amplitude*width, i.e.
            # the covariance scaling of theory eq. 4.4, plus noise.
            observables_by_model[name] = {
                "rdf_error": float(amplitude * width * (1.0 + 0.05 * rng.normal()))
            }
        table = proxy_quality_matrix(
            metrics_by_model, observables_by_model, n_bootstrap=200, seed=0
        )
        assert table.shape == (len(METRIC_INFO), 1)
        assert table.metric_names == metric_names()
        # metrics that are all-NaN across the zoo yield NaN cells, not crashes
        assert np.isnan(table.rho[table.metric_names.index("extrapolation_grade"), 0])
        assert np.isfinite(table.rho[table.metric_names.index("force_rmse"), 0])
        best = top_k_agreement(
            [metrics_by_model[m]["force_rmse"] for m in table.model_names],
            [observables_by_model[m]["rdf_error"] for m in table.model_names],
            k=2,
        )
        assert 0.0 <= best.fraction <= 1.0
