"""Validation of the canonical samplers.

These samplers are the source of every reference-ensemble average in the study,
so a bias here would propagate into every result.  They are checked in three
independent ways, which is the point: a sampler that is wrong is usually wrong
in a way that is invisible to any single test.

1. **Against a closed form.**  An Einstein crystal has ``<U> = (3N/2) k_B T`` and
   ``var(U) = (3N/2) (k_B T)^2`` exactly, for every temperature and spring
   constant.  Both moments are checked, because the mean alone is reproduced by
   several incorrect samplers while the variance is not.
2. **Against each other.**  Hybrid Monte Carlo and single-particle Metropolis
   share no code beyond the potential.  Agreement on a Lennard-Jones liquid is
   therefore evidence rather than a self-consistency check.
3. **Against the perturbation identity.**  Reweighting between two temperatures
   is exact, so a sampler run at one temperature must predict the energy at a
   nearby one.  This closes the loop with :mod:`atomlab.analysis.response`.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.analysis.statistics import blocking_analysis
from atomlab.build import fcc
from atomlab.potentials.harmonic import EinsteinCrystal
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.sampling import hybrid_monte_carlo, maxwell_velocities, metropolis_nvt
from atomlab.units import KB, MVV2E


def einstein_system(reps=(2, 2, 2), spring=1.0):
    cfg = fcc(4.05, "Al", reps)
    return cfg, EinsteinCrystal(cfg.positions.copy(), spring)


class TestMaxwellVelocities:
    def test_kinetic_energy_matches_equipartition(self):
        rng = np.random.default_rng(0)
        masses = np.full(2000, 39.948)
        v = maxwell_velocities(masses, 300.0, rng, remove_com=False)
        kinetic = 0.5 * MVV2E * (masses[:, None] * v**2).sum()
        expected = 1.5 * masses.size * KB * 300.0
        assert kinetic == pytest.approx(expected, rel=0.03)

    def test_com_removal(self):
        rng = np.random.default_rng(1)
        masses = np.linspace(1.0, 50.0, 100)
        v = maxwell_velocities(masses, 300.0, rng, remove_com=True)
        assert np.allclose((masses[:, None] * v).sum(axis=0), 0.0, atol=1e-10)

    def test_temperature_scaling(self):
        """Velocity width must scale as sqrt(T), which pins the unit conversion."""
        rng = np.random.default_rng(2)
        masses = np.full(5000, 39.948)
        cold = maxwell_velocities(masses, 100.0, rng, remove_com=False).std()
        hot = maxwell_velocities(masses, 400.0, rng, remove_com=False).std()
        assert hot / cold == pytest.approx(2.0, rel=0.05)


class TestHMCAgainstClosedForm:
    """An Einstein crystal fixes both moments of the potential energy exactly."""

    def test_mean_energy_is_equipartition(self):
        cfg, potential = einstein_system()
        traj, report = hybrid_monte_carlo(
            cfg, potential, 100.0, n_samples=6000, n_leapfrog=8,
            step_size=3e-3, burn_in=400, seed=1, remove_com=False,
        )
        u = traj.scalars["potential_energy"]
        estimate = blocking_analysis(u)
        exact = 1.5 * cfg.n_atoms * KB * 100.0
        assert 0.4 < report.acceptance < 0.95
        assert abs(estimate.value - exact) < 3.0 * estimate.error

    def test_energy_variance_is_equipartition(self):
        """var(U) = (3N/2)(kT)^2 -- the moment a wrong temperature scale breaks."""
        cfg, potential = einstein_system()
        traj, _ = hybrid_monte_carlo(
            cfg, potential, 100.0, n_samples=8000, n_leapfrog=8,
            step_size=3e-3, burn_in=400, seed=2, remove_com=False,
        )
        u = traj.scalars["potential_energy"]
        exact = 1.5 * cfg.n_atoms * (KB * 100.0) ** 2
        assert u.var() == pytest.approx(exact, rel=0.10)

    def test_energy_is_linear_in_temperature(self):
        """<U> must double when T doubles, for any harmonic system."""
        cfg, potential = einstein_system()
        means = []
        for temperature in (80.0, 160.0):
            traj, _ = hybrid_monte_carlo(
                cfg, potential, temperature, n_samples=4000, n_leapfrog=8,
                step_size=3e-3, burn_in=300, seed=3, remove_com=False,
            )
            means.append(traj.scalars["potential_energy"].mean())
        assert means[1] / means[0] == pytest.approx(2.0, rel=0.05)

    def test_result_is_independent_of_spring_constant_in_energy(self):
        """Equipartition does not care how stiff the springs are."""
        results = []
        for spring in (0.5, 4.0):
            cfg, potential = einstein_system(spring=spring)
            traj, _ = hybrid_monte_carlo(
                cfg, potential, 100.0, n_samples=4000, n_leapfrog=8,
                step_size=3e-3, burn_in=300, seed=4, remove_com=False,
            )
            results.append(traj.scalars["potential_energy"].mean())
        assert results[0] == pytest.approx(results[1], rel=0.06)


class TestSamplerAgreement:
    """Two samplers sharing no code must agree on a real interacting system."""

    @pytest.mark.slow
    def test_hmc_matches_metropolis_on_lennard_jones(self):
        cfg = fcc(5.4, "Ar", (2, 2, 2))
        potential = LennardJones.argon()
        # Cutoff must fit the cell; a 2x2x2 fcc argon cell is 10.8 A across.
        if 2 * potential.cutoff > 10.8:
            potential = LennardJones(epsilon=0.0103, sigma=3.405, cutoff=5.0, mode="shifted_force")

        hmc_traj, hmc_report = hybrid_monte_carlo(
            cfg, potential, 60.0, n_samples=3000, n_leapfrog=10,
            step_size=2e-3, burn_in=300, seed=5,
        )
        mc_traj, mc_report = metropolis_nvt(
            cfg, potential, 60.0, n_sweeps=1500, max_displacement=0.12, burn_in=150, seed=6,
        )

        hmc = blocking_analysis(hmc_traj.scalars["potential_energy"])
        mc = blocking_analysis(mc_traj.scalars["potential_energy"])
        combined = np.hypot(hmc.error, mc.error)
        assert abs(hmc.value - mc.value) < 4.0 * combined, (
            f"HMC {hmc.value:.4f}+/-{hmc.error:.4f} vs Metropolis {mc.value:.4f}+/-{mc.error:.4f}; "
            f"acceptances {hmc_report.acceptance:.2f} / {mc_report.acceptance:.2f}"
        )


class TestReweightingConsistency:
    """Samples at one temperature must predict the energy at a nearby one."""

    def test_temperature_reweighting_matches_direct_sampling(self):
        cfg, potential = einstein_system()
        t0, t1 = 100.0, 110.0

        traj, _ = hybrid_monte_carlo(
            cfg, potential, t0, n_samples=8000, n_leapfrog=8,
            step_size=3e-3, burn_in=400, seed=7, remove_com=False,
        )
        u = traj.scalars["potential_energy"]

        # Reweighting from beta0 to beta1 uses weights exp(-(beta1-beta0) U).
        delta_beta = 1.0 / (KB * t1) - 1.0 / (KB * t0)
        log_w = -delta_beta * u
        w = np.exp(log_w - log_w.max())
        reweighted = (w * u).sum() / w.sum()

        exact_t1 = 1.5 * cfg.n_atoms * KB * t1
        assert reweighted == pytest.approx(exact_t1, rel=0.03)


class TestBookkeeping:
    def test_trajectory_is_marked_non_dynamical(self):
        cfg, potential = einstein_system()
        traj, _ = hybrid_monte_carlo(cfg, potential, 100.0, n_samples=20, burn_in=10, seed=8)
        assert traj.info["dynamical"] is False
        assert traj.info["sampler"] == "hybrid_monte_carlo"
        assert traj.velocities is None

    def test_deterministic_given_a_seed(self):
        cfg, potential = einstein_system()
        kwargs = dict(n_samples=50, n_leapfrog=6, step_size=3e-3, burn_in=20, seed=9)
        a, _ = hybrid_monte_carlo(cfg, potential, 100.0, **kwargs)
        b, _ = hybrid_monte_carlo(cfg, potential, 100.0, **kwargs)
        assert np.array_equal(a.positions, b.positions)

    def test_different_seeds_differ(self):
        cfg, potential = einstein_system()
        a, _ = hybrid_monte_carlo(cfg, potential, 100.0, n_samples=50, burn_in=20, seed=10)
        b, _ = hybrid_monte_carlo(cfg, potential, 100.0, n_samples=50, burn_in=20, seed=11)
        assert not np.array_equal(a.positions, b.positions)

    def test_frame_count_and_shapes(self):
        cfg, potential = einstein_system()
        traj, _ = hybrid_monte_carlo(
            cfg, potential, 100.0, n_samples=37, stride=3, burn_in=10, seed=12
        )
        assert traj.n_frames == 37
        assert traj.positions.shape == (37, cfg.n_atoms, 3)
        assert traj.scalars["potential_energy"].shape == (37,)

    def test_low_acceptance_is_reported(self):
        """A step size far too large must produce a warning, not silent garbage.

        The springs here have an oscillation period of about 330 fs, so velocity
        Verlet loses stability past roughly 100 fs. 60 fs already pushes the
        integration error well beyond k_B T for a 32-atom system and acceptance
        collapses; note the transition is sharp rather than gradual, and is not
        even monotone in the timestep (leapfrog on a harmonic system has
        resonances where a longer step happens to land back near its start).
        """
        cfg, potential = einstein_system()
        _, report = hybrid_monte_carlo(
            cfg, potential, 100.0, n_samples=200, n_leapfrog=20,
            step_size=6e-2, burn_in=0, seed=13, adapt=False, remove_com=False,
        )
        assert report.acceptance < 0.2
        assert any("acceptance" in note for note in report.notes)

    def test_hmc_does_not_adapt_without_burn_in(self):
        """`adapt=True` cannot change the production kernel when burn-in is zero."""
        cfg, potential = einstein_system()
        initial = 3e-3
        _, report = hybrid_monte_carlo(
            cfg, potential, 100.0, n_samples=25, n_leapfrog=4,
            step_size=initial, burn_in=0, seed=130, adapt=True,
            remove_com=False,
        )
        assert report.final_step_size == pytest.approx(initial)

    def test_hmc_adaptation_uses_complete_windows_and_freezes(self):
        """Two ten-proposal burn-in windows adapt twice; production adds none."""
        cfg, potential = einstein_system()
        initial = 1e-4  # essentially unit acceptance, so both windows scale up
        reports = []
        for n_samples in (1, 30):
            _, report = hybrid_monte_carlo(
                cfg, potential, 100.0, n_samples=n_samples, n_leapfrog=2,
                step_size=initial, burn_in=20, seed=131, adapt=True,
                remove_com=False,
            )
            reports.append(report)
        assert reports[0].final_step_size == pytest.approx(initial * 1.1**2)
        assert reports[1].final_step_size == pytest.approx(reports[0].final_step_size)

    def test_metropolis_adapts_toward_half_acceptance(self):
        cfg, potential = einstein_system()
        _, report = metropolis_nvt(
            cfg, potential, 100.0, n_sweeps=60, max_displacement=2.0, burn_in=40, seed=14
        )
        assert 0.2 < report.acceptance < 0.8
