"""Validation of the molecular dynamics engine.

The tests are organised around the properties ``docs/design.md`` §5.3 declares
non-negotiable, and they are deliberately *quantitative*: an integrator that is
merely stable is not evidence of anything, because a wrong mass factor or a
wrong sign is stable too.  What discriminates a correct implementation is

* agreement with an exactly solvable trajectory, and the **observed order** of
  the error under timestep refinement;
* exact time reversibility of the symplectic scheme;
* the **conserved quantity** of the extended-system thermostats, which is
  sensitive to every part of the chain propagator;
* the sampled kinetic-energy **distribution**, not only its mean -- the mean is
  what a broken thermostat gets right (see the Berendsen tests, which are here
  as a negative control and are expected to fail the distribution test while
  passing the mean test).

Fitted exponents and measured drifts are printed so that a run of the suite
doubles as the numerical report.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from atomlab.build import fcc
from atomlab.md.integrators import (
    BerendsenThermostat,
    Langevin,
    MTKBarostat,
    NoseHooverChain,
    VelocityVerlet,
    suzuki_yoshida_weights,
)
from atomlab.md.state import MDState
from atomlab.md.velocities import (
    instantaneous_temperature,
    kinetic_energy,
    maxwell_boltzmann,
    n_dof,
    remove_com_momentum,
    rescale_to_temperature,
)
from atomlab.potentials.base import ZeroPotential
from atomlab.potentials.harmonic import EinsteinCrystal, HarmonicPair
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.types import Configuration
from atomlab.units import EV_A3_TO_BAR, KB, MVV2E

ARGON_MASS = 39.948
ARGON_A0 = 5.256  # A, fcc lattice constant used throughout the LJ-argon tests


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _block_error(series, n_blocks: int = 16) -> float:
    """Standard error of the mean from block averaging.

    Correlated MD samples make the naive ``std/sqrt(n)`` a fantasy; blocking
    into ``n_blocks`` contiguous chunks whose length exceeds the correlation
    time restores an honest error bar.
    """
    x = np.asarray(series, dtype=float)
    usable = (x.size // n_blocks) * n_blocks
    blocks = x[:usable].reshape(n_blocks, -1).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def _fitted_order(step_sizes, errors) -> float:
    """Slope of ``log(error)`` against ``log(dt)``, i.e. the observed order."""
    return float(np.polyfit(np.log(np.asarray(step_sizes)), np.log(np.asarray(errors)), 1)[0])


def _argon(reps=(3, 3, 3), cutoff=6.0, skin=1.5):
    """fcc argon plus a matching LJ potential.

    ``skin`` enables the Verlet-list reuse inside the potential; it changes the
    trajectory by exactly nothing (bitwise) and speeds these tests up ~8x.
    """
    return fcc(ARGON_A0, "Ar", reps), LennardJones.argon(cutoff=cutoff, skin=skin)


def _einstein_system(n_atoms: int, spring: float, seed: int = 0):
    """``n_atoms`` independent 3D harmonic oscillators, exactly solvable."""
    ref = np.random.default_rng(seed).uniform(0.0, 30.0, (n_atoms, 3))
    cfg = Configuration(
        positions=ref,
        cell=np.eye(3) * 60.0,
        pbc=False,
        masses=np.full(n_atoms, ARGON_MASS),
    )
    return cfg, EinsteinCrystal(ref, spring), ref


@pytest.fixture(scope="module")
def equilibrated_argon():
    """256-atom fcc argon equilibrated at 80 K, shared by the slow NVE tests."""
    cfg, lj = _argon(reps=(4, 4, 4), cutoff=8.5, skin=1.5)
    state = MDState.from_configuration(cfg, lj, 160.0, seed=3)
    # Start hot: half the kinetic energy is converted to potential energy as the
    # lattice relaxes, so a 160 K Maxwellian on a perfect lattice equilibrates
    # near 80 K.  Langevin removes the rest of the transient.
    Langevin(0.002, 80.0, 2.0, virial=False).run(state, lj, 2500)
    return state, lj


# --------------------------------------------------------------------------
# velocities and degrees of freedom
# --------------------------------------------------------------------------


def test_n_dof_bookkeeping():
    assert n_dof(256, fixed_com=True) == 3 * 256 - 3
    assert n_dof(256, fixed_com=False) == 3 * 256
    with pytest.raises(ValueError):
        n_dof(1, fixed_com=True)


def test_kinetic_energy_uses_metal_unit_conversion():
    # One atom of mass m moving at 1 A/ps has KE = 0.5 * MVV2E * m exactly.
    masses = np.array([ARGON_MASS])
    velocities = np.array([[1.0, 0.0, 0.0]])
    assert kinetic_energy(masses, velocities) == pytest.approx(0.5 * MVV2E * ARGON_MASS)


def test_maxwell_boltzmann_moments_and_com():
    rng = np.random.default_rng(0)
    masses = np.full(4000, ARGON_MASS)
    target = 300.0
    v = maxwell_boltzmann(masses, target, rng)

    # zero total momentum by construction
    assert np.abs((masses[:, None] * v).sum(axis=0)).max() < 1e-9

    t_meas = instantaneous_temperature(masses, v, fixed_com=True)
    # relative sd of T is sqrt(2/n_dof) = 1.3% here; 4 sd is a fair tolerance
    assert t_meas == pytest.approx(target, rel=4 * np.sqrt(2.0 / n_dof(masses.size)))

    # per-component variance must be kT/(m*MVV2E), not kT/m
    assert v[:, 0].var() == pytest.approx(KB * target / (ARGON_MASS * MVV2E), rel=0.05)

    exact = maxwell_boltzmann(masses, target, rng, exact_temperature=True)
    assert instantaneous_temperature(masses, exact) == pytest.approx(target, rel=1e-12)


def test_remove_com_and_rescale():
    rng = np.random.default_rng(1)
    masses = rng.uniform(1.0, 50.0, 64)
    v = rng.normal(size=(64, 3)) + 3.0
    # Absolute tolerances are meaningless here: the momenta being cancelled are
    # of order sum(m)*|v| ~ 1e3, so "zero" means ~1e-13 in double precision.
    scale = float(masses.sum() * np.abs(v).max())
    v0 = remove_com_momentum(v, masses)
    assert np.abs((masses[:, None] * v0).sum(axis=0)).max() < 1e-13 * scale
    v1 = rescale_to_temperature(v0, masses, 250.0)
    assert instantaneous_temperature(masses, v1) == pytest.approx(250.0)
    # rescaling must not reintroduce drift
    assert np.abs((masses[:, None] * v1).sum(axis=0)).max() < 1e-13 * scale


def test_suzuki_yoshida_weights_are_symmetric_and_normalised():
    for order in (1, 3, 5, 7):
        w = suzuki_yoshida_weights(order)
        assert w.size == order
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w, w[::-1])
    with pytest.raises(ValueError):
        suzuki_yoshida_weights(4)


# --------------------------------------------------------------------------
# MDState
# --------------------------------------------------------------------------


def test_state_from_configuration_and_roundtrip():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 90.0, seed=11)
    assert state.n_dof == 3 * cfg.n_atoms - 3
    assert state.temperature() > 0.0
    assert np.abs(state.com_velocity()).max() < 1e-12

    back = state.to_configuration()
    assert np.allclose(back.positions, cfg.positions)
    assert back.energy == pytest.approx(state.potential_energy)

    clone = state.copy()
    clone.positions[0, 0] += 1.0
    clone.thermostat["probe"] = 1
    assert state.positions[0, 0] != clone.positions[0, 0]
    assert "probe" not in state.thermostat


def test_state_forces_raise_before_evaluation():
    cfg, _ = _argon()
    state = MDState.from_configuration(cfg, None, 0.0, seed=0)
    with pytest.raises(RuntimeError):
        _ = state.forces


def test_pressure_has_both_kinetic_and_virial_terms():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 120.0, seed=4)

    manual = (2.0 * state.kinetic_energy() + np.trace(state.virial)) / (3.0 * state.volume)
    assert state.pressure(unit="eV/A^3") == pytest.approx(manual)
    assert state.pressure(unit="bar") == pytest.approx(manual * EV_A3_TO_BAR)
    assert state.pressure(unit="bar") == pytest.approx(
        state.pressure(unit="GPa") * 1e4, rel=1e-10
    )

    # The virial term is not negligible: dropping it changes the answer by more
    # than the kinetic term itself for a cold dense solid.
    kinetic_only = 2.0 * state.kinetic_energy() / (3.0 * state.volume)
    assert abs(manual - kinetic_only) > 0.1 * abs(kinetic_only)


def test_ideal_gas_pressure_is_exact_for_a_free_particle_system():
    # With no interactions the virial vanishes and P V = n_dof k_B T / 3 * 3.
    cfg, _ = _argon()
    zero = ZeroPotential()
    state = MDState.from_configuration(cfg, zero, 150.0, seed=6)
    expected = state.n_dof * KB * state.temperature() / (3.0 * state.volume) * 3.0 / 3.0
    assert state.pressure(unit="eV/A^3") == pytest.approx(
        2.0 * state.kinetic_energy() / (3.0 * state.volume)
    )
    assert state.pressure(unit="eV/A^3") == pytest.approx(expected)


# --------------------------------------------------------------------------
# velocity Verlet against the exactly solvable harmonic oscillator
# --------------------------------------------------------------------------


def _harmonic_pair_system(k=1.0, r0=3.0, d0=3.4):
    """Two atoms on a spring: the one MD problem with a closed-form solution."""
    mu = ARGON_MASS / 2.0
    omega = np.sqrt(k / (mu * MVV2E))  # rad/ps

    def analytic_positions(t):
        d = r0 + (d0 - r0) * np.cos(omega * t)
        return np.array([[-0.5 * d, 0.0, 0.0], [0.5 * d, 0.0, 0.0]])

    cfg = Configuration(
        positions=analytic_positions(0.0),
        cell=np.eye(3) * 100.0,
        pbc=False,
        masses=[ARGON_MASS, ARGON_MASS],
    )
    return cfg, HarmonicPair(k=k, r0=r0, cutoff=20.0), analytic_positions, omega


def test_velocity_verlet_matches_analytic_harmonic_trajectory():
    cfg, pot, analytic, omega = _harmonic_pair_system()
    dt = 0.0005  # ps, ~570 steps per period
    state = MDState.from_configuration(cfg, pot, 0.0, seed=0, fixed_com=False)
    vv = VelocityVerlet(dt)

    worst = 0.0
    for _ in range(int(round(2.0 / dt))):  # 2 ps = 7 periods
        vv.step(state, pot)
        worst = max(worst, np.abs(state.positions - analytic(state.time)).max())
    print(f"\n[harmonic] omega = {omega:.4f} rad/ps, max |r - r_exact| = {worst:.3e} A")
    assert worst < 1e-4


def test_velocity_verlet_global_error_is_second_order():
    cfg, pot, analytic, _ = _harmonic_pair_system()
    step_sizes = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    errors = []
    for dt in step_sizes:
        state = MDState.from_configuration(cfg, pot, 0.0, seed=0, fixed_com=False)
        vv = VelocityVerlet(dt)
        worst = 0.0
        for _ in range(int(round(2.0 / dt))):
            vv.step(state, pot)
            worst = max(worst, np.abs(state.positions - analytic(state.time)).max())
        errors.append(worst)

    order = _fitted_order(step_sizes, errors)
    print("\n[VV order] dt/ps, max error/A:")
    for dt, e in zip(step_sizes, errors):
        print(f"    {dt:8.5f}  {e:.4e}")
    print(f"    fitted exponent = {order:.4f}")
    assert 1.9 < order < 2.1


def test_velocity_verlet_is_time_reversible():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 100.0, seed=5)
    r0 = state.positions.copy()
    v0 = state.velocities.copy()

    vv = VelocityVerlet(0.002)
    vv.run(state, lj, 300)
    state.velocities = -state.velocities  # forces are unchanged, so the cache stays valid
    vv.run(state, lj, 300)

    dr = float(np.abs(state.positions - r0).max())
    dv = float(np.abs(-state.velocities - v0).max())
    print(f"\n[reversibility] max |dr| = {dr:.3e} A, max |dv| = {dv:.3e} A/ps after 2x300 steps")
    assert dr < 1e-9
    assert dv < 1e-9


def test_velocity_verlet_conserves_energy_short_run():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 120.0, seed=8)
    vv = VelocityVerlet(0.002, virial=False)
    vv.step(state, lj)
    e0 = state.total_energy()
    worst = 0.0
    for _ in range(500):
        vv.step(state, lj)
        worst = max(worst, abs(state.total_energy() - e0))
    assert worst / state.n_atoms < 1e-5


# --------------------------------------------------------------------------
# NVE: the design-document targets
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.physics
def test_nve_energy_drift_lj_argon_50ps(equilibrated_argon):
    """256 atoms, 50 ps at dt = 1 fs: drift must stay under 1e-4 eV/atom."""
    ref_state, lj = equilibrated_argon
    state = ref_state.copy()
    dt = 0.001
    n_steps = 50_000

    vv = VelocityVerlet(dt, virial=False)
    vv.step(state, lj)
    e0 = state.total_energy()
    energies = np.empty(n_steps)
    for i in range(n_steps):
        vv.step(state, lj)
        energies[i] = state.total_energy()

    n = state.n_atoms
    times = np.arange(n_steps) * dt
    slope = float(np.polyfit(times, energies, 1)[0]) / n
    max_dev = float(np.abs(energies - e0).max()) / n
    rms = float(energies.std()) / n
    print(
        f"\n[NVE 50 ps, {n} atoms, dt = 1 fs]"
        f"\n    max |E - E0|      = {max_dev:.4e} eV/atom"
        f"\n    rms E fluctuation = {rms:.4e} eV/atom"
        f"\n    secular drift     = {slope:.4e} eV/atom/ps"
        f"  ({slope * 50.0:.4e} eV/atom over the run)"
        f"\n    T = {state.temperature():.2f} K"
    )
    assert max_dev < 1e-4
    assert abs(slope) * 50.0 < 1e-4


@pytest.mark.slow
@pytest.mark.physics
def test_nve_energy_drift_scales_as_dt_squared(equilibrated_argon):
    ref_state, lj = equilibrated_argon
    step_sizes = [0.008, 0.004, 0.002, 0.001]
    span = 5.0  # ps per run
    max_devs, slopes = [], []
    for dt in step_sizes:
        state = ref_state.copy()
        vv = VelocityVerlet(dt, virial=False)
        vv.step(state, lj)
        e0 = state.total_energy()
        n_steps = int(round(span / dt))
        energies = np.empty(n_steps)
        for i in range(n_steps):
            vv.step(state, lj)
            energies[i] = state.total_energy()
        max_devs.append(float(np.abs(energies - e0).max()) / state.n_atoms)
        slopes.append(abs(float(np.polyfit(np.arange(n_steps) * dt, energies, 1)[0])) / state.n_atoms)

    order = _fitted_order(step_sizes, max_devs)
    print("\n[NVE dt-scaling, 5 ps runs] dt/fs, max |dE|/atom, |secular drift|/atom/ps:")
    for dt, d, s in zip(step_sizes, max_devs, slopes):
        print(f"    {dt * 1000:5.1f}  {d:.4e}  {s:.4e}")
    print(f"    fitted exponent of max |dE| = {order:.4f}")
    assert 1.8 < order < 2.2


# --------------------------------------------------------------------------
# Langevin / BAOAB
# --------------------------------------------------------------------------


def test_baoab_harmonic_configurational_average_is_exact():
    r"""BAOAB samples <x^2> = kT/k exactly, while <mv^2> = kT (1 - w^2 dt^2/4).

    This pair of statements is the signature of the BAOAB ordering: the
    configurational marginal is exact at *any* timestep for a harmonic system,
    and the whole :math:`O(dt^2)` error sits in the momenta.  Running at
    :math:`\omega dt = 0.6` -- far larger than any production timestep -- turns
    a 9% kinetic bias into a decisive test that the ``O`` block is the exact
    Ornstein-Uhlenbeck update and that it sits between the two ``A`` half-drifts.
    """
    spring = 0.8  # eV/A^2
    temperature = 100.0
    cfg, pot, ref = _einstein_system(400, spring, seed=0)
    omega = np.sqrt(spring / (ARGON_MASS * MVV2E))

    print(f"\n[BAOAB harmonic] omega = {omega:.3f} rad/ps")
    for omega_dt in (0.2, 0.6):
        dt = omega_dt / omega
        state = MDState.from_configuration(cfg, pot, temperature, seed=4, fixed_com=False)
        lgv = Langevin(dt, temperature, 5.0, virial=False)
        lgv.run(state, pot, 3000)

        x2, kin_t = [], []
        for i in range(30_000):
            lgv.step(state, pot)
            if i % 10 == 0:
                x2.append(((state.positions - ref) ** 2).sum() / (3 * state.n_atoms))
                kin_t.append(state.temperature())

        conf_ratio = np.mean(x2) / (KB * temperature / spring)
        kin_ratio = np.mean(kin_t) / temperature
        print(
            f"    omega*dt = {omega_dt}: <x^2>/(kT/k) = {conf_ratio:.5f} (exact 1)"
            f"   <T_kin>/T = {kin_ratio:.5f} (predicted {1 - omega_dt**2 / 4:.5f})"
        )
        assert conf_ratio == pytest.approx(1.0, abs=0.004)
        assert kin_ratio == pytest.approx(1.0 - omega_dt**2 / 4.0, abs=0.004)


def test_langevin_harmonic_mean_square_displacement():
    """<x^2> = kT/k for the Einstein crystal, with a block-averaged error bar."""
    spring = 0.8
    temperature = 100.0
    cfg, pot, ref = _einstein_system(216, spring, seed=1)
    state = MDState.from_configuration(cfg, pot, temperature, seed=2, fixed_com=False)
    lgv = Langevin(0.005, temperature, 4.0, virial=False)
    lgv.run(state, pot, 2000)

    x2, f2, ndotf = [], [], []
    for i in range(40_000):
        lgv.step(state, pot)
        if i % 20 == 0:
            delta = state.positions - ref
            x2.append((delta**2).sum() / (3 * state.n_atoms))
            f2.append((state.forces**2).sum())
            # For an Einstein crystal div F = -3 N k analytically.
            ndotf.append(3.0 * state.n_atoms * spring)

    predicted = KB * temperature / spring
    mean = float(np.mean(x2))
    err = _block_error(x2)
    print(
        f"\n[Langevin harmonic] <x^2> = {mean:.6e} +/- {err:.1e} A^2"
        f"   kT/k = {predicted:.6e} A^2"
        f"   ratio = {mean / predicted:.5f} +/- {err / predicted:.5f}"
    )
    assert abs(mean - predicted) < 4.0 * err + 1e-3 * predicted

    # Configurational temperature <|F|^2>/(k_B <div F>) -- the diagnostic
    # docs/design.md asks for, and a genuinely independent one: it uses only
    # forces, so it is blind to the velocity distribution that the kinetic
    # temperature measures.
    t_conf = np.mean(f2) / (KB * np.mean(ndotf))
    print(f"    configurational temperature = {t_conf:.3f} K (target {temperature} K)")
    assert t_conf == pytest.approx(temperature, rel=0.02)


def test_langevin_requires_an_explicit_generator():
    cfg, pot, _ = _einstein_system(8, 0.5)
    state = MDState.from_configuration(cfg, pot, 50.0, seed=0, fixed_com=False)
    state.rng = None
    with pytest.raises(ValueError):
        Langevin(0.002, 50.0, 1.0).step(state, pot)


def test_stochastic_integrator_is_deterministic_given_a_seed():
    cfg, pot, _ = _einstein_system(32, 0.6)
    finals = []
    for seed in (7, 7, 8):
        state = MDState.from_configuration(cfg, pot, 90.0, seed=seed, fixed_com=False)
        Langevin(0.004, 90.0, 3.0, rng=np.random.default_rng(seed), virial=False).run(
            state, pot, 200
        )
        finals.append(state.positions.copy())
    assert np.array_equal(finals[0], finals[1])
    assert not np.allclose(finals[0], finals[2])


# --------------------------------------------------------------------------
# thermostat distributions
# --------------------------------------------------------------------------


def _sample_kinetic_energies(integrator, potential, cfg, *, n_eq, n_prod, stride, seed, t_init):
    state = MDState.from_configuration(cfg, potential, t_init, seed=seed)
    integrator.run(state, potential, n_eq)
    samples = []
    for i in range(n_prod):
        integrator.step(state, potential)
        if i % stride == 0:
            samples.append(state.kinetic_energy())
    return np.array(samples), state


def _report_kinetic_distribution(label, samples, dof, temperature):
    """Compare the sampled KE against Gamma(dof/2, kT) and print the numbers.

    The canonical kinetic energy of ``dof`` quadratic degrees of freedom is
    ``KE = (kT/2) chi^2_dof``, i.e. Gamma-distributed with shape ``dof/2`` and
    scale ``kT``.  Its mean is ``dof kT / 2`` and its variance ``dof (kT)^2 / 2``
    -- the variance is the discriminating moment, because a thermostat that
    merely rescales gets the mean right by construction.
    """
    kt = KB * temperature
    reference = stats.gamma(a=dof / 2.0, scale=kt)
    mean_pred, var_pred = dof * kt / 2.0, dof * kt**2 / 2.0
    mean, var = float(samples.mean()), float(samples.var(ddof=1))
    ks = stats.kstest(samples, reference.cdf)
    print(
        f"\n[{label}] n = {samples.size} decorrelated samples, dof = {dof}"
        f"\n    <T>       = {2 * mean / (dof * KB):.3f} K (target {temperature} K)"
        f"\n    mean(KE)  = {mean:.6f} eV vs {mean_pred:.6f}  ratio {mean / mean_pred:.4f}"
        f"\n    var(KE)   = {var:.4e} eV^2 vs {var_pred:.4e}  ratio {var / var_pred:.4f}"
        f"\n    KS statistic = {ks.statistic:.4f}, p = {ks.pvalue:.3g}"
    )
    return mean / mean_pred, var / var_pred, ks.pvalue


@pytest.mark.slow
@pytest.mark.physics
def test_langevin_samples_the_maxwell_boltzmann_kinetic_distribution():
    cfg, lj = _argon()
    temperature = 80.0
    dof = 3 * cfg.n_atoms - 3
    samples, state = _sample_kinetic_energies(
        Langevin(0.004, temperature, 1.0, virial=False),
        lj,
        cfg,
        n_eq=2500,
        n_prod=60_000,
        stride=250,
        seed=11,
        t_init=160.0,
    )
    mean_ratio, var_ratio, pvalue = _report_kinetic_distribution(
        "Langevin BAOAB", samples, dof, temperature
    )
    assert mean_ratio == pytest.approx(1.0, abs=0.02)
    assert 0.7 < var_ratio < 1.35
    assert pvalue > 0.005


@pytest.mark.slow
@pytest.mark.physics
def test_nhc_samples_the_maxwell_boltzmann_kinetic_distribution():
    cfg, lj = _argon()
    temperature = 80.0
    dof = 3 * cfg.n_atoms - 3
    samples, state = _sample_kinetic_energies(
        NoseHooverChain(0.004, temperature, 0.2, chain_length=3, n_sy=3, virial=False),
        lj,
        cfg,
        n_eq=2500,
        n_prod=60_000,
        stride=250,
        seed=11,
        t_init=160.0,
    )
    mean_ratio, var_ratio, pvalue = _report_kinetic_distribution(
        "Nose-Hoover chain", samples, dof, temperature
    )
    assert mean_ratio == pytest.approx(1.0, abs=0.02)
    assert 0.7 < var_ratio < 1.35
    assert pvalue > 0.005


@pytest.mark.slow
@pytest.mark.physics
def test_berendsen_gets_the_mean_right_and_the_distribution_wrong():
    """The negative control: this is what a plausible-looking failure looks like."""
    cfg, lj = _argon()
    temperature = 80.0
    dof = 3 * cfg.n_atoms - 3
    samples, _ = _sample_kinetic_energies(
        BerendsenThermostat(0.004, temperature, 0.1, virial=False),
        lj,
        cfg,
        n_eq=2500,
        n_prod=60_000,
        stride=250,
        seed=11,
        t_init=160.0,
    )
    mean_ratio, var_ratio, pvalue = _report_kinetic_distribution(
        "Berendsen (NOT canonical)", samples, dof, temperature
    )
    # The mean temperature -- the metric everybody reports -- is fine.
    assert mean_ratio == pytest.approx(1.0, abs=0.02)
    # The fluctuations, i.e. the heat capacity, are badly suppressed.
    assert var_ratio < 0.6
    assert pvalue < 1e-3


@pytest.mark.slow
@pytest.mark.physics
def test_berendsen_flying_ice_cube():
    """Energy leaks into the centre-of-mass modes while the thermometer looks fine.

    Seeded with a small drift (1.5% of the kinetic energy), an unprotected
    Berendsen run funnels energy into the three zero-frequency translational
    modes: rescaling multiplies *every* momentum by lambda, but only the
    internal modes give their share back to the potential energy.  The reported
    temperature stays on target throughout, which is the whole point.
    """
    cfg, lj = _argon()
    total_mass = float(cfg.masses.sum())

    def com_fraction(state):
        v = state.com_velocity()
        return 0.5 * MVV2E * total_mass * float(v @ v) / state.kinetic_energy()

    state = MDState.from_configuration(cfg, lj, 160.0, seed=9, fixed_com=False)
    state.velocities += np.array([0.5, 0.0, 0.0])
    thermostat = BerendsenThermostat(0.004, 80.0, 0.05, remove_com=False, virial=False)
    start = com_fraction(state)
    temps = []
    for i in range(30_000):
        thermostat.step(state, lj)
        if i % 1000 == 0:
            temps.append(state.temperature())
    end = com_fraction(state)
    print(
        f"\n[flying ice cube] COM kinetic fraction {start:.4f} -> {end:.4f}"
        f" over 120 ps, while <T> = {np.mean(temps):.2f} K (target 80 K)"
    )
    assert end > 4.0 * start
    assert np.mean(temps) == pytest.approx(80.0, rel=0.1)

    # With the standard patch applied the symptom disappears -- which is exactly
    # how a broken method survives in production.
    protected = MDState.from_configuration(cfg, lj, 160.0, seed=9, fixed_com=False)
    protected.velocities += np.array([0.5, 0.0, 0.0])
    BerendsenThermostat(0.004, 80.0, 0.05, remove_com=True, virial=False).run(
        protected, lj, 2000
    )
    assert com_fraction(protected) < 1e-20


# --------------------------------------------------------------------------
# Nose-Hoover chain: the conserved quantity
# --------------------------------------------------------------------------


def test_nhc_reaches_the_target_temperature():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 300.0, seed=12)
    nhc = NoseHooverChain(0.002, 80.0, 0.1, virial=False)
    nhc.run(state, lj, 4000)
    temps = []
    for _ in range(4000):
        nhc.step(state, lj)
        temps.append(state.temperature())
    mean = float(np.mean(temps))
    print(f"\n[NHC equilibration] <T> over the last 8 ps = {mean:.2f} K (target 80 K)")
    assert mean == pytest.approx(80.0, rel=0.06)


@pytest.mark.slow
@pytest.mark.physics
def test_nhc_conserved_quantity_drift_scales_as_dt_squared():
    """The strongest single test of a Nose-Hoover chain implementation."""
    cfg, lj = _argon()
    seed_state = MDState.from_configuration(cfg, lj, 160.0, seed=3)
    Langevin(0.002, 80.0, 2.0, virial=False).run(seed_state, lj, 3000)

    step_sizes = [0.008, 0.004, 0.002, 0.001]
    span = 10.0  # ps
    devs, slopes = [], []
    for dt in step_sizes:
        state = seed_state.copy()
        nhc = NoseHooverChain(dt, 80.0, 0.2, chain_length=3, n_sy=3, virial=False)
        nhc.step(state, lj)
        c0 = nhc.conserved_quantity(state)
        n_steps = int(round(span / dt))
        conserved = np.empty(n_steps)
        for i in range(n_steps):
            nhc.step(state, lj)
            conserved[i] = nhc.conserved_quantity(state)
        devs.append(float(np.abs(conserved - c0).max()) / state.n_atoms)
        slopes.append(
            abs(float(np.polyfit(np.arange(n_steps) * dt, conserved, 1)[0])) / state.n_atoms
        )

    order = _fitted_order(step_sizes, devs)
    print("\n[NHC conserved quantity, 10 ps runs] dt/fs, max |dH'|/atom, |drift|/atom/ps:")
    for dt, d, s in zip(step_sizes, devs, slopes):
        print(f"    {dt * 1000:5.1f}  {d:.4e}  {s:.4e}")
    print(f"    fitted exponent = {order:.4f}")
    assert devs[-1] < 1e-6  # eV/atom at dt = 1 fs
    assert 1.8 < order < 2.2


def test_nhc_chain_is_rebuilt_when_the_settings_change():
    """Chain masses depend on N_f, T and tau; stale ones must not be reused."""
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 80.0, seed=2)
    NoseHooverChain(0.002, 80.0, 0.1, virial=False).step(state, lj)
    q_first = state.thermostat["Q"].copy()
    NoseHooverChain(0.002, 80.0, 0.4, virial=False).step(state, lj)
    assert not np.allclose(q_first, state.thermostat["Q"])
    assert state.thermostat["Q"][1] == pytest.approx(q_first[1] * 16.0)


def test_single_nose_hoover_is_not_ergodic_for_a_harmonic_system():
    """Why the chain exists: chain_length = 1 fails on the Einstein crystal.

    A single Nose-Hoover thermostat coupled to a harmonic system has an extra
    conserved structure and samples a torus, not the canonical distribution.
    With one oscillator the failure is stark; a chain of three fixes it.
    """
    spring = 0.8
    temperature = 100.0
    cfg, pot, ref = _einstein_system(1, spring, seed=3)
    predicted = KB * temperature / spring

    ratios = {}
    for length in (1, 3):
        state = MDState.from_configuration(cfg, pot, temperature, seed=5, fixed_com=False)
        nhc = NoseHooverChain(0.002, temperature, 0.05, chain_length=length, virial=False)
        nhc.run(state, pot, 20_000)
        x2 = []
        for i in range(400_000):
            nhc.step(state, pot)
            if i % 10 == 0:
                x2.append(((state.positions - ref) ** 2).sum() / 3.0)
        ratios[length] = float(np.mean(x2)) / predicted
    print(
        f"\n[NHC ergodicity, single oscillator] <x^2>/(kT/k):"
        f" chain_length=1 -> {ratios[1]:.4f}, chain_length=3 -> {ratios[3]:.4f}"
    )
    assert ratios[3] == pytest.approx(1.0, rel=0.15)
    assert abs(ratios[1] - 1.0) > abs(ratios[3] - 1.0)


# --------------------------------------------------------------------------
# MTK barostat
# --------------------------------------------------------------------------


def test_mtk_conserved_quantity_is_stable():
    cfg, lj = _argon()
    state = MDState.from_configuration(cfg, lj, 80.0, seed=3)
    mtk = MTKBarostat(0.002, 80.0, 1000.0, 0.2, 1.0)
    mtk.step(state, lj)
    c0 = mtk.conserved_quantity(state)
    worst = 0.0
    for _ in range(2000):
        mtk.step(state, lj)
        worst = max(worst, abs(mtk.conserved_quantity(state) - c0))
    print(f"\n[MTK 4 ps] max |dH'| = {worst / state.n_atoms:.3e} eV/atom")
    assert worst / state.n_atoms < 1e-5


@pytest.mark.slow
@pytest.mark.physics
def test_mtk_conserved_quantity_scales_as_dt_squared_and_hits_the_setpoint():
    cfg, lj = _argon()
    seed_state = MDState.from_configuration(cfg, lj, 160.0, seed=3)
    Langevin(0.002, 80.0, 2.0, virial=False).run(seed_state, lj, 3000)

    step_sizes = [0.004, 0.002, 0.001]
    devs = []
    final = None
    for dt in step_sizes:
        state = seed_state.copy()
        mtk = MTKBarostat(dt, 80.0, 1000.0, 0.2, 1.0)
        mtk.step(state, lj)
        c0 = mtk.conserved_quantity(state)
        n_steps = int(round(10.0 / dt))
        worst = 0.0
        pressures, temps = [], []
        for i in range(n_steps):
            mtk.step(state, lj)
            worst = max(worst, abs(mtk.conserved_quantity(state) - c0))
            if i > n_steps // 2:
                pressures.append(state.pressure(unit="bar"))
                temps.append(state.temperature())
        devs.append(worst / state.n_atoms)
        final = (float(np.mean(pressures)), _block_error(pressures), float(np.mean(temps)))

    order = _fitted_order(step_sizes, devs)
    print("\n[MTK conserved quantity, 10 ps runs] dt/fs, max |dH'|/atom:")
    for dt, d in zip(step_sizes, devs):
        print(f"    {dt * 1000:5.1f}  {d:.4e}")
    print(f"    fitted exponent = {order:.4f}")
    print(
        f"    dt = 1 fs second half: <P> = {final[0]:.1f} +/- {final[1]:.1f} bar"
        f" (target 1000 bar), <T> = {final[2]:.2f} K (target 80 K)"
    )
    assert 1.8 < order < 2.2
    # The instantaneous pressure of a few hundred atoms fluctuates by kilobars,
    # so the tolerance here is set by statistics, not by the integrator.
    assert abs(final[0] - 1000.0) < max(300.0, 4.0 * final[1])
    assert final[2] == pytest.approx(80.0, rel=0.1)


def test_mtk_moves_the_volume_in_the_right_direction():
    """A compressive target pressure must shrink a cell that is under tension."""
    cfg, lj = _argon()
    v0 = cfg.volume
    state = MDState.from_configuration(cfg, lj, 20.0, seed=1)
    MTKBarostat(0.002, 20.0, 50_000.0, 0.1, 0.4).run(state, lj, 1500)
    print(f"\n[MTK compression] V: {v0:.1f} -> {state.volume:.1f} A^3 at 50 kbar")
    assert state.volume < v0

    state = MDState.from_configuration(cfg, lj, 20.0, seed=1)
    MTKBarostat(0.002, 20.0, -20_000.0, 0.1, 0.4).run(state, lj, 1500)
    assert state.volume > v0


def test_mtk_requires_periodic_boundaries():
    cfg, pot, _ = _einstein_system(8, 0.5)
    state = MDState.from_configuration(cfg, pot, 50.0, seed=0, fixed_com=False)
    with pytest.raises(ValueError):
        MTKBarostat(0.002, 50.0, 1.0, 0.1, 0.5).step(state, pot)
