"""Tests for :mod:`atomlab.potentials.perturbations`.

The non-negotiable test is the first one: analytic derivatives against central
differences, for every perturbation, on a rattled fcc cell, a rattled diamond
cell and a sheared triclinic cell.  A perturbation with subtly wrong forces
would still run in MD and would still produce a plausible-looking observable
shift -- and since this module *is* the instrument that measures how error
reaches observables, a wrong instrument would not fail loudly, it would quietly
produce the paper's conclusion.

Beyond that, the tests check the quantitative predictions the module exists to
make: the ``w^{3/2}`` width scaling of ``docs/theory.md`` section 4.1, the
covariance suppression of the null-space construction, and the amplification of
its aligned partner at matched force error.

Numbers that the report quotes are printed (run with ``-s`` to see them).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atomlab.build import diamond, fcc, random_gas, rattle, scale_cell, shear
from atomlab.cell import min_cell_width
from atomlab.neighbors import build_neighbor_list, pair_vectors
from atomlab.potentials.base import SumPotential, check_forces, check_virial
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import (
    AlignedPerturbation,
    AngularPerturbation,
    HighEnergyPerturbation,
    HighFrequencyPerturbation,
    LinearShellPerturbation,
    NullSpacePerturbation,
    PairPerturbation,
    RadialShellPerturbation,
    RandomShellPerturbation,
    ShellBasis,
    build_shell_design,
    observable_basis_covariance,
    perturbation_covariance,
    predicted_shift,
    radial_coupling,
    smooth_cutoff,
    _solve_weights,
)

CUTOFF = 5.0
#: Atoms sampled by the finite-difference force check.  The full 6N evaluation
#: is unnecessary -- a wrong derivative is wrong for every atom -- and this
#: keeps the three-body check to a second or so.
FD_ATOMS = (0, 3, 11)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _min_pair_distance(configuration, cutoff=CUTOFF) -> float:
    nl = build_neighbor_list(configuration, cutoff, half=True)
    _, r = pair_vectors(configuration, nl)
    return float(r.min())


@pytest.fixture(scope="module")
def rattled_fcc():
    """32-atom fcc argon, displaced enough to break every symmetry."""
    return rattle(fcc(5.26, "Ar", (2, 2, 2)), 0.12, seed=11)


@pytest.fixture(scope="module")
def rattled_diamond():
    """64-atom diamond silicon; three-body terms are non-trivial here."""
    return rattle(diamond(5.431, "Si", (2, 2, 2)), 0.10, seed=12)


@pytest.fixture(scope="module")
def sheared_triclinic():
    """A strongly sheared fcc cell: pairs are seen through skewed images."""
    cfg = shear(rattle(fcc(5.26, "Ar", (2, 2, 2)), 0.12, seed=13), 0.22, "xy")
    cfg = shear(cfg, 0.12, "xz")
    assert min_cell_width(cfg.cell, cfg.pbc) > 2.0 * CUTOFF
    return cfg


@pytest.fixture(scope="module")
def all_cells(rattled_fcc, rattled_diamond, sheared_triclinic):
    return {
        "rattled_fcc": rattled_fcc,
        "rattled_diamond": rattled_diamond,
        "sheared_triclinic": sheared_triclinic,
    }


def _pair_perturbations(configuration):
    """Every pair perturbation, sized so it has real support on ``configuration``."""
    r_min = _min_pair_distance(configuration)
    r0 = 0.5 * (r_min + CUTOFF)
    basis = ShellBasis.for_configurations(configuration, CUTOFF, 8)
    rng = np.random.default_rng(5)
    return {
        "shell": RadialShellPerturbation(r0, 0.35, 0.07, CUTOFF),
        "shell_wide": RadialShellPerturbation(r0, 0.9, -0.04, CUTOFF),
        "high_frequency": HighFrequencyPerturbation(0.55, 0.02, CUTOFF),
        "high_energy": HighEnergyPerturbation(1.12 * r_min, 0.03, CUTOFF),
        "linear_shell": LinearShellPerturbation(basis, 0.02 * rng.standard_normal(8)),
    }


# --------------------------------------------------------------------------
# 1. analytic vs numerical derivatives -- the load-bearing test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cell_name", ["rattled_fcc", "rattled_diamond", "sheared_triclinic"])
def test_pair_perturbation_forces_and_virial(all_cells, cell_name):
    """Analytic forces and virial of every pair perturbation match differences."""
    cfg = all_cells[cell_name]
    worst = {}
    for name, pert in _pair_perturbations(cfg).items():
        # A perturbation with no support would pass vacuously; refuse that.
        assert abs(pert.energy(cfg)) > 0.0, f"{name} has no support on {cell_name}"
        df = check_forces(pert, cfg, delta=1e-5, atoms=FD_ATOMS)
        dw = check_virial(pert, cfg, delta=1e-6)
        worst[name] = (df, dw)
        assert df < 1e-6, f"{name} on {cell_name}: force error {df:.3e}"
        assert dw < 1e-6, f"{name} on {cell_name}: virial error {dw:.3e}"
    print(f"\n[pair derivatives on {cell_name}]")
    for name, (df, dw) in worst.items():
        print(f"  {name:16s} max|dF| = {df:9.3e} eV/A   max|dW| = {dw:9.3e} eV")


@pytest.mark.parametrize("cell_name", ["rattled_fcc", "rattled_diamond", "sheared_triclinic"])
def test_angular_perturbation_forces_and_virial(all_cells, cell_name):
    """The three-body angular term differentiated with respect to all three atoms."""
    cfg = all_cells[cell_name]
    pert = AngularPerturbation(0.06, -1.0 / 3.0, CUTOFF)
    assert abs(pert.energy(cfg)) > 0.0
    df = check_forces(pert, cfg, delta=1e-5, atoms=FD_ATOMS)
    dw = check_virial(pert, cfg, delta=1e-6)
    print(f"\n[angular derivatives on {cell_name}] max|dF| = {df:.3e} eV/A, max|dW| = {dw:.3e} eV")
    assert df < 1e-6
    assert dw < 1e-6


@pytest.mark.parametrize("cell_name", ["rattled_fcc", "rattled_diamond", "sheared_triclinic"])
def test_angular_perturbation_at_a_non_tetrahedral_reference(all_cells, cell_name):
    """A different ``cos_theta0`` exercises a different part of the derivative."""
    cfg = all_cells[cell_name]
    pert = AngularPerturbation(-0.05, 0.25, CUTOFF)
    assert check_forces(pert, cfg, delta=1e-5, atoms=FD_ATOMS) < 1e-6
    assert check_virial(pert, cfg, delta=1e-6) < 1e-6


def test_forces_sum_to_zero_and_virial_is_symmetric(rattled_diamond):
    """Newton's third law and the absence of a spurious antisymmetric stress."""
    cfg = rattled_diamond
    perts = list(_pair_perturbations(cfg).values()) + [AngularPerturbation(0.06, -1 / 3, CUTOFF)]
    for pert in perts:
        res = pert.compute(cfg)
        assert np.max(np.abs(res.forces.sum(axis=0))) < 1e-10
        assert np.max(np.abs(res.virial - res.virial.T)) < 1e-9
        assert math.isclose(res.energies.sum(), res.energy, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------
# 2. everything vanishes, with its derivative, at the cutoff
# --------------------------------------------------------------------------


def test_smooth_cutoff_is_c2_at_both_ends():
    s, ds = smooth_cutoff(np.array([0.0, 3.0, 4.0, 5.0, 6.0]), 4.0, 5.0)
    assert np.allclose(s[:3], 1.0)
    assert np.allclose(ds[:3], 0.0)
    assert s[3] == 0.0 and ds[3] == 0.0
    assert s[4] == 0.0 and ds[4] == 0.0


@pytest.mark.parametrize(
    "name", ["shell", "shell_wide", "high_frequency", "high_energy", "linear_shell"]
)
def test_pair_perturbations_vanish_at_cutoff(rattled_fcc, name):
    """``delta_u`` and ``delta_u'`` are exactly zero at ``Rc`` and vanish smoothly.

    Exact zeros beyond the cutoff are checked directly.  The *order* of
    vanishing is checked by the decay of the one-sided values as the offset
    shrinks by ten: a ``C^2`` switch gives ``u ~ h^3`` and ``u' ~ h^2``, so the
    ratios must be near ``1e-3`` and ``1e-2``.  A bare truncation would give
    ratios of 1 (a jump), and a merely ``C^0`` switch would give 1 for ``u'``.
    """
    pert = _pair_perturbations(rattled_fcc)[name]
    rc = pert.cutoff

    beyond = np.array([rc, rc + 1e-9, rc + 0.1, rc + 5.0])
    u, du = pert.pair(beyond)
    assert np.all(u == 0.0)
    assert np.all(du == 0.0)

    if name == "high_energy":
        # Compactly supported below r_onset: check the vanishing there instead,
        # which is the boundary that actually carries the C^2 requirement.
        edge = pert.r_onset
    else:
        edge = rc

    u_coarse, du_coarse = pert.pair(np.array([edge - 1e-2]))
    u_fine, du_fine = pert.pair(np.array([edge - 1e-3]))
    scale = max(np.max(np.abs(pert.pair(np.linspace(0.5, rc, 400))[0])), 1e-30)
    ratio_u = abs(u_fine[0]) / max(abs(u_coarse[0]), 1e-300)
    ratio_du = abs(du_fine[0]) / max(abs(du_coarse[0]), 1e-300)
    print(
        f"\n[{name}] |u(rc-1e-3)|/|u(rc-1e-2)| = {ratio_u:.3e} (cubic -> 1e-3), "
        f"|u'| ratio = {ratio_du:.3e} (quadratic -> 1e-2)"
    )
    assert abs(u_coarse[0]) < 1e-4 * scale
    assert ratio_u < 3e-3
    assert ratio_du < 3e-2


def test_angular_perturbation_vanishes_at_cutoff():
    """A triplet whose third atom crosses ``Rc``: energy and force die smoothly.

    The geometry is chosen so that only the ``i``-centred triplet exists near
    the crossing -- ``j`` and ``k`` are more than ``Rc`` apart, so neither of
    them has two neighbours -- which isolates the ``S(r_ik)`` factor.
    """
    from atomlab.types import Configuration

    pert = AngularPerturbation(0.5, -1.0 / 3.0, CUTOFF)

    def energy_at(d: float) -> float:
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, d, 0.0]])
        cfg = Configuration(positions=pos, cell=None, pbc=False, symbols=("Si",))
        return pert.energy(cfg)

    # Sanity: the j-k separation stays outside the cutoff over the whole scan.
    assert math.hypot(2.0, CUTOFF - 0.02) > CUTOFF

    assert energy_at(CUTOFF) == 0.0
    assert energy_at(CUTOFF + 0.5) == 0.0
    e_coarse = energy_at(CUTOFF - 1e-2)
    e_fine = energy_at(CUTOFF - 1e-3)
    assert e_coarse != 0.0
    ratio = abs(e_fine) / abs(e_coarse)
    # dE/dd by central difference just inside the cutoff must also be tiny.
    h = 1e-5
    slope = (energy_at(CUTOFF - 1e-3 + h) - energy_at(CUTOFF - 1e-3 - h)) / (2 * h)
    print(f"\n[angular cutoff] E ratio = {ratio:.3e} (cubic -> 1e-3), dE/dd = {slope:.3e} eV/A")
    assert ratio < 3e-3
    assert abs(slope) < 1e-4


def test_high_energy_vanishes_at_onset_and_explodes_under_compression(rattled_fcc):
    """Invisible in-distribution, divergent out of it (theory section 5)."""
    cfg = rattled_fcc
    r_min = _min_pair_distance(cfg)
    # cutoff defaults to r_onset: the support is all the neighbour search needs,
    # which also keeps the minimum-image guard happy on the compressed cell.
    pert = HighEnergyPerturbation(0.90 * r_min, 1.0)

    # Nothing at all on the reference configuration ...
    assert pert.energy(cfg) == 0.0
    assert pert.force_rms(cfg) == 0.0

    # ... still nothing at 10% compression ...
    assert pert.energy(scale_cell(cfg, 0.90)) == 0.0

    # ... and then it explodes.  The point is the *rate*: a further 15% of
    # linear compression multiplies the error by three orders of magnitude,
    # which is exactly the behaviour no in-distribution metric can anticipate.
    print("\n[high energy] linear scale -> energy (eV), force RMSE (eV/A)")
    energies = {}
    for scale_factor in (1.00, 0.90, 0.85, 0.80, 0.75, 0.70):
        comp = scale_cell(cfg, scale_factor)
        energies[scale_factor] = pert.energy(comp)
        print(
            f"  x{scale_factor:.2f}: E = {energies[scale_factor]:10.5f}   "
            f"RMSE_F = {pert.force_rms(comp):9.5f}"
        )
    assert energies[0.70] > 0.5
    assert energies[0.70] / energies[0.85] > 100.0

    compressed = scale_cell(cfg, 0.75)
    assert check_forces(pert, compressed, delta=1e-6, atoms=FD_ATOMS) < 1e-6
    assert check_virial(pert, compressed, delta=1e-6) < 1e-6

    # C^2 at the onset: u, u' and u'' all vanish there.
    onset = pert.r_onset
    u, du = pert.pair(np.array([onset - 1e-2, onset - 1e-3, onset, onset + 1e-3]))
    assert u[2] == 0.0 and du[2] == 0.0 and u[3] == 0.0 and du[3] == 0.0
    assert abs(u[1]) / abs(u[0]) < 3e-3


# --------------------------------------------------------------------------
# 3. matched force error
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gas_ensemble():
    """Dense-gas frames with a smooth ``g(r)`` around 4.5 A.

    The width-scaling argument assumes the pair density is roughly flat across
    the shell; a crystalline ensemble has ``g(r)`` peaks narrower than the
    shells being compared, which would contaminate the exponent with structure
    rather than with the effect being measured.
    """
    cfgs = [random_gas(64, 0.0212, "Ar", seed=1000 + i, min_distance=2.9) for i in range(40)]
    assert min_cell_width(cfgs[0].cell, cfgs[0].pbc) > 2.0 * 6.5
    return cfgs


def test_matched_force_error_hits_the_target(gas_ensemble):
    """The amplitude solve is exact in sample and close out of sample."""
    train = gas_ensemble[:20]
    held_out = gas_ensemble[20:]
    target = 0.05
    print("\n[matched force error] target = 0.0500 eV/A")
    for width in (0.10, 0.20, 0.35):
        pert = RadialShellPerturbation.matched_force_error(
            width, target, train, r0=4.5, cutoff=6.5
        )
        achieved = pert.force_rms(train)
        out = pert.force_rms(held_out)
        print(
            f"  w = {width:.2f} A: amplitude = {pert.amplitude:.5f} eV, "
            f"in-sample = {achieved:.6f}, held-out = {out:.6f} eV/A "
            f"({100 * abs(out - target) / target:.2f}% off)"
        )
        assert abs(achieved - target) / target < 1e-10
        assert abs(out - target) / target < 0.05

    # The generic version on a different family.
    hf = HighFrequencyPerturbation(0.6, 1.0, 6.5).matched_to_force_rms(target, train)
    assert abs(hf.force_rms(train) - target) / target < 1e-10

    with pytest.raises(ValueError):
        # A shell far below any sampled separation carries no force.
        RadialShellPerturbation.matched_force_error(0.05, target, train, r0=0.6, cutoff=6.5)


# --------------------------------------------------------------------------
# 4. the width-scaling prediction of theory section 4.1
# --------------------------------------------------------------------------


def _measured_g(configurations, cutoff, n_bins=90):
    """Crude histogram RDF of an ensemble, for the coupling weight."""
    edges = np.linspace(0.0, cutoff, n_bins + 1)
    counts = np.zeros(n_bins)
    n_atoms = 0
    volume = 0.0
    for cfg in configurations:
        nl = build_neighbor_list(cfg, cutoff, half=True)
        _, r = pair_vectors(cfg, nl)
        counts += np.histogram(r, bins=edges)[0]
        n_atoms += cfg.n_atoms
        volume += cfg.volume
    density = n_atoms / volume
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = 0.5 * n_atoms * density * shell
    g = np.divide(counts, ideal, out=np.zeros_like(counts), where=ideal > 0)
    return lambda r: np.interp(r, centers, g, left=0.0, right=g[-1])


@pytest.mark.slow
def test_width_scaling_exponent(gas_ensemble):
    """At matched force error, coupling to ``g(r)`` scales as ``w^{3/2}``.

    Equation (4.5) of ``docs/theory.md``, measured directly.  Two exponents are
    reported: the amplitude needed to hold the force error fixed (predicted
    ``w^{1/2}``) and the resulting radial coupling (predicted ``w^{3/2}``).
    """
    cfgs = gas_ensemble
    cutoff, r0, target = 6.5, 4.5, 0.05
    density = 64 / cfgs[0].volume
    g_fn = _measured_g(cfgs, cutoff)

    widths = np.array([0.08, 0.12, 0.18, 0.27])
    amplitudes, couplings, couplings_g, force_rms = [], [], [], []
    for w in widths:
        pert = RadialShellPerturbation.matched_force_error(
            w, target, cfgs, r0=r0, cutoff=cutoff
        )
        # The bump must be untruncated for the derivation to apply.
        assert r0 + 3 * w < pert.r_on and r0 - 3 * w > 2.5
        amplitudes.append(pert.amplitude)
        force_rms.append(pert.force_rms(cfgs))
        couplings.append(abs(radial_coupling(pert, density=density, r_min=2.0)))
        couplings_g.append(abs(radial_coupling(pert, density=density, g_of_r=g_fn, r_min=2.0)))

    log_w = np.log(widths)
    slope_a = float(np.polyfit(log_w, np.log(amplitudes), 1)[0])
    slope_c = float(np.polyfit(log_w, np.log(couplings), 1)[0])
    slope_cg = float(np.polyfit(log_w, np.log(couplings_g), 1)[0])

    print("\n[width scaling at fixed force RMSE = 0.05 eV/A]")
    print("   w (A)   amplitude (eV)  RMSE_F (eV/A)   coupling (eV/atom)  coupling x g(r)")
    for w, a, f, c, cg in zip(widths, amplitudes, force_rms, couplings, couplings_g):
        print(f"  {w:6.3f}  {a:14.6f}  {f:13.6f}  {c:18.6e}  {cg:15.6e}")
    print(f"  fitted amplitude exponent = {slope_a:.4f}   (predicted 0.5)")
    print(f"  fitted coupling  exponent = {slope_c:.4f}   (predicted 1.5)")
    print(f"  same with measured g(r)   = {slope_cg:.4f}   (predicted 1.5)")

    assert np.allclose(force_rms, target, rtol=1e-10)
    assert abs(slope_a - 0.5) < 0.1
    assert abs(slope_c - 1.5) < 0.15
    assert abs(slope_cg - 1.5) < 0.25

    # Quadrature convergence: the reported couplings are not integration error.
    fine = abs(
        radial_coupling(
            RadialShellPerturbation.matched_force_error(
                widths[0], target, cfgs, r0=r0, cutoff=cutoff
            ),
            density=density,
            r_min=2.0,
            n_quad=1024,
        )
    )
    assert abs(fine - couplings[0]) / couplings[0] < 1e-8


def test_high_frequency_error_is_a_false_alarm(gas_ensemble):
    """At equal force error, an oscillatory error couples far less than a shell.

    This is the "false alarm" of theory section 4 made quantitative: the two
    perturbations are indistinguishable to a force-RMSE-based evaluation and
    differ by orders of magnitude in what they do to the radial distribution.
    """
    cfgs = gas_ensemble
    density = 64 / cfgs[0].volume
    target = 0.05
    shell = RadialShellPerturbation.matched_force_error(
        0.35, target, cfgs, r0=4.5, cutoff=6.5
    )
    hf = HighFrequencyPerturbation(0.45, 1.0, 6.5).matched_to_force_rms(target, cfgs)
    c_shell = abs(radial_coupling(shell, density=density, r_min=2.0))
    c_hf = abs(radial_coupling(hf, density=density, r_min=2.0))
    print(
        f"\n[false alarm] both at RMSE_F = {target} eV/A: shell coupling = {c_shell:.4e}, "
        f"high-frequency coupling = {c_hf:.4e} eV/atom, ratio = {c_shell / c_hf:.1f}"
    )
    assert c_shell / c_hf > 20.0


# --------------------------------------------------------------------------
# 5. the null-space / aligned pair -- the sharpest falsification test
# --------------------------------------------------------------------------


def _binned_gr(cfg):
    """A coarse ``g(r)``: six bin occupancies between 3 and 5 A."""
    nl = build_neighbor_list(cfg, CUTOFF, half=True)
    _, r = pair_vectors(cfg, nl)
    return np.histogram(r, bins=6, range=(3.0, 5.0))[0].astype(float)


@pytest.fixture(scope="module")
def designed_setup():
    """Reference and held-out ensembles, a shared basis, and a vector observable.

    The observable is evaluated once on each ensemble and passed to the
    constructions as an array; every construction then sees *exactly* the same
    covariance estimate, which is what makes "matched force error, different
    covariance" a controlled comparison rather than two separate experiments.
    """
    base = fcc(5.26, "Ar", (2, 2, 2))
    train = [rattle(base, 0.22, seed=100 + i) for i in range(600)]
    test = [rattle(base, 0.22, seed=500_000 + i) for i in range(400)]

    a_train = np.array([_binned_gr(c) for c in train])
    a_test = np.array([_binned_gr(c) for c in test])

    basis = ShellBasis.for_configurations(train[:20], CUTOFF, 24)
    design = build_shell_design(train, basis)
    return {
        "train": train,
        "test": test,
        "a_train": a_train,
        "a_test": a_test,
        "basis": basis,
        "design": design,
    }


def _designed(cls, setup, seed, target=0.08):
    return cls(
        setup["train"],
        setup["a_train"],
        150.0,
        cutoff=CUTOFF,
        basis=setup["basis"],
        design=setup["design"],
        target_force_rms=target,
        seed=seed,
    )


def _cov_norm(pert, a_samples, cfgs):
    """``||Cov(A, delta_U)||`` with a precomputed observable matrix."""
    du = np.array([pert.energy(c) for c in cfgs])
    return float(np.linalg.norm(observable_basis_covariance(a_samples, du[:, None])[:, 0]))


def test_designed_perturbation_derivatives(all_cells, designed_setup):
    """The designed error fields are ordinary potentials and must behave like it."""
    perts = [
        _designed(cls, designed_setup, seed=3)
        for cls in (NullSpacePerturbation, AlignedPerturbation, RandomShellPerturbation)
    ]
    for cell_name, cfg in all_cells.items():
        for pert in perts:
            df = check_forces(pert, cfg, delta=1e-5, atoms=FD_ATOMS)
            dw = check_virial(pert, cfg, delta=1e-6)
            assert df < 1e-6, f"{pert.name} on {cell_name}: {df:.3e}"
            assert dw < 1e-6, f"{pert.name} on {cell_name}: {dw:.3e}"
            u, du = pert.pair(np.array([CUTOFF, CUTOFF + 0.5]))
            assert np.all(u == 0.0) and np.all(du == 0.0)


@pytest.mark.slow
def test_null_space_and_aligned_suppression(designed_setup):
    """Cov(A, delta_U) is numerically zero for the null-space construction.

    Three error fields with **identical force RMSE** on the reference frames:
    one projected onto the null space of ``Cov(A, .)``, one aligned with it,
    one random.  Force RMSE cannot tell them apart; the covariance -- and
    hence, by Result 1a, the observable error -- differs by many orders of
    magnitude.  That is falsification criterion 3 of theory section 8, run
    forwards.

    Both an in-sample and a held-out suppression factor are reported.  The
    in-sample number is what the construction guarantees (orthogonality is
    exact on the frames the covariance was measured on); the held-out number is
    limited by the ``1/sqrt(M)`` sampling error of that covariance estimate and
    is the honest figure for a practitioner.
    """
    train, test = designed_setup["train"], designed_setup["test"]
    a_train, a_test = designed_setup["a_train"], designed_setup["a_test"]
    seeds = (3, 17, 101, 202)

    in_sample = {"null": [], "aligned": [], "random": []}
    out_sample = {"null": [], "aligned": [], "random": []}
    force_rms = []
    report = None

    for seed in seeds:
        perts = {
            "null": _designed(NullSpacePerturbation, designed_setup, seed),
            "aligned": _designed(AlignedPerturbation, designed_setup, seed),
            "random": _designed(RandomShellPerturbation, designed_setup, seed),
        }
        report = perts["null"].report()
        for key, pert in perts.items():
            force_rms.append(pert.design_force_rms)
            in_sample[key].append(_cov_norm(pert, a_train, train))
            out_sample[key].append(_cov_norm(pert, a_test, test))

    # Matched force error is an equality, not an approximation.
    assert np.allclose(force_rms, 0.08, rtol=1e-9)

    med = {k: {n: float(np.median(v[n])) for n in v} for k, v in
           (("in", in_sample), ("out", out_sample))}
    supp_in = med["in"]["random"] / med["in"]["null"]
    supp_out = med["out"]["random"] / med["out"]["null"]
    ampl_in = med["in"]["aligned"] / med["in"]["random"]
    ampl_out = med["out"]["aligned"] / med["out"]["random"]

    print("\n[null-space / aligned pair, all at RMSE_F = 0.0800 eV/A exactly]")
    print(f"  design frames = {report['n_frames']}, basis = {report['n_basis']} shells, "
          f"observable dim = {report['observable_dim']}, "
          f"covariance rank = {report['covariance_rank']}, "
          f"null dimension = {report['null_dimension']}")
    print("  median ||Cov(A, dU)||  (over 4 random draws)")
    for key in ("null", "aligned", "random"):
        print(f"    {key:8s} in-sample = {med['in'][key]:.6e}   held-out = {med['out'][key]:.6e}")
    print(f"  SUPPRESSION  null vs random: in-sample = {supp_in:.3e}, held-out = {supp_out:.2f}")
    print(f"  AMPLIFICATION aligned vs random: in-sample = {ampl_in:.2f}, held-out = {ampl_out:.2f}")
    print(f"  aligned/null covariance ratio (in-sample) = "
          f"{med['in']['aligned'] / med['in']['null']:.3e}")

    # The required claim: at least an order of magnitude, by direct computation.
    assert supp_in > 10.0
    assert supp_in > 1e6, "in-sample orthogonality should be at round-off level"
    # Held-out suppression is finite and improves as sqrt(M); assert a floor
    # rather than a value, and report the number above.
    assert supp_out > 2.5
    # Aligned must be substantially worse than random at the same force error.
    assert ampl_in > 3.0
    assert ampl_out > 3.0


def test_null_space_predicted_shift_is_zero(designed_setup):
    """The first-order predicted observable shift vanishes for the null field."""
    train, a_train = designed_setup["train"], designed_setup["a_train"]
    null = _designed(NullSpacePerturbation, designed_setup, seed=3)
    aligned = _designed(AlignedPerturbation, designed_setup, seed=3)
    d_null = predicted_shift(null, a_train, train, 150.0)
    d_aligned = predicted_shift(aligned, a_train, train, 150.0)
    print(
        f"\n[first-order shift at T = 150 K] ||D<A>||: null = {np.linalg.norm(d_null):.3e}, "
        f"aligned = {np.linalg.norm(d_aligned):.3e} (bin counts)"
    )
    assert np.linalg.norm(d_null) < 1e-8
    assert np.linalg.norm(d_aligned) > 1.0


def test_aligned_direction_is_invariant_to_svd_sign(monkeypatch):
    """The signed aligned field must not inherit LAPACK's arbitrary SVD sign."""
    covariance = np.array([[2.0, -1.0, 0.5], [0.2, 1.5, -0.3]])
    transform = np.eye(3)
    original_svd = np.linalg.svd

    y_reference, _, _ = _solve_weights(
        covariance, transform, "aligned", np.random.default_rng(1), rcond=1e-12
    )

    def sign_flipped_svd(matrix, full_matrices=True):
        u, sv, vt = original_svd(matrix, full_matrices=full_matrices)
        u = u.copy()
        vt = vt.copy()
        u[:, 0] *= -1.0
        vt[0] *= -1.0
        return u, sv, vt

    monkeypatch.setattr(np.linalg, "svd", sign_flipped_svd)
    y_flipped, _, _ = _solve_weights(
        covariance, transform, "aligned", np.random.default_rng(1), rcond=1e-12
    )

    assert np.allclose(y_reference, y_flipped, atol=1e-14)
    projected = covariance @ y_flipped
    anchor = int(np.argmax(np.abs(projected)))
    assert projected[anchor] < 0.0


def test_designed_construction_is_deterministic(designed_setup):
    a = _designed(NullSpacePerturbation, designed_setup, seed=7)
    b = _designed(NullSpacePerturbation, designed_setup, seed=7)
    c = _designed(NullSpacePerturbation, designed_setup, seed=8)
    assert np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.weights, c.weights)


def test_scalar_observable_null_space(designed_setup):
    """The construction also works for a scalar observable (L = 1)."""
    train = designed_setup["train"]
    basis, design = designed_setup["basis"], designed_setup["design"]

    def coordination(cfg):
        nl = build_neighbor_list(cfg, CUTOFF, half=True)
        _, r = pair_vectors(cfg, nl)
        return float(np.sum(r < 4.3)) / cfg.n_atoms

    kw = dict(cutoff=CUTOFF, basis=basis, design=design, target_force_rms=0.08, seed=21)
    null = NullSpacePerturbation(train, coordination, 150.0, **kw)
    aligned = AlignedPerturbation(train, coordination, 150.0, **kw)
    rand = RandomShellPerturbation(train, coordination, 150.0, **kw)
    assert null.observable_dim == 1
    assert null.null_dimension == basis.n_functions - 1
    c_null = abs(float(perturbation_covariance(null, coordination, train)[0]))
    c_rand = abs(float(perturbation_covariance(rand, coordination, train)[0]))
    print(f"\n[scalar observable] in-sample suppression = {c_rand / c_null:.3e}")
    assert c_rand / c_null > 1e6
    # The sign convention makes a one-sided aligned-minus-null rule portable:
    # Cov <= 0, so the leading shift -beta*Cov is >= 0.
    assert aligned.design_covariance[0] < 0.0
    assert predicted_shift(aligned, coordination, train, 150.0)[0] > 0.0


def test_null_space_refuses_impossible_requests(designed_setup):
    """No silent fallbacks: too few shells for the observable dimension raises."""
    train = designed_setup["train"]
    a_train = designed_setup["a_train"]

    with pytest.raises(ValueError, match="null space is empty"):
        NullSpacePerturbation(
            train[:100],
            a_train[:100],  # 6 components
            150.0,
            cutoff=CUTOFF,
            n_functions=4,  # fewer well-conditioned directions than components
            target_force_rms=0.05,
            seed=0,
        )

    with pytest.raises(ValueError, match="no measurable covariance"):
        NullSpacePerturbation(
            train[:50],
            lambda cfg: 1.0,  # constant observable: zero covariance with everything
            150.0,
            cutoff=CUTOFF,
            n_functions=12,
            target_force_rms=0.05,
            seed=0,
        )


# --------------------------------------------------------------------------
# 6. composition with a reference potential
# --------------------------------------------------------------------------


def test_sum_potential_matches_separate_evaluation(rattled_fcc):
    """``U_0 + delta_U`` is what the experiments actually run; check the algebra."""
    cfg = rattled_fcc
    reference = LennardJones.argon(cutoff=CUTOFF, mode="shifted_force")
    for pert in (
        RadialShellPerturbation(4.0, 0.35, 0.05, CUTOFF),
        AngularPerturbation(0.04, -1 / 3, CUTOFF),
        HighFrequencyPerturbation(0.6, 0.01, CUTOFF),
    ):
        total = reference + pert
        assert isinstance(total, SumPotential)
        r_total = total.compute(cfg)
        r_ref = reference.compute(cfg)
        r_pert = pert.compute(cfg)

        assert abs(r_total.energy - (r_ref.energy + r_pert.energy)) < 1e-12
        assert np.max(np.abs(r_total.forces - (r_ref.forces + r_pert.forces))) < 1e-12
        assert np.max(np.abs(r_total.virial - (r_ref.virial + r_pert.virial))) < 1e-12

        # And the sum is still a correct potential in its own right.
        assert check_forces(total, cfg, delta=1e-5, atoms=FD_ATOMS) < 1e-6
        assert check_virial(total, cfg, delta=1e-6) < 1e-6

        # delta_U recovered by subtraction is exactly the perturbation energy.
        assert abs((total - reference).energy(cfg) - pert.energy(cfg)) < 1e-12


def test_rescaling_is_exactly_linear(rattled_fcc):
    cfg = rattled_fcc
    for pert in (
        RadialShellPerturbation(4.0, 0.3, 0.05, CUTOFF),
        AngularPerturbation(0.04, -1 / 3, CUTOFF),
        HighFrequencyPerturbation(0.6, 0.01, CUTOFF),
        HighEnergyPerturbation(0.95 * CUTOFF, 0.01, CUTOFF),
    ):
        doubled = pert.rescaled(2.0)
        assert type(doubled) is type(pert)
        assert abs(doubled.energy(cfg) - 2.0 * pert.energy(cfg)) < 1e-12
        assert np.max(np.abs(doubled.forces(cfg) - 2.0 * pert.forces(cfg))) < 1e-12
        # The original is untouched (rescaled returns a copy).
        assert pert.energy(cfg) != doubled.energy(cfg) or pert.energy(cfg) == 0.0


# --------------------------------------------------------------------------
# 7. plumbing
# --------------------------------------------------------------------------


def test_shell_design_gram_matches_direct_force_rms(rattled_fcc):
    """The Gram-matrix force RMSE agrees with evaluating the potential."""
    cfgs = [rattle(rattled_fcc, 0.05, seed=200 + i) for i in range(12)]
    basis = ShellBasis.for_configurations(cfgs, CUTOFF, 10)
    design = build_shell_design(cfgs, basis)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = 0.01 * rng.standard_normal(basis.n_functions)
        pert = LinearShellPerturbation(basis, w)
        assert abs(design.force_rms(w) - pert.force_rms(cfgs)) < 1e-12
        direct = np.array([pert.energy(c) for c in cfgs])
        assert np.max(np.abs(design.perturbation_energies(w) - direct)) < 1e-12


def test_observable_basis_covariance_matches_numpy():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((50, 3))
    b = rng.standard_normal((50, 7))
    got = observable_basis_covariance(a, b)
    want = np.cov(np.hstack([a, b]).T, ddof=1)[:3, 3:]
    assert np.max(np.abs(got - want)) < 1e-12
    # Scalar observables are accepted as a plain (M,) array.
    assert observable_basis_covariance(a[:, 0], b).shape == (1, 7)


def test_minimum_image_guard_raises(rattled_fcc):
    """A cutoff that does not fit in the cell must raise, never truncate."""
    small = fcc(5.26, "Ar", (1, 1, 1))
    pert = RadialShellPerturbation(4.0, 0.3, 0.05, CUTOFF)
    with pytest.raises(ValueError, match="too large for this cell"):
        pert.energy(small)
    with pytest.raises(ValueError, match="too large for this cell"):
        AngularPerturbation(0.05, -1 / 3, CUTOFF).energy(small)


def test_constructor_validation():
    with pytest.raises(ValueError):
        RadialShellPerturbation(4.0, -0.1, 0.05, 5.0)
    with pytest.raises(ValueError):
        RadialShellPerturbation(4.0, 0.3, 0.05, 5.0, r_on=6.0)
    with pytest.raises(ValueError):
        HighFrequencyPerturbation(0.0, 0.01, 5.0)
    with pytest.raises(ValueError):
        HighEnergyPerturbation(4.0, 0.01, 3.0)
    with pytest.raises(ValueError):
        AngularPerturbation(0.05, 1.5, 5.0)
    with pytest.raises(ValueError):
        ShellBasis.uniform(5.0, 6, r_min=4.5, r_max=4.0)
    with pytest.raises(ValueError):
        LinearShellPerturbation(ShellBasis.uniform(5.0, 6, r_min=2.0), np.ones(5))
    with pytest.raises(ValueError):
        smooth_cutoff(np.array([1.0]), 5.0, 5.0)


def test_perturbation_is_translation_and_rotation_invariant(rattled_fcc):
    """Sanity: these are ordinary potentials of interatomic distances."""
    cfg = rattled_fcc
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for pert in (
        RadialShellPerturbation(4.0, 0.3, 0.05, CUTOFF),
        AngularPerturbation(0.04, -1 / 3, CUTOFF),
    ):
        e0 = pert.energy(cfg)
        assert abs(pert.energy(cfg.translated([1.3, -0.7, 2.2])) - e0) < 1e-10
        assert abs(pert.energy(cfg.rotated(rot)) - e0) < 1e-10


def test_pair_perturbation_is_an_abstract_hook():
    class Broken(PairPerturbation):
        pass

    with pytest.raises(NotImplementedError):
        Broken(5.0).pair(np.array([1.0]))
