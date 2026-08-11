"""Validation of the two-body reference potentials.

These are the ground-truth physics of the whole study, so the tests here are
not smoke tests: they check analytic derivatives against finite differences,
the jitted kernels against independent NumPy implementations, exact symmetry
invariances, the behaviour of every cutoff mode at the cutoff, the
Lennard-Jones fcc lattice sum against its literature value, and the whole thing
against ASE's independent implementation.

``ase`` is imported inside this file only -- it is a test-only cross-check and
must never appear inside ``atomlab``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atomlab.build import fcc, random_gas, rattle, shear
from atomlab.cell import min_cell_width
from atomlab.potentials.base import check_forces, check_virial
from atomlab.potentials.harmonic import EinsteinCrystal, HarmonicPair, harmonic_pair
from atomlab.potentials.lennard_jones import (
    ARGON_CUTOFF,
    ARGON_EPSILON,
    ARGON_SIGMA,
    CUTOFF_MODES,
    FCC_LATTICE_SUM_A6,
    FCC_LATTICE_SUM_A12,
    FCC_MIN_A0_OVER_SIGMA,
    FCC_MIN_ENERGY_PER_EPSILON,
    FCC_MIN_NN_OVER_SIGMA,
    LennardJones,
    lj_pair,
    mix_pair_parameters,
    switching_function,
)
from atomlab.potentials.morse import Morse, morse_pair
from atomlab.types import Configuration
from atomlab.units import MVV2E

# Argon fcc lattice constant near the LJ minimum, in A.
A_FCC = FCC_MIN_A0_OVER_SIGMA * ARGON_SIGMA  # 5.2496 A


# --------------------------------------------------------------------------
# configurations under test
# --------------------------------------------------------------------------


def rattled_fcc(reps: int = 3, sigma: float = 0.12, seed: int = 11) -> Configuration:
    """Thermally disordered fcc argon: the condensed-phase test case."""
    return rattle(fcc(A_FCC, "Ar", reps), sigma, seed=seed)


def gas(n: int = 64, seed: int = 17) -> Configuration:
    """Dilute random gas: samples the repulsive wall and long distances."""
    return random_gas(n, 0.016, "Ar", seed=seed, min_distance=2.9)


def triclinic(reps: int = 3, gamma: float = 0.22, seed: int = 23) -> Configuration:
    """Strongly sheared cell: exercises the non-orthogonal image machinery."""
    return shear(rattled_fcc(reps, 0.10, seed), gamma, "xz")


def open_pair(distance: float) -> Configuration:
    """Two argon atoms at a given separation along x, no periodicity."""
    return Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        cell=None,
        pbc=False,
        symbols=("Ar",),
    )


CONFIG_FACTORIES = {
    "rattled_fcc": rattled_fcc,
    "gas": gas,
    "triclinic": triclinic,
}


def small_cutoff(configuration: Configuration, requested: float) -> float:
    """Largest usable cutoff: honour the minimum-image bound with a margin."""
    return min(requested, 0.45 * min_cell_width(configuration.cell, configuration.pbc))


# --------------------------------------------------------------------------
# analytic vs numerical derivatives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_lennard_jones_forces_and_virial(config_name, mode):
    cfg = CONFIG_FACTORIES[config_name]()
    lj = LennardJones.argon(cutoff=small_cutoff(cfg, 6.0), mode=mode)
    assert check_forces(lj, cfg, atoms=range(6)) < 1e-6
    assert check_virial(lj, cfg) < 1e-6


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_morse_forces_and_virial(config_name, mode):
    cfg = CONFIG_FACTORIES[config_name]()
    mo = Morse.argon(cutoff=small_cutoff(cfg, 6.0), mode=mode)
    assert check_forces(mo, cfg, atoms=range(6)) < 1e-6
    assert check_virial(mo, cfg) < 1e-6


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
def test_harmonic_pair_forces_and_virial(config_name):
    cfg = CONFIG_FACTORIES[config_name]()
    hp = HarmonicPair(0.8, 3.7, small_cutoff(cfg, 5.0))
    assert check_forces(hp, cfg, atoms=range(6)) < 1e-6
    assert check_virial(hp, cfg) < 1e-6


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
def test_einstein_crystal_forces_and_virial(config_name):
    cfg = CONFIG_FACTORIES[config_name]()
    reference = cfg.positions + 0.05  # sites deliberately offset from the atoms
    ec = EinsteinCrystal(reference, 1.7)
    assert check_forces(ec, cfg) < 1e-6
    assert check_virial(ec, cfg) < 1e-6


@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_multispecies_forces_and_virial(mode):
    """Lorentz-Berthelot mixing must be differentiated correctly too."""
    cfg = rattled_fcc(3, 0.12, seed=31)
    rng = np.random.default_rng(5)
    cfg.species = rng.integers(0, 3, size=cfg.n_atoms).astype(np.int32)
    cfg.symbols = ("Ar", "Ne", "Kr")
    lj = LennardJones(
        [0.0103, 0.0031, 0.0140],
        [3.405, 2.780, 3.640],
        small_cutoff(cfg, 6.0),
        mode=mode,
    )
    assert check_forces(lj, cfg, atoms=range(8)) < 1e-6
    assert check_virial(lj, cfg) < 1e-6


# --------------------------------------------------------------------------
# jitted kernel vs pure-NumPy reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_lj_kernel_matches_numpy_reference(config_name, mode):
    cfg = CONFIG_FACTORIES[config_name]()
    rc = small_cutoff(cfg, 6.0)
    jit = LennardJones.argon(cutoff=rc, mode=mode, kernel="numba").compute(cfg)
    ref = LennardJones.argon(cutoff=rc, mode=mode, kernel="numpy").compute(cfg)
    assert abs(jit.energy - ref.energy) < 1e-12 * max(1.0, abs(ref.energy))
    assert np.max(np.abs(jit.forces - ref.forces)) < 1e-12
    assert np.max(np.abs(jit.virial - ref.virial)) < 1e-12
    assert np.max(np.abs(jit.energies - ref.energies)) < 1e-13


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_morse_kernel_matches_numpy_reference(config_name, mode):
    cfg = CONFIG_FACTORIES[config_name]()
    rc = small_cutoff(cfg, 6.0)
    jit = Morse.argon(cutoff=rc, mode=mode, kernel="numba").compute(cfg)
    ref = Morse.argon(cutoff=rc, mode=mode, kernel="numpy").compute(cfg)
    assert abs(jit.energy - ref.energy) < 1e-12 * max(1.0, abs(ref.energy))
    assert np.max(np.abs(jit.forces - ref.forces)) < 1e-12
    assert np.max(np.abs(jit.virial - ref.virial)) < 1e-12


def test_kernel_agreement_for_mixed_species():
    cfg = rattled_fcc(3, 0.12, seed=41)
    rng = np.random.default_rng(2)
    cfg.species = rng.integers(0, 2, size=cfg.n_atoms).astype(np.int32)
    cfg.symbols = ("Ar", "Kr")
    args = ([0.0103, 0.0140], [3.405, 3.640], small_cutoff(cfg, 6.0))
    a = LennardJones(*args, mode="switched", kernel="numba").compute(cfg)
    b = LennardJones(*args, mode="switched", kernel="numpy").compute(cfg)
    assert abs(a.energy - b.energy) < 1e-13
    assert np.max(np.abs(a.forces - b.forces)) < 1e-13


# --------------------------------------------------------------------------
# exact symmetries
# --------------------------------------------------------------------------


def _rotation(axis, angle) -> np.ndarray:
    """Rodrigues rotation matrix for a unit axis and angle in radians."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _potentials_for(cfg):
    rc = small_cutoff(cfg, 6.0)
    return [
        LennardJones.argon(cutoff=rc, mode="shifted_force"),
        LennardJones.argon(cutoff=rc, mode="switched"),
        Morse.argon(cutoff=rc, mode="shifted"),
        HarmonicPair(0.8, 3.7, min(rc, 5.0)),
    ]


def test_energy_invariant_under_translation():
    cfg = rattled_fcc()
    shifted = cfg.translated(np.array([1.7, -3.1, 0.4]))
    for pot in _potentials_for(cfg):
        a, b = pot.compute(cfg), pot.compute(shifted)
        assert abs(a.energy - b.energy) < 1e-12 * max(1.0, abs(a.energy))
        assert np.max(np.abs(a.forces - b.forces)) < 1e-12
        assert np.max(np.abs(a.virial - b.virial)) < 1e-11


def test_forces_rotate_covariantly():
    cfg = rattled_fcc()
    rot = _rotation([0.3, -0.7, 0.5], 0.9)
    rotated = cfg.rotated(rot)
    for pot in _potentials_for(cfg):
        a, b = pot.compute(cfg), pot.compute(rotated)
        assert abs(a.energy - b.energy) < 1e-11 * max(1.0, abs(a.energy))
        # Forces are stored as row vectors, so f -> f R^T.
        assert np.max(np.abs(b.forces - a.forces @ rot.T)) < 1e-12
        assert np.max(np.abs(b.virial - rot @ a.virial @ rot.T)) < 1e-11


def test_energy_invariant_under_permutation():
    cfg = rattled_fcc()
    perm = np.random.default_rng(7).permutation(cfg.n_atoms)
    shuffled = cfg.copy()
    shuffled.positions = cfg.positions[perm]
    shuffled.species = cfg.species[perm]
    shuffled.masses = cfg.masses[perm]
    for pot in _potentials_for(cfg):
        a, b = pot.compute(cfg), pot.compute(shuffled)
        assert abs(a.energy - b.energy) < 1e-12 * max(1.0, abs(a.energy))
        assert np.max(np.abs(b.forces - a.forces[perm])) < 1e-12
        assert np.max(np.abs(a.virial - b.virial)) < 1e-11


@pytest.mark.parametrize("config_name", sorted(CONFIG_FACTORIES))
def test_virial_is_symmetric_for_central_potentials(config_name):
    """W = -sum phi'(r) D (x) D / r is manifestly symmetric; check the code is."""
    cfg = CONFIG_FACTORIES[config_name]()
    for pot in _potentials_for(cfg):
        w = pot.virial(cfg)
        assert np.max(np.abs(w - w.T)) < 1e-12 * max(1.0, np.max(np.abs(w)))


def test_einstein_virial_is_not_symmetric():
    """An external field has no reason to give a symmetric virial, and does not.

    This is the counterpart of the test above: it pins down that the symmetry
    checked there is a property of central pair interactions rather than an
    artefact of how the virial is accumulated.
    """
    cfg = rattled_fcc()
    ec = EinsteinCrystal(fcc(A_FCC, "Ar", 3).positions, 1.3)
    w = ec.virial(cfg)
    assert np.max(np.abs(w - w.T)) > 1e-3
    assert check_virial(ec, cfg) < 1e-6  # ... and it is still the correct one


def test_compute_does_not_mutate_the_configuration():
    cfg = rattled_fcc()
    before = (cfg.positions.copy(), cfg.cell.copy(), cfg.species.copy())
    for pot in _potentials_for(cfg):
        pot.compute(cfg)
    assert np.array_equal(cfg.positions, before[0])
    assert np.array_equal(cfg.cell, before[1])
    assert np.array_equal(cfg.species, before[2])


def test_per_atom_energies_sum_to_the_total():
    cfg = rattled_fcc()
    for pot in _potentials_for(cfg):
        res = pot.compute(cfg)
        assert abs(res.energies.sum() - res.energy) < 1e-12 * max(1.0, abs(res.energy))


# --------------------------------------------------------------------------
# behaviour at the cutoff
# --------------------------------------------------------------------------


def test_switching_function_is_c2_at_both_ends():
    r_on, rc = 6.0, 8.0
    s, ds = switching_function(np.array([r_on, rc]), r_on, rc)
    assert np.allclose(s, [1.0, 0.0], atol=0.0)
    assert np.allclose(ds, [0.0, 0.0], atol=0.0)

    # Second derivative by central difference at both ends, from the inside.
    h = 1e-4
    for r in (r_on + h, rc - h):
        _, dplus = switching_function(np.array([r + h]), r_on, rc)
        _, dminus = switching_function(np.array([r - h]), r_on, rc)
        d2 = (dplus - dminus) / (2.0 * h)
        assert abs(float(d2[0])) < 1e-3  # -> 0 as h -> 0; O(h) here


@pytest.mark.parametrize("mode", CUTOFF_MODES)
def test_potential_is_exactly_zero_beyond_the_cutoff(mode):
    rc = 7.0
    lj = LennardJones.argon(cutoff=rc, mode=mode)
    for r in (rc + 1e-9, rc + 0.5, 3.0 * rc):
        u, du = lj.pair(np.array([r]))
        assert float(u[0]) == 0.0
        assert float(du[0]) == 0.0
        assert lj.energy(open_pair(r)) == 0.0


def _dimer_force(pot, r):
    """Magnitude of the force on atom 0 of an isolated dimer at separation r."""
    return float(np.abs(pot.forces(open_pair(r))[0, 0]))


@pytest.mark.parametrize("mode,continuous", [("shifted_force", True), ("switched", True)])
def test_smooth_modes_have_continuous_forces_at_the_cutoff(mode, continuous):
    """Force -> 0 as r -> rc from below, at the analytically expected rate."""
    rc = 7.0
    pot = LennardJones.argon(cutoff=rc, mode=mode)
    f1 = _dimer_force(pot, rc - 1e-3)
    f2 = _dimer_force(pot, rc - 1e-4)
    assert _dimer_force(pot, rc + 1e-6) == 0.0
    assert f1 < 1e-6 and f2 < f1
    # shifted_force: phi'(rc-d) ~ d phi''(rc)  -> ratio ~ 10
    # switched:      phi'(rc-d) ~ d^2          -> ratio ~ 100
    ratio = f1 / f2
    expected = 10.0 if mode == "shifted_force" else 100.0
    assert 0.5 * expected < ratio < 2.0 * expected


@pytest.mark.parametrize("mode", ["truncated", "shifted"])
def test_bare_and_shifted_modes_have_a_force_jump_at_the_cutoff(mode):
    """The discontinuity is exactly |u'(rc)|: documented, not accidental."""
    rc = 7.0
    pot = LennardJones.argon(cutoff=rc, mode=mode)
    _, du_rc = lj_pair(rc, ARGON_EPSILON, ARGON_SIGMA)
    # Evaluated exactly at rc, which the kernel counts as inside.
    jump = _dimer_force(pot, rc) - _dimer_force(pot, rc + 1e-9)
    assert abs(jump - abs(float(du_rc))) < 1e-15
    assert abs(jump) > 1e-6


def test_energy_jump_at_the_cutoff_distinguishes_the_modes():
    """Only the bare truncation leaves an energy step -- the impulsive force."""
    rc = 7.0
    u_rc, _ = lj_pair(rc, ARGON_EPSILON, ARGON_SIGMA)
    steps = {}
    for mode in CUTOFF_MODES:
        pot = LennardJones.argon(cutoff=rc, mode=mode)
        steps[mode] = pot.energy(open_pair(rc)) - pot.energy(open_pair(rc + 1e-9))
    assert abs(steps["truncated"] - float(u_rc)) < 1e-16
    assert abs(steps["truncated"]) > 1e-6
    for mode in ("shifted", "shifted_force", "switched"):
        assert abs(steps[mode]) < 1e-15


# --------------------------------------------------------------------------
# closed-form pair values
# --------------------------------------------------------------------------


def test_lennard_jones_pair_function_landmarks():
    eps, sig = ARGON_EPSILON, ARGON_SIGMA
    u_sigma, _ = lj_pair(sig, eps, sig)
    assert abs(float(u_sigma)) < 1e-18
    r_min = 2.0 ** (1.0 / 6.0) * sig
    u_min, du_min = lj_pair(r_min, eps, sig)
    assert abs(float(u_min) + eps) < 1e-15
    assert abs(float(du_min)) < 1e-15
    assert abs(LennardJones.argon().r_min - r_min) < 1e-12


def test_morse_pair_function_landmarks():
    d_e, alpha, r_e = 0.5, 1.4, 2.3
    u, du = morse_pair(r_e, d_e, alpha, r_e)
    assert abs(float(u) + d_e) < 1e-15
    assert abs(float(du)) < 1e-15
    assert abs(float(morse_pair(1e4, d_e, alpha, r_e)[0])) < 1e-30


def test_morse_matched_to_lennard_jones_shares_depth_position_curvature():
    eps, sig = ARGON_EPSILON, ARGON_SIGMA
    mo = Morse.matched_to_lennard_jones(eps, sig, 9.0)
    r_e = float(mo.r_e[0, 0])
    assert abs(r_e - 2.0 ** (1.0 / 6.0) * sig) < 1e-12
    assert abs(float(morse_pair(r_e, mo.d_e[0, 0], mo.alpha[0, 0], r_e)[0]) + eps) < 1e-15

    # Curvatures at the shared minimum, by central difference of du/dr.
    h = 1e-4
    lj_curv = (lj_pair(r_e + h, eps, sig)[1] - lj_pair(r_e - h, eps, sig)[1]) / (2 * h)
    mo_curv = (
        morse_pair(r_e + h, mo.d_e[0, 0], mo.alpha[0, 0], r_e)[1]
        - morse_pair(r_e - h, mo.d_e[0, 0], mo.alpha[0, 0], r_e)[1]
    ) / (2 * h)
    assert abs(float(lj_curv) - float(mo_curv)) < 1e-6 * abs(float(lj_curv))
    assert abs(float(lj_curv) - 72.0 * eps / (2.0 ** (1.0 / 3.0) * sig**2)) < 1e-6


def test_dimer_energy_equals_the_pair_function():
    rc = 8.0
    for mode in CUTOFF_MODES:
        pot = LennardJones.argon(cutoff=rc, mode=mode)
        for r in (3.2, 3.822, 5.0, 7.5):
            expected = float(pot.pair(np.array([r]))[0][0])
            assert abs(pot.energy(open_pair(r)) - expected) < 1e-15


def test_dimer_virial_matches_the_analytic_pair_form():
    """W = -phi'(r) D (x) D / r for one pair, checked componentwise."""
    rc, r = 8.0, 4.4
    pot = Morse.argon(cutoff=rc, mode="shifted")
    cfg = open_pair(r)
    _, dphi = pot.pair(np.array([r]))
    w = pot.virial(cfg)
    expected = np.zeros((3, 3))
    expected[0, 0] = -float(dphi[0]) * r  # D = (r, 0, 0)
    assert np.max(np.abs(w - expected)) < 1e-15


# --------------------------------------------------------------------------
# mixing rules
# --------------------------------------------------------------------------


def test_lorentz_berthelot_mixing_matrices():
    eps = np.array([0.01, 0.04])
    sig = np.array([3.0, 4.0])
    lj = LennardJones(eps, sig, 8.0)
    assert lj.n_types == 2
    assert abs(lj.epsilon[0, 1] - math.sqrt(0.01 * 0.04)) < 1e-15
    assert abs(lj.sigma[0, 1] - 3.5) < 1e-15
    assert np.allclose(np.diag(lj.epsilon), eps)
    assert np.allclose(np.diag(lj.sigma), sig)
    assert np.allclose(lj.epsilon, lj.epsilon.T)


def test_uniform_species_reduces_to_single_species():
    """Two types with identical parameters must be indistinguishable from one."""
    cfg = rattled_fcc(3, 0.12, seed=51)
    rc = small_cutoff(cfg, 6.0)
    single = LennardJones.argon(cutoff=rc, mode="shifted_force").compute(cfg)

    two = cfg.copy()
    two.species = (np.arange(cfg.n_atoms) % 2).astype(np.int32)
    two.symbols = ("Ar", "Ar")
    multi = LennardJones(
        [ARGON_EPSILON, ARGON_EPSILON], [ARGON_SIGMA, ARGON_SIGMA], rc, mode="shifted_force"
    ).compute(two)
    assert abs(single.energy - multi.energy) < 1e-12
    assert np.max(np.abs(single.forces - multi.forces)) < 1e-13


def test_explicit_cross_matrix_overrides_mixing():
    """A full (T, T) matrix must be used verbatim -- the perturbations need it."""
    eps = np.array([[0.01, 0.05], [0.05, 0.04]])  # 0.05 != sqrt(0.01*0.04) = 0.02
    sig = np.array([[3.0, 3.9], [3.9, 4.0]])  # 3.9 != (3+4)/2 = 3.5
    lj = LennardJones(eps, sig, 8.0)
    assert lj.epsilon[0, 1] == 0.05
    assert lj.sigma[0, 1] == 3.9


def test_morse_mixing_rules():
    mo = Morse([1.0, 4.0], [2.0, 3.0], [1.5, 2.5], 8.0)
    assert abs(mo.d_e[0, 1] - 2.0) < 1e-15  # geometric
    assert abs(mo.alpha[0, 1] - 2.5) < 1e-15  # arithmetic
    assert abs(mo.r_e[0, 1] - 2.0) < 1e-15  # arithmetic


def test_mixing_rejects_asymmetric_matrices():
    with pytest.raises(ValueError, match="symmetric"):
        mix_pair_parameters(np.array([[1.0, 2.0], [3.0, 4.0]]), rule="geometric")


# --------------------------------------------------------------------------
# the fcc Lennard-Jones lattice sum
# --------------------------------------------------------------------------


def _fcc_lattice_energy(rc_over_sigma: float, nn_over_sigma: float) -> tuple[float, float]:
    """Truncated fcc lattice energy per atom, and its mean-field tail, in eps.

    Works in reduced units (``eps = sigma = 1``).  The supercell is sized so
    that ``2 rc`` fits inside the minimum cell width, which is precisely the
    condition under which the periodic energy equals the *infinite* lattice sum
    truncated at ``rc`` -- so the supercell size affects cost and nothing else.
    """
    a0 = nn_over_sigma * math.sqrt(2.0)
    reps = int(math.ceil((2.0 * rc_over_sigma + 0.2) / a0))
    cfg = fcc(a0, "Ar", reps)
    lj = LennardJones(1.0, 1.0, rc_over_sigma, mode="truncated")
    return lj.energy_per_atom(cfg), lj.tail_energy_per_atom(cfg.density)


@pytest.mark.physics
@pytest.mark.slow
def test_fcc_lattice_energy_converges_to_the_lattice_sum():
    """E/N -> -8.6102 eps at d = 1.0902 sigma, by extrapolation in the cutoff.

    Convergence is slow and *oscillatory*, for two separate reasons:

    * the missing tail is ``-8 pi rho eps sigma^6 / (3 rc^3)`` -- the attractive
      ``r^-6`` branch integrated over the neglected shell -- so the truncated
      energy converges only as ``rc^-3``.  At ``rc = 6 sigma`` this is still
      0.042 eps, five times the tolerance we are asked to meet;
    * the lattice is discrete, so ``E(rc)`` is a *step* function of ``rc`` while
      the mean-field tail is smooth.  The residual after tail correction
      therefore oscillates with the shell structure instead of decaying
      monotonically, and a naive least-squares fit in ``rc^-3`` is limited by
      that oscillation rather than by the neglected ``rc^-9`` term.

    The estimator used here is consequently: correct each truncated energy with
    the analytic tail, then average over a ladder of cutoffs, which cancels the
    (zero-mean) shell oscillation.  The observed numbers are printed.
    """
    d = FCC_MIN_NN_OVER_SIGMA
    ladder = [5.5, 6.0, 6.5, 7.0]
    raw, corrected = [], []
    print("\n  rc/sigma      E_trunc      tail       E+tail      error")
    for rc in ladder:
        e, tail = _fcc_lattice_energy(rc, d)
        raw.append(e)
        corrected.append(e + tail)
        print(
            f"  {rc:6.2f}   {e:+.6f}  {tail:+.6f}  {e + tail:+.6f}  "
            f"{e + tail - FCC_MIN_ENERGY_PER_EPSILON:+.2e}"
        )

    corrected = np.array(corrected)
    estimate = float(corrected.mean())
    error = estimate - FCC_MIN_ENERGY_PER_EPSILON

    # A plain rc^-3 least-squares extrapolation of the *uncorrected* energies,
    # reported for comparison: it is an order of magnitude worse.
    rc = np.array(ladder)
    design = np.vstack([np.ones_like(rc), rc**-3.0]).T
    coeff, *_ = np.linalg.lstsq(design, np.array(raw), rcond=None)
    fit_error = coeff[0] - FCC_MIN_ENERGY_PER_EPSILON
    print(f"  tail-corrected mean over ladder : {estimate:+.6f}  (error {error:+.2e})")
    print(f"  plain rc^-3 least-squares fit   : {coeff[0]:+.6f}  (error {fit_error:+.2e})")

    assert abs(error) < 1e-3, "3-decimal target on the extrapolated lattice energy"
    assert np.max(np.abs(corrected - FCC_MIN_ENERGY_PER_EPSILON)) < 2e-3
    assert abs(fit_error) < 5e-3


@pytest.mark.physics
@pytest.mark.slow
def test_design_doc_lattice_constant_is_not_the_lj_minimum():
    """``a0 = 1.5496 sigma`` (docs/design.md) is not where the minimum sits.

    The nearest-neighbour distance ``1.0902 sigma`` quoted in the same document
    corresponds to ``a0 = 1.0902 * sqrt(2) = 1.5417 sigma``.  Evaluating the
    lattice sum at both settles it: only the latter reproduces ``-8.6102 eps``.
    """
    rc = 6.5
    e_ref, tail_ref = _fcc_lattice_energy(rc, FCC_MIN_NN_OVER_SIGMA)
    e_doc, tail_doc = _fcc_lattice_energy(rc, 1.5496 / math.sqrt(2.0))
    correct = e_ref + tail_ref
    quoted = e_doc + tail_doc
    print(f"\n  a0 = {FCC_MIN_A0_OVER_SIGMA:.4f} sigma -> E/N = {correct:+.5f} eps")
    print(f"  a0 = 1.5496 sigma -> E/N = {quoted:+.5f} eps")

    assert abs(correct - FCC_MIN_ENERGY_PER_EPSILON) < 1e-3
    assert quoted > correct  # the doc's spacing is off the minimum ...
    assert abs(quoted - FCC_MIN_ENERGY_PER_EPSILON) > 5e-3  # ... by a lot


def test_lattice_sum_constants_are_self_consistent():
    """The stored fcc constants satisfy the closed-form relations they claim."""
    x = FCC_LATTICE_SUM_A6 / (2.0 * FCC_LATTICE_SUM_A12)
    assert abs(x ** (-1.0 / 6.0) - FCC_MIN_NN_OVER_SIGMA) < 1e-6
    assert abs(FCC_MIN_NN_OVER_SIGMA * math.sqrt(2.0) - FCC_MIN_A0_OVER_SIGMA) < 1e-6
    e_min = 2.0 * (FCC_LATTICE_SUM_A12 * x * x - FCC_LATTICE_SUM_A6 * x)
    assert abs(e_min - FCC_MIN_ENERGY_PER_EPSILON) < 1e-5


def test_tail_correction_matches_numerical_integration():
    """The analytic long-range correction against direct quadrature of 4 pi r^2 u."""
    from scipy.integrate import quad

    lj = LennardJones.argon(cutoff=8.5)
    density = 0.021
    integral, _ = quad(
        lambda r: 4.0 * math.pi * r * r * float(lj_pair(r, ARGON_EPSILON, ARGON_SIGMA)[0]),
        lj.cutoff,
        np.inf,
        limit=200,
    )
    assert abs(lj.tail_energy_per_atom(density) - 0.5 * density * integral) < 1e-12


# --------------------------------------------------------------------------
# cross-check against ASE (test-only dependency)
# --------------------------------------------------------------------------


def _to_ase(cfg: Configuration):
    from ase import Atoms

    return Atoms(
        symbols=["Ar"] * cfg.n_atoms,
        positions=cfg.positions,
        cell=cfg.cell,
        pbc=cfg.pbc,
    )


@pytest.mark.parametrize(
    "config_name,rc",
    [("rattled_fcc", 8.5), ("gas", 6.5), ("triclinic", 6.0)],
)
def test_agrees_with_ase_lennard_jones(config_name, rc):
    """ASE's LennardJones(smooth=False) is exactly our ``mode="shifted"``.

    ASE's stress is ``+dU/d eps / V`` while our virial is ``-dU/d eps``, so the
    comparison also pins down the sign convention of the virial against an
    independent implementation.
    """
    from ase.calculators.lj import LennardJones as ASELennardJones

    cfg = rattled_fcc(4) if config_name == "rattled_fcc" else CONFIG_FACTORIES[config_name]()
    rc = small_cutoff(cfg, rc)
    ours = LennardJones.argon(cutoff=rc, mode="shifted").compute(cfg)

    atoms = _to_ase(cfg)
    atoms.calc = ASELennardJones(
        epsilon=ARGON_EPSILON, sigma=ARGON_SIGMA, rc=rc, smooth=False
    )
    d_energy = abs(ours.energy - atoms.get_potential_energy())
    d_forces = float(np.max(np.abs(ours.forces - atoms.get_forces())))
    d_stress = float(
        np.max(np.abs(-ours.virial / cfg.volume - atoms.get_stress(voigt=False)))
    )
    d_atomic = float(np.max(np.abs(ours.energies - atoms.get_potential_energies())))
    print(
        f"\n  {config_name} rc={rc:.3f}: dE={d_energy:.2e} eV, dF={d_forces:.2e} eV/A, "
        f"dsigma={d_stress:.2e} eV/A^3, dE_atom={d_atomic:.2e} eV"
    )

    assert d_energy < 1e-11
    assert d_forces < 1e-12
    assert d_stress < 1e-14
    assert d_atomic < 1e-13


def test_agrees_with_ase_for_an_open_cluster():
    """No cell, no periodicity: the aperiodic code path against ASE."""
    from ase.calculators.lj import LennardJones as ASELennardJones

    src = rattle(fcc(A_FCC, "Ar", 2), 0.15, seed=61)
    cfg = Configuration(positions=src.positions, cell=None, pbc=False, symbols=("Ar",))
    ours = LennardJones.argon(cutoff=8.5, mode="shifted").compute(cfg)
    atoms = _to_ase(cfg)
    atoms.calc = ASELennardJones(
        epsilon=ARGON_EPSILON, sigma=ARGON_SIGMA, rc=8.5, smooth=False
    )
    assert abs(ours.energy - atoms.get_potential_energy()) < 1e-13
    assert np.max(np.abs(ours.forces - atoms.get_forces())) < 1e-13


# --------------------------------------------------------------------------
# harmonic potentials against closed forms
# --------------------------------------------------------------------------


def test_harmonic_pair_dimer_closed_form():
    k, r0 = 1.25, 3.6
    hp = HarmonicPair(k, r0, 6.0)
    for r in (2.9, 3.6, 4.4, 5.5):
        cfg = open_pair(r)
        res = hp.compute(cfg)
        assert abs(res.energy - 0.5 * k * (r - r0) ** 2) < 1e-14
        # Force on atom 0 points along +x when the bond is stretched.
        assert abs(res.forces[0, 0] - k * (r - r0)) < 1e-14
        assert np.allclose(res.forces[0], -res.forces[1], atol=1e-15)
        assert abs(res.virial[0, 0] + k * (r - r0) * r) < 1e-13


def test_harmonic_pair_matches_the_bare_pair_function():
    k, r0 = 0.7, 3.3
    hp = HarmonicPair(k, r0, 6.0)  # default mode="truncated": no modification
    r = np.linspace(1.0, 5.9, 25)
    u, du = hp.pair(r)
    u_ref, du_ref = harmonic_pair(r, k, r0)
    assert np.max(np.abs(u - u_ref)) < 1e-15
    assert np.max(np.abs(du - du_ref)) < 1e-15


def test_harmonic_pair_angular_frequency():
    k = 2.0
    mu = 39.948 / 2.0
    hp = HarmonicPair(k, 3.6, 6.0)
    assert abs(hp.angular_frequency(mu) - math.sqrt(k / (mu * MVV2E))) < 1e-12


def test_einstein_crystal_closed_form():
    rng = np.random.default_rng(3)
    ref = fcc(A_FCC, "Ar", 2).positions
    disp = rng.normal(0.0, 0.09, size=ref.shape)
    cfg = Configuration(positions=ref + disp, cell=None, pbc=False, symbols=("Ar",))
    k = 1.6
    ec = EinsteinCrystal(ref, k)
    res = ec.compute(cfg)
    assert abs(res.energy - 0.5 * k * float(np.sum(disp**2))) < 1e-13
    assert np.max(np.abs(res.forces + k * disp)) < 1e-14
    assert np.max(np.abs(res.energies - 0.5 * k * np.sum(disp**2, axis=1))) < 1e-15
    # W = sum_i f_i (x) r_i for an external field.
    assert np.max(np.abs(res.virial - np.einsum("ia,ib->ab", -k * disp, cfg.positions))) < 1e-12


def test_einstein_crystal_per_atom_spring_constants():
    ref = fcc(A_FCC, "Ar", 2).positions
    k = np.linspace(0.5, 2.0, ref.shape[0])
    disp = np.random.default_rng(9).normal(0.0, 0.07, size=ref.shape)
    cfg = Configuration(positions=ref + disp, cell=None, pbc=False, symbols=("Ar",))
    ec = EinsteinCrystal(ref, k)
    res = ec.compute(cfg)
    assert abs(res.energy - 0.5 * float(np.sum(k * np.sum(disp**2, axis=1)))) < 1e-13
    assert check_forces(ec, cfg) < 1e-6


def test_einstein_crystal_tethers_through_the_minimum_image():
    """An atom that leaves the box is pulled to the nearest image of its site."""
    ref = fcc(A_FCC, "Ar", 2)
    moved = ref.copy()
    # Move atom 0 one full lattice vector: it is now on top of its own image,
    # so the tether energy must be exactly zero, not k|a1|^2/2.
    moved.positions[0] += ref.cell[0]
    ec = EinsteinCrystal(ref.positions, 1.0)
    assert abs(ec.energy(moved)) < 1e-18

    ec_naive = EinsteinCrystal(ref.positions, 1.0, minimum_image=False)
    assert ec_naive.energy(moved) > 1.0


def test_einstein_free_energy_is_thermodynamically_consistent():
    """d(beta F)/d beta must be the mean energy 3 N k T of 3N classical dof.

    This exercises the whole expression, including the temperature dependence
    of the de Broglie wavelength (which supplies half of the 3 N k T), so it
    catches a botched metal-unit conversion rather than only a botched prefactor.
    """
    from atomlab.units import KB

    ref = fcc(A_FCC, "Ar", 2)
    n = ref.n_atoms
    ec = EinsteinCrystal(ref.positions, 1.4)
    t = 120.0
    h = 0.5

    def beta_f(temperature):
        return ec.free_energy(temperature, ref.masses) / (KB * temperature)

    beta = 1.0 / (KB * t)
    d_beta = 1.0 / (KB * (t - h)) - 1.0 / (KB * (t + h))
    mean_energy = (beta_f(t - h) - beta_f(t + h)) / d_beta
    assert abs(mean_energy - 3.0 * n * KB * t) < 1e-4 * abs(3.0 * n * KB * t)

    # Doubling every spring lowers F by (3/2) N k T ln 2.
    stiffer = EinsteinCrystal(ref.positions, 2.8)
    delta = stiffer.free_energy(t, ref.masses) - ec.free_energy(t, ref.masses)
    assert abs(delta - 1.5 * n * KB * t * math.log(2.0)) < 1e-12


def test_einstein_angular_frequency():
    ec = EinsteinCrystal(np.zeros((1, 3)), 3.0)
    assert abs(ec.angular_frequency(39.948) - math.sqrt(3.0 / (39.948 * MVV2E))) < 1e-12


# --------------------------------------------------------------------------
# plumbing: guards, algebra, neighbour-list caching
# --------------------------------------------------------------------------


def test_cutoff_larger_than_half_the_cell_raises():
    cfg = fcc(A_FCC, "Ar", 2)  # width 10.50 A
    lj = LennardJones.argon(cutoff=6.0)
    with pytest.raises(ValueError, match="minimum image"):
        lj.compute(cfg)


def test_species_index_beyond_the_parameterisation_raises():
    cfg = rattled_fcc()
    cfg.species = np.ones(cfg.n_atoms, dtype=np.int32)
    lj = LennardJones.argon(cutoff=6.0)
    with pytest.raises(ValueError, match="parameterised"):
        lj.compute(cfg)


def test_invalid_construction_arguments_raise():
    with pytest.raises(ValueError, match="cutoff"):
        LennardJones(1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="mode"):
        LennardJones(1.0, 1.0, 5.0, mode="clipped")
    with pytest.raises(ValueError, match="r_on"):
        LennardJones(1.0, 1.0, 5.0, mode="switched", r_on=5.0)
    with pytest.raises(ValueError, match="sigma"):
        LennardJones(1.0, -1.0, 5.0)
    with pytest.raises(ValueError, match="jitted"):
        HarmonicPair(1.0, 1.0, 5.0, kernel="numba")
    with pytest.raises(ValueError, match="alpha"):
        Morse(1.0, 0.0, 1.0, 5.0)


def test_verlet_skin_does_not_change_the_answer():
    """A skin changes which pairs are listed, never the energy or forces."""
    cfg = rattled_fcc()
    rc = small_cutoff(cfg, 6.0)
    plain = LennardJones.argon(cutoff=rc, mode="switched").compute(cfg)
    skinned_pot = LennardJones.argon(cutoff=rc, mode="switched", skin=0.8)
    first = skinned_pot.compute(cfg)
    assert abs(plain.energy - first.energy) < 1e-13
    assert np.max(np.abs(plain.forces - first.forces)) < 1e-14

    # Move the atoms a little; the cached list must still give the right answer.
    moved = cfg.copy()
    moved.positions += np.random.default_rng(4).normal(0.0, 0.05, size=cfg.positions.shape)
    cached = skinned_pot.compute(moved)
    fresh = LennardJones.argon(cutoff=rc, mode="switched").compute(moved)
    assert abs(cached.energy - fresh.energy) < 1e-13
    assert np.max(np.abs(cached.forces - fresh.forces)) < 1e-14


def test_potential_algebra_composes_and_stays_differentiable():
    """U0 + delta_U is how every perturbation experiment is built."""
    cfg = rattled_fcc()
    rc = small_cutoff(cfg, 6.0)
    lj = LennardJones.argon(cutoff=rc, mode="shifted_force")
    mo = Morse.argon(cutoff=rc, mode="shifted_force")

    total = lj + 0.25 * mo
    expected = lj.energy(cfg) + 0.25 * mo.energy(cfg)
    assert abs(total.energy(cfg) - expected) < 1e-12
    assert check_forces(total, cfg, atoms=range(4)) < 1e-6
    assert check_virial(total, cfg) < 1e-6

    difference = lj - lj
    assert abs(difference.energy(cfg)) < 1e-14
    assert np.max(np.abs(difference.forces(cfg))) < 1e-15


def test_labelling_round_trip():
    cfg = rattled_fcc()
    lj = LennardJones.argon(cutoff=small_cutoff(cfg, 6.0))
    (labelled,) = lj.label([cfg])
    assert labelled.has_labels
    assert abs(labelled.energy - lj.energy(cfg)) < 1e-14
    assert cfg.energy is None  # the input is untouched


def test_argon_preset_parameters():
    lj = LennardJones.argon()
    assert lj.epsilon[0, 0] == ARGON_EPSILON == 0.0103
    assert lj.sigma[0, 0] == ARGON_SIGMA == 3.405
    assert lj.cutoff == ARGON_CUTOFF == 8.5
    assert lj.mode == "shifted_force"
    assert lj.n_types == 1
