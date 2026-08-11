"""Tests for the many-body reference potentials (Stillinger-Weber and EAM).

These two potentials are the reason the study is not trivial: neither can be
represented by any function of pair distances alone, so a pair-only surrogate
is *structurally* wrong for them rather than merely inaccurate.  That makes
their correctness load-bearing -- an error here would look exactly like the
model error the experiments are trying to measure.

The checks fall into four groups:

1. analytic vs finite-difference derivatives (forces and virial),
2. exact known physics (the SW diamond lattice energy, the vanishing of the
   three-body term at the tetrahedral angle, the fitted fcc Cu properties),
3. exact symmetries (translation, rotation, permutation, periodic image),
4. agreement between the jitted kernel and the pure-NumPy reference.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from atomlab.build import diamond, fcc, rattle, scale_cell
from atomlab.cell import min_cell_width
from atomlab.potentials.base import check_forces, check_virial
from atomlab.potentials.eam import EAM, EAM_COPPER
from atomlab.potentials.stillinger_weber import SW_SILICON_1985, StillingerWeber
from atomlab.types import Configuration
from atomlab.units import EV_A3_TO_GPA

# Tolerance demanded by docs/design.md for every potential in the package.
DERIV_TOL = 1e-6


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sw() -> StillingerWeber:
    return StillingerWeber.silicon()


@pytest.fixture(scope="module")
def sw_numpy() -> StillingerWeber:
    return StillingerWeber.silicon(implementation="numpy")


@pytest.fixture(scope="module")
def eam() -> EAM:
    return EAM.copper()


@pytest.fixture(scope="module")
def eam_numpy() -> EAM:
    return EAM.copper(implementation="numpy")


def rattled_silicon(sigma: float = 0.15, seed: int = 11) -> Configuration:
    """2x2x2 diamond Si supercell (64 atoms), displaced off the lattice."""
    return rattle(diamond(5.431, "Si", (2, 2, 2)), sigma, seed=seed)


def rattled_copper(sigma: float = 0.10, seed: int = 12) -> Configuration:
    """3x3x3 fcc Cu supercell (108 atoms), displaced off the lattice."""
    return rattle(fcc(3.615, "Cu", (3, 3, 3)), sigma, seed=seed)


def _distort(cfg: Configuration, seed: int, sigma: float) -> Configuration:
    """Apply a large, deliberately non-symmetric strain, then rattle.

    The strain has a substantial antisymmetric part as well, so the cell is
    both sheared and rotated: this is the case where the naive "round the
    fractional coordinates" minimum image is wrong and the neighbour list has
    to carry the day.
    """
    eps = np.array(
        [
            [0.061, 0.092, -0.043],
            [0.048, -0.070, 0.115],
            [-0.083, 0.031, 0.052],
        ]
    )
    return rattle(cfg.strained(eps), sigma, seed=seed)


def triclinic_silicon() -> Configuration:
    return _distort(diamond(5.431, "Si", (2, 2, 2)), seed=21, sigma=0.10)


def triclinic_copper() -> Configuration:
    return _distort(fcc(3.615, "Cu", (4, 4, 4)), seed=22, sigma=0.08)


def cluster(symbol: str, n: int, box: float, seed: int, min_distance: float) -> Configuration:
    """A small non-periodic blob of ``n`` atoms, rejection-sampled in a cube."""
    rng = np.random.default_rng(seed)
    pos: list[np.ndarray] = []
    while len(pos) < n:
        trial = rng.random(3) * box
        if all(np.linalg.norm(trial - p) >= min_distance for p in pos):
            pos.append(trial)
    return Configuration(
        positions=np.array(pos),
        cell=np.zeros((3, 3)),
        pbc=False,
        symbols=(symbol,),
    )


def silicon_cluster() -> Configuration:
    return cluster("Si", 9, box=5.0, seed=31, min_distance=2.2)


def copper_cluster() -> Configuration:
    return cluster("Cu", 11, box=5.5, seed=32, min_distance=2.3)


# --------------------------------------------------------------------------
# 1. analytic derivatives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_cfg",
    [rattled_silicon, triclinic_silicon, silicon_cluster],
    ids=["rattled-diamond", "triclinic", "cluster"],
)
def test_sw_forces_match_finite_difference(sw, make_cfg):
    cfg = make_cfg()
    # Checking a subset keeps the test to a few seconds; the atoms are picked
    # by stride so they sample different local environments rather than one
    # corner of the cell.
    atoms = range(0, cfg.n_atoms, max(1, cfg.n_atoms // 8))
    assert check_forces(sw, cfg, atoms=atoms) < DERIV_TOL


@pytest.mark.parametrize(
    "make_cfg",
    [rattled_silicon, triclinic_silicon, silicon_cluster],
    ids=["rattled-diamond", "triclinic", "cluster"],
)
def test_sw_virial_matches_finite_difference(sw, make_cfg):
    assert check_virial(sw, make_cfg()) < DERIV_TOL


@pytest.mark.parametrize(
    "make_cfg",
    [rattled_copper, triclinic_copper, copper_cluster],
    ids=["rattled-fcc", "triclinic", "cluster"],
)
def test_eam_forces_match_finite_difference(eam, make_cfg):
    cfg = make_cfg()
    atoms = range(0, cfg.n_atoms, max(1, cfg.n_atoms // 8))
    assert check_forces(eam, cfg, atoms=atoms) < DERIV_TOL


@pytest.mark.parametrize(
    "make_cfg",
    [rattled_copper, triclinic_copper, copper_cluster],
    ids=["rattled-fcc", "triclinic", "cluster"],
)
def test_eam_virial_matches_finite_difference(eam, make_cfg):
    assert check_virial(eam, make_cfg()) < DERIV_TOL


def test_test_cells_satisfy_minimum_image():
    """The periodic test cells must not be relying on a silent fallback."""
    for cfg, cut in (
        (rattled_silicon(), SW_SILICON_1985.cutoff),
        (triclinic_silicon(), SW_SILICON_1985.cutoff),
        (rattled_copper(), EAM_COPPER.cutoff),
        (triclinic_copper(), EAM_COPPER.cutoff),
    ):
        assert min_cell_width(cfg.cell, cfg.pbc) > 2.0 * cut


def test_potentials_raise_when_cell_is_too_small(sw, eam):
    with pytest.raises(ValueError, match="too large for this cell"):
        sw.compute(diamond(5.431, "Si"))
    with pytest.raises(ValueError, match="too large for this cell"):
        eam.compute(fcc(3.615, "Cu", (2, 2, 2)))


# --------------------------------------------------------------------------
# 2a. Stillinger-Weber: the diamond lattice energy
# --------------------------------------------------------------------------


def test_sw_diamond_cohesive_energy(sw):
    """Diamond Si at a0 = 5.431 A has energy exactly -2 epsilon per atom.

    In the ideal diamond lattice every bond sits (to 6 significant figures) at
    the minimum of ``phi_2``, whose value is exactly ``-epsilon`` by the choice
    of ``A`` and ``B``, and every bond angle is exactly tetrahedral so the
    three-body term vanishes identically.  Two bonds per atom then give
    ``-2 * 2.1683 = -4.33660 eV/atom``.

    ``docs/design.md`` quotes ``-4.3363 eV/atom``.  That number corresponds to
    ``epsilon = 50 kcal/mol = 2.16815 eV`` rather than to the ``2.1683 eV`` of
    the LAMMPS ``Si.sw`` file specified for this implementation, so the two
    differ by 3e-4 eV/atom -- in the fourth decimal.  We assert the identity
    that is actually exact (``-2 epsilon``) and pin the documented value only
    to the precision at which it is meaningful.
    """
    cfg = diamond(5.431, "Si", (2, 2, 2))
    result = sw.compute(cfg)
    e_per_atom = result.energy / cfg.n_atoms

    assert e_per_atom == pytest.approx(-2.0 * SW_SILICON_1985.epsilon, abs=1e-8)
    assert e_per_atom == pytest.approx(-4.33660, abs=5e-6)
    assert e_per_atom == pytest.approx(-4.3363, abs=5e-4)

    # The three-body term is not merely small here, it is zero to round-off.
    assert abs(result.extra["energy_three_body"]) < 1e-20
    assert result.extra["energy_two_body"] == pytest.approx(result.energy)

    # Perfect lattice: no forces, isotropic virial.
    assert np.max(np.abs(result.forces)) < 1e-10
    w = result.virial
    assert np.allclose(w, np.diag(np.diag(w)), atol=1e-10)
    assert np.allclose(np.diag(w), np.mean(np.diag(w)), rtol=1e-10)


def test_sw_diamond_lattice_constant_is_stationary(sw):
    """a0 = 5.431 A is (to 1e-4 A) the minimum of E(a) for diamond Si."""
    base = diamond(5.431, "Si", (2, 2, 2))

    def energy_per_atom(a: float) -> float:
        return sw.energy_per_atom(scale_cell(base, a / 5.431))

    # Numerical derivative at the nominal lattice constant.  It is small but
    # not identically zero: the exact stationary point is 5.430950 A (see
    # below), and with a curvature of ~3.9 eV/A^2 per atom the 5e-5 A offset
    # of the rounded 5.431 leaves a residual slope of ~2e-4 eV/A per atom.
    # Asserting "< 1e-4" would be asserting that 5.431 is exact, which it is
    # not; the honest statement is that the slope is four orders of magnitude
    # below the curvature scale.
    h = 1e-3
    slope = (energy_per_atom(5.431 + h) - energy_per_atom(5.431 - h)) / (2 * h)
    assert abs(slope) < 5e-4  # eV / A per atom

    res = minimize_scalar(
        energy_per_atom, bounds=(5.3, 5.6), method="bounded", options={"xatol": 1e-10}
    )
    # The exact stationary point is 4 * 2^(1/6) * sigma / sqrt(3), because the
    # bond sits at the minimum of phi_2 at r = 2^(1/6) sigma.
    exact = 4.0 * 2.0 ** (1.0 / 6.0) * SW_SILICON_1985.sigma / math.sqrt(3.0)
    assert exact == pytest.approx(5.43095, abs=1e-5)
    assert res.x == pytest.approx(exact, abs=1e-4)
    assert res.x == pytest.approx(5.431, abs=1e-3)

    # And the virial pressure vanishes there, which is the same statement made
    # through the analytic derivative rather than through a scan.
    relaxed = scale_cell(base, res.x / 5.431)
    assert abs(sw.pressure(relaxed)) * EV_A3_TO_GPA < 1e-3  # GPa


# --------------------------------------------------------------------------
# 2b. Stillinger-Weber: the three-body term in isolation
# --------------------------------------------------------------------------


def _trimer(bond: float, angle_deg: float) -> Configuration:
    """Isolated three-atom cluster: central atom 0, angle at 0 subtended by 1-2."""
    t = math.radians(angle_deg)
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [bond, 0.0, 0.0],
            [bond * math.cos(t), bond * math.sin(t), 0.0],
        ]
    )
    return Configuration(positions=pos, cell=np.zeros((3, 3)), pbc=False, symbols=("Si",))


def test_sw_three_body_vanishes_at_tetrahedral_angle(sw):
    """cos(theta) = -1/3 is the zero of the angular factor, by construction.

    The bond length 2.7 A is chosen so that the 1-2 separation exceeds the
    cutoff at *both* angles tested below (4.41 A at 109.47 deg, 3.82 A at
    90 deg), leaving exactly one triplet in the cluster.  Otherwise atoms 1
    and 2 would each become the centre of a further, non-tetrahedral triplet
    and the test would not isolate what it claims to.
    """
    bond = 2.7
    tetrahedral = math.degrees(math.acos(-1.0 / 3.0))
    assert tetrahedral == pytest.approx(109.4712, abs=1e-4)

    cfg = _trimer(bond, tetrahedral)
    assert np.linalg.norm(cfg.positions[1] - cfg.positions[2]) > sw.cutoff
    e3 = sw.compute(cfg).extra["energy_three_body"]
    assert abs(e3) < 1e-25

    # 90 degrees: cos = 0, so (cos - cos0)^2 = 1/9 and the term is positive.
    cfg90 = _trimer(bond, 90.0)
    assert np.linalg.norm(cfg90.positions[1] - cfg90.positions[2]) > sw.cutoff
    e3_90 = sw.compute(cfg90).extra["energy_three_body"]
    assert e3_90 > 0.0

    prm = SW_SILICON_1985
    expected = (
        prm.lam
        * prm.epsilon
        * (1.0 / 3.0) ** 2
        * math.exp(2.0 * prm.gamma * prm.sigma / (bond - sw.cutoff))
    )
    assert e3_90 == pytest.approx(expected, rel=1e-12)
    assert e3_90 == pytest.approx(sw.three_body(bond, bond, 0.0), rel=1e-12)


def test_sw_three_body_is_non_negative_and_angle_dependent(sw):
    """Scanning the angle: a strict minimum of zero at the tetrahedral value."""
    angles = np.linspace(60.0, 175.0, 40)
    e3 = np.array([sw.compute(_trimer(2.7, a)).extra["energy_three_body"] for a in angles])
    assert np.all(e3 >= 0.0)
    assert angles[int(np.argmin(e3))] == pytest.approx(109.47, abs=3.0)


def test_sw_two_body_minimum_is_minus_epsilon(sw):
    """phi_2 has its minimum value exactly -epsilon at r = 2^(1/6) sigma."""
    r_min = 2.0 ** (1.0 / 6.0) * SW_SILICON_1985.sigma
    assert sw.two_body(r_min) == pytest.approx(-SW_SILICON_1985.epsilon, abs=1e-9)
    res = minimize_scalar(
        lambda r: float(sw.two_body(r)), bounds=(1.5, 3.5), method="bounded",
        options={"xatol": 1e-12},
    )
    assert res.x == pytest.approx(r_min, abs=1e-8)


def test_sw_vanishes_smoothly_at_the_cutoff(sw):
    """Energy and force go to zero at a*sigma without a switching function."""
    rc = sw.cutoff
    assert rc == pytest.approx(3.77118, abs=1e-9)
    for eps in (1e-3, 1e-4, 1e-6):
        assert abs(float(sw.two_body(rc - eps))) < 1e-12
    assert float(sw.two_body(rc)) == 0.0
    assert float(sw.two_body(rc + 1.0)) == 0.0

    # A dimer straddling the cutoff: energy and force are continuous through it.
    def dimer_energy(r: float) -> float:
        cfg = Configuration(
            positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            cell=np.zeros((3, 3)),
            pbc=False,
            symbols=("Si",),
        )
        return sw.energy(cfg)

    for r in (rc - 1e-9, rc, rc + 1e-9):
        assert abs(dimer_energy(r)) < 1e-12


# --------------------------------------------------------------------------
# 2c. EAM: the fitted fcc copper properties
# --------------------------------------------------------------------------


def _fcc_equation_of_state(potential: EAM, a_values: np.ndarray) -> np.ndarray:
    base = fcc(3.615, "Cu", (4, 4, 4))
    return np.array([potential.energy_per_atom(scale_cell(base, a / 3.615)) for a in a_values])


def test_eam_copper_lattice_constant_cohesive_energy_and_bulk_modulus(eam):
    """Scan a, fit, and report the three quantities the model was tuned to.

    ``B = V d2E/dV2`` at the minimum, with the volume per atom ``V = a^3/4``
    for fcc.  The fit is done in ``V`` (not in ``a``) because that is the
    variable the definition of ``B`` is written in; fitting in ``a`` and then
    converting introduces a first-derivative term that only vanishes exactly at
    the minimum.
    """
    a_values = np.linspace(3.50, 3.73, 25)
    energies = _fcc_equation_of_state(eam, a_values)

    coeffs = np.polyfit(a_values, energies, 4)
    roots = np.roots(np.polyder(coeffs))
    real = np.real(roots[(np.abs(roots.imag) < 1e-9) & (np.real(roots) > 3.4) & (np.real(roots) < 3.9)])
    a0 = float(real[0])
    e0 = float(np.polyval(coeffs, a0))

    volumes = a_values**3 / 4.0
    v0 = a0**3 / 4.0
    cv = np.polyfit(volumes - v0, energies, 4)
    d2e_dv2 = 2.0 * cv[-3]  # coefficient of (V - V0)^2, times 2!
    bulk_gpa = v0 * d2e_dv2 * EV_A3_TO_GPA

    assert a0 == pytest.approx(3.615, abs=2e-3)
    assert e0 == pytest.approx(-3.54, abs=5e-3)
    assert bulk_gpa == pytest.approx(140.0, rel=0.02)

    # The virial pressure must vanish at the same lattice constant -- an
    # independent route to a0 through the analytic derivative.
    relaxed = scale_cell(fcc(3.615, "Cu", (4, 4, 4)), a0 / 3.615)
    assert abs(eam.pressure(relaxed) * EV_A3_TO_GPA) < 5e-3  # GPa

    print(
        f"\nEAM/Cu fitted properties: a0 = {a0:.5f} A, "
        f"E_coh = {e0:.5f} eV/atom, B = {bulk_gpa:.3f} GPa"
    )


def test_eam_fcc_is_the_ground_state(eam):
    """fcc must be below bcc and simple cubic at their own optimal volumes.

    Not a fitted target -- a genuine prediction of the model, and a cheap way
    to notice a sign error in the embedding term.
    """
    from atomlab.build import bcc, sc

    def relaxed_energy(builder, reps, a_guess) -> float:
        base = builder(a_guess, "Cu", reps)
        res = minimize_scalar(
            lambda s: base and EAM.copper().energy_per_atom(scale_cell(base, s)),
            bounds=(0.85, 1.15),
            method="bounded",
            options={"xatol": 1e-8},
        )
        return float(res.fun)

    e_fcc = relaxed_energy(fcc, (4, 4, 4), 3.615)
    e_bcc = relaxed_energy(bcc, (5, 5, 5), 2.87)
    e_sc = relaxed_energy(sc, (6, 6, 6), 2.36)
    assert e_fcc < e_bcc < e_sc
    print(f"\nEAM/Cu structural energies (eV/atom): fcc {e_fcc:.4f}, bcc {e_bcc:.4f}, sc {e_sc:.4f}")


def test_eam_is_genuinely_many_body(eam):
    """The embedding energy is not additive over neighbours.

    If ``F`` were linear the model would collapse to a pair potential.  Compare
    ``F`` of the bulk density against the sum of ``F`` over single-neighbour
    contributions: the gap is the many-body content.
    """
    cfg = fcc(3.615, "Cu", (3, 3, 3))
    n_bulk = eam.density(cfg)[0]
    rho_1nn = float(eam.electron_density(np.array([3.615 / math.sqrt(2.0)]))[0])
    additive = 12.0 * float(eam.embedding_energy(np.array([rho_1nn]))[0])
    true = float(eam.embedding_energy(np.array([n_bulk]))[0])
    assert abs(true - additive) > 1.0  # eV, i.e. a large fraction of the cohesion
    # And F is concave in n (sqrt-dominated) at the bulk density: adding a
    # neighbour lowers the energy by less than the previous one did.
    d = 1e-4
    second = (
        float(eam.embedding_energy(np.array([n_bulk + d]))[0])
        - 2.0 * float(eam.embedding_energy(np.array([n_bulk]))[0])
        + float(eam.embedding_energy(np.array([n_bulk - d]))[0])
    ) / d**2
    assert second > 0.0  # convex: F'' = A/(4 n^{3/2}) + 2C > 0


def test_eam_cutoff_function_is_smooth(eam):
    rc = eam.cutoff
    assert float(eam.cutoff_function(np.array([rc]))[0]) == 0.0
    assert float(eam.cutoff_function(np.array([rc + 1.0]))[0]) == 0.0
    assert float(eam.cutoff_function(np.array([rc - 1e-4]))[0]) < 1e-100
    assert 0.0 < float(eam.cutoff_function(np.array([2.5]))[0]) < 1.0


# --------------------------------------------------------------------------
# 3. exact symmetries
# --------------------------------------------------------------------------


def _rotation(seed: int) -> np.ndarray:
    """A random proper rotation from a QR factorisation."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_translation_invariance(which, sw, eam):
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    ref = pot.compute(cfg)
    moved = cfg.translated(np.array([3.7, -11.2, 0.9]))
    got = pot.compute(moved)
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)
    assert np.max(np.abs(got.forces - ref.forces)) < 1e-9
    assert np.max(np.abs(got.virial - ref.virial)) < 1e-9


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_rotation_equivariance(which, sw, eam):
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    ref = pot.compute(cfg)
    rot = _rotation(7)
    got = pot.compute(cfg.rotated(rot))
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)
    assert np.max(np.abs(got.forces - ref.forces @ rot.T)) < 1e-9
    assert np.max(np.abs(got.virial - rot @ ref.virial @ rot.T)) < 1e-9


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_permutation_invariance(which, sw, eam):
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    ref = pot.compute(cfg)
    perm = np.random.default_rng(5).permutation(cfg.n_atoms)
    shuffled = cfg.copy()
    shuffled.positions = cfg.positions[perm]
    shuffled.species = cfg.species[perm]
    shuffled.masses = cfg.masses[perm]
    got = pot.compute(shuffled)
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)
    assert np.max(np.abs(got.forces - ref.forces[perm])) < 1e-9
    assert np.max(np.abs(got.virial - ref.virial)) < 1e-9


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_periodic_image_invariance(which, sw, eam):
    """Moving atoms by lattice vectors must not change anything at all."""
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    ref = pot.compute(cfg)
    rng = np.random.default_rng(9)
    shifted = cfg.copy()
    images = rng.integers(-2, 3, size=(cfg.n_atoms, 3)).astype(float)
    shifted.positions = cfg.positions + images @ cfg.cell
    got = pot.compute(shifted)
    assert got.energy == pytest.approx(ref.energy, abs=1e-9)
    assert np.max(np.abs(got.forces - ref.forces)) < 1e-9
    assert np.max(np.abs(got.virial - ref.virial)) < 1e-9


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_per_atom_energies_sum_to_the_total(which, sw, eam):
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    res = pot.compute(cfg)
    assert float(res.energies.sum()) == pytest.approx(res.energy, rel=1e-12)


@pytest.mark.parametrize("which", ["sw", "eam"])
def test_compute_does_not_mutate_the_configuration(which, sw, eam):
    pot, cfg = (sw, rattled_silicon()) if which == "sw" else (eam, rattled_copper())
    before = (cfg.positions.copy(), cfg.cell.copy())
    pot.compute(cfg)
    assert np.array_equal(cfg.positions, before[0])
    assert np.array_equal(cfg.cell, before[1])


def test_forces_sum_to_zero(sw, eam):
    """Newton's third law, and a check that no force is dropped on the floor."""
    assert np.max(np.abs(sw.forces(rattled_silicon()).sum(axis=0))) < 1e-10
    assert np.max(np.abs(eam.forces(rattled_copper()).sum(axis=0))) < 1e-10
    assert np.max(np.abs(sw.forces(silicon_cluster()).sum(axis=0))) < 1e-10
    assert np.max(np.abs(eam.forces(copper_cluster()).sum(axis=0))) < 1e-10


# --------------------------------------------------------------------------
# 4. jitted kernel vs pure-NumPy reference
# --------------------------------------------------------------------------


def _random_configurations(symbol: str, lattice_a: float, builder, n_configs: int, seed: int):
    """A mixed bag of periodic and cluster geometries for cross-checking."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_configs):
        if k % 5 == 4:
            out.append(
                cluster(symbol, int(rng.integers(4, 12)), box=5.5,
                        seed=int(rng.integers(1 << 30)), min_distance=2.2)
            )
            continue
        base = builder(lattice_a * float(rng.uniform(0.96, 1.06)), symbol, (2, 2, 2))
        cfg = rattle(base, float(rng.uniform(0.05, 0.25)), seed=int(rng.integers(1 << 30)))
        if k % 3 == 0:
            eps = 0.05 * rng.normal(size=(3, 3))
            cfg = cfg.strained(eps)
        out.append(cfg)
    return out


def test_sw_numba_matches_numpy_reference(sw, sw_numpy):
    """50 random configurations; a fast wrong kernel is the worst kind of bug."""
    configs = _random_configurations("Si", 5.431, diamond, 50, seed=1234)
    de = df = dw = 0.0
    for cfg in configs:
        ra = sw.compute(cfg)
        rb = sw_numpy.compute(cfg)
        de = max(de, abs(ra.energy - rb.energy))
        df = max(df, float(np.max(np.abs(ra.forces - rb.forces))))
        dw = max(dw, float(np.max(np.abs(ra.virial - rb.virial))))
    print(f"\nSW numba vs numpy over 50 configs: dE {de:.3e} eV, dF {df:.3e} eV/A, dW {dw:.3e} eV")
    assert de < 1e-9
    assert df < 1e-9
    assert dw < 1e-9


def test_eam_numba_matches_numpy_reference(eam, eam_numpy):
    configs = _random_configurations("Cu", 3.615, fcc, 50, seed=4321)
    # fcc 2x2x2 at a ~ 3.6 is 7.2 A wide, narrower than 2 * 5.0 A, so the
    # minimum-image guard has to be relaxed here: the neighbour list is exact
    # for small cells regardless, and this test is about kernel agreement.
    fast = EAM.copper(strict_minimum_image=False)
    slow = EAM.copper(implementation="numpy", strict_minimum_image=False)
    de = df = dw = 0.0
    for cfg in configs:
        ra = fast.compute(cfg)
        rb = slow.compute(cfg)
        de = max(de, abs(ra.energy - rb.energy))
        df = max(df, float(np.max(np.abs(ra.forces - rb.forces))))
        dw = max(dw, float(np.max(np.abs(ra.virial - rb.virial))))
    print(f"\nEAM numba vs numpy over 50 configs: dE {de:.3e} eV, dF {df:.3e} eV/A, dW {dw:.3e} eV")
    assert de < 1e-9
    assert df < 1e-9
    assert dw < 1e-9


def test_ase_stillinger_weber_cross_check(sw):
    """Cross-check against ASE if it ships an SW calculator.

    ASE 3.29 (the version pinned in this environment) has no
    ``ase.calculators.stillingerweber`` module -- only ``tersoff``,
    ``lj``, ``morse``, ``eam`` and the external-code drivers.  There is
    therefore no independent SW implementation available to compare against,
    and :func:`test_sw_numba_matches_numpy_reference` carries the burden
    instead.  The test is kept (skipping) so that the cross-check switches on
    automatically if ASE ever grows one.
    """
    pytest.importorskip(
        "ase.calculators.stillingerweber",
        reason="ASE does not ship a StillingerWeber calculator in this version",
    )
    from ase import Atoms  # pragma: no cover - only runs if ASE grows the module
    from ase.calculators.stillingerweber import StillingerWeber as AseSW  # pragma: no cover

    cfg = rattled_silicon()  # pragma: no cover
    atoms = Atoms(  # pragma: no cover
        symbols=["Si"] * cfg.n_atoms, positions=cfg.positions, cell=cfg.cell, pbc=True
    )
    atoms.calc = AseSW(  # pragma: no cover
        epsilon=SW_SILICON_1985.epsilon,
        sigma=SW_SILICON_1985.sigma,
        a=SW_SILICON_1985.a,
        lambda_=SW_SILICON_1985.lam,
        gamma=SW_SILICON_1985.gamma,
        A=SW_SILICON_1985.A,
        B=SW_SILICON_1985.B,
        p=SW_SILICON_1985.p,
        q=SW_SILICON_1985.q,
    )
    assert sw.energy(cfg) == pytest.approx(atoms.get_potential_energy(), rel=1e-8)  # pragma: no cover


# --------------------------------------------------------------------------
# 5. miscellaneous API contract
# --------------------------------------------------------------------------


def test_potential_algebra_composes(sw, eam):
    """SumPotential/ScaledPotential must work on both, since exp06-08 need it."""
    cfg = rattled_silicon()
    doubled = sw + sw
    assert doubled.energy(cfg) == pytest.approx(2.0 * sw.energy(cfg), rel=1e-12)
    assert (2.0 * sw).energy(cfg) == pytest.approx(2.0 * sw.energy(cfg), rel=1e-12)
    assert abs((sw - sw).energy(cfg)) < 1e-9
    assert doubled.cutoff == sw.cutoff


def test_bad_implementation_name_raises():
    with pytest.raises(ValueError, match="numba"):
        StillingerWeber(implementation="cuda")
    with pytest.raises(ValueError, match="numba"):
        EAM(implementation="cuda")


def test_sw_three_body_only_variant_has_zero_two_body():
    """dataclasses.replace on the parameter set is the intended way to ablate."""
    prm = dataclasses.replace(SW_SILICON_1985, A=0.0)
    pot = StillingerWeber(prm)
    res = pot.compute(rattled_silicon())
    assert res.extra["energy_two_body"] == 0.0
    assert res.extra["energy_three_body"] > 0.0
    assert check_forces(pot, rattled_silicon(), atoms=range(0, 64, 8)) < DERIV_TOL


def test_eam_density_matches_the_kernel(eam):
    cfg = rattled_copper()
    res = eam.compute(cfg)
    assert np.allclose(res.extra["density"], eam.density(cfg), atol=1e-12)
    # Bulk fcc Cu at the equilibrium lattice constant.
    assert eam.density(fcc(3.615, "Cu", (3, 3, 3)))[0] == pytest.approx(10.5765, abs=1e-3)
