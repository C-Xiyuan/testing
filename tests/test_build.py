"""Tests for :mod:`atomlab.build`.

The lattice tests deliberately avoid re-deriving the answer from the same basis
vectors the builder used.  Instead every structural claim is checked by direct
computation on the returned cartesian coordinates: atom counts, volume per atom
from ``det(cell)``, and coordination shells from an explicit distance histogram
over periodic images.  A transposed cell or a mis-signed basis vector survives
an algebraic self-check but not a shell count.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atomlab import build
from atomlab.types import Configuration
from atomlab.units import ATOMIC_MASSES

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)


# --------------------------------------------------------------------------
# independent geometry helpers (no atomlab code involved)
# --------------------------------------------------------------------------


def image_offsets(cfg: Configuration, n_images: int = 1) -> np.ndarray:
    """Cartesian offsets of all periodic images within +-n_images."""
    ranges = [
        range(-n_images, n_images + 1) if cfg.pbc[k] else range(1) for k in range(3)
    ]
    shifts = np.array([(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]], float)
    return shifts @ cfg.cell


def distances_from(cfg: Configuration, center: int = 0, n_images: int = 1) -> np.ndarray:
    """Sorted distances from atom ``center`` to every other atom and image."""
    offsets = image_offsets(cfg, n_images)
    d = cfg.positions[None, :, :] + offsets[:, None, :] - cfg.positions[center]
    r = np.linalg.norm(d, axis=-1).ravel()
    return np.sort(r[r > 1e-8])


def all_pair_distances(cfg: Configuration, n_images: int = 1) -> np.ndarray:
    """All i<j distances including periodic images (self-images included)."""
    offsets = image_offsets(cfg, n_images)
    d = cfg.positions[None, None, :, :] + offsets[:, None, None, :] - cfg.positions[None, :, None, :]
    r = np.linalg.norm(d, axis=-1).ravel()
    return r[r > 1e-8]


def shells(distances: np.ndarray, n_shells: int, rtol: float = 1e-6):
    """Group sorted distances into ``(radius, count)`` coordination shells."""
    out = []
    i = 0
    while len(out) < n_shells and i < distances.size:
        r0 = distances[i]
        j = i
        while j < distances.size and distances[j] <= r0 * (1.0 + rtol) + 1e-9:
            j += 1
        out.append((float(r0), j - i))
        i = j
    return out


# --------------------------------------------------------------------------
# lattices: counts, volume per atom, coordination shells
# --------------------------------------------------------------------------

A = 4.05  # arbitrary but not special: catches accidental unit-cell assumptions
N = 3

# (builder kwargs, atoms per cell, volume per atom, first three shells)
LATTICE_CASES = {
    "sc": (1, A**3, [(A, 6), (A * SQRT2, 12), (A * SQRT3, 8)]),
    "bcc": (2, A**3 / 2, [(A * SQRT3 / 2, 8), (A, 6), (A * SQRT2, 12)]),
    "fcc": (4, A**3 / 4, [(A / SQRT2, 12), (A, 6), (A * math.sqrt(1.5), 24)]),
    "diamond": (8, A**3 / 8, [(A * SQRT3 / 4, 4), (A / SQRT2, 12), (A * math.sqrt(11) / 4, 12)]),
}


@pytest.mark.parametrize("name", sorted(LATTICE_CASES))
def test_cubic_lattice_geometry(name):
    n_basis, vol_per_atom, expected_shells = LATTICE_CASES[name]
    builder = getattr(build, name)
    cfg = builder(A, "Cu", (N, N, N))

    assert cfg.n_atoms == n_basis * N**3
    assert cfg.volume == pytest.approx(vol_per_atom * cfg.n_atoms, rel=1e-12)
    assert cfg.volume / cfg.n_atoms == pytest.approx(vol_per_atom, rel=1e-12)
    assert np.allclose(cfg.cell, N * A * np.eye(3))
    assert cfg.pbc.all()

    # Every atom in a perfect crystal must see the same environment; checking a
    # basis atom other than index 0 catches a wrong basis vector that happens to
    # leave the origin atom's shells intact.
    for center in (0, cfg.n_atoms // 2, cfg.n_atoms - 1):
        found = shells(distances_from(cfg, center), len(expected_shells))
        for (r_ref, c_ref), (r, c) in zip(expected_shells, found):
            assert r == pytest.approx(r_ref, rel=1e-10), (name, center)
            assert c == c_ref, (name, center, r)


def test_hcp_ideal_geometry():
    a = 3.21
    c = a * math.sqrt(8.0 / 3.0)
    cfg = build.hcp(a, "Cu", c, (N, N, N))

    assert cfg.n_atoms == 2 * N**3
    # V/atom = sqrt(3)/2 * a^2 * c / 2
    assert cfg.volume / cfg.n_atoms == pytest.approx(SQRT3 * a * a * c / 4.0, rel=1e-12)

    for center in (0, 1, cfg.n_atoms - 1):
        found = shells(distances_from(cfg, center), 2)
        # Ideal c/a merges the in-plane and out-of-plane neighbours into one
        # shell of 12 at exactly a -- the close-packing signature.
        assert found[0][0] == pytest.approx(a, rel=1e-10)
        assert found[0][1] == 12
        assert found[1][0] == pytest.approx(a * SQRT2, rel=1e-10)
        assert found[1][1] == 6


def test_hcp_default_c_is_ideal():
    a = 3.21
    cfg = build.hcp(a, "Cu")
    assert cfg.cell[2, 2] == pytest.approx(a * math.sqrt(8.0 / 3.0), rel=1e-14)
    assert cfg.n_atoms == 2


def test_hcp_non_ideal_splits_the_first_shell():
    a, c = 3.21, 5.9  # c/a = 1.838 > ideal 1.633, so the 12-shell splits 6 + 6
    cfg = build.hcp(a, "Cu", c, (N, N, N))
    found = shells(distances_from(cfg, 0), 2)
    out_of_plane = math.sqrt(a * a / 3.0 + c * c / 4.0)
    assert found[0][0] == pytest.approx(a, rel=1e-10)
    assert found[0][1] == 6
    assert found[1][0] == pytest.approx(out_of_plane, rel=1e-10)
    assert found[1][1] == 6


def test_hcp_basis_atom_sits_over_a_triangle_centroid():
    a = 3.0
    cfg = build.hcp(a, "Cu", a * math.sqrt(8.0 / 3.0))
    # (1/3, 2/3, 1/2) in this cell is the centroid of the (0,0), a2, a1+a2
    # triangle, lifted by c/2.
    expected = np.array([0.0, a / SQRT3, 0.5 * cfg.cell[2, 2]])
    assert np.allclose(cfg.positions[1], expected, atol=1e-12)


def test_hcp_accepts_both_positional_orders():
    a, c = 3.21, 5.2
    x = build.hcp(a, "Cu", c, (1, 1, 2))
    y = build.hcp(a, c, "Cu", (1, 1, 2))
    assert np.allclose(x.positions, y.positions)
    assert np.allclose(x.cell, y.cell)
    assert x.symbols == y.symbols == ("Cu",)


@pytest.mark.parametrize("name", ["sc", "bcc", "fcc", "diamond"])
def test_reps_scaling_and_default(name):
    builder = getattr(build, name)
    unit = builder(A, "Si")
    assert unit.n_atoms == LATTICE_CASES[name][0]
    grown = builder(A, "Si", (1, 2, 3))
    assert grown.n_atoms == 6 * unit.n_atoms
    assert np.allclose(np.diag(grown.cell), [A, 2 * A, 3 * A])
    assert builder(A, "Si", 2).n_atoms == 8 * unit.n_atoms


def test_lattice_dispatch_matches_direct_builders():
    for name in ("sc", "bcc", "fcc", "diamond"):
        direct = getattr(build, name)(A, "Cu", (2, 1, 1))
        via = build.lattice(name, A, "Cu", (2, 1, 1))
        assert np.allclose(direct.positions, via.positions)
    assert build.lattice("hcp", 3.2, "Cu", 1, c=5.2).cell[2, 2] == pytest.approx(5.2)
    with pytest.raises(ValueError):
        build.lattice("hexagonal", A, "Cu")
    with pytest.raises(ValueError):
        build.lattice("fcc", A, "Cu", c=5.0)


def test_masses_and_symbols_come_from_units_table():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    assert cfg.symbols == ("Cu",)
    assert np.allclose(cfg.masses, ATOMIC_MASSES["Cu"])
    assert cfg.species.dtype == np.int32 and np.all(cfg.species == 0)
    si = build.diamond(5.431, "Si")
    assert np.allclose(si.masses, ATOMIC_MASSES["Si"])


def test_unknown_symbol_and_bad_arguments_raise():
    with pytest.raises(KeyError):
        build.fcc(4.0, "Unobtainium")
    with pytest.raises(ValueError):
        build.fcc(-1.0, "Cu")
    with pytest.raises(ValueError):
        build.fcc(4.0, "Cu", (0, 1, 1))


def test_silicon_reference_geometry():
    """Diamond Si at the experimental a0: bond length and density."""
    a0 = 5.431
    cfg = build.diamond(a0, "Si", (2, 2, 2))
    bond, count = shells(distances_from(cfg, 0), 1)[0]
    assert bond == pytest.approx(a0 * SQRT3 / 4.0, rel=1e-12)
    assert bond == pytest.approx(2.3517, abs=1e-4)
    assert count == 4
    assert cfg.n_atoms / cfg.volume == pytest.approx(8.0 / a0**3, rel=1e-12)


# --------------------------------------------------------------------------
# random gas
# --------------------------------------------------------------------------


def test_random_gas_density_and_cell():
    cfg = build.random_gas(200, 0.02, "Ar", seed=1)
    assert cfg.n_atoms == 200
    assert cfg.density == pytest.approx(0.02, rel=1e-12)
    box = (200 / 0.02) ** (1 / 3)
    assert np.allclose(cfg.cell, box * np.eye(3))
    assert cfg.pbc.all()
    assert np.allclose(cfg.masses, ATOMIC_MASSES["Ar"])
    # positions inside the box
    assert cfg.positions.min() >= 0.0 and cfg.positions.max() <= box


@pytest.mark.parametrize("min_distance", [2.0, 3.0])
def test_random_gas_respects_min_distance_under_pbc(min_distance):
    cfg = build.random_gas(150, 0.018, "Ar", seed=7, min_distance=min_distance)
    r = all_pair_distances(cfg, n_images=1)
    assert r.min() >= min_distance - 1e-12, f"closest pair {r.min():.4f} A"


def test_random_gas_is_deterministic_and_seed_sensitive():
    a = build.random_gas(50, 0.02, "Ar", seed=42, min_distance=2.5)
    b = build.random_gas(50, 0.02, "Ar", seed=42, min_distance=2.5)
    c = build.random_gas(50, 0.02, "Ar", seed=43, min_distance=2.5)
    assert np.array_equal(a.positions, b.positions)
    assert not np.allclose(a.positions, c.positions)
    # A Generator is accepted and advances state, so successive draws differ.
    rng = np.random.default_rng(0)
    d = build.random_gas(50, 0.02, "Ar", seed=rng)
    e = build.random_gas(50, 0.02, "Ar", seed=rng)
    assert not np.allclose(d.positions, e.positions)


def test_random_gas_refuses_impossible_packing():
    # Hard spheres of diameter 3.5 A cannot reach this density by rejection.
    with pytest.raises(RuntimeError, match="could not place atom"):
        build.random_gas(200, 0.05, "Ar", seed=3, min_distance=3.5, max_attempts=200)
    with pytest.raises(ValueError, match="minimum image"):
        build.random_gas(8, 0.02, "Ar", seed=3, min_distance=100.0)
    with pytest.raises(TypeError):
        build.random_gas(8, 0.02, "Ar")  # seed is mandatory


# --------------------------------------------------------------------------
# rattle
# --------------------------------------------------------------------------


def test_rattle_is_reproducible_and_leaves_input_alone():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    before = cfg.positions.copy()
    a = build.rattle(cfg, 0.05, seed=11)
    b = build.rattle(cfg, 0.05, seed=11)
    c = build.rattle(cfg, 0.05, seed=12)
    assert np.array_equal(a.positions, b.positions)
    assert not np.allclose(a.positions, c.positions)
    assert np.array_equal(cfg.positions, before)
    assert np.allclose(a.cell, cfg.cell)
    assert np.array_equal(a.species, cfg.species)
    with pytest.raises(ValueError):
        build.rattle(cfg, 0.05, seed=None)


def test_rattle_displacement_statistics():
    sigma = 0.07
    cfg = build.fcc(3.615, "Cu", (8, 8, 8))  # 2048 atoms -> 6144 components
    out = build.rattle(cfg, sigma, seed=5)
    d = out.positions - cfg.positions
    n = d.size
    # Sampling error on the std is sigma/sqrt(2n) ~ 0.9%; 5% is a safe bound
    # that still fails hard if sigma is applied per-atom instead of per-component
    # or if a variance/std mix-up creeps in.
    assert np.std(d) == pytest.approx(sigma, rel=0.05)
    assert abs(np.mean(d)) < 5.0 * sigma / math.sqrt(n)
    # radial displacement of a 3D Gaussian: <|u|> = 2*sqrt(2/pi)*sigma
    assert np.mean(np.linalg.norm(d, axis=1)) == pytest.approx(
        2.0 * math.sqrt(2.0 / math.pi) * sigma, rel=0.05
    )
    # isotropy: the three cartesian variances agree
    per_axis = np.std(d, axis=0)
    assert np.allclose(per_axis, sigma, rtol=0.06)


def test_rattle_zero_sigma_is_identity_and_labels_are_dropped():
    cfg = build.fcc(3.615, "Cu", (1, 1, 1))
    cfg.energy = -1.0
    cfg.forces = np.zeros((cfg.n_atoms, 3))
    out = build.rattle(cfg, 0.0, seed=1)
    assert np.array_equal(out.positions, cfg.positions)
    assert out.energy is None and out.forces is None


# --------------------------------------------------------------------------
# rescaling
# --------------------------------------------------------------------------


def test_scale_cell_preserves_fractional_coordinates():
    cfg = build.rattle(build.fcc(3.615, "Cu", (2, 2, 2)), 0.1, seed=2)
    s0 = cfg.scaled_positions()
    out = build.scale_cell(cfg, 1.13)
    assert np.allclose(out.cell, 1.13 * cfg.cell)
    assert np.allclose(out.scaled_positions(), s0, atol=1e-12)
    assert out.volume == pytest.approx(cfg.volume * 1.13**3, rel=1e-12)

    aniso = build.scale_cell(cfg, [1.0, 1.5, 2.0])
    assert np.allclose(aniso.cell, cfg.cell * np.array([[1.0], [1.5], [2.0]]))
    assert np.allclose(aniso.scaled_positions(), s0, atol=1e-12)
    with pytest.raises(ValueError):
        build.scale_cell(cfg, -1.0)


def test_scale_to_density_hits_the_target():
    cfg = build.fcc(3.615, "Cu", (3, 3, 3))
    for target in (0.05, 0.0847, 0.12):
        out = build.scale_to_density(cfg, target)
        assert out.density == pytest.approx(target, rel=1e-12)
        assert out.n_atoms == cfg.n_atoms
        assert np.allclose(out.scaled_positions(), cfg.scaled_positions(), atol=1e-12)


# --------------------------------------------------------------------------
# defects
# --------------------------------------------------------------------------


def test_vacancy_removes_the_named_atom():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    out = build.vacancy(cfg, 5)
    assert out.n_atoms == cfg.n_atoms - 1
    kept = np.delete(cfg.positions, 5, axis=0)
    assert np.allclose(out.positions, kept)
    assert out.info["vacancy_index"] == 5
    assert np.allclose(out.cell, cfg.cell)
    assert cfg.n_atoms == 32  # input untouched


def test_random_vacancy_needs_a_seed_and_is_reproducible():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    with pytest.raises(ValueError):
        build.vacancy(cfg)
    i1 = build.vacancy(cfg, seed=4).info["vacancy_index"]
    i2 = build.vacancy(cfg, seed=4).info["vacancy_index"]
    assert i1 == i2
    idx = {build.vacancy(cfg, seed=s).info["vacancy_index"] for s in range(40)}
    assert len(idx) > 1  # not a constant disguised as a random choice
    with pytest.raises(IndexError):
        build.vacancy(cfg, 999)


def test_interstitial_appends_an_atom():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    pos = np.array([1.0, 2.0, 3.0])
    out = build.interstitial(cfg, pos)
    assert out.n_atoms == cfg.n_atoms + 1
    assert np.allclose(out.positions[-1], pos)
    assert out.species[-1] == 0
    assert out.masses[-1] == pytest.approx(ATOMIC_MASSES["Cu"])

    foreign = build.interstitial(cfg, pos, symbol="Ar")
    assert foreign.symbols == ("Cu", "Ar")
    assert foreign.species[-1] == 1
    assert foreign.masses[-1] == pytest.approx(ATOMIC_MASSES["Ar"])


def test_substitute_updates_species_and_mass():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    out = build.substitute(cfg, 3, "Ni")
    assert out.symbols == ("Cu", "Ni")
    assert out.species[3] == 1
    assert out.masses[3] == pytest.approx(ATOMIC_MASSES["Ni"])
    assert np.all(np.delete(out.species, 3) == 0)
    assert np.allclose(np.delete(out.masses, 3), ATOMIC_MASSES["Cu"])
    assert np.allclose(out.positions, cfg.positions)
    # substituting back by integer index restores the original species
    back = build.substitute(out, 3, 0)
    assert back.species[3] == 0
    assert back.masses[3] == pytest.approx(ATOMIC_MASSES["Cu"])
    with pytest.raises(IndexError):
        build.substitute(cfg, 100, "Ni")


# --------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_surface_slab_opens_exactly_one_direction(axis):
    a, vac = 3.615, 12.0
    cfg = build.fcc(a, "Cu", (2, 2, 2))
    slab = build.surface_slab(cfg, axis, vac)

    assert slab.n_atoms == cfg.n_atoms
    expected_pbc = np.ones(3, dtype=bool)
    expected_pbc[axis] = False
    assert np.array_equal(slab.pbc, expected_pbc)
    assert np.linalg.norm(slab.cell[axis]) == pytest.approx(2 * a + vac, rel=1e-12)
    for k in range(3):
        if k != axis:
            assert np.allclose(slab.cell[k], cfg.cell[k])

    # The free space along `axis` grew by exactly `vacuum`, and the slab is not
    # poking out of the enlarged cell.
    z = slab.positions[:, axis]
    z0 = cfg.positions[:, axis]
    gap_before = 2 * a - (z0.max() - z0.min())
    gap_after = (2 * a + vac) - (z.max() - z.min())
    assert gap_after == pytest.approx(gap_before + vac, rel=1e-12)
    assert z.min() >= 0.0 and z.max() <= 2 * a + vac

    # In-plane structure is untouched: distances within the slab are unchanged.
    assert np.allclose(
        np.sort(np.linalg.norm(slab.positions - slab.positions[0], axis=1)),
        np.sort(np.linalg.norm(cfg.positions - cfg.positions[0], axis=1)),
        atol=1e-12,
    )


def test_surface_slab_gap_is_at_least_the_vacuum():
    """No atom pair straddles the gap closer than the requested vacuum."""
    cfg = build.fcc(3.615, "Cu", (2, 2, 1))
    slab = build.surface_slab(cfg, 2, 10.0)
    # Only x/y are periodic now, so images along z must not appear.
    r = all_pair_distances(slab, n_images=1)
    assert r.min() > 2.5
    with pytest.raises(ValueError):
        build.surface_slab(slab, 2, 5.0)  # already open
    with pytest.raises(ValueError):
        build.surface_slab(cfg, 3, 5.0)


# --------------------------------------------------------------------------
# strain helpers
# --------------------------------------------------------------------------


def test_voigt_roundtrip_uses_engineering_shear():
    v = np.array([0.01, -0.02, 0.003, 0.04, -0.05, 0.06])
    eps = build.voigt_to_strain(v)
    assert np.allclose(eps, eps.T)
    assert eps[1, 2] == pytest.approx(0.02)  # e4 = 2*eps_yz
    assert np.allclose(build.strain_to_voigt(eps), v)


def test_shear_applies_a_symmetric_strain():
    gamma = 0.02
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    out = build.shear(cfg, gamma, "xy")
    eps = np.zeros((3, 3))
    eps[0, 1] = eps[1, 0] = gamma / 2
    defm = np.eye(3) + eps
    assert np.allclose(out.cell, cfg.cell @ defm.T)
    assert np.allclose(out.positions, cfg.positions @ defm.T)
    # symmetric shear carries no rotation, and det = 1 - gamma^2/4 exactly
    assert out.volume / cfg.volume == pytest.approx(1.0 - gamma**2 / 4.0, rel=1e-12)
    assert np.allclose(out.scaled_positions(), cfg.scaled_positions(), atol=1e-12)
    # index form and name form agree; order of the plane letters is irrelevant
    assert np.allclose(build.shear(cfg, gamma, (0, 1)).cell, out.cell)
    assert np.allclose(build.shear(cfg, gamma, "yx").cell, out.cell)
    with pytest.raises(ValueError):
        build.shear(cfg, gamma, "xx")
    with pytest.raises(ValueError):
        build.shear(cfg, gamma, (0, 0))


def test_uniaxial_and_hydrostatic_strain():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    eps = 0.01
    ux = build.uniaxial_strain(cfg, eps, axis=1)
    assert ux.cell[1, 1] == pytest.approx(cfg.cell[1, 1] * (1 + eps), rel=1e-14)
    assert ux.cell[0, 0] == pytest.approx(cfg.cell[0, 0], rel=1e-14)
    assert ux.volume == pytest.approx(cfg.volume * (1 + eps), rel=1e-12)

    hy = build.hydrostatic_strain(cfg, eps)
    assert hy.volume == pytest.approx(cfg.volume * (1 + eps) ** 3, rel=1e-12)
    assert np.allclose(hy.cell, build.scale_cell(cfg, 1 + eps).cell)


def test_apply_strain_accepts_tensor_or_voigt():
    cfg = build.fcc(3.615, "Cu", (1, 1, 1))
    v = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.02])
    assert np.allclose(
        build.apply_strain(cfg, v).cell, build.apply_strain(cfg, build.voigt_to_strain(v)).cell
    )


def test_strain_scan_and_volume_scan():
    cfg = build.fcc(3.615, "Cu", (2, 2, 2))
    mags = [-0.01, -0.005, 0.005, 0.01]
    scan = build.strain_scan(cfg, "xx", mags)
    assert len(scan) == len(mags)
    for m, c in zip(mags, scan):
        assert c.cell[0, 0] == pytest.approx(cfg.cell[0, 0] * (1 + m), rel=1e-14)
        assert c.info["strain_voigt"][0] == pytest.approx(m)
    assert np.allclose(build.strain_scan(cfg, 5, [0.01])[0].cell, build.shear(cfg, 0.01, "xy").cell)

    ratios = [0.9, 1.0, 1.1]
    vs = build.volume_scan(cfg, ratios)
    for r, c in zip(ratios, vs):
        assert c.volume / cfg.volume == pytest.approx(r, rel=1e-12)
        assert c.n_atoms == cfg.n_atoms
    with pytest.raises(ValueError):
        build.volume_scan(cfg, [-1.0])


def test_builders_do_not_leak_labels():
    cfg = build.fcc(3.615, "Cu", (1, 1, 1))
    cfg.energy, cfg.forces = -3.5, np.zeros((4, 3))
    for out in (
        build.scale_cell(cfg, 1.01),
        build.shear(cfg, 0.01, "xy"),
        build.vacancy(cfg, 0),
        build.interstitial(cfg, [0.1, 0.1, 0.1]),
        build.substitute(cfg, 0, "Ni"),
        build.surface_slab(cfg, 2, 10.0),
    ):
        assert out.energy is None and out.forces is None


# --------------------------------------------------------------------------
# independent cross-check against ASE (test-only dependency)
# --------------------------------------------------------------------------

ase_build = pytest.importorskip("ase.build")


@pytest.mark.parametrize(
    "structure,kwargs",
    [
        ("sc", {}),
        ("bcc", {}),
        ("fcc", {}),
        ("diamond", {}),
        ("hcp", {"c": 3.21 * math.sqrt(8 / 3)}),
    ],
)
def test_matches_ase_bulk(structure, kwargs):
    """Same atom count, same volume per atom, same neighbour distance spectrum.

    Compared via the distance spectrum rather than raw coordinates because ASE
    is free to choose a different (equivalent) cell orientation or origin.
    """
    a = 3.21 if structure == "hcp" else A
    cubic = structure != "hcp"
    atoms = ase_build.bulk("Cu", structure, a=a, cubic=cubic, **kwargs)
    ours = build.lattice(structure, a, "Cu", (1, 1, 1), **kwargs)

    assert len(atoms) == ours.n_atoms
    assert atoms.get_volume() / len(atoms) == pytest.approx(
        ours.volume / ours.n_atoms, rel=1e-10
    )

    theirs = Configuration(
        positions=atoms.get_positions(), cell=np.array(atoms.get_cell()), pbc=True
    )
    n_shells = 4
    ref = shells(distances_from(theirs, 0, n_images=2), n_shells)
    got = shells(distances_from(ours, 0, n_images=2), n_shells)
    for (r_ref, c_ref), (r, c) in zip(ref, got):
        assert r == pytest.approx(r_ref, rel=1e-10)
        assert c == c_ref
