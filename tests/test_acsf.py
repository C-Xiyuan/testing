"""Tests for :mod:`atomlab.models.descriptors.acsf`.

The load-bearing test in this file is the finite-difference check on the
descriptor derivatives.  A model whose descriptor derivatives are subtly wrong
still trains to a plausible energy error and then produces wrong physics, which
is exactly the failure mode this repository exists to characterise -- so it must
not be committed here.  Everything else (invariances, kernel agreement, cutoff
continuity) is there to catch the ways a derivative check can pass while the
descriptor is still not the function it claims to be.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from atomlab.build import diamond, fcc, random_gas, rattle, shear
from atomlab.models.descriptors.acsf import ACSF, cutoff_function
from atomlab.types import Configuration

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def dense_derivatives(out, n_atoms: int) -> np.ndarray:
    """Expand the sparse ``(P, D, 3)`` layout into ``(N, N, D, 3)``.

    ``dense[i, a, d, k] = dG[i, d] / dr[a, k]``.  Duplicate ``(i, a)`` rows (an
    atom seen through more than one periodic image contributes one row, but the
    self rows can coincide with a self-image row) are summed, which is exactly
    what :meth:`DescriptorOutput.forces_from_energy_gradient` does.
    """
    n_features = out.n_features
    flat = np.zeros((n_atoms * n_atoms, n_features, 3))
    np.add.at(flat, out.pair_i * n_atoms + out.pair_j, out.derivatives)
    return flat.reshape(n_atoms, n_atoms, n_features, 3)


def numerical_derivatives(desc, cfg, atoms, delta: float = 1e-5) -> np.ndarray:
    """Central-difference ``dG[i, d] / dr[a, k]`` for ``a in atoms``.

    Returns ``(N, len(atoms), D, 3)``.  ``delta = 1e-5 A`` balances truncation
    against round-off for features of order 1.
    """
    out = np.zeros((cfg.n_atoms, len(atoms), desc.n_features, 3))
    for col, a in enumerate(atoms):
        for k in range(3):
            plus = cfg.copy()
            plus.positions[a, k] += delta
            minus = cfg.copy()
            minus.positions[a, k] -= delta
            fp = desc.compute(plus, derivatives=False).features
            fm = desc.compute(minus, derivatives=False).features
            out[:, col, :, k] = (fp - fm) / (2.0 * delta)
    return out


def small_acsf(species, cutoff, *, angular_kind="G4", cutoff_function_name="cosine"):
    """A deliberately tiny parameter set, so the O(N_neigh^2) tests stay fast."""
    radial = [[1.0, 0.0], [0.8, 1.8], [2.0, 3.0]]
    angular = [[0.02, 1.0, 1.0], [0.02, 4.0, -1.0], [0.15, 2.0, 1.0], [0.0, 16.0, -1.0]]
    return ACSF(
        species,
        cutoff,
        radial,
        angular,
        angular_kind=angular_kind,
        cutoff_function=cutoff_function_name,
    )


def random_rotation(seed: int) -> np.ndarray:
    """Uniform random rotation in SO(3) from a QR decomposition of a Gaussian."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


# -- the five reference geometries -----------------------------------------


def rattled_fcc() -> Configuration:
    return rattle(fcc(4.05, "Al", (2, 2, 2)), 0.08, seed=11)


def rattled_diamond() -> Configuration:
    # 5.431 A cell with a 4.2 A cutoff: the cell width is between Rc and 2 Rc,
    # so atoms see the same neighbour through more than one periodic image.
    # Rc also has to exceed the 3.84 A second-neighbour distance, otherwise
    # every G4 triple is killed by fc(r_jk) and the angular test is vacuous.
    return rattle(diamond(5.431, "Si", (2, 1, 1)), 0.10, seed=12)


def random_gas_config() -> Configuration:
    return random_gas(24, 0.02, "Ar", seed=13, min_distance=2.2)


def triclinic_config() -> Configuration:
    return rattle(shear(fcc(4.05, "Al", (2, 2, 2)), 0.18, "xy"), 0.07, seed=14)


def two_species_config() -> Configuration:
    base = rattle(fcc(4.05, "Al", (2, 2, 2)), 0.08, seed=15)
    species = np.zeros(base.n_atoms, dtype=np.int32)
    species[::3] = 1  # not a symmetric decoration, so the blocks really differ
    return Configuration(
        positions=base.positions,
        cell=base.cell,
        pbc=True,
        species=species,
        symbols=("Al", "Ni"),
    )


GEOMETRIES = {
    "rattled_fcc": (rattled_fcc, (0,), 4.0),
    "rattled_diamond": (rattled_diamond, (0,), 4.2),
    "random_gas": (random_gas_config, (0,), 4.0),
    "triclinic": (triclinic_config, (0,), 4.0),
    "two_species": (two_species_config, (0, 1), 4.0),
}


# --------------------------------------------------------------------------
# 1. analytic derivatives vs central differences  -- the non-negotiable test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("kind", ["G4", "G5"])
def test_descriptor_derivatives_match_finite_differences(geometry, kind):
    builder, species, cutoff = GEOMETRIES[geometry]
    cfg = builder()
    desc = small_acsf(species, cutoff, angular_kind=kind)

    out = desc.compute(cfg)
    analytic = dense_derivatives(out, cfg.n_atoms)

    # Guard against a vacuous check: if the angular block were identically zero
    # (which it is for G4 whenever Rc is below the second-neighbour distance)
    # the test would pass while exercising nothing.
    n_radial_cols = len(species) * desc.n_radial
    assert np.abs(out.features[:, :n_radial_cols]).max() > 1e-2
    assert np.abs(out.features[:, n_radial_cols:]).max() > 1e-2

    atoms = list(range(min(4, cfg.n_atoms)))
    numeric = numerical_derivatives(desc, cfg, atoms)

    err = np.max(np.abs(analytic[:, atoms] - numeric))
    assert err < 1e-6, f"{geometry}/{kind}: max |analytic - FD| = {err:.3e}"
    err_ang = np.max(np.abs(analytic[:, atoms, n_radial_cols:] - numeric[:, :, n_radial_cols:]))
    assert err_ang < 1e-6, f"{geometry}/{kind}: max angular |analytic - FD| = {err_ang:.3e}"


def test_derivatives_with_mixed_g4_g5_and_tanh_cutoff():
    """The two angular forms and the tanh cutoff share the same gradient code."""
    cfg = rattled_fcc()
    desc = ACSF(
        (0,),
        4.0,
        [[1.0, 0.0], [1.5, 2.2]],
        [[0.02, 1.0, 1.0], [0.05, 4.0, -1.0], [0.02, 2.0, -1.0], [0.10, 8.0, 1.0]],
        angular_kind=["G4", "G5", "G5", "G4"],
        cutoff_function="tanh",
    )
    out = desc.compute(cfg)
    analytic = dense_derivatives(out, cfg.n_atoms)
    atoms = [0, 1, 5]
    numeric = numerical_derivatives(desc, cfg, atoms)
    err = np.max(np.abs(analytic[:, atoms] - numeric))
    assert err < 1e-6, f"max |analytic - FD| = {err:.3e}"


def test_default_parameter_set_derivatives():
    """The set the models actually use, on a system with a full first shell."""
    cfg = rattled_fcc()
    desc = ACSF.default((0,), 4.6)
    assert 30 <= desc.n_features <= 60

    out = desc.compute(cfg)
    analytic = dense_derivatives(out, cfg.n_atoms)
    atoms = [0, 3]
    numeric = numerical_derivatives(desc, cfg, atoms)
    err = np.max(np.abs(analytic[:, atoms] - numeric))
    assert err < 1e-6, f"max |analytic - FD| = {err:.3e}"


def test_forces_from_energy_gradient_path():
    """Validate the sparse layout through the exact path the models use.

    A linear "energy" ``E = sum_i w . G_i`` has forces ``-dE/dr`` that
    :meth:`DescriptorOutput.forces_from_energy_gradient` must reproduce.  This
    is where a transposed index or a missing self term hides.
    """
    cfg = two_species_config()
    desc = small_acsf((0, 1), 4.0)
    rng = np.random.default_rng(7)
    w = rng.normal(size=desc.n_features)

    out = desc.compute(cfg)
    forces = out.forces_from_energy_gradient(np.tile(w, (cfg.n_atoms, 1)))

    delta = 1e-5
    numeric = np.zeros_like(forces)
    for a in range(cfg.n_atoms):
        for k in range(3):
            plus = cfg.copy()
            plus.positions[a, k] += delta
            minus = cfg.copy()
            minus.positions[a, k] -= delta
            ep = float((desc.compute(plus, derivatives=False).features @ w).sum())
            em = float((desc.compute(minus, derivatives=False).features @ w).sum())
            numeric[a, k] = -(ep - em) / (2.0 * delta)

    err = np.max(np.abs(forces - numeric))
    assert err < 1e-6, f"max |analytic - FD| forces = {err:.3e}"


# --------------------------------------------------------------------------
# 2. exact invariances
# --------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_translation_invariance(geometry):
    builder, species, cutoff = GEOMETRIES[geometry]
    cfg = builder()
    desc = small_acsf(species, cutoff)
    ref = desc.compute(cfg, derivatives=False).features
    moved = desc.compute(cfg.translated([1.234, -5.678, 9.1011]), derivatives=False).features
    assert np.max(np.abs(ref - moved)) < 1e-12


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_rotation_invariance(geometry):
    builder, species, cutoff = GEOMETRIES[geometry]
    cfg = builder()
    desc = small_acsf(species, cutoff)
    ref = desc.compute(cfg, derivatives=False).features
    for seed in (1, 2, 3):
        rot = random_rotation(seed)
        turned = desc.compute(cfg.rotated(rot), derivatives=False).features
        assert np.max(np.abs(ref - turned)) < 1e-12


def test_derivatives_rotate_equivariantly():
    """``dG/dr`` is a vector: rotating the system must rotate it."""
    cfg = rattled_fcc()
    desc = small_acsf((0,), 4.0)
    rot = random_rotation(4)

    a = dense_derivatives(desc.compute(cfg), cfg.n_atoms)
    b = dense_derivatives(desc.compute(cfg.rotated(rot)), cfg.n_atoms)
    assert np.max(np.abs(a @ rot.T - b)) < 1e-12


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_permutation_invariance(geometry):
    """Relabelling atoms permutes the rows of the feature matrix and nothing else."""
    builder, species, cutoff = GEOMETRIES[geometry]
    cfg = builder()
    desc = small_acsf(species, cutoff)
    ref = desc.compute(cfg, derivatives=False).features

    rng = np.random.default_rng(21)
    perm = rng.permutation(cfg.n_atoms)
    shuffled = cfg.copy()
    shuffled.positions = cfg.positions[perm]
    shuffled.species = cfg.species[perm]
    shuffled.masses = cfg.masses[perm]

    got = desc.compute(shuffled, derivatives=False).features
    assert np.max(np.abs(ref[perm] - got)) < 1e-12


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_periodic_image_invariance(geometry):
    """Moving one atom to an equivalent periodic image changes nothing."""
    builder, species, cutoff = GEOMETRIES[geometry]
    cfg = builder()
    desc = small_acsf(species, cutoff)
    ref = desc.compute(cfg, derivatives=False).features

    for atom, lattice in ((0, 0), (1, 1), (2, 2), (3, 0)):
        moved = cfg.copy()
        moved.positions[atom] += cfg.cell[lattice]
        got = desc.compute(moved, derivatives=False).features
        assert np.max(np.abs(ref - got)) < 1e-12, f"atom {atom} + a{lattice}"


def test_species_resolution_actually_resolves():
    """A two-species descriptor must not be a one-species descriptor in disguise."""
    cfg = two_species_config()
    two = small_acsf((0, 1), 4.0)
    one = small_acsf((0,), 4.0)

    collapsed = cfg.copy()
    collapsed.species = np.zeros(cfg.n_atoms, dtype=np.int32)

    f_two = two.compute(cfg, derivatives=False).features
    f_one = one.compute(collapsed, derivatives=False).features

    assert two.n_features == 2 * 3 + 3 * 4  # 2 radial blocks, 3 species pairs
    assert one.n_features == 3 + 4
    # Summing the per-species radial blocks must reproduce the species-blind one.
    summed = f_two[:, 0:3] + f_two[:, 3:6]
    assert np.max(np.abs(summed - f_one[:, 0:3])) < 1e-12
    # ... but the resolved features carry strictly more information.
    assert np.max(np.abs(f_two[:, 0:3] - f_two[:, 3:6])) > 1e-3


# --------------------------------------------------------------------------
# 3. numba kernel vs pure-numpy reference
# --------------------------------------------------------------------------


def test_kernel_matches_numpy_reference_on_random_configurations():
    """50 random configurations, features and derivatives, to 1e-12."""
    desc = ACSF(
        (0, 1),
        4.2,
        [[1.0, 0.0], [0.9, 1.6], [1.8, 3.0]],
        [[0.02, 1.0, 1.0], [0.05, 4.0, -1.0], [0.02, 2.0, 1.0], [0.0, 8.0, -1.0]],
        angular_kind=["G4", "G5", "G4", "G5"],
    )
    rng = np.random.default_rng(31)
    worst_f = 0.0
    worst_d = 0.0
    for trial in range(50):
        n = int(rng.integers(6, 16))
        box = float(rng.uniform(6.0, 11.0))
        cell = box * np.eye(3)
        if trial % 3 == 0:  # exercise triclinic cells too
            cell = cell + np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.7, -0.9, 0.0]])
        cfg = Configuration(
            positions=rng.uniform(0.0, box, size=(n, 3)),
            cell=cell,
            pbc=True,
            species=rng.integers(0, 2, size=n).astype(np.int32),
            symbols=("Al", "Ni"),
        )
        fast = desc.compute(cfg, implementation="numba")
        slow = desc.compute(cfg, implementation="numpy")
        worst_f = max(worst_f, float(np.max(np.abs(fast.features - slow.features))))
        worst_d = max(worst_d, float(np.max(np.abs(fast.derivatives - slow.derivatives))))
        assert np.array_equal(fast.pair_i, slow.pair_i)
        assert np.array_equal(fast.pair_j, slow.pair_j)
    assert worst_f < 1e-12, f"features differ by {worst_f:.3e}"
    assert worst_d < 1e-12, f"derivatives differ by {worst_d:.3e}"


# --------------------------------------------------------------------------
# 4. cutoff behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cosine", "tanh"])
def test_cutoff_function_and_derivative_vanish_at_rc(kind):
    rc = 4.3
    f, df = cutoff_function(np.array([rc, rc + 1e-9, rc + 1.0, 2 * rc]), rc, kind)
    assert np.all(f == 0.0)
    assert np.all(df == 0.0)

    # approaching Rc from inside (r increases towards Rc), both go smoothly to
    # zero -- fc like (Rc - r)^2 for the cosine form, (Rc - r)^3 for tanh.
    r = rc - np.geomspace(1e-1, 1e-6, 20)
    f, df = cutoff_function(r, rc, kind)
    assert f[-1] < 1e-11 and abs(df[-1]) < 1e-5
    assert np.all(np.diff(f) < 0.0)  # monotonically decreasing towards Rc

    # analytic vs finite-difference derivative inside the cutoff
    r = np.linspace(0.2, rc - 0.2, 40)
    h = 1e-6
    fp, _ = cutoff_function(r + h, rc, kind)
    fm, _ = cutoff_function(r - h, rc, kind)
    _, dfa = cutoff_function(r, rc, kind)
    assert np.max(np.abs((fp - fm) / (2 * h) - dfa)) < 1e-7


def test_isolated_atom_has_exactly_zero_features():
    """No neighbours, no features -- and no derivatives either."""
    for pbc in (False, True):
        cfg = Configuration(
            positions=np.array([[0.3, -0.2, 0.7]]),
            cell=60.0 * np.eye(3),
            pbc=pbc,
            species=np.zeros(1, dtype=np.int32),
            symbols=("Ar",),
        )
        desc = ACSF.default((0,), 5.0)
        out = desc.compute(cfg)
        assert np.all(out.features == 0.0)
        assert np.all(out.derivatives == 0.0)

    # a dimer at exactly the cutoff is also exactly zero
    rc = 5.0
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [rc, 0.0, 0.0]]),
        cell=60.0 * np.eye(3),
        pbc=True,
        species=np.zeros(2, dtype=np.int32),
        symbols=("Ar",),
    )
    out = ACSF.default((0,), rc).compute(cfg)
    assert np.max(np.abs(out.features)) == 0.0
    assert np.max(np.abs(out.derivatives)) == 0.0


def test_features_and_derivatives_are_continuous_across_the_cutoff():
    """Scan a neighbour through ``Rc`` and look for a step in G or in dG/dr.

    A bare truncation would put an O(1) jump in the feature at the crossing; a
    cutoff with ``fc(Rc) = 0`` but ``fc'(Rc) != 0`` would put one in the
    derivative.  The assertion is therefore not "the jump is small" (every
    consecutive-sample difference is small on a fine grid) but "the change at
    the crossing is no larger than the changes happening well inside the
    cutoff", which fails loudly for both broken variants.
    """
    rc = 3.0
    desc = ACSF(
        (0,),
        rc,
        [[1.0, 0.0], [1.5, 2.0]],
        [[0.05, 1.0, 1.0], [0.05, 4.0, -1.0]],
        angular_kind=["G4", "G5"],
    )
    # Atom 0 is the centre, atom 1 a fixed spectator (so angular terms exist),
    # atom 2 is scanned outwards along a generic direction through Rc.
    direction = np.array([0.6, 0.8, 0.0])
    radii = np.linspace(rc - 0.5, rc + 0.5, 501)
    step = float(radii[1] - radii[0])

    feats, derivs = [], []
    for r in radii:
        pos = np.array([[0.0, 0.0, 0.0], [1.9, 0.4, 0.0], list(r * direction)])
        cfg = Configuration(positions=pos, cell=60.0 * np.eye(3), pbc=True)
        out = desc.compute(cfg)
        feats.append(out.features[0].copy())
        derivs.append(dense_derivatives(out, 3)[0, 2].copy())  # dG_0 / dr_2
    feats = np.asarray(feats)
    derivs = np.asarray(derivs)

    df = np.abs(np.diff(feats, axis=0)).max(axis=1)
    dd = np.abs(np.diff(derivs, axis=0)).reshape(len(radii) - 1, -1).max(axis=1)

    cross = int(np.searchsorted(radii, rc))
    inside = radii[:-1] < rc - 0.05

    assert df[cross - 1] <= 2.0 * df[inside].max()
    assert dd[cross - 1] <= 2.0 * dd[inside].max()
    # Lipschitz sanity: the step in the feature is what the analytic derivative
    # says it should be, everywhere including at the crossing.
    assert df.max() <= 3.0 * step * np.abs(derivs).max()


# --------------------------------------------------------------------------
# 5. construction, validation and bookkeeping
# --------------------------------------------------------------------------


def test_default_sizes_and_labels():
    one = ACSF.default((0,), 6.0)
    assert one.n_features == 36 == len(one.feature_labels())
    two = ACSF.default((0, 1), 6.0)
    assert two.n_features == 44 == len(two.feature_labels())
    assert two.n_features == 2 * two.n_radial + 3 * two.n_angular


def test_default_transfers_between_cutoffs():
    """Parameters scale with Rc, so the same call works for LJ and for SW."""
    for rc in (3.77, 6.0, 8.5):
        desc = ACSF.default((0,), rc)
        assert np.all(desc.radial[:, 1] < rc)
        assert np.all(desc.radial[:, 1] > 0.0)
        assert np.all(desc.angular[:, 1] >= 1.0)
        assert set(np.unique(desc.angular[:, 2])) == {-1.0, 1.0}


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        ACSF((0,), -1.0, [[1.0, 0.0]])
    with pytest.raises(ValueError):
        ACSF((0,), 4.0, [[1.0, 0.0, 3.0]])
    with pytest.raises(ValueError):
        ACSF((0,), 4.0, [[1.0, 0.0]], [[0.1, 0.5, 1.0]])  # zeta < 1
    with pytest.raises(ValueError):
        ACSF((0,), 4.0, [[1.0, 0.0]], [[0.1, 2.0, 0.5]])  # lambda not +-1
    with pytest.raises(ValueError):
        ACSF((0,), 4.0, [[1.0, 0.0]], [[0.1, 2.0, 1.0]], angular_kind="G6")
    with pytest.raises(ValueError):
        ACSF((0, 0), 4.0, [[1.0, 0.0]])  # duplicate species
    with pytest.raises(ValueError):
        ACSF((0,), 4.0, [[1.0, 0.0]], cutoff_function="gaussian")


def test_unknown_species_is_rejected_not_silently_featurised():
    desc = small_acsf((0,), 4.0)
    cfg = two_species_config()
    with pytest.raises(ValueError, match="species"):
        desc.compute(cfg)


def test_compute_does_not_mutate_the_configuration():
    cfg = rattled_fcc()
    before = cfg.positions.copy()
    small_acsf((0,), 4.0).compute(cfg)
    assert np.array_equal(cfg.positions, before)


def test_bad_implementation_name_raises():
    with pytest.raises(ValueError):
        small_acsf((0,), 4.0).compute(rattled_fcc(), implementation="cython")


# --------------------------------------------------------------------------
# 6. cost
# --------------------------------------------------------------------------


def test_timing_256_atoms(capsys):
    """Report the cost of features + derivatives for a 256-atom configuration."""
    cfg = rattle(fcc(4.05, "Al", (4, 4, 4)), 0.05, seed=41)
    assert cfg.n_atoms == 256
    desc = ACSF.default((0,), 6.0)

    desc.compute(cfg)  # JIT warm-up, not part of the measurement
    t0 = time.perf_counter()
    out = desc.compute(cfg, derivatives=True)
    t_both = time.perf_counter() - t0
    t0 = time.perf_counter()
    desc.compute(cfg, derivatives=False)
    t_feat = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\n[acsf timing] N=256, D={desc.n_features}, Rc={desc.cutoff} A, "
            f"rows={out.derivatives.shape[0]}: "
            f"features+derivatives {t_both * 1e3:.1f} ms, features only "
            f"{t_feat * 1e3:.1f} ms"
        )
    # Generous bound: this is a regression guard against an accidental O(N^2)
    # rewrite, not a benchmark.
    assert t_both < 5.0
