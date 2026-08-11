"""Tests for :mod:`atomlab.neighbors`.

Three things are checked hard, because everything downstream inherits them:

1. the jitted cell-linked-list and the pure-NumPy reference agree *exactly*,
   across densities, cutoffs, cell shapes and periodicities;
2. the awkward periodic cases are right -- a cell smaller than ``2*cutoff``
   where a neighbour is seen through several images at once, and an atom
   seeing itself through its own image;
3. the pair displacement points from ``i`` to ``j``.

The simple-cubic coordination numbers give an absolute, hand-countable answer
for case (2): the number of lattice points at distance^2 = m a^2 is 6, 12, 8,
6, 24, 24, 0, 12, 30 for m = 1..9.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.cell import reduce_cell, wrap_positions
from atomlab.neighbors import (
    NeighborList,
    VerletList,
    build_neighbor_list,
    build_neighbor_list_naive,
    pair_vectors,
)
from atomlab.types import Configuration

ORTHO = np.diag([8.0, 9.0, 10.0])
TRICLINIC = np.array([[6.0, 0.0, 0.0], [2.0, 5.5, 0.0], [1.5, -2.0, 7.0]])
SKEWED = np.array([[5.0, 0.0, 0.0], [4.4, 2.2, 0.0], [4.1, 1.7, 3.4]])

#: Number of simple-cubic lattice points at squared distance ``m * a^2``,
#: i.e. the number of ways m is a sum of three signed squares.  Hand-countable:
#: m=1 -> (+-1,0,0) and permutations = 6; m=2 -> (+-1,+-1,0) = 3*4 = 12;
#: m=3 -> (+-1,+-1,+-1) = 8; m=4 -> (+-2,0,0) = 6; m=5 -> (+-2,+-1,0) = 24;
#: m=6 -> (+-2,+-1,+-1) = 24; m=7 -> none; m=8 -> (+-2,+-2,0) = 12;
#: m=9 -> (+-3,0,0) plus (+-2,+-2,+-1) = 6 + 24 = 30.
SC_SHELLS = {1: 6, 2: 12, 3: 8, 4: 6, 5: 24, 6: 24, 7: 0, 8: 12, 9: 30}


def _configuration(cell, n, rng, pbc=True):
    return Configuration(positions=rng.uniform(size=(n, 3)) @ cell, cell=cell, pbc=pbc)


def _identical(a: NeighborList, b: NeighborList) -> bool:
    return (
        np.array_equal(a.i, b.i)
        and np.array_equal(a.j, b.j)
        and np.array_equal(a.shift, b.shift)
    )


# --------------------------------------------------------------------------
# fast vs reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cell_name", ["ortho", "triclinic", "skewed"])
@pytest.mark.parametrize("n_atoms", [1, 8, 60])
@pytest.mark.parametrize("cutoff", [1.5, 3.0, 6.5])
def test_cell_list_matches_naive(cell_name, n_atoms, cutoff):
    """The jitted kernel and the O(N^2) reference must agree array-for-array.

    The 6.5 A cutoff exceeds half the width of every cell here, so this also
    covers the multiple-image regime at several densities.
    """
    cell = {"ortho": ORTHO, "triclinic": TRICLINIC, "skewed": SKEWED}[cell_name]
    rng = np.random.default_rng(hash((cell_name, n_atoms)) % 2**32)
    cfg = _configuration(cell, n_atoms, rng)
    fast = build_neighbor_list(cfg, cutoff)
    ref = build_neighbor_list_naive(cfg, cutoff)
    assert _identical(fast, ref), f"{fast.n_pairs} vs {ref.n_pairs} pairs"
    # Canary against the parametrisation quietly drifting out of the hard
    # regime.  Once the cutoff exceeds the shortest lattice vector, a neighbour
    # and its next image are both comfortably inside range, so multi-image
    # pairs must actually appear -- which is the case this comparison exists to
    # police.  (The converse is not asserted: below that threshold multi-image
    # pairs are possible but not guaranteed.)
    shortest = float(np.min(np.linalg.norm(reduce_cell(cell, True)[0], axis=1)))
    if fast.n_pairs > 0 and cutoff > shortest:
        _, counts = np.unique(np.stack([fast.i, fast.j], axis=1), axis=0, return_counts=True)
        assert counts.max() > 1, "expected multi-image pairs at this cutoff"


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False), (True, False, False)])
def test_cell_list_matches_naive_for_mixed_pbc(pbc, half):
    rng = np.random.default_rng(42)
    cfg = _configuration(TRICLINIC, 40, rng, pbc=pbc)
    fast = build_neighbor_list(cfg, 4.0, half=half)
    ref = build_neighbor_list_naive(cfg, 4.0, half=half)
    assert _identical(fast, ref)
    # No images may be generated along an open direction.
    open_dirs = [k for k in range(3) if not pbc[k]]
    assert np.all(fast.shift[:, open_dirs] == 0)


def test_slab_has_no_pairs_through_the_open_direction():
    """A slab with vacuum: nothing may interact across the vacuum gap."""
    rng = np.random.default_rng(3)
    cell = np.diag([7.0, 7.0, 30.0])
    pos = rng.uniform(size=(40, 3)) * np.array([7.0, 7.0, 6.0])  # 6 A thick slab
    cfg = Configuration(positions=pos, cell=cell, pbc=[True, True, False])
    nl = build_neighbor_list(cfg, 5.0)
    assert np.all(nl.shift[:, 2] == 0)
    d, r = pair_vectors(cfg, nl)
    assert np.max(np.abs(d[:, 2])) <= 6.0
    assert _identical(nl, build_neighbor_list_naive(cfg, 5.0))


def test_open_cluster_matches_brute_force_distance_matrix():
    rng = np.random.default_rng(4)
    pos = rng.uniform(size=(45, 3)) * 6.0
    cfg = Configuration(positions=pos, cell=np.eye(3) * 20.0, pbc=False)
    nl = build_neighbor_list(cfg, 2.0)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert nl.n_pairs == int((d <= 2.0).sum())
    assert np.all(nl.shift == 0)
    assert _identical(nl, build_neighbor_list_naive(cfg, 2.0))


def test_works_for_unwrapped_positions_far_outside_the_cell():
    """MD carries unwrapped coordinates; shifts must be reported against those."""
    rng = np.random.default_rng(5)
    inside = rng.uniform(size=(30, 3)) @ TRICLINIC
    offset = np.array([3.0, -5.0, 2.0]) @ TRICLINIC * 40.0
    cfg_in = Configuration(positions=inside, cell=TRICLINIC, pbc=True)
    cfg_out = Configuration(positions=inside + offset, cell=TRICLINIC, pbc=True)
    a = build_neighbor_list(cfg_in, 4.0)
    b = build_neighbor_list(cfg_out, 4.0)
    # A rigid translation of the whole system leaves the pair geometry alone.
    assert np.array_equal(a.i, b.i) and np.array_equal(a.j, b.j)
    assert np.allclose(np.sort(pair_vectors(cfg_in, a)[1]), np.sort(pair_vectors(cfg_out, b)[1]))
    assert _identical(b, build_neighbor_list_naive(cfg_out, 4.0))


def test_individually_shifted_atoms_give_the_same_geometry():
    """Which image an atom is stored in must not change the physics."""
    rng = np.random.default_rng(6)
    pos = rng.uniform(size=(25, 3)) @ TRICLINIC
    jumps = rng.integers(-2, 3, size=(25, 3)).astype(float)
    cfg_a = Configuration(positions=pos, cell=TRICLINIC, pbc=True)
    cfg_b = Configuration(positions=pos + jumps @ TRICLINIC, cell=TRICLINIC, pbc=True)
    nl_a = build_neighbor_list(cfg_a, 4.5)
    nl_b = build_neighbor_list(cfg_b, 4.5)
    assert np.array_equal(nl_a.i, nl_b.i) and np.array_equal(nl_a.j, nl_b.j)
    assert np.allclose(pair_vectors(cfg_a, nl_a)[0], pair_vectors(cfg_b, nl_b)[0])


# --------------------------------------------------------------------------
# small cells: multiple images and self-images
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", sorted(SC_SHELLS))
def test_simple_cubic_coordination_shells(m):
    """One atom in a cubic cell: every neighbour is one of its own images."""
    a = 2.3
    cfg = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * a, pbc=True)
    cutoff = np.sqrt(m) * a * 1.0001  # just outside shell m, well inside m+1
    expected = sum(c for k, c in SC_SHELLS.items() if k <= m)
    nl = build_neighbor_list(cfg, cutoff)
    assert nl.n_pairs == expected
    assert np.all(nl.i == 0) and np.all(nl.j == 0)  # case (b): i == j, shift != 0
    assert np.all(np.any(nl.shift != 0, axis=1))  # never the trivial self pair
    assert _identical(nl, build_neighbor_list_naive(cfg, cutoff))
    # The shells themselves, not just the cumulative count.
    _, r = pair_vectors(cfg, nl)
    counts = np.bincount(np.round((r / a) ** 2).astype(int), minlength=m + 1)
    for k, expect in SC_SHELLS.items():
        if k <= m:
            assert counts[k] == expect, f"shell m={k}"


def test_simple_cubic_supercell_has_the_same_coordination_per_atom():
    """The 2x2x2 supercell must reproduce the primitive answer atom by atom."""
    a = 2.3
    prim = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * a, pbc=True)
    super_cfg = prim.repeated((2, 2, 2))
    cutoff = np.sqrt(6.0) * a * 1.0001  # 6+12+8+6+24+24 = 80 neighbours
    nl = build_neighbor_list(super_cfg, cutoff)
    assert np.all(nl.counts() == 80)
    assert _identical(nl, build_neighbor_list_naive(super_cfg, cutoff))


def test_the_same_neighbour_is_seen_through_several_distinct_images():
    """Case (a): one (i, j) pair, several shifts, all of them real."""
    a = 2.3
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]), cell=np.eye(3) * a, pbc=True
    )
    nl = build_neighbor_list(cfg, 4.0)
    mask = (nl.i == 0) & (nl.j == 1)
    shifts = nl.shift[mask]
    assert len(shifts) > 1
    assert len({tuple(s) for s in shifts}) == len(shifts)  # each image listed once
    d, r = pair_vectors(cfg, nl)
    assert np.all(r <= 4.0 + 1e-12)
    # Distinct shifts must give genuinely distinct displacement vectors.
    assert len({tuple(np.round(v, 9)) for v in d[mask]}) == len(shifts)


def test_a_single_atom_in_a_large_cell_has_no_neighbours():
    cfg = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * 20.0, pbc=True)
    assert build_neighbor_list(cfg, 5.0).n_pairs == 0


# --------------------------------------------------------------------------
# half lists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cell", [ORTHO, TRICLINIC])
@pytest.mark.parametrize("cutoff", [2.5, 5.5])
def test_half_list_doubled_equals_full_list(cell, cutoff):
    rng = np.random.default_rng(7)
    cfg = _configuration(cell, 30, rng)
    full = build_neighbor_list(cfg, cutoff)
    half = build_neighbor_list(cfg, cutoff, half=True)
    assert 2 * half.n_pairs == full.n_pairs
    assert _identical(half.to_full(), full)


def test_half_list_for_self_images_keeps_exactly_one_of_each_pair():
    """The i == j tie-break: shifts s and -s describe the same interaction."""
    a = 2.3
    cfg = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * a, pbc=True)
    full = build_neighbor_list(cfg, 3.3)  # shells m = 1, 2 -> 18 pairs
    half = build_neighbor_list(cfg, 3.3, half=True)
    assert full.n_pairs == 18 and half.n_pairs == 9
    for shift in half.shift:
        first = shift[np.flatnonzero(shift)[0]]
        assert first > 0, "half list must keep the lexicographically positive image"
    seen = {tuple(s) for s in half.shift}
    assert not any(tuple(-np.asarray(s)) in seen for s in seen)
    assert _identical(half.to_full(), full)


def test_full_list_is_symmetric():
    """Every (i, j, s) must have its partner (j, i, -s); half lists rely on it."""
    rng = np.random.default_rng(8)
    cfg = _configuration(TRICLINIC, 35, rng)
    nl = build_neighbor_list(cfg, 5.0)
    forward = nl.as_set()
    backward = {
        (int(b), int(a), -int(s[0]), -int(s[1]), -int(s[2]))
        for a, b, s in zip(nl.i, nl.j, nl.shift)
    }
    assert forward == backward


def test_half_list_energy_sum_is_half_the_full_one():
    """The practical consequence: sum over a half list, double it, get the full."""
    rng = np.random.default_rng(9)
    cfg = _configuration(ORTHO, 40, rng)
    full = build_neighbor_list(cfg, 4.0)
    half = build_neighbor_list(cfg, 4.0, half=True)
    _, r_full = pair_vectors(cfg, full)
    _, r_half = pair_vectors(cfg, half)
    u = lambda r: np.sum(r**-6)  # noqa: E731 - a stand-in pair energy
    assert 2.0 * u(r_half) == pytest.approx(u(r_full), rel=1e-12)


# --------------------------------------------------------------------------
# pair geometry
# --------------------------------------------------------------------------


def test_pair_vectors_point_from_i_to_j():
    rng = np.random.default_rng(10)
    cfg = _configuration(TRICLINIC, 20, rng)
    nl = build_neighbor_list(cfg, 4.0)
    d, r = pair_vectors(cfg, nl)
    expected = (
        cfg.positions[nl.j] + nl.shift.astype(float) @ cfg.cell - cfg.positions[nl.i]
    )
    assert np.allclose(d, expected)
    assert np.allclose(r, np.linalg.norm(expected, axis=1))
    assert np.all(r <= nl.cutoff + 1e-12)
    assert np.all(r > 0.0)


def test_pair_vectors_are_translation_invariant():
    rng = np.random.default_rng(11)
    cfg = _configuration(ORTHO, 25, rng)
    nl = build_neighbor_list(cfg, 3.5)
    moved = cfg.translated([1.234, -8.5, 0.77])
    d0, _ = pair_vectors(cfg, nl)
    d1, _ = pair_vectors(moved, nl)
    assert np.allclose(d0, d1)


def test_pair_vectors_rejects_a_mismatched_configuration():
    rng = np.random.default_rng(12)
    cfg = _configuration(ORTHO, 20, rng)
    nl = build_neighbor_list(cfg, 3.0)
    other = _configuration(ORTHO, 19, rng)
    with pytest.raises(ValueError, match="built for"):
        pair_vectors(other, nl)


def test_every_pair_within_the_cutoff_is_listed():
    """Completeness, checked against an independent all-images enumeration."""
    rng = np.random.default_rng(13)
    cfg = _configuration(SKEWED, 20, rng)
    cutoff = 5.0
    pos = wrap_positions(cfg.positions, cfg.cell, cfg.pbc)
    found = set()
    for nx in range(-3, 4):
        for ny in range(-3, 4):
            for nz in range(-3, 4):
                image = np.array([nx, ny, nz], dtype=float)
                d = pos[None, :, :] + image @ cfg.cell - pos[:, None, :]
                r = np.linalg.norm(d, axis=-1)
                for a, b in zip(*np.nonzero(r <= cutoff)):
                    if a == b and nx == ny == nz == 0:
                        continue
                    found.add((int(a), int(b), nx, ny, nz))
    assert build_neighbor_list(cfg, cutoff).as_set() == found


# --------------------------------------------------------------------------
# cross-check against ASE (test-only dependency)
# --------------------------------------------------------------------------


# Cutoffs deliberately avoid |a1| = 5.0 A of SKEWED: this package includes
# pairs at exactly r == cutoff and ASE excludes them, and an exact tie would be
# comparing conventions rather than correctness.
@pytest.mark.parametrize("cell", [ORTHO, TRICLINIC, SKEWED])
@pytest.mark.parametrize("cutoff", [2.5, 4.7])
def test_matches_ase_primitive_neighbor_list(cell, cutoff):
    ase_nl = pytest.importorskip("ase.neighborlist")
    rng = np.random.default_rng(14)
    cfg = _configuration(cell, 20, rng)
    i, j, s = ase_nl.primitive_neighbor_list(
        "ijS", [True, True, True], cfg.cell, cfg.positions, cutoff
    )
    reference = {(int(a), int(b), *map(int, sh)) for a, b, sh in zip(i, j, s)}
    assert build_neighbor_list(cfg, cutoff).as_set() == reference


def test_matches_ase_for_a_one_atom_cell_with_self_images():
    ase_nl = pytest.importorskip("ase.neighborlist")
    a = 2.3
    cutoff = float(np.sqrt(5.0) * a * 1.0001)  # just past the m = 5 shell
    cfg = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * a, pbc=True)
    i, j, s = ase_nl.primitive_neighbor_list(
        "ijS", [True, True, True], cfg.cell, cfg.positions, cutoff
    )
    reference = {(int(x), int(y), *map(int, sh)) for x, y, sh in zip(i, j, s)}
    assert build_neighbor_list(cfg, cutoff).as_set() == reference
    assert len(reference) == 56  # shells 1..5: 6 + 12 + 8 + 6 + 24


# --------------------------------------------------------------------------
# skin and Verlet bookkeeping
# --------------------------------------------------------------------------


def test_skin_widens_the_list_and_is_recorded():
    rng = np.random.default_rng(15)
    cfg = _configuration(ORTHO, 40, rng)
    plain = build_neighbor_list(cfg, 3.0)
    padded = build_neighbor_list(cfg, 3.0, skin=1.0)
    assert padded.cutoff == pytest.approx(4.0)
    assert padded.skin == pytest.approx(1.0)
    assert padded.interaction_cutoff == pytest.approx(3.0)
    assert padded.n_pairs > plain.n_pairs
    # Everything in the plain list must be in the padded one.
    assert plain.as_set() <= padded.as_set()
    _, r = pair_vectors(cfg, padded)
    assert plain.n_pairs == int((r <= 3.0).sum())


def test_verlet_rebuild_criterion_is_half_the_skin():
    rng = np.random.default_rng(16)
    cfg = _configuration(ORTHO, 30, rng)
    verlet = VerletList(cutoff=3.0, skin=0.6)
    verlet.build(cfg)
    assert verlet.n_builds == 1
    assert not verlet.needs_rebuild(cfg.positions)

    small = cfg.positions.copy()
    small[3, 0] += 0.29
    assert not verlet.needs_rebuild(small)
    big = cfg.positions.copy()
    big[3, 0] += 0.31
    assert verlet.needs_rebuild(big)
    assert verlet.max_displacement(big) == pytest.approx(0.31)


def test_verlet_update_rebuilds_only_when_needed():
    rng = np.random.default_rng(17)
    cfg = _configuration(ORTHO, 30, rng)
    verlet = VerletList(cutoff=3.0, skin=0.8)
    verlet.update(cfg)
    assert verlet.n_builds == 1
    nudged = cfg.copy()
    nudged.positions += 0.05
    verlet.update(nudged)
    assert verlet.n_builds == 1  # within the skin: reuse
    jumped = cfg.copy()
    jumped.positions += 1.0
    verlet.update(jumped)
    assert verlet.n_builds == 2


def test_verlet_rebuilds_when_the_cell_changes():
    rng = np.random.default_rng(18)
    cfg = _configuration(ORTHO, 20, rng)
    verlet = VerletList(cutoff=3.0, skin=0.5)
    verlet.build(cfg)
    squeezed = cfg.copy()
    squeezed.cell = cfg.cell * 0.99
    assert verlet.needs_rebuild(cfg.positions, squeezed.cell)


def test_verlet_list_remains_complete_while_atoms_drift_within_the_skin():
    """The point of the skin: the stale list still contains every real pair."""
    rng = np.random.default_rng(19)
    cfg = _configuration(ORTHO, 50, rng)
    cutoff, skin = 3.0, 0.6
    verlet = VerletList(cutoff=cutoff, skin=skin)
    verlet.build(cfg)
    for _ in range(5):
        moved = cfg.copy()
        step = rng.normal(size=cfg.positions.shape)
        step *= (0.49 * skin) / np.max(np.linalg.norm(step, axis=1))
        moved.positions = cfg.positions + step
        assert not verlet.needs_rebuild(moved.positions)
        stale = verlet.update(moved)
        assert verlet.n_builds == 1
        _, r_stale = pair_vectors(moved, stale)
        inside = np.flatnonzero(r_stale <= cutoff)
        stale_pairs = {
            (int(stale.i[p]), int(stale.j[p]), *map(int, stale.shift[p])) for p in inside
        }
        exact = build_neighbor_list(moved, cutoff).as_set()
        assert exact <= stale_pairs


# --------------------------------------------------------------------------
# input validation and edge cases
# --------------------------------------------------------------------------


def test_rejects_nonsensical_cutoffs():
    cfg = Configuration(positions=np.zeros((2, 3)) + 0.1, cell=np.eye(3) * 10.0, pbc=True)
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            build_neighbor_list(cfg, bad)
    with pytest.raises(ValueError):
        build_neighbor_list(cfg, 2.0, skin=-0.1)


def test_open_system_with_no_cell_at_all():
    """A gas-phase cluster carries a zero cell; nothing may divide by it."""
    rng = np.random.default_rng(22)
    cfg = Configuration(positions=rng.uniform(size=(20, 3)) * 5.0, cell=None, pbc=False)
    nl = build_neighbor_list(cfg, 2.0)
    assert _identical(nl, build_neighbor_list_naive(cfg, 2.0))
    assert np.all(nl.shift == 0)
    assert np.all(pair_vectors(cfg, nl)[1] <= 2.0)


def test_small_cells_are_supported_not_rejected():
    """The cutoff-vs-cell check belongs to potentials, not to the list itself.

    ``atomlab.cell.check_minimum_image`` is what raises; a neighbour list is
    exact for arbitrarily small cells because it enumerates images explicitly.
    """
    cfg = Configuration(positions=np.zeros((1, 3)), cell=np.eye(3) * 1.0, pbc=True)
    nl = build_neighbor_list(cfg, 6.0)  # 12x the minimum width: must not raise
    assert nl.n_pairs == 924  # lattice points with 0 < |n| <= 6 in a unit sc lattice
    assert _identical(nl, build_neighbor_list_naive(cfg, 6.0))


def test_empty_configuration_gives_an_empty_list():
    cfg = Configuration(positions=np.zeros((0, 3)), cell=np.eye(3) * 5.0, pbc=True)
    nl = build_neighbor_list(cfg, 2.0)
    assert nl.n_pairs == 0 and nl.n_atoms == 0
    assert pair_vectors(cfg, nl)[0].shape == (0, 3)


def test_dtypes_are_as_advertised():
    rng = np.random.default_rng(20)
    cfg = _configuration(ORTHO, 10, rng)
    nl = build_neighbor_list(cfg, 3.0)
    assert nl.i.dtype == np.int32 and nl.j.dtype == np.int32
    assert nl.shift.dtype == np.int32 and nl.shift.shape == (nl.n_pairs, 3)
    assert isinstance(nl.n_atoms, int) and isinstance(nl.half, bool)


def test_counts_sum_to_the_pair_count():
    rng = np.random.default_rng(21)
    cfg = _configuration(TRICLINIC, 30, rng)
    nl = build_neighbor_list(cfg, 4.0)
    assert nl.counts().sum() == nl.n_pairs
    assert nl.counts().shape == (cfg.n_atoms,)
