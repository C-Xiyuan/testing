"""Tests for :mod:`atomlab.cell`.

The emphasis is on the two things that break silently rather than loudly:
the minimum-image convention in skewed triclinic cells, and the definition of
"how big is this cell" that guards every periodic potential.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from atomlab.cell import (
    apply_strain,
    cartesian_to_fractional,
    cell_from_parameters,
    cell_to_parameters,
    cell_widths,
    check_minimum_image,
    fractional_to_cartesian,
    full_to_voigt,
    is_orthorhombic,
    min_cell_width,
    minimum_image,
    minimum_image_naive,
    minimum_image_shift,
    reciprocal_cell,
    reduce_cell,
    voigt_to_full,
    volume,
    wrap_positions,
)
from atomlab.types import Configuration

# A triclinic cell whose widths are exactly computable by hand:
#   V = 120, |a2 x a3| = sqrt(1000), |a3 x a1| = sqrt(720), |a1 x a2| = 20
# giving widths 12/sqrt(10), 2*sqrt(5) and 6.  Note the shortest lattice vector
# is |a1| = 4, which is *larger* than the smallest width -- the whole point.
HAND_CELL = np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [2.0, 3.0, 6.0]])

# Strongly sheared cells, where fractional rounding is not the minimum image.
SKEWED = np.array([[5.0, 0.0, 0.0], [4.4, 2.0, 0.0], [4.1, 1.7, 3.0]])


def _random_cells(n, rng, *, shear=6.0):
    """Random lower-triangular cells with adjustable shear, all non-degenerate."""
    out = []
    while len(out) < n:
        c = np.diag(rng.uniform(3.0, 8.0, size=3))
        c[1, 0] = rng.uniform(-shear, shear)
        c[2, 0] = rng.uniform(-shear, shear)
        c[2, 1] = rng.uniform(-shear, shear)
        if abs(np.linalg.det(c)) > 5.0:
            out.append(c)
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_volume_matches_determinant_and_triple_product():
    v = volume(HAND_CELL)
    assert v == pytest.approx(120.0)
    triple = abs(np.dot(HAND_CELL[0], np.cross(HAND_CELL[1], HAND_CELL[2])))
    assert v == pytest.approx(triple)


def test_volume_is_positive_for_left_handed_cell():
    flipped = HAND_CELL[[1, 0, 2]]
    assert np.linalg.det(flipped) < 0
    assert volume(flipped) == pytest.approx(120.0)


def test_reciprocal_cell_uses_the_two_pi_convention():
    b = reciprocal_cell(HAND_CELL)
    assert np.allclose(HAND_CELL @ b.T, 2.0 * math.pi * np.eye(3))
    # 1/A for a cell in A.
    assert b.shape == (3, 3)


def test_reciprocal_cell_rejects_degenerate_cell():
    with pytest.raises(ValueError):
        reciprocal_cell(np.zeros((3, 3)))


def test_cell_widths_hand_computed():
    w = cell_widths(HAND_CELL, True)
    assert w[0] == pytest.approx(12.0 / math.sqrt(10.0))
    assert w[1] == pytest.approx(2.0 * math.sqrt(5.0))
    assert w[2] == pytest.approx(6.0)


def test_cell_widths_match_volume_over_cross_product():
    """The projection formula must agree with V / |a_j x a_k| for full pbc."""
    rng = np.random.default_rng(0)
    for cell in _random_cells(8, rng):
        w = cell_widths(cell, True)
        v = volume(cell)
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            expected = v / np.linalg.norm(np.cross(cell[j], cell[k]))
            assert w[i] == pytest.approx(expected)


def test_cell_widths_equal_inverse_reciprocal_length():
    """w_i = 1 / |g_i| with g the reciprocal vectors *without* the 2 pi."""
    g = reciprocal_cell(HAND_CELL) / (2.0 * math.pi)
    assert np.allclose(cell_widths(HAND_CELL, True), 1.0 / np.linalg.norm(g, axis=1))


def test_min_cell_width_is_smaller_than_the_shortest_lattice_vector():
    """The trap this function exists to avoid."""
    shortest = float(np.min(np.linalg.norm(HAND_CELL, axis=1)))
    assert shortest == pytest.approx(4.0)
    assert min_cell_width(HAND_CELL) == pytest.approx(12.0 / math.sqrt(10.0))
    assert min_cell_width(HAND_CELL) < shortest


def test_min_cell_width_orthorhombic_is_the_shortest_edge():
    cell = np.diag([7.0, 3.0, 11.0])
    assert min_cell_width(cell) == pytest.approx(3.0)


def test_min_cell_width_partial_pbc_ignores_open_directions():
    # Periodic in a1, a2 only: widths are |a1 x a2| / |a2| = 20/5 = 4 and 20/4 = 5.
    w = cell_widths(HAND_CELL, [True, True, False])
    assert w[0] == pytest.approx(4.0)
    assert w[1] == pytest.approx(5.0)
    assert not np.isfinite(w[2])
    assert min_cell_width(HAND_CELL, [True, True, False]) == pytest.approx(4.0)


def test_min_cell_width_single_periodic_direction_is_the_vector_length():
    assert min_cell_width(HAND_CELL, [False, False, True]) == pytest.approx(7.0)
    assert not np.isfinite(min_cell_width(HAND_CELL, False))


def test_check_minimum_image_raises_when_cutoff_too_large():
    width = min_cell_width(HAND_CELL)
    check_minimum_image(HAND_CELL, True, 0.4 * width)
    with pytest.raises(ValueError, match="minimum image"):
        check_minimum_image(HAND_CELL, True, 0.5 * width)
    # A cutoff that passes the (wrong) shortest-lattice-vector test but fails
    # the correct one must still raise.
    assert 2.0 * 1.95 < 4.0 and 2.0 * 1.95 > width
    with pytest.raises(ValueError):
        check_minimum_image(HAND_CELL, True, 1.95)


def test_is_orthorhombic_is_rotation_invariant():
    cell = np.diag([4.0, 5.0, 6.0])
    theta = 0.7
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert is_orthorhombic(cell)
    assert is_orthorhombic(cell @ rot.T)
    assert not is_orthorhombic(HAND_CELL)
    # ...but the sheared direction is not periodic, so the sublattice is.
    assert is_orthorhombic(HAND_CELL, [True, True, False])


# --------------------------------------------------------------------------
# coordinate transforms
# --------------------------------------------------------------------------


def test_fractional_cartesian_round_trip():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=(37, 3)) * 10.0
    s = cartesian_to_fractional(pos, HAND_CELL)
    assert np.allclose(fractional_to_cartesian(s, HAND_CELL), pos)
    # r = s @ cell is the stated convention.
    assert np.allclose(s @ HAND_CELL, pos)


def test_fractional_preserves_shape():
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(4, 5, 3))
    assert cartesian_to_fractional(pos, HAND_CELL).shape == (4, 5, 3)


def test_wrap_positions_lands_inside_the_cell():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(50, 3)) * 30.0
    wrapped = wrap_positions(pos, HAND_CELL, True)
    s = cartesian_to_fractional(wrapped, HAND_CELL)
    assert np.all(s > -1e-12) and np.all(s < 1.0 + 1e-12)


def test_wrap_positions_shift_is_exact():
    rng = np.random.default_rng(4)
    pos = rng.normal(size=(50, 3)) * 30.0
    wrapped, shift = wrap_positions(pos, HAND_CELL, True, return_shift=True)
    assert shift.dtype == np.int32
    # Bitwise, not merely close: the neighbour list relies on wrapped positions
    # and reported shifts describing exactly the same points.
    assert np.array_equal(wrapped, pos + shift.astype(np.float64) @ HAND_CELL)


def test_wrap_positions_is_idempotent_with_eps():
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(20, 3)) * 12.0
    once = wrap_positions(pos, HAND_CELL, True, eps=1e-12)
    twice = wrap_positions(once, HAND_CELL, True, eps=1e-12)
    assert np.allclose(once, twice)


def test_wrap_positions_leaves_open_directions_untouched():
    rng = np.random.default_rng(6)
    pos = rng.normal(size=(20, 3)) * 20.0
    cell = np.diag([5.0, 5.0, 5.0])
    wrapped = wrap_positions(pos, cell, [True, True, False])
    assert np.allclose(wrapped[:, 2], pos[:, 2])
    assert np.all(np.abs(wrapped[:, :2]) < 5.0 + 1e-9)


def test_wrap_matches_configuration_wrapped():
    rng = np.random.default_rng(7)
    pos = rng.normal(size=(15, 3)) * 15.0
    cfg = Configuration(positions=pos, cell=HAND_CELL, pbc=True)
    assert np.allclose(cfg.wrapped().positions, wrap_positions(pos, HAND_CELL, True, eps=1e-12))


# --------------------------------------------------------------------------
# minimum image
# --------------------------------------------------------------------------


def _brute_force_min_image(disp, cell, pbc, n=4):
    """Exhaustive search over a large image box: the ground truth."""
    pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    ranges = [range(-n, n + 1) if pbc[k] else range(1) for k in range(3)]
    best = np.full(disp.shape[0], np.inf)
    for image in itertools.product(*ranges):
        cand = disp + np.array(image, dtype=float) @ cell
        best = np.minimum(best, np.linalg.norm(cand, axis=1))
    return best


def test_minimum_image_orthorhombic_matches_explicit_rounding():
    cell = np.diag([4.0, 5.0, 6.0])
    rng = np.random.default_rng(8)
    d = rng.normal(size=(200, 3)) * 15.0
    got = minimum_image(d, cell, True)
    expected = d - np.round(d / np.diag(cell)) * np.diag(cell)
    assert np.allclose(got, expected)


def test_minimum_image_safe_matches_brute_force_on_vicious_cells():
    """The safe path must be exact for arbitrary shear, not merely better."""
    rng = np.random.default_rng(9)
    worst = 0.0
    for cell in _random_cells(25, rng, shear=12.0):
        d = rng.normal(size=(60, 3)) * 12.0
        got = np.linalg.norm(minimum_image(d, cell, True, method="safe"), axis=1)
        truth = _brute_force_min_image(d, cell, True, n=4)
        worst = max(worst, float(np.max(got - truth)))
    assert worst < 1e-9, f"safe minimum image is not minimal, excess {worst}"


def test_minimum_image_naive_is_wrong_for_skewed_cells():
    """Documented failure mode: rounding is not the Wigner-Seitz reduction."""
    rng = np.random.default_rng(10)
    d = rng.normal(size=(400, 3)) * 8.0
    safe_vectors = minimum_image(d, SKEWED, True, method="safe")
    naive = np.linalg.norm(minimum_image_naive(d, SKEWED, True), axis=1)
    safe = np.linalg.norm(safe_vectors, axis=1)
    assert np.all(safe <= naive + 1e-12)
    assert np.max(naive - safe) > 0.5, "expected the naive recipe to fail visibly here"
    # And "auto" must not pick the broken one for this cell.
    assert np.allclose(minimum_image(d, SKEWED, True), safe_vectors)


def test_minimum_image_auto_uses_the_fast_path_when_it_is_exact():
    cell = np.diag([4.0, 5.0, 6.0])
    rng = np.random.default_rng(11)
    d = rng.normal(size=(100, 3)) * 20.0
    assert np.allclose(
        minimum_image(d, cell, True, method="naive"),
        minimum_image(d, cell, True, method="safe"),
    )


def test_minimum_image_shift_reconstructs_the_displacement():
    rng = np.random.default_rng(12)
    d = rng.normal(size=(80, 3)) * 20.0
    reduced, shift = minimum_image_shift(d, SKEWED, True)
    assert shift.dtype == np.int32
    assert np.allclose(reduced, d + shift.astype(float) @ SKEWED)


def test_minimum_image_partial_pbc_leaves_open_component_alone():
    rng = np.random.default_rng(13)
    d = rng.normal(size=(50, 3)) * 20.0
    cell = np.diag([4.0, 5.0, 6.0])
    out = minimum_image(d, cell, [True, True, False])
    assert np.allclose(out[:, 2], d[:, 2])
    assert np.all(np.abs(out[:, 0]) <= 2.0 + 1e-9)


def test_minimum_image_partial_pbc_matches_brute_force_for_a_sheared_slab():
    rng = np.random.default_rng(14)
    cell = np.array([[5.0, 0.0, 0.0], [4.6, 2.2, 0.0], [0.0, 0.0, 30.0]])
    d = rng.normal(size=(60, 3)) * 8.0
    got = np.linalg.norm(minimum_image(d, cell, [True, True, False]), axis=1)
    truth = _brute_force_min_image(d, cell, [True, True, False], n=5)
    assert np.max(got - truth) < 1e-9


def test_minimum_image_no_pbc_is_the_identity():
    rng = np.random.default_rng(15)
    d = rng.normal(size=(10, 3))
    assert np.allclose(minimum_image(d, np.eye(3), False), d)


def test_minimum_image_preserves_shape():
    rng = np.random.default_rng(16)
    d = rng.normal(size=(6, 7, 3)) * 10.0
    assert minimum_image(d, SKEWED, True).shape == (6, 7, 3)


def test_minimum_image_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        minimum_image(np.zeros((1, 3)), np.eye(3), True, method="fast")


# --------------------------------------------------------------------------
# lattice reduction
# --------------------------------------------------------------------------


def test_reduce_cell_is_unimodular_and_shortens():
    rng = np.random.default_rng(17)
    for cell in _random_cells(15, rng, shear=15.0):
        basis, transform = reduce_cell(cell, True)
        assert np.allclose(basis, transform @ cell)
        assert abs(round(float(np.linalg.det(transform)))) == 1
        assert abs(abs(np.linalg.det(basis)) - volume(cell)) < 1e-8
        assert np.sum(np.linalg.norm(basis, axis=1)) <= np.sum(np.linalg.norm(cell, axis=1)) + 1e-9


def test_reduce_cell_of_partial_pbc_has_the_right_rank():
    basis, transform = reduce_cell(HAND_CELL, [True, False, True])
    assert basis.shape == (2, 3)
    assert transform.shape == (2, 2)
    assert np.allclose(basis, transform @ HAND_CELL[[0, 2]])


# --------------------------------------------------------------------------
# lattice parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        (3.0, 3.0, 3.0, 90.0, 90.0, 90.0),  # cubic
        (3.2, 3.2, 5.2, 90.0, 90.0, 120.0),  # hexagonal
        (4.0, 5.0, 6.0, 80.0, 95.0, 110.0),  # triclinic
        (4.05, 4.05, 4.05, 60.0, 60.0, 60.0),  # fcc primitive
        (2.87, 2.87, 2.87, 109.4712206, 109.4712206, 109.4712206),  # bcc primitive
    ],
)
def test_cell_parameters_round_trip(params):
    cell = cell_from_parameters(*params)
    assert np.allclose(cell_to_parameters(cell), params)


def test_cell_from_parameters_uses_the_standard_orientation():
    cell = cell_from_parameters(4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
    assert cell[0, 1] == 0.0 and cell[0, 2] == 0.0
    assert cell[1, 2] == 0.0
    assert cell[1, 1] > 0.0 and cell[2, 2] > 0.0
    assert np.linalg.det(cell) > 0.0  # right handed


def test_cell_parameters_round_trip_from_a_matrix_up_to_rotation():
    params = cell_to_parameters(HAND_CELL)
    rebuilt = cell_from_parameters(*params)
    # Same metric tensor => same cell up to a rigid rotation.
    assert np.allclose(rebuilt @ rebuilt.T, HAND_CELL @ HAND_CELL.T)
    assert np.allclose(cell_to_parameters(rebuilt), params)


def test_cell_from_parameters_known_hexagonal():
    cell = cell_from_parameters(1.0, 1.0, 2.0, 90.0, 90.0, 120.0)
    assert np.allclose(cell[0], [1.0, 0.0, 0.0])
    assert np.allclose(cell[1], [-0.5, math.sqrt(3) / 2, 0.0])
    assert np.allclose(cell[2], [0.0, 0.0, 2.0])
    assert volume(cell) == pytest.approx(math.sqrt(3.0))


def test_cell_from_parameters_rejects_impossible_angles():
    with pytest.raises(ValueError):
        cell_from_parameters(1.0, 1.0, 1.0, 20.0, 20.0, 150.0)
    with pytest.raises(ValueError):
        cell_from_parameters(-1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# strain and Voigt helpers
# --------------------------------------------------------------------------


def test_voigt_round_trip_stress_convention():
    rng = np.random.default_rng(18)
    t = rng.normal(size=(3, 3))
    t = 0.5 * (t + t.T)
    assert np.allclose(voigt_to_full(full_to_voigt(t)), t)


def test_voigt_round_trip_strain_convention():
    rng = np.random.default_rng(19)
    t = rng.normal(size=(3, 3))
    t = 0.5 * (t + t.T)
    assert np.allclose(voigt_to_full(full_to_voigt(t, strain=True), strain=True), t)


def test_voigt_strain_flag_is_the_factor_of_two_in_engineering_shear():
    eps = np.zeros((3, 3))
    eps[1, 2] = eps[2, 1] = 0.01
    v = full_to_voigt(eps, strain=True)
    assert v[3] == pytest.approx(0.02)  # gamma_yz = 2 eps_yz
    assert full_to_voigt(eps)[3] == pytest.approx(0.01)


def test_voigt_index_order_is_xx_yy_zz_yz_xz_xy():
    t = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
    assert np.allclose(full_to_voigt(t), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_full_to_voigt_discards_the_antisymmetric_part():
    t = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(full_to_voigt(t), np.zeros(6))


def test_apply_strain_matches_configuration_strained():
    rng = np.random.default_rng(20)
    pos = rng.normal(size=(12, 3)) * 3.0
    cfg = Configuration(positions=pos, cell=HAND_CELL, pbc=True)
    eps = np.array([[0.01, 0.002, 0.0], [0.002, -0.005, 0.001], [0.0, 0.001, 0.003]])
    strained = cfg.strained(eps)
    assert np.allclose(apply_strain(pos, eps), strained.positions)
    assert np.allclose(apply_strain(HAND_CELL, eps), strained.cell)


def test_apply_strain_accepts_voigt_engineering_strain():
    eps_voigt = np.array([0.01, 0.0, 0.0, 0.02, 0.0, 0.0])
    full = voigt_to_full(eps_voigt, strain=True)
    rng = np.random.default_rng(21)
    v = rng.normal(size=(8, 3))
    assert np.allclose(apply_strain(v, eps_voigt), apply_strain(v, full))


def test_apply_strain_hydrostatic_scales_volume():
    e = 1e-3
    strained = apply_strain(HAND_CELL, e * np.eye(3))
    assert volume(strained) == pytest.approx(volume(HAND_CELL) * (1 + e) ** 3)


def test_apply_strain_zero_is_the_identity():
    assert np.allclose(apply_strain(HAND_CELL, np.zeros((3, 3))), HAND_CELL)
