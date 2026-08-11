"""Validation of the structural observables: g(r), S(q), ADF, q, Q4/Q6.

The tests here are physics, not smoke tests.  Each one pins a number that is
known independently of this code base:

* an ideal gas has ``g(r) = 1`` at *every* ``r`` -- this is the test that
  catches a linearised shell volume, a wrong ``rho``, or a missing factor of
  two in a partial;
* a perfect fcc crystal has delta peaks at ``a/sqrt(2), a, a sqrt(3/2), ...``
  with exactly 12, 6, 24, 12, 24, 8 neighbours under them;
* the Fourier transform of ``h(r)`` and the direct reciprocal-lattice sum are
  two implementations of the same quantity that share no code;
* the tetrahedral order parameter is exactly 1 for diamond and 0 in the mean
  for random directions;
* Steinhardt ``Q4``/``Q6`` for fcc/bcc/hcp/sc are tabulated in Steinhardt,
  Nelson and Ronchetti, Phys. Rev. B 28, 784 (1983).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import sph_harm_y

from atomlab.build import bcc, diamond, fcc, hcp, random_gas, sc
from atomlab.observables.adf import (
    bond_angle_distribution,
    neighbor_groups,
    steinhardt,
    steinhardt_per_atom,
    tetrahedral_order,
    tetrahedral_order_per_atom,
)
from atomlab.observables.base import (
    FrameSample,
    ObservableResult,
    accumulate_frames,
    estimate_from_samples,
    frame_estimator,
    iter_frames,
)
from atomlab.observables.rdf import (
    coordination_number,
    radial_distribution,
    shell_centroids,
    shell_volumes,
    structure_factor,
    structure_factor_direct,
)
from atomlab.types import Configuration, Trajectory

# Literature values (Steinhardt/Nelson/Ronchetti 1983, Table I).  The bcc entry
# is the 8 + 6 neighbour convention; with only the 8 first neighbours one gets
# (0.5093, 0.6285) instead, which is the diamond/tetrahedral value.
STEINHARDT_REFERENCE = {
    "fcc": (0.190941, 0.574524),
    "hcp": (0.097222, 0.484762),
    "bcc": (0.036370, 0.510688),
    "sc": (0.763763, 0.353553),
}

IDEAL_DENSITY = 0.02  # A^-3, dilute enough that a hard-core-free gas is safe


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _gas_frames(n_frames: int, n_atoms: int, density: float, seed: int, min_distance: float = 0.0):
    """Independent random configurations: an exactly known reference ensemble."""
    return [
        random_gas(n_atoms, density, "Ar", seed=seed + k, min_distance=min_distance)
        for k in range(n_frames)
    ]


def _as_trajectory(frames):
    """Wrap a list of configurations as a Trajectory (times are fictitious)."""
    return Trajectory(
        positions=np.stack([f.positions for f in frames]),
        cells=np.stack([f.cell for f in frames]),
        times=np.arange(len(frames), dtype=float),
        template=frames[0],
        info={"dynamical": False},
    )


@pytest.fixture(scope="module")
def gas_frames():
    return _gas_frames(30, 400, IDEAL_DENSITY, seed=1000)


@pytest.fixture(scope="module")
def dense_frames():
    """Random sequential addition: a genuine, non-trivial pair structure."""
    return _gas_frames(10, 400, 0.05, seed=2000, min_distance=2.2)


@pytest.fixture(scope="module")
def fcc_crystal():
    return fcc(4.05, "Al", (4, 4, 4))


# --------------------------------------------------------------------------
# base machinery
# --------------------------------------------------------------------------


def test_bin_geometry_is_exact():
    edges = np.linspace(0.0, 10.0, 101)
    volumes = shell_volumes(edges)
    # The shells must tile the sphere exactly.
    assert volumes.sum() == pytest.approx((4.0 / 3.0) * math.pi * 10.0**3, rel=1e-14)
    # And the linearised 4 pi r^2 dr is wrong by a lot in the first bin, which
    # is precisely why the exact form is used.
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    linearised = 4.0 * math.pi * midpoints**2 * np.diff(edges)
    rel = np.abs(linearised / volumes - 1.0)
    assert rel[0] > 0.03, "first-bin error of the linearised shell volume"
    assert rel[-1] < 1e-4
    centroids = shell_centroids(edges)
    assert np.all(centroids > midpoints)  # volume weighting pushes outward


def test_estimate_from_samples_respects_correlation():
    """Blocking must inflate the error bar of a correlated series."""
    rng = np.random.default_rng(7)
    n = 4096
    phi = 0.9  # AR(1): tau = (1+phi)/(2(1-phi)) = 9.5 frame spacings
    noise = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = noise[0]
    for k in range(1, n):
        x[k] = phi * x[k - 1] + noise[k]

    result = estimate_from_samples(x[:, None], name="ar1")
    naive = x.std(ddof=1) / math.sqrt(n)
    _, error = result.scalar()
    assert error > 2.0 * naive
    # tau for AR(1) with phi = 0.9 is 9.5; Sokal windowing is biased low a
    # little, so accept a factor of two either way.
    assert 4.0 < result.correlation_time < 20.0
    assert 100 < result.n_effective < 700


def test_single_frame_error_is_nan_not_zero():
    result = estimate_from_samples(np.array([[1.0, 2.0]]))
    assert np.all(np.isnan(result.error))
    assert result.n_samples == 1


def test_frame_estimator_decorator_and_transform():
    @frame_estimator
    def mean_x(cfg, *, axis=0):
        return FrameSample(values=[cfg.positions[:, axis].mean()], metadata={"n": cfg.n_atoms})

    frames = _gas_frames(6, 50, IDEAL_DENSITY, seed=11)
    result = mean_x(frames, axis=0)
    assert isinstance(result, ObservableResult)
    assert result.n_samples == 6
    assert result.metadata["per_frame"][0]["n"] == 50

    doubled = result.transform(lambda v: 2.0 * v, name="2x")
    assert doubled.value[0] == pytest.approx(2.0 * result.value[0])
    assert doubled.error[0] == pytest.approx(2.0 * result.error[0], rel=1e-12)

    dropped = mean_x(frames, keep_samples=False)
    with pytest.raises(ValueError, match="keep_samples"):
        dropped.transform(lambda v: v)


def test_iter_frames_accepts_every_source(gas_frames):
    traj = _as_trajectory(gas_frames[:4])
    assert len(iter_frames(traj)) == 4
    assert len(iter_frames(traj, stride=2)) == 2
    assert len(iter_frames(gas_frames[0])) == 1
    assert len(iter_frames(gas_frames[:3])) == 3
    with pytest.raises(TypeError):
        iter_frames([1, 2, 3])


def test_accumulate_rejects_changing_grid(gas_frames):
    calls = {"n": 0}

    def kernel(cfg):
        calls["n"] += 1
        return FrameSample(values=np.zeros(3), bins=np.arange(3) + calls["n"])

    with pytest.raises(ValueError, match="bin grid"):
        accumulate_frames(gas_frames[:3], kernel)


# --------------------------------------------------------------------------
# g(r): normalisation
# --------------------------------------------------------------------------


def test_ideal_gas_rdf_is_unity(gas_frames):
    """The normalisation test: g(r) = 1 everywhere for an ideal gas.

    Sensitive to the shell volume, to the (N-1)/V counting density and to the
    exclusion of self-image pairs all at once.
    """
    result = radial_distribution(gas_frames, 8.0, 40)
    assert result.value.shape == (40,)
    z = (result.value - 1.0) / result.error
    assert np.all(np.isfinite(z))
    # 40 bins: the largest |z| of 40 standard normals is ~2.7 on average, so 4
    # is a real failure rather than bad luck.
    assert np.max(np.abs(z)) < 4.0, f"max |z| = {np.max(np.abs(z)):.2f}"
    # The mean over the outer half is a much tighter test of any systematic
    # offset (e.g. rho = N/V instead of (N-1)/V, which biases by 1/N = 0.25%).
    tail = result.value[len(result.value) // 2 :]
    assert abs(tail.mean() - 1.0) < 0.005


def test_ideal_gas_rdf_beyond_half_the_box():
    """r_max past L/2 is allowed and still normalises to 1 (multi-image counting)."""
    frames = _gas_frames(12, 200, IDEAL_DENSITY, seed=3000)
    box = float(frames[0].cell[0, 0])
    result = radial_distribution(frames, 0.9 * box, 40)
    assert result.metadata["r_max"] > 0.5 * box
    z = (result.value - 1.0) / result.error
    assert np.max(np.abs(z)) < 4.0, f"max |z| = {np.max(np.abs(z)):.2f}"
    beyond = result.bins > 0.5 * box
    assert beyond.sum() > 5
    assert abs(result.value[beyond].mean() - 1.0) < 0.01


def test_rdf_raise_policy_beyond_half_box(gas_frames):
    box = float(gas_frames[0].cell[0, 0])
    with pytest.raises(ValueError, match="exceeds half the minimum cell width"):
        radial_distribution(gas_frames[:2], 0.9 * box, 20, beyond_half_cell="raise")


def test_partial_rdf_prefactors(gas_frames):
    """A randomly labelled binary ideal gas: every partial must be 1.

    This is the test of the a == b versus a != b combinatorial prefactor.  The
    like partials use ``N_a (N_a - 1)`` and the unlike one ``N_a N_b``; getting
    the unlike case wrong is a clean factor of two, and getting the like case
    wrong is a 1/N bias.
    """
    rng = np.random.default_rng(4242)
    labelled = []
    for cfg in gas_frames[:20]:
        species = (rng.random(cfg.n_atoms) < 0.5).astype(np.int32)
        labelled.append(
            Configuration(
                positions=cfg.positions,
                cell=cfg.cell,
                pbc=cfg.pbc,
                species=species,
                symbols=("Ar", "Kr"),
            )
        )
    for pair in [(0, 0), (0, 1), (1, 0), (1, 1), ("Ar", "Kr")]:
        result = radial_distribution(labelled, 8.0, 20, pairs=pair)
        z = (result.value - 1.0) / result.error
        assert np.max(np.abs(z)) < 4.0, f"pair {pair}: max |z| = {np.max(np.abs(z)):.2f}"
        assert abs(result.value[5:].mean() - 1.0) < 0.01, f"pair {pair} offset"

    # g_ab and g_ba are the same observable and must agree to the last bit.
    ab = radial_distribution(labelled, 8.0, 20, pairs=(0, 1))
    ba = radial_distribution(labelled, 8.0, 20, pairs=(1, 0))
    assert np.allclose(ab.value, ba.value, rtol=1e-12, atol=1e-12)


def test_rdf_accepts_a_trajectory(gas_frames):
    traj = _as_trajectory(gas_frames[:10])
    from_traj = radial_distribution(traj, 6.0, 20)
    from_list = radial_distribution(gas_frames[:10], 6.0, 20)
    assert np.allclose(from_traj.value, from_list.value)
    assert from_traj.n_samples == 10
    strided = radial_distribution(traj, 6.0, 20, stride=2)
    assert strided.n_samples == 5


def test_rdf_requires_a_periodic_cell():
    cfg = Configuration(positions=np.random.default_rng(0).random((10, 3)) * 5.0, pbc=False)
    with pytest.raises(ValueError, match="fully periodic"):
        radial_distribution(cfg, 2.0, 10)


# --------------------------------------------------------------------------
# g(r): a perfect crystal
# --------------------------------------------------------------------------


def test_fcc_peaks_and_coordination_numbers(fcc_crystal):
    """Delta peaks at the known shell radii, with exact integer coordination."""
    a = 4.05
    result = radial_distribution(fcc_crystal, 8.0, 400)

    occupied = np.nonzero(result.value)[0]
    expected_radii = a * np.sqrt(np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]))
    found = result.bins[occupied]
    assert len(found) == len(expected_radii)
    assert np.allclose(found, expected_radii, atol=8.0 / 400)  # within one bin width

    # Cumulative coordination numbers: 12, 6, 24, 12, 24, 8 in the shells.
    cumulative = np.cumsum([12, 6, 24, 12, 24, 8])
    midpoints = 0.5 * (expected_radii[:-1] + expected_radii[1:])
    for r_cut, expected in zip(midpoints, cumulative):
        value, _ = coordination_number(result, float(r_cut)).scalar()
        assert value == pytest.approx(expected, abs=1e-9), f"n({r_cut:.3f}) = {value}"


def test_coordination_number_of_an_ideal_gas(gas_frames):
    """n(r) = rho * 4/3 pi r^3 for an ideal gas, with a real error bar."""
    result = radial_distribution(gas_frames, 8.0, 80)
    for r_cut in (3.0, 5.0, 7.5):
        estimate = coordination_number(result, r_cut)
        value, error = estimate.scalar()
        rho = (gas_frames[0].n_atoms - 1) / gas_frames[0].volume
        expected = rho * (4.0 / 3.0) * math.pi * r_cut**3
        assert error > 0.0
        assert abs(value - expected) < 4.0 * error, f"n({r_cut}) = {value} +/- {error}"


def test_coordination_number_rejects_r_cut_beyond_r_max(fcc_crystal):
    result = radial_distribution(fcc_crystal, 5.0, 100)
    with pytest.raises(ValueError, match="r_max"):
        coordination_number(result, 6.0)


# --------------------------------------------------------------------------
# structure factor
# --------------------------------------------------------------------------


def test_structure_factor_routes_agree(dense_frames):
    """The load-bearing cross-check: FT of h(r) versus the direct sum.

    The two share no code beyond the frame loop, so agreement validates the
    shell volumes, the density convention and the transform simultaneously.
    Compared for q >= 1 1/A: below that the sphere-truncated transform and the
    cell-periodic sum genuinely differ (finite-size, not a bug), which is why
    S(q -> 0) is not claimed by either route.
    """
    rdf = radial_distribution(dense_frames, 10.0, 200)
    direct = structure_factor_direct(dense_frames, 4.0, n_bins=40)
    q = direct.bins
    ft_raw = structure_factor(rdf, q, window="none")
    ft_lorch = structure_factor(rdf, q, window="lorch")

    window = q >= 1.0
    assert window.sum() > 20
    d_raw = np.abs(direct.value[window] - ft_raw.value[window])
    d_lorch = np.abs(direct.value[window] - ft_lorch.value[window])
    assert d_raw.mean() < 0.03, f"mean |dS| = {d_raw.mean():.4f}"
    assert d_raw.max() < 0.08, f"max |dS| = {d_raw.max():.4f}"
    # The Lorch window trades resolution for smoothness, so it is allowed to
    # differ a little more -- but not qualitatively.
    assert d_lorch.mean() < 0.05
    assert d_lorch.max() < 0.12
    # Both routes must see the same first peak, to within the Lorch broadening.
    assert abs(q[np.argmax(direct.value)] - q[np.argmax(ft_raw.value)]) < 0.25
    # Sanity: the structure is real, not noise.
    assert direct.value.max() > 1.2


def test_structure_factor_of_an_ideal_gas(gas_frames):
    """S(q) = 1 for uncorrelated positions, by both routes."""
    rdf = radial_distribution(gas_frames, 8.0, 80)
    direct = structure_factor_direct(gas_frames[:10], 3.0, n_bins=20)
    assert np.all(np.abs(direct.value - 1.0) < 4.0 * direct.error + 0.05)
    ft = structure_factor(rdf, direct.bins, window="lorch")
    assert np.max(np.abs(ft.value - 1.0)) < 0.15


def test_structure_factor_direct_rejects_varying_cells(gas_frames):
    stretched = gas_frames[1].copy()
    stretched.cell = stretched.cell * 1.01
    with pytest.raises(ValueError, match="different cell"):
        structure_factor_direct([gas_frames[0], stretched], 2.0)


def test_structure_factor_direct_uses_commensurate_q(dense_frames):
    """Every q must be a reciprocal-lattice vector of the cell."""
    box = float(dense_frames[0].cell[0, 0])
    result = structure_factor_direct(dense_frames[:3], 2.0, n_bins=1000)
    # Bin centres are means over shells of commensurate vectors, so each must
    # be expressible as |2 pi n / L| with integer n components.
    n_squared = (result.bins * box / (2.0 * math.pi)) ** 2
    assert np.allclose(n_squared, np.round(n_squared), atol=1e-8)
    assert result.bins.min() == pytest.approx(2.0 * math.pi / box, rel=1e-10)


# --------------------------------------------------------------------------
# bond angle distribution
# --------------------------------------------------------------------------


def test_bond_angle_normalisations(gas_frames):
    """theta-density integrates to 1; the solid-angle density is flat at 1/2."""
    frames = gas_frames[:8]
    theta = bond_angle_distribution(frames, 6.0, 36)
    assert (theta.value * np.diff(theta.metadata["edges_deg"])).sum() == pytest.approx(1.0)

    solid = bond_angle_distribution(frames, 6.0, 36, normalization="solid_angle")
    # Random directions give exactly 1/2 per unit cos(theta).  The end bins
    # subtend a tiny solid angle and are correspondingly noisy, so judge them
    # by their own error bars rather than by an absolute tolerance.
    z = (solid.value - 0.5) / solid.error
    assert np.max(np.abs(z)) < 4.0, f"max |z| = {np.max(np.abs(z)):.2f}"
    assert abs(solid.value[2:-2].mean() - 0.5) < 0.01

    # In the theta representation the same random gas is *not* flat: it follows
    # the sin(theta) Jacobian.  A peak at 90 degrees here means nothing.
    centres = np.radians(theta.bins)
    jacobian = 0.5 * np.sin(centres) * math.pi / 180.0
    assert np.max(np.abs(theta.value - jacobian)) < 5.0 * np.max(theta.error)


def test_bond_angle_of_diamond_is_tetrahedral():
    cfg = diamond(5.431, "Si", (2, 2, 2))
    result = bond_angle_distribution(cfg, 3.0, 720)
    occupied = np.nonzero(result.value)[0]
    assert len(occupied) == 1
    assert result.bins[occupied][0] == pytest.approx(math.degrees(math.acos(-1 / 3)), abs=0.25)


def test_bond_angle_requires_two_neighbours():
    cfg = fcc(4.05, "Al", (3, 3, 3))
    with pytest.raises(ValueError, match="two neighbours"):
        bond_angle_distribution(cfg, 1.0, 10)


# --------------------------------------------------------------------------
# tetrahedral order
# --------------------------------------------------------------------------


def test_tetrahedral_order_of_diamond_is_one():
    cfg = diamond(5.431, "Si", (2, 2, 2))
    q = tetrahedral_order(cfg, 3.0)
    value, _ = q.scalar()
    assert value == pytest.approx(1.0, abs=1e-12)
    per_atom = tetrahedral_order_per_atom(cfg, 3.0)
    assert np.allclose(per_atom, 1.0, atol=1e-12)


def test_tetrahedral_order_of_a_gas_is_zero(gas_frames):
    q = tetrahedral_order(gas_frames[:10], 8.0)
    value, error = q.scalar()
    assert abs(value) < 0.05, f"<q> = {value:.4f} +/- {error:.4f}"
    assert error < 0.02


def test_tetrahedral_order_needs_four_neighbours():
    cfg = diamond(5.431, "Si", (2, 2, 2))
    with pytest.raises(ValueError, match="neighbours within"):
        tetrahedral_order(cfg, 1.5)


# --------------------------------------------------------------------------
# Steinhardt bond order
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lattice,r_cut",
    [("fcc", 3.4), ("bcc", 3.0), ("hcp", 3.0), ("sc", 3.5)],
)
def test_steinhardt_literature_values(lattice, r_cut):
    """Q4 and Q6 of perfect lattices against Steinhardt et al. (1983)."""
    builders = {
        "fcc": lambda: fcc(4.05, "Al", (3, 3, 3)),
        "bcc": lambda: bcc(2.87, "Cu", (3, 3, 3)),
        "hcp": lambda: hcp(2.55, "Cu", None, (4, 4, 3)),
        "sc": lambda: sc(3.0, "Cu", (4, 4, 4)),
    }
    cfg = builders[lattice]()
    q4 = steinhardt(cfg, 4, r_cut).scalar()[0]
    q6 = steinhardt(cfg, 6, r_cut).scalar()[0]
    ref4, ref6 = STEINHARDT_REFERENCE[lattice]
    assert q4 == pytest.approx(ref4, abs=5e-4), f"{lattice} Q4 = {q4:.6f}"
    assert q6 == pytest.approx(ref6, abs=5e-4), f"{lattice} Q6 = {q6:.6f}"
    # Every atom is equivalent in a perfect lattice, so the local parameter has
    # no spread and the global one coincides with it.
    per_atom = steinhardt_per_atom(cfg, 6, r_cut)
    assert per_atom.std() < 1e-12
    assert steinhardt(cfg, 6, r_cut, kind="global").scalar()[0] == pytest.approx(q6, abs=1e-9)


def test_steinhardt_of_diamond():
    """Four tetrahedral neighbours give the same invariants as bcc's 8 shell."""
    cfg = diamond(5.431, "Si", (2, 2, 2))
    q4 = steinhardt(cfg, 4, 3.0).scalar()[0]
    q6 = steinhardt(cfg, 6, 3.0).scalar()[0]
    assert q4 == pytest.approx(0.509329, abs=5e-4), f"diamond Q4 = {q4:.6f}"
    assert q6 == pytest.approx(0.628539, abs=5e-4), f"diamond Q6 = {q6:.6f}"


def test_steinhardt_is_rotationally_invariant(fcc_crystal):
    angle = 0.7
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
    rotated = fcc_crystal.rotated(rotation)
    for l in (4, 6):
        a = steinhardt_per_atom(fcc_crystal, l, 3.4)
        b = steinhardt_per_atom(rotated, l, 3.4)
        assert np.allclose(np.sort(a), np.sort(b), atol=1e-12)


def test_steinhardt_real_and_complex_harmonics_agree(fcc_crystal):
    """Q_l is basis-independent: real and complex harmonics must give one number.

    ``sum_m |q_lm|^2`` is a squared norm, and the real and complex spherical
    harmonic bases are related by a unitary matrix, so the invariant cannot
    depend on the choice.  Checking it numerically guards against a wrong
    Condon-Shortley phase or a swapped (theta, phi) argument order in
    ``sph_harm_y``, neither of which the perfect-lattice values alone would
    catch for every l.
    """
    l, r_cut = 6, 3.4
    d, r, offsets = neighbor_groups(fcc_crystal, r_cut)
    theta = np.arccos(np.clip(d[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0])
    counts = np.diff(offsets)
    central = np.repeat(np.arange(fcc_crystal.n_atoms), counts)

    columns = []
    for m in range(-l, l + 1):
        y_pos = sph_harm_y(l, abs(m), theta, phi)
        if m == 0:
            columns.append(y_pos.real)
        elif m > 0:
            columns.append(math.sqrt(2.0) * (-1) ** m * y_pos.real)
        else:
            columns.append(math.sqrt(2.0) * (-1) ** m * y_pos.imag)
    y_real = np.stack(columns, axis=1)

    accum = np.zeros((fcc_crystal.n_atoms, 2 * l + 1))
    np.add.at(accum, central, y_real)
    q_lm = accum / counts[:, None]
    q_real = np.sqrt(4.0 * math.pi / (2 * l + 1) * (q_lm**2).sum(axis=1))

    assert np.allclose(q_real, steinhardt_per_atom(fcc_crystal, l, r_cut), atol=1e-12)


def test_steinhardt_discriminates_solid_from_gas(gas_frames):
    """Q6 is the melting diagnostic: fcc ~0.575, disordered ~0.3 with a spread."""
    crystal = steinhardt(fcc(4.05, "Al", (3, 3, 3)), 6, 3.4).scalar()[0]
    gas = steinhardt(gas_frames[:8], 6, 4.5)
    gas_value, gas_error = gas.scalar()
    assert gas_value < 0.45
    assert crystal - gas_value > 0.1
    assert gas_error > 0.0
    # The *global* parameter is the sharper discriminator: bond orientations of
    # a disordered system cancel, so it goes to zero as 1/sqrt(n_bonds), while
    # the local mean stays finite because each atom has few neighbours.
    global_value, _ = gas.metadata["global"]
    assert global_value < 0.1 * gas_value


def test_averaged_steinhardt_narrows_the_distribution():
    """Lechner-Dellago averaging must reduce the per-atom spread of a rattled solid."""
    from atomlab.build import rattle

    cfg = rattle(fcc(4.05, "Al", (4, 4, 4)), 0.12, seed=5)
    bare = steinhardt_per_atom(cfg, 6, 3.4)
    averaged = steinhardt_per_atom(cfg, 6, 3.4, averaged=True)
    assert averaged.std() < bare.std()
    assert abs(averaged.mean() - 0.5745) < abs(bare.mean() - 0.5745) + 0.05
