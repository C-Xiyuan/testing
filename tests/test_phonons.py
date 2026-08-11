"""Phonon machinery validated against a lattice whose spectrum is known exactly.

The test system is a simple-cubic lattice with nearest-neighbour central-force
springs at zero tension (the spring rest length equals the lattice constant, so
the transverse restoring force vanishes at first order).  Its force-constant
matrix is then ``k * xhat (x) xhat`` for the two x-neighbours and likewise for y
and z, the dynamical matrix is diagonal, and

    omega_x(q) = 2 sqrt(k/m) |sin(q_x a / 2)|

with the other two branches independent of ``q_x``.  That closed form pins down
the Fourier sum, the mass weighting, the unit conversion and the sign convention
all at once -- there is nowhere for an error to hide.

The second test is band folding: describing the *same* crystal with a doubled
cell must reproduce the same physics, with the zone-boundary modes of the small
cell appearing at the zone centre of the large one.  That exercises the
multi-atom-basis path, which the one-atom test cannot reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.observables.phonons import (
    compute_force_constants,
    cubic_q_path,
    debye_temperature_from_dos,
    dynamical_matrix,
    harmonic_free_energy,
    harmonic_heat_capacity,
    phonon_dos,
    phonon_frequencies,
    zero_point_energy,
)
from atomlab.potentials.base import Potential
from atomlab.types import Configuration, Result
from atomlab.units import E2MVV, KB, RADPS_TO_THZ

A = 3.0            # lattice constant, A
K = 2.5            # spring constant, eV/A^2
MASS = 40.0        # amu
CUTOFF = 1.2 * A   # between the first shell (a) and the second (a sqrt(2))


class NearestNeighbourSpring(Potential):
    """u(r) = k (r - r0)^2 / 2 for r < cutoff, zero beyond.

    Defined here rather than imported so that the analytic phonon answer is
    tested against a potential whose form is visible in this file.  The
    discontinuity at the cutoff is harmless: the cutoff is chosen to sit between
    coordination shells, so no pair ever crosses it in these tests.
    """

    def __init__(self, k: float = K, r0: float = A, cutoff: float = CUTOFF):
        self.k, self.r0, self.cutoff = float(k), float(r0), float(cutoff)
        self.name = "nn-spring"

    def compute(self, configuration, *, forces=True, virial=True) -> Result:
        from atomlab.neighbors import build_neighbor_list, pair_vectors

        nl = build_neighbor_list(configuration, self.cutoff, half=True)
        d, r = pair_vectors(configuration, nl)

        energy = float((0.5 * self.k * (r - self.r0) ** 2).sum())
        f = np.zeros((configuration.n_atoms, 3))
        w = np.zeros((3, 3))

        # du/dr = k (r - r0); the force on i from j points along -d/r since d
        # points from i to j, so f_i = +k(r - r0) * (d/r).
        coefficient = self.k * (r - self.r0) / r
        pair_force = coefficient[:, None] * d          # force on i due to j
        np.add.at(f, nl.i, pair_force)
        np.add.at(f, nl.j, -pair_force)
        # W_ab = sum_{i<j} f_ij (x) r_ij with r_ij = r_i - r_j = -d.
        w = -(pair_force[:, :, None] * d[:, None, :]).sum(axis=0)
        return Result(energy=energy, forces=f, virial=w if virial else None)


def simple_cubic(a: float = A) -> Configuration:
    return Configuration(
        positions=np.zeros((1, 3)),
        cell=np.eye(3) * a,
        pbc=True,
        species=np.zeros(1, dtype=np.int32),
        masses=np.full(1, MASS),
    )


def doubled_cell(a: float = A) -> Configuration:
    """The same crystal described with two atoms per cell, doubled along x."""
    return Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0]]),
        cell=np.diag([2 * a, a, a]),
        pbc=True,
        species=np.zeros(2, dtype=np.int32),
        masses=np.full(2, MASS),
    )


def analytic_branch_thz(q_component: float, a: float = A) -> float:
    """The longitudinal branch 2 sqrt(k/m) |sin(q a / 2)|, converted to THz."""
    omega = 2.0 * np.sqrt(K / MASS * E2MVV) * abs(np.sin(q_component * a / 2.0))
    return omega * RADPS_TO_THZ


@pytest.fixture(scope="module")
def fc_sc():
    return compute_force_constants(NearestNeighbourSpring(), simple_cubic(), (3, 3, 3), delta=0.005)


#: Tolerance on frequencies that are exactly zero in the continuum limit.
#: Finite displacement puts the transverse springs under a spurious tension of
#: order ``k delta^2 / 2a``, which lifts the flat transverse branches to a small
#: non-zero frequency.  That is a property of the finite-difference scheme, not
#: a bug, and :meth:`TestForceConstants.test_transverse_artefact_scales_as_delta_squared`
#: verifies it behaves as the analysis says it should.
ZERO_TOL_THZ = 2e-2


class TestForceConstants:
    def test_analytic_force_constants(self, fc_sc):
        """For zero-tension NN springs, Phi for an x-neighbour is k xhat (x) xhat."""
        phi = fc_sc.values[0]  # (3, n_super, 3)
        # The self term must be -sum of the neighbour terms, i.e. 2k on each
        # diagonal component (two neighbours per axis).  The residual 1e-5
        # relative deviation is the O(delta^2) transverse tension described above.
        assert phi[0, 0, 0] == pytest.approx(2 * K, rel=1e-4)
        assert phi[1, 0, 1] == pytest.approx(2 * K, rel=1e-4)
        assert phi[2, 0, 2] == pytest.approx(2 * K, rel=1e-4)
        # Off-diagonal cartesian blocks vanish for a cubic lattice of central
        # springs aligned with the axes.
        assert abs(phi[0, 0, 1]) < 1e-8

    def test_transverse_artefact_scales_as_delta_squared(self):
        """The spurious transverse stiffness must vanish as delta^2, not slower.

        A bug in the Fourier sum or the image weighting would produce a
        transverse frequency that does *not* shrink with the displacement, so
        this scaling is what distinguishes a numerical artefact from an error.
        """
        residuals = []
        for delta in (0.02, 0.01, 0.005):
            fc = compute_force_constants(
                NearestNeighbourSpring(), simple_cubic(), (3, 3, 3), delta=delta
            )
            freq = sorted(phonon_frequencies(fc, np.array([[np.pi / A, 0.0, 0.0]]))[0])
            residuals.append(freq[0])
        # omega ~ sqrt(Phi) and Phi ~ delta^2, so omega ~ delta: halving delta
        # halves the residual.
        ratios = [residuals[i] / residuals[i + 1] for i in range(len(residuals) - 1)]
        assert all(1.6 < r < 2.5 for r in ratios), residuals

    def test_acoustic_sum_rule_is_nearly_satisfied_before_enforcement(self, fc_sc):
        """A correct implementation barely needs the correction it applies."""
        assert fc_sc.asr_residual < 1e-6

    def test_supercell_too_small_raises(self):
        with pytest.raises(ValueError, match="minimum width"):
            compute_force_constants(NearestNeighbourSpring(), simple_cubic(), (2, 2, 2))

    def test_unrelaxed_structure_raises(self):
        """A one-atom cell has zero forces by symmetry, so displace a two-atom one."""
        # A modest displacement is not enough: the atom's own periodic images
        # move with it, so the springs along the displacement direction stay at
        # their rest length and only the perpendicular bonds are stretched.
        cfg = doubled_cell()
        cfg.positions[1, 1] += 0.5
        with pytest.raises(ValueError, match="not relaxed"):
            compute_force_constants(NearestNeighbourSpring(), cfg, (2, 3, 3))


class TestDispersionAgainstClosedForm:
    def test_gamma_point_has_three_acoustic_zeros(self, fc_sc):
        freq = phonon_frequencies(fc_sc, np.zeros((1, 3)))[0]
        assert np.allclose(freq, 0.0, atol=ZERO_TOL_THZ)

    @pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 1.0])
    def test_longitudinal_branch_matches_analytic(self, fc_sc, fraction):
        """omega_x(q) = 2 sqrt(k/m) |sin(q a/2)| along the x axis."""
        qx = fraction * np.pi / A
        freq = phonon_frequencies(fc_sc, np.array([[qx, 0.0, 0.0]]))[0]
        expected_longitudinal = analytic_branch_thz(qx)
        # The two transverse branches are flat at zero along this direction,
        # because a central-force cubic lattice has no transverse restoring
        # force for a purely longitudinal wavevector.
        assert np.allclose(sorted(freq)[:2], 0.0, atol=ZERO_TOL_THZ)
        assert sorted(freq)[2] == pytest.approx(expected_longitudinal, rel=1e-5)

    def test_full_dispersion_along_a_general_direction(self, fc_sc):
        """Each cartesian branch depends only on its own q component."""
        q = np.array([0.7, 0.3, 0.9]) * np.pi / A
        freq = sorted(phonon_frequencies(fc_sc, q[None, :])[0])
        expected = sorted(analytic_branch_thz(qi) for qi in q)
        assert np.allclose(freq, expected, rtol=1e-4, atol=ZERO_TOL_THZ)

    def test_zone_boundary_maximum(self, fc_sc):
        """The band maximum is 2 sqrt(k/m), reached at the zone boundary."""
        freq = phonon_frequencies(fc_sc, np.array([[np.pi / A, 0.0, 0.0]]))[0]
        omega_max = 2.0 * np.sqrt(K / MASS * E2MVV) * RADPS_TO_THZ
        assert freq.max() == pytest.approx(omega_max, rel=1e-5)

    def test_dynamical_matrix_is_hermitian(self, fc_sc):
        d = dynamical_matrix(fc_sc, np.array([0.4, -0.2, 0.7]))
        assert np.allclose(d, d.conj().T, atol=1e-12)


class TestBandFolding:
    """The same crystal in a doubled cell must give the folded spectrum."""

    def test_doubled_cell_reproduces_zone_boundary_at_gamma(self):
        fc = compute_force_constants(
            NearestNeighbourSpring(), doubled_cell(), (2, 3, 3), delta=0.005
        )
        freq = sorted(phonon_frequencies(fc, np.zeros((1, 3)))[0])

        # Six branches: the three q=0 modes of the small cell (all zero) plus
        # the three modes at q = (pi/a, 0, 0), which are two transverse zeros
        # and one longitudinal mode at the band maximum.
        omega_max = 2.0 * np.sqrt(K / MASS * E2MVV) * RADPS_TO_THZ
        assert len(freq) == 6
        assert np.allclose(freq[:5], 0.0, atol=ZERO_TOL_THZ)
        assert freq[5] == pytest.approx(omega_max, rel=1e-4)

    def test_doubled_cell_matches_small_cell_at_a_shared_wavevector(self, fc_sc):
        """At a q commensurate with both cells the physical branches agree."""
        fc2 = compute_force_constants(
            NearestNeighbourSpring(), doubled_cell(), (2, 3, 3), delta=0.005
        )
        q = np.array([[0.0, 0.5 * np.pi / A, 0.0]])
        small = sorted(phonon_frequencies(fc_sc, q)[0])
        large = sorted(phonon_frequencies(fc2, q)[0])
        # The doubled cell shows each small-cell branch twice (once folded from
        # q and once from q + pi/a x, which for a y-directed q gives the same
        # set of frequencies).
        assert np.allclose(large[::2], small, atol=ZERO_TOL_THZ)


class TestDensityOfStates:
    def test_dos_integrates_to_three_times_the_basis_size(self, fc_sc):
        dos = phonon_dos(fc_sc, mesh=(10, 10, 10), n_bins=200)
        assert dos["integral"] == pytest.approx(3.0, rel=0.05)
        assert dos["n_imaginary"] == 0

    def test_dos_upper_edge_is_the_band_maximum(self, fc_sc):
        dos = phonon_dos(fc_sc, mesh=(10, 10, 10), n_bins=200)
        omega_max = 2.0 * np.sqrt(K / MASS * E2MVV) * RADPS_TO_THZ
        support = dos["frequencies"][dos["dos"] > 0.01 * dos["dos"].max()]
        assert support.max() == pytest.approx(omega_max, rel=0.08)

    def test_debye_temperature_is_physically_sized(self, fc_sc):
        dos = phonon_dos(fc_sc, mesh=(8, 8, 8))
        theta = debye_temperature_from_dos(dos)
        assert 50.0 < theta < 2000.0


class TestHarmonicThermodynamics:
    """Tested on an Einstein spectrum, where every quantity is closed-form.

    The spring lattice above is a poor vehicle for these functions: two thirds of
    its branches are identically flat at zero frequency (a central-force cubic
    lattice has no transverse restoring force), so it is mechanically unstable
    and its low-temperature limit is dominated by modes that should not be there.
    Feeding the thermodynamic functions a synthetic spectrum instead tests them
    against exact answers and keeps the two concerns separate.
    """

    N_MODES = 300
    NU = 5.0  # THz

    @property
    def spectrum(self) -> np.ndarray:
        return np.full(self.N_MODES, self.NU)

    def test_zero_point_energy_closed_form(self):
        from atomlab.units import HPLANCK

        assert zero_point_energy(self.spectrum) == pytest.approx(
            0.5 * HPLANCK * self.NU * self.N_MODES, rel=1e-12
        )

    def test_dulong_petit_limit(self):
        """C_v -> N k_B when k T greatly exceeds h nu."""
        cv = harmonic_heat_capacity(self.spectrum, temperature=20_000.0)
        assert cv == pytest.approx(self.N_MODES * KB, rel=1e-3)

    def test_einstein_heat_capacity_closed_form(self):
        """C_v = N k_B x^2 e^x / (e^x - 1)^2 with x = h nu / k T."""
        from atomlab.units import HPLANCK

        temperature = 150.0
        x = HPLANCK * self.NU / (KB * temperature)
        expected = self.N_MODES * KB * x**2 * np.exp(x) / (np.exp(x) - 1.0) ** 2
        assert harmonic_heat_capacity(self.spectrum, temperature) == pytest.approx(expected, rel=1e-10)

    def test_heat_capacity_is_exponentially_suppressed_at_low_temperature(self):
        low = harmonic_heat_capacity(self.spectrum, 5.0)
        high = harmonic_heat_capacity(self.spectrum, 20_000.0)
        assert low < 1e-6 * high

    def test_free_energy_tends_to_the_zero_point_energy(self):
        zpe = zero_point_energy(self.spectrum)
        assert harmonic_free_energy(self.spectrum, 1.0) == pytest.approx(zpe, rel=1e-12)

    def test_free_energy_high_temperature_limit(self):
        """F -> N k T ln(h nu / k T) in the classical regime."""
        from atomlab.units import HPLANCK

        temperature = 20_000.0
        x = HPLANCK * self.NU / (KB * temperature)
        expected = self.N_MODES * KB * temperature * np.log(x)
        assert harmonic_free_energy(self.spectrum, temperature) == pytest.approx(expected, rel=2e-3)

    def test_imaginary_and_zero_modes_are_excluded(self):
        """Soft and unstable modes have no thermal occupation and must be dropped."""
        contaminated = np.concatenate([self.spectrum, [0.0, -3.0, -0.5]])
        assert zero_point_energy(contaminated) == pytest.approx(zero_point_energy(self.spectrum))
        assert harmonic_heat_capacity(contaminated, 300.0) == pytest.approx(
            harmonic_heat_capacity(self.spectrum, 300.0)
        )


class TestQPath:
    def test_cubic_path_endpoints_and_monotone_distance(self):
        q, d, ticks, labels = cubic_q_path(4.05, path="GXWLGK", n_points=20)
        assert labels == list("GXWLGK")
        assert np.all(np.diff(d) >= -1e-12)
        assert np.allclose(q[0], 0.0)
        assert len(ticks) == 6

    def test_unknown_symmetry_point_raises(self):
        with pytest.raises(ValueError, match="unknown high-symmetry"):
            cubic_q_path(4.05, path="GZ")
