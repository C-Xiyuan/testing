"""Harmonic phonons from finite displacements.

Phonons are the sharpest observable in this study.  They are a *curvature*
property -- second derivatives of the potential at a minimum -- whereas force
error is a first-derivative property, so they probe a part of the error field
that force RMSE is structurally blind to.  A model can reproduce forces on
thermally-sampled configurations well and still get the curvature at the
equilibrium structure wrong, and the phonon spectrum is where that shows up.

Method
------
The supercell finite-displacement approach.  Displace each atom of the
primitive cell in turn inside a supercell, read off the forces on every atom,
and assemble the force-constant matrix

.. math:: \\Phi^{\\alpha\\beta}_{0\\kappa, R\\kappa'} = -\\frac{\\partial F^{\\beta}_{R\\kappa'}}{\\partial u^{\\alpha}_{0\\kappa}}

Forces are differentiated, not energies: the analytic forces are already exact,
so one finite difference is enough and the ``1/delta^2`` noise amplification of a
double difference is avoided entirely.

The dynamical matrix at wavevector ``q`` follows as

.. math:: D^{\\alpha\\beta}_{\\kappa\\kappa'}(q) = \\frac{1}{\\sqrt{m_\\kappa m_{\\kappa'}}} \\sum_R \\Phi^{\\alpha\\beta}_{0\\kappa,R\\kappa'} e^{i q \\cdot R}

and the frequencies are the square roots of its eigenvalues.

Two details separate a correct implementation from a plausible one, and both are
handled explicitly below:

* **Image assignment.**  In a finite supercell the pair ``(0κ, Rκ')`` may be
  reachable through several lattice translations of equal length.  Assigning the
  force constant to just one of them breaks the symmetry of the Fourier sum and
  produces band structures that look reasonable but are wrong away from the
  commensurate q-points.  Equidistant images are found and the force constant is
  split equally among them.
* **The acoustic sum rule.**  Translating the whole crystal costs no energy, so
  ``sum_j Phi_ij = 0`` exactly.  Finite-difference error violates it by a small
  amount, which shows up as acoustic modes with non-zero frequency at Gamma --
  and, if the violation is negative, as spurious imaginary frequencies that
  would be misread as a structural instability.  The rule is imposed on the
  self-term, and the residual before imposition is reported so that a large
  violation is visible rather than absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..types import Configuration
from ..units import HPLANCK, KB, E2MVV, RADPS_TO_THZ, THZ_TO_CM1, THZ_TO_MEV

__all__ = [
    "ForceConstants",
    "compute_force_constants",
    "dynamical_matrix",
    "phonon_frequencies",
    "phonon_dos",
    "band_structure",
    "cubic_q_path",
    "harmonic_free_energy",
    "harmonic_heat_capacity",
    "zero_point_energy",
    "debye_temperature_from_dos",
]


@dataclass
class ForceConstants:
    """Force constants of a crystal, in the layout the Fourier sum needs.

    Attributes
    ----------
    values:
        ``(n_prim, 3, n_super, 3)`` array.  ``values[k, a, s, b]`` is
        ``-dF[s, b] / du[k, a]``, in eV/A^2, where ``k`` indexes the primitive
        basis and ``s`` indexes every atom of the supercell.
    primitive:
        The primitive :class:`~atomlab.types.Configuration`.
    reps:
        Supercell repetitions ``(n1, n2, n3)`` used to build the supercell.
    images:
        ``(n_super,)`` array of lattice-translation indices; ``images[s]`` gives
        the row of :attr:`translations` that supercell atom ``s`` belongs to.
    translations:
        ``(n_images, 3)`` cartesian lattice translations of the primitive cell.
    asr_residual:
        Largest ``|sum_s Phi[k, a, s, b]|`` before the acoustic sum rule was
        imposed, in eV/A^2.  A value comparable to the diagonal force constants
        means the displacement was too large, the structure was not relaxed, or
        the supercell is too small for the interaction range.
    """

    values: np.ndarray
    primitive: Configuration
    reps: tuple[int, int, int]
    images: np.ndarray
    translations: np.ndarray
    asr_residual: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n_prim(self) -> int:
        return self.primitive.n_atoms

    @property
    def n_super(self) -> int:
        return self.values.shape[2]

    @property
    def masses(self) -> np.ndarray:
        """``(n_prim,)`` masses of the primitive basis, in amu."""
        return self.primitive.masses


def _supercell_layout(primitive: Configuration, reps: Sequence[int]):
    """Reproduce the atom ordering that :meth:`Configuration.repeated` produces.

    The mapping has to match exactly, because the force-constant array is
    indexed by supercell atom.  ``repeated`` iterates images in ``i, j, k`` order
    and lays out the whole primitive basis within each image, so supercell atom
    ``s`` is primitive atom ``s % n_prim`` in image ``s // n_prim``.
    """
    nx, ny, nz = (int(r) for r in reps)
    offsets = np.array(
        [
            i * primitive.cell[0] + j * primitive.cell[1] + k * primitive.cell[2]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ]
    )
    n_prim = primitive.n_atoms
    n_images = len(offsets)
    images = np.repeat(np.arange(n_images), n_prim)
    basis = np.tile(np.arange(n_prim), n_images)
    return offsets, images, basis


def compute_force_constants(
    potential,
    primitive: Configuration,
    reps: Sequence[int] = (3, 3, 3),
    *,
    delta: float = 0.01,
    enforce_asr: bool = True,
    symmetrize: bool = True,
) -> ForceConstants:
    """Finite-displacement force constants.

    Parameters
    ----------
    potential:
        Any :class:`~atomlab.potentials.base.Potential`.  Learned models work
        here exactly as analytic ones do, which is the point.
    primitive:
        The **relaxed** primitive cell.  Phonons of an unrelaxed structure are
        not meaningful: the linear term in the energy expansion does not vanish,
        so the quadratic coefficients are not the force constants.  Relax with
        :func:`atomlab.md.simulate.minimize` first; this function checks the
        residual forces and raises if they are large.
    reps:
        Supercell repetitions.  Must be large enough that the supercell exceeds
        twice the potential's cutoff in every direction, or interactions wrap
        around and the force constants are contaminated.  Checked and raised on.
    delta:
        Displacement in angstrom.  ``0.01`` is the usual choice: large enough
        that the force response dominates numerical noise, small enough that
        cubic anharmonicity contributes negligibly.  Because analytic forces are
        differenced (not energies), the error is ``O(delta^2)`` from
        anharmonicity alone.
    enforce_asr:
        Impose ``sum_s Phi[k, :, s, :] = 0`` on the self-term.
    symmetrize:
        Average ``Phi`` with its transpose under exchange of the two atom-index
        pairs.  Exact for a conservative potential; imposing it halves the
        finite-difference noise at no cost in correctness.

    Returns
    -------
    ForceConstants
    """
    reps = tuple(int(r) for r in reps)
    supercell = primitive.repeated(reps)

    cutoff = getattr(potential, "cutoff", 0.0)
    if np.isfinite(cutoff) and cutoff > 0:
        from ..cell import min_cell_width

        width = min_cell_width(supercell.cell, supercell.pbc)
        if width < 2.0 * cutoff:
            raise ValueError(
                f"supercell {reps} has minimum width {width:.2f} A, which is below "
                f"2 x cutoff = {2 * cutoff:.2f} A. Force constants would be contaminated "
                "by interactions wrapping through the periodic boundary; increase reps."
            )

    residual = np.abs(potential.forces(supercell)).max()
    if residual > 1e-3:
        raise ValueError(
            f"structure is not relaxed: residual force {residual:.2e} eV/A exceeds 1e-3. "
            "Phonons of an unrelaxed structure are not force constants -- relax first."
        )

    offsets, images, _ = _supercell_layout(primitive, reps)
    n_prim, n_super = primitive.n_atoms, supercell.n_atoms

    phi = np.zeros((n_prim, 3, n_super, 3))
    for k in range(n_prim):
        for a in range(3):
            plus = supercell.copy()
            plus.positions[k, a] += delta
            minus = supercell.copy()
            minus.positions[k, a] -= delta
            f_plus = potential.forces(plus)
            f_minus = potential.forces(minus)
            phi[k, a] = -(f_plus - f_minus) / (2.0 * delta)

    asr_residual = float(np.abs(phi.sum(axis=2)).max())
    if enforce_asr:
        # The self-term is the one that finite differencing determines least
        # accurately (it is the largest number, obtained as a sum of many small
        # ones), so correcting it is also the statistically sensible choice.
        for k in range(n_prim):
            correction = phi[k, :, :, :].sum(axis=1)
            phi[k, :, k, :] -= correction

    if symmetrize:
        # Phi_{0k,Rk'} = Phi_{0k',-Rk}; within the supercell that is the
        # transpose over the (atom, cartesian) index pair for the atoms of image
        # zero, which is the part we can symmetrise without extra evaluations.
        block = phi[:, :, :n_prim, :]
        phi[:, :, :n_prim, :] = 0.5 * (block + block.transpose(2, 3, 0, 1))

    return ForceConstants(
        values=phi,
        primitive=primitive,
        reps=reps,
        images=images,
        translations=offsets,
        asr_residual=asr_residual,
        meta={"delta": delta, "enforce_asr": enforce_asr, "symmetrize": symmetrize},
    )


def _image_weights(fc: ForceConstants, tolerance: float = 1e-4):
    """Assign each supercell atom to its minimum-image lattice translations.

    Returns a list, one entry per supercell atom, of ``(translations, weight)``
    with ``translations`` an ``(m, 3)`` array of equally-short cartesian lattice
    vectors and ``weight = 1/m``.  Splitting between degenerate images is what
    keeps the Fourier sum symmetric; picking one arbitrarily gives band
    structures that are right at the commensurate q-points and wrong between
    them, which is the hardest kind of error to notice.
    """
    prim, reps = fc.primitive, fc.reps
    supercell_vectors = prim.cell * np.array(reps, dtype=float)[:, None]

    # Search neighbouring supercell images; +/-1 suffices because the atom is
    # already inside the supercell by construction.
    shifts = np.array([
        i * supercell_vectors[0] + j * supercell_vectors[1] + k * supercell_vectors[2]
        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
    ])

    n_prim = prim.n_atoms
    out = []
    for s in range(fc.n_super):
        kappa_prime = s % n_prim
        base = fc.translations[fc.images[s]]
        # Vector from the primitive-cell atom to this supercell atom, for every
        # candidate image of the supercell.
        candidates = base[None, :] + shifts
        tau = prim.positions[kappa_prime] - prim.positions  # (n_prim, 3), per source atom
        # Distance depends on which source atom kappa we came from, so keep the
        # full (n_prim, n_shift) structure.
        d = np.linalg.norm(candidates[None, :, :] + tau[:, None, :], axis=2)
        entry = []
        for kappa in range(n_prim):
            dmin = d[kappa].min()
            sel = np.abs(d[kappa] - dmin) < tolerance
            entry.append((candidates[sel], 1.0 / sel.sum()))
        out.append(entry)
    return out


def dynamical_matrix(fc: ForceConstants, q: np.ndarray, *, weights=None) -> np.ndarray:
    """Dynamical matrix at a single wavevector.

    Parameters
    ----------
    q:
        ``(3,)`` wavevector in **cartesian reciprocal** units of 1/A, i.e.
        including the factor of ``2 pi``.  Use :func:`cubic_q_path` or convert
        fractional coordinates with the reciprocal cell from
        :func:`atomlab.cell.reciprocal_cell`.
    weights:
        Precomputed output of :func:`_image_weights`; pass it when evaluating
        many q-points so the minimum-image search is done once.

    Returns
    -------
    ndarray
        ``(3 n_prim, 3 n_prim)`` complex Hermitian matrix in eV/(A^2 amu).
    """
    q = np.asarray(q, dtype=float).reshape(3)
    if weights is None:
        weights = _image_weights(fc)

    n_prim = fc.n_prim
    masses = fc.masses
    d = np.zeros((n_prim, 3, n_prim, 3), dtype=complex)

    for s in range(fc.n_super):
        kappa_prime = s % n_prim
        for kappa in range(n_prim):
            translations, w = weights[s][kappa]
            phase = w * np.exp(1j * (translations @ q)).sum()
            d[kappa, :, kappa_prime, :] += fc.values[kappa, :, s, :] * phase

    mass_factor = 1.0 / np.sqrt(masses[:, None] * masses[None, :])
    d *= mass_factor[:, None, :, None]
    matrix = d.reshape(3 * n_prim, 3 * n_prim)

    # Hermitise: the imaginary antisymmetric residue is finite-difference noise,
    # and leaving it in would give complex eigenvalues that have no meaning.
    return 0.5 * (matrix + matrix.conj().T)


def phonon_frequencies(fc: ForceConstants, qpoints, *, unit: str = "THz") -> np.ndarray:
    """Phonon frequencies at a set of wavevectors.

    Parameters
    ----------
    qpoints:
        ``(n_q, 3)`` cartesian wavevectors in 1/A.
    unit:
        ``"THz"``, ``"cm-1"``, ``"meV"`` or ``"rad/ps"``.

    Returns
    -------
    ndarray
        ``(n_q, 3 n_prim)`` frequencies, sorted ascending at each q.  **Imaginary
        frequencies are returned as negative numbers**, the standard convention:
        a soft mode is physically meaningful information about an unstable
        structure and must not be hidden by taking an absolute value.
    """
    qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
    weights = _image_weights(fc)

    out = np.empty((len(qpoints), 3 * fc.n_prim))
    for i, q in enumerate(qpoints):
        eigenvalues = np.linalg.eigvalsh(dynamical_matrix(fc, q, weights=weights))
        # eigenvalues are omega^2 in eV/(A^2 amu); E2MVV converts to 1/ps^2.
        omega_sq = eigenvalues * E2MVV
        out[i] = np.sign(omega_sq) * np.sqrt(np.abs(omega_sq))

    if unit == "rad/ps":
        return out
    thz = out * RADPS_TO_THZ
    if unit == "THz":
        return thz
    if unit == "cm-1":
        return thz * THZ_TO_CM1
    if unit == "meV":
        return thz * THZ_TO_MEV
    raise ValueError(f"unknown unit {unit!r}; use THz, cm-1, meV or rad/ps")


def cubic_q_path(lattice_constant: float, *, path: str = "GXWLGK", n_points: int = 60):
    """A standard high-symmetry path through the fcc Brillouin zone.

    Returns ``(qpoints, distances, tick_positions, tick_labels)`` with
    ``qpoints`` in cartesian 1/A including the ``2 pi``, and ``distances`` the
    cumulative path length so bands can be plotted against a single axis.

    The fcc points are given in units of ``2 pi / a``:
    ``G = (0,0,0)``, ``X = (0,1,0)``, ``W = (1/2,1,0)``, ``L = (1/2,1/2,1/2)``,
    ``K = (3/4,3/4,0)``.  These are the conventional-cell coordinates, which is
    what a cubic lattice constant refers to.
    """
    special = {
        "G": np.array([0.0, 0.0, 0.0]),
        "X": np.array([0.0, 1.0, 0.0]),
        "W": np.array([0.5, 1.0, 0.0]),
        "L": np.array([0.5, 0.5, 0.5]),
        "K": np.array([0.75, 0.75, 0.0]),
        "U": np.array([0.25, 1.0, 0.25]),
    }
    unknown = set(path) - set(special)
    if unknown:
        raise ValueError(f"unknown high-symmetry points {sorted(unknown)}")

    scale = 2.0 * np.pi / lattice_constant
    corners = [special[c] * scale for c in path]

    qpoints, distances, ticks = [], [], [0.0]
    total = 0.0
    for start, end in zip(corners[:-1], corners[1:]):
        segment_length = np.linalg.norm(end - start)
        fractions = np.linspace(0.0, 1.0, n_points, endpoint=False)
        for f in fractions:
            qpoints.append(start + f * (end - start))
            distances.append(total + f * segment_length)
        total += segment_length
        ticks.append(total)
    qpoints.append(corners[-1])
    distances.append(total)

    return np.array(qpoints), np.array(distances), np.array(ticks), list(path)


def band_structure(fc: ForceConstants, lattice_constant: float, *, path: str = "GXWLGK",
                   n_points: int = 60, unit: str = "THz") -> dict:
    """Phonon bands along a high-symmetry path, ready to plot."""
    qpoints, distances, ticks, labels = cubic_q_path(lattice_constant, path=path, n_points=n_points)
    frequencies = phonon_frequencies(fc, qpoints, unit=unit)
    return {
        "distances": distances,
        "frequencies": frequencies,
        "tick_positions": ticks,
        "tick_labels": labels,
        "unit": unit,
        "n_imaginary": int((frequencies < -1e-3).sum()),
    }


def phonon_dos(fc: ForceConstants, *, mesh: Sequence[int] = (8, 8, 8), n_bins: int = 120,
               sigma: float | None = None, unit: str = "THz") -> dict:
    """Phonon density of states on a Monkhorst-Pack mesh.

    Gaussian smearing is applied with a default width of two bin widths, which
    is enough to make a coarse mesh look continuous without erasing the van Hove
    singularities that distinguish one model's spectrum from another's.

    Returns a dict with ``frequencies`` (bin centres), ``dos`` normalised so that
    its integral is ``3 n_prim``, and ``n_imaginary`` -- the count of modes with
    negative frequency, which must be reported rather than silently dropped.
    """
    mesh = tuple(int(m) for m in mesh)
    reciprocal = 2.0 * np.pi * np.linalg.inv(fc.primitive.cell).T

    fractional = np.stack(
        np.meshgrid(*[(np.arange(m) + 0.5) / m - 0.5 for m in mesh], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    qpoints = fractional @ reciprocal

    frequencies = phonon_frequencies(fc, qpoints, unit=unit)
    n_imaginary = int((frequencies < -1e-3).sum())

    positive = frequencies[frequencies > -1e-3]
    hi = float(positive.max()) * 1.05 if positive.size else 1.0
    edges = np.linspace(0.0, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    sigma = sigma if sigma is not None else 2.0 * width

    flat = frequencies.ravel()
    flat = flat[flat > -1e-3]
    # Gaussian-smeared histogram, normalised so the integral is 3*n_prim.
    dos = np.exp(-0.5 * ((centres[:, None] - flat[None, :]) / sigma) ** 2).sum(axis=1)
    dos /= (sigma * np.sqrt(2.0 * np.pi)) * len(qpoints)

    return {
        "frequencies": centres,
        "dos": dos,
        "unit": unit,
        "n_imaginary": n_imaginary,
        "mesh": mesh,
        "integral": float(np.trapezoid(dos, centres)),
    }


# --------------------------------------------------------------------------
# Thermodynamics from the harmonic spectrum
# --------------------------------------------------------------------------


def _positive_thz(frequencies_thz: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Drop acoustic zeros and imaginary modes, which have no thermal occupation."""
    f = np.asarray(frequencies_thz, dtype=float).ravel()
    return f[f > floor]


def zero_point_energy(frequencies_thz) -> float:
    """Harmonic zero-point energy ``sum_i h nu_i / 2`` in eV.

    Included because it is the cheapest quantum correction to a classical MD
    result and because, being a pure function of the spectrum, it is a sensitive
    scalar summary of how well a model reproduces curvature.
    """
    return float(0.5 * HPLANCK * _positive_thz(frequencies_thz).sum())


def harmonic_free_energy(frequencies_thz, temperature: float) -> float:
    """Quantum harmonic vibrational free energy in eV.

    ``F = sum_i [ h nu_i / 2 + kT ln(1 - exp(-h nu_i / kT)) ]``.
    """
    nu = _positive_thz(frequencies_thz)
    x = HPLANCK * nu / (KB * temperature)
    return float((0.5 * HPLANCK * nu).sum() + KB * temperature * np.log1p(-np.exp(-x)).sum())


def harmonic_heat_capacity(frequencies_thz, temperature: float) -> float:
    """Quantum harmonic heat capacity ``C_v`` in eV/K.

    Reduces to ``3 N k_B`` (Dulong-Petit) at high temperature, which is the
    obvious check and the one the tests use.
    """
    nu = _positive_thz(frequencies_thz)
    x = HPLANCK * nu / (KB * temperature)
    # Guard the small-x limit, where exp(x)-1 loses precision catastrophically.
    with np.errstate(over="ignore"):
        expx = np.exp(np.clip(x, 0.0, 500.0))
    term = np.where(x < 1e-6, 1.0, x**2 * expx / (expx - 1.0) ** 2)
    return float(KB * term.sum())


def debye_temperature_from_dos(dos_result: dict) -> float:
    """Debye temperature from the second moment of the density of states.

    Uses ``Theta_D = (h/k_B) sqrt(5/3 <nu^2>)``, the moment-matching definition,
    which is well defined for any spectrum rather than only for a genuinely
    Debye-like one.  State which definition is meant whenever this number is
    quoted -- there are several in circulation and they differ by tens of
    percent for real solids.
    """
    nu = np.asarray(dos_result["frequencies"], dtype=float)
    g = np.asarray(dos_result["dos"], dtype=float)
    norm = np.trapezoid(g, nu)
    if norm <= 0:
        raise ValueError("density of states integrates to zero")
    mean_sq = np.trapezoid(g * nu**2, nu) / norm
    return float(HPLANCK / KB * np.sqrt(5.0 / 3.0 * mean_sq))
