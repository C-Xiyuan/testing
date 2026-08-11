"""Velocity initialisation, momentum removal and degrees-of-freedom bookkeeping.

Everything here is in metal units, which means the mass-to-energy conversion
``MVV2E`` appears in every formula that relates a velocity to a temperature::

    KE [eV] = 0.5 * MVV2E * sum_i m_i [amu] * v_i^2 [A^2/ps^2]

so a Maxwell-Boltzmann velocity component has standard deviation
``sqrt(k_B T / (m * MVV2E))`` and **not** ``sqrt(k_B T / m)``.  Dropping the
conversion is the classic metal-units bug: it leaves temperatures wrong by
roughly four orders of magnitude, which is large enough to notice immediately
and therefore, paradoxically, less dangerous than the degrees-of-freedom
mistakes this module also exists to prevent.

Degrees of freedom
------------------

If the total momentum is removed once and the dynamics conserves it (as
Newtonian dynamics with periodic boundary conditions does), the centre-of-mass
velocity stays zero forever and the system has ``3N - 3`` momentum degrees of
freedom, not ``3N``.  Using the wrong count biases every temperature estimate
by a factor ``3N / (3N - 3)``: 1.2% for 256 atoms, 4.9% for 64 atoms.  That is
small enough to survive a casual eyeball check of a thermostat and large enough
to shift a melting point, so :func:`n_dof` is the single source of truth and
every temperature in this package goes through it.
"""

from __future__ import annotations

import numpy as np

from ..units import KB, MVV2E

__all__ = [
    "n_dof",
    "maxwell_boltzmann",
    "com_velocity",
    "remove_com_momentum",
    "kinetic_energy",
    "instantaneous_temperature",
    "rescale_to_temperature",
]


def n_dof(n_atoms: int, fixed_com: bool = True) -> int:
    """Number of momentum degrees of freedom.

    Parameters
    ----------
    n_atoms : int
        Number of atoms ``N``.
    fixed_com : bool
        True if the centre-of-mass momentum has been removed and is conserved
        by the dynamics, in which case three degrees of freedom are lost.

    Returns
    -------
    int
        ``3 N - 3`` if ``fixed_com`` else ``3 N``.

    Raises
    ------
    ValueError
        If the result would be non-positive, which happens for a single atom
        with the centre of mass fixed -- a system with no dynamics at all.
    """
    dof = 3 * int(n_atoms) - (3 if fixed_com else 0)
    if dof <= 0:
        raise ValueError(
            f"n_atoms={n_atoms} with fixed_com={fixed_com} leaves {dof} degrees of freedom"
        )
    return dof


def maxwell_boltzmann(
    masses: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
    *,
    remove_com: bool = True,
    exact_temperature: bool = False,
) -> np.ndarray:
    """Draw velocities from the Maxwell-Boltzmann distribution.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Atomic masses in amu.
    temperature : float
        Target temperature in kelvin.  ``0`` returns zero velocities.
    rng : numpy.random.Generator
        Explicit generator; there is no module-level randomness anywhere in
        this package.
    remove_com : bool
        Subtract the centre-of-mass velocity after drawing.  This changes the
        distribution -- the remaining ``3N - 3`` degrees of freedom are still
        exactly Maxwellian, but the drawn ``3N`` are no longer independent.
    exact_temperature : bool
        If True, rescale so the instantaneous temperature equals ``temperature``
        exactly.  This is convenient for reproducible starts but it puts the
        system on a measure-zero shell of the canonical distribution, so it must
        not be used when generating independent canonical samples.

    Returns
    -------
    ndarray, shape (N, 3)
        Velocities in A/ps.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 1:
        raise ValueError(f"masses must be (N,), got {masses.shape}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0 K, got {temperature}")
    if temperature == 0.0:
        return np.zeros((masses.size, 3))

    # sigma^2 = k_B T / (m * MVV2E): the MVV2E converts amu*A^2/ps^2 to eV.
    sigma = np.sqrt(KB * temperature / (masses * MVV2E))
    velocities = rng.standard_normal((masses.size, 3)) * sigma[:, None]

    if remove_com:
        velocities = remove_com_momentum(velocities, masses)
    if exact_temperature:
        velocities = rescale_to_temperature(
            velocities, masses, temperature, fixed_com=remove_com
        )
    return velocities


def com_velocity(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Centre-of-mass velocity ``sum m v / sum m``, shape ``(3,)`` in A/ps."""
    masses = np.asarray(masses, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    return (masses[:, None] * velocities).sum(axis=0) / masses.sum()


def remove_com_momentum(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return a copy of ``velocities`` with zero total momentum.

    Parameters
    ----------
    velocities : ndarray, shape (N, 3)
        Velocities in A/ps.
    masses : ndarray, shape (N,)
        Masses in amu.

    Returns
    -------
    ndarray, shape (N, 3)
        ``v - V_com`` in A/ps.  The removal lowers the kinetic energy, so the
        temperature drops slightly; that is correct, because the removed
        degrees of freedom are exactly the three that :func:`n_dof` stops
        counting.
    """
    masses = np.asarray(masses, dtype=np.float64)
    velocities = np.array(velocities, dtype=np.float64, copy=True)
    velocities -= com_velocity(velocities, masses)
    return velocities


def kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> float:
    """Kinetic energy in eV: ``0.5 * MVV2E * sum m v^2``.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Masses in amu.
    velocities : ndarray, shape (N, 3)
        Velocities in A/ps.
    """
    masses = np.asarray(masses, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    return float(0.5 * MVV2E * np.einsum("i,ik,ik->", masses, velocities, velocities))


def instantaneous_temperature(
    masses: np.ndarray,
    velocities: np.ndarray,
    *,
    fixed_com: bool = True,
    dof: int | None = None,
) -> float:
    """Instantaneous temperature in K from equipartition ``2 KE = dof k_B T``.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Masses in amu.
    velocities : ndarray, shape (N, 3)
        Velocities in A/ps.
    fixed_com : bool
        Passed to :func:`n_dof` when ``dof`` is not given explicitly.
    dof : int, optional
        Override the degree-of-freedom count (e.g. when extra constraints are
        active).
    """
    masses = np.asarray(masses, dtype=np.float64)
    ndof = n_dof(masses.size, fixed_com) if dof is None else int(dof)
    return 2.0 * kinetic_energy(masses, velocities) / (ndof * KB)


def rescale_to_temperature(
    velocities: np.ndarray,
    masses: np.ndarray,
    temperature: float,
    *,
    fixed_com: bool = True,
    dof: int | None = None,
) -> np.ndarray:
    """Uniformly scale velocities so the instantaneous temperature is exact.

    Parameters
    ----------
    velocities : ndarray, shape (N, 3)
        Velocities in A/ps.
    masses : ndarray, shape (N,)
        Masses in amu.
    temperature : float
        Target temperature in K.
    fixed_com, dof
        Degrees-of-freedom bookkeeping, see :func:`instantaneous_temperature`.

    Returns
    -------
    ndarray, shape (N, 3)
        Rescaled copy.

    Notes
    -----
    Rescaling is *not* a thermostat.  It fixes the first moment of the kinetic
    energy and does nothing about the distribution, which is the point of the
    Berendsen negative control in :mod:`atomlab.md.integrators`.  Use it only to
    prepare an initial state.
    """
    masses = np.asarray(masses, dtype=np.float64)
    current = instantaneous_temperature(masses, velocities, fixed_com=fixed_com, dof=dof)
    if current <= 0.0:
        raise ValueError("cannot rescale velocities with zero kinetic energy")
    return np.asarray(velocities, dtype=np.float64) * np.sqrt(temperature / current)
