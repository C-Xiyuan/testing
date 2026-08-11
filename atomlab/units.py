"""Unit system for :mod:`atomlab`.

We use the LAMMPS ``metal`` unit system throughout the entire codebase.  Every
public function takes and returns quantities in these units unless its
docstring explicitly says otherwise.

===============  =========================================
Quantity         Unit
===============  =========================================
length           angstrom (A)
energy           electronvolt (eV)
mass             atomic mass unit (amu, i.e. g/mol)
time             picosecond (ps)
force            eV / A
velocity         A / ps
temperature      kelvin (K)
pressure         bar
stress           eV / A^3  (internal); ``bar`` at the API surface
===============  =========================================

The only non-obvious consequence of this choice is that Newton's second law
picks up a conversion factor, because ``amu * A / ps^2`` is not ``eV / A``::

    a [A/ps^2] = F [eV/A] / (m [amu] * MVV2E)

and kinetic energy is::

    KE [eV] = 0.5 * MVV2E * sum(m [amu] * v^2 [A^2/ps^2])

This mirrors LAMMPS exactly, which lets us cross-check against published
LAMMPS/ASE numbers without any unit gymnastics.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Fundamental constants (CODATA 2018), expressed in metal units
# --------------------------------------------------------------------------

#: Boltzmann constant, eV / K.
KB = 8.617333262e-5

#: Converts ``amu * A^2 / ps^2`` to ``eV``.  Equal to ``1e4 / (e * N_A)`` with
#: ``e`` in coulomb and ``N_A`` in 1/mol.  This is LAMMPS' ``mvv2e`` for metal
#: units.
MVV2E = 1.0364269e-4

#: Inverse of :data:`MVV2E`; converts ``eV`` to ``amu * A^2 / ps^2``.
E2MVV = 1.0 / MVV2E

#: Converts ``eV / A^3`` to ``bar``.  1 eV/A^3 = 1.602176634e11 Pa = 1.602176634e6 bar.
EV_A3_TO_BAR = 1.602176634e6

#: Converts ``bar`` to ``eV / A^3``.
BAR_TO_EV_A3 = 1.0 / EV_A3_TO_BAR

#: Converts ``eV / A^3`` to ``GPa``.
EV_A3_TO_GPA = 160.2176634

#: Converts ``GPa`` to ``eV / A^3``.
GPA_TO_EV_A3 = 1.0 / EV_A3_TO_GPA

#: Planck constant, eV * ps.
HPLANCK = 4.135667696e-3

#: Reduced Planck constant, eV * ps.
HBAR = HPLANCK / (2.0 * math.pi)

#: Speed of light, A / ps.
CLIGHT = 2.99792458e6

#: Avogadro constant, 1 / mol.
NA = 6.02214076e23

# --------------------------------------------------------------------------
# Spectroscopic conversions (used by the vibrational density of states)
# --------------------------------------------------------------------------

#: Converts an angular frequency in ``rad/ps`` to ``THz``.
RADPS_TO_THZ = 1.0 / (2.0 * math.pi)

#: Converts ``THz`` to wavenumbers ``cm^-1``.
THZ_TO_CM1 = 1e10 / CLIGHT  # 1 THz = 1e12 Hz; c in cm/s = CLIGHT*1e-8*1e12

#: Converts ``THz`` to ``meV``.
THZ_TO_MEV = HPLANCK * 1e3

# --------------------------------------------------------------------------
# Element data used by the reference potentials
# --------------------------------------------------------------------------

#: Atomic masses in amu for the (few) elements this project simulates.
ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.002602,
    "Ne": 20.1797,
    "Ar": 39.948,
    "Kr": 83.798,
    "Xe": 131.293,
    "Si": 28.0855,
    "Ge": 72.630,
    "C": 12.011,
    "Cu": 63.546,
    "Ni": 58.6934,
    "Al": 26.9815385,
    "Au": 196.966569,
    "Ag": 107.8682,
}


def beta(temperature: float) -> float:
    """Return the inverse temperature ``1 / (k_B T)`` in ``1/eV``.

    Parameters
    ----------
    temperature:
        Temperature in kelvin.  Must be strictly positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature}")
    return 1.0 / (KB * temperature)


def temperature_from_kinetic(kinetic_energy: float, n_dof: int) -> float:
    """Instantaneous temperature from kinetic energy via equipartition.

    Parameters
    ----------
    kinetic_energy:
        Total kinetic energy in eV.
    n_dof:
        Number of momentum degrees of freedom actually sampled.  For an
        ``N``-atom periodic system with the centre of mass fixed this is
        ``3 * N - 3``.
    """
    if n_dof <= 0:
        raise ValueError(f"n_dof must be > 0, got {n_dof}")
    return 2.0 * kinetic_energy / (n_dof * KB)
