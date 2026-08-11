"""Analytic embedded-atom model for an fcc metal (parameterised for copper).

Why an *analytic* EAM
---------------------

Production EAM potentials tabulate ``F``, ``rho`` and ``phi`` on a grid and
interpolate with cubic splines.  That is fine for simulation but poor for this
project: the spline knots put small discontinuities in the third derivative,
which contaminate finite-difference force constants and elastic constants at
exactly the level of precision we need to resolve model errors.  Everything
here is therefore closed-form, so energies, forces, virials and Hessians are
exact to machine precision.

Functional form
---------------

.. math::

    U = \\sum_i F(n_i) + \\tfrac12 \\sum_{i \\ne j} \\phi(r_{ij}),
    \\qquad n_i = \\sum_{j \\ne i} \\rho(r_{ij})

with, writing :math:`f_c(r) = \\exp[-h/(r_c - r)]` for :math:`r < r_c` and
:math:`0` otherwise,

.. math::

    \\rho(r) = e^{-\\chi (r/r_e - 1)} f_c(r), \\qquad
    \\phi(r) = D\\, e^{-2\\alpha (r/r_e - 1)} f_c(r), \\qquad
    F(n) = -A\\sqrt{n} + C n^2 .

The pair term is purely repulsive: in the EAM picture cohesion comes from the
embedding energy, and :math:`\\phi` is the core repulsion that stops the
lattice collapsing.  The :math:`-A\\sqrt{n}` embedding is the second-moment
tight-binding form; the :math:`+C n^2` correction is what lets one fit the
bulk modulus independently of the cohesive energy.

:math:`f_c` vanishes together with all of its derivatives at :math:`r_c` (the
same device the Stillinger-Weber potential uses), so the potential is
:math:`C^\\infty` at the cutoff with no switching function and no energy shift.
As with SW, the expression *diverges* for :math:`r > r_c`, so every evaluation
is guarded by a strict ``r < rc`` test.

Why this is a many-body potential
---------------------------------

:math:`F` is a nonlinear function of a *sum* over neighbours.  Expanding
:math:`-A\\sqrt{\\sum_j \\rho_j}` generates couplings among all neighbours of an
atom at every order, so no pair potential reproduces it -- yet each input to
:math:`F` is a plain pairwise quantity, which is what keeps the forces cheap
and the two-pass algorithm below possible.  Physically it is the reason EAM
metals have the correct (negative) Cauchy pressure and correct surface
relaxations, which pair potentials get qualitatively wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from ..cell import check_minimum_image
from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Result
from .base import Potential

__all__ = [
    "EAMParameters",
    "EAM_COPPER",
    "EAM",
]


@dataclass(frozen=True)
class EAMParameters:
    """Analytic EAM parameters for a single species.

    Attributes
    ----------
    r_e : float
        Reference length in angstrom, taken as the nearest-neighbour distance
        of the equilibrium fcc lattice.  Only sets where the exponentials are
        normalised; it is not fitted.
    alpha : float
        Dimensionless decay of the pair repulsion (``phi ~ exp(-2 alpha r/r_e)``).
    chi : float
        Dimensionless decay of the electron density.
    D : float
        Pair prefactor in eV, i.e. ``phi(r_e)`` before the cutoff factor.
    A : float
        Embedding prefactor in eV (``F = -A sqrt(n) + C n^2``).
    C : float
        Quadratic embedding coefficient in eV (``n`` is dimensionless).
    cutoff : float
        Interaction range in angstrom.
    h : float
        Width in angstrom of the smooth cutoff ``exp(-h/(rc - r))``.  Larger
        ``h`` pulls the taper further inside ``cutoff``.
    """

    r_e: float
    alpha: float
    chi: float
    D: float
    A: float
    C: float
    cutoff: float
    h: float


#: fcc copper.  ``alpha``, ``chi``, ``cutoff`` and ``h`` were fixed by hand at
#: physically sensible values; ``D``, ``A`` and ``C`` were then solved (by
#: least squares on the fcc lattice sum, see ``tests/test_manybody_potentials.py``
#: for the verification) to reproduce
#:
#: ===========================  ==========  ==========
#: quantity                     target      achieved
#: ===========================  ==========  ==========
#: lattice constant a0 (A)      3.615       3.6150
#: cohesive energy (eV/atom)    -3.54       -3.5400
#: bulk modulus (GPa)           140         140.00
#: ===========================  ==========  ==========
#:
#: The model is not fitted to anything else, so its elastic anisotropy,
#: vacancy energy and phonon spectrum are predictions rather than inputs --
#: which is what we want from a "ground truth" in this study.  For the record
#: it also puts fcc below bcc (by 0.020 eV/atom) and below simple cubic (by
#: 0.194 eV/atom), so the reference structure is the ground state.
EAM_COPPER = EAMParameters(
    r_e=2.556,
    alpha=6.0,
    chi=4.0,
    D=0.0635652317,
    A=1.4201475775,
    C=0.0069687821,
    cutoff=5.0,
    h=0.6,
)


# --------------------------------------------------------------------------
# jitted kernel
# --------------------------------------------------------------------------


@njit(cache=True)
def _eam_kernel(first, pair_j, D, r, n_atoms, r_e, alpha, chi, dpair, aemb, cemb, rc, h):
    """Two-pass analytic EAM over a *full* neighbour list.

    Pass 1 accumulates the host electron density ``n_i``; the embedding
    derivative ``F'(n_i)`` is then known for every atom, and pass 2 turns it
    into forces.  The two passes are unavoidable: the force on atom ``i``
    depends on ``F'`` evaluated at *its neighbours'* densities, which are not
    known until every pair has been visited once.

    Returns ``(energy, forces, virial, energies, density)``.
    """
    dens = np.zeros(n_atoms)
    pair_energy = np.zeros(n_atoms)
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))

    # ---- pass 1: densities and the (pairwise) pair energy -----------------
    # Each unordered bond appears twice in a full list, so half of phi lands
    # on each partner and the totals come out right without a half list.
    for i in range(n_atoms):
        for p in range(first[i], first[i + 1]):
            rij = r[p]
            if rij >= rc:
                continue
            fc = np.exp(-h / (rc - rij))
            u = np.exp(-chi * (rij / r_e - 1.0))
            v = np.exp(-2.0 * alpha * (rij / r_e - 1.0))
            dens[i] += u * fc
            pair_energy[i] += 0.5 * dpair * v * fc

    # ---- embedding energy and its derivative ------------------------------
    embed = np.zeros(n_atoms)
    dembed = np.zeros(n_atoms)
    energies = np.zeros(n_atoms)
    total = 0.0
    for i in range(n_atoms):
        n_i = dens[i]
        if n_i > 1e-300:
            s = np.sqrt(n_i)
            embed[i] = -aemb * s + cemb * n_i * n_i
            dembed[i] = -0.5 * aemb / s + 2.0 * cemb * n_i
        # An isolated atom has n = 0; F(0) = 0 and F' is formally infinite
        # there, but it multiplies rho' summed over an empty neighbour set, so
        # leaving both at zero is exact rather than a regularisation.
        energies[i] = embed[i] + pair_energy[i]
        total += energies[i]

    # ---- pass 2: forces and virial ----------------------------------------
    for i in range(n_atoms):
        for p in range(first[i], first[i + 1]):
            rij = r[p]
            if rij >= rc:
                continue
            j = pair_j[p]
            gap = rc - rij
            fc = np.exp(-h / gap)
            dfc = -h / (gap * gap) * fc
            u = np.exp(-chi * (rij / r_e - 1.0))
            v = np.exp(-2.0 * alpha * (rij / r_e - 1.0))
            drho = u * (-chi / r_e) * fc + u * dfc
            dphi = dpair * (v * (-2.0 * alpha / r_e) * fc + v * dfc)
            # mu = [F'(n_i) + F'(n_j)] rho'(r) + phi'(r); the two embedding
            # derivatives are what make this many-body.
            mu = (dembed[i] + dembed[j]) * drho + dphi
            c = mu / rij
            for t in range(3):
                forces[i, t] += c * D[p, t]
            # Half weight: the mirror pair contributes the identical dyad, so
            # summing mu/2 over the full list gives sum_{i<j} f_ij (x) r_ij.
            for s in range(3):
                for t in range(3):
                    virial[s, t] -= 0.5 * c * D[p, s] * D[p, t]

    return total, forces, virial, energies, dens


# --------------------------------------------------------------------------
# pure-numpy reference
# --------------------------------------------------------------------------


def _eam_numpy(first, pair_j, D, r, n_atoms, r_e, alpha, chi, dpair, aemb, cemb, rc, h):
    """Vectorised NumPy reference for :func:`_eam_kernel`; same signature."""
    counts = np.diff(first)
    pair_i = np.repeat(np.arange(n_atoms), counts)

    inside = r < rc
    idx = np.flatnonzero(inside)
    dens = np.zeros(n_atoms)
    pair_energy = np.zeros(n_atoms)
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))

    if idx.size:
        rr = r[idx]
        ii = pair_i[idx]
        jj = pair_j[idx]
        gap = rc - rr
        fc = np.exp(-h / gap)
        dfc = -h / gap**2 * fc
        u = np.exp(-chi * (rr / r_e - 1.0))
        v = np.exp(-2.0 * alpha * (rr / r_e - 1.0))
        np.add.at(dens, ii, u * fc)
        np.add.at(pair_energy, ii, 0.5 * dpair * v * fc)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(dens)
        embed = np.where(dens > 0.0, -aemb * s + cemb * dens**2, 0.0)
        dembed = np.where(dens > 0.0, -0.5 * aemb / np.where(s > 0, s, 1.0) + 2.0 * cemb * dens, 0.0)

    energies = embed + pair_energy
    total = float(energies.sum())

    if idx.size:
        drho = u * (-chi / r_e) * fc + u * dfc
        dphi = dpair * (v * (-2.0 * alpha / r_e) * fc + v * dfc)
        mu = (dembed[ii] + dembed[jj]) * drho + dphi
        c = mu / rr
        dd = D[idx]
        np.add.at(forces, ii, c[:, None] * dd)
        virial -= 0.5 * np.einsum("p,pa,pb->ab", c, dd, dd)

    return total, forces, virial, energies, dens


# --------------------------------------------------------------------------
# the potential
# --------------------------------------------------------------------------


class EAM(Potential):
    """Analytic embedded-atom model with exact forces and virial.

    Parameters
    ----------
    parameters : EAMParameters, optional
        Defaults to :data:`EAM_COPPER`.
    implementation : {"numba", "numpy"}
        Which kernel to run; both are tested against each other.
    name : str
        Short identifier for tables and figures.
    strict_minimum_image : bool
        Raise when ``2 * cutoff >= min_cell_width`` (package contract).

    Notes
    -----
    Metal units throughout: A, eV, eV/A, eV.
    """

    def __init__(
        self,
        parameters: EAMParameters = EAM_COPPER,
        *,
        implementation: str = "numba",
        name: str = "EAM",
        strict_minimum_image: bool = True,
    ) -> None:
        if implementation not in ("numba", "numpy"):
            raise ValueError(
                f"implementation must be 'numba' or 'numpy', got {implementation!r}"
            )
        self.parameters = parameters
        self.implementation = implementation
        self.cutoff = float(parameters.cutoff)
        self.name = name
        self.strict_minimum_image = bool(strict_minimum_image)

    @classmethod
    def copper(cls, **kwargs) -> "EAM":
        """The fcc copper parameterisation (:data:`EAM_COPPER`)."""
        return cls(EAM_COPPER, name=kwargs.pop("name", "EAM/Cu"), **kwargs)

    # -- the interface -----------------------------------------------------

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Evaluate the potential on ``configuration``.

        Parameters
        ----------
        configuration : Configuration
            Positions ``(N, 3)`` in angstrom.
        forces, virial : bool
            Both are always computed; the second neighbour pass is the same
            cost as the density pass, so skipping it would not pay.

        Returns
        -------
        Result
            ``energy`` (eV), ``forces`` ``(N, 3)`` eV/A, ``virial`` ``(3, 3)``
            eV, ``energies`` ``(N,)`` eV (embedding energy of the atom plus
            half of each of its bonds), and ``extra["density"]`` the ``(N,)``
            dimensionless host electron densities ``n_i``.
        """
        if self.strict_minimum_image and bool(np.asarray(configuration.pbc).any()):
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

        n = configuration.n_atoms
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        D, r = pair_vectors(configuration, nl)
        first = np.searchsorted(nl.i, np.arange(n + 1)).astype(np.int64)

        prm = self.parameters
        kernel = _eam_kernel if self.implementation == "numba" else _eam_numpy
        energy, f, w, e_at, dens = kernel(
            first,
            np.ascontiguousarray(nl.j.astype(np.int64)),
            np.ascontiguousarray(D),
            np.ascontiguousarray(r),
            n,
            prm.r_e,
            prm.alpha,
            prm.chi,
            prm.D,
            prm.A,
            prm.C,
            self.cutoff,
            prm.h,
        )

        return Result(
            energy=energy,
            forces=f,
            virial=w if virial else None,
            energies=e_at,
            extra={"density": dens},
        )

    # -- analytic pieces ---------------------------------------------------

    def cutoff_function(self, r: np.ndarray) -> np.ndarray:
        """``f_c(r)``, dimensionless; zero (with all derivatives) at the cutoff."""
        rr = np.asarray(r, dtype=float)
        out = np.zeros_like(rr)
        m = rr < self.cutoff
        out[m] = np.exp(-self.parameters.h / (self.cutoff - rr[m]))
        return out

    def electron_density(self, r: np.ndarray) -> np.ndarray:
        """``rho(r)``, dimensionless, for distances ``r`` in angstrom."""
        prm = self.parameters
        rr = np.asarray(r, dtype=float)
        return np.exp(-prm.chi * (rr / prm.r_e - 1.0)) * self.cutoff_function(rr)

    def pair_energy(self, r: np.ndarray) -> np.ndarray:
        """``phi(r)`` in eV for distances ``r`` in angstrom."""
        prm = self.parameters
        rr = np.asarray(r, dtype=float)
        return prm.D * np.exp(-2.0 * prm.alpha * (rr / prm.r_e - 1.0)) * self.cutoff_function(rr)

    def embedding_energy(self, n: np.ndarray) -> np.ndarray:
        """``F(n) = -A sqrt(n) + C n^2`` in eV for dimensionless densities ``n``."""
        prm = self.parameters
        nn = np.asarray(n, dtype=float)
        return -prm.A * np.sqrt(np.clip(nn, 0.0, None)) + prm.C * nn**2

    def density(self, configuration: Configuration) -> np.ndarray:
        """``(N,)`` host electron densities ``n_i`` (dimensionless).

        Exposed because the embedding argument is the natural coordinate for
        judging whether a configuration is an extrapolation for a metal: a
        surface atom sits at roughly half the bulk density.
        """
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        _, r = pair_vectors(configuration, nl)
        out = np.zeros(configuration.n_atoms)
        np.add.at(out, nl.i, self.electron_density(r))
        return out
