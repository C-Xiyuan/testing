"""Stillinger-Weber potential (silicon, original 1985 parameters).

The Stillinger-Weber form is the smallest interatomic potential that is
*genuinely* many-body: its three-body term penalises deviations of the bond
angle from the tetrahedral value, and no function of pair distances alone can
reproduce it.  That is exactly why it is a reference potential in this project.
A learned pair-only surrogate (``atomlab/models/pair_spline.py``) is *exactly*
right for Lennard-Jones and *structurally* wrong for Stillinger-Weber, which
gives us a controlled instance of the error mode the whole study is about.

Functional form
---------------

With :math:`r_c = a \\sigma` the cutoff,

.. math::

    U = \\sum_{i<j} \\phi_2(r_{ij})
      + \\sum_i \\sum_{j<k} \\phi_3(r_{ij}, r_{ik}, \\theta_{jik})

.. math::

    \\phi_2(r) = A \\epsilon
        \\left[ B (\\sigma/r)^p - (\\sigma/r)^q \\right]
        \\exp\\!\\left( \\frac{\\sigma}{r - a\\sigma} \\right)

.. math::

    \\phi_3 = \\lambda \\epsilon (\\cos\\theta_{jik} - \\cos\\theta_0)^2
        \\exp\\!\\left( \\frac{\\gamma\\sigma}{r_{ij} - a\\sigma} \\right)
        \\exp\\!\\left( \\frac{\\gamma\\sigma}{r_{ik} - a\\sigma} \\right)

The angle :math:`\\theta_{jik}` is subtended at the *central* atom ``i``.

Why the cutoff needs no switching function
------------------------------------------

As :math:`r \\to (a\\sigma)^-` the argument of every exponential tends to
:math:`-\\infty`, so the factor and *all* of its derivatives vanish there: the
potential is :math:`C^\\infty` at the cutoff without any taper.  This is the
single most elegant feature of the SW form and it is why we may truncate at
``r >= a*sigma`` with a bare ``if``.  The corresponding trap is that the same
expression *diverges* for :math:`r > a\\sigma` (the denominator changes sign and
the exponential overflows), so every evaluation below is guarded by a strict
``r < rc`` test rather than by an ``r <= rc`` test or by clipping.

Three-body virial
-----------------

The virial is accumulated as :math:`W_{ab} = -\\sum G_a D_b` summed over the
*two bond vectors* of each triplet, where :math:`G = \\partial\\phi_3/\\partial D`
and :math:`D` is the (minimum-image / shifted) bond vector.  The tempting
alternative :math:`\\sum_i r_i \\otimes f_i` is wrong under periodic boundary
conditions, because absolute positions are only defined modulo a lattice
vector while bond vectors are not.  The two agree only for an isolated cluster.
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
    "SWParameters",
    "SW_SILICON_1985",
    "StillingerWeber",
]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SWParameters:
    """Stillinger-Weber parameter set for a single species.

    Attributes
    ----------
    epsilon : float
        Energy scale in eV.
    sigma : float
        Length scale in angstrom.
    a : float
        Dimensionless cutoff, in units of ``sigma``.  The interaction range is
        ``a * sigma``.
    lam : float
        Three-body strength ``lambda`` (dimensionless; multiplies ``epsilon``).
    gamma : float
        Dimensionless range parameter of the three-body radial factor.
    A, B : float
        Dimensionless two-body prefactors.
    p, q : float
        Two-body repulsive/attractive exponents.  ``q = 0`` in the original
        silicon set, which makes the "attractive" term a constant ``-1`` inside
        the bracket; the exponential envelope alone then produces the well.
    cos_theta0 : float
        Cosine of the preferred bond angle.  ``-1/3`` is the ideal tetrahedral
        angle 109.47 deg.
    """

    epsilon: float
    sigma: float
    a: float
    lam: float
    gamma: float
    A: float
    B: float
    p: float
    q: float
    cos_theta0: float

    @property
    def cutoff(self) -> float:
        """Interaction range ``a * sigma`` in angstrom."""
        return self.a * self.sigma


#: The original silicon parameters of Stillinger & Weber, Phys. Rev. B **31**,
#: 5262 (1985), in metal units.  ``epsilon = 2.1683 eV`` and
#: ``sigma = 2.0951 A`` are the values distributed with LAMMPS as ``Si.sw``.
#: ``A`` and ``B`` are chosen so that ``phi_2`` has its minimum value exactly
#: ``-epsilon`` at ``r = 2^(1/6) sigma``, which is what makes the diamond
#: lattice energy come out at exactly ``-2 epsilon`` per atom (see
#: :meth:`StillingerWeber.diamond_reference_energy`).
SW_SILICON_1985 = SWParameters(
    epsilon=2.1683,
    sigma=2.0951,
    a=1.80,
    lam=21.0,
    gamma=1.20,
    A=7.049556277,
    B=0.6022245584,
    p=4.0,
    q=0.0,
    cos_theta0=-1.0 / 3.0,
)


# --------------------------------------------------------------------------
# jitted kernel
# --------------------------------------------------------------------------


@njit(cache=True)
def _sw_kernel(first, pair_j, D, r, n_atoms, epsilon, sigma, a, lam, gamma, A, B, p, q, cos0):
    """Scalar Stillinger-Weber kernel over a *full* neighbour list.

    ``first`` is a CSR-style offset array of length ``n_atoms + 1`` into the
    pair arrays, which must be sorted by central atom (the neighbour list is
    returned in exactly that canonical order).

    Returns ``(e2, e3, forces, virial, energies)``.
    """
    rc = a * sigma
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))
    energies = np.zeros(n_atoms)
    e2_total = 0.0
    e3_total = 0.0
    a_eps = A * epsilon
    lam_eps = lam * epsilon

    for i in range(n_atoms):
        lo = first[i]
        hi = first[i + 1]

        # ---- two-body ----------------------------------------------------
        # Every unordered bond appears twice in a full list, so each visit
        # carries *half* the bond energy (and half the virial dyad, which is
        # symmetric under the exchange and so is counted twice).  The force is
        # the exception: atom i feels the bond both as the pair's centre and,
        # through the mirror entry, as its neighbour, and those two
        # contributions are equal -- hence the full dphi/r here against the
        # 0.5 elsewhere.  Self-image pairs (i == j) then cancel exactly, as
        # they must.
        for m in range(lo, hi):
            rij = r[m]
            if rij >= rc:
                continue
            x = sigma / rij
            xp = x**p
            xq = x**q
            dr = rij - rc  # strictly negative, so the exponential decays
            env = np.exp(sigma / dr)
            bracket = B * xp - xq
            phi = a_eps * bracket * env
            # d/dr [ (B x^p - x^q) exp(sigma/(r-rc)) ]
            dphi = a_eps * env * (
                (q * xq - p * B * xp) / rij - bracket * sigma / (dr * dr)
            )
            e2_total += 0.5 * phi
            energies[i] += 0.5 * phi
            c = dphi / rij
            for t in range(3):
                forces[i, t] += c * D[m, t]
            for s in range(3):
                for t in range(3):
                    virial[s, t] -= 0.5 * c * D[m, s] * D[m, t]

        # ---- three-body --------------------------------------------------
        # Unordered neighbour pairs (m < n) of the central atom i.
        for m in range(lo, hi):
            rij = r[m]
            if rij >= rc:
                continue
            j = pair_j[m]
            dij = rij - rc
            eij = np.exp(gamma * sigma / dij)
            # d/dr_ij of the radial factor, divided by the factor itself
            lij = -gamma * sigma / (dij * dij)

            for n in range(m + 1, hi):
                rik = r[n]
                if rik >= rc:
                    continue
                k = pair_j[n]
                dik = rik - rc
                eik = np.exp(gamma * sigma / dik)
                lik = -gamma * sigma / (dik * dik)

                dot = D[m, 0] * D[n, 0] + D[m, 1] * D[n, 1] + D[m, 2] * D[n, 2]
                cos_t = dot / (rij * rik)
                delta = cos_t - cos0

                pref = lam_eps * eij * eik
                h = pref * delta * delta
                e3_total += h
                energies[i] += h

                dh_drij = h * lij
                dh_drik = h * lik
                dh_dcos = 2.0 * pref * delta

                for t in range(3):
                    uij = D[m, t] / rij
                    uik = D[n, t] / rik
                    # G = dh/dD ; the angular part is the component of the
                    # other unit vector perpendicular to this one, over r.
                    gij = dh_drij * uij + dh_dcos * (uik - cos_t * uij) / rij
                    gik = dh_drik * uik + dh_dcos * (uij - cos_t * uik) / rik
                    # h depends on positions only through D_ij = r_j - r_i and
                    # D_ik = r_k - r_i, hence these three force terms.
                    forces[j, t] -= gij
                    forces[k, t] -= gik
                    forces[i, t] += gij + gik
                    for s in range(3):
                        virial[t, s] -= gij * D[m, s] + gik * D[n, s]

    return e2_total, e3_total, forces, virial, energies


# --------------------------------------------------------------------------
# pure-numpy reference
# --------------------------------------------------------------------------


def _sw_numpy(first, pair_j, D, r, n_atoms, epsilon, sigma, a, lam, gamma, A, B, p, q, cos0):
    """Vectorised NumPy reference for :func:`_sw_kernel`.

    Deliberately written with a different loop structure (array expressions
    over all pairs, and over all neighbour *combinations* of one atom at a
    time) so that a mistake in the jitted kernel is unlikely to be mirrored
    here.  Same return signature.
    """
    rc = a * sigma
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))
    energies = np.zeros(n_atoms)

    inside = r < rc
    idx = np.flatnonzero(inside)
    e2_total = 0.0

    if idx.size:
        rr = r[idx]
        dd = D[idx]
        ii = np.repeat(np.arange(n_atoms), np.diff(first))[idx]
        x = sigma / rr
        xp = x**p
        xq = x**q
        dr = rr - rc
        env = np.exp(sigma / dr)
        bracket = B * xp - xq
        phi = A * epsilon * bracket * env
        dphi = A * epsilon * env * ((q * xq - p * B * xp) / rr - bracket * sigma / dr**2)

        e2_total = 0.5 * float(phi.sum())
        np.add.at(energies, ii, 0.5 * phi)
        c = dphi / rr
        np.add.at(forces, ii, c[:, None] * dd)
        virial -= 0.5 * np.einsum("p,pa,pb->ab", c, dd, dd)

    e3_total = 0.0
    for i in range(n_atoms):
        lo, hi = int(first[i]), int(first[i + 1])
        local = np.flatnonzero(inside[lo:hi]) + lo
        if local.size < 2:
            continue
        ia, ib = np.triu_indices(local.size, k=1)
        m = local[ia]
        n = local[ib]
        dij, dik = D[m], D[n]
        rij, rik = r[m], r[n]

        eij = np.exp(gamma * sigma / (rij - rc))
        eik = np.exp(gamma * sigma / (rik - rc))
        lij = -gamma * sigma / (rij - rc) ** 2
        lik = -gamma * sigma / (rik - rc) ** 2

        cos_t = np.einsum("pa,pa->p", dij, dik) / (rij * rik)
        delta = cos_t - cos0
        pref = lam * epsilon * eij * eik
        h = pref * delta * delta

        e3_total += float(h.sum())
        energies[i] += float(h.sum())

        uij = dij / rij[:, None]
        uik = dik / rik[:, None]
        dh_dcos = 2.0 * pref * delta
        gij = (h * lij)[:, None] * uij + (dh_dcos / rij)[:, None] * (uik - cos_t[:, None] * uij)
        gik = (h * lik)[:, None] * uik + (dh_dcos / rik)[:, None] * (uij - cos_t[:, None] * uik)

        np.add.at(forces, pair_j[m], -gij)
        np.add.at(forces, pair_j[n], -gik)
        forces[i] += (gij + gik).sum(axis=0)
        virial -= np.einsum("pa,pb->ab", gij, dij) + np.einsum("pa,pb->ab", gik, dik)

    return e2_total, e3_total, forces, virial, energies


# --------------------------------------------------------------------------
# the potential
# --------------------------------------------------------------------------


class StillingerWeber(Potential):
    """Stillinger-Weber potential with analytic forces and virial.

    Parameters
    ----------
    parameters : SWParameters, optional
        Parameter set; defaults to :data:`SW_SILICON_1985`.
    implementation : {"numba", "numpy"}
        Which kernel to run.  ``"numpy"`` is the reference implementation and
        is a few times slower; both are tested against each other.
    name : str, optional
        Short identifier used in tables and figures.
    strict_minimum_image : bool
        If True (the default) raise when ``2 * cutoff >= min_cell_width``, per
        the package-wide contract.  The neighbour list itself is correct for
        arbitrarily small cells, so this is a guard against the *caller*
        silently comparing a small-cell result against a minimum-image one,
        not a limitation of the algorithm.

    Notes
    -----
    All quantities are in metal units: positions in A, energy in eV, forces in
    eV/A, virial in eV.
    """

    def __init__(
        self,
        parameters: SWParameters = SW_SILICON_1985,
        *,
        implementation: str = "numba",
        name: str = "SW",
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

    # -- presets -----------------------------------------------------------

    @classmethod
    def silicon(cls, **kwargs) -> "StillingerWeber":
        """The original 1985 silicon parameterisation."""
        return cls(SW_SILICON_1985, name=kwargs.pop("name", "SW/Si"), **kwargs)

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
            Ignored beyond the return contract: both are always computed,
            because the three-body pass costs the same either way.

        Returns
        -------
        Result
            ``energy`` (eV), ``forces`` ``(N, 3)`` eV/A, ``virial`` ``(3, 3)``
            eV, ``energies`` ``(N,)`` eV (two-body bonds split evenly between
            partners, three-body triplets assigned to the central atom), and
            ``extra`` carrying the ``"energy_two_body"`` / ``"energy_three_body"``
            decomposition in eV.
        """
        if self.strict_minimum_image and bool(np.asarray(configuration.pbc).any()):
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

        n = configuration.n_atoms
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        D, r = pair_vectors(configuration, nl)
        # The list is sorted by (i, j, shift), so a searchsorted gives CSR
        # offsets straight away -- no grouping pass needed.
        first = np.searchsorted(nl.i, np.arange(n + 1)).astype(np.int64)

        prm = self.parameters
        kernel = _sw_kernel if self.implementation == "numba" else _sw_numpy
        e2, e3, f, w, e_at = kernel(
            first,
            np.ascontiguousarray(nl.j.astype(np.int64)),
            np.ascontiguousarray(D),
            np.ascontiguousarray(r),
            n,
            prm.epsilon,
            prm.sigma,
            prm.a,
            prm.lam,
            prm.gamma,
            prm.A,
            prm.B,
            float(prm.p),
            float(prm.q),
            prm.cos_theta0,
        )

        return Result(
            energy=e2 + e3,
            forces=f,
            virial=w if virial else None,
            energies=e_at,
            extra={"energy_two_body": float(e2), "energy_three_body": float(e3)},
        )

    # -- analytic pieces, exposed for tests and for documentation ----------

    def two_body(self, r: np.ndarray | float) -> np.ndarray:
        """``phi_2(r)`` in eV for distances ``r`` in angstrom (0 beyond cutoff)."""
        prm = self.parameters
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(rr)
        m = rr < self.cutoff
        x = prm.sigma / rr[m]
        out[m] = (
            prm.A
            * prm.epsilon
            * (prm.B * x**prm.p - x**prm.q)
            * np.exp(prm.sigma / (rr[m] - self.cutoff))
        )
        return out if np.ndim(r) else float(out[0])

    def three_body(
        self, r_ij: float, r_ik: float, cos_theta: float
    ) -> float:
        """``phi_3`` in eV for one triplet; distances in A, angle as its cosine."""
        prm = self.parameters
        if r_ij >= self.cutoff or r_ik >= self.cutoff:
            return 0.0
        return float(
            prm.lam
            * prm.epsilon
            * (cos_theta - prm.cos_theta0) ** 2
            * np.exp(prm.gamma * prm.sigma / (r_ij - self.cutoff))
            * np.exp(prm.gamma * prm.sigma / (r_ik - self.cutoff))
        )

    def diamond_reference_energy(self) -> float:
        """Energy per atom of the ideal diamond lattice, in eV.

        In the diamond structure at the SW equilibrium every bond sits at the
        minimum of ``phi_2`` (value ``-epsilon``) and every bond angle is
        exactly tetrahedral, so every three-body term vanishes identically.
        With four bonds per atom shared between two atoms each, the energy per
        atom is therefore exactly ``-2 epsilon``.  Second neighbours in silicon
        sit at ``a0 / sqrt(2) = 3.840 A``, outside the ``3.771 A`` cutoff, so
        nothing else contributes.
        """
        return -2.0 * self.parameters.epsilon
