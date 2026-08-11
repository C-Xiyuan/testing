"""Morse pair potential -- the second two-body reference.

.. math::

    u(r) = D_e \\left[ e^{-2 \\alpha (r - r_e)} - 2 e^{-\\alpha (r - r_e)} \\right]

with ``u(r_e) = -D_e`` the well depth and ``u(inf) = 0``.  It exists in this
package as a *second* analytic ground truth alongside Lennard-Jones, for two
reasons that both matter to the argument the repository is making:

* it has the same two-body symmetry as LJ but a different anharmonicity and a
  much softer repulsive wall (exponential rather than ``r^-12``), so a
  pair-only surrogate that is exactly right for LJ is exactly right for Morse
  too, while descriptor models with a fixed radial basis are not equally good
  at both.  Any conclusion that survives both is not an artefact of the 12-6
  form;
* the harmonic frequency at the minimum, ``omega = alpha sqrt(2 D_e / mu)``, is
  analytic, which gives the phonon and VDOS estimators a closed-form target.

Structure, conventions and the four cutoff modes are exactly those of
:mod:`atomlab.potentials.lennard_jones`; see that module's docstring for the
statement of which mode conserves energy in molecular dynamics and why.

Multi-species mixing
--------------------
Lorentz-Berthelot has no meaning for a Morse potential, so the analogous rules
are used and are stated explicitly here rather than inherited by implication:
``D_e`` mixes geometrically (it is an energy scale, like ``epsilon``), while
``r_e`` and ``alpha`` mix arithmetically (they are a length and an inverse
length attached to the individual species).  This is LAMMPS' ``pair_style
morse`` mixing.  As always a full symmetric ``(T, T)`` matrix may be passed
instead, which is the only way to build a cross interaction that no mixing rule
generates.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..units import MVV2E
from .lennard_jones import PairPotential, _switch_jit, mix_pair_parameters

__all__ = ["Morse", "morse_pair"]


def morse_pair(r, d_e: float = 1.0, alpha: float = 1.0, r_e: float = 1.0):
    """Bare Morse pair energy and its radial derivative, without any cutoff.

    Parameters
    ----------
    r : array_like
        Pair distances in angstrom.
    d_e : float
        Well depth in eV (the potential minimum is ``-d_e``).
    alpha : float
        Stiffness parameter in 1/A.
    r_e : float
        Equilibrium pair distance in angstrom.

    Returns
    -------
    u : ndarray
        ``d_e [exp(-2 a (r - r_e)) - 2 exp(-a (r - r_e))]`` in eV.
    du : ndarray
        ``du/dr = -2 a d_e [exp(-2 a (r - r_e)) - exp(-a (r - r_e))]`` in eV/A.
    """
    r = np.asarray(r, dtype=np.float64)
    ex = np.exp(-alpha * (r - r_e))
    u = d_e * (ex * ex - 2.0 * ex)
    du = -2.0 * alpha * d_e * (ex * ex - ex)
    return u, du


@njit(cache=True)
def _morse_kernel(
    pi,
    pj,
    D,
    r,
    species,
    d_e,
    alpha,
    r_e,
    u_rc,
    du_rc,
    rc,
    r_on,
    mode,
    n_atoms,
    want_forces,
    want_virial,
):
    """Half-list Morse pair loop; see :func:`_lj_kernel` for the conventions.

    Returns ``(energy, energies, forces, virial)`` in (eV, eV, eV/A, eV).
    """
    energy = 0.0
    energies = np.zeros(n_atoms)
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))

    inv_span = 0.0
    if mode == 3:
        inv_span = 1.0 / (rc - r_on)

    for p in range(pi.shape[0]):
        rp = r[p]
        if rp > rc:
            continue
        a = pi[p]
        b = pj[p]
        ta = species[a]
        tb = species[b]
        de = d_e[ta, tb]
        al = alpha[ta, tb]
        re = r_e[ta, tb]

        ex = math.exp(-al * (rp - re))
        u = de * (ex * ex - 2.0 * ex)
        du = -2.0 * al * de * (ex * ex - ex)

        if mode == 1:
            u -= u_rc[ta, tb]
        elif mode == 2:
            u = u - u_rc[ta, tb] - (rp - rc) * du_rc[ta, tb]
            du = du - du_rc[ta, tb]
        elif mode == 3:
            sw, dsw = _switch_jit(rp, r_on, inv_span)
            du = du * sw + u * dsw
            u = u * sw

        energy += u
        energies[a] += 0.5 * u
        energies[b] += 0.5 * u

        if want_forces or want_virial:
            gr = du / rp
            fx = gr * D[p, 0]
            fy = gr * D[p, 1]
            fz = gr * D[p, 2]
            if want_forces:
                forces[a, 0] += fx
                forces[a, 1] += fy
                forces[a, 2] += fz
                forces[b, 0] -= fx
                forces[b, 1] -= fy
                forces[b, 2] -= fz
            if want_virial:
                virial[0, 0] -= fx * D[p, 0]
                virial[0, 1] -= fx * D[p, 1]
                virial[0, 2] -= fx * D[p, 2]
                virial[1, 0] -= fy * D[p, 0]
                virial[1, 1] -= fy * D[p, 1]
                virial[1, 2] -= fy * D[p, 2]
                virial[2, 0] -= fz * D[p, 0]
                virial[2, 1] -= fz * D[p, 1]
                virial[2, 2] -= fz * D[p, 2]

    return energy, energies, forces, virial


class Morse(PairPotential):
    """Morse pair potential with a choice of cutoff treatment.

    Parameters
    ----------
    d_e : float or array_like
        Well depth in eV.  Scalar, ``(T,)`` mixed geometrically, or a symmetric
        ``(T, T)`` matrix used as given.
    alpha : float or array_like
        Stiffness in 1/A.  ``(T,)`` values mix arithmetically.
    r_e : float or array_like
        Equilibrium pair distance in angstrom.  ``(T,)`` values mix
        arithmetically.
    cutoff : float
        Cutoff radius in angstrom.
    mode : {"truncated", "shifted", "shifted_force", "switched"}
        Cutoff treatment, identical in meaning to
        :class:`~atomlab.potentials.lennard_jones.LennardJones`.
    r_on, kernel, skin, name
        As in :class:`~atomlab.potentials.lennard_jones.PairPotential`.

    Attributes
    ----------
    d_e, alpha, r_e : ndarray, shape (T, T)
        Mixed parameters in eV, 1/A and A.

    Notes
    -----
    Unlike Lennard-Jones the Morse potential is *finite* at ``r = 0``
    (``u(0) = D_e[e^{2 a r_e} - 2 e^{a r_e}]``, large but not infinite), so a
    Morse system has no hard core and can be driven through an overlap by a bad
    integrator without the energy blowing up to signal it.  That is convenient
    for testing and dangerous for production; it is the reason the MD driver
    checks forces rather than energies for sanity.
    """

    has_jit_kernel = True

    def __init__(
        self,
        d_e,
        alpha,
        r_e,
        cutoff: float,
        *,
        mode: str = "shifted_force",
        r_on: float | None = None,
        kernel: str | None = None,
        skin: float = 0.0,
        name: str | None = None,
    ) -> None:
        de = mix_pair_parameters(d_e, rule="geometric", name="d_e")
        al = mix_pair_parameters(alpha, rule="arithmetic", name="alpha")
        re = mix_pair_parameters(r_e, rule="arithmetic", name="r_e")
        shapes = {de.shape[0], al.shape[0], re.shape[0]}
        if len(shapes) != 1:
            raise ValueError(
                f"d_e, alpha and r_e imply different numbers of types: "
                f"{de.shape[0]}, {al.shape[0]}, {re.shape[0]}"
            )
        if np.any(de < 0.0):
            raise ValueError("d_e must be non-negative")
        if np.any(al <= 0.0):
            raise ValueError("alpha must be strictly positive")
        if np.any(re <= 0.0):
            raise ValueError("r_e must be strictly positive")

        self.d_e = de
        self.alpha = al
        self.r_e = re
        self._n_types = int(de.shape[0])

        super().__init__(
            cutoff,
            mode=mode,
            r_on=r_on,
            kernel=kernel,
            skin=skin,
            name=name or f"Morse({mode})",
        )
        self._cache_cutoff_values()

    # -- presets -----------------------------------------------------------

    @classmethod
    def matched_to_lennard_jones(
        cls,
        epsilon: float,
        sigma: float,
        cutoff: float,
        **kwargs,
    ) -> "Morse":
        """Morse parameters reproducing an LJ well: same depth, position, curvature.

        Matching the first three terms of the Taylor expansion about the minimum
        is what makes the two potentials a *controlled* pair: they agree on the
        harmonic physics (elastic constants, phonons at low temperature) and
        differ only in anharmonicity and in the repulsive wall, so any observable
        that separates them is separating exactly that.

        With ``r_min = 2^{1/6} sigma`` and the LJ curvature
        ``u''(r_min) = 72 eps / 2^{1/3} sigma^2``, and the Morse curvature
        ``u''(r_e) = 2 alpha^2 D_e``, the match is

        ``D_e = eps``,  ``r_e = 2^{1/6} sigma``,
        ``alpha = sqrt(36 / 2^{1/3}) / sigma = 6 / (2^{1/6} sigma)``.

        Parameters
        ----------
        epsilon : float
            LJ well depth in eV.
        sigma : float
            LJ length parameter in angstrom.
        cutoff : float
            Cutoff radius in angstrom.
        **kwargs
            Forwarded to :class:`Morse`.

        Returns
        -------
        Morse
        """
        r_e = 2.0 ** (1.0 / 6.0) * float(sigma)
        alpha = 6.0 / r_e
        kwargs.setdefault("name", "Morse-LJ")
        return cls(float(epsilon), alpha, r_e, cutoff, **kwargs)

    @classmethod
    def argon(cls, *, cutoff: float = 8.5, **kwargs) -> "Morse":
        """Morse fitted to the well of standard Lennard-Jones argon.

        Convenience wrapper around :meth:`matched_to_lennard_jones` with
        ``epsilon = 0.0103 eV`` and ``sigma = 3.405 A``.

        Returns
        -------
        Morse
        """
        from .lennard_jones import ARGON_EPSILON, ARGON_SIGMA

        kwargs.setdefault("name", "Morse-Ar")
        return cls.matched_to_lennard_jones(ARGON_EPSILON, ARGON_SIGMA, cutoff, **kwargs)

    # -- parameters --------------------------------------------------------

    @property
    def n_types(self) -> int:
        """Number of species this potential is parameterised for."""
        return self._n_types

    def _raw(self, r, ti, tj):
        de = self.d_e[ti, tj]
        al = self.alpha[ti, tj]
        re = self.r_e[ti, tj]
        ex = np.exp(-al * (r - re))
        u = de * (ex * ex - 2.0 * ex)
        du = -2.0 * al * de * (ex * ex - ex)
        return u, du

    def _call_kernel(self, pi, pj, D, r, species, n_atoms, want_forces, want_virial):
        return _morse_kernel(
            pi,
            pj,
            D,
            r,
            species,
            self.d_e,
            self.alpha,
            self.r_e,
            self._u_rc,
            self._du_rc,
            self.cutoff,
            self.r_on,
            self.mode_code,
            n_atoms,
            want_forces,
            want_virial,
        )

    # -- analytic diagnostics ---------------------------------------------

    def angular_frequency(self, reduced_mass: float, *, type_pair=(0, 0)) -> float:
        """Harmonic frequency of an isolated Morse dimer, in rad/ps.

        ``omega = alpha sqrt(2 D_e / mu)`` follows from ``u''(r_e) = 2 alpha^2
        D_e``.  The unit conversion is the one of ``docs/design.md`` -- an
        acceleration in A/ps^2 is ``F / (m * MVV2E)`` -- so

        ``omega [rad/ps] = alpha sqrt(2 D_e / (mu * MVV2E))``

        with ``alpha`` in 1/A, ``D_e`` in eV and ``mu`` in amu.

        Parameters
        ----------
        reduced_mass : float
            Reduced mass ``m1 m2 / (m1 + m2)`` in amu.
        type_pair : sequence of int
            Which parameter pair to use.

        Returns
        -------
        float
            Angular frequency in rad/ps.  Note this is the frequency of the
            *bare* potential; the shifted-force mode adds a constant to the
            force but not to its gradient, so it leaves this unchanged, whereas
            the switched mode changes it if ``r_e > r_on``.
        """
        a, b = (int(t) for t in type_pair)
        if reduced_mass <= 0.0:
            raise ValueError(f"reduced_mass must be > 0 amu, got {reduced_mass}")
        return float(
            self.alpha[a, b]
            * math.sqrt(2.0 * self.d_e[a, b] / (float(reduced_mass) * MVV2E))
        )
