"""Harmonic reference potentials: an Einstein crystal and a harmonic pair.

Two unrelated jobs, both of which need a potential whose answer is known in
closed form.

:class:`EinsteinCrystal`
    Tethers every atom to a fixed lattice site with an isotropic spring.  It is
    the standard *reference state* for absolute free energies: the classical
    free energy of a set of independent harmonic oscillators is analytic (see
    :meth:`EinsteinCrystal.free_energy`), so a thermodynamic-integration path
    from it to the real potential yields the free energy of the real solid.  It
    is also the cheapest possible sanity check on a thermostat, since its exact
    canonical distribution is a Gaussian of known width.

:class:`HarmonicPair`
    A pair potential ``u(r) = k (r - r0)^2 / 2``.  Two atoms interacting
    through it perform an exactly solvable oscillation about ``r0``, so the MD
    integrators can be validated against an analytic trajectory rather than
    against each other.  For that use the default cutoff mode is ``"truncated"``
    -- a bare truncation -- because any of the smooth modes would modify the
    force away from ``-k (r - r0)`` and destroy the closed-form solution.  This
    is the exception ``docs/design.md`` allows: the truncation is the intended
    physics, and the test configurations never let a pair approach the cutoff.

Both use the sign and virial conventions of
:mod:`atomlab.potentials.lennard_jones`.
"""

from __future__ import annotations

import math

import numpy as np

from ..cell import minimum_image as _minimum_image
from ..types import Configuration, Result
from ..units import HPLANCK, KB, MVV2E
from .base import Potential
from .lennard_jones import PairPotential, mix_pair_parameters

__all__ = ["EinsteinCrystal", "HarmonicPair", "harmonic_pair"]


# --------------------------------------------------------------------------
# harmonic pair
# --------------------------------------------------------------------------


def harmonic_pair(r, k: float = 1.0, r0: float = 1.0):
    """Bare harmonic pair energy and its radial derivative.

    Parameters
    ----------
    r : array_like
        Pair distances in angstrom.
    k : float
        Spring constant in eV/A^2.
    r0 : float
        Equilibrium distance in angstrom.

    Returns
    -------
    u : ndarray
        ``k (r - r0)^2 / 2`` in eV.
    du : ndarray
        ``k (r - r0)`` in eV/A.
    """
    r = np.asarray(r, dtype=np.float64)
    d = r - r0
    return 0.5 * k * d * d, k * d


class HarmonicPair(PairPotential):
    """Harmonic springs between every pair inside the cutoff.

    Parameters
    ----------
    k : float or array_like
        Spring constant in eV/A^2.  Scalar, ``(T,)`` mixed geometrically, or a
        symmetric ``(T, T)`` matrix.
    r0 : float or array_like
        Equilibrium pair distance in angstrom.  ``(T,)`` values mix
        arithmetically.
    cutoff : float
        Cutoff radius in angstrom.
    mode : {"truncated", "shifted", "shifted_force", "switched"}
        Defaults to ``"truncated"``, i.e. no modification at all inside the
        cutoff, so that the force is exactly ``-k (r - r0)`` along the pair and
        the two-body dynamics has the closed-form solution the integrator tests
        need.  The energy is then discontinuous at ``rc``; see the
        :mod:`~atomlab.potentials.lennard_jones` module docstring for why that
        is fatal in general and harmless here.
    kernel : {"numpy"}
        Only the pure-NumPy path exists.  This class is a measuring instrument,
        not a production potential: there is nothing to gain from a second
        implementation of a parabola and something to lose if the two disagree.
    skin, r_on, name
        As in :class:`~atomlab.potentials.lennard_jones.PairPotential`.

    Attributes
    ----------
    k : ndarray, shape (T, T)
        Spring constants in eV/A^2.
    r0 : ndarray, shape (T, T)
        Equilibrium distances in angstrom.

    Notes
    -----
    ``u`` is *not* zero at large ``r`` -- it grows without bound -- so with more
    than two atoms in range this is a confining potential rather than anything
    physical.  That is intentional: for the integrator tests the system is a
    single isolated pair.
    """

    has_jit_kernel = False

    def __init__(
        self,
        k,
        r0,
        cutoff: float,
        *,
        mode: str = "truncated",
        r_on: float | None = None,
        kernel: str | None = None,
        skin: float = 0.0,
        name: str | None = None,
    ) -> None:
        kk = mix_pair_parameters(k, rule="geometric", name="k")
        rr = mix_pair_parameters(r0, rule="arithmetic", name="r0")
        if kk.shape != rr.shape:
            raise ValueError(
                f"k and r0 imply different numbers of types: {kk.shape[0]} vs {rr.shape[0]}"
            )
        if np.any(kk < 0.0):
            raise ValueError("spring constants must be non-negative")
        if np.any(rr < 0.0):
            raise ValueError("equilibrium distances must be non-negative")

        self.k = kk
        self.r0 = rr
        self._n_types = int(kk.shape[0])

        super().__init__(
            cutoff,
            mode=mode,
            r_on=r_on,
            kernel=kernel,
            skin=skin,
            name=name or f"harmonic-pair({mode})",
        )
        self._cache_cutoff_values()

    @property
    def n_types(self) -> int:
        """Number of species this potential is parameterised for."""
        return self._n_types

    def _raw(self, r, ti, tj):
        d = r - self.r0[ti, tj]
        k = self.k[ti, tj]
        return 0.5 * k * d * d, k * d

    def angular_frequency(self, reduced_mass: float, *, type_pair=(0, 0)) -> float:
        """Vibrational frequency of an isolated pair, in rad/ps.

        ``omega = sqrt(k / mu)`` with the metal-unit conversion of
        ``docs/design.md`` (``a = F / (m * MVV2E)``):

        ``omega [rad/ps] = sqrt(k [eV/A^2] / (mu [amu] * MVV2E))``.

        The relative coordinate of a two-atom system obeys
        ``mu d^2 r / dt^2 = -k (r - r0)`` exactly for a purely radial
        displacement, so this is the frequency an integrator test must
        reproduce; transverse motion of the pair is free rotation, not
        oscillation, which is why the analytic test displaces along the bond.

        Parameters
        ----------
        reduced_mass : float
            ``m1 m2 / (m1 + m2)`` in amu.
        type_pair : sequence of int
            Which parameter pair to use.

        Returns
        -------
        float
            Angular frequency in rad/ps.
        """
        a, b = (int(t) for t in type_pair)
        if reduced_mass <= 0.0:
            raise ValueError(f"reduced_mass must be > 0 amu, got {reduced_mass}")
        return float(math.sqrt(self.k[a, b] / (float(reduced_mass) * MVV2E)))


# --------------------------------------------------------------------------
# Einstein crystal
# --------------------------------------------------------------------------


class EinsteinCrystal(Potential):
    """Independent isotropic springs tethering atoms to fixed reference sites.

    .. math::  U = \\tfrac{1}{2} \\sum_i k_i \\, |\\Delta_i|^2 ,
               \\qquad \\Delta_i = r_i - R_i

    There is no interaction between atoms, so this is an *external field* and
    not a pair potential.  Two consequences worth stating, because both are
    easy to get wrong:

    * the reference sites ``R_i`` are anchored in the laboratory frame and do
      **not** move under strain, so the virial ``W_ab = -dU/d eps_ab`` is
      ``sum_i f_ia p_ib`` -- the ``sum f (x) r`` of an external field -- and is
      in general **not symmetric**.  :meth:`Potential.numerical_virial` is the
      arbiter, and it agrees;
    * the energy is not translationally invariant.  That is the entire point of
      a tether, but it means this potential must never be used on its own in an
      MD run that removes the centre-of-mass drift as if it were a symmetry.

    Parameters
    ----------
    reference_positions : array_like, shape (N, 3)
        Lattice sites in angstrom.  Copied.
    spring_constant : float or array_like, shape (N,)
        ``k`` in eV/A^2, uniform or per atom.
    minimum_image : bool
        If True (the default) the displacement ``Delta_i`` is reduced to its
        shortest periodic image before the spring energy is formed, so an atom
        that wanders across a cell face is pulled back to the *nearest* image of
        its site rather than dragged across the box.  This is what makes the
        potential usable in an MD run with wrapped coordinates.  It is exact as
        long as no atom strays further than half the cell width from its site,
        which for a bound reference state is never in question.
    name : str
        Short identifier.

    Attributes
    ----------
    cutoff : float
        ``0.0``.  The potential is on-site, so it needs no neighbours; the
        attribute exists only because :class:`~atomlab.potentials.base.Potential`
        and :class:`~atomlab.potentials.base.SumPotential` read it.
    """

    def __init__(
        self,
        reference_positions,
        spring_constant,
        *,
        minimum_image: bool = True,
        name: str = "einstein",
    ) -> None:
        ref = np.ascontiguousarray(np.asarray(reference_positions, dtype=np.float64))
        if ref.ndim != 2 or ref.shape[1] != 3:
            raise ValueError(f"reference_positions must be (N, 3), got {ref.shape}")
        n = ref.shape[0]

        k = np.asarray(spring_constant, dtype=np.float64)
        if k.ndim == 0:
            k = np.full(n, float(k))
        k = np.ascontiguousarray(k)
        if k.shape != (n,):
            raise ValueError(f"spring_constant must be scalar or ({n},), got {k.shape}")
        if np.any(k < 0.0):
            raise ValueError("spring constants must be non-negative")

        self.reference_positions = ref
        self.spring_constant = k
        self.minimum_image = bool(minimum_image)
        self.cutoff = 0.0
        self.name = str(name)

    @classmethod
    def from_configuration(
        cls, configuration: Configuration, spring_constant, **kwargs
    ) -> "EinsteinCrystal":
        """Tether atoms to the positions they currently occupy.

        Parameters
        ----------
        configuration : Configuration
            Its positions become the reference sites (copied).
        spring_constant : float or array_like
            ``k`` in eV/A^2.
        **kwargs
            Forwarded to :class:`EinsteinCrystal`.

        Returns
        -------
        EinsteinCrystal
        """
        return cls(configuration.positions.copy(), spring_constant, **kwargs)

    @property
    def n_atoms(self) -> int:
        """Number of tethered sites."""
        return self.reference_positions.shape[0]

    # -- evaluation --------------------------------------------------------

    def _displacements(self, configuration: Configuration) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Delta, p)``: site displacements and the positions used.

        ``p = R + Delta`` is the periodic image of each atom nearest to its own
        reference site.  Under an affine strain both ``r`` and ``cell`` scale,
        hence ``p -> (1 + eps) p`` with the image index held fixed, which is
        what makes the analytic virial below exact.
        """
        if configuration.n_atoms != self.n_atoms:
            raise ValueError(
                f"{self.name} was built for {self.n_atoms} sites but the configuration "
                f"has {configuration.n_atoms} atoms"
            )
        delta = configuration.positions - self.reference_positions
        if self.minimum_image and np.asarray(configuration.pbc).any():
            delta = _minimum_image(delta, configuration.cell, configuration.pbc)
        return delta, self.reference_positions + delta

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Energy (eV), forces (eV/A) and virial (eV) of ``configuration``.

        Returns
        -------
        Result
            ``energies[i] = k_i |Delta_i|^2 / 2`` is an exact per-atom
            decomposition here, unlike for a pair potential where it is a
            convention.
        """
        delta, p = self._displacements(configuration)
        k = self.spring_constant

        per_atom = 0.5 * k * np.einsum("ia,ia->i", delta, delta)
        f = -k[:, None] * delta
        # W_ab = -dU/d eps_ab with r -> (1+eps) r and the sites R held fixed.
        w = np.einsum("ia,ib->ab", f, p) if virial else None

        return Result(
            energy=float(per_atom.sum()),
            forces=f,
            virial=w,
            energies=per_atom,
        )

    # -- analytic thermodynamics ------------------------------------------

    def free_energy(self, temperature: float, masses) -> float:
        """Classical Helmholtz free energy of the tethered oscillators, in eV.

        Each atom is an independent 3-D harmonic oscillator, so the canonical
        partition function factorises and

        ``F = -k_B T sum_i ln[ (2 pi / (beta k_i))^{3/2} / Lambda_i^3 ]``

        with ``Lambda_i = h / sqrt(2 pi m_i k_B T)`` the thermal de Broglie
        wavelength.  In metal units the momentum integral needs the same
        conversion as the kinetic energy (``1 eV = MVV2E^-1 amu A^2 / ps^2``),
        which gives

        ``Lambda [A] = h / sqrt(2 pi m k_B T * MVV2E)``

        with ``h`` in eV*ps, ``m`` in amu and ``k_B T`` in eV.  The ``Lambda``
        term is the only place Planck's constant enters a classical simulation;
        it cancels exactly in any free-energy *difference* between two states
        with the same atoms, which is how this class is actually used.

        Parameters
        ----------
        temperature : float
            Temperature in K.  Must be positive.
        masses : array_like, shape (N,)
            Atomic masses in amu.

        Returns
        -------
        float
            Total free energy in eV, including the ideal-gas momentum
            contribution.

        Raises
        ------
        ValueError
            If any spring constant is zero, in which case the configurational
            integral diverges (a free particle has no harmonic free energy) --
            silently returning ``-inf`` would be a worse answer than an error.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 K, got {temperature}")
        m = np.asarray(masses, dtype=np.float64)
        if m.shape != (self.n_atoms,):
            raise ValueError(f"masses must be ({self.n_atoms},), got {m.shape}")
        if np.any(m <= 0.0):
            raise ValueError("masses must be strictly positive")
        if np.any(self.spring_constant <= 0.0):
            raise ValueError(
                "free_energy needs strictly positive spring constants; a zero spring "
                "makes the configurational integral divergent"
            )

        kt = KB * float(temperature)
        lam = HPLANCK / np.sqrt(2.0 * math.pi * m * kt * MVV2E)
        z = (2.0 * math.pi * kt / self.spring_constant) ** 1.5 / lam**3
        return float(-kt * np.sum(np.log(z)))

    def angular_frequency(self, mass: float, *, atom: int = 0) -> float:
        """Oscillation frequency of one tethered atom, in rad/ps.

        ``omega = sqrt(k / m)`` with the metal-unit conversion
        ``omega [rad/ps] = sqrt(k [eV/A^2] / (m [amu] * MVV2E))``.

        Parameters
        ----------
        mass : float
            Atomic mass in amu.
        atom : int
            Which site's spring constant to use.

        Returns
        -------
        float
        """
        if mass <= 0.0:
            raise ValueError(f"mass must be > 0 amu, got {mass}")
        return float(math.sqrt(self.spring_constant[int(atom)] / (float(mass) * MVV2E)))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EinsteinCrystal(name={self.name!r}, n_atoms={self.n_atoms}, "
            f"k={self.spring_constant[0]:.4g} eV/A^2)"
        )
