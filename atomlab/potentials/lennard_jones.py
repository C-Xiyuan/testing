"""Lennard-Jones 12-6, and the shared machinery for isotropic pair potentials.

The 12-6 potential

.. math::  u(r) = 4 \\epsilon \\left[ (\\sigma/r)^{12} - (\\sigma/r)^{6} \\right]

is the ground-truth ``U_0`` for the argon half of this study, so it has to be
exact: analytic energy, forces and virial, a jitted kernel and an independent
pure-NumPy path that the test suite cross-checks against it, and a documented
statement of *which* of the four standard cutoff treatments is being used.

Why ``PairPotential`` lives in this file
---------------------------------------
Every two-body potential in the package (this one, :mod:`~atomlab.potentials.morse`,
:mod:`~atomlab.potentials.harmonic`) shares the same skeleton: build a half
neighbour list, evaluate ``u(r)`` and ``u'(r)`` per pair, accumulate energy,
forces and virial, and apply one of four cutoff treatments.  That skeleton is
written once, here, and imported by the others.  The module map in
``docs/design.md`` gives this file the Lennard-Jones row, so the shared base is
kept in it rather than in a new file.

Cutoff modes, and which one conserves energy in MD
--------------------------------------------------
Let ``u`` be the bare pair function, ``rc`` the cutoff and ``phi`` what this
class actually evaluates.  ``phi(r) = 0`` for ``r > rc`` in every mode.

``"truncated"``
    ``phi(r) = u(r)``.  The energy **jumps** by ``-u(rc)`` whenever a pair
    crosses the cutoff.  In exact mechanics that jump is produced by an
    impulsive force ``-u(rc) \\delta(r - rc)``; a finite-difference integrator
    never sees the delta function, so every crossing silently adds or removes
    ``u(rc)`` of energy.  The result is a random walk of the total energy whose
    rate is set by the crossing frequency, **not** by the timestep: halving
    ``dt`` does not halve it, because it is not a discretisation error at all.
    This is why a bare truncated Lennard-Jones fails an NVE drift test no
    matter how carefully it is integrated.  Use this mode for static energies
    only -- lattice sums, where the truncation *is* the definition being
    tested.

``"shifted"``
    ``phi(r) = u(r) - u(rc)``.  The energy is continuous, so the impulse is
    gone and NVE drift is bounded and does shrink with ``dt``.  The *force*
    still jumps by ``u'(rc)`` at the cutoff, so the force field is only
    piecewise smooth; velocity Verlet's ``O(dt^2)`` energy behaviour is
    formally lost at crossing events, and in practice the drift is larger than
    for the two smooth modes at the same ``dt``.  Acceptable when ``rc`` is
    large enough that ``u'(rc)`` is negligible.  This is what ASE's
    ``LennardJones(smooth=False)`` computes, which is why the cross-check test
    uses it.

``"shifted_force"``
    ``phi(r) = u(r) - u(rc) - (r - rc) u'(rc)``.  Both ``phi(rc) = 0`` and
    ``phi'(rc) = 0``, so energy *and* force are continuous: no impulse, no
    kink, excellent NVE conservation.  The price is that the force is changed
    at **every** ``r < rc`` by the constant ``u'(rc)``, so the thermodynamics
    is a slightly different model rather than a truncation of the same one.

``"switched"``
    ``phi(r) = u(r) S(r)`` with ``S`` the quintic smoothstep, exactly 1 below
    ``r_on`` and exactly 0 at ``rc`` with vanishing first and second
    derivatives at both ends (see :func:`switching_function`).  ``phi`` is
    therefore ``C^2``: energy, force and force gradient are all continuous, so
    this conserves energy at least as well as ``shifted_force`` *and* leaves
    the potential untouched below ``r_on``.  The price is a distortion of the
    force inside ``[r_on, rc]``, which grows as the window is narrowed.

**Recommendation, and the one the downstream NVE test depends on:** use
``"shifted_force"`` or ``"switched"`` for molecular dynamics; ``"shifted"`` only
with a generous cutoff; ``"truncated"`` never for dynamics.

Sign conventions
----------------
Pair displacements come from :func:`atomlab.neighbors.pair_vectors` and point
**from i to j**: ``D = r_j + shift @ cell - r_i``, ``r = |D|``.  With
``U = sum_pairs phi(r)``,

* force on ``i``: ``f_i = -dU/dr_i = +phi'(r) D / r`` (and ``-`` that on ``j``);
* virial: ``W_ab = -dU/d eps_ab = -sum_pairs phi'(r) D_a D_b / r``, which is the
  ``sum_{i<j} f_ij (x) r_ij`` of ``docs/design.md`` since ``r_ij = r_i - r_j =
  -D``.  It is manifestly symmetric for a central potential.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit

from ..cell import check_minimum_image
from ..neighbors import VerletList, build_neighbor_list, pair_vectors
from ..types import Configuration, Result
from .base import Potential

__all__ = [
    "CUTOFF_MODES",
    "MODE_CODES",
    "PairPotential",
    "LennardJones",
    "switching_function",
    "lj_pair",
    "mix_pair_parameters",
    "ARGON_EPSILON",
    "ARGON_SIGMA",
    "ARGON_CUTOFF",
    "FCC_LATTICE_SUM_A12",
    "FCC_LATTICE_SUM_A6",
    "FCC_MIN_ENERGY_PER_EPSILON",
    "FCC_MIN_NN_OVER_SIGMA",
    "FCC_MIN_A0_OVER_SIGMA",
]

# --------------------------------------------------------------------------
# Literature parameters and validation targets
# --------------------------------------------------------------------------

#: Well depth of the standard Lennard-Jones argon parameterisation, in eV.
ARGON_EPSILON = 0.0103
#: Length parameter of the standard Lennard-Jones argon parameterisation, in A.
ARGON_SIGMA = 3.405
#: Conventional cutoff for LJ argon, in A (= 2.4964 sigma).
ARGON_CUTOFF = 8.5

#: fcc lattice sums ``A_n = sum'_i (d/r_i)^n`` over all neighbours of one site,
#: with ``d`` the nearest-neighbour distance.  Standard values (Kittel; Ashcroft
#: & Mermin), reproduced here to 6 figures by direct enumeration out to 60 d.
#: Both series converge, ``A_6`` only as ``1/R^3``, which is the same slow
#: convergence that forces the cutoff extrapolation in the lattice-energy test.
FCC_LATTICE_SUM_A12 = 12.131880
FCC_LATTICE_SUM_A6 = 14.453921

#: Minimum of the fcc Lennard-Jones lattice energy, in units of ``epsilon`` per
#: atom.  ``E/N = 2 eps [A12 (sig/d)^12 - A6 (sig/d)^6]`` is stationary at
#: ``(sig/d)^6 = A6 / (2 A12)``, where it takes the closed-form value
#: ``-A6^2 / (2 A12) = -8.6102 eps``.
FCC_MIN_ENERGY_PER_EPSILON = -8.610200

#: Nearest-neighbour distance at that minimum, in units of ``sigma``:
#: ``(2 A12 / A6)^{1/6}``.
FCC_MIN_NN_OVER_SIGMA = 1.0901734

#: The corresponding **conventional cubic** lattice constant, ``sqrt(2)`` times
#: the nearest-neighbour distance, in units of ``sigma``.
#:
#: Note: ``docs/design.md`` quotes ``a0 = 1.5496 sigma`` for this quantity.
#: That value is not consistent with the nearest-neighbour distance
#: ``1.0902 sigma`` quoted in the same place (``1.0902 * sqrt(2) = 1.5418``),
#: and evaluating the lattice sum at ``a0 = 1.5496 sigma`` gives
#: ``-8.6024 eps/atom``, not ``-8.6102``.  The nearest-neighbour form is the
#: correct one and is what the test suite checks; see
#: ``tests/test_pair_potentials.py``.
FCC_MIN_A0_OVER_SIGMA = 1.5417374

#: The four supported cutoff treatments, in the order they are documented above.
CUTOFF_MODES: tuple[str, ...] = ("truncated", "shifted", "shifted_force", "switched")

#: Integer codes handed to the jitted kernels (numba cannot switch on strings
#: cheaply, and an integer keeps the inner loop branch-predictable).
MODE_CODES: dict[str, int] = {name: k for k, name in enumerate(CUTOFF_MODES)}


# --------------------------------------------------------------------------
# parameter mixing
# --------------------------------------------------------------------------


def mix_pair_parameters(values, *, rule: str, name: str = "parameter") -> np.ndarray:
    """Expand per-type pair parameters into a full ``(T, T)`` matrix.

    Parameters
    ----------
    values : array_like
        Scalar (single species), ``(T,)`` per-type values to be mixed, or an
        already-mixed symmetric ``(T, T)`` matrix which is used as given.
    rule : {"geometric", "arithmetic"}
        Mixing rule for the ``(T,)`` case.  Lorentz-Berthelot is
        ``geometric`` for the energy scale (``eps_ij = sqrt(eps_i eps_j)``) and
        ``arithmetic`` for the length scale (``sig_ij = (sig_i + sig_j)/2``).
    name : str
        Used in error messages.

    Returns
    -------
    ndarray, shape (T, T)
        Symmetric matrix of pair parameters, same units as the input.

    Notes
    -----
    Passing a full matrix is not a convenience but a requirement of the
    perturbation experiments: they need to break Lorentz-Berthelot deliberately
    (a cross term that no mixing rule produces is the cheapest way to build a
    ``delta U`` that is invisible to single-species observables).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.ascontiguousarray(arr.reshape(1, 1).copy())
    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError(f"{name} must have at least one entry")
        if rule == "geometric":
            if np.any(arr < 0.0):
                raise ValueError(f"geometric mixing needs non-negative {name}, got {arr}")
            mixed = np.sqrt(np.outer(arr, arr))
        elif rule == "arithmetic":
            mixed = 0.5 * (arr[:, None] + arr[None, :])
        else:
            raise ValueError(f"unknown mixing rule {rule!r}; use 'geometric' or 'arithmetic'")
        return np.ascontiguousarray(mixed)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} matrix must be square, got {arr.shape}")
        if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} matrix must be symmetric (u_ij == u_ji)")
        return np.ascontiguousarray(arr.copy())
    raise ValueError(f"{name} must be scalar, (T,) or (T, T), got shape {arr.shape}")


# --------------------------------------------------------------------------
# switching function
# --------------------------------------------------------------------------


def switching_function(r, r_on: float, r_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Quintic ``C^2`` switching function and its derivative.

    With ``x = (r - r_on) / (r_cut - r_on)`` clamped to ``[0, 1]``,

    ``S(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5``

    which satisfies ``S(0) = 1``, ``S(1) = 0`` and ``S' = S'' = 0`` at both
    ends.  The vanishing *second* derivative is what makes ``u * S`` a ``C^2``
    function of ``r``: a ``C^1`` switch (such as the cubic one ASE uses) leaves
    a discontinuity in ``dF/dr``, which is harmless for energy conservation but
    shows up as a spurious feature in finite-difference force constants and
    hence in the phonon spectrum.

    Parameters
    ----------
    r : array_like
        Pair distances in angstrom.
    r_on : float
        Radius at which the switch starts, in angstrom.
    r_cut : float
        Radius at which the switch reaches zero, in angstrom.  Must exceed
        ``r_on``.

    Returns
    -------
    S : ndarray
        Dimensionless switch value, same shape as ``r``.
    dS : ndarray
        ``dS/dr`` in 1/A, same shape as ``r``.
    """
    if not r_cut > r_on:
        raise ValueError(f"switching needs r_on < r_cut, got r_on={r_on}, r_cut={r_cut}")
    r = np.asarray(r, dtype=np.float64)
    span = float(r_cut) - float(r_on)
    x = np.clip((r - r_on) / span, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    s = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    # Zero outside the window (the clamp above already flattens x, so dS is 0
    # there automatically; written out for clarity).
    ds = (-30.0 * x2 + 60.0 * x3 - 30.0 * x3 * x) / span
    return s, ds


@njit(cache=True, inline="always")
def _switch_jit(r: float, r_on: float, inv_span: float) -> tuple[float, float]:
    """Scalar version of :func:`switching_function` for the jitted kernels."""
    x = (r - r_on) * inv_span
    if x <= 0.0:
        return 1.0, 0.0
    if x >= 1.0:
        return 0.0, 0.0
    x2 = x * x
    x3 = x2 * x
    s = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    ds = (-30.0 * x2 + 60.0 * x3 - 30.0 * x3 * x) * inv_span
    return s, ds


# --------------------------------------------------------------------------
# bare pair function
# --------------------------------------------------------------------------


def lj_pair(r, epsilon: float = 1.0, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Bare 12-6 pair energy and its radial derivative, without any cutoff.

    Parameters
    ----------
    r : array_like
        Pair distances in angstrom.  Must be strictly positive.
    epsilon : float
        Well depth in eV.
    sigma : float
        Zero crossing of the potential in angstrom (the minimum is at
        ``2**(1/6) sigma``).

    Returns
    -------
    u : ndarray
        ``4 eps [(sig/r)^12 - (sig/r)^6]`` in eV.
    du : ndarray
        ``du/dr = -(24 eps / r) [2 (sig/r)^12 - (sig/r)^6]`` in eV/A.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r <= 0.0):
        raise ValueError("Lennard-Jones is singular at r = 0; all distances must be > 0")
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    u = 4.0 * epsilon * (sr12 - sr6)
    du = -24.0 * epsilon * (2.0 * sr12 - sr6) / r
    return u, du


# --------------------------------------------------------------------------
# shared pair-potential machinery
# --------------------------------------------------------------------------


class PairPotential(Potential):
    """Base class for isotropic two-body potentials with a cutoff treatment.

    A subclass supplies

    * ``_raw(r, ti, tj) -> (u, du)``, the bare pair function and its radial
      derivative for the given type pairs (pure NumPy, vectorised);
    * ``_type_matrices()``, the ``(T, T)`` parameter matrices its jitted kernel
      needs, and ``_call_kernel(...)`` to invoke that kernel -- optional, and
      only if ``has_jit_kernel`` is set.

    Everything else -- neighbour lists, the minimum-image guard, the four
    cutoff modes, accumulation of energy/forces/virial/per-atom energies -- is
    handled here, identically for every subclass.

    Parameters
    ----------
    cutoff : float
        Interaction range ``rc`` in angstrom.  ``phi(r) = 0`` exactly for
        ``r > rc``.
    mode : {"truncated", "shifted", "shifted_force", "switched"}
        Cutoff treatment; see the module docstring for the energy-conservation
        consequences of each.
    r_on : float, optional
        Onset radius of the ``"switched"`` mode, in angstrom.  Defaults to
        ``0.9 * cutoff``: switching late leaves the physically important part
        of the potential untouched, at the cost of a stronger (but still
        smooth) force distortion inside the narrow window.  Ignored by the
        other modes.
    kernel : {"numba", "numpy"}
        Which implementation ``compute`` uses.  The two are independent code
        paths over the same neighbour list and are required by the test suite
        to agree to machine precision -- a fast wrong kernel is the most
        expensive kind of bug in this package.
    skin : float
        Verlet skin in angstrom.  ``0`` (the default) rebuilds the neighbour
        list on every call, which is the safe behaviour for one-shot
        evaluations; a positive skin caches a list out to ``cutoff + skin`` and
        rebuilds it only once an atom has moved by more than ``skin/2``.  Pairs
        between ``cutoff`` and ``cutoff + skin`` contribute exactly zero, so the
        skin changes performance and nothing else.
    name : str
        Short identifier used in tables and figures.
    """

    #: Set True by subclasses that provide ``_call_kernel``.
    has_jit_kernel: bool = False

    def __init__(
        self,
        cutoff: float,
        *,
        mode: str = "shifted_force",
        r_on: float | None = None,
        kernel: str | None = None,
        skin: float = 0.0,
        name: str = "pair",
    ) -> None:
        cutoff = float(cutoff)
        if not cutoff > 0.0 or not math.isfinite(cutoff):
            raise ValueError(f"cutoff must be a finite positive length in A, got {cutoff}")
        if mode not in MODE_CODES:
            raise ValueError(f"unknown cutoff mode {mode!r}; expected one of {CUTOFF_MODES}")

        self.cutoff = cutoff
        self.mode = mode
        self.mode_code = MODE_CODES[mode]
        self.name = str(name)

        r_on = 0.9 * cutoff if r_on is None else float(r_on)
        if mode == "switched" and not 0.0 <= r_on < cutoff:
            raise ValueError(
                f"switched mode needs 0 <= r_on < cutoff, got r_on={r_on}, cutoff={cutoff}"
            )
        self.r_on = r_on

        if kernel is None:
            kernel = "numba" if self.has_jit_kernel else "numpy"
        if kernel not in ("numba", "numpy"):
            raise ValueError(f"kernel must be 'numba' or 'numpy', got {kernel!r}")
        if kernel == "numba" and not self.has_jit_kernel:
            raise ValueError(f"{type(self).__name__} has no jitted kernel; use kernel='numpy'")
        self.kernel = kernel

        skin = float(skin)
        if skin < 0.0:
            raise ValueError(f"skin must be >= 0 A, got {skin}")
        self.skin = skin
        self._verlet = (
            VerletList(cutoff=cutoff, skin=skin, half=True) if skin > 0.0 else None
        )

    # -- subclass hooks ----------------------------------------------------

    @property
    def n_types(self) -> int:
        """Number of species this potential is parameterised for."""
        raise NotImplementedError

    def _raw(self, r: np.ndarray, ti: np.ndarray, tj: np.ndarray):
        """Bare ``(u, du)`` in (eV, eV/A) for distances ``r`` and type pairs."""
        raise NotImplementedError

    def _call_kernel(self, pi, pj, D, r, species, n_atoms, want_forces, want_virial):
        """Jitted evaluation; only defined when ``has_jit_kernel``."""
        raise NotImplementedError

    # -- the modified pair function ---------------------------------------

    def pair(self, r, type_i=0, type_j=0) -> tuple[np.ndarray, np.ndarray]:
        """Cutoff-modified pair energy ``phi(r)`` and its derivative.

        This is the function the potential actually sums; it is public because
        the learned pair-spline model and the perturbation constructions both
        need to see exactly what the reference potential does at the cutoff.

        Parameters
        ----------
        r : array_like
            Pair distances in angstrom.
        type_i, type_j : int or array_like
            Species type indices, broadcast against ``r``.

        Returns
        -------
        phi : ndarray
            Pair energy in eV, exactly zero for ``r > cutoff``.
        dphi : ndarray
            ``dphi/dr`` in eV/A, exactly zero for ``r > cutoff``.
        """
        r = np.asarray(r, dtype=np.float64)
        ti = np.broadcast_to(np.asarray(type_i, dtype=np.int64), r.shape)
        tj = np.broadcast_to(np.asarray(type_j, dtype=np.int64), r.shape)
        if np.any(ti >= self.n_types) or np.any(tj >= self.n_types):
            raise ValueError(f"{self.name} is parameterised for {self.n_types} type(s) only")

        inside = r <= self.cutoff
        # Evaluate on a clamped copy so that a pair sitting far outside the
        # cutoff cannot raise (or overflow) inside the bare pair function.
        r_eval = np.where(inside, r, self.cutoff)
        u, du = self._raw(r_eval, ti, tj)

        if self.mode == "shifted":
            u = u - self._u_rc[ti, tj]
        elif self.mode == "shifted_force":
            u = u - self._u_rc[ti, tj] - (r_eval - self.cutoff) * self._du_rc[ti, tj]
            du = du - self._du_rc[ti, tj]
        elif self.mode == "switched":
            s, ds = switching_function(r_eval, self.r_on, self.cutoff)
            du = du * s + u * ds
            u = u * s

        return np.where(inside, u, 0.0), np.where(inside, du, 0.0)

    def _cache_cutoff_values(self) -> None:
        """Precompute ``u(rc)`` and ``u'(rc)`` for every type pair.

        Must be called by the subclass once its parameters are set.  These are
        the only quantities the shifted and shifted-force modes need, and
        caching them keeps the inner loops free of ``**`` calls.
        """
        t = self.n_types
        ti, tj = np.meshgrid(np.arange(t), np.arange(t), indexing="ij")
        rc = np.full((t, t), self.cutoff)
        u_rc, du_rc = self._raw(rc, ti, tj)
        self._u_rc = np.ascontiguousarray(np.asarray(u_rc, dtype=np.float64))
        self._du_rc = np.ascontiguousarray(np.asarray(du_rc, dtype=np.float64))

    # -- evaluation --------------------------------------------------------

    def _neighbor_list(self, configuration: Configuration):
        """Half neighbour list for ``configuration``, with the image guard."""
        if np.asarray(configuration.pbc).any():
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )
        if self._verlet is not None:
            return self._verlet.update(configuration)
        return build_neighbor_list(configuration, self.cutoff, half=True)

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Energy (eV), forces (eV/A) and virial (eV) of ``configuration``.

        Parameters
        ----------
        configuration : Configuration
            Geometry; not modified.
        forces, virial : bool
            Requesting less work is honoured only where it is free; both are
            in practice always computed, because the pair loop that produces
            the energy produces them at negligible extra cost.

        Returns
        -------
        Result
            With ``energies`` set to the symmetric per-atom decomposition
            ``u_i = 1/2 sum_j phi(r_ij)``.
        """
        species = np.asarray(configuration.species, dtype=np.int32)
        if species.size and int(species.max()) >= self.n_types:
            raise ValueError(
                f"configuration has species index {int(species.max())} but {self.name} "
                f"is parameterised for {self.n_types} type(s)"
            )

        nl = self._neighbor_list(configuration)
        D, r = pair_vectors(configuration, nl)
        n = configuration.n_atoms

        if self.kernel == "numba":
            energy, energies, f, w = self._call_kernel(
                nl.i, nl.j, D, r, species, n, bool(forces), bool(virial)
            )
        else:
            energy, energies, f, w = self._compute_numpy(
                nl.i, nl.j, D, r, species, n, bool(forces), bool(virial)
            )

        return Result(
            energy=float(energy),
            forces=f,
            virial=w if virial else None,
            energies=energies,
        )

    def _compute_numpy(self, pi, pj, D, r, species, n_atoms, want_forces, want_virial):
        """Pure-NumPy reference evaluation over a half neighbour list.

        Kept deliberately naive and vectorised: it is the arbiter the jitted
        kernel is validated against, so it must contain no optimisation that
        could itself be wrong.
        """
        ti = species[pi].astype(np.int64)
        tj = species[pj].astype(np.int64)
        u, du = self.pair(r, ti, tj)

        energy = float(u.sum())
        energies = np.zeros(n_atoms)
        np.add.at(energies, pi, 0.5 * u)
        np.add.at(energies, pj, 0.5 * u)

        # phi'(r) D / r is the force on i; the neighbour list guarantees r > 0,
        # but guard anyway so that a degenerate input fails loudly elsewhere
        # rather than producing a NaN here.
        safe_r = np.where(r > 0.0, r, 1.0)
        fvec = (du / safe_r)[:, None] * D

        forces = np.zeros((n_atoms, 3))
        np.add.at(forces, pi, fvec)
        np.add.at(forces, pj, -fvec)

        # W_ab = -sum_p phi'(r) D_a D_b / r = -sum_p f_a D_b.
        virial = -np.einsum("pa,pb->ab", fvec, D) if len(r) else np.zeros((3, 3))
        return energy, energies, forces, virial

    # -- diagnostics -------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(name={self.name!r}, cutoff={self.cutoff:.4f}, "
            f"mode={self.mode!r}, kernel={self.kernel!r})"
        )


# --------------------------------------------------------------------------
# jitted kernel
# --------------------------------------------------------------------------


@njit(cache=True)
def _lj_kernel(
    pi,
    pj,
    D,
    r,
    species,
    eps,
    sig,
    u_rc,
    du_rc,
    rc,
    r_on,
    mode,
    n_atoms,
    want_forces,
    want_virial,
):
    """Half-list Lennard-Jones pair loop.

    Parameters mirror :meth:`PairPotential._call_kernel`.  ``mode`` is the
    integer code from :data:`MODE_CODES`.  Returns
    ``(energy, energies, forces, virial)`` in (eV, eV, eV/A, eV).
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
        e = eps[ta, tb]
        s = sig[ta, tb]

        sr2 = (s * s) / (rp * rp)
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        u = 4.0 * e * (sr12 - sr6)
        du = -24.0 * e * (2.0 * sr12 - sr6) / rp

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


# --------------------------------------------------------------------------
# the potential
# --------------------------------------------------------------------------


class LennardJones(PairPotential):
    """Lennard-Jones 12-6 potential with a choice of cutoff treatment.

    Parameters
    ----------
    epsilon : float or array_like
        Well depth in eV.  Scalar for one species, ``(T,)`` for per-type values
        combined by the Berthelot geometric rule, or a symmetric ``(T, T)``
        matrix used as given.
    sigma : float or array_like
        Length parameter in angstrom, with the Lorentz arithmetic rule for the
        ``(T,)`` case.  Together these two are Lorentz-Berthelot mixing:
        ``eps_ij = sqrt(eps_i eps_j)``, ``sig_ij = (sig_i + sig_j) / 2``.
    cutoff : float
        Cutoff radius ``rc`` in angstrom.
    mode : {"truncated", "shifted", "shifted_force", "switched"}
        See the module docstring.  Defaults to ``"shifted_force"``, the cheapest
        treatment with continuous forces, because the dominant use of this class
        is molecular dynamics.
    r_on, kernel, skin, name
        As in :class:`PairPotential`.

    Attributes
    ----------
    epsilon : ndarray, shape (T, T)
        Mixed well depths in eV.
    sigma : ndarray, shape (T, T)
        Mixed length parameters in angstrom.

    Examples
    --------
    >>> lj = LennardJones.argon()
    >>> round(lj.r_min, 3)
    3.822
    """

    has_jit_kernel = True

    def __init__(
        self,
        epsilon,
        sigma,
        cutoff: float,
        *,
        mode: str = "shifted_force",
        r_on: float | None = None,
        kernel: str | None = None,
        skin: float = 0.0,
        name: str | None = None,
    ) -> None:
        eps = mix_pair_parameters(epsilon, rule="geometric", name="epsilon")
        sig = mix_pair_parameters(sigma, rule="arithmetic", name="sigma")
        if eps.shape != sig.shape:
            raise ValueError(
                f"epsilon and sigma imply different numbers of types: "
                f"{eps.shape[0]} vs {sig.shape[0]}"
            )
        if np.any(eps < 0.0):
            raise ValueError("epsilon must be non-negative (a negative well depth is not LJ)")
        if np.any(sig <= 0.0):
            raise ValueError("sigma must be strictly positive")

        self.epsilon = eps
        self.sigma = sig
        self._n_types = int(eps.shape[0])

        super().__init__(
            cutoff,
            mode=mode,
            r_on=r_on,
            kernel=kernel,
            skin=skin,
            name=name or f"LJ({mode})",
        )
        self._cache_cutoff_values()

    # -- presets -----------------------------------------------------------

    @classmethod
    def argon(
        cls,
        *,
        cutoff: float = ARGON_CUTOFF,
        mode: str = "shifted_force",
        **kwargs,
    ) -> "LennardJones":
        """Standard Lennard-Jones argon.

        ``epsilon = 0.0103 eV`` (119.5 K), ``sigma = 3.405 A``, ``rc = 8.5 A``
        (2.496 sigma).  These are the Rahman/Verlet parameters that essentially
        all published LJ-argon results use, so observables computed with this
        preset are directly comparable with the literature.

        Parameters
        ----------
        cutoff : float
            Cutoff in angstrom.
        mode : str
            Cutoff treatment; the default gives continuous forces, which the
            NVE drift target in ``docs/design.md`` requires.
        **kwargs
            Forwarded to :class:`LennardJones`.

        Returns
        -------
        LennardJones
        """
        kwargs.setdefault("name", f"LJ-Ar({mode})")
        return cls(ARGON_EPSILON, ARGON_SIGMA, cutoff, mode=mode, **kwargs)

    # -- parameters --------------------------------------------------------

    @property
    def n_types(self) -> int:
        """Number of species this potential is parameterised for."""
        return self._n_types

    @property
    def r_min(self) -> float:
        """Distance of the potential minimum, ``2**(1/6) sigma``, in angstrom.

        Only defined for a single-species parameterisation.
        """
        if self._n_types != 1:
            raise ValueError("r_min is only defined for a single-species potential")
        return float(2.0 ** (1.0 / 6.0) * self.sigma[0, 0])

    def _raw(self, r, ti, tj):
        eps = self.epsilon[ti, tj]
        sig = self.sigma[ti, tj]
        sr6 = (sig / r) ** 6
        sr12 = sr6 * sr6
        u = 4.0 * eps * (sr12 - sr6)
        du = -24.0 * eps * (2.0 * sr12 - sr6) / r
        return u, du

    def _call_kernel(self, pi, pj, D, r, species, n_atoms, want_forces, want_virial):
        return _lj_kernel(
            pi,
            pj,
            D,
            r,
            species,
            self.epsilon,
            self.sigma,
            self._u_rc,
            self._du_rc,
            self.cutoff,
            self.r_on,
            self.mode_code,
            n_atoms,
            want_forces,
            want_virial,
        )

    # -- analytic long-range correction -----------------------------------

    def tail_energy_per_atom(self, density: float, *, type_pair: Sequence[int] = (0, 0)) -> float:
        """Mean-field estimate of the energy per atom missing beyond the cutoff.

        Assuming ``g(r) = 1`` for ``r > rc`` -- exact in the limit of large
        ``rc``, since the pair correlations of any short-ranged fluid or crystal
        average to the mean density there --

        ``e_tail = (rho/2) int_rc^inf 4 pi r^2 u(r) dr
                 = 8 pi rho eps [ sig^12 / (9 rc^9) - sig^6 / (3 rc^3) ]``

        The ``1/rc^3`` term dominates for any sensible cutoff, which is why the
        truncated fcc lattice energy converges only as ``rc^-3`` and has to be
        extrapolated rather than merely evaluated at a big cutoff.

        Parameters
        ----------
        density : float
            Number density in atoms / A^3.
        type_pair : sequence of int
            Which ``(i, j)`` parameter pair to use; only meaningful for a
            single-component estimate.

        Returns
        -------
        float
            Energy per atom in eV.  Negative for any ``rc`` beyond the
            repulsive core.
        """
        a, b = (int(t) for t in type_pair)
        eps = float(self.epsilon[a, b])
        sig = float(self.sigma[a, b])
        rc = self.cutoff
        return float(
            8.0 * math.pi * density * eps * (sig**12 / (9.0 * rc**9) - sig**6 / (3.0 * rc**3))
        )
