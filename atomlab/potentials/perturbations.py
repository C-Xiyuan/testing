"""Designed error fields ``delta_U`` -- the experimental instrument of this study.

``docs/theory.md`` argues that the error of an observable is controlled by
``Cov_0(A, delta_U)`` and not by any norm of ``delta_U`` or of its gradient.
That argument only becomes a *test* if we can build error fields whose norm and
whose covariance with a chosen observable are dialled independently.  This
module builds them.

Everything here is a :class:`~atomlab.potentials.base.Potential` with analytic
forces and virial, so a surrogate is literally ``U_0 + delta_U`` via
``SumPotential`` (``reference + perturbation``) and can be handed straight to
molecular dynamics, to the observable estimators, or to
:mod:`atomlab.analysis.response`.

The five constructions
----------------------

``RadialShellPerturbation``
    A Gaussian bump of amplitude ``a``, centre ``r0`` and width ``w`` in the
    pair energy.  Section 4.1 of ``docs/theory.md`` predicts

    ``RMSE_F ~ a / sqrt(w)``   and   ``|Cov(g, delta_U)| ~ a * w``

    so **at fixed force error** the damage to the radial distribution function
    scales as ``w**(3/2)``.  The family is parameterised for exactly that
    measurement by :meth:`RadialShellPerturbation.matched_force_error`, which
    solves for the amplitude giving a prescribed force RMSE on a reference
    ensemble.  Because every perturbation in this module is *linear* in its
    amplitude, that solve is one evaluation, not a root find.

``HighFrequencyPerturbation``
    ``a sin(2 pi r / lambda)`` under a smooth envelope.  Large gradient, tiny
    integral against any smooth observable: the concrete realisation of the
    "false alarm" failure mode of section 4.

``AngularPerturbation``
    A Stillinger-Weber-shaped three-body term.  It moves bond *angles* while
    leaving the pair distribution nearly untouched, which no pair-only
    perturbation can do.

``HighEnergyPerturbation``
    Supported only below ``r_onset``, a separation rarely sampled at the target
    temperature.  Invisible to an in-distribution test-set force error *and* to
    equilibrium observables, but divergent under compression -- the case of
    section 5 that argues for out-of-distribution probes rather than for better
    in-distribution norms.

``NullSpacePerturbation`` / ``AlignedPerturbation``
    The key construction, and its opposite.  Both expand ``delta_u(r)`` in a
    basis of narrow shell bumps ``b_k(r)``, measure the covariance
    ``c_lk = Cov(A_l, sum_pairs b_k)`` over reference samples, and then choose
    the coefficients either orthogonal to ``c`` (numerically zero covariance,
    large force error) or parallel to it (maximal covariance per unit force
    error).  The two at matched force error are the sharpest falsification test
    in the programme: they invert the usual ranking by construction.

Force error is measured in the *metric of the force error itself*
----------------------------------------------------------------
"Coefficients orthogonal to ``c``" is ambiguous until one says orthogonal in
which inner product.  The physically meaningful constraint is a fixed force
RMSE, and force RMSE is the quadratic form ``w^T G w`` with

``G_kl = (1 / n_dof) * sum_frames sum_atoms  F_k . F_l``

the Gram matrix of the per-basis-function force fields over the reference
frames.  This module therefore whitens with ``G`` first (:func:`whiten_gram`),
does all of its linear algebra in the coordinates where ``||y||_2`` *is* the
force RMSE, and transforms back.  The consequence is that
:class:`AlignedPerturbation` genuinely maximises covariance per unit force
error (``w ~ G^-1 c``, not ``w ~ c``), and that the null-space and aligned
members of a matched pair have *exactly* equal force RMSE on the reference set
rather than approximately equal.

Conventions
-----------
Metal units throughout: distances A, energies eV, forces eV/A, virial eV.
Pair displacements come from :func:`atomlab.neighbors.pair_vectors` and point
**from i to j**, ``D = r_j + shift @ cell - r_i``.  With ``U = sum_pairs u(r)``
the force on ``i`` is ``+u'(r) D / r`` and the virial is
``W_ab = -sum_pairs u'(r) D_a D_b / r``, matching
``atomlab/potentials/lennard_jones.py``.

Every perturbation and its first derivative vanish at ``cutoff``.  A bare
truncation would put a delta-function impulse in the force at ``rc`` (see the
``lennard_jones`` module docstring); since these objects are meant to be run in
MD, the smooth quintic switch :func:`smooth_cutoff` is not optional.

Sizing
------
The kernels are pure vectorised NumPy, with no ``numba``.  On the 32-64 atom
cells this study uses, a perturbation evaluation is well under a millisecond
and the perturbation is never the bottleneck next to the reference potential or
the model; a jitted kernel here would buy nothing and would need its own
reference implementation to be trusted.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..cell import check_minimum_image
from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Result
from ..units import beta as inverse_temperature
from .base import Potential

__all__ = [
    "smooth_cutoff",
    "Perturbation",
    "PairPerturbation",
    "RadialShellPerturbation",
    "HighFrequencyPerturbation",
    "HighEnergyPerturbation",
    "AngularPerturbation",
    "ShellBasis",
    "ShellDesign",
    "LinearShellPerturbation",
    "NullSpacePerturbation",
    "AlignedPerturbation",
    "RandomShellPerturbation",
    "build_shell_design",
    "whiten_gram",
    "observable_matrix",
    "observable_basis_covariance",
    "perturbation_covariance",
    "predicted_shift",
    "radial_coupling",
]


#: What the designed perturbations accept as a target observable: a callable on
#: a configuration returning a scalar or a length-``L`` vector.  An
#: already-evaluated ``(M, L)`` array is accepted everywhere a callable is (see
#: :func:`observable_matrix`), which is how several constructions share one
#: expensive observable.
ObservableFn = Callable[[Configuration], "float | np.ndarray"]


# --------------------------------------------------------------------------
# smooth cutoff
# --------------------------------------------------------------------------


def smooth_cutoff(r, r_on: float, r_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Quintic ``C^2`` switch that is 1 below ``r_on`` and 0 at ``r_cut``.

    With ``x = clip((r - r_on) / (r_cut - r_on), 0, 1)``,

    ``S(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5``,

    which satisfies ``S(0) = 1``, ``S(1) = 0`` and ``S' = S'' = 0`` at both
    ends.  The vanishing *second* derivative matters: a ``C^1`` switch leaves a
    kink in ``dF/dr``, which is invisible in an energy-conservation test but
    shows up as a spurious feature in finite-displacement force constants --
    and phonons are one of the observables this study measures.

    The clamp makes the zeros exact rather than merely small: ``x = 1`` gives
    ``S = 1 - 10 + 15 - 6 = 0`` and ``dS = (-30 + 60 - 30) / span = 0`` in exact
    arithmetic, so ``S`` and ``dS`` are identically zero at and beyond
    ``r_cut``.

    Parameters
    ----------
    r : array_like
        Pair distances in angstrom.
    r_on : float
        Radius at which the switch starts, in angstrom.
    r_cut : float
        Radius at which the switch reaches zero, in angstrom; must exceed
        ``r_on``.

    Returns
    -------
    s : ndarray
        Dimensionless switch value, same shape as ``r``.
    ds : ndarray
        ``dS/dr`` in 1/A, same shape as ``r``.
    """
    if not r_cut > r_on:
        raise ValueError(f"smooth_cutoff needs r_on < r_cut, got {r_on} and {r_cut}")
    r = np.asarray(r, dtype=np.float64)
    span = float(r_cut) - float(r_on)
    x = np.clip((r - r_on) / span, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    s = 1.0 - 10.0 * x3 + 15.0 * x3 * x - 6.0 * x3 * x2
    ds = (-30.0 * x2 + 60.0 * x3 - 30.0 * x3 * x) / span
    return s, ds


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not (v > 0.0 and math.isfinite(v)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


def _as_configurations(configurations) -> list[Configuration]:
    """Accept a single configuration or any iterable of them."""
    if isinstance(configurations, Configuration):
        return [configurations]
    out = list(configurations)
    if not out:
        raise ValueError("need at least one reference configuration")
    for cfg in out:
        if not isinstance(cfg, Configuration):
            raise TypeError(f"expected Configuration objects, got {type(cfg).__name__}")
    return out


# --------------------------------------------------------------------------
# shared base: anything whose energy is linear in one amplitude
# --------------------------------------------------------------------------


class Perturbation(Potential):
    """Base class for designed error fields.

    Every perturbation in this module is **linear in its amplitude**, which is
    the property that makes the whole experimental design cheap: matching a
    prescribed force RMSE is a division rather than a root find, and a family
    at fixed force error can be swept over any other parameter in one pass.

    Subclasses declare which attribute carries that linear scale via
    ``_scaling_attribute`` and must implement
    :meth:`~atomlab.potentials.base.Potential.compute`.

    Parameters
    ----------
    cutoff : float
        Interaction range in angstrom.  The perturbation and its first
        derivative are exactly zero at and beyond it.
    name : str
        Short identifier used in tables and figures.
    """

    #: Attribute holding the linear amplitude (see :meth:`rescaled`).
    _scaling_attribute: str = "amplitude"

    def force_rms(self, configurations) -> float:
        """Force RMSE of this error field over reference configurations.

        Computes ``sqrt( sum_frames sum_atoms |F|^2 / sum_frames 3 N )``, i.e.
        exactly the ``RMSE_F`` of ``docs/theory.md`` eq. (4.2) with the
        reference forces set to zero -- which is what a *perturbation's* force
        error is, since ``delta_F = -grad delta_U``.

        Parameters
        ----------
        configurations : Configuration or sequence of Configuration
            Reference-ensemble frames.

        Returns
        -------
        float
            Force RMSE in eV/A.
        """
        total = 0.0
        n_dof = 0
        for cfg in _as_configurations(configurations):
            f = self.compute(cfg, forces=True, virial=False).forces
            total += float(np.sum(f * f))
            n_dof += 3 * cfg.n_atoms
        return math.sqrt(total / n_dof)

    def rescaled(self, factor: float) -> "Perturbation":
        """Return a copy whose energy is multiplied by ``factor``.

        Exploits linearity in ``_scaling_attribute``; the returned object is a
        genuine instance of the same class (not a ``ScaledPotential``), so it
        keeps its constructor-specific diagnostics.
        """
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError(f"scale factor must be finite, got {factor}")
        out = copy.copy(self)
        current = getattr(self, self._scaling_attribute)
        setattr(out, self._scaling_attribute, current * factor)
        return out

    def matched_to_force_rms(self, target_force_rms: float, configurations) -> "Perturbation":
        """Return a copy whose force RMSE on ``configurations`` equals the target.

        Parameters
        ----------
        target_force_rms : float
            Desired force RMSE in eV/A, strictly positive.
        configurations : Configuration or sequence of Configuration
            Frames on which the force RMSE is defined.  Because the match is
            exact *in sample*, a held-out ensemble will reproduce it only to
            within sampling error -- report that separately rather than
            assuming it.

        Returns
        -------
        Perturbation
        """
        target = _positive(target_force_rms, "target_force_rms")
        current = self.force_rms(configurations)
        if current <= 0.0:
            raise ValueError(
                f"{self.name} produces no force on the reference configurations, so it "
                "cannot be matched to a nonzero force RMSE (its support probably does "
                "not overlap the sampled pair distances)"
            )
        return self.rescaled(target / current)


# --------------------------------------------------------------------------
# pair perturbations
# --------------------------------------------------------------------------


class PairPerturbation(Perturbation):
    """Isotropic two-body error field ``delta_U = sum_{i<j} delta_u(r_ij)``.

    Subclasses implement :meth:`pair`, which must return an energy and its
    radial derivative that are *exactly* zero at and beyond ``cutoff``.

    Parameters
    ----------
    cutoff : float
        Interaction range in angstrom.
    r_on : float, optional
        Onset of the smooth cutoff switch, in angstrom.  Defaults to
        ``0.85 * cutoff``.  Keeping the switch late means a shell bump centred
        well inside ``r_on`` is *exactly* the Gaussian it claims to be, which
        matters because the width-scaling prediction of theory section 4.1 is
        derived for an untruncated Gaussian.
    name : str
        Short identifier.
    """

    def __init__(self, cutoff: float, *, r_on: float | None = None, name: str = "pair_perturbation"):
        self.cutoff = _positive(cutoff, "cutoff")
        r_on = 0.85 * self.cutoff if r_on is None else float(r_on)
        if not 0.0 <= r_on < self.cutoff:
            raise ValueError(f"need 0 <= r_on < cutoff, got r_on={r_on}, cutoff={self.cutoff}")
        self.r_on = r_on
        self.name = str(name)

    # -- subclass hook -----------------------------------------------------

    def pair(self, r) -> tuple[np.ndarray, np.ndarray]:
        """``delta_u(r)`` in eV and ``d delta_u/dr`` in eV/A.

        Parameters
        ----------
        r : array_like
            Pair distances in angstrom.

        Returns
        -------
        u : ndarray
            Pair energy in eV, exactly zero for ``r >= cutoff``.
        du : ndarray
            Radial derivative in eV/A, exactly zero for ``r >= cutoff``.
        """
        raise NotImplementedError

    def envelope(self, r) -> tuple[np.ndarray, np.ndarray]:
        """The module's smooth cutoff switch evaluated for this perturbation."""
        return smooth_cutoff(r, self.r_on, self.cutoff)

    # -- evaluation --------------------------------------------------------

    def compute(self, configuration: Configuration, *, forces: bool = True, virial: bool = True) -> Result:
        """Energy (eV), forces (eV/A) and virial (eV) of the pair error field.

        Parameters
        ----------
        configuration : Configuration
            Geometry; not modified.
        forces, virial : bool
            Both are always computed -- the pair loop that gives the energy
            gives them for free -- but ``virial=False`` suppresses the returned
            tensor to honour the base-class contract.

        Returns
        -------
        Result
            ``energies`` carries the symmetric per-atom split
            ``u_i = 1/2 sum_j delta_u(r_ij)``.
        """
        if np.asarray(configuration.pbc).any():
            check_minimum_image(configuration.cell, configuration.pbc, self.cutoff, what=self.name)

        nl = build_neighbor_list(configuration, self.cutoff, half=True)
        D, r = pair_vectors(configuration, nl)
        n = configuration.n_atoms

        u, du = self.pair(r)
        energy = float(np.sum(u))

        energies = np.zeros(n)
        f = np.zeros((n, 3))
        w = np.zeros((3, 3))
        if r.size:
            np.add.at(energies, nl.i, 0.5 * u)
            np.add.at(energies, nl.j, 0.5 * u)
            # The neighbour list never returns r == 0, but a guarded divide
            # keeps a degenerate input failing loudly upstream rather than
            # silently producing NaN forces here.
            safe_r = np.where(r > 0.0, r, 1.0)
            fvec = (du / safe_r)[:, None] * D  # force on i; -fvec on j
            np.add.at(f, nl.i, fvec)
            np.add.at(f, nl.j, -fvec)
            # W_ab = -dU/d eps_ab = -sum_p u'(r) D_a D_b / r = -sum_p f_a D_b.
            w = -np.einsum("pa,pb->ab", fvec, D)

        return Result(
            energy=energy,
            forces=f,
            virial=w if virial else None,
            energies=energies,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, cutoff={self.cutoff:.3f})"


class RadialShellPerturbation(PairPerturbation):
    r"""A smooth Gaussian bump in the pair energy, localised in ``r``.

    .. math::

        \delta u(r) = a \, \exp\!\left[-\frac{(r-r_0)^2}{2 w^2}\right] S(r)

    with ``S`` the quintic switch of :func:`smooth_cutoff`.  Its coupling to
    ``g(r)`` is concentrated near ``r0``, so it moves a *chosen* part of the
    radial distribution function and leaves the rest alone.

    The scaling argument (``docs/theory.md`` section 4.1)
    ----------------------------------------------------
    The gradient of the bump is ``|delta u'| ~ a / w`` and the fraction of pairs
    inside the shell grows as ``w``, so

    ``RMSE_F^2 ~ (1/3N) * n_pairs_in_shell * (a/w)^2 ~ a^2 / w``  ->  ``RMSE_F ~ a / sqrt(w)``

    while the coupling to any smooth radial observable is an integral over the
    shell,

    ``|Cov_0(g, delta_U)| ~ integral 4 pi r^2 rho g(r) delta u(r) dr ~ a * w``.

    Eliminating ``a`` between the two gives the falsifiable prediction

    ``|Delta<g>| / RMSE_F  ~  w^{3/2}``

    -- unbounded in both directions, so two models can be built with identical
    force RMSE and observable errors differing by any factor one likes.  The
    experiment that measures this exponent needs a family at *fixed force
    error* parameterised by ``w``, which is exactly what
    :meth:`matched_force_error` returns.

    Parameters
    ----------
    r0 : float
        Centre of the shell in angstrom.
    width : float
        Gaussian standard deviation ``w`` in angstrom.
    amplitude : float
        Peak pair energy ``a`` in eV.  May be negative.
    cutoff : float
        Interaction range in angstrom.
    r_on : float, optional
        Onset of the smooth switch, in angstrom (default ``0.85 * cutoff``).
        For the scaling argument to apply cleanly the bump must be untruncated,
        i.e. ``r0 + 3 w < r_on``; this is not enforced, because a deliberately
        truncated bump is itself a useful probe, but it is what the
        width-scaling experiment should respect.
    name : str, optional
    """

    def __init__(
        self,
        r0: float,
        width: float,
        amplitude: float,
        cutoff: float,
        *,
        r_on: float | None = None,
        name: str | None = None,
    ):
        r0 = float(r0)
        if r0 < 0.0:
            raise ValueError(f"r0 must be >= 0 A, got {r0}")
        width = _positive(width, "width")
        amplitude = float(amplitude)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite, got {amplitude}")
        super().__init__(
            cutoff,
            r_on=r_on,
            name=name or f"shell(r0={r0:.2f},w={width:.3f})",
        )
        self.r0 = r0
        self.width = width
        self.amplitude = amplitude

    def pair(self, r) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=np.float64)
        z = (r - self.r0) / self.width
        # exp(-z^2/2) underflows smoothly to exactly 0.0 for large |z|, which is
        # the behaviour we want: no clipping, no spurious tail.
        g = np.exp(-0.5 * z * z)
        dg = -(z / self.width) * g
        s, ds = self.envelope(r)
        return self.amplitude * g * s, self.amplitude * (dg * s + g * ds)

    @classmethod
    def matched_force_error(
        cls,
        width: float,
        target_force_rms: float,
        reference_config,
        *,
        r0: float,
        cutoff: float,
        r_on: float | None = None,
        name: str | None = None,
    ) -> "RadialShellPerturbation":
        """Solve for the amplitude giving a prescribed force RMSE.

        This is the key instrument for the width-scaling experiment: sweeping
        ``width`` with ``target_force_rms`` held fixed produces a family of
        error fields that a force-RMSE-based evaluation cannot distinguish, and
        whose observable damage nevertheless spans orders of magnitude.

        The solve is exact and costs one force evaluation per reference frame,
        because the pair energy is linear in ``amplitude`` and hence so is the
        force field: ``a = target / RMSE_F(a = 1)``.

        Parameters
        ----------
        width : float
            Gaussian width ``w`` in angstrom.
        target_force_rms : float
            Desired force RMSE in eV/A over ``reference_config``.
        reference_config : Configuration or sequence of Configuration
            The ensemble on which the force error is defined.  A single
            configuration is accepted, but a handful of frames from the
            reference ensemble gives a match that transfers to held-out frames.
        r0 : float
            Shell centre in angstrom.
        cutoff : float
            Interaction range in angstrom.
        r_on : float, optional
            Switch onset in angstrom.
        name : str, optional

        Returns
        -------
        RadialShellPerturbation
            With ``amplitude`` solved so that
            ``perturbation.force_rms(reference_config) == target_force_rms``
            to round-off.

        Raises
        ------
        ValueError
            If the shell has no support on the reference configurations, i.e.
            no sampled pair distance lies inside the bump.  Silently returning
            a huge amplitude would poison every downstream number.
        """
        probe = cls(r0, width, 1.0, cutoff, r_on=r_on, name=name)
        current = probe.force_rms(reference_config)
        target = _positive(target_force_rms, "target_force_rms")
        if current <= 0.0:
            raise ValueError(
                f"a shell at r0={r0} A of width {width} A produces no force on the "
                "reference configurations: no sampled pair distance lies inside it, so "
                "no amplitude can reach the requested force RMSE"
            )
        return cls(r0, width, target / current, cutoff, r_on=r_on, name=name)


class HighFrequencyPerturbation(PairPerturbation):
    r"""An oscillatory pair error for testing gradient/projection mismatch.

    .. math::  \delta u(r) = a \, \sin(2\pi r / \lambda) \, S(r)

    Force error weights a radial oscillation's gradient, so shrinking
    ``lambda`` at fixed ``a`` can inflate ``RMSE_F``. Under additional
    smooth-coupling assumptions a declared observable may average over those
    oscillations. This class constructs that hypothesis for a bounded test; it
    does not imply that high-frequency errors are generally harmless or that
    fitted MLIPs occupy this direction.

    Parameters
    ----------
    wavelength : float
        Oscillation wavelength ``lambda`` in angstrom.
    amplitude : float
        Peak pair energy ``a`` in eV.
    cutoff : float
        Interaction range in angstrom.
    r_on : float, optional
        Switch onset (default ``0.85 * cutoff``).
    name : str, optional

    Notes
    -----
    The envelope is applied as a multiplicative switch rather than as a window
    that also suppresses short ``r``: the oscillation is meant to be present
    wherever pairs are, and truncating it at short range would turn it into a
    localised feature with a nonzero mean, which is precisely the thing it is
    supposed *not* to be.
    """

    def __init__(
        self,
        wavelength: float,
        amplitude: float,
        cutoff: float,
        *,
        r_on: float | None = None,
        name: str | None = None,
    ):
        wavelength = _positive(wavelength, "wavelength")
        super().__init__(
            cutoff, r_on=r_on, name=name or f"highfreq(lambda={wavelength:.3f})"
        )
        self.wavelength = wavelength
        self.amplitude = float(amplitude)
        self.wavenumber = 2.0 * math.pi / wavelength

    def pair(self, r) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=np.float64)
        k = self.wavenumber
        sn = np.sin(k * r)
        cs = np.cos(k * r)
        s, ds = self.envelope(r)
        return self.amplitude * sn * s, self.amplitude * (k * cs * s + sn * ds)


class HighEnergyPerturbation(PairPerturbation):
    r"""An error field supported only at rarely-sampled short separations.

    .. math::

        \delta u(r) = a \left(\frac{r_\mathrm{onset}}{r} - 1\right)^{3},
        \qquad r < r_\mathrm{onset}, \quad 0 \text{ otherwise}

    The cubic power is chosen so that ``delta_u``, ``delta_u'`` and
    ``delta_u''`` all vanish at ``r_onset``: the perturbation is ``C^2`` there
    with no switch needed, and it is *identically* zero above it.  As
    ``r -> 0`` it grows as ``r^{-3}``, so it "explodes" under compression or
    heating while remaining invisible at the target temperature.

    This is the case of ``docs/theory.md`` section 5 that argues for
    out-of-distribution probes rather than for better in-distribution norms:
    if ``r_onset`` sits below the inner edge of the first coordination shell at
    temperature ``T``, then *no* quantity computed on samples from that
    ensemble -- neither a test-set force RMSE nor any equilibrium observable --
    can see this error.  It is the honest steel-manning of force error, and
    also its clearest limitation.

    Parameters
    ----------
    r_onset : float
        Separation below which the perturbation turns on, in angstrom.  Choose
        it below the smallest pair distance the reference ensemble samples.
    amplitude : float
        Scale ``a`` in eV.  Positive gives a repulsive (stiffening) error.
    cutoff : float, optional
        Interaction range in angstrom, used only to size the neighbour search.
        Defaults to ``r_onset``.  Must be at least ``r_onset``; the support of
        the perturbation is ``[0, r_onset)`` regardless.
    name : str, optional
    """

    def __init__(
        self,
        r_onset: float,
        amplitude: float,
        cutoff: float | None = None,
        *,
        name: str | None = None,
    ):
        r_onset = _positive(r_onset, "r_onset")
        cutoff = r_onset if cutoff is None else _positive(cutoff, "cutoff")
        if cutoff < r_onset:
            raise ValueError(
                f"cutoff {cutoff} A must be >= r_onset {r_onset} A, otherwise the "
                "neighbour search would miss pairs the perturbation acts on"
            )
        # r_on is irrelevant here (no switch is used) but the base class wants a
        # valid value; put it at the cutoff-adjacent end so it can never bite.
        super().__init__(
            cutoff,
            r_on=0.999 * cutoff,
            name=name or f"highenergy(r_onset={r_onset:.2f})",
        )
        self.r_onset = r_onset
        self.amplitude = float(amplitude)

    def pair(self, r) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=np.float64)
        inside = r < self.r_onset
        # Evaluate on a clamped copy: r is strictly positive from the neighbour
        # list, but clamping keeps the r -> 0 divergence out of the masked-off
        # branch so that no inf * 0 = nan can appear.
        safe_r = np.where(inside, np.maximum(r, 1e-300), self.r_onset)
        x = self.r_onset / safe_r - 1.0
        u = self.amplitude * x**3
        du = self.amplitude * 3.0 * x * x * (-self.r_onset / (safe_r * safe_r))
        return np.where(inside, u, 0.0), np.where(inside, du, 0.0)


# --------------------------------------------------------------------------
# three-body angular perturbation
# --------------------------------------------------------------------------


class AngularPerturbation(Perturbation):
    r"""A three-body error field that moves bond angles, not bond lengths.

    Stillinger-Weber form, with the exponential envelope replaced by the
    quintic switch of :func:`smooth_cutoff`:

    .. math::

        \delta U = a \sum_i \sum_{j<k} S(r_{ij}) S(r_{ik})
                   \left(\cos\theta_{jik} - \cos\theta_0\right)^2

    Because ``S = 1`` below ``r_on``, the *radial* derivative of every term
    vanishes there and the perturbation exerts purely angular forces on the
    bulk of the neighbour shell.  Its effect on the pair distribution is
    therefore second order (it acts on ``g(r)`` only through the angular
    rearrangement it induces), while its effect on the bond-angle distribution
    is first order.  No pair-only error field can do that, which is why an
    observable-dependent notion of model quality is unavoidable: a model can be
    excellent for ``g(r)`` and poor for the angular distribution function.

    Derivatives
    -----------
    Write ``D_ij = r_j - r_i`` (the package-wide "from i to j" convention),
    ``u_ij = D_ij / r_ij`` and ``c = cos theta_jik = u_ij . u_ik``.  For one
    triplet term ``h = a S_ij S_ik (c - c0)^2``,

    ``dc / dD_ij = (u_ik - c u_ij) / r_ij``     (and symmetrically in ik)

    -- the projection of ``u_ik`` orthogonal to ``u_ij``, divided by ``r_ij``,
    which is the standard result and the one place a factor of ``r`` is easy to
    lose.  Hence

    ``g_ij = dh/dD_ij = a S'_ij S_ik (c-c0)^2 u_ij + (dh/dc) (u_ik - c u_ij)/r_ij``

    and by ``D_ij = r_j - r_i`` the forces are ``F_j = -g_ij``, ``F_k = -g_ik``
    and ``F_i = +(g_ij + g_ik)``, which manifestly sum to zero.  The virial
    follows from ``D -> (1+eps) D`` under strain:

    ``W_ab = -sum_triplets (g_ij,a D_ij,b + g_ik,a D_ik,b)``.

    All of this is checked against central differences in
    ``tests/test_perturbations.py``; a three-body derivative that is subtly
    wrong still trains and still runs, which is exactly the failure mode this
    repository exists to expose.

    Parameters
    ----------
    amplitude : float
        Scale ``a`` in eV.  Positive penalises departures from ``cos_theta0``.
    cos_theta0 : float
        Cosine of the preferred bond angle; ``-1/3`` is tetrahedral.
    cutoff : float
        Interaction range in angstrom.
    r_on : float, optional
        Switch onset (default ``0.85 * cutoff``).
    name : str, optional

    Notes
    -----
    The cost is ``O(N * n_neighbours^2)`` in pure NumPy with one vectorised
    pass per central atom.  For the 32-64 atom cells of this study that is a
    fraction of a millisecond; it is not sized for thousands of atoms and is
    not meant to be.
    """

    def __init__(
        self,
        amplitude: float,
        cos_theta0: float = -1.0 / 3.0,
        cutoff: float = 3.8,
        *,
        r_on: float | None = None,
        name: str | None = None,
    ):
        self.cutoff = _positive(cutoff, "cutoff")
        r_on = 0.85 * self.cutoff if r_on is None else float(r_on)
        if not 0.0 <= r_on < self.cutoff:
            raise ValueError(f"need 0 <= r_on < cutoff, got r_on={r_on}, cutoff={self.cutoff}")
        cos_theta0 = float(cos_theta0)
        if not -1.0 <= cos_theta0 <= 1.0:
            raise ValueError(f"cos_theta0 must lie in [-1, 1], got {cos_theta0}")
        self.r_on = r_on
        self.amplitude = float(amplitude)
        self.cos_theta0 = cos_theta0
        self.name = name or f"angular(cos0={cos_theta0:.3f})"

    def compute(self, configuration: Configuration, *, forces: bool = True, virial: bool = True) -> Result:
        """Energy (eV), forces (eV/A) and virial (eV) of the angular error field.

        Parameters
        ----------
        configuration : Configuration
            Geometry; not modified.
        forces, virial : bool
            Both are always computed; ``virial=False`` only suppresses the
            returned tensor.

        Returns
        -------
        Result
            ``energies`` assigns each triplet to its central atom.
        """
        if np.asarray(configuration.pbc).any():
            check_minimum_image(configuration.cell, configuration.pbc, self.cutoff, what=self.name)

        n = configuration.n_atoms
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        D, r = pair_vectors(configuration, nl)
        # The list is canonically sorted by (i, j, shift), so searchsorted gives
        # CSR offsets for free -- the same trick the Stillinger-Weber module uses.
        first = np.searchsorted(nl.i, np.arange(n + 1))

        s_env, ds_env = (smooth_cutoff(r, self.r_on, self.cutoff) if r.size
                         else (np.zeros(0), np.zeros(0)))

        energy = 0.0
        energies = np.zeros(n)
        f = np.zeros((n, 3))
        w = np.zeros((3, 3))
        a = self.amplitude
        c0 = self.cos_theta0

        for i in range(n):
            lo, hi = int(first[i]), int(first[i + 1])
            if hi - lo < 2:
                continue
            # Pairs at or beyond the cutoff have S = S' = 0 and contribute
            # nothing; dropping them is a pure optimisation.
            local = np.flatnonzero(s_env[lo:hi] > 0.0) + lo
            if local.size < 2:
                continue
            ia, ib = np.triu_indices(local.size, k=1)
            m = local[ia]
            q = local[ib]

            dij, dik = D[m], D[q]
            rij, rik = r[m], r[q]
            sij, sik = s_env[m], s_env[q]
            dsij, dsik = ds_env[m], ds_env[q]

            uij = dij / rij[:, None]
            uik = dik / rik[:, None]
            cos_t = np.einsum("pa,pa->p", uij, uik)
            delta = cos_t - c0
            delta2 = delta * delta

            h = a * sij * sik * delta2
            dh_dcos = 2.0 * a * sij * sik * delta

            # Radial part (from the switch only) + angular part.
            gij = (a * dsij * sik * delta2)[:, None] * uij + (dh_dcos / rij)[:, None] * (
                uik - cos_t[:, None] * uij
            )
            gik = (a * sij * dsik * delta2)[:, None] * uik + (dh_dcos / rik)[:, None] * (
                uij - cos_t[:, None] * uik
            )

            total_h = float(h.sum())
            energy += total_h
            energies[i] += total_h

            np.add.at(f, nl.j[m], -gij)
            np.add.at(f, nl.j[q], -gik)
            f[i] += (gij + gik).sum(axis=0)
            w -= np.einsum("pa,pb->ab", gij, dij) + np.einsum("pa,pb->ab", gik, dik)

        return Result(
            energy=energy,
            forces=f,
            virial=w if virial else None,
            energies=energies,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AngularPerturbation(name={self.name!r}, amplitude={self.amplitude:g}, "
            f"cos_theta0={self.cos_theta0:g}, cutoff={self.cutoff:.3f})"
        )


# --------------------------------------------------------------------------
# shell basis and the designed (null-space / aligned) perturbations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellBasis:
    """A basis of narrow Gaussian shell bumps spanning the cutoff range.

    Attributes
    ----------
    centers : ndarray, shape (K,)
        Shell centres in angstrom.
    widths : ndarray, shape (K,)
        Gaussian widths in angstrom.
    cutoff : float
        Interaction range in angstrom.
    r_on : float
        Onset of the shared smooth switch, in angstrom.  The switch is part of
        the basis function, so every linear combination automatically vanishes
        with its derivative at ``cutoff``.
    """

    centers: np.ndarray
    widths: np.ndarray
    cutoff: float
    r_on: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centers", np.ascontiguousarray(self.centers, dtype=np.float64))
        object.__setattr__(self, "widths", np.ascontiguousarray(self.widths, dtype=np.float64))
        if self.centers.ndim != 1 or self.centers.shape != self.widths.shape:
            raise ValueError("centers and widths must be 1-D arrays of the same length")
        if self.centers.size == 0:
            raise ValueError("shell basis needs at least one function")
        if np.any(self.widths <= 0.0):
            raise ValueError("all shell widths must be > 0 A")
        if not self.cutoff > self.r_on >= 0.0:
            raise ValueError(f"need 0 <= r_on < cutoff, got {self.r_on} and {self.cutoff}")

    @property
    def n_functions(self) -> int:
        """Number of basis functions ``K``."""
        return int(self.centers.size)

    def evaluate(self, r) -> tuple[np.ndarray, np.ndarray]:
        """Basis values and radial derivatives at pair distances ``r``.

        Parameters
        ----------
        r : array_like, shape (P,)
            Pair distances in angstrom.

        Returns
        -------
        b : ndarray, shape (P, K)
            ``b_k(r)`` (dimensionless; multiply by a weight in eV), exactly
            zero for ``r >= cutoff``.
        db : ndarray, shape (P, K)
            ``d b_k / dr`` in 1/A, exactly zero for ``r >= cutoff``.
        """
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        z = (r[:, None] - self.centers[None, :]) / self.widths[None, :]
        g = np.exp(-0.5 * z * z)
        dg = -(z / self.widths[None, :]) * g
        s, ds = smooth_cutoff(r, self.r_on, self.cutoff)
        return g * s[:, None], dg * s[:, None] + g * ds[:, None]

    @classmethod
    def uniform(
        cls,
        cutoff: float,
        n_functions: int = 24,
        *,
        r_min: float,
        r_max: float | None = None,
        r_on: float | None = None,
        width_factor: float = 1.0,
    ) -> "ShellBasis":
        """Evenly spaced shells with overlapping widths.

        Parameters
        ----------
        cutoff : float
            Interaction range in angstrom.
        n_functions : int
            Number of shells ``K``.  It must exceed the number of observable
            components for a null space to exist at all.
        r_min : float
            Centre of the innermost shell in angstrom.  Put it at (or just
            below) the smallest pair distance the reference ensemble samples:
            shells with no support carry no force, and a projection that put
            weight on them would be rescaled to a meaningless amplitude.
        r_max : float, optional
            Centre of the outermost shell (default ``r_on``).
        r_on : float, optional
            Switch onset (default ``0.85 * cutoff``).
        width_factor : float
            Width in units of the shell spacing.  ``1.0`` gives neighbouring
            shells crossing near ``exp(-1/2)``, i.e. a smooth, well-conditioned
            partition of the range without being so broad that the basis loses
            radial resolution.

        Returns
        -------
        ShellBasis
        """
        cutoff = _positive(cutoff, "cutoff")
        r_on = 0.85 * cutoff if r_on is None else float(r_on)
        r_max = r_on if r_max is None else float(r_max)
        n_functions = int(n_functions)
        if n_functions < 2:
            raise ValueError(f"need at least 2 shell functions, got {n_functions}")
        if not 0.0 <= r_min < r_max:
            raise ValueError(f"need 0 <= r_min < r_max, got {r_min} and {r_max}")
        centers = np.linspace(float(r_min), r_max, n_functions)
        spacing = centers[1] - centers[0]
        widths = np.full(n_functions, _positive(width_factor, "width_factor") * spacing)
        return cls(centers=centers, widths=widths, cutoff=cutoff, r_on=r_on)

    @classmethod
    def for_configurations(
        cls,
        configurations,
        cutoff: float,
        n_functions: int = 24,
        *,
        r_on: float | None = None,
        width_factor: float = 1.0,
        margin: float = 0.98,
    ) -> "ShellBasis":
        """Build a uniform basis whose inner edge tracks the sampled pair distances.

        Placing ``r_min`` by hand is the easiest way to end up with dead basis
        functions.  This scans the reference frames for the smallest pair
        separation actually present and starts the basis just below it.

        Parameters
        ----------
        configurations : Configuration or sequence of Configuration
        cutoff : float
            Interaction range in angstrom.
        n_functions : int
        r_on : float, optional
        width_factor : float
        margin : float
            Fraction of the observed minimum pair distance at which the
            innermost shell is centred.

        Returns
        -------
        ShellBasis
        """
        cutoff = _positive(cutoff, "cutoff")
        r_min = np.inf
        for cfg in _as_configurations(configurations):
            nl = build_neighbor_list(cfg, cutoff, half=True)
            _, r = pair_vectors(cfg, nl)
            if r.size:
                r_min = min(r_min, float(r.min()))
        if not np.isfinite(r_min):
            raise ValueError("no pairs inside the cutoff in any reference configuration")
        return cls.uniform(
            cutoff,
            n_functions,
            r_min=margin * r_min,
            r_on=r_on,
            width_factor=width_factor,
        )


class LinearShellPerturbation(PairPerturbation):
    """A pair error field expanded in a :class:`ShellBasis`.

    ``delta_u(r) = sum_k w_k b_k(r)``, with ``w`` in eV.  This is the generic
    container the designed perturbations return; it is also useful on its own
    for hand-built error fields.

    Parameters
    ----------
    basis : ShellBasis
    weights : array_like, shape (K,)
        Coefficients in eV.
    name : str, optional
    """

    _scaling_attribute = "weights"

    def __init__(self, basis: ShellBasis, weights, *, name: str = "shell_combination"):
        super().__init__(basis.cutoff, r_on=basis.r_on, name=name)
        self.basis = basis
        weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64)).reshape(-1)
        if weights.shape[0] != basis.n_functions:
            raise ValueError(
                f"weights has length {weights.shape[0]} but the basis has "
                f"{basis.n_functions} functions"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must all be finite")
        self.weights = weights

    def pair(self, r) -> tuple[np.ndarray, np.ndarray]:
        b, db = self.basis.evaluate(r)
        return b @ self.weights, db @ self.weights


@dataclass
class ShellDesign:
    """Everything a designed perturbation needs from the reference ensemble.

    Attributes
    ----------
    basis : ShellBasis
    energies : ndarray, shape (M, K)
        ``delta_U_k`` for each frame at unit weight, in eV -- i.e.
        ``sum_{i<j} b_k(r_ij)``.  The columns are the "basis error fields"
        whose covariance with the observable is being controlled.
    force_gram : ndarray, shape (K, K)
        ``G_kl = (1/n_dof) sum_frames sum_atoms F_k . F_l`` in (eV/A)^2, with
        ``F_k`` the force field of basis function ``k``.  The force RMSE of a
        weight vector is exactly ``sqrt(w^T G w)``, which is what makes matched
        force error an equality rather than an approximation.
    n_dof : int
        ``sum_frames 3 N``.
    n_frames : int
    """

    basis: ShellBasis
    energies: np.ndarray
    force_gram: np.ndarray
    n_dof: int
    n_frames: int

    def force_rms(self, weights) -> float:
        """Force RMSE in eV/A of ``delta_u = sum_k w_k b_k`` over the design frames."""
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        value = float(w @ self.force_gram @ w)
        # A Gram matrix is positive semidefinite; a tiny negative value is pure
        # round-off, and clipping it is safe (a genuinely negative value would
        # be a bug, and would be many orders of magnitude larger).
        return math.sqrt(max(value, 0.0))

    def perturbation_energies(self, weights) -> np.ndarray:
        """``(M,)`` values of ``delta_U`` on the design frames, in eV."""
        return self.energies @ np.asarray(weights, dtype=np.float64).reshape(-1)


def build_shell_design(configurations, basis: ShellBasis) -> ShellDesign:
    """Evaluate the shell basis and its force Gram matrix over reference frames.

    One pass over the frames produces both the ``(M, K)`` matrix of basis
    energies and the ``(K, K)`` force Gram matrix.  Doing them together is not
    just an optimisation: the Gram matrix is what defines "force error" as an
    inner product on coefficient space, and it must be measured on exactly the
    frames the covariance is measured on for the matched-force-error claim to
    be exact.

    Parameters
    ----------
    configurations : Configuration or sequence of Configuration
        Reference-ensemble frames.
    basis : ShellBasis

    Returns
    -------
    ShellDesign
    """
    cfgs = _as_configurations(configurations)
    k = basis.n_functions
    energies = np.zeros((len(cfgs), k))
    gram = np.zeros((k, k))
    n_dof = 0

    for m, cfg in enumerate(cfgs):
        if np.asarray(cfg.pbc).any():
            check_minimum_image(cfg.cell, cfg.pbc, basis.cutoff, what="shell basis")
        nl = build_neighbor_list(cfg, basis.cutoff, half=True)
        D, r = pair_vectors(cfg, nl)
        n = cfg.n_atoms
        n_dof += 3 * n
        if r.size == 0:
            continue
        b, db = basis.evaluate(r)
        energies[m] = b.sum(axis=0)
        # (P, K*3) pair force contributions; the same +f on i, -f on j split as
        # in PairPerturbation.compute, so the two agree to round-off.
        fvec = ((db / r[:, None])[:, :, None] * D[:, None, :]).reshape(r.size, k * 3)
        # Scatter-add with bincount rather than np.add.at: the latter is an
        # unbuffered ufunc call and is ~50x slower here, and the design pass is
        # run over hundreds of frames.
        forces_k = np.empty((n, k * 3))
        for col in range(k * 3):
            wcol = fvec[:, col]
            forces_k[:, col] = np.bincount(nl.i, weights=wcol, minlength=n) - np.bincount(
                nl.j, weights=wcol, minlength=n
            )
        forces_k = forces_k.reshape(n, k, 3)
        gram += np.einsum("nka,nla->kl", forces_k, forces_k)

    gram /= n_dof
    # Symmetrise: the einsum is symmetric in exact arithmetic, and enforcing it
    # keeps the later eigendecomposition from picking up a spurious imaginary
    # part from round-off asymmetry.
    gram = 0.5 * (gram + gram.T)
    return ShellDesign(
        basis=basis, energies=energies, force_gram=gram, n_dof=n_dof, n_frames=len(cfgs)
    )


def whiten_gram(force_gram: np.ndarray, *, rcond: float = 1e-10) -> np.ndarray:
    """Transform to coordinates in which the force RMSE is the Euclidean norm.

    Returns ``T`` (shape ``(K, K')``) with ``T^T G T = I``, obtained from the
    eigendecomposition ``G = Q L Q^T`` as ``T = Q_keep L_keep^{-1/2}``.
    Directions with ``lambda <= rcond * lambda_max`` are dropped: they are basis
    functions (or combinations of them) that exert essentially no force on the
    reference frames, and keeping them would let the null-space projection hide
    in a direction that is invisible to the force-error metric -- which would
    make the whole matched-force-error comparison vacuous.

    Parameters
    ----------
    force_gram : ndarray, shape (K, K)
        Positive semidefinite Gram matrix in (eV/A)^2.
    rcond : float
        Relative eigenvalue floor.

    Returns
    -------
    ndarray, shape (K, K')
        Whitening transform; ``w = T y`` has force RMSE ``||y||_2``.
    """
    g = np.asarray(force_gram, dtype=np.float64)
    lam, q = np.linalg.eigh(0.5 * (g + g.T))
    lam_max = float(lam.max()) if lam.size else 0.0
    if lam_max <= 0.0:
        raise ValueError(
            "the shell basis exerts no force on the reference configurations; "
            "check that the basis range overlaps the sampled pair distances"
        )
    keep = lam > rcond * lam_max
    if not np.any(keep):
        raise ValueError("no well-conditioned force directions survive the rcond cut")
    return q[:, keep] / np.sqrt(lam[keep])[None, :]


def observable_matrix(observable, configurations) -> np.ndarray:
    """Evaluate a scalar or vector observable on reference frames.

    Parameters
    ----------
    observable : callable or array_like
        ``A(configuration) -> float`` or ``-> ndarray of shape (L,)``.  An
        already-evaluated ``(M,)`` or ``(M, L)`` array is accepted as-is, which
        matters when several constructions share one expensive observable: a
        binned ``g(r)`` over a thousand frames costs more than the whole
        covariance design, and recomputing it per construction is the easiest
        way to make this module look slow.
    configurations : Configuration or sequence of Configuration

    Returns
    -------
    ndarray, shape (M, L)
        Always 2-D; a scalar observable gives ``L = 1``.
    """
    cfgs = _as_configurations(configurations)
    if not callable(observable):
        arr = np.atleast_2d(np.asarray(observable, dtype=np.float64))
        if arr.shape[0] == 1 and len(cfgs) != 1:
            arr = arr.T
        if arr.shape[0] != len(cfgs):
            raise ValueError(
                f"precomputed observable has {arr.shape[0]} rows but {len(cfgs)} "
                "configurations were given"
            )
        return arr
    rows = [np.atleast_1d(np.asarray(observable(cfg), dtype=np.float64)).reshape(-1) for cfg in cfgs]
    lengths = {row.size for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"observable returned inconsistent lengths across frames: {sorted(lengths)}")
    return np.vstack(rows)


def observable_basis_covariance(a_samples: np.ndarray, basis_energies: np.ndarray) -> np.ndarray:
    """Sample covariance ``Cov(A_l, delta_U_k)`` over frames.

    Parameters
    ----------
    a_samples : ndarray, shape (M,) or (M, L)
        Observable values on the frames.
    basis_energies : ndarray, shape (M, K)
        Basis error-field energies in eV.

    Returns
    -------
    ndarray, shape (L, K)
        Covariance with ``ddof = 1``; units are those of ``A`` times eV.
    """
    a = np.atleast_2d(np.asarray(a_samples, dtype=np.float64))
    if a.shape[0] == 1 and basis_energies.shape[0] != 1:
        a = a.T
    b = np.asarray(basis_energies, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"observable has {a.shape[0]} frames, basis has {b.shape[0]}")
    m = a.shape[0]
    if m < 2:
        raise ValueError("a covariance needs at least two frames")
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    return (ac.T @ bc) / (m - 1)


def perturbation_covariance(potential: Potential, observable, configurations) -> np.ndarray:
    """``Cov_0(A, delta_U)`` measured directly, by evaluating both on frames.

    This is the honest arbiter for the null-space construction: the designed
    perturbation is orthogonal *by construction* in the basis coordinates, so
    the claim has to be checked by evaluating the potential itself, on the
    design frames and (more informatively) on held-out ones.

    Parameters
    ----------
    potential : Potential
        The error field ``delta_U``.
    observable : callable or array_like
        ``A(configuration) -> float`` or ``(L,)``, or a precomputed ``(M, L)``
        array of observable values on the same frames.
    configurations : Configuration or sequence of Configuration

    Returns
    -------
    ndarray, shape (L,)
        Covariance in (units of A) * eV.
    """
    cfgs = _as_configurations(configurations)
    a = observable_matrix(observable, cfgs)
    du = np.array([potential.energy(cfg) for cfg in cfgs])
    return observable_basis_covariance(a, du[:, None])[:, 0]


def predicted_shift(
    potential: Potential,
    observable,
    configurations,
    temperature: float,
) -> np.ndarray:
    """First-order predicted observable shift ``-beta Cov_0(A, delta_U)``.

    Result 1a of ``docs/theory.md``.  Provided here so that a designed
    perturbation can report the damage it is *supposed* to do without pulling
    in :mod:`atomlab.analysis.response` (which owns the error bars, the
    second-order remainder and the reweighting cross-check -- use it for
    anything beyond a quick diagnostic).

    Parameters
    ----------
    potential : Potential
    observable : callable
    configurations : Configuration or sequence of Configuration
    temperature : float
        Temperature in kelvin.

    Returns
    -------
    ndarray, shape (L,)
        Predicted ``Delta<A>`` in units of ``A``.
    """
    return -inverse_temperature(temperature) * perturbation_covariance(
        potential, observable, configurations
    )


def _solve_weights(
    covariance: np.ndarray,
    transform: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    *,
    rcond: float,
) -> tuple[np.ndarray, int, int]:
    """Choose basis coefficients in the whitened (force-metric) coordinates.

    Returns ``(y, rank, n_null)`` with ``y`` a unit vector in whitened space
    (so that a later rescale to a target force RMSE is a plain multiplication).
    """
    m_tilde = np.asarray(covariance, dtype=np.float64) @ transform  # (L, K')
    k_prime = transform.shape[1]
    u, sv, vt = np.linalg.svd(m_tilde, full_matrices=True)
    rank = int(np.sum(sv > (sv.max() * rcond))) if sv.size else 0
    n_null = k_prime - rank

    if rank == 0:
        raise ValueError(
            "the target observable has no measurable covariance with any shell basis "
            "function on these frames, so neither an orthogonal nor an aligned "
            "construction is meaningful; use a basis that overlaps the observable's "
            "support, or more frames"
        )

    if mode == "null":
        if n_null == 0:
            raise ValueError(
                f"the covariance matrix has full rank {rank} in a {k_prime}-dimensional "
                "well-conditioned basis, so the null space is empty; use more shell "
                "functions than observable components"
            )
        null_basis = vt[rank:].T  # (K', n_null)
        y = null_basis @ rng.standard_normal(n_null)
    elif mode == "aligned":
        # In whitened coordinates ||y|| IS the force RMSE, so the direction
        # maximising ||M y|| per unit force error is the top right singular
        # vector.  Transformed back this is w ~ G^-1 c for a scalar observable
        # -- not w ~ c, which would only be optimal if force error happened to
        # be the Euclidean norm on coefficients.  It is not.
        y = vt[0].copy()
    elif mode == "random":
        y = rng.standard_normal(k_prime)
    else:  # pragma: no cover - guarded by the subclasses
        raise ValueError(f"unknown mode {mode!r}")

    norm = float(np.linalg.norm(y))
    if norm <= 0.0:
        raise ValueError(f"degenerate weight vector for mode {mode!r}")
    return y / norm, rank, n_null


class _DesignedShellPerturbation(LinearShellPerturbation):
    """Shared machinery for the covariance-designed perturbations.

    Subclasses set ``_mode`` to ``"null"``, ``"aligned"`` or ``"random"``.
    """

    _mode: str = "random"

    def __init__(
        self,
        configurations,
        observable,
        temperature: float,
        *,
        cutoff: float,
        basis: ShellBasis | None = None,
        n_functions: int = 24,
        target_force_rms: float | None = 0.05,
        seed: int | np.random.Generator = 0,
        rcond: float = 1e-10,
        r_on: float | None = None,
        width_factor: float = 1.0,
        name: str | None = None,
        design: ShellDesign | None = None,
    ):
        cfgs = _as_configurations(configurations)
        if basis is None:
            basis = (
                design.basis
                if design is not None
                else ShellBasis.for_configurations(
                    cfgs, cutoff, n_functions, r_on=r_on, width_factor=width_factor
                )
            )
        if design is None:
            design = build_shell_design(cfgs, basis)
        elif design.basis is not basis:
            raise ValueError("the supplied design was built for a different basis")

        a_samples = observable_matrix(observable, cfgs)
        covariance = observable_basis_covariance(a_samples, design.energies)  # (L, K)
        transform = whiten_gram(design.force_gram, rcond=rcond)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        y, rank, n_null = _solve_weights(covariance, transform, self._mode, rng, rcond=rcond)
        if target_force_rms is not None:
            y = y * _positive(target_force_rms, "target_force_rms")
        weights = transform @ y

        super().__init__(basis, weights, name=name or f"{self._mode}_space")

        self.temperature = float(temperature)
        self.design = design
        self.covariance = covariance
        self.observable_dim = int(a_samples.shape[1])
        self.rank = rank
        self.null_dimension = n_null
        #: Number of basis directions that survived the force-Gram rcond cut,
        #: i.e. the dimension the construction actually had to work in.
        self.n_well_conditioned = int(transform.shape[1])
        self.seed = seed

    # -- diagnostics -------------------------------------------------------

    @property
    def design_force_rms(self) -> float:
        """Force RMSE in eV/A on the design frames (exact, from the Gram matrix)."""
        return self.design.force_rms(self.weights)

    @property
    def design_covariance(self) -> np.ndarray:
        """``(L,)`` in-sample ``Cov(A, delta_U)``, from the design matrices."""
        return self.covariance @ self.weights

    def report(self) -> dict:
        """A dictionary of everything a reader would need to trust this object."""
        cov = self.design_covariance
        return {
            "name": self.name,
            "mode": self._mode,
            "temperature": self.temperature,
            "n_frames": self.design.n_frames,
            "n_basis": self.basis.n_functions,
            "n_well_conditioned": self.n_well_conditioned,
            "observable_dim": self.observable_dim,
            "covariance_rank": self.rank,
            "null_dimension": self.null_dimension,
            "force_rms": self.design_force_rms,
            "covariance_norm": float(np.linalg.norm(cov)),
            "predicted_shift_norm": float(
                inverse_temperature(self.temperature) * np.linalg.norm(cov)
            ),
        }


class NullSpacePerturbation(_DesignedShellPerturbation):
    """Large force error, numerically zero covariance with a target observable.

    **The key construction of the programme.**  Given reference frames, a
    target observable ``A`` and a temperature:

    1. lay a basis of narrow shell bumps ``b_k(r)`` across the cutoff range;
    2. evaluate ``c_lk = Cov(A_l, sum_pairs b_k(r_ij))`` over the frames;
    3. whiten with the force Gram matrix so that "unit coefficient norm" means
       "unit force RMSE" (see the module docstring);
    4. project a random vector onto the null space of ``c`` in those
       coordinates, and rescale to the requested force RMSE.

    By construction ``Cov_0(A, delta_U) = 0`` to round-off on the design
    frames, while ``RMSE_F`` is whatever was asked for.  Result 1a of
    ``docs/theory.md`` then predicts ``Delta<A> = 0`` at first order despite an
    arbitrarily large force error -- and falsification criterion 3 of section 8
    is precisely the failure of that prediction.

    Honesty about in-sample versus out-of-sample
    --------------------------------------------
    The orthogonality is exact *on the frames it was fitted to*.  On held-out
    frames the residual covariance is finite and set by the sampling error of
    ``c``, i.e. it falls as ``1/sqrt(M)``.  Both numbers should be reported;
    :func:`perturbation_covariance` computes either, and the suppression factor
    against a random perturbation of equal force error is the figure of merit.

    Parameters
    ----------
    configurations : Configuration or sequence of Configuration
        Reference-ensemble frames.  More frames give better out-of-sample
        suppression.
    observable : callable or array_like
        ``A(configuration) -> float`` or ``-> ndarray (L,)``, or an
        already-evaluated ``(M, L)`` array.  A vector observable such as a
        binned ``g(r)`` is handled by taking the null space of the whole
        ``(L, K)`` covariance matrix, which requires ``K > L``.
    temperature : float
        Temperature in kelvin.  Used for the reported first-order shift; it
        does not enter the construction, because ``Cov = 0`` kills the
        prediction at any temperature.
    cutoff : float
        Interaction range in angstrom (ignored if ``basis`` is given).
    basis : ShellBasis, optional
        Supply one to share a basis (and hence an exactly comparable force
        metric) across several constructions.
    n_functions : int
        Number of shells when the basis is built automatically.
    target_force_rms : float, optional
        Force RMSE in eV/A to match on the design frames.  ``None`` leaves the
        coefficients at unit norm in whitened coordinates, i.e. force RMSE 1.
    seed : int or numpy.random.Generator
        Seed for the random vector that is projected.  Required for
        determinism; there is no module-level RNG anywhere in this package.
    rcond : float
        Relative floor for both the force-Gram eigenvalues and the covariance
        singular values.
    r_on, width_factor, name, design
        Passed through to the basis / design construction.

    Attributes
    ----------
    covariance : ndarray, shape (L, K)
        The measured ``Cov(A_l, delta_U_k)`` the projection was built from.
    null_dimension : int
        Dimension of the null space that was available.
    """

    _mode = "null"


class AlignedPerturbation(_DesignedShellPerturbation):
    """Maximal covariance with a target observable per unit force error.

    The complement of :class:`NullSpacePerturbation`, and the other half of the
    sharpest falsification test in the programme: a *pair* of error fields with
    identical force RMSE whose observable damage differs by orders of
    magnitude, which inverts the ranking that force RMSE would assign.

    Because force RMSE is the quadratic form ``w^T G w`` and not ``||w||^2``,
    "parallel to ``c``" is the wrong direction; the maximiser of
    ``|c . w| / sqrt(w^T G w)`` is ``w ~ G^{-1} c``.  This class computes that
    (via the whitening transform, so the ill-conditioned directions of ``G``
    are dropped rather than inverted).  For a vector observable it takes the
    direction maximising ``||C w||_2`` per unit force error, i.e. the top right
    singular vector of the whitened covariance matrix.

    Parameters
    ----------
    Identical to :class:`NullSpacePerturbation`.
    """

    _mode = "aligned"


class RandomShellPerturbation(_DesignedShellPerturbation):
    """A random direction in the same basis at the same force error.

    The control arm.  A designed perturbation is only interesting relative to
    what an *undesigned* error field of the same size does to the observable,
    so the suppression and amplification factors quoted in the experiments are
    both measured against this.

    Parameters
    ----------
    Identical to :class:`NullSpacePerturbation`.  The observable is still
    required, because the class shares the design pass and reports the same
    diagnostics -- it simply ignores the covariance when choosing a direction.
    """

    _mode = "random"


# --------------------------------------------------------------------------
# radial coupling diagnostic
# --------------------------------------------------------------------------


def radial_coupling(
    perturbation: PairPerturbation,
    *,
    density: float = 1.0,
    g_of_r: Callable[[np.ndarray], np.ndarray] | None = None,
    r_min: float = 0.0,
    n_quad: int = 512,
) -> float:
    r"""Mean-field coupling of a pair error field to the radial distribution.

    .. math::

        C = \tfrac12 \int_{r_\mathrm{min}}^{R_c} 4\pi r^2 \rho\, g(r)\,
            \delta u(r)\, \mathrm{d}r

    This is the pair-energy shift per atom that ``delta_u`` produces in a fluid
    of density ``rho`` with radial distribution ``g``, and it is the quantity
    that appears (up to the ``-beta`` and the fluctuation structure) in the
    coupling ``|Cov_0(g, delta_U)| ~ a w`` of ``docs/theory.md`` eq. (4.4).
    It is deterministic, so the width-scaling exponent can be measured without
    the sampling noise of a covariance estimate on top of it.

    ``g_of_r = None`` uses the ideal-gas weighting ``g = 1``, which is the
    limit in which the scaling argument of section 4.1 is derived.  Passing a
    measured ``g(r)`` gives the same exponent with a system-specific prefactor.

    Parameters
    ----------
    perturbation : PairPerturbation
        The error field.
    density : float
        Number density in atoms / A^3.
    g_of_r : callable, optional
        ``g(r)`` for an array of distances in angstrom.
    r_min : float
        Lower integration limit in angstrom (set it to the excluded-volume
        radius to avoid integrating over separations no pair ever reaches).
    n_quad : int
        Gauss-Legendre nodes.  The integrand is smooth but can be narrow (a
        shell of width 0.05 A inside a 8 A range), so the default is generous;
        the test suite checks convergence against a doubled node count.

    Returns
    -------
    float
        Coupling in eV per atom.
    """
    if not isinstance(perturbation, PairPerturbation):
        raise TypeError("radial_coupling is defined for pair perturbations only")
    lo = float(r_min)
    hi = float(perturbation.cutoff)
    if not hi > lo:
        raise ValueError(f"need r_min < cutoff, got {lo} and {hi}")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    r = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    jac = 0.5 * (hi - lo)
    u, _ = perturbation.pair(r)
    g = np.ones_like(r) if g_of_r is None else np.asarray(g_of_r(r), dtype=np.float64)
    integrand = 4.0 * np.pi * r * r * float(density) * g * u
    return 0.5 * float(np.sum(weights * integrand) * jac)
