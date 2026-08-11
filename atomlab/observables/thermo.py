"""Thermodynamic and mechanical observables: EOS, elastic response, Cv, alpha.

These are the observables a practitioner reports when asked whether a potential
is any good, and they probe the potential in ways that force error on a
thermally-sampled test set does not:

* the **equation of state** probes the energy over a range of densities that a
  room-temperature training set never visits;
* the **elastic constants** are second derivatives of the energy with respect to
  a *homogeneous* deformation, i.e. a long-wavelength curvature that no
  finite-range force sample constrains directly;
* the **heat capacity** is a fluctuation, which depends on the whole
  distribution rather than on any mean.

Sign conventions (the part that goes wrong silently)
----------------------------------------------------
This package stores a **virial** ``W = -dU/d(eps)`` in eV (see
:class:`atomlab.types.Configuration`) and defines
``Potential.stress = W / V``, which is pressure-positive:
``P = tr(W) / (3V)`` is positive under compression.

The *mechanical* stress tensor used in elasticity is the other sign,

``sigma_ab = (1/V) dU/d(eps_ab) = -W_ab / V``

so that stretching a crystal (``eps_xx > 0``) gives a positive ``sigma_xx``
(tension) and hence a positive ``C11``.  Every stress in this module is
``sigma``; the conversion happens in exactly one place,
:func:`stress_tensor`, and nowhere else.

Units: energies eV, volumes A^3, stresses and elastic constants eV/A^3
internally and GPa at the API surface (always named in the attribute), pressures
in bar where they meet the MD layer, temperatures K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from ..analysis.statistics import blocking_analysis, integrated_autocorrelation_time
from ..cell import voigt_to_full
from ..potentials.base import Potential
from ..types import Configuration, Trajectory
from ..units import EV_A3_TO_GPA, KB
from .base import ObservableResult
from .dynamics import _as_observable_result

__all__ = [
    "EOSResult",
    "BulkModulusResult",
    "ElasticResult",
    "ThermalExpansionResult",
    "HeatCapacityResult",
    "birch_murnaghan_energy",
    "birch_murnaghan_pressure",
    "fit_birch_murnaghan",
    "equation_of_state",
    "bulk_modulus",
    "elastic_constants",
    "stress_tensor",
    "scaled_to_volume",
    "thermal_expansion",
    "heat_capacity",
]


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def stress_tensor(
    potential: Potential,
    configuration: Configuration,
    *,
    relax_positions: bool = False,
    fmax: float = 1e-5,
    max_steps: int = 500,
) -> np.ndarray:
    """Mechanical stress ``sigma = -W / V`` in eV/A^3.

    Note the sign: this is the negative of :meth:`Potential.stress`, which
    follows the pressure convention of this package.  Tension is positive here,
    so a stretched crystal has ``sigma_xx > 0`` and ``C11 > 0``.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
        Periodic; the volume is taken from its cell.
    relax_positions : bool
        Relax the atomic positions at **fixed cell** before measuring.  This is
        the internal relaxation that non-primitive lattices require, see
        :func:`elastic_constants`.
    fmax, max_steps : float, int
        Convergence controls passed to :func:`atomlab.md.simulate.minimize`.

    Returns
    -------
    ndarray, shape (3, 3)
        Symmetric stress tensor in eV/A^3.
    """
    cfg = configuration
    if relax_positions:
        from ..md.simulate import minimize  # local: md imports nothing from here

        cfg, report = minimize(cfg, potential, relax_cell=False, fmax=fmax, max_steps=max_steps)
        if not report.converged:
            raise RuntimeError(
                "internal relaxation did not converge "
                f"(fmax = {report.fmax:.3e} eV/A after {report.n_iterations} iterations); "
                "an unrelaxed strained cell gives a wrong C44, so this is not "
                "something to continue past"
            )
    virial = potential.virial(cfg)
    return -0.5 * (virial + virial.T) / cfg.volume


def scaled_to_volume(configuration: Configuration, volume: float) -> Configuration:
    """Isotropically scale a configuration to a target volume.

    Positions and lattice vectors are scaled by the same factor
    ``(V / V_0)^(1/3)``, which keeps fractional coordinates fixed -- the correct
    move for an equation of state, where the internal coordinates are either
    fixed by symmetry or relaxed afterwards.

    Parameters
    ----------
    configuration : Configuration
    volume : float
        Target cell volume in A^3.

    Returns
    -------
    Configuration
        Copy at the new volume, with reference labels stripped.
    """
    if volume <= 0.0:
        raise ValueError(f"target volume must be > 0 A^3, got {volume}")
    factor = (volume / configuration.volume) ** (1.0 / 3.0)
    return configuration.strained(np.eye(3) * (factor - 1.0))


# --------------------------------------------------------------------------
# Birch-Murnaghan equation of state
# --------------------------------------------------------------------------


def birch_murnaghan_energy(volume, v0: float, e0: float, b0: float, b0_prime: float):
    """Third-order Birch--Murnaghan energy.

    .. math::
        E(V) = E_0 + \\frac{9 V_0 B_0}{16}\\Big\\{ \\big[(V_0/V)^{2/3}-1\\big]^3 B_0'
             + \\big[(V_0/V)^{2/3}-1\\big]^2\\big[6 - 4 (V_0/V)^{2/3}\\big]\\Big\\}

    The Birch--Murnaghan form is a Taylor expansion of the energy in the
    *Eulerian finite strain* ``f = [(V0/V)^{2/3} - 1] / 2`` rather than in the
    volume.  That is what makes it usable over a wide compression range: a
    polynomial in ``V`` truncated at the same order goes badly wrong under
    compression, because the true energy diverges as ``V -> 0`` and a polynomial
    cannot.

    Parameters
    ----------
    volume : array_like
        Volumes in A^3.
    v0 : float
        Equilibrium volume in A^3.
    e0 : float
        Energy at ``v0`` in eV.
    b0 : float
        Bulk modulus at ``v0`` in eV/A^3.
    b0_prime : float
        Pressure derivative ``dB/dP`` at ``v0``, dimensionless.  ``4`` is the
        value for which the third-order term vanishes.

    Returns
    -------
    ndarray
        Energies in eV.
    """
    v = np.asarray(volume, dtype=float)
    eta = (v0 / v) ** (2.0 / 3.0) - 1.0
    return e0 + (9.0 * v0 * b0 / 16.0) * (eta**3 * b0_prime + eta**2 * (6.0 - 4.0 * (eta + 1.0)))


def birch_murnaghan_pressure(volume, v0: float, b0: float, b0_prime: float):
    """Third-order Birch--Murnaghan pressure ``P = -dE/dV``.

    .. math::
        P(V) = \\frac{3 B_0}{2}\\big[x^{7/3} - x^{5/3}\\big]
               \\Big\\{1 + \\tfrac34 (B_0' - 4)\\big[x^{2/3} - 1\\big]\\Big\\},
        \\qquad x = V_0/V

    Parameters
    ----------
    volume : array_like
        Volumes in A^3.
    v0 : float
        Equilibrium volume in A^3.
    b0 : float
        Bulk modulus in eV/A^3.
    b0_prime : float
        Dimensionless.

    Returns
    -------
    ndarray
        Pressures in eV/A^3 (multiply by
        :data:`atomlab.units.EV_A3_TO_GPA` for GPa).
    """
    x = v0 / np.asarray(volume, dtype=float)
    return (
        1.5
        * b0
        * (x ** (7.0 / 3.0) - x ** (5.0 / 3.0))
        * (1.0 + 0.75 * (b0_prime - 4.0) * (x ** (2.0 / 3.0) - 1.0))
    )


@dataclass
class EOSResult:
    """Birch--Murnaghan fit of an energy-volume curve.

    Attributes
    ----------
    volumes : ndarray, shape (n,)
        Sampled volumes in A^3.
    energies : ndarray, shape (n,)
        Total energies in eV at those volumes.
    v0, e0 : float
        Equilibrium volume in A^3 and energy in eV.
    b0 : float
        Bulk modulus at ``v0`` in eV/A^3.
    b0_prime : float
        ``dB/dP``, dimensionless.
    errors : dict
        One standard error for each of ``v0``, ``e0``, ``b0``, ``b0_prime``,
        from the covariance ``s^2 (J^T J)^{-1}`` of the least-squares fit with
        ``s^2`` the reduced chi-square.  These are *fit* errors: they say how
        well the four parameters are determined by the sampled points, and
        nothing about whether the BM3 form is the right model -- for that, look
        at ``residual_rms``.
    residual_rms : float
        RMS fit residual in eV per atom.
    n_atoms : int
    converged : bool
    message : str
        Optimiser message; recorded rather than raised so that a failed fit is
        visible in a results table instead of aborting a sweep.
    meta : dict
    """

    volumes: np.ndarray
    energies: np.ndarray
    v0: float
    e0: float
    b0: float
    b0_prime: float
    errors: dict
    residual_rms: float
    n_atoms: int
    converged: bool
    message: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def b0_gpa(self) -> float:
        """Bulk modulus in GPa."""
        return self.b0 * EV_A3_TO_GPA

    @property
    def b0_gpa_error(self) -> float:
        return self.errors["b0"] * EV_A3_TO_GPA

    @property
    def v0_per_atom(self) -> float:
        """Equilibrium volume per atom in A^3."""
        return self.v0 / self.n_atoms

    @property
    def e0_per_atom(self) -> float:
        """Equilibrium energy per atom in eV."""
        return self.e0 / self.n_atoms

    def energy_at(self, volume):
        """Fitted energy in eV at ``volume`` (A^3)."""
        return birch_murnaghan_energy(volume, self.v0, self.e0, self.b0, self.b0_prime)

    def pressure_at(self, volume):
        """Fitted pressure in eV/A^3 at ``volume`` (A^3)."""
        return birch_murnaghan_pressure(volume, self.v0, self.b0, self.b0_prime)

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "bulk_modulus[eos]",
            self.b0_gpa,
            self.b0_gpa_error,
            "GPa",
            {"v0": self.v0, "e0": self.e0, "b0_prime": self.b0_prime},
        )


def fit_birch_murnaghan(volumes, energies, *, n_atoms: int = 1) -> EOSResult:
    """Least-squares fit of the third-order Birch--Murnaghan EOS.

    Parameters
    ----------
    volumes : array_like, shape (n,)
        Volumes in A^3, at least four distinct values (four parameters).
    energies : array_like, shape (n,)
        Total energies in eV.
    n_atoms : int
        Used only to report per-atom quantities.

    Returns
    -------
    EOSResult

    Notes
    -----
    Initial guesses come from a parabola through the data: its minimum gives
    ``V0`` and ``E0``, its curvature gives ``B0 = V0 (d^2E/dV^2)``, and ``B0'``
    starts at 4 (the value at which the third-order term vanishes, so the fit
    starts from the second-order EOS and adds the cubic term only if the data
    ask for it).  Without a real starting point the fit happily runs off to
    negative ``V0``; with it, convergence is immediate for any sensible scan.

    ``least_squares`` is used rather than ``curve_fit`` so that the bounds
    ``V0 > 0``, ``B0 > 0`` can be imposed -- an unbounded fit to a noisy or
    too-narrow scan can return a negative bulk modulus, which is not a physical
    answer that should be reported with an error bar.
    """
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    if v.shape != e.shape or v.ndim != 1:
        raise ValueError(f"volumes and energies must be matching 1-D arrays, got {v.shape}, {e.shape}")
    if v.size < 4:
        raise ValueError(f"need at least 4 points to fit 4 BM3 parameters, got {v.size}")

    quad = np.polyfit(v, e, 2)
    if quad[0] <= 0.0:
        raise ValueError(
            "the energy-volume data have non-positive curvature; there is no "
            "equilibrium volume to expand about (is the scan on the right side "
            "of the minimum?)"
        )
    v0_guess = float(-quad[1] / (2.0 * quad[0]))
    if not (v.min() * 0.2 < v0_guess < v.max() * 5.0):
        v0_guess = float(v[np.argmin(e)])
    e0_guess = float(np.polyval(quad, v0_guess))
    b0_guess = float(v0_guess * 2.0 * quad[0])
    guess = np.array([v0_guess, e0_guess, max(b0_guess, 1e-4), 4.0])

    def residual(p):
        return birch_murnaghan_energy(v, *p) - e

    fit = least_squares(
        residual,
        guess,
        bounds=([1e-8, -np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )

    dof = max(v.size - 4, 1)
    chi2_reduced = 2.0 * float(fit.cost) / dof
    # Covariance of a least-squares fit: s^2 (J^T J)^{-1}.  Computed through the
    # SVD of J because J^T J is the square of the condition number of J, and for
    # an EOS scan (where V0 and B0 are strongly correlated) that squaring is
    # enough to matter.
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    tol = np.finfo(float).eps * max(fit.jac.shape) * singular[0]
    keep = singular > tol
    cov = (vt[keep].T / singular[keep] ** 2) @ vt[keep] * chi2_reduced
    errors = np.sqrt(np.abs(np.diag(cov)))

    v0, e0, b0, b0p = (float(x) for x in fit.x)
    return EOSResult(
        volumes=v,
        energies=e,
        v0=v0,
        e0=e0,
        b0=b0,
        b0_prime=b0p,
        errors={
            "v0": float(errors[0]),
            "e0": float(errors[1]),
            "b0": float(errors[2]),
            "b0_prime": float(errors[3]),
        },
        residual_rms=float(np.sqrt(np.mean(residual(fit.x) ** 2)) / n_atoms),
        n_atoms=int(n_atoms),
        converged=bool(fit.success),
        message=str(fit.message),
        meta={"covariance": cov, "guess": guess, "chi2_reduced": chi2_reduced},
    )


def equation_of_state(
    potential: Potential,
    configuration: Configuration,
    volume_range: tuple[float, float] | Sequence[float],
    n_points: int = 11,
    *,
    relative: bool = False,
    relax_positions: bool = False,
    fmax: float = 1e-5,
) -> EOSResult:
    """Energy versus volume, fitted with the third-order Birch--Murnaghan EOS.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
        Periodic starting structure.  Its cell sets the reference volume when
        ``relative=True``.
    volume_range : tuple of float
        ``(V_min, V_max)`` in A^3, or as multiples of the input volume if
        ``relative=True``.  A range of roughly +/-10% in volume is the sweet
        spot: narrower and ``B0'`` is undetermined, wider and the BM3 truncation
        error starts to show up in ``residual_rms``.
    n_points : int
        Number of volumes, spaced uniformly in volume.
    relative : bool
        Interpret ``volume_range`` as multiples of the input cell volume.
    relax_positions : bool
        Relax internal coordinates at each volume, at fixed cell.  Unnecessary
        for a lattice whose basis is fixed by symmetry (fcc, bcc, sc, diamond),
        essential for anything else.
    fmax : float
        Force convergence for that relaxation, eV/A.

    Returns
    -------
    EOSResult

    Raises
    ------
    ValueError
        For a non-periodic configuration or a degenerate volume range.
    """
    if not np.asarray(configuration.pbc).any():
        raise ValueError("an equation of state needs a periodic configuration")
    lo, hi = (float(volume_range[0]), float(volume_range[1]))
    if relative:
        lo *= configuration.volume
        hi *= configuration.volume
    if not lo < hi:
        raise ValueError(f"volume_range must be increasing, got ({lo}, {hi})")
    if int(n_points) < 4:
        raise ValueError(f"need at least 4 volumes to fit BM3, got {n_points}")

    volumes = np.linspace(lo, hi, int(n_points))
    energies = np.empty_like(volumes)
    pressures = np.empty_like(volumes)
    for k, vol in enumerate(volumes):
        cfg = scaled_to_volume(configuration, float(vol))
        if relax_positions:
            from ..md.simulate import minimize

            cfg, _ = minimize(cfg, potential, relax_cell=False, fmax=fmax)
        result = potential.compute(cfg, forces=False, virial=True)
        energies[k] = result.energy
        pressures[k] = np.trace(result.virial) / (3.0 * cfg.volume) if result.virial is not None else np.nan

    out = fit_birch_murnaghan(volumes, energies, n_atoms=configuration.n_atoms)
    out.meta["virial_pressures"] = pressures  # eV/A^3, pressure-positive
    out.meta["relax_positions"] = bool(relax_positions)
    return out


# --------------------------------------------------------------------------
# bulk modulus
# --------------------------------------------------------------------------


@dataclass
class BulkModulusResult:
    """Bulk modulus by two independent routes, which must agree.

    Attributes
    ----------
    from_eos : float
        ``B0`` of the Birch--Murnaghan fit, in eV/A^3.
    from_finite_difference : float
        ``B = -V dP/dV`` evaluated at ``volume`` by central differences of the
        virial pressure, in eV/A^3.
    error_eos : float
        Fit error on ``from_eos``, eV/A^3.
    volume : float
        Volume at which the finite difference was taken, A^3.  This is the
        fitted ``V0``, so that both numbers refer to the same state -- ``B``
        depends on volume, and comparing values taken at different volumes would
        be a meaningless test.
    pressure : float
        Virial pressure at ``volume`` in eV/A^3.  Should be ~0 if ``V0`` really
        is the equilibrium volume; a non-zero value means the two routes are
        being compared away from equilibrium, where the EOS fit's ``B0`` (which
        is defined *at* ``V0``) is not the same quantity.
    eos : EOSResult
    meta : dict

    Notes
    -----
    Why the agreement is a real test and not a tautology: the EOS route
    integrates the *energy* over a wide volume range and extracts a curvature
    from a four-parameter global fit, while the finite-difference route
    differentiates the *virial* -- an independently computed quantity, from the
    forces rather than from the energy -- over a narrow interval.  They agree
    only if the virial is consistent with the energy, which is precisely the
    property ``Potential.numerical_virial`` exists to check.  A potential with a
    subtly wrong virial passes an energy test and fails this one.
    """

    from_eos: float
    from_finite_difference: float
    error_eos: float
    volume: float
    pressure: float
    eos: EOSResult
    meta: dict = field(default_factory=dict)

    @property
    def from_eos_gpa(self) -> float:
        return self.from_eos * EV_A3_TO_GPA

    @property
    def from_finite_difference_gpa(self) -> float:
        return self.from_finite_difference * EV_A3_TO_GPA

    @property
    def relative_difference(self) -> float:
        """``|B_eos - B_fd| / B_eos``, the number the agreement test looks at."""
        return float(abs(self.from_eos - self.from_finite_difference) / abs(self.from_eos))

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "bulk_modulus",
            self.from_eos_gpa,
            self.error_eos * EV_A3_TO_GPA,
            "GPa",
            {
                "from_finite_difference_gpa": self.from_finite_difference_gpa,
                "relative_difference": self.relative_difference,
                "volume": self.volume,
            },
        )


def bulk_modulus(
    potential: Potential,
    configuration: Configuration,
    *,
    volume_range: tuple[float, float] = (0.90, 1.10),
    n_points: int = 11,
    relative: bool = True,
    volume_step: float = 1e-3,
    relax_positions: bool = False,
    fmax: float = 1e-5,
) -> BulkModulusResult:
    """Bulk modulus from the EOS fit **and** from ``-V dP/dV``, for comparison.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
    volume_range, n_points, relative, relax_positions, fmax
        Passed to :func:`equation_of_state`.
    volume_step : float
        Relative volume step of the central difference, ``dV/V``.  ``1e-3`` sits
        in the flat part of the usual accuracy-versus-round-off curve: the
        truncation error grows as ``(dV/V)^2 ~ 1e-6`` while the round-off error
        of a pressure difference divided by ``dV/V`` falls as ``1e-16 / 1e-3``.

    Returns
    -------
    BulkModulusResult
    """
    eos = equation_of_state(
        potential,
        configuration,
        volume_range,
        n_points,
        relative=relative,
        relax_positions=relax_positions,
        fmax=fmax,
    )

    v0 = eos.v0
    step = float(volume_step) * v0

    def pressure(vol: float) -> float:
        cfg = scaled_to_volume(configuration, vol)
        if relax_positions:
            from ..md.simulate import minimize

            cfg, _ = minimize(cfg, potential, relax_cell=False, fmax=fmax)
        result = potential.compute(cfg, forces=False, virial=True)
        if result.virial is None:
            raise ValueError(f"{potential.name} does not provide a virial, so -V dP/dV is unavailable")
        return float(np.trace(result.virial) / (3.0 * cfg.volume))

    p_plus = pressure(v0 + step)
    p_minus = pressure(v0 - step)
    p_zero = pressure(v0)
    # B = -V dP/dV with P the (pressure-positive) virial pressure.
    b_fd = -v0 * (p_plus - p_minus) / (2.0 * step)

    return BulkModulusResult(
        from_eos=eos.b0,
        from_finite_difference=float(b_fd),
        error_eos=eos.errors["b0"],
        volume=v0,
        pressure=p_zero,
        eos=eos,
        meta={"volume_step": step, "pressure_plus": p_plus, "pressure_minus": p_minus},
    )


# --------------------------------------------------------------------------
# elastic constants
# --------------------------------------------------------------------------

#: Voigt strain patterns used for a cubic crystal.  Each entry is
#: ``(label, voigt index of the applied engineering strain, list of measured
#: (stress voigt index, which constant it reports))``.
_UNIAXIAL_AXES = (0, 1, 2)
_SHEAR_AXES = (3, 4, 5)


@dataclass
class ElasticResult:
    """Cubic elastic constants from a strain-stress fit.

    Attributes
    ----------
    c11, c12, c44 : float
        Elastic constants in eV/A^3 (see ``*_gpa`` for GPa).
    errors : dict
        Standard errors of the fitted slopes, eV/A^3.
    scatter : dict
        Spread over symmetry-equivalent directions (``max - min`` over the three
        equivalent patterns) in eV/A^3.  For a cubic crystal this is zero in
        exact arithmetic, so it is a direct check that the structure really is
        cubic and that the relaxation converged.  ``nan`` if only one direction
        was used.
    linearity : dict
        For each constant, the fraction of the stress response at the largest
        applied strain that the quadratic term accounts for.  Small (<~1%) means
        the strain magnitudes are inside the linear regime, which is the
        assumption the whole method rests on.
    r_squared : dict
        Coefficient of determination of each linear fit.
    residual_stress : float
        Largest component of the stress at zero strain, eV/A^3.  The elastic
        constants are defined at zero stress; a residual means the structure was
        not at its equilibrium lattice constant, and the numbers are then
        "Birch coefficients" that differ from the true ``C_ij`` by terms of
        order the residual stress itself.
    relax_internal : bool
        Whether atomic positions were relaxed at fixed strained cell.
    internal_relaxation_shift : dict
        ``C_unrelaxed - C_relaxed`` for each constant, eV/A^3, when both were
        computed.  For a lattice with a basis (diamond) the ``C44`` entry is
        large -- that is the point of the option, and it is measured rather
        than asserted.
    strains : ndarray
        Applied engineering strain magnitudes (dimensionless).
    meta : dict

    Notes
    -----
    The Cauchy relation ``C12 = C44`` holds exactly for any potential built from
    central pair interactions, when every atom sits at a centre of inversion
    symmetry and the crystal is at zero stress.  It is not a consequence of
    cubic symmetry -- three-body or embedding terms break it, and real metals
    violate it by tens of per cent -- so it is a sharp, non-obvious test of a
    pair-potential implementation and of this routine at the same time.
    """

    c11: float
    c12: float
    c44: float
    errors: dict
    scatter: dict
    linearity: dict
    r_squared: dict
    residual_stress: float
    relax_internal: bool
    internal_relaxation_shift: dict = field(default_factory=dict)
    strains: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    @property
    def c11_gpa(self) -> float:
        return self.c11 * EV_A3_TO_GPA

    @property
    def c12_gpa(self) -> float:
        return self.c12 * EV_A3_TO_GPA

    @property
    def c44_gpa(self) -> float:
        return self.c44 * EV_A3_TO_GPA

    @property
    def bulk_modulus(self) -> float:
        """``B = (C11 + 2 C12) / 3`` in eV/A^3, the cubic Voigt/Reuss average.

        For a cubic crystal this is exact (not an average over orientations),
        which makes it a cross-check against the EOS bulk modulus computed by a
        completely different route.
        """
        return (self.c11 + 2.0 * self.c12) / 3.0

    @property
    def bulk_modulus_gpa(self) -> float:
        return self.bulk_modulus * EV_A3_TO_GPA

    @property
    def cauchy_deviation(self) -> float:
        """``(C12 - C44) / C12``: zero for a central pair potential at ``P = 0``."""
        return float((self.c12 - self.c44) / self.c12)

    @property
    def zener_anisotropy(self) -> float:
        """``2 C44 / (C11 - C12)``; 1 for an elastically isotropic cubic crystal."""
        return float(2.0 * self.c44 / (self.c11 - self.c12))

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "elastic_constants",
            np.array([self.c11_gpa, self.c12_gpa, self.c44_gpa]),
            np.array([self.errors["c11"], self.errors["c12"], self.errors["c44"]]) * EV_A3_TO_GPA,
            "GPa",
            {
                "labels": ("C11", "C12", "C44"),
                "cauchy_deviation": self.cauchy_deviation,
                "relax_internal": self.relax_internal,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ElasticResult(C11={self.c11_gpa:.2f}, C12={self.c12_gpa:.2f}, "
            f"C44={self.c44_gpa:.2f} GPa, Cauchy dev={self.cauchy_deviation:+.3%})"
        )


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit ``y = a x + b`` and also a quadratic; return diagnostics.

    Returns ``(slope, slope_error, r_squared, curvature_fraction)`` where
    ``curvature_fraction`` is the quadratic term's contribution at the largest
    ``|x|`` relative to the linear term's -- the quantitative statement of
    "are we still in the linear regime".
    """
    design = np.vstack([x, np.ones_like(x)]).T
    coeff, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    resid = y - (slope * x + intercept)
    dof = max(x.size - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(design.T @ design)
    total = y - y.mean()
    denom = float(total @ total)
    r2 = 1.0 - float(resid @ resid) / denom if denom > 0 else float("nan")

    curvature = 0.0
    if x.size >= 3:
        quad = np.polyfit(x, y, 2)
        biggest = float(np.max(np.abs(x)))
        linear_part = abs(quad[1] * biggest)
        if linear_part > 0:
            curvature = float(abs(quad[0] * biggest**2) / linear_part)
    return slope, float(np.sqrt(cov[0, 0])), r2, curvature


def _strain_stress_scan(
    potential: Potential,
    configuration: Configuration,
    voigt_index: int,
    magnitudes: np.ndarray,
    *,
    relax_internal: bool,
    fmax: float,
    max_steps: int,
) -> np.ndarray:
    """Stress tensors (eV/A^3) for one strain pattern at several magnitudes.

    ``voigt_index`` selects the engineering strain component that is switched
    on; :func:`atomlab.cell.voigt_to_full` with ``strain=True`` halves the
    off-diagonal components, so a shear entry ``gamma`` produces
    ``eps_yz = eps_zy = gamma/2`` and the measured ``sigma_yz`` has slope
    ``C44`` in ``gamma`` -- the factor of two that this convention exists to
    keep straight.

    Returns an array of shape ``(len(magnitudes), 3, 3)``.
    """
    out = np.empty((magnitudes.size, 3, 3))
    for k, eps in enumerate(magnitudes):
        voigt = np.zeros(6)
        voigt[voigt_index] = eps
        strained = configuration.strained(voigt_to_full(voigt, strain=True))
        out[k] = stress_tensor(
            potential,
            strained,
            relax_positions=relax_internal,
            fmax=fmax,
            max_steps=max_steps,
        )
    return out


def elastic_constants(
    potential: Potential,
    configuration: Configuration,
    *,
    strain_magnitude: float = 0.005,
    relax_internal: bool = True,
    n_strains: int = 5,
    average_equivalent: bool = True,
    measure_relaxation_effect: bool = False,
    fmax: float = 1e-5,
    max_steps: int = 1000,
) -> ElasticResult:
    """Cubic elastic constants ``C11``, ``C12``, ``C44`` by strain-stress fitting.

    Two strain families are applied and the stress response is fitted linearly:

    * uniaxial ``eps_xx = e``: ``sigma_xx = C11 e`` and
      ``sigma_yy = sigma_zz = C12 e``;
    * engineering shear ``gamma_yz = g`` (i.e. ``eps_yz = eps_zy = g/2``):
      ``sigma_yz = C44 g``.

    Several magnitudes of each sign are used and the slope is fitted rather than
    taken from a single point, both because that averages down the round-off in
    the stress and because it produces the diagnostics -- ``r_squared`` and the
    quadratic contribution in ``linearity`` -- that say whether the assumed
    linear regime was actually reached.  Symmetric ``+/-`` magnitudes also cancel
    the leading anharmonic error in the slope.

    **Internal relaxation.**  For a lattice with more than one atom per
    primitive cell (diamond, zincblende, hcp) a homogeneous strain does not
    determine the atomic positions: the sublattices shift relative to one
    another, and the energy is lowered by letting them.  Omitting this
    relaxation gives the "unrelaxed" or "frozen-ion" constants, which for
    diamond-structure ``C44`` are wrong by tens of per cent -- the internal
    strain is precisely what the Kleinman parameter measures.  ``relax_internal``
    (on by default) relaxes the positions at fixed strained cell before reading
    the stress.  For fcc/bcc/sc, where every atom is at an inversion centre, the
    relaxation is identically zero and only costs time.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
        A cubic crystal **at its equilibrium lattice constant for this
        potential**.  Relax it first (``atomlab.md.simulate.minimize`` with
        ``relax_cell=True, hydrostatic=True``); the residual stress is reported
        so that a caller who forgot can see it.
    strain_magnitude : float
        Largest applied strain (dimensionless).  ``0.005`` is small enough to
        stay linear for the potentials here and large enough that the stress
        difference is far above numerical noise.
    relax_internal : bool
        Relax atomic positions at fixed strained cell before measuring stress.
    n_strains : int
        Number of magnitudes, spaced uniformly in ``[-strain_magnitude,
        +strain_magnitude]``.  Must be at least 3; an odd value includes zero,
        which is what gives ``residual_stress``.
    average_equivalent : bool
        Average over the three symmetry-equivalent uniaxial axes and the three
        equivalent shears.  Costs three times as much and returns the scatter
        between them, which is the cheapest available check that the input
        really is cubic and that the relaxations converged.
    measure_relaxation_effect : bool
        Also compute the constants without internal relaxation and report the
        difference in ``internal_relaxation_shift``.
    fmax, max_steps : float, int
        Convergence controls for the internal relaxation.

    Returns
    -------
    ElasticResult
        Constants in eV/A^3, with ``.c11_gpa`` etc. for GPa.

    Raises
    ------
    ValueError
        For a non-periodic configuration or an invalid strain grid.
    RuntimeError
        If an internal relaxation fails to converge.
    """
    if not np.asarray(configuration.pbc).all():
        raise ValueError("elastic constants require a fully periodic configuration")
    if int(n_strains) < 3:
        raise ValueError(f"need at least 3 strain magnitudes to judge linearity, got {n_strains}")
    if strain_magnitude <= 0.0:
        raise ValueError(f"strain_magnitude must be > 0, got {strain_magnitude}")

    magnitudes = np.linspace(-float(strain_magnitude), float(strain_magnitude), int(n_strains))
    cfg = configuration.stripped()

    def measure(relax: bool) -> dict:
        c11_values, c12_values, c44_values = [], [], []
        diagnostics: dict[str, list] = {"c11": [], "c12": [], "c44": []}
        residual = 0.0

        axes = _UNIAXIAL_AXES if average_equivalent else _UNIAXIAL_AXES[:1]
        for axis in axes:
            stresses = _strain_stress_scan(
                potential, cfg, axis, magnitudes,
                relax_internal=relax, fmax=fmax, max_steps=max_steps,
            )
            # Longitudinal response -> C11; the two transverse responses -> C12.
            other = [a for a in (0, 1, 2) if a != axis]
            slope, err, r2, curv = _linear_fit(magnitudes, stresses[:, axis, axis])
            c11_values.append(slope)
            diagnostics["c11"].append((err, r2, curv))
            for a in other:
                slope, err, r2, curv = _linear_fit(magnitudes, stresses[:, a, a])
                c12_values.append(slope)
                diagnostics["c12"].append((err, r2, curv))
            zero = np.argmin(np.abs(magnitudes))
            residual = max(residual, float(np.max(np.abs(stresses[zero]))))

        shears = _SHEAR_AXES if average_equivalent else _SHEAR_AXES[:1]
        for shear in shears:
            stresses = _strain_stress_scan(
                potential, cfg, shear, magnitudes,
                relax_internal=relax, fmax=fmax, max_steps=max_steps,
            )
            a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[shear]
            slope, err, r2, curv = _linear_fit(magnitudes, stresses[:, a, b])
            c44_values.append(slope)
            diagnostics["c44"].append((err, r2, curv))

        values = {
            "c11": np.array(c11_values),
            "c12": np.array(c12_values),
            "c44": np.array(c44_values),
        }
        return {"values": values, "diagnostics": diagnostics, "residual": residual}

    relaxed = measure(relax_internal)
    values = relaxed["values"]
    diag = relaxed["diagnostics"]

    def summarise(key: str) -> tuple[float, float, float, float, float]:
        v = values[key]
        errs = np.array([d[0] for d in diag[key]])
        r2 = float(np.min([d[1] for d in diag[key]]))
        curv = float(np.max([d[2] for d in diag[key]]))
        # Errors on the mean of n_dir independent fits add in quadrature.
        err = float(np.sqrt(np.sum(errs**2)) / errs.size)
        scatter = float(v.max() - v.min()) if v.size > 1 else float("nan")
        return float(v.mean()), err, r2, curv, scatter

    c11, e11, r11, k11, s11 = summarise("c11")
    c12, e12, r12, k12, s12 = summarise("c12")
    c44, e44, r44, k44, s44 = summarise("c44")

    shift: dict = {}
    if measure_relaxation_effect:
        unrelaxed = measure(False)["values"]
        shift = {
            "c11": float(unrelaxed["c11"].mean() - c11),
            "c12": float(unrelaxed["c12"].mean() - c12),
            "c44": float(unrelaxed["c44"].mean() - c44),
            "c11_unrelaxed": float(unrelaxed["c11"].mean()),
            "c12_unrelaxed": float(unrelaxed["c12"].mean()),
            "c44_unrelaxed": float(unrelaxed["c44"].mean()),
        }

    return ElasticResult(
        c11=c11,
        c12=c12,
        c44=c44,
        errors={"c11": e11, "c12": e12, "c44": e44},
        scatter={"c11": s11, "c12": s12, "c44": s44},
        linearity={"c11": k11, "c12": k12, "c44": k44},
        r_squared={"c11": r11, "c12": r12, "c44": r44},
        residual_stress=float(relaxed["residual"]),
        relax_internal=bool(relax_internal),
        internal_relaxation_shift=shift,
        strains=magnitudes,
        meta={
            "average_equivalent": bool(average_equivalent),
            "c11_directions": values["c11"],
            "c12_directions": values["c12"],
            "c44_directions": values["c44"],
        },
    )


# --------------------------------------------------------------------------
# thermal expansion (NPT) and heat capacity (fluctuations)
# --------------------------------------------------------------------------


@dataclass
class ThermalExpansionResult:
    """Volumetric and linear thermal expansion from a temperature series.

    Attributes
    ----------
    temperatures : ndarray, shape (n,)
        Target temperatures in K.
    volumes, volume_errors : ndarray, shape (n,)
        Mean cell volume and its blocking error bar in A^3.
    measured_temperatures : ndarray, shape (n,)
        Mean instantaneous temperature actually sampled, K.  A systematic
        difference from ``temperatures`` means the barostat/thermostat did not
        equilibrate, and the expansion coefficient inherits it.
    alpha_volumetric : float
        ``(1/V) dV/dT`` in 1/K, evaluated at the reference volume.
    alpha_linear : float
        ``alpha_volumetric / 3`` in 1/K.  Exact for a cubic crystal under
        isotropic barostatting.
    alpha_error : float
        One standard error on ``alpha_volumetric``, propagated from the fit, 1/K.
    reference_volume, reference_temperature : float
        Where the derivative was evaluated, A^3 and K.
    trajectories : list of Trajectory or None
        Retained only if ``keep_trajectories``; NPT runs are expensive and
        throwing them away by default is deliberate.
    meta : dict
    """

    temperatures: np.ndarray
    volumes: np.ndarray
    volume_errors: np.ndarray
    measured_temperatures: np.ndarray
    alpha_volumetric: float
    alpha_linear: float
    alpha_error: float
    reference_volume: float
    reference_temperature: float
    trajectories: list | None = None
    meta: dict = field(default_factory=dict)

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "thermal_expansion",
            self.alpha_linear,
            self.alpha_error / 3.0,
            "1/K",
            {
                "temperatures": self.temperatures,
                "volumes": self.volumes,
                "alpha_volumetric": self.alpha_volumetric,
            },
        )


def thermal_expansion(
    potential: Potential,
    configuration: Configuration,
    temperatures: Sequence[float],
    *,
    dt: float = 0.002,
    n_equilibrate: int = 2000,
    n_production: int = 5000,
    seed: int = 0,
    pressure_bar: float = 0.0,
    tau_t: float = 0.1,
    tau_p: float = 1.0,
    frame_stride: int = 10,
    log_stride: int = 1,
    keep_trajectories: bool = False,
    progress: bool = False,
) -> ThermalExpansionResult:
    """Thermal expansion coefficient from a series of NPT simulations.

    One isothermal-isobaric run per temperature; the mean volume is measured
    with a blocking error bar, and ``alpha_V = (1/V) dV/dT`` comes from a
    straight-line fit of ``V(T)`` weighted by those errors.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
        Starting structure; the same one is used for every temperature, so each
        run is independent rather than a continuation (which would correlate the
        temperatures and understate the error).
    temperatures : sequence of float
        Target temperatures in K, at least two.
    dt : float
        Timestep in ps.
    n_equilibrate, n_production : int
        Steps discarded and recorded per temperature.  The volume is the slowest
        variable in an NPT run -- it relaxes on ``tau_p``, not on ``dt`` -- so
        ``n_equilibrate * dt`` should be several times ``tau_p``, and the check
        is ``meta["equilibration_ratio"]``.
    seed : int
        Base seed; run ``k`` uses ``seed + k`` so the temperatures are
        independent and the whole series is reproducible.
    pressure_bar : float
        Target pressure in bar.
    tau_t, tau_p : float
        Thermostat and barostat time constants in ps.
    frame_stride, log_stride : int
        Passed to :func:`atomlab.md.simulate.run_md`.  The volume series comes
        from the logged scalars.
    keep_trajectories : bool
        Retain the trajectories in the result.
    progress : bool
        Progress lines from the MD driver.

    Returns
    -------
    ThermalExpansionResult

    Raises
    ------
    ValueError
        For fewer than two temperatures, or a non-periodic configuration.
    """
    from ..md.integrators import MTKBarostat
    from ..md.simulate import run_md

    temps = np.asarray(temperatures, dtype=float)
    if temps.size < 2:
        raise ValueError("need at least two temperatures to fit a thermal expansion coefficient")
    if np.any(temps <= 0.0):
        raise ValueError("temperatures must be > 0 K")
    if not np.asarray(configuration.pbc).all():
        raise ValueError("thermal expansion needs a fully periodic configuration")

    volumes = np.empty(temps.size)
    errors = np.empty(temps.size)
    measured = np.empty(temps.size)
    taus = np.empty(temps.size)
    kept: list = []

    for k, temperature in enumerate(temps):
        integrator = MTKBarostat(dt, float(temperature), pressure_bar, tau_t, tau_p)
        traj = run_md(
            configuration,
            potential,
            integrator,
            n_steps=int(n_production),
            thermalize_steps=int(n_equilibrate),
            temperature=float(temperature),
            seed=int(seed) + k,
            frame_stride=int(frame_stride),
            log_stride=int(log_stride),
            progress=progress,
        )
        log = traj.info.get("log", {})
        series = np.asarray(log.get("volume", traj.scalars["volume"]), dtype=float)
        estimate = blocking_analysis(series)
        volumes[k] = float(estimate.value)
        errors[k] = float(estimate.error)
        taus[k] = float(estimate.extra["tau"])
        measured[k] = float(np.mean(np.asarray(log.get("temperature", traj.scalars["temperature"]))))
        if keep_trajectories:
            kept.append(traj)

    # Weighted straight-line fit V(T) = V_ref + slope * (T - T_ref).
    reference_temperature = float(temps.mean())
    weights = 1.0 / np.where(errors > 0, errors, np.finfo(float).eps)
    design = np.vstack([(temps - reference_temperature) * weights, weights]).T
    coeff, *_ = np.linalg.lstsq(design, volumes * weights, rcond=None)
    slope, reference_volume = float(coeff[0]), float(coeff[1])
    cov = np.linalg.inv(design.T @ design)
    slope_error = float(np.sqrt(cov[0, 0]))

    alpha = slope / reference_volume
    # dV and V are correlated through the same data; the V term contributes at
    # relative order (sigma_V / V) ~ 1e-3, so the slope error dominates and is
    # what is propagated.
    alpha_error = slope_error / reference_volume

    return ThermalExpansionResult(
        temperatures=temps,
        volumes=volumes,
        volume_errors=errors,
        measured_temperatures=measured,
        alpha_volumetric=float(alpha),
        alpha_linear=float(alpha / 3.0),
        alpha_error=float(alpha_error),
        reference_volume=reference_volume,
        reference_temperature=reference_temperature,
        trajectories=kept if keep_trajectories else None,
        meta={
            "volume_tau_frames": taus,
            "equilibration_ratio": float(n_equilibrate * dt / tau_p),
            "pressure_bar": float(pressure_bar),
            "seed": int(seed),
        },
    )


@dataclass
class HeatCapacityResult:
    """Constant-volume heat capacity from energy fluctuations.

    Attributes
    ----------
    cv : float
        Total ``C_V`` in eV/K.
    error : float
        One standard error in eV/K.
    cv_per_atom_kb : float
        ``C_V / (N k_B)``, dimensionless -- the form to compare against the
        Dulong--Petit value of 3.
    configurational : float
        The ``var(U) / (k_B T^2)`` part alone, eV/K.
    kinetic : float
        The ideal ``(n_dof / 2) k_B`` part, eV/K.  Separated because only the
        configurational part carries statistical error worth arguing about, and
        because a potential is judged on that part.
    temperature : float
        Temperature used, K.
    tau_frames : float
        Integrated autocorrelation time of the energy series, in frames.  A
        fluctuation converges like ``sqrt(2 tau / M)``, so this is not a detail:
        with ``tau = 50`` frames a 10000-frame run has 100 independent samples
        and a 10% error bar on ``C_V``, however smooth the running average
        looks.
    n_effective : float
        ``M / (2 tau)``, the number of independent samples.
    ensemble : {"nvt", "nve"}
    meta : dict
    """

    cv: float
    error: float
    cv_per_atom_kb: float
    configurational: float
    kinetic: float
    temperature: float
    tau_frames: float
    n_effective: float
    ensemble: str
    meta: dict = field(default_factory=dict)

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "heat_capacity",
            self.cv,
            self.error,
            "eV/K",
            {
                "cv_per_atom_kb": self.cv_per_atom_kb,
                "temperature": self.temperature,
                "tau_frames": self.tau_frames,
                "ensemble": self.ensemble,
            },
        )


def heat_capacity(
    trajectory: Trajectory,
    *,
    temperature: float | None = None,
    ensemble: str = "nvt",
    n_dof: int | None = None,
    n_blocks: int = 8,
    energy_key: str = "potential_energy",
) -> HeatCapacityResult:
    """Heat capacity ``C_V`` from energy fluctuations.

    In the canonical ensemble the fluctuation-dissipation relation gives

    ``C_V = var(U) / (k_B T^2) + (n_dof / 2) k_B``

    where the second term is the kinetic contribution, which is exactly ideal
    for a classical system and carries no statistical error.  In the
    microcanonical ensemble the same information is in the *kinetic* energy
    fluctuation (Lebowitz--Percus--Verlet),

    ``var(KE) / <KE>^2 = (2 / n_dof) [1 - n_dof k_B / (2 C_V)]``

    which is the ``ensemble="nve"`` branch.

    Fluctuation estimators converge slowly -- the variance of a variance -- so
    the error bar is computed by blocking the *squared deviations*, and the
    correlation time is reported alongside.  A heat capacity quoted without its
    correlation time cannot be judged.

    Parameters
    ----------
    trajectory : Trajectory
        Must carry the relevant scalar log.  The finer-grained
        ``info["log"]`` series is preferred over the frame-aligned
        ``scalars`` when present, because a fluctuation estimator wants every
        sample it can get.
    temperature : float, optional
        Temperature in K.  Defaults to the mean of the logged instantaneous
        temperature, which is the right choice in NVE and an adequate one in
        NVT.
    ensemble : {"nvt", "nve"}
        Which fluctuation formula to use.  Getting this wrong is a factor-level
        error, not a small one, so there is no default guess from the data.
    n_dof : int, optional
        Momentum degrees of freedom.  Defaults to the value the MD driver
        recorded, else ``3N - 3``.
    n_blocks : int
        Blocks used for the error bar on the fluctuation.
    energy_key : str
        Which scalar series to use for ``U``.

    Returns
    -------
    HeatCapacityResult

    Raises
    ------
    ValueError
        For an unknown ensemble or a missing energy series.
    """
    if ensemble not in ("nvt", "nve"):
        raise ValueError(f"unknown ensemble {ensemble!r}; expected 'nvt' or 'nve'")

    log = trajectory.info.get("log", {})

    def series(key: str) -> np.ndarray:
        if key in log:
            return np.asarray(log[key], dtype=float)
        if key in trajectory.scalars:
            return np.asarray(trajectory.scalars[key], dtype=float)
        raise ValueError(
            f"trajectory carries no {key!r} series; heat_capacity needs the "
            "thermodynamic log that run_md writes"
        )

    n_atoms = trajectory.n_atoms
    if n_dof is None:
        n_dof = int(trajectory.info.get("n_dof", 3 * n_atoms - 3))
    if temperature is None:
        temperature = float(np.mean(series("temperature")))
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature}")

    kinetic_term = 0.5 * n_dof * KB

    if ensemble == "nvt":
        energy = series(energy_key)
        deviation = energy - energy.mean()
        squared = deviation**2
        variance = float(squared.mean())
        configurational = variance / (KB * temperature**2)
        # Error on a variance: block the squared deviations, which is the same
        # blocking analysis applied to the quantity actually being averaged.
        block_estimate = blocking_analysis(squared, min_blocks=max(2, n_blocks))
        error = float(block_estimate.error) / (KB * temperature**2)
        cv = configurational + kinetic_term
        tau = float(integrated_autocorrelation_time(energy))
        n_eff = energy.size / (2.0 * tau)
    else:
        kinetic = series("kinetic_energy")
        mean_ke = float(kinetic.mean())
        squared = (kinetic - mean_ke) ** 2
        ratio = float(squared.mean()) / mean_ke**2
        denominator = 1.0 - 0.5 * n_dof * ratio
        if denominator <= 0.0:
            raise ValueError(
                "the microcanonical kinetic-energy fluctuation is too large for the "
                "Lebowitz-Percus-Verlet formula (implied negative heat capacity); "
                "the run is either far too short or not microcanonical"
            )
        cv = 0.5 * n_dof * KB / denominator
        block_estimate = blocking_analysis(squared, min_blocks=max(2, n_blocks))
        # dCv/d(var) propagated: Cv = a/(1 - b var) => dCv = Cv^2 b / a * dvar.
        b = 0.5 * n_dof / mean_ke**2
        error = float(cv**2 * b / (0.5 * n_dof * KB) * block_estimate.error)
        configurational = cv - kinetic_term
        tau = float(integrated_autocorrelation_time(kinetic))
        n_eff = kinetic.size / (2.0 * tau)

    return HeatCapacityResult(
        cv=float(cv),
        error=float(error),
        cv_per_atom_kb=float(cv / (n_atoms * KB)),
        configurational=float(configurational),
        kinetic=float(kinetic_term),
        temperature=float(temperature),
        tau_frames=tau,
        n_effective=float(n_eff),
        ensemble=ensemble,
        meta={
            "n_samples": int(squared.size),
            "n_dof": int(n_dof),
            "blocking_method": block_estimate.method,
        },
    )
