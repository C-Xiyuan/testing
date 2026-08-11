"""Dynamical observables: mean squared displacement, VACF, self-diffusion.

These are the observables that molecular dynamics is *irreplaceable* for.  Every
static quantity in this project is measured from HMC samples (see
``docs/design.md`` 5.3a), but a diffusion coefficient is a statement about real
time evolution and Monte Carlo cannot produce one at any price.  Accordingly
every estimator here refuses a trajectory whose ``info["dynamical"]`` is False.

Two independent routes to the self-diffusion coefficient are provided, and the
fact that they are independent is the point:

*Einstein*      ``D = lim_{t->inf} <|r(t) - r(0)|^2> / (6 t)``
*Green--Kubo*   ``D = (1/3) int_0^inf <v(0) . v(t)> dt``

They are equal in the limit of infinite sampling, but they weight the data
completely differently -- the Einstein route is dominated by the long-lag
behaviour of a few decorrelated windows, the Green--Kubo route by the short-lag
behaviour of the velocity correlation.  A bug in the unwrapping, in the
integrator, or in the origin averaging will generally move one and not the
other, so their agreement *within error bars* is the strongest cheap check
available on the whole dynamical pipeline.

Algorithm note (why the MSD is done with FFTs)
----------------------------------------------
The definition

``MSD(m) = 1/(T-m) sum_{k=0}^{T-m-1} |r(k+m) - r(k)|^2``

evaluated directly costs ``O(T^2 N)``, which for the ``T ~ 10^4-10^5`` frame
trajectories this study needs is minutes-to-hours per trajectory per model --
multiplied by the model zoo, that is the difference between a feasible and an
infeasible experiment.  The Fast Correlation Algorithm (Kalus et al., the
algorithm in nMoldyn) rewrites the definition as

``MSD(m) = S1(m) - 2 S2(m)``

with ``S1(m) = 1/(T-m) sum_k [|r(k)|^2 + |r(k+m)|^2]`` -- computable for all
``m`` at once by a running sum -- and ``S2(m) = 1/(T-m) sum_k r(k).r(k+m)``, the
autocorrelation of the position vector, which the Wiener--Khinchin theorem gives
in ``O(T log T)`` via one FFT.  The naive form is kept in :func:`msd_naive` and
tested against the fast one, because a fast wrong kernel is the most expensive
kind of bug in a project whose conclusions are numerical comparisons.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.integrate import simpson

from ..analysis.statistics import integrated_autocorrelation_time
from ..cell import min_cell_width
from ..types import Trajectory
from .base import ObservableResult

__all__ = [
    "MSDResult",
    "VACFResult",
    "DiffusionResult",
    "mean_squared_displacement",
    "msd_fft",
    "msd_naive",
    "diffusion_coefficient",
    "velocity_autocorrelation",
    "vacf_fft",
    "vacf_naive",
    "diffusion_from_vacf",
    "A2_PS_TO_CM2_S",
]

#: Converts a diffusion coefficient from ``A^2/ps`` to ``cm^2/s``.  This is a
#: definitional conversion between length and time units (1 A = 1e-8 cm,
#: 1 ps = 1e-12 s), not a measured constant, so it is written out here rather
#: than imported from :mod:`atomlab.units`.
A2_PS_TO_CM2_S = (1.0e-8) ** 2 / 1.0e-12  # = 1e-4


# --------------------------------------------------------------------------
# interoperability with the shared observable container
# --------------------------------------------------------------------------


def _as_observable_result(name: str, value, error, unit: str, meta: dict) -> ObservableResult:
    """Wrap a scalar estimate in the package-wide :class:`ObservableResult`.

    ``observables/base.py`` is authored independently of this module, so the
    exact field list of :class:`ObservableResult` is not something this file
    should assume.  We therefore offer the standard set of keywords and pass
    only the ones the container actually accepts; anything left over is folded
    into the metadata dictionary.  This is an interface adapter, not a physics
    fallback -- if the container cannot represent ``value`` and ``error`` at all
    the construction is allowed to fail loudly.
    """
    accepted = set(inspect.signature(ObservableResult).parameters)
    offered = {
        "name": name,
        "value": value,
        "error": error,
        "unit": unit,
        "units": unit,
        "meta": dict(meta),
        "metadata": dict(meta),
        "extra": dict(meta),
        "info": dict(meta),
    }
    return ObservableResult(**{k: v for k, v in offered.items() if k in accepted})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _require_dynamical(trajectory: Trajectory, what: str) -> None:
    """Raise if the trajectory was not produced by real time evolution.

    ``atomlab.sampling`` stamps ``info["dynamical"] = False`` on Monte Carlo
    output.  Feeding that to a diffusion estimator would return a number that
    depends on the proposal distribution and nothing else, which is exactly the
    kind of quiet nonsense ``docs/design.md`` 5.3a forbids.
    """
    if not trajectory.info.get("dynamical", True):
        raise ValueError(
            f"{what} needs a dynamical trajectory, but this one carries "
            f"info['dynamical'] = False (sampler: {trajectory.info.get('sampler', '?')}). "
            "Monte Carlo trajectories have no physical time axis."
        )


def _check_unwrapped(trajectory: Trajectory, *, tolerance: float = 0.5) -> float:
    """Verify that stored positions are unwrapped; return the worst step size.

    :class:`~atomlab.types.Trajectory` promises unwrapped positions, and the
    MSD is meaningless if that promise is broken -- a wrapped coordinate makes
    the MSD saturate at a value set by the cell size, which looks like a solid
    rather than like a bug.  The check is the standard one: no single-frame
    displacement may exceed ``tolerance`` times the minimum perpendicular cell
    width, since a wrap always produces a jump of at least one cell width.

    This is necessary but not sufficient.  A genuinely ballistic atom in a small
    cell with a long frame stride can trip it, and a wrap can hide below it if
    the stride is long enough that real motion is comparable to the cell.  Both
    failure modes are loud (an exception, or a visibly saturating MSD) rather
    than silent, which is the design rule.

    Returns
    -------
    float
        Largest single-step displacement in angstrom.

    Raises
    ------
    ValueError
        If a displacement exceeds ``tolerance * min_cell_width``.
    """
    if trajectory.n_frames < 2:
        return 0.0
    steps = np.diff(trajectory.positions, axis=0)
    largest = float(np.sqrt(np.max(np.einsum("tna,tna->tn", steps, steps))))

    widths = np.array(
        [min_cell_width(trajectory.cells[t], trajectory.template.pbc) for t in range(trajectory.n_frames)]
    )
    limit = tolerance * float(np.min(widths))
    if np.isfinite(limit) and largest > limit:
        frame, atom = np.unravel_index(
            int(np.argmax(np.einsum("tna,tna->tn", steps, steps))), steps.shape[:2]
        )
        raise ValueError(
            f"trajectory positions do not look unwrapped: atom {atom} moves "
            f"{largest:.3f} A between frames {frame} and {frame + 1}, which exceeds "
            f"{tolerance:g} * min_cell_width = {limit:.3f} A. Trajectory.positions must "
            "be stored unwrapped (atomlab.md.simulate does this); wrapping makes the "
            "mean squared displacement saturate at the cell size."
        )
    return largest


def _fft_autocorrelation(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of a vector series, summed over components.

    Parameters
    ----------
    x : ndarray, shape (T, N, 3)
        Vector series (positions in A, or velocities in A/ps).

    Returns
    -------
    ndarray, shape (T, N)
        ``C[m, i] = 1/(T-m) sum_k x[k, i] . x[k+m, i]``, in the square of the
        input unit.

    Notes
    -----
    Wiener--Khinchin: the autocorrelation is the inverse transform of the power
    spectrum.  The transform length is padded to a power of two ``>= 2T`` for
    two separate reasons -- ``>= 2T`` makes the circular correlation the linear
    one (without it, lag ``m`` would be contaminated by lag ``T-m``), and the
    power of two is what makes the FFT actually fast.  The ``1/(T-m)``
    normalisation counts the pairs that exist at each lag; the alternative
    ``1/T`` convention tapers the estimate towards zero at long lag, which for
    an MSD would masquerade as sub-diffusive behaviour.
    """
    n_frames = x.shape[0]
    n_fft = 1 << int(2 * n_frames - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n_fft, axis=0)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=0)[:n_frames]
    corr = corr.sum(axis=-1)
    counts = (n_frames - np.arange(n_frames)).astype(np.float64)[:, None]
    return corr / counts


def _as_series(positions) -> np.ndarray:
    p = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
    if p.ndim != 3 or p.shape[2] != 3:
        raise ValueError(f"positions must have shape (T, N, 3), got {p.shape}")
    if p.shape[0] < 2:
        raise ValueError("need at least two frames")
    return p


def _resolve_max_lag(max_lag: int | None, n_frames: int) -> int:
    if max_lag is None:
        # Half the trajectory: beyond that fewer than T/2 origins contribute and
        # the estimator's variance grows without any gain in information.
        max_lag = n_frames // 2
    max_lag = int(max_lag)
    if max_lag < 1 or max_lag > n_frames - 1:
        raise ValueError(f"max_lag must be in [1, {n_frames - 1}], got {max_lag}")
    return max_lag


# --------------------------------------------------------------------------
# mean squared displacement
# --------------------------------------------------------------------------


def msd_fft(positions, max_lag: int | None = None) -> np.ndarray:
    """Per-atom MSD via the Fast Correlation Algorithm (FFT based).

    Parameters
    ----------
    positions : array_like, shape (T, N, 3)
        **Unwrapped** cartesian coordinates in angstrom.
    max_lag : int, optional
        Largest lag returned, in frames.  Defaults to ``T // 2``.

    Returns
    -------
    ndarray, shape (max_lag + 1, N)
        ``MSD[m, i]`` in A^2, averaged over all ``T - m`` time origins.

    Notes
    -----
    Positions are shifted by each atom's time-mean before the transform.  The
    MSD is exactly invariant under a per-atom constant shift, but the FFT route
    computes it as the difference ``S1 - 2 S2`` of two quantities that both
    scale as ``|r|^2``; for a trajectory sitting 100 A from the origin that
    cancellation costs four significant digits.  Centring makes the two terms
    the same size as their difference and the loss disappears.
    """
    r = _as_series(positions)
    n_frames = r.shape[0]
    max_lag = _resolve_max_lag(max_lag, n_frames)
    r = r - r.mean(axis=0, keepdims=True)

    sq = np.einsum("tna,tna->tn", r, r)  # (T, N) = |r(k)|^2
    # Running sum for S1(m) = 1/(T-m) sum_{k=0}^{T-m-1} [|r(k)|^2 + |r(k+m)|^2].
    # Going from lag m-1 to lag m drops the origin |r(m-1)|^2 from the first sum
    # and the endpoint |r(T-m)|^2 from the second, which is the whole trick.
    padded = np.concatenate([sq, np.zeros((1, sq.shape[1]))], axis=0)
    running = 2.0 * sq.sum(axis=0)
    s1 = np.empty((max_lag + 1, sq.shape[1]))
    for m in range(max_lag + 1):
        running = running - (padded[m - 1] if m > 0 else 0.0) - padded[n_frames - m]
        s1[m] = running / (n_frames - m)

    s2 = _fft_autocorrelation(r)[: max_lag + 1]
    return s1 - 2.0 * s2


def msd_naive(positions, max_lag: int | None = None) -> np.ndarray:
    """Per-atom MSD by direct summation. ``O(T^2 N)`` -- the reference.

    Parameters
    ----------
    positions : array_like, shape (T, N, 3)
        Unwrapped cartesian coordinates in angstrom.
    max_lag : int, optional
        Largest lag returned, in frames.  Defaults to ``T // 2``.

    Returns
    -------
    ndarray, shape (max_lag + 1, N)
        ``MSD[m, i]`` in A^2.

    Notes
    -----
    This is the textbook definition, transcribed.  It exists only so that
    :func:`msd_fft` can be validated against it; it is far too slow for the
    trajectory lengths this study uses.  The loop over time origins is expressed
    as an array operation rather than a Python loop -- the arithmetic performed
    is identical, and a reference implementation that takes an hour would not be
    run often enough to be a reference.  The same per-atom centring as
    :func:`msd_fft` is applied so that the two are compared on bit-identical
    input.
    """
    r = _as_series(positions)
    n_frames = r.shape[0]
    max_lag = _resolve_max_lag(max_lag, n_frames)
    r = r - r.mean(axis=0, keepdims=True)

    out = np.empty((max_lag + 1, r.shape[1]))
    for m in range(max_lag + 1):
        d = r[m:] - r[: n_frames - m]
        out[m] = np.einsum("tna,tna->tn", d, d).mean(axis=0)
    return out


@dataclass
class MSDResult:
    """Mean squared displacement as a function of lag time.

    Attributes
    ----------
    lags : ndarray, shape (L+1,), int
        Lags in frames.
    times : ndarray, shape (L+1,)
        Lag times in ps.
    msd : ndarray, shape (L+1,)
        MSD in A^2, averaged over atoms and over time origins.
    msd_per_atom : ndarray, shape (L+1, N)
        Per-atom MSD in A^2.  Kept because the spread over atoms is one of the
        two independent handles on the statistical error.
    msd_blocks : ndarray, shape (B, L+1) or None
        MSD computed independently in ``B`` contiguous, non-overlapping segments
        of the trajectory, in A^2.  These are the "independent time origins"
        used for the other error estimate.  ``None`` when the trajectory is too
        short to be split.
    dt : float
        Frame spacing in ps.
    n_frames, n_atoms : int
        Shape of the source trajectory.
    method : str
        ``"fft"`` or ``"naive"``.
    max_step : float
        Largest single-frame displacement seen, in angstrom (the unwrapping
        diagnostic).
    meta : dict
    """

    lags: np.ndarray
    times: np.ndarray
    msd: np.ndarray
    msd_per_atom: np.ndarray
    msd_blocks: np.ndarray | None
    dt: float
    n_frames: int
    n_atoms: int
    method: str
    max_step: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return 0 if self.msd_blocks is None else int(self.msd_blocks.shape[0])

    def to_observable_result(self) -> ObservableResult:
        """Expose the MSD curve through the package-wide container."""
        spread = self.msd_per_atom.std(axis=1, ddof=1) / np.sqrt(self.n_atoms)
        return _as_observable_result(
            "mean_squared_displacement",
            self.msd,
            spread,
            "A^2",
            {"times_ps": self.times, "lags": self.lags, "method": self.method},
        )


def mean_squared_displacement(
    trajectory: Trajectory,
    *,
    max_lag: int | None,
    use_fft: bool = True,
    n_blocks: int = 5,
    atom_indices: Sequence[int] | None = None,
    check_unwrapped: bool = True,
) -> MSDResult:
    """Mean squared displacement of a trajectory, averaged over time origins.

    Parameters
    ----------
    trajectory : Trajectory
        Positions must be **unwrapped** (they are, coming from
        :func:`atomlab.md.simulate.run_md`); this is asserted, see
        ``check_unwrapped``.
    max_lag : int or None
        Largest lag in frames.  ``None`` means ``T // 2``.  Keyword-only and
        mandatory: the right value depends on what the caller intends to fit and
        there is no defensible silent default.
    use_fft : bool
        Use the ``O(T log T)`` Fast Correlation Algorithm (:func:`msd_fft`).
        ``False`` selects the ``O(T^2)`` reference (:func:`msd_naive`), which
        exists for validation only.
    n_blocks : int
        Number of contiguous segments in which the MSD is *also* computed
        independently, to give an error bar over independent time origins.
        Reduced automatically (possibly to zero) if the trajectory is too short
        to give each block more than ``max_lag`` frames.
    atom_indices : sequence of int, optional
        Restrict the average to a subset of atoms, e.g. one species.
    check_unwrapped : bool
        Run the unwrapping assertion.  Only turn it off for a trajectory you
        have unwrapped yourself by another route.

    Returns
    -------
    MSDResult
        MSD in A^2 versus lag time in ps.

    Raises
    ------
    ValueError
        If the trajectory is not dynamical, or fails the unwrapping check.
    """
    _require_dynamical(trajectory, "mean_squared_displacement")
    max_step = _check_unwrapped(trajectory) if check_unwrapped else 0.0

    positions = trajectory.positions
    if atom_indices is not None:
        positions = positions[:, np.asarray(atom_indices, dtype=int), :]
    n_frames = positions.shape[0]
    max_lag = _resolve_max_lag(max_lag, n_frames)

    kernel = msd_fft if use_fft else msd_naive
    per_atom = kernel(positions, max_lag)
    msd = per_atom.mean(axis=1)

    # Independent time origins: each block must be long enough that its own
    # longest lag is still averaged over a few origins, hence the +2.
    usable = min(int(n_blocks), n_frames // (max_lag + 2))
    blocks: np.ndarray | None = None
    if usable >= 2:
        size = n_frames // usable
        blocks = np.stack(
            [kernel(positions[b * size : (b + 1) * size], max_lag).mean(axis=1) for b in range(usable)]
        )

    dt = trajectory.dt
    lags = np.arange(max_lag + 1)
    return MSDResult(
        lags=lags,
        times=lags * dt,
        msd=msd,
        msd_per_atom=per_atom,
        msd_blocks=blocks,
        dt=dt,
        n_frames=n_frames,
        n_atoms=positions.shape[1],
        method="fft" if use_fft else "naive",
        max_step=max_step,
        meta={"n_blocks_requested": int(n_blocks), "n_blocks_used": 0 if blocks is None else usable},
    )


# --------------------------------------------------------------------------
# diffusion: Einstein route
# --------------------------------------------------------------------------


@dataclass
class DiffusionResult:
    """Self-diffusion coefficient with its fit diagnostics.

    Attributes
    ----------
    value : float
        ``D`` in A^2/ps.
    error : float
        One standard error in A^2/ps.  See ``error_origins`` / ``error_atoms``.
    error_origins, error_atoms : float
        Error from block resampling over independent time origins, and from
        bootstrap resampling over atoms, both in A^2/ps.  ``error`` is their
        quadrature sum, which double counts and is therefore conservative --
        the two resamplings are different views of the same fluctuations, not
        independent contributions.  For a study whose entire content is a
        comparison between numbers, an error bar that is too large is the safe
        failure mode and one that is too small is not.
    route : str
        ``"einstein"`` or ``"green-kubo"``.
    r_squared : float
        Coefficient of determination of the straight-line fit (Einstein) or
        ``nan`` (Green--Kubo).
    exponent : float
        Local power-law exponent ``d log MSD / d log t`` over the fit window.
        ``1`` is diffusive, ``2`` ballistic, ``<1`` sub-diffusive or not yet
        converged.  This is the number that tells the caller whether the fit
        window was chosen in the right regime, and it is the reason the fit
        quality is returned rather than just the slope.
    intercept : float
        Fit intercept in A^2 (Einstein).  A large positive intercept means the
        window still contains the ballistic/caged transient.
    fit_range : tuple of float
        Window actually used, in ps.
    n_points : int
        Number of lag points in the window.
    meta : dict
    """

    value: float
    error: float
    error_origins: float
    error_atoms: float
    route: str
    r_squared: float = float("nan")
    exponent: float = float("nan")
    intercept: float = float("nan")
    fit_range: tuple[float, float] = (float("nan"), float("nan"))
    n_points: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def value_cm2_s(self) -> float:
        """``D`` in cm^2/s, the unit the transport literature quotes."""
        return self.value * A2_PS_TO_CM2_S

    @property
    def error_cm2_s(self) -> float:
        return self.error * A2_PS_TO_CM2_S

    def agrees_with(self, other: "DiffusionResult", n_sigma: float = 2.0) -> bool:
        """True if two estimates overlap within ``n_sigma`` combined errors."""
        sigma = float(np.hypot(self.error, other.error))
        return bool(abs(self.value - other.value) <= n_sigma * sigma)

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            f"diffusion_coefficient[{self.route}]",
            self.value,
            self.error,
            "A^2/ps",
            {
                "route": self.route,
                "r_squared": self.r_squared,
                "exponent": self.exponent,
                "fit_range_ps": self.fit_range,
                "value_cm2_s": self.value_cm2_s,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DiffusionResult({self.route}: D = {self.value:.6g} +/- {self.error:.3g} A^2/ps"
            f", exponent = {self.exponent:.3f})"
        )


def _fit_slope(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least squares ``y = a t + b``; returns ``(a, b, r_squared)``."""
    design = np.vstack([t, np.ones_like(t)]).T
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    residual = y - (slope * t + intercept)
    total = y - y.mean()
    denom = float(total @ total)
    r2 = 1.0 - float(residual @ residual) / denom if denom > 0 else float("nan")
    return slope, intercept, r2


def diffusion_coefficient(
    msd_result: MSDResult,
    fit_range: tuple[float, float],
    *,
    fit_range_units: Literal["ps", "lag"] = "ps",
    n_bootstrap: int = 200,
    seed: int = 0,
) -> DiffusionResult:
    """Self-diffusion coefficient from the Einstein relation.

    ``MSD(t) -> 6 D t`` in three dimensions, so ``D`` is one sixth of the slope
    of a straight line fitted over a lag window that the caller chooses.  The
    window matters: at short lag the motion is ballistic (``MSD ~ v^2 t^2``) and
    at long lag the estimator is dominated by the handful of independent origins
    left, so a fit over "all lags" is wrong at both ends.  The returned
    ``exponent`` and ``r_squared`` are what let the caller check the window was
    in the diffusive regime rather than assume it.

    Parameters
    ----------
    msd_result : MSDResult
        Output of :func:`mean_squared_displacement`.
    fit_range : tuple of float
        ``(start, stop)`` of the fit window, inclusive, in ps (or in frames if
        ``fit_range_units="lag"``).
    fit_range_units : {"ps", "lag"}
        Units of ``fit_range``.
    n_bootstrap : int
        Bootstrap resamples over atoms.
    seed : int
        Seeds a dedicated :class:`numpy.random.Generator`; nothing here touches
        global random state.

    Returns
    -------
    DiffusionResult
        ``value`` in A^2/ps.

    Raises
    ------
    ValueError
        If the window contains fewer than three lag points.
    """
    axis = msd_result.times if fit_range_units == "ps" else msd_result.lags.astype(float)
    lo, hi = float(fit_range[0]), float(fit_range[1])
    if hi <= lo:
        raise ValueError(f"fit_range must be increasing, got {fit_range}")
    mask = (axis >= lo) & (axis <= hi)
    if int(mask.sum()) < 3:
        raise ValueError(
            f"fit window {fit_range} ({fit_range_units}) contains {int(mask.sum())} lag points; "
            "need at least 3 to fit a line and judge its quality"
        )

    t = msd_result.times[mask]
    slope, intercept, r2 = _fit_slope(t, msd_result.msd[mask])
    value = slope / 6.0

    # Local power-law exponent: the diagnostic that separates ballistic from
    # diffusive.  Fitted on logs, so it needs strictly positive lag times and
    # MSD values -- lag zero is dropped rather than nudged.
    positive = mask & (msd_result.times > 0) & (msd_result.msd > 0)
    if int(positive.sum()) >= 2:
        exponent, _, _ = _fit_slope(
            np.log(msd_result.times[positive]), np.log(msd_result.msd[positive])
        )
    else:
        exponent = float("nan")

    # Error 1: independent time origins.  Each block is a separate stretch of
    # trajectory, so the spread of the per-block slopes is a direct measure of
    # how much the answer depends on which piece of the run was used.
    error_origins = 0.0
    block_values: np.ndarray | None = None
    if msd_result.msd_blocks is not None and msd_result.msd_blocks.shape[0] >= 2:
        block_values = np.array(
            [_fit_slope(t, curve[mask])[0] / 6.0 for curve in msd_result.msd_blocks]
        )
        error_origins = float(block_values.std(ddof=1) / np.sqrt(block_values.size))

    # Error 2: atoms.  Bootstrap whole per-atom MSD curves, which keeps each
    # atom's own time correlations intact and only resamples the population.
    rng = np.random.default_rng(seed)
    n_atoms = msd_result.msd_per_atom.shape[1]
    per_atom = msd_result.msd_per_atom[mask]
    idx = rng.integers(0, n_atoms, size=(n_bootstrap, n_atoms))
    resampled = per_atom[:, idx].mean(axis=2)  # (n_window, n_bootstrap)
    design = np.vstack([t, np.ones_like(t)]).T
    slopes, *_ = np.linalg.lstsq(design, resampled, rcond=None)
    error_atoms = float(np.std(slopes[0] / 6.0, ddof=1))

    error = float(np.hypot(error_origins, error_atoms))
    return DiffusionResult(
        value=value,
        error=error,
        error_origins=error_origins,
        error_atoms=error_atoms,
        route="einstein",
        r_squared=r2,
        exponent=float(exponent),
        intercept=intercept,
        fit_range=(float(t[0]), float(t[-1])),
        n_points=int(mask.sum()),
        meta={
            "slope_A2_per_ps": slope,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "block_values": block_values,
            "n_blocks": msd_result.n_blocks,
        },
    )


# --------------------------------------------------------------------------
# velocity autocorrelation and the Green-Kubo route
# --------------------------------------------------------------------------


def vacf_fft(velocities, max_lag: int | None = None) -> np.ndarray:
    """Per-atom velocity autocorrelation via FFT (Wiener--Khinchin).

    Parameters
    ----------
    velocities : array_like, shape (T, N, 3)
        Velocities in A/ps.
    max_lag : int, optional
        Largest lag in frames; defaults to ``T // 2``.

    Returns
    -------
    ndarray, shape (max_lag + 1, N)
        ``C[m, i] = <v_i(k) . v_i(k+m)>`` in A^2/ps^2, **not** normalised --
        the absolute scale is what the Green--Kubo integral needs.

    Notes
    -----
    Velocities are *not* mean-centred.  For the MSD the per-atom mean is a gauge
    freedom, but here the mean velocity is physical: subtracting it would remove
    a genuine drift contribution to the correlation and silently change the
    diffusion coefficient.  Any centre-of-mass drift should be removed in the
    dynamics (``run_md(..., fixed_com=True)``), not here.
    """
    v = _as_series(velocities)
    max_lag = _resolve_max_lag(max_lag, v.shape[0])
    return _fft_autocorrelation(v)[: max_lag + 1]


def vacf_naive(velocities, max_lag: int | None = None) -> np.ndarray:
    """Per-atom velocity autocorrelation by direct summation. ``O(T^2 N)``.

    The reference implementation for :func:`vacf_fft`, with the same
    ``1/(T-m)`` normalisation and the same units (A^2/ps^2).
    """
    v = _as_series(velocities)
    n_frames = v.shape[0]
    max_lag = _resolve_max_lag(max_lag, n_frames)
    out = np.empty((max_lag + 1, v.shape[1]))
    for m in range(max_lag + 1):
        out[m] = np.einsum("tna,tna->tn", v[m:], v[: n_frames - m]).mean(axis=0)
    return out


@dataclass
class VACFResult:
    """Normalised velocity autocorrelation function.

    Attributes
    ----------
    lags : ndarray, shape (L+1,), int
    times : ndarray, shape (L+1,)
        Lag times in ps.
    vacf : ndarray, shape (L+1,)
        Mass-weighted-free, atom-averaged VACF **normalised to 1 at zero lag**
        (dimensionless).
    c0 : float
        ``<|v|^2>`` at zero lag in A^2/ps^2, i.e. the scale that ``vacf`` was
        divided by.  ``c0 * vacf`` is the unnormalised correlation, which is
        what the Green--Kubo integral consumes.
    vacf_per_atom : ndarray, shape (L+1, N)
        Unnormalised per-atom correlations in A^2/ps^2.  Needed for the
        mass-weighted spectrum (:mod:`atomlab.observables.vdos`) and for the
        error bar over atoms.
    vacf_blocks : ndarray, shape (B, L+1) or None
        Unnormalised atom-averaged correlations from ``B`` independent
        contiguous segments, in A^2/ps^2.
    masses : ndarray, shape (N,)
        Atomic masses in amu, carried so that the spectrum can be mass weighted
        without needing the trajectory again.
    dt : float
        Frame spacing in ps.
    n_frames, n_atoms : int
    method : str
        ``"fft"`` or ``"naive"``.
    meta : dict
    """

    lags: np.ndarray
    times: np.ndarray
    vacf: np.ndarray
    c0: float
    vacf_per_atom: np.ndarray
    vacf_blocks: np.ndarray | None
    masses: np.ndarray
    dt: float
    n_frames: int
    n_atoms: int
    method: str
    meta: dict = field(default_factory=dict)

    @property
    def unnormalized(self) -> np.ndarray:
        """``<v(0).v(t)>`` in A^2/ps^2, shape ``(L+1,)``."""
        return self.c0 * self.vacf

    def mass_weighted(self) -> np.ndarray:
        """``sum_i m_i <v_i(0).v_i(t)> / sum_i m_i <|v_i|^2>``, shape ``(L+1,)``.

        This is the correlation function whose power spectrum is the vibrational
        density of states: mass weighting is what makes the spectral weight of a
        mode proportional to its number of degrees of freedom rather than to its
        amplitude, so that the integral counts modes.  For a single-species
        system it is identical to the plain average.
        """
        weighted = self.vacf_per_atom @ self.masses
        if weighted[0] == 0.0:
            raise ValueError("zero-lag mass-weighted VACF is zero; are all velocities zero?")
        return weighted / weighted[0]

    def to_observable_result(self) -> ObservableResult:
        spread = (self.vacf_per_atom / self.c0).std(axis=1, ddof=1) / np.sqrt(self.n_atoms)
        return _as_observable_result(
            "velocity_autocorrelation",
            self.vacf,
            spread,
            "dimensionless",
            {"times_ps": self.times, "c0_A2_ps2": self.c0, "method": self.method},
        )


def velocity_autocorrelation(
    trajectory: Trajectory,
    max_lag: int | None = None,
    use_fft: bool = True,
    *,
    n_blocks: int = 5,
    atom_indices: Sequence[int] | None = None,
) -> VACFResult:
    """Velocity autocorrelation function, normalised to 1 at zero lag.

    Parameters
    ----------
    trajectory : Trajectory
        Must carry velocities (``run_md(..., store_velocities=True)``).
    max_lag : int, optional
        Largest lag in frames; defaults to ``T // 2``.
    use_fft : bool
        FFT algorithm (default) or the ``O(T^2)`` reference.
    n_blocks : int
        Independent contiguous segments used for the time-origin error bar.
    atom_indices : sequence of int, optional
        Restrict to a subset of atoms.

    Returns
    -------
    VACFResult
        ``vacf`` is dimensionless with ``vacf[0] == 1``; the physical scale is
        in ``c0`` (A^2/ps^2).

    Raises
    ------
    ValueError
        If the trajectory is not dynamical or has no velocities.
    """
    _require_dynamical(trajectory, "velocity_autocorrelation")
    if trajectory.velocities is None:
        raise ValueError(
            "trajectory has no velocities; the velocity autocorrelation cannot be "
            "reconstructed from positions without differentiating them, which would "
            "change the answer at short lag. Re-run with store_velocities=True."
        )

    velocities = trajectory.velocities
    masses = trajectory.template.masses
    if atom_indices is not None:
        sel = np.asarray(atom_indices, dtype=int)
        velocities = velocities[:, sel, :]
        masses = masses[sel]

    n_frames = velocities.shape[0]
    max_lag = _resolve_max_lag(max_lag, n_frames)
    kernel = vacf_fft if use_fft else vacf_naive
    per_atom = kernel(velocities, max_lag)
    mean_curve = per_atom.mean(axis=1)
    c0 = float(mean_curve[0])
    if c0 <= 0.0:
        raise ValueError("zero-lag VACF is not positive; the velocities are all zero")

    usable = min(int(n_blocks), n_frames // (max_lag + 2))
    blocks: np.ndarray | None = None
    if usable >= 2:
        size = n_frames // usable
        blocks = np.stack(
            [
                kernel(velocities[b * size : (b + 1) * size], max_lag).mean(axis=1)
                for b in range(usable)
            ]
        )

    dt = trajectory.dt
    lags = np.arange(max_lag + 1)
    return VACFResult(
        lags=lags,
        times=lags * dt,
        vacf=mean_curve / c0,
        c0=c0,
        vacf_per_atom=per_atom,
        vacf_blocks=blocks,
        masses=np.asarray(masses, dtype=float),
        dt=dt,
        n_frames=n_frames,
        n_atoms=velocities.shape[1],
        method="fft" if use_fft else "naive",
        meta={"n_blocks_used": 0 if blocks is None else usable},
    )


def _integrate(curve: np.ndarray, dt: float) -> float:
    """``int_0^{t_max} C(t) dt`` by Simpson's rule (trapezoid for even length).

    Simpson's rule is exact for the locally-parabolic shape a VACF has near
    ``t = 0``, where the integrand is largest, so it beats the trapezoid rule
    where it matters most.
    """
    if curve.size < 3:
        return float(np.trapezoid(curve, dx=dt))
    return float(simpson(curve, dx=dt))


def diffusion_from_vacf(
    vacf_result: VACFResult,
    *,
    t_max: float | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> DiffusionResult:
    """Self-diffusion coefficient from the Green--Kubo relation.

    ``D = (1/3) int_0^inf <v(0).v(t)> dt``.  The factor ``1/3`` is the number of
    spatial dimensions, and it appears because ``vacf`` here is the correlation
    of the full vector ``v``, not of one component.

    The upper limit is finite in practice, and where it is truncated is the main
    systematic in this route: too early and the tail is lost, too late and the
    integral accumulates noise linearly in ``t_max`` while the signal has
    already decayed.  ``meta["running_integral"]`` holds ``D(t)`` for every
    intermediate upper limit so that the caller can see whether a plateau was
    reached instead of trusting one number.

    Parameters
    ----------
    vacf_result : VACFResult
        Output of :func:`velocity_autocorrelation`.
    t_max : float, optional
        Upper integration limit in ps.  Defaults to the full available lag
        range.
    n_bootstrap : int
        Bootstrap resamples over atoms.
    seed : int
        Seeds a dedicated generator.

    Returns
    -------
    DiffusionResult
        ``value`` in A^2/ps, ``route = "green-kubo"``.
    """
    times = vacf_result.times
    limit = float(times[-1] if t_max is None else t_max)
    mask = times <= limit + 1e-12
    if int(mask.sum()) < 3:
        raise ValueError(f"t_max = {limit} ps leaves fewer than 3 lag points to integrate")

    dt = vacf_result.dt
    curve = vacf_result.unnormalized[mask]
    value = _integrate(curve, dt) / 3.0

    running = np.array(
        [_integrate(curve[: k + 1], dt) / 3.0 if k >= 2 else 0.0 for k in range(curve.size)]
    )

    error_origins = 0.0
    block_values: np.ndarray | None = None
    if vacf_result.vacf_blocks is not None and vacf_result.vacf_blocks.shape[0] >= 2:
        block_values = np.array(
            [_integrate(curve_b[mask], dt) / 3.0 for curve_b in vacf_result.vacf_blocks]
        )
        error_origins = float(block_values.std(ddof=1) / np.sqrt(block_values.size))

    rng = np.random.default_rng(seed)
    n_atoms = vacf_result.vacf_per_atom.shape[1]
    per_atom = vacf_result.vacf_per_atom[mask]
    idx = rng.integers(0, n_atoms, size=(n_bootstrap, n_atoms))
    resampled = per_atom[:, idx].mean(axis=2)  # (n_window, n_bootstrap)
    boot = np.array([_integrate(resampled[:, b], dt) / 3.0 for b in range(n_bootstrap)])
    error_atoms = float(boot.std(ddof=1))

    # How many frames of the VACF are statistically independent: a long
    # correlation time relative to the integration window means the truncation
    # is cutting into signal, not noise.
    tau = integrated_autocorrelation_time(vacf_result.vacf)

    return DiffusionResult(
        value=value,
        error=float(np.hypot(error_origins, error_atoms)),
        error_origins=error_origins,
        error_atoms=error_atoms,
        route="green-kubo",
        fit_range=(0.0, float(times[mask][-1])),
        n_points=int(mask.sum()),
        meta={
            "running_integral": running,
            "running_times": times[mask],
            "tau_frames": tau,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "block_values": block_values,
        },
    )
