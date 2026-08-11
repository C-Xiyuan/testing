"""Shared machinery for trajectory-averaged observables.

Every structural observable in this package has the same shape: a *per-frame
kernel* maps one :class:`~atomlab.types.Configuration` to a vector of numbers
(one radial bin, one angular bin, one atom, or just one scalar), and the
estimator is the average of that vector over frames.  What is not trivial is
the error bar.  Successive frames of a trajectory -- MD or Monte Carlo -- are
correlated, so the naive standard error ``sigma / sqrt(T)`` underestimates the
true uncertainty by a factor ``sqrt(2 tau)`` with ``tau`` the integrated
autocorrelation time in units of the frame spacing.  Since the entire claim of
this repository is a *comparison* between numbers that are close together, an
error bar that is wrong by that factor is worse than no error bar at all.

This module therefore centralises three things:

1. :class:`ObservableResult` -- the value, its error, the x-axis it lives on,
   the number of samples, the measured correlation time, and (optionally) the
   per-frame samples themselves.  Keeping the samples is what makes derived
   quantities -- a coordination number integrated from ``g(r)``, a structure
   factor Fourier-transformed from it -- carry a *propagated* error rather than
   an invented one: the derived quantity is recomputed frame by frame and
   re-blocked, which automatically respects the correlations between bins.
2. :func:`estimate_from_samples` -- Flyvbjerg--Petersen blocking (via
   :mod:`atomlab.analysis.statistics`) applied column by column.
3. :func:`accumulate_frames` and the :func:`frame_estimator` decorator, so that
   a new observable only has to define its per-frame kernel.

Conventions
-----------
Frames are the unit of statistical weight: all blocking is over frames, never
over atoms or bins.  Atoms within a frame are strongly correlated with each
other, so treating them as independent samples would produce error bars that
are far too small; the per-frame average is the object that decorrelates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..analysis.statistics import blocking_analysis, integrated_autocorrelation_time
from ..types import Configuration, Trajectory

__all__ = [
    "ObservableResult",
    "FrameSample",
    "iter_frames",
    "estimate_from_samples",
    "accumulate_frames",
    "frame_estimator",
]


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------


@dataclass
class FrameSample:
    """What a per-frame kernel returns.

    Attributes
    ----------
    values : ndarray, shape (K,)
        The per-frame estimate.  ``K = 1`` for a scalar observable.  Units are
        whatever the observable is in; the kernel documents them.
    bins : ndarray, shape (K,) or None
        The x-axis the values live on (bin centres in A, in degrees, in 1/A,
        ...).  Must be identical for every frame; :func:`accumulate_frames`
        checks that and raises if a kernel silently changes its grid.
    metadata : dict
        Per-frame diagnostics.  Anything the frame-dependent normalisation
        needs later -- the cell volume, the effective density -- belongs here,
        because a constant-pressure run changes them frame by frame.
    """

    values: np.ndarray
    bins: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.atleast_1d(np.asarray(self.values, dtype=np.float64))
        if self.values.ndim != 1:
            raise ValueError(f"frame values must be 1-D, got shape {self.values.shape}")
        if self.bins is not None:
            self.bins = np.asarray(self.bins, dtype=np.float64)
            if self.bins.shape != self.values.shape:
                raise ValueError(
                    f"bins {self.bins.shape} and values {self.values.shape} must match"
                )


@dataclass
class ObservableResult:
    """A measured observable with a statistical error bar.

    Attributes
    ----------
    value : ndarray, shape (K,) or float
        Trajectory average of the per-frame estimates.
    error : ndarray, shape (K,) or float
        One standard error on ``value``, from blocking over frames.  ``nan``
        when a single frame was supplied -- one sample carries no information
        about its own scatter, and reporting zero there would be a lie.
    bins : ndarray, shape (K,) or None
        The x-axis: bin centres in A for ``g(r)``, degrees for an angular
        distribution, 1/A for ``S(q)``.  ``None`` for scalar observables.
    metadata : dict
        Free-form provenance: the normalisation used, densities, cutoffs, the
        per-frame metadata list under ``"per_frame"``.
    n_samples : int
        Number of frames that entered the average.
    correlation_time : float
        Integrated autocorrelation time ``tau`` in units of the *frame
        spacing* (not ps).  ``tau = 0.5`` means uncorrelated frames.
    samples : ndarray, shape (T, K) or None
        The per-frame estimates.  Retained by default so derived quantities can
        be re-blocked rather than error-propagated by hand.
    name : str
        Short identifier used in tables and figures.
    """

    value: np.ndarray | float
    error: np.ndarray | float
    bins: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    n_samples: int = 0
    correlation_time: float = float("nan")
    samples: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=np.float64)
        self.error = np.asarray(self.error, dtype=np.float64)
        if self.value.shape != self.error.shape:
            raise ValueError(
                f"value {self.value.shape} and error {self.error.shape} must have "
                "the same shape"
            )
        if self.bins is not None:
            self.bins = np.asarray(self.bins, dtype=np.float64)
            if self.bins.shape != self.value.shape:
                raise ValueError(
                    f"bins {self.bins.shape} must match value {self.value.shape}"
                )
        if self.samples is not None:
            self.samples = np.asarray(self.samples, dtype=np.float64)
            if self.samples.ndim != 2 or self.samples.shape[1:] != self.value.shape:
                raise ValueError(
                    f"samples {self.samples.shape} must be (T, {self.value.shape[0] if self.value.ndim else 1})"
                )
        self.n_samples = int(self.n_samples)
        self.correlation_time = float(self.correlation_time)

    # -- convenience -------------------------------------------------------

    @property
    def n_effective(self) -> float:
        """``T / (2 tau)``: the number of statistically independent frames."""
        if not np.isfinite(self.correlation_time) or self.correlation_time <= 0.0:
            return float("nan")
        return float(self.n_samples / (2.0 * self.correlation_time))

    @property
    def is_scalar(self) -> bool:
        """True if the observable is a single number rather than a curve."""
        return self.value.ndim == 0 or self.value.size == 1

    def scalar(self) -> tuple[float, float]:
        """Return ``(value, error)`` as floats; raises for a vector observable."""
        if not self.is_scalar:
            raise ValueError(
                f"{self.name or 'observable'} has {self.value.size} components; "
                "it is not a scalar"
            )
        return float(np.ravel(self.value)[0]), float(np.ravel(self.error)[0])

    def require_samples(self) -> np.ndarray:
        """``(T, K)`` per-frame samples, or a clear error if they were dropped."""
        if self.samples is None:
            raise ValueError(
                f"{self.name or 'observable'} was computed with keep_samples=False, "
                "so derived quantities cannot propagate its error; recompute with "
                "keep_samples=True"
            )
        return self.samples

    def transform(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        *,
        bins: np.ndarray | None = None,
        name: str = "",
        metadata: dict | None = None,
    ) -> "ObservableResult":
        """Apply ``function`` frame by frame and re-block the result.

        This is how every derived observable (coordination number, structure
        factor) gets its error bar.  Applying the functional to each frame and
        re-blocking is exact for linear functionals and correct to leading
        order for nonlinear ones, and -- unlike propagating the per-bin error
        bars -- it automatically accounts for the strong correlations *between*
        bins of the same frame, which for an integral over ``g(r)`` are the
        dominant contribution.

        Parameters
        ----------
        function : callable
            Maps a ``(K,)`` per-frame sample to a ``(K',)`` array (or scalar).
        bins : ndarray, shape (K',), optional
            x-axis of the transformed quantity.
        name : str
        metadata : dict, optional

        Returns
        -------
        ObservableResult
        """
        samples = self.require_samples()
        out = np.stack([np.atleast_1d(np.asarray(function(row), dtype=float)) for row in samples])
        meta = {"derived_from": self.name, **(metadata or {})}
        return estimate_from_samples(out, bins=bins, name=name, metadata=meta)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = f"ObservableResult({self.name!r}, T={self.n_samples}, tau={self.correlation_time:.2f}"
        if self.is_scalar:
            v, e = self.scalar()
            return head + f", {v:.6g} +/- {e:.3g})"
        return head + f", {self.value.size} bins)"


# --------------------------------------------------------------------------
# frame iteration
# --------------------------------------------------------------------------


def iter_frames(
    source: Trajectory | Configuration | Sequence[Configuration],
    stride: int = 1,
    *,
    start: int = 0,
    stop: int | None = None,
) -> list[Configuration]:
    """Normalise any frame source into a list of configurations.

    Accepting a bare :class:`~atomlab.types.Configuration` (or a list of them)
    alongside a :class:`~atomlab.types.Trajectory` is deliberate: a perfect
    crystal is a one-frame "trajectory", and the validation tests -- and the
    static reference calculations in ``exp01`` -- are much cleaner if they do
    not have to fabricate a Trajectory wrapper.

    Parameters
    ----------
    source : Trajectory or Configuration or sequence of Configuration
    stride : int
        Keep every ``stride``-th frame.  Striding is a legitimate way to
        decorrelate samples, but it throws information away; the blocking
        analysis handles correlation correctly at ``stride = 1``.
    start, stop : int
        Frame range, as for a slice.

    Returns
    -------
    list of Configuration
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if isinstance(source, Trajectory):
        indices = range(*slice(start, stop, stride).indices(source.n_frames))
        frames = [source.frame(i) for i in indices]
    elif isinstance(source, Configuration):
        frames = [source]
    else:
        seq = list(source)
        if not all(isinstance(c, Configuration) for c in seq):
            raise TypeError(
                "source must be a Trajectory, a Configuration, or a sequence of "
                "Configurations"
            )
        frames = seq[slice(start, stop, stride)]
    if not frames:
        raise ValueError("no frames selected")
    return frames


# --------------------------------------------------------------------------
# blocking
# --------------------------------------------------------------------------


def estimate_from_samples(
    samples: np.ndarray,
    *,
    bins: np.ndarray | None = None,
    name: str = "",
    metadata: dict | None = None,
    min_blocks: int = 8,
    keep_samples: bool = True,
) -> ObservableResult:
    """Average ``(T, K)`` per-frame samples and attach a blocking error bar.

    Parameters
    ----------
    samples : ndarray, shape (T, K)
        Per-frame estimates; rows are frames.  A ``(T,)`` input is treated as
        ``(T, 1)``.
    bins : ndarray, shape (K,), optional
        x-axis carried into the result.
    name : str
    metadata : dict, optional
    min_blocks : int
        Passed to :func:`~atomlab.analysis.statistics.blocking_analysis`: the
        plateau is taken over blocking levels that still leave at least this
        many blocks, since the "error on the error" beyond that is larger than
        the effect being measured.
    keep_samples : bool
        Store the samples in the result (needed by
        :meth:`ObservableResult.transform`).

    Returns
    -------
    ObservableResult
        With ``error = nan`` if only one frame was supplied.
    """
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"samples must be (T, K), got shape {a.shape}")
    n_frames = a.shape[0]
    mean = a.mean(axis=0)

    if n_frames == 1:
        # One sample says nothing about its own scatter.  NaN, not zero.
        error = np.full(mean.shape, np.nan)
        tau = float("nan")
    else:
        est = blocking_analysis(a, min_blocks=min_blocks)
        error = np.atleast_1d(np.asarray(est.error, dtype=float))
        # tau over the columns that are informative: a constant column (a bin
        # that never receives a count) has an undefined correlation time and
        # would otherwise poison the maximum taken across columns.
        varying = np.asarray(a.std(axis=0) > 0.0)
        tau = (
            float(integrated_autocorrelation_time(a[:, varying]))
            if varying.any()
            else 0.5
        )

    return ObservableResult(
        value=mean,
        error=error,
        bins=None if bins is None else np.asarray(bins, dtype=float),
        metadata=dict(metadata or {}),
        n_samples=n_frames,
        correlation_time=tau,
        samples=a if keep_samples else None,
        name=name,
    )


# --------------------------------------------------------------------------
# the accumulator and its decorator
# --------------------------------------------------------------------------


def accumulate_frames(
    source: Trajectory | Configuration | Sequence[Configuration],
    kernel: Callable[[Configuration], FrameSample | np.ndarray],
    *,
    stride: int = 1,
    name: str = "",
    metadata: dict | None = None,
    keep_samples: bool = True,
    min_blocks: int = 8,
) -> ObservableResult:
    """Run a per-frame kernel over a trajectory and block-average the result.

    Parameters
    ----------
    source : Trajectory or Configuration or sequence of Configuration
    kernel : callable
        ``kernel(configuration) -> FrameSample`` (or a bare ``(K,)`` array).
        It must return the *same* ``K`` and the same bins for every frame.
    stride : int
    name : str
    metadata : dict, optional
        Merged into the result's metadata, below the kernel's own first-frame
        metadata.
    keep_samples : bool
    min_blocks : int

    Returns
    -------
    ObservableResult
        ``metadata["per_frame"]`` is the list of per-frame metadata dicts, in
        frame order, so that frame-dependent normalisations (cell volume under
        NPT, for instance) remain available to derived quantities.
    """
    frames = iter_frames(source, stride)
    values: list[np.ndarray] = []
    per_frame: list[dict] = []
    bins: np.ndarray | None = None
    first_meta: dict = {}

    for index, cfg in enumerate(frames):
        out = kernel(cfg)
        if not isinstance(out, FrameSample):
            out = FrameSample(values=np.asarray(out, dtype=float))
        if index == 0:
            bins = out.bins
            first_meta = dict(out.metadata)
        else:
            if out.values.shape != values[0].shape:
                raise ValueError(
                    f"kernel returned {out.values.shape} on frame {index} but "
                    f"{values[0].shape} on frame 0"
                )
            if bins is not None and out.bins is not None and not np.allclose(bins, out.bins):
                raise ValueError(
                    f"kernel changed its bin grid at frame {index}; a trajectory "
                    "average is only meaningful on a fixed grid"
                )
        values.append(out.values)
        per_frame.append(out.metadata)

    meta = {**first_meta, **(metadata or {}), "per_frame": per_frame, "stride": int(stride)}
    return estimate_from_samples(
        np.stack(values),
        bins=bins,
        name=name,
        metadata=meta,
        min_blocks=min_blocks,
        keep_samples=keep_samples,
    )


def frame_estimator(kernel: Callable[..., FrameSample | np.ndarray]) -> Callable[..., ObservableResult]:
    """Decorator turning a per-frame kernel into a trajectory estimator.

    The decorated function is called as ``kernel(configuration, **params)`` and
    returns a :class:`FrameSample`.  The wrapper it produces has the signature

    ``wrapper(source, *, stride=1, keep_samples=True, min_blocks=8, name="", **params)``

    and returns an :class:`ObservableResult` with blocking error bars and a
    measured correlation time.  Observables in this package are written this
    way so that the statistics live in exactly one place: an estimator that
    invents its own error bar is the failure mode this whole module exists to
    prevent.

    Examples
    --------
    >>> @frame_estimator
    ... def mean_height(cfg, *, axis=2):
    ...     return FrameSample(values=[cfg.positions[:, axis].mean()])
    """

    @wraps(kernel)
    def wrapper(
        source: Trajectory | Configuration | Sequence[Configuration],
        *,
        stride: int = 1,
        keep_samples: bool = True,
        min_blocks: int = 8,
        name: str = "",
        metadata: dict | None = None,
        **params: Any,
    ) -> ObservableResult:
        return accumulate_frames(
            source,
            lambda cfg: kernel(cfg, **params),
            stride=stride,
            name=name or kernel.__name__.lstrip("_"),
            metadata=metadata,
            keep_samples=keep_samples,
            min_blocks=min_blocks,
        )

    wrapper.kernel = kernel  # type: ignore[attr-defined]
    return wrapper


def _require_periodic(configuration: Configuration, what: str) -> float:
    """Return the cell volume in A^3, or raise if the cell is not 3-D periodic.

    Structural observables that normalise by a number density are meaningless
    without a volume, and quietly returning an unnormalised histogram would
    contaminate every comparison made downstream.
    """
    if not np.asarray(configuration.pbc).all():
        raise ValueError(
            f"{what} requires a fully periodic cell (pbc = True in all three "
            f"directions); got pbc = {np.asarray(configuration.pbc)}"
        )
    volume = configuration.volume
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"{what} requires a non-degenerate cell, got volume {volume}")
    return float(volume)
