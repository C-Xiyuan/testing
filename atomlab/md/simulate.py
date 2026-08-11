"""The molecular dynamics driver: :func:`run_md`, :func:`minimize`, :func:`equilibrate`.

This is the entry point every experiment in the repository goes through, so it
carries the bookkeeping that is easy to get subtly wrong and expensive to
discover later.

Unwrapping
----------

:class:`~atomlab.types.Trajectory` stores **unwrapped** positions, because mean
squared displacements, diffusion coefficients and the velocity autocorrelation
are meaningless without them.  The naive way to produce them -- store whatever
the integrator has, then walk the stored frames afterwards folding any jump
larger than half a box back -- is wrong whenever an atom moves more than half a
box length *between stored frames*.  With ``frame_stride = 50`` at 1 fs that
needs a displacement of only a few angstrom per stored frame, which a hot light
atom in a small cell reaches easily, and the failure is silent: the trajectory
looks perfectly reasonable and the diffusion coefficient is wrong by a factor
of order one.

This driver therefore tracks the image counter at **every integrator step**,
not at every stored frame, and reports the unwrapped positions as
``r_raw + n_image @ cell``.  The integrators in this package keep unwrapped
positions themselves (see :class:`~atomlab.md.state.MDState`), in which case the
counter stays exactly zero and the recorded positions are bitwise the state's
own; the machinery costs one comparison per step and makes the driver correct
for a wrapping integrator too.

Neighbour lists
---------------

:class:`NeighborCache` is a Verlet-skin cache with a rebuild criterion that
survives a *deforming* cell, which the plain
:class:`~atomlab.neighbors.VerletList` does not: that class rebuilds whenever
the cell changes at all, so under NPT -- where the cell changes every single
step -- it degenerates into rebuilding from scratch every step, silently paying
the full cost the skin exists to avoid.  See
:meth:`SkinnedVerletList.needs_rebuild` for the bound that replaces it.

Pressure sign conventions
-------------------------

Everything logged in ``bar``; the virial ``W = -dU/d(eps)`` is in eV and the
pressure is ``(2 KE + tr W) / (3 V)``, positive under compression, exactly as
:class:`~atomlab.md.state.MDState` defines it.
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable as _CallableT, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from ..cell import check_minimum_image, min_cell_width, minimum_image_shift
from ..neighbors import NeighborList, VerletList, build_neighbor_list
from ..potentials.base import Potential
from ..types import Configuration, Trajectory
from ..units import BAR_TO_EV_A3, EV_A3_TO_BAR, KB
from .integrators import Integrator, Langevin, NoseHooverChain, VelocityVerlet
from .state import MDState

__all__ = [
    "Callback",
    "StepInfo",
    "MeasureEvery",
    "SkinnedVerletList",
    "NeighborCache",
    "run_md",
    "minimize",
    "MinimizeReport",
    "equilibrate",
]


# ==========================================================================
# callbacks
# ==========================================================================


@dataclass
class StepInfo:
    """Everything a per-step callback might need that is not on the state.

    Attributes
    ----------
    step : int
        Index within the current phase, ``0`` for the state before the first
        step of that phase.
    absolute_step : int
        ``state.step``, which keeps counting across thermalisation and across
        restarts.
    time : float
        Time in ps since the start of the *production* segment (so a trajectory
        always starts at ``t = 0``, which is what the correlation-function
        estimators assume).
    phase : {"thermalize", "production"}
        Thermalisation frames are never recorded; a callback still sees them
        unless it was registered with ``production_only``.
    positions : ndarray, shape (N, 3)
        Unwrapped positions in angstrom -- **not** ``state.positions``, which is
        whatever convention the integrator uses.
    scalars : dict
        The thermodynamic scalars logged at this step, or an empty dict if this
        was not a logging step.  Keys as in :func:`run_md`.
    """

    step: int
    absolute_step: int
    time: float
    phase: str
    positions: np.ndarray
    scalars: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Callback(Protocol):
    """Anything that can be hooked into :func:`run_md` per step.

    A callback is any callable ``f(state, info)``; it is invoked once per
    integrator step (including step 0, the initial state) and its return value
    is ignored.  The point of the hook is that an experiment measuring a cheap
    per-step quantity -- an order parameter, a bond count, a running histogram
    -- can accumulate it *online* instead of storing a whole ``(T, N, 3)``
    trajectory and post-processing it, which for a nanosecond run is the
    difference between megabytes and gigabytes.

    Callbacks that carry state may additionally define ``begin(state, info)``
    and ``finish(state, info)``; the driver calls them, when present, before the
    first step and after the last one.  They are looked up with ``getattr``, so
    a plain function remains a perfectly good callback.

    Notes
    -----
    ``state`` is the live :class:`~atomlab.md.state.MDState` and is mutated in
    place by the integrator.  A callback that wants to keep a snapshot must copy
    what it needs; a callback that mutates the state is changing the dynamics,
    which is occasionally the point (steered MD) and usually a bug.
    """

    def __call__(self, state: MDState, info: StepInfo) -> None:  # pragma: no cover - protocol
        ...


class MeasureEvery:
    """Call ``fn(state, info)`` every ``stride`` steps and keep the results.

    The standard way to use the :class:`Callback` hook: measure something cheap
    often, store only the numbers.

    Parameters
    ----------
    stride : int
        Call ``fn`` every ``stride`` steps of the production segment.
    fn : callable
        ``fn(state, info)`` returning whatever should be recorded (a float, an
        array, anything).
    production_only : bool
        Skip thermalisation steps.  Default True, because a measurement taken
        while the system is still equilibrating is not a sample of anything.

    Attributes
    ----------
    values : list
        Whatever ``fn`` returned, in order.
    times : list of float
        Simulation time in ps at which each value was taken.
    """

    def __init__(
        self,
        stride: int,
        fn: _CallableT[[MDState, StepInfo], Any],
        *,
        production_only: bool = True,
    ) -> None:
        if int(stride) < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = int(stride)
        self.fn = fn
        self.production_only = bool(production_only)
        self.values: list[Any] = []
        self.times: list[float] = []

    def __call__(self, state: MDState, info: StepInfo) -> None:
        if self.production_only and info.phase != "production":
            return
        if info.step % self.stride:
            return
        self.values.append(self.fn(state, info))
        self.times.append(info.time)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times, values)`` as arrays; times in ps."""
        return np.asarray(self.times, dtype=np.float64), np.asarray(self.values)


def _call_hook(callbacks: Sequence[Any], hook: str, state: MDState, info: StepInfo) -> None:
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is not None:
            fn(state, info)


# ==========================================================================
# neighbour-list caching
# ==========================================================================


class SkinnedVerletList(VerletList):
    """Verlet list whose rebuild criterion is valid under cell deformation.

    :class:`~atomlab.neighbors.VerletList` invalidates its list whenever the
    cell changes by so much as a bit, which is correct but useless under a
    barostat: the MTK propagator rescales the cell every step, so every step
    triggers a rebuild and the skin buys nothing.  The bound below replaces that
    all-or-nothing test.

    Let a pair carry image ``n`` (fixed for the lifetime of the list), and write
    positions in fractional coordinates of the *current* cell, ``r = s C``.  The
    pair vector is ``D = (s_j - s_i + n) C``, so between build time (subscript
    0) and now

    ``D - D0 = (du) C + u0 (C - C0)``

    with ``du = ds_j - ds_i`` and ``u0`` the fractional pair vector at build
    time.  Taking norms,

    ``|D - D0| <= 2 max_i |ds_i| smax(C) + r_search * smax(C - C0) / smin(C0)``

    using ``|u0| <= |D0| / smin(C0) <= r_search / smin(C0)``.  When that bound
    exceeds the skin, a pair that was outside ``cutoff + skin`` at build time
    could now be inside ``cutoff``, and only then is a rebuild required.  The
    bound is rigorous, not heuristic: no pair is ever missed.

    Two properties worth noting.  Pure affine scaling of the positions -- what a
    barostat does to every atom -- leaves ``ds`` exactly zero, so it contributes
    only through the second term, which grows like the accumulated strain and
    permits several percent of volume change per rebuild.  And for an unchanged
    cell the class falls back to the exact cartesian criterion
    ``2 max|dr| > skin`` rather than the (looser, anisotropy-inflated) bound.

    Parameters
    ----------
    cutoff : float
        Physical interaction range in angstrom.
    skin : float
        Verlet skin in angstrom.
    half : bool
        Build half lists (one entry per unordered pair).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._scaled: np.ndarray | None = None
        self._smin: float = 0.0
        #: Number of rebuilds that were forced by cell deformation rather than
        #: by atomic motion.  Diagnostic only, but it is the number that tells
        #: an NPT run whether ``skin`` or ``tau_p`` is the thing to change.
        self.n_deformation_builds: int = 0

    def build(self, configuration: Configuration) -> NeighborList:
        """Force a rebuild against ``configuration`` and return the new list."""
        nl = super().build(configuration)
        cell = np.asarray(configuration.cell, dtype=np.float64)
        if np.abs(np.linalg.det(cell)) > 1e-20:
            self._scaled = np.linalg.solve(cell.T, configuration.positions.T).T
            self._smin = float(np.linalg.svd(cell, compute_uv=False)[-1])
        else:  # aperiodic: fractional coordinates do not exist, and are not needed
            self._scaled = None
            self._smin = 0.0
        return nl

    def deformation_bound(self, positions, cell) -> float:
        """Upper bound on the change of any pair separation since the build, in A.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Current positions in angstrom (unwrapped or wrapped consistently
            with the build).
        cell : array_like, shape (3, 3)
            Current lattice vectors as rows, in angstrom.

        Returns
        -------
        float
            The bound described in the class docstring, in angstrom, or ``inf``
            if no list has been built yet.
        """
        if self.nl is None or self._scaled is None or self._cell is None:
            return float("inf")
        cell = np.asarray(cell, dtype=np.float64)
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != self._scaled.shape:
            return float("inf")
        scaled = np.linalg.solve(cell.T, pos.T).T
        d_scaled = np.sqrt(np.max(np.sum((scaled - self._scaled) ** 2, axis=1)))
        smax_cell = float(np.linalg.svd(cell, compute_uv=False)[0])
        smax_dcell = float(np.linalg.svd(cell - self._cell, compute_uv=False)[0])
        r_search = self.cutoff + self.skin
        return float(2.0 * d_scaled * smax_cell + r_search * smax_dcell / max(self._smin, 1e-300))

    def needs_rebuild(self, positions, cell=None) -> bool:
        """True if the cached list can no longer be trusted.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Current positions in angstrom.  Must be **unwrapped** (or at least
            continuous), for the reason given in
            :class:`~atomlab.neighbors.VerletList`.
        cell : array_like, shape (3, 3), optional
            Current cell.  ``None`` means "unchanged".

        Returns
        -------
        bool
        """
        if self.nl is None or self._positions is None:
            return True
        if cell is None or np.array_equal(np.asarray(cell, dtype=np.float64), self._cell):
            return self.max_displacement(positions) > 0.5 * self.skin
        return self.deformation_bound(positions, cell) > self.skin

    def update(self, configuration: Configuration) -> NeighborList:
        """Return a valid list for ``configuration``, rebuilding only if needed."""
        if self.needs_rebuild(configuration.positions, configuration.cell):
            deformed = self._cell is not None and not np.array_equal(
                np.asarray(configuration.cell, dtype=np.float64), self._cell
            )
            nl = self.build(configuration)
            if deformed:
                self.n_deformation_builds += 1
            return nl
        assert self.nl is not None
        return self.nl


def _walk_potentials(potential: Potential) -> Iterator[Potential]:
    """Yield ``potential`` and every sub-potential of a sum/scaled expression.

    The perturbation experiments run ``U0 + dU`` as a
    :class:`~atomlab.potentials.base.SumPotential`, and each term keeps its own
    neighbour list, so the cache has to reach inside.
    """
    yield potential
    for term in getattr(potential, "terms", ()) or ():
        yield from _walk_potentials(term)
    base = getattr(potential, "base", None)
    if isinstance(base, Potential):
        yield from _walk_potentials(base)


class NeighborCache:
    """Driver-level Verlet neighbour-list cache with rebuild accounting.

    Two ways to use it.

    *Standalone.*  ``cache.update(configuration)`` returns a
    :class:`~atomlab.neighbors.NeighborList` valid at that geometry, rebuilding
    only when :class:`SkinnedVerletList` says it must.

    *Attached to a potential.*  ``with cache.attached(potential):`` installs a
    :class:`SkinnedVerletList` into every sub-potential that keeps a Verlet slot
    (the pair potentials of this package do, via their ``skin`` constructor
    argument), so the saving is realised inside the force loop where the pairs
    are actually consumed, and restores the potential's own lists on exit --
    ``run_md`` must not leave a potential object it was handed in a different
    state than it found it.  Each attached potential gets a list built at *its*
    own cutoff, which is what makes ``U0 + dU`` with different ranges work.

    Parameters
    ----------
    cutoff : float
        Cutoff for the standalone list, in angstrom.
    skin : float
        Verlet skin in angstrom.  0.3-1.0 A is usual for condensed phases: too
        small and it rebuilds constantly, too large and every force evaluation
        drags useless pairs around.
    half : bool
        Build half lists for the standalone list.

    Attributes
    ----------
    n_rebuilds : int
        Total list constructions, summed over the standalone list and every
        attached potential.  A run in which this is comparable to the step count
        is a run whose skin is doing nothing.
    n_queries : int
        Number of times a list was requested from the standalone path.
    """

    def __init__(self, cutoff: float, *, skin: float = 0.5, half: bool = True) -> None:
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        if self.skin <= 0.0:
            raise ValueError(f"a neighbour cache needs a positive skin, got {skin}")
        self.half = bool(half)
        self.list = SkinnedVerletList(cutoff=self.cutoff, skin=self.skin, half=self.half)
        self.n_queries = 0
        self._attached: list[tuple[Potential, Any]] = []
        self._attached_lists: list[SkinnedVerletList] = []

    # -- standalone --------------------------------------------------------

    def update(self, configuration: Configuration) -> NeighborList:
        """Return a list valid at ``configuration``, rebuilding only if needed."""
        self.n_queries += 1
        return self.list.update(configuration)

    @property
    def n_rebuilds(self) -> int:
        """Total number of neighbour-list constructions."""
        return int(self.list.n_builds + sum(v.n_builds for v in self._attached_lists))

    @property
    def n_deformation_rebuilds(self) -> int:
        """Rebuilds triggered by cell deformation rather than atomic motion."""
        return int(
            self.list.n_deformation_builds
            + sum(v.n_deformation_builds for v in self._attached_lists)
        )

    # -- attachment --------------------------------------------------------

    def attach(self, potential: Potential) -> list[Potential]:
        """Install skinned lists into every sub-potential that keeps one.

        Returns
        -------
        list of Potential
            The sub-potentials that accepted a cache.  An empty list means the
            model does its own neighbour bookkeeping (or none), which is not an
            error but is worth knowing: ``run_md`` records it in
            ``Trajectory.info``.
        """
        targets: list[Potential] = []
        for sub in _walk_potentials(potential):
            if not hasattr(sub, "_verlet"):
                continue
            cutoff = float(getattr(sub, "cutoff", self.cutoff))
            if not np.isfinite(cutoff) or cutoff <= 0.0:
                continue
            half = getattr(getattr(sub, "_verlet", None), "half", True)
            fresh = SkinnedVerletList(cutoff=cutoff, skin=self.skin, half=bool(half))
            self._attached.append((sub, sub._verlet))
            self._attached_lists.append(fresh)
            sub._verlet = fresh
            targets.append(sub)
        return targets

    def detach(self) -> None:
        """Restore every attached potential's original neighbour-list slot."""
        for sub, original in reversed(self._attached):
            sub._verlet = original
        self._attached.clear()

    def attached(self, potential: Potential):
        """Context manager wrapping :meth:`attach` / :meth:`detach`."""

        class _Ctx:
            def __init__(ctx, cache: "NeighborCache") -> None:
                ctx.cache = cache
                ctx.targets: list[Potential] = []

            def __enter__(ctx):
                ctx.targets = ctx.cache.attach(potential)
                return ctx.targets

            def __exit__(ctx, *exc):
                ctx.cache.detach()
                return False

        return _Ctx(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NeighborCache(cutoff={self.cutoff:.3f}, skin={self.skin:.3f}, "
            f"rebuilds={self.n_rebuilds})"
        )


# ==========================================================================
# the driver
# ==========================================================================


def _is_stochastic(integrator: Integrator) -> bool:
    """True if the integrator consumes random numbers during a step."""
    if isinstance(integrator, Langevin):
        return integrator.friction > 0.0
    return bool(getattr(integrator, "stochastic", False))


class _Progress:
    """Throttled progress reporting to stderr.

    Prints at most every ``interval`` seconds so that a fast, cheap potential
    does not spend its time formatting strings.
    """

    def __init__(self, enabled: bool, total: int, *, interval: float = 2.0, stream=None) -> None:
        self.enabled = bool(enabled) and total > 0
        self.total = int(total)
        self.interval = float(interval)
        self.stream = stream if stream is not None else sys.stderr
        self.t0 = _time.monotonic()
        self._last = self.t0

    def update(self, step: int, state: MDState, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = _time.monotonic()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        elapsed = now - self.t0
        rate = step / elapsed if elapsed > 0 and step > 0 else 0.0
        eta = (self.total - step) / rate if rate > 0 else float("nan")
        try:
            temp = f"{state.temperature():8.2f} K"
        except Exception:  # pragma: no cover - defensive
            temp = "       ? K"
        self.stream.write(
            f"[md] step {step:>9d}/{self.total:<9d} t={state.time:10.4f} ps  T={temp}"
            f"  {rate:8.1f} steps/s  eta {eta:8.1f} s\n"
        )
        self.stream.flush()


def _scalar_bundle(
    state: MDState,
    integrator: Integrator,
    *,
    want_pressure: bool,
    want_conserved: bool,
) -> dict[str, float]:
    """Thermodynamic scalars at the current state, in metal units (bar for P)."""
    kinetic = state.kinetic_energy()
    potential_energy = state.potential_energy
    out = {
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic,
        "total_energy": kinetic + potential_energy,
        "temperature": 2.0 * kinetic / (state.n_dof * KB),
        "volume": state.volume,
    }
    if want_pressure:
        out["pressure"] = state.pressure(unit="bar")
    if want_conserved:
        out["conserved_quantity"] = float(integrator.conserved_quantity(state))
    return out


class _Unwrapper:
    """Incremental image accounting, evaluated once per integrator step.

    Holds an integer image counter ``n`` such that the unwrapped position is
    ``r_raw + n @ cell``.  Each step it compares the raw displacement against
    half the smallest perpendicular cell width: anything smaller cannot be a
    boundary crossing (a lattice translation is never shorter than that width),
    anything larger is folded by :func:`atomlab.cell.minimum_image_shift` and
    the fold is accumulated.

    For the integrators in this package -- which keep unwrapped positions
    themselves -- the counter stays exactly zero and the returned positions are
    bitwise the state's own, so nothing is lost by running this unconditionally.
    """

    def __init__(self, state: MDState) -> None:
        self.pbc = np.asarray(state.pbc, dtype=bool)
        self.n_image = np.zeros(state.positions.shape, dtype=np.int64)
        self.previous = state.positions.copy()
        self.events = 0
        self._cell_cache: np.ndarray | None = None
        self._half_width = float("inf")

    def _threshold(self, cell: np.ndarray) -> float:
        if self._cell_cache is None or not np.array_equal(cell, self._cell_cache):
            self._cell_cache = cell.copy()
            self._half_width = 0.5 * min_cell_width(cell, self.pbc)
        return self._half_width

    def observe(self, state: MDState) -> None:
        """Record one integrator step's worth of motion."""
        positions = state.positions
        if not self.pbc.any():
            self.previous = positions.copy()
            return
        delta = positions - self.previous
        if float(np.max(np.sum(delta * delta, axis=1))) > self._threshold(state.cell) ** 2:
            _, shift = minimum_image_shift(delta, state.cell, self.pbc)
            self.n_image += shift.astype(np.int64)
            self.events += int(np.count_nonzero(shift.any(axis=1)))
        self.previous = positions.copy()

    def positions(self, state: MDState) -> np.ndarray:
        """``(N, 3)`` unwrapped positions in angstrom."""
        if not self.n_image.any():
            return state.positions.copy()
        return state.positions + self.n_image.astype(np.float64) @ state.cell


def run_md(
    configuration: Configuration,
    potential: Potential,
    integrator: Integrator,
    *,
    n_steps: int,
    frame_stride: int = 1,
    log_stride: int = 1,
    seed: int | np.random.Generator | None = None,
    thermalize_steps: int = 0,
    callbacks: Sequence[Any] = (),
    progress: bool = True,
    store_velocities: bool = True,
    max_wall_seconds: float | None = None,
    temperature: float | None = None,
    velocities: np.ndarray | None = None,
    fixed_com: bool = True,
    remove_com_every: int = 0,
    neighbor_skin: float | None = None,
    state: MDState | None = None,
    progress_stream=None,
) -> Trajectory:
    """Run molecular dynamics and return the recorded trajectory.

    Parameters
    ----------
    configuration : Configuration
        Starting geometry; positions in angstrom, cell rows as lattice vectors.
        Not modified.  Ignored (except for provenance) if ``state`` is given.
    potential : Potential
        Energy model.  Restored untouched on exit even when a neighbour cache
        was attached to it.
    integrator : Integrator
        Any :class:`~atomlab.md.integrators.Integrator`; its ``dt`` is in ps.
    n_steps : int
        Number of production steps.  ``0`` is legal and records the initial
        state only.
    frame_stride : int
        Store a frame every ``frame_stride`` production steps.  Frame 0 is the
        state at the start of production, so ``T = n_steps // frame_stride + 1``
        frames are recorded for a run that is not cut short.
    log_stride : int
        Compute the thermodynamic scalars every ``log_stride`` steps.  The
        frame-aligned subset lands in ``Trajectory.scalars`` (which the type
        contract requires to be ``(T,)``); the full, finer-grained series lands
        in ``Trajectory.info["log"]`` together with its own ``"time"`` axis.
    seed : int or numpy.random.Generator, optional
        Seeds the state's generator.  Required whenever randomness is actually
        used (drawing initial velocities, or a stochastic integrator): passing
        ``None`` there raises rather than silently producing an unreproducible
        run.
    thermalize_steps : int
        Steps to run and discard before recording starts.  Callbacks see them
        with ``info.phase == "thermalize"``; nothing else does.
    callbacks : sequence of Callback
        Per-step hooks, see :class:`Callback`.
    progress : bool
        Write a throttled progress line to stderr.
    store_velocities : bool
        Store ``(T, N, 3)`` velocities in A/ps.  Turn off for long runs where
        only structure is wanted; the velocity autocorrelation obviously needs
        them.
    max_wall_seconds : float, optional
        Stop cleanly once this much wall-clock time has elapsed since the call
        started (thermalisation included) and return the frames recorded so far,
        instead of being killed by a scheduler with nothing to show.  Whether it
        triggered is recorded in ``Trajectory.info["wall_time_exceeded"]``, and
        an analysis script that silently accepts a truncated run is a bug -- so
        the flag is always present, not only when it fired.
    temperature : float, optional
        Temperature in K for the initial Maxwell-Boltzmann draw.  Defaults to
        the integrator's setpoint if it has one, else to a start from rest.
    velocities : ndarray, shape (N, 3), optional
        Explicit initial velocities in A/ps; overrides ``temperature``.
    fixed_com : bool
        Remove the centre-of-mass momentum at the start and count ``3N - 3``
        degrees of freedom.
    remove_com_every : int
        Re-zero the centre-of-mass momentum every this many steps (``0``
        disables).  Newtonian dynamics conserves it exactly, so this is only
        needed to mop up round-off in very long runs -- and doing it *often* is
        a way to hide a broken thermostat, so it defaults to off.
    neighbor_skin : float, optional
        Attach a :class:`NeighborCache` with this skin (in angstrom) to the
        potential for the duration of the run.
    state : MDState, optional
        Continue from an existing state (restart, or the production leg of
        :func:`equilibrate`).  ``configuration``, ``temperature``,
        ``velocities`` and ``seed`` are then not used to build one.
    progress_stream : file-like, optional
        Where progress goes; defaults to ``sys.stderr``.

    Returns
    -------
    Trajectory
        Positions ``(T, N, 3)`` **unwrapped**, in angstrom; ``cells``
        ``(T, 3, 3)``; ``times`` ``(T,)`` in ps measured from the start of
        production; ``velocities`` ``(T, N, 3)`` in A/ps if requested.
        ``scalars`` holds ``potential_energy``, ``kinetic_energy``,
        ``total_energy`` (eV), ``temperature`` (K), ``pressure`` (bar),
        ``volume`` (A^3) and ``conserved_quantity`` (eV) when the integrator
        exposes one.  ``info`` records the run parameters, the neighbour-list
        rebuild count, the final :class:`~atomlab.md.state.MDState` under
        ``"final_state"`` for restarts, and ``"dynamical" = True`` so that the
        dynamical estimators of :mod:`atomlab.observables` accept it.

    Raises
    ------
    ValueError
        If randomness is required and no seed was given, or if the strides are
        not positive.
    """
    n_steps = int(n_steps)
    thermalize_steps = int(thermalize_steps)
    frame_stride = int(frame_stride)
    log_stride = int(log_stride)
    if n_steps < 0 or thermalize_steps < 0:
        raise ValueError("n_steps and thermalize_steps must be >= 0")
    if frame_stride < 1 or log_stride < 1:
        raise ValueError("frame_stride and log_stride must be >= 1")

    t_start = _time.monotonic()

    if state is None:
        if temperature is None:
            temperature = getattr(integrator, "temperature", None)
        needs_random = velocities is None and temperature is not None and temperature > 0.0
        if seed is None and (needs_random or _is_stochastic(integrator)):
            raise ValueError(
                "run_md needs an explicit seed: this run draws random numbers "
                "(initial velocities and/or a stochastic integrator) and an "
                "unseeded run is not reproducible"
            )
        state = MDState.from_configuration(
            configuration,
            potential=None,
            temperature=temperature,
            seed=seed,
            fixed_com=fixed_com,
            velocities=velocities,
        )
    elif seed is not None:
        raise ValueError("pass either `state` or `seed`, not both: reseeding a live state "
                         "would silently restart its random stream")

    cache = None if neighbor_skin is None else NeighborCache(
        cutoff=float(getattr(potential, "cutoff", 0.0)) or 1.0, skin=float(neighbor_skin)
    )
    attached: list[Potential] = []

    try:
        if cache is not None:
            attached = cache.attach(potential)

        integrator.initialize(state, potential)

        want_pressure = bool(getattr(integrator, "virial", True)) and state.result.virial is not None
        try:
            integrator.conserved_quantity(state)
            want_conserved = True
        except (NotImplementedError, RuntimeError):
            want_conserved = False

        unwrapper = _Unwrapper(state)
        reporter = _Progress(
            progress, thermalize_steps + n_steps, stream=progress_stream
        )
        wall_exceeded = False

        def _elapsed() -> float:
            return _time.monotonic() - t_start

        def _out_of_time() -> bool:
            return max_wall_seconds is not None and _elapsed() >= float(max_wall_seconds)

        # -- thermalisation: stepped, never recorded -----------------------
        info0 = StepInfo(
            step=0,
            absolute_step=state.step,
            time=0.0,
            phase="thermalize" if thermalize_steps else "production",
            positions=unwrapper.positions(state),
        )
        _call_hook(callbacks, "begin", state, info0)

        done_thermal = 0
        for k in range(thermalize_steps):
            if _out_of_time():
                wall_exceeded = True
                break
            for cb in callbacks:
                cb(
                    state,
                    StepInfo(
                        step=k,
                        absolute_step=state.step,
                        time=0.0,
                        phase="thermalize",
                        positions=unwrapper.positions(state),
                    ),
                )
            integrator.step(state, potential)
            unwrapper.observe(state)
            if remove_com_every and (k + 1) % remove_com_every == 0:
                state.zero_com_momentum()
            done_thermal = k + 1
            reporter.update(done_thermal, state)

        # -- production ----------------------------------------------------
        t0 = state.time
        frames: list[np.ndarray] = []
        cells: list[np.ndarray] = []
        times: list[float] = []
        vels: list[np.ndarray] = []
        frame_scalars: dict[str, list[float]] = {}
        log_scalars: dict[str, list[float]] = {}
        log_times: list[float] = []
        done_production = 0

        for k in range(n_steps + 1):
            if wall_exceeded or (k and _out_of_time()):
                wall_exceeded = True
                break

            is_frame = (k % frame_stride) == 0
            is_log = (k % log_stride) == 0
            scalars: dict[str, float] = {}
            if is_frame or is_log:
                scalars = _scalar_bundle(
                    state,
                    integrator,
                    want_pressure=want_pressure,
                    want_conserved=want_conserved,
                )
            if is_log:
                log_times.append(state.time - t0)
                for key, value in scalars.items():
                    log_scalars.setdefault(key, []).append(value)
            if is_frame:
                frames.append(unwrapper.positions(state))
                cells.append(state.cell.copy())
                times.append(state.time - t0)
                if store_velocities:
                    vels.append(state.velocities.copy())
                for key, value in scalars.items():
                    frame_scalars.setdefault(key, []).append(value)

            if callbacks:
                info = StepInfo(
                    step=k,
                    absolute_step=state.step,
                    time=state.time - t0,
                    phase="production",
                    positions=frames[-1] if is_frame else unwrapper.positions(state),
                    scalars=scalars,
                )
                for cb in callbacks:
                    cb(state, info)

            if k == n_steps:
                break
            integrator.step(state, potential)
            unwrapper.observe(state)
            if remove_com_every and (k + 1) % remove_com_every == 0:
                state.zero_com_momentum()
            done_production = k + 1
            reporter.update(done_thermal + done_production, state)

        reporter.update(done_thermal + done_production, state, force=True)

        n_atoms = state.n_atoms
        empty = np.zeros((0, n_atoms, 3))
        trajectory = Trajectory(
            positions=np.asarray(frames) if frames else empty,
            cells=np.asarray(cells) if cells else np.zeros((0, 3, 3)),
            times=np.asarray(times) if times else np.zeros(0),
            template=state.to_configuration(labels=False).stripped(),
            velocities=(
                (np.asarray(vels) if vels else empty) if store_velocities else None
            ),
            scalars={k: np.asarray(v, dtype=np.float64) for k, v in frame_scalars.items()},
        )
        final_info = StepInfo(
            step=done_production,
            absolute_step=state.step,
            time=state.time - t0,
            phase="production",
            positions=unwrapper.positions(state),
        )
        _call_hook(callbacks, "finish", state, final_info)

        trajectory.info.update(
            dynamical=True,
            sampler=f"md:{getattr(integrator, 'name', type(integrator).__name__)}",
            integrator=getattr(integrator, "name", type(integrator).__name__),
            potential=getattr(potential, "name", type(potential).__name__),
            dt=integrator.dt,
            n_steps_requested=n_steps,
            n_steps_completed=done_production,
            thermalize_steps_requested=thermalize_steps,
            thermalize_steps_completed=done_thermal,
            frame_stride=frame_stride,
            log_stride=log_stride,
            seed=None if isinstance(seed, np.random.Generator) else seed,
            fixed_com=state.fixed_com,
            n_dof=state.n_dof,
            temperature_setpoint=getattr(integrator, "temperature", None),
            pressure_logged=want_pressure,
            wall_time_exceeded=wall_exceeded,
            wall_seconds=_elapsed(),
            max_wall_seconds=max_wall_seconds,
            unwrap_events=unwrapper.events,
            neighbor_skin=neighbor_skin,
            neighbor_rebuilds=None if cache is None else cache.n_rebuilds,
            neighbor_cache_targets=[getattr(p, "name", "?") for p in attached],
            log={"time": np.asarray(log_times, dtype=np.float64)}
            | {k: np.asarray(v, dtype=np.float64) for k, v in log_scalars.items()},
            final_state=state,
        )
        return trajectory
    finally:
        if cache is not None:
            cache.detach()


# ==========================================================================
# relaxation
# ==========================================================================


@dataclass
class MinimizeReport:
    """Outcome of a :func:`minimize` call.

    Attributes
    ----------
    converged : bool
        Whether **both** criteria were met: max per-atom force norm below
        ``fmax`` (eV/A), and, for a variable-cell relaxation, the scaled cell
        residual below ``fmax`` too.
    fmax : float
        Final max per-atom force norm in eV/A.
    initial_fmax : float
        The same before relaxation, in eV/A.
    cell_residual : float
        ``max |W - P V I| / cell_factor`` in eV, the quantity the cell degrees
        of freedom are driven to zero; ``0`` for a fixed-cell relaxation.
    stress_residual : float
        ``max |W/V - P I|`` in eV/A^3 -- the same thing without the conditioning
        factor, which is the physically meaningful deviation from the target
        stress.
    energy, initial_energy : float
        Potential energy in eV (not enthalpy, even when a target pressure was
        applied).
    pressure_bar : float
        Virial pressure ``tr(W)/(3V)`` of the relaxed structure, in bar.
    target_pressure_bar : float
        What was asked for, in bar.
    volume : float
        Final cell volume in A^3.
    n_iterations, n_evaluations : int
        Optimiser iterations and potential evaluations used.
    method : str
        Optimiser actually used.
    message : str
        Terminating message from SciPy, or the reason we stopped.
    """

    converged: bool
    fmax: float
    initial_fmax: float
    cell_residual: float
    stress_residual: float
    energy: float
    initial_energy: float
    pressure_bar: float
    target_pressure_bar: float
    volume: float
    n_iterations: int
    n_evaluations: int
    method: str
    message: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "converged" if self.converged else "NOT converged"
        return (
            f"MinimizeReport({status}, fmax={self.fmax:.3e} eV/A, "
            f"E={self.energy:.6f} eV, {self.n_iterations} it, "
            f"{self.n_evaluations} evals)"
        )


_SCIPY_METHODS = {"lbfgs": "L-BFGS-B", "bfgs": "BFGS", "cg": "CG"}


def _max_force(forces: np.ndarray) -> float:
    """Largest per-atom force norm in eV/A (``0`` for an empty system)."""
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces, axis=1)))


def minimize(
    configuration: Configuration,
    potential: Potential,
    *,
    method: str = "lbfgs",
    fmax: float = 1e-4,
    max_steps: int = 500,
    relax_cell: bool = False,
    pressure_bar: float = 0.0,
    cell_factor: float | None = None,
    hydrostatic: bool = False,
    max_restarts: int = 8,
) -> tuple[Configuration, MinimizeReport]:
    """Relax a structure with SciPy and analytic gradients.

    Phonon calculations are the reason this exists: the dynamical matrix is only
    the second derivative of the energy *at a stationary point*, and a residual
    force of 1e-2 eV/A on a rattled starting structure puts imaginary modes into
    an otherwise perfectly good band structure.  ``fmax`` defaults accordingly to
    ``1e-4 eV/A``, which is tighter than a typical geometry optimisation and
    cheap for an analytic potential.

    Parameters
    ----------
    configuration : Configuration
        Starting geometry; not modified.
    potential : Potential
        Energy model.  Its ``compute`` supplies the analytic gradient, so no
        finite differences are involved anywhere in this routine.
    method : {"lbfgs", "bfgs", "cg"}
        SciPy optimiser.  ``lbfgs`` (L-BFGS-B) is the default: it is the only
        one of the three whose memory cost stays linear in ``3N``.
    fmax : float
        Force convergence threshold in eV/A, applied to the largest per-atom
        force *norm* (not component).
    max_steps : int
        Maximum optimiser iterations, summed over restarts.
    relax_cell : bool
        Also relax the lattice vectors, driving the virial to the target
        pressure.  Requires a periodic cell.
    pressure_bar : float
        Target pressure in **bar**.  The minimised objective is then the
        enthalpy ``U + P V``; its stationary point satisfies
        ``tr(W)/(3V) = P`` with ``W`` the virial, which is what makes this a
        pressure and not a Lagrange multiplier with the wrong sign.
    cell_factor : float, optional
        Conditioning factor between the position and strain degrees of freedom.
        The strain gradient is an extensive quantity (it scales with ``N``)
        while a force is intensive, so without rescaling L-BFGS builds a
        hopeless Hessian estimate for anything but a handful of atoms.
        Defaults to ``n_atoms``.
    hydrostatic : bool
        Restrict the cell degrees of freedom to a uniform dilation.  Useful for
        a cubic crystal, where the six-component strain is free to break the
        symmetry along a soft direction and produce a slightly triclinic
        "cubic" cell.
    max_restarts : int
        L-BFGS-B stops on its own function-change criterion long before the
        gradient criterion on a stiff problem; when that happens with the force
        criterion still unmet, the optimiser is restarted from where it stopped
        (which resets its Hessian memory) up to this many times.

    Returns
    -------
    relaxed : Configuration
        Copy carrying the relaxed positions, cell, and the final energy/forces/
        virial as labels.
    report : MinimizeReport

    Raises
    ------
    ValueError
        For an unknown method, or ``relax_cell`` on an aperiodic configuration.
    """
    if method not in _SCIPY_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_SCIPY_METHODS)}")
    if fmax <= 0.0:
        raise ValueError(f"fmax must be > 0 eV/A, got {fmax}")
    cfg0 = configuration.stripped()
    n_atoms = cfg0.n_atoms
    if relax_cell and not np.asarray(cfg0.pbc).all():
        raise ValueError("variable-cell relaxation needs a fully periodic configuration")

    pressure = float(pressure_bar) * BAR_TO_EV_A3  # eV/A^3
    factor = float(n_atoms if cell_factor is None else cell_factor)
    if factor <= 0.0:
        raise ValueError(f"cell_factor must be > 0, got {cell_factor}")
    n_cell = (1 if hydrostatic else 6) if relax_cell else 0

    cell0 = cfg0.cell.copy()
    counter = {"n": 0, "infeasible": 0}

    def deformation(x: np.ndarray) -> np.ndarray:
        """``F = 1 + eps`` from the strain part of the variable vector."""
        if not relax_cell:
            return np.eye(3)
        v = x[3 * n_atoms:] / factor
        if hydrostatic:
            return np.eye(3) * (1.0 + v[0])
        return np.eye(3) + np.array(
            [[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]]
        )

    def unpack(x: np.ndarray) -> Configuration:
        defm = deformation(x)
        cfg = cfg0.copy()
        # Row-vector storage: r_i^T = u_i^T F^T, and the cell transforms the same
        # way, which is exactly Configuration.strained's convention.
        cfg.positions = x[: 3 * n_atoms].reshape(n_atoms, 3) @ defm.T
        cfg.cell = cell0 @ defm.T
        return cfg

    def objective(x: np.ndarray):
        cfg = unpack(x)
        try:
            result = potential.compute(cfg, forces=True, virial=relax_cell)
        except ValueError:
            # A variable-cell line search can probe a cell so small that the
            # potential's minimum-image guard fires.  That point is infeasible
            # for this code, not merely expensive, so it is reported as such and
            # the line search backs off.  Nothing is silently degraded: the
            # count is returned in the report, and if the *final* point is
            # infeasible the exception propagates from `diagnostics`.
            counter["infeasible"] += 1
            return np.inf, np.zeros(3 * n_atoms + n_cell)
        counter["n"] += 1
        volume = abs(float(np.linalg.det(cfg.cell)))
        energy = result.energy + pressure * volume
        defm = deformation(x)
        grad = np.empty(3 * n_atoms + n_cell)
        # dE/du = (dE/dr) F  in row-vector form, with dE/dr = -forces.
        grad[: 3 * n_atoms] = (-result.forces @ defm).ravel()
        if relax_cell:
            # dU/dF = -W F^-T (the virial is -dU/d eps at eps = 0, and an
            # incremental strain composes as F -> (1 + d eps) F), and
            # dV/dF = V F^-T, so the enthalpy gradient is (P V I - W) F^-T.
            finv_t = np.linalg.inv(defm).T
            a = (pressure * volume * np.eye(3) - result.virial) @ finv_t
            if hydrostatic:
                grad[3 * n_atoms] = float(np.trace(a)) / factor
            else:
                grad[3 * n_atoms:] = (
                    np.array(
                        [
                            a[0, 0],
                            a[1, 1],
                            a[2, 2],
                            a[1, 2] + a[2, 1],
                            a[0, 2] + a[2, 0],
                            a[0, 1] + a[1, 0],
                        ]
                    )
                    / factor
                )
        return energy, grad

    def diagnostics(x: np.ndarray):
        cfg = unpack(x)
        result = potential.compute(cfg, forces=True, virial=True)
        counter["n"] += 1
        volume = abs(float(np.linalg.det(cfg.cell)))
        residual = result.virial - pressure * volume * np.eye(3)
        cell_res = float(np.max(np.abs(residual))) / factor if relax_cell else 0.0
        stress_res = float(np.max(np.abs(residual))) / volume if relax_cell else 0.0
        return cfg, result, _max_force(result.forces), cell_res, stress_res, volume

    x = np.concatenate([cfg0.positions.ravel(), np.zeros(n_cell)])
    _, result0, fmax0, cell_res0, _, _ = diagnostics(x)
    energy0 = result0.energy

    # L-BFGS-B's `gtol` is on the max |gradient component|; a per-atom force norm
    # is at most sqrt(3) times that, so asking for fmax/sqrt(3) guarantees the
    # criterion we actually report.
    gtol = fmax / np.sqrt(3.0)
    scipy_method = _SCIPY_METHODS[method]
    used, message = 0, "not started"
    for _ in range(max(1, int(max_restarts))):
        budget = int(max_steps) - used
        if budget <= 0:
            message = f"iteration budget of {max_steps} exhausted"
            break
        options: dict[str, Any] = {"maxiter": budget, "gtol": gtol}
        if scipy_method == "L-BFGS-B":
            # ftol ~ 0: stopping on a relative energy change is exactly the
            # premature termination this restart loop exists to undo.
            options.update(ftol=1e-18, maxfun=20 * budget + 100)
        res = _scipy_minimize(objective, x, jac=True, method=scipy_method, options=options)
        x = np.asarray(res.x, dtype=np.float64)
        used += max(int(getattr(res, "nit", 0)), 1)
        message = str(getattr(res, "message", ""))
        _, _, current_fmax, current_cell_res, _, _ = diagnostics(x)
        if current_fmax <= fmax and current_cell_res <= fmax:
            break

    cfg, result, fmax_final, cell_res, stress_res, volume = diagnostics(x)
    converged = bool(fmax_final <= fmax and cell_res <= fmax)
    relaxed = cfg.with_labels(result)
    relaxed.info = dict(configuration.info)
    relaxed.info["relaxed"] = True

    report = MinimizeReport(
        converged=converged,
        fmax=fmax_final,
        initial_fmax=fmax0,
        cell_residual=cell_res,
        stress_residual=stress_res,
        energy=float(result.energy),
        initial_energy=float(energy0),
        pressure_bar=float(np.trace(result.virial) / (3.0 * volume) * EV_A3_TO_BAR)
        if result.virial is not None
        else float("nan"),
        target_pressure_bar=float(pressure_bar),
        volume=volume,
        n_iterations=used,
        n_evaluations=counter["n"],
        method=scipy_method,
        message=message,
    )
    return relaxed, report


# ==========================================================================
# convenience wrapper
# ==========================================================================


def equilibrate(
    configuration: Configuration,
    potential: Potential,
    *,
    temperature: float,
    dt: float,
    n_equilibrate: int,
    n_production: int,
    seed: int | np.random.Generator,
    thermostat: str = "langevin",
    production: str = "nve",
    friction: float = 5.0,
    tau: float | None = None,
    frame_stride: int = 1,
    log_stride: int = 1,
    callbacks: Sequence[Any] = (),
    progress: bool = True,
    store_velocities: bool = True,
    max_wall_seconds: float | None = None,
    neighbor_skin: float | None = None,
    fixed_com: bool = True,
    progress_stream=None,
) -> Trajectory:
    """Thermalise under a thermostat, then record a production segment.

    The usual recipe, in one call: draw Maxwell-Boltzmann velocities, run an NVT
    segment long enough to forget them, then hand the *same state* to the
    production integrator and record from there.  Only the production trajectory
    is returned, so an equilibration transient cannot leak into an average by
    accident.

    Whether production should be NVE or NVT is a real choice and not a default
    worth hiding.  NVE preserves the true dynamics, which is what transport
    coefficients require; NVT fixes the temperature but the thermostat's own
    timescale contaminates the velocity autocorrelation.  Static observables in
    this project come from HMC rather than from either (see docs/design.md
    5.3a), so ``production="nve"`` is the default here.

    Parameters
    ----------
    configuration : Configuration
        Starting geometry.
    potential : Potential
        Energy model.
    temperature : float
        Target temperature in K.
    dt : float
        Timestep in **ps** (1 fs is ``0.001``).
    n_equilibrate, n_production : int
        Steps in the discarded and recorded segments.
    seed : int or numpy.random.Generator
        Required: this function always draws velocities.
    thermostat : {"langevin", "nhc"}
        Which NVT integrator does the equilibration.
    production : {"nve", "nvt"}
        ``"nvt"`` continues with the same thermostat.
    friction : float
        Langevin friction in 1/ps.
    tau : float, optional
        Nose-Hoover time constant in ps; defaults to ``100 * dt``.
    frame_stride, log_stride, callbacks, progress, store_velocities,
    max_wall_seconds, neighbor_skin, fixed_com, progress_stream
        Passed through to :func:`run_md` for the production segment.

    Returns
    -------
    Trajectory
        The production segment only, with ``times`` starting at 0 ps.
        ``info["n_equilibrate"]`` records the discarded segment length.

    Raises
    ------
    ValueError
        For an unknown thermostat or production ensemble.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature}")
    tau = 100.0 * float(dt) if tau is None else float(tau)

    if thermostat == "langevin":
        nvt: Integrator = Langevin(dt, temperature, friction)
    elif thermostat == "nhc":
        nvt = NoseHooverChain(dt, temperature, tau)
    else:
        raise ValueError(f"unknown thermostat {thermostat!r}; expected 'langevin' or 'nhc'")

    if production == "nve":
        prod: Integrator = VelocityVerlet(dt)
    elif production == "nvt":
        prod = nvt
    else:
        raise ValueError(f"unknown production ensemble {production!r}; expected 'nve' or 'nvt'")

    warmup = run_md(
        configuration,
        potential,
        nvt,
        n_steps=0,
        thermalize_steps=int(n_equilibrate),
        seed=seed,
        temperature=temperature,
        fixed_com=fixed_com,
        progress=progress,
        progress_stream=progress_stream,
        store_velocities=False,
        neighbor_skin=neighbor_skin,
        max_wall_seconds=max_wall_seconds,
    )
    state = warmup.info["final_state"]

    trajectory = run_md(
        configuration,
        potential,
        prod,
        n_steps=int(n_production),
        frame_stride=frame_stride,
        log_stride=log_stride,
        callbacks=callbacks,
        progress=progress,
        progress_stream=progress_stream,
        store_velocities=store_velocities,
        neighbor_skin=neighbor_skin,
        max_wall_seconds=(
            None
            if max_wall_seconds is None
            else max(0.0, float(max_wall_seconds) - warmup.info["wall_seconds"])
        ),
        state=state,
    )
    trajectory.info.update(
        n_equilibrate=int(n_equilibrate),
        equilibration_integrator=nvt.name,
        equilibration_temperature=float(temperature),
        production_ensemble=production,
        equilibration_wall_seconds=warmup.info["wall_seconds"],
        wall_time_exceeded=bool(
            warmup.info["wall_time_exceeded"] or trajectory.info["wall_time_exceeded"]
        ),
    )
    return trajectory
