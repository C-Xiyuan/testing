"""Wall-clock instrumentation.

MD runs and model fits in this project are minutes-to-hours long, and the
experiment manifests are expected to record where that time went.  The registry
here is deliberately a plain accumulator keyed by name -- it answers "how much
total wall time did neighbour-list construction cost across the whole run", not
"what is the flame graph", which is a profiler's job.

Wall clock (``time.perf_counter``) is used rather than CPU time on purpose: the
numbers are meant to predict how long a re-run takes, and the numba/BLAS parts
are multi-threaded, so CPU time would over-report them by up to a factor of the
core count.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Callable, Iterator, TypeVar

__all__ = [
    "Timer",
    "timed",
    "format_duration",
    "TimerRegistry",
    "REGISTRY",
    "get_timings",
    "reset_timings",
    "timing_report",
]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingRecord:
    """Accumulated statistics for one timer name.

    Attributes
    ----------
    name:
        Timer label.
    count:
        Number of completed intervals.
    total:
        Total wall-clock seconds.
    best, worst:
        Fastest / slowest single interval, in seconds.
    """

    name: str
    count: int = 0
    total: float = 0.0
    best: float = float("inf")
    worst: float = 0.0

    @property
    def mean(self) -> float:
        """Mean seconds per interval (``0.0`` if never run)."""
        return self.total / self.count if self.count else 0.0

    def add(self, dt: float) -> None:
        self.count += 1
        self.total += dt
        self.best = min(self.best, dt)
        self.worst = max(self.worst, dt)


class TimerRegistry:
    """Thread-safe accumulator mapping timer name -> :class:`TimingRecord`."""

    def __init__(self) -> None:
        self._records: dict[str, TimingRecord] = {}
        self._lock = threading.Lock()

    def add(self, name: str, dt: float) -> None:
        """Record one interval of ``dt`` seconds under ``name``."""
        with self._lock:
            rec = self._records.get(name)
            if rec is None:
                rec = self._records[name] = TimingRecord(name)
            rec.add(dt)

    def get(self, name: str) -> TimingRecord | None:
        return self._records.get(name)

    def totals(self) -> dict[str, float]:
        """Return ``{name: total_seconds}``, sorted by descending total."""
        with self._lock:
            items = sorted(self._records.items(), key=lambda kv: -kv[1].total)
        return {k: v.total for k, v in items}

    def records(self) -> dict[str, TimingRecord]:
        with self._lock:
            return dict(self._records)

    def reset(self, name: str | None = None) -> None:
        """Clear one timer, or all of them when ``name`` is ``None``."""
        with self._lock:
            if name is None:
                self._records.clear()
            else:
                self._records.pop(name, None)

    def report(self, *, top: int | None = None) -> str:
        """Human-readable table, sorted by total time descending."""
        recs = sorted(self.records().values(), key=lambda r: -r.total)
        if top is not None:
            recs = recs[:top]
        if not recs:
            return "(no timings recorded)"
        width = max(len(r.name) for r in recs)
        head = f"{'timer'.ljust(width)}  {'calls':>7}  {'total':>10}  {'mean':>10}"
        lines = [head, "-" * len(head)]
        for r in recs:
            lines.append(
                f"{r.name.ljust(width)}  {r.count:>7d}  "
                f"{format_duration(r.total):>10}  {format_duration(r.mean):>10}"
            )
        return "\n".join(lines)

    def __iter__(self) -> Iterator[TimingRecord]:
        return iter(self.records().values())

    def __len__(self) -> int:
        return len(self._records)


#: Process-wide default registry used by :class:`Timer` and :func:`timed`.
REGISTRY = TimerRegistry()


def get_timings(registry: TimerRegistry | None = None) -> dict[str, float]:
    """Return ``{name: total_seconds}`` from ``registry`` (default: global)."""
    return (registry or REGISTRY).totals()


def reset_timings(name: str | None = None, registry: TimerRegistry | None = None) -> None:
    """Clear accumulated timings."""
    (registry or REGISTRY).reset(name)


def timing_report(*, top: int | None = None, registry: TimerRegistry | None = None) -> str:
    """Formatted timing table from ``registry`` (default: global)."""
    return (registry or REGISTRY).report(top=top)


class Timer:
    """Context manager measuring wall-clock time for a named block.

    Parameters
    ----------
    name:
        Label under which the interval is accumulated.  ``None`` measures the
        block without touching any registry.
    registry:
        Where to accumulate; defaults to the module-global :data:`REGISTRY`.
    logger:
        If given, ``logger.debug`` is called with the formatted duration on
        exit.  Kept at debug level so that instrumenting an inner loop does not
        drown a production log.
    accumulate:
        Set False to time a block without adding to the registry.

    Attributes
    ----------
    elapsed:
        Seconds spent in the block.  Readable *during* the block too, in which
        case it reports the time so far.

    Examples
    --------
    >>> with Timer("demo") as t:
    ...     pass
    >>> t.elapsed >= 0.0
    True
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        registry: TimerRegistry | None = None,
        logger: Any = None,
        accumulate: bool = True,
    ):
        self.name = name
        self.registry = registry if registry is not None else REGISTRY
        self.logger = logger
        self.accumulate = bool(accumulate) and name is not None
        self._t0: float | None = None
        self._elapsed: float = 0.0

    @property
    def elapsed(self) -> float:
        """Seconds elapsed (running total if the block is still open)."""
        if self._t0 is not None:
            return time.perf_counter() - self._t0
        return self._elapsed

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        assert self._t0 is not None
        self._elapsed = time.perf_counter() - self._t0
        self._t0 = None
        # Accumulate even when the body raised: a crash after 40 minutes is
        # exactly the timing you want in the log.
        if self.accumulate and self.name is not None:
            self.registry.add(self.name, self._elapsed)
        if self.logger is not None:
            self.logger.debug("%s took %s", self.name or "block", format_duration(self._elapsed))
        return False


def timed(
    func: F | None = None,
    *,
    name: str | None = None,
    registry: TimerRegistry | None = None,
    logger: Any = None,
) -> Any:
    """Decorator accumulating a function's wall-clock time into a registry.

    Usable bare (``@timed``) or parameterised (``@timed(name="rdf")``).

    Parameters
    ----------
    name:
        Registry key.  Defaults to ``f"{module}.{qualname}"``, which keeps two
        same-named methods on different classes from colliding.
    registry:
        Target registry; defaults to the global :data:`REGISTRY`.
    logger:
        Optional logger for a per-call debug line.

    Returns
    -------
    callable
        The wrapped function.  ``wrapper.timer_name`` exposes the key used, so
        callers can look the record up without re-deriving the name.
    """

    def decorate(f: F) -> F:
        key = name or f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', f.__name__)}"

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(key, registry=registry, logger=logger):
                return f(*args, **kwargs)

        wrapper.timer_name = key  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorate(func)
    return decorate


def format_duration(seconds: float) -> str:
    """Format a duration for logs and manifests.

    Chooses the unit by magnitude so that a table of timings stays readable:
    sub-millisecond values print in us, then ms, then ``s``, then ``m s``, then
    ``h m s``.  Always three significant-ish digits, never scientific notation.

    Parameters
    ----------
    seconds:
        Duration in seconds.  Negative values are formatted with a leading
        ``-`` rather than rejected, since they only arise from clock weirdness
        and a hard failure in a logging path would be worse.

    Returns
    -------
    str

    Examples
    --------
    >>> format_duration(0.0000123)
    '12.3 us'
    >>> format_duration(0.25)
    '250 ms'
    >>> format_duration(3.5)
    '3.500 s'
    >>> format_duration(3725.0)
    '1 h 02 m 05 s'
    """
    if seconds != seconds:  # NaN
        return "nan"
    sign = "-" if seconds < 0 else ""
    s = abs(float(seconds))
    if s < 1e-3:
        return f"{sign}{s * 1e6:.3g} us"
    if s < 1.0:
        return f"{sign}{s * 1e3:.3g} ms"
    if s < 60.0:
        return f"{sign}{s:.3f} s"
    if s < 3600.0:
        m, rem = divmod(s, 60.0)
        return f"{sign}{int(m)} m {rem:04.1f} s"
    h, rem = divmod(s, 3600.0)
    m, sec = divmod(rem, 60.0)
    return f"{sign}{int(h)} h {int(m):02d} m {int(sec):02d} s"
