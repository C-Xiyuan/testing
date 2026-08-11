"""Logging and progress reporting.

Everything here writes to **stderr**, never stdout.  Experiment scripts emit
machine-readable results on stdout (or straight to ``results/``), and mixing
human commentary into that stream is how a results file ends up with a progress
bar in the middle of it.

The other deliberate choice is that :class:`ProgressReporter` throttles by
*time*, not by iteration count.  A stride of "every 100 steps" produces one line
per second in a cheap system and one line per hour in an expensive one; a stride
of "every 30 seconds" produces a log of predictable length regardless of the
workload, which is what you want for a multi-hour MD run whose log a human will
skim afterwards.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Iterable, Iterator, TypeVar

from .timing import format_duration

__all__ = [
    "get_logger",
    "set_level",
    "add_file_handler",
    "ProgressReporter",
    "progress",
    "DEFAULT_FORMAT",
]

T = TypeVar("T")

#: Timestamp, level, logger name, message.  Timestamps are load-bearing: they
#: are how a slow run is diagnosed after the fact from a captured log.
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

#: Root logger name.  All package loggers hang off this so that a single call
#: to :func:`set_level` controls the whole package without touching the global
#: root logger (which would also reconfigure the host application's logging).
ROOT_NAME = "atomlab"

_ENV_LEVEL = "ATOMLAB_LOG_LEVEL"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(ROOT_NAME)
    if not getattr(root, "_atomlab_configured", False):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
        root.addHandler(handler)
        level = os.environ.get(_ENV_LEVEL, "INFO").upper()
        root.setLevel(getattr(logging, level, logging.INFO))
        # Do not also hand records to the global root logger: that would
        # duplicate every line whenever the embedding application has its own
        # handler installed.
        root.propagate = False
        root._atomlab_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger writing to stderr.

    Parameters
    ----------
    name:
        Logger name.  Names not already under ``"atomlab"`` are placed there,
        so ``get_logger(__name__)`` from inside the package and
        ``get_logger("exp04")`` from a script both end up on the same handler.
        ``None`` returns the package root logger.

    Returns
    -------
    logging.Logger

    Notes
    -----
    The default level is ``INFO``, overridable process-wide with the
    ``ATOMLAB_LOG_LEVEL`` environment variable so that a batch job can be made
    verbose without editing code.
    """
    root = _configure_root()
    if name is None or name == ROOT_NAME:
        return root
    if name.startswith(ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def set_level(level: int | str) -> None:
    """Set the level of the whole ``atomlab`` logger tree."""
    root = _configure_root()
    root.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else int(level))


def add_file_handler(path: Any, *, level: int | str = "DEBUG", mode: str = "w") -> logging.Handler:
    """Also write the package log to ``path``, and return the handler.

    Used by experiment drivers so that each run leaves a complete log next to
    its results, independent of how the job's stderr was captured.
    """
    root = _configure_root()
    handler = logging.FileHandler(str(path), mode=mode, encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
    handler.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else int(level))
    root.addHandler(handler)
    return handler


class ProgressReporter:
    """Time-throttled progress logging for long loops.

    Emits at most one line per ``interval`` seconds (plus a first and a final
    line), which makes it safe to call from the innermost MD step loop: the
    common path is one ``time.perf_counter()`` call and a float comparison.

    Parameters
    ----------
    total:
        Expected number of items, if known.  Enables percentage and ETA.
        ``None`` reports rate and count only.
    label:
        Prefix identifying what is being counted, e.g. ``"md steps"``.
    interval:
        Minimum seconds between lines.  30 s is a reasonable default for runs
        measured in hours; drop it for interactive work.
    logger:
        Destination logger; a child of the package logger by default.
    level:
        Level at which progress lines are emitted (default ``INFO``).
    unit:
        Noun used in the rate, e.g. ``"steps"`` -> ``"12.3 steps/s"``.

    Examples
    --------
    >>> rep = ProgressReporter(total=3, label="demo", interval=1e9)
    >>> [rep.update() for _ in range(3)]   # throttled: nothing emitted yet
    [False, False, False]
    >>> rep.count
    3
    """

    def __init__(
        self,
        total: int | None = None,
        *,
        label: str = "progress",
        interval: float = 30.0,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        unit: str = "it",
    ):
        if interval <= 0:
            raise ValueError(f"interval must be > 0 s, got {interval}")
        self.total = None if total is None else int(total)
        self.label = label
        self.interval = float(interval)
        self.logger = logger if logger is not None else get_logger("progress")
        self.level = level
        self.unit = unit

        self.count = 0
        self.n_reports = 0
        self._t_start = time.perf_counter()
        self._t_last = self._t_start
        self._count_last = 0

    # -- core ------------------------------------------------------------

    def update(self, n: int = 1, *, extra: str = "", force: bool = False) -> bool:
        """Advance the counter by ``n`` and log if the interval has elapsed.

        Parameters
        ----------
        n:
            Increment.
        extra:
            Extra text appended to the line, e.g. ``"T=300.1 K"``.  Only
            evaluated by the caller when it decides to pass it, so build it
            cheaply or guard with :meth:`due`.
        force:
            Emit regardless of the throttle.

        Returns
        -------
        bool
            True if a line was emitted, so a caller can piggyback expensive
            diagnostics onto the same cadence.
        """
        self.count += int(n)
        now = time.perf_counter()
        if not force and (now - self._t_last) < self.interval:
            return False
        self._emit(now, extra=extra)
        return True

    def due(self) -> bool:
        """True if the throttle interval has elapsed (no counter change)."""
        return (time.perf_counter() - self._t_last) >= self.interval

    def finish(self, *, extra: str = "") -> None:
        """Emit a final summary line with the overall rate."""
        elapsed = time.perf_counter() - self._t_start
        rate = self.count / elapsed if elapsed > 0 else float("inf")
        msg = (
            f"{self.label}: done {self.count}"
            + (f"/{self.total}" if self.total is not None else "")
            + f" in {format_duration(elapsed)} ({rate:.3g} {self.unit}/s)"
        )
        if extra:
            msg += f" | {extra}"
        self.logger.log(self.level, msg)
        self.n_reports += 1

    # -- internals -------------------------------------------------------

    def _emit(self, now: float, *, extra: str = "") -> None:
        window = now - self._t_last
        # Instantaneous rate over the last window, not the running average:
        # for MD it is the current rate that tells you whether the run has
        # started thrashing (neighbour rebuilds, swap, a thermostat blowup).
        rate = (self.count - self._count_last) / window if window > 0 else float("inf")

        parts = [f"{self.label}: {self.count}"]
        if self.total:
            pct = 100.0 * self.count / self.total
            parts[0] += f"/{self.total} ({pct:5.1f}%)"
            if rate > 0 and self.count <= self.total:
                eta = (self.total - self.count) / rate
                parts.append(f"eta {format_duration(eta)}")
        parts.append(f"{rate:.3g} {self.unit}/s")
        parts.append(f"elapsed {format_duration(now - self._t_start)}")
        if extra:
            parts.append(extra)
        self.logger.log(self.level, " | ".join(parts))

        self._t_last = now
        self._count_last = self.count
        self.n_reports += 1

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish()
        return False


def progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    label: str = "progress",
    interval: float = 30.0,
    logger: logging.Logger | None = None,
    unit: str = "it",
) -> Iterator[T]:
    """Wrap ``iterable`` in a :class:`ProgressReporter`.

    ``total`` is taken from ``len(iterable)`` when available.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    rep = ProgressReporter(total, label=label, interval=interval, logger=logger, unit=unit)
    for item in iterable:
        yield item
        rep.update()
    rep.finish()
