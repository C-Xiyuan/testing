"""Deterministic random-number plumbing.

Every stochastic entry point in :mod:`atomlab` takes an explicit ``seed`` or
``rng``; this module is what turns that promise into something usable.

Three separate problems are solved here, and they are genuinely different:

1. *Getting a generator.*  :func:`make_rng` accepts whatever a caller happens
   to have -- an ``int``, ``None``, a :class:`numpy.random.SeedSequence`, or an
   already-constructed :class:`numpy.random.Generator` -- and always returns a
   ``Generator``.  Passing a ``Generator`` through unchanged is deliberate: it
   lets a function draw from its caller's stream rather than silently
   restarting a private one, which is the usual way "deterministic" code ends
   up correlated across calls.

2. *Splitting a stream for parallel work.*  :func:`spawn_rngs` uses
   ``SeedSequence.spawn``.  Naive alternatives (``seed + worker_index``, or
   drawing sub-seeds with ``randint``) give streams whose independence is not
   guaranteed; ``SeedSequence`` runs the child entropy through a hash-based
   mixing function precisely so that the children are statistically
   independent, and it is reproducible regardless of *how many* workers
   actually run or in what order they finish.

3. *Containing global state.*  The libraries we depend on keep hidden global
   RNGs (``random``, NumPy's legacy ``mtrand`` singleton, torch).  A helper
   that seeds torch for its own reproducibility would otherwise reach out and
   perturb an enclosing experiment.  :class:`SeedScope` saves and restores
   those globals so that cannot happen.
"""

from __future__ import annotations

import contextlib
import random as _py_random
from types import TracebackType
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "SeedLike",
    "make_rng",
    "spawn_rngs",
    "seed_everything",
    "SeedScope",
    "capture_random_state",
    "restore_random_state",
]

#: Anything :func:`make_rng` knows how to turn into a ``Generator``.
SeedLike = "int | np.random.SeedSequence | np.random.Generator | np.random.BitGenerator | None"


def _torch_module():
    """Return the imported ``torch`` module, or ``None`` if unavailable.

    torch is an optional-at-runtime dependency for the pure-NumPy parts of the
    package, and importing it costs a second or so, so the import is deferred
    to the point of use rather than done at module import.
    """
    try:  # pragma: no cover - trivial
        import torch  # noqa: PLC0415
    except Exception:  # pragma: no cover - torch missing or broken install
        return None
    return torch


def make_rng(seed: Any = None) -> np.random.Generator:
    """Coerce ``seed`` into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed:
        One of

        * ``None`` -- a fresh generator seeded from OS entropy.  Use only where
          irreproducibility is genuinely acceptable.
        * ``int`` -- a reproducible generator (PCG64).
        * :class:`numpy.random.SeedSequence` -- used directly, which is how
          spawned child streams are turned into generators.
        * :class:`numpy.random.Generator` -- returned *unchanged*, so the
          callee shares the caller's stream.
        * :class:`numpy.random.BitGenerator` -- wrapped in a ``Generator``.

    Returns
    -------
    numpy.random.Generator

    Examples
    --------
    >>> int(make_rng(0).integers(0, 100))
    85
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.BitGenerator):
        return np.random.Generator(seed)
    if isinstance(seed, np.random.SeedSequence):
        return np.random.Generator(np.random.PCG64(seed))
    if seed is None or isinstance(seed, (int, np.integer, Sequence)):
        return np.random.default_rng(seed)
    raise TypeError(f"cannot make a Generator from {type(seed).__name__}")


def seed_sequence(seed: Any = None) -> np.random.SeedSequence:
    """Return a :class:`numpy.random.SeedSequence` for ``seed``.

    Accepts the same inputs as :func:`make_rng` except a bare ``Generator``,
    whose internal seed sequence is not part of the public API and therefore
    cannot be split reproducibly.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if seed is None or isinstance(seed, (int, np.integer, Sequence)):
        return np.random.SeedSequence(seed)
    raise TypeError(f"cannot make a SeedSequence from {type(seed).__name__}")


def spawn_rngs(seed: Any, n: int) -> list[np.random.Generator]:
    """Create ``n`` independent, reproducible generators from one ``seed``.

    Parameters
    ----------
    seed:
        Root seed (see :func:`seed_sequence`).
    n:
        Number of child streams, one per parallel worker / replica / MD chain.

    Returns
    -------
    list of numpy.random.Generator
        Length ``n``.  Child ``k`` depends only on ``(seed, k)``, never on how
        many children were requested, so a run with 4 workers and a run with 8
        workers share the first four streams.  This matters when re-running an
        experiment on a machine with a different core count.

    Notes
    -----
    ``SeedSequence.spawn`` is used rather than arithmetic on the seed.  Nearby
    integer seeds are *not* guaranteed to produce uncorrelated PCG64 streams;
    the spawn key is mixed by the same hashing step that protects the root
    entropy, which is guaranteed.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    root = seed_sequence(seed)
    # spawn() advances the parent's n_children_spawned counter, so spawn from a
    # copy: repeated calls with the same root seed must be identical.
    fresh = np.random.SeedSequence(
        entropy=root.entropy, spawn_key=root.spawn_key, pool_size=root.pool_size
    )
    return [np.random.Generator(np.random.PCG64(child)) for child in fresh.spawn(n)]


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed every global RNG we can reach, and report what was done.

    This exists for the top-level entry points of experiment scripts, where a
    third-party library may reach for a global RNG we do not control.  Library
    code inside :mod:`atomlab` must **not** call it -- it takes an explicit
    ``seed``/``rng`` argument instead.

    Parameters
    ----------
    seed:
        Root seed.  Must be a non-negative integer.
    deterministic_torch:
        If True, also request deterministic cuDNN/cuBLAS kernel selection from
        torch.  Off by default because it can make training several times
        slower and, on some builds, raises for ops with no deterministic
        implementation.

    Returns
    -------
    dict
        Provenance record with keys ``"seed"``, ``"python_random"``,
        ``"numpy_legacy"``, ``"torch"``, ``"torch_deterministic"``.  Values are
        the per-library seeds actually applied, or ``None`` where the library
        was unavailable.  Write this straight into a ``manifest.json``.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    record: dict[str, Any] = {"seed": seed}

    # Derive an independent sub-seed per library rather than handing the same
    # integer to all of them: identical seeds in different PRNGs are harmless
    # in principle but make cross-library coincidences hard to rule out when
    # debugging a suspicious result.
    sub = seed_sequence(seed).generate_state(4, dtype=np.uint32)

    _py_random.seed(int(sub[0]))
    record["python_random"] = int(sub[0])

    np.random.seed(int(sub[1]) % (2**32))
    record["numpy_legacy"] = int(sub[1]) % (2**32)

    torch = _torch_module()
    if torch is None:
        record["torch"] = None
        record["torch_deterministic"] = None
    else:
        torch_seed = int(sub[2])
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(torch_seed)
        record["torch"] = torch_seed
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):  # pragma: no cover
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        record["torch_deterministic"] = bool(deterministic_torch)

    return record


def capture_random_state(*, include_torch: bool = True) -> dict[str, Any]:
    """Snapshot the global RNG states of python / numpy-legacy / torch.

    Returns
    -------
    dict
        Opaque state blob accepted by :func:`restore_random_state`.  Keys with
        value ``None`` mean that library was not present.
    """
    state: dict[str, Any] = {
        "python_random": _py_random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch": None,
    }
    if include_torch:
        torch = _torch_module()
        if torch is not None:
            # Clone: get_rng_state returns a tensor that torch may later reuse.
            state["torch"] = torch.get_rng_state().clone()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_random_state`."""
    if state.get("python_random") is not None:
        _py_random.setstate(state["python_random"])
    if state.get("numpy_legacy") is not None:
        np.random.set_state(state["numpy_legacy"])
    if state.get("torch") is not None:
        torch = _torch_module()
        if torch is not None:
            torch.set_rng_state(state["torch"])


class SeedScope:
    """Context manager that isolates global RNG state.

    On entry the global states of ``random``, NumPy's legacy singleton and
    torch are saved (and optionally reseeded); on exit they are restored
    exactly, *including* if the body raised.

    The motivating failure is subtle and worth spelling out.  A model's
    ``fit()`` may call ``torch.manual_seed`` so that its own initialisation is
    reproducible.  If that happens in the middle of an experiment that is also
    drawing from torch's global RNG (dropout, data shuffling), then the outer
    experiment's stream silently depends on how many models were fitted, and
    re-running with one extra model changes results everywhere.  Wrapping the
    inner call in a ``SeedScope`` makes the perturbation impossible.

    Parameters
    ----------
    seed:
        If given, :func:`seed_everything` is applied on entry.  If ``None`` the
        state is merely saved and restored, not reseeded.
    include_torch:
        Set False to skip torch entirely (avoids importing it).

    Attributes
    ----------
    rng:
        A :class:`numpy.random.Generator` derived from ``seed``, available
        inside the block as ``with SeedScope(0) as scope: scope.rng``.  It is
        ``None`` when ``seed is None``.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> before = np.random.rand()
    >>> np.random.seed(1234)
    >>> with SeedScope(99):
    ...     _ = np.random.rand()
    >>> np.random.rand() == before
    True
    """

    def __init__(self, seed: int | None = None, *, include_torch: bool = True):
        self.seed = seed
        self.include_torch = bool(include_torch)
        self.rng: np.random.Generator | None = None
        self.record: dict[str, Any] | None = None
        self._saved: dict[str, Any] | None = None

    def __enter__(self) -> "SeedScope":
        self._saved = capture_random_state(include_torch=self.include_torch)
        if self.seed is not None:
            self.record = seed_everything(self.seed)
            self.rng = make_rng(self.seed)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        assert self._saved is not None
        restore_random_state(self._saved)
        self._saved = None
        return False  # never swallow exceptions


@contextlib.contextmanager
def seeded(seed: int | None = None, *, include_torch: bool = True) -> Iterator[np.random.Generator]:
    """Function form of :class:`SeedScope` yielding the derived generator.

    Examples
    --------
    >>> with seeded(0) as rng:
    ...     x = rng.normal(size=3)
    """
    with SeedScope(seed, include_torch=include_torch) as scope:
        yield scope.rng if scope.rng is not None else make_rng(None)
