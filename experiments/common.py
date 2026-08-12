"""Shared harness for the experiment programme.

Every experiment in this directory is a claim about the world, so every
experiment has to be re-runnable by someone who does not trust the person who
ran it first.  That means three things, all of which this module supplies so
that no individual experiment has to remember them:

1. **A manifest.**  Git commit, dirty-tree flag, full configuration, wall time,
   host and library versions, written next to the results.  A results file
   without a manifest cannot be checked and should not be believed.
2. **Explicit seeds.**  Derived by ``SeedSequence`` from a single experiment
   seed, so sub-components get independent streams that are nonetheless a pure
   function of that one number.
3. **Atomic, resumable output.**  Long MD campaigns get interrupted.  Stages
   write to their own files and are skipped on re-run unless ``--force``, so a
   crashed campaign resumes instead of restarting.

Usage::

    from experiments.common import ExperimentContext, main

    def run(ctx: ExperimentContext) -> dict:
        cfg = ctx.config
        ...
        return {"headline_number": 42.0}

    if __name__ == "__main__":
        main(run, default_config=DEFAULTS, description="...")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _git(*args: str, strip: bool = True) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() if strip else out
    except Exception:
        return "unknown"


def _code_digest() -> dict:
    """Hash the source that actually produced the result.

    `git_dirty` alone is a bare boolean: it says the tree differed from the
    commit but not how, so a reader cannot tell an edited sampler from an edited
    README. These two fields make the provenance checkable. `dirty_paths` is the
    porcelain listing, and `sha256` is over the content of every tracked Python
    file under the library and experiment trees, sorted by path -- the digest
    changes if and only if code that can affect a number changed.
    """
    # Not _git(...) with the default strip: it would eat the leading status
    # column of the first line and truncate that path by one character.
    status = _git("status", "--porcelain", strip=False)
    dirty = [] if status.strip() in ("", "unknown") else sorted(
        line[3:] for line in status.splitlines() if line[3:])
    digest = hashlib.sha256()
    sources = sorted(
        p for d in ("atomlab", "experiments", "scripts")
        for p in (REPO_ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts)
    for path in sources:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "dirty_paths": dirty,
        "n_source_files": len(sources),
        "sha256": digest.hexdigest(),
    }


def _library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for module in ("numpy", "scipy", "torch", "numba", "matplotlib"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "absent"
    return versions


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles the numpy scalars results are full of."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


@dataclass
class ExperimentContext:
    """Everything an experiment needs in order to be reproducible.

    Attributes
    ----------
    name:
        Experiment identifier, e.g. ``"exp01_reference_physics"``.  Determines
        the output directory.
    config:
        The merged configuration (defaults, overridden by a config file,
        overridden by ``--set key=value`` command-line overrides).
    seed:
        Master seed.  Use :meth:`rng` rather than this directly.
    output_dir, figure_dir:
        Where to write.  Created on construction.
    force:
        If True, cached stage outputs are recomputed rather than reused.
    """

    name: str
    config: dict
    seed: int = 0
    output_dir: Path = field(default=None)  # type: ignore[assignment]
    figure_dir: Path = field(default=None)  # type: ignore[assignment]
    force: bool = False
    quick: bool = False
    _timings: dict = field(default_factory=dict)
    _start: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = RESULTS_ROOT / self.name
        if self.figure_dir is None:
            self.figure_dir = FIGURES_ROOT
        self.output_dir = Path(self.output_dir)
        self.figure_dir = Path(self.figure_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    # -- randomness --------------------------------------------------------

    def rng(self, *stream: Any) -> np.random.Generator:
        """A reproducible generator for a named sub-stream.

        ``ctx.rng("dataset", 3)`` always returns the same generator for a given
        master seed, and a different one from ``ctx.rng("dataset", 4)``.  Naming
        streams rather than incrementing a counter means adding a new stage in
        the middle of an experiment does not silently change every result after
        it -- which is the kind of thing that makes "reproducible" experiments
        quietly not be.
        """
        key = [self.seed] + [
            int(s) if isinstance(s, (int, np.integer)) else
            int.from_bytes(str(s).encode()[:8].ljust(8, b"\0"), "little")
            for s in stream
        ]
        return np.random.default_rng(np.random.SeedSequence(key))

    # -- configuration -----------------------------------------------------

    def get(self, key: str, default=None):
        """Fetch a possibly-nested config value with ``"a.b.c"`` syntax."""
        node = self.config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def scaled(self, key: str, quick_factor: float = 0.05):
        """A config value, shrunk when running under ``--quick``.

        ``--quick`` exists so the whole programme can be smoke-tested end to end
        in minutes before committing hours to it.  Results produced under
        ``--quick`` are marked as such in the manifest so they cannot be
        mistaken for production numbers.
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"config has no key {key!r}")
        if not self.quick:
            return value
        if isinstance(value, int):
            return max(1, int(value * quick_factor))
        if isinstance(value, float):
            return value * quick_factor
        return value

    # -- output ------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        p = self.output_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def figure_path(self, stem: str, ext: str = "png") -> Path:
        return self.figure_dir / f"{self.name}_{stem}.{ext}"

    def save_json(self, name: str, payload) -> Path:
        """Write JSON atomically, so an interrupted run cannot leave a half file."""
        target = self.path(name if name.endswith(".json") else f"{name}.json")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder))
        tmp.replace(target)
        return target

    def save_npz(self, name: str, **arrays) -> Path:
        target = self.path(name if name.endswith(".npz") else f"{name}.npz")
        tmp = target.with_suffix(".npz.tmp")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(target)
        return target

    def load_json(self, name: str):
        target = self.path(name if name.endswith(".json") else f"{name}.json")
        return json.loads(target.read_text()) if target.exists() else None

    # -- stage caching -----------------------------------------------------

    def stage(self, name: str, compute: Callable[[], Any], *, kind: str = "json"):
        """Run ``compute`` unless its output already exists.

        Long campaigns are interrupted by wall-clock limits far more often than
        by bugs.  Caching per stage turns a five-hour experiment into five
        one-hour experiments that can be run whenever there is time, without any
        loss of reproducibility -- the manifest records which stages were reused.
        """
        suffix = ".json" if kind == "json" else ".npz"
        target = self.path(f"{name}{suffix}")
        if target.exists() and not self.force:
            self._timings.setdefault("reused_stages", []).append(name)
            if kind == "json":
                return json.loads(target.read_text())
            with np.load(target, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}

        with self.timed(name):
            payload = compute()
        if kind == "json":
            self.save_json(name, payload)
        else:
            self.save_npz(name, **payload)
        return payload

    @contextmanager
    def timed(self, label: str) -> Iterator[None]:
        """Record wall time for a labelled section into the manifest."""
        start = time.time()
        print(f"  [{self.name}] {label} ...", flush=True)
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._timings[label] = elapsed
            print(f"  [{self.name}] {label} done in {elapsed:.1f} s", flush=True)

    # -- provenance --------------------------------------------------------

    def write_manifest(self, extra: dict | None = None) -> Path:
        manifest = {
            "experiment": self.name,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "code": _code_digest(),
            "seed": self.seed,
            "quick_mode": self.quick,
            "config": self.config,
            "timings_seconds": self._timings,
            "total_seconds": time.time() - self._start,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "libraries": _library_versions(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if extra:
            manifest["results"] = extra
        return self.save_json("manifest", manifest)


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``key.path=value`` overrides, parsing values as JSON when possible."""
    out = json.loads(json.dumps(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be of the form key=value")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def main(
    run: Callable[[ExperimentContext], dict],
    *,
    default_config: dict,
    description: str = "",
    name: str | None = None,
) -> dict:
    """Standard entry point: parse arguments, build a context, run, write manifest."""
    script = Path(sys.argv[0]).resolve()
    experiment_name = name or script.parent.name

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=script.parent / "config.json",
                        help="JSON config file; missing file means use defaults")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="recompute cached stages")
    parser.add_argument("--quick", action="store_true",
                        help="shrink every size parameter for a fast smoke test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config entry")
    args = parser.parse_args()

    config = json.loads(json.dumps(default_config))
    if args.config and Path(args.config).exists():
        config.update(json.loads(Path(args.config).read_text()))
    config = _apply_overrides(config, args.overrides)

    ctx = ExperimentContext(
        name=experiment_name,
        config=config,
        seed=args.seed,
        output_dir=args.output,
        force=args.force,
        quick=args.quick,
    )

    banner = f"=== {experiment_name}{' (QUICK)' if args.quick else ''} ==="
    print(banner, flush=True)
    results = run(ctx)
    ctx.write_manifest(results if isinstance(results, dict) else None)
    print(f"=== {experiment_name} complete: {ctx.output_dir} ===", flush=True)
    return results
