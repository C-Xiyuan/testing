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
import os
import platform
import stat
import subprocess
import sys
import time
import traceback
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


def _git_bytes(*args: str) -> bytes | None:
    """Run Git without decoding paths or blobs; return ``None`` on failure."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None


SOURCE_DIRS = ("atomlab", "experiments", "scripts", "protocols")
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml"}


def require_fresh_production_seed(
    ctx: "ExperimentContext", *, expected: int, legacy: tuple[int, ...] = (0,)
) -> None:
    """Prevent a corrected confirmatory run from replaying legacy random streams.

    Quick smoke runs may use any seed because they cannot become evidence. A
    production rerun must use the prospectively fixed seed and must not silently
    inherit the harness default (historically zero in this repository).
    """
    if ctx.quick:
        return
    if ctx.seed in legacy:
        raise RuntimeError(
            f"production seed {ctx.seed} is a legacy seed; use the frozen v2 "
            f"master seed {expected}"
        )
    if ctx.seed != expected:
        raise RuntimeError(
            f"production master seed must equal the prospectively frozen value "
            f"{expected}, got {ctx.seed}"
        )


def _canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes used for exact protocol/config attestations."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _config_mismatches(actual: Any, expected: Any, path: str = "config") -> list[str]:
    """Describe exact, type-sensitive differences including extra/missing keys."""
    if type(actual) is not type(expected):
        return [
            f"{path}: expected {type(expected).__name__} {expected!r}, "
            f"got {type(actual).__name__} {actual!r}"
        ]
    if isinstance(expected, dict):
        mismatches = []
        actual_keys, expected_keys = set(actual), set(expected)
        mismatches.extend(
            f"{path}.{key}: unexpected key" for key in sorted(actual_keys - expected_keys)
        )
        mismatches.extend(
            f"{path}.{key}: missing key" for key in sorted(expected_keys - actual_keys)
        )
        for key in sorted(actual_keys & expected_keys):
            mismatches.extend(_config_mismatches(
                actual[key], expected[key], f"{path}.{key}"
            ))
        return mismatches
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: expected {len(expected)} items, got {len(actual)}"]
        mismatches = []
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatches.extend(_config_mismatches(
                actual_item, expected_item, f"{path}[{index}]"
            ))
        return mismatches
    return [] if actual == expected else [
        f"{path}: expected {expected!r}, got {actual!r}"
    ]


def _load_protocol_document(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    document = json.loads(
        raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number {token!r}")
        ),
    )
    if not isinstance(document, dict):
        raise ValueError("protocol root must be a JSON object")
    return document, raw


def require_frozen_protocol(
    ctx: "ExperimentContext",
    *,
    protocol_path: str | Path,
    expected_protocol: str | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Load the independent protocol artifact and attest the executable config.

    The protocol JSON, not the module-level ``DEFAULTS`` object, is authoritative.
    Production comparison is exact and recursive: extra keys, missing keys, type
    changes, list changes, and value changes all fail. Quick runs may use a
    shrunken config, but their manifest records that it did not match and remains
    ``smoke_only``.
    """
    path = Path(protocol_path).resolve()
    document, raw = _load_protocol_document(path)
    required = {"protocol", "protocol_version", "master_seed", "executable_config"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"protocol {path} is missing required keys: {missing}")
    if not isinstance(document["protocol_version"], int):
        raise ValueError("protocol_version must be an integer")
    if not isinstance(document["master_seed"], int):
        raise ValueError("master_seed must be an integer")
    if not isinstance(document["executable_config"], dict):
        raise ValueError("executable_config must be a JSON object")
    if expected_protocol is not None and document["protocol"] != expected_protocol:
        raise ValueError(
            f"expected protocol {expected_protocol!r}, got {document['protocol']!r}"
        )
    if (expected_version is not None
            and document["protocol_version"] != expected_version):
        raise ValueError(
            f"expected protocol version {expected_version}, "
            f"got {document['protocol_version']}"
        )

    config_mismatches = _config_mismatches(
        ctx.config, document["executable_config"]
    )
    seed_matches = int(ctx.seed) == document["master_seed"]
    mismatches = list(config_mismatches)
    if not seed_matches:
        mismatches.append(
            f"master_seed: expected {document['master_seed']!r}, got {ctx.seed!r}"
        )
    try:
        recorded_path = path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        recorded_path = str(path)
    attestation = {
        "path": recorded_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": document["protocol"],
        "protocol_version": document["protocol_version"],
        "master_seed": document["master_seed"],
        "seed_matches": seed_matches,
        "executable_config_sha256": hashlib.sha256(
            _canonical_json(document["executable_config"])
        ).hexdigest(),
        "config_matches": not config_mismatches,
        "config_mismatches": config_mismatches,
        "protocol_matches": not mismatches,
        "protocol_mismatches": mismatches,
    }
    ctx._frozen_protocol = attestation
    if mismatches and not ctx.quick:
        preview = "; ".join(mismatches[:20])
        if len(mismatches) > 20:
            preview += f"; ... ({len(mismatches) - 20} more)"
        raise RuntimeError(
            "production config violates the frozen protocol artifact; create a "
            "new protocol/version for exploratory changes: " + preview
        )
    return document


def _status_entries() -> list[dict[str, Any]]:
    """Return NUL-safe porcelain entries, retaining both sides of renames."""
    status = _git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
    )
    if status in ("", "unknown"):
        return []
    tokens = status.split("\0")
    entries, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 4:
            continue
        state, path = token[:2], token[3:]
        paths = [path]
        if "R" in state or "C" in state:
            if index >= len(tokens) or not tokens[index]:
                raise RuntimeError("malformed NUL-delimited git rename status")
            paths.append(tokens[index])
            index += 1
        entries.append({"status": state, "path": path, "paths": paths})
    return entries


def _source_files() -> list[Path]:
    """Files whose working-tree contents can change a numerical result."""
    sources = [
        path
        for directory in SOURCE_DIRS
        for path in (REPO_ROOT / directory).rglob("*")
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.parts
        and path.suffix.lower() in SOURCE_SUFFIXES
    ]
    for top_level in (REPO_ROOT / "pyproject.toml",):
        if top_level.exists():
            sources.append(top_level)
    return sorted(set(sources), key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _is_numerical_source_path(relative: str) -> bool:
    path = Path(relative)
    if relative == "pyproject.toml":
        return True
    return (
        bool(path.parts)
        and path.parts[0] in SOURCE_DIRS
        and path.suffix.lower() in SOURCE_SUFFIXES
        and "__pycache__" not in path.parts
    )


def _entry_touches_numerical_source(entry: dict[str, Any]) -> bool:
    return any(_is_numerical_source_path(path)
               for path in entry.get("paths", [entry["path"]]))


def _head_source_entries() -> dict[str, dict[str, str]] | None:
    """Return mode/object metadata for every numerical source blob in HEAD."""
    raw = _git_bytes(
        "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--",
        *SOURCE_DIRS, "pyproject.toml",
    )
    if raw is None:
        return None
    entries = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
        relative = os.fsdecode(path_bytes)
        if object_type == "blob" and _is_numerical_source_path(relative):
            entries[relative] = {"mode": mode, "object_id": object_id}
    return entries


def _index_source_flags() -> list[dict[str, Any]]:
    """Expose flags that can make porcelain omit working-tree modifications."""
    raw = _git_bytes(
        "ls-files", "-v", "-z", "--", *SOURCE_DIRS, "pyproject.toml"
    )
    if raw is None:
        return []
    flags = []
    for record in raw.split(b"\0"):
        if len(record) < 3:
            continue
        tag = chr(record[0])
        relative = os.fsdecode(record[2:])
        if not _is_numerical_source_path(relative):
            continue
        assume_unchanged = tag.islower()
        skip_worktree = tag.upper() == "S"
        if assume_unchanged or skip_worktree:
            flags.append({
                "path": relative,
                "tag": tag,
                "assume_unchanged": assume_unchanged,
                "skip_worktree": skip_worktree,
            })
    return sorted(flags, key=lambda item: item["path"])


def _worktree_blob(path: Path) -> tuple[str, bytes] | None:
    """Return Git-style mode and bytes for a regular file or symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        return "120000", os.fsencode(os.readlink(path))
    if stat.S_ISREG(metadata.st_mode):
        mode = "100755" if metadata.st_mode & 0o111 else "100644"
        return mode, path.read_bytes()
    raise RuntimeError(f"unsupported numerical source file type: {path}")


def _verify_sources_against_head(
    sources: list[Path],
) -> dict[str, Any]:
    """Compare each working-tree source's bytes/type/mode directly with HEAD.

    This deliberately does not trust porcelain or index flags. In particular,
    ``assume-unchanged`` and ``skip-worktree`` can suppress a modified file from
    ``git status``; blob comparison still exposes it.
    """
    head_entries = _head_source_entries()
    index_flags = _index_source_flags()
    flags_by_path = {entry["path"]: entry for entry in index_flags}
    current = {
        path.relative_to(REPO_ROOT).as_posix(): path
        for path in sources
    }
    mismatches, verified = [], 0
    if head_entries is None:
        mismatches.extend({
            "status": "HEAD_MISMATCH",
            "path": relative,
            "paths": [relative],
            "reason": "head_tree_unavailable",
            "index_flags": flags_by_path.get(relative),
        } for relative in sorted(current))
        return {
            "n_head_source_files": None,
            "n_worktree_source_files": len(current),
            "n_verified_identical": 0,
            "index_flags": index_flags,
            "mismatches": mismatches,
        }

    for relative in sorted(set(head_entries) | set(current)):
        head = head_entries.get(relative)
        path = current.get(relative)
        reason = None
        head_mode = head["mode"] if head else None
        worktree_mode = None
        if head is None:
            reason = "not_in_head"
        elif path is None:
            reason = "missing_from_worktree"
        else:
            worktree = _worktree_blob(path)
            if worktree is None:
                reason = "missing_from_worktree"
            else:
                worktree_mode, content = worktree
                head_content = _git_bytes("cat-file", "blob", head["object_id"])
                if head_content is None:
                    reason = "head_blob_unavailable"
                elif worktree_mode != head_mode:
                    reason = "mode_or_file_type_differs_from_head"
                elif content != head_content:
                    reason = "content_differs_from_head"
                else:
                    verified += 1
        if reason is not None:
            mismatches.append({
                "status": "HEAD_MISMATCH",
                "path": relative,
                "paths": [relative],
                "reason": reason,
                "head_mode": head_mode,
                "worktree_mode": worktree_mode,
                "index_flags": flags_by_path.get(relative),
            })
    return {
        "n_head_source_files": len(head_entries),
        "n_worktree_source_files": len(current),
        "n_verified_identical": verified,
        "index_flags": index_flags,
        "mismatches": mismatches,
    }


def _code_digest() -> dict:
    """Hash the complete working-tree source used by an experiment.

    `git_dirty` alone is a bare boolean: it says the tree differed from the
    commit but not how, so a reader cannot tell an edited sampler from an edited
    README.  The digest covers tracked *and untracked* source/configuration files
    in the numerical code paths.  Length delimiters prevent ambiguous
    concatenations of paths or contents.
    """
    porcelain_entries = _status_entries()
    digest = hashlib.sha256()
    sources = _source_files()
    for path in sources:
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        mode, content = _worktree_blob(path)  # type: ignore[misc]
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(mode.encode("ascii"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    head_verification = _verify_sources_against_head(sources)
    porcelain_source_dirty = [
        entry for entry in porcelain_entries
        if _entry_touches_numerical_source(entry)
    ]
    porcelain_source_paths = {
        path for entry in porcelain_source_dirty
        for path in entry.get("paths", [entry["path"]])
        if _is_numerical_source_path(path)
    }
    hidden_mismatches = [
        entry for entry in head_verification["mismatches"]
        if entry["path"] not in porcelain_source_paths
    ]
    entries = [*porcelain_entries, *hidden_mismatches]
    source_dirty = [*porcelain_source_dirty, *hidden_mismatches]
    unreconstructable = [
        {
            "path": entry["path"],
            "reason": entry["reason"],
            "index_flags": entry.get("index_flags"),
        }
        for entry in head_verification["mismatches"]
    ]
    return {
        "dirty_paths": sorted({path for entry in entries
                               for path in entry.get("paths", [entry["path"]])}),
        "dirty_entries": entries,
        "source_dirty_entries": source_dirty,
        "unreconstructable_source_entries": unreconstructable,
        "head_verification": head_verification,
        "n_source_files": len(sources),
        "sha256": digest.hexdigest(),
        "scope": [*SOURCE_DIRS, "pyproject.toml"],
    }


def _source_snapshot() -> dict:
    """Capture the immutable launch/end provenance used to validate a run."""
    code = _code_digest()
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(code["dirty_entries"]),
        "source_dirty": bool(code["source_dirty_entries"]),
        "source_reconstructable": not bool(code["unreconstructable_source_entries"]),
        "head_tree": _git("rev-parse", "HEAD^{tree}"),
        "code": code,
    }


def _artifact_inventory(output_dir: Path, registered_paths=None) -> list[dict[str, Any]]:
    """Content hashes for completed evidence files, excluding manifests."""
    inventory = []
    if not output_dir.exists():
        return inventory
    paths = (sorted(Path(path) for path in registered_paths)
             if registered_paths is not None
             else sorted(p for p in output_dir.rglob("*") if p.is_file()))
    for path in paths:
        if not path.is_file() or output_dir not in path.parents:
            continue
        if path.name in {"manifest.json", "manifest.inprogress.json"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append({
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return inventory


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
    _source_start: dict = field(default_factory=dict, init=False, repr=False)
    _started_at: str = field(default="", init=False, repr=False)
    _written_artifacts: set[Path] = field(default_factory=set, init=False, repr=False)
    _lock_path: Path | None = field(default=None, init=False, repr=False)
    _frozen_protocol: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Own an immutable-at-construction copy of the executable configuration.
        # Besides preventing callers from mutating a module-level DEFAULTS mapping,
        # this keeps the frozen protocol object independent from the configuration
        # that command-line/config-file merging is allowed to modify.
        self.config = json.loads(json.dumps(self.config))
        # Capture before creating an output directory: otherwise a clean launch
        # is made dirty by the harness itself and the manifest cannot distinguish
        # numerical source edits from its own result files.
        self._source_start = _source_snapshot()
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if self.output_dir is None:
            self.output_dir = RESULTS_ROOT / self.name
        if self.figure_dir is None:
            self.figure_dir = FIGURES_ROOT
        self.output_dir = Path(self.output_dir)
        self.figure_dir = Path(self.figure_dir)
        existing = list(self.output_dir.iterdir()) if self.output_dir.exists() else []
        if existing and not self.quick:
            raise RuntimeError(
                "production output directory must be fresh and empty; choose a new "
                f"--output path instead of reusing {self.output_dir}"
            )
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
        if target.name not in {"manifest.json", "manifest.inprogress.json"}:
            self._written_artifacts.add(target)
        return target

    def save_npz(self, name: str, **arrays) -> Path:
        target = self.path(name if name.endswith(".npz") else f"{name}.npz")
        tmp = target.with_suffix(".npz.tmp")
        # Passing a path whose suffix is not ``.npz`` makes NumPy silently add
        # another suffix, leaving ``tmp`` absent and breaking the atomic rename.
        # A binary file handle preserves the exact temporary path.
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(target)
        self._written_artifacts.add(target)
        return target

    def acquire_run_lock(self) -> None:
        """Prevent two processes from interleaving one evidence directory."""
        lock = self.output_dir / ".run.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"another run holds {lock}") from exc
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"pid={os.getpid()} started={self._started_at}\n")
        self._lock_path = lock

    def release_run_lock(self) -> None:
        if self._lock_path is not None:
            self._lock_path.unlink(missing_ok=True)
            self._lock_path = None

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

    def write_provisional_manifest(self) -> Path:
        """Write launch provenance immediately, so killed runs remain auditable."""
        return self.save_json("manifest.inprogress", {
            "experiment": self.name,
            "status": "running",
            "started_at": self._started_at,
            "source_start": self._source_start,
            "seed": self.seed,
            "quick_mode": self.quick,
            "config": self.config,
            "frozen_protocol": self._frozen_protocol,
            "invocation": list(sys.argv),
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "libraries": _library_versions(),
        })

    def write_manifest(
        self,
        extra: dict | None = None,
        *,
        status: str = "complete",
        exception: BaseException | None = None,
    ) -> Path:
        source_end = _source_snapshot()
        code_stable = (
            self._source_start["code"]["sha256"]
            == source_end["code"]["sha256"]
        )
        provenance_valid = bool(
            code_stable
            and not self._source_start["source_dirty"]
            and self._source_start["source_reconstructable"]
            and source_end["source_reconstructable"]
            and self._source_start["git_commit"] != "unknown"
            and self._source_start["head_tree"] != "unknown"
            and self._source_start["head_tree"] == source_end["head_tree"]
        )
        final_status = status
        if status == "complete" and self.quick:
            final_status = "smoke_only"
        elif status == "complete" and not provenance_valid:
            final_status = "invalid_provenance"
        manifest = {
            "experiment": self.name,
            "status": final_status,
            # Legacy top-level fields intentionally refer to launch, not finish.
            "git_commit": self._source_start["git_commit"],
            "git_branch": self._source_start["git_branch"],
            "git_dirty": self._source_start["git_dirty"],
            "code": self._source_start["code"],
            "source_start": self._source_start,
            "source_end": source_end,
            "source_code_stable": code_stable,
            "provenance_valid": provenance_valid,
            "seed": self.seed,
            "quick_mode": self.quick,
            "config": self.config,
            "frozen_protocol": self._frozen_protocol,
            "invocation": list(sys.argv),
            "timings_seconds": self._timings,
            "total_seconds": time.time() - self._start,
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "libraries": _library_versions(),
            "artifacts": _artifact_inventory(
                self.output_dir, self._written_artifacts
            ),
            "started_at": self._started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if exception is not None:
            manifest["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": "".join(traceback.format_exception(exception)),
            }
        if extra:
            manifest["results"] = extra
        target = self.save_json("manifest", manifest)
        self.path("manifest.inprogress.json").unlink(missing_ok=True)
        return target


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
    protocol_path: str | Path | None = None,
    protocol_name: str | None = None,
    protocol_version: int | None = None,
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
    if protocol_path is not None:
        require_frozen_protocol(
            ctx,
            protocol_path=protocol_path,
            expected_protocol=protocol_name,
            expected_version=protocol_version,
        )
    if not args.quick and ctx._source_start["source_dirty"]:
        dirty = ", ".join(
            entry["path"] for entry in ctx._source_start["code"]["source_dirty_entries"]
        )
        raise RuntimeError(
            "refusing a production run from dirty numerical source; commit or "
            f"stash these files first: {dirty}"
        )
    if not args.quick and not ctx._source_start["source_reconstructable"]:
        bad = ", ".join(
            f"{entry['path']} ({entry['reason']})"
            for entry in ctx._source_start["code"]["unreconstructable_source_entries"]
        )
        raise RuntimeError(
            "refusing a production run with source that cannot be reconstructed "
            f"from HEAD: {bad}"
        )
    ctx.acquire_run_lock()
    try:
        ctx.write_provisional_manifest()
        banner = f"=== {experiment_name}{' (QUICK)' if args.quick else ''} ==="
        print(banner, flush=True)
        try:
            results = run(ctx)
        except BaseException as exc:
            ctx.write_manifest(status="failed", exception=exc)
            raise
        manifest_path = ctx.write_manifest(results if isinstance(results, dict) else None)
        manifest = json.loads(manifest_path.read_text())
        if not args.quick and not manifest["provenance_valid"]:
            raise RuntimeError(
                "experiment finished but provenance is invalid: launch source was "
                "dirty, unreconstructable, or numerical source changed during the "
                "run; results must not be used as production evidence"
            )
        print(f"=== {experiment_name} complete: {ctx.output_dir} ===", flush=True)
        return results
    finally:
        ctx.release_run_lock()
