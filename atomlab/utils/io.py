"""Persistence for configurations, datasets and trajectories.

Two storage formats are provided, for two different jobs.

**``.npz`` (archival).**  :func:`save_configurations` / :func:`load_configurations`
and :func:`save_trajectory` / :func:`load_trajectory` write a single compressed
NumPy archive that round-trips every field *bitwise*.  This is the format the
experiments use, because a reference dataset that is only accurate to the 15th
digit is not a reference dataset: the perturbation analysis subtracts energies
of nearly identical models, and a 1-ulp error introduced by re-reading a file
would show up as spurious ``delta_U``.  Systems of different sizes are stored
ragged -- all per-atom arrays concatenated along axis 0, plus an ``offsets``
array of length ``M + 1`` -- which avoids both padding and object arrays.

**extended XYZ (inspection).**  :func:`write_extxyz` / :func:`read_extxyz` speak
the standard format understood by OVITO, VMD and ASE, so a trajectory can be
looked at with external tools.  Floats are written with ``repr``, which is the
shortest decimal string that round-trips a float64 exactly, so in practice this
format is also lossless -- but it is text, it is large, and the *format* only
promises "to float precision", so it is not what the archival path uses.

Both paths preserve the ``info`` dictionary.  ``info`` is JSON-encoded; NumPy
arrays and scalars inside it survive via a tagged base64 encoding (see
:func:`encode_tree`), which keeps them bit-exact for any dtype.
"""

from __future__ import annotations

import base64
import gzip
import io as _io
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..types import Configuration, Dataset, Trajectory
from ..units import ATOMIC_MASSES

__all__ = [
    "save_configurations",
    "load_configurations",
    "save_dataset",
    "load_dataset",
    "save_trajectory",
    "load_trajectory",
    "write_extxyz",
    "read_extxyz",
    "save_json",
    "load_json",
    "NumpyJSONEncoder",
    "encode_tree",
    "decode_tree",
]

PathLike = "str | os.PathLike[str]"

#: Bumped whenever the on-disk layout changes incompatibly.
FORMAT_VERSION = 1


# ==========================================================================
# JSON
# ==========================================================================


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars, arrays and paths.

    Arrays become nested lists and NumPy scalars become Python scalars, i.e.
    the output is *readable* JSON with no atomlab-specific tags.  The cost is
    that ``load_json`` returns lists, not arrays, and integer/float widths are
    lost.  Use ``save_json(..., preserve_numpy=True)`` when exact round-tripping
    matters more than readability.
    """

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, complex):
            return {"real": o.real, "imag": o.imag}
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def encode_tree(obj: Any) -> Any:
    """Recursively convert ``obj`` to a JSON-safe, *losslessly decodable* tree.

    NumPy arrays and scalars are stored as base64 of their raw buffer together
    with dtype and shape, so every dtype -- float32, complex128, int8 -- comes
    back bit-identical.  ``tuple``/``bytes`` are tagged so they do not decay to
    ``list``/``str``.  Python floats are left alone: :mod:`json` writes them
    with ``repr``, which round-trips float64 exactly.

    Raises
    ------
    TypeError
        On a dict key that is not a string (JSON has no such thing, and a
        silent ``str()`` coercion would corrupt the data), or on a type with no
        lossless representation.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
        }
    if isinstance(obj, np.generic):
        arr = np.asarray(obj)
        return {
            "__npscalar__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": arr.dtype.str,
        }
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, tuple):
        return {"__tuple__": [encode_tree(v) for v in obj]}
    if isinstance(obj, list):
        return [encode_tree(v) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"info dict keys must be str, got {type(k).__name__}: {k!r}")
            out[k] = encode_tree(v)
        return out
    if isinstance(obj, Path):
        return {"__path__": str(obj)}
    raise TypeError(f"cannot losslessly encode {type(obj).__name__}")


def decode_tree(obj: Any) -> Any:
    """Inverse of :func:`encode_tree`."""
    if isinstance(obj, list):
        return [decode_tree(v) for v in obj]
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            raw = base64.b64decode(obj["__ndarray__"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        if "__npscalar__" in obj:
            raw = base64.b64decode(obj["__npscalar__"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"]))[0]
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"])
        if "__tuple__" in obj:
            return tuple(decode_tree(v) for v in obj["__tuple__"])
        if "__path__" in obj:
            return Path(obj["__path__"])
        return {k: decode_tree(v) for k, v in obj.items()}
    return obj


def save_json(path: "PathLike", obj: Any, *, indent: int = 2, preserve_numpy: bool = False) -> Path:
    """Write ``obj`` as JSON, tolerating NumPy types.

    Parameters
    ----------
    path:
        Destination.  A ``.gz`` suffix triggers gzip compression.
    obj:
        Any JSON-able structure, possibly containing NumPy values.
    indent:
        Passed to :func:`json.dump`.  Keep it non-zero: these files are meant
        to be read by humans auditing an experiment.
    preserve_numpy:
        If True, encode with :func:`encode_tree` so that :func:`load_json`
        restores NumPy arrays/dtypes and tuples exactly.  If False (default)
        arrays are flattened to plain lists.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    if preserve_numpy:
        payload = json.dumps(
            {"__atomlab_json__": FORMAT_VERSION, "data": encode_tree(obj)},
            indent=indent,
            sort_keys=False,
        )
    else:
        payload = json.dumps(obj, indent=indent, cls=NumpyJSONEncoder, sort_keys=False)
    _write_text(p, payload + "\n")
    return p


def load_json(path: "PathLike") -> Any:
    """Read a file written by :func:`save_json` (either encoding)."""
    raw = json.loads(_read_text(Path(path)))
    if isinstance(raw, dict) and "__atomlab_json__" in raw:
        return decode_tree(raw["data"])
    return raw


def _write_text(path: Path, text: str) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


# ==========================================================================
# npz: configurations
# ==========================================================================


def _pack_configurations(configs: Sequence[Configuration], prefix: str = "") -> dict[str, Any]:
    """Flatten a ragged list of configurations into a flat array dict.

    All per-atom quantities are concatenated along axis 0 and indexed by
    ``offsets`` (length ``M + 1``, ``offsets[0] == 0``).  Presence of the
    optional labels is recorded in boolean masks rather than by a sentinel
    value, because ``NaN`` is a legitimate (if alarming) energy for a diverged
    model and must survive the round trip as itself.
    """
    m = len(configs)
    counts = np.array([c.n_atoms for c in configs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])

    positions = (
        np.concatenate([c.positions for c in configs], axis=0)
        if m
        else np.zeros((0, 3), dtype=np.float64)
    )
    masses = (
        np.concatenate([c.masses for c in configs]) if m else np.zeros(0, dtype=np.float64)
    )
    species = (
        np.concatenate([c.species for c in configs]).astype(np.int32)
        if m
        else np.zeros(0, dtype=np.int32)
    )

    cells = np.stack([c.cell for c in configs]) if m else np.zeros((0, 3, 3))
    pbc = np.stack([c.pbc for c in configs]) if m else np.zeros((0, 3), dtype=bool)

    has_energy = np.array([c.energy is not None for c in configs], dtype=bool)
    energy = np.array([0.0 if c.energy is None else c.energy for c in configs], dtype=np.float64)

    has_forces = np.array([c.forces is not None for c in configs], dtype=bool)
    forces = np.zeros((total, 3), dtype=np.float64)
    for k, c in enumerate(configs):
        if c.forces is not None:
            forces[offsets[k] : offsets[k + 1]] = c.forces

    has_virial = np.array([c.virial is not None for c in configs], dtype=bool)
    virial = np.zeros((m, 3, 3), dtype=np.float64)
    for k, c in enumerate(configs):
        if c.virial is not None:
            virial[k] = c.virial

    meta = {
        "symbols": [list(c.symbols) for c in configs],
        "info": [encode_tree(c.info) for c in configs],
    }

    return {
        f"{prefix}offsets": offsets,
        f"{prefix}positions": positions,
        f"{prefix}cells": cells,
        f"{prefix}pbc": pbc,
        f"{prefix}species": species,
        f"{prefix}masses": masses,
        f"{prefix}energy": energy,
        f"{prefix}has_energy": has_energy,
        f"{prefix}forces": forces,
        f"{prefix}has_forces": has_forces,
        f"{prefix}virial": virial,
        f"{prefix}has_virial": has_virial,
        f"{prefix}meta_json": np.array(json.dumps(meta)),
    }


def _unpack_configurations(z: Any, prefix: str = "") -> list[Configuration]:
    offsets = np.asarray(z[f"{prefix}offsets"])
    positions = np.asarray(z[f"{prefix}positions"])
    cells = np.asarray(z[f"{prefix}cells"])
    pbc = np.asarray(z[f"{prefix}pbc"])
    species = np.asarray(z[f"{prefix}species"])
    masses = np.asarray(z[f"{prefix}masses"])
    energy = np.asarray(z[f"{prefix}energy"])
    has_energy = np.asarray(z[f"{prefix}has_energy"])
    forces = np.asarray(z[f"{prefix}forces"])
    has_forces = np.asarray(z[f"{prefix}has_forces"])
    virial = np.asarray(z[f"{prefix}virial"])
    has_virial = np.asarray(z[f"{prefix}has_virial"])
    meta = json.loads(str(z[f"{prefix}meta_json"].item()))

    out: list[Configuration] = []
    for k in range(len(offsets) - 1):
        lo, hi = int(offsets[k]), int(offsets[k + 1])
        out.append(
            Configuration(
                positions=positions[lo:hi].copy(),
                cell=cells[k].copy(),
                pbc=pbc[k].copy(),
                species=species[lo:hi].copy(),
                symbols=tuple(meta["symbols"][k]),
                masses=masses[lo:hi].copy(),
                energy=float(energy[k]) if bool(has_energy[k]) else None,
                forces=forces[lo:hi].copy() if bool(has_forces[k]) else None,
                virial=virial[k].copy() if bool(has_virial[k]) else None,
                info=decode_tree(meta["info"][k]),
            )
        )
    return out


def save_configurations(
    path: "PathLike",
    configurations: Iterable[Configuration],
    *,
    info: dict | None = None,
    compress: bool = True,
) -> Path:
    """Write a list of configurations to a single ``.npz`` archive.

    Parameters
    ----------
    path:
        Destination file.  ``.npz`` is appended if missing.
    configurations:
        Any iterable of :class:`~atomlab.types.Configuration`.  Sizes may
        differ between entries (ragged storage).
    info:
        Optional archive-level metadata dict (provenance: potential name,
        temperature grid, git commit ...).  Stored alongside the per-config
        ``info`` dicts.
    compress:
        Use ``savez_compressed``.  Deflate is lossless, so this never affects
        fidelity; it typically shrinks MD snapshots by ~30%.

    Returns
    -------
    pathlib.Path
        The path actually written.

    Notes
    -----
    Round trip is **bitwise** for every float64 field (positions, cell, masses,
    energy, forces, virial) because the values go through NumPy's binary
    format, not text.  ``info`` values go through JSON with the lossless
    encoding of :func:`encode_tree`.
    """
    configs = list(configurations)
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(p.suffix + ".npz")
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    payload = _pack_configurations(configs)
    payload["format_version"] = np.array(FORMAT_VERSION)
    payload["kind"] = np.array("configurations")
    payload["archive_info_json"] = np.array(json.dumps(encode_tree(info or {})))

    saver = np.savez_compressed if compress else np.savez
    saver(p, **payload)
    return p


def load_configurations(path: "PathLike") -> list[Configuration]:
    """Read configurations written by :func:`save_configurations`.

    Returns
    -------
    list of Configuration
        In the order they were written.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        kind = str(z["kind"].item()) if "kind" in z.files else "configurations"
        if kind not in ("configurations", "dataset"):
            raise ValueError(f"{path} is a {kind!r} archive, not configurations")
        return _unpack_configurations(z)


def load_archive_info(path: "PathLike") -> dict:
    """Return the archive-level ``info`` dict stored by the ``save_*`` helpers."""
    with np.load(Path(path), allow_pickle=False) as z:
        if "archive_info_json" not in z.files:
            return {}
        return decode_tree(json.loads(str(z["archive_info_json"].item())))


def save_dataset(path: "PathLike", dataset: Dataset, *, compress: bool = True) -> Path:
    """Write a :class:`~atomlab.types.Dataset` (configurations + dataset info)."""
    return save_configurations(
        path, dataset.configurations, info=dataset.info, compress=compress
    )


def load_dataset(path: "PathLike") -> Dataset:
    """Read a :class:`~atomlab.types.Dataset` written by :func:`save_dataset`."""
    return Dataset(
        configurations=load_configurations(path), info=dict(load_archive_info(path))
    )


# ==========================================================================
# npz: trajectories
# ==========================================================================


def save_trajectory(path: "PathLike", trajectory: Trajectory, *, compress: bool = True) -> Path:
    """Write a :class:`~atomlab.types.Trajectory` to a single ``.npz`` archive.

    Positions are stored exactly as held in memory, i.e. **unwrapped**, so that
    mean-squared displacements computed from a reloaded trajectory match those
    computed in-process to the last bit.

    Parameters
    ----------
    path:
        Destination; ``.npz`` appended if missing.
    trajectory:
        The trajectory.  ``velocities`` may be ``None``; ``scalars`` may hold
        any number of ``(T,)`` logs.
    compress:
        Use deflate (lossless).

    Returns
    -------
    pathlib.Path
    """
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(p.suffix + ".npz")
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": np.array(FORMAT_VERSION),
        "kind": np.array("trajectory"),
        "positions": trajectory.positions,
        "cells": trajectory.cells,
        "times": trajectory.times,
        "has_velocities": np.array(trajectory.velocities is not None),
        "velocities": (
            trajectory.velocities
            if trajectory.velocities is not None
            else np.zeros((0, 0, 3), dtype=np.float64)
        ),
        "scalar_keys_json": np.array(json.dumps(list(trajectory.scalars.keys()))),
        "archive_info_json": np.array(json.dumps(encode_tree(trajectory.info))),
    }
    for key, val in trajectory.scalars.items():
        # Prefixed to keep the key namespace disjoint from the fixed fields; a
        # scalar log named "times" would otherwise silently clobber the axis.
        payload[f"scalar__{key}"] = np.asarray(val)
    # The template is a single-configuration block reusing the exact same
    # packing code, so template fidelity cannot drift from dataset fidelity.
    payload.update(_pack_configurations([trajectory.template], prefix="template_"))

    saver = np.savez_compressed if compress else np.savez
    saver(p, **payload)
    return p


def load_trajectory(path: "PathLike") -> Trajectory:
    """Read a trajectory written by :func:`save_trajectory`."""
    with np.load(Path(path), allow_pickle=False) as z:
        kind = str(z["kind"].item()) if "kind" in z.files else "trajectory"
        if kind != "trajectory":
            raise ValueError(f"{path} is a {kind!r} archive, not a trajectory")
        keys = json.loads(str(z["scalar_keys_json"].item()))
        scalars = {k: np.asarray(z[f"scalar__{k}"]).copy() for k in keys}
        template = _unpack_configurations(z, prefix="template_")[0]
        return Trajectory(
            positions=np.asarray(z["positions"]).copy(),
            cells=np.asarray(z["cells"]).copy(),
            times=np.asarray(z["times"]).copy(),
            template=template,
            velocities=(
                np.asarray(z["velocities"]).copy() if bool(z["has_velocities"].item()) else None
            ),
            scalars=scalars,
            info=decode_tree(json.loads(str(z["archive_info_json"].item()))),
        )


# ==========================================================================
# extended XYZ
# ==========================================================================

# Dummy element names used when a configuration carries no `symbols`.  They are
# real element symbols so that external viewers do not choke, but the authoritative
# type->symbol mapping is always written into the `atomlab_symbols` key as well.
_PLACEHOLDER_SYMBOLS = ("X", "Xe", "Kr", "Ne", "He", "Ar")


def _fmt(x: float) -> str:
    """Shortest decimal string that reads back as the same float64.

    ``repr`` of a Python float has been shortest-round-trip since CPython 3.1,
    so this is lossless despite being a text format.
    """
    return repr(float(x))


def _symbols_for(cfg: Configuration) -> tuple[str, ...]:
    if cfg.symbols:
        return tuple(cfg.symbols)
    n_types = int(cfg.species.max()) + 1 if cfg.n_atoms else 0
    if n_types > len(_PLACEHOLDER_SYMBOLS):
        raise ValueError(
            "cannot write extxyz: configuration has more species than placeholder "
            "symbols; set Configuration.symbols explicitly"
        )
    return _PLACEHOLDER_SYMBOLS[:n_types]


def _masses_recoverable(cfg: Configuration, symbols: tuple[str, ...]) -> bool:
    """True if masses can be regenerated from ``symbols`` on read-back."""
    if not cfg.symbols:
        return False
    try:
        lut = np.array([ATOMIC_MASSES[s] for s in symbols])
    except KeyError:
        return False
    return bool(np.array_equal(lut[cfg.species], cfg.masses))


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_extxyz(
    path: "PathLike",
    configurations: Iterable[Configuration] | Configuration | Trajectory,
    *,
    write_forces: bool = True,
    append: bool = False,
) -> Path:
    """Write configurations in the extended-XYZ format.

    The comment (second) line of each frame carries ``key=value`` pairs::

        Lattice="ax ay az bx by bz cx cy cz"
        Properties=species:S:1:pos:R:3:forces:R:3
        pbc="T T T" energy=... virial="..."

    The ``Lattice`` string is the row-major flattening of the ``(3, 3)`` cell,
    i.e. lattice vector ``a1`` first -- the same row convention used throughout
    :mod:`atomlab` (``r = s @ cell``) and by ASE, so files interchange without
    a transpose.

    Two non-standard keys are added, both ignored by other readers:
    ``atomlab_symbols`` (preserves the *type index* ordering, which a bare
    species column cannot) and ``atomlab_info`` (JSON of the ``info`` dict).
    A ``masses:R:1`` column is appended only when the masses cannot be
    regenerated from the chemical symbols, so the common case stays canonical.

    Parameters
    ----------
    path:
        Destination.  ``.gz`` triggers gzip.
    configurations:
        A single configuration, an iterable of them, or a
        :class:`~atomlab.types.Trajectory` (iterated frame by frame).
    write_forces:
        Include a ``forces:R:3`` column.  Only honoured for frames that
        actually carry forces.
    append:
        Append frames to an existing file instead of truncating.

    Returns
    -------
    pathlib.Path
    """
    if isinstance(configurations, Configuration):
        configs: Iterable[Configuration] = [configurations]
    else:
        configs = configurations

    p = Path(path)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    buf = _io.StringIO()
    for cfg in configs:
        buf.write(_frame_to_extxyz(cfg, write_forces=write_forces))

    text = buf.getvalue()
    if append and p.exists():
        text = _read_text(p) + text
    _write_text(p, text)
    return p


def _frame_to_extxyz(cfg: Configuration, *, write_forces: bool) -> str:
    symbols = _symbols_for(cfg)
    with_forces = write_forces and cfg.forces is not None
    with_masses = not _masses_recoverable(cfg, symbols)

    props = "species:S:1:pos:R:3"
    if with_forces:
        props += ":forces:R:3"
    if with_masses:
        props += ":masses:R:1"

    lattice = " ".join(_fmt(v) for v in np.asarray(cfg.cell).reshape(-1))
    pbc = " ".join("T" if b else "F" for b in cfg.pbc)

    parts = [f'Lattice="{lattice}"', f"Properties={props}", f'pbc="{pbc}"']
    if cfg.energy is not None:
        parts.append(f"energy={_fmt(cfg.energy)}")
    if cfg.virial is not None:
        parts.append('virial="' + " ".join(_fmt(v) for v in cfg.virial.reshape(-1)) + '"')
    parts.append('atomlab_symbols="' + " ".join(symbols) + '"')
    if cfg.info:
        parts.append('atomlab_info="' + _escape(json.dumps(encode_tree(cfg.info))) + '"')

    lines = [str(cfg.n_atoms), " ".join(parts)]
    for i in range(cfg.n_atoms):
        row = [symbols[int(cfg.species[i])]]
        row += [_fmt(v) for v in cfg.positions[i]]
        if with_forces:
            row += [_fmt(v) for v in cfg.forces[i]]
        if with_masses:
            row.append(_fmt(cfg.masses[i]))
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


_KV_RE = re.compile(
    r"""\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S+))"""
)


def _parse_comment(line: str) -> dict[str, str]:
    """Split an extxyz comment line into ``key -> raw string value``.

    Handles double-quoted values containing spaces and backslash escapes, which
    is required for the JSON ``info`` blob and for ``Lattice``.
    """
    out: dict[str, str] = {}
    for m in _KV_RE.finditer(line):
        key = m.group(1)
        if m.group(2) is not None:
            val = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
        elif m.group(3) is not None:
            val = m.group(3)
        else:
            val = m.group(4)
        out[key] = val
    return out


def _parse_properties(spec: str) -> list[tuple[str, str, int]]:
    """Parse ``name:type:count:...`` into a list of ``(name, type, count)``."""
    fields = spec.split(":")
    if len(fields) % 3 != 0:
        raise ValueError(f"malformed Properties spec: {spec!r}")
    return [
        (fields[i], fields[i + 1], int(fields[i + 2])) for i in range(0, len(fields), 3)
    ]


def read_extxyz(path: "PathLike", *, index: int | slice | None = None) -> list[Configuration]:
    """Read an extended-XYZ file into configurations.

    Recognised comment-line keys: ``Lattice``, ``Properties``, ``pbc``,
    ``energy``, ``virial``, plus the atomlab extensions ``atomlab_symbols`` and
    ``atomlab_info``.  Unknown keys are collected into ``Configuration.info``
    (as strings), so nothing written by another code is silently dropped.

    Recognised columns: ``species`` (or ``Z``/``symbol``), ``pos``, ``forces``
    (or ``force``), ``masses``.  Any other column is read and discarded.

    Parameters
    ----------
    path:
        Source file; ``.gz`` is decompressed transparently.
    index:
        Optional frame selection applied after parsing.

    Returns
    -------
    list of Configuration

    Notes
    -----
    ``pbc`` defaults to ``T T T`` when a ``Lattice`` is present and to all-False
    when it is not -- an open cluster and a periodic cell are genuinely
    different physics, so we never guess a cell that was not written.
    """
    text = _read_text(Path(path))
    lines = text.splitlines()

    configs: list[Configuration] = []
    k = 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].strip())
        comment = lines[k + 1] if k + 1 < len(lines) else ""
        body = lines[k + 2 : k + 2 + n]
        if len(body) != n:
            raise ValueError(f"{path}: frame starting at line {k + 1} is truncated")
        configs.append(_frame_from_extxyz(n, comment, body))
        k += 2 + n

    if index is None:
        return configs
    if isinstance(index, int):
        return [configs[index]]
    return configs[index]


_KNOWN_KEYS = {"Lattice", "Properties", "pbc", "energy", "virial", "atomlab_symbols", "atomlab_info"}


def _frame_from_extxyz(n: int, comment: str, body: Sequence[str]) -> Configuration:
    kv = _parse_comment(comment)

    if "Lattice" in kv:
        cell = np.array([float(x) for x in kv["Lattice"].split()], dtype=np.float64)
        if cell.size != 9:
            raise ValueError(f"Lattice must have 9 entries, got {cell.size}")
        cell = cell.reshape(3, 3)
        default_pbc = np.array([True, True, True])
    else:
        cell = np.zeros((3, 3))
        default_pbc = np.array([False, False, False])

    if "pbc" in kv:
        toks = kv["pbc"].replace(",", " ").split()
        pbc = np.array([t.upper() in ("T", "TRUE", "1") for t in toks])
        if pbc.size != 3:
            raise ValueError(f"pbc must have 3 entries, got {pbc.size}")
    else:
        pbc = default_pbc

    props = _parse_properties(kv.get("Properties", "species:S:1:pos:R:3"))
    n_cols = sum(c for _, _, c in props)

    table = [ln.split() for ln in body]
    for row in table:
        if len(row) < n_cols:
            raise ValueError(f"expected {n_cols} columns, got {len(row)}: {row}")

    col = 0
    raw: dict[str, list[list[str]]] = {}
    for name, _typ, count in props:
        raw[name] = [row[col : col + count] for row in table]
        col += count

    sym_key = next((k for k in ("species", "symbol", "symbols", "Z") if k in raw), None)
    if sym_key is None:
        raise ValueError("extxyz frame has no species column")
    atom_symbols = [r[0] for r in raw[sym_key]]

    pos_key = next((k for k in ("pos", "positions") if k in raw), None)
    if pos_key is None:
        raise ValueError("extxyz frame has no pos column")
    positions = np.array([[float(v) for v in r] for r in raw[pos_key]], dtype=np.float64)

    force_key = next((k for k in ("forces", "force") if k in raw), None)
    forces = (
        np.array([[float(v) for v in r] for r in raw[force_key]], dtype=np.float64)
        if force_key
        else None
    )

    if "atomlab_symbols" in kv:
        # Authoritative ordering: preserves the original type indices even when
        # a species happens not to appear in this particular frame.
        symbols = tuple(kv["atomlab_symbols"].split())
    else:
        seen: list[str] = []
        for s in atom_symbols:
            if s not in seen:
                seen.append(s)
        symbols = tuple(seen)
    index_of = {s: i for i, s in enumerate(symbols)}
    species = np.array([index_of[s] for s in atom_symbols], dtype=np.int32)

    masses = None
    if "masses" in raw:
        masses = np.array([float(r[0]) for r in raw["masses"]], dtype=np.float64)
    elif not all(s in ATOMIC_MASSES for s in symbols):
        # Unknown element: Configuration would raise looking the mass up, and a
        # silently wrong mass would corrupt any dynamics run from this file.
        masses = np.ones(n, dtype=np.float64)

    energy = float(kv["energy"]) if "energy" in kv else None
    virial = (
        np.array([float(x) for x in kv["virial"].split()], dtype=np.float64).reshape(3, 3)
        if "virial" in kv
        else None
    )

    if "atomlab_info" in kv:
        info = decode_tree(json.loads(kv["atomlab_info"]))
    else:
        info = {}
    for key, val in kv.items():
        if key not in _KNOWN_KEYS and key not in info:
            info[key] = val

    return Configuration(
        positions=positions,
        cell=cell,
        pbc=pbc,
        species=species,
        symbols=symbols,
        masses=masses,
        energy=energy,
        forces=forces,
        virial=virial,
        info=info,
    )
