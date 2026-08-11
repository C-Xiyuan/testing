"""Core data structures shared by every module in :mod:`atomlab`.

Everything in this project speaks in terms of :class:`Configuration`: a set of
atomic positions in a (possibly periodic) simulation cell, optionally decorated
with reference labels (energy, forces, virial).  Potentials consume
configurations and produce :class:`Result` objects; molecular dynamics produces
:class:`Trajectory` objects, which are sequences of configurations plus
velocities.

Sign and index conventions
--------------------------

*Cell.*  ``cell`` is a ``(3, 3)`` array whose **rows** are the lattice vectors
``a1, a2, a3``.  Fractional coordinates ``s`` and cartesian coordinates ``r``
are therefore related by ``r = s @ cell``.

*Virial.*  ``virial`` is the ``(3, 3)`` tensor

.. math::  W_{ab} = -\\left.\\frac{\\partial U}{\\partial \\epsilon_{ab}}\\right|_{\\epsilon=0}

where the strain :math:`\\epsilon` acts as :math:`r \\to (1 + \\epsilon) r` on
both atomic positions and lattice vectors.  Units are eV (not eV/A^3).  For a
pair potential this equals :math:`\\sum_{i<j} f_{ij} \\otimes r_{ij}` with
:math:`f_{ij}` the force on atom *i* due to atom *j* and
:math:`r_{ij} = r_i - r_j`.  The potential part of the stress tensor is
``W / V``; :func:`atomlab.md.observables` adds the kinetic part.

*Species.*  ``species`` holds integer type indices ``0 .. n_types-1``, not
atomic numbers.  Potentials define their own mapping from type index to
chemistry, and :attr:`Configuration.symbols` records the human-readable names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

import numpy as np

from .units import ATOMIC_MASSES

__all__ = [
    "Configuration",
    "Result",
    "Trajectory",
    "Dataset",
]


def _as_f64(a, name: str, shape=None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


@dataclass
class Configuration:
    """An atomic configuration in a periodic (or open) simulation cell.

    Parameters
    ----------
    positions:
        ``(N, 3)`` cartesian coordinates in angstrom.  Positions are *not*
        automatically wrapped into the cell; use :meth:`wrapped` for that.
    cell:
        ``(3, 3)`` matrix whose rows are the lattice vectors, in angstrom.  For
        a non-periodic system pass zeros (or ``None``) and set ``pbc`` False.
    pbc:
        Length-3 boolean array, periodicity along each lattice vector.
    species:
        ``(N,)`` integer type indices.  Defaults to all zeros.
    symbols:
        Optional tuple mapping type index to chemical symbol, e.g.
        ``("Ar",)``.  Used for masses and for pretty printing.
    masses:
        ``(N,)`` masses in amu.  If omitted they are looked up from
        ``symbols`` via :data:`atomlab.units.ATOMIC_MASSES`.
    energy, forces, virial:
        Optional reference labels.  ``energy`` in eV (total, not per atom),
        ``forces`` ``(N, 3)`` in eV/A, ``virial`` ``(3, 3)`` in eV.
    info:
        Free-form metadata dictionary.  Used to record provenance such as the
        temperature a snapshot was drawn at.
    """

    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: np.ndarray | bool = True
    species: np.ndarray | None = None
    symbols: tuple[str, ...] = ()
    masses: np.ndarray | None = None

    energy: float | None = None
    forces: np.ndarray | None = None
    virial: np.ndarray | None = None

    info: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        self.positions = _as_f64(self.positions, "positions")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        n = self.positions.shape[0]

        if self.cell is None:
            self.cell = np.zeros((3, 3))
        self.cell = _as_f64(self.cell, "cell", (3, 3))

        if isinstance(self.pbc, (bool, np.bool_)):
            self.pbc = np.array([bool(self.pbc)] * 3)
        self.pbc = np.ascontiguousarray(np.asarray(self.pbc, dtype=bool))
        if self.pbc.shape != (3,):
            raise ValueError(f"pbc must be (3,), got {self.pbc.shape}")
        if self.pbc.any() and abs(np.linalg.det(self.cell)) < 1e-12:
            raise ValueError("periodic configuration requires a non-degenerate cell")

        if self.species is None:
            self.species = np.zeros(n, dtype=np.int32)
        self.species = np.ascontiguousarray(np.asarray(self.species, dtype=np.int32))
        if self.species.shape != (n,):
            raise ValueError(f"species must be ({n},), got {self.species.shape}")

        self.symbols = tuple(self.symbols)
        if self.symbols and int(self.species.max(initial=0)) >= len(self.symbols):
            raise ValueError("species index exceeds the number of symbols provided")

        if self.masses is None:
            if self.symbols:
                lut = np.array([ATOMIC_MASSES[s] for s in self.symbols])
                self.masses = lut[self.species]
            else:
                self.masses = np.ones(n)
        self.masses = _as_f64(self.masses, "masses", (n,))
        if np.any(self.masses <= 0.0):
            raise ValueError("all masses must be strictly positive")

        if self.forces is not None:
            self.forces = _as_f64(self.forces, "forces", (n, 3))
        if self.virial is not None:
            self.virial = _as_f64(self.virial, "virial", (3, 3))
        if self.energy is not None:
            self.energy = float(self.energy)

    # -- basic properties --------------------------------------------------

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self.positions.shape[0]

    def __len__(self) -> int:
        return self.n_atoms

    @property
    def volume(self) -> float:
        """Cell volume in A^3 (``inf`` for a fully open system)."""
        if not self.pbc.any():
            return float("inf")
        return float(abs(np.linalg.det(self.cell)))

    @property
    def density(self) -> float:
        """Number density in atoms / A^3."""
        return self.n_atoms / self.volume

    @property
    def n_types(self) -> int:
        """Number of distinct species indices present."""
        return int(self.species.max()) + 1 if self.n_atoms else 0

    # -- geometry ----------------------------------------------------------

    def scaled_positions(self) -> np.ndarray:
        """Fractional coordinates ``s`` with ``r = s @ cell``."""
        return np.linalg.solve(self.cell.T, self.positions.T).T

    def wrapped(self, eps: float = 1e-12) -> "Configuration":
        """Return a copy with all atoms wrapped into the primary cell.

        Only periodic directions are wrapped.  ``eps`` nudges coordinates off
        the exact boundary so that repeated wrapping is idempotent.
        """
        if not self.pbc.any():
            return self.copy()
        s = self.scaled_positions()
        for k in range(3):
            if self.pbc[k]:
                s[:, k] = np.mod(s[:, k] + eps, 1.0)
        return replace(self.copy(), positions=s @ self.cell)

    def strained(self, strain: np.ndarray) -> "Configuration":
        """Apply an affine strain ``r -> (1 + eps) r`` to atoms and cell.

        Parameters
        ----------
        strain:
            ``(3, 3)`` strain tensor ``eps``.  It is used as given (no
            symmetrisation), so antisymmetric parts produce rotations.

        Notes
        -----
        The returned configuration has its reference labels stripped, because
        they no longer correspond to the geometry.
        """
        eps = _as_f64(strain, "strain", (3, 3))
        defm = np.eye(3) + eps
        out = self.copy()
        out.positions = self.positions @ defm.T
        out.cell = self.cell @ defm.T
        out.energy = None
        out.forces = None
        out.virial = None
        return out

    def rotated(self, rotation: np.ndarray) -> "Configuration":
        """Rigidly rotate positions, cell, and any labelled forces/virial."""
        R = _as_f64(rotation, "rotation", (3, 3))
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthogonal")
        out = self.copy()
        out.positions = self.positions @ R.T
        out.cell = self.cell @ R.T
        if out.forces is not None:
            out.forces = out.forces @ R.T
        if out.virial is not None:
            out.virial = R @ out.virial @ R.T
        return out

    def translated(self, shift: np.ndarray) -> "Configuration":
        """Rigidly translate all atoms by ``shift`` (labels are preserved)."""
        out = self.copy()
        out.positions = self.positions + np.asarray(shift, dtype=np.float64)
        return out

    def repeated(self, reps: Sequence[int]) -> "Configuration":
        """Return a supercell replicated ``reps = (nx, ny, nz)`` times.

        Reference labels are dropped: energy and virial would need scaling and
        forces would need tiling, which is only correct for perfect crystals.
        """
        nx, ny, nz = (int(r) for r in reps)
        if min(nx, ny, nz) < 1:
            raise ValueError("repetitions must be >= 1")
        offsets = np.array(
            [
                i * self.cell[0] + j * self.cell[1] + k * self.cell[2]
                for i in range(nx)
                for j in range(ny)
                for k in range(nz)
            ]
        )
        pos = (self.positions[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        species = np.tile(self.species, len(offsets))
        masses = np.tile(self.masses, len(offsets))
        cell = self.cell * np.array([[nx], [ny], [nz]], dtype=float)
        return Configuration(
            positions=pos,
            cell=cell,
            pbc=self.pbc.copy(),
            species=species,
            symbols=self.symbols,
            masses=masses,
            info=dict(self.info),
        )

    # -- labels ------------------------------------------------------------

    @property
    def has_labels(self) -> bool:
        """True if energy and forces are both present."""
        return self.energy is not None and self.forces is not None

    def with_labels(self, result: "Result") -> "Configuration":
        """Return a copy carrying the energy/forces/virial from ``result``."""
        out = self.copy()
        out.energy = result.energy
        out.forces = result.forces.copy()
        out.virial = None if result.virial is None else result.virial.copy()
        return out

    def stripped(self) -> "Configuration":
        """Return a copy with all reference labels removed."""
        out = self.copy()
        out.energy = None
        out.forces = None
        out.virial = None
        return out

    # -- misc --------------------------------------------------------------

    def copy(self) -> "Configuration":
        """Deep copy (all arrays are copied)."""
        return Configuration(
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc.copy(),
            species=self.species.copy(),
            symbols=self.symbols,
            masses=self.masses.copy(),
            energy=self.energy,
            forces=None if self.forces is None else self.forces.copy(),
            virial=None if self.virial is None else self.virial.copy(),
            info=dict(self.info),
        )

    def fingerprint(self) -> str:
        """Stable content hash, used to detect accidental dataset reuse."""
        h = hashlib.sha256()
        for arr in (self.positions, self.cell, self.species.astype(np.float64)):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lab = "labelled" if self.has_labels else "unlabelled"
        sym = "/".join(self.symbols) if self.symbols else "?"
        return f"Configuration({self.n_atoms} atoms, {sym}, V={self.volume:.2f} A^3, {lab})"


@dataclass
class Result:
    """What a :class:`~atomlab.potentials.base.Potential` returns.

    Attributes
    ----------
    energy:
        Total potential energy in eV.
    forces:
        ``(N, 3)`` forces in eV/A, i.e. ``-dU/dr``.
    virial:
        Optional ``(3, 3)`` virial tensor in eV (see :class:`Configuration`).
    energies:
        Optional ``(N,)`` decomposition of the total energy into per-atom
        contributions.  Only some potentials define one; it is never unique.
    extra:
        Free-form dictionary for model-specific diagnostics, e.g. per-atom
        uncertainty from an ensemble.
    """

    energy: float
    forces: np.ndarray
    virial: np.ndarray | None = None
    energies: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.energy = float(self.energy)
        self.forces = _as_f64(self.forces, "forces")
        if self.forces.ndim != 2 or self.forces.shape[1] != 3:
            raise ValueError(f"forces must be (N, 3), got {self.forces.shape}")
        if self.virial is not None:
            self.virial = _as_f64(self.virial, "virial", (3, 3))
        if self.energies is not None:
            self.energies = _as_f64(self.energies, "energies", (self.forces.shape[0],))

    @property
    def n_atoms(self) -> int:
        return self.forces.shape[0]

    def pressure(self, volume: float) -> float:
        """Potential (virial-only) contribution to the pressure, in eV/A^3."""
        if self.virial is None:
            raise ValueError("result carries no virial")
        return float(np.trace(self.virial) / (3.0 * volume))


@dataclass
class Trajectory:
    """A time series produced by molecular dynamics.

    Positions are stored **unwrapped** so that mean squared displacements are
    meaningful without post-hoc unwrapping.

    Attributes
    ----------
    positions:
        ``(T, N, 3)`` unwrapped coordinates in A.
    velocities:
        Optional ``(T, N, 3)`` velocities in A/ps.
    cells:
        ``(T, 3, 3)`` cell matrices (constant for NVE/NVT runs).
    times:
        ``(T,)`` simulation times in ps.
    template:
        A :class:`Configuration` carrying species, symbols, masses and pbc.
    scalars:
        Dictionary of ``(T,)`` arrays: ``"potential_energy"``,
        ``"kinetic_energy"``, ``"total_energy"``, ``"temperature"``,
        ``"pressure"`` and anything else the driver logged.
    """

    positions: np.ndarray
    cells: np.ndarray
    times: np.ndarray
    template: Configuration
    velocities: np.ndarray | None = None
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = _as_f64(self.positions, "positions")
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be (T, N, 3), got {self.positions.shape}")
        n_frames = self.positions.shape[0]
        self.cells = _as_f64(self.cells, "cells", (n_frames, 3, 3))
        self.times = _as_f64(self.times, "times", (n_frames,))
        if self.velocities is not None:
            self.velocities = _as_f64(self.velocities, "velocities", self.positions.shape)

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[1]

    @property
    def dt(self) -> float:
        """Spacing between stored frames in ps (assumes uniform sampling)."""
        if self.n_frames < 2:
            raise ValueError("need at least two frames to define a frame spacing")
        return float(self.times[1] - self.times[0])

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, index) -> "Configuration | Trajectory":
        if isinstance(index, (int, np.integer)):
            return self.frame(int(index))
        return Trajectory(
            positions=self.positions[index],
            cells=self.cells[index],
            times=self.times[index],
            template=self.template,
            velocities=None if self.velocities is None else self.velocities[index],
            scalars={k: v[index] for k, v in self.scalars.items()},
            info=dict(self.info),
        )

    def frame(self, i: int) -> Configuration:
        """Return frame ``i`` as a :class:`Configuration`."""
        cfg = self.template.copy()
        cfg.positions = self.positions[i].copy()
        cfg.cell = self.cells[i].copy()
        cfg.info = dict(self.template.info)
        cfg.info["time"] = float(self.times[i])
        cfg.info["frame"] = int(i)
        return cfg

    def __iter__(self) -> Iterator[Configuration]:
        for i in range(self.n_frames):
            yield self.frame(i)

    def subsample(self, stride: int = 1, start: int = 0, stop: int | None = None) -> "Trajectory":
        """Return every ``stride``-th frame in ``[start, stop)``."""
        return self[slice(start, stop, stride)]


@dataclass
class Dataset:
    """A labelled collection of configurations used for fitting and testing."""

    configurations: list[Configuration] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.configurations)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self.configurations[int(index)]
        return Dataset(configurations=list(self.configurations[index]), info=dict(self.info))

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self.configurations)

    def append(self, configuration: Configuration) -> None:
        self.configurations.append(configuration)

    def extend(self, configurations: Iterable[Configuration]) -> None:
        self.configurations.extend(configurations)

    @property
    def n_atoms_total(self) -> int:
        return sum(c.n_atoms for c in self.configurations)

    def split(self, fractions: Sequence[float], seed: int = 0) -> list["Dataset"]:
        """Randomly partition into datasets with the requested fractions."""
        fr = np.asarray(fractions, dtype=float)
        if np.any(fr < 0) or not np.isclose(fr.sum(), 1.0):
            raise ValueError("fractions must be non-negative and sum to 1")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self))
        edges = np.concatenate([[0], np.cumsum(np.round(fr * len(self)).astype(int))])
        edges[-1] = len(self)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append(Dataset([self.configurations[i] for i in order[lo:hi]], dict(self.info)))
        return out

    def energies(self) -> np.ndarray:
        """``(M,)`` total energies; raises if any configuration is unlabelled."""
        return np.array([self._require(c).energy for c in self.configurations])

    def energies_per_atom(self) -> np.ndarray:
        return np.array([self._require(c).energy / c.n_atoms for c in self.configurations])

    def forces(self) -> np.ndarray:
        """``(sum_i N_i, 3)`` stacked forces over the whole dataset."""
        return np.concatenate([self._require(c).forces for c in self.configurations], axis=0)

    @staticmethod
    def _require(cfg: Configuration) -> Configuration:
        if not cfg.has_labels:
            raise ValueError("dataset contains unlabelled configurations")
        return cfg
