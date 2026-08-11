"""Neighbour lists: cell-linked-list construction, half/full lists, Verlet skin.

Every potential in this package consumes a :class:`NeighborList`, so the two
conventions fixed here propagate everywhere:

*Shift.*  A pair is the triple ``(i, j, shift)`` and means "atom ``i`` in the
home cell interacting with atom ``j`` displaced by ``shift @ cell``".  Shifts
are integers in lattice units and are relative to the **original** positions
handed in, not to any internally wrapped copy, so
``pair_vectors`` is meaningful even for the unwrapped coordinates that
molecular dynamics carries around.

*Direction.*  ``D[p] = r_j + shift @ cell - r_i`` points **from i to j**.  A
pair force is then applied as ``+f`` on ``j`` and ``-f`` on ``i`` for a
repulsive interaction with ``f`` along ``D``; getting this backwards flips
every force in the package, so it is asserted in the tests.

Small cells are a first-class case, not an error case.  When the cell is
smaller than ``2 * cutoff`` an atom sees a given neighbour through several
periodic images at once, and it sees *itself* through its own images.  Both
appear here as ordinary pairs distinguished only by their ``shift``; the
``i == j`` self-image pair is real physics (it is exactly the term that makes
the energy of a one-atom periodic cell finite and nonzero) and is included.
The minimum-image convention is never invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .cell import _pinv_basis, cell_widths, wrap_positions
from .types import Configuration

__all__ = [
    "NeighborList",
    "build_neighbor_list",
    "build_neighbor_list_naive",
    "pair_vectors",
    "VerletList",
]


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------


@dataclass
class NeighborList:
    """A list of interacting pairs, in the fixed package-wide format.

    Attributes
    ----------
    i : ndarray, shape (P,), int32
        Central atom index of each pair.
    j : ndarray, shape (P,), int32
        Neighbour atom index.  May equal ``i`` when an atom interacts with its
        own periodic image.
    shift : ndarray, shape (P, 3), int32
        Periodic image of ``j``, in lattice units.  The cartesian offset is
        ``shift @ cell`` (angstrom).  Always zero along non-periodic
        directions.
    n_atoms : int
        Number of atoms in the configuration the list was built from.
    cutoff : float
        Radius in angstrom within which **all** pairs are guaranteed present.
        For a list built with a Verlet skin this is ``cutoff + skin``, i.e. the
        radius actually searched; the physically meaningful interaction range
        is ``cutoff - skin``.
    half : bool
        If True each unordered pair appears exactly once; if False both
        ``(i, j, s)`` and ``(j, i, -s)`` are present.
    skin : float
        Verlet skin included in ``cutoff``, in angstrom.  Zero for a plain
        list.
    """

    i: np.ndarray
    j: np.ndarray
    shift: np.ndarray
    n_atoms: int
    cutoff: float
    half: bool
    skin: float = 0.0

    def __post_init__(self) -> None:
        self.i = np.ascontiguousarray(np.asarray(self.i, dtype=np.int32)).reshape(-1)
        self.j = np.ascontiguousarray(np.asarray(self.j, dtype=np.int32)).reshape(-1)
        self.shift = np.ascontiguousarray(np.asarray(self.shift, dtype=np.int32)).reshape(-1, 3)
        if self.i.shape != self.j.shape or self.shift.shape[0] != self.i.shape[0]:
            raise ValueError(
                f"inconsistent pair arrays: i {self.i.shape}, j {self.j.shape}, "
                f"shift {self.shift.shape}"
            )
        self.n_atoms = int(self.n_atoms)
        self.cutoff = float(self.cutoff)
        self.half = bool(self.half)
        self.skin = float(self.skin)

    @property
    def n_pairs(self) -> int:
        """Number of pairs ``P`` in the list."""
        return int(self.i.shape[0])

    def __len__(self) -> int:
        return self.n_pairs

    @property
    def interaction_cutoff(self) -> float:
        """The physical cutoff requested by the caller, ``cutoff - skin`` (A)."""
        return self.cutoff - self.skin

    def counts(self) -> np.ndarray:
        """``(n_atoms,)`` number of pairs in which each atom appears as ``i``."""
        return np.bincount(self.i, minlength=self.n_atoms)

    def as_set(self) -> set[tuple[int, int, int, int, int]]:
        """Return the pairs as a ``set`` of ``(i, j, sx, sy, sz)`` tuples.

        Used by the tests to compare implementations independently of pair
        ordering.
        """
        return {
            (int(a), int(b), int(s[0]), int(s[1]), int(s[2]))
            for a, b, s in zip(self.i, self.j, self.shift)
        }

    def to_half(self) -> "NeighborList":
        """Return the half-list version of a full list (see :func:`_half_mask`)."""
        if self.half:
            return self
        keep = _half_mask(self.i, self.j, self.shift)
        return NeighborList(
            i=self.i[keep],
            j=self.j[keep],
            shift=self.shift[keep],
            n_atoms=self.n_atoms,
            cutoff=self.cutoff,
            half=True,
            skin=self.skin,
        )

    def to_full(self) -> "NeighborList":
        """Return the full list obtained by adding the reversed pairs."""
        if not self.half:
            return self
        i = np.concatenate([self.i, self.j])
        j = np.concatenate([self.j, self.i])
        shift = np.concatenate([self.shift, -self.shift])
        return _sorted_list(i, j, shift, self.n_atoms, self.cutoff, False, self.skin)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "half" if self.half else "full"
        return (
            f"NeighborList({self.n_pairs} {kind} pairs, {self.n_atoms} atoms, "
            f"cutoff={self.cutoff:.3f} A)"
        )


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------


def _half_mask(i: np.ndarray, j: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Boolean mask selecting one representative of each unordered pair.

    The full list contains ``(i, j, s)`` and ``(j, i, -s)`` for the same
    physical interaction.  The tie-break is:

    * ``i < j`` -- keeps exactly one of the two for distinct atoms;
    * ``i == j`` -- an atom with its own image, where the partner is the *same*
      atom, so the ordering must come from the shift instead.  We keep the
      lexicographically positive shift, i.e. the one whose first nonzero
      component is positive.  ``s`` and ``-s`` differ in that component's sign,
      so exactly one of the two survives.

    ``i > j`` is always dropped.  ``i == j`` with ``s == 0`` cannot occur (it is
    excluded at construction, being an atom with itself).
    """
    first = np.zeros(i.shape[0], dtype=np.int32)
    nonzero = shift != 0
    # Index of the first nonzero shift component, then its sign.
    idx = np.argmax(nonzero, axis=1)
    any_nz = nonzero.any(axis=1)
    first[any_nz] = shift[np.arange(shift.shape[0])[any_nz], idx[any_nz]]
    return (i < j) | ((i == j) & (first > 0))


def _sorted_list(
    i: np.ndarray,
    j: np.ndarray,
    shift: np.ndarray,
    n_atoms: int,
    cutoff: float,
    half: bool,
    skin: float,
) -> NeighborList:
    """Build a :class:`NeighborList` with pairs in a canonical order.

    Sorting by ``(i, j, sx, sy, sz)`` makes the output independent of the
    traversal order of whichever algorithm produced it, so the jitted kernel
    and the reference implementation return *identical arrays*, not merely
    equal sets, and so downstream reductions are bitwise reproducible.
    """
    i = np.asarray(i, dtype=np.int32).reshape(-1)
    j = np.asarray(j, dtype=np.int32).reshape(-1)
    shift = np.asarray(shift, dtype=np.int32).reshape(-1, 3)
    if i.size:
        order = np.lexsort((shift[:, 2], shift[:, 1], shift[:, 0], j, i))
        i, j, shift = i[order], j[order], shift[order]
    return NeighborList(
        i=i, j=j, shift=shift, n_atoms=n_atoms, cutoff=cutoff, half=half, skin=skin
    )


def _image_range(cell: np.ndarray, pbc: np.ndarray, r_search: float) -> np.ndarray:
    """Number of periodic images to search along each lattice direction.

    Both atoms of a pair are wrapped into the primary cell, so their fractional
    separation along direction ``k`` lies in ``[n_k - 1, n_k + 1]`` for image
    ``n``.  The cartesian distance is at least ``|f_k| * w_k`` with ``w_k`` the
    perpendicular width, hence a pair inside ``r_search`` requires
    ``|n_k| - 1 <= r_search / w_k``.

    Returns
    -------
    ndarray, shape (3,), int
        Half-width of the image box; zero along non-periodic directions.
    """
    widths = cell_widths(cell, pbc)
    out = np.zeros(3, dtype=np.int64)
    for k in range(3):
        if not pbc[k]:
            continue
        if not np.isfinite(widths[k]) or widths[k] <= 0.0:
            raise ValueError(f"lattice vector {k} is degenerate but marked periodic")
        out[k] = int(np.floor(r_search / widths[k])) + 1
    return out


def _prepare(configuration: Configuration, cutoff: float, skin: float):
    """Validate inputs and wrap atoms; shared by both builders."""
    cutoff = float(cutoff)
    skin = float(skin)
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be > 0 A, got {cutoff}")
    if skin < 0.0:
        raise ValueError(f"skin must be >= 0 A, got {skin}")
    if not np.isfinite(cutoff):
        raise ValueError("cannot build a neighbour list for an infinite cutoff")
    cell = np.ascontiguousarray(configuration.cell, dtype=np.float64)
    pbc = np.asarray(configuration.pbc, dtype=bool)
    r_search = cutoff + skin
    # Wrap so that the image-range bound above applies; the returned integer
    # shift lets us report images relative to the caller's own positions.
    pos, wrap_shift = wrap_positions(configuration.positions, cell, pbc, return_shift=True)
    return pos, wrap_shift.astype(np.int64), cell, pbc, cutoff, skin, r_search


# --------------------------------------------------------------------------
# reference implementation
# --------------------------------------------------------------------------


def build_neighbor_list_naive(
    configuration: Configuration,
    cutoff: float,
    *,
    half: bool = False,
    skin: float = 0.0,
) -> NeighborList:
    """Pure-NumPy O(N^2) reference neighbour list over all relevant images.

    Deliberately simple: enumerate every periodic image that could possibly
    contribute, form the full ``N x N`` displacement matrix for each, and keep
    what is inside the cutoff.  Slow, but it has no binning logic to get wrong,
    so it is the arbiter for :func:`build_neighbor_list` in the test suite.  A
    fast wrong kernel is the most expensive kind of bug in this package.

    Parameters
    ----------
    configuration : Configuration
        Geometry; ``positions`` in angstrom, ``cell`` rows as lattice vectors.
    cutoff : float
        Interaction range in angstrom.
    half : bool
        Return a half list (each unordered pair once) instead of a full one.
    skin : float
        Extra radius in angstrom, searched on top of ``cutoff``.

    Returns
    -------
    NeighborList
        Pairs with ``|D| <= cutoff + skin``, excluding the trivial self pair
        ``(i, i, 0)`` but including genuine self-image pairs.
    """
    pos, wrap_shift, cell, pbc, cutoff, skin, r_search = _prepare(configuration, cutoff, skin)
    n = pos.shape[0]
    if n == 0:
        return _sorted_list(
            np.empty(0), np.empty(0), np.empty((0, 3)), 0, r_search, half, skin
        )

    nmax = _image_range(cell, pbc, r_search)
    images = np.array(
        np.meshgrid(
            *[np.arange(-nmax[k], nmax[k] + 1) for k in range(3)],
            indexing="ij",
        )
    ).reshape(3, -1).T.astype(np.int64)

    r2max = r_search * r_search
    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    for image in images:
        offset = image.astype(np.float64) @ cell
        d = (pos[None, :, :] + offset) - pos[:, None, :]  # (N, N, 3), from i to j
        r2 = np.einsum("ijk,ijk->ij", d, d)
        mask = r2 <= r2max
        if not image.any():
            np.fill_diagonal(mask, False)  # an atom is not its own neighbour
        ii, jj = np.nonzero(mask)
        if ii.size == 0:
            continue
        out_i.append(ii)
        out_j.append(jj)
        # Report the image relative to the caller's unwrapped positions.
        out_s.append(wrap_shift[jj] + image[None, :] - wrap_shift[ii])

    if not out_i:
        nl = _sorted_list(np.empty(0), np.empty(0), np.empty((0, 3)), n, r_search, False, skin)
    else:
        nl = _sorted_list(
            np.concatenate(out_i),
            np.concatenate(out_j),
            np.concatenate(out_s, axis=0),
            n,
            r_search,
            False,
            skin,
        )
    return nl.to_half() if half else nl


# --------------------------------------------------------------------------
# cell-linked-list implementation
# --------------------------------------------------------------------------


@njit(cache=True)
def _scan_bins(
    pos,
    bin_x,
    bin_y,
    bin_z,
    ghost_pos,
    ghost_atom,
    ghost_home,
    bin_start,
    bin_count,
    bin_order,
    nbx,
    nby,
    nbz,
    r2max,
    out_i,
    out_g,
    count_only,
):
    """Inner loop: for each real atom, test the ghosts in the 27 nearby bins.

    Bins are at least ``r_search`` wide in every cartesian direction, so a
    3x3x3 stencil around an atom's own bin provably contains every ghost
    within the search radius.  Run twice: once with ``count_only`` to size the
    output, once to fill it.  Two cheap passes beat one pass with a guessed
    capacity, because a resize-and-retry would make the pair ordering depend on
    the guess.
    """
    n = pos.shape[0]
    p = 0
    for i in range(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        bx = bin_x[i]
        by = bin_y[i]
        bz = bin_z[i]
        for dx in range(-1, 2):
            cx = bx + dx
            if cx < 0 or cx >= nbx:
                continue
            for dy in range(-1, 2):
                cy = by + dy
                if cy < 0 or cy >= nby:
                    continue
                for dz in range(-1, 2):
                    cz = bz + dz
                    if cz < 0 or cz >= nbz:
                        continue
                    b = (cx * nby + cy) * nbz + cz
                    start = bin_start[b]
                    for t in range(bin_count[b]):
                        g = bin_order[start + t]
                        # Skip only the atom paired with itself in the home
                        # image; self-image pairs (ghost_home == 0) are real.
                        if ghost_home[g] == 1 and ghost_atom[g] == i:
                            continue
                        ddx = ghost_pos[g, 0] - xi
                        ddy = ghost_pos[g, 1] - yi
                        ddz = ghost_pos[g, 2] - zi
                        r2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if r2 <= r2max:
                            if not count_only:
                                out_i[p] = i
                                out_g[p] = g
                            p += 1
    return p


def _build_ghosts(pos, wrap_shift, cell, pbc, r_search):
    """Replicate the wrapped atoms into every image that can reach the cell.

    Returns ``(ghost_pos, ghost_atom, ghost_shift, ghost_home)`` where
    ``ghost_shift`` is already expressed relative to the caller's original
    positions (it folds in the wrap), and ``ghost_home`` flags the ``n == 0``
    copies.
    """
    n = pos.shape[0]
    nmax = _image_range(cell, pbc, r_search)
    dims = np.flatnonzero(pbc)

    images = np.array(
        np.meshgrid(
            *[np.arange(-nmax[k], nmax[k] + 1) for k in range(3)],
            indexing="ij",
        )
    ).reshape(3, -1).T.astype(np.int64)

    # Fractional coordinates along the periodic directions only; used to drop
    # images that provably cannot come within r_search of the primary cell.
    if dims.size:
        frac = pos @ _pinv_basis(cell[dims])  # (N, d), in [0, 1)
        widths = cell_widths(cell, pbc)[dims]

    pos_list, atom_list, shift_list, home_list = [], [], [], []
    for image in images:
        offset = image.astype(np.float64) @ cell
        if image.any() and dims.size:
            # Distance from this ghost to the [0,1] slab in each periodic
            # direction, converted to a length.  Conservative (it ignores the
            # other directions), so it never discards a needed ghost.
            f = frac + image[dims][None, :]
            gap = np.maximum(np.maximum(f - 1.0, -f), 0.0) * widths[None, :]
            keep = np.all(gap <= r_search, axis=1)
            if not keep.any():
                continue
        else:
            keep = np.ones(n, dtype=bool)
        idx = np.flatnonzero(keep)
        pos_list.append(pos[idx] + offset)
        atom_list.append(idx.astype(np.int32))
        shift_list.append(wrap_shift[idx] + image[None, :])
        home_list.append(np.full(idx.size, 1 if not image.any() else 0, dtype=np.int8))

    return (
        np.ascontiguousarray(np.concatenate(pos_list, axis=0)),
        np.ascontiguousarray(np.concatenate(atom_list)),
        np.ascontiguousarray(np.concatenate(shift_list, axis=0)),
        np.ascontiguousarray(np.concatenate(home_list)),
    )


def _bin_atoms(ghost_pos, pos, r_search):
    """Assign ghosts and real atoms to a uniform cartesian grid.

    The grid spans the bounding box of all ghosts (which contains the real
    atoms, since the home image is among them).  Bin edges are at least
    ``r_search`` so the 3x3x3 stencil in :func:`_scan_bins` is sufficient.
    """
    lo = ghost_pos.min(axis=0)
    hi = ghost_pos.max(axis=0)
    extent = np.maximum(hi - lo, 1e-12)
    nbins = np.maximum(np.floor(extent / r_search).astype(np.int64), 1)
    # A very dilute or slab-like system could ask for far more bins than there
    # are atoms to put in them; coarsen uniformly rather than allocating (and
    # walking) a grid that is mostly empty.  Correctness is unaffected: bins
    # only ever get wider than r_search.
    max_bins = max(64, 8 * ghost_pos.shape[0])
    while int(np.prod(nbins)) > max_bins:
        nbins = np.maximum(nbins // 2, 1)
        if np.all(nbins == 1):
            break
    width = extent / nbins

    def index(p):
        k = np.floor((p - lo) / width).astype(np.int64)
        return np.clip(k, 0, nbins - 1)

    gidx = index(ghost_pos)
    aidx = index(pos)
    flat = (gidx[:, 0] * nbins[1] + gidx[:, 1]) * nbins[2] + gidx[:, 2]
    total = int(np.prod(nbins))
    counts = np.bincount(flat, minlength=total)
    start = np.zeros(total, dtype=np.int64)
    np.cumsum(counts[:-1], out=start[1:])
    order = np.argsort(flat, kind="stable").astype(np.int64)
    return aidx, start, counts.astype(np.int64), order, nbins


def build_neighbor_list(
    configuration: Configuration,
    cutoff: float,
    *,
    half: bool = False,
    skin: float = 0.0,
) -> NeighborList:
    """Build a neighbour list with a cell-linked-list algorithm.

    Atoms are wrapped into the primary cell and replicated into every periodic
    image that can reach it; the replicas are binned on a uniform cartesian
    grid of edge >= ``cutoff + skin`` and each real atom is tested against the
    27 surrounding bins by a ``numba`` kernel.  Working with explicit image
    replicas (rather than applying the minimum image convention) is what makes
    small cells, triclinic cells and mixed periodicity all the same code path:
    an atom that sees a neighbour through three different images simply finds
    three ghosts.

    Parameters
    ----------
    configuration : Configuration
        Geometry; positions in angstrom, ``cell`` rows as lattice vectors,
        ``pbc`` per lattice vector.
    cutoff : float
        Interaction range in angstrom.  Must be finite and positive.
    half : bool
        If True each unordered pair is returned once, with the tie-break
        documented in :func:`_half_mask`.
    skin : float
        Verlet skin in angstrom.  Pairs out to ``cutoff + skin`` are listed, so
        the list stays valid while atoms move by up to ``skin / 2`` each.

    Returns
    -------
    NeighborList
        Canonically ordered by ``(i, j, shift)``, identical array-for-array to
        :func:`build_neighbor_list_naive` on the same input.

    Notes
    -----
    No check is made that the cutoff fits inside the cell; that restriction
    belongs to potentials that assume a single image
    (:func:`atomlab.cell.check_minimum_image`), not to the neighbour list,
    which is correct for arbitrarily small cells by construction.
    """
    pos, wrap_shift, cell, pbc, cutoff, skin, r_search = _prepare(configuration, cutoff, skin)
    n = pos.shape[0]
    if n == 0:
        return _sorted_list(
            np.empty(0), np.empty(0), np.empty((0, 3)), 0, r_search, half, skin
        )

    ghost_pos, ghost_atom, ghost_shift, ghost_home = _build_ghosts(
        pos, wrap_shift, cell, pbc, r_search
    )
    aidx, bin_start, bin_count, bin_order, nbins = _bin_atoms(ghost_pos, pos, r_search)

    args = (
        pos,
        np.ascontiguousarray(aidx[:, 0]),
        np.ascontiguousarray(aidx[:, 1]),
        np.ascontiguousarray(aidx[:, 2]),
        ghost_pos,
        ghost_atom,
        ghost_home,
        bin_start,
        bin_count,
        bin_order,
        int(nbins[0]),
        int(nbins[1]),
        int(nbins[2]),
        r_search * r_search,
    )
    empty_i = np.empty(0, dtype=np.int64)
    n_pairs = _scan_bins(*args, empty_i, empty_i, True)

    out_i = np.empty(n_pairs, dtype=np.int64)
    out_g = np.empty(n_pairs, dtype=np.int64)
    _scan_bins(*args, out_i, out_g, False)

    nl = _sorted_list(
        out_i.astype(np.int32),
        ghost_atom[out_g],
        ghost_shift[out_g] - wrap_shift[out_i],
        n,
        r_search,
        False,
        skin,
    )
    return nl.to_half() if half else nl


# --------------------------------------------------------------------------
# pair geometry
# --------------------------------------------------------------------------


def pair_vectors(
    configuration: Configuration, nl: NeighborList
) -> tuple[np.ndarray, np.ndarray]:
    """Displacement vectors and distances for every pair in ``nl``.

    Parameters
    ----------
    configuration : Configuration
        Must be the geometry the list describes (same atom count).  Positions
        need not be the exact ones used at build time -- that is the point of a
        Verlet skin -- but the atom indexing must match.
    nl : NeighborList

    Returns
    -------
    D : ndarray, shape (P, 3)
        ``D[p] = r_j + shift @ cell - r_i``, in angstrom, pointing **from i to
        j**.
    r : ndarray, shape (P,)
        ``|D[p]|`` in angstrom.
    """
    if configuration.n_atoms != nl.n_atoms:
        raise ValueError(
            f"neighbour list was built for {nl.n_atoms} atoms but the configuration "
            f"has {configuration.n_atoms}"
        )
    pos = configuration.positions
    if nl.n_pairs == 0:
        return np.zeros((0, 3)), np.zeros(0)
    offsets = nl.shift.astype(np.float64) @ configuration.cell
    d = pos[nl.j] + offsets - pos[nl.i]
    return d, np.linalg.norm(d, axis=1)


# --------------------------------------------------------------------------
# Verlet wrapper
# --------------------------------------------------------------------------


@dataclass
class VerletList:
    """A neighbour list with a skin, rebuilt only when the atoms have moved.

    The list is built out to ``cutoff + skin``.  If no atom has moved further
    than ``skin / 2`` since the build, then no pair separation can have changed
    by more than ``skin`` (the triangle inequality, with both partners moving),
    so every pair now inside ``cutoff`` was inside ``cutoff + skin`` at build
    time and is therefore already in the list.  That bound is the whole
    justification for the ``skin / 2`` criterion, and it is why the criterion
    uses the max over atoms rather than an rms.

    Displacements are measured on the **unwrapped** positions, which is what
    :class:`atomlab.md.state.MDState` carries.  Feeding wrapped coordinates
    would make an atom crossing a cell face look like a huge displacement and
    trigger harmless but constant rebuilds.

    Parameters
    ----------
    cutoff : float
        Physical interaction range in angstrom.
    skin : float
        Verlet skin in angstrom.  ~0.3-1.0 A is usual for condensed phases;
        larger skins mean fewer rebuilds and more pairs per step.
    half : bool
        Build half lists.
    """

    cutoff: float
    skin: float = 0.5
    half: bool = False

    nl: NeighborList | None = field(default=None, init=False, repr=False)
    n_builds: int = field(default=0, init=False)
    _positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _cell: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cutoff = float(self.cutoff)
        self.skin = float(self.skin)
        if self.skin < 0.0:
            raise ValueError(f"skin must be >= 0 A, got {self.skin}")
        self.half = bool(self.half)

    def build(self, configuration: Configuration) -> NeighborList:
        """Force a rebuild against ``configuration`` and return the new list."""
        self.nl = build_neighbor_list(
            configuration, self.cutoff, half=self.half, skin=self.skin
        )
        self._positions = configuration.positions.copy()
        self._cell = np.array(configuration.cell, dtype=np.float64, copy=True)
        self.n_builds += 1
        return self.nl

    def max_displacement(self, positions) -> float:
        """Largest distance any atom has moved since the last build, in A."""
        if self._positions is None:
            return float("inf")
        p = np.asarray(positions, dtype=np.float64)
        if p.shape != self._positions.shape:
            return float("inf")
        return float(np.sqrt(np.max(np.sum((p - self._positions) ** 2, axis=1))))

    def needs_rebuild(self, positions, cell=None) -> bool:
        """True if the cached list can no longer be trusted.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Current **unwrapped** positions in angstrom.
        cell : array_like, shape (3, 3), optional
            Current cell.  Any change (as under a barostat) invalidates the
            list, because the stored shifts are in lattice units and the
            cartesian image offsets move with the cell.

        Returns
        -------
        bool
        """
        if self.nl is None or self._positions is None:
            return True
        if cell is not None and not np.array_equal(
            np.asarray(cell, dtype=np.float64), self._cell
        ):
            return True
        return self.max_displacement(positions) > 0.5 * self.skin

    def update(self, configuration: Configuration) -> NeighborList:
        """Return a valid list for ``configuration``, rebuilding only if needed."""
        if self.needs_rebuild(configuration.positions, configuration.cell):
            return self.build(configuration)
        assert self.nl is not None
        return self.nl
