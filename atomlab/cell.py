"""Simulation-cell geometry: minimum image, wrapping, metrics, strain.

Everything in :mod:`atomlab` that touches periodic boundary conditions goes
through this module, so the conventions here are the conventions everywhere:

*Row vectors.*  ``cell`` is ``(3, 3)`` with the **rows** being the lattice
vectors ``a1, a2, a3`` (angstrom), so cartesian and fractional coordinates are
related by ``r = s @ cell`` -- see :class:`atomlab.types.Configuration`.

*Partial periodicity.*  ``pbc`` is a length-3 boolean array indexed by lattice
vector, not by cartesian axis.  A slab periodic in ``a1`` and ``a2`` has
``pbc = (True, True, False)``; only those two lattice vectors generate images,
and ``a3`` is then allowed to be zero or meaningless.  Every routine here works
off the periodic *sublattice* rather than assuming a full-rank cell, which is
what makes slabs, wires and clusters fall out of the same code path.

*Why there are two minimum-image implementations.*  See
:func:`minimum_image`; the short version is that the textbook
"round the fractional coordinates" recipe is simply wrong for skewed triclinic
cells, and this module implements both so that the wrong one is never reached
by accident.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "volume",
    "reciprocal_cell",
    "is_orthorhombic",
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    "wrap_positions",
    "minimum_image",
    "minimum_image_naive",
    "minimum_image_shift",
    "cell_widths",
    "min_cell_width",
    "check_minimum_image",
    "reduce_cell",
    "cell_from_parameters",
    "cell_to_parameters",
    "apply_strain",
    "voigt_to_full",
    "full_to_voigt",
    "VOIGT_ORDER",
]

#: Voigt index order used throughout the package: ``(xx, yy, zz, yz, xz, xy)``.
#: This matches LAMMPS and ASE, so stress tensors can be compared without
#: reshuffling.
VOIGT_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# --------------------------------------------------------------------------
# small internal helpers
# --------------------------------------------------------------------------


def _as_cell(cell) -> np.ndarray:
    c = np.ascontiguousarray(np.asarray(cell, dtype=np.float64))
    if c.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {c.shape}")
    return c


def _as_pbc(pbc) -> np.ndarray:
    if isinstance(pbc, (bool, np.bool_)):
        return np.array([bool(pbc)] * 3)
    p = np.ascontiguousarray(np.asarray(pbc, dtype=bool))
    if p.shape != (3,):
        raise ValueError(f"pbc must have shape (3,), got {p.shape}")
    return p


def _periodic_basis(cell: np.ndarray, pbc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A_p, dims)``: the periodic lattice vectors and their indices.

    ``A_p`` has shape ``(d, 3)`` with ``d = pbc.sum()``.  Deliberately does not
    look at the non-periodic rows at all, so a slab may carry a zero ``a3``.
    """
    dims = np.flatnonzero(pbc)
    return cell[dims], dims


def _pinv_basis(basis: np.ndarray) -> np.ndarray:
    """Dual basis of a ``(d, 3)`` row-vector basis, as a ``(3, d)`` matrix.

    ``s = r @ dual`` gives the coordinates of the projection of ``r`` onto
    ``span(basis)``, i.e. ``basis[k] @ dual == e_k``.  For ``d = 3`` this is
    exactly ``inv(basis)``.
    """
    if basis.shape[0] == 0:
        return np.zeros((3, 0))
    gram = basis @ basis.T
    if abs(np.linalg.det(gram)) < 1e-20:
        raise ValueError("periodic lattice vectors are linearly dependent")
    return basis.T @ np.linalg.inv(gram)


# --------------------------------------------------------------------------
# cell metrics
# --------------------------------------------------------------------------


def volume(cell) -> float:
    """Cell volume in A^3.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.

    Returns
    -------
    float
        ``|det(cell)|`` in A^3.  Zero for a degenerate cell; the absolute value
        means a left-handed cell is not an error.
    """
    return float(abs(np.linalg.det(_as_cell(cell))))


def reciprocal_cell(cell) -> np.ndarray:
    """Reciprocal lattice with the **2 pi convention**.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Direct lattice vectors as rows, in angstrom.

    Returns
    -------
    ndarray, shape (3, 3)
        Reciprocal lattice vectors ``b1, b2, b3`` as rows, in 1/A, satisfying

        ``a_i . b_j = 2 pi delta_ij``

        i.e. ``B = 2 pi inv(A).T``.  The 2 pi is included because every
        consumer in this package (structure factor ``S(q)``, phonon dynamical
        matrices, Ewald-style sums) writes plane waves as ``exp(i q . r)`` with
        ``q`` a reciprocal lattice vector; the crystallographic convention
        without the 2 pi would put stray factors in all of them.
    """
    c = _as_cell(cell)
    if abs(np.linalg.det(c)) < 1e-20:
        raise ValueError("cannot invert a degenerate cell")
    return 2.0 * math.pi * np.linalg.inv(c).T


def is_orthorhombic(cell, pbc=True, tol: float = 1e-10) -> bool:
    """True if the *periodic* lattice vectors are mutually orthogonal.

    Note this is orthogonality of the lattice vectors, not axis alignment: a
    rotated cubic cell is orthorhombic by this test, as it should be, because
    the property that matters downstream is that fractional rounding gives the
    true minimum image.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)
    tol : float
        Relative tolerance on the off-diagonal Gram matrix elements.
    """
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    basis, _ = _periodic_basis(c, p)
    if basis.shape[0] < 2:
        return True
    gram = basis @ basis.T
    scale = float(np.max(np.abs(np.diag(gram))))
    if scale <= 0.0:
        return True
    off = gram - np.diag(np.diag(gram))
    return bool(np.max(np.abs(off)) <= tol * scale)


def cell_widths(cell, pbc=True) -> np.ndarray:
    """Perpendicular width of the cell along each lattice direction.

    The width along direction ``i`` is the distance between the two periodic
    faces normal to it, i.e. the length of the component of ``a_i``
    perpendicular to the span of the *other periodic* lattice vectors.  For a
    fully periodic cell this is the familiar

    ``w_i = V / |a_j x a_k|``

    and it equals ``1 / |g_i|`` with ``g_i`` the reciprocal vector without the
    2 pi.  For a slab (``pbc = (True, True, False)``) the same definition gives
    ``w_1 = |a1 x a2| / |a2|`` -- the non-periodic direction correctly plays no
    part.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.
    pbc : bool or array_like of bool, shape (3,)

    Returns
    -------
    ndarray, shape (3,)
        Widths in angstrom, with ``inf`` in every non-periodic direction (a
        non-periodic direction imposes no image constraint at all).
    """
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    dims = np.flatnonzero(p)
    out = np.full(3, np.inf)
    for i in dims:
        others = np.array([k for k in dims if k != i], dtype=int)
        a_i = c[i]
        if others.size == 0:
            out[i] = float(np.linalg.norm(a_i))
            continue
        basis = c[others]  # (m, 3)
        # Component of a_i orthogonal to span(basis), via least squares so that
        # the formula is uniform in the number of remaining periodic vectors.
        coeff, *_ = np.linalg.lstsq(basis.T, a_i, rcond=None)
        out[i] = float(np.linalg.norm(a_i - coeff @ basis))
    return out


def min_cell_width(cell, pbc=True) -> float:
    """Smallest perpendicular width between periodic faces, in angstrom.

    This -- and **not** the shortest lattice vector -- is the quantity that
    must exceed ``2 * cutoff`` for the minimum image convention to be valid.
    The two differ for any sheared cell, and the shortest lattice vector is
    always the larger (hence more permissive, hence dangerous) of the two.  A
    concrete example: rows ``(4,0,0), (0,5,0), (2,3,6)`` has shortest lattice
    vector 4 A but minimum width ``12/sqrt(10) = 3.795 A``, so a 1.9 A cutoff
    looks safe by the naive test and is not.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)

    Returns
    -------
    float
        Minimum width in angstrom over the periodic directions, or ``inf`` for
        a fully aperiodic system.
    """
    return float(np.min(cell_widths(cell, pbc)))


def check_minimum_image(cell, pbc, cutoff: float, *, what: str = "potential") -> None:
    """Raise if ``cutoff`` is too large for the minimum image convention.

    A potential that assumes each pair is seen through exactly one image is
    only correct when ``2 * cutoff < min_cell_width``.  Design rule: no silent
    fallbacks -- quiet degradation here would contaminate every observable
    measured downstream, so this raises rather than truncating.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)
    cutoff : float
        Interaction range in angstrom.
    what : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If ``2 * cutoff >= min_cell_width(cell, pbc)``.
    """
    width = min_cell_width(cell, pbc)
    if 2.0 * cutoff >= width:
        raise ValueError(
            f"{what} cutoff {cutoff:.4f} A is too large for this cell: the minimum "
            f"perpendicular width is {width:.4f} A and the minimum image convention "
            f"requires 2*cutoff < width. Use a larger supercell."
        )


# --------------------------------------------------------------------------
# coordinate transforms
# --------------------------------------------------------------------------


def cartesian_to_fractional(positions, cell) -> np.ndarray:
    """Fractional coordinates ``s`` such that ``r = s @ cell``.

    Parameters
    ----------
    positions : array_like, shape (..., 3)
        Cartesian coordinates in angstrom.
    cell : array_like, shape (3, 3)

    Returns
    -------
    ndarray, shape (..., 3)
        Dimensionless fractional coordinates.  Solved with ``np.linalg.solve``
        rather than by forming the inverse, which is both more accurate and
        cheaper for the near-singular cells that show up during barostat runs.
    """
    r = np.asarray(positions, dtype=np.float64)
    c = _as_cell(cell)
    if abs(np.linalg.det(c)) < 1e-20:
        raise ValueError("cannot convert to fractional coordinates with a degenerate cell")
    flat = r.reshape(-1, 3)
    s = np.linalg.solve(c.T, flat.T).T
    return s.reshape(r.shape)


def fractional_to_cartesian(fractional, cell) -> np.ndarray:
    """Cartesian coordinates ``r = s @ cell`` in angstrom.

    Parameters
    ----------
    fractional : array_like, shape (..., 3)
    cell : array_like, shape (3, 3)

    Returns
    -------
    ndarray, shape (..., 3)
    """
    s = np.asarray(fractional, dtype=np.float64)
    c = _as_cell(cell)
    return (s.reshape(-1, 3) @ c).reshape(s.shape)


def wrap_positions(positions, cell, pbc=True, *, eps: float = 0.0, return_shift: bool = False):
    """Wrap positions into the primary cell along periodic directions only.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Cartesian coordinates in angstrom.
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)
    eps : float
        Fractional nudge applied before flooring.  ``0`` is exact; a tiny
        positive value (``1e-12``) pushes atoms sitting exactly on the upper
        face onto the lower one, which makes repeated wrapping idempotent.
    return_shift : bool
        If True also return the integer image ``n`` that was applied, with
        ``wrapped == positions + n @ cell`` holding to machine precision.  The
        neighbour-list builder needs it to report shifts relative to the
        *original* (possibly unwrapped) positions.

    Returns
    -------
    ndarray, shape (N, 3)
        Wrapped positions in angstrom, and if ``return_shift`` the ``(N, 3)``
        int32 image vector as a second element.

    Notes
    -----
    The wrapped positions are formed as ``positions + n @ cell`` rather than as
    ``frac_wrapped @ cell``.  The two are equal in exact arithmetic, but only
    the former is *bitwise* consistent with the returned shift, which matters
    when a pair sits within rounding distance of a cutoff.
    """
    r = np.asarray(positions, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {r.shape}")
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    n = np.zeros((r.shape[0], 3), dtype=np.int64)
    if p.any():
        basis, dims = _periodic_basis(c, p)
        s = r @ _pinv_basis(basis)  # (N, d) coordinates along periodic vectors
        n[:, dims] = -np.floor(s + eps).astype(np.int64)
    wrapped = r + n.astype(np.float64) @ c
    if return_shift:
        return wrapped, n.astype(np.int32)
    return wrapped


# --------------------------------------------------------------------------
# lattice reduction
# --------------------------------------------------------------------------


def reduce_cell(cell, pbc=True) -> tuple[np.ndarray, np.ndarray]:
    """Greedily reduce the periodic sublattice to a nearly-Minkowski basis.

    Repeatedly sorts the periodic lattice vectors by length and subtracts the
    nearest integer multiple of each from the others.  In dimension <= 3 this
    greedy scheme returns a Minkowski-reduced basis for all but contrived
    inputs, and a Minkowski-reduced basis is what makes "round the fractional
    coordinates, then look at the 26 surrounding images" an *exact* solution of
    the closest-lattice-point problem.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)

    Returns
    -------
    basis : ndarray, shape (d, 3)
        Reduced periodic lattice vectors, ``d = pbc.sum()``, in angstrom.
    transform : ndarray, shape (d, d)
        Unimodular integer matrix with ``basis == transform @ cell[pbc]``.
        Needed to translate an image index in the reduced basis back into the
        caller's original lattice units.
    """
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    basis, dims = _periodic_basis(c, p)
    d = basis.shape[0]
    b = basis.copy()
    t = np.eye(d, dtype=np.int64)
    if d < 2:
        return b, t

    for _ in range(64):
        order = np.argsort(np.einsum("ij,ij->i", b, b), kind="stable")
        b, t = b[order], t[order]
        changed = False
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                njj = float(b[j] @ b[j])
                if njj <= 0.0:
                    continue
                mu = int(round(float(b[i] @ b[j]) / njj))
                if mu == 0:
                    continue
                cand = b[i] - mu * b[j]
                # Only accept a strict shortening; the 1e-12 relative guard
                # stops round-off from driving an infinite swap loop.
                if float(cand @ cand) < float(b[i] @ b[i]) * (1.0 - 1e-12):
                    b[i] = cand
                    t[i] = t[i] - mu * t[j]
                    changed = True
        if not changed:
            break
    return b, t


# --------------------------------------------------------------------------
# minimum image
# --------------------------------------------------------------------------


def minimum_image_naive(displacements, cell, pbc=True) -> np.ndarray:
    """Minimum image by rounding fractional coordinates. **Not always correct.**

    Computes ``s = d @ inv(cell)``, subtracts ``round(s)`` in the periodic
    components, and transforms back.  This is exact if and only if the periodic
    lattice vectors are mutually orthogonal.  For a sheared cell the fractional
    cube ``[-1/2, 1/2)^3`` is not the Wigner-Seitz cell, and rounding can
    return a vector that is longer than the true minimum image -- badly so for
    strong shear.  Provided because it is the textbook recipe, it is the fast
    path for orthorhombic cells, and having it named makes the failure mode
    testable instead of latent.

    Parameters
    ----------
    displacements : array_like, shape (..., 3)
        Cartesian displacement vectors in angstrom.
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)

    Returns
    -------
    ndarray, shape (..., 3)
        Displacements in angstrom, reduced by lattice translations.
    """
    d = np.asarray(displacements, dtype=np.float64)
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    if not p.any():
        return d.copy()
    flat = d.reshape(-1, 3)
    s = np.linalg.solve(c.T, flat.T).T
    n = np.zeros_like(s)
    n[:, p] = np.round(s[:, p])
    return (flat - n @ c).reshape(d.shape)


def minimum_image_shift(displacements, cell, pbc=True, *, n_search: int = 1):
    """Minimum image with the integer image index, correct for any cell.

    Reduces the periodic sublattice (see :func:`reduce_cell`), rounds in the
    reduced basis, then exhaustively searches the ``(2*n_search+1)^d``
    surrounding images and keeps the shortest candidate.  With a
    Minkowski-reduced basis ``n_search = 1`` provably contains the true closest
    lattice point in dimension <= 3.

    Parameters
    ----------
    displacements : array_like, shape (..., 3)
        Cartesian displacements in angstrom.
    cell : array_like, shape (3, 3)
    pbc : bool or array_like of bool, shape (3,)
    n_search : int
        Half-width of the image search around the rounded image.

    Returns
    -------
    reduced : ndarray, shape (..., 3)
        Shortest equivalent displacement, in angstrom.
    shift : ndarray, shape (..., 3), int32
        Image index ``n`` in the caller's lattice units, with
        ``reduced == displacements + n @ cell``.
    """
    d = np.asarray(displacements, dtype=np.float64)
    c = _as_cell(cell)
    p = _as_pbc(pbc)
    out_shape = d.shape
    flat = d.reshape(-1, 3)
    shift = np.zeros((flat.shape[0], 3), dtype=np.int64)
    if not p.any():
        return flat.reshape(out_shape).copy(), shift.astype(np.int32).reshape(out_shape)

    basis, transform = reduce_cell(c, p)
    dims = np.flatnonzero(p)
    dim = basis.shape[0]

    s = flat @ _pinv_basis(basis)  # (P, dim)
    m = np.round(s)
    base_vec = flat - m @ basis  # rounded solution: the search is centred here
    base_n = -m.astype(np.int64)  # image index in the reduced basis
    best_vec = base_vec.copy()
    best_r2 = np.einsum("ij,ij->i", best_vec, best_vec)
    best_n = base_n.copy()

    # Loop over candidate images rather than materialising them all at once:
    # keeps the memory O(P) instead of O(P * (2n+1)^d) for large pair counts.
    ranges = [range(-n_search, n_search + 1)] * dim
    grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(dim, -1).T
    for offset in grid:
        if not offset.any():
            continue
        cand = base_vec + offset @ basis
        r2 = np.einsum("ij,ij->i", cand, cand)
        better = r2 < best_r2 - 1e-14
        if better.any():
            best_r2 = np.where(better, r2, best_r2)
            best_vec = np.where(better[:, None], cand, best_vec)
            best_n = np.where(better[:, None], base_n + offset, best_n)

    shift[:, dims] = best_n @ transform
    return best_vec.reshape(out_shape), shift.astype(np.int32).reshape(out_shape)


def minimum_image(displacements, cell, pbc=True, *, method: str = "auto") -> np.ndarray:
    """Shortest periodic image of a set of displacement vectors.

    Two implementations live behind this function:

    ``"naive"``
        Round the fractional coordinates (:func:`minimum_image_naive`).  Exact
        only for mutually orthogonal periodic lattice vectors; for a strongly
        skewed triclinic cell it can return a vector that is much longer than
        the true minimum image, which shows up downstream as missing
        neighbours and a silently wrong energy.

    ``"safe"``
        Lattice-reduce, round, and search the surrounding images
        (:func:`minimum_image_shift`).  Correct for any cell.

    ``"auto"`` (the default, and what every caller should use) picks ``naive``
    when the periodic sublattice is orthogonal -- where the two agree exactly
    and the naive path is a few times cheaper -- and ``safe`` otherwise.

    Parameters
    ----------
    displacements : array_like, shape (..., 3)
        Cartesian displacement vectors in angstrom.  Typically ``r_j - r_i``.
    cell : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.
    pbc : bool or array_like of bool, shape (3,)
        Periodicity per lattice vector.
    method : {"auto", "safe", "naive"}

    Returns
    -------
    ndarray, shape (..., 3)
        Minimum-image displacements in angstrom, same shape as the input.

    Notes
    -----
    This reduces each displacement independently; it says nothing about
    whether the reduction is *unique*.  Uniqueness needs
    ``2 * cutoff < min_cell_width`` -- use :func:`check_minimum_image`.  When it
    does not hold, do not reach for this function at all: build a neighbour
    list, which represents every image as a separate pair.
    """
    if method == "auto":
        method = "naive" if is_orthorhombic(cell, pbc) else "safe"
    if method == "naive":
        return minimum_image_naive(displacements, cell, pbc)
    if method == "safe":
        return minimum_image_shift(displacements, cell, pbc)[0]
    raise ValueError(f"unknown method {method!r}; expected 'auto', 'safe' or 'naive'")


# --------------------------------------------------------------------------
# lattice parameters
# --------------------------------------------------------------------------


def cell_from_parameters(
    a: float, b: float, c: float, alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0
) -> np.ndarray:
    """Build a cell matrix from lattice parameters.

    Uses the standard crystallographic setting: ``a1`` along x, ``a2`` in the
    xy plane with positive y, ``a3`` completing a right-handed set.  This fixes
    the orientation, which the six parameters alone do not.

    Parameters
    ----------
    a, b, c : float
        Lattice vector lengths in angstrom.
    alpha, beta, gamma : float
        Angles in **degrees**: ``alpha`` between ``a2`` and ``a3``, ``beta``
        between ``a1`` and ``a3``, ``gamma`` between ``a1`` and ``a2``.

    Returns
    -------
    ndarray, shape (3, 3)
        Lattice vectors as rows, in angstrom.

    Raises
    ------
    ValueError
        If the parameters are not geometrically realisable, which happens when
        the angles violate the triangle-like inequality that keeps the metric
        tensor positive definite.
    """
    if min(a, b, c) <= 0.0:
        raise ValueError(f"lattice lengths must be positive, got {(a, b, c)}")
    ra, rb, rg = (math.radians(x) for x in (alpha, beta, gamma))
    ca, cb, cg = math.cos(ra), math.cos(rb), math.cos(rg)
    sg = math.sin(rg)
    if abs(sg) < 1e-12:
        raise ValueError(f"gamma = {gamma} is degenerate")
    cx = cb
    cy = (ca - cb * cg) / sg
    cz2 = 1.0 - cx * cx - cy * cy
    if cz2 <= 0.0:
        raise ValueError(
            f"lattice angles ({alpha}, {beta}, {gamma}) do not define a valid cell "
            "(the metric tensor is not positive definite)"
        )
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cx, c * cy, c * math.sqrt(cz2)],
        ]
    )


def cell_to_parameters(cell) -> tuple[float, float, float, float, float, float]:
    """Lattice parameters ``(a, b, c, alpha, beta, gamma)`` of a cell.

    Parameters
    ----------
    cell : array_like, shape (3, 3)
        Lattice vectors as rows, in angstrom.

    Returns
    -------
    tuple of float
        Lengths in angstrom and angles in **degrees**, the exact inverse of
        :func:`cell_from_parameters`.  Note the round trip
        ``cell -> parameters -> cell`` reproduces the original cell only up to
        a rigid rotation, since the parameters carry no orientation; the round
        trip ``parameters -> cell -> parameters`` is exact.
    """
    m = _as_cell(cell)
    lengths = np.linalg.norm(m, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("cannot extract parameters from a cell with a zero lattice vector")

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        cosine = float(u @ v) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))
        return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))

    return (
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        angle(m[1], m[2]),
        angle(m[0], m[2]),
        angle(m[0], m[1]),
    )


# --------------------------------------------------------------------------
# strain and stress helpers
# --------------------------------------------------------------------------


def voigt_to_full(voigt, *, strain: bool = False) -> np.ndarray:
    """Expand a 6-component Voigt vector into a symmetric 3x3 tensor.

    Parameters
    ----------
    voigt : array_like, shape (6,)
        Components in the order ``(xx, yy, zz, yz, xz, xy)``.
    strain : bool
        If True the input is interpreted as **engineering** strain, whose
        shear components are ``gamma_yz = 2 * eps_yz``, so the off-diagonals
        are halved on expansion.  If False (the default, appropriate for
        stress and for the virial) the components are copied unchanged.  This
        factor of two is the single most common sign-of-life bug in elastic
        constant code, which is why it is an explicit flag rather than a
        convention buried in a docstring.

    Returns
    -------
    ndarray, shape (3, 3)
        Symmetric tensor, in the same units as the input.
    """
    v = np.asarray(voigt, dtype=np.float64)
    if v.shape != (6,):
        raise ValueError(f"voigt vector must have shape (6,), got {v.shape}")
    off = v[3:] * (0.5 if strain else 1.0)
    return np.array(
        [
            [v[0], off[2], off[1]],
            [off[2], v[1], off[0]],
            [off[1], off[0], v[2]],
        ]
    )


def full_to_voigt(tensor, *, strain: bool = False) -> np.ndarray:
    """Contract a symmetric 3x3 tensor into Voigt notation.

    Parameters
    ----------
    tensor : array_like, shape (3, 3)
        Symmetric tensor; any antisymmetric part is discarded by averaging,
        since Voigt notation cannot represent it.
    strain : bool
        If True the off-diagonal components are doubled to give engineering
        shear strains, the inverse of :func:`voigt_to_full` with the same flag.

    Returns
    -------
    ndarray, shape (6,)
        Components in the order ``(xx, yy, zz, yz, xz, xy)``.
    """
    t = np.asarray(tensor, dtype=np.float64)
    if t.shape != (3, 3):
        raise ValueError(f"tensor must have shape (3, 3), got {t.shape}")
    sym = 0.5 * (t + t.T)
    factor = 2.0 if strain else 1.0
    return np.array(
        [
            sym[0, 0],
            sym[1, 1],
            sym[2, 2],
            factor * sym[1, 2],
            factor * sym[0, 2],
            factor * sym[0, 1],
        ]
    )


def apply_strain(vectors, strain) -> np.ndarray:
    """Apply an affine strain ``r -> (1 + eps) r`` to a stack of row vectors.

    Works for positions and for cell matrices alike, because both are stored
    as stacks of row vectors: with ``r' = (1 + eps) r`` acting on column
    vectors, the row-vector form is ``V' = V @ (1 + eps).T``.  Straining the
    cell with the same call is what keeps the virial
    ``W = -dU/d(eps)`` consistent with
    :meth:`atomlab.types.Configuration.strained`.

    Parameters
    ----------
    vectors : array_like, shape (..., 3)
        Positions (angstrom) or lattice vectors (angstrom), as rows.
    strain : array_like, shape (3, 3) or (6,)
        Strain tensor ``eps``.  A ``(6,)`` input is read as **engineering**
        Voigt strain and expanded with :func:`voigt_to_full`.  A full ``(3, 3)``
        input is used as given without symmetrisation, so an antisymmetric part
        produces an (infinitesimal) rotation -- occasionally useful, and never
        silently removed.

    Returns
    -------
    ndarray, shape (..., 3)
        Strained vectors, same shape and units as the input.
    """
    v = np.asarray(vectors, dtype=np.float64)
    e = np.asarray(strain, dtype=np.float64)
    if e.shape == (6,):
        e = voigt_to_full(e, strain=True)
    if e.shape != (3, 3):
        raise ValueError(f"strain must have shape (3, 3) or (6,), got {e.shape}")
    defm = np.eye(3) + e
    return (v.reshape(-1, 3) @ defm.T).reshape(v.shape)
