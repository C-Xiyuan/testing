"""Angular structure: bond angles, tetrahedral order, Steinhardt bond order.

Pair correlations cannot see three-body structure at all: a Stillinger-Weber
silicon melt and a Lennard-Jones liquid can be tuned to almost the same
``g(r)`` while differing completely in their bond angles, and a pair-only
surrogate model (``models/pair_spline.py``) is *designed* to fail exactly here.
The observables in this module are what make that failure visible, and the
melting diagnostics of ``observables/melting.py`` are built on ``Q6``.

Neighbour convention
--------------------
All three estimators define "neighbour" by a fixed cutoff ``r_cut`` and take
pairs from :func:`atomlab.neighbors.build_neighbor_list`, which enumerates
periodic images separately.  Self-image pairs (an atom seeing its own replica
in a very small cell) are kept here, unlike in ``g(r)``: they are genuine
bonds of the periodic structure, and the bond angles they subtend are the
angles of that structure.

Sensitivity to ``r_cut`` is real and is not hidden: a cutoff placed on the
flank of a coordination shell makes every quantity in this module a smooth
function of it.  Put it in the minimum of ``g(r)`` between shells, and record
it -- it is part of the definition of the observable, not an implementation
detail.  All results carry it in ``metadata["r_cut"]``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.special import sph_harm_y

from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Trajectory
from .base import FrameSample, ObservableResult, estimate_from_samples, frame_estimator

__all__ = [
    "bond_angle_distribution",
    "tetrahedral_order",
    "tetrahedral_order_per_atom",
    "steinhardt",
    "steinhardt_per_atom",
    "neighbor_groups",
]


# --------------------------------------------------------------------------
# shared neighbour bookkeeping
# --------------------------------------------------------------------------


def neighbor_groups(
    configuration: Configuration, r_cut: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atom neighbour vectors, grouped by central atom.

    Parameters
    ----------
    configuration : Configuration
    r_cut : float
        Neighbour cutoff in angstrom.

    Returns
    -------
    d : ndarray, shape (P, 3)
        Displacement vectors ``r_j + shift @ cell - r_i`` in angstrom, pointing
        from the central atom to the neighbour, sorted by central atom.
    r : ndarray, shape (P,)
        Their lengths in angstrom.
    offsets : ndarray, shape (N+1,), int
        ``d[offsets[i]:offsets[i+1]]`` are the neighbours of atom ``i``.

    Notes
    -----
    :func:`atomlab.neighbors.build_neighbor_list` already returns pairs sorted
    by ``(i, j, shift)``, so the grouping is a cumulative count rather than a
    sort; the assertion that this holds is cheap and worth keeping, because a
    change in that guarantee would silently scramble every angle.
    """
    if r_cut <= 0.0:
        raise ValueError(f"r_cut must be > 0 A, got {r_cut}")
    nl = build_neighbor_list(configuration, float(r_cut), half=False)
    d, r = pair_vectors(configuration, nl)
    i = nl.i
    if i.size and np.any(np.diff(i) < 0):  # pragma: no cover - contract check
        order = np.argsort(i, kind="stable")
        i, d, r = i[order], d[order], r[order]
    counts = np.bincount(i, minlength=configuration.n_atoms)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return d, r, offsets


def _nearest_unit_vectors(
    d: np.ndarray, r: np.ndarray, offsets: np.ndarray, index: int, n_take: int | None
) -> np.ndarray | None:
    """Unit vectors to atom ``index``'s neighbours, optionally the ``n_take`` nearest.

    Returns ``None`` when the atom has fewer than ``n_take`` neighbours, so the
    caller can decide whether that is an error or a skip.
    """
    lo, hi = int(offsets[index]), int(offsets[index + 1])
    if hi - lo == 0:
        return None
    dd = d[lo:hi]
    rr = r[lo:hi]
    if n_take is not None:
        if hi - lo < n_take:
            return None
        pick = np.argpartition(rr, n_take - 1)[:n_take]
        dd, rr = dd[pick], rr[pick]
    return dd / rr[:, None]


# --------------------------------------------------------------------------
# bond angle distribution
# --------------------------------------------------------------------------


@frame_estimator
def _adf_kernel(
    configuration: Configuration,
    *,
    r_cut: float,
    edges_deg: np.ndarray,
    centres_deg: np.ndarray,
    normalization: str,
) -> FrameSample:
    """Single-frame bond-angle histogram."""
    d, r, offsets = neighbor_groups(configuration, r_cut)
    n_bins = len(centres_deg)
    hist = np.zeros(n_bins)
    n_angles = 0

    for atom in range(configuration.n_atoms):
        u = _nearest_unit_vectors(d, r, offsets, atom, None)
        if u is None or u.shape[0] < 2:
            continue
        cos = np.clip(u @ u.T, -1.0, 1.0)
        iu = np.triu_indices(u.shape[0], k=1)
        angles = np.degrees(np.arccos(cos[iu]))
        n_angles += angles.size
        hist += np.histogram(angles, bins=edges_deg)[0]

    if n_angles == 0:
        raise ValueError(
            f"no atom has two neighbours within r_cut = {r_cut} A; the bond angle "
            "distribution is undefined"
        )

    if normalization == "theta":
        # Probability density in theta: sum(P * dtheta) = 1, units 1/degree.
        # The reference for uncorrelated directions is sin(theta)/2 (per radian).
        measure = np.diff(edges_deg)
    elif normalization == "solid_angle":
        # Density per unit cos(theta): sum(P * dcos) = 1, so uncorrelated
        # directions give exactly 1/2 at every angle.  The bin measure is the
        # exact solid-angle measure cos(theta_lo) - cos(theta_hi), never
        # sin(theta_centre) * dtheta -- the two differ by O(dtheta^2) in the
        # interior and catastrophically in the end bins, where sin -> 0.
        c = np.cos(np.radians(edges_deg))
        measure = c[:-1] - c[1:]
    else:
        raise ValueError(
            f"normalization must be 'theta' or 'solid_angle', got {normalization!r}"
        )

    values = hist / (n_angles * measure)
    return FrameSample(
        values=values,
        bins=centres_deg,
        metadata={"n_angles": int(n_angles), "n_atoms": configuration.n_atoms},
    )


def bond_angle_distribution(
    trajectory: Trajectory | Configuration | Sequence[Configuration],
    r_cut: float,
    n_bins: int = 180,
    stride: int = 1,
    *,
    normalization: str = "theta",
    keep_samples: bool = True,
    name: str = "",
) -> ObservableResult:
    """Distribution of bond angles ``theta_jik`` within a cutoff.

    Every unordered pair of neighbours ``(j, k)`` of every central atom ``i``
    contributes one angle, so an atom with ``n`` neighbours contributes
    ``n(n-1)/2`` angles and better-coordinated atoms are weighted more heavily
    -- the standard convention, and the one that makes the integral of the
    distribution proportional to the number of angular triples.

    Parameters
    ----------
    trajectory : Trajectory or Configuration or sequence of Configuration
    r_cut : float
        Neighbour cutoff in angstrom.  Place it in the first minimum of
        ``g(r)``; it is part of the observable's definition.
    n_bins : int
        Uniform bins over ``[0, 180]`` degrees.
    stride : int
    normalization : {"theta", "solid_angle"}
        ``"theta"`` returns a probability density in the angle itself, in
        1/degree, normalised so ``sum(P * dtheta) = 1``.  This is what is
        plotted in the literature, and randomly oriented neighbours give the
        Jacobian curve ``sin(theta)/2`` (per radian), peaked at 90 degrees --
        so a peak near 90 degrees in this representation is not evidence of
        structure.
        ``"solid_angle"`` divides by the exact solid-angle measure of each bin,
        returning a density in ``cos(theta)`` normalised so
        ``sum(P * dcos) = 1``.  Randomly oriented neighbours then give exactly
        ``1/2`` everywhere, which makes deviations from randomness readable by
        eye and is the right representation for a quantitative comparison
        between models.
    keep_samples : bool
    name : str

    Returns
    -------
    ObservableResult
        ``(n_bins,)`` density with ``bins`` the bin-centre angles in degrees.
    """
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    result = _adf_kernel(
        trajectory,
        stride=stride,
        keep_samples=keep_samples,
        name=name or "bond angle distribution",
        r_cut=float(r_cut),
        edges_deg=edges,
        centres_deg=centres,
        normalization=normalization,
    )
    result.metadata.update(
        {
            "r_cut": float(r_cut),
            "normalization": normalization,
            "edges_deg": edges,
            "units": "1/degree" if normalization == "theta" else "per unit cos(theta)",
            "random_reference": (
                "sin(theta)/2 per radian" if normalization == "theta" else "0.5 everywhere"
            ),
        }
    )
    return result


# --------------------------------------------------------------------------
# tetrahedral order parameter
# --------------------------------------------------------------------------


def tetrahedral_order_per_atom(
    configuration: Configuration, r_cut: float, *, n_neighbors: int = 4
) -> np.ndarray:
    """Chau--Hardwick / Errington--Debenedetti ``q`` for every atom.

    .. math::

        q_i = 1 - \\frac{3}{8} \\sum_{j<k}
              \\left(\\cos\\theta_{jik} + \\frac{1}{3}\\right)^2

    over the four nearest neighbours.  The ``3/8`` and the ``1/3`` are chosen
    so that ``q = 1`` for a perfect tetrahedron (every ``cos theta = -1/3``)
    and ``<q> = 0`` for randomly oriented neighbours: with ``cos theta``
    uniform on ``[-1, 1]``, ``E[(cos theta + 1/3)^2] = 1/3 + 1/9 = 4/9`` and
    the six pairs give ``1 - (3/8)(6)(4/9) = 0`` exactly.

    Parameters
    ----------
    configuration : Configuration
    r_cut : float
        Search radius in angstrom.  Must be large enough to contain the four
        nearest neighbours of every atom; the parameter selects candidates, and
        the four *nearest* of those are used.
    n_neighbors : int
        Number of nearest neighbours entering the sum.  Four is the definition;
        the parameter exists so the tests can probe the machinery, not so that
        the observable can be redefined casually.

    Returns
    -------
    ndarray, shape (N,)
        Dimensionless ``q`` per atom.
    """
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
    d, r, offsets = neighbor_groups(configuration, r_cut)
    out = np.empty(configuration.n_atoms)
    # 9 / (2 n (n-1)) is the prefactor that keeps <q> = 0 for random directions
    # at any n: there are n(n-1)/2 pairs, each contributing 4/9 on average.
    # For the standard n = 4 it is exactly 3/8.
    norm = 9.0 / (2.0 * n_neighbors * (n_neighbors - 1))
    for atom in range(configuration.n_atoms):
        u = _nearest_unit_vectors(d, r, offsets, atom, n_neighbors)
        if u is None:
            have = int(offsets[atom + 1] - offsets[atom])
            raise ValueError(
                f"atom {atom} has only {have} neighbours within r_cut = {r_cut} A "
                f"but {n_neighbors} are needed for the tetrahedral order parameter; "
                "increase r_cut"
            )
        cos = np.clip(u @ u.T, -1.0, 1.0)
        iu = np.triu_indices(n_neighbors, k=1)
        out[atom] = 1.0 - norm * np.sum((cos[iu] + 1.0 / 3.0) ** 2)
    return out


@frame_estimator
def _tetrahedral_kernel(
    configuration: Configuration, *, r_cut: float, n_neighbors: int
) -> FrameSample:
    q = tetrahedral_order_per_atom(configuration, r_cut, n_neighbors=n_neighbors)
    return FrameSample(values=[float(q.mean())], metadata={"q_per_atom": q})


def tetrahedral_order(
    trajectory: Trajectory | Configuration | Sequence[Configuration],
    r_cut: float,
    *,
    n_neighbors: int = 4,
    stride: int = 1,
    keep_samples: bool = True,
    name: str = "",
) -> ObservableResult:
    """Trajectory-averaged tetrahedral order parameter ``<q>``.

    ``q = 1`` for perfect diamond/tetrahedral coordination and ``<q> = 0`` for
    an ideal gas, by construction (see :func:`tetrahedral_order_per_atom`).
    For liquid silicon it sits around 0.5-0.6 and it is the sharpest cheap
    discriminator between a tetrahedral network and a simple liquid -- which is
    precisely the distinction a pair-only surrogate cannot make.

    Parameters
    ----------
    trajectory : Trajectory or Configuration or sequence of Configuration
    r_cut : float
        Neighbour search radius in angstrom.
    n_neighbors : int
        Neighbours entering the sum; 4 is the definition.
    stride : int
    keep_samples : bool
    name : str

    Returns
    -------
    ObservableResult
        Scalar ``<q>`` with a blocking error bar over frames.  The per-atom
        values of each frame are in ``metadata["per_frame"][t]["q_per_atom"]``,
        ``(N,)`` per frame, for callers that want the distribution rather than
        the mean.
    """
    result = _tetrahedral_kernel(
        trajectory,
        stride=stride,
        keep_samples=keep_samples,
        name=name or "tetrahedral order q",
        r_cut=float(r_cut),
        n_neighbors=int(n_neighbors),
    )
    result.metadata.update({"r_cut": float(r_cut), "n_neighbors": int(n_neighbors)})
    return result


# --------------------------------------------------------------------------
# Steinhardt bond-order parameters
# --------------------------------------------------------------------------


def _q_lm_per_atom(
    configuration: Configuration, l: int, r_cut: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex ``q_lm(i)`` for every atom, plus bond counts and the bond sum.

    Returns
    -------
    q_lm : ndarray, shape (N, 2l+1), complex
        ``q_lm(i) = (1/n_i) sum_j Y_lm(theta_ij, phi_ij)``; rows with no
        neighbour are zero.
    counts : ndarray, shape (N,), int
        Number of neighbours of each atom.
    bond_sum : ndarray, shape (2l+1,), complex
        Sum of ``Y_lm`` over *all* bonds in the frame, used for the global
        parameter.
    """
    d, r, offsets = neighbor_groups(configuration, r_cut)
    counts = np.diff(offsets)
    n_atoms = configuration.n_atoms
    m_values = np.arange(-l, l + 1)
    q_lm = np.zeros((n_atoms, 2 * l + 1), dtype=np.complex128)
    if d.shape[0] == 0:
        return q_lm, counts, np.zeros(2 * l + 1, dtype=np.complex128)

    # scipy's sph_harm_y(n, m, theta, phi) takes theta as the POLAR angle and
    # phi as the azimuth -- the opposite naming from the deprecated sph_harm.
    theta = np.arccos(np.clip(d[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0])
    y = np.stack([sph_harm_y(l, int(m), theta, phi) for m in m_values], axis=1)  # (P, 2l+1)

    central = np.repeat(np.arange(n_atoms), counts)
    accum = np.zeros((n_atoms, 2 * l + 1), dtype=np.complex128)
    np.add.at(accum, central, y)
    nz = counts > 0
    q_lm[nz] = accum[nz] / counts[nz][:, None]
    return q_lm, counts, y.sum(axis=0)


def _invariant(q_lm: np.ndarray, l: int) -> np.ndarray:
    """``Q_l = sqrt(4 pi / (2l+1) * sum_m |q_lm|^2)`` for each row.

    This contraction over ``m`` is what makes ``Q_l`` a rotational invariant,
    and it is also why the complex and the real spherical-harmonic conventions
    give identical answers: the two bases are related by a unitary matrix, and
    ``sum_m |q_lm|^2`` is the squared norm of a vector, which no unitary change
    of basis can alter.  The test suite checks both statements numerically.
    """
    weight = 4.0 * math.pi / (2 * l + 1)
    return np.sqrt(weight * np.sum(np.abs(q_lm) ** 2, axis=-1))


def steinhardt_per_atom(
    configuration: Configuration, l: int, r_cut: float, *, averaged: bool = False
) -> np.ndarray:
    """Local Steinhardt bond-order parameter ``Q_l(i)`` for every atom.

    Parameters
    ----------
    configuration : Configuration
    l : int
        Degree; 4 and 6 are the useful ones for close-packed structures.
    r_cut : float
        Neighbour cutoff in angstrom.  ``Q_l`` depends on it strongly, because
        it depends on *which* shells are included: for bcc the canonical
        literature values need the 8 first *and* 6 second neighbours (a cutoff
        between ``a sqrt(2)`` and the third shell), whereas fcc, hcp and sc
        need only their first shell.
    averaged : bool
        If True use the Lechner--Dellago averaged variant, in which ``q_lm`` is
        first averaged over the atom *and its neighbours* before the invariant
        is formed.  This suppresses the thermal noise that makes bare ``Q6``
        distributions of a warm solid and a liquid overlap, at the cost of
        blurring interfaces over one neighbour shell.

    Returns
    -------
    ndarray, shape (N,)
        Dimensionless ``Q_l`` per atom; atoms with no neighbour inside
        ``r_cut`` get 0.
    """
    l = int(l)
    if l < 1:
        raise ValueError(f"l must be >= 1, got {l}")
    q_lm, counts, _ = _q_lm_per_atom(configuration, l, r_cut)
    if averaged:
        nl = build_neighbor_list(configuration, float(r_cut), half=False)
        accum = q_lm.copy()
        np.add.at(accum, nl.i, q_lm[nl.j])
        # The atom itself counts as one member of its own neighbourhood, hence
        # the +1 in the denominator (Lechner and Dellago 2008, eq. 6).
        q_lm = accum / (counts + 1)[:, None]
    return _invariant(q_lm, l)


@frame_estimator
def _steinhardt_kernel(
    configuration: Configuration, *, l: int, r_cut: float, averaged: bool
) -> FrameSample:
    q_lm, counts, bond_sum = _q_lm_per_atom(configuration, l, r_cut)
    local = steinhardt_per_atom(configuration, l, r_cut, averaged=averaged)
    total_bonds = int(counts.sum())
    if total_bonds == 0:
        raise ValueError(f"no neighbours within r_cut = {r_cut} A; Q_{l} is undefined")
    # Global parameter: average q_lm over every bond in the system *before*
    # taking the invariant.  It vanishes for an isotropic liquid (the bond
    # orientations cancel) whereas the local mean does not, which is exactly
    # why both are reported.
    global_q = float(_invariant(bond_sum / total_bonds, l))
    return FrameSample(
        values=[float(local.mean()), global_q],
        metadata={"q_per_atom": local, "n_bonds": total_bonds},
    )


def steinhardt(
    trajectory: Trajectory | Configuration | Sequence[Configuration],
    l: int,
    r_cut: float,
    *,
    kind: str = "local",
    averaged: bool = False,
    stride: int = 1,
    keep_samples: bool = True,
    name: str = "",
) -> ObservableResult:
    """Steinhardt bond-order parameter ``Q_l``, averaged over frames.

    Reference values for perfect structures, which the test suite reproduces to
    better than ``1e-3`` (Steinhardt, Nelson and Ronchetti 1983):

    ======  ============  ==========  ==========
    Lattice  neighbours    ``Q4``      ``Q6``
    ======  ============  ==========  ==========
    fcc      12            0.190941    0.574524
    hcp      12            0.097222    0.484762
    bcc      8 + 6         0.036370    0.510688
    sc       6             0.763763    0.353553
    ideal    -             ~0          ~0
    ======  ============  ==========  ==========

    Note the bcc entry: the canonical numbers require the second shell as well,
    which is a statement about ``r_cut``, not about the estimator.

    Parameters
    ----------
    trajectory : Trajectory or Configuration or sequence of Configuration
    l : int
        Degree of the harmonic.
    r_cut : float
        Neighbour cutoff in angstrom -- part of the definition; see
        :func:`steinhardt_per_atom`.
    kind : {"local", "global"}
        ``"local"`` returns the mean over atoms of the per-atom ``Q_l(i)``,
        which is what discriminates fcc/bcc/hcp/liquid environments and what
        ``melting.py`` uses.  ``"global"`` first averages ``q_lm`` over every
        bond in the frame and then takes the invariant, which vanishes for an
        isotropic liquid and is therefore a crystallinity order parameter for
        the system as a whole.  Whichever is not requested is still reported in
        the metadata.
    averaged : bool
        Lechner--Dellago neighbour-averaged variant (local only).
    stride : int
    keep_samples : bool
    name : str

    Returns
    -------
    ObservableResult
        Scalar ``Q_l`` with a blocking error bar.  ``metadata`` carries
        ``"local"``/``"global"`` values and errors, and the per-frame per-atom
        arrays under ``metadata["per_frame"][t]["q_per_atom"]`` ``(N,)``.
    """
    if kind not in ("local", "global"):
        raise ValueError(f"kind must be 'local' or 'global', got {kind!r}")
    l = int(l)
    # The kernel always returns both variants, so the samples are needed here
    # even when the caller does not want them kept in the returned result.
    both = _steinhardt_kernel(
        trajectory,
        stride=stride,
        keep_samples=True,
        name=name or f"Q{l} ({kind})",
        l=l,
        r_cut=float(r_cut),
        averaged=bool(averaged),
    )
    column = 0 if kind == "local" else 1
    samples = both.require_samples()
    result = estimate_from_samples(
        samples[:, [column]],
        name=both.name,
        metadata={
            **{k: v for k, v in both.metadata.items()},
            "l": l,
            "r_cut": float(r_cut),
            "kind": kind,
            "averaged": bool(averaged),
            "local": (float(both.value[0]), float(both.error[0])),
            "global": (float(both.value[1]), float(both.error[1])),
        },
        keep_samples=keep_samples,
    )
    return result
