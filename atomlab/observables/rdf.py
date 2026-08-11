"""Radial distribution function, coordination numbers, and the structure factor.

``g(r)`` is the observable this study leans on hardest: it is cheap, it is the
thing a pair potential is most directly responsible for, and it is a *vector*
observable, which is what makes the linear-response machinery of
:mod:`atomlab.analysis.response` interesting (see ``docs/design.md`` 6.3).  It
is also the observable whose normalisation is most often quietly wrong, so the
conventions are spelled out here rather than left implicit.

Normalisation
-------------
For species ``a`` and ``b`` in a periodic cell of volume ``V`` holding ``N_a``
and ``N_b`` atoms, this module defines

.. math::

    g_{ab}(r_k) = \\frac{\\langle n_{ab}(k) \\rangle}
                       {N_a \\, \\rho_b^{\\rm eff} \\, V_k},
    \\qquad
    V_k = \\frac{4\\pi}{3}\\left(r_{k+1}^3 - r_k^3\\right),
    \\qquad
    \\rho_b^{\\rm eff} = \\frac{N_b - \\delta_{ab}}{V}

where ``n_ab(k)`` counts *ordered* pairs ``(i in a, j in b)`` -- over all
periodic images -- whose separation falls in bin ``k``.

Three choices in that formula are load-bearing.

* **The shell volume is exact**, ``4/3 pi (r_out^3 - r_in^3)``, not the
  linearised ``4 pi r^2 dr``.  The two differ by a relative
  ``(dr/r)^2 / 4``, which is 6% in the first bin of a typical 100-bin,
  10 A histogram: enough to fake a spurious shoulder at small ``r`` and far
  more than the statistical error of a converged run.
* **The counting density carries the** ``delta_ab``.  For a like-species
  partial an atom is not its own neighbour, so the exact ideal-gas expectation
  is ``N_a (N_a - 1) / V`` ordered pairs per unit volume, not ``N_a^2 / V``.
  Using ``N/V`` leaves a ``-1/N`` bias in ``g(r)`` -- 0.5% at ``N = 200``,
  which is larger than the statistical error bar of a well-converged run and
  therefore shows up as a systematic offset in exactly the model-to-model
  comparisons this repository is built to make.  With this convention an ideal
  gas gives ``g(r) = 1`` for every ``r``, with no finite-size correction, and
  the coordination number of a perfect crystal comes out as an exact integer.
* **Ordered pairs with the symmetric prefactor.**  For ``a != b`` we count
  ``(i in a, j in b)`` and normalise by ``N_a rho_b``; counting the reverse
  ordering instead and normalising by ``N_b rho_a`` gives the identical number,
  so ``g_ab == g_ba`` by construction.  This is the combinatorial factor that
  the "count each unordered pair once" recipe gets wrong by a factor of two for
  ``a != b``.

``r_max`` beyond half the cell width
------------------------------------
The minimum-image convention is *not* used here.  Pairs come from
:func:`atomlab.neighbors.build_neighbor_list`, which enumerates every periodic
image separately, so a shell that no longer fits inside the cell is still
counted correctly: at large ``r`` a central atom simply sees several images of
the same partner, and that is the physically right answer for the periodic
system being simulated.  The default policy is therefore ``"images"`` -- allow
it and count correctly -- and the ideal-gas test in
``tests/test_structure_observables.py`` verifies ``g(r) = 1`` out to ``0.9 L``,
well past ``L/2``.

Two caveats, both handled explicitly rather than silently:

* Self-image pairs ``i == j`` (an atom and its own periodic replica) are
  **excluded**.  They are delta functions at the lattice-vector lengths, an
  artefact of the periodic replication rather than a physical pair correlation,
  and including them would put spikes into the ``g(r)`` of an ideal gas.
* Beyond ``L/2`` the periodic replication correlates a pair with itself, so the
  *statistical* error bar is no longer independent bin to bin.  The blocking is
  over frames, so the reported error is still correct; it is the interpretation
  of structure beyond ``L/2`` that needs care.

Pass ``beyond_half_cell="raise"`` to forbid the situation instead.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..cell import min_cell_width, reciprocal_cell
from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Trajectory
from .base import (
    FrameSample,
    ObservableResult,
    _require_periodic,
    estimate_from_samples,
    frame_estimator,
    iter_frames,
)

__all__ = [
    "radial_distribution",
    "coordination_number",
    "structure_factor",
    "structure_factor_direct",
    "shell_volumes",
    "shell_centroids",
]


# --------------------------------------------------------------------------
# bin geometry
# --------------------------------------------------------------------------


def shell_volumes(edges: np.ndarray) -> np.ndarray:
    """Exact volumes of the spherical shells defined by ``edges``.

    Parameters
    ----------
    edges : ndarray, shape (K+1,)
        Bin edges in angstrom, increasing.

    Returns
    -------
    ndarray, shape (K,)
        ``4/3 pi (r_out^3 - r_in^3)`` in A^3.  This is the whole point of the
        module docstring: the linearised ``4 pi r^2 dr`` is wrong by 6% in the
        first bin of a typical histogram.
    """
    e = np.asarray(edges, dtype=np.float64)
    return (4.0 / 3.0) * math.pi * (e[1:] ** 3 - e[:-1] ** 3)


def shell_centroids(edges: np.ndarray) -> np.ndarray:
    """Volume-weighted mean radius of each shell, in angstrom.

    ``rbar = (3/4) (r_out^4 - r_in^4) / (r_out^3 - r_in^3)`` is the radius at
    which a smooth integrand should be sampled when a shell integral is
    approximated by ``V_shell * f(rbar)``; using the arithmetic bin midpoint
    instead biases every shell integral (the structure-factor transform, most
    visibly) at order ``(dr/r)^2``.
    """
    e = np.asarray(edges, dtype=np.float64)
    num = e[1:] ** 4 - e[:-1] ** 4
    den = e[1:] ** 3 - e[:-1] ** 3
    out = np.empty_like(num)
    good = den > 0.0
    out[good] = 0.75 * num[good] / den[good]
    out[~good] = 0.5 * (e[1:] + e[:-1])[~good]
    return out


def _species_index(configuration: Configuration, key) -> int:
    """Resolve a species given as an int index or a chemical symbol."""
    if isinstance(key, str):
        if key not in configuration.symbols:
            raise ValueError(
                f"symbol {key!r} is not in this configuration's symbols "
                f"{configuration.symbols}"
            )
        return int(configuration.symbols.index(key))
    index = int(key)
    if index < 0:
        raise ValueError(f"species index must be >= 0, got {index}")
    return index


# --------------------------------------------------------------------------
# g(r)
# --------------------------------------------------------------------------


@frame_estimator
def _rdf_kernel(
    configuration: Configuration,
    *,
    edges: np.ndarray,
    volumes: np.ndarray,
    centres: np.ndarray,
    pairs,
    beyond_half_cell: str,
) -> FrameSample:
    """Single-frame ``g(r)`` histogram, normalised as in the module docstring."""
    volume = _require_periodic(configuration, "radial_distribution")
    r_max = float(edges[-1])
    width = min_cell_width(configuration.cell, configuration.pbc)
    if 2.0 * r_max > width and beyond_half_cell == "raise":
        raise ValueError(
            f"r_max = {r_max:.4f} A exceeds half the minimum cell width "
            f"({0.5 * width:.4f} A). Pass beyond_half_cell='images' to count all "
            "periodic images correctly, or use a larger cell."
        )

    species = configuration.species
    if pairs is None:
        mask_a = np.ones(configuration.n_atoms, dtype=bool)
        mask_b = mask_a
        same = True
    else:
        type_a = _species_index(configuration, pairs[0])
        type_b = _species_index(configuration, pairs[1])
        mask_a = species == type_a
        mask_b = species == type_b
        same = type_a == type_b
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"partial RDF requested for species {pairs} but the frame holds "
            f"{n_a} and {n_b} atoms of them"
        )
    if same and n_b < 2:
        raise ValueError("a like-species partial RDF needs at least two atoms of that species")

    nl = build_neighbor_list(configuration, r_max, half=False)
    _, r = pair_vectors(configuration, nl)
    # Drop the self-image pairs: they are lattice delta functions from the
    # periodic replication, not pair correlations (see module docstring).
    keep = nl.i != nl.j
    if pairs is not None:
        keep &= mask_a[nl.i] & mask_b[nl.j]
    r = r[keep]

    n_bins = len(volumes)
    index = np.floor(r * (n_bins / r_max)).astype(np.int64)
    inside = (index >= 0) & (index < n_bins)  # r == r_max lands in no bin
    counts = np.bincount(index[inside], minlength=n_bins).astype(np.float64)

    rho_eff = (n_b - (1 if same else 0)) / volume
    g = counts / (n_a * rho_eff * volumes)
    return FrameSample(
        values=g,
        bins=centres,
        metadata={
            "volume": volume,
            "rho_effective": rho_eff,
            "n_a": n_a,
            "n_b": n_b,
            "counts_total": float(counts.sum()),
            "min_cell_width": float(width),
        },
    )


def radial_distribution(
    trajectory: Trajectory | Configuration | Sequence[Configuration],
    r_max: float,
    n_bins: int,
    *,
    pairs: tuple | None = None,
    stride: int = 1,
    beyond_half_cell: str = "images",
    keep_samples: bool = True,
    name: str = "",
) -> ObservableResult:
    """Radial distribution function ``g(r)``, block-averaged over frames.

    Parameters
    ----------
    trajectory : Trajectory or Configuration or sequence of Configuration
        Frames to average over.  A single configuration is accepted (a perfect
        crystal is a one-frame trajectory) and then the error bar is ``nan``.
    r_max : float
        Largest separation binned, in angstrom.  May exceed half the cell
        width; see ``beyond_half_cell`` and the module docstring.
    n_bins : int
        Number of uniform bins covering ``[0, r_max]``.
    pairs : tuple, optional
        ``(a, b)`` species selector for a partial ``g_ab(r)``.  Each entry is
        either an integer species index or a chemical symbol.  ``None`` (the
        default) uses all atoms.
    stride : int
        Use every ``stride``-th frame.
    beyond_half_cell : {"images", "raise"}
        What to do when ``2 * r_max`` exceeds the minimum perpendicular cell
        width.  ``"images"`` counts every periodic image separately, which is
        exact for the periodic system; ``"raise"`` refuses.
    keep_samples : bool
        Keep the per-frame ``g(r)`` curves, which
        :func:`coordination_number` and :func:`structure_factor` need in order
        to propagate the error bar.
    name : str

    Returns
    -------
    ObservableResult
        ``value`` and ``error`` are ``(n_bins,)`` dimensionless, ``bins`` are
        the volume-weighted shell centroids in angstrom (so that
        ``sum_k rho V_k g_k`` over the bins up to ``r`` is exactly the mean
        neighbour count).  ``metadata`` records the bin edges, shell volumes,
        the effective density used, and the per-frame normalisations.

    Notes
    -----
    ``g(r) -> 1`` at large ``r`` is the normalisation self-test; the ideal-gas
    test in the suite checks it bin by bin against the statistical error.
    """
    r_max = float(r_max)
    n_bins = int(n_bins)
    if r_max <= 0.0:
        raise ValueError(f"r_max must be > 0 A, got {r_max}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if beyond_half_cell not in ("images", "raise"):
        raise ValueError(
            f"beyond_half_cell must be 'images' or 'raise', got {beyond_half_cell!r}"
        )
    if pairs is not None and len(tuple(pairs)) != 2:
        raise ValueError(f"pairs must be a 2-tuple (a, b), got {pairs!r}")

    edges = np.linspace(0.0, r_max, n_bins + 1)
    volumes = shell_volumes(edges)
    centres = shell_centroids(edges)

    result = _rdf_kernel(
        trajectory,
        stride=stride,
        keep_samples=keep_samples,
        name=name or ("g(r)" if pairs is None else f"g_{pairs[0]}{pairs[1]}(r)"),
        edges=edges,
        volumes=volumes,
        centres=centres,
        pairs=None if pairs is None else tuple(pairs),
        beyond_half_cell=beyond_half_cell,
    )
    result.metadata.update(
        {
            "edges": edges,
            "shell_volumes": volumes,
            "r_max": r_max,
            "n_bins": n_bins,
            "pairs": None if pairs is None else tuple(pairs),
            "normalisation": (
                "g_ab = <n_ab> / (N_a * (N_b - delta_ab)/V * 4/3 pi (r_out^3 - r_in^3)); "
                "ordered pairs over all periodic images; self-image pairs excluded"
            ),
        }
    )
    return result


# --------------------------------------------------------------------------
# coordination number
# --------------------------------------------------------------------------


def coordination_number(
    rdf_result: ObservableResult,
    r_cut: float,
    *,
    name: str = "",
) -> ObservableResult:
    """Mean number of ``b`` neighbours within ``r_cut`` of an ``a`` atom.

    Computes ``n(r_cut) = 4 pi rho_b int_0^{r_cut} g(r) r^2 dr`` by summing the
    *exact* shell integrals ``rho_b V_k g_k``, with the bin straddling
    ``r_cut`` contributing its partial shell volume
    ``4/3 pi (r_cut^3 - r_in^3)``.  Because the same ``rho_b`` and the same
    exact shell volumes were used to normalise ``g``, this sum telescopes back
    to (counted pairs)/(number of central atoms): for a perfect crystal it
    returns exactly 12, 6, 24, ... with no discretisation error at all, which
    is the strongest possible check that the normalisation is right.

    Parameters
    ----------
    rdf_result : ObservableResult
        Output of :func:`radial_distribution`, computed with
        ``keep_samples=True`` so the error can be propagated.
    r_cut : float
        Integration limit in angstrom; must not exceed the RDF's ``r_max``.
        Choose it in the minimum between two shells -- the answer is
        insensitive to it there, which is what makes a coordination number a
        well-defined observable at all.
    name : str

    Returns
    -------
    ObservableResult
        Scalar (dimensionless) with a blocking error bar obtained by
        integrating each frame's ``g(r)`` separately and re-blocking, so
        correlations between bins are handled exactly.
    """
    meta = rdf_result.metadata
    if "edges" not in meta:
        raise ValueError("coordination_number expects the result of radial_distribution()")
    edges = np.asarray(meta["edges"], dtype=float)
    r_cut = float(r_cut)
    if r_cut <= 0.0 or r_cut > edges[-1]:
        raise ValueError(
            f"r_cut = {r_cut} A must lie in (0, r_max = {edges[-1]} A]; the RDF "
            "carries no information beyond r_max"
        )

    # Partial shell volumes: full shells below r_cut, the straddling bin
    # truncated exactly at r_cut, nothing above.
    lo = np.minimum(edges[:-1], r_cut)
    hi = np.minimum(edges[1:], r_cut)
    partial = (4.0 / 3.0) * math.pi * np.maximum(hi**3 - lo**3, 0.0)

    per_frame = meta.get("per_frame")
    if per_frame is None:
        raise ValueError("RDF result carries no per-frame metadata")
    rho = np.array([m["rho_effective"] for m in per_frame], dtype=float)

    samples = rdf_result.require_samples()
    coordination = (samples * partial[None, :]).sum(axis=1) * rho

    return estimate_from_samples(
        coordination[:, None],
        name=name or f"n({r_cut:.3f} A)",
        metadata={
            "r_cut": r_cut,
            "rho_effective": float(rho.mean()),
            "derived_from": rdf_result.name,
            "method": "exact shell integration of g(r), error re-blocked frame by frame",
        },
    )


# --------------------------------------------------------------------------
# structure factor: Fourier route
# --------------------------------------------------------------------------


def _lorch_window(r: np.ndarray, r_max: float) -> np.ndarray:
    """Lorch modification function ``sin(pi r / r_max) / (pi r / r_max)``.

    Truncating ``h(r)`` at ``r_max`` multiplies it by a box, whose transform is
    a ``sinc`` with slowly decaying side lobes; the result is the familiar
    ripple in ``S(q)`` at the period ``2 pi / r_max``.  The Lorch window tapers
    the integrand smoothly to zero at ``r_max``, trading that ripple for a loss
    of ``q`` resolution of order ``2 pi / r_max`` -- the standard bargain in
    diffraction data analysis, and the right one when the question is "do two
    models predict the same S(q)" rather than "where exactly is the peak".
    """
    x = math.pi * np.asarray(r, dtype=float) / r_max
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0.0, 1.0, x))


def structure_factor(
    rdf_result: ObservableResult,
    q_values: np.ndarray,
    density: float | None = None,
    *,
    window: str = "lorch",
    name: str = "",
) -> ObservableResult:
    """Structure factor from the Fourier transform of ``h(r) = g(r) - 1``.

    .. math::

        S(q) = 1 + \\rho \\int_0^{r_{\\rm max}} 4\\pi r^2 h(r)
               \\frac{\\sin qr}{qr} W(r)\\, dr

    evaluated as an exact shell sum, ``rho * sum_k V_k h_k j0(q rbar_k) W_k``,
    with ``rbar_k`` the volume-weighted shell centroid.

    Parameters
    ----------
    rdf_result : ObservableResult
        Output of :func:`radial_distribution` with ``keep_samples=True``.
    q_values : ndarray, shape (Q,)
        Wavevector magnitudes in 1/A.  ``q = 0`` is not allowed: the ``q -> 0``
        limit is a compressibility, which a truncated ``h(r)`` cannot deliver.
    density : float, optional
        Number density in 1/A^3 of the species being counted.  Defaults to the
        effective density used to normalise the RDF, which is the value that
        makes this transform consistent with the counting convention.
    window : {"lorch", "none"}
        Truncation window; see :func:`_lorch_window`.  ``"none"`` gives the
        sharper but ripplier raw transform, and is what the agreement test
        against the direct route uses.
    name : str

    Returns
    -------
    ObservableResult
        ``(Q,)`` dimensionless ``S(q)`` with ``bins`` the ``q`` values in 1/A.
        The error bar comes from transforming each frame's ``g(r)`` and
        re-blocking.
    """
    meta = rdf_result.metadata
    if "edges" not in meta:
        raise ValueError("structure_factor expects the result of radial_distribution()")
    edges = np.asarray(meta["edges"], dtype=float)
    volumes = np.asarray(meta["shell_volumes"], dtype=float)
    centres = shell_centroids(edges)
    r_max = float(edges[-1])

    q = np.atleast_1d(np.asarray(q_values, dtype=float))
    if np.any(q <= 0.0):
        raise ValueError("q_values must be strictly positive; S(q->0) is not accessible here")

    if density is None:
        per_frame = meta.get("per_frame")
        if per_frame is None:
            raise ValueError("no per-frame metadata: pass `density` explicitly")
        density = float(np.mean([m["rho_effective"] for m in per_frame]))
    density = float(density)
    if density <= 0.0:
        raise ValueError(f"density must be > 0 A^-3, got {density}")

    if window == "lorch":
        w = _lorch_window(centres, r_max)
    elif window == "none":
        w = np.ones_like(centres)
    else:
        raise ValueError(f"window must be 'lorch' or 'none', got {window!r}")

    # kernel[q, k] = rho * V_k * j0(q rbar_k) * W_k  ->  S = 1 + kernel @ h
    x = q[:, None] * centres[None, :]
    j0 = np.where(x < 1e-12, 1.0, np.sin(x) / np.where(x == 0.0, 1.0, x))
    kernel = density * volumes[None, :] * j0 * w[None, :]

    result = rdf_result.transform(
        lambda g: 1.0 + kernel @ (g - 1.0),
        bins=q,
        name=name or "S(q) [FT of g(r)]",
        metadata={
            "route": "fourier",
            "window": window,
            "density": density,
            "r_max": r_max,
        },
    )
    return result


# --------------------------------------------------------------------------
# structure factor: direct route on the reciprocal lattice
# --------------------------------------------------------------------------


def _reciprocal_vectors(cell: np.ndarray, q_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal-lattice vectors with ``0 < |q| <= q_max``, one per +/- pair.

    Only wavevectors commensurate with the cell are allowed: ``exp(i q . r)``
    must be periodic under ``r -> r + a_i``, otherwise the sum over the
    simulation box is not the sum over the periodic system it represents and
    ``S(q)`` acquires a spurious self-interference term.

    Returns
    -------
    q : ndarray, shape (M, 3)
        Cartesian wavevectors in 1/A.
    q_norm : ndarray, shape (M,)
        Their magnitudes in 1/A.
    """
    b = reciprocal_cell(cell)  # rows b1, b2, b3, with a_i . b_j = 2 pi delta_ij
    # |n . B| >= |n_k| * (2 pi / w_k) is false in general, so bound each index
    # generously using the direct lattice vector lengths and filter afterwards.
    lengths = np.linalg.norm(cell, axis=1)
    n_max = np.maximum(1, np.ceil(q_max * lengths / (2.0 * math.pi)).astype(int))
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in n_max], indexing="ij")
    n = np.stack([g.ravel() for g in grids], axis=1)

    # Keep one of each +/- pair: S(-q) = S(q) for real densities, so the other
    # half is redundant work.  "First nonzero index positive" picks exactly one.
    first = np.zeros(len(n), dtype=int)
    nonzero = n != 0
    any_nz = nonzero.any(axis=1)
    idx = np.argmax(nonzero, axis=1)
    first[any_nz] = n[np.arange(len(n))[any_nz], idx[any_nz]]
    n = n[first > 0]

    q = n @ b
    q_norm = np.linalg.norm(q, axis=1)
    keep = q_norm <= q_max
    return q[keep], q_norm[keep]


def structure_factor_direct(
    trajectory: Trajectory | Configuration | Sequence[Configuration],
    q_max: float,
    *,
    n_bins: int = 60,
    stride: int = 1,
    q_min: float = 0.0,
    chunk: int = 4096,
    keep_samples: bool = True,
    name: str = "",
) -> ObservableResult:
    """Structure factor measured directly as ``<|sum_i e^{i q.r_i}|^2> / N``.

    This is the independent route: it never touches ``g(r)``, has no binning of
    real-space distances, no truncation and no window, and it is evaluated only
    on wavevectors commensurate with the simulation cell.  Agreement between
    this and :func:`structure_factor` is the strongest available check that the
    RDF normalisation, the shell volumes and the transform are all correct --
    the two routes share no code path beyond the frame loop.

    Parameters
    ----------
    trajectory : Trajectory or Configuration or sequence of Configuration
        All frames must share one cell: ``S(q)`` on a reciprocal lattice that
        moves between frames is not a single observable, so a varying cell
        raises rather than being silently averaged.
    q_max : float
        Largest wavevector magnitude, in 1/A.  Cost grows as ``q_max^3``.
    n_bins : int
        Number of uniform ``|q|`` bins in ``[q_min, q_max]``.  Bins containing
        no reciprocal-lattice vector are dropped from the output.
    stride : int
    q_min : float
        Lower edge of the binning, 1/A.  Defaults to 0, which starts at the
        smallest commensurate wavevector ``2 pi / L``.
    chunk : int
        Wavevectors processed per batch; bounds peak memory at
        ``chunk * N`` complex numbers.
    keep_samples : bool
    name : str

    Returns
    -------
    ObservableResult
        ``S(q)`` (dimensionless) with ``bins`` the count-weighted mean ``|q|``
        of each bin in 1/A.  ``metadata["n_vectors"]`` records how many
        reciprocal-lattice vectors contributed to each bin, which is the right
        thing to look at when a bin's error bar looks surprising.

    Notes
    -----
    The all-atom (single-species) definition is used; a partial ``S_ab(q)``
    would need Faber--Ziman weights and is not required by this study.
    """
    frames = iter_frames(trajectory, stride)
    cell = frames[0].cell
    for k, cfg in enumerate(frames[1:], start=1):
        if not np.allclose(cfg.cell, cell, rtol=1e-10, atol=1e-10):
            raise ValueError(
                f"frame {k} has a different cell; the direct structure factor is "
                "defined on the cell's reciprocal lattice and cannot average over "
                "changing lattices"
            )
    _require_periodic(frames[0], "structure_factor_direct")

    q_max = float(q_max)
    q_min = float(q_min)
    if q_max <= 0.0 or q_max <= q_min:
        raise ValueError(f"need 0 <= q_min < q_max, got ({q_min}, {q_max})")

    q_vec, q_norm = _reciprocal_vectors(cell, q_max)
    if q_vec.shape[0] == 0:
        raise ValueError(
            f"no reciprocal-lattice vector with |q| <= {q_max} 1/A; the smallest "
            "one is 2 pi / L for a cubic cell of edge L"
        )

    edges = np.linspace(q_min, q_max, int(n_bins) + 1)
    which = np.clip(np.digitize(q_norm, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    occupied = counts > 0
    q_centres = np.zeros(n_bins)
    q_centres[occupied] = np.bincount(which, weights=q_norm, minlength=n_bins)[occupied] / counts[occupied]

    def kernel(cfg: Configuration) -> FrameSample:
        n_atoms = cfg.n_atoms
        acc = np.zeros(n_bins)
        pos = cfg.positions
        for start in range(0, q_vec.shape[0], chunk):
            block = q_vec[start : start + chunk]
            phase = block @ pos.T  # (m, N), dimensionless
            rho_q = np.exp(1j * phase).sum(axis=1)
            power = (rho_q.real**2 + rho_q.imag**2) / n_atoms
            acc += np.bincount(which[start : start + chunk], weights=power, minlength=n_bins)
        values = np.zeros(n_bins)
        values[occupied] = acc[occupied] / counts[occupied]
        return FrameSample(
            values=values[occupied],
            bins=q_centres[occupied],
            metadata={"n_atoms": n_atoms},
        )

    from .base import accumulate_frames  # local import: keeps the public API flat

    result = accumulate_frames(
        frames,
        kernel,
        name=name or "S(q) [direct]",
        keep_samples=keep_samples,
        metadata={
            "route": "direct",
            "q_max": q_max,
            "n_vectors": counts[occupied],
            "n_vectors_total": int(q_vec.shape[0]),
            "definition": "S(q) = <|sum_i exp(i q.r_i)|^2> / N on the reciprocal lattice",
        },
    )
    return result
