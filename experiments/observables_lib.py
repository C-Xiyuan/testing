"""Observable functions shared by the experiment scripts.

These are deliberately built from the neighbour list alone, with no dependency
on the richer estimators in :mod:`atomlab.observables`.  The reason is not
convenience but interpretability: the response theory of ``docs/theory.md``
applies to an observable ``A(x)`` that is a *plain function of the
configuration*, and the quantities here are exactly that.  A normalised ``g(r)``
divides by a density and a shell volume, both of which are constants at fixed
``N`` and ``V``, so the pair count in a bin and ``g(r)`` in that bin differ only
by a known factor -- and working with the raw count keeps the correspondence
between what the theory predicts and what is measured completely explicit.

The bin counts, taken together, are a coarse radial distribution function and
are handled by the response machinery as a vector observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomlab.neighbors import build_neighbor_list, pair_vectors


@dataclass
class PairBinObservable:
    """Vector observable: the number of pairs falling in each radial bin.

    Parameters
    ----------
    edges:
        ``(K+1,)`` bin edges in angstrom.  The outermost edge must not exceed
        the cutoff used to build the neighbour list, or the outer bins are
        truncated by the neighbour search rather than by physics.
    cutoff:
        Neighbour-list cutoff.  Defaults to the outermost bin edge.

    Notes
    -----
    ``g(r_k) = count_k / (N * rho_other * V_shell_k / 2)`` with
    ``rho_other = (N-1)/V`` and
    ``V_shell_k = (4/3) pi (r_out^3 - r_in^3)``.  :meth:`to_g_of_r` applies that
    conversion so figures can be drawn in the units readers expect, while the
    response analysis operates on the counts.
    """

    edges: np.ndarray
    cutoff: float | None = None

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("edges must be a 1-D array of at least two values")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        if self.cutoff is None:
            self.cutoff = float(self.edges[-1])
        elif self.cutoff < self.edges[-1] - 1e-12:
            raise ValueError(
                f"cutoff {self.cutoff} is inside the outermost bin edge {self.edges[-1]}; "
                "the outer bins would be truncated by the neighbour search"
            )

    @property
    def n_bins(self) -> int:
        return self.edges.size - 1

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def __call__(self, configuration) -> np.ndarray:
        """``(K,)`` pair counts for one configuration."""
        nl = build_neighbor_list(configuration, self.cutoff, half=True)
        _, r = pair_vectors(configuration, nl)
        return np.histogram(r, bins=self.edges)[0].astype(float)

    def evaluate_trajectory(self, trajectory) -> np.ndarray:
        """``(T, K)`` counts over every frame."""
        return np.array([self(cfg) for cfg in trajectory])

    def to_g_of_r(self, counts, configuration) -> np.ndarray:
        """Convert pair counts to ``g(r)``.

        The shell volume is computed exactly as ``(4/3) pi (r_out^3 - r_in^3)``
        rather than approximated by ``4 pi r^2 dr``; the approximation is wrong
        by several percent in the innermost bins, which is where the first peak
        of a liquid lives.
        """
        counts = np.asarray(counts, dtype=float)
        n = configuration.n_atoms
        if not np.isfinite(configuration.volume) or configuration.volume <= 0:
            raise ValueError("g(r) normalization requires a finite positive cell volume")
        if n < 2:
            raise ValueError("g(r) normalization requires at least two atoms")
        # For a tagged atom there are N-1 possible partners, not N.  Using
        # N/V leaves a finite-size (N-1)/N bias even for an ideal gas.
        other_density = (n - 1) / configuration.volume
        shell = (4.0 / 3.0) * np.pi * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
        return counts / (0.5 * n * other_density * shell)


def coordination_number(configuration, r_cut: float) -> float:
    """Mean number of neighbours within ``r_cut``, per atom."""
    nl = build_neighbor_list(configuration, r_cut, half=True)
    _, r = pair_vectors(configuration, nl)
    return float(2.0 * (r < r_cut).sum() / configuration.n_atoms)


def mean_pair_energy(configuration, potential) -> float:
    """Potential energy per atom in eV -- the cheapest scalar observable."""
    return potential.energy(configuration) / configuration.n_atoms


def nearest_neighbour_distance(configuration, cutoff: float) -> float:
    """Mean distance to each atom's closest neighbour, in angstrom.

    A different kind of observable from the bin counts: it is a nonlinear,
    order-statistic function of the configuration rather than a sum over pairs.
    Response theory makes no assumption of linearity in the observable, so this
    is a useful check that the machinery is not quietly relying on one.
    """
    nl = build_neighbor_list(configuration, cutoff, half=False)
    _, r = pair_vectors(configuration, nl)
    closest = np.full(configuration.n_atoms, np.inf)
    np.minimum.at(closest, nl.i, r)
    if not np.all(np.isfinite(closest)):
        raise ValueError(f"some atoms have no neighbour within {cutoff} A")
    return float(closest.mean())


def bond_orientational_order(configuration, r_cut: float, l: int = 6) -> float:
    """Global Steinhardt ``Q_l``, the standard solid/liquid discriminator.

    Uses spherical harmonics via ``scipy.special.sph_harm_y``.  Reference values
    for perfect lattices: fcc ``Q6 = 0.5745``, ``Q4 = 0.1909``; bcc
    ``Q6 = 0.5106``, ``Q4 = 0.0364``; hcp ``Q6 = 0.4848``; simple cubic
    ``Q6 = 0.3536``, ``Q4 = 0.7638``.  A liquid gives a small value that
    decreases with system size.

    .. warning::
       **The bcc reference values assume the first two coordination shells.**
       bcc has 8 neighbours at ``a√3/2 = 0.866 a`` and 6 more at ``a``, which is
       only 15 % further out.  A cutoff that captures the first shell alone
       gives ``Q6 = 0.6285``, ``Q4 = 0.5092`` -- values so far from the quoted
       ones that a bcc crystal would be classified as something else entirely.
       Verified here: at ``r_cut`` giving ``Z = 14`` this function returns
       ``Q4 = 0.0364``, ``Q6 = 0.5107``, against the literature 0.0364 and
       0.5106.  Choose ``r_cut`` from the first minimum of ``g(r)`` and record
       the coordination number it yields alongside any ``Q_l`` reported.
    """
    from scipy.special import sph_harm_y

    nl = build_neighbor_list(configuration, r_cut, half=False)
    d, r = pair_vectors(configuration, nl)
    if r.size == 0:
        raise ValueError(f"no pairs within {r_cut} A")

    theta = np.arccos(np.clip(d[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(d[:, 1], d[:, 0])

    total = 0.0
    for m in range(-l, l + 1):
        q_lm = sph_harm_y(l, m, theta, phi).mean()
        total += np.abs(q_lm) ** 2
    return float(np.sqrt(4.0 * np.pi / (2 * l + 1) * total))
