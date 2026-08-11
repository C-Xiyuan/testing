"""Polynomial (ACE/SNAP-style) invariant basis for the linear model.

This is the deliberately simple sibling of :mod:`atomlab.models.descriptors.soap`.
Two blocks of features are built for every atom ``i``:

**Two-body.**  A radial-only sum over neighbours,

.. math::  B^{(2)}_{k}(i) = \\sum_{j} T_k\\!\\big(x(r_{ij})\\big)\\, f_c(r_{ij}),
           \\qquad x(r) = \\frac{2r}{r_{\\rm cut}} - 1 \\in [-1, 1],

with :math:`T_k` the Chebyshev polynomial of the first kind.

**Three-body.**  A sum over *ordered pairs* of distinct neighbours, coupling a
radial function to a Legendre polynomial of the bond angle,

.. math::  B^{(3)}_{q l}(i) = \\sum_{j \\ne k}
           \\psi_q(r_{ij}) \\psi_q(r_{ik}) P_l(\\cos\\theta_{jik}),
           \\qquad \\psi_q(r) = T_q(x(r)) f_c(r).

Both blocks are invariant under translation, rotation, permutation of identical
atoms, and the choice of periodic image, exactly and by construction:
:math:`r_{ij}` and :math:`\\cos\\theta_{jik}` are the only geometric inputs and
both are invariants, and the sums do not depend on neighbour ordering.

Why this descriptor exists
--------------------------

It is the basis of the "ACE-lite" baseline in ``atomlab/models/linear.py``.  The
model built on it is

.. math::  U = \\sum_i \\mathbf{w} \\cdot \\mathbf{B}(i)

which is *exactly linear in the fitted coefficients*, so the fit is a solved
least-squares problem with no optimiser, no seed dependence and no convergence
question, and the forces are linear in ``w`` as well.  The value of that for
this repository is not accuracy: it is that the error field ``delta U`` of a
linear model is a known, low-frequency object living in the span of the basis
functions the model omits.  Every other model in the zoo has an error structure
one can only measure; this one has an error structure one can reason about,
which makes it the control against which the rest are read.

For that reason the features are **not** normalised to unit length (unlike
SOAP): normalisation is a nonlinear operation on the environment and would
destroy the exact linearity that is the entire point.

Derivatives
-----------

Both blocks differentiate in closed form.  Writing
:math:`\\mathbf{d}_a = \\mathbf{r}_a - \\mathbf{r}_i` for the neighbour vectors,
:math:`\\hat{\\mathbf{d}}_a = \\mathbf{d}_a / r_a` and
:math:`u_{ab} = \\hat{\\mathbf{d}}_a \\cdot \\hat{\\mathbf{d}}_b`,

.. math::

    \\frac{\\partial B^{(2)}_k}{\\partial \\mathbf{d}_a}
      &= \\psi_k'(r_a)\\, \\hat{\\mathbf{d}}_a \\\\
    \\frac{\\partial u_{ab}}{\\partial \\mathbf{d}_a}
      &= \\frac{\\hat{\\mathbf{d}}_b - u_{ab} \\hat{\\mathbf{d}}_a}{r_a} \\\\
    \\frac{\\partial B^{(3)}_{ql}}{\\partial \\mathbf{d}_a}
      &= 2 \\sum_{b \\ne a} \\Big[ \\psi_q'(r_a) \\psi_q(r_b) P_l(u_{ab})
         \\hat{\\mathbf{d}}_a
         + \\psi_q(r_a)\\psi_q(r_b) P_l'(u_{ab})
           \\frac{\\hat{\\mathbf{d}}_b - u_{ab}\\hat{\\mathbf{d}}_a}{r_a} \\Big]

The factor 2 in the last line is the one that is easy to lose: the double sum
runs over *ordered* pairs, so index ``a`` appears both as the first and as the
second member and the summand is symmetric under exchange.  Dropping it gives a
force that is exactly half of the three-body contribution, which a fit then
compensates for by doubling the corresponding weights -- producing a model that
is self-consistently wrong and looks fine on energies.  The finite-difference
test in ``tests/test_soap.py`` is what stands between that bug and the study.

The self derivative is ``dB_i/dr_i = -sum_a dB_i/dd_a`` because every ``d_a``
has its origin at atom ``i``.

Cost
----

Defaults ``n_radial_2b = 8``, ``n_radial_3b = 4``, ``l_max = 4`` give
``8 + 4 * 5 = 28`` features per atom.  The three-body block is the expensive
one: it is quadratic in the number of neighbours, so the cost scales as
``N * n_neigh^2 * n_radial_3b * l_max``.  With ``r_cut = 5 A`` in a condensed
phase (~50 neighbours) that is affordable on 4 cores for the few-thousand
64-atom configurations this study fits; pushing ``r_cut`` past ~6 A or
``n_radial_3b`` past ~6 is not.
"""

from __future__ import annotations

import numpy as np

from ...neighbors import build_neighbor_list, pair_vectors
from ...types import Configuration
from ..base import Descriptor, DescriptorOutput
from .soap import segment_sum

__all__ = ["Bispectrum", "PolynomialBasis", "ACEBasis", "chebyshev", "legendre"]


# --------------------------------------------------------------------------
# polynomial families and their derivatives
# --------------------------------------------------------------------------


def chebyshev(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev polynomials of the first kind ``T_0 .. T_{n-1}`` and ``dT/dx``.

    Parameters
    ----------
    x : ndarray, shape (P,)
        Arguments, expected in ``[-1, 1]`` (dimensionless).
    n : int
        Number of polynomials, ``>= 1``.

    Returns
    -------
    t : ndarray, shape (P, n)
        Values, dimensionless.
    dt : ndarray, shape (P, n)
        Derivatives with respect to ``x``.

    Notes
    -----
    Generated by the three-term recurrences
    ``T_k = 2 x T_{k-1} - T_{k-2}`` and
    ``T_k' = 2 T_{k-1} + 2 x T_{k-1}' - T_{k-2}'`` (the derivative of the
    first), so the derivative returned is the exact derivative of the value
    returned, term by term.  The closed forms ``T_k = cos(k arccos x)`` and
    ``T_k' = k U_{k-1}(x)`` are avoided because they are singular at
    ``x = +-1``, which is precisely where the cutoff puts the outermost
    neighbours.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    x = np.asarray(x, dtype=np.float64)
    t = np.empty(x.shape + (n,))
    dt = np.empty(x.shape + (n,))
    t[..., 0] = 1.0
    dt[..., 0] = 0.0
    if n > 1:
        t[..., 1] = x
        dt[..., 1] = 1.0
    for k in range(2, n):
        t[..., k] = 2.0 * x * t[..., k - 1] - t[..., k - 2]
        dt[..., k] = 2.0 * t[..., k - 1] + 2.0 * x * dt[..., k - 1] - dt[..., k - 2]
    return t, dt


def legendre(u: np.ndarray, l_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Legendre polynomials ``P_0 .. P_{l_max}`` and ``dP/du``.

    Parameters
    ----------
    u : ndarray
        Arguments, ``cos(theta)`` in ``[-1, 1]`` (dimensionless).
    l_max : int
        Maximum degree, ``>= 0``.

    Returns
    -------
    p : ndarray, shape u.shape + (l_max + 1,)
    dp : ndarray, shape u.shape + (l_max + 1,)

    Notes
    -----
    Bonnet's recurrence ``l P_l = (2l-1) u P_{l-1} - (l-1) P_{l-2}``, with its
    term-by-term derivative
    ``l P_l' = (2l-1)(P_{l-1} + u P_{l-1}') - (l-1) P_{l-2}'``.  The usual
    closed form ``P_l'(u) = l (u P_l - P_{l-1}) / (u^2 - 1)`` is not used: it is
    0/0 at ``u = +-1``, i.e. for collinear neighbour triples, which occur in
    every cubic crystal.
    """
    l_max = int(l_max)
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    u = np.asarray(u, dtype=np.float64)
    p = np.empty(u.shape + (l_max + 1,))
    dp = np.empty(u.shape + (l_max + 1,))
    p[..., 0] = 1.0
    dp[..., 0] = 0.0
    if l_max >= 1:
        p[..., 1] = u
        dp[..., 1] = 1.0
    for l in range(2, l_max + 1):
        p[..., l] = ((2 * l - 1) * u * p[..., l - 1] - (l - 1) * p[..., l - 2]) / l
        dp[..., l] = (
            (2 * l - 1) * (p[..., l - 1] + u * dp[..., l - 1]) - (l - 1) * dp[..., l - 2]
        ) / l
    return p, dp


# --------------------------------------------------------------------------
# the descriptor
# --------------------------------------------------------------------------


class Bispectrum(Descriptor):
    """Radial (two-body) + radial x Legendre (three-body) invariant basis.

    Parameters
    ----------
    r_cut : float
        Cutoff radius in angstrom.  Default 5.0.
    n_radial_2b : int
        Number of Chebyshev functions in the two-body block.  Default 8.
    n_radial_3b : int
        Number of Chebyshev functions in the three-body block.  Default 4.
        Both members of a neighbour pair use the *same* radial index; giving
        them independent indices would square this block's size for a gain that
        is not affordable at this compute budget.
    l_max : int
        Maximum Legendre degree in the three-body block; ``l`` runs
        ``0 .. l_max``.  Default 4.
    name : str
        Identifier used in tables and figures.

    Attributes
    ----------
    cutoff : float
        Interaction range in angstrom.
    n_features : int
        ``n_radial_2b + n_radial_3b * (l_max + 1)``; 28 for the defaults.

    Notes
    -----
    Feature ordering is the two-body block first (``k = 0 .. n_radial_2b-1``),
    then the three-body block in ``(q, l)`` row-major order.

    Units: the features carry no units at all -- ``T_k`` and ``P_l`` are
    dimensionless and the cutoff function is dimensionless -- so the two-body
    block is a pure neighbour count weighted by a polynomial, and the three-body
    block is a count of neighbour pairs.  Derivatives are therefore in ``1/A``.

    The ``l = 0`` three-body features are not redundant with the two-body ones:
    ``P_0 = 1`` makes them ``(sum_a psi_q)^2 - sum_a psi_q^2``, a genuinely
    quadratic function of the environment.
    """

    def __init__(
        self,
        r_cut: float = 5.0,
        n_radial_2b: int = 8,
        n_radial_3b: int = 4,
        l_max: int = 4,
        *,
        name: str = "bispectrum",
    ) -> None:
        self.cutoff = float(r_cut)
        if self.cutoff <= 0.0:
            raise ValueError(f"r_cut must be > 0 A, got {r_cut}")
        self.n_radial_2b = int(n_radial_2b)
        self.n_radial_3b = int(n_radial_3b)
        self.l_max = int(l_max)
        if self.n_radial_2b < 1 or self.n_radial_3b < 1:
            raise ValueError("n_radial_2b and n_radial_3b must both be >= 1")
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        self.name = str(name)
        self._n_features = self.n_radial_2b + self.n_radial_3b * (self.l_max + 1)

    @property
    def n_features(self) -> int:
        """Descriptor dimension per atom."""
        return self._n_features

    def feature_labels(self) -> list[str]:
        """Human-readable label for every feature, in order."""
        out = [f"2b_k{k}" for k in range(self.n_radial_2b)]
        for q in range(self.n_radial_3b):
            for l in range(self.l_max + 1):
                out.append(f"3b_q{q}_l{l}")
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Bispectrum(r_cut={self.cutoff}, n_radial_2b={self.n_radial_2b}, "
            f"n_radial_3b={self.n_radial_3b}, l_max={self.l_max}, "
            f"n_features={self.n_features})"
        )

    # -- pieces ------------------------------------------------------------

    def _cutoff_function(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine cutoff and its derivative; both vanish at ``r_cut``.

        Returns ``(f, df)`` of shape ``r.shape``, ``f`` dimensionless and ``df``
        in ``1/A``.  Because ``f(r_cut) = f'(r_cut) = 0`` and every feature is a
        product of at least one ``f``, a neighbour crossing the cutoff changes
        neither the descriptor nor its derivative discontinuously.
        """
        s = np.pi / self.cutoff
        f = 0.5 * (1.0 + np.cos(s * r))
        df = -0.5 * s * np.sin(s * r)
        outside = r >= self.cutoff
        return np.where(outside, 0.0, f), np.where(outside, 0.0, df)

    def _radial(self, r: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``psi_k(r) = T_k(x(r)) f_c(r)`` and ``dpsi_k/dr``.

        Parameters
        ----------
        r : ndarray, shape (P,)
            Distances in angstrom.
        n : int
            Number of Chebyshev orders.

        Returns
        -------
        psi : ndarray, shape (P, n)      dimensionless
        dpsi : ndarray, shape (P, n)     1/A
        """
        x = 2.0 * r / self.cutoff - 1.0
        dx_dr = 2.0 / self.cutoff
        t, dt = chebyshev(x, n)
        f, df = self._cutoff_function(r)
        psi = t * f[:, None]
        dpsi = dt * (dx_dr * f)[:, None] + t * df[:, None]
        return psi, dpsi

    # -- the public entry point -------------------------------------------

    def compute(
        self, configuration: Configuration, *, derivatives: bool = True
    ) -> DescriptorOutput:
        """Featurise every atom of ``configuration``.

        Parameters
        ----------
        configuration : Configuration
            Geometry; not modified.
        derivatives : bool
            Also return ``dB/dr`` in the sparse pair layout of
            :class:`~atomlab.models.base.DescriptorOutput`.

        Returns
        -------
        DescriptorOutput
            ``features`` is ``(N, n_features)``, dimensionless.
            ``derivatives`` is ``(P + N, n_features, 3)`` in ``1/A``: one block
            per neighbour pair giving ``dB_i/dr_j``, then one block per atom
            giving the self term ``dB_i/dr_i``.

        Notes
        -----
        The three-body block is accumulated atom by atom.  That Python loop
        looks like a performance mistake and is not: the sum runs over *pairs*
        of neighbours of the same atom, so grouping by centre turns the
        scatter-add that a flat pair layout would need into a plain contraction
        over a dense ``(n_i, n_i)`` block.  The neighbour list is already sorted
        by centre (see :mod:`atomlab.neighbors`), so the blocks are contiguous
        slices.
        """
        n_atoms = configuration.n_atoms
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        dvec, r = pair_vectors(configuration, nl)
        n_pairs = nl.n_pairs

        if n_pairs and np.any(r <= 0.0):
            raise ValueError(
                "Bispectrum found a pair at zero separation; two atoms are "
                "coincident, which makes the bond direction undefined"
            )

        n2 = self.n_radial_2b
        n3 = self.n_radial_3b
        n_ang = self.l_max + 1
        features = np.zeros((n_atoms, self.n_features))
        dfeat = np.zeros((n_pairs, self.n_features, 3)) if derivatives else None

        if n_pairs == 0:
            if not derivatives:
                return DescriptorOutput(features=features)
            idx = np.arange(n_atoms, dtype=np.int32)
            return DescriptorOutput(
                features=features,
                derivatives=np.zeros((n_atoms, self.n_features, 3)),
                pair_i=idx,
                pair_j=idx,
            )

        dhat = dvec / r[:, None]

        # -- two-body block ------------------------------------------------
        psi2, dpsi2 = self._radial(r, n2)
        features[:, :n2] = segment_sum(psi2, nl.i, n_atoms)
        if derivatives:
            dfeat[:, :n2, :] = dpsi2[:, :, None] * dhat[:, None, :]

        # -- three-body block ----------------------------------------------
        psi3, dpsi3 = self._radial(r, n3)
        counts = np.bincount(nl.i, minlength=n_atoms)
        ends = np.cumsum(counts)
        starts = ends - counts

        for i in range(n_atoms):
            lo, hi = int(starts[i]), int(ends[i])
            if hi - lo < 2:
                continue  # a single neighbour forms no pair
            sl = slice(lo, hi)
            e = dhat[sl]  # (n, 3) unit bond vectors
            ri = r[sl]  # (n,)
            pa = psi3[sl]  # (n, n3)
            dpa = dpsi3[sl]  # (n, n3)

            u = e @ e.T  # (n, n) cos(theta)
            # Round-off can push a self-overlap or a collinear pair marginally
            # outside [-1, 1]; the Legendre recurrence is a polynomial and is
            # perfectly happy there, so no clipping is applied -- clipping would
            # introduce a kink with a zero derivative and break the force test.
            pl, dpl = legendre(u, self.l_max)  # (n, n, L+1)
            # Drop the a == b terms: an atom does not form a bond angle with
            # itself.  Zeroing here rather than masking later means the
            # exclusion is carried into every derivative contraction below.
            diag = np.arange(hi - lo)
            pl[diag, diag, :] = 0.0
            dpl[diag, diag, :] = 0.0

            # B_{q,l} = sum_{a != b} psi_q(a) psi_q(b) P_l(u_ab)
            m = np.einsum("bq,abl->aql", pa, pl, optimize=True)  # (n, n3, L+1)
            features[i, n2:] = np.einsum("aq,aql->ql", pa, m, optimize=True).reshape(-1)

            if not derivatives:
                continue

            # term 1: radial derivative on leg a
            t1 = (2.0 * dpa)[:, :, None, None] * m[:, :, :, None] * e[:, None, None, :]
            # term 2: angular derivative on leg a, via du_ab/dd_a
            v1 = np.einsum("bq,abl,bk->aqlk", pa, dpl, e, optimize=True)
            v2 = np.einsum("bq,abl->aql", pa, dpl * u[:, :, None], optimize=True)
            t2 = (2.0 * pa / ri[:, None])[:, :, None, None] * (
                v1 - v2[:, :, :, None] * e[:, None, None, :]
            )
            dfeat[sl, n2:, :] = (t1 + t2).reshape(hi - lo, n3 * n_ang, 3)

        if not derivatives:
            return DescriptorOutput(features=features)

        self_deriv = -segment_sum(dfeat, nl.i, n_atoms)
        pair_i = np.concatenate([nl.i, np.arange(n_atoms, dtype=np.int32)])
        pair_j = np.concatenate([nl.j, np.arange(n_atoms, dtype=np.int32)])
        derivs = np.concatenate([dfeat, self_deriv], axis=0)
        return DescriptorOutput(
            features=features, derivatives=derivs, pair_i=pair_i, pair_j=pair_j
        )


#: The design document names this file's contribution the "polynomial/ACE-style
#: invariant basis"; both spellings are exported so downstream imports do not
#: have to guess which noun was meant.
PolynomialBasis = Bispectrum
ACEBasis = Bispectrum
