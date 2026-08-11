"""Learned **pair-only** potential: a cubic spline in ``r``, fitted by least squares.

.. math::  U(x) = \\sum_{i<j} \\phi_{s_i s_j}(r_{ij}) \\; + \\; \\sum_i c_{s_i}

with :math:`\\phi` a cubic spline on knots between ``r_min`` and the cutoff,
constrained to vanish smoothly at the cutoff (value and first derivative both
zero there, so energy *and* force are continuous when a pair crosses it).

This model is deliberately under-expressive
-------------------------------------------

That is its entire scientific function.  It can represent Lennard-Jones
**exactly** -- LJ is a pair potential, and a spline with enough knots reproduces
any smooth pair function to interpolation accuracy -- and it can represent
Stillinger-Weber **not at all**, because SW's three-body term is not a function
of pair distances alone.  So the model zoo contains a matched pair of controls:

* *right in kind, wrong in detail* -- this spline on Lennard-Jones data, whose
  residual is pure interpolation error, spread smoothly over ``r``, and
  systematically reducible by adding knots;
* *wrong in kind* -- this spline on Stillinger-Weber data, whose residual
  contains an irreducible floor no knot count can touch, because the missing
  physics is angular and the model has no angular coordinate.

Matched at the same force error, those two models are the sharpest available
test of the repository's thesis (``docs/theory.md`` sections 4-5): a norm of the
error cannot distinguish them, but the *structure* of their error fields is
categorically different, and their observable errors should be too.  The SW
plateau in ``tests/test_linear_models.py`` is therefore a **result**, not a
failure of the fit.

Fitting
-------

The energy is linear in the spline coefficients, so -- exactly as for
:mod:`atomlab.models.linear` -- energies and forces give linear equations in the
same unknowns and are stacked into one design matrix and solved by
ridge-regularised least squares through a QR reduction and an explicit SVD (see
:class:`atomlab.models.linear.ReducedSystem`; forming the normal equations would
square an already large condition number).  An optional second-derivative
penalty ``mu * \\int \\phi''(r)^2 dr`` is available: it is the natural
smoothness control for a spline, and it acts on the same axis the ridge
parameter does in the linear model -- the spatial-frequency content of the
fitted function, which ``docs/theory.md`` section 4 identifies as precisely the
axis along which force error and observable error disagree.

Definition outside the fitted window
------------------------------------

``phi`` is defined piecewise and is ``C^1`` everywhere:

* ``r >= r_cut``: exactly zero (the last two B-splines, the only ones with a
  nonzero value or slope at the cutoff, are dropped from the basis);
* ``r_min <= r < r_cut``: the cubic spline;
* ``r < r_min``: the linear continuation ``phi(r_min) + phi'(r_min)(r - r_min)``.

The last clause is part of the model definition, not a fallback: a spline is
undefined below its first knot, and a molecular-dynamics run that briefly
compresses a pair past ``r_min`` must get a finite, continuous force rather than
a crash or a NaN.  Because a linear continuation is still linear in the
coefficients, the fit stays exact even if training pairs fall below ``r_min``.
It is *not* physically trustworthy there -- a linear repulsive wall is far too
soft -- so :meth:`PairSpline.compute` counts such pairs in
``Result.extra["n_below_r_min"]`` and :meth:`PairSpline.fit` records them in the
``FitReport`` notes.

Cost
----

Trivial by the standards of this study: no descriptor, no optimiser.  A fit to a
few thousand 64-atom configurations takes seconds on one core, dominated by
neighbour-list construction.  The basis has ``n_knots + 2`` free coefficients per
species pair (cubic B-splines on ``n_knots`` knots, less the two end
constraints).

Units: ``r`` in A, ``phi`` in eV, forces eV/A, virial eV.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import BSpline

from ..cell import check_minimum_image
from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Dataset, Result
from .base import FitReport, MLModel
from .linear import (
    DEFAULT_RIDGE_GRID,
    ReducedSystem,
    RidgePath,
    StackedDesign,
    _scatter_sum,
)

__all__ = ["PairSpline", "SplineBasis"]


# --------------------------------------------------------------------------
# the basis
# --------------------------------------------------------------------------


class SplineBasis:
    """Clamped B-spline basis on ``[r_min, r_cut]`` with vanishing end conditions.

    Parameters
    ----------
    r_min : float
        Innermost knot in angstrom.  Put it below the shortest pair distance the
        model will ever see; below it the basis functions continue linearly.
    r_cut : float
        Outermost knot, i.e. the interaction cutoff, in angstrom.
    n_knots : int
        Number of knots, including both ends.  Uniformly spaced in ``r``
        (``spacing="uniform"``) or in ``1/r`` (``spacing="inverse"``, which puts
        more resolution on the steep repulsive wall where a pair potential
        varies fastest).
    degree : int
        Spline degree; 3 (cubic) unless you have a reason.
    end_conditions : int
        How many of value, slope, curvature at ``r_cut`` are forced to zero, by
        dropping that many basis functions from the clamped basis.  ``2`` (the
        default) gives ``phi(r_cut) = phi'(r_cut) = 0``, i.e. continuous energy
        and force at the cutoff -- the minimum a potential used in molecular
        dynamics may have (``docs/design.md`` section 4).  ``3`` additionally
        gives ``phi''(r_cut) = 0``, which matters for phonons but costs one
        basis function and prevents the model from reproducing a shifted-force
        reference exactly near the cutoff.
    spacing : {'uniform', 'inverse'}
        Knot placement.

    Attributes
    ----------
    knots : ndarray, shape (n_knots,)
        Knot positions in angstrom.
    n_basis : int
        Number of usable basis functions, ``n_knots + degree - 1 -
        end_conditions``.

    Notes
    -----
    A clamped knot vector repeats both ends ``degree`` times.  At the right end
    exactly one basis function has a nonzero value, exactly two have a nonzero
    slope, and exactly three a nonzero curvature; dropping the last
    ``end_conditions`` of them is therefore an exact way to impose the boundary
    conditions, with no constraint equations and no loss of the partition-of-
    unity structure elsewhere.  ``tests/test_linear_models.py`` checks this
    numerically rather than trusting the argument.
    """

    def __init__(
        self,
        r_min: float,
        r_cut: float,
        n_knots: int,
        *,
        degree: int = 3,
        end_conditions: int = 2,
        spacing: str = "uniform",
    ) -> None:
        self.r_min = float(r_min)
        self.r_cut = float(r_cut)
        self.degree = int(degree)
        self.end_conditions = int(end_conditions)
        self.spacing = str(spacing)

        if not 0.0 < self.r_min < self.r_cut:
            raise ValueError(f"need 0 < r_min < r_cut, got {r_min} and {r_cut}")
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        n_knots = int(n_knots)
        if n_knots < self.degree + 1:
            raise ValueError(f"n_knots must be >= degree + 1 = {self.degree + 1}, got {n_knots}")
        if not 0 <= self.end_conditions <= self.degree:
            raise ValueError(
                f"end_conditions must be in 0..degree ({self.degree}), got {end_conditions}"
            )

        if self.spacing == "uniform":
            self.knots = np.linspace(self.r_min, self.r_cut, n_knots)
        elif self.spacing == "inverse":
            self.knots = 1.0 / np.linspace(1.0 / self.r_min, 1.0 / self.r_cut, n_knots)
        else:
            raise ValueError(f"spacing must be 'uniform' or 'inverse', got {self.spacing!r}")

        k = self.degree
        t = np.concatenate([np.full(k, self.r_min), self.knots, np.full(k, self.r_cut)])
        n_full = t.size - k - 1
        self.n_basis = n_full - self.end_conditions
        if self.n_basis < 1:
            raise ValueError("end conditions leave no free basis function")

        coefficients = np.eye(n_full)[:, : self.n_basis]
        self._spline = BSpline(t, coefficients, k, extrapolate=False)
        self._d1 = self._spline.derivative(1)
        self._d2 = self._spline.derivative(2) if k >= 2 else None
        # Values and slopes at the inner knot, for the linear continuation.
        self._value_min = np.asarray(self._spline(self.r_min), dtype=np.float64)
        self._slope_min = np.asarray(self._d1(self.r_min), dtype=np.float64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SplineBasis(r_min={self.r_min}, r_cut={self.r_cut}, "
            f"n_knots={self.knots.size}, n_basis={self.n_basis}, "
            f"spacing={self.spacing!r})"
        )

    def evaluate(self, r: np.ndarray, *, derivative: bool = True):
        """Basis functions and their radial derivatives at ``r``.

        Parameters
        ----------
        r : ndarray, shape (P,)
            Distances in angstrom.  Values below ``r_min`` use the linear
            continuation; values at or beyond ``r_cut`` give exactly zero.
        derivative : bool
            Also return ``dB/dr``.

        Returns
        -------
        b : ndarray, shape (P, n_basis)
            Dimensionless.
        db : ndarray, shape (P, n_basis) or None
            In ``1/A``.
        """
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        clamped = np.clip(r, self.r_min, self.r_cut)
        b = np.asarray(self._spline(clamped), dtype=np.float64).reshape(r.size, self.n_basis)
        db = np.asarray(self._d1(clamped), dtype=np.float64).reshape(r.size, self.n_basis)

        below = r < self.r_min
        if below.any():
            # Linear continuation: B(r) = B(r_min) + B'(r_min) (r - r_min),
            # which keeps the basis (and therefore the model) C^1 at r_min.
            b[below] = self._value_min[None, :] + self._slope_min[None, :] * (
                r[below] - self.r_min
            )[:, None]
            db[below] = self._slope_min[None, :]

        outside = r >= self.r_cut
        if outside.any():
            b[outside] = 0.0
            db[outside] = 0.0
        return (b, db) if derivative else (b, None)

    def curvature_penalty(self, n_quadrature: int = 4) -> np.ndarray:
        """Gram matrix of the second derivative, ``Omega_bb' = int B''_b B''_b' dr``.

        Parameters
        ----------
        n_quadrature : int
            Gauss-Legendre points per knot span.  For a cubic spline ``B''`` is
            piecewise linear, so the integrand is piecewise quadratic and two
            points are already exact; the default 4 covers higher degrees too.

        Returns
        -------
        ndarray, shape (n_basis, n_basis)
            Symmetric positive semi-definite, in units of ``1/A^3``.
        """
        if self._d2 is None:
            raise ValueError("a curvature penalty needs degree >= 2")
        x, w = np.polynomial.legendre.leggauss(int(n_quadrature))
        lo, hi = self.knots[:-1], self.knots[1:]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        points = (mid[:, None] + half[:, None] * x[None, :]).reshape(-1)
        weights = (half[:, None] * w[None, :]).reshape(-1)
        d2 = np.asarray(self._d2(points), dtype=np.float64).reshape(points.size, self.n_basis)
        return d2.T @ (weights[:, None] * d2)

    def penalty_factor(self, n_quadrature: int = 4, eps: float = 1e-14) -> np.ndarray:
        """Square-root factor ``L`` of the curvature penalty, ``L^T L = Omega``.

        Returned as rows to be appended to a least-squares design matrix, which
        is how a quadratic penalty enters a QR/SVD solve without ever forming a
        normal-equation matrix.  Obtained from a symmetric eigendecomposition
        with negative eigenvalues (round-off only) clipped to zero.

        Parameters
        ----------
        n_quadrature : int
            As in :meth:`curvature_penalty`.
        eps : float
            Eigenvalues below ``eps * max`` are dropped.

        Returns
        -------
        ndarray, shape (rank, n_basis)
        """
        omega = self.curvature_penalty(n_quadrature)
        vals, vecs = np.linalg.eigh(omega)
        keep = vals > eps * max(vals.max(), 0.0)
        return (np.sqrt(vals[keep])[:, None] * vecs[:, keep].T)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class PairSpline(MLModel):
    """Cubic-spline pair potential fitted to energies and forces.

    Parameters
    ----------
    cutoff : float
        Interaction cutoff in angstrom.  ``phi`` and ``dphi/dr`` are exactly
        zero beyond it.
    r_min : float
        Innermost knot in angstrom; see the module docstring for what happens
        below it.
    n_knots : int
        Knots between ``r_min`` and ``cutoff``.  This is the model's capacity
        knob, and the one the Stillinger-Weber plateau test sweeps.
    n_species : int
        Number of species indices.  One independent spline is fitted per
        unordered species pair, so the model has ``n_species (n_species + 1) / 2``
        splines.
    fit_offsets : bool
        Fit a per-species constant energy offset.  Default **False**: a pure
        pair potential has none, and at fixed composition and density the
        constant is nearly degenerate with the mean of the pair block, which
        inflates the condition number for no gain.  Turn it on only when the
        training set spans compositions or densities wide enough to identify it.
    ridge : float
        Default ridge penalty on the column-scaled coefficients.
    smoothness : float
        Strength ``mu`` of the second-derivative penalty
        ``mu * int phi''(r)^2 dr``, in eV^-2 A^3 relative to the (normalised)
        data loss.  Zero disables it.
    degree, end_conditions, spacing
        Passed to :class:`SplineBasis`.
    name : str
        Identifier used in tables and figures.

    Attributes
    ----------
    basis : SplineBasis
    coefficients : ndarray, shape (n_pair_types, n_basis)
        Fitted spline coefficients in eV.
    offsets : ndarray, shape (n_species,)
        Per-species energy offsets in eV (zeros when ``fit_offsets`` is False).
    report : FitReport or None
        Report from the most recent :meth:`fit`.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        r_min: float = 2.0,
        n_knots: int = 20,
        *,
        n_species: int = 1,
        fit_offsets: bool = False,
        ridge: float = 1e-8,
        smoothness: float = 0.0,
        degree: int = 3,
        end_conditions: int = 2,
        spacing: str = "uniform",
        name: str = "pair_spline",
    ) -> None:
        self.cutoff = float(cutoff)
        self.basis = SplineBasis(
            r_min,
            self.cutoff,
            n_knots,
            degree=degree,
            end_conditions=end_conditions,
            spacing=spacing,
        )
        self.n_species = int(n_species)
        if self.n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        self.fit_offsets = bool(fit_offsets)
        self.ridge = float(ridge)
        self.smoothness = float(smoothness)
        if self.smoothness < 0.0:
            raise ValueError(f"smoothness must be >= 0, got {smoothness}")
        self.name = str(name)

        # Unordered species pair -> block index.
        table = np.zeros((self.n_species, self.n_species), dtype=np.int64)
        n_pt = 0
        for a in range(self.n_species):
            for b in range(a, self.n_species):
                table[a, b] = table[b, a] = n_pt
                n_pt += 1
        self._pair_type = table
        self.n_pair_types = n_pt

        self.coefficients = np.zeros((n_pt, self.basis.n_basis))
        self.offsets = np.zeros(self.n_species)
        self.is_fitted = False
        self.report: FitReport | None = None

    # -- bookkeeping -------------------------------------------------------

    @property
    def r_min(self) -> float:
        """Innermost knot in angstrom."""
        return self.basis.r_min

    @property
    def n_basis(self) -> int:
        """Free spline coefficients per species pair."""
        return self.basis.n_basis

    @property
    def n_parameters(self) -> int:
        """Total fitted coefficients."""
        return self.n_pair_types * self.n_basis + (self.n_species if self.fit_offsets else 0)

    @property
    def flat_coefficients(self) -> np.ndarray:
        """``(K,)`` flat coefficient vector: splines first, then offsets."""
        parts = [self.coefficients.reshape(-1)]
        if self.fit_offsets:
            parts.append(self.offsets)
        return np.concatenate(parts)

    def _unpack(self, flat: np.ndarray) -> None:
        n = self.n_pair_types * self.n_basis
        self.coefficients = flat[:n].reshape(self.n_pair_types, self.n_basis).copy()
        self.offsets = (
            flat[n : n + self.n_species].copy() if self.fit_offsets else np.zeros(self.n_species)
        )

    def set_coefficients(self, flat: np.ndarray, *, ridge: float | None = None) -> None:
        """Install a flat coefficient vector (e.g. one row of a ridge path)."""
        flat = np.asarray(flat, dtype=np.float64).reshape(-1)
        if flat.shape[0] != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} coefficients, got {flat.shape[0]}")
        self._unpack(flat)
        if ridge is not None:
            self.ridge = float(ridge)
        self.is_fitted = True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "untrained"
        return (
            f"PairSpline(name={self.name!r}, r_min={self.r_min}, cutoff={self.cutoff}, "
            f"n_knots={self.basis.knots.size}, n_basis={self.n_basis}, {state})"
        )

    # -- the pair function -------------------------------------------------

    def pair(self, r, type_i=0, type_j=0) -> tuple[np.ndarray, np.ndarray]:
        """Fitted pair energy ``phi(r)`` and its derivative.

        Deliberately mirrors
        :meth:`atomlab.potentials.lennard_jones.PairPotential.pair`, so a fitted
        spline can be compared with a reference pair function directly rather
        than through an energy difference.

        Parameters
        ----------
        r : array_like
            Distances in angstrom.
        type_i, type_j : int or array_like
            Species type indices, broadcast against ``r``.

        Returns
        -------
        phi : ndarray
            Pair energy in eV; exactly zero for ``r >= cutoff``.
        dphi : ndarray
            ``dphi/dr`` in eV/A; exactly zero for ``r >= cutoff``.
        """
        self._require_fitted()
        r_arr = np.asarray(r, dtype=np.float64)
        ti = np.broadcast_to(np.asarray(type_i, dtype=np.int64), r_arr.shape).reshape(-1)
        tj = np.broadcast_to(np.asarray(type_j, dtype=np.int64), r_arr.shape).reshape(-1)
        if ti.size and (ti.max() >= self.n_species or tj.max() >= self.n_species):
            raise ValueError(f"{self.name} is parameterised for {self.n_species} species only")
        b, db = self.basis.evaluate(r_arr.reshape(-1))
        coef = self.coefficients[self._pair_type[ti, tj]]  # (P, n_basis)
        phi = np.einsum("pb,pb->p", b, coef)
        dphi = np.einsum("pb,pb->p", db, coef)
        return phi.reshape(r_arr.shape), dphi.reshape(r_arr.shape)

    # -- the Potential interface -------------------------------------------

    def _validate(self, configuration: Configuration) -> None:
        if configuration.species.size and int(configuration.species.max()) >= self.n_species:
            raise ValueError(
                f"configuration contains species index {int(configuration.species.max())} "
                f"but {self.name} was built for {self.n_species} species"
            )
        if np.asarray(configuration.pbc).any():
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

    def _pairs(self, configuration: Configuration):
        """Half neighbour list plus geometry: ``(pi, pj, D, r, pair_type)``."""
        nl = build_neighbor_list(configuration, self.cutoff, half=True)
        d_vec, r = pair_vectors(configuration, nl)
        if r.size and np.any(r <= 0.0):
            raise ValueError(
                f"{self.name} found a pair at zero separation; two atoms are coincident"
            )
        species = configuration.species
        pt = self._pair_type[species[nl.i], species[nl.j]]
        return nl.i, nl.j, d_vec, r, pt

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Evaluate energy, forces and virial.

        Parameters
        ----------
        configuration : Configuration
            Geometry; not modified.
        forces, virial : bool
            Both come from the same pair loop, so neither is skipped.

        Returns
        -------
        Result
            ``energy`` eV, ``forces`` ``(N, 3)`` eV/A, ``virial`` ``(3, 3)`` eV,
            ``energies`` ``(N,)`` eV with each pair split evenly between its two
            atoms, and ``extra["n_below_r_min"]`` counting pairs evaluated in the
            linear-continuation region.
        """
        self._require_fitted()
        self._validate(configuration)
        n = configuration.n_atoms
        pi, pj, d_vec, r, pt = self._pairs(configuration)

        energies = np.zeros(n)
        f_out = np.zeros((n, 3))
        w_out = np.zeros((3, 3)) if virial else None
        energy = 0.0
        n_below = 0

        if r.size:
            b, db = self.basis.evaluate(r)
            coef = self.coefficients[pt]
            phi = np.einsum("pb,pb->p", b, coef)
            dphi = np.einsum("pb,pb->p", db, coef)
            energy = float(phi.sum())
            np.add.at(energies, pi, 0.5 * phi)
            np.add.at(energies, pj, 0.5 * phi)
            n_below = int(np.count_nonzero(r < self.basis.r_min))

            # Force on i is +phi'(r) D/r with D pointing from i to j; force on j
            # is its negative.  Same convention as every pair potential in the
            # package (see atomlab.potentials.lennard_jones._compute_numpy).
            fvec = (dphi / r)[:, None] * d_vec
            np.add.at(f_out, pi, fvec)
            np.add.at(f_out, pj, -fvec)
            if virial:
                # W_ab = sum_{i<j} f_ij (x) r_ij with f_ij the force on i from j
                # and r_ij = r_i - r_j = -D, hence the minus sign.
                w_out = -np.einsum("pa,pb->ab", fvec, d_vec)

        if self.fit_offsets:
            per_species = self.offsets[configuration.species]
            energies = energies + per_species
            energy += float(per_species.sum())

        return Result(
            energy=energy,
            forces=f_out,
            virial=w_out,
            energies=energies,
            extra={"n_below_r_min": n_below},
        )

    # -- design matrix -----------------------------------------------------

    def design(
        self,
        configurations: Iterable[Configuration],
        *,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        require_forces: bool = True,
    ) -> StackedDesign:
        """Assemble the stacked energy + force design matrix.

        Parameters
        ----------
        configurations : iterable of Configuration
            Labelled configurations.
        energy_weight, force_weight : float
            Block weights; see :meth:`atomlab.models.linear.LinearPotential.fit`
            for the normalisation, which is identical here.
        require_forces : bool
            Include the ``3N`` force rows per configuration.

        Returns
        -------
        StackedDesign
            Energy rows in eV/atom first, then force rows in eV/A.
        """
        configs = list(configurations)
        if not configs:
            raise ValueError("no configurations to fit")
        nb = self.n_basis
        n_pt = self.n_pair_types
        k = self.n_parameters

        n_atoms = np.array([c.n_atoms for c in configs], dtype=int)
        n_e = len(configs)
        n_f = 3 * int(n_atoms.sum()) if require_forces else 0

        matrix = np.zeros((n_e + n_f, k))
        target = np.zeros(n_e + n_f)
        row = n_e
        self._n_below_r_min_fit = 0
        for m, cfg in enumerate(configs):
            if cfg.energy is None or (require_forces and cfg.forces is None):
                raise ValueError(
                    f"{self.name}.fit needs labelled configurations; configuration "
                    f"{m} is missing {'forces' if cfg.energy is not None else 'an energy'}"
                )
            self._validate(cfg)
            n = cfg.n_atoms
            pi, pj, d_vec, r, pt = self._pairs(cfg)
            b, db = self.basis.evaluate(r)
            self._n_below_r_min_fit += int(np.count_nonzero(r < self.basis.r_min))

            # -- energy row, eV/atom ---------------------------------------
            summed = _scatter_sum(pt, b, n_pt)  # (n_pair_types, n_basis)
            matrix[m, : n_pt * nb] = summed.reshape(-1) / n
            if self.fit_offsets:
                matrix[m, n_pt * nb :] = np.bincount(cfg.species, minlength=self.n_species) / n
            target[m] = cfg.energy / n

            if not require_forces:
                continue

            # -- force rows -------------------------------------------------
            # dU/dr_j = phi'(r) D_hat, so F_j = -sum_b c_b B'_b(r) D_hat and the
            # design entry for coefficient b at row (j, a) is -B'_b(r) D_hat_a;
            # atom i gets the opposite sign.
            dhat = d_vec / r[:, None]
            block = (db[:, :, None] * dhat[:, None, :]).reshape(r.size, nb * 3)
            index = np.concatenate([pj.astype(np.int64) * n_pt + pt, pi.astype(np.int64) * n_pt + pt])
            values = np.concatenate([-block, block], axis=0)
            acc = _scatter_sum(index, values, n * n_pt)
            acc = acc.reshape(n, n_pt, nb, 3).transpose(0, 3, 1, 2).reshape(3 * n, n_pt * nb)
            matrix[row : row + 3 * n, : n_pt * nb] = acc
            target[row : row + 3 * n] = cfg.forces.reshape(-1)
            row += 3 * n

        weight = np.empty(n_e + n_f)
        weight[:n_e] = energy_weight / np.sqrt(n_e)
        if n_f:
            weight[n_e:] = force_weight / np.sqrt(n_f)
        return StackedDesign(
            matrix=matrix,
            target=target,
            weight=weight,
            n_energy=n_e,
            n_force=n_f,
            n_atoms=n_atoms,
        )

    def _penalty_rows(self) -> np.ndarray | None:
        """Curvature-penalty rows for the full coefficient vector, or ``None``."""
        if self.smoothness <= 0.0:
            return None
        factor = self.basis.penalty_factor()
        rows = np.zeros((factor.shape[0] * self.n_pair_types, self.n_parameters))
        for p in range(self.n_pair_types):
            r0 = p * factor.shape[0]
            c0 = p * self.n_basis
            rows[r0 : r0 + factor.shape[0], c0 : c0 + self.n_basis] = factor
        return np.sqrt(self.smoothness) * rows

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        train: Dataset,
        *,
        val: Dataset | None = None,
        ridge: float | str | None = None,
        ridge_grid: Sequence[float] | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        fit_forces: bool = True,
        scale_columns: bool = True,
    ) -> FitReport:
        """Fit energies and forces jointly by ridge-regularised least squares.

        Parameters
        ----------
        train : Dataset
            Labelled configurations.
        val : Dataset, optional
            Held-out set for the reported validation errors and for
            ``ridge='heldout'``.
        ridge : float or {'gcv', 'heldout'}, optional
            Penalty on the column-scaled coefficients; defaults to the
            constructor value.
        ridge_grid : sequence of float, optional
            Grid for the selection routes; defaults to
            :data:`atomlab.models.linear.DEFAULT_RIDGE_GRID`.
        energy_weight, force_weight : float
            Block weights, normalised exactly as in
            :meth:`atomlab.models.linear.LinearPotential.fit`.
        fit_forces : bool
            Include force rows (default True).
        scale_columns : bool
            Scale design columns to unit RMS before solving.

        Returns
        -------
        FitReport
            ``condition_number`` is that of the column-scaled weighted design
            matrix; ``history`` carries the ridge path, the raw condition number
            and the singular-value spectrum.

        Notes
        -----
        Deterministic: no random numbers are used, so repeated fits on the same
        data give bitwise identical coefficients.
        """
        t0 = time.perf_counter()
        notes: list[str] = []
        if ridge is None:
            ridge = self.ridge
        grid = np.asarray(DEFAULT_RIDGE_GRID if ridge_grid is None else ridge_grid, float)

        design = self.design(
            train,
            energy_weight=energy_weight,
            force_weight=force_weight,
            require_forces=fit_forces,
        )
        n_below = self._n_below_r_min_fit
        if n_below:
            notes.append(
                f"{n_below} training pairs lie below r_min = {self.r_min} A and were fitted "
                "through the linear continuation, which is not a physical repulsive wall; "
                "lower r_min if this is more than a handful"
            )
        if not fit_forces:
            notes.append(
                "fitted on energies only: the model's gradient is unconstrained by the data"
            )

        val_design = None
        if val is not None and len(list(val)):
            val_design = self.design(
                val,
                energy_weight=energy_weight,
                force_weight=force_weight,
                require_forces=fit_forces,
            )

        penalized = np.ones(self.n_parameters, dtype=bool)
        if self.fit_offsets:
            penalized[self.n_pair_types * self.n_basis :] = False

        system = ReducedSystem(
            design.matrix,
            design.target,
            weights=design.weight,
            penalized=penalized,
            extra_penalty=self._penalty_rows(),
            scale_columns=scale_columns,
        )

        history: dict = {
            "singular_values": system.singular_values,
            "condition_number_raw": [system.raw_condition_number],
        }

        if isinstance(ridge, str):
            criterion = ridge.lower()
            if criterion not in ("gcv", "heldout"):
                raise ValueError(f"ridge must be a float, 'gcv' or 'heldout', got {ridge!r}")
            heldout = None
            if criterion == "heldout":
                if val_design is None:
                    raise ValueError("ridge='heldout' needs a non-empty `val` dataset")
                heldout = (val_design.matrix, val_design.target, val_design.weight)
            path = system.path(grid, heldout=heldout)
            best = path.best(criterion)
            solution = system.solve(float(path.ridge[best]))
            history.update(path.as_dict())
            history["selected_ridge"] = [float(path.ridge[best])]
            if best in (0, len(path.ridge) - 1):
                notes.append(
                    f"the {criterion} optimum sits at the edge of the ridge grid "
                    f"({path.ridge[best]:.3g}); widen `ridge_grid` before trusting it"
                )
        else:
            solution = system.solve(float(ridge))
            history["selected_ridge"] = [solution.ridge]

        self._unpack(solution.coefficients)
        self.ridge = solution.ridge
        self.is_fitted = True

        train_e, train_f = design.rmse(solution.coefficients)
        if val_design is not None:
            val_e, val_f = val_design.rmse(solution.coefficients)
        else:
            val_e, val_f = train_e, train_f
            notes.append("no validation set: the reported 'val' metrics are training-set values")

        if solution.rank < self.n_parameters:
            notes.append(
                f"the regularised system is rank {solution.rank} of {self.n_parameters}: "
                "some spline coefficients are not identified by this data (typically "
                "knots in a distance range the training set never samples)"
            )

        report = FitReport(
            converged=True,
            n_epochs=1,
            wall_seconds=time.perf_counter() - t0,
            n_parameters=self.n_parameters,
            n_train=design.n_energy,
            n_val=0 if val_design is None else val_design.n_energy,
            train_energy_rmse=train_e,
            train_force_rmse=train_f,
            val_energy_rmse=val_e,
            val_force_rmse=val_f,
            condition_number=system.condition_number,
            history=history,
            notes=notes,
        )
        self.report = report
        return report

    def ridge_sweep(
        self,
        train: Dataset,
        ridges: Sequence[float],
        *,
        val: Dataset | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
    ) -> RidgePath:
        """Evaluate a ridge path without changing the fitted state.

        Parameters
        ----------
        train : Dataset
            Labelled configurations.
        ridges : sequence of float
            Penalties to evaluate.
        val : Dataset, optional
            Held-out set; its weighted residual is reported per ridge value.
        energy_weight, force_weight : float
            As in :meth:`fit`.

        Returns
        -------
        RidgePath
        """
        design = self.design(train, energy_weight=energy_weight, force_weight=force_weight)
        penalized = np.ones(self.n_parameters, dtype=bool)
        if self.fit_offsets:
            penalized[self.n_pair_types * self.n_basis :] = False
        system = ReducedSystem(
            design.matrix,
            design.target,
            weights=design.weight,
            penalized=penalized,
            extra_penalty=self._penalty_rows(),
        )
        heldout = None
        if val is not None and len(list(val)):
            vd = self.design(val, energy_weight=energy_weight, force_weight=force_weight)
            heldout = (vd.matrix, vd.target, vd.weight)
        return system.path(ridges, heldout=heldout)
