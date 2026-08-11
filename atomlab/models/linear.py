"""Linear (ridge) potential on an invariant descriptor -- the "ACE-lite" baseline.

The model is

.. math::  U(x) = \\sum_i \\big[ \\mathbf{w}_{s_i} \\cdot \\mathbf{G}_i(x) + c_{s_i} \\big]

with :math:`\\mathbf{G}_i` the invariant descriptor of atom ``i`` (any
:class:`~atomlab.models.base.Descriptor`; the default is
:class:`~atomlab.models.descriptors.bispectrum.Bispectrum`), :math:`s_i` its
species, and :math:`(\\mathbf{w}_s, c_s)` the fitted coefficients.

Why this model is in the zoo
----------------------------

Not for its accuracy.  It is here because **its error structure is fully
understood**, which is what makes it the control the neural models are read
against:

* The energy is linear in the coefficients, so the forces
  :math:`\\mathbf{F}_j = -\\sum_{i,d} w_{s_i d}\\, \\partial G_{id}/\\partial
  \\mathbf{r}_j` are linear in the *same* coefficients.  Energy and force labels
  therefore give linear equations in one unknown vector and can be stacked into a
  single design matrix -- no optimiser, no seed, no convergence question, and a
  fit that is reproducible to the last bit.
* The residual error field ``delta U = U_model - U_true`` lives, by
  construction, in the span of the basis functions the descriptor omits.  Every
  other model in this study has an error structure one can only measure; this
  one has an error structure one can reason about.

Ridge, smoothness, and why the sweep is an instrument
----------------------------------------------------

The ridge parameter is exposed, swept and reported rather than tuned once and
forgotten.  ``docs/theory.md`` section 4 decomposes the error field into spatial
frequencies and shows that force error weights mode ``k`` by ``k^2`` while an
observable's response weights it by ``Cov(A, phi_k)``, which *decays* with
``k``.  Ridge regression moves the fitted model along exactly that axis: a large
ridge suppresses the high-frequency, large-coefficient components of the fit and
leaves a smooth, low-frequency residual (the "false confidence" regime -- small
force error, potentially large observable error), while a small ridge lets the
fit chase high-frequency structure (the "false alarm" regime).  Sweeping the
ridge therefore generates a family of models that differ in *where* their error
lives at roughly matched magnitude, which is precisely what experiments
``exp05``/``exp07`` need.  It is a knob on the physics, not a nuisance
hyper-parameter.

Numerics
--------

The stacked design matrix is ill-conditioned: descriptor columns are strongly
correlated (a bond-length polynomial basis is very far from orthogonal on a
condensed-phase distance distribution), and the energy and force blocks probe
different linear combinations.  Condition numbers of ``1e6``-``1e10`` are normal
here.  Consequently:

* The solve goes through an **orthogonal reduction (Householder QR) followed by
  an explicit SVD**, never through the normal equations ``G^T G``.  Forming
  ``G^T G`` squares the condition number, which at ``cond(G) = 1e9`` means a
  numerically singular matrix in double precision and coefficients that are
  noise.
* The condition number that governs the fit is reported in
  :class:`~atomlab.models.base.FitReport`, both for the column-scaled matrix
  that is actually solved and for the raw one.
* Columns are scaled to unit RMS before the solve.  A shared ridge parameter is
  meaningless otherwise: descriptor components differ by orders of magnitude in
  scale, and an unscaled penalty would silently regularise only the small ones.

The QR/SVD machinery (:class:`ReducedSystem`) is written once here and reused by
:mod:`atomlab.models.pair_spline`, which poses the same mathematical problem
with a different basis.

Cost
----

Sized for 4 CPU cores, per ``docs/design.md`` section 5.5.  The expensive step is
the descriptor evaluation, not the algebra: for ``M`` configurations of ``N``
atoms with ``D`` features, the design matrix is ``M(1 + 3N) x (n_species D + 1)``
and its QR costs ``O(M N K^2)`` -- a few seconds for a few thousand 64-atom
configurations at ``D = 28``.  Once reduced, an entire ridge path costs
milliseconds, because every ridge value reuses the same ``K x K`` triangular
factor.

Units: energies eV, forces eV/A, positions A.  Descriptor features are
dimensionless (see :mod:`atomlab.models.descriptors.bispectrum`), so the
coefficients ``w`` are in eV and ``c`` is in eV/atom.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse

from ..cell import check_minimum_image, minimum_image
from ..types import Configuration, Dataset, Result
from .base import Descriptor, DescriptorOutput, FitReport, MLModel

__all__ = [
    "LinearPotential",
    "ReducedSystem",
    "RidgeSolution",
    "RidgePath",
    "StackedDesign",
    "DEFAULT_RIDGE_GRID",
]

#: Default ridge grid for the GCV / held-out sweeps.  The design matrix is
#: column-scaled to unit RMS and the row weights are normalised (see
#: :meth:`LinearPotential.fit`), so the useful range of the penalty is
#: dimensionless and spans roughly ``1e-12`` (numerically unregularised) to
#: ``1e0`` (coefficients crushed to a smooth mean field).
DEFAULT_RIDGE_GRID = np.logspace(-12.0, 0.0, 25)


# --------------------------------------------------------------------------
# the numerical core: QR reduction + SVD ridge solve
# --------------------------------------------------------------------------


@dataclass
class RidgeSolution:
    """One point of a ridge path.

    Attributes
    ----------
    coefficients : ndarray, shape (K,)
        Solution in the caller's (unscaled) coefficient space.
    ridge : float
        The penalty strength used, acting on the column-scaled coefficients.
    rss : float
        Weighted residual sum of squares on the **data** rows only (any
        smoothness rows folded into the reduction are subtracted off).
    effective_dof : float
        ``tr[A (A^T A + lambda P^T P)^{-1} A^T]``, the effective number of
        parameters the fit actually used.  Runs from ``K`` at zero ridge down
        towards the number of unpenalised columns as the ridge grows.
    gcv : float
        Generalised cross-validation score ``n rss / (n - dof)^2``; lower is
        better.  ``inf`` when ``dof >= n``.
    coefficient_norm : float
        Euclidean norm of the penalised part of the scaled coefficient vector.
        This is the quantity ridge regression is guaranteed to shrink
        monotonically as the penalty grows.
    rank : int
        Numerical rank of the augmented system.
    """

    coefficients: np.ndarray
    ridge: float
    rss: float
    effective_dof: float
    gcv: float
    coefficient_norm: float
    rank: int


@dataclass
class RidgePath:
    """A whole ridge sweep, evaluated from a single QR reduction.

    Attributes
    ----------
    ridge : ndarray, shape (L,)
        Penalty values, ascending.
    coefficients : ndarray, shape (L, K)
        Solution for each penalty, in unscaled coefficient space.
    rss, effective_dof, gcv, coefficient_norm : ndarray, shape (L,)
        Per-penalty diagnostics; see :class:`RidgeSolution`.
    heldout_rss : ndarray, shape (L,) or None
        Weighted residual sum of squares on a held-out design, when one was
        supplied.
    """

    ridge: np.ndarray
    coefficients: np.ndarray
    rss: np.ndarray
    effective_dof: np.ndarray
    gcv: np.ndarray
    coefficient_norm: np.ndarray
    heldout_rss: np.ndarray | None = None

    def best(self, criterion: str = "gcv") -> int:
        """Index of the best penalty under ``criterion`` (``'gcv'``/``'heldout'``)."""
        if criterion == "gcv":
            return int(np.argmin(self.gcv))
        if criterion == "heldout":
            if self.heldout_rss is None:
                raise ValueError("no held-out design was supplied to the path")
            return int(np.argmin(self.heldout_rss))
        raise ValueError(f"criterion must be 'gcv' or 'heldout', got {criterion!r}")

    def as_dict(self) -> dict:
        """Plain-array view, for the ``history`` field of a ``FitReport``."""
        out = {
            "ridge": self.ridge,
            "rss": self.rss,
            "effective_dof": self.effective_dof,
            "gcv": self.gcv,
            "coefficient_norm": self.coefficient_norm,
        }
        if self.heldout_rss is not None:
            out["heldout_rss"] = self.heldout_rss
        return out


class ReducedSystem:
    """Orthogonal reduction of a weighted, ridge-regularised least-squares problem.

    Solves, for any penalty ``lambda``,

    .. math::  \\min_x \\; \\|W(Ax - b)\\|^2 + \\lambda \\|P x_s\\|^2 + \\|E x\\|^2

    where ``W`` is a diagonal row weighting, ``x_s`` the column-scaled
    coefficients, ``P`` selects the penalised columns, and ``E`` is an optional
    fixed extra penalty (used for the second-derivative smoothness term of
    :class:`~atomlab.models.pair_spline.PairSpline`).

    The reduction is a Householder QR of the *augmented* matrix ``[WA | Wb]``.
    Writing that QR as ``[WA | Wb] = Q R_f`` gives, exactly,

    .. math::  \\|W(Ax-b)\\|^2 = \\|R x - c\\|^2 + \\rho_0^2

    with ``R = R_f[:K,:K]``, ``c = R_f[:K,K]`` and ``rho_0 = R_f[K,K]``.  Every
    subsequent operation -- singular values, condition number, the whole ridge
    path -- then happens in the ``K x K`` triangular factor, at negligible cost
    and, crucially, *without ever forming* ``A^T A``.  That matters here and is
    not pedantry: these design matrices routinely have ``cond ~ 1e8``, so the
    normal equations would have ``cond ~ 1e16`` and return noise.

    Parameters
    ----------
    design : ndarray, shape (M, K)
        Design matrix, unweighted, in the caller's coefficient units.
    target : ndarray, shape (M,)
        Right-hand side, unweighted.
    weights : ndarray, shape (M,), optional
        Per-row weights ``W``.  Applied as ``W A``, ``W b``, i.e. the weight
        multiplies the residual, so the loss is ``sum_m w_m^2 r_m^2``.
    penalized : ndarray of bool, shape (K,), optional
        Which columns the ridge acts on.  Defaults to all.  Unpenalised columns
        (per-species energy offsets) must be excluded: shrinking a constant
        energy offset towards zero is not regularisation, it is a bias of order
        the cohesive energy.
    extra_penalty : ndarray, shape (Kx, K), optional
        Additional fixed penalty rows in *unscaled* coefficient space, already
        multiplied by the square root of their strength.  Contributes
        ``\\|E x\\|^2`` to the objective at every ridge value.
    scale_columns : bool
        Scale each column to unit RMS before the solve (default True).  Without
        it a single ridge value regularises descriptor components of different
        magnitude by wildly different relative amounts.

    Attributes
    ----------
    n_rows : int
        Number of data rows ``M``.
    n_columns : int
        Number of coefficients ``K``.
    scale : ndarray, shape (K,)
        Column scales that were divided out.
    singular_values : ndarray, shape (K,)
        Singular values of the weighted, column-scaled data design matrix.
    raw_singular_values : ndarray, shape (K,)
        Singular values of the weighted design matrix before column scaling.
    condition_number : float
        ``s_max / s_min`` of the matrix actually solved (column-scaled).
    raw_condition_number : float
        The same for the unscaled matrix -- typically orders of magnitude
        larger, which is the honest measure of how badly posed the raw problem
        is.
    """

    def __init__(
        self,
        design: np.ndarray,
        target: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        penalized: np.ndarray | None = None,
        extra_penalty: np.ndarray | None = None,
        scale_columns: bool = True,
    ) -> None:
        a = np.asarray(design, dtype=np.float64)
        b = np.asarray(target, dtype=np.float64).reshape(-1)
        if a.ndim != 2:
            raise ValueError(f"design must be 2-D, got shape {a.shape}")
        m, k = a.shape
        if b.shape[0] != m:
            raise ValueError(f"target has {b.shape[0]} rows, design has {m}")
        if m < k + 1:
            raise ValueError(
                f"least squares needs at least K+1 = {k + 1} rows, got {m}; "
                "with fewer rows than parameters the fit is not identified and "
                "reporting a condition number for it would be meaningless"
            )

        w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
        if w is not None and w.shape[0] != m:
            raise ValueError(f"weights has {w.shape[0]} entries, design has {m} rows")

        # -- column scaling, measured on the weighted data rows --------------
        if scale_columns:
            wa_sq = (a * a) if w is None else (a * a) * (w * w)[:, None]
            scale = np.sqrt(wa_sq.mean(axis=0))
            # A column that is identically zero over the whole training set
            # carries no information; leave it unscaled rather than dividing by
            # a numerical zero and manufacturing noise (cf. models.base.standardize).
            scale = np.where(scale > 0.0, scale, 1.0)
        else:
            scale = np.ones(k)

        # -- QR of [WA/scale | Wb] ------------------------------------------
        # Built in one preallocated block so that the peak memory is a single
        # copy of the design matrix rather than three.
        z = np.empty((m, k + 1))
        np.divide(a, scale, out=z[:, :k])
        z[:, k] = b
        if w is not None:
            z *= w[:, None]
        r_data = np.linalg.qr(z, mode="r")  # (K+1, K+1)
        del z

        r_block = np.ascontiguousarray(r_data[:k, :k])
        self.singular_values = np.linalg.svd(r_block, compute_uv=False)
        # R_scaled @ diag(scale) is the triangular factor of the *unscaled*
        # weighted design (Q is unchanged by a column scaling applied on the
        # right), so its singular values come for free.
        self.raw_singular_values = np.linalg.svd(r_block * scale[None, :], compute_uv=False)

        # -- fold in the fixed extra penalty --------------------------------
        if extra_penalty is not None:
            e = np.asarray(extra_penalty, dtype=np.float64)
            if e.ndim != 2 or e.shape[1] != k:
                raise ValueError(f"extra_penalty must be (Kx, {k}), got {e.shape}")
            self._extra_scaled = e / scale
            stack = np.zeros((r_data.shape[0] + e.shape[0], k + 1))
            stack[: r_data.shape[0]] = r_data
            stack[r_data.shape[0] :, :k] = self._extra_scaled
            r_full = np.linalg.qr(stack, mode="r")
        else:
            self._extra_scaled = None
            r_full = r_data

        self.r = np.ascontiguousarray(r_full[:k, :k])
        self.c = np.ascontiguousarray(r_full[:k, k])
        #: Residual of the reduction: the part of ``Wb`` orthogonal to the
        #: column space, which no coefficient vector can remove.
        self.rss_floor = float(r_full[k, k] ** 2)

        self.scale = scale
        self.n_rows = int(m)
        self.n_columns = int(k)
        self.penalized = (
            np.ones(k, dtype=bool) if penalized is None else np.asarray(penalized, dtype=bool)
        )
        if self.penalized.shape != (k,):
            raise ValueError(f"penalized must be ({k},), got {self.penalized.shape}")

    # -- diagnostics -------------------------------------------------------

    @staticmethod
    def _cond(s: np.ndarray) -> float:
        return float(s[0] / s[-1]) if s[-1] > 0.0 else float("inf")

    @property
    def condition_number(self) -> float:
        """Condition number of the column-scaled weighted design matrix."""
        return self._cond(self.singular_values)

    @property
    def raw_condition_number(self) -> float:
        """Condition number before column scaling."""
        return self._cond(self.raw_singular_values)

    # -- the solve ---------------------------------------------------------

    def solve(self, ridge: float) -> RidgeSolution:
        """Solve at one ridge value by an explicit SVD of the augmented system.

        Parameters
        ----------
        ridge : float
            Penalty ``lambda >= 0`` on the column-scaled penalised columns.

        Returns
        -------
        RidgeSolution

        Notes
        -----
        The augmented matrix ``B = [R ; sqrt(lambda) P]`` is ``(K + Kp, K)`` --
        tiny -- so a full SVD of it is free, and the minimum-norm solution it
        gives is well defined even when the problem is rank-deficient (a
        descriptor column that no training configuration excites).  The
        effective degrees of freedom follow from the same factorisation:
        ``tr H = ||R V S^{-1}||_F^2``.
        """
        ridge = float(ridge)
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
        k = self.n_columns
        idx = np.flatnonzero(self.penalized)
        n_pen = idx.size

        b_aug = np.zeros((k + n_pen, k))
        b_aug[:k] = self.r
        if n_pen:
            b_aug[np.arange(k, k + n_pen), idx] = np.sqrt(ridge)
        d_aug = np.zeros(k + n_pen)
        d_aug[:k] = self.c

        u, s, vt = np.linalg.svd(b_aug, full_matrices=False)
        tol = max(b_aug.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        keep = s > tol
        rank = int(keep.sum())
        v_inv = vt[keep].T / s[keep]  # (K, rank)
        x_scaled = v_inv @ (u[:, keep].T @ d_aug)

        # Effective dof: tr[A (A^T A + lambda P^T P)^{-1} A^T] = ||R V S^{-1}||_F^2.
        g = self.r @ v_inv
        dof = float((g * g).sum())

        resid = self.r @ x_scaled - self.c
        rss = float(resid @ resid) + self.rss_floor
        if self._extra_scaled is not None:
            # The smoothness rows were folded into R, so subtract their share to
            # leave the residual on the data rows alone.
            pen = self._extra_scaled @ x_scaled
            rss -= float(pen @ pen)
        rss = max(rss, 0.0)

        n = self.n_rows
        gcv = float(n * rss / (n - dof) ** 2) if dof < n else float("inf")
        pen_norm = float(np.linalg.norm(x_scaled[self.penalized]))
        return RidgeSolution(
            coefficients=x_scaled / self.scale,
            ridge=ridge,
            rss=rss,
            effective_dof=dof,
            gcv=gcv,
            coefficient_norm=pen_norm,
            rank=rank,
        )

    def path(
        self,
        ridges: Sequence[float],
        *,
        heldout: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    ) -> RidgePath:
        """Solve at every ridge in ``ridges``, reusing this one reduction.

        Parameters
        ----------
        ridges : sequence of float
            Penalty values.  Sorted ascending in the returned path.
        heldout : tuple, optional
            ``(design, target, weights)`` of a held-out set; its weighted
            residual sum of squares is evaluated for every ridge value, giving
            the cross-validation route to selecting the penalty.

        Returns
        -------
        RidgePath
        """
        grid = np.sort(np.asarray(ridges, dtype=np.float64).reshape(-1))
        sols = [self.solve(lam) for lam in grid]
        coefs = np.array([s.coefficients for s in sols])
        held = None
        if heldout is not None:
            a_h, b_h, w_h = heldout
            resid = coefs @ np.asarray(a_h).T - np.asarray(b_h)[None, :]
            if w_h is not None:
                resid = resid * np.asarray(w_h)[None, :]
            held = (resid**2).sum(axis=1)
        return RidgePath(
            ridge=grid,
            coefficients=coefs,
            rss=np.array([s.rss for s in sols]),
            effective_dof=np.array([s.effective_dof for s in sols]),
            gcv=np.array([s.gcv for s in sols]),
            coefficient_norm=np.array([s.coefficient_norm for s in sols]),
            heldout_rss=held,
        )


# --------------------------------------------------------------------------
# assembled design matrix for a dataset
# --------------------------------------------------------------------------


@dataclass
class StackedDesign:
    """Stacked energy + force design matrix for one dataset.

    ``matrix`` is ``(M, K)`` with ``M = n_config + 3 * n_atoms_total`` rows:
    one energy row per configuration (in eV/atom, i.e. already divided by ``N``)
    followed by all force rows (eV/A).  ``weight`` carries the block weighting
    described in :meth:`LinearPotential.fit`.
    """

    matrix: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    n_energy: int
    n_force: int
    n_atoms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def energy_slice(self) -> slice:
        return slice(0, self.n_energy)

    @property
    def force_slice(self) -> slice:
        return slice(self.n_energy, self.n_energy + self.n_force)

    def rmse(self, coefficients: np.ndarray) -> tuple[float, float]:
        """``(energy RMSE in eV/atom, force RMSE in eV/A)`` for ``coefficients``."""
        pred = self.matrix @ coefficients
        de = pred[self.energy_slice] - self.target[self.energy_slice]
        df = pred[self.force_slice] - self.target[self.force_slice]
        e_rmse = float(np.sqrt((de**2).mean())) if de.size else float("nan")
        f_rmse = float(np.sqrt((df**2).mean())) if df.size else float("nan")
        return e_rmse, f_rmse


def _scatter_sum(index: np.ndarray, values: np.ndarray, n_rows: int) -> np.ndarray:
    """``out[index[p]] += values[p]``, vectorised.

    Parameters
    ----------
    index : ndarray, shape (P,)
        Destination row of each block of values.
    values : ndarray, shape (P, C)
    n_rows : int

    Returns
    -------
    ndarray, shape (n_rows, C)

    Notes
    -----
    Implemented as a sparse 0/1 matrix product rather than ``np.add.at``, which
    is ~12x slower at the sizes this module scatters (a 64-atom cell has a few
    thousand descriptor pairs and this runs once per configuration per fit).
    """
    p = index.shape[0]
    if p == 0:
        return np.zeros((n_rows,) + values.shape[1:])
    picker = scipy.sparse.csr_matrix(
        (np.ones(p), (index, np.arange(p))), shape=(n_rows, p)
    )
    return picker @ values


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class LinearPotential(MLModel):
    """Ridge regression of energies **and** forces on an invariant descriptor.

    Parameters
    ----------
    descriptor : Descriptor, optional
        Featuriser.  Defaults to
        :class:`~atomlab.models.descriptors.bispectrum.Bispectrum` with its
        default settings (28 features, 5 A cutoff).  Any descriptor obeying the
        :class:`~atomlab.models.base.Descriptor` contract works, including
        :class:`~atomlab.models.descriptors.soap.SOAP` and
        :class:`~atomlab.models.descriptors.acsf.ACSF`.
    n_species : int
        Number of species indices the model is parameterised for.  Each gets its
        own weight vector and energy offset, so the model is linear in
        ``n_species * (D + 1)`` coefficients.
    fit_offsets : bool
        Fit a per-species constant energy offset (default True).  These columns
        are **not** penalised by the ridge: an offset is the cohesive-energy
        zero, and shrinking it towards zero would be a bias of several eV/atom
        rather than a regularisation.
    ridge : float
        Default ridge parameter, on the column-scaled coefficients.
    name : str
        Identifier used in tables and figures.

    Attributes
    ----------
    weights : ndarray, shape (n_species, D)
        Fitted descriptor coefficients in eV (features are dimensionless).
    offsets : ndarray, shape (n_species,)
        Fitted per-species energy offsets in eV.
    cutoff : float
        Taken from the descriptor, in A.
    report : FitReport or None
        The report from the most recent :meth:`fit`.

    Examples
    --------
    >>> from atomlab.models.linear import LinearPotential
    >>> model = LinearPotential()                        # doctest: +SKIP
    >>> report = model.fit(train, val=val, ridge="gcv")  # doctest: +SKIP
    >>> report.condition_number                          # doctest: +SKIP
    """

    def __init__(
        self,
        descriptor: Descriptor | None = None,
        *,
        n_species: int = 1,
        fit_offsets: bool = True,
        ridge: float = 1e-8,
        name: str = "linear",
    ) -> None:
        if descriptor is None:
            from .descriptors.bispectrum import Bispectrum

            descriptor = Bispectrum()
        self.descriptor = descriptor
        self.n_species = int(n_species)
        if self.n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        self.fit_offsets = bool(fit_offsets)
        self.ridge = float(ridge)
        self.name = str(name)
        self.cutoff = float(descriptor.cutoff)

        d = int(descriptor.n_features)
        self.weights = np.zeros((self.n_species, d))
        self.offsets = np.zeros(self.n_species)
        self.is_fitted = False
        self.report: FitReport | None = None

    # -- bookkeeping -------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Descriptor dimension ``D``."""
        return int(self.descriptor.n_features)

    @property
    def n_parameters(self) -> int:
        """Number of fitted coefficients, ``n_species * (D + 1)`` with offsets."""
        return self.n_species * self.n_features + (self.n_species if self.fit_offsets else 0)

    @property
    def coefficients(self) -> np.ndarray:
        """``(K,)`` flat coefficient vector, weights first then offsets."""
        flat = [self.weights.reshape(-1)]
        if self.fit_offsets:
            flat.append(self.offsets)
        return np.concatenate(flat)

    def _unpack(self, coefficients: np.ndarray) -> None:
        d = self.n_features
        n = self.n_species
        self.weights = coefficients[: n * d].reshape(n, d).copy()
        self.offsets = (
            coefficients[n * d : n * d + n].copy() if self.fit_offsets else np.zeros(n)
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "untrained"
        return (
            f"LinearPotential(name={self.name!r}, D={self.n_features}, "
            f"n_species={self.n_species}, ridge={self.ridge:g}, {state})"
        )

    # -- the Potential interface -------------------------------------------

    def _validate(self, configuration: Configuration) -> None:
        if configuration.species.size and int(configuration.species.max()) >= self.n_species:
            raise ValueError(
                f"configuration contains species index {int(configuration.species.max())} "
                f"but {self.name} was built for {self.n_species} species"
            )
        if np.asarray(configuration.pbc).any():
            # Required by the pair-resolved virial below (a pair reaching through
            # two images would be counted once with the wrong displacement) and
            # by the package contract in docs/design.md section 4.
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

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
            Whether those quantities are needed.  Both share one descriptor
            derivative evaluation, so asking for either costs the same.

        Returns
        -------
        Result
            ``energy`` eV, ``forces`` ``(N, 3)`` eV/A, ``virial`` ``(3, 3)`` eV,
            ``energies`` ``(N,)`` eV per atom (exact here, unlike for most
            models: the model *is* a sum of per-atom terms).
        """
        self._require_fitted()
        self._validate(configuration)
        want_deriv = bool(forces or virial)
        desc = self.descriptor.compute(configuration, derivatives=want_deriv)

        species = configuration.species
        # dE/dG for atom i is just w[species_i] -- the model is linear.
        dE_dG = self.weights[species]
        per_atom = np.einsum("nd,nd->n", desc.features, dE_dG) + self.offsets[species]
        energy = float(per_atom.sum())

        f_out = np.zeros((configuration.n_atoms, 3))
        w_out = None
        if want_deriv:
            f_out = desc.forces_from_energy_gradient(dE_dG)
            if virial:
                w_out = self._virial_from_pairs(configuration, desc, dE_dG)

        return Result(energy=energy, forces=f_out, virial=w_out, energies=per_atom)

    @staticmethod
    def _virial_from_pairs(
        configuration: Configuration, desc: DescriptorOutput, dE_dG: np.ndarray
    ) -> np.ndarray:
        """``W_ab = -sum_p (dU/dD_p)_a D_p,b`` over the descriptor's pairs.

        ``D_p`` is the displacement from the centre ``pair_i[p]`` to the mover
        ``pair_j[p]``; the self rows have ``D_p = 0`` and drop out, as they must
        -- a self-derivative is attached to no displacement.  The minimum image
        is the right ``D_p`` only because :meth:`_validate` has already refused
        cells in which a pair could reach through two images.

        Returns
        -------
        ndarray, shape (3, 3)
            Virial in eV, sign convention of :class:`atomlab.types.Configuration`.
        """
        pi = np.asarray(desc.pair_i)
        if pi.size == 0:
            return np.zeros((3, 3))
        pj = np.asarray(desc.pair_j)
        du_dd = np.einsum("pd,pda->pa", dE_dG[pi], desc.derivatives, optimize=True)
        dr = configuration.positions[pj] - configuration.positions[pi]
        if np.asarray(configuration.pbc).any():
            dr = minimum_image(dr, configuration.cell, configuration.pbc)
        return -np.einsum("pa,pb->ab", du_dd, dr, optimize=True)

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
            Block weights; see :meth:`fit`.
        require_forces : bool
            Refuse configurations without force labels.  Energy-only fitting is
            possible but leaves the gradient -- the quantity molecular dynamics
            integrates -- essentially unconstrained, so it must be asked for
            explicitly.

        Returns
        -------
        StackedDesign
            ``matrix`` rows are energies **per atom** (eV/atom) followed by
            force components (eV/A).
        """
        configs = list(configurations)
        if not configs:
            raise ValueError("no configurations to fit")
        d = self.n_features
        n_s = self.n_species
        k = self.n_parameters

        n_atoms = np.array([c.n_atoms for c in configs], dtype=int)
        n_e = len(configs)
        n_f = 3 * int(n_atoms.sum()) if require_forces else 0

        matrix = np.zeros((n_e + n_f, k))
        target = np.zeros(n_e + n_f)
        row = n_e
        for m, cfg in enumerate(configs):
            if cfg.energy is None or (require_forces and cfg.forces is None):
                raise ValueError(
                    f"{self.name}.fit needs labelled configurations; configuration "
                    f"{m} is missing {'forces' if cfg.energy is not None else 'an energy'}"
                )
            self._validate(cfg)
            desc = self.descriptor.compute(cfg, derivatives=require_forces)
            n = cfg.n_atoms
            species = cfg.species

            # -- energy row, in eV/atom ------------------------------------
            summed = _scatter_sum(species, desc.features, n_s)  # (n_species, D)
            matrix[m, : n_s * d] = summed.reshape(-1) / n
            if self.fit_offsets:
                matrix[m, n_s * d :] = np.bincount(species, minlength=n_s) / n
            target[m] = cfg.energy / n

            if not require_forces:
                continue

            # -- force rows -------------------------------------------------
            # F_j = -sum_{i,d} w[s_i,d] dG_id/dr_j, so the design entry for
            # coefficient (s, d) at row (j, a) is -sum over pairs whose centre
            # has species s of dG[i,d]/dr[j,a].  The species of the *centre*
            # selects the block, not the species of the atom being moved.
            pi = np.asarray(desc.pair_i)
            pj = np.asarray(desc.pair_j)
            deriv = desc.derivatives.reshape(pi.shape[0], d * 3)
            flat_index = pj.astype(np.int64) * n_s + species[pi]
            block = -_scatter_sum(flat_index, deriv, n * n_s)  # (N*n_species, D*3)
            block = block.reshape(n, n_s, d, 3).transpose(0, 3, 1, 2).reshape(3 * n, n_s * d)
            matrix[row : row + 3 * n, : n_s * d] = block
            target[row : row + 3 * n] = cfg.forces.reshape(-1)
            row += 3 * n

        # -- block weighting ---------------------------------------------
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
            Held-out set.  Used for the reported validation errors and, when
            ``ridge="heldout"``, to select the penalty.
        ridge : float or {'gcv', 'heldout'}, optional
            Penalty on the column-scaled coefficients.  A float fixes it;
            ``'gcv'`` minimises the generalised cross-validation score over
            ``ridge_grid``; ``'heldout'`` minimises the weighted residual on
            ``val``.  Defaults to the value passed to the constructor.
        ridge_grid : sequence of float, optional
            Grid for the selection routes.  Defaults to
            :data:`DEFAULT_RIDGE_GRID`.  The entire grid is evaluated from one
            QR reduction, so a fine grid is nearly free.
        energy_weight, force_weight : float
            Relative weights of the two blocks.  Each block is normalised by the
            square root of its own row count first, so the loss is

            ``w_E^2 * mean_config (dE/N)^2 + w_F^2 * mean_component (dF)^2``

            with the energy residual in eV/atom and the force residual in eV/A.
            The normalisation makes the ratio ``w_F / w_E`` mean the same thing
            regardless of dataset size or system size, which it does not if the
            raw rows are stacked (``3N`` force rows would otherwise drown one
            energy row).
        fit_forces : bool
            Include the force rows.  Switching this off is supported for
            diagnostic experiments only; a model fitted on energies alone has
            essentially no constraint on its gradient, and the ``FitReport``
            says so.
        scale_columns : bool
            Scale design columns to unit RMS before solving (default True).

        Returns
        -------
        FitReport
            With ``condition_number`` set to the condition number of the
            column-scaled weighted design matrix actually solved, and
            ``history`` carrying the full ridge path plus the raw condition
            number and singular-value spectrum.

        Notes
        -----
        Deterministic: no random number is drawn anywhere in this routine, and
        repeated calls on the same dataset return bitwise identical
        coefficients.
        """
        t0 = time.perf_counter()
        notes: list[str] = []
        if ridge is None:
            ridge = self.ridge
        grid = np.asarray(DEFAULT_RIDGE_GRID if ridge_grid is None else ridge_grid, float)

        train_design = self.design(
            train,
            energy_weight=energy_weight,
            force_weight=force_weight,
            require_forces=fit_forces,
        )
        if not fit_forces:
            notes.append(
                "fitted on energies only: the model's gradient is unconstrained by "
                "the data, so its force error is not a fitted quantity"
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
            penalized[self.n_species * self.n_features :] = False

        system = ReducedSystem(
            train_design.matrix,
            train_design.target,
            weights=train_design.weight,
            penalized=penalized,
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

        train_e, train_f = train_design.rmse(solution.coefficients)
        if val_design is not None:
            val_e, val_f = val_design.rmse(solution.coefficients)
        else:
            val_e, val_f = train_e, train_f
            notes.append("no validation set: the reported 'val' metrics are training-set values")

        if solution.rank < self.n_parameters:
            notes.append(
                f"the regularised system is rank {solution.rank} of {self.n_parameters}: "
                "some coefficient directions are not identified by this data"
            )
        if system.condition_number > 1e10:
            notes.append(
                f"design-matrix condition number {system.condition_number:.3g} after column "
                "scaling; coefficients in the smallest singular directions are noise, which "
                "shows up as extrapolation error rather than as training error"
            )

        report = FitReport(
            converged=True,  # a direct solve either succeeds or raises
            n_epochs=1,
            wall_seconds=time.perf_counter() - t0,
            n_parameters=self.n_parameters,
            n_train=train_design.n_energy,
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
        """Evaluate a whole ridge path without changing the fitted state.

        This is the study's instrument, not a tuning loop: per
        ``docs/theory.md`` section 4 the ridge controls the **smoothness** of the
        fitted error field, and force error and observable error respond to
        smoothness in opposite directions.  A sweep therefore produces a family
        of models whose force errors are similar and whose observable errors need
        not be -- exactly the comparison ``exp05`` and ``exp07`` are built on.

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
            ``coefficients`` has shape ``(L, K)``; feed a row to
            :meth:`set_coefficients` to instantiate that model.
        """
        design = self.design(train, energy_weight=energy_weight, force_weight=force_weight)
        penalized = np.ones(self.n_parameters, dtype=bool)
        if self.fit_offsets:
            penalized[self.n_species * self.n_features :] = False
        system = ReducedSystem(
            design.matrix, design.target, weights=design.weight, penalized=penalized
        )
        heldout = None
        if val is not None and len(list(val)):
            vd = self.design(val, energy_weight=energy_weight, force_weight=force_weight)
            heldout = (vd.matrix, vd.target, vd.weight)
        return system.path(ridges, heldout=heldout)

    def set_coefficients(self, coefficients: np.ndarray, *, ridge: float | None = None) -> None:
        """Install a coefficient vector (e.g. one row of a :class:`RidgePath`).

        Parameters
        ----------
        coefficients : ndarray, shape (K,)
            Flat coefficients, weights first then offsets.
        ridge : float, optional
            Recorded for provenance only.
        """
        coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coefficients.shape[0] != self.n_parameters:
            raise ValueError(
                f"expected {self.n_parameters} coefficients, got {coefficients.shape[0]}"
            )
        self._unpack(coefficients)
        if ridge is not None:
            self.ridge = float(ridge)
        self.is_fitted = True
