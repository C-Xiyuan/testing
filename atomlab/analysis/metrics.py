"""The proxy-metric zoo: every cheap number a practitioner might rank models by.

``docs/theory.md`` argues that the error in a physical observable is controlled
by ``Cov_0(A, dU)`` -- a *projection* of the error field onto the observable --
while every metric in routine use is a *norm* of that field or of its gradient.
Prediction **P1** is that norms therefore rank models badly against physics.
Testing P1 requires the norms, computed exactly as the field computes them, in
one place, uniformly, so that no metric gets an unfair advantage from a
different test set, a different alignment convention, or a different definition
of "force error".  That is what this module is.

Nothing here is an improvement on standard practice.  Several entries are
deliberately naive.  The point is to measure what they do predict, so the
metrics are implemented in their conventional form and the arguments about them
are left to the analysis in :mod:`atomlab.analysis.correlation`.

Groups
------

**Standard.**  Energy MAE/RMSE per atom (raw and after removing a constant
offset -- theory section 5 notes that a constant ``dU`` shifts no observable, so
an unaligned energy error is partly measuring something known to be harmless),
force RMSE/MAE per component, force cosine similarity, per-vector force-error
magnitude, force *magnitude* error, and virial/stress error.

**Distributional.**  Tail statistics rather than means: the 95th and 99th
percentile of per-atom force error, and force error restricted to the
highest-energy decile and to the shortest-bond decile of test configurations.
These probe the corners of the test distribution where a mean error is blind,
and are the closest a purely in-distribution metric gets to the
out-of-distribution probes theory section 5 argues for.

**Curvature.**  The Hessian, obtained by finite-differencing **analytic
forces** -- never by double finite-differencing the energy, whose error scales
as ``eps/h^2`` and which loses roughly half the available digits.  Both the
Frobenius error and the error in the *low-lying* eigenvalues are reported,
because those are the ones phonons and elastic constants actually see: a model
can match the stiff modes and be useless for thermal properties.

**Smoothness.**  ``std(dU) / force_RMSE`` (:func:`smoothness_ratio`), which
theory section 4 identifies as the empirical stand-in for the inverse spectral
weighting.  Force error weights the error field by the square of its spatial
frequency; observables do not.  A model whose error is high-frequency wiggle has
much force error per unit energy error and so a *small* ratio; one whose error
is smooth and systematic -- the dangerous kind -- has a *large* one.  This is
the one metric here that is not a norm, and P1 predicts it should behave
differently from the rest.

**Uncertainty.**  Ensemble force and energy disagreement, model predictive
variance (GAP-style), and an extrapolation grade: the maximum Mahalanobis
distance of a test descriptor from the training descriptor distribution.  These
need no reference labels at all, which is what makes them attractive in
practice and worth ranking alongside the oracle metrics.

Every metric appears in :data:`METRIC_INFO` with its description, units and
direction, and :func:`compute_all_metrics` returns exactly the keys of that
dictionary.  The correlation study iterates over them without special-casing;
that uniformity is a requirement, not a convenience, because a metric handled
specially is a metric with a thumb on the scale.

Units are metal units throughout (eV, angstrom, ps, amu); each metric's units
are stated in :data:`METRIC_INFO` and in the docstring of the function that
computes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..cell import minimum_image
from ..potentials.base import Potential
from ..types import Configuration, Dataset
from ..units import EV_A3_TO_GPA

__all__ = [
    "METRIC_INFO",
    "LOWER_IS_BETTER",
    "HIGHER_IS_BETTER",
    "DIAGNOSTIC",
    "PairEvaluation",
    "metric_names",
    "metric_direction",
    "describe_metrics",
    "evaluate_pair",
    "standard_metrics",
    "distributional_metrics",
    "curvature_metrics",
    "uncertainty_metrics",
    "smoothness_ratio",
    "hessian",
    "hessian_eigenvalues",
    "hessian_error",
    "mahalanobis_distances",
    "extrapolation_grade",
    "descriptor_features",
    "ensemble_disagreement",
    "compute_all_metrics",
]


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

#: Direction tags used by :data:`METRIC_INFO`.
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"
#: Neither: the number characterises *where* a model's error lives rather than
#: how much of it there is.  ``smoothness_ratio`` is the archetype -- a large
#: value is not "good", it means the error is smooth, which is precisely the
#: kind that damages observables while hiding from force RMSE.
DIAGNOSTIC = "diagnostic"


#: ``name -> (description, units, direction)`` for every metric this module
#: computes.  :func:`compute_all_metrics` returns exactly these keys, so a
#: results table can be documented mechanically and no metric can appear in an
#: analysis without a stated meaning, unit and orientation.
METRIC_INFO: dict[str, tuple[str, str, str]] = {
    # -- standard -----------------------------------------------------------
    "energy_mae": (
        "Mean absolute error of the total energy per atom, as reported.",
        "eV/atom",
        LOWER_IS_BETTER,
    ),
    "energy_rmse": (
        "Root-mean-square error of the total energy per atom, as reported.",
        "eV/atom",
        LOWER_IS_BETTER,
    ),
    "energy_rmse_shifted": (
        "Energy RMSE per atom after removing the mean offset between model and "
        "reference. A constant error field shifts no observable (theory sec. 5), "
        "so the gap between this and energy_rmse is error known to be harmless.",
        "eV/atom",
        LOWER_IS_BETTER,
    ),
    "force_rmse": (
        "Per-component force RMSE over all atoms and Cartesian directions. The "
        "number the field selects and reports models on.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_mae": (
        "Per-component mean absolute force error.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_vector_rmse": (
        "RMS of the per-atom force error vector magnitude |dF_i|. Identically "
        "sqrt(3) x force_rmse -- it is the same number in the other of the two "
        "conventions the literature uses without saying which, and it is here so "
        "that a published force RMSE can be compared with the right one.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_magnitude_rmse": (
        "RMS error of the force *magnitude* |F_i|, ignoring direction. Small "
        "here with a large force_rmse means the errors are mostly rotations of "
        "the force, not changes in its size.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_cosine": (
        "Mean cosine similarity between predicted and reference force vectors.",
        "dimensionless",
        HIGHER_IS_BETTER,
    ),
    "virial_rmse": (
        "RMS error of the virial tensor components per atom (all 9 components).",
        "eV/atom",
        LOWER_IS_BETTER,
    ),
    "stress_rmse": (
        "RMS error of the potential stress tensor components W/V.",
        "GPa",
        LOWER_IS_BETTER,
    ),
    "pressure_mae": (
        "Mean absolute error of the virial pressure tr(W)/3V.",
        "GPa",
        LOWER_IS_BETTER,
    ),
    # -- distributional -----------------------------------------------------
    "force_error_p95": (
        "95th percentile of the per-atom force error magnitude |dF_i|. Tail "
        "error, not mean error: a model can have an excellent RMSE and a heavy "
        "tail, and it is the tail that drives rare events.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_error_p99": (
        "99th percentile of the per-atom force error magnitude |dF_i|.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_error_max": (
        "Maximum per-atom force error magnitude over the test set.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_rmse_high_energy": (
        "Per-component force RMSE restricted to the highest-energy decile of "
        "test configurations (by reference energy per atom).",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "force_rmse_short_bond": (
        "Per-component force RMSE restricted to the decile of test "
        "configurations with the shortest minimum bond length -- the repulsive "
        "wall, which is under-sampled in equilibrium data and controls both the "
        "equation of state and the stability of a hot run.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    # -- curvature ----------------------------------------------------------
    "hessian_frobenius_error": (
        "Frobenius norm of (H_model - H_reference) for a small test system, "
        "with H obtained by central-differencing analytic forces.",
        "eV/A^2",
        LOWER_IS_BETTER,
    ),
    "hessian_frobenius_relative": (
        "The same, divided by ||H_reference||_F.",
        "dimensionless",
        LOWER_IS_BETTER,
    ),
    "hessian_eigenvalue_rmse": (
        "RMS difference between the sorted eigenvalue spectra of the model and "
        "reference Hessians (all modes).",
        "eV/A^2",
        LOWER_IS_BETTER,
    ),
    "hessian_low_eigenvalue_rmse": (
        "RMS difference over the lowest non-trivial eigenvalues only -- the soft "
        "modes that phonon spectra, elastic constants and thermal expansion "
        "actually sample.",
        "eV/A^2",
        LOWER_IS_BETTER,
    ),
    # -- smoothness ---------------------------------------------------------
    "smoothness_ratio": (
        "std(dU) / force_rmse over the test set, with dU the total-energy error "
        "per configuration. The empirical stand-in for the inverse spectral "
        "weighting of theory sec. 4: large means smooth, systematic error "
        "(dangerous, invisible to force RMSE), small means high-frequency "
        "wiggle (loud in force RMSE, nearly harmless to observables).",
        "A",
        DIAGNOSTIC,
    ),
    "delta_u_std": (
        "std(dU), the standard deviation of the total-energy error field over "
        "the test set. One of the three factors in the Cauchy-Schwarz bound "
        "|shift| <= beta sigma(A) sigma(dU) |rho|.",
        "eV",
        LOWER_IS_BETTER,
    ),
    # -- uncertainty --------------------------------------------------------
    "ensemble_force_disagreement": (
        "RMS over atoms and components of the ensemble standard deviation of "
        "the predicted force. Needs no reference labels.",
        "eV/A",
        LOWER_IS_BETTER,
    ),
    "ensemble_energy_disagreement": (
        "Mean over configurations of the ensemble standard deviation of the "
        "predicted energy per atom.",
        "eV/atom",
        LOWER_IS_BETTER,
    ),
    "predictive_variance": (
        "Mean model-reported predictive variance of the energy per atom "
        "(GAP-style posterior variance), i.e. Var[E]/N^2 averaged over the test "
        "set.",
        "eV^2/atom^2",
        LOWER_IS_BETTER,
    ),
    "extrapolation_grade": (
        "Maximum Mahalanobis distance of a test-set atomic descriptor from the "
        "training descriptor distribution. Measures how far outside its training "
        "data the model is being asked to work, not how wrong it is there.",
        "dimensionless",
        LOWER_IS_BETTER,
    ),
}


def metric_names() -> list:
    """Names of every metric, in catalogue order."""
    return list(METRIC_INFO)


def metric_direction(name: str) -> str:
    """Direction tag for one metric: lower/higher is better, or diagnostic."""
    return METRIC_INFO[name][2]


def describe_metrics() -> str:
    """Human-readable table of the whole zoo, for a report appendix."""
    width = max(len(k) for k in METRIC_INFO)
    lines = [f"{'metric'.ljust(width)}  {'units':<16} {'direction':<16} description"]
    for name, (desc, units, direction) in METRIC_INFO.items():
        lines.append(f"{name.ljust(width)}  {units:<16} {direction:<16} {desc}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Evaluating a model against a reference on a test set
# --------------------------------------------------------------------------


def _min_bond_length(configuration: Configuration) -> float:
    """Shortest interatomic distance in angstrom, under the minimum image.

    Computed over all ``N(N-1)/2`` pairs rather than from a neighbour list,
    because a neighbour list needs a cutoff and this quantity is used to *find*
    the configurations whose bonds are shorter than usual -- exactly the ones a
    cutoff-based shortcut would be least reliable about.  ``O(N^2)`` is fine for
    the 64-atom cells this study uses.
    """
    n = configuration.n_atoms
    if n < 2:
        return float("inf")
    d = configuration.positions[:, None, :] - configuration.positions[None, :, :]
    if np.asarray(configuration.pbc).any():
        d = minimum_image(d.reshape(-1, 3), configuration.cell, configuration.pbc).reshape(n, n, 3)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return float(r.min())


@dataclass
class PairEvaluation:
    """Model and reference predictions on one test set, evaluated once.

    Every metric in this module is a function of this object, so the potentially
    expensive part (calling the model on every test configuration) happens
    exactly once per model no matter how many metrics are requested.

    Attributes
    ----------
    n_atoms : ndarray, shape (M,) int
        Atoms per configuration.
    energy_model, energy_reference : ndarray, shape (M,)
        Total energies in eV.
    delta_u : ndarray, shape (M,)
        ``U_model - U_reference`` in eV: the error field sampled on the test
        set.  This is the object theory.md is about, and it is kept as a total
        energy (not per atom) so that it matches the convention of
        :func:`atomlab.analysis.response.predict_shift`.
    forces_model, forces_reference : ndarray, shape (A, 3)
        Forces in eV/A, concatenated over configurations, with ``A = sum N_i``.
    config_index : ndarray, shape (A,) int
        Which configuration each row of the force arrays came from.
    virial_model, virial_reference : ndarray, shape (M, 3, 3) or None
        Virials in eV; None if either potential declines to provide one.
    volumes : ndarray, shape (M,)
        Cell volumes in A^3.
    min_bond : ndarray, shape (M,)
        Shortest interatomic distance per configuration, in A.
    energy_per_atom_reference : ndarray, shape (M,)
        Reference energy per atom in eV/atom, used for the high-energy decile.
    """

    n_atoms: np.ndarray
    energy_model: np.ndarray
    energy_reference: np.ndarray
    forces_model: np.ndarray
    forces_reference: np.ndarray
    config_index: np.ndarray
    volumes: np.ndarray
    min_bond: np.ndarray
    virial_model: np.ndarray | None = None
    virial_reference: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_configurations(self) -> int:
        return int(self.n_atoms.size)

    @property
    def delta_u(self) -> np.ndarray:
        """``(M,)`` total-energy error field ``U_model - U_reference``, eV."""
        return self.energy_model - self.energy_reference

    @property
    def energy_per_atom_reference(self) -> np.ndarray:
        """``(M,)`` reference energy per atom, eV/atom."""
        return self.energy_reference / self.n_atoms

    @property
    def delta_forces(self) -> np.ndarray:
        """``(A, 3)`` force error in eV/A."""
        return self.forces_model - self.forces_reference

    def force_rmse_over(self, mask: np.ndarray) -> float:
        """Per-component force RMSE restricted to a boolean mask over configurations."""
        if not np.any(mask):
            return float("nan")
        rows = mask[self.config_index]
        df = self.delta_forces[rows]
        return float(np.sqrt((df**2).mean())) if df.size else float("nan")


def evaluate_pair(
    model: Potential,
    reference_potential: Potential,
    test_dataset,
    *,
    use_labels: bool = False,
    virial: bool = True,
) -> PairEvaluation:
    """Evaluate a model and its reference on the same configurations.

    Parameters
    ----------
    model : Potential
        The surrogate under test.  Any :class:`~atomlab.potentials.base.Potential`
        works -- a fitted :class:`~atomlab.models.base.MLModel`, a deliberately
        designed perturbation, or a sum of the two.
    reference_potential : Potential
        The analytic ground truth ``U_0``.
    test_dataset : Dataset or sequence of Configuration
        Held-out configurations.
    use_labels : bool
        If True, reuse ``configuration.energy`` / ``.forces`` / ``.virial``
        instead of calling ``reference_potential``, when they are present.
        Default False: stored labels may have come from a different potential
        or a different cutoff, and a silent mismatch there would contaminate
        every correlation this module feeds.  Turn it on only when the dataset
        is known to have been labelled by this exact reference.
    virial : bool
        Request virials.  If either potential returns None the virial metrics
        become NaN rather than raising, since several models legitimately do not
        define one.

    Returns
    -------
    PairEvaluation
    """
    configurations = list(test_dataset)
    if not configurations:
        raise ValueError("test_dataset is empty")

    n_atoms, e_model, e_ref, volumes, min_bond = [], [], [], [], []
    f_model, f_ref, index = [], [], []
    w_model, w_ref = [], []
    have_virial = virial

    for c, cfg in enumerate(configurations):
        res_m = model.compute(cfg, forces=True, virial=virial)
        if use_labels and cfg.has_labels:
            ref_energy = float(cfg.energy)
            ref_forces = cfg.forces
            ref_virial = cfg.virial
        else:
            res_r = reference_potential.compute(cfg, forces=True, virial=virial)
            ref_energy, ref_forces, ref_virial = res_r.energy, res_r.forces, res_r.virial

        n_atoms.append(cfg.n_atoms)
        e_model.append(res_m.energy)
        e_ref.append(ref_energy)
        f_model.append(res_m.forces)
        f_ref.append(ref_forces)
        index.append(np.full(cfg.n_atoms, c, dtype=np.int64))
        volumes.append(cfg.volume)
        min_bond.append(_min_bond_length(cfg))
        if have_virial and res_m.virial is not None and ref_virial is not None:
            w_model.append(res_m.virial)
            w_ref.append(ref_virial)
        else:
            have_virial = False

    return PairEvaluation(
        n_atoms=np.asarray(n_atoms, dtype=np.int64),
        energy_model=np.asarray(e_model, dtype=float),
        energy_reference=np.asarray(e_ref, dtype=float),
        forces_model=np.concatenate(f_model, axis=0),
        forces_reference=np.concatenate(f_ref, axis=0),
        config_index=np.concatenate(index),
        volumes=np.asarray(volumes, dtype=float),
        min_bond=np.asarray(min_bond, dtype=float),
        virial_model=np.stack(w_model) if have_virial else None,
        virial_reference=np.stack(w_ref) if have_virial else None,
        meta={"model": getattr(model, "name", "?"), "reference": getattr(reference_potential, "name", "?")},
    )


# --------------------------------------------------------------------------
# Standard metrics
# --------------------------------------------------------------------------


def standard_metrics(evaluation: PairEvaluation) -> dict:
    """Energy, force and virial errors in their conventional forms.

    Parameters
    ----------
    evaluation : PairEvaluation

    Returns
    -------
    dict
        Keys ``energy_mae``, ``energy_rmse``, ``energy_rmse_shifted``,
        ``force_rmse``, ``force_mae``, ``force_vector_rmse``,
        ``force_magnitude_rmse``, ``force_cosine``, ``virial_rmse``,
        ``stress_rmse``, ``pressure_mae``.  Units are as in
        :data:`METRIC_INFO`.

    Notes
    -----
    Energy errors are per atom because total-energy error scales with system
    size.  ``energy_rmse_shifted`` additionally removes the mean per-atom offset
    over the test set: theory section 5 shows a constant ``dU`` moves no
    observable, so the difference between the two numbers is error that is known
    in advance to be harmless -- and a model penalised only by that difference is
    being mismeasured.

    Force errors are per Cartesian component, matching universal practice.
    ``force_vector_rmse`` is the same quantity organised per atom and is
    *identically* ``sqrt(3)`` times larger -- summing the squared components and
    then averaging over atoms is the same sum in a different order.  Both are
    reported because published "force RMSE" values silently use one convention
    or the other, and a factor of 1.73 between two papers is otherwise
    indistinguishable from a real difference in model quality.  The genuinely
    independent third number is ``force_magnitude_rmse``, which ignores
    direction entirely: a model whose forces have the right magnitudes but the
    wrong directions scores well on it and badly on the other two.
    """
    n = evaluation.n_atoms
    de = evaluation.delta_u / n
    df = evaluation.delta_forces

    fm, fr = evaluation.forces_model, evaluation.forces_reference
    norms = np.linalg.norm(fm, axis=1) * np.linalg.norm(fr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(norms > 0, (fm * fr).sum(axis=1) / np.maximum(norms, 1e-300), 1.0)

    out = {
        "energy_mae": float(np.abs(de).mean()),
        "energy_rmse": float(np.sqrt((de**2).mean())),
        "energy_rmse_shifted": float(np.sqrt(((de - de.mean()) ** 2).mean())),
        "force_rmse": float(np.sqrt((df**2).mean())),
        "force_mae": float(np.abs(df).mean()),
        "force_vector_rmse": float(np.sqrt((df**2).sum(axis=1).mean())),
        "force_magnitude_rmse": float(
            np.sqrt(((np.linalg.norm(fm, axis=1) - np.linalg.norm(fr, axis=1)) ** 2).mean())
        ),
        "force_cosine": float(cosine.mean()),
    }

    if evaluation.virial_model is None or evaluation.virial_reference is None:
        out.update(virial_rmse=float("nan"), stress_rmse=float("nan"), pressure_mae=float("nan"))
        return out

    dw = evaluation.virial_model - evaluation.virial_reference
    per_atom = dw / n[:, None, None]
    stress = dw / evaluation.volumes[:, None, None] * EV_A3_TO_GPA
    dp = np.trace(dw, axis1=1, axis2=2) / (3.0 * evaluation.volumes) * EV_A3_TO_GPA
    out.update(
        virial_rmse=float(np.sqrt((per_atom**2).mean())),
        stress_rmse=float(np.sqrt((stress**2).mean())),
        pressure_mae=float(np.abs(dp).mean()),
    )
    return out


# --------------------------------------------------------------------------
# Distributional metrics
# --------------------------------------------------------------------------


def distributional_metrics(evaluation: PairEvaluation, *, decile: float = 0.1) -> dict:
    """Tail and subset force errors.

    Parameters
    ----------
    evaluation : PairEvaluation
    decile : float
        Fraction of configurations retained by the two subset metrics.  0.1 by
        definition of "decile"; exposed because a small test set may need a
        larger fraction to keep more than one configuration in the subset.

    Returns
    -------
    dict
        ``force_error_p95``, ``force_error_p99``, ``force_error_max``,
        ``force_rmse_high_energy``, ``force_rmse_short_bond``, all in eV/A.

    Notes
    -----
    The percentiles are of the **per-atom error magnitude** ``|dF_i|``, not of
    the per-component error: an atom is the unit that experiences a wrong force,
    and a component-wise percentile would dilute a badly-wrong atom across its
    three components.

    The two subsets are chosen by configuration-level properties that are
    computable without the reference labels the metric itself needs -- energy
    ranking uses the reference energy, and the bond ranking uses geometry alone.
    The shortest-bond decile probes the repulsive wall, which equilibrium
    sampling under-covers and which controls the equation of state; the
    highest-energy decile probes the same corner from the thermodynamic side.
    Both are in-distribution stand-ins for the out-of-distribution probes theory
    section 5 argues are the only way to see error off the typical set.
    """
    per_atom_error = np.linalg.norm(evaluation.delta_forces, axis=1)
    n_config = evaluation.n_configurations
    keep = max(1, int(round(decile * n_config)))

    energy_order = np.argsort(evaluation.energy_per_atom_reference, kind="stable")
    high_mask = np.zeros(n_config, dtype=bool)
    high_mask[energy_order[-keep:]] = True

    bond_order = np.argsort(evaluation.min_bond, kind="stable")
    short_mask = np.zeros(n_config, dtype=bool)
    short_mask[bond_order[:keep]] = True

    return {
        "force_error_p95": float(np.percentile(per_atom_error, 95.0)),
        "force_error_p99": float(np.percentile(per_atom_error, 99.0)),
        "force_error_max": float(per_atom_error.max()),
        "force_rmse_high_energy": evaluation.force_rmse_over(high_mask),
        "force_rmse_short_bond": evaluation.force_rmse_over(short_mask),
    }


# --------------------------------------------------------------------------
# Curvature
# --------------------------------------------------------------------------


def hessian(
    potential: Potential,
    configuration: Configuration,
    *,
    delta: float = 1e-4,
    symmetrize: bool = True,
) -> np.ndarray:
    """Force-constant matrix by central-differencing **analytic forces**.

    Parameters
    ----------
    potential : Potential
    configuration : Configuration
        The system to differentiate about.  ``3N x 3N`` entries and ``6N`` force
        evaluations, so keep ``N`` small (8-32 atoms); this is a diagnostic, not
        a production phonon calculation.
    delta : float
        Displacement in angstrom.  ``1e-4`` is chosen deliberately: the
        truncation error of a central difference is ``O(delta^2 U'''')`` and the
        round-off is ``O(eps |F| / delta)``, so with analytic forces
        (``|F| ~ 1 eV/A``, ``eps ~ 1e-16``) the two are balanced near
        ``1e-5``-``1e-4`` and the total error is ~1e-8 eV/A^2.
    symmetrize : bool
        Average with the transpose.  The exact Hessian is symmetric; the finite
        difference is not, and the antisymmetric part is a free estimate of the
        discretisation error (available as ``0.5*|H - H.T|`` before this is
        applied).

    Returns
    -------
    ndarray, shape (3N, 3N)
        ``H[3i+a, 3j+b] = d^2 U / dr_ia dr_jb`` in eV/A^2, in row-major
        ``(atom, component)`` order.

    Notes
    -----
    **Never double finite-difference the energy for this.**  A second difference
    of the energy has round-off ``eps U / delta^2``, which for ``delta = 1e-4``
    and ``U ~ 100 eV`` is ``~1e-6`` eV/A^2 -- two orders worse than the route
    taken here, and worse still if ``delta`` is reduced.  Differencing the
    analytic forces uses derivative information the potential already has, and
    is the only version of this metric whose error is negligible compared with
    the model errors it is meant to measure.

    This is the bare force-constant matrix, *not* mass-weighted.  For a
    single-species system the dynamical matrix is ``H / m``, a constant
    rescaling that leaves every relative eigenvalue error unchanged; for a
    multi-species system use ``mass_weighted=True`` in
    :func:`hessian_eigenvalues` if the phonon spectrum itself is wanted.
    """
    n = configuration.n_atoms
    h = np.empty((3 * n, 3 * n))
    for i in range(n):
        for a in range(3):
            plus = configuration.copy()
            plus.positions[i, a] += delta
            minus = configuration.copy()
            minus.positions[i, a] -= delta
            # H_{ia,jb} = dU^2/dr_ia dr_jb = -dF_jb/dr_ia
            h[3 * i + a] = -(potential.forces(plus) - potential.forces(minus)).ravel() / (2.0 * delta)
    return 0.5 * (h + h.T) if symmetrize else h


def hessian_eigenvalues(
    potential: Potential,
    configuration: Configuration,
    *,
    delta: float = 1e-4,
    mass_weighted: bool = False,
) -> np.ndarray:
    """Ascending eigenvalues of the (optionally mass-weighted) Hessian.

    Parameters
    ----------
    potential, configuration, delta :
        As in :func:`hessian`.
    mass_weighted : bool
        If True, diagonalise ``M^{-1/2} H M^{-1/2}`` with ``M`` the diagonal
        mass matrix in amu, giving squared frequencies in eV/(A^2 amu); the
        conversion to THz belongs to :mod:`atomlab.observables.phonons`, not
        here.

    Returns
    -------
    ndarray, shape (3N,)
        Eigenvalues in ascending order, eV/A^2 (or eV/A^2/amu if mass weighted).
        ``eigvalsh`` is used, which assumes symmetry -- guaranteed by
        :func:`hessian`'s symmetrisation.
    """
    h = hessian(potential, configuration, delta=delta)
    if mass_weighted:
        inv_sqrt_m = 1.0 / np.sqrt(np.repeat(configuration.masses, 3))
        h = h * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    return np.linalg.eigvalsh(h)


def hessian_error(
    model: Potential,
    reference_potential: Potential,
    configuration: Configuration,
    *,
    delta: float = 1e-4,
    n_skip_modes: int = 3,
    n_low_modes: int = 6,
) -> dict:
    """Compare model and reference curvature on one small configuration.

    Parameters
    ----------
    model, reference_potential : Potential
    configuration : Configuration
        Small system (``6N`` force evaluations per potential).
    delta : float
        Finite-difference step, angstrom.
    n_skip_modes : int
        Number of lowest modes to discard before taking the "low-lying" set.
        For a translationally invariant potential in a periodic cell the three
        acoustic modes at ``q = 0`` are exactly zero and their *difference* is
        pure noise, so including them would dilute the metric with a
        deterministic zero.  Pass 0 for a potential without translational
        invariance (an Einstein crystal, an external field).
    n_low_modes : int
        How many of the remaining lowest eigenvalues to compare.  These are the
        soft modes: phonon spectra, elastic constants and thermal expansion are
        dominated by them, and a model can reproduce the stiff modes perfectly
        while getting these wrong.

    Returns
    -------
    dict
        ``hessian_frobenius_error`` (eV/A^2), ``hessian_frobenius_relative``
        (dimensionless), ``hessian_eigenvalue_rmse`` (eV/A^2, all modes),
        ``hessian_low_eigenvalue_rmse`` (eV/A^2, the ``n_low_modes`` lowest
        non-trivial modes).

    Notes
    -----
    Eigenvalues are compared **index by index after sorting both spectra**, not
    by mode overlap.  Sorted comparison is the right choice for the question
    asked here -- "does this model reproduce the spectrum?" -- and is insensitive
    to the arbitrary phase and degeneracy structure of the eigenvectors.  It does
    mean that a model which reproduces the spectrum but permutes the modes scores
    perfectly; that is a deliberate scoping choice, and mode-resolved comparison
    belongs in :mod:`atomlab.observables.phonons`.
    """
    h_model = hessian(model, configuration, delta=delta)
    h_ref = hessian(reference_potential, configuration, delta=delta)
    diff = h_model - h_ref
    ref_norm = float(np.linalg.norm(h_ref))

    ev_model = np.linalg.eigvalsh(h_model)
    ev_ref = np.linalg.eigvalsh(h_ref)
    d_ev = ev_model - ev_ref

    lo = int(np.clip(n_skip_modes, 0, d_ev.size))
    hi = int(min(d_ev.size, lo + max(1, n_low_modes)))
    low = d_ev[lo:hi]

    return {
        "hessian_frobenius_error": float(np.linalg.norm(diff)),
        "hessian_frobenius_relative": float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else float("nan"),
        "hessian_eigenvalue_rmse": float(np.sqrt((d_ev**2).mean())),
        "hessian_low_eigenvalue_rmse": float(np.sqrt((low**2).mean())) if low.size else float("nan"),
    }


def _pick_hessian_configuration(
    evaluation_configs: Sequence[Configuration], max_atoms: int
) -> Configuration | None:
    """Smallest configuration at or below ``max_atoms`` atoms, or None."""
    small = [c for c in evaluation_configs if c.n_atoms <= max_atoms]
    if not small:
        return None
    return min(small, key=lambda c: c.n_atoms)


def curvature_metrics(
    model: Potential,
    reference_potential: Potential,
    configurations: Sequence[Configuration],
    *,
    hessian_configuration: Configuration | None = None,
    max_hessian_atoms: int = 16,
    delta: float = 1e-4,
    n_skip_modes: int = 3,
    n_low_modes: int = 6,
) -> dict:
    """Curvature metrics on one small configuration, chosen or supplied.

    If ``hessian_configuration`` is None the smallest test configuration with at
    most ``max_hessian_atoms`` atoms is used; if there is none, every curvature
    metric is NaN.  The cost is ``6N`` force evaluations *per potential* and it
    grows linearly in ``N`` with a ``(3N)^2`` memory footprint, so silently
    running it on a 64-atom cell for every model in a zoo would dominate the
    study's runtime -- hence the cap and the explicit opt-in for anything bigger.
    """
    cfg = hessian_configuration or _pick_hessian_configuration(configurations, max_hessian_atoms)
    if cfg is None:
        return {
            "hessian_frobenius_error": float("nan"),
            "hessian_frobenius_relative": float("nan"),
            "hessian_eigenvalue_rmse": float("nan"),
            "hessian_low_eigenvalue_rmse": float("nan"),
        }
    return hessian_error(
        model,
        reference_potential,
        cfg,
        delta=delta,
        n_skip_modes=n_skip_modes,
        n_low_modes=n_low_modes,
    )


# --------------------------------------------------------------------------
# Smoothness
# --------------------------------------------------------------------------


def smoothness_ratio(delta_u: np.ndarray, force_rmse: float) -> float:
    """``std(dU) / force_RMSE``: where in frequency the error lives.

    Parameters
    ----------
    delta_u : ndarray, shape (M,)
        Total-energy error ``U_model - U_reference`` on ``M`` test
        configurations, in eV.  The standard deviation is used, not the mean, so
        a constant offset -- which theory section 5 shows moves no observable --
        does not enter.
    force_rmse : float
        Per-component force RMSE on the same set, eV/A.

    Returns
    -------
    float
        Ratio in angstrom.  ``inf`` if the force RMSE is exactly zero.

    Notes
    -----
    Theory section 4: force error weights the error field by ``k^2`` in spatial
    frequency, observables by roughly ``k^0``.  For a Gaussian shell
    perturbation of width ``w`` and amplitude ``a`` the theory predicts
    ``sigma(dU) ~ a*w`` and ``RMSE_F ~ a/sqrt(w)`` (eqs. 4.3-4.4), so this ratio
    grows monotonically with the width of the error -- it is a *measurement of
    the width*, using only quantities a practitioner already computes.

    A large ratio means smooth, low-frequency, systematic error: the dangerous
    kind, which force RMSE under-reports.  A small ratio means high-frequency
    interpolation wiggle: loud in force RMSE, nearly invisible to observables.
    The direction tag in :data:`METRIC_INFO` is therefore ``diagnostic``, not
    ``lower_is_better`` -- treating it as a quality score would be a category
    error, and P1 predicts it should correlate with observable error in a way
    the norms do not.

    This is the same quantity returned as ``smoothness_ratio`` by
    :func:`atomlab.analysis.response.spectral_decomposition`, computed from the
    same convention (total-energy dU) so the two agree numerically.
    """
    du = np.asarray(delta_u, dtype=float).ravel()
    if du.size < 2:
        return float("nan")
    sigma = float(du.std(ddof=1))
    return sigma / force_rmse if force_rmse > 0 else float("inf")


# --------------------------------------------------------------------------
# Uncertainty-based metrics
# --------------------------------------------------------------------------


def ensemble_disagreement(
    ensemble: Sequence[Potential],
    configurations: Sequence[Configuration],
) -> dict:
    """Spread of an ensemble's predictions -- an uncertainty with no ground truth.

    Parameters
    ----------
    ensemble : sequence of Potential
        Committee members, at least two.  In this study they are independently
        seeded fits, or an architecture-diverse committee; the module does not
        care which, but theory section 7.4 does: shared systematic error cancels
        in a homogeneous committee and this metric under-reports it.
    configurations : sequence of Configuration

    Returns
    -------
    dict
        ``ensemble_force_disagreement`` (eV/A): RMS over atoms and components of
        the across-member standard deviation of the force.
        ``ensemble_energy_disagreement`` (eV/atom): mean over configurations of
        the across-member standard deviation of the energy per atom.

    Notes
    -----
    The sample standard deviation (``ddof=1``) is used, so a two-member
    "ensemble" gives ``|a-b|/sqrt(2)`` -- an honest, if noisy, estimate rather
    than the systematically small ``ddof=0`` value.
    """
    members = list(ensemble)
    if len(members) < 2:
        raise ValueError("ensemble disagreement needs at least two members")

    energy_std, force_sq, force_n = [], 0.0, 0
    for cfg in configurations:
        energies = np.empty(len(members))
        forces = np.empty((len(members), cfg.n_atoms, 3))
        for k, member in enumerate(members):
            res = member.compute(cfg, forces=True, virial=False)
            energies[k] = res.energy / cfg.n_atoms
            forces[k] = res.forces
        energy_std.append(energies.std(ddof=1))
        s = forces.std(axis=0, ddof=1)
        force_sq += float((s**2).sum())
        force_n += s.size

    return {
        "ensemble_energy_disagreement": float(np.mean(energy_std)),
        "ensemble_force_disagreement": float(np.sqrt(force_sq / force_n)) if force_n else float("nan"),
    }


def _predictive_variance(model, configurations: Sequence[Configuration], variance_fn=None) -> float:
    """Mean per-atom predictive variance, eV^2/atom^2, or NaN if unavailable.

    Looks for, in order: an explicit ``variance_fn(configuration) -> float``, a
    ``model.predictive_variance(configuration)`` method (the GAP interface), or a
    ``Result.extra`` entry named ``variance`` / ``energy_variance`` /
    ``predictive_variance``.  A model that offers none of these gets NaN rather
    than a zero, because "this model reports no uncertainty" and "this model is
    certain" are opposite statements and conflating them would put a
    perfectly-confident-looking row in the correlation table.
    """
    values = []
    for cfg in configurations:
        var = None
        if variance_fn is not None:
            var = float(variance_fn(cfg))
        elif hasattr(model, "predictive_variance"):
            var = float(model.predictive_variance(cfg))
        else:
            extra = model.compute(cfg, forces=False, virial=False).extra or {}
            for key in ("variance", "energy_variance", "predictive_variance"):
                if key in extra:
                    var = float(np.sum(extra[key]))
                    break
        if var is None:
            return float("nan")
        values.append(var / cfg.n_atoms**2)
    return float(np.mean(values)) if values else float("nan")


def descriptor_features(descriptor, configurations: Sequence[Configuration]) -> np.ndarray:
    """Stack per-atom descriptor features over configurations.

    Parameters
    ----------
    descriptor : Descriptor
        Anything implementing :class:`atomlab.models.base.Descriptor`.
    configurations : sequence of Configuration

    Returns
    -------
    ndarray, shape (sum_i N_i, D)
        Per-atom features.  Derivatives are not requested, which is much
        cheaper: the extrapolation grade is a property of the *distribution* of
        environments, not of the model's gradients.
    """
    return np.concatenate(
        [np.asarray(descriptor.compute(c, derivatives=False).features, dtype=float) for c in configurations],
        axis=0,
    )


def mahalanobis_distances(test_features, train_features, *, ridge: float = 1e-6) -> np.ndarray:
    """Mahalanobis distance of each test descriptor from the training distribution.

    Parameters
    ----------
    test_features : array_like, shape (T, D)
        Per-atom descriptors of the test environments.
    train_features : array_like, shape (S, D)
        Per-atom descriptors of the training environments.
    ridge : float
        Relative Tikhonov regularisation added to the training covariance, as a
        fraction of its mean diagonal.  Descriptor bases are routinely rank
        deficient -- SOAP and ACSF components are analytically dependent, and a
        crystal training set spans far fewer directions than the basis has -- so
        the raw covariance is singular and the naive inverse would report
        astronomically large distances along directions the training set simply
        never varied in.  The ridge makes those directions cheap rather than
        infinitely expensive, which is the conservative choice for a metric
        whose purpose is to flag genuine extrapolation.

    Returns
    -------
    ndarray, shape (T,)
        Distances, dimensionless (the covariance carries the descriptor units).

    Notes
    -----
    Distances are computed from the eigendecomposition of the regularised
    covariance rather than by inverting it, so a near-singular basis degrades
    smoothly instead of producing negative squared distances from round-off.
    """
    x = np.atleast_2d(np.asarray(test_features, dtype=float))
    y = np.atleast_2d(np.asarray(train_features, dtype=float))
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dimensions differ: test {x.shape[1]}, train {y.shape[1]}")
    if y.shape[0] < 2:
        raise ValueError("need at least two training environments to estimate a covariance")

    mean = y.mean(axis=0)
    cov = np.cov(y, rowvar=False)
    cov = np.atleast_2d(cov)
    scale = float(np.trace(cov) / cov.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + ridge * scale * np.eye(cov.shape[0])

    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, ridge * scale * 1e-6)
    projected = (x - mean) @ evecs
    return np.sqrt(np.maximum((projected**2 / evals).sum(axis=1), 0.0))


def extrapolation_grade(test_features, train_features, *, ridge: float = 1e-6, quantile: float = 1.0) -> float:
    """Worst-case Mahalanobis distance of a test environment from training data.

    Parameters
    ----------
    test_features, train_features, ridge :
        As in :func:`mahalanobis_distances`.
    quantile : float
        1.0 (the default) gives the maximum, which is the quantity of interest:
        a single environment far outside the training distribution is enough to
        destabilise a molecular dynamics run, and averaging would hide it.  Set
        below 1 for a robustified version if a few outliers in the test set are
        known artefacts.

    Returns
    -------
    float
        Dimensionless distance.  This measures *where the model is being asked
        to work*, not how wrong it is there -- which is exactly why it is worth
        having in the zoo: theory section 5 notes that error off the typical set
        is invisible to every in-distribution metric, including force RMSE, so a
        metric that at least detects the geometry of the excursion is the only
        in-distribution warning available.
    """
    d = mahalanobis_distances(test_features, train_features, ridge=ridge)
    return float(d.max()) if quantile >= 1.0 else float(np.quantile(d, quantile))


def uncertainty_metrics(
    model: Potential,
    configurations: Sequence[Configuration],
    *,
    ensemble: Sequence[Potential] | None = None,
    variance_fn: Callable[[Configuration], float] | None = None,
    train_features=None,
    test_features=None,
    descriptor=None,
    train_configurations: Sequence[Configuration] | None = None,
    ridge: float = 1e-6,
) -> dict:
    """The label-free metrics: committee spread, predictive variance, extrapolation.

    Parameters
    ----------
    model : Potential
        Used for the predictive variance and, if ``ensemble`` is None, searched
        for a committee (attribute ``models`` or ``members``).
    configurations : sequence of Configuration
        Test configurations.
    ensemble : sequence of Potential, optional
        Committee members.  If None, ``model.models`` / ``model.members`` is
        used when present; otherwise the ensemble metrics are NaN.
    variance_fn : callable, optional
        ``configuration -> variance of the total energy in eV^2``.
    train_features, test_features : array_like, optional
        Pre-computed ``(S, D)`` and ``(T, D)`` per-atom descriptors.
    descriptor, train_configurations : optional
        Alternative to the above: compute features with this descriptor from
        these training configurations (and from ``configurations`` for the test
        side).
    ridge : float
        Covariance regularisation for the Mahalanobis distance.

    Returns
    -------
    dict
        ``ensemble_force_disagreement``, ``ensemble_energy_disagreement``,
        ``predictive_variance``, ``extrapolation_grade``.  Any metric whose
        inputs are unavailable is NaN -- never zero.
    """
    out = {
        "ensemble_force_disagreement": float("nan"),
        "ensemble_energy_disagreement": float("nan"),
        "predictive_variance": float("nan"),
        "extrapolation_grade": float("nan"),
    }

    members = ensemble
    if members is None:
        # Duck-typed committee discovery.  Checked for list/tuple explicitly
        # rather than with truthiness, because a model attribute called
        # ``models`` could be an array, whose truth value is an exception.
        for attribute in ("models", "members"):
            candidate = getattr(model, attribute, None)
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                members = candidate
                break
    if members is not None and len(list(members)) >= 2:
        out.update(ensemble_disagreement(list(members), configurations))

    out["predictive_variance"] = _predictive_variance(model, configurations, variance_fn)

    if train_features is None and descriptor is not None and train_configurations is not None:
        train_features = descriptor_features(descriptor, train_configurations)
    if test_features is None and descriptor is not None and train_features is not None:
        test_features = descriptor_features(descriptor, configurations)
    if train_features is not None and test_features is not None:
        out["extrapolation_grade"] = extrapolation_grade(test_features, train_features, ridge=ridge)
    return out


# --------------------------------------------------------------------------
# The single entry point
# --------------------------------------------------------------------------


def compute_all_metrics(
    model: Potential,
    reference_potential: Potential,
    test_dataset,
    *,
    ensemble: Sequence[Potential] | None = None,
    variance_fn: Callable[[Configuration], float] | None = None,
    descriptor=None,
    train_configurations: Sequence[Configuration] | None = None,
    train_features=None,
    test_features=None,
    hessian_configuration: Configuration | None = None,
    max_hessian_atoms: int = 16,
    hessian_delta: float = 1e-4,
    n_skip_modes: int = 3,
    n_low_modes: int = 6,
    decile: float = 0.1,
    use_labels: bool = False,
    ridge: float = 1e-6,
) -> dict:
    """Compute every proxy metric for one model, uniformly.

    Parameters
    ----------
    model : Potential
        The surrogate under test.
    reference_potential : Potential
        The analytic ground truth.
    test_dataset : Dataset or sequence of Configuration
        Held-out configurations, the same set for every model in a zoo -- the
        correlation study compares metrics *across models*, so any variation in
        the test set between models would enter as noise indistinguishable from
        the effect being measured.
    ensemble, variance_fn :
        Optional uncertainty sources, see :func:`uncertainty_metrics`.
    descriptor, train_configurations, train_features, test_features :
        Optional inputs for the extrapolation grade, see
        :func:`uncertainty_metrics`.
    hessian_configuration, max_hessian_atoms, hessian_delta, n_skip_modes,
    n_low_modes :
        Curvature settings, see :func:`curvature_metrics`.
    decile : float
        Subset fraction for the distributional metrics.
    use_labels : bool
        Reuse stored reference labels instead of recomputing them; see
        :func:`evaluate_pair`.
    ridge : float
        Covariance regularisation for the extrapolation grade.

    Returns
    -------
    dict
        ``{metric_name: float}`` with **exactly** the keys of
        :data:`METRIC_INFO`, in catalogue order.  Metrics whose inputs were not
        supplied (no ensemble, no descriptor, no small configuration for the
        Hessian) are NaN.  The key set never depends on the inputs, so the
        correlation study can iterate over metrics without special-casing and a
        missing metric shows up as a missing *value* rather than as a missing
        column.

    Notes
    -----
    Cost is dominated by one force evaluation per test configuration for the
    model and one for the reference, plus ``6N`` more of each for the Hessian and
    ``K`` more per configuration for a ``K``-member ensemble.  Nothing here is
    fitted or sampled, so the whole zoo of metrics for one model on a few hundred
    64-atom configurations takes seconds with an analytic reference.
    """
    configurations = list(test_dataset)
    evaluation = evaluate_pair(model, reference_potential, configurations, use_labels=use_labels)

    out: dict[str, float] = {}
    out.update(standard_metrics(evaluation))
    out.update(distributional_metrics(evaluation, decile=decile))
    out.update(
        curvature_metrics(
            model,
            reference_potential,
            configurations,
            hessian_configuration=hessian_configuration,
            max_hessian_atoms=max_hessian_atoms,
            delta=hessian_delta,
            n_skip_modes=n_skip_modes,
            n_low_modes=n_low_modes,
        )
    )
    out["delta_u_std"] = float(np.std(evaluation.delta_u, ddof=1)) if evaluation.n_configurations > 1 else float("nan")
    out["smoothness_ratio"] = smoothness_ratio(evaluation.delta_u, out["force_rmse"])
    out.update(
        uncertainty_metrics(
            model,
            configurations,
            ensemble=ensemble,
            variance_fn=variance_fn,
            train_features=train_features,
            test_features=test_features,
            descriptor=descriptor,
            train_configurations=train_configurations,
            ridge=ridge,
        )
    )

    missing = set(METRIC_INFO) - set(out)
    extra = set(out) - set(METRIC_INFO)
    if missing or extra:  # pragma: no cover - guards a programming error, not user input
        raise RuntimeError(
            f"metric catalogue and computed metrics disagree: missing {sorted(missing)}, "
            f"undocumented {sorted(extra)}"
        )
    return {name: float(out[name]) for name in METRIC_INFO}
