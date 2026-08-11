"""Interface for learned potentials, and the descriptor contract they share.

A learned model is a :class:`~atomlab.potentials.base.Potential` first and a
machine-learning object second.  That ordering is deliberate: it means the
molecular dynamics driver, the observable estimators and the response analysis
cannot tell a fitted model from the analytic reference, which is precisely what
makes the controlled comparison in this repository possible.

Two extra contracts appear here on top of ``Potential``:

* :class:`MLModel` -- fitting, serialisation, parameter counting, and a
  :class:`FitReport` that records what actually happened during training rather
  than only the final number.  A quietly non-converged fit would contaminate the
  correlations this study measures, so non-convergence has to be visible.
* :class:`Descriptor` -- the local-environment featurisation shared by the
  descriptor-based models.  Its defining property is that it returns *both* the
  features and their derivatives with respect to atomic positions, because
  forces come from the chain rule through the descriptor and a model whose
  descriptor derivatives are wrong will still train to a plausible energy error.
"""

from __future__ import annotations

import abc
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..potentials.base import Potential
from ..types import Configuration, Dataset

__all__ = ["FitReport", "MLModel", "Descriptor", "DescriptorOutput", "standardize"]


@dataclass
class FitReport:
    """Everything about a fit that a later reader would need in order to trust it.

    Attributes
    ----------
    converged:
        Whether the optimiser met its own stopping criterion.  ``False`` here is
        not a failure to be hidden -- several experiments deliberately train
        under-resourced models, and the distinction between "converged and
        inaccurate" and "not converged" changes what a correlation means.
    n_epochs, wall_seconds, n_parameters, n_train, n_val:
        Basic accounting.
    train_energy_rmse, train_force_rmse, val_energy_rmse, val_force_rmse:
        Energy errors in eV/atom, force errors in eV/A.  Per atom for energy
        because total-energy error scales with system size and would otherwise
        be incomparable across datasets.
    history:
        Per-epoch losses, so a training curve can be plotted after the fact.
    condition_number:
        For linear/kernel models, the conditioning of the normal equations.  A
        huge value means the fitted coefficients are not identified, which shows
        up as wild extrapolation rather than as a bad training error -- a
        distinction this study cares about a great deal.
    notes:
        Free-form warnings raised during fitting.
    """

    converged: bool
    n_epochs: int = 0
    wall_seconds: float = 0.0
    n_parameters: int = 0
    n_train: int = 0
    n_val: int = 0
    train_energy_rmse: float = float("nan")
    train_force_rmse: float = float("nan")
    val_energy_rmse: float = float("nan")
    val_force_rmse: float = float("nan")
    condition_number: float = float("nan")
    history: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = {k: np.asarray(v).tolist() for k, v in self.history.items()}
        return d

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flag = "" if self.converged else " NOT-CONVERGED"
        return (
            f"FitReport(E={self.val_energy_rmse*1e3:.2f} meV/atom, "
            f"F={self.val_force_rmse:.4f} eV/A, p={self.n_parameters}{flag})"
        )


class MLModel(Potential, abc.ABC):
    """A potential whose parameters are fitted to labelled configurations."""

    #: Set by :meth:`fit`; used to refuse evaluation of an untrained model.
    is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, train: Dataset, *, val: Dataset | None = None, **kwargs) -> FitReport:
        """Fit to labelled configurations.

        Implementations must fit to **energies and forces jointly** unless the
        model cannot represent forces.  Force labels carry ``3N`` numbers per
        configuration against the energy's one, and a model fitted on energies
        alone has essentially no constraint on its gradient -- which is the very
        quantity molecular dynamics integrates.
        """

    @property
    @abc.abstractmethod
    def n_parameters(self) -> int:
        """Number of fitted parameters, for capacity-versus-error comparisons."""

    def save(self, path) -> None:
        """Serialise to ``path``.

        The default uses pickle, which is adequate here because models are only
        ever reloaded by the same code that wrote them within a single study.
        Subclasses holding torch modules should override with a state-dict-based
        format so that a saved model survives a refactor of the class.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "MLModel":
        with open(Path(path), "rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} contains a {type(model).__name__}, expected {cls.__name__}")
        return model

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted; call fit() before evaluating it. "
                "Evaluating an unfitted model would return arbitrary numbers that "
                "look like predictions."
            )

    def evaluate(self, dataset: Dataset) -> dict:
        """Standard error metrics against a labelled dataset.

        Returns energy RMSE/MAE in eV/atom, force RMSE/MAE in eV/A, and the mean
        cosine similarity between predicted and reference force vectors.  These
        are the metrics the field reports; this project's argument is about what
        they do and do not imply, so they are computed here in exactly their
        conventional form and nowhere improved upon.
        """
        self._require_fitted()
        e_pred, e_ref, f_pred, f_ref = [], [], [], []
        for cfg in dataset:
            if not cfg.has_labels:
                raise ValueError("evaluate() requires labelled configurations")
            result = self.compute(cfg, forces=True, virial=False)
            e_pred.append(result.energy / cfg.n_atoms)
            e_ref.append(cfg.energy / cfg.n_atoms)
            f_pred.append(result.forces)
            f_ref.append(cfg.forces)

        e_pred, e_ref = np.array(e_pred), np.array(e_ref)
        f_pred, f_ref = np.concatenate(f_pred), np.concatenate(f_ref)
        de, df = e_pred - e_ref, f_pred - f_ref

        norms = np.linalg.norm(f_pred, axis=1) * np.linalg.norm(f_ref, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine = np.where(norms > 0, (f_pred * f_ref).sum(axis=1) / np.maximum(norms, 1e-300), 1.0)

        return {
            "energy_rmse": float(np.sqrt((de**2).mean())),
            "energy_mae": float(np.abs(de).mean()),
            "force_rmse": float(np.sqrt((df**2).mean())),
            "force_mae": float(np.abs(df).mean()),
            "force_cosine": float(cosine.mean()),
            "n_configurations": len(dataset),
            "n_atoms": int(f_ref.shape[0]),
        }


@dataclass
class DescriptorOutput:
    """Features and their position derivatives for one configuration.

    Attributes
    ----------
    features:
        ``(N, D)`` descriptor values, one row per atom.
    derivatives:
        ``(P, D, 3)`` derivative of feature ``d`` of atom ``pair_i[p]`` with
        respect to the position of atom ``pair_j[p]``.  The sparse pair layout is
        used rather than a dense ``(N, D, N, 3)`` tensor because the latter is
        both enormous and almost entirely zero: a descriptor depends only on
        atoms inside its cutoff.
    pair_i, pair_j:
        ``(P,)`` index arrays defining which (centre, mover) pair each derivative
        block belongs to.  The diagonal ``i == j`` entry is included and is the
        self-derivative.
    """

    features: np.ndarray
    derivatives: np.ndarray | None = None
    pair_i: np.ndarray | None = None
    pair_j: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def forces_from_energy_gradient(self, dE_dfeature: np.ndarray) -> np.ndarray:
        """Chain-rule assembly of forces from per-atom energy gradients.

        Parameters
        ----------
        dE_dfeature:
            ``(N, D)`` derivative of the total energy with respect to each atom's
            descriptor.

        Returns
        -------
        ndarray
            ``(N, 3)`` forces, ``F_j = -sum_{i,d} dE/dG_id * dG_id/dr_j``.

        Notes
        -----
        This is where a factor of two or a transposed index silently produces
        forces that are wrong but plausible, so every descriptor is tested
        against :meth:`Potential.numerical_forces` through this exact path
        rather than through a bespoke one.
        """
        if self.derivatives is None:
            raise ValueError("descriptor was computed without derivatives")
        contributions = (dE_dfeature[self.pair_i][:, :, None] * self.derivatives).sum(axis=1)
        forces = np.zeros((self.n_atoms, 3))
        np.add.at(forces, self.pair_j, -contributions)
        return forces


class Descriptor(abc.ABC):
    """Maps a configuration to per-atom invariant features and their derivatives."""

    cutoff: float
    name: str = "descriptor"

    @abc.abstractmethod
    def compute(self, configuration: Configuration, *, derivatives: bool = True) -> DescriptorOutput:
        """Featurise every atom.

        Implementations must be exactly invariant under translation, rotation,
        permutation of identical atoms, and the choice of periodic image.  These
        are not approximations to be checked loosely: an inexact invariance shows
        up in molecular dynamics as a spurious force with no counterpart in any
        error metric computed on a static test set.
        """

    @property
    @abc.abstractmethod
    def n_features(self) -> int:
        """Descriptor dimension per atom."""

    def compute_many(self, configurations: Sequence[Configuration], *, derivatives: bool = True):
        return [self.compute(c, derivatives=derivatives) for c in configurations]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(n_features={self.n_features}, cutoff={self.cutoff})"


def standardize(features: np.ndarray, *, mean=None, scale=None, eps: float = 1e-12):
    """Centre and scale a feature matrix, returning the statistics used.

    Descriptor components routinely differ by many orders of magnitude -- a
    narrow Gaussian symmetry function and a broad one are not remotely
    comparable in scale -- and an unstandardised design matrix turns a
    well-posed ridge regression into an ill-conditioned one.  Constant columns
    (scale below ``eps``) are left unscaled rather than divided by ~0; they carry
    no information and dividing would manufacture numerical noise.
    """
    if mean is None:
        mean = features.mean(axis=0)
    if scale is None:
        scale = features.std(axis=0)
        scale = np.where(scale > eps, scale, 1.0)
    return (features - mean) / scale, mean, scale
