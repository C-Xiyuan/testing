"""Abstract interface every energy model in :mod:`atomlab` implements.

The whole project rests on one abstraction: a :class:`Potential` maps a
:class:`~atomlab.types.Configuration` to a :class:`~atomlab.types.Result`.
Analytic reference potentials (Lennard-Jones, Stillinger-Weber, EAM) and
machine-learned models alike implement it, which is what lets the molecular
dynamics driver, the observable estimators and the response analysis treat
"ground truth" and "surrogate" interchangeably.

Implementers only have to provide :meth:`Potential.compute`.  Everything else
-- convenience accessors, finite-difference self-checks, arithmetic on
potentials -- is derived from it.
"""

from __future__ import annotations

import abc
from typing import Iterable, Sequence

import numpy as np

from ..types import Configuration, Result

__all__ = [
    "Potential",
    "SumPotential",
    "ScaledPotential",
    "ZeroPotential",
    "check_forces",
    "check_virial",
]


class Potential(abc.ABC):
    """Base class for anything that assigns an energy to a configuration.

    Attributes
    ----------
    cutoff:
        Interaction range in angstrom.  Used by neighbour-list construction and
        by consistency checks on the minimum image convention.  Subclasses must
        set it (``float('inf')`` is allowed for long-range models, but then the
        neighbour machinery cannot be used).
    name:
        Short human-readable identifier, used in results tables and figures.
    """

    cutoff: float = 0.0
    name: str = "potential"

    # -- the one method subclasses must implement --------------------------

    @abc.abstractmethod
    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Evaluate the potential.

        Parameters
        ----------
        configuration:
            The geometry to evaluate.  Implementations must not modify it.
        forces:
            If False the implementation may skip force evaluation, but it is
            always allowed to compute them anyway.  When False, the returned
            ``Result.forces`` must still be a valid ``(N, 3)`` array (zeros are
            acceptable only if the caller is known not to use them -- prefer
            returning the real forces).
        virial:
            Same contract as ``forces``.  When False, ``Result.virial`` may be
            ``None``.

        Returns
        -------
        Result
            Energy in eV, forces in eV/A, virial in eV.
        """

    # -- convenience -------------------------------------------------------

    def energy(self, configuration: Configuration) -> float:
        """Total potential energy in eV."""
        return self.compute(configuration, forces=False, virial=False).energy

    def energy_per_atom(self, configuration: Configuration) -> float:
        """Potential energy per atom in eV."""
        return self.energy(configuration) / configuration.n_atoms

    def forces(self, configuration: Configuration) -> np.ndarray:
        """``(N, 3)`` forces in eV/A."""
        return self.compute(configuration, forces=True, virial=False).forces

    def virial(self, configuration: Configuration) -> np.ndarray:
        """``(3, 3)`` virial tensor in eV."""
        w = self.compute(configuration, forces=False, virial=True).virial
        if w is None:
            raise NotImplementedError(f"{self.name} does not provide a virial")
        return w

    def stress(self, configuration: Configuration) -> np.ndarray:
        """Potential contribution to the stress tensor, in eV/A^3.

        This is ``W / V``; it excludes the kinetic (ideal-gas) term, which is a
        property of the dynamical state rather than of the potential.
        """
        return self.virial(configuration) / configuration.volume

    def pressure(self, configuration: Configuration) -> float:
        """Virial pressure ``tr(W) / (3 V)`` in eV/A^3."""
        return float(np.trace(self.virial(configuration)) / (3.0 * configuration.volume))

    def label(self, configurations: Iterable[Configuration]) -> list[Configuration]:
        """Return copies of ``configurations`` carrying this potential's labels."""
        out = []
        for cfg in configurations:
            out.append(cfg.with_labels(self.compute(cfg, forces=True, virial=True)))
        return out

    # -- self-consistency checks -------------------------------------------

    def numerical_forces(
        self,
        configuration: Configuration,
        delta: float = 1e-5,
        atoms: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Central-difference forces, for validating :meth:`compute`.

        Parameters
        ----------
        delta:
            Displacement in angstrom.  ``1e-5`` balances truncation against
            round-off for double-precision energies of order 1-100 eV.
        atoms:
            Restrict the check to these atom indices (the full calculation is
            ``6 N`` energy evaluations, which is slow for large systems).
        """
        idx = range(configuration.n_atoms) if atoms is None else atoms
        out = np.zeros((configuration.n_atoms, 3))
        for i in idx:
            for k in range(3):
                plus = configuration.copy()
                plus.positions[i, k] += delta
                minus = configuration.copy()
                minus.positions[i, k] -= delta
                out[i, k] = -(self.energy(plus) - self.energy(minus)) / (2.0 * delta)
        return out

    def numerical_virial(self, configuration: Configuration, delta: float = 1e-6) -> np.ndarray:
        """Central-difference virial ``-dU/d(strain)``, for validating :meth:`compute`."""
        w = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                eps = np.zeros((3, 3))
                eps[a, b] = delta
                e_plus = self.energy(configuration.strained(eps))
                eps[a, b] = -delta
                e_minus = self.energy(configuration.strained(eps))
                w[a, b] = -(e_plus - e_minus) / (2.0 * delta)
        return w

    # -- algebra -----------------------------------------------------------

    def __add__(self, other: "Potential") -> "SumPotential":
        return SumPotential([self, other])

    def __mul__(self, factor: float) -> "ScaledPotential":
        return ScaledPotential(self, float(factor))

    __rmul__ = __mul__

    def __sub__(self, other: "Potential") -> "SumPotential":
        return SumPotential([self, ScaledPotential(other, -1.0)])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, cutoff={self.cutoff})"


class SumPotential(Potential):
    """Sum of several potentials, evaluated on the same configuration.

    Used pervasively in the perturbation experiments, where the surrogate is
    written as ``U_true + delta_U`` with a deliberately constructed ``delta_U``.
    """

    def __init__(self, terms: Sequence[Potential], name: str | None = None):
        if not terms:
            raise ValueError("SumPotential needs at least one term")
        self.terms = list(terms)
        self.cutoff = max(t.cutoff for t in self.terms)
        self.name = name or " + ".join(t.name for t in self.terms)

    def compute(self, configuration, *, forces=True, virial=True) -> Result:
        total_e = 0.0
        total_f = np.zeros((configuration.n_atoms, 3))
        total_w = np.zeros((3, 3)) if virial else None
        have_virial = virial
        for term in self.terms:
            r = term.compute(configuration, forces=forces, virial=virial)
            total_e += r.energy
            total_f += r.forces
            if have_virial:
                if r.virial is None:
                    have_virial = False
                    total_w = None
                else:
                    total_w += r.virial
        return Result(energy=total_e, forces=total_f, virial=total_w)


class ScaledPotential(Potential):
    """A potential multiplied by a scalar."""

    def __init__(self, base: Potential, factor: float, name: str | None = None):
        self.base = base
        self.factor = float(factor)
        self.cutoff = base.cutoff
        self.name = name or f"{self.factor:g}*{base.name}"

    def compute(self, configuration, *, forces=True, virial=True) -> Result:
        r = self.base.compute(configuration, forces=forces, virial=virial)
        return Result(
            energy=self.factor * r.energy,
            forces=self.factor * r.forces,
            virial=None if r.virial is None else self.factor * r.virial,
            energies=None if r.energies is None else self.factor * r.energies,
        )


class ZeroPotential(Potential):
    """Identically zero; useful as a neutral element and in tests."""

    cutoff = 0.0
    name = "zero"

    def compute(self, configuration, *, forces=True, virial=True) -> Result:
        return Result(
            energy=0.0,
            forces=np.zeros((configuration.n_atoms, 3)),
            virial=np.zeros((3, 3)) if virial else None,
        )


# --------------------------------------------------------------------------
# Validation helpers shared by the test suite
# --------------------------------------------------------------------------


def check_forces(
    potential: Potential,
    configuration: Configuration,
    delta: float = 1e-5,
    atoms: Sequence[int] | None = None,
) -> float:
    """Return the max abs difference between analytic and numerical forces.

    A correct implementation gives ``~1e-7 eV/A`` or better for a well-scaled
    system; anything above ``1e-5`` indicates a genuine bug rather than
    finite-difference noise.
    """
    analytic = potential.forces(configuration)
    numerical = potential.numerical_forces(configuration, delta=delta, atoms=atoms)
    if atoms is not None:
        idx = np.asarray(list(atoms))
        analytic = analytic[idx]
        numerical = numerical[idx]
    return float(np.max(np.abs(analytic - numerical)))


def check_virial(potential: Potential, configuration: Configuration, delta: float = 1e-6) -> float:
    """Return the max abs difference between analytic and numerical virial."""
    analytic = potential.virial(configuration)
    numerical = potential.numerical_virial(configuration, delta=delta)
    return float(np.max(np.abs(analytic - numerical)))
