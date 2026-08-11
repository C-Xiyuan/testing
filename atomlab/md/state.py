"""The mutable state a molecular dynamics integrator advances.

:class:`MDState` carries everything needed to take one step and to restart from
disk: unwrapped positions, velocities, the cell, the cached
:class:`~atomlab.types.Result` of the last force evaluation, the extended-system
variables of whatever thermostat or barostat is attached, and the random
generator that drives any stochastic integrator.

Two decisions here are load-bearing.

*Positions are unwrapped.*  The state stores the continuous trajectory
``r(t)``, never the wrapped image.  Mean squared displacements and diffusion
coefficients are then correct by construction, and wrapping happens only where
it is needed -- inside :meth:`MDState.to_configuration`, on the way to a
potential, where the minimum-image convention makes it irrelevant anyway.

*The last force evaluation is cached.*  A velocity-Verlet step needs the forces
at the beginning and at the end of the interval, and the forces at the end of
one step are the forces at the beginning of the next.  Caching them halves the
cost of a run; forgetting to invalidate the cache after moving atoms silently
integrates the wrong equations of motion, so every mutation of ``positions`` or
``cell`` in this package goes through code that immediately recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Configuration, Result
from ..units import EV_A3_TO_BAR, EV_A3_TO_GPA, KB, MVV2E
from .velocities import (
    com_velocity,
    kinetic_energy,
    maxwell_boltzmann,
    n_dof,
    remove_com_momentum,
)

__all__ = ["MDState"]

_PRESSURE_UNITS = {
    "bar": EV_A3_TO_BAR,
    "GPa": EV_A3_TO_GPA,
    "eV/A^3": 1.0,
}


@dataclass
class MDState:
    """Complete dynamical state of a molecular dynamics run.

    Attributes
    ----------
    positions : ndarray, shape (N, 3)
        **Unwrapped** cartesian coordinates in A.
    velocities : ndarray, shape (N, 3)
        Velocities in A/ps.
    cell : ndarray, shape (3, 3)
        Lattice vectors as rows, in A.  Mutated by barostats.
    masses : ndarray, shape (N,)
        Masses in amu.
    species : ndarray, shape (N,)
        Integer type indices.
    pbc : ndarray, shape (3,)
        Periodicity flags.
    symbols : tuple of str
        Type index to chemical symbol map, carried through for provenance.
    result : Result or None
        Cached energy (eV), forces (eV/A) and virial (eV) at ``positions``.
        ``None`` means "not yet evaluated"; use :meth:`refresh`.
    step : int
        Number of integrator steps taken.
    time : float
        Elapsed simulation time in ps.
    fixed_com : bool
        Whether the centre-of-mass momentum is being held at zero.  Controls
        the degree-of-freedom count used by every temperature and pressure.
    thermostat : dict
        Extended-system variables owned by the attached thermostat (Nose-Hoover
        chain positions/velocities/masses, and so on).  Stored on the state
        rather than on the integrator so that a restart is a single object and
        so that :meth:`copy` really copies everything.
    barostat : dict
        The same, for barostat variables.
    rng : numpy.random.Generator or None
        Source of randomness for stochastic integrators.  Required by
        :class:`~atomlab.md.integrators.Langevin`; unused by the deterministic
        ones.
    info : dict
        Free-form metadata (temperature setpoint, provenance, ...).
    """

    positions: np.ndarray
    velocities: np.ndarray
    cell: np.ndarray
    masses: np.ndarray
    species: np.ndarray | None = None
    pbc: np.ndarray | bool = True
    symbols: tuple[str, ...] = ()

    result: Result | None = None

    step: int = 0
    time: float = 0.0
    fixed_com: bool = True

    thermostat: dict = field(default_factory=dict)
    barostat: dict = field(default_factory=dict)
    rng: np.random.Generator | None = None
    info: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        n = self.positions.shape[0]

        self.velocities = np.ascontiguousarray(self.velocities, dtype=np.float64)
        if self.velocities.shape != (n, 3):
            raise ValueError(f"velocities must be ({n}, 3), got {self.velocities.shape}")

        self.cell = np.ascontiguousarray(self.cell, dtype=np.float64)
        if self.cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3), got {self.cell.shape}")

        self.masses = np.ascontiguousarray(self.masses, dtype=np.float64)
        if self.masses.shape != (n,):
            raise ValueError(f"masses must be ({n},), got {self.masses.shape}")
        if np.any(self.masses <= 0.0):
            raise ValueError("all masses must be strictly positive")

        if self.species is None:
            self.species = np.zeros(n, dtype=np.int32)
        self.species = np.ascontiguousarray(self.species, dtype=np.int32)

        if isinstance(self.pbc, (bool, np.bool_)):
            self.pbc = np.array([bool(self.pbc)] * 3)
        self.pbc = np.ascontiguousarray(np.asarray(self.pbc, dtype=bool))

        self.symbols = tuple(self.symbols)

    @classmethod
    def from_configuration(
        cls,
        configuration: Configuration,
        potential=None,
        temperature: float | None = None,
        seed: int | np.random.Generator | None = None,
        *,
        fixed_com: bool = True,
        velocities: np.ndarray | None = None,
        exact_temperature: bool = False,
    ) -> "MDState":
        """Build a state from a configuration, drawing Maxwell-Boltzmann velocities.

        Parameters
        ----------
        configuration : Configuration
            Geometry, masses, species and cell.  Not modified.
        potential : Potential, optional
            If given, the forces/energy/virial are evaluated once so the state
            arrives ready to step.
        temperature : float, optional
            Temperature in K for the initial velocities.  ``None`` or ``0``
            starts from rest.  Ignored if ``velocities`` is given.
        seed : int or numpy.random.Generator, optional
            Seeds :attr:`rng`.  An integer is passed to
            ``numpy.random.default_rng``.  ``None`` still produces a generator
            (an unseeded one), so stochastic integrators always work -- but for
            reproducible science, pass the seed.
        fixed_com : bool
            Remove the centre-of-mass momentum and drop three degrees of
            freedom.  Set False for a system with an external field (an
            Einstein crystal, say), where the centre of mass is not a
            symmetry and its momentum is not conserved.
        velocities : ndarray, shape (N, 3), optional
            Use these velocities instead of drawing them.
        exact_temperature : bool
            Rescale the drawn velocities so the initial instantaneous
            temperature is exactly ``temperature``.

        Returns
        -------
        MDState
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        if velocities is None:
            if temperature is None or temperature == 0.0:
                vel = np.zeros((configuration.n_atoms, 3))
            else:
                vel = maxwell_boltzmann(
                    configuration.masses,
                    temperature,
                    rng,
                    remove_com=fixed_com,
                    exact_temperature=exact_temperature,
                )
        else:
            vel = np.array(velocities, dtype=np.float64, copy=True)
            if fixed_com:
                vel = remove_com_momentum(vel, configuration.masses)

        state = cls(
            positions=configuration.positions.copy(),
            velocities=vel,
            cell=configuration.cell.copy(),
            masses=configuration.masses.copy(),
            species=configuration.species.copy(),
            pbc=configuration.pbc.copy(),
            symbols=configuration.symbols,
            fixed_com=fixed_com,
            rng=rng,
            info=dict(configuration.info),
        )
        if temperature is not None:
            state.info.setdefault("temperature_setpoint", float(temperature))
        if potential is not None:
            state.refresh(potential)
        return state

    # -- basic properties --------------------------------------------------

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return self.positions.shape[0]

    @property
    def volume(self) -> float:
        """Cell volume in A^3 (``inf`` for a fully open system)."""
        if not np.asarray(self.pbc).any():
            return float("inf")
        return float(abs(np.linalg.det(self.cell)))

    @property
    def n_dof(self) -> int:
        """Momentum degrees of freedom, ``3N - 3`` when the COM is fixed."""
        return n_dof(self.n_atoms, self.fixed_com)

    @property
    def forces(self) -> np.ndarray:
        """``(N, 3)`` cached forces in eV/A.

        Raises
        ------
        RuntimeError
            If no force evaluation has been cached.  Silently returning zeros
            here would let a whole run integrate free particles.
        """
        if self.result is None:
            raise RuntimeError(
                "MDState has no cached Result; call refresh(potential) before stepping"
            )
        return self.result.forces

    @property
    def potential_energy(self) -> float:
        """Cached potential energy in eV."""
        if self.result is None:
            raise RuntimeError("MDState has no cached Result; call refresh(potential)")
        return self.result.energy

    @property
    def virial(self) -> np.ndarray:
        """Cached ``(3, 3)`` virial in eV."""
        if self.result is None or self.result.virial is None:
            raise RuntimeError("MDState has no cached virial; call refresh(potential)")
        return self.result.virial

    # -- thermodynamics ----------------------------------------------------

    def kinetic_energy(self) -> float:
        """Kinetic energy in eV, ``0.5 * MVV2E * sum m v^2``."""
        return kinetic_energy(self.masses, self.velocities)

    def total_energy(self) -> float:
        """Kinetic plus cached potential energy, in eV.

        This is the quantity conserved by :class:`~atomlab.md.integrators.VelocityVerlet`.
        Thermostatted integrators conserve a larger quantity instead; ask the
        integrator for it via ``integrator.conserved_quantity(state)``.
        """
        return self.kinetic_energy() + self.potential_energy

    def temperature(self) -> float:
        """Instantaneous temperature in K.

        Uses ``2 KE = n_dof k_B T`` with :attr:`n_dof`, i.e. ``3N - 3`` degrees
        of freedom whenever :attr:`fixed_com` is set.  Getting this count wrong
        biases the temperature by ``3N / (3N - 3)``, which is 1.2% at 256 atoms
        -- invisible in a plot, fatal in a comparison.
        """
        return 2.0 * self.kinetic_energy() / (self.n_dof * KB)

    def com_velocity(self) -> np.ndarray:
        """``(3,)`` centre-of-mass velocity in A/ps."""
        return com_velocity(self.velocities, self.masses)

    def zero_com_momentum(self) -> None:
        """Remove the centre-of-mass momentum in place."""
        self.velocities = remove_com_momentum(self.velocities, self.masses)

    def kinetic_pressure_tensor(self) -> np.ndarray:
        """``(3, 3)`` kinetic contribution ``MVV2E * sum m v (x) v / V`` in eV/A^3.

        This is a mechanical identity, not a thermodynamic one: no degree-of-
        freedom count enters.  Its trace is exactly ``2 KE / V``, and because
        ``<sum m v^2> = n_dof k_B T`` the ensemble average of the trace is
        ``n_dof k_B T / V`` -- which is where the dof bookkeeping enters the
        pressure, implicitly and correctly.
        """
        mv = self.masses[:, None] * self.velocities
        return MVV2E * (mv.T @ self.velocities) / self.volume

    def pressure_tensor(self, *, unit: str = "bar") -> np.ndarray:
        """``(3, 3)`` pressure tensor: kinetic plus virial contributions.

        Parameters
        ----------
        unit : {'bar', 'GPa', 'eV/A^3'}
            Output unit.  ``bar`` matches the LAMMPS metal-unit convention.

        Returns
        -------
        ndarray, shape (3, 3)
            ``(MVV2E * sum m v (x) v + W) / V`` with ``W = -dU/d eps`` the
            virial of :mod:`atomlab.types`.  Positive under compression.
        """
        if unit not in _PRESSURE_UNITS:
            raise ValueError(f"unknown pressure unit {unit!r}; use one of {sorted(_PRESSURE_UNITS)}")
        return (self.kinetic_pressure_tensor() + self.virial / self.volume) * _PRESSURE_UNITS[unit]

    def pressure(self, *, unit: str = "bar") -> float:
        """Scalar instantaneous pressure ``tr(P)/3``.

        Parameters
        ----------
        unit : {'bar', 'GPa', 'eV/A^3'}
            Output unit; ``bar`` by default.

        Returns
        -------
        float
            ``(2 KE + tr W) / (3 V)`` converted to ``unit``.  Both terms are
            included: the virial alone is not the pressure, and for a dense
            liquid the two terms are of comparable magnitude and opposite sign,
            so omitting the kinetic part is not a small error.
        """
        return float(np.trace(self.pressure_tensor(unit=unit)) / 3.0)

    # -- interoperation ----------------------------------------------------

    def to_configuration(self, *, wrap: bool = False, labels: bool = True) -> Configuration:
        """Return the current geometry as a :class:`~atomlab.types.Configuration`.

        Parameters
        ----------
        wrap : bool
            Wrap positions into the primary cell.  The default is False because
            the state's positions are unwrapped by contract and potentials do
            not care (they apply the minimum image convention).
        labels : bool
            Attach the cached energy/forces/virial if available.

        Returns
        -------
        Configuration
        """
        cfg = Configuration(
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=np.asarray(self.pbc).copy(),
            species=self.species.copy(),
            symbols=self.symbols,
            masses=self.masses.copy(),
            info=dict(self.info),
        )
        cfg.info["time"] = self.time
        cfg.info["step"] = self.step
        if labels and self.result is not None:
            cfg.energy = self.result.energy
            cfg.forces = self.result.forces.copy()
            cfg.virial = None if self.result.virial is None else self.result.virial.copy()
        return cfg.wrapped() if wrap else cfg

    def refresh(self, potential, *, virial: bool = True) -> Result:
        """Recompute and cache the potential at the current geometry.

        Parameters
        ----------
        potential : Potential
            Energy model.
        virial : bool
            Request the virial.  Barostats need it every step; NVE and NVT runs
            need it only when the pressure is logged, and for expensive models
            skipping it is worth the branch.

        Returns
        -------
        Result
        """
        self.result = potential.compute(self.to_configuration(labels=False), forces=True, virial=virial)
        return self.result

    def copy(self) -> "MDState":
        """Deep copy of everything except :attr:`rng`, which is shared.

        The generator is deliberately *not* copied: two states sharing one
        generator continue the same stochastic stream, whereas a copied
        generator would silently replay it.  Pass a fresh ``rng`` explicitly if
        you want an independent branch.
        """
        return MDState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            cell=self.cell.copy(),
            masses=self.masses.copy(),
            species=self.species.copy(),
            pbc=np.asarray(self.pbc).copy(),
            symbols=self.symbols,
            result=None
            if self.result is None
            else Result(
                energy=self.result.energy,
                forces=self.result.forces.copy(),
                virial=None if self.result.virial is None else self.result.virial.copy(),
                energies=None if self.result.energies is None else self.result.energies.copy(),
                extra=dict(self.result.extra),
            ),
            step=self.step,
            time=self.time,
            fixed_com=self.fixed_com,
            thermostat={k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.thermostat.items()},
            barostat={k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.barostat.items()},
            rng=self.rng,
            info=dict(self.info),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        try:
            t = f"{self.temperature():.1f} K"
        except Exception:  # pragma: no cover
            t = "?"
        return f"MDState({self.n_atoms} atoms, step={self.step}, t={self.time:.4f} ps, T={t})"
