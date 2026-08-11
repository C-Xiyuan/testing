"""Canonical-ensemble samplers built only on the :class:`Potential` interface.

The response theory in :mod:`atomlab.analysis.response` needs samples from the
canonical distribution ``exp(-beta U0)`` and nothing else.  Molecular dynamics
can supply them, but only if its thermostat is correct, and thermostat
correctness is exactly the kind of subtle thing that would contaminate a study
whose subject is subtle errors.  These samplers sidestep that: they sample the
canonical distribution *exactly* by construction, with the Metropolis criterion
absorbing any integration error into the acceptance rate rather than into the
distribution.

Two are provided:

* :func:`hybrid_monte_carlo` -- the workhorse.  Proposes a whole new
  configuration by running a short Hamiltonian trajectory from randomised
  momenta, then accepts or rejects on the change in the total Hamiltonian.
  Because it moves every atom at once along the forces, it decorrelates far
  faster than local moves in a dense liquid, and it needs nothing from a
  potential beyond energy and forces -- so a learned model works here exactly as
  an analytic one does.
* :func:`metropolis_nvt` -- single-particle displacement moves.  Slow for large
  systems because a general potential offers no local energy, but it depends on
  *nothing* except the total energy, which makes it the independent check that
  the HMC implementation is sampling what it claims to.

.. warning::
   Neither sampler produces dynamics.  Configurations are drawn from the right
   distribution, but the sequence carries no physical time, so mean squared
   displacements, velocity autocorrelations and transport coefficients must not
   be computed from these trajectories.  :attr:`Trajectory.info` records
   ``dynamical=False`` so downstream code can refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Configuration, Trajectory
from .units import KB, MVV2E, E2MVV

__all__ = ["SamplerReport", "hybrid_monte_carlo", "metropolis_nvt", "maxwell_velocities"]


@dataclass
class SamplerReport:
    """Diagnostics that decide whether a sampler run can be trusted.

    Attributes
    ----------
    acceptance:
        Fraction of proposals accepted.  For HMC the efficient range is roughly
        0.6-0.9; far below that means the timestep is too large, and near 1.0
        means it is too small and the sampler is wasting force evaluations
        taking timid steps.
    n_proposals, n_accepted:
        Raw counts.
    energy_drift_per_step:
        Mean ``|dH|`` over the accepted trajectories, in eV.  This is the
        integrator error the Metropolis test is correcting for; if it is large
        compared with ``k_B T`` the acceptance will collapse.
    final_step_size:
        Timestep after adaptation, in ps.
    temperature_measured:
        Temperature inferred from the sampled potential-energy fluctuations via
        the configurational relation, used as an independent check that the
        sampler produced the requested ensemble.
    """

    acceptance: float
    n_proposals: int
    n_accepted: int
    energy_drift_per_step: float
    final_step_size: float
    temperature_measured: float = float("nan")
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SamplerReport(acceptance={self.acceptance:.2f}, "
            f"dt={self.final_step_size*1e3:.2f} fs, |dH|={self.energy_drift_per_step:.2e} eV)"
        )


def maxwell_velocities(masses: np.ndarray, temperature: float, rng, *, remove_com: bool = True):
    """Draw velocities in A/ps from the Maxwell-Boltzmann distribution.

    The width is ``sqrt(k_B T / (m * MVV2E))`` because kinetic energy in metal
    units is ``0.5 * MVV2E * m v^2``; dropping that conversion is the classic
    metal-units bug and produces temperatures wrong by four orders of magnitude.
    """
    masses = np.asarray(masses, dtype=float)
    sigma = np.sqrt(KB * temperature / (masses * MVV2E))
    velocities = rng.normal(0.0, 1.0, size=(masses.size, 3)) * sigma[:, None]
    if remove_com:
        total_mass = masses.sum()
        velocities -= (masses[:, None] * velocities).sum(axis=0) / total_mass
    return velocities


def _kinetic(masses: np.ndarray, velocities: np.ndarray) -> float:
    return float(0.5 * MVV2E * (masses[:, None] * velocities**2).sum())


def hybrid_monte_carlo(
    configuration: Configuration,
    potential,
    temperature: float,
    *,
    n_samples: int = 500,
    n_leapfrog: int = 12,
    step_size: float = 2e-3,
    stride: int = 1,
    burn_in: int = 50,
    seed: int = 0,
    adapt: bool = True,
    target_acceptance: float = 0.75,
    remove_com: bool = True,
    progress_every: int | None = None,
) -> tuple[Trajectory, SamplerReport]:
    """Sample the canonical ensemble by Hamiltonian Monte Carlo.

    Each proposal draws fresh momenta from Maxwell-Boltzmann, integrates
    ``n_leapfrog`` velocity-Verlet steps of size ``step_size``, and accepts with
    probability ``min(1, exp(-beta dH))``.  Velocity Verlet is symplectic and
    time-reversible, which is what makes the acceptance test exact: the proposal
    is symmetric, so no Jacobian correction is needed and the stationary
    distribution is exactly ``exp(-beta U)`` regardless of the integration error.

    Parameters
    ----------
    configuration:
        Starting geometry.  Should already be near equilibrium; ``burn_in``
        proposals are discarded before recording, but a wildly wrong starting
        density will not fix itself in an NVT sampler.
    temperature:
        Kelvin.
    n_samples:
        Number of *recorded* frames.  The number of proposals is
        ``burn_in + n_samples * stride``.
    n_leapfrog, step_size:
        Trajectory length and timestep (ps).  Longer trajectories decorrelate
        more per proposal but cost proportionally more force evaluations and
        accumulate more integration error.
    stride:
        Record every ``stride``-th proposal.  Successive HMC samples are
        correlated; recording every one wastes memory without adding
        information, but note that the error-bar machinery estimates the
        correlation itself, so a stride is an optimisation and not a fix.
    adapt:
        During burn-in, scale ``step_size`` toward ``target_acceptance``.
        Adaptation stops before recording begins -- an adaptive proposal is no
        longer a fixed Markov kernel and would bias the samples.
    remove_com:
        Draw momenta with zero net momentum.  Correct and desirable for a
        translationally invariant potential, where the centre-of-mass position
        is not a physical degree of freedom.  It must be ``False`` for a
        potential that breaks translational invariance -- an Einstein crystal or
        any tethered system -- because there the centre of mass *is* a real
        degree of freedom, and suppressing its momentum under-samples it and
        biases the potential energy low by roughly ``3/(3N)``.

    Returns
    -------
    (Trajectory, SamplerReport)
        The trajectory carries ``info["dynamical"] = False``.

    Notes
    -----
    The momenta are fully refreshed every proposal, which is the simplest
    correct choice and makes successive proposals independent apart from the
    position correlation.  Partial momentum refreshment would decorrelate faster
    but introduces a tuning parameter that would have to be justified, and the
    systems here are small enough that it does not pay.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / (KB * temperature)
    masses = configuration.masses
    inv_mass = 1.0 / (masses * MVV2E)          # F/m in A/ps^2 per eV/A

    current = configuration.copy()
    result = potential.compute(current, forces=True, virial=False)
    u_current, f_current = result.energy, result.forces

    frames, cells, scalars_u, scalars_p = [], [], [], []
    n_accepted = 0
    drift_sum = 0.0
    n_proposals = burn_in + n_samples * stride
    dt = float(step_size)
    window_accepted = 0   # acceptances since the last adaptation step

    for step in range(n_proposals):
        velocities = maxwell_velocities(masses, temperature, rng, remove_com=remove_com)
        k_current = _kinetic(masses, velocities)

        positions = current.positions.copy()
        f = f_current.copy()
        proposal = current.copy()

        # Velocity Verlet. The half-kick/drift/half-kick form is what makes the
        # map volume-preserving and reversible; an Euler proposal would need a
        # Jacobian factor and would not satisfy detailed balance as written.
        for _ in range(n_leapfrog):
            velocities += 0.5 * dt * f * inv_mass[:, None]
            positions += dt * velocities
            proposal.positions = positions
            f = potential.compute(proposal, forces=True, virial=False).forces
            velocities += 0.5 * dt * f * inv_mass[:, None]

        proposal.positions = positions
        u_proposal = potential.compute(proposal, forces=False, virial=False).energy
        k_proposal = _kinetic(masses, velocities)

        delta_h = (u_proposal + k_proposal) - (u_current + k_current)
        drift_sum += abs(delta_h)

        if delta_h <= 0.0 or rng.random() < np.exp(-beta * delta_h):
            current = proposal
            u_current, f_current = u_proposal, f
            n_accepted += 1
            window_accepted += 1

        if adapt and step < burn_in and step > 0 and step % 10 == 0:
            # Adapt on the acceptance in the last window only. A cumulative rate
            # lags badly: the high acceptance of the first few timid steps keeps
            # pushing dt up long after it has become too large, and the sampler
            # settles at an acceptance far below target.
            rate = window_accepted / 10.0
            window_accepted = 0
            # Acceptance depends on dt through an integration error growing as
            # dt^2, so a gentle multiplicative factor converges without ringing.
            dt *= 1.1 if rate > target_acceptance else 0.9
            dt = float(np.clip(dt, 1e-5, 5e-2))

        if step >= burn_in and (step - burn_in) % stride == 0:
            frames.append(current.positions.copy())
            cells.append(current.cell.copy())
            scalars_u.append(u_current)
            if progress_every and len(frames) % progress_every == 0:
                print(f"    hmc: {len(frames)}/{n_samples} frames, "
                      f"acceptance {n_accepted/(step+1):.2f}, dt {dt*1e3:.2f} fs", flush=True)

    acceptance = n_accepted / n_proposals
    report = SamplerReport(
        acceptance=acceptance,
        n_proposals=n_proposals,
        n_accepted=n_accepted,
        energy_drift_per_step=drift_sum / n_proposals,
        final_step_size=dt,
    )
    if acceptance < 0.2:
        report.notes.append(
            f"acceptance {acceptance:.2f} is low; the sampler is barely moving and the "
            "samples are strongly correlated. Reduce step_size or n_leapfrog."
        )
    if acceptance > 0.98:
        report.notes.append(
            f"acceptance {acceptance:.2f} is near unity; step_size is likely too small "
            "and force evaluations are being wasted on tiny moves."
        )

    trajectory = Trajectory(
        positions=np.array(frames),
        cells=np.array(cells),
        times=np.arange(len(frames), dtype=float) * stride,
        template=configuration.stripped(),
        velocities=None,
        scalars={"potential_energy": np.array(scalars_u)},
        info={
            "sampler": "hybrid_monte_carlo",
            "dynamical": False,
            "temperature": temperature,
            "acceptance": acceptance,
            "seed": seed,
        },
    )
    return trajectory, report


def metropolis_nvt(
    configuration: Configuration,
    potential,
    temperature: float,
    *,
    n_sweeps: int = 200,
    max_displacement: float = 0.1,
    stride: int = 1,
    burn_in: int = 20,
    seed: int = 0,
    adapt: bool = True,
) -> tuple[Trajectory, SamplerReport]:
    """Single-particle Metropolis sampling, as an independent check on HMC.

    One sweep attempts ``N`` single-atom displacements.  Because a general
    :class:`Potential` exposes only the total energy, each attempt costs a full
    energy evaluation, making this ``O(N)`` times more expensive per sweep than
    a specialised pair-potential implementation.  That is acceptable for its
    purpose: it shares no code with :func:`hybrid_monte_carlo` beyond the
    potential itself, so agreement between the two is real evidence rather than
    a consistency check of one implementation with itself.

    ``max_displacement`` is adapted during burn-in toward 50% acceptance, the
    standard heuristic for local moves.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / (KB * temperature)

    current = configuration.copy()
    u_current = potential.energy(current)
    n_atoms = current.n_atoms
    delta = float(max_displacement)

    frames, cells, energies = [], [], []
    n_accepted = 0
    n_proposals = 0

    for sweep in range(burn_in + n_sweeps * stride):
        for _ in range(n_atoms):
            i = int(rng.integers(n_atoms))
            shift = rng.uniform(-delta, delta, size=3)
            trial = current.copy()
            trial.positions[i] += shift
            u_trial = potential.energy(trial)
            n_proposals += 1
            if u_trial <= u_current or rng.random() < np.exp(-beta * (u_trial - u_current)):
                current, u_current = trial, u_trial
                n_accepted += 1

        if adapt and sweep < burn_in:
            rate = n_accepted / max(n_proposals, 1)
            delta *= 1.1 if rate > 0.5 else 0.9
            delta = float(np.clip(delta, 1e-3, 1.0))

        if sweep >= burn_in and (sweep - burn_in) % stride == 0:
            frames.append(current.positions.copy())
            cells.append(current.cell.copy())
            energies.append(u_current)

    report = SamplerReport(
        acceptance=n_accepted / max(n_proposals, 1),
        n_proposals=n_proposals,
        n_accepted=n_accepted,
        energy_drift_per_step=0.0,
        final_step_size=delta,
    )
    trajectory = Trajectory(
        positions=np.array(frames),
        cells=np.array(cells),
        times=np.arange(len(frames), dtype=float) * stride,
        template=configuration.stripped(),
        velocities=None,
        scalars={"potential_energy": np.array(energies)},
        info={
            "sampler": "metropolis_nvt",
            "dynamical": False,
            "temperature": temperature,
            "acceptance": report.acceptance,
            "seed": seed,
        },
    )
    return trajectory, report
