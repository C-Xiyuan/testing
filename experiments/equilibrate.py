"""Equilibration protocol and the check that refuses to proceed without it.

This module exists because of a bug that nearly invalidated the whole study.

The experiments build a liquid by taking an fcc lattice and scaling it to liquid
density.  That configuration is at the right density and the right temperature,
and it is a *crystal*.  Sampling it produces configurations that look plausible,
give smooth radial distribution functions, and are wrong: measured directly, the
global bond-order parameter ``Q6`` decayed monotonically from 0.5745 (perfect
fcc) to 0.43 over 600 Monte Carlo proposals and was still falling, while the
potential energy climbed monotonically from -0.0434 to -0.0357 eV/atom and was
still climbing.  The "reference ensemble" was a slowly melting crystal, and its
energy was 30-40 % away from the equilibrated liquid's -0.0311 eV/atom.

Nothing in the sampler was wrong.  The diagnostics it reports -- acceptance,
integration error -- were all healthy, because they measure whether the Markov
chain is being simulated correctly, not whether it has reached its stationary
distribution.  Those are different questions and only the second one matters for
an ensemble average.

Hence two things here, and the second is the important one:

* :func:`equilibrated_configuration` melts the crystal at an elevated
  temperature, anneals to the target, and returns a configuration drawn from the
  liquid.
* :func:`check_equilibrated` compares the first and second halves of a
  production trajectory and **raises** if either the energy or the order
  parameter is still drifting.  A silent pass would reintroduce exactly the
  failure above, so it is not a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atomlab.analysis.statistics import blocking_analysis
from atomlab.sampling import hybrid_monte_carlo
from experiments.observables_lib import bond_orientational_order

__all__ = ["EquilibrationReport", "equilibrated_configuration", "check_equilibrated"]


@dataclass
class EquilibrationReport:
    """What the protocol did, and the evidence that it worked."""

    melt_temperature: float
    n_melt: int
    n_anneal: int
    q6_start: float
    q6_after_melt: float
    q6_final: float
    energy_per_atom: float
    melt_acceptance: float
    anneal_acceptance: float
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EquilibrationReport(Q6 {self.q6_start:.3f} -> {self.q6_final:.3f}, "
            f"U/atom {self.energy_per_atom:.5f} eV, "
            f"acceptance {self.melt_acceptance:.2f}/{self.anneal_acceptance:.2f})"
        )


def equilibrated_configuration(
    configuration,
    potential,
    temperature: float,
    *,
    melt_temperature: float | None = None,
    n_melt: int = 400,
    n_anneal: int = 800,
    n_leapfrog: int = 8,
    step_size: float = 2e-3,
    seed: int = 0,
    q6_cutoff: float = 4.3,
    q6_target: float = 0.20,
) -> tuple:
    """Melt a crystalline starting structure, anneal to the target temperature.

    Parameters
    ----------
    melt_temperature:
        Defaults to five times the target, which for liquid argon at 120 K puts
        the melting stage far above the critical temperature and destroys the
        lattice in a few hundred proposals.  Melting at the target temperature
        works too but takes an order of magnitude longer, because the barrier
        the system has to cross is exactly the one that makes the crystal
        metastable.
    q6_target:
        Maximum acceptable global ``Q6`` after annealing.  A 108-atom liquid
        gives roughly 0.08; a perfect fcc crystal gives 0.5745.  A value above
        this means residual crystalline order and raises.

    Returns
    -------
    (Configuration, EquilibrationReport)
    """
    melt_temperature = melt_temperature or 5.0 * temperature
    q6_start = bond_orientational_order(configuration, q6_cutoff, 6)

    melted, melt_report = hybrid_monte_carlo(
        configuration, potential, melt_temperature,
        n_samples=n_melt, n_leapfrog=n_leapfrog, step_size=step_size,
        burn_in=max(50, n_melt // 2), seed=seed,
    )
    hot = melted.frame(-1)
    q6_hot = bond_orientational_order(hot, q6_cutoff, 6)

    annealed, anneal_report = hybrid_monte_carlo(
        hot, potential, temperature,
        n_samples=n_anneal, n_leapfrog=n_leapfrog, step_size=step_size,
        burn_in=max(100, n_anneal // 2), seed=seed + 1,
    )
    final = annealed.frame(-1)
    q6_final = float(np.mean([
        bond_orientational_order(annealed.frame(i), q6_cutoff, 6)
        for i in range(max(0, annealed.n_frames - 20), annealed.n_frames)
    ]))

    report = EquilibrationReport(
        melt_temperature=melt_temperature,
        n_melt=n_melt,
        n_anneal=n_anneal,
        q6_start=float(q6_start),
        q6_after_melt=float(q6_hot),
        q6_final=q6_final,
        energy_per_atom=float(annealed.scalars["potential_energy"][-20:].mean()
                              / configuration.n_atoms),
        melt_acceptance=melt_report.acceptance,
        anneal_acceptance=anneal_report.acceptance,
    )

    if q6_final > q6_target:
        raise RuntimeError(
            f"annealing left Q6 = {q6_final:.3f}, above the {q6_target} threshold: "
            f"the configuration still has crystalline order and is not a liquid. "
            f"(started at {q6_start:.3f}, after melting {q6_hot:.3f}.) "
            "Increase n_melt or melt_temperature."
        )
    return final, report


def check_equilibrated(
    trajectory,
    *,
    n_sigma: float = 3.0,
    q6_cutoff: float = 4.3,
    check_order: bool = True,
    label: str = "trajectory",
) -> dict:
    """Refuse a production trajectory that is still drifting.

    Splits the trajectory in half and compares the two halves' mean potential
    energy, and optionally their mean ``Q6``.  A stationary chain gives halves
    that agree within their error bars; a chain still relaxing toward its
    stationary distribution gives a systematic difference.

    This is deliberately a hard failure rather than a warning.  The failure it
    guards against does not announce itself -- every other diagnostic looked
    healthy while the reference ensemble was a melting crystal -- so anything
    softer than an exception would eventually be ignored.

    Raises
    ------
    RuntimeError
        If either quantity differs between halves by more than ``n_sigma``
        combined standard errors.
    """
    u = np.asarray(trajectory.scalars["potential_energy"], dtype=float)
    half = u.size // 2
    if half < 8:
        raise ValueError(f"{label}: need at least 16 frames to test for drift, got {u.size}")

    first, second = blocking_analysis(u[:half]), blocking_analysis(u[half:])
    gap = abs(first.value - second.value)
    sigma = float(np.hypot(first.error, second.error))
    result = {
        "energy_first_half": first.value,
        "energy_second_half": second.value,
        "energy_gap": gap,
        "energy_gap_sigma": gap / sigma if sigma > 0 else float("inf"),
    }
    if sigma > 0 and gap > n_sigma * sigma:
        raise RuntimeError(
            f"{label} is not equilibrated: potential energy differs between the first "
            f"and second halves by {gap:.5f} eV = {gap / sigma:.1f} sigma "
            f"({first.value:.5f} vs {second.value:.5f}). The chain is still relaxing, "
            "so its averages are not ensemble averages. Equilibrate for longer."
        )

    if check_order:
        stride = max(1, trajectory.n_frames // 40)
        q6 = np.array([
            bond_orientational_order(trajectory.frame(i), q6_cutoff, 6)
            for i in range(0, trajectory.n_frames, stride)
        ])
        h = q6.size // 2
        q6_gap = abs(q6[:h].mean() - q6[h:].mean())
        q6_sigma = float(np.hypot(q6[:h].std(ddof=1) / np.sqrt(h),
                                  q6[h:].std(ddof=1) / np.sqrt(q6.size - h)))
        result.update({
            "q6_mean": float(q6.mean()),
            "q6_first_half": float(q6[:h].mean()),
            "q6_second_half": float(q6[h:].mean()),
            "q6_gap_sigma": q6_gap / q6_sigma if q6_sigma > 0 else float("inf"),
        })
        if q6_sigma > 0 and q6_gap > n_sigma * q6_sigma:
            raise RuntimeError(
                f"{label} is not equilibrated: bond-order parameter Q6 drifts from "
                f"{q6[:h].mean():.4f} to {q6[h:].mean():.4f} between halves "
                f"({q6_gap / q6_sigma:.1f} sigma). Structural relaxation is incomplete."
            )
    return result
