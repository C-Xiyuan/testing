#!/usr/bin/env python3
"""How large is the error on a difference between two independent MC chains?

exp06 turned up a discrepancy worth chasing. At the smallest perturbation
amplitude, three estimates of the same observable shift were:

    first order     -3.835
    reweighted      -3.884    (exact to all orders; ESS fraction 0.83)
    direct sampling -1.948 +/- 0.821

The first two share the reference samples and agree to 1.3 %. The third
requires an independent chain, and it is the one that disagrees. That points at
the *direct* estimate rather than at the theory, and specifically at its error
bar: blocking estimates the error within a chain, but the difference between two
chains also carries the offset between their slow modes, which no within-chain
estimator can see.

This script measures that offset directly. The same perturbed potential is
sampled from several independent seeds; the scatter of the resulting observable
across seeds is the between-chain error. If it substantially exceeds the
within-chain blocking error, the exp06 discrepancy is explained and the error
bars there were too small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

TEMPERATURE = 120.0
DENSITY = 0.0200
CUTOFF = 7.0
N_SAMPLES = 1500
N_SEEDS = 6


def main():
    cfg = scale_to_density(fcc(5.4, "Ar", (3, 3, 3)), DENSITY)
    potential = LennardJones(epsilon=0.0103, sigma=3.405, cutoff=CUTOFF, mode="shifted_force")
    cfg, eq = equilibrated_configuration(cfg, potential, TEMPERATURE,
                                         n_melt=400, n_anneal=1000, seed=0)
    print(f"equilibration: {eq}", flush=True)

    observable = PairBinObservable(np.linspace(3.0, 7.0, 9), cutoff=CUTOFF)
    perturbation = RadialShellPerturbation(4.2, 0.35, 0.0005, CUTOFF)
    surrogate = potential + perturbation

    def measure(pot, seed):
        traj, report = hybrid_monte_carlo(
            cfg, pot, TEMPERATURE, n_samples=N_SAMPLES, n_leapfrog=8,
            step_size=2e-3, burn_in=300, seed=seed)
        check_equilibrated(traj, label=f"seed{seed}", check_order=False)
        a = observable.evaluate_trajectory(traj)
        return a, blocking_analysis(a), report

    print(f"\nsampling {N_SEEDS} independent chains of each potential "
          f"({N_SAMPLES} frames each)", flush=True)
    reference, perturbed = [], []
    for seed in range(N_SEEDS):
        a_ref, est_ref, rep_ref = measure(potential, 1000 + seed)
        a_pert, est_pert, rep_pert = measure(surrogate, 2000 + seed)
        reference.append((a_ref, est_ref))
        perturbed.append((a_pert, est_pert))
        print(f"  seed {seed}: reference bin4 = {est_ref.value[4]:8.3f} "
              f"+/- {est_ref.error[4]:.3f}   perturbed = {est_pert.value[4]:8.3f} "
              f"+/- {est_pert.error[4]:.3f}   (acceptance {rep_ref.acceptance:.2f})",
              flush=True)

    bin_index = 4
    ref_means = np.array([e.value[bin_index] for _, e in reference])
    pert_means = np.array([e.value[bin_index] for _, e in perturbed])
    within = np.mean([e.error[bin_index] for _, e in reference])

    between_ref = ref_means.std(ddof=1)
    between_pert = pert_means.std(ddof=1)

    print(f"\n--- bin {bin_index} ({observable.centres[bin_index]:.2f} A) ---")
    print(f"within-chain blocking error (mean over chains) : {within:.3f} pairs")
    print(f"between-chain scatter, reference               : {between_ref:.3f} pairs")
    print(f"between-chain scatter, perturbed               : {between_pert:.3f} pairs")
    print(f"ratio between/within                           : "
          f"{max(between_ref, between_pert) / within:.2f}")

    # The estimate exp06 would have reported, and its true spread.
    diffs = pert_means - ref_means
    print(f"\nmeasured shift per seed pair: {np.round(diffs, 3)}")
    print(f"mean {diffs.mean():.3f}, standard error of the mean "
          f"{diffs.std(ddof=1) / np.sqrt(N_SEEDS):.3f}, "
          f"spread of a single pair {diffs.std(ddof=1):.3f}")
    print(f"exp06 quoted +/- {np.hypot(within, within):.3f} for one such pair")

    # The two same-sample estimators, for comparison.
    a_ref_all = np.concatenate([a for a, _ in reference], axis=0)
    frames_traj, _ = hybrid_monte_carlo(cfg, potential, TEMPERATURE,
                                        n_samples=N_SAMPLES * 2, n_leapfrog=8,
                                        step_size=2e-3, burn_in=300, seed=7)
    frames = [frames_traj.frame(i) for i in range(frames_traj.n_frames)]
    a_ref2 = observable.evaluate_trajectory(frames_traj)
    du = np.array([perturbation.energy(c) for c in frames])
    pred = predict_shift(a_ref2, du, TEMPERATURE, n_resamples=400, seed=0)
    rw = reweight(a_ref2, du, TEMPERATURE, n_resamples=200, seed=0)
    print(f"\nsame-sample estimators on a fresh {frames_traj.n_frames}-frame chain:")
    print(f"  first order  {np.asarray(pred.value)[bin_index]:8.3f} "
          f"+/- {np.asarray(pred.error)[bin_index]:.3f}")
    print(f"  reweighted   {np.asarray(rw.shift.value)[bin_index]:8.3f} "
          f"(ESS fraction {rw.ess_fraction:.2f})")


if __name__ == "__main__":
    main()
