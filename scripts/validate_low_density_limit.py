#!/usr/bin/env python3
"""Validate the response estimator against the exact low-density limit.

In the dilute limit of a pair fluid the radial distribution function is known in
closed form,

    g(r) = exp(-beta u(r)) + O(rho)

so perturbing the pair potential by ``du(r)`` changes it by, exactly to first
order,

    dg(r) = -beta g(r) du(r)

This is the strongest available end-to-end test of
:mod:`atomlab.analysis.response`, because it checks the sampled covariance
estimator against a *closed form on a real interacting system* rather than on a
harmonic toy.  Everything is exercised at once: the sampler, the equilibration
protocol, the observable, the covariance estimator, and the block-bootstrap
error bars.

The comparison is made on pair counts rather than on ``g(r)`` itself, so the
prediction becomes

    dA_k = -(N/2) rho integral_k 4 pi r^2 g(r) beta du(r) dr

with the shell integral evaluated by quadrature from the analytic ``u`` and
``du``.  Any disagreement beyond the stated error bars is a defect in the
estimator, not a modelling choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis
from atomlab.build import random_gas
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo
from atomlab.units import KB
from experiments.equilibrate import check_equilibrated
from experiments.observables_lib import PairBinObservable

# A dilute gas: rho* = 0.079, well inside the regime where g = exp(-beta u).
DENSITY = 0.0020        # atoms / A^3
TEMPERATURE = 300.0     # K -- high enough that clustering is negligible
CUTOFF = 7.0
EPSILON, SIGMA = 0.0103, 3.405


def exact_bin_shift(edges, perturbation, n_atoms, volume, temperature, cutoff):
    """Closed-form first-order shift in each bin count, in the dilute limit.

    The naive version of this calculation -- g(r) = exp(-beta u) so
    dg = -beta g du, hence dA_k proportional to the shell integral of g du --
    is **wrong**, and the error is instructive enough to spell out.

    The total number of pairs is exactly N(N-1)/2 whatever the potential. A
    perturbation that depletes one shell cannot simply remove those pairs; they
    reappear elsewhere. The pair separation distribution in a finite closed
    system is therefore *normalised*,

        p(r) = exp(-beta u(r)) / Omega,   Omega = integral over the cell of exp(-beta u)

    and perturbing it gives a difference of two terms,

        dA_k = M I_k / Omega  -  A_k I_tot / Omega

    with M = N(N-1)/2, ``I_k`` the shell integral of ``-beta du exp(-beta u)``
    and ``I_tot`` the same integral over all separations. The second term is the
    sum-rule compensation, and dropping it produces a prediction that is wrong
    by a factor of ten in the outer bins and has the wrong sign in the tail.

    The covariance estimator in :mod:`atomlab.analysis.response` gets this right
    automatically: a covariance is a mean-subtracted quantity, and the
    subtraction *is* the normalisation term. That the two agree only after the
    hand calculation is corrected is a point in the estimator's favour.
    """
    beta = 1.0 / (KB * temperature)
    n_pairs = 0.5 * n_atoms * (n_atoms - 1)

    def shell_integrals(lo, hi):
        r = np.linspace(lo, hi, 800)
        w = np.exp(-beta * pair_energy_lj(r))
        du = np.asarray(perturbation.pair(r)[0])
        weight = np.trapezoid(4.0 * np.pi * r**2 * w, r)
        response = np.trapezoid(4.0 * np.pi * r**2 * w * (-beta * du), r)
        return weight, response

    # Omega: the cell volume corrected for the excluded/attracted volume the
    # potential creates inside the cutoff. Beyond the cutoff w = 1 exactly.
    r_all = np.linspace(1e-3, cutoff, 4000)
    w_all = np.exp(-beta * pair_energy_lj(r_all))
    omega = volume + np.trapezoid(4.0 * np.pi * r_all**2 * (w_all - 1.0), r_all)
    du_all = np.asarray(perturbation.pair(r_all)[0])
    i_tot = np.trapezoid(4.0 * np.pi * r_all**2 * w_all * (-beta * du_all), r_all)

    out = np.empty(len(edges) - 1)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        weight, response = shell_integrals(lo, hi)
        a_k = n_pairs * weight / omega
        out[k] = n_pairs * response / omega - a_k * i_tot / omega
    return out


def pair_energy_lj(r):
    """Shifted-force Lennard-Jones pair energy, evaluated analytically."""
    sr6 = (SIGMA / r) ** 6
    u = 4.0 * EPSILON * (sr6**2 - sr6)
    src6 = (SIGMA / CUTOFF) ** 6
    u_c = 4.0 * EPSILON * (src6**2 - src6)
    f_c = 24.0 * EPSILON * (2 * src6**2 - src6) / CUTOFF
    return np.where(r < CUTOFF, u - u_c + (r - CUTOFF) * f_c, 0.0)


def main():
    n_reference = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

    # A dilute gas has no lattice to melt and no bond-order parameter to
    # measure -- at this density the nearest neighbour of an fcc arrangement
    # would sit 8.9 A away, outside any sensible Q6 cutoff. So the crystalline
    # starting point and its melt/anneal protocol are both inapplicable here;
    # a random gas at a safe minimum separation is already the right ensemble's
    # support, and a long burn-in plus the drift check does the rest.
    cfg = random_gas(64, DENSITY, "Ar", seed=0, min_distance=3.0)
    potential = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF, mode="shifted_force")
    print(f"system: {cfg.n_atoms} atoms, L = {cfg.cell[0, 0]:.2f} A, "
          f"rho* = {DENSITY * SIGMA**3:.3f}, T* = {KB * TEMPERATURE / EPSILON:.3f}")

    traj, report = hybrid_monte_carlo(
        cfg, potential, TEMPERATURE, n_samples=n_reference, n_leapfrog=8,
        step_size=2e-3, burn_in=2000, seed=1, progress_every=2000,
    )
    print(f"sampling: {report}")
    check_equilibrated(traj, label="dilute gas", check_order=False)

    edges = np.linspace(3.2, 6.0, 8)
    observable = PairBinObservable(edges, cutoff=CUTOFF)
    a_samples = observable.evaluate_trajectory(traj)
    counts = blocking_analysis(a_samples)

    # A weak shell perturbation, well inside linear response.
    perturbation = RadialShellPerturbation(4.2, 0.4, 0.0008, CUTOFF)
    frames = [traj.frame(i) for i in range(traj.n_frames)]
    du = np.array([perturbation.energy(c) for c in frames])

    prediction = predict_shift(a_samples, du, TEMPERATURE, n_resamples=600, seed=0)
    rw = reweight(a_samples, du, TEMPERATURE, n_resamples=300, seed=0)
    exact = exact_bin_shift(edges, perturbation, cfg.n_atoms, cfg.volume, TEMPERATURE, CUTOFF)

    predicted = np.asarray(prediction.value, dtype=float)
    errors = np.asarray(prediction.error, dtype=float)

    print(f"\nperturbation force RMSE = {perturbation.force_rms(frames):.3e} eV/A, "
          f"beta*sd(dU) = {prediction.beta_sigma_dU:.4f}")
    print(f"{'r (A)':>8} {'<A>':>9} {'exact dA':>10} {'estimated dA':>18} "
          f"{'reweighted':>11} {'sigma off':>10}")
    deviations = []
    for k, r in enumerate(observable.centres):
        sigma_off = (predicted[k] - exact[k]) / errors[k] if errors[k] > 0 else np.nan
        deviations.append(sigma_off)
        print(f"{r:8.2f} {counts.value[k]:9.3f} {exact[k]:10.4f} "
              f"{predicted[k]:10.4f} +/-{errors[k]:6.4f} {np.asarray(rw.shift.value)[k]:11.4f} "
              f"{sigma_off:10.2f}")

    deviations = np.array(deviations)
    print(f"\nmax |deviation| = {np.nanmax(np.abs(deviations)):.2f} sigma "
          f"over {len(deviations)} bins")
    print(f"rms deviation   = {np.sqrt(np.nanmean(deviations**2)):.2f} sigma "
          f"(expect ~1 if the estimator and its error bars are both right)")
    print(f"second-order ratio (median) = "
          f"{np.median(np.asarray(prediction.second_order_ratio)):.4f}")
    print(f"reweighting ESS fraction    = {rw.ess_fraction:.3f}")


if __name__ == "__main__":
    main()
