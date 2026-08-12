#!/usr/bin/env python3
"""An exact benchmark for the response estimator, with no density expansion.

The review's P0-1 makes a point that the manuscript conceded without acting on.
`scripts/validate_low_density_limit.py` compares the estimator against
`g(r) = exp(-beta u)`, which is the leading term of a density series and is
wrong at O(rho); at rho* = 0.079 the correction is not negligible, so a
disagreement of 4.7 sigma there is as easily the benchmark's error as the
estimator's. A test whose reference is approximate cannot certify an estimator
that is not.

Two particles in a periodic box fixes this, because at N = 2 the pair
distribution is not a series -- it is a one-dimensional integral. With minimum
image and a potential of range `r_c < L/2`, the separation distribution is
exactly proportional to `4 pi r^2 exp(-beta u(r))` for all `r < L/2`, and the
normalisation is a 1-D quadrature as well:

    <A_bin> = 4 pi Int_a^b r^2 e^{-beta u} dr
              / ( L^3 + 4 pi Int_0^{r_c} r^2 (e^{-beta u} - 1) dr )

because `e^{-beta u} = 1` outside the cutoff, so the whole box contributes `L^3`
and only the cutoff sphere carries a correction. Every quantity on the right is
exact to quadrature precision. There is no O(rho) term to blame a disagreement
on: at N = 2 the density expansion terminates.

The test then asks three things of the estimator, at a perturbation small
enough that linear response should be exact for practical purposes:

  1. does the reference ensemble sampled by HMC reproduce the exact <A>?
     -- if not, the sampler is wrong and nothing downstream means anything;
  2. does `-beta Cov(A, dU)` reproduce the exact shift?
  3. does exponential reweighting, which is exact to all orders, reproduce it?

A failure of (1) is a sampler bug. A failure of (2) with (3) passing is the
truncation. A failure of both with (1) passing is an estimator bug, and would be
the most serious outcome available in this repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo
from atomlab.types import Configuration
from atomlab.units import beta as inverse_temperature
from experiments.observables_lib import PairBinObservable

TEMPERATURE = 120.0
BOX = 12.0            # A; minimum image needs r_c <= L/2 = 6.0
CUTOFF = 5.0          # A
EPSILON, SIGMA = 0.0103, 3.405
BINS = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
N_SAMPLES = 60000
BURN_IN = 2000
AMPLITUDE = 5.0e-4    # eV, the amplitude of the disputed exp06 row


def exact_bin_means(pair_energy, edges, box, cutoff, temperature):
    """Exact <A_k> for two particles, by 1-D quadrature.

    ``pair_energy(r)`` must vanish for ``r >= cutoff`` and ``edges[-1]`` must not
    exceed ``box / 2``, or the spherical shell leaves the minimum-image cell and
    the integral below stops being the whole story.
    """
    if edges[-1] > 0.5 * box:
        raise ValueError("bins must lie inside the minimum-image sphere")
    b = inverse_temperature(temperature)

    def boltzmann(r):
        return np.exp(-b * pair_energy(r))

    # Normalisation: L^3 outside the cutoff plus the correction inside it.
    correction, err_c = quad(lambda r: r * r * (boltzmann(r) - 1.0), 0.0, cutoff,
                             limit=200)
    z = box**3 + 4.0 * np.pi * correction

    means, errors = [], []
    for a, c in zip(edges[:-1], edges[1:]):
        value, err = quad(lambda r: r * r * boltzmann(r), a, c, limit=200)
        means.append(4.0 * np.pi * value / z)
        errors.append(4.0 * np.pi * (err + abs(err_c) * value / z) / z)
    return np.array(means), np.array(errors)


def two_particle_configuration(box, rng):
    positions = np.array([[0.25 * box, 0.5 * box, 0.5 * box],
                          [0.65 * box, 0.5 * box, 0.5 * box]])
    return Configuration(positions=positions, cell=np.eye(3) * box, pbc=True,
                         symbols=("Ar", "Ar"),
                         masses=np.array([39.948, 39.948]))


def main():
    rng = np.random.default_rng(0)
    cfg = two_particle_configuration(BOX, rng)
    reference = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF,
                             mode="shifted_force")
    field = RadialShellPerturbation(r0=4.2, width=0.35, amplitude=AMPLITUDE,
                                    cutoff=CUTOFF)
    surrogate = reference + field
    observable = PairBinObservable(BINS, cutoff=CUTOFF)
    centres = 0.5 * (BINS[:-1] + BINS[1:])

    print(f"two particles, L = {BOX} A, cutoff = {CUTOFF} A, T = {TEMPERATURE} K")
    print(f"bins {BINS}, all inside the minimum-image sphere at {0.5 * BOX} A")
    print(f"perturbation: Gaussian shell r0 = 4.2, w = 0.35, a = {AMPLITUDE:.1e} eV\n")

    # `pair` returns (energy, force); the composite has no `pair`, so the two
    # terms are summed here rather than relying on an accessor that only the
    # leaf potentials have.
    def u_reference(r):
        return float(np.asarray(reference.pair(r)).ravel()[0])

    def u_surrogate(r):
        return u_reference(r) + float(np.asarray(field.pair(r)).ravel()[0])

    exact_ref, q_ref = exact_bin_means(u_reference, BINS, BOX, CUTOFF, TEMPERATURE)
    exact_sur, q_sur = exact_bin_means(u_surrogate, BINS, BOX, CUTOFF, TEMPERATURE)
    exact_shift = exact_sur - exact_ref
    print(f"quadrature error on <A>: {max(q_ref.max(), q_sur.max()):.2e} "
          f"(negligible against everything below)")

    traj, report = hybrid_monte_carlo(cfg, reference, TEMPERATURE,
                                      n_samples=N_SAMPLES, n_leapfrog=8,
                                      step_size=5e-3, burn_in=BURN_IN, seed=1,
                                      remove_com=False)
    print(f"reference chain: {report}")
    a = observable.evaluate_trajectory(traj)
    sampled = a.mean(axis=0)
    sampled_err = np.asarray(blocking_analysis(a).error)

    frames = [traj.frame(i) for i in range(traj.n_frames)]
    du = np.array([field.energy(c) for c in frames])
    pred = predict_shift(a, du, TEMPERATURE, n_resamples=600, seed=2)
    rw = reweight(a, du, TEMPERATURE, n_resamples=300, seed=3)

    print(f"\n(1) does the sampler reproduce the exact reference ensemble?")
    print(f"  {'r (A)':>7}{'exact':>12}{'sampled':>18}{'sigma':>8}")
    for i, r in enumerate(centres):
        z = (sampled[i] - exact_ref[i]) / sampled_err[i]
        print(f"  {r:>7.2f}{exact_ref[i]:>12.5f}"
              f"{sampled[i]:>11.5f} +/-{sampled_err[i]:.5f}{z:>8.2f}")
    z_sampler = (sampled - exact_ref) / sampled_err
    print(f"  rms {np.sqrt((z_sampler**2).mean()):.2f} sigma over {len(centres)} bins")

    print(f"\n(2,3) do the estimators reproduce the exact shift?")
    print(f"  {'r (A)':>7}{'exact shift':>14}{'linear':>20}{'sigma':>8}"
          f"{'reweighted':>14}{'sigma':>8}")
    lin = np.asarray(pred.value)
    lin_e = np.asarray(pred.error)
    rew = np.asarray(rw.shift.value)
    rew_e = np.asarray(rw.shift.error)
    for i, r in enumerate(centres):
        zl = (lin[i] - exact_shift[i]) / lin_e[i]
        zr = (rew[i] - exact_shift[i]) / rew_e[i]
        print(f"  {r:>7.2f}{exact_shift[i]:>14.6f}"
              f"{lin[i]:>13.6f} +/-{lin_e[i]:.6f}{zl:>8.2f}"
              f"{rew[i]:>14.6f}{zr:>8.2f}")
    z_lin = (lin - exact_shift) / lin_e
    z_rew = (rew - exact_shift) / rew_e
    print(f"  rms: linear {np.sqrt((z_lin**2).mean()):.2f} sigma, "
          f"reweighted {np.sqrt((z_rew**2).mean()):.2f} sigma")
    print(f"  second-order ratio (median) {np.median(np.asarray(pred.second_order_ratio)):.4f}, "
          f"beta*sd(dU) {pred.beta_sigma_dU:.4f}, "
          f"reweighting ESS fraction {rw.ess_fraction:.3f}")

    ok_sampler = np.sqrt((z_sampler**2).mean()) < 2.0
    ok_linear = np.sqrt((z_lin**2).mean()) < 2.0
    ok_reweight = np.sqrt((z_rew**2).mean()) < 2.0
    print()
    if not ok_sampler:
        print("VERDICT: the sampler does not reproduce an exactly known ensemble. "
              "Nothing downstream of it can be interpreted until that is fixed.")
    elif ok_linear and ok_reweight:
        print("VERDICT: sampler, linear response and exact reweighting all agree "
              "with a reference that involves no density expansion. The 4.7 sigma "
              "of the dilute-gas check is a property of that check's O(rho) "
              "reference, not of the estimator.")
    elif ok_reweight and not ok_linear:
        print("VERDICT: reweighting agrees and linear response does not, which "
              "localises the disagreement in the truncation rather than the "
              "machinery.")
    else:
        print("VERDICT: the estimators disagree with an exact reference while the "
              "sampler reproduces it. This is an estimator defect and the "
              "prediction claims must be withdrawn.")

    out = ROOT / "results/validation/two_particle_exact.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"two particles, L={BOX}, cutoff={CUTOFF}, T={TEMPERATURE}, "
        f"n_samples={N_SAMPLES}, amplitude={AMPLITUDE}\n"
        f"bins {BINS.tolist()}\n"
        f"exact reference {exact_ref.tolist()}\n"
        f"sampled         {sampled.tolist()}\n"
        f"sampled error   {sampled_err.tolist()}\n"
        f"exact shift     {exact_shift.tolist()}\n"
        f"linear          {lin.tolist()}\n"
        f"linear error    {lin_e.tolist()}\n"
        f"reweighted      {rew.tolist()}\n"
        f"reweighted err  {rew_e.tolist()}\n"
        f"z sampler {z_sampler.tolist()}\nz linear {z_lin.tolist()}\n"
        f"z reweight {z_rew.tolist()}\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
