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

import json
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
# The first is exp06's disputed row. The rest climb until linear response has
# to fail: with a single pair, beta*sd(dU) is far smaller than the 0.43 of the
# 108-atom system at the same amplitude, so reaching the regime where the
# truncation is actually in question needs a much larger bump. The exact
# reference does not care how large -- that is the point of using it.
AMPLITUDES = [5.0e-4, 2.0e-3, 8.0e-3, 2.0e-2, 5.0e-2, 1.0e-1]


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


def two_particle_configuration(box):
    positions = np.array([[0.25 * box, 0.5 * box, 0.5 * box],
                          [0.65 * box, 0.5 * box, 0.5 * box]])
    return Configuration(positions=positions, cell=np.eye(3) * box, pbc=True,
                         symbols=("Ar", "Ar"),
                         masses=np.array([39.948, 39.948]))


def main():
    cfg = two_particle_configuration(BOX)
    reference = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF,
                             mode="shifted_force")
    observable = PairBinObservable(BINS, cutoff=CUTOFF)
    centres = 0.5 * (BINS[:-1] + BINS[1:])

    print(f"two particles, L = {BOX} A, cutoff = {CUTOFF} A, T = {TEMPERATURE} K")
    print(f"bins {BINS}, all inside the minimum-image sphere at {0.5 * BOX} A\n")

    def u_reference(r):
        return float(np.asarray(reference.pair(r)).ravel()[0])

    # One reference chain serves every amplitude: the perturbation enters only
    # through dU evaluated on stored frames, so the sweep costs nothing beyond
    # the first chain.
    traj, report = hybrid_monte_carlo(cfg, reference, TEMPERATURE,
                                      n_samples=N_SAMPLES, n_leapfrog=8,
                                      step_size=5e-3, burn_in=BURN_IN, seed=1,
                                      remove_com=False)
    print(f"reference chain: {report}")
    a = observable.evaluate_trajectory(traj)
    frames = [traj.frame(i) for i in range(traj.n_frames)]
    sampled = a.mean(axis=0)
    sampled_err = np.asarray(blocking_analysis(a).error)

    exact_ref, q_ref = exact_bin_means(u_reference, BINS, BOX, CUTOFF, TEMPERATURE)
    z_sampler = (sampled - exact_ref) / sampled_err
    print(f"\n(1) does the sampler reproduce the exactly known reference ensemble?")
    print(f"  {'r (A)':>7}{'exact':>12}{'sampled':>18}{'sigma':>8}")
    for i, r in enumerate(centres):
        print(f"  {r:>7.2f}{exact_ref[i]:>12.5f}"
              f"{sampled[i]:>11.5f} +/-{sampled_err[i]:.5f}{z_sampler[i]:>8.2f}")
    print(f"  rms {np.sqrt((z_sampler**2).mean()):.2f} sigma over {len(centres)} bins")
    if np.sqrt((z_sampler**2).mean()) > 2.0:
        print("\n  The sampler does not reproduce an exactly known ensemble. "
              "Nothing below can be interpreted until that is fixed.")
        return

    print(f"\n(2,3) the estimators against an exact shift, as the perturbation grows")
    print(f"  {'amplitude':>10}{'beta*sd(dU)':>13}{'2nd/1st':>9}{'ESS':>7}"
          f"{'|exact shift|':>15}{'linear':>10}{'reweighted':>13}"
          f"{'lin bias':>11}")
    print(f"  {'(eV)':>10}{'':>13}{'':>9}{'':>7}{'(pairs)':>15}"
          f"{'(sigma)':>10}{'(sigma)':>13}{'(%)':>11}")
    rows = []
    for k, amplitude in enumerate(AMPLITUDES):
        field = RadialShellPerturbation(r0=4.2, width=0.35, amplitude=amplitude,
                                        cutoff=CUTOFF)
        exact_sur, _ = exact_bin_means(
            lambda r, f=field: u_reference(r) + float(np.asarray(f.pair(r)).ravel()[0]),
            BINS, BOX, CUTOFF, TEMPERATURE)
        exact_shift = exact_sur - exact_ref
        du = np.array([field.energy(c) for c in frames])
        pred = predict_shift(a, du, TEMPERATURE, n_resamples=400, seed=2 + k)
        rw = reweight(a, du, TEMPERATURE, n_resamples=200, seed=100 + k)
        lin, lin_e = np.asarray(pred.value), np.asarray(pred.error)
        rew, rew_e = np.asarray(rw.shift.value), np.asarray(rw.shift.error)
        z_lin = np.sqrt((((lin - exact_shift) / lin_e) ** 2).mean())
        z_rew = np.sqrt((((rew - exact_shift) / rew_e) ** 2).mean())
        # Fractional bias of the linear prediction on the bin that moves most,
        # which is the number a practitioner would actually be misled by.
        j = int(np.argmax(np.abs(exact_shift)))
        bias = 100.0 * (lin[j] - exact_shift[j]) / exact_shift[j]
        ratio = float(np.median(np.asarray(pred.second_order_ratio)))
        print(f"  {amplitude:>10.1e}{pred.beta_sigma_dU:>13.3f}{ratio:>9.3f}"
              f"{rw.ess_fraction:>7.3f}{abs(exact_shift[j]):>15.5f}"
              f"{z_lin:>10.2f}{z_rew:>13.2f}{bias:>11.1f}")
        rows.append({"amplitude": amplitude, "beta_sigma_dU": pred.beta_sigma_dU,
                     "second_order_ratio": ratio, "ess_fraction": rw.ess_fraction,
                     "exact_shift": exact_shift.tolist(),
                     "linear": lin.tolist(), "linear_error": lin_e.tolist(),
                     "reweighted": rew.tolist(), "reweighted_error": rew_e.tolist(),
                     "rms_sigma_linear": float(z_lin),
                     "rms_sigma_reweighted": float(z_rew),
                     "linear_bias_percent": float(bias)})

    ok_small = rows[0]["rms_sigma_linear"] < 2.0 and rows[0]["rms_sigma_reweighted"] < 2.0
    print()
    if ok_small:
        print("VERDICT at the disputed amplitude: sampler, linear response and exact")
        print("  reweighting all agree with a reference carrying no density expansion.")
        print("  The 4.7 sigma of the dilute-gas check is a property of that check's")
        print("  O(rho) reference, not a defect in the estimator.")
    else:
        print("VERDICT: the estimators disagree with an exact reference while the")
        print("  sampler reproduces it. This is an estimator defect and the")
        print("  prediction claims must be withdrawn.")

    breaks = [r for r in rows if abs(r["linear_bias_percent"]) > 10.0]
    if breaks:
        first = breaks[0]
        print(f"\n  Linear response first exceeds 10% bias at beta*sd(dU) = "
              f"{first['beta_sigma_dU']:.2f}, where the second-order ratio reads "
              f"{first['second_order_ratio']:.3f}.")
    else:
        print(f"\n  Linear response stays within 10% bias over the whole sweep, up "
              f"to beta*sd(dU) = {rows[-1]['beta_sigma_dU']:.2f}.")
    print("  Caveat: with one pair, beta*sd(dU) at a given amplitude is far below")
    print("  its value in the 108-atom system, so the exact test reaches the")
    print("  breakdown regime only through amplitudes no fitted model would have.")

    out = ROOT / "results/validation/two_particle_exact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "box": BOX, "cutoff": CUTOFF, "temperature": TEMPERATURE,
        "n_samples": N_SAMPLES, "bins": BINS.tolist(),
        "exact_reference": exact_ref.tolist(), "sampled": sampled.tolist(),
        "sampled_error": sampled_err.tolist(),
        "sampler_z": z_sampler.tolist(),
        "amplitudes": rows,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
