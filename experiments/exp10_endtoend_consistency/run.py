#!/usr/bin/env python3
"""exp10 -- the release blocker: do the reference-based estimators and direct MD agree?

Review item P0-1. `results/exp06_response_validation/amplitude_records.json`
contains a disagreement the manuscript did not resolve. At the smallest
amplitude, where the second-order ratio is 0.006 and linear response should be
exact for practical purposes, the two reference-based estimators agree with each
other to 1% -- linear -3.835, reweighted -3.884 pairs -- and direct sampling
says -1.948 +/- 0.821. Two estimators built from the reference ensemble agree,
and the ensemble they claim to describe does not.

Either the reference-based prediction is wrong, in which case the central claim
of this repository fails, or the direct measurement is wrong, in which case
every "measured" column in the manuscript is suspect. Both possibilities are
worse than the missing experiment, so this is the experiment.

Two candidate explanations survive a look at the raw curves, and they point in
opposite directions, which is why this needs an experiment rather than an
argument.

*Incomplete relaxation.* In the reported bin the direct shift is smaller in
magnitude than the linear prediction at every one of the seven amplitudes --
-1.95 against -3.84, -6.61 against -7.67, -12.64 against -15.34, and so on. A
random measurement error does not have a sign. A direct chain that has not
finished relaxing into the perturbed ensemble does: it still partly reports the
reference ensemble it started from, biasing every measured shift toward zero.
exp06 started its surrogate chains from a reference-equilibrated configuration,
discarded 300 moves, and ran no stationarity check on them at all.

*A single-chain realisation.* Against that, the discrepancy does not have a
consistent sign across bins. `scripts/check_between_chain_scatter.py` measured
the same perturbation at 5.25 A with six independent chains on each side and
found the direct shift *larger* than the prediction, +2.377 +/- 0.144 against
+1.725 +/- 0.101. Under-relaxation cannot make one bin under-shift and another
over-shift. And exp06's smallest-amplitude row is one chain against one chain,
with several bins the perturbation barely touches also displaced by two to four
standard errors -- the signature of one chain's slow modes, not of an estimator.

There is a third possibility neither the manuscript nor the review priced in:
the +1.725 prediction carries a within-chain blocking error from a *single*
reference chain, so its quoted +/-0.101 says nothing about how much the
prediction itself moves between reference realisations. If that scatter is
comparable to the discrepancy, the 3.7 sigma is arithmetic rather than physics.
This experiment measures it directly by predicting from each reference chain
separately.

Distinguishing these requires measurement, not argument:

  1. **Bracket the answer from both sides.** Half the surrogate chains start
     from configurations equilibrated under the *reference*, and half from
     configurations equilibrated under the *surrogate*. Incomplete relaxation
     biases the first toward zero shift and the second toward the full shift, so
     the truth is between them. When the two arms meet, relaxation is complete;
     where they have not met, the gap is the size of the artefact.
  2. **Resolve it in time.** Both arms are measured over a ladder of discarded
     prefixes from the same chains, which costs nothing extra and turns "is it
     equilibrated" into a curve rather than a verdict.
  3. **Use an estimator that can see the other ensemble.** Forward reweighting
     from the reference is variance-limited by a tail the reference barely
     samples. With surrogate chains in hand, BAR and two-state MBAR
     (`atomlab.analysis.fep`) use both directions, and their overlap
     diagnostics say whether the answer deserves belief.
  4. **Change the sampler.** Everything in exp06 came from one HMC
     implementation. A single-particle Metropolis arm shares no propagation
     machinery, so agreement between them is evidence and disagreement localises
     the fault.
  5. **Change the observable.** Two bins, 4.25 A and 5.25 A -- the bin exp06
     reported and the bin the six-chain follow-up used -- since the two gave
     discrepancies of opposite sign and a single experiment should cover both.
  6. **Propagate the prediction's own chain-to-chain uncertainty.** The linear
     prediction is computed separately from each reference chain, so its
     between-chain scatter is measured rather than assumed to be the blocking
     error of one chain.

Pre-registered before the confirmatory numbers were seen
--------------------------------------------------------
*Estimand.* For each bin k, D_k = <A_k>_U - <A_k>_0, the shift in the mean
occupancy of that pair-distance bin between the reference and surrogate
canonical ensembles at fixed N, V, T.

*Primary comparison.* The paired difference between the direct estimate (arm
means, fully relaxed ladder point) and the MBAR estimate, per bin.

*Equivalence bound.* +/- 0.5 pairs. The contrast this measurement has to be able
to support is the aligned-null difference of 10.1 pairs in exp07; a systematic
error of 0.5 pairs is 5% of that, below the point where any conclusion in the
manuscript would change. Chosen from that downstream requirement, not from the
observed scatter.

*Decision rule.* Consistency is declared only if the 95% interval of the paired
difference is contained in the bound, for both bins, in both samplers. If the
interval is wider than the bound the result is "underpowered", not "consistent".
If it excludes zero and exceeds the bound, the estimators genuinely disagree and
the manuscript's prediction claims are withdrawn.

*Chain count.* Eight reference and eight surrogate HMC chains per arm. The exp06
pilot gives a per-chain standard deviation of about 0.8 pairs on this
observable, so eight chains give a standard error near 0.28 and a paired
interval near +/- 0.8 pairs -- wider than the equivalence bound. This is
expected and is why the ladder and the bracketing arms carry the argument: the
question "is the direct estimate drifting toward the prediction as burn-in
grows" is answered by the *shape* of the ladder, which is a within-chain
comparison and far better determined than the absolute level.

*Multiplicity.* Two bins x two samplers x two reference-based estimators = eight
comparisons. Reported with Bonferroni-adjusted intervals alongside the nominal
95%.

*What was decided after seeing data, and what it was.* A smoke-scale run
(100 frames per chain, far below production) was used to debug the plumbing, and
two design decisions were taken after seeing it, so they are recorded here
rather than presented as prior:

  - the primary direct arm is the one started from surrogate-equilibrated
    configurations, on the principle that a chain beginning in the ensemble it
    is measuring carries no relaxation transient; the other arm is reported
    beside it and the difference between them is the artefact;
  - a convergence gate was added: if the two initialisations have not met to
    within the equivalence bound at the deepest discard, no comparison against
    the direct estimate is reported as estimator disagreement, because an
    unrelaxed chain is not a measurement of the ensemble in question.

Both make the test harder to pass, not easier, and neither depends on which way
the answer comes out. The smoke run's numbers are not used anywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atomlab.analysis.fep import (
    bennett_acceptance_ratio,
    mbar_two_state_average,
    overlap_diagnostics,
)
from atomlab.analysis.response import predict_shift, reweight
from atomlab.analysis.statistics import blocking_analysis, integrated_autocorrelation_time
from atomlab.build import fcc, scale_to_density
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.potentials.perturbations import RadialShellPerturbation
from atomlab.sampling import hybrid_monte_carlo, metropolis_nvt
from experiments.common import ExperimentContext, main
from experiments.equilibrate import check_equilibrated, equilibrated_configuration
from experiments.observables_lib import PairBinObservable

DEFAULTS = {
    "system": {"lattice_constant": 5.4, "reps": [3, 3, 3],
               "density": 0.0200, "temperature": 120.0},
    "potential": {"epsilon": 0.0103, "sigma": 3.405, "cutoff": 7.0,
                  "mode": "shifted_force"},
    # exp06's bin edges, so bin 2 is 4.0-4.5 A (centre 4.25) and bin 4 is
    # 5.0-5.5 A (centre 5.25) -- the two the review asks for.
    "observable": {"r_min": 3.0, "r_max": 7.0, "n_bins": 8,
                   "report_bins": [2, 4]},
    # The disputed point: exp06's smallest amplitude, where linear response is
    # least in doubt and the disagreement is therefore most damaging.
    "field": {"r0": 4.2, "width": 0.35, "amplitude": 0.0005},
    # Chain lengths are set by the wall-clock available rather than by what
    # would be ideal, and the consequence is visible in the output: the
    # decision rule reports "underpowered" when the interval is wider than the
    # bound, which is the honest outcome of a short chain rather than a defect
    # to be hidden. Sixteen surrogate chains at 1500 frames is the budget.
    "sampling": {"n_reference": 1500, "n_surrogate": 1500, "n_presoak": 2000,
                 "n_leapfrog": 8, "step_size": 2e-3, "burn_in": 300,
                 "n_melt": 400, "n_anneal": 1000},
    "replication": {"n_reference_chains": 8, "n_surrogate_chains": 8},
    # A Metropolis sweep costs a full energy evaluation per attempted move and
    # is about thirteen times an HMC move here, so this arm is sized as a
    # cross-check rather than a second primary measurement: enough to say
    # whether the HMC answer is a property of the ensemble or of the sampler,
    # not enough to resolve the equivalence bound. Its interval is reported and
    # will read "underpowered" if it is wider than the bound, which is the
    # correct outcome rather than a defect.
    "metropolis": {"n_chains": 4, "n_sweeps": 350, "burn_in": 150,
                   "max_displacement": 0.12},
    # Fractions of each surrogate chain discarded before averaging.
    "relaxation": {"discard_fractions": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]},
    "analysis": {"equivalence_bound_pairs": 0.5, "n_resamples": 400},
}


def build_system(ctx):
    s, p = ctx.config["system"], ctx.config["potential"]
    cfg = scale_to_density(fcc(s["lattice_constant"], "Ar", tuple(s["reps"])), s["density"])
    return cfg, LennardJones(epsilon=p["epsilon"], sigma=p["sigma"],
                             cutoff=p["cutoff"], mode=p["mode"])


def tune_step_size(ctx, cfg, potential, seed, *, n_steps=300):
    """Adapt the HMC step size on a throwaway chain, and return it.

    The bracketing arms below must record their chains from the first move, so
    they run with ``burn_in=0`` -- and step-size adaptation only happens during
    burn-in, because an adapting proposal is not a valid MCMC kernel. Without
    this helper those chains would keep the configured 2 fs while every other
    chain in the experiment adapts to roughly 30 fs, sampling the surrogate
    ensemble more than an order of magnitude more slowly than the reference
    ensemble it is compared against. That is not a small inefficiency: the whole
    question is whether the surrogate chains reach stationarity, and answering
    it with a deliberately crippled kernel would answer a different question.

    The tuning chain starts from the same configuration and is discarded; only
    its step size is kept, so the recorded chain is a fixed kernel started at
    the intended point.
    """
    s = ctx.config["sampling"]
    _, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"], n_samples=1,
        n_leapfrog=s["n_leapfrog"], step_size=s["step_size"],
        burn_in=int(n_steps), seed=seed)
    return float(report.final_step_size)


def hmc(ctx, cfg, potential, n_samples, seed, *, burn_in=None, label="", check=True,
        step_size=None):
    s = ctx.config["sampling"]
    traj, report = hybrid_monte_carlo(
        cfg, potential, ctx.config["system"]["temperature"],
        n_samples=int(n_samples), n_leapfrog=s["n_leapfrog"],
        step_size=s["step_size"] if step_size is None else float(step_size),
        burn_in=s["burn_in"] if burn_in is None else int(burn_in), seed=seed,
        adapt=step_size is None)
    if report.acceptance < 0.2:
        raise RuntimeError(f"{label}: acceptance {report.acceptance:.2f} too low")
    if check:
        check_equilibrated(traj, label=label, check_order=False)
    return traj, report


def chain_values(traj, observable, bins):
    """Per-frame values of the reported bins, shape (n_frames, len(bins))."""
    return observable.evaluate_trajectory(traj)[:, bins]


MIN_FRAMES_FOR_MEAN = 20


def usable_fractions(n_frames, fractions):
    """Which ladder points leave enough frames to average over.

    Under ``--quick`` the chains are short enough that the deepest discards
    would average a handful of frames, so they are dropped rather than reported
    with a meaningless error bar.  Returned explicitly so the caller uses the
    same list everywhere instead of assuming the deepest requested point exists.
    """
    keep = [f for f in fractions
            if n_frames - int(round(f * n_frames)) >= MIN_FRAMES_FOR_MEAN]
    if not keep:
        raise RuntimeError(f"no ladder point leaves {MIN_FRAMES_FOR_MEAN} frames "
                           f"out of {n_frames}")
    return keep


def ladder(values, fractions):
    """Mean of each bin after discarding a leading fraction of the chain."""
    n = values.shape[0]
    return {f: values[int(round(f * n)):].mean(axis=0)
            for f in usable_fractions(n, fractions)}


def run(ctx: ExperimentContext) -> dict:
    cfg, potential = build_system(ctx)
    temperature = ctx.config["system"]["temperature"]
    o, s_cfg = ctx.config["observable"], ctx.config["sampling"]
    rep, an = ctx.config["replication"], ctx.config["analysis"]
    bins = list(o["report_bins"])
    edges = np.linspace(o["r_min"], o["r_max"], o["n_bins"] + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    observable = PairBinObservable(edges, cutoff=ctx.config["potential"]["cutoff"])
    f = ctx.config["field"]
    field = RadialShellPerturbation(r0=f["r0"], width=f["width"],
                                    amplitude=f["amplitude"],
                                    cutoff=ctx.config["potential"]["cutoff"])
    surrogate = potential + field
    fractions = list(ctx.config["relaxation"]["discard_fractions"])
    print(f"  system: {cfg.n_atoms} atoms, T = {temperature} K")
    print(f"  reporting bins {bins} at r = {np.round(centres[bins], 2)} A")
    print(f"  field: Gaussian shell r0={f['r0']} w={f['width']} a={f['amplitude']:.1e} eV")

    with ctx.timed("equilibration"):
        cfg, eq = equilibrated_configuration(
            cfg, potential, temperature, n_melt=int(ctx.get("sampling.n_melt")),
            n_anneal=int(ctx.get("sampling.n_anneal")), n_leapfrog=s_cfg["n_leapfrog"],
            step_size=s_cfg["step_size"], seed=ctx.seed)
    print(f"    {eq}")

    # -- reference ensemble --------------------------------------------------
    with ctx.timed("reference_chains"):
        refs = []
        for i in range(int(rep["n_reference_chains"])):
            traj, _ = hmc(ctx, cfg, potential, ctx.scaled("sampling.n_reference"),
                          seed=ctx.seed + 3000 + 41 * i, label=f"reference {i}")
            a = chain_values(traj, observable, bins)
            du = np.array([field.energy(traj.frame(j)) for j in range(traj.n_frames)])
            refs.append({"index": i, "a": a, "du": du, "traj": traj,
                         "mean": a.mean(axis=0)})
            print(f"      reference {i}: <A> = {np.round(refs[-1]['mean'], 2)}")
    ref_means = np.array([r["mean"] for r in refs])
    ref_grand = ref_means.mean(axis=0)
    ref_sem = ref_means.std(axis=0, ddof=1) / np.sqrt(len(refs))
    print(f"    reference grand mean {np.round(ref_grand, 3)} "
          f"+/- {np.round(ref_sem, 3)} (over {len(refs)} chains)")

    # A configuration properly equilibrated under the *surrogate*, used both to
    # start the second bracketing arm and to supply reverse-direction samples.
    with ctx.timed("surrogate_presoak"):
        presoak, _ = hmc(ctx, cfg, surrogate, ctx.scaled("sampling.n_presoak"),
                         seed=ctx.seed + 7777, label="surrogate presoak")
        soak_frames = [presoak.frame(j) for j in
                       np.linspace(presoak.n_frames // 2, presoak.n_frames - 1,
                                   int(rep["n_surrogate_chains"])).astype(int)]
    print(f"    presoak: {presoak.n_frames} frames, "
          f"<A> = {np.round(chain_values(presoak, observable, bins).mean(axis=0), 2)}")

    # -- surrogate ensemble, bracketed ---------------------------------------
    # burn_in=0 on purpose: the relaxation is the measurement, so nothing may be
    # silently discarded inside the sampler.
    with ctx.timed("surrogate_chains"):
        arms = {}
        starts = {
            "from_reference": [refs[i % len(refs)]["traj"].frame(
                refs[i % len(refs)]["traj"].n_frames - 1 - 7 * i)
                for i in range(int(rep["n_surrogate_chains"]))],
            "from_surrogate": soak_frames,
        }
        tuned = tune_step_size(ctx, cfg, surrogate, seed=ctx.seed + 6006)
        print(f"      step size tuned on a throwaway chain: {tuned * 1e3:.2f} fs "
              f"(configured {s_cfg['step_size'] * 1e3:.2f} fs)")
        deepest = None
        for arm, start_configs in starts.items():
            chains = []
            for i, start in enumerate(start_configs):
                traj, _ = hmc(ctx, start, surrogate, ctx.scaled("sampling.n_surrogate"),
                              seed=ctx.seed + 4000 + 53 * i + (0 if arm == "from_reference" else 1),
                              burn_in=0, label=f"{arm} {i}", check=False,
                              step_size=tuned)
                a = chain_values(traj, observable, bins)
                du = np.array([field.energy(traj.frame(j)) for j in range(traj.n_frames)])
                chains.append({"a": a, "du": du, "ladder": ladder(a, fractions)})
                deepest = max(chains[-1]["ladder"]) if deepest is None \
                    else min(deepest, max(chains[-1]["ladder"]))
            arms[arm] = chains
            final = np.array([c["ladder"][deepest] for c in chains])
            print(f"      {arm}: <A> after discarding {deepest:.0%} = "
                  f"{np.round(final.mean(axis=0), 2)}")
    fractions = [f for f in fractions if f <= deepest]

    # -- the relaxation curve ------------------------------------------------
    relaxation = []
    for frac in fractions:
        row = {"discard_fraction": frac}
        for arm, chains in arms.items():
            vals = np.array([c["ladder"][frac] for c in chains if frac in c["ladder"]])
            if vals.size == 0:
                continue
            row[arm] = (vals.mean(axis=0) - ref_grand).tolist()
            row[f"{arm}_sem"] = (vals.std(axis=0, ddof=1) / np.sqrt(len(vals))).tolist()
        if "from_reference" in row and "from_surrogate" in row:
            row["arm_gap"] = (np.array(row["from_surrogate"])
                              - np.array(row["from_reference"])).tolist()
        relaxation.append(row)

    print("\n    relaxation ladder (shift relative to the reference grand mean):")
    print(f"      {'discard':>8} {'from_reference':>22} {'from_surrogate':>22} {'gap':>16}")
    for row in relaxation:
        if "arm_gap" not in row:
            continue
        print(f"      {row['discard_fraction']:8.0%} "
              f"{str(np.round(row['from_reference'], 2)):>22} "
              f"{str(np.round(row['from_surrogate'], 2)):>22} "
              f"{str(np.round(row['arm_gap'], 2)):>16}")

    # -- estimators ----------------------------------------------------------
    with ctx.timed("estimators"):
        estimates = compare_estimators(
            refs, arms["from_surrogate"], arms["from_reference"], temperature,
            fractions[-1], n_resamples=int(an["n_resamples"]), seed=ctx.seed)

    # -- independent sampler -------------------------------------------------
    with ctx.timed("metropolis_cross_check"):
        metro = metropolis_arm(ctx, cfg, potential, surrogate, observable, bins,
                               field, temperature)

    summary = assemble(estimates, metro, relaxation, ref_grand, ref_sem,
                       centres[bins], float(an["equivalence_bound_pairs"]),
                       fractions[-1])
    summary["tuned_step_size_fs"] = tuned * 1e3
    ctx.save_json("relaxation", relaxation)
    ctx.save_json("estimates", estimates)
    ctx.save_json("metropolis", metro)
    ctx.save_json("summary", summary)
    report_summary(summary)
    return summary


def compare_estimators(refs, surrogate_chains, other_chains, temperature, discard, *,
                       n_resamples, seed):
    """Every estimator of the same shift, on the same data, side by side."""
    a_ref = np.concatenate([r["a"] for r in refs], axis=0)
    du_ref = np.concatenate([r["du"] for r in refs])
    n_keep = [int(round(discard * c["a"].shape[0])) for c in surrogate_chains]
    a_sur = np.concatenate([c["a"][k:] for c, k in zip(surrogate_chains, n_keep)], axis=0)
    du_sur = np.concatenate([c["du"][k:] for c, k in zip(surrogate_chains, n_keep)])

    out = {"n_reference_samples": int(a_ref.shape[0]),
           "n_surrogate_samples": int(a_sur.shape[0])}

    linear = predict_shift(a_ref, du_ref, temperature, n_resamples=n_resamples, seed=seed)
    # Per-chain predictions as well as the pooled one. The manuscript quoted a
    # within-chain blocking error for the prediction, which says how well one
    # chain determines its own covariance and nothing about how much the
    # covariance moves between chains -- the quantity a difference against an
    # independently sampled ensemble actually needs.
    per_chain = np.array([np.asarray(predict_shift(r["a"], r["du"], temperature,
                                                   n_resamples=max(100, n_resamples // 4),
                                                   seed=seed + 10 + r["index"]).value)
                          for r in refs])
    out["linear"] = {"value": np.asarray(linear.value).tolist(),
                     "error": np.asarray(linear.error).tolist(),
                     "second_order": np.asarray(linear.second_order.value).tolist(),
                     "second_order_ratio": np.asarray(linear.second_order_ratio).tolist(),
                     "per_chain": per_chain.tolist(),
                     "between_chain_scatter": per_chain.std(axis=0, ddof=1).tolist(),
                     "between_chain_sem": (per_chain.std(axis=0, ddof=1)
                                           / np.sqrt(len(per_chain))).tolist()}

    fwd = reweight(a_ref, du_ref, temperature, n_resamples=n_resamples, seed=seed)
    out["forward_fep"] = {"value": np.asarray(fwd.shift.value).tolist(),
                          "error": np.asarray(fwd.shift.error).tolist(),
                          "ess_fraction": fwd.ess_fraction,
                          "max_weight_fraction": fwd.max_weight_fraction}

    # Reverse FEP: reweight the surrogate ensemble back onto the reference, with
    # the sign of the energy difference flipped because the roles swap. Its
    # `shift` is then <A>_0 - <A>_U, so the sign is flipped back to keep every
    # row of the table pointing the same way.
    rev = reweight(a_sur, -du_sur, temperature, n_resamples=n_resamples, seed=seed + 1)
    out["reverse_fep"] = {"value": (-np.asarray(rev.shift.value)).tolist(),
                          "error": np.asarray(rev.shift.error).tolist(),
                          "ess_fraction": rev.ess_fraction,
                          "max_weight_fraction": rev.max_weight_fraction}

    bar = bennett_acceptance_ratio(du_ref, du_sur, temperature)
    out["bar_delta_f"] = {"value": bar.value, "error": bar.error}

    mbar = mbar_two_state_average(a_ref, du_ref, a_sur, du_sur, temperature,
                                  n_resamples=n_resamples, seed=seed + 2)
    out["mbar"] = {"value": np.asarray(mbar["shift"].value).tolist(),
                   "error": np.asarray(mbar["shift"].error).tolist(),
                   "kish_reference": mbar["kish_reference"],
                   "kish_surrogate": mbar["kish_surrogate"],
                   "max_weight_reference": mbar["max_weight_reference"],
                   "max_weight_surrogate": mbar["max_weight_surrogate"],
                   "delta_f": mbar["delta_f"]}

    # Direct: chain means are the experimental unit, so the error is their
    # scatter, not a within-chain blocking estimate. The primary arm is the one
    # started from configurations already equilibrated under the surrogate --
    # not because it gives the answer one wants, but because a chain that begins
    # in the ensemble it is measuring has no relaxation transient to contaminate
    # it. The other arm is reported beside it, and the difference between them
    # *is* the relaxation artefact rather than an error bar on it.
    ref_chain = np.array([r["a"].mean(axis=0) for r in refs])
    for arm_name, chains in (("direct", surrogate_chains),
                             ("direct_from_reference", other_chains)):
        keep = [int(round(discard * c["a"].shape[0])) for c in chains]
        sur_chain = np.array([c["a"][k:].mean(axis=0) for c, k in zip(chains, keep)])
        value = sur_chain.mean(axis=0) - ref_chain.mean(axis=0)
        err = np.sqrt(sur_chain.var(axis=0, ddof=1) / len(sur_chain)
                      + ref_chain.var(axis=0, ddof=1) / len(ref_chain))
        out[arm_name] = {"value": value.tolist(), "error": err.tolist(),
                         "n_reference_chains": len(ref_chain),
                         "n_surrogate_chains": len(sur_chain)}

    out["overlap"] = overlap_diagnostics(du_ref, du_sur, temperature)
    out["tau_reference"] = float(integrated_autocorrelation_time(a_ref))
    out["tau_surrogate"] = float(integrated_autocorrelation_time(a_sur))
    out["blocking_reference"] = np.asarray(blocking_analysis(a_ref).error).tolist()
    return out


def metropolis_arm(ctx, cfg, potential, surrogate, observable, bins, field, temperature):
    """The same shift from a sampler that shares no propagation machinery.

    Single-particle Metropolis costs a full energy evaluation per attempted move
    and is roughly thirteen times more expensive per sweep than an HMC move
    here, so this arm is deliberately smaller.  It is a check on whether the
    HMC answer is a property of the ensemble or of the sampler, and for that a
    wider error bar is acceptable; it is not a second primary measurement.
    """
    m = ctx.config["metropolis"]
    out = {"n_chains": int(m["n_chains"]), "n_sweeps": int(ctx.scaled("metropolis.n_sweeps"))}
    for label, pot in (("reference", potential), ("surrogate", surrogate)):
        means, accs = [], []
        for i in range(int(m["n_chains"])):
            traj, report = metropolis_nvt(
                cfg, pot, temperature, n_sweeps=int(ctx.scaled("metropolis.n_sweeps")),
                burn_in=int(m["burn_in"]), max_displacement=m["max_displacement"],
                seed=ctx.seed + 8000 + 61 * i + (0 if label == "reference" else 1))
            means.append(observable.evaluate_trajectory(traj)[:, bins].mean(axis=0))
            accs.append(report.acceptance)
        means = np.array(means)
        out[label] = {"chain_means": means.tolist(),
                      "mean": means.mean(axis=0).tolist(),
                      "sem": (means.std(axis=0, ddof=1) / np.sqrt(len(means))).tolist(),
                      "acceptance": float(np.mean(accs))}
        print(f"      metropolis {label}: {np.round(means.mean(axis=0), 2)} "
              f"+/- {np.round(means.std(axis=0, ddof=1) / np.sqrt(len(means)), 2)} "
              f"(acceptance {np.mean(accs):.2f})")
    shift = np.array(out["surrogate"]["mean"]) - np.array(out["reference"]["mean"])
    err = np.sqrt(np.array(out["surrogate"]["sem"])**2 + np.array(out["reference"]["sem"])**2)
    out["shift"] = shift.tolist()
    out["shift_error"] = err.tolist()
    return out


def assemble(est, metro, relaxation, ref_grand, ref_sem, radii, bound, discard):
    """Apply the pre-registered decision rule, per bin and per sampler."""
    # Two reference-based estimators are compared against direct sampling. MBAR
    # is the primary one because it uses both ensembles; the linear prediction
    # is included because it is the estimator the manuscript's claims rest on,
    # and its uncertainty is taken as the scatter of the per-chain predictions,
    # not the blocking error of the pooled samples.
    targets = {
        "mbar": (np.array(est["mbar"]["value"]), np.array(est["mbar"]["error"])),
        "linear": (np.array(est["linear"]["value"]),
                   np.array(est["linear"]["between_chain_sem"])),
    }
    comparisons = []
    for sampler, value, error in (
            ("hmc", np.array(est["direct"]["value"]), np.array(est["direct"]["error"])),
            ("metropolis", np.array(metro["shift"]), np.array(metro["shift_error"]))):
        for name, (target, target_err) in targets.items():
            diff = value - target
            # The estimators share the reference samples in the HMC case, so
            # their errors are positively correlated and adding in quadrature is
            # conservative for the difference. Stated rather than silently
            # assumed.
            err = np.sqrt(error**2 + target_err**2)
            for j, r in enumerate(radii):
                half95, half9875 = 1.96 * err[j], 2.50 * err[j]
                comparisons.append({
                    "sampler": sampler, "against": name, "bin_radius": float(r),
                    "direct": float(value[j]), "direct_error": float(error[j]),
                    "reference_based": float(target[j]),
                    "reference_based_error": float(target_err[j]),
                    "difference": float(diff[j]), "difference_error": float(err[j]),
                    "ci95": [float(diff[j] - half95), float(diff[j] + half95)],
                    "ci_bonferroni": [float(diff[j] - half9875), float(diff[j] + half9875)],
                    "within_bound": bool(abs(diff[j]) + half95 <= bound),
                    "excludes_zero": bool(abs(diff[j]) > half95),
                    "underpowered": bool(half95 > bound),
                })
    first, last = relaxation[0], relaxation[-1]
    # Whether the two initialisations have met. This gates everything else: if
    # they have not, the direct estimate still carries a relaxation transient
    # and no comparison against it means anything, whatever its error bar says.
    gap = np.abs(np.array(last["arm_gap"])) if last.get("arm_gap") else None
    gap_error = (np.sqrt(np.array(last["from_surrogate_sem"])**2
                         + np.array(last["from_reference_sem"])**2)
                 if last.get("arm_gap") else None)
    arms_converged = bool(gap is not None and np.all(gap - 1.96 * gap_error <= bound))
    return {
        "equivalence_bound_pairs": bound,
        "arms_converged": arms_converged,
        "final_gap": gap.tolist() if gap is not None else None,
        "final_gap_error": gap_error.tolist() if gap_error is not None else None,
        "discard_fraction_used": discard,
        "bin_radii": np.asarray(radii).tolist(),
        "reference_grand_mean": np.asarray(ref_grand).tolist(),
        "reference_sem": np.asarray(ref_sem).tolist(),
        "estimators": {
            "linear": est["linear"]["value"],
            "linear_error": est["linear"]["error"],
            "linear_between_chain_sem": est["linear"]["between_chain_sem"],
            "forward_fep": est["forward_fep"]["value"],
            "reverse_fep": est["reverse_fep"]["value"],
            "mbar": est["mbar"]["value"],
            "mbar_error": est["mbar"]["error"],
            "direct_hmc": est["direct"]["value"],
            "direct_hmc_error": est["direct"]["error"],
            "direct_metropolis": metro["shift"],
            "direct_metropolis_error": metro["shift_error"],
        },
        "overlap": est["overlap"],
        "mbar_kish": [est["mbar"]["kish_reference"], est["mbar"]["kish_surrogate"]],
        "relaxation_gap_start": first.get("arm_gap"),
        "relaxation_gap_end": last.get("arm_gap"),
        "comparisons": comparisons,
        "all_within_bound": all(c["within_bound"] for c in comparisons),
        "any_underpowered": any(c["underpowered"] for c in comparisons),
        "any_disagreement": any(c["excludes_zero"] and not c["within_bound"]
                                for c in comparisons),
    }


def report_summary(s):
    print("\n  --- P0-1: end-to-end consistency ---")
    e = s["estimators"]
    print(f"  {'estimator':<24}" + "".join(f"{r:>12.2f} A" for r in s["bin_radii"]))
    for label, key, ekey in (("linear response", "linear", "linear_error"),
                             ("forward FEP", "forward_fep", None),
                             ("reverse FEP", "reverse_fep", None),
                             ("MBAR (both directions)", "mbar", "mbar_error"),
                             ("direct, HMC", "direct_hmc", "direct_hmc_error"),
                             ("direct, Metropolis", "direct_metropolis",
                              "direct_metropolis_error")):
        vals = np.atleast_1d(e[key])
        errs = np.atleast_1d(e[ekey]) if ekey else None
        cells = "".join(
            f"{v:>9.2f}" + (f" +/-{errs[i]:>4.2f}" if errs is not None else "     ")
            for i, v in enumerate(vals))
        print(f"  {label:<24}{cells}")

    print(f"  {'linear, between-chain SEM':<24}"
          + "".join(f"{v:>14.2f}" for v in np.atleast_1d(e["linear_between_chain_sem"])))
    print("    (the manuscript quoted the within-chain blocking error instead; "
          "compare the two)")

    o = s["overlap"]
    print(f"\n  overlap: work_overlap {o['work_overlap']:.2f}, "
          f"Kish forward {o['kish_forward']:.3f}, reverse {o['kish_reverse']:.3f}, "
          f"max weight {max(o['max_weight_forward'], o['max_weight_reverse']):.3f}")
    if s["relaxation_gap_start"] is not None:
        print(f"  bracketing gap (from_surrogate - from_reference): "
              f"{np.round(s['relaxation_gap_start'], 2)} at no discard -> "
              f"{np.round(s['relaxation_gap_end'], 2)} at "
              f"{s['discard_fraction_used']:.0%}")

    print(f"\n  pre-registered bound +/-{s['equivalence_bound_pairs']} pairs; "
          f"direct minus reference-based:")
    for c in s["comparisons"]:
        verdict = ("EQUIVALENT" if c["within_bound"] else
                   "DISAGREE" if c["excludes_zero"] else
                   "underpowered" if c["underpowered"] else "inconclusive")
        print(f"    {c['sampler']:<11} vs {c['against']:<7} {c['bin_radius']:.2f} A  "
              f"{c['difference']:+7.3f}  95% [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"{verdict}")
    if not s["arms_converged"]:
        print(f"\n  GATE FAILED: the two initialisations are still "
              f"{np.round(s['final_gap'], 2)} pairs apart (error "
              f"{np.round(s['final_gap_error'], 2)}) after the deepest discard, "
              f"against a bound of {s['equivalence_bound_pairs']}.")
        print("  The surrogate chains have not relaxed, so the direct estimate is "
              "not a measurement of the surrogate ensemble and none of the "
              "comparisons above can be read as estimator disagreement. Longer "
              "chains are needed; the size of the gap is the size of the artefact.")
        return
    if s["any_disagreement"]:
        print("  VERDICT: at least one comparison is outside the bound and excludes "
              "zero -- the estimators genuinely disagree.")
    elif s["all_within_bound"]:
        print("  VERDICT: every comparison is inside the pre-registered bound.")
    else:
        print("  VERDICT: no comparison establishes disagreement, but the intervals "
              "are wider than the bound -- underpowered, not consistent.")


if __name__ == "__main__":
    main(run, default_config=DEFAULTS,
         description="Reconcile reference-based estimators with direct sampling")
