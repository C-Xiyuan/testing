# Response to the referee

**Manuscript:** "Error fields, not error norms: how the error in a fitted
interatomic potential reaches a physical observable"

We are grateful for a report that did the one thing we most wanted a referee to
do: re-analyse the deposit rather than the prose. Four of the referee's findings
are, in our judgement, correct and consequential, and we have changed the
paper's claims — not merely its wording — in response. Two more we accept as
disclosure obligations we had failed to discharge. Where we disagree we say so
and give the reason.

Section numbers below are those of the revised manuscript. Almost every quantity
added in revision is recomputed from records the referee already had, by
`scripts/compute_revision_statistics.py`, which writes
`results/revision_statistics.json`. The one exception is the eight-chain
calibration replication described under R1, which is a new measurement, is
deposited in full at `results/exp09_calibration_replication`, and is flagged as
new wherever it is used.

---

## Part 1 — Points we accept, and what changed

### R1. The shared reference baseline (referee §2). Accepted in full.

The referee is right, and this is the most useful thing in the report. We
confirm the arithmetic independently: in `exp07` all eight signed residuals
`(measured − predicted)/SE` are negative, with mean **−0.861σ**, scatter about
that mean **0.571σ**, rms 1.013σ, and a one-sample *t* of **−4.27** on 7 d.o.f.
The mechanism is in the code exactly where the referee says it is
(`experiments/exp07_designed_counterexamples/run.py`, where `t_reference` is the
same evaluation half for every field; likewise `t_ref_mean` in
`experiments/exp03_model_zoo/run.py`).

Two pieces of evidence the referee did not have make the diagnosis firmer.

First, the 40-configuration arm — which we had not reported, see R5 —
reproduces the pattern almost exactly on a different set of surrogates: offset
**−0.851σ**, scatter **0.567σ**, again eight negatives out of eight. Two
independent arms agreeing to the third decimal on both components is what a
shared baseline predicts and what a genuine mis-calibration of the estimator
would not.

Second, and decisively, we have since run the replication the referee's argument
implies: eight designed fields re-measured against **eight** independently
equilibrated reference chains with four direct chains each, 64 measurements on
an 8 × 8 grid (`results/exp09_calibration_replication`). The per-chain mean
residual runs from **−2.47σ to +1.64σ**, a spread of **1.30σ** against the
**0.89σ** the quoted errors allow; the grand mean is **−0.34 ± 0.46** over eight
chains, consistent with zero; and 36 of the 64 residuals are negative, so the
eight-out-of-eight sign pattern is not a general property of the estimator but a
property of one reference chain. **The offset is a property of the reference
chain, not a bias in the estimator** — which is the referee's diagnosis, now
measured rather than inferred. Two honest qualifications travel with it: the arm
ran in quick mode (1500 reference and 1500 direct frames per chain against 4000
and 3000 in Table I), so its own residuals are noisier (rms 1.71σ); and a
per-field effect survives the two-way decomposition, observed spread 0.78σ
against 0.45σ allowed, so the reference chain accounts for most of the excess
but not all of it. Both are stated in §4.2 and §7.

Changes: §3.7 now states the shared-baseline construction as a property of the
measurements. §4.2 no longer offers 1.01σ as a calibration; it decomposes it,
gives the sign pattern, the offset, the scatter, the *t* statistic and the
1/128, and states the two defensible claims (scatter 0.57σ, *n* = 8, better than
the quoted uncertainties; a common offset of about half a pair), then reports
the eight-chain replication that assigns the offset to the reference chain.
§4.5 gives the same decomposition for the ten-model arm (7/10 negative, mean
−0.49σ, *t* = −1.5, not significant alone). The abstract, contribution 2, the
conclusion and the caption of Fig. 2 all now carry the decomposition rather than
the bare figure. We report the offset rather than correct for it: the
replication settles its provenance, not its size.

### R2. The Kish-ESS paragraph in §5.3 is a non-sequitur. Accepted in full.

The referee's argument is decisive and we should have seen it: the effective
sample size diagnoses the reweighting estimator alone, and the first-order
estimator — which carries no importance weights — deviates from direct sampling
by the same amount. An explanation confined to the exponential average cannot
account for a deviation shared by an estimator that contains no exponential
average.

Changes: §5.3 is retitled "with **four** explanations excluded" and the ESS is
now the fourth exclusion, with the referee's reasoning given. The remaining
untested candidate is now the one the referee proposes and R1 makes concrete:
both same-sample estimators run on one reference chain and would move together
with an unlucky realisation of it, while the direct estimate averages six
independent chain pairs and would not. We give the size comparison honestly —
between-chain scatter 0.312 pairs against a discrepancy of 0.652, "the right
order, but only that" — and name the two tests that would settle it, neither of
which was run. Table VI's caption is updated to match.

### R3. The second-order "warning light" (referee §6). Accepted in full; the claim is withdrawn.

We withdraw it. The referee is right on both counts, and the second is the one
that matters: the second-order term does not merely fail to be resolved from
zero, it has the **opposite sign** to the residual it would have to explain, so
adding it makes the agreement worse (residual after first order −0.107 pairs at
the upper level, −0.439 after first plus second). The eight-chain replication of
R1 says the same thing at *n* = 64 rather than *n* = 2: adding the second-order
term raises the root-mean-square residual from **1.71σ to 2.12σ**. Both figures
are now in §4.2.

We also accept both of the referee's other two strikes against the diagnostic:

* §5.4 no longer cites the amplitude sweep in its support. Of the two amplitudes
  the diagnostic certified, one is the unresolved discrepancy of §5.3; the
  deposited summary records this under `flagged_cases_all_agree: false`, and the
  manuscript now says so and quotes the key.
* §4.5 no longer says the diagnostic "fired on exactly the two models where the
  linear prediction is least reliable". The largest residual is `linear_l2` at
  −1.60σ, which the diagnostic flags trustworthy. The corrected statement is
  that it fired on two of the three least reliable predictions, not on exactly
  the two least reliable, and this appears in the text and in a footnote to
  Table IV.

§2.3(iii) is rewritten to present the diagnostic as a hypothesis rather than a
result, with a forward reference to the mixed evidence, and the conclusion no
longer asserts that the estimator's own second-order term gives advance warning.

### R4. The large-δU arm lacks the stationarity flag (referee §5b). Accepted.

Confirmed: every record in `results/exp03_model_zoo_n400/` lacks
`direct_equilibrated` and `drift_note`, while the sibling 40-configuration arm
used exactly that check to discard two `egnn_c8` runs whose apparent shifts were
−95.5 and −112.8 pairs. We cannot re-run the arm within this revision, so we
have done what the referee's second option requires without deleting the
evidence: §5.4 now discloses the missing flag in bold, sets it beside the
sibling arm's exclusion, gives the `linear_l3`/`linear_l4` comparison the referee
identified, and states the conclusion at the strength the evidence carries —
linear response must break down at large δU and this arm is consistent with
that, but it is not a clean demonstration, and **30.4σ is an upper bound on the
disagreement attributable to the truncation** rather than a measurement of it.
The same concession appears in §7 as limitation (ii). Table V's caption carries
the missing-field disclosure and the cross-arm comparison. Nothing else in the
paper depends on this arm.

### R5. The undisclosed 40-configuration arm and its exclusion (referee §5a). Accepted.

This was an omission and we are glad to have it pointed out. §4.5 now reports
the arm as a third validation (1.00σ over the eight equilibrated models,
decomposing as in R1), reports the exclusion of the two `egnn_c8` runs and their
apparent shifts of −95.5 ± 1.7 and −112.8 ± 2.8 pairs, and states in terms that
excluding models from an arm on a diagnostic is a disclosure obligation. The
introduction's contribution 6 now names the occasion on which the equilibration
guard fired, and Table IV's caption records the arm and the exclusion.

### R6. Two different objects called the "null field" (referee §7). Accepted.

Confirmed: the `exp07` construction targets the single 3.4–3.9 Å bin, while
`exp05`'s `null_f4e-03` is built against the full eight-bin observable, and its
difference curve `[0.27, −1.13, 1.29, 0.16, −0.29, −1.36, −0.28, 1.39]` has norm
2.645 pairs against a noise level of 2.428 — the smallest in the zoo and not
resolved from zero.

We now name them apart throughout: **scalar-null** and **curve-null**, defined
in §3.5 and used consistently in §4.1, §4.3, §4.4, Table I's caption and Fig. 3's
caption. §4.3 is rewritten around the referee's better claim: orthogonality is
specific to the *set* of observables a field was built against, enlarging the
set works — nulling the vector silences the curve — and it is therefore **not**
true that no error field can be harmless in general. That is a cleaner statement
and it is the one our own recommendation (a vector of response scores) actually
needs.

### R7. The width exponent needs an interval; the sweep saturates (referee §2, smaller points). Accepted.

We agree that a paper insisting on `ρ = value [low, high], n` cannot quote a bare
fitted exponent as a refutation. Propagating the deposited blocking errors on
the six direct-sampling points through 20,000 Monte Carlo draws (seed 20240517)
gives **0.56 [0.09, 1.34]**, with 1.4% of resamples reaching 1.5. The interval
is quoted in the abstract, §2.6, §5.1, the conclusion and Fig. 6's caption. §5.1
now leads with the interval, states that the exponent is about two standard
errors from zero, states that the sweep is **not monotone** (2.1, 5.3, 4.9, 7.7,
9.9, 8.4 pairs, turning over at the largest width — the saturation §2.6
predicted analytically), and reports that dropping the saturating point moves
the exponent to 0.72. §4.6 also notes the non-monotonicity.

One qualification. The referee suggests leading with the first-order exponent as
"far better determined". We report 0.45 and its smaller point-to-point scatter,
but the deposit carries **no per-point uncertainty** for the first-order series,
so we quote no interval for it and claim none. Quoting "better determined"
without an interval would be the error the referee is asking us to fix.

### R8. The 30× band spread, the ties, and the robustness analysis (referee §1). Accepted.

Confirmed in every particular. The denominator of the 30× ratio is the
curve-null field, whose noise-subtracted error of 1.05 pairs is a near-total
cancellation of a raw 2.645 against a noise level of 2.428 and is not resolved
from zero; the raw-series spread is 12.05×; removing the two designed fields
leaves 6.51×. Six of the fifteen band members share a force RMSE of exactly
4.0 × 10⁻³ eV/Å by construction.

Changes: §4.4 has a new paragraph, "Three features of that band restrict what
those numbers mean, and all cut against us", giving all three. The referee's
robustness analysis is now in the paper: band minus designed fields, ρ(force
RMSE) = 0.33 [−0.42, 0.84] against ρ(response prediction) = 0.91 [0.64, 1.00],
*n* = 13; shell family alone, 0.28 [−0.63, 1.00] against 0.98 [0.86, 1.00],
*n* = 9. An earlier draft of this response said we would quote these as point
estimates, on the grounds that a percentile bootstrap at *n* = 13 and *n* = 9 is
uninformative; we withdraw that, since the honest move is to show how wide the
intervals are rather than to assert it. They are computed by the same estimator
as every other interval in the paper (`scripts/compute_revision_statistics.py`,
2000 resamples, seed 20240517), and they are wide — but in both subsets the
force-RMSE interval covers zero and the response-prediction interval does not,
which is the comparison the paragraph rests on. We label them robustness
checks, not results. The section
now closes on the 6.5× figure rather than the 30×. The abstract, the conclusion,
Table III's caption and Fig. 4's caption all carry the qualification. §7 lists
the ties and the designed members among the band's limitations.

We keep the 30× as the headline of the noise-subtracted convention, because it
is the convention used for every correlation in the paper and switching
conventions for one ratio would be worse. But it is never now stated without its
alternatives.

### R9. Present the headline as a difference (referee, recommended 11). Accepted.

A good suggestion and we have taken it. §4.1 has a new paragraph: at the upper
level the aligned field sits **10.11 pairs** above the scalar-null field, and
because both are measured against the same reference mean, that term cancels
identically in the difference. Combining the two tabulated errors in quadrature
double-counts the shared term, so the quoted **11.4σ** is a conservative
lower bound. The conclusion now leads with this rather than with two separately
quoted shifts.

### R10. Prevalence versus possibility (referee §4, §9, recommended 13). Accepted.

We agree that this is the paper's deepest weakness and that our language slid.
§6 has a new paragraph, "Possibility, not prevalence", which states that
everything demonstrated here works on error fields we built, that they establish
what force RMSE *can* miss and say nothing about how often fitting produces
error in that direction, that the one arm that could have spoken to prevalence
returns a null for a reason internal to our design (a two-parameter pair
potential is fitted to ~4 × 10⁻⁴ eV/Å by everything we tried), and that the
seed pair is *n* = 2. §4.1 now ends "This is an existence proof, not a frequency
claim"; §7 closes on the same point; the conclusion says the demonstration is of
possibility rather than prevalence. §3.5 states that the null construction uses
the reference-ensemble covariance a practitioner does not have, so these fields
are an instrument, not a recipe.

### R11. `n_resolvable = 26/34` (recommended 12). Accepted; stated in §4.4 where ρ = 0.83 is first reported.

### R12. Dirty working trees (referee §8). Accepted.

All six manifests carry `git_dirty: true`, from five distinct commits
(`8a2802c`, `400b2cb`, `2dd024a`, `964dbe8`, `590a1a1` twice). §3.8 now says so
in bold and draws the consequence — no number here can be regenerated from a
named revision — and names the fix (a clean re-run from a tagged commit) as not
done. §7 lists it as limitation (iv). We have not re-run, because a clean re-run
would change every number in the paper and could not be checked against the
present deposit within this revision.

### R13. The misnamed directory (recommended 10) and the outer bins (recommended 14). Accepted.

`results/exp03_model_zoo_n400/` holds the budget-20 arm; we have not renamed it,
because the deposited scripts and manifests refer to it by that name and a
rename would break the traceability the paper depends on. Instead §3.8 and
Table V's caption state plainly that the directory is misnamed. On the outer
bins: averaged over the 34-member zoo, the two bins at the 7.0 Å cutoff carry
28% of the squared norm of the measured difference curve against 29% for the
single bin at 4.25 Å, so the norm is not a cutoff artefact. That is now in
Table III's caption.

### R14. Points raised by the compliance review, accepted alongside the referee's

* The four Behler–Parrinello records carry `fit_converged: false`. Disclosed in
  §3.6 and in a footnote to Table IV.
* The 30-construction-frame result (1.2–1.6 pairs) exists only in narrative
  notes, not in the deposit. Under the study's own no-invented-numbers rule the
  number is now **removed**: §4.2 and Table II record that the construction
  failed to generalise and that the artefact was not retained, and quote no
  figure.
* "Wrong in energy by 30–40%" is not supported by its own endpoints
  (−0.036 to −0.043 against −0.0311 is 16% to 38%). §3.3 now says 16 to 38% and
  notes the departure from our notes.
* The dilute-gas disagreement is 4.7σ rms / 6.5σ worst in the deposited output,
  not the 5.4σ our notes quote. §5.5 now states the departure explicitly.
* `0.652 pairs` is 38% of the first-order value; the 35% belongs to the
  reweighted difference of 0.620. §5.3 now gives both differences and both
  percentages.
* The `egnn_c8` force-RMSE difference is 35%, not 36%.
* Uncertainties are now rounded to two significant figures throughout, with
  central values to the same decimal place; the exception previously claimed for
  pair-count shifts is withdrawn.
* Fig. 3's caption said the target bin is "left alone" at −0.220 ± 0.640 pairs,
  which cannot be read off the figure because 3.4–3.9 Å is deliberately offset
  from the plotted eight-bin grid. The caption now says so in bold, and §3.4 and
  §4.3 make the same point.
* Fig. 4 plots a sampled quantity without error bars; the caption now says so
  and points to where the per-member standard errors live.
* Fig. 5 is drawn on the raw convention while the text uses the noise-subtracted
  one; its caption now discloses this and gives both.
* Fig. 6(b) does not plot the first-order series or either fitted line; the
  caption now describes what the panel contains.
* Fig. 1(b)'s printed panel title rounds 4.66 to "5×"; flagged in the caption.
* Table V's σ column is a prediction residual, not |measured|/SE; both captions
  now say which is which.
* The second author of Ref. Wu2024 is **Wenjiang** Zhou, not Yuwen Zhou;
  corrected in `refs.bib` against arXiv:2401.11427 and the published record.
* `main.md` carried an orphan numeric citation and a false statement about its
  reference ordering. It is now generated from `main.tex` by `paper/tex2md.py`,
  so the two cannot drift apart, and its citation convention is stated at the
  top.

---

## Part 2 — Points where we have not made the change asked for, with reasons

### D1. "Either re-run the large-δU arm with the guard, or withdraw §5.4 and Table V."

We have done neither, and we think the middle course is the right one. Deleting
the arm would remove from the paper the only place where the boundary of the
estimator's validity is visible at all, and would remove it on the grounds that
we cannot rule out an artefact — which is an argument for stating that we cannot
rule it out, not for hiding the observation. The revision therefore keeps the
table, discloses the missing flag as prominently as the referee's report does,
supplies the internal evidence that points at unrelaxed chains, and downgrades
the claim to an upper bound. A reader can now reach the referee's conclusion
from the paper's own text. If the editor prefers outright withdrawal we will
comply, but we believe the disclosed version is more informative than either
alternative.

### D2. "Lead the refutation with the first-order exponent (0.45, far better determined)."

We report both exponents and lead with the interval on the direct one, but we do
not claim the first-order exponent is better determined, because the deposit
carries no per-point uncertainty for that series. Asserting "far better
determined" would be an unbacked statistical claim of exactly the kind the rest
of the report asks us to remove. This is a disagreement about which of us is
being more conservative, not about the data.

### D3. "Report 6.5× rather than 30×."

We report both, in every place the ratio appears, and we let §4.4 and the
conclusion end on 6.5×. We do not replace 30× outright, because it is the
correct value on the noise-subtracted series — the convention used for every
correlation in the paper — and quietly switching conventions for one quantity
while keeping it for the others would introduce an inconsistency of exactly the
kind the compliance review flagged elsewhere. The qualification now travels with
the number.

### D4. "Rename `results/exp03_model_zoo_n400/`."

Not done, deliberately. The deposited manifests, scripts and the run logs all
refer to that path; renaming it after the fact would make the manuscript's
traceability comments false and would be an undocumented edit to the archived
record. We disclose the misnaming in the methods and in the table caption
instead. We would rather ship a deposit with an ugly name than one whose
contents no longer match what produced them.

### D5. The suggested Stillinger–Weber experiment (referee §10).

We agree with the referee's reasoning entirely — it is the right experiment and
for the right three reasons — and we have said so in §6 in the referee's own
terms: this paper motivates that experiment rather than answering the question
it would settle. We have not run it. It is a new campaign, not a revision, and
running it would change what the paper is about. The same applies to the
*N* = 256 finite-size repeat and to the *n* ≥ 8 seed sweep, both of which remain
named in §7 as outstanding.

### D6. "The rms residual over n = 8 has an intrinsic uncertainty of about ±25%; 'exactly the level the error bars claim' is over-precise."

Accepted in substance — the sentence is gone — but we note for the record that
the referee's own decomposition makes the point more sharply than the ±25%
argument does, and it is the decomposition we have reported.

### D7. One factual correction to the report.

The referee writes that the fitted-model zoo's residual "currently establishes
*nothing at all*" until re-derived. We think that is slightly too strong. After
the decomposition, the ten-model arm establishes that the first-order prediction
tracks fitted models with a scatter of 1.00σ about a common offset that is not
individually significant (*t* = −1.5), and the eight-model 40-configuration arm
establishes the same with a scatter of 0.57σ. What is destroyed is the reading
of 1.06σ as a calibration; what survives is that the response formula applies to
objects produced by fitting. That is the claim §4.5 now makes and no more.

---

## Part 3 — An honest summary of where the manuscript stands

**What it establishes.**

1. A derivation, standard and claimed as such, that the error a potential makes
   in a static observable is a covariance between the observable and the error
   field, while force RMSE is a norm of that field's gradient.
2. A constructive existence proof that the two come apart: four error fields
   agreeing in out-of-sample force RMSE to 0.52%, whose target-observable shifts
   separate by 10.11 pairs, at least 11.4σ on a conservative error in which the
   shared reference term cancels. The ordering is set by ρ(*A*, δ*U*),
   monotonically, at both force levels. This survives the obvious circularity
   objections: the fields are built on one half of the reference trajectory and
   evaluated on the other, the random controls were not tuned and land where the
   formula says they should, and the null-space suppression holds out of sample
   at 360× from 2000 construction frames.
3. That the first-order prediction, computed from reference-ensemble samples at
   one energy evaluation per stored frame, tracks direct measurement across
   three independent arms with a scatter of 0.57σ, 1.00σ and 0.57σ about a
   common per-arm offset — and that the offset itself is a property of the
   reference chain, not of the estimator, on an eight-chain replication whose
   grand mean is consistent with zero. This is the paper's practical claim and
   it is intact, though now stated more carefully than before.
4. That force error is a coarse filter and not a selector: strong across a
   361-fold range of force RMSE on a 34-member designed zoo, not resolved from
   zero inside a factor-2.6 band where the response prediction is resolved and
   high. This refutes the study's own pre-registered P1.
5. Three refuted predictions of the study's own — P1, the smoothness ratio, the
   *w*^{3/2} scaling — reported as results.
6. A methodological result of independent interest: an unrelaxed reference
   ensemble that no sampler diagnostic caught, and a stationarity guard that
   later excluded two fitted models whose apparent shifts were twenty times any
   real effect in the study.

**What it does not establish.**

1. **Prevalence.** Every positive result rests on error fields we designed. The
   paper shows that the pathology can exist; it does not show how often fitting
   produces it. The one arm that could have spoken to prevalence returned a
   null, for a reason internal to the design: the reference is easy enough that
   every architecture fits it to the noise floor.
2. **The refutation of P1 on the registered population.** P1's falsification
   criterion named an *architecture-diverse* zoo. The zoo on which P1 is refuted
   is a zoo of designed error fields. The substitution is now stated; it is a
   real weakening.
3. **Any ranking claim from the fitted models.** Every interval from that arm
   spans zero, and the paper says so.
4. **The breakdown of linear response at large δU.** The 30.4σ arm is consistent
   with the expected breakdown but its stationarity check was not recorded, and
   an unrelaxed chain explains its most extreme rows at least as well.
5. **The second-order warning light.** Withdrawn to a hypothesis. Our own tests
   are one non-rejection with no power, one false positive out of two positives,
   and one partial success out of ten models.
6. **The source of the 35% discrepancy of §5.3.** Four explanations excluded,
   one candidate untested.
7. **Anything about transport, angular or many-body error, other state points,
   other reference potentials, published potentials, or system sizes other than
   *N* = 108.** None were measured.
8. **Reproducibility from a named revision.** Six runs, five commits, every one
   from a dirty tree.

One consequence of R1's replication deserves flagging to the editor: it is a new
measurement, made after the report was written, and it is the only number in the
revision that is not a re-analysis of data the referee already had. It is
deposited in full (`results/exp09_calibration_replication`, 64 records and a
manifest) and it ran in quick mode; we would not object to a referee asking for
it at production settings before acceptance.

**Was "major revision" fair?**

Yes, and the report was better than fair — it was useful. Four of the referee's
findings were correct and load-bearing, and two of them (the shared reference
baseline; the sign of the second-order term) we would probably not have found on
our own, because both are cases where a number that looked good was good for the
wrong reason. The first of those also turned out to be testable directly, which
is the best thing a referee's objection can be. The recommendation was also correctly calibrated in the other
direction: the referee separated the central existence proof, which survives
re-analysis, from the four quantitative claims that did not, and declined to
treat the second as evidence against the first. We think the revised manuscript
is the paper the report asked for, with the two exceptions argued in D1 and D2
and the three experiments in D5 that belong to a different campaign.

We would add one thing the referee could not have known. The 40-configuration
arm, which we had failed to report, turns out to corroborate the referee's own
central diagnosis to three decimal places. Had we disclosed it in the first
submission, the shared-baseline defect would have been visible on the face of
the paper. That is a small illustration of the argument the paper is trying to
make: the number that looked fine was fine because two errors cancelled, and
what exposed it was a completeness of reporting we had not quite managed.
