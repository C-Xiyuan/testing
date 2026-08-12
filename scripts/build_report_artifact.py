#!/usr/bin/env python3
"""Emit the published report with figures inlined as data URIs."""
import base64
import json
from pathlib import Path

ROOT = Path("/home/user/testing")
OUT = Path("/tmp/claude-0/-home-user-testing/27fb9859-976b-5117-b57e-5a0a7a305adc/scratchpad/report.html")


def img(name):
    data = base64.b64encode((ROOT / "figures" / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


records = json.loads((ROOT / "results/exp07_designed_counterexamples/records.json").read_text())
gen = json.loads((ROOT / "results/exp07_designed_counterexamples/generalisation.json").read_text())
widths = json.loads((ROOT / "results/exp06_response_validation/width_records.json").read_text())
zoo = json.loads((ROOT / "results/exp05_proxy_correlation/records.json").read_text())

high = sorted((r for r in records if r["force_rms_level"] == 4e-3),
              key=lambda r: r["target_correlation"])

rows = "\n".join(
    f"""      <tr>
        <td class="name">{r['name'].replace('random0','random (a)').replace('random1','random (b)')}</td>
        <td class="num quiet">{r['force_rms_out_of_sample']*1e3:.3f}</td>
        <td class="num">{r['target_correlation']:+.3f}</td>
        <td class="num">{r['target_predicted']:+.2f}</td>
        <td class="num strong">{r['target_measured']:+.2f} <span class="pm">± {r['target_error']:.2f}</span></td>
        <td class="num quiet">{abs(r['target_measured'])/r['target_error']:.1f}σ</td>
      </tr>"""
    for r in high
)

gen_rows = "\n".join(
    f"""      <tr><td class="num">{g['n_construct']}</td>
        <td class="num strong">{g['out_of_sample']:.3f} <span class="pm">± {g['out_of_sample_error']:.3f}</span></td></tr>"""
    for g in gen
)

HTML = f"""<title>Error Fields, Not Error Norms</title>
<style>
  :root {{
    color-scheme: light;
    --ground:  #F3F5F5;
    --surface: #FFFFFF;
    --ink:     #10171A;
    --ink-2:   #35464A;
    --muted:   #5C686B;
    --rule:    #DCE3E3;
    --rule-2:  #EDF1F1;
    --accent:  #0E6A74;
    --accent-soft: #E2EFF0;
    --warn:    #94301F;
    --warn-soft: #F7E9E5;

    --display: ui-sans-serif, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --body: "Iowan Old Style", "Charter", "Bitstream Charter", Georgia, "Times New Roman", serif;
    --data: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Menlo, Consolas, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground:  #0D1315;
      --surface: #141C1F;
      --ink:     #E7EDED;
      --ink-2:   #C0CCCE;
      --muted:   #90A0A3;
      --rule:    #253134;
      --rule-2:  #1B2427;
      --accent:  #46AAB3;
      --accent-soft: #12292C;
      --warn:    #D9836F;
      --warn-soft: #2A1B18;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground:  #0D1315;
    --surface: #141C1F;
    --ink:     #E7EDED;
    --ink-2:   #C0CCCE;
    --muted:   #90A0A3;
    --rule:    #253134;
    --rule-2:  #1B2427;
    --accent:  #46AAB3;
    --accent-soft: #12292C;
    --warn:    #D9836F;
    --warn-soft: #2A1B18;
  }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: var(--body);
    font-size: 17px;
    line-height: 1.62;
    margin: 0;
    padding: 0 1.25rem 6rem;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 68ch; margin: 0 auto; }}
  .wide {{ max-width: 92ch; margin-inline: auto; }}

  header {{ padding: 5.5rem 0 3rem; border-bottom: 1px solid var(--rule); margin-bottom: 3rem; }}
  .eyebrow {{
    font-family: var(--display); font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
    margin: 0 0 1.1rem;
  }}
  h1 {{
    font-family: var(--display); font-weight: 680; font-size: clamp(2.1rem, 5.2vw, 3.05rem);
    line-height: 1.06; letter-spacing: -0.025em; text-wrap: balance; margin: 0 0 1.3rem;
  }}
  .standfirst {{ font-size: 1.19rem; line-height: 1.55; color: var(--ink-2); margin: 0; text-wrap: pretty; }}

  h2 {{
    font-family: var(--display); font-weight: 650; font-size: 1.32rem; letter-spacing: -0.014em;
    margin: 3.6rem 0 0.35rem; text-wrap: balance;
  }}
  h2 + .lede {{ color: var(--muted); font-size: 0.95rem; margin: 0 0 1.4rem; }}
  h3 {{ font-family: var(--display); font-weight: 620; font-size: 1.0rem; margin: 2.2rem 0 0.5rem; }}
  p {{ margin: 0 0 1.15rem; text-wrap: pretty; }}
  a {{ color: var(--accent); text-underline-offset: 2px; }}
  strong {{ font-weight: 600; }}
  em {{ font-style: italic; }}

  .eq {{
    font-family: var(--data); font-size: 1.02rem; text-align: center;
    background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
    padding: 1.15rem 1rem; margin: 1.8rem 0; overflow-x: auto;
  }}
  .eq b {{ color: var(--accent); font-weight: 600; }}

  /* The comparison block: the layout encodes the finding. One row is
     deliberately unremarkable because the quantity it shows is unremarkable. */
  .compare {{
    background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
    margin: 2rem 0; overflow: hidden;
  }}
  .compare-row {{ display: grid; grid-template-columns: 1fr auto auto; gap: 1rem 1.5rem;
    align-items: baseline; padding: 1.05rem 1.3rem; }}
  .compare-row + .compare-row {{ border-top: 1px solid var(--rule-2); }}
  .compare-row.headline {{ background: var(--accent-soft); }}
  .compare-label {{ font-family: var(--display); font-size: 0.83rem; font-weight: 600;
    letter-spacing: 0.02em; color: var(--ink-2); }}
  .compare-label span {{ display: block; font-weight: 400; color: var(--muted);
    font-size: 0.76rem; letter-spacing: 0; margin-top: 0.12rem; }}
  .compare-val {{ font-family: var(--data); font-variant-numeric: tabular-nums;
    font-size: 1.12rem; text-align: right; white-space: nowrap; }}
  .compare-row.headline .compare-val {{ color: var(--accent); font-weight: 600; font-size: 1.3rem; }}
  .compare-row.quiet .compare-val {{ color: var(--muted); font-size: 1.0rem; }}

  .tablewrap {{ overflow-x: auto; margin: 1.6rem 0 0.6rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th {{
    font-family: var(--display); font-size: 0.71rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--muted); text-align: right;
    padding: 0 0.7rem 0.55rem; border-bottom: 1px solid var(--rule); white-space: nowrap;
  }}
  th:first-child, td.name {{ text-align: left; }}
  td {{ padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--rule-2); }}
  td.name {{ font-family: var(--display); font-size: 0.86rem; font-weight: 500; white-space: nowrap; }}
  td.num {{ font-family: var(--data); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
  td.strong {{ color: var(--accent); font-weight: 600; }}
  td.quiet {{ color: var(--muted); }}
  .pm {{ color: var(--muted); font-size: 0.86em; }}
  caption {{ caption-side: bottom; text-align: left; color: var(--muted); font-size: 0.83rem;
    padding-top: 0.8rem; line-height: 1.5; }}

  figure {{ margin: 2.4rem 0; }}
  figure img {{ width: 100%; max-width: 100%; height: auto; display: block;
    background: #fff; border: 1px solid var(--rule); border-radius: 3px; }}
  figcaption {{ color: var(--muted); font-size: 0.86rem; line-height: 1.5; margin-top: 0.75rem; }}
  figcaption b {{ color: var(--ink-2); font-weight: 600; }}

  .note {{
    border-left: 2px solid var(--warn); background: var(--warn-soft);
    padding: 1rem 1.2rem; margin: 1.8rem 0; border-radius: 0 3px 3px 0;
  }}
  .note p:last-child {{ margin-bottom: 0; }}
  .note .tag {{ font-family: var(--display); font-size: 0.7rem; font-weight: 650;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--warn); display: block;
    margin-bottom: 0.45rem; }}

  ul {{ margin: 0 0 1.15rem; padding-left: 1.2rem; }}
  li {{ margin-bottom: 0.5rem; }}
  li::marker {{ color: var(--muted); }}

  footer {{ margin-top: 4.5rem; padding-top: 1.6rem; border-top: 1px solid var(--rule);
    color: var(--muted); font-size: 0.86rem; }}
  code {{ font-family: var(--data); font-size: 0.88em; background: var(--rule-2);
    padding: 0.1em 0.34em; border-radius: 2px; }}
</style>

<header class="wrap">
  <p class="eyebrow">Machine-learned interatomic potentials</p>
  <h1>Error fields, not error norms</h1>
  <p class="standfirst">Machine-learned potentials are selected by force error. Here are four
  models whose force error agrees to half a percent, and whose physics does not agree at all —
  along with the formula that predicted each one in advance, without simulating any of them.</p>
</header>

<main class="wrap">

<h2>The claim</h2>
<p class="lede">What controls the damage is a projection, not a magnitude.</p>

<p>A machine-learned interatomic potential is fitted, its force RMSE on held-out
configurations is reported, and it is then used to run molecular dynamics from which
some physical quantity is extracted. The implicit chain is <em>small force error ⇒ good
physics</em>.</p>

<p>For a static observable <em>A</em>, writing the fitted model as
<code>U = U₀ + δU</code>, first-order perturbation theory in the canonical ensemble gives</p>

<div class="eq">Δ⟨A⟩ = −β · <b>Cov₀(A, δU)</b> + O(δU²)</div>

<p>The observable error is a <strong>covariance between the observable and the error
field</strong> — a projection. Force RMSE is a <strong>norm of that field's gradient</strong>.
Norms and projections are related by an inequality, and the gradient weights the error
spectrum by <em>k</em>² while a smooth observable weights it by roughly <em>k</em>⁰. The two
functionals rank models by nearly opposite criteria.</p>

<h2>Four models, one force error</h2>
<p class="lede">Liquid argon, 108 atoms, 120 K. Error fields built to differ only in projection.</p>

<p>Each field below was added to the reference potential and scaled so that its force RMSE on
the reference ensemble is the same. The observable is the number of atom pairs in one radial
bin across the first peak of <em>g(r)</em>.</p>

<div class="compare">
  <div class="compare-row quiet">
    <div class="compare-label">Force RMSE, across all four fields
      <span>the number a practitioner would report</span></div>
    <div class="compare-val">3.98 – 4.00 × 10⁻³ eV/Å</div>
    <div class="compare-val">± 0.5 %</div>
  </div>
  <div class="compare-row headline">
    <div class="compare-label">Measured shift in the observable
      <span>the thing that number is supposed to stand for</span></div>
    <div class="compare-val">−0.22 → +9.89 pairs</div>
    <div class="compare-val">45×</div>
  </div>
</div>

<div class="wide">
<div class="tablewrap">
<table>
  <caption>Each row is a surrogate potential differing from the reference by a designed error
  field. <em>ρ</em> is the correlation between that field and the observable. The predicted
  column comes from the reference trajectory alone — no simulation of the surrogate was
  involved in producing it. Significance is the measured shift in units of its own
  uncertainty.</caption>
  <thead>
    <tr>
      <th>error field</th>
      <th>force RMSE<br>(10⁻³ eV/Å)</th>
      <th>ρ(A, δU)</th>
      <th>predicted<br>shift</th>
      <th>measured shift<br>(pairs)</th>
      <th>from zero</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
</div>
</div>

<p>The field labelled <em>null</em> was constructed in the null space of the covariance with
that observable; <em>aligned</em> was constructed parallel to it. Both carry the same force
error. One is invisible to the physics and the other moves it by sixteen standard errors.</p>

<figure class="wide">
  <img src="{img('headline_mechanism.png')}"
       alt="Two panels. Left: measured shift in pair count against the signed correlation between error field and observable, for two force-error levels; both fall on straight monotone lines passing through zero shift at zero correlation. Right: observable error against the width of the error field on a log axis, rising by a factor of five while the force error stays constant.">
  <figcaption><b>(a)</b> At fixed force error, the measured shift is a monotone, near-linear
  function of the correlation — the axis force RMSE cannot see. <b>(b)</b> Varying only the
  width of the error field, with force RMSE held constant to 0.0 %, the observable error still
  spans a factor of five.</figcaption>
</figure>

<h2>Where force error works, and where it stops</h2>
<p class="lede">A coarse filter, not a selector. The distinction is the whole practical point.</p>

<p>The result above says nothing about whether force error is useful <em>in general</em> — it
was measured on fields built to have identical force error. So the question was asked
separately, on a zoo of 34 surrogates spanning three decades of force RMSE.</p>

<p><strong>Across that range force error works.</strong> Its rank correlation with the measured
observable error is ρ = 0.83 [0.68, 0.92]. A model a hundred times worse in force error really
is worse, and no theory was needed to say so. This contradicts the weak-correlation prediction
this study set out to test, and the prediction is withdrawn.</p>

<p>What it cannot do is choose among models of <em>comparable</em> force error — the only
situation a practitioner is ever in.</p>

<div class="compare">
  <div class="compare-row quiet">
    <div class="compare-label">Restricted to models within a 2.6× band of force error
      <span>15 of the 34 surrogates</span></div>
    <div class="compare-val">ρ = 0.34</div>
    <div class="compare-val">force RMSE</div>
  </div>
  <div class="compare-row headline">
    <div class="compare-label">Their physics, over that same band
      <span>what the metric was standing in for</span></div>
    <div class="compare-val">spans 30×</div>
    <div class="compare-val">ρ = 0.94</div>
  </div>
</div>

<figure class="wide">
  <img src="{img('headline_regimes.png')}"
       alt="Two panels. Left: log-log scatter of observable error against force RMSE for 34 surrogates, rising overall, with a shaded band marking models of comparable force error. Right: bar chart of observable error for the fifteen models inside that band, ordered by force RMSE, showing no trend; six models sharing a force RMSE of 4.0e-3 range from 1 to 25 pairs.">
  <figcaption><b>(a)</b> Across three decades the trend is real. <b>(b)</b> Inside the band it
  is not: six of these models share a force RMSE of 4.0 × 10⁻³ eV/Å to two significant figures,
  and their observable errors run from 1.0 to 24.9 pairs.</figcaption>
</figure>

<p>So force error will tell you that a badly fitted model is bad. It will not tell you which of
your good models to use — and the designed fields above show why: within a band, the ordering
is set by the projection, which force error does not measure.</p>

<div class="note">
  <span class="tag">Also refuted</span>
  <p>The theory document proposed <code>std(δU)/force_RMSE</code> as a cheap empirical
  stand-in for the inverse spectral weighting. Measured against observable error across the
  same zoo it gives ρ = <strong>−0.01</strong> [−0.37, 0.37]. It carries no information here.
  Proposed and refuted within the same study.</p>
</div>

<h2>The formula predicts it</h2>
<p class="lede">One energy evaluation per stored frame, instead of a simulation per model.</p>

<p>Every predicted value above is <code>−β Cov₀(A, δU)</code> evaluated on the reference
trajectory. Across all eight fields the residual between prediction and direct measurement is
<strong>1.01 σ rms</strong> — agreement at exactly the level the error bars claim, neither
better nor worse.</p>

<figure>
  <img src="{img('headline_prediction.png')}"
       alt="Parity plot of measured shift against predicted shift for eight error fields, spanning minus one and a half to plus ten pairs. All points lie on the diagonal within their error bars.">
  <figcaption>Predicted against measured, for every field at both force levels. The dashed line
  is equality, not a fit.</figcaption>
</figure>

<p>This is the practical consequence. Estimating what a model will do to an observable costs
one energy evaluation per frame of an existing trajectory — no molecular dynamics with the
model at all.</p>

<h3>The second-order term is right too</h3>
<p>For a null-space field the first-order term vanishes by construction, so any residual
effect must be second order — and the same expansion predicts that. At both force levels the
measured residual agreed with the second-order estimate to within its error bar. The formula
is not merely right at leading order; its own correction term is right as well.</p>

<h3>How much data the construction needs</h3>
<p>A null-space field is orthogonal to the observable <em>on the frames it was built from</em>
by definition, so the construction is worthless unless it survives on fresh frames. Building on
one half of the reference trajectory and evaluating on the other:</p>

<div class="tablewrap">
<table>
  <caption>Out-of-sample predicted shift of the null-space field, against the aligned field's
  10.2 pairs at the same force error. An earlier attempt using 30 construction frames gave
  1.2–1.6 pairs: the null space of a covariance estimated from 30 samples is the null space of
  the noise.</caption>
  <thead><tr><th>construction frames</th><th>out-of-sample shift (pairs)</th></tr></thead>
  <tbody>
{gen_rows}
  </tbody>
</table>
</div>

<h2>Orthogonality belongs to one observable</h2>
<p class="lede">There is no error field that is harmless in general.</p>

<p>The null-space field leaves its target bin alone and moves the <em>rest</em> of the
<em>g(r)</em> curve by +11.4 pairs — as much as the aligned field moves it (+11.2). This is
the point rather than a caveat. Harmlessness is a relation between an error and a particular
question, so no single scalar can summarise model quality: not force RMSE, and not any
replacement for it either. What can be computed is a <em>vector</em> of response scores, one
per observable anyone actually cares about.</p>

<h2>What did not work</h2>
<p class="lede">Three results that went against the project, reported at the same length as the ones that did not.</p>

<div class="note">
  <span class="tag">Retracted</span>
  <p>An early version of this work proposed that at fixed force error the observable damage
  should grow as <code>width^3/2</code>. Measured on a verified ensemble, the exponent is
  <strong>0.56</strong> — off by a factor of three. The claim was withdrawn on analytic grounds
  before the measurement, and the measurement confirms the withdrawal was right. What survives
  is the weaker statement the data supports: force error and observable error depend
  differently on the shape of the error field, so their ratio is not a constant.</p>
</div>

<div class="note">
  <span class="tag">Open — three causes ruled out</span>
  <p>In a separate sweep, three estimates of the same shift were: first order −3.835,
  exact reweighting −3.884, direct sampling −1.948 ± 0.821. The two sharing the reference
  samples agreed to 1.3 %; the one requiring an independent chain disagreed by a factor of two.</p>
  <p>The obvious diagnosis was that the direct estimate's error bar was too small, since
  blocking cannot see the offset between two chains' slow modes. <strong>That diagnosis was
  wrong.</strong> Sampling both potentials from six independent seeds gives a between-chain
  scatter of 0.312 pairs against a within-chain blocking error of 0.589 — blocking is
  <em>conservative</em> by a factor of two. The measured shift is +2.377 ± 0.144 across six
  chain pairs, against +1.725 ± 0.101 from first order and +1.757 from exact reweighting on a
  fresh 3000-frame chain: a 35 % systematic difference.</p>
  <p>Excluded by measurement: the error bars, the second-order truncation (the two same-sample
  estimators agree to 2 %), and incomplete relaxation of the perturbed chain (which would bias
  the measurement <em>toward</em> the reference, and it is biased away). What remains is either
  the reweighting estimator's finite-sample behaviour under correlated samples — the effective
  sample size assumes independence — or something not yet identified. Recorded unresolved
  rather than attributed to the nearest plausible cause.</p>
</div>

<div class="note">
  <span class="tag">Unconfirmed</span>
  <p>Checked against the exact dilute-gas limit, the estimator and the exact reweighting
  identity agree to under 2 % in every bin, and both differ from a hand-derived closed form by
  5.4 σ. The most likely explanation is the <code>O(ρ)</code> correction to
  <code>g = exp(−βu)</code>, which is not negligible at this density — but the test that would
  confirm it, repeating at a quarter of the density, has not been run.</p>
</div>

<h2>What is not here</h2>

<ul>
  <li><strong>Fitted models.</strong> Three of the planned model implementations did not
  complete, so every result above uses <em>designed</em> error fields. That is the better test
  of the mechanism — it decouples how large an error is from what shape it has — but it is a
  weaker claim about external validity. This study does not show that real fitted models occupy
  the regime these designed fields explore.</li>
  <li><strong>Dynamical observables.</strong> Diffusion and vibrational spectra need real time
  evolution, which Monte Carlo cannot provide. The theory above covers static averages only,
  and says so.</li>
  <li><strong>A melting point.</strong> Two-phase coexistence is not affordable at the number
  of models compared here. Omitted deliberately rather than reported badly converged.</li>
</ul>

<h2>Method, in brief</h2>

<p>The ground truth is an <em>analytic</em> potential rather than reference data. That looks
like a limitation and is the point: <code>U₀</code> is then known everywhere in configuration
space, so <code>δU</code> is exactly computable rather than estimated, reference observables
converge to arbitrary precision, and error fields can be <em>designed</em> — which turns an
observation into a constructive test.</p>

<p>Static observables are sampled by Hamiltonian Monte Carlo, not molecular dynamics. Because
velocity Verlet is symplectic and reversible, the Metropolis test makes the stationary
distribution exactly <code>exp(−βU)</code> whatever the integration error, which appears only
as a reduced acceptance rate. A thermostat that samples something slightly off-canonical would
introduce exactly the kind of small systematic difference this study attributes to model error.</p>

<div class="note">
  <span class="tag">A bug worth recording</span>
  <p>The liquid was originally built by scaling an fcc lattice to liquid density and sampling
  it. That configuration is a <em>crystal</em>. The bond-order parameter fell monotonically from
  0.5745 toward 0.43 and was still falling; the energy was 30–40 % from the equilibrated
  liquid's. No sampler diagnostic caught it — acceptance and integration error stayed healthy
  throughout, because they measure whether the chain is <em>simulated</em> correctly, not
  whether it has reached its stationary distribution. Only the second question bears on an
  ensemble average, and nothing was asking it. Every trajectory is now split in half and checked
  for drift, and a drifting one raises rather than warning.</p>
</div>

<h2>Supporting validation</h2>
<p class="lede">Checks against targets from outside the codebase.</p>

<div class="tablewrap">
<table>
  <thead><tr><th>check</th><th>target</th><th>obtained</th></tr></thead>
  <tbody>
    <tr><td class="name">Stillinger–Weber Si cohesive energy</td><td class="num quiet">−4.33660 eV/atom</td><td class="num">−4.3366000</td></tr>
    <tr><td class="name">Lennard-Jones fcc lattice sum</td><td class="num quiet">−8.6108 ε/atom</td><td class="num">−8.603</td></tr>
    <tr><td class="name">EAM Cu constant / cohesion / bulk modulus</td><td class="num quiet">3.615 Å / −3.54 eV / 140 GPa</td><td class="num">exact on all three</td></tr>
    <tr><td class="name">Steinhardt Q₆, fcc / bcc / sc</td><td class="num quiet">0.5745 / 0.5106 / 0.3536</td><td class="num">0.5745 / 0.5107 / 0.3536</td></tr>
    <tr><td class="name">SOAP, ACSF rotational invariance</td><td class="num quiet">0</td><td class="num">8×10⁻¹⁵, 2×10⁻¹⁵</td></tr>
    <tr><td class="name">HMC ⟨U⟩ vs equipartition</td><td class="num quiet">(3N/2)k<sub>B</sub>T</td><td class="num">0.5 σ</td></tr>
    <tr><td class="name">Phonon dispersion vs closed form</td><td class="num quiet">2√(k/m)|sin(qa/2)|</td><td class="num">10⁻⁵ relative</td></tr>
  </tbody>
</table>
</div>

<p>And one worth stating although it is not a defect: the analytic EAM reproduces its three
fitted targets exactly and still gives a maximum phonon frequency 30 % below copper's. Fitting
a quantity does not constrain the quantities you did not fit, even closely related ones —
the same statement this study makes about force error, arriving from a different direction.</p>

<footer>
  <p>All numbers produced by the <code>atomlab</code> repository on an equilibrated reference
  ensemble whose stationarity was verified. Raw outputs carry a manifest recording the commit
  that produced them.</p>
</footer>

</main>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} ({len(HTML)/1e6:.2f} MB)")
