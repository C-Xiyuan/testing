# Independent validation log

The test suite in `tests/` is written by whoever implements each module, which
makes it good at catching mistakes the author can imagine and poor at catching
the ones they cannot. This file records checks run **separately from the test
suite**, against values that come from outside the codebase: closed-form
results, published lattice sums, and physical measurements.

Every number below was produced by running the code in this repository, not
copied from a report. Where the code disagrees with a target, the disagreement
is stated rather than resolved by adjusting the target.

---

## 1. Geometry and neighbour lists

| Check | Expected | Obtained |
|---|---|---|
| fcc coordination shells (a = 4.05 Å) | 12, 6, 24, 12, 24 | 12, 6, 24, 12, 24 |
| fcc first-shell radius | a/√2 = 2.86378 Å | 2.8638 Å |
| diamond first shell (a = 5.431 Å) | 4 neighbours at a√3/4 = 2.35169 Å | 4 at 2.3517 Å |
| fcc volume per atom | a³/4 = 16.60753 Å³ | 16.60753 Å³ |
| perpendicular width, heavily skewed triclinic cell<br>rows (10,0,0), (7,8,0), (6,5,7) | min(V/\|aᵢ×aⱼ\|) = 6.50987 Å | 6.50987 Å |

The last row matters more than it looks. The shortest *lattice vector* of that
cell is 7.0 Å, so an implementation that uses lattice-vector length instead of
perpendicular face separation would permit a cutoff of 3.5 Å where the true
limit is 3.25 Å, and would silently miss pairs.

## 2. Reference potentials

### Stillinger–Weber silicon

Diamond structure at a₀ = 5.431 Å gives **−4.3366000 eV/atom**, against the
exact value −2ε = −4.33660 eV/atom for ε = 2.1683 eV. The three-body energy is
6.7 × 10⁻²⁹ eV, i.e. identically zero as it must be at the tetrahedral angle.

The value is exact rather than fitted-approximate, for three independent
reasons that all have to hold at once: the diamond nearest-neighbour distance
a₀√3/4 = 2.3517 Å coincides with 2^{1/6}σ where the two-body term attains
exactly −ε; every bond angle is exactly tetrahedral so the three-body term
vanishes; and the second-neighbour shell at 3.8403 Å lies outside the
aσ = 3.77118 Å cutoff. a₀ = 5.431 Å is confirmed to be a stationary point
(E = −4.336408, −4.336600, −4.336405 eV/atom at a = 5.421, 5.431, 5.441 Å).

*This check corrected the design document*, which had specified −4.3363 — the
value belonging to ε = 2.16815 eV (50 kcal/mol) rather than to the ε the
contract itself specifies.

### Lennard-Jones fcc lattice sum

Energy per atom at a₀ = 1.5496 σ, as a function of cutoff:

| r_c / σ | 3.0 | 4.0 | 5.0 | 6.0 |
|---|---|---|---|---|
| E/atom (ε) | −8.30375 | −8.47675 | −8.52939 | −8.56030 |

Extrapolating the last two points with the expected E(r_c) = E_∞ + C/r_c³ tail
gives **E_∞ = −8.603 ε/atom**, against the exact lattice-sum value

  E_min = 2ε[A₁₂ (σ/r)¹² − A₆ (σ/r)⁶] = **−8.6108 ε/atom** at r = 1.09026 σ,

with the fcc lattice sums A₆ = 14.4539, A₁₂ = 12.1319. The 0.09 % gap is
consistent with the crudeness of a two-point 1/r_c³ extrapolation, which
neglects both the repulsive tail and lattice discreteness.

### Lennard-Jones cutoff modes

Energy and force discontinuity at r_c, measured at r_c ∓ 10⁻⁷ Å:

| mode | \|Δu\| (eV) | \|ΔF\| (eV/Å) |
|---|---|---|
| `truncated` | 2.4 × 10⁻⁴ | 1.8 × 10⁻⁴ |
| `shifted` | 1.8 × 10⁻¹¹ | 1.8 × 10⁻⁴ |
| `shifted_force` | 8.7 × 10⁻¹⁹ | 1.6 × 10⁻¹¹ |
| `switched` | 2.2 × 10⁻¹⁹ | 1.4 × 10⁻¹⁶ |

This is why `shifted` is not good enough for molecular dynamics despite having a
continuous energy: the force still jumps, delivering an impulse every time a
pair crosses the cutoff, and the resulting energy drift is a systematic
artefact rather than integrator noise.

### Analytic EAM copper

Fitted targets, recovered by an independent E(V) scan and parabolic fit:

| Property | Target | Obtained |
|---|---|---|
| lattice constant | 3.615 Å | 3.6150 Å |
| cohesive energy | −3.54 eV/atom | −3.5400 eV/atom |
| bulk modulus | 140 GPa | 140.0 GPa |

**And a discrepancy worth stating plainly.** The same potential, which
reproduces those three numbers exactly, gives a maximum phonon frequency of
**5.14 THz** against roughly **7.5 THz** for real copper — about 30 % soft at
the zone boundary.

This is not a defect. The reference potentials define the ground truth of this
study; they are not claims about experiment, and a surrogate is judged against
the reference, never against copper. But it is a striking miniature of the
thesis: a potential fitted to three properties exactly, including one
second-derivative property (the bulk modulus, a long-wavelength curvature),
still misses the short-wavelength curvature badly. Fitting a quantity does not
constrain the quantities you did not fit, even when they are closely related —
which is the same statement the rest of this repository makes about force RMSE.

## 3. Phonon machinery

| Check | Expected | Obtained |
|---|---|---|
| simple-cubic NN springs, ω(q) along x | 2√(k/m)\|sin(qa/2)\| | matches to 1 × 10⁻⁵ relative at every q tested |
| Γ-point acoustic modes | 3 zeros | 3 zeros |
| band folding into a doubled cell | 5 zeros + zone-boundary mode | reproduced |
| acoustic sum rule residual, EAM Cu | 0 | 7.5 × 10⁻¹⁴ eV/Å² |
| DOS normalisation, 4-atom cell | 3N = 12 | 12.000 |
| imaginary modes, relaxed fcc Cu | none | none |

The residual transverse stiffness of the spring lattice (which should be
exactly zero) is confirmed to scale as δ², identifying it as the
finite-displacement artefact it is rather than leaving it as an unexplained
tolerance.

## 4. Descriptors

Measured on a rattled 32-atom fcc cell, over 60 random SO(3) rotations:

| Descriptor | dimension | max relative rotation error | permutation error |
|---|---|---|---|
| SOAP (n_max = 4, l_max = 3) | 40 | 8.0 × 10⁻¹⁵ | 1.9 × 10⁻¹⁴ |
| ACSF (default) | 36 | 2.3 × 10⁻¹⁵ | 8.5 × 10⁻¹⁶ |

Both are invariant to machine precision, and both discriminate the three cubic
structures — the Euclidean separation between fcc, bcc and diamond central-atom
descriptors is non-zero and well outside the invariance noise for every pair.

## 5. Samplers

Hamiltonian Monte Carlo against exact equipartition on a 32-atom Einstein
crystal at 100 K (spring constant 1 eV/Å²):

| Quantity | Exact | Obtained |
|---|---|---|
| ⟨U⟩ | (3N/2) k_BT = 0.41363 eV | 0.41462 ± 0.00196 eV (0.5 σ) |
| var(U) | (3N/2)(k_BT)² = 0.003564 eV² | 0.003459 eV² |

Both moments matter: several incorrect samplers reproduce the mean and not the
variance, because the variance is what a wrong temperature scale breaks.

Two real defects were found and fixed by these checks. Centre-of-mass momentum
removal, which is correct for a translationally invariant potential, biases a
*tethered* system low by 3/(3N) — the centre of mass is a genuine degree of
freedom there and suppressing its momentum under-samples it. And step-size
adaptation driven by a cumulative acceptance rate lags so badly that the
sampler settled at acceptance 0.35 against a target of 0.75; adapting on a
windowed rate brings it to 0.74.
