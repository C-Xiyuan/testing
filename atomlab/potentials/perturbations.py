"""Designed error fields: the experimental instrument of this study.

``docs/theory.md`` argues that the damage a potential error does to an
observable is governed by ``Cov(A, dU)`` -- a projection -- while force RMSE is a
norm of ``grad dU``.  Observing that real models behave that way would be
suggestive.  *Constructing* error fields with chosen projections, and watching
the observables move exactly as predicted while the force error stays fixed, is
a test.  That is what this module is for.

Everything here is a full :class:`~atomlab.potentials.base.Potential` with
analytic forces and virial, so a perturbed model is built by ordinary addition,
``surrogate = reference + perturbation``, and can be sampled or integrated
without any special handling.

The constructions, in increasing order of how much they give away:

* :class:`RadialShellPerturbation` -- a smooth bump in the pair energy at a
  chosen separation.  Its width is the control parameter for the frequency
  argument of theory section 4.1.
* :class:`HighFrequencyPerturbation` -- an oscillation in ``r``.  Large gradient,
  negligible projection onto anything smooth.  The concrete realisation of
  "force error over-weights high-frequency error by ``k^2``".
* :class:`HighEnergyPerturbation` -- supported only at separations that thermal
  sampling almost never visits, so it is invisible to in-distribution force
  error *and* to equilibrium observables, and appears only under extrapolation.
* :class:`SplinePerturbation` -- an arbitrary pair perturbation expanded in a
  basis of shell bumps.  Being linear in its coefficients is what makes the next
  two constructions possible.
* :func:`null_space_perturbation` -- coefficients chosen numerically orthogonal
  to a target observable.  Large force error, no predicted effect.
* :func:`aligned_perturbation` -- coefficients parallel to the covariance
  direction.  Maximal effect per unit force error.

The null-space/aligned pair, built at *matched force RMSE*, is the sharpest
falsification test in the programme: if force error were the right summary the
two would damage the observable equally, and the theory says one will do
essentially nothing while the other does a great deal.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..neighbors import build_neighbor_list, pair_vectors
from ..types import Configuration, Result
from .base import Potential

__all__ = [
    "PairPerturbation",
    "RadialShellPerturbation",
    "HighFrequencyPerturbation",
    "HighEnergyPerturbation",
    "SplinePerturbation",
    "AngularPerturbation",
    "force_rms",
    "match_force_error",
    "build_shell_basis",
    "basis_energy_matrix",
    "null_space_perturbation",
    "aligned_perturbation",
    "random_perturbation",
]


def _smooth_taper(r: np.ndarray, cutoff: float, width: float):
    """C^2 taper going 1 -> 0 over the last ``width`` angstrom before ``cutoff``.

    Uses the quintic smoothstep ``x^3(10 - 15x + 6x^2)``, whose value and first
    two derivatives vanish at both ends.  A perturbation that did not vanish
    smoothly at the cutoff would inject an impulsive force every time a pair
    crossed it, and that artefact -- not the intended error field -- would
    dominate the dynamics.

    Returns ``(s, ds/dr)``.
    """
    x = np.clip((cutoff - r) / width, 0.0, 1.0)
    s = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
    ds = -(30.0 * x**2 * (1.0 - x) ** 2) / width
    return s, ds


class PairPerturbation(Potential):
    """Base class for a perturbation that is a sum over pairs of ``du(r)``.

    Subclasses implement :meth:`radial`, returning ``(du, d(du)/dr)`` for an
    array of separations.  Everything else -- neighbour handling, force
    assembly, virial, the amplitude scaling used to match a force error -- is
    provided here, so a new designed error field is a few lines.
    """

    amplitude: float = 1.0

    def __init__(self, cutoff: float, *, taper_width: float = 1.0, name: str = "pair-perturbation"):
        self.cutoff = float(cutoff)
        self.taper_width = float(taper_width)
        self.name = name

    def radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(du, du')`` in eV and eV/A, *before* the taper is applied."""
        raise NotImplementedError

    def _tapered(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, du = self.radial(r)
        s, ds = _smooth_taper(r, self.cutoff, self.taper_width)
        return u * s, du * s + u * ds

    def pair_energy(self, r) -> np.ndarray:
        """The perturbation's pair function, for plotting and for diagnostics."""
        return self._tapered(np.atleast_1d(np.asarray(r, dtype=float)))[0]

    def compute(self, configuration: Configuration, *, forces=True, virial=True) -> Result:
        nl = build_neighbor_list(configuration, self.cutoff, half=True)
        d, r = pair_vectors(configuration, nl)
        if r.size == 0:
            return Result(
                energy=0.0,
                forces=np.zeros((configuration.n_atoms, 3)),
                virial=np.zeros((3, 3)) if virial else None,
            )

        u, du = self._tapered(r)
        f = np.zeros((configuration.n_atoms, 3))
        # d points from i to j, so r = |r_j - r_i| and dr/dr_i = -d/r. Hence
        # F_i = -dU/dr_i = +u'(r) * d/r. The sign is worth spelling out: with a
        # repulsive core (u' < 0) this pushes i along -d, away from j, which is
        # the check to run mentally before trusting it.
        pair_force = (du / r)[:, None] * d
        np.add.at(f, nl.i, pair_force)
        np.add.at(f, nl.j, -pair_force)

        w = None
        if virial:
            # W_ab = sum_{i<j} f_ij (x) r_ij with r_ij = r_i - r_j = -d.
            w = -(pair_force[:, :, None] * d[:, None, :]).sum(axis=0)
        return Result(energy=float(u.sum()), forces=f, virial=w)

    def scaled(self, factor: float) -> "PairPerturbation":
        """Return a copy with the amplitude multiplied by ``factor``.

        Every perturbation here is linear in its amplitude, so the force error
        is too, which is what makes :func:`match_force_error` a one-shot
        rescaling rather than a root find.
        """
        import copy

        other = copy.deepcopy(self)
        other.amplitude = self.amplitude * float(factor)
        return other


class RadialShellPerturbation(PairPerturbation):
    """A Gaussian bump in the pair energy at separation ``r0``.

    ``du(r) = amplitude * exp(-(r - r0)^2 / 2 width^2)``, tapered to zero at the
    cutoff.

    The width is the instrument.  Section 4.1 of ``docs/theory.md`` predicts
    that force error scales as ``amplitude / sqrt(width)`` -- the gradient goes
    as ``amplitude/width`` and the fraction of pairs inside the shell as
    ``width`` -- while the coupling to ``g(r)`` scales as ``amplitude * width``.
    At *fixed* force error the observable damage should therefore scale as
    ``width^{3/2}``, which is unbounded: two models with identical force RMSE
    can differ in observable error by any factor one likes.
    """

    def __init__(self, r0: float, width: float, amplitude: float = 1.0,
                 cutoff: float = 8.0, *, taper_width: float = 1.0):
        super().__init__(cutoff, taper_width=taper_width,
                         name=f"shell(r0={r0:.2f},w={width:.3f})")
        self.r0 = float(r0)
        self.width = float(width)
        self.amplitude = float(amplitude)

    def radial(self, r):
        z = (r - self.r0) / self.width
        g = np.exp(-0.5 * z**2)
        return self.amplitude * g, self.amplitude * g * (-z / self.width)


class HighFrequencyPerturbation(PairPerturbation):
    """An oscillation in ``r``: ``du(r) = amplitude * sin(2 pi r / wavelength)``.

    Its gradient carries a factor ``2 pi / wavelength``, so its contribution to
    force RMSE grows without bound as the wavelength shrinks, while its integral
    against any smooth function of ``r`` -- which is what an observable's
    covariance with it looks like -- oscillates to nearly nothing.  Making the
    wavelength small is therefore a way to manufacture arbitrarily bad-looking
    force error with arbitrarily little physical consequence.
    """

    def __init__(self, wavelength: float, amplitude: float = 1.0, cutoff: float = 8.0,
                 *, r_min: float = 2.0, taper_width: float = 1.0):
        super().__init__(cutoff, taper_width=taper_width,
                         name=f"highfreq(lambda={wavelength:.2f})")
        self.wavelength = float(wavelength)
        self.amplitude = float(amplitude)
        self.r_min = float(r_min)

    def radial(self, r):
        k = 2.0 * np.pi / self.wavelength
        # An envelope that switches on above r_min keeps the perturbation out of
        # the steeply repulsive core, where it would change the effective atomic
        # size rather than test what it is meant to test.
        x = np.clip((r - self.r_min) / self.taper_width, 0.0, 1.0)
        env = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        denv = (30.0 * x**2 * (1.0 - x) ** 2) / self.taper_width
        s, c = np.sin(k * r), np.cos(k * r)
        return self.amplitude * s * env, self.amplitude * (k * c * env + s * denv)


class HighEnergyPerturbation(PairPerturbation):
    """Error supported only at separations thermal sampling rarely visits.

    ``du(r) = amplitude * (r_onset - r)^3`` for ``r < r_onset``, zero above.  At
    a temperature where pairs almost never come closer than ``r_onset``, this
    contributes nothing to a test-set force error drawn from that same ensemble
    and nothing to any equilibrium observable -- and then dominates under
    compression, heating, or any rare event that pushes atoms together.

    It is the honest steel-man of force error: this error is invisible not
    because force RMSE is a bad summary but because *nothing* evaluated
    in-distribution can see it.  The remedy is out-of-distribution probes, not a
    better norm.  Section 5 of ``docs/theory.md``.
    """

    def __init__(self, r_onset: float, amplitude: float = 1.0, cutoff: float = 8.0):
        # The support ends at r_onset, well inside the cutoff, so no taper is
        # needed: du and du' already vanish there by construction.
        super().__init__(cutoff, taper_width=1.0, name=f"highE(r<{r_onset:.2f})")
        self.r_onset = float(r_onset)
        self.amplitude = float(amplitude)

    def radial(self, r):
        inside = r < self.r_onset
        z = np.where(inside, self.r_onset - r, 0.0)
        return self.amplitude * z**3, -3.0 * self.amplitude * z**2

    def _tapered(self, r):
        # Override: the cubic already vanishes with its derivative at r_onset,
        # and applying the outer taper would be a no-op that costs time.
        return self.radial(r)


class SplinePerturbation(PairPerturbation):
    """An arbitrary pair perturbation expanded in a basis of shell bumps.

    ``du(r) = sum_k w_k b_k(r)`` with ``b_k`` narrow Gaussians on a grid.  Being
    linear in ``w`` is the whole point: the covariance of the perturbation with
    an observable is then linear in ``w`` too, so choosing ``w`` to make that
    covariance vanish -- or to maximise it -- is a linear-algebra problem rather
    than an optimisation.
    """

    def __init__(self, centres, width: float, coefficients, cutoff: float = 8.0,
                 *, taper_width: float = 1.0, name: str = "spline-perturbation"):
        super().__init__(cutoff, taper_width=taper_width, name=name)
        self.centres = np.asarray(centres, dtype=float)
        self.width = float(width)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != self.centres.shape:
            raise ValueError(
                f"got {self.centres.size} centres and {self.coefficients.size} coefficients"
            )
        self.amplitude = 1.0

    def radial(self, r):
        z = (r[:, None] - self.centres[None, :]) / self.width
        g = np.exp(-0.5 * z**2)
        w = self.amplitude * self.coefficients
        return g @ w, (g * (-z / self.width)) @ w


class AngularPerturbation(Potential):
    """A three-body error field that moves bond angles, not pair distances.

    ``dU = lambda * sum_{j<k} fc(r_ij) fc(r_ik) (cos theta_jik - cos0)^2``

    Included because it is qualitatively different from everything else here: it
    is invisible to a pair-only surrogate's functional form, and it perturbs the
    angular distribution while leaving the radial distribution almost untouched.
    That makes it the natural instrument for showing that "which observable"
    matters as much as "how big the error is".
    """

    def __init__(self, amplitude: float, cos_theta0: float = -1.0 / 3.0,
                 cutoff: float = 3.8, *, taper_width: float = 0.8):
        self.amplitude = float(amplitude)
        self.cos_theta0 = float(cos_theta0)
        self.cutoff = float(cutoff)
        self.taper_width = float(taper_width)
        self.name = f"angular(lam={amplitude:.3g})"

    def compute(self, configuration: Configuration, *, forces=True, virial=True) -> Result:
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        d, r = pair_vectors(configuration, nl)
        n_atoms = configuration.n_atoms

        f = np.zeros((n_atoms, 3))
        w = np.zeros((3, 3))
        energy = 0.0

        fc, dfc = _smooth_taper(r, self.cutoff, self.taper_width)
        order = np.argsort(nl.i, kind="stable")
        starts = np.searchsorted(nl.i[order], np.arange(n_atoms))
        ends = np.searchsorted(nl.i[order], np.arange(n_atoms), side="right")

        for i in range(n_atoms):
            idx = order[starts[i]:ends[i]]
            if idx.size < 2:
                continue
            dj, rj = d[idx], r[idx]
            fcj, dfcj = fc[idx], dfc[idx]
            a, b = np.triu_indices(idx.size, k=1)

            v1, v2 = dj[a], dj[b]
            r1, r2 = rj[a], rj[b]
            cos = (v1 * v2).sum(axis=1) / (r1 * r2)
            h = (cos - self.cos_theta0) ** 2
            dh = 2.0 * (cos - self.cos_theta0)
            g = fcj[a] * fcj[b]
            energy += float(self.amplitude * (g * h).sum())

            u1, u2 = v1 / r1[:, None], v2 / r2[:, None]
            # d(cos)/d r_j = (u2 - cos u1)/r1, and symmetrically for k.
            dcos_dj = (u2 - cos[:, None] * u1) / r1[:, None]
            dcos_dk = (u1 - cos[:, None] * u2) / r2[:, None]

            grad_j = self.amplitude * (
                (h * dfcj[a] * fcj[b])[:, None] * u1 + (g * dh)[:, None] * dcos_dj
            )
            grad_k = self.amplitude * (
                (h * fcj[a] * dfcj[b])[:, None] * u2 + (g * dh)[:, None] * dcos_dk
            )

            fj, fk = -grad_j, -grad_k
            np.add.at(f, nl.j[idx][a], fj)
            np.add.at(f, nl.j[idx][b], fk)
            f[i] -= (fj + fk).sum(axis=0)

            if virial:
                # W_ab = sum over triplets of (F_j (x) r_ij + F_k (x) r_ik),
                # accumulated on the bond vectors rather than on absolute
                # positions, which is the only form correct under periodicity.
                w += fj.T @ v1 + fk.T @ v2

        return Result(energy=energy, forces=f, virial=w if virial else None)


# --------------------------------------------------------------------------
# Matching force error, and the designed constructions
# --------------------------------------------------------------------------


def force_rms(potential: Potential, configurations: Sequence[Configuration]) -> float:
    """Root-mean-square force component in eV/A over a set of configurations.

    This is the ordinary force RMSE, computed exactly as a practitioner would
    report it: the mean is over every cartesian component of every atom of every
    configuration.
    """
    total, count = 0.0, 0
    for cfg in configurations:
        f = potential.forces(cfg)
        total += float((f**2).sum())
        count += f.size
    if count == 0:
        raise ValueError("no configurations supplied")
    return float(np.sqrt(total / count))


def match_force_error(perturbation, configurations, target_force_rms: float):
    """Rescale a perturbation so its force RMSE equals ``target_force_rms``.

    Every perturbation in this module is linear in its amplitude, so this is an
    exact one-shot rescaling.  Matching force error is what makes the
    comparisons meaningful: it holds fixed precisely the number a practitioner
    would use to choose between models, so that any difference in observable
    error is a difference the practitioner could not have seen.
    """
    current = force_rms(perturbation, configurations)
    if current <= 0.0:
        raise ValueError(f"{perturbation.name} has zero force on these configurations")
    return perturbation.scaled(target_force_rms / current)


def build_shell_basis(r_min: float, r_max: float, n_basis: int, cutoff: float,
                      *, width: float | None = None) -> list[RadialShellPerturbation]:
    """A basis of narrow, evenly spaced shell bumps spanning ``[r_min, r_max]``.

    ``width`` defaults to the spacing, which makes neighbouring basis functions
    overlap at about ``exp(-1/2)`` -- enough that any smooth pair perturbation
    is representable, narrow enough that the basis is not badly conditioned.
    """
    centres = np.linspace(r_min, r_max, n_basis)
    if width is None:
        width = float(centres[1] - centres[0]) if n_basis > 1 else 0.2
    return [RadialShellPerturbation(c, width, 1.0, cutoff) for c in centres]


def basis_energy_matrix(basis: Sequence[PairPerturbation],
                        configurations: Sequence[Configuration]) -> np.ndarray:
    """``(M, K)`` matrix of each basis perturbation's energy on each frame.

    This is the design matrix for the covariance calculations below.  Computing
    it once and reusing it is what keeps the null-space construction cheap.
    """
    return np.array([[b.energy(cfg) for b in basis] for cfg in configurations])


def _covariance(a_samples: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """``(K, J)`` covariance between each basis energy and each observable."""
    a = np.asarray(a_samples, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    ac = a - a.mean(axis=0, keepdims=True)
    ec = energies - energies.mean(axis=0, keepdims=True)
    return ec.T @ ac / (a.shape[0] - 1)


def null_space_perturbation(
    basis: Sequence[PairPerturbation],
    a_samples,
    configurations: Sequence[Configuration],
    *,
    target_force_rms: float,
    seed: int = 0,
    energies: np.ndarray | None = None,
    rank_tolerance: float = 1e-10,
) -> tuple[SplinePerturbation, dict]:
    """Build a pair perturbation numerically orthogonal to a target observable.

    Because a :class:`SplinePerturbation` is linear in its coefficients, the
    covariance of its energy with the observable is ``C^T w`` where ``C`` is the
    ``(K, J)`` covariance matrix between basis energies and observable
    components.  Choosing ``w`` in the null space of ``C^T`` therefore makes the
    predicted first-order shift vanish identically, at any force error we care
    to impose.

    Parameters
    ----------
    basis:
        Shell-bump basis from :func:`build_shell_basis`.  Needs at least one
        more member than the observable has components, or the null space is
        empty.
    a_samples:
        ``(M,)`` or ``(M, J)`` observable values on the reference frames.
    configurations:
        The same ``M`` reference frames, in the same order.
    target_force_rms:
        Force RMSE the result is scaled to, in eV/A.

    Returns
    -------
    (SplinePerturbation, dict)
        The dict reports the achieved covariance, the covariance of a random
        perturbation of equal force error, and the resulting suppression factor
        -- the number that says whether the construction actually worked.
    """
    basis = list(basis)
    if energies is None:
        energies = basis_energy_matrix(basis, configurations)
    a = np.asarray(a_samples, dtype=float)
    if a.ndim == 1:
        a = a[:, None]

    cov = _covariance(a, energies)                       # (K, J)
    # Left null space of cov: vectors w with w^T cov = 0.
    u, s, _ = np.linalg.svd(cov, full_matrices=True)
    rank = int((s > rank_tolerance * max(s.max(), 1e-300)).sum()) if s.size else 0
    null_basis = u[:, rank:]
    if null_basis.shape[1] == 0:
        raise ValueError(
            f"the {len(basis)}-member basis has no null space against a "
            f"{a.shape[1]}-component observable; use more basis functions"
        )

    rng = np.random.default_rng(seed)
    w = null_basis @ rng.normal(size=null_basis.shape[1])
    w /= np.linalg.norm(w)

    centres = np.array([b.r0 for b in basis])
    perturbation = SplinePerturbation(
        centres, basis[0].width, w, basis[0].cutoff, name="null-space"
    )
    perturbation = match_force_error(perturbation, configurations, target_force_rms)

    reference = random_perturbation(
        basis, configurations, target_force_rms=target_force_rms, seed=seed + 1
    )
    achieved = float(np.abs(_covariance(a, (energies @ (perturbation.amplitude *
                                                        perturbation.coefficients))[:, None])).max())
    random_cov = float(np.abs(_covariance(a, (energies @ (reference.amplitude *
                                                          reference.coefficients))[:, None])).max())

    diagnostics = {
        "null_space_dimension": int(null_basis.shape[1]),
        "covariance_rank": rank,
        "achieved_covariance": achieved,
        "random_covariance": random_cov,
        "suppression_factor": random_cov / achieved if achieved > 0 else float("inf"),
        "force_rms": force_rms(perturbation, configurations),
    }
    return perturbation, diagnostics


def aligned_perturbation(
    basis: Sequence[PairPerturbation],
    a_samples,
    configurations: Sequence[Configuration],
    *,
    target_force_rms: float,
    component: int = 0,
    energies: np.ndarray | None = None,
) -> tuple[SplinePerturbation, dict]:
    """Build the pair perturbation of maximal effect per unit force error.

    Coefficients parallel to the covariance direction.  This is the complement
    of :func:`null_space_perturbation`: same basis, same force RMSE, opposite
    projection.  Reporting the two side by side is the cleanest statement the
    study can make, because the only thing that differs between them is the one
    quantity the theory says matters and the reported metric cannot see.
    """
    basis = list(basis)
    if energies is None:
        energies = basis_energy_matrix(basis, configurations)
    a = np.asarray(a_samples, dtype=float)
    if a.ndim == 1:
        a = a[:, None]

    cov = _covariance(a, energies)
    if a.shape[1] == 1:
        w = cov[:, 0]
    else:
        u, _, _ = np.linalg.svd(cov, full_matrices=False)
        w = u[:, component]
    norm = np.linalg.norm(w)
    if norm == 0:
        raise ValueError("observable has zero covariance with every basis function")
    w = w / norm

    centres = np.array([b.r0 for b in basis])
    perturbation = SplinePerturbation(
        centres, basis[0].width, w, basis[0].cutoff, name="aligned"
    )
    perturbation = match_force_error(perturbation, configurations, target_force_rms)
    achieved = float(np.abs(_covariance(a, (energies @ (perturbation.amplitude *
                                                        perturbation.coefficients))[:, None])).max())
    return perturbation, {
        "achieved_covariance": achieved,
        "force_rms": force_rms(perturbation, configurations),
    }


def random_perturbation(
    basis: Sequence[PairPerturbation],
    configurations: Sequence[Configuration],
    *,
    target_force_rms: float,
    seed: int = 0,
) -> SplinePerturbation:
    """A perturbation with random coefficients, scaled to a given force error.

    The control arm: whatever the designed constructions achieve has to be
    compared against what an arbitrary error field of the same reported quality
    does.
    """
    basis = list(basis)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=len(basis))
    w /= np.linalg.norm(w)
    centres = np.array([b.r0 for b in basis])
    perturbation = SplinePerturbation(
        centres, basis[0].width, w, basis[0].cutoff, name=f"random(seed={seed})"
    )
    return match_force_error(perturbation, configurations, target_force_rms)
