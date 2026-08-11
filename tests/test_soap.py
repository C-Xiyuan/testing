"""Tests for the many-body descriptors: SOAP and the polynomial/ACE basis.

The load-bearing tests here are, in order of importance:

1. **Exact rotational invariance** under 100 random SO(3) rotations.  A wrong
   spherical-harmonic convention, a mis-ordered ``(l, m)`` packing, or a
   forgotten Condon-Shortley phase all survive every other test in this file
   and are caught only by this one.
2. **Analytic descriptor derivatives against central differences.**  A model
   trained on a descriptor whose derivatives are subtly wrong reaches a
   perfectly plausible energy error and then produces wrong forces, wrong
   dynamics and wrong observables -- which is exactly the failure mode this
   repository exists to study, so it must not be committed here.
3. Translation, permutation and periodic-image invariance, which are cheap and
   catch index bugs in the neighbour-list plumbing.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.stats import special_ortho_group

from atomlab.build import bcc, diamond, fcc, random_gas, rattle, sc
from atomlab.cell import cell_from_parameters
from atomlab.models.descriptors.bispectrum import Bispectrum, chebyshev, legendre
from atomlab.models.descriptors.soap import (
    SOAP,
    polynomial_radial_basis,
    real_solid_harmonics,
    real_spherical_harmonic_scipy,
)
from atomlab.types import Configuration

# Small parameter sets: these tests do O(6 N) descriptor evaluations for the
# finite-difference checks, so the systems are kept deliberately tiny.
SMALL_SOAP = dict(r_cut=4.2, n_max=3, l_max=3, sigma_atom=0.5, n_quadrature=24)
SMALL_BISPECTRUM = dict(r_cut=4.2, n_radial_2b=4, n_radial_3b=3, l_max=3)


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------


def _triclinic(n_atoms: int = 12, seed: int = 5) -> Configuration:
    """A strongly skewed triclinic cell with random, non-overlapping atoms.

    Triclinic cells are the case where a naive "round the fractional
    coordinates" minimum image is wrong, so a descriptor that quietly assumed
    an orthorhombic cell would fail here and nowhere else.
    """
    cell = cell_from_parameters(9.0, 9.6, 10.4, 71.0, 84.0, 66.0)
    rng = np.random.default_rng(seed)
    frac = rng.random((n_atoms, 3))
    positions = frac @ cell
    return Configuration(
        positions=positions, cell=cell, pbc=True, symbols=("Ar",), masses=np.full(n_atoms, 39.948)
    )


def _systems() -> dict[str, Configuration]:
    """The four geometry classes every derivative check is run on."""
    return {
        "rattled_fcc": rattle(fcc(5.26, "Ar", (2, 2, 2)), 0.12, seed=1),
        "rattled_diamond": rattle(diamond(5.431, "Si", (1, 1, 1)), 0.10, seed=2),
        "random_gas": random_gas(20, 0.020, "Ar", seed=3, min_distance=2.4),
        "triclinic": _triclinic(),
    }


def _descriptors() -> dict[str, object]:
    return {
        "soap": SOAP(**SMALL_SOAP),
        "soap_unnormalised": SOAP(**SMALL_SOAP, normalize=False),
        "bispectrum": Bispectrum(**SMALL_BISPECTRUM),
    }


# --------------------------------------------------------------------------
# derivative helpers
# --------------------------------------------------------------------------


def _dense_derivative(descriptor, configuration: Configuration) -> np.ndarray:
    """Expand the sparse ``(P, D, 3)`` derivative into ``(N, D, N, 3)``.

    Duplicate ``(i, j)`` blocks -- which occur whenever a pair is seen through
    more than one periodic image -- are summed, exactly as
    :meth:`DescriptorOutput.forces_from_energy_gradient` does.
    """
    out = descriptor.compute(configuration, derivatives=True)
    n, d = out.features.shape
    dense = np.zeros((n, d, n, 3))
    np.add.at(dense, (out.pair_i, slice(None), out.pair_j), out.derivatives)
    return dense


def _numerical_derivative(descriptor, configuration: Configuration, delta: float = 1e-5):
    """Central-difference ``(N, D, N, 3)`` derivative of the descriptor itself.

    ``delta = 1e-5 A`` balances truncation error ``O(delta^2)`` against the
    ``O(eps/delta)`` round-off of descriptor values of order unity, giving a
    floor around ``1e-9``.
    """
    base = descriptor.compute(configuration, derivatives=False).features
    n, d = base.shape
    num = np.zeros((n, d, n, 3))
    for j in range(n):
        for k in range(3):
            plus = configuration.copy()
            plus.positions[j, k] += delta
            minus = configuration.copy()
            minus.positions[j, k] -= delta
            fp = descriptor.compute(plus, derivatives=False).features
            fm = descriptor.compute(minus, derivatives=False).features
            num[:, :, j, k] = (fp - fm) / (2.0 * delta)
    return num


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------


class TestSphericalHarmonics:
    def test_solid_harmonics_match_scipy(self):
        """The recurrence must reproduce ``r**l * Y_lm`` from ``sph_harm_y``.

        This pins the convention (no Condon-Shortley phase, ``sqrt(2)`` on
        ``m != 0``, ``m`` running ``-l .. +l``).  Everything else about SOAP is
        downstream of it.
        """
        rng = np.random.default_rng(0)
        d = rng.normal(size=(200, 3))
        l_max = 5
        values, _ = real_solid_harmonics(d, l_max)
        r = np.linalg.norm(d, axis=1)
        worst = 0.0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                reference = real_spherical_harmonic_scipy(l, m, d) * r**l
                got = values[:, l * l + (m + l)]
                worst = max(worst, float(np.max(np.abs(reference - got))))
        assert worst < 1e-12, f"solid harmonics disagree with scipy by {worst:.3e}"

    def test_solid_harmonics_at_the_poles(self):
        """No singularity on the z axis, where spherical-coordinate forms fail."""
        d = np.array([[0.0, 0.0, 1.7], [0.0, 0.0, -2.3], [1e-13, 0.0, 3.0]])
        values, grads = real_solid_harmonics(d, 4)
        assert np.all(np.isfinite(values))
        assert np.all(np.isfinite(grads))

    def test_solid_harmonic_gradients(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=(60, 3)) * 2.0
        l_max = 5
        _, grads = real_solid_harmonics(d, l_max)
        h = 1e-6
        num = np.zeros_like(grads)
        for k in range(3):
            dp = d.copy()
            dp[:, k] += h
            dm = d.copy()
            dm[:, k] -= h
            num[:, :, k] = (
                real_solid_harmonics(dp, l_max, gradient=False)[0]
                - real_solid_harmonics(dm, l_max, gradient=False)[0]
            ) / (2.0 * h)
        rel = np.max(np.abs(num - grads)) / np.max(np.abs(grads))
        assert rel < 1e-7, f"solid-harmonic gradient relative error {rel:.3e}"

    def test_orthonormality_of_harmonics_on_the_sphere(self):
        """A Monte-Carlo check that the normalisation constant is right.

        ``int Y_lm Y_l'm' dOmega = delta``, so the average of ``4 pi Y Y'`` over
        uniformly sampled directions must be the identity.
        """
        rng = np.random.default_rng(2)
        d = rng.normal(size=(200000, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        values, _ = real_solid_harmonics(d, 3, gradient=False)  # r = 1 => Y_lm
        gram = 4.0 * math.pi * (values.T @ values) / d.shape[0]
        assert np.max(np.abs(gram - np.eye(gram.shape[0]))) < 0.05


class TestRadialBasis:
    def test_polynomial_basis_is_orthonormal(self):
        r_cut, n_max = 5.0, 4
        w = polynomial_radial_basis(n_max, r_cut)
        nodes, weights = np.polynomial.legendre.leggauss(200)
        r = 0.5 * r_cut * (nodes + 1.0)
        wq = 0.5 * r_cut * weights
        powers = np.arange(1, n_max + 1) + 2.0
        phi = np.power((r_cut - r)[None, :], powers[:, None])
        g = w @ phi
        gram = (g * (wq * r**2)[None, :]) @ g.T
        # The primitive (r_cut - r)^(n+2) basis is Hilbert-like: its Gram matrix
        # has condition number ~1e5 at n_max = 4, so the orthonormalised basis
        # is only orthonormal to ~cond * eps, not to eps.  That is a property of
        # the basis, not a bug, and it is why n_max is kept small.
        cond = np.linalg.cond(polynomial_radial_basis(n_max, r_cut))
        assert np.max(np.abs(gram - np.eye(n_max))) < 1e-8, f"cond(W) = {cond:.2e}"

    def test_polynomial_basis_vanishes_at_the_cutoff(self):
        r_cut, n_max = 5.0, 4
        w = polynomial_radial_basis(n_max, r_cut)
        powers = np.arange(1, n_max + 1) + 2.0
        phi = np.power(np.array([0.0]), powers[:, None])  # (r_cut - r) == 0
        assert np.max(np.abs(w @ phi)) == 0.0

    def test_chebyshev_and_legendre_derivatives(self):
        x = np.linspace(-0.999, 0.999, 41)
        h = 1e-6
        t, dt = chebyshev(x, 8)
        num = (chebyshev(x + h, 8)[0] - chebyshev(x - h, 8)[0]) / (2 * h)
        assert np.max(np.abs(num - dt)) < 1e-6
        p, dp = legendre(x, 6)
        num = (legendre(x + h, 6)[0] - legendre(x - h, 6)[0]) / (2 * h)
        assert np.max(np.abs(num - dp)) < 1e-6

    def test_legendre_matches_known_values(self):
        u = np.array([-1.0, -0.5, 0.0, 0.3, 1.0])
        p, _ = legendre(u, 4)
        assert np.allclose(p[:, 0], 1.0)
        assert np.allclose(p[:, 1], u)
        assert np.allclose(p[:, 2], 0.5 * (3 * u**2 - 1))
        assert np.allclose(p[:, 3], 0.5 * (5 * u**3 - 3 * u))
        assert np.allclose(p[:, 4], (35 * u**4 - 30 * u**2 + 3) / 8)
        # P_l(1) == 1 for every l -- the collinear-triple limit.
        assert np.allclose(p[-1], 1.0)


# --------------------------------------------------------------------------
# THE test: analytic derivatives vs central differences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("descriptor_name", list(_descriptors()))
@pytest.mark.parametrize("system_name", list(_systems()))
def test_derivatives_match_central_differences(descriptor_name, system_name):
    """Analytic ``dG/dr`` must match central differences to < 1e-6.

    Run over the full cross product of {SOAP normalised, SOAP unnormalised,
    polynomial basis} x {rattled fcc, rattled diamond, random gas, triclinic}.
    The unnormalised SOAP is included because the normalisation projector
    ``(1 - p p^T)/|p|`` is a separate piece of algebra that can be right or
    wrong independently of everything upstream of it.
    """
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()[system_name]
    analytic = _dense_derivative(descriptor, configuration)
    numerical = _numerical_derivative(descriptor, configuration)
    err = float(np.max(np.abs(analytic - numerical)))
    scale = float(np.max(np.abs(numerical)))
    assert scale > 1e-3, "the finite-difference reference is suspiciously flat"
    assert err < 1e-6, (
        f"{descriptor_name} on {system_name}: max |analytic - numerical| = "
        f"{err:.3e} (feature derivative scale {scale:.3e})"
    )


def test_soap_derivative_needs_the_normalisation_projector():
    """Dropping the projector must actually break the derivative test.

    Guards against the projector being applied but numerically irrelevant --
    if this passes trivially then the test above is not proving what it claims.
    """
    configuration = _systems()["rattled_fcc"]
    normalised = SOAP(**SMALL_SOAP).compute(configuration, derivatives=True)
    raw = SOAP(**SMALL_SOAP, normalize=False).compute(configuration, derivatives=True)

    unit = normalised.features[normalised.pair_i]
    norms = np.linalg.norm(raw.features, axis=1)[raw.pair_i]

    # (a) The projector really is applied: a unit vector has no derivative
    #     component along itself, so p_hat . dp_hat/dx must vanish identically.
    along = np.einsum("pd,pdk->pk", unit, normalised.derivatives)
    assert np.max(np.abs(along)) < 1e-10

    # (b) It is not a negligible correction: the naive "divide by the norm"
    #     derivative differs from the correct one by an O(1) relative amount,
    #     so the finite-difference test above genuinely exercises it.
    naive = raw.derivatives / norms[:, None, None]
    difference = np.max(np.abs(naive - normalised.derivatives))
    scale = np.max(np.abs(normalised.derivatives))
    assert difference > 0.05 * scale, (
        "the normalisation projector contributes negligibly here, so the "
        "derivative test cannot be said to check it"
    )


# --------------------------------------------------------------------------
# invariances
# --------------------------------------------------------------------------


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_exact_rotational_invariance(descriptor_name):
    """100 random SO(3) rotations must leave the descriptor unchanged.

    This is worth more than every other test in the file combined: it is the
    only one that sees a wrong spherical-harmonic convention.
    """
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()["rattled_fcc"]
    reference = descriptor.compute(configuration, derivatives=False).features
    scale = float(np.max(np.abs(reference)))
    rotations = special_ortho_group.rvs(3, size=100, random_state=20240)
    worst = 0.0
    for rotation in rotations:
        rotated = descriptor.compute(
            configuration.rotated(rotation), derivatives=False
        ).features
        worst = max(worst, float(np.max(np.abs(rotated - reference))))
    assert worst < 1e-10 * max(scale, 1.0), (
        f"{descriptor_name} rotational invariance violated by {worst:.3e} "
        f"(feature scale {scale:.3e})"
    )


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_invariance_under_improper_and_identity_rotations(descriptor_name):
    """A rotation by the identity changes nothing; inversion is a separate story.

    SOAP's power spectrum and the Legendre three-body basis are both invariant
    under inversion as well as rotation (they contain no odd-parity invariant),
    which is a real limitation worth recording explicitly rather than
    discovering later: two enantiomeric environments are indistinguishable.
    """
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()["random_gas"]
    reference = descriptor.compute(configuration, derivatives=False).features
    identity = descriptor.compute(
        configuration.rotated(np.eye(3)), derivatives=False
    ).features
    assert np.allclose(identity, reference, atol=0.0, rtol=0.0)

    inverted = configuration.copy()
    inverted.positions = -inverted.positions
    inverted.cell = -inverted.cell
    got = descriptor.compute(inverted, derivatives=False).features
    assert np.max(np.abs(got - reference)) < 1e-10


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_translation_invariance(descriptor_name):
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()["triclinic"]
    reference = descriptor.compute(configuration, derivatives=False).features
    for shift in ([1.3, -2.7, 0.4], [100.0, 0.0, 0.0], [-31.7, 12.2, -8.8]):
        got = descriptor.compute(
            configuration.translated(shift), derivatives=False
        ).features
        assert np.max(np.abs(got - reference)) < 1e-10


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_periodic_image_invariance(descriptor_name):
    """Moving atoms by whole lattice vectors must change nothing.

    The neighbour list reports shifts relative to the caller's own (possibly
    unwrapped) coordinates, so this checks the descriptor consumes them
    correctly rather than assuming wrapped input.
    """
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()["triclinic"]
    reference = descriptor.compute(configuration, derivatives=False).features

    rng = np.random.default_rng(11)
    shifted = configuration.copy()
    images = rng.integers(-2, 3, size=(configuration.n_atoms, 3))
    shifted.positions = shifted.positions + images @ configuration.cell
    got = descriptor.compute(shifted, derivatives=False).features
    assert np.max(np.abs(got - reference)) < 1e-10


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_permutation_invariance(descriptor_name):
    """Relabelling atoms permutes the rows and changes nothing else."""
    descriptor = _descriptors()[descriptor_name]
    configuration = _systems()["random_gas"]
    reference = descriptor.compute(configuration, derivatives=False).features

    rng = np.random.default_rng(12)
    order = rng.permutation(configuration.n_atoms)
    permuted = configuration.copy()
    permuted.positions = permuted.positions[order]
    permuted.species = permuted.species[order]
    permuted.masses = permuted.masses[order]
    got = descriptor.compute(permuted, derivatives=False).features
    assert np.max(np.abs(got - reference[order])) < 1e-10


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_supercell_replication_gives_identical_environments(descriptor_name):
    """Every atom of a perfect crystal has the same environment, in any cell.

    Compares a 1x1x1 fcc cell against a 2x2x2 supercell of it: the descriptor
    must be the same for all atoms in both, which exercises the multi-image
    pair handling (in the small cell an atom sees the same neighbour through
    several images) against the single-image case.
    """
    descriptor = _descriptors()[descriptor_name]
    small = fcc(4.05, "Al", (1, 1, 1))
    large = fcc(4.05, "Al", (2, 2, 2))
    f_small = descriptor.compute(small, derivatives=False).features
    f_large = descriptor.compute(large, derivatives=False).features
    assert np.max(np.abs(f_small - f_small[0])) < 1e-10
    assert np.max(np.abs(f_large - f_large[0])) < 1e-10
    assert np.max(np.abs(f_large[0] - f_small[0])) < 1e-9


# --------------------------------------------------------------------------
# discrimination -- invariance is worthless without it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_rotated_copies_are_identical_but_real_structures_are_not(descriptor_name):
    """Two configurations differing only by a rotation must agree exactly;
    fcc / bcc / diamond / simple cubic must not.

    A descriptor that returned a constant would pass every invariance test in
    this file, so the discrimination half has to be asserted too.
    """
    descriptor = _descriptors()[descriptor_name]
    rotation = special_ortho_group.rvs(3, random_state=99)
    base = rattle(fcc(4.05, "Al", (2, 2, 2)), 0.05, seed=4)
    same = descriptor.compute(base.rotated(rotation), derivatives=False).features
    assert np.max(np.abs(same - descriptor.compute(base, derivatives=False).features)) < 1e-10

    structures = {
        "fcc": fcc(4.05, "Al", (3, 3, 3)),
        "bcc": bcc(3.22, "Al", (3, 3, 3)),
        "diamond": diamond(5.431, "Si", (2, 2, 2)),
        "sc": sc(2.60, "Al", (4, 4, 4)),
    }
    fingerprints = {
        key: descriptor.compute(cfg, derivatives=False).features[0]
        for key, cfg in structures.items()
    }
    keys = list(structures)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            fa, fb = fingerprints[keys[a]], fingerprints[keys[b]]
            relative = np.linalg.norm(fa - fb) / max(np.linalg.norm(fa), 1e-30)
            assert relative > 1e-3, (
                f"{descriptor_name} cannot tell {keys[a]} from {keys[b]} "
                f"(relative distance {relative:.2e})"
            )


# --------------------------------------------------------------------------
# bookkeeping and edge cases
# --------------------------------------------------------------------------


def test_feature_counts_and_labels():
    soap = SOAP()  # documented defaults
    assert soap.n_features == 40  # (l_max+1) * n_max (n_max+1)/2 = 4 * 10
    assert soap.n_features == len(soap.feature_labels())
    assert soap.compute(fcc(4.05, "Al", (1, 1, 1)), derivatives=False).features.shape[1] == 40

    bispectrum = Bispectrum()
    assert bispectrum.n_features == 8 + 4 * 5  # n_radial_2b + n_radial_3b*(l_max+1) = 28
    assert bispectrum.n_features == len(bispectrum.feature_labels())


def test_soap_features_are_unit_norm_when_normalised():
    descriptor = SOAP(**SMALL_SOAP)
    features = descriptor.compute(_systems()["random_gas"], derivatives=False).features
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)

    raw = SOAP(**SMALL_SOAP, normalize=False)
    assert not np.allclose(
        np.linalg.norm(raw.compute(_systems()["random_gas"], derivatives=False).features, axis=1),
        1.0,
    )


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_isolated_atom_gives_zero_features(descriptor_name):
    """An empty environment is a state, not an error, and must not divide by zero."""
    descriptor = _descriptors()[descriptor_name]
    cfg = Configuration(
        positions=np.zeros((1, 3)),
        cell=np.eye(3) * 40.0,
        pbc=True,
        symbols=("Ar",),
        masses=np.array([39.948]),
    )
    out = descriptor.compute(cfg, derivatives=True)
    assert np.all(out.features == 0.0)
    assert np.all(np.isfinite(out.features))
    assert np.all(out.derivatives == 0.0)


@pytest.mark.parametrize("descriptor_name", ["soap", "bispectrum"])
def test_two_atom_environment_has_no_angular_content(descriptor_name):
    """A single neighbour forms no bond angle; the three-body block must vanish.

    For the polynomial basis this is exact (the ``a != b`` sum is empty).  For
    SOAP it is *not* -- a single Gaussian still has ``l > 0`` content -- so only
    the polynomial basis is asserted to be zero, and SOAP is asserted to be
    nonzero, which is the physically correct distinction between the two
    constructions.
    """
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cfg = Configuration(
        positions=positions,
        cell=np.eye(3) * 40.0,
        pbc=True,
        symbols=("Ar",),
        masses=np.full(2, 39.948),
    )
    descriptor = _descriptors()[descriptor_name]
    features = descriptor.compute(cfg, derivatives=False).features
    if descriptor_name == "bispectrum":
        angular = features[:, descriptor.n_radial_2b :]
        assert np.max(np.abs(angular)) < 1e-14
        assert np.max(np.abs(features[:, : descriptor.n_radial_2b])) > 1e-6
    else:
        assert np.max(np.abs(features)) > 1e-6


def test_forces_from_energy_gradient_round_trip():
    """The chain rule through DescriptorOutput must reproduce -dU/dr.

    Uses a linear "model" ``U = sum_i w . G_i`` with random weights, which is
    the exact path :mod:`atomlab.models.linear` takes, so this checks the sparse
    derivative layout and not just the derivative values.
    """
    rng = np.random.default_rng(7)
    configuration = _systems()["rattled_fcc"]
    for descriptor in (SOAP(**SMALL_SOAP), Bispectrum(**SMALL_BISPECTRUM)):
        weights = rng.normal(size=descriptor.n_features)
        out = descriptor.compute(configuration, derivatives=True)
        forces = out.forces_from_energy_gradient(
            np.tile(weights, (configuration.n_atoms, 1))
        )

        delta = 1e-5
        numerical = np.zeros_like(forces)
        for atom in range(configuration.n_atoms):
            for k in range(3):
                plus = configuration.copy()
                plus.positions[atom, k] += delta
                minus = configuration.copy()
                minus.positions[atom, k] -= delta
                e_plus = float(
                    (descriptor.compute(plus, derivatives=False).features @ weights).sum()
                )
                e_minus = float(
                    (descriptor.compute(minus, derivatives=False).features @ weights).sum()
                )
                numerical[atom, k] = -(e_plus - e_minus) / (2.0 * delta)
        err = float(np.max(np.abs(forces - numerical)))
        assert err < 1e-6, f"{descriptor.name} force round trip off by {err:.3e}"


def test_soap_quadrature_is_converged():
    """Increasing the number of quadrature nodes must stop changing the answer.

    The descriptor is *defined* as its quadrature sum, so this is a statement
    about how well the features approximate the ideal SOAP integral -- not
    about correctness of the derivatives, which are exact at any node count.
    """
    configuration = _systems()["rattled_fcc"]
    coarse = SOAP(r_cut=4.2, n_max=3, l_max=3, n_quadrature=24)
    fine = SOAP(r_cut=4.2, n_max=3, l_max=3, n_quadrature=96)
    a = coarse.compute(configuration, derivatives=False).features
    b = fine.compute(configuration, derivatives=False).features
    assert np.max(np.abs(a - b)) < 1e-8


def test_soap_species_weights_change_the_descriptor():
    """A species weight scales that species' contribution to the density."""
    configuration = _systems()["rattled_fcc"]
    two_species = configuration.copy()
    two_species.species = (np.arange(configuration.n_atoms) % 2).astype(np.int32)
    two_species.symbols = ("Ar", "Kr")
    plain = SOAP(**SMALL_SOAP)
    weighted = SOAP(**SMALL_SOAP, species_weights=[1.0, 2.0])
    a = plain.compute(two_species, derivatives=False).features
    b = weighted.compute(two_species, derivatives=False).features
    assert np.max(np.abs(a - b)) > 1e-3
    with pytest.raises(ValueError):
        SOAP(**SMALL_SOAP, species_weights=[1.0]).compute(two_species, derivatives=False)


def test_rejects_coincident_atoms():
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        cell=np.eye(3) * 30.0,
        pbc=True,
        symbols=("Ar",),
        masses=np.full(2, 39.948),
    )
    with pytest.raises(ValueError, match="zero separation"):
        SOAP(**SMALL_SOAP).compute(cfg, derivatives=False)
    with pytest.raises(ValueError, match="zero separation"):
        Bispectrum(**SMALL_BISPECTRUM).compute(cfg, derivatives=False)


def test_constructor_validation():
    for kwargs in (dict(r_cut=0.0), dict(n_max=0), dict(l_max=-1), dict(sigma_atom=-1.0)):
        with pytest.raises(ValueError):
            SOAP(**kwargs)
    for kwargs in (dict(r_cut=-1.0), dict(n_radial_2b=0), dict(l_max=-2)):
        with pytest.raises(ValueError):
            Bispectrum(**kwargs)


# --------------------------------------------------------------------------
# cost, reported rather than merely bounded
# --------------------------------------------------------------------------


def test_timing_and_feature_counts_for_256_atoms(capsys):
    """Report the wall-clock cost of both descriptors on a 256-atom cell.

    The bound is generous: the assertion exists to catch an accidental
    quadratic blow-up, while the printed numbers are the thing worth reading.
    Run with ``pytest -s`` to see them.
    """
    configuration = rattle(fcc(5.26, "Ar", (4, 4, 4)), 0.10, seed=8)
    assert configuration.n_atoms == 256

    report = []
    for descriptor in (SOAP(), Bispectrum()):
        descriptor.compute(configuration, derivatives=True)  # warm caches
        start = time.perf_counter()
        out = descriptor.compute(configuration, derivatives=True)
        elapsed = time.perf_counter() - start
        report.append(
            f"{descriptor.name:12s} n_features={descriptor.n_features:3d} "
            f"n_pairs={out.pair_i.size - 256:6d} "
            f"t(256 atoms, with derivatives)={elapsed:6.3f} s"
        )
        assert elapsed < 20.0
        assert out.features.shape == (256, descriptor.n_features)

    with capsys.disabled():
        print("\n" + "\n".join(report))
