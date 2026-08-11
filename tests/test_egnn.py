"""Tests for the from-scratch E(3)-equivariant network in :mod:`atomlab.models.egnn`.

Equivariance is the entire point of the architecture, so it is tested harder
than anything else here, and always on a **randomly initialised** network.  That
choice is load-bearing: a *trained* network fitted to an invariant target can be
accidentally near-invariant while the architecture is not, so training it first
would hide exactly the bug we are looking for.

The other load-bearing checks, in order of how much damage a failure would do:

1. **Analytic (autograd) forces against central differences**, and the
   **strain-derivative virial against the finite-difference virial on a
   triclinic cell under shear**.  A model whose derivatives are subtly wrong
   trains to a plausible energy error and then produces wrong dynamics -- the
   failure mode this whole repository is about.
2. **Parity.**  The module claims O(3), not SO(3).  An energy that is secretly a
   pseudoscalar passes every rotation test and fails only here.
3. The Clebsch-Gordan tables, verified numerically by checking that a tensor
   product of two equivariant features transforms with the same Wigner matrix as
   the harmonics themselves.

Precision note: the strict ``1e-6`` relative tolerances and the ``1e-5``
derivative tolerances are checked in **float64**.  float32 autograd noise on
this network is of order ``1e-4`` eV/A on forces and ``1e-6`` relative on
energies, which is fine for training and useless for validating a derivative;
``test_float32_equivariance_is_looser`` records the float32 numbers separately
so the difference is documented rather than assumed.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch
from scipy.stats import special_ortho_group

from atomlab.build import fcc, rattle
from atomlab.cell import cell_from_parameters
from atomlab.models.egnn import (
    DEFAULT_TORCH_THREADS,
    EGNN,
    bessel_basis,
    clebsch_gordan,
    polynomial_envelope,
    real_spherical_harmonics,
    tensor_product_paths,
)
from atomlab.potentials.base import check_forces, check_virial
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.types import Configuration, Dataset

N_ROTATIONS = 100

#: A randomly initialised readout produces per-atom energies of order 1 eV but
#: very small gradients, which would make a finite-difference force check pass
#: trivially (comparing ~0 with ~0).  Amplifying the energy scale gives forces
#: of order 1 eV/A -- the magnitude a real fitted model has -- so the derivative
#: tests have something to actually get wrong.
UNTRAINED_SCALE = 50.0


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------


def _triclinic(n_atoms: int = 12, seed: int = 5, cutoff: float = 4.0) -> Configuration:
    """A strongly skewed triclinic cell, comfortably wider than ``2 * cutoff``."""
    cell = cell_from_parameters(10.5, 11.1, 11.8, 74.0, 83.0, 68.0)
    rng = np.random.default_rng(seed)
    cfg = Configuration(
        positions=rng.random((n_atoms, 3)) @ cell,
        cell=cell,
        pbc=True,
        symbols=("Ar",),
    )
    from atomlab.cell import min_cell_width

    assert min_cell_width(cell, cfg.pbc) > 2 * cutoff
    return cfg


def _model(l_max: int = 1, *, dtype: str = "float64", seed: int = 3, cutoff: float = 4.0) -> EGNN:
    m = EGNN(
        cutoff=cutoff,
        l_max=l_max,
        channels=16,
        n_layers=2,
        seed=seed,
        dtype=dtype,
        allow_untrained=True,
    )
    m.set_scaling(0.0, UNTRAINED_SCALE)
    return m


def _rotations(n: int = N_ROTATIONS, seed: int = 11) -> np.ndarray:
    return special_ortho_group.rvs(3, size=n, random_state=seed)


# ==========================================================================
# 1. Spherical harmonics
# ==========================================================================


def test_spherical_harmonics_normalisation_and_form():
    """``sum_m Y_lm^2 = 1`` per degree, and ``l = 1`` is ``(y, z, x)``."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(64, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    y = real_spherical_harmonics(torch.as_tensor(v), 2).numpy()
    for l in range(3):
        block = y[:, l * l : (l + 1) ** 2]
        assert np.allclose((block**2).sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(y[:, 1:4], v[:, [1, 2, 0]], atol=1e-14)
    assert np.allclose(y[:, 0], 1.0, atol=1e-14)


def test_spherical_harmonics_orthogonality_on_the_sphere():
    """``<Y_lm, Y_l'm'> = delta / (2l+1)`` under the chosen normalisation."""
    n = 4000
    rng = np.random.default_rng(1)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    y = real_spherical_harmonics(torch.as_tensor(v), 2).numpy()
    gram = y.T @ y / n
    expect = np.diag(
        np.concatenate([np.full(2 * l + 1, 1.0 / (2 * l + 1)) for l in range(3)])
    )
    assert np.max(np.abs(gram - expect)) < 0.05  # Monte-Carlo, 1/sqrt(n) noise


def test_spherical_harmonics_reject_unsupported_degree():
    with pytest.raises(ValueError):
        real_spherical_harmonics(torch.zeros(1, 3), 3)


def _wigner_from_harmonics(R: np.ndarray, l: int, seed: int = 0) -> np.ndarray:
    """Extract the real Wigner matrix ``D^l(R)`` from the harmonics themselves.

    Solves ``Y_l(R n_k) = D^l(R) Y_l(n_k)`` in the least-squares sense over a
    set of probe directions.  If ``Y_l`` did *not* transform as a
    ``(2l+1)``-dimensional representation, no such matrix would exist and the
    residual on **fresh** directions (checked by the caller) would be large --
    which is what makes the CG test below non-circular.
    """
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(40, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    a = real_spherical_harmonics(torch.as_tensor(n), 2).numpy()[:, l * l : (l + 1) ** 2]
    b = real_spherical_harmonics(torch.as_tensor(n @ R.T), 2).numpy()[
        :, l * l : (l + 1) ** 2
    ]
    # b = a @ D^T  =>  D^T = lstsq(a, b)
    dt, *_ = np.linalg.lstsq(a, b, rcond=None)
    return dt.T


def test_harmonics_transform_as_irreps():
    """The extracted Wigner matrices are orthogonal and work on unseen directions."""
    rng = np.random.default_rng(2)
    for R in _rotations(10, seed=4):
        for l in (1, 2):
            d = _wigner_from_harmonics(R, l)
            assert np.max(np.abs(d @ d.T - np.eye(2 * l + 1))) < 1e-10
            fresh = rng.normal(size=(7, 3))
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            y = real_spherical_harmonics(torch.as_tensor(fresh), 2).numpy()[
                :, l * l : (l + 1) ** 2
            ]
            y_rot = real_spherical_harmonics(torch.as_tensor(fresh @ R.T), 2).numpy()[
                :, l * l : (l + 1) ** 2
            ]
            assert np.max(np.abs(y_rot - y @ d.T)) < 1e-12


# ==========================================================================
# 2. Clebsch-Gordan coefficients
# ==========================================================================


def test_paths_obey_triangle_and_parity_selection_rules():
    paths = tensor_product_paths(2, 2, 2)
    assert (1, 1, 1) not in paths, "the l=1 x l=1 -> l=1 cross product is parity-odd"
    assert (2, 2, 1) not in paths
    for l1, l2, l3 in paths:
        assert abs(l1 - l2) <= l3 <= l1 + l2
        assert (l1 + l2 + l3) % 2 == 0
    assert tensor_product_paths(1, 1, 1) == [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]


def test_parity_forbidden_gaunt_coefficients_vanish():
    """The quadrature must return an exact zero, not dust, for odd triples."""
    for triple in [(1, 1, 1), (2, 2, 1), (1, 2, 2), (0, 1, 0)]:
        assert np.all(clebsch_gordan(*triple) == 0.0)


def test_cg_identity_paths():
    """``0 (x) l -> l`` and ``l (x) l -> 0`` are proportional to the identity."""
    for l in (1, 2):
        g = clebsch_gordan(0, l, l)[0]
        assert np.max(np.abs(g - np.eye(2 * l + 1) / math.sqrt(2 * l + 1))) < 1e-12
        g = clebsch_gordan(l, l, 0)[:, :, 0]
        assert np.max(np.abs(g - np.eye(2 * l + 1) / math.sqrt(2 * l + 1))) < 1e-12


@pytest.mark.parametrize("path", tensor_product_paths(2, 2, 2))
def test_clebsch_gordan_is_an_equivariant_intertwiner(path):
    """THE CG verification: the tensor product transforms like an ``l3`` object.

    Build two genuine equivariant features ``u = Y_l1(a)``, ``v = Y_l2(b)`` from
    random directions, contract them with the CG tensor, and check that rotating
    the *inputs* is the same as applying ``D^{l3}(R)`` to the *output*.  The
    Wigner matrix is obtained independently from the harmonics (and separately
    validated in ``test_harmonics_transform_as_irreps``), so a wrong CG table --
    a transposed index, a wrong ``m`` order, a missing sign -- fails here.
    """
    l1, l2, l3 = path
    cg = clebsch_gordan(l1, l2, l3)
    rng = np.random.default_rng(10 * l1 + 3 * l2 + l3)
    a = rng.normal(size=(6, 3))
    b = rng.normal(size=(6, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    def feats(v, l):
        return real_spherical_harmonics(torch.as_tensor(v), 2).numpy()[
            :, l * l : (l + 1) ** 2
        ]

    w = np.einsum("pm,pn,mno->po", feats(a, l1), feats(b, l2), cg)
    worst = 0.0
    for R in _rotations(20, seed=23):
        w_rot = np.einsum(
            "pm,pn,mno->po", feats(a @ R.T, l1), feats(b @ R.T, l2), cg
        )
        d3 = _wigner_from_harmonics(R, l3)
        worst = max(worst, float(np.max(np.abs(w_rot - w @ d3.T))))
    scale = max(float(np.max(np.abs(w))), 1e-12)
    assert worst / scale < 1e-10, f"path {path}: relative error {worst / scale:.2e}"


# ==========================================================================
# 3. Radial basis and cutoff
# ==========================================================================


def test_envelope_is_c2_at_the_cutoff():
    rc = 4.0
    r = torch.tensor([rc - 1e-9], dtype=torch.float64, requires_grad=True)
    u = polynomial_envelope(r, rc)
    (du,) = torch.autograd.grad(u.sum(), r, create_graph=True)
    (d2u,) = torch.autograd.grad(du.sum(), r)
    assert abs(float(u)) < 1e-12
    assert abs(float(du)) < 1e-8
    assert abs(float(d2u)) < 1e-4
    assert float(polynomial_envelope(torch.tensor([rc + 0.1]), rc)) == 0.0
    assert abs(float(polynomial_envelope(torch.tensor([1e-6], dtype=torch.float64), rc)) - 1.0) < 1e-12


def test_bessel_basis_vanishes_at_the_cutoff():
    rc = 4.0
    b = bessel_basis(torch.tensor([rc], dtype=torch.float64), rc, 8)
    assert np.max(np.abs(b.numpy())) < 1e-12


def test_messages_vanish_continuously_at_the_cutoff():
    """Energy must be smooth as a pair crosses ``rc`` -- a dimer scan.

    A bare truncation would show a step here; a ``C^0`` cutoff would show a kink
    in the force.  Both are invisible to any static error metric and both ruin
    energy conservation in MD.
    """
    m = _model(cutoff=4.0)
    energies = []
    for r in np.linspace(3.90, 4.10, 41):
        cfg = Configuration(
            positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            cell=np.eye(3) * 40.0,
            pbc=False,
            symbols=("Ar",),
        )
        energies.append(m.energy(cfg))
    e = np.array(energies)
    # Third differences: a C^2 function sampled on a uniform grid has third
    # differences of order h^3 * f''' -- tiny.  A kink would blow this up.
    d3 = np.diff(e, 3)
    assert np.max(np.abs(d3)) < 1e-6, f"max |third difference| = {np.max(np.abs(d3)):.2e}"
    assert np.all(np.abs(e[-6:] - e[-1]) < 1e-12), "energy must be flat beyond rc"


# ==========================================================================
# 4. Equivariance of the untrained network -- the core of the file
# ==========================================================================


@pytest.mark.parametrize("l_max", [1, 2])
def test_energy_is_invariant_under_100_random_rotations(l_max):
    cfg = _triclinic()
    m = _model(l_max)
    e0 = m.energy(cfg)
    assert abs(e0) > 1e-3, "degenerate test: the untrained energy is ~0"
    worst = max(abs(m.energy(cfg.rotated(R)) - e0) for R in _rotations())
    assert worst / abs(e0) < 1e-6, f"relative energy drift {worst / abs(e0):.3e}"


@pytest.mark.parametrize("l_max", [1, 2])
def test_forces_are_equivariant_under_100_random_rotations(l_max):
    cfg = _triclinic()
    m = _model(l_max)
    f0 = m.forces(cfg)
    scale = np.max(np.abs(f0))
    assert scale > 1e-3, "degenerate test: the untrained forces are ~0"
    worst = 0.0
    for R in _rotations():
        # A rotation acts on row vectors as f -> f R^T.
        worst = max(worst, float(np.max(np.abs(m.forces(cfg.rotated(R)) - f0 @ R.T))))
    assert worst / scale < 1e-6, f"relative force drift {worst / scale:.3e}"


@pytest.mark.parametrize("l_max", [1, 2])
def test_energy_is_invariant_under_reflection_o3_not_just_so3(l_max):
    """The module claims **O(3)**: the energy is a true scalar, not a pseudoscalar.

    A network that admitted the parity-odd ``1 x 1 -> 1`` path could learn an
    energy that changes sign (or simply changes) under inversion.  That is
    invisible to every proper-rotation test above and is a classic bug, so it is
    tested separately with (a) pure inversion and (b) general improper rotations
    ``-R``, which additionally rule out an accidental cancellation special to
    inversion.
    """
    cfg = _triclinic()
    m = _model(l_max)
    e0 = m.energy(cfg)
    f0 = m.forces(cfg)
    fscale = np.max(np.abs(f0))

    parity = -np.eye(3)
    assert abs(m.energy(cfg.rotated(parity)) - e0) / abs(e0) < 1e-6
    # Forces are polar vectors: F(-x) = -F(x).
    assert np.max(np.abs(m.forces(cfg.rotated(parity)) + f0)) / fscale < 1e-6

    worst_e, worst_f = 0.0, 0.0
    for R in _rotations(20, seed=31):
        improper = -R  # det = -1
        assert np.linalg.det(improper) < 0
        worst_e = max(worst_e, abs(m.energy(cfg.rotated(improper)) - e0))
        worst_f = max(
            worst_f,
            float(np.max(np.abs(m.forces(cfg.rotated(improper)) - f0 @ improper.T))),
        )
    assert worst_e / abs(e0) < 1e-6
    assert worst_f / fscale < 1e-6


def test_translation_invariance():
    cfg = _triclinic()
    m = _model()
    e0, f0 = m.energy(cfg), m.forces(cfg)
    rng = np.random.default_rng(3)
    for _ in range(10):
        shift = rng.normal(scale=5.0, size=3)
        moved = cfg.translated(shift)
        assert abs(m.energy(moved) - e0) / abs(e0) < 1e-10
        assert np.max(np.abs(m.forces(moved) - f0)) / np.max(np.abs(f0)) < 1e-10


def test_periodic_image_invariance():
    """Translating one atom by a lattice vector changes nothing."""
    cfg = _triclinic()
    m = _model()
    e0, f0 = m.energy(cfg), m.forces(cfg)
    moved = cfg.copy()
    moved.positions[3] += cfg.cell[1] - 2.0 * cfg.cell[2]
    assert abs(m.energy(moved) - e0) / abs(e0) < 1e-10
    assert np.max(np.abs(m.forces(moved) - f0)) / np.max(np.abs(f0)) < 1e-10


def test_permutation_invariance():
    cfg = _triclinic()
    m = _model()
    e0, f0 = m.energy(cfg), m.forces(cfg)
    perm = np.random.default_rng(9).permutation(cfg.n_atoms)
    permuted = cfg.copy()
    permuted.positions = cfg.positions[perm]
    permuted.species = cfg.species[perm]
    permuted.masses = cfg.masses[perm]
    assert abs(m.energy(permuted) - e0) / abs(e0) < 1e-10
    assert np.max(np.abs(m.forces(permuted) - f0[perm])) / np.max(np.abs(f0)) < 1e-10


def test_float32_equivariance_is_looser_but_still_sound():
    """Document the float32 numbers rather than assume them.

    float32 is what training runs in; it is ~2x faster and its equivariance
    error (~1e-6 relative) is far below any fitting error.  It is simply not
    precise enough to *validate* the architecture, which is why every strict
    test above uses float64.
    """
    cfg = _triclinic()
    m32 = EGNN(cutoff=4.0, seed=3, dtype="float32", allow_untrained=True)
    m32.set_scaling(0.0, UNTRAINED_SCALE)
    e0 = m32.energy(cfg)
    worst = max(abs(m32.energy(cfg.rotated(R)) - e0) for R in _rotations(25, seed=17))
    rel = worst / abs(e0)
    print(f"\nfloat32 rotational energy drift: {rel:.3e} relative")
    assert rel < 1e-4


# ==========================================================================
# 5. Derivatives -- the non-negotiable test
# ==========================================================================


@pytest.mark.parametrize("l_max", [1, 2])
def test_autograd_forces_match_finite_differences(l_max):
    """float64: analytic vs central-difference forces on a triclinic cell."""
    cfg = _triclinic(n_atoms=8, seed=7)
    m = _model(l_max)
    f = m.forces(cfg)
    err = check_forces(m, cfg, delta=1e-5)
    print(
        f"\nl_max={l_max}: max |F_analytic - F_numeric| = {err:.3e} eV/A "
        f"(|F|max = {np.max(np.abs(f)):.3f} eV/A)"
    )
    assert np.max(np.abs(f)) > 0.1, "degenerate test: forces are ~0"
    assert err < 1e-5


@pytest.mark.parametrize("l_max", [1, 2])
def test_strain_virial_matches_finite_differences_on_a_sheared_triclinic_cell(l_max):
    """The strain-derivative virial, on the hardest cell available.

    A triclinic cell under an additional shear is where a hand-assembled virial
    that forgets a periodic-image term, or that assumes an orthorhombic cell,
    goes wrong.  The strain trick cannot make that mistake by construction --
    which is the point of using it -- but that claim still has to be checked.
    """
    from atomlab.build import shear

    cfg = shear(_triclinic(n_atoms=8, seed=7), 0.08, "xz")
    m = _model(l_max)
    w = m.virial(cfg)
    err = check_virial(m, cfg, delta=1e-6)
    print(
        f"\nl_max={l_max}: max |W_analytic - W_numeric| = {err:.3e} eV "
        f"(|W|max = {np.max(np.abs(w)):.3f} eV)"
    )
    assert np.max(np.abs(w)) > 0.1, "degenerate test: the virial is ~0"
    assert err < 1e-5
    # The strain variable is symmetrised, so the virial is symmetric exactly.
    assert np.max(np.abs(w - w.T)) == 0.0


def test_virial_is_symmetric_and_consistent_with_pressure():
    cfg = _triclinic(n_atoms=10, seed=2)
    m = _model()
    res = m.compute(cfg)
    assert np.allclose(res.virial, res.virial.T, atol=0.0)
    assert abs(res.pressure(cfg.volume) - m.pressure(cfg)) < 1e-12


def test_forces_sum_to_zero():
    """Translation invariance implies Newton's third law in aggregate."""
    cfg = _triclinic(n_atoms=12, seed=4)
    f = _model().forces(cfg)
    assert np.max(np.abs(f.sum(axis=0))) < 1e-10 * np.max(np.abs(f))


# ==========================================================================
# 6. Contract / plumbing
# ==========================================================================


def test_unfitted_model_refuses_to_predict():
    m = EGNN(cutoff=4.0, dtype="float64")
    with pytest.raises(RuntimeError, match="has not been fitted"):
        m.energy(_triclinic())


def test_cutoff_larger_than_half_the_cell_raises():
    cfg = fcc(5.3, "Ar", (1, 1, 1))  # 5.3 A cell, half width 2.65 A
    m = _model(cutoff=4.0)
    with pytest.raises(ValueError, match="minimum image"):
        m.energy(cfg)


def test_compute_does_not_mutate_the_configuration():
    cfg = _triclinic()
    before = cfg.positions.copy(), cfg.cell.copy()
    _model().compute(cfg)
    assert np.array_equal(cfg.positions, before[0])
    assert np.array_equal(cfg.cell, before[1])


def test_seeding_is_deterministic_and_does_not_leak():
    cfg = _triclinic()
    torch.manual_seed(12345)
    before = torch.rand(3)
    a = _model(seed=42).energy(cfg)
    torch.manual_seed(12345)
    _ = torch.rand(3)
    b = _model(seed=42).energy(cfg)
    c = _model(seed=43).energy(cfg)
    assert a == b
    assert a != c
    # The global stream must be exactly where it was left.
    torch.manual_seed(12345)
    assert torch.equal(before, torch.rand(3))


def test_parameter_count_is_small_enough_to_train_here():
    m = EGNN(cutoff=4.0, allow_untrained=True)
    n = m.n_parameters
    print(f"\nEGNN(l_max=1, C=16, 2 layers) parameters: {n}")
    assert 1000 < n < 30000
    assert EGNN(cutoff=4.0, l_max=2, allow_untrained=True).n_parameters < 60000
    assert DEFAULT_TORCH_THREADS == 1


def test_energies_decompose_into_per_atom_contributions():
    cfg = _triclinic()
    res = _model().compute(cfg)
    assert res.energies.shape == (cfg.n_atoms,)
    assert abs(res.energies.sum() - res.energy) < 1e-9


def test_open_boundary_system_works():
    rng = np.random.default_rng(6)
    cfg = Configuration(
        positions=rng.normal(scale=2.0, size=(9, 3)),
        cell=np.eye(3) * 100.0,
        pbc=False,
        symbols=("Ar",),
    )
    m = _model()
    assert check_forces(m, cfg, delta=1e-5) < 1e-5


# ==========================================================================
# 7. Timing and a real (small) training run
# ==========================================================================


def _lj_dataset(n_configs: int, reps=(2, 2, 2), sigma: float = 0.12, seed: int = 0):
    """Rattled fcc argon labelled by a smoothly switched Lennard-Jones."""
    ref = LennardJones.argon()
    lj = LennardJones(
        epsilon=float(ref.epsilon[0, 0]),
        sigma=float(ref.sigma[0, 0]),
        cutoff=5.0,
        mode="switched",
        r_on=4.0,
    )
    base = fcc(5.30, "Ar", reps)
    rng = np.random.default_rng(seed)
    cfgs = []
    for _ in range(n_configs):
        c = rattle(base, sigma, seed=int(rng.integers(1 << 30)))
        cfgs.append(c.with_labels(lj.compute(c)))
    return Dataset(cfgs), lj


def test_timing_for_a_64_atom_configuration():
    """Report forward+backward wall time; the assertion is only a sanity bound."""
    cfg = rattle(fcc(5.30, "Ar", (2, 2, 2)).repeated((2, 1, 1)), 0.12, seed=1)
    assert cfg.n_atoms == 64
    m = EGNN(cutoff=5.0, allow_untrained=True, dtype="float32")
    m.compute(cfg)  # warm up
    t0 = time.perf_counter()
    for _ in range(20):
        m.compute(cfg, forces=True, virial=True)
    dt = (time.perf_counter() - t0) / 20
    print(f"\n64-atom forward+backward (E+F+W, float32): {dt * 1e3:.1f} ms")
    assert dt < 0.5


@pytest.mark.slow
def test_short_training_run_reaches_useful_accuracy():
    """A genuine, if modest, fit: 120 train / 40 val 32-atom LJ configurations.

    The target here is not state of the art, it is *evidence that the model
    learns at all* within a budget compatible with the rest of the study.  The
    numbers printed are the ones quoted in the report.
    """
    ds, lj = _lj_dataset(160, seed=0)
    train, val = ds.split([0.75, 0.25], seed=0)
    e_std = float(np.std(ds.energies_per_atom()))
    f_rms = float(np.sqrt((ds.forces() ** 2).mean()))

    m = EGNN(cutoff=5.0, seed=1, dtype="float32", allow_untrained=True)
    t0 = time.perf_counter()
    report = m.fit(train, val=val, epochs=60, batch_size=4, learning_rate=0.01)
    wall = time.perf_counter() - t0

    print(
        f"\ntraining: {report.n_epochs} epochs in {wall:.1f} s "
        f"({m.n_parameters} parameters)\n"
        f"  target spread: E/atom std = {e_std * 1e3:.3f} meV/atom, "
        f"F rms = {f_rms:.4f} eV/A\n"
        f"  train: E {report.train_energy_rmse * 1e3:.3f} meV/atom, "
        f"F {report.train_force_rmse:.4f} eV/A\n"
        f"  val:   E {report.val_energy_rmse * 1e3:.3f} meV/atom, "
        f"F {report.val_force_rmse:.4f} eV/A\n"
        f"  notes: {report.notes}"
    )
    # "Useful" here means a large fraction of the target variance is explained.
    assert report.val_force_rmse < 0.35 * f_rms
    assert report.val_energy_rmse < 0.5 * e_std
    assert report.n_parameters == m.n_parameters

    # A fitted model must still be a valid Potential: check the derivative of
    # the *trained* weights too, since training moves the radial MLP into a
    # regime the random init never visits.
    m64 = EGNN(cutoff=5.0, seed=1, dtype="float64", allow_untrained=True)
    m64.net.load_state_dict(m.net.state_dict())
    for layer64, layer32 in zip(m64.net.layers, m.net.layers):
        layer64.avg_neighbors = layer32.avg_neighbors
    m64.net.to(torch.float64)
    small = rattle(fcc(5.30, "Ar", (2, 2, 2)), 0.12, seed=99)
    err_f = check_forces(m64, small, delta=1e-5, atoms=range(4))
    print(f"  trained-model force check: {err_f:.3e} eV/A")
    assert err_f < 1e-5


@pytest.mark.slow
def test_fit_is_reproducible_given_a_seed():
    ds, _ = _lj_dataset(24, seed=1)
    out = []
    for _ in range(2):
        m = EGNN(cutoff=5.0, seed=5, dtype="float32", allow_untrained=True)
        rep = m.fit(ds, epochs=4, batch_size=4, seed=7)
        out.append(rep.train_force_rmse)
    assert out[0] == out[1]
