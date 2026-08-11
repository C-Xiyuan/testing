"""Tests for the Behler-Parrinello neural network in :mod:`atomlab.models.bpnn`.

The load-bearing test in this file is the first one: **analytic forces against
central differences on a randomly initialised network**.  It is deliberately
run before any training, because a trained network fitted to a smooth target
has small, well-behaved gradients and can hide a chain-rule bug that an
untrained one exposes immediately.  For the same reason the untrained network's
energy scale is amplified (:data:`UNTRAINED_SCALE`) so that its forces are of
order 1 eV/A -- the magnitude a fitted model has -- rather than the ~0.03 eV/A a
freshly initialised readout produces.  Comparing ~0 with ~0 passes any
tolerance and tests nothing.

The failure this guards against is specific and is the subject of the whole
repository: forces assembled by the chain rule
``dE/dr = (dE/dG) (dG/dr)`` can be wrong by a constant factor, by a dropped
centre-atom term, or by a missing ``1/sd`` from the feature standardisation,
and every one of those still trains to a plausible energy error and then
produces wrong physics.  The virial has the same property one derivative
further out, so it is checked against ``numerical_virial`` on an orthorhombic
*and* on a sheared triclinic cell -- an off-diagonal error is invisible in a
cubic box.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from scipy.stats import special_ortho_group

from atomlab.build import fcc, rattle
from atomlab.cell import cell_from_parameters, min_cell_width
from atomlab.models.bpnn import BPNN, DEFAULT_TORCH_THREADS
from atomlab.models.descriptors.acsf import ACSF
from atomlab.potentials.base import check_forces, check_virial
from atomlab.potentials.lennard_jones import LennardJones
from atomlab.types import Configuration, Dataset

#: Descriptor cutoff, in angstrom.  Chosen so that the 2x2x2 fcc argon cell
#: used throughout (a = 5.4 A, so 10.8 A wide) still satisfies the
#: minimum-image requirement ``2 * cutoff < min_cell_width`` that the model
#: enforces.  It covers the first two neighbour shells of fcc argon.
CUTOFF = 5.0

#: fcc lattice parameter for argon, in angstrom.  ``a / sqrt(2) = 3.82 A`` is
#: the Lennard-Jones minimum ``2^(1/6) sigma``, so the perfect crystal sits at
#: the bottom of the pair well and rattling it produces forces of both signs.
A_FCC = 5.4

#: Amplification of the untrained readout; see the module docstring.
UNTRAINED_SCALE = 50.0

#: Tolerance on analytic-vs-numerical derivatives, eV/A and eV.  This is the
#: number ``docs/design.md`` section 5.5 requires of every model.
DERIV_TOL = 1e-6


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _descriptor(species=(0,)) -> ACSF:
    return ACSF.default(species, CUTOFF)


def _untrained(seed: int = 3, species=(0,), **kw) -> BPNN:
    """A randomly initialised model with forces of a realistic magnitude."""
    model = BPNN(_descriptor(species), seed=seed, allow_untrained=True, **kw)
    model.set_scaling(np.zeros(model.n_species), UNTRAINED_SCALE)
    return model


def _argon(sigma: float = 0.12, seed: int = 1, reps=(2, 2, 2)) -> Configuration:
    return rattle(fcc(A_FCC, "Ar", reps), sigma, seed=seed)


def _triclinic(n_atoms: int = 16, seed: int = 5) -> Configuration:
    """A strongly sheared cell, still comfortably wider than ``2 * CUTOFF``."""
    cell = cell_from_parameters(11.5, 12.2, 12.8, 74.0, 83.0, 68.0)
    rng = np.random.default_rng(seed)
    cfg = Configuration(
        positions=rng.random((n_atoms, 3)) @ cell,
        cell=cell,
        pbc=True,
        symbols=("Ar",),
    )
    assert min_cell_width(cell, cfg.pbc) > 2 * CUTOFF
    return cfg


def _binary(n_atoms: int = 16, seed: int = 7) -> Configuration:
    """A two-species cell, exercising the per-species MLP dispatch."""
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 11.0
    return Configuration(
        positions=rng.random((n_atoms, 3)) @ cell,
        cell=cell,
        pbc=True,
        species=np.array([i % 2 for i in range(n_atoms)], dtype=np.int32),
        symbols=("Ar", "Kr"),
    )


def _argon_dataset(n: int, *, seed: int, sigma_range=(0.05, 0.20)) -> Dataset:
    """``n`` rattled fcc argon cells labelled with :class:`LennardJones`.

    The rattle amplitude is drawn per configuration so the set spans a range of
    force magnitudes rather than a single displacement shell; a model fitted to
    one amplitude only is not being asked anything.
    """
    lj = LennardJones.argon(cutoff=CUTOFF)
    base = fcc(A_FCC, "Ar", (2, 2, 2))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sigma = float(rng.uniform(*sigma_range))
        cfg = rattle(base, sigma, seed=int(rng.integers(1 << 30)))
        out.append(cfg.with_labels(lj.compute(cfg)))
    return Dataset(out)


# --------------------------------------------------------------------------
# 1. derivatives -- the tests that must pass before anything else matters
# --------------------------------------------------------------------------


def test_analytic_forces_match_finite_differences_untrained():
    """dE/dr through the descriptor chain rule, on a random network."""
    model = _untrained()
    cfg = _argon()
    forces = model.forces(cfg)
    # Guard against a vacuous pass: the check is only meaningful if the model
    # actually produces forces of a physically relevant size.
    assert np.abs(forces).max() > 0.1, "untrained forces too small to test anything"
    err = check_forces(model, cfg, atoms=[0, 5, 17, 31])
    assert err < DERIV_TOL, f"analytic vs numerical forces differ by {err:.3e} eV/A"


def test_analytic_virial_matches_finite_differences_untrained():
    """The pair-outer-product virial against -dU/d(strain)."""
    model = _untrained()
    cfg = _argon()
    err = check_virial(model, cfg)
    assert np.abs(model.virial(cfg)).max() > 0.1
    assert err < DERIV_TOL, f"analytic vs numerical virial differ by {err:.3e} eV"


def test_virial_matches_finite_differences_on_a_sheared_triclinic_cell():
    """Off-diagonal virial components are invisible in a cubic box."""
    model = _untrained(seed=11)
    cfg = _triclinic()
    err = check_virial(model, cfg)
    assert err < DERIV_TOL, f"triclinic virial error {err:.3e} eV"


def test_forces_and_virial_on_an_open_boundary_cluster():
    """No periodic images at all: the same chain rule must still close."""
    rng = np.random.default_rng(2)
    cfg = Configuration(
        positions=rng.normal(0.0, 2.0, size=(10, 3)),
        cell=None,
        pbc=False,
        symbols=("Ar",),
    )
    model = _untrained(seed=4)
    assert check_forces(model, cfg) < DERIV_TOL
    assert check_virial(model, cfg) < DERIV_TOL


def test_derivatives_for_a_two_species_model():
    """The per-species MLP dispatch must not scramble the chain rule."""
    model = _untrained(seed=6, species=(0, 1))
    cfg = _binary()
    assert model.n_species == 2
    assert check_forces(model, cfg, atoms=[0, 1, 9]) < DERIV_TOL
    assert check_virial(model, cfg) < DERIV_TOL


def test_descriptor_derivatives_match_central_differences():
    """Check the descriptor's own dG/dr before trusting anything built on it.

    The BPNN chain rule is only as good as the array it contracts against, and
    a failure here versus a failure in the force test above mean completely
    different things -- descriptor bug versus assembly bug.  ``tests/test_acsf``
    owns the descriptor, but the model has to interpret its ``(P, D, 3)``
    layout, so the layout is re-derived here from first principles rather than
    assumed: summing the rows with ``pair_j == j`` and ``pair_i == i`` must give
    ``dG[i]/dr[j]``.
    """
    desc = _descriptor()
    cfg = _argon(sigma=0.1, seed=12)
    out = desc.compute(cfg, derivatives=True)
    pair_i = np.asarray(out.pair_i)
    pair_j = np.asarray(out.pair_j)

    delta = 1e-5
    worst = 0.0
    rng = np.random.default_rng(0)
    for j in rng.choice(cfg.n_atoms, size=3, replace=False):
        for k in range(3):
            plus, minus = cfg.copy(), cfg.copy()
            plus.positions[j, k] += delta
            minus.positions[j, k] -= delta
            fd = (
                desc.compute(plus, derivatives=False).features
                - desc.compute(minus, derivatives=False).features
            ) / (2.0 * delta)                                   # (N, D)
            analytic = np.zeros_like(fd)
            rows = np.flatnonzero(pair_j == j)
            np.add.at(analytic, pair_i[rows], out.derivatives[rows, :, k])
            worst = max(worst, float(np.abs(analytic - fd).max()))
    assert worst < 1e-6, f"descriptor derivatives differ from finite differences by {worst:.3e}"


def test_forces_sum_to_zero_and_virial_is_symmetric():
    """Newton's third law and the absence of a net torque, respectively."""
    model = _untrained(seed=8)
    result = model.compute(_argon())
    assert np.abs(result.forces.sum(axis=0)).max() < 1e-11
    assert np.abs(result.virial - result.virial.T).max() < 1e-11


def test_trained_model_derivatives_are_also_exact():
    """A fit changes the weights, not the chain rule -- but check anyway."""
    model, _ = _quick_fit(n=24, epochs=12, seed=2)
    cfg = _argon(sigma=0.15, seed=99)
    assert check_forces(model, cfg, atoms=[0, 7, 23]) < DERIV_TOL
    assert check_virial(model, cfg) < DERIV_TOL


# --------------------------------------------------------------------------
# 2. symmetries
# --------------------------------------------------------------------------


def test_energy_is_invariant_under_translation():
    model = _untrained()
    cfg = _argon()
    e0 = model.energy(cfg)
    f0 = model.forces(cfg)
    moved = cfg.translated(np.array([3.1, -7.7, 12.4]))
    assert abs(model.energy(moved) - e0) < 1e-10 * max(abs(e0), 1.0)
    assert np.abs(model.forces(moved) - f0).max() < 1e-10


def test_energy_is_invariant_and_forces_equivariant_under_rotation():
    model = _untrained()
    cfg = _argon()
    e0 = model.energy(cfg)
    f0 = model.forces(cfg)
    rotations = special_ortho_group.rvs(3, size=20, random_state=0)
    worst_e, worst_f = 0.0, 0.0
    for rot in rotations:
        turned = cfg.rotated(rot)
        worst_e = max(worst_e, abs(model.energy(turned) - e0) / max(abs(e0), 1.0))
        worst_f = max(worst_f, np.abs(model.forces(turned) - f0 @ rot.T).max())
    assert worst_e < 1e-10, f"rotation changed the energy by {worst_e:.2e} (relative)"
    assert worst_f < 1e-9, f"forces are not equivariant: {worst_f:.2e} eV/A"


def test_energy_is_invariant_under_permutation_and_forces_follow():
    model = _untrained()
    cfg = _argon()
    e0 = model.energy(cfg)
    f0 = model.forces(cfg)
    perm = np.random.default_rng(4).permutation(cfg.n_atoms)
    shuffled = cfg.copy()
    shuffled.positions = cfg.positions[perm]
    shuffled.species = cfg.species[perm]
    shuffled.masses = cfg.masses[perm]
    assert abs(model.energy(shuffled) - e0) < 1e-10 * max(abs(e0), 1.0)
    assert np.abs(model.forces(shuffled) - f0[perm]).max() < 1e-10


def test_energy_is_invariant_under_the_choice_of_periodic_image():
    """Moving one atom by a lattice vector is not a physical change."""
    model = _untrained()
    cfg = _argon()
    e0 = model.energy(cfg)
    f0 = model.forces(cfg)
    moved = cfg.copy()
    moved.positions[3] += cfg.cell[1] - 2.0 * cfg.cell[2]
    assert abs(model.energy(moved) - e0) < 1e-10 * max(abs(e0), 1.0)
    assert np.abs(model.forces(moved) - f0).max() < 1e-10


# --------------------------------------------------------------------------
# 3. contract / hygiene
# --------------------------------------------------------------------------


def test_unfitted_model_refuses_to_predict():
    model = BPNN(_descriptor(), seed=0)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.energy(_argon())


def test_cell_narrower_than_twice_the_cutoff_raises():
    """The virial assembly needs one image per pair; a narrow cell breaks it."""
    model = _untrained()
    tiny = rattle(fcc(A_FCC, "Ar", (1, 1, 1)), 0.05, seed=0)  # 5.4 A cell
    with pytest.raises(ValueError):
        model.energy(tiny)


def test_compute_does_not_mutate_the_configuration():
    model = _untrained()
    cfg = _argon()
    before = (cfg.positions.copy(), cfg.cell.copy(), cfg.species.copy())
    model.compute(cfg)
    assert np.array_equal(cfg.positions, before[0])
    assert np.array_equal(cfg.cell, before[1])
    assert np.array_equal(cfg.species, before[2])


def test_per_atom_energies_sum_to_the_total():
    model = _untrained()
    result = model.compute(_argon())
    assert result.energies is not None
    assert abs(result.energies.sum() - result.energy) < 1e-10 * max(abs(result.energy), 1.0)


def test_seeding_is_deterministic_and_does_not_leak():
    cfg = _argon()
    a = _untrained(seed=17)
    b = _untrained(seed=17)
    c = _untrained(seed=18)
    assert a.energy(cfg) == b.energy(cfg)
    assert a.energy(cfg) != c.energy(cfg)

    # Constructing a model must not disturb the global torch RNG stream.
    torch.manual_seed(123)
    first = torch.randn(4)
    torch.manual_seed(123)
    _untrained(seed=99)
    assert torch.equal(first, torch.randn(4))


def test_parameter_count_is_small_enough_to_train_here():
    model = _untrained()
    # 36 -> 64 -> 64 -> 1 with biases.
    expected = 36 * 64 + 64 + 64 * 64 + 64 + 64 + 1
    assert model.n_parameters == expected
    assert model.n_parameters < 20_000, "architecture is too large for a 4-core study"


def test_default_thread_count_is_polite():
    assert DEFAULT_TORCH_THREADS == 2


def test_fit_restores_the_global_thread_count():
    before = torch.get_num_threads()
    _quick_fit(n=8, epochs=2, seed=0)
    assert torch.get_num_threads() == before


# --------------------------------------------------------------------------
# 4. fitting
# --------------------------------------------------------------------------


def _quick_fit(*, n: int, epochs: int, seed: int, **kw):
    """Small helper: build a dataset, fit, return ``(model, report)``."""
    data = _argon_dataset(n, seed=seed)
    model = BPNN(_descriptor(), seed=seed)
    report = model.fit(data, epochs=epochs, batch_size=8, patience=10**6, **kw)
    return model, report


def test_fit_rejects_unlabelled_configurations():
    model = BPNN(_descriptor(), seed=0)
    with pytest.raises(ValueError, match="labelled"):
        model.fit(Dataset([_argon()]), epochs=1)


def test_fit_overfits_twenty_configurations():
    """The sharpest end-to-end check that loss and gradients are consistent.

    Twenty configurations against 6593 parameters is a memorisation task.  If
    the force loss cannot be driven below 0.01 eV/A here then either the loss
    or the derivative path through it is broken -- there is nothing left to
    blame on generalisation.
    """
    data = _argon_dataset(20, seed=42, sigma_range=(0.2, 0.3))
    reference = float(np.sqrt((data.forces() ** 2).mean()))
    model = BPNN(_descriptor(), seed=1)
    report = model.fit(
        data, epochs=250, batch_size=5, learning_rate=5e-3, patience=10**6
    )
    print(
        f"\n[overfit] reference force RMS {reference:.4f} eV/A -> "
        f"train force RMSE {report.train_force_rmse:.5f} eV/A, "
        f"energy RMSE {report.train_energy_rmse * 1e3:.4f} meV/atom, "
        f"{report.wall_seconds:.1f} s"
    )
    assert reference > 0.05, "test data has no forces worth fitting"
    assert report.train_force_rmse < 0.01, (
        f"could not overfit 20 configurations: train force RMSE "
        f"{report.train_force_rmse:.4f} eV/A"
    )


def test_energy_only_fit_runs_but_is_not_the_default():
    """``weight_force = 0`` skips the derivative machinery entirely.

    It is supported because the response experiments want an energy-only arm,
    and it is not the default because a model with no constraint on its
    gradient has no constraint on the quantity molecular dynamics integrates --
    which shows up here as a force error comparable to the labels themselves.
    """
    data = _argon_dataset(20, seed=55)
    model = BPNN(_descriptor(), seed=2)
    report = model.fit(
        data, epochs=30, batch_size=5, weight_energy=1.0, weight_force=0.0, patience=10**6
    )
    assert np.isnan(report.train_force_rmse)
    metrics = model.evaluate(data)
    reference = float(np.sqrt((data.forces() ** 2).mean()))
    print(
        f"\n[energy-only fit] train energy RMSE "
        f"{report.train_energy_rmse * 1e3:.4f} meV/atom, force RMSE "
        f"{metrics['force_rmse']:.4f} eV/A (label RMS {reference:.4f} eV/A)"
    )
    assert report.train_energy_rmse < 1e-3


def test_model_rejects_species_it_was_not_built_for():
    model = _untrained()
    cfg = _argon()
    cfg.species = np.ones(cfg.n_atoms, dtype=np.int32)
    with pytest.raises(ValueError, match="species"):
        model.energy(cfg)


def test_fit_report_records_the_training_curve():
    _, report = _quick_fit(n=16, epochs=5, seed=0)
    assert report.n_epochs == 5
    assert len(report.history["train_loss"]) == 5
    assert len(report.history["val_loss"]) == 5
    assert len(report.history["lr"]) == 5
    assert report.history["train_loss"][-1] < report.history["train_loss"][0]
    assert report.n_parameters > 0
    # Exhausting the epoch budget is *not* convergence, and must be said so.
    assert not report.converged
    assert any("without triggering early stopping" in note for note in report.notes)


def test_early_stopping_fires_and_reports_convergence():
    data = _argon_dataset(24, seed=3)
    train, val = data.split([0.75, 0.25], seed=0)
    model = BPNN(_descriptor(), seed=5)
    report = model.fit(
        train, val=val, epochs=300, batch_size=8, patience=5, lr_schedule="plateau"
    )
    assert report.converged
    assert report.n_epochs < 300
    assert any("early stop" in note for note in report.notes)


def test_fit_is_reproducible_given_a_seed():
    cfg = _argon(sigma=0.15, seed=77)
    a, ra = _quick_fit(n=16, epochs=6, seed=9)
    b, rb = _quick_fit(n=16, epochs=6, seed=9)
    assert ra.train_force_rmse == rb.train_force_rmse
    assert a.energy(cfg) == b.energy(cfg)


def test_independently_seeded_members_differ():
    """The ensembling hook: same data, different seed, different model."""
    data = _argon_dataset(16, seed=11)
    cfg = _argon(sigma=0.15, seed=78)
    members = []
    for seed in (0, 1, 2):
        model = BPNN(_descriptor(), seed=seed)
        model.fit(data, epochs=8, batch_size=8, patience=10**6)
        members.append(model.energy(cfg))
    spread = float(np.std(members))
    assert spread > 0.0, "independently seeded members collapsed onto one prediction"


@pytest.mark.slow
@pytest.mark.physics
def test_small_fit_on_lennard_jones_argon_reports_honest_numbers():
    """Train on ~200 rattled fcc argon cells and report what is achieved.

    The numbers are printed rather than only asserted, because the point of the
    test is the record: this is the accuracy the study's BPNN arm actually
    reaches on this machine, at this cost.
    """
    data = _argon_dataset(200, seed=1234)
    train, val = data.split([0.8, 0.2], seed=0)
    reference_f = float(np.sqrt((val.forces() ** 2).mean()))
    reference_e = float(np.std(val.energies_per_atom()))

    model = BPNN(_descriptor(), seed=0)
    t0 = time.perf_counter()
    report = model.fit(
        train,
        val=val,
        epochs=60,
        batch_size=16,
        learning_rate=3e-3,
        patience=10**6,
    )
    wall = time.perf_counter() - t0
    metrics = model.evaluate(val)

    print(
        "\n[fit 200 rattled fcc-Ar, 32 atoms each]\n"
        f"  train / val configurations : {report.n_train} / {report.n_val}\n"
        f"  parameters                 : {report.n_parameters}\n"
        f"  wall time (fit, 2 threads) : {wall:.1f} s "
        f"({report.n_epochs} epochs)\n"
        f"  val energy RMSE            : {metrics['energy_rmse'] * 1e3:.4f} meV/atom "
        f"(label spread {reference_e * 1e3:.3f} meV/atom)\n"
        f"  val force  RMSE            : {metrics['force_rmse']:.5f} eV/A "
        f"(label RMS {reference_f:.5f} eV/A)\n"
        f"  val force  cosine          : {metrics['force_cosine']:.6f}\n"
        f"  converged                  : {report.converged}\n"
    )

    # Deliberately loose: the assertion exists to catch a broken fit, not to
    # certify an accuracy.  The printed numbers are the actual result.
    assert metrics["force_rmse"] < 0.25 * reference_f
    assert metrics["energy_rmse"] < 0.5 * reference_e
    assert wall < 300.0


def test_compute_timing_is_usable_for_molecular_dynamics():
    """Record the single-point cost; MD needs thousands of these.

    Measured at several thread counts because the answer is not monotone: the
    tensors in a single-point evaluation are tiny, and torch's OpenMP fork/join
    costs more than the arithmetic it parallelises.  The number that matters
    for MD is the single-threaded one, and the way to use four cores is four
    independent trajectories.
    """
    model, _ = _quick_fit(n=8, epochs=2, seed=0)
    cfg = _argon()
    model.compute(cfg)  # warm the numba kernel

    entry = torch.get_num_threads()
    timings = {}
    try:
        for n_threads in (1, 2, 4):
            torch.set_num_threads(n_threads)
            model.compute(cfg)
            t0 = time.perf_counter()
            n_calls = 20
            for _ in range(n_calls):
                model.compute(cfg, forces=True, virial=True)
            timings[n_threads] = (time.perf_counter() - t0) / n_calls
        torch.set_num_threads(1)
        t0 = time.perf_counter()
        for _ in range(20):
            model.descriptor.compute(cfg, derivatives=True)
        acsf = (time.perf_counter() - t0) / 20
    finally:
        torch.set_num_threads(entry)

    print(
        f"\n[timing] compute() on {cfg.n_atoms} atoms, forces+virial: "
        + ", ".join(f"{k} thread(s) {v * 1e3:.2f} ms" for k, v in timings.items())
        + f"; of which ACSF alone {acsf * 1e3:.2f} ms"
    )
    assert timings[1] < 0.05, "single-point evaluation is too slow to drive MD"


# --------------------------------------------------------------------------
# 5. serialisation
# --------------------------------------------------------------------------


def test_save_load_round_trip_is_bitwise_identical(tmp_path):
    """Reload must reproduce predictions exactly, not approximately.

    Approximate agreement would mean some piece of state -- the feature
    standardisation, the energy scale -- was being re-derived rather than
    restored, and a model whose predictions drift on reload is not a model.
    """
    model, _ = _quick_fit(n=16, epochs=6, seed=13)
    cfg = _argon(sigma=0.18, seed=321)
    before = model.compute(cfg)

    path = tmp_path / "bpnn.pt"
    model.save(path)
    reloaded = BPNN.load(path)

    after = reloaded.compute(cfg)
    assert reloaded.is_fitted
    assert after.energy == before.energy
    assert np.array_equal(after.forces, before.forces)
    assert np.array_equal(after.virial, before.virial)
    assert np.array_equal(after.energies, before.energies)
    assert reloaded.n_parameters == model.n_parameters
    assert reloaded.name == model.name
    assert reloaded.descriptor.n_features == model.descriptor.n_features
    assert np.array_equal(reloaded.descriptor.radial, model.descriptor.radial)


def test_load_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "not_a_bpnn.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="not an atomlab BPNN checkpoint"):
        BPNN.load(path)


def test_saved_model_keeps_its_feature_statistics(tmp_path):
    model, _ = _quick_fit(n=12, epochs=3, seed=21)
    path = tmp_path / "bpnn.pt"
    model.save(path)
    reloaded = BPNN.load(path)
    assert torch.equal(reloaded.net.feature_mean, model.net.feature_mean)
    assert torch.equal(reloaded.net.feature_scale, model.net.feature_scale)
    assert torch.equal(reloaded.net.energy_shift, model.net.energy_shift)
    assert torch.equal(reloaded.net.energy_scale, model.net.energy_scale)
