"""Tests for the molecular dynamics driver, :mod:`atomlab.md.simulate`.

The properties checked here are the ones whose failure is *silent* in a
production run: an unwrapping bug shows up only as a wrong diffusion
coefficient, a neighbour-list bug only as a wrong energy, and a relaxation that
stops short of the force criterion only as imaginary phonon modes.  Each test
therefore compares against something independently computed rather than against
a recorded value.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from atomlab.build import fcc, rattle
from atomlab.cell import wrap_positions
from atomlab.md.integrators import (
    Langevin,
    MTKBarostat,
    NoseHooverChain,
    VelocityVerlet,
)
from atomlab.md.simulate import (
    MeasureEvery,
    NeighborCache,
    SkinnedVerletList,
    equilibrate,
    minimize,
    run_md,
)
from atomlab.md.state import MDState
from atomlab.neighbors import VerletList, build_neighbor_list
from atomlab.potentials import LennardJones
from atomlab.potentials.base import Potential, ZeroPotential
from atomlab.types import Configuration, Result
from atomlab.units import EV_A3_TO_BAR, KB, MVV2E

# A supercell large enough that 2 * cutoff < min cell width with room to spare,
# which the minimum-image guard in every pair potential insists on.
ARGON_A0 = 5.26
CUTOFF = 7.0


def argon_crystal(reps=(3, 3, 3), a0: float = ARGON_A0) -> Configuration:
    return fcc(a0, "Ar", reps)


def argon_potential(**kwargs) -> LennardJones:
    return LennardJones.argon(cutoff=CUTOFF, **kwargs)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class WrappingVelocityVerlet(VelocityVerlet):
    """Velocity Verlet that wraps positions into the primary cell each step.

    Most production MD codes store wrapped coordinates.  The driver is not
    allowed to assume otherwise, and this class is how that assumption is
    tested: with it, ``state.positions`` jumps by a lattice vector whenever an
    atom crosses a face, and only a driver that accounts for images *every step*
    can reconstruct the continuous trajectory.
    """

    name = "wrapping-velocity-verlet"

    def step(self, state, potential):
        state = super().step(state, potential)
        state.positions = wrap_positions(state.positions, state.cell, state.pbc)
        return state


class SlowPotential(Potential):
    """Wraps a potential and burns wall-clock time, for the timeout test."""

    def __init__(self, base: Potential, seconds: float) -> None:
        self.base = base
        self.seconds = float(seconds)
        self.cutoff = base.cutoff
        self.name = f"slow({base.name})"
        self.n_calls = 0

    def compute(self, configuration, *, forces=True, virial=True) -> Result:
        self.n_calls += 1
        time.sleep(self.seconds)
        return self.base.compute(configuration, forces=forces, virial=virial)


def naive_post_hoc_unwrap(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Unwrap stored frames by folding jumps larger than half a box.

    This is the recipe :mod:`atomlab.md.simulate` deliberately does *not* use.
    It is here so that the test can demonstrate it failing on exactly the case
    the driver handles, rather than merely asserting that the driver agrees with
    itself.
    """
    out = positions.copy()
    inv = np.linalg.inv(cell)
    for t in range(1, out.shape[0]):
        frac = (positions[t] - positions[t - 1]) @ inv
        out[t] = out[t - 1] + (frac - np.round(frac)) @ cell
    return out


# --------------------------------------------------------------------------
# unwrapping
# --------------------------------------------------------------------------


def test_unwrapping_free_particle_across_many_images():
    """An atom driven across many images is unwrapped exactly.

    Free particles under a zero potential move as ``r0 + v t``, so the correct
    unwrapped trajectory is known in closed form.  The velocity and stride are
    chosen so the atom moves *more than half a box length between stored
    frames*, which is precisely the regime where post-hoc unwrapping fails.
    """
    box = 10.0
    cfg = Configuration(
        positions=np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]),
        cell=np.eye(3) * box,
        pbc=True,
        symbols=("Ar",),
        species=np.zeros(2, dtype=np.int32),
    )
    velocity = np.array([[37.0, -23.0, 11.0], [0.0, 0.0, 0.0]])
    dt = 0.01
    n_steps = 200
    stride = 25

    traj = run_md(
        cfg,
        ZeroPotential(),
        WrappingVelocityVerlet(dt),
        n_steps=n_steps,
        frame_stride=stride,
        seed=0,
        velocities=velocity,
        fixed_com=False,
        progress=False,
    )

    # The atom crosses many images, and more than half a box between frames.
    per_frame = np.linalg.norm(velocity[0]) * dt * stride
    assert per_frame > 0.5 * box
    total = np.linalg.norm(velocity[0]) * dt * n_steps
    assert total > 5 * box

    expected = cfg.positions[None] + velocity[None] * traj.times[:, None, None]
    assert np.max(np.abs(traj.positions - expected)) < 1e-9

    # The integrator really did wrap: raw frames stay inside the cell.
    assert traj.info["unwrap_events"] > 0

    # ...and the naive recipe really does fail here, by whole lattice vectors.
    naive = naive_post_hoc_unwrap(traj.positions, cfg.cell)
    assert np.max(np.abs(naive - expected)) > 0.4 * box


def test_unwrapping_matches_every_step_reference():
    """Striding the recording must not change the unwrapped positions.

    A run recorded every step and the same run recorded every 37th step are the
    same trajectory; the driver unwraps per step, so the strided frames must
    reproduce the corresponding dense frames to round-off.
    """
    cfg = argon_crystal((2, 2, 2), a0=5.0)
    potential = LennardJones.argon(cutoff=4.5)
    stride = 37
    n_steps = 37 * 8

    def make():
        return MDState.from_configuration(
            cfg, temperature=600.0, seed=17, fixed_com=True
        )

    dense = run_md(
        cfg,
        potential,
        WrappingVelocityVerlet(0.002),
        n_steps=n_steps,
        frame_stride=1,
        seed=None,
        state=make(),
        progress=False,
    )
    strided = run_md(
        cfg,
        potential,
        WrappingVelocityVerlet(0.002),
        n_steps=n_steps,
        frame_stride=stride,
        seed=None,
        state=make(),
        progress=False,
    )

    assert dense.n_frames == n_steps + 1
    assert strided.n_frames == n_steps // stride + 1
    assert np.allclose(strided.times, dense.times[::stride], atol=1e-12)
    assert np.max(np.abs(strided.positions - dense.positions[::stride])) < 1e-9

    # Atoms genuinely cross the boundary, so the raw (wrapped) coordinates the
    # integrator carries would look discontinuous if stored as they are.
    assert dense.info["unwrap_events"] > 0
    wrapped = wrap_positions(
        dense.positions.reshape(-1, 3), cfg.cell, cfg.pbc
    ).reshape(dense.positions.shape)
    assert np.linalg.norm(np.diff(wrapped, axis=0), axis=2).max() > 4.0


def test_unwrapped_positions_are_continuous():
    """No stored frame-to-frame jump may look like a boundary crossing."""
    cfg = argon_crystal((2, 2, 2), a0=5.0)
    traj = run_md(
        cfg,
        LennardJones.argon(cutoff=4.5),
        WrappingVelocityVerlet(0.002),
        n_steps=400,
        frame_stride=1,
        seed=5,
        temperature=600.0,
        progress=False,
    )
    jumps = np.linalg.norm(np.diff(traj.positions, axis=0), axis=2)
    # 0.002 ps at 600 K moves an argon atom well under 0.1 A per step.
    assert jumps.max() < 0.5


# --------------------------------------------------------------------------
# neighbour-list caching
# --------------------------------------------------------------------------


def test_neighbor_cache_misses_no_pairs_and_matches_forces_exactly():
    """The cached list contains every pair inside the physical cutoff.

    Two checks, at randomly chosen steps of a real trajectory: the *set* of
    pairs within ``cutoff`` reconstructed from the cached (skinned) list is
    exactly the set a from-scratch build produces, and the forces computed
    through the cached list are bitwise identical to those from a build with no
    skin at all.  Bitwise, not merely close: the extra pairs in the skin
    contribute exactly zero and the lists are canonically ordered, so any
    difference at all would signal a real discrepancy.
    """
    cfg = argon_crystal()
    hot = argon_potential()
    traj = run_md(
        cfg,
        hot,
        VelocityVerlet(0.002),
        n_steps=200,
        frame_stride=1,
        seed=11,
        temperature=150.0,
        progress=False,
    )

    cache = NeighborCache(cutoff=CUTOFF, skin=1.0, half=True)
    fresh = LennardJones.argon(cutoff=CUTOFF, skin=0.0)
    skinned = LennardJones.argon(cutoff=CUTOFF, skin=1.0)

    rng = np.random.default_rng(0)
    check_at = set(rng.choice(traj.n_frames, size=12, replace=False).tolist())

    for step in range(traj.n_frames):
        frame = traj.frame(step)
        nl = cache.update(frame)
        if step not in check_at:
            continue

        # -- no pair inside the cutoff is missing from the cached list
        offsets = nl.shift.astype(float) @ frame.cell
        r = np.linalg.norm(frame.positions[nl.j] + offsets - frame.positions[nl.i], axis=1)
        inside = r <= CUTOFF
        cached_pairs = {
            (int(i), int(j), int(s[0]), int(s[1]), int(s[2]))
            for i, j, s in zip(nl.i[inside], nl.j[inside], nl.shift[inside])
        }
        reference = build_neighbor_list(frame, CUTOFF, half=True).as_set()
        assert cached_pairs == reference

        # -- and the forces the cache feeds a potential are exact
        assert np.array_equal(skinned.forces(frame), fresh.forces(frame))
        assert skinned.energy(frame) == fresh.energy(frame)

    # The whole point of the skin: far fewer builds than queries.
    assert cache.n_queries == traj.n_frames
    assert cache.n_rebuilds < 0.2 * cache.n_queries


def test_neighbor_cache_rebuild_count_in_run_md():
    """``run_md`` reports the rebuild count and it is far below the step count."""
    cfg = argon_crystal()
    potential = argon_potential()
    n_steps = 400
    traj = run_md(
        cfg,
        potential,
        VelocityVerlet(0.002),
        n_steps=n_steps,
        frame_stride=50,
        seed=3,
        temperature=100.0,
        progress=False,
        neighbor_skin=1.0,
    )
    rebuilds = traj.info["neighbor_rebuilds"]
    assert traj.info["neighbor_cache_targets"] == [potential.name]
    assert 0 < rebuilds < 0.1 * n_steps
    # The potential must be handed back exactly as it was given.
    assert potential._verlet is None


def test_neighbor_cache_survives_cell_deformation_under_npt():
    """Under a barostat the skin criterion must account for the cell change.

    ``VerletList`` invalidates on any cell change whatsoever, so with an MTK
    barostat -- which rescales the cell every single step -- it rebuilds every
    step and the skin does nothing.  ``SkinnedVerletList`` uses a bound that
    tolerates deformation; it must still never miss a pair.
    """
    cfg = argon_crystal()
    potential = argon_potential()
    barostat = MTKBarostat(0.002, 100.0, 500.0, 0.1, 0.5)
    n_steps = 250

    traj = run_md(
        cfg,
        potential,
        barostat,
        n_steps=n_steps,
        frame_stride=1,
        seed=7,
        temperature=100.0,
        progress=False,
    )
    # The cell really is changing every step.
    volumes = traj.scalars["volume"]
    assert np.all(np.diff(volumes) != 0.0)
    assert abs(volumes[-1] / volumes[0] - 1.0) > 1e-3

    skinned = SkinnedVerletList(cutoff=CUTOFF, skin=1.0, half=True)
    plain = VerletList(cutoff=CUTOFF, skin=1.0, half=True)
    for step in range(traj.n_frames):
        frame = traj.frame(step)
        nl = skinned.update(frame)
        plain.update(frame)
        if step % 25 == 0:
            offsets = nl.shift.astype(float) @ frame.cell
            r = np.linalg.norm(
                frame.positions[nl.j] + offsets - frame.positions[nl.i], axis=1
            )
            inside = r <= CUTOFF
            cached = {
                (int(i), int(j), int(s[0]), int(s[1]), int(s[2]))
                for i, j, s in zip(nl.i[inside], nl.j[inside], nl.shift[inside])
            }
            assert cached == build_neighbor_list(frame, CUTOFF, half=True).as_set()

    assert plain.n_builds == traj.n_frames  # the pathology being fixed
    assert skinned.n_builds < 0.1 * traj.n_frames


def test_skinned_verlet_bound_is_conservative():
    """The deformation bound must exceed the true worst-case pair change."""
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    lst = SkinnedVerletList(cutoff=4.0, skin=0.8, half=True)
    nl = lst.build(cfg)
    d0 = nl.shift.astype(float) @ cfg.cell
    d0 = cfg.positions[nl.j] + d0 - cfg.positions[nl.i]

    rng = np.random.default_rng(2)
    for scale in (1.002, 1.01, 0.99):
        moved = cfg.copy()
        moved.cell = cfg.cell * scale
        moved.positions = cfg.positions * scale + rng.normal(0.0, 0.05, cfg.positions.shape)
        d = nl.shift.astype(float) @ moved.cell
        d = moved.positions[nl.j] + d - moved.positions[nl.i]
        true_change = float(np.max(np.linalg.norm(d - d0, axis=1)))
        bound = lst.deformation_bound(moved.positions, moved.cell)
        assert bound >= true_change


# --------------------------------------------------------------------------
# logging and determinism
# --------------------------------------------------------------------------


def test_run_md_is_deterministic_given_a_seed():
    """Same seed, same trajectory -- bitwise, including the stochastic path."""
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)

    def go():
        return run_md(
            cfg,
            potential,
            Langevin(0.002, 90.0, 4.0),
            n_steps=60,
            frame_stride=3,
            log_stride=2,
            seed=1234,
            thermalize_steps=10,
            temperature=90.0,
            progress=False,
        )

    a, b = go(), go()
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.velocities, b.velocities)
    for key in a.scalars:
        assert np.array_equal(a.scalars[key], b.scalars[key])

    different = run_md(
        cfg,
        potential,
        Langevin(0.002, 90.0, 4.0),
        n_steps=60,
        frame_stride=3,
        seed=4321,
        thermalize_steps=10,
        temperature=90.0,
        progress=False,
    )
    assert not np.allclose(different.positions, a.positions)


def test_run_md_refuses_unseeded_randomness():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    with pytest.raises(ValueError, match="explicit seed"):
        run_md(
            cfg,
            potential,
            Langevin(0.002, 90.0, 4.0),
            n_steps=1,
            seed=None,
            progress=False,
        )
    with pytest.raises(ValueError, match="explicit seed"):
        run_md(
            cfg,
            potential,
            VelocityVerlet(0.002),
            n_steps=1,
            seed=None,
            temperature=90.0,
            progress=False,
        )


def test_scalars_shapes_units_and_conserved_quantity():
    """Scalars are frame-aligned, in the documented units, and complete."""
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    n_steps, frame_stride, log_stride = 120, 12, 4
    traj = run_md(
        cfg,
        potential,
        NoseHooverChain(0.002, 100.0, 0.2),
        n_steps=n_steps,
        frame_stride=frame_stride,
        log_stride=log_stride,
        seed=9,
        temperature=100.0,
        progress=False,
    )

    expected_keys = {
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "temperature",
        "pressure",
        "volume",
        "conserved_quantity",
    }
    assert expected_keys <= set(traj.scalars)
    for key, values in traj.scalars.items():
        assert values.shape == (traj.n_frames,), key
    assert traj.n_frames == n_steps // frame_stride + 1

    # The finer log lives beside the frames, with its own time axis.
    log = traj.info["log"]
    assert log["time"].shape == (n_steps // log_stride + 1,)
    assert expected_keys <= set(log)
    assert np.allclose(log["time"][:: frame_stride // log_stride], traj.times)

    # Independent recomputation of frame 0 from the potential and the state.
    state0 = MDState.from_configuration(cfg, temperature=100.0, seed=9, fixed_com=True)
    result = potential.compute(cfg)
    kinetic = 0.5 * MVV2E * float(
        np.sum(state0.masses[:, None] * state0.velocities**2)
    )
    assert traj.scalars["potential_energy"][0] == pytest.approx(result.energy, rel=1e-12)
    assert traj.scalars["kinetic_energy"][0] == pytest.approx(kinetic, rel=1e-12)
    assert traj.scalars["temperature"][0] == pytest.approx(
        2.0 * kinetic / ((3 * cfg.n_atoms - 3) * KB), rel=1e-12
    )
    pressure = (2.0 * kinetic + np.trace(result.virial)) / (3.0 * cfg.volume)
    assert traj.scalars["pressure"][0] == pytest.approx(pressure * EV_A3_TO_BAR, rel=1e-10)
    assert traj.scalars["volume"][0] == pytest.approx(cfg.volume, rel=1e-14)

    # A Nose-Hoover chain conserves H'; a Langevin run has nothing to conserve.
    conserved = traj.scalars["conserved_quantity"]
    assert np.std(conserved) / cfg.n_atoms < 1e-4
    stochastic = run_md(
        cfg,
        potential,
        Langevin(0.002, 100.0, 3.0),
        n_steps=10,
        seed=1,
        temperature=100.0,
        progress=False,
    )
    assert "conserved_quantity" not in stochastic.scalars


def test_nve_energy_is_conserved_and_trajectory_metadata_is_set():
    cfg = argon_crystal()
    potential = argon_potential()
    traj = run_md(
        cfg,
        potential,
        VelocityVerlet(0.001),
        n_steps=500,
        frame_stride=10,
        seed=2,
        temperature=80.0,
        progress=False,
    )
    drift = np.abs(traj.scalars["total_energy"] - traj.scalars["total_energy"][0]).max()
    assert drift / cfg.n_atoms < 1e-6
    assert traj.info["dynamical"] is True
    assert traj.info["sampler"].startswith("md:")
    assert traj.info["n_steps_completed"] == 500
    assert traj.info["wall_time_exceeded"] is False
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(0.5, rel=1e-12)
    assert isinstance(traj.info["final_state"], MDState)


def test_thermalize_steps_are_discarded():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    traj = run_md(
        cfg,
        potential,
        VelocityVerlet(0.002),
        n_steps=20,
        frame_stride=5,
        seed=4,
        thermalize_steps=50,
        temperature=120.0,
        progress=False,
    )
    assert traj.n_frames == 5
    assert traj.times[0] == 0.0
    assert traj.info["thermalize_steps_completed"] == 50
    # Production starts after the thermalisation, so the state has moved on.
    assert traj.info["final_state"].step == 70
    assert not np.allclose(traj.positions[0], cfg.positions)


# --------------------------------------------------------------------------
# graceful degradation
# --------------------------------------------------------------------------


def test_max_wall_seconds_triggers_cleanly():
    """A run that runs out of wall time returns a valid, flagged prefix."""
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = SlowPotential(LennardJones.argon(cutoff=5.0), 0.004)
    budget = 0.25
    traj = run_md(
        cfg,
        potential,
        VelocityVerlet(0.002),
        n_steps=100_000,
        frame_stride=2,
        seed=6,
        temperature=100.0,
        progress=False,
        max_wall_seconds=budget,
    )
    assert traj.info["wall_time_exceeded"] is True
    assert 0 < traj.info["n_steps_completed"] < 100_000
    assert traj.info["wall_seconds"] < 10 * budget
    # Whatever was recorded is a consistent trajectory.
    assert traj.n_frames >= 1
    assert traj.positions.shape == (traj.n_frames, cfg.n_atoms, 3)
    assert traj.velocities.shape == traj.positions.shape
    for values in traj.scalars.values():
        assert values.shape == (traj.n_frames,)
    assert np.all(np.diff(traj.times) > 0)
    assert np.isfinite(traj.scalars["total_energy"]).all()


def test_max_wall_seconds_not_triggered_is_recorded_too():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    traj = run_md(
        cfg,
        LennardJones.argon(cutoff=5.0),
        VelocityVerlet(0.002),
        n_steps=20,
        seed=6,
        temperature=100.0,
        progress=False,
        max_wall_seconds=600.0,
    )
    assert traj.info["wall_time_exceeded"] is False
    assert traj.info["n_steps_completed"] == 20


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------


def test_callbacks_see_every_step_and_can_avoid_storing_frames():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)

    seen = []

    def record(state, info):
        seen.append((info.phase, info.step, state.step))

    measured = MeasureEvery(5, lambda state, info: state.temperature())

    class Hooked:
        def __init__(self):
            self.began = self.finished = 0

        def __call__(self, state, info):
            pass

        def begin(self, state, info):
            self.began += 1

        def finish(self, state, info):
            self.finished += 1

    hooked = Hooked()
    traj = run_md(
        cfg,
        potential,
        VelocityVerlet(0.002),
        n_steps=30,
        frame_stride=30,  # store almost nothing...
        seed=8,
        thermalize_steps=7,
        temperature=100.0,
        callbacks=(record, measured, hooked),
        progress=False,
    )

    assert traj.n_frames == 2  # ...yet the measurement is dense
    assert sum(1 for phase, *_ in seen if phase == "thermalize") == 7
    assert sum(1 for phase, *_ in seen if phase == "production") == 31
    assert hooked.began == 1 and hooked.finished == 1

    times, values = measured.as_arrays()
    assert values.shape == (7,)  # steps 0, 5, ..., 30
    assert np.allclose(times, np.arange(7) * 5 * 0.002)
    assert np.all(values > 0.0)
    # The callback saw the same temperatures the driver logged.
    assert values[0] == pytest.approx(traj.scalars["temperature"][0], rel=1e-12)


# --------------------------------------------------------------------------
# relaxation
# --------------------------------------------------------------------------


def test_minimize_relaxes_a_rattled_crystal_below_fmax():
    """Forces must actually fall below ``fmax``, and the crystal must come back."""
    perfect = argon_crystal()
    potential = argon_potential()
    disturbed = rattle(perfect, 0.08, seed=3)

    relaxed, report = minimize(disturbed, potential, fmax=1e-5, max_steps=500)

    assert report.converged
    assert report.initial_fmax > 1e-2
    assert report.fmax < 1e-5
    # Independent check on the returned configuration, not on the report.
    assert np.max(np.linalg.norm(potential.forces(relaxed), axis=1)) < 1e-5
    assert relaxed.energy < potential.energy(disturbed)
    # An fcc crystal is a genuine minimum: rattling and relaxing returns to it.
    assert relaxed.energy / relaxed.n_atoms == pytest.approx(
        potential.energy_per_atom(perfect), abs=1e-8
    )
    assert report.cell_residual == 0.0
    assert np.array_equal(relaxed.cell, disturbed.cell)


@pytest.mark.parametrize("method", ["lbfgs", "cg", "bfgs"])
def test_minimize_methods_all_reach_the_criterion(method):
    potential = LennardJones.argon(cutoff=5.0)
    disturbed = rattle(argon_crystal((2, 2, 2), a0=5.4), 0.05, seed=11)
    relaxed, report = minimize(disturbed, potential, method=method, fmax=1e-4, max_steps=800)
    assert report.converged, report.message
    assert np.max(np.linalg.norm(potential.forces(relaxed), axis=1)) < 1e-4


def test_variable_cell_relaxation_recovers_the_equilibrium_lattice_constant():
    """Relaxing the cell at P = 0 must find the lattice constant of the potential.

    The reference is not a literature number but the same potential's own
    minimum, located independently by a one-dimensional scan of ``E(a)``: the
    two routes share no code beyond ``compute``.
    """
    from scipy.optimize import minimize_scalar

    potential = argon_potential()
    reps = (3, 3, 3)
    n_cells = reps[0] * reps[1] * reps[2]

    scan = minimize_scalar(
        lambda a: potential.energy_per_atom(fcc(a, "Ar", reps)),
        bracket=(5.2, 5.4),
        method="brent",
        options={"xtol": 1e-12},
    )
    a_reference = float(scan.x)

    for a_start in (5.05, 5.60):
        relaxed, report = minimize(
            fcc(a_start, "Ar", reps),
            potential,
            fmax=1e-6,
            relax_cell=True,
            max_steps=500,
        )
        assert report.converged, report.message
        a_relaxed = (relaxed.volume / n_cells) ** (1.0 / 3.0)
        assert a_relaxed == pytest.approx(a_reference, abs=1e-5)
        # Zero target pressure means a zero virial pressure, in bar.
        assert report.pressure_bar == pytest.approx(0.0, abs=1e-2)
        assert report.stress_residual < 1e-8
        # Cubic symmetry is not broken by the six-component strain.
        off = relaxed.cell - np.diag(np.diag(relaxed.cell))
        assert np.max(np.abs(off)) < 1e-8
        assert np.ptp(np.diag(relaxed.cell)) < 1e-8


def test_variable_cell_relaxation_hits_a_target_pressure():
    potential = argon_potential()
    target = 4000.0
    relaxed, report = minimize(
        fcc(5.45, "Ar", (3, 3, 3)),
        potential,
        fmax=1e-6,
        relax_cell=True,
        pressure_bar=target,
        max_steps=500,
    )
    assert report.converged, report.message
    assert report.pressure_bar == pytest.approx(target, rel=1e-5)
    # Independent recomputation from the returned configuration.
    assert potential.pressure(relaxed) * EV_A3_TO_BAR == pytest.approx(target, rel=1e-5)
    # Compression, as a positive target pressure demands.
    assert relaxed.volume < fcc(5.45, "Ar", (3, 3, 3)).volume


def test_variable_cell_hydrostatic_matches_full_strain_for_a_cubic_crystal():
    potential = argon_potential()
    start = fcc(5.5, "Ar", (3, 3, 3))
    iso, report_iso = minimize(start, potential, fmax=1e-6, relax_cell=True, hydrostatic=True)
    full, report_full = minimize(start, potential, fmax=1e-6, relax_cell=True)
    assert report_iso.converged and report_full.converged
    assert iso.volume == pytest.approx(full.volume, rel=1e-6)


def test_minimize_rejects_bad_arguments():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    with pytest.raises(ValueError, match="unknown method"):
        minimize(cfg, potential, method="fire")
    open_cfg = Configuration(positions=cfg.positions, cell=cfg.cell, pbc=False, symbols=("Ar",))
    with pytest.raises(ValueError, match="fully periodic"):
        minimize(open_cfg, potential, relax_cell=True)


# --------------------------------------------------------------------------
# equilibrate
# --------------------------------------------------------------------------


def test_equilibrate_returns_only_the_production_segment():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    n_production = 120
    frame_stride = 10

    traj = equilibrate(
        cfg,
        potential,
        temperature=90.0,
        dt=0.002,
        n_equilibrate=150,
        n_production=n_production,
        seed=21,
        thermostat="langevin",
        production="nve",
        friction=8.0,
        frame_stride=frame_stride,
        progress=False,
    )

    assert traj.n_frames == n_production // frame_stride + 1
    assert traj.times[0] == 0.0
    assert traj.times[-1] == pytest.approx(n_production * 0.002, rel=1e-12)
    assert traj.info["n_equilibrate"] == 150
    assert traj.info["production_ensemble"] == "nve"
    assert traj.info["integrator"] == "velocity-verlet"
    assert traj.info["dynamical"] is True
    # Production is NVE: the total energy is conserved, so the thermostat is
    # genuinely off during the recorded segment.
    total = traj.scalars["total_energy"]
    assert np.abs(total - total[0]).max() / cfg.n_atoms < 1e-6
    # Equilibration did its job: the crystal is near the target, not at 0 K or
    # at twice the setpoint.
    assert 30.0 < traj.scalars["temperature"].mean() < 150.0


def test_equilibrate_nvt_production_keeps_the_thermostat():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    traj = equilibrate(
        cfg,
        potential,
        temperature=100.0,
        dt=0.002,
        n_equilibrate=100,
        n_production=100,
        seed=22,
        thermostat="nhc",
        production="nvt",
        frame_stride=10,
        progress=False,
    )
    assert traj.info["integrator"] == "nose-hoover-chain"
    assert "conserved_quantity" in traj.scalars


def test_equilibrate_rejects_unknown_ensembles():
    cfg = argon_crystal((2, 2, 2), a0=5.4)
    potential = LennardJones.argon(cutoff=5.0)
    with pytest.raises(ValueError, match="unknown thermostat"):
        equilibrate(
            cfg, potential, temperature=90.0, dt=0.002, n_equilibrate=1,
            n_production=1, seed=1, thermostat="berendsen", progress=False,
        )
    with pytest.raises(ValueError, match="unknown production ensemble"):
        equilibrate(
            cfg, potential, temperature=90.0, dt=0.002, n_equilibrate=1,
            n_production=1, seed=1, production="npt", progress=False,
        )
