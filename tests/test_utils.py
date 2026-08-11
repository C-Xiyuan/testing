"""Tests for the reproducibility and I/O plumbing (:mod:`atomlab.utils`).

The bar these tests hold the code to is *bitwise* round-tripping for the npz
archive.  That is not pedantry: the response analysis subtracts energies of two
nearly identical potentials, so an error of a few ulp introduced by writing a
dataset to disk and reading it back would appear as a spurious ``delta_U`` of
the same order as the signal the project is trying to measure.
"""

from __future__ import annotations

import json
import logging
import random as py_random
import textwrap

import numpy as np
import pytest

from atomlab.types import Configuration, Dataset, Trajectory
from atomlab.utils.io import (
    decode_tree,
    encode_tree,
    load_configurations,
    load_dataset,
    load_json,
    load_trajectory,
    read_extxyz,
    save_configurations,
    save_dataset,
    save_json,
    save_trajectory,
    write_extxyz,
)
from atomlab.utils.logging import ProgressReporter, get_logger, progress
from atomlab.utils.seeds import (
    SeedScope,
    capture_random_state,
    make_rng,
    seed_everything,
    spawn_rngs,
)
from atomlab.utils.timing import (
    REGISTRY,
    Timer,
    TimerRegistry,
    format_duration,
    get_timings,
    reset_timings,
    timed,
)


# ==========================================================================
# fixtures
# ==========================================================================


def _make_configs(seed: int = 0) -> list[Configuration]:
    """Two labelled configurations of *different* sizes and species counts.

    Ragged sizes are the interesting case for the offsets machinery, and a
    triclinic cell exercises the fact that ``cell`` rows are lattice vectors.
    """
    rng = make_rng(seed)

    cell_a = np.array([[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]])
    cfg_a = Configuration(
        positions=rng.uniform(0.0, 5.4, size=(8, 3)),
        cell=cell_a,
        pbc=True,
        species=np.zeros(8, dtype=np.int32),
        symbols=("Ar",),
        energy=-3.14159265358979,
        forces=rng.normal(size=(8, 3)),
        virial=rng.normal(size=(3, 3)),
        info={
            "temperature": 300.0,
            "source": "rattled fcc",
            "step": 17,
            "converged": True,
            "weights": np.array([0.5, 0.25, 0.125]),
            "float32": np.float32(0.1),
            "nested": {"tuple": (1, 2.5, "x"), "none": None},
        },
    )

    # Triclinic, two species, no labels: exercises the optional-field masks.
    cell_b = np.array([[4.0, 0.0, 0.0], [1.3, 3.7, 0.0], [0.7, -0.4, 4.2]])
    cfg_b = Configuration(
        positions=rng.uniform(0.0, 4.0, size=(5, 3)),
        cell=cell_b,
        pbc=np.array([True, True, False]),
        species=np.array([0, 1, 0, 1, 1], dtype=np.int32),
        symbols=("Si", "Ge"),
        info={},
    )
    return [cfg_a, cfg_b]


def _assert_configs_identical(a: Configuration, b: Configuration, *, exact: bool = True) -> None:
    cmp = np.testing.assert_array_equal if exact else _allclose
    cmp(a.positions, b.positions)
    cmp(a.cell, b.cell)
    np.testing.assert_array_equal(a.pbc, b.pbc)
    np.testing.assert_array_equal(a.species, b.species)
    assert a.symbols == b.symbols
    cmp(a.masses, b.masses)
    assert (a.energy is None) == (b.energy is None)
    if a.energy is not None:
        if exact:
            assert a.energy == b.energy
        else:
            assert abs(a.energy - b.energy) <= 1e-12 * max(1.0, abs(a.energy))
    assert (a.forces is None) == (b.forces is None)
    if a.forces is not None:
        cmp(a.forces, b.forces)
    assert (a.virial is None) == (b.virial is None)
    if a.virial is not None:
        cmp(a.virial, b.virial)


def _allclose(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def _assert_info_identical(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            assert isinstance(vb, np.ndarray)
            assert va.dtype == vb.dtype
            assert va.tobytes() == vb.tobytes()
        elif isinstance(va, dict):
            _assert_info_identical(va, vb)
        else:
            assert type(va) is type(vb), f"{k}: {type(va)} != {type(vb)}"
            assert va == vb


# ==========================================================================
# npz round trip
# ==========================================================================


def test_configurations_npz_roundtrip_is_bitwise(tmp_path):
    configs = _make_configs()
    path = save_configurations(tmp_path / "configs.npz", configs)
    back = load_configurations(path)

    assert len(back) == len(configs)
    for orig, new in zip(configs, back):
        _assert_configs_identical(orig, new, exact=True)
        _assert_info_identical(orig.info, new.info)

    # Explicitly assert bitwise equality of the float payloads, since
    # assert_array_equal would also accept a value that merely prints the same.
    assert back[0].positions.tobytes() == configs[0].positions.tobytes()
    assert back[0].forces.tobytes() == configs[0].forces.tobytes()
    assert back[0].virial.tobytes() == configs[0].virial.tobytes()
    assert back[0].cell.tobytes() == configs[0].cell.tobytes()


def test_npz_roundtrip_survives_pathological_floats(tmp_path):
    """NaN/inf energies must come back as themselves, not as "missing"."""
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        energy=float("nan"),
        forces=np.array([[np.inf, -np.inf, 0.0], [1e-308, -0.0, 5e-324]]),
    )
    back = load_configurations(save_configurations(tmp_path / "odd.npz", [cfg]))[0]
    assert back.energy is not None and np.isnan(back.energy)
    assert back.forces.tobytes() == cfg.forces.tobytes()
    # Signed zero survives, which array_equal alone would not detect.
    assert np.signbit(back.forces[1, 1])


def test_unlabelled_fields_stay_none(tmp_path):
    configs = _make_configs()
    back = load_configurations(save_configurations(tmp_path / "c.npz", configs))
    assert back[1].energy is None
    assert back[1].forces is None
    assert back[1].virial is None


def test_empty_configuration_list(tmp_path):
    back = load_configurations(save_configurations(tmp_path / "empty.npz", []))
    assert back == []


def test_dataset_roundtrip(tmp_path):
    ds = Dataset(configurations=_make_configs(), info={"budget": 128, "split": "train"})
    back = load_dataset(save_dataset(tmp_path / "ds.npz", ds))
    assert back.info == ds.info
    assert len(back) == len(ds)
    _assert_configs_identical(ds[0], back[0], exact=True)


def test_trajectory_roundtrip_is_bitwise(tmp_path):
    rng = make_rng(3)
    template = _make_configs()[0].stripped()
    t, n = 6, template.n_atoms
    traj = Trajectory(
        positions=rng.normal(size=(t, n, 3)) * 3.0,
        cells=np.repeat(template.cell[None], t, axis=0),
        times=np.arange(t) * 0.001,
        template=template,
        velocities=rng.normal(size=(t, n, 3)),
        scalars={
            "potential_energy": rng.normal(size=t),
            "temperature": rng.uniform(280, 320, size=t),
        },
        info={"integrator": "VelocityVerlet", "dt_ps": 0.001},
    )
    back = load_trajectory(save_trajectory(tmp_path / "traj.npz", traj))

    assert back.positions.tobytes() == traj.positions.tobytes()
    assert back.cells.tobytes() == traj.cells.tobytes()
    assert back.times.tobytes() == traj.times.tobytes()
    assert back.velocities.tobytes() == traj.velocities.tobytes()
    assert set(back.scalars) == set(traj.scalars)
    for k in traj.scalars:
        assert back.scalars[k].tobytes() == traj.scalars[k].tobytes()
    assert back.info == traj.info
    _assert_configs_identical(traj.template, back.template, exact=True)


def test_trajectory_without_velocities(tmp_path):
    template = _make_configs()[0].stripped()
    traj = Trajectory(
        positions=np.zeros((2, template.n_atoms, 3)),
        cells=np.repeat(template.cell[None], 2, axis=0),
        times=np.array([0.0, 0.5]),
        template=template,
    )
    back = load_trajectory(save_trajectory(tmp_path / "t2.npz", traj))
    assert back.velocities is None
    assert back.n_frames == 2


def test_loading_wrong_archive_kind_raises(tmp_path):
    p = save_configurations(tmp_path / "c.npz", _make_configs())
    with pytest.raises(ValueError, match="not a trajectory"):
        load_trajectory(p)


# ==========================================================================
# extxyz
# ==========================================================================


def test_extxyz_roundtrip_to_float_precision(tmp_path):
    configs = _make_configs()
    path = write_extxyz(tmp_path / "frames.xyz", configs)
    back = read_extxyz(path)

    assert len(back) == len(configs)
    for orig, new in zip(configs, back):
        _assert_configs_identical(orig, new, exact=False)
        _assert_info_identical(orig.info, new.info)

    # The format only promises float precision, but because floats are written
    # with ``repr`` (shortest round-tripping decimal) it is in fact exact.  Lock
    # that in: a future switch to a fixed "%.8f" would silently degrade every
    # dataset that goes through this path.
    assert back[0].positions.tobytes() == configs[0].positions.tobytes()
    assert back[0].forces.tobytes() == configs[0].forces.tobytes()
    assert back[0].energy == configs[0].energy


def test_extxyz_header_is_standard(tmp_path):
    cfg = _make_configs()[0]
    text = (tmp_path / "one.xyz").with_suffix(".xyz")
    write_extxyz(text, [cfg])
    lines = text.read_text().splitlines()
    assert lines[0] == str(cfg.n_atoms)
    header = lines[1]
    assert "Lattice=\"" in header
    assert "Properties=species:S:1:pos:R:3:forces:R:3" in header
    assert 'pbc="T T T"' in header
    assert "energy=" in header
    # Ar masses come from the symbol table, so no masses column is needed.
    assert "masses" not in header
    assert lines[2].split()[0] == "Ar"


def test_extxyz_parses_handwritten_triclinic_file(tmp_path):
    """A literal file, written by hand, with a triclinic Lattice string.

    This is the interoperability test: it must parse without any atomlab
    extension keys present, and the cell must come back with lattice vectors as
    *rows* (a1 = (4.0, 0, 0)), matching ``r = s @ cell``.
    """
    text = textwrap.dedent(
        """\
        3
        Lattice="4.0 0.0 0.0 1.5 3.8 0.0 0.25 -0.75 4.5" Properties=species:S:1:pos:R:3:forces:R:3 energy=-12.5 pbc="T T F"
        Si 0.00000 0.00000 0.00000 0.10000 -0.20000 0.30000
        Si 1.35750 1.35750 1.35750 -0.10000 0.20000 -0.30000
        Ge 2.71500 2.71500 0.00000 0.00000 0.00000 0.00000
        """
    )
    path = tmp_path / "hand.xyz"
    path.write_text(text)

    (cfg,) = read_extxyz(path)

    assert cfg.n_atoms == 3
    np.testing.assert_allclose(
        cfg.cell,
        np.array([[4.0, 0.0, 0.0], [1.5, 3.8, 0.0], [0.25, -0.75, 4.5]]),
    )
    # Triclinic volume = |det(cell)| = 4.0 * 3.8 * 4.5.
    assert cfg.volume == pytest.approx(4.0 * 3.8 * 4.5)
    np.testing.assert_array_equal(cfg.pbc, np.array([True, True, False]))
    assert cfg.symbols == ("Si", "Ge")
    np.testing.assert_array_equal(cfg.species, np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(cfg.masses, [28.0855, 28.0855, 72.630])
    assert cfg.energy == pytest.approx(-12.5)
    np.testing.assert_allclose(cfg.positions[1], [1.35750, 1.35750, 1.35750])
    np.testing.assert_allclose(cfg.forces[0], [0.1, -0.2, 0.3])


def test_extxyz_open_cluster_has_no_pbc(tmp_path):
    """No Lattice key => an open system, never a guessed cell."""
    path = tmp_path / "cluster.xyz"
    path.write_text("2\nProperties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\nAr 0.0 0.0 3.4\n")
    (cfg,) = read_extxyz(path)
    assert not cfg.pbc.any()
    assert cfg.volume == float("inf")


def test_extxyz_multiframe_and_trajectory(tmp_path):
    template = _make_configs()[0].stripped()
    rng = make_rng(11)
    traj = Trajectory(
        positions=rng.uniform(0, 5.4, size=(4, template.n_atoms, 3)),
        cells=np.repeat(template.cell[None], 4, axis=0),
        times=np.arange(4) * 0.002,
        template=template,
    )
    path = write_extxyz(tmp_path / "traj.xyz", traj)
    frames = read_extxyz(path)
    assert len(frames) == 4
    np.testing.assert_allclose(frames[2].positions, traj.positions[2], rtol=0, atol=0)


def test_extxyz_gz_roundtrip(tmp_path):
    configs = _make_configs()
    back = read_extxyz(write_extxyz(tmp_path / "f.xyz.gz", configs))
    _assert_configs_identical(configs[0], back[0], exact=False)


# ==========================================================================
# json
# ==========================================================================


def test_save_json_numpy_aware(tmp_path):
    obj = {"a": np.arange(3), "b": np.float64(1.5), "c": [np.int32(7)], "d": "x"}
    back = load_json(save_json(tmp_path / "j.json", obj))
    assert back == {"a": [0, 1, 2], "b": 1.5, "c": [7], "d": "x"}


def test_save_json_preserve_numpy_roundtrip(tmp_path):
    obj = {"a": np.arange(3, dtype=np.int16), "b": np.float32(0.1), "t": (1, "two")}
    back = load_json(save_json(tmp_path / "j2.json", obj, preserve_numpy=True))
    assert back["a"].dtype == np.int16
    np.testing.assert_array_equal(back["a"], obj["a"])
    assert isinstance(back["b"], np.float32) and back["b"] == obj["b"]
    assert back["t"] == (1, "two")


def test_encode_tree_rejects_non_string_keys():
    with pytest.raises(TypeError, match="keys must be str"):
        encode_tree({1: "a"})


def test_encode_tree_is_json_serialisable_and_exact():
    obj = {"x": np.linspace(0, 1, 5), "y": b"raw"}
    round_tripped = decode_tree(json.loads(json.dumps(encode_tree(obj))))
    assert round_tripped["x"].tobytes() == obj["x"].tobytes()
    assert round_tripped["y"] == b"raw"


# ==========================================================================
# seeds
# ==========================================================================


def test_make_rng_accepts_the_documented_inputs():
    assert isinstance(make_rng(0), np.random.Generator)
    assert isinstance(make_rng(None), np.random.Generator)
    assert isinstance(make_rng(np.random.SeedSequence(3)), np.random.Generator)
    g = make_rng(7)
    assert make_rng(g) is g  # passing a Generator must share, not restart
    with pytest.raises(TypeError):
        make_rng("not-a-seed")


def test_make_rng_is_reproducible():
    assert make_rng(42).normal(size=5).tolist() == make_rng(42).normal(size=5).tolist()
    assert make_rng(42).normal() != make_rng(43).normal()


def test_spawn_rngs_reproducible_and_independent():
    a = spawn_rngs(1234, 4)
    b = spawn_rngs(1234, 4)
    xa = np.array([g.normal(size=2000) for g in a])
    xb = np.array([g.normal(size=2000) for g in b])

    # Reproducible: same root seed, same streams.
    np.testing.assert_array_equal(xa, xb)

    # Distinct: no two children produce the same stream.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(xa[i], xa[j])
            # Independent to within sampling error: 2000 samples give a
            # correlation standard error of 1/sqrt(2000) = 0.022, so |r| > 0.15
            # (about 7 sigma) would indicate a genuine relationship.
            r = float(np.corrcoef(xa[i], xa[j])[0, 1])
            assert abs(r) < 0.15, f"children {i},{j} correlated: r={r}"


def test_spawn_rngs_prefix_stable_under_worker_count():
    """A 4-worker rerun of an 8-worker job must reuse the first four streams."""
    few = spawn_rngs(99, 4)
    many = spawn_rngs(99, 8)
    for g1, g2 in zip(few, many):
        np.testing.assert_array_equal(g1.normal(size=10), g2.normal(size=10))


def test_spawn_rngs_repeated_calls_do_not_advance():
    first = [g.normal() for g in spawn_rngs(5, 3)]
    second = [g.normal() for g in spawn_rngs(5, 3)]
    assert first == second


def test_spawn_rngs_zero_and_negative():
    assert spawn_rngs(0, 0) == []
    with pytest.raises(ValueError):
        spawn_rngs(0, -1)


def test_seed_everything_records_what_it_set():
    rec = seed_everything(7)
    assert rec["seed"] == 7
    assert isinstance(rec["python_random"], int)
    assert isinstance(rec["numpy_legacy"], int)
    # torch is a declared dependency of this project, so it must be recorded.
    assert rec["torch"] is not None
    assert json.dumps(rec)  # must be manifest-serialisable

    a = (py_random.random(), float(np.random.rand()))
    seed_everything(7)
    b = (py_random.random(), float(np.random.rand()))
    assert a == b

    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seedscope_restores_python_and_numpy_state():
    seed_everything(1234)
    expected = (py_random.random(), float(np.random.rand()))

    seed_everything(1234)
    with SeedScope(9999):
        # Burn a different amount of randomness inside the scope.
        py_random.random()
        py_random.random()
        np.random.rand(17)
    got = (py_random.random(), float(np.random.rand()))

    assert got == expected


def test_seedscope_restores_torch_state():
    torch = pytest.importorskip("torch")

    torch.manual_seed(1234)
    expected = torch.randn(4).tolist()

    torch.manual_seed(1234)
    with SeedScope(4321):
        torch.randn(11)
    got = torch.randn(4).tolist()

    assert got == expected


def test_seedscope_restores_on_exception():
    seed_everything(2)
    expected = float(np.random.rand())
    seed_everything(2)
    with pytest.raises(RuntimeError):
        with SeedScope(3):
            np.random.rand(5)
            raise RuntimeError("boom")
    assert float(np.random.rand()) == expected


def test_seedscope_without_seed_only_saves():
    """seed=None must not reseed anything, merely protect the outer state."""
    seed_everything(8)
    expected = float(np.random.rand())
    seed_everything(8)
    with SeedScope() as scope:
        assert scope.rng is None
        np.random.rand(3)
    assert float(np.random.rand()) == expected


def test_seedscope_exposes_a_generator():
    with SeedScope(123) as scope:
        x = scope.rng.normal(size=3)
    np.testing.assert_array_equal(x, make_rng(123).normal(size=3))


def test_capture_random_state_is_a_snapshot_not_a_view():
    seed_everything(5)
    state = capture_random_state()
    a = float(np.random.rand())
    np.random.rand(10)
    from atomlab.utils.seeds import restore_random_state

    restore_random_state(state)
    assert float(np.random.rand()) == a


# ==========================================================================
# timing
# ==========================================================================


def test_timer_accumulates_into_registry():
    reg = TimerRegistry()
    for _ in range(3):
        with Timer("block", registry=reg) as t:
            sum(range(1000))
    assert t.elapsed > 0.0
    rec = reg.get("block")
    assert rec is not None and rec.count == 3
    assert rec.total >= rec.best
    assert rec.worst >= rec.mean >= rec.best
    assert list(reg.totals()) == ["block"]


def test_timer_records_even_when_the_body_raises():
    reg = TimerRegistry()
    with pytest.raises(ValueError):
        with Timer("boom", registry=reg):
            raise ValueError
    assert reg.get("boom").count == 1


def test_timer_without_a_name_does_not_pollute_the_registry():
    reg = TimerRegistry()
    with Timer(registry=reg) as t:
        pass
    assert len(reg) == 0
    assert t.elapsed >= 0.0


def test_timed_decorator_bare_and_parameterised():
    reg = TimerRegistry()

    @timed(registry=reg)
    def cheap(x):
        return x * 2

    @timed(name="custom", registry=reg)
    def other():
        return None

    assert cheap(3) == 6
    cheap(4)
    other()
    assert reg.get(cheap.timer_name).count == 2
    assert reg.get("custom").count == 1
    assert cheap.__name__ == "cheap"


def test_global_registry_helpers():
    reset_timings()
    with Timer("global-test"):
        pass
    assert "global-test" in get_timings()
    assert "global-test" in REGISTRY.report()
    reset_timings("global-test")
    assert "global-test" not in get_timings()


def test_format_duration_units():
    assert format_duration(1.23e-5) == "12.3 us"
    assert format_duration(0.25) == "250 ms"
    assert format_duration(3.5) == "3.500 s"
    assert format_duration(90.0) == "1 m 30.0 s"
    assert format_duration(3725.0) == "1 h 02 m 05 s"
    assert format_duration(-0.25).startswith("-")
    assert format_duration(float("nan")) == "nan"


# ==========================================================================
# logging
# ==========================================================================


def test_get_logger_writes_to_stderr_under_the_package_root():
    log = get_logger("test-module")
    assert log.name == "atomlab.test-module"
    root = logging.getLogger("atomlab")
    assert root.handlers, "package logger must have a handler"
    assert any(getattr(h, "stream", None) is not None for h in root.handlers)
    import sys

    assert root.handlers[0].stream is sys.stderr
    assert root.propagate is False


def test_progress_reporter_is_throttled_by_time(caplog):
    """A huge interval must produce exactly one line: the final summary."""
    rep = ProgressReporter(total=100, label="steps", interval=1e6, unit="steps")
    with caplog.at_level(logging.INFO, logger="atomlab.progress"):
        for _ in range(100):
            rep.update()
        rep.finish()
    assert rep.count == 100
    assert rep.n_reports == 1
    assert "done 100/100" in caplog.text


def test_progress_reporter_forced_line_has_percent_and_eta(caplog):
    rep = ProgressReporter(total=10, label="md", interval=1e6)
    with caplog.at_level(logging.INFO, logger="atomlab.progress"):
        emitted = rep.update(5, extra="T=300.0 K", force=True)
    assert emitted is True
    assert "md: 5/10" in caplog.text
    assert "50.0%" in caplog.text
    assert "eta" in caplog.text
    assert "T=300.0 K" in caplog.text


def test_progress_reporter_without_total(caplog):
    rep = ProgressReporter(label="unknown-length", interval=1e6)
    with caplog.at_level(logging.INFO, logger="atomlab.progress"):
        rep.update(3, force=True)
        rep.finish()
    assert "unknown-length: 3" in caplog.text
    assert "%" not in caplog.text


def test_progress_reporter_rejects_bad_interval():
    with pytest.raises(ValueError):
        ProgressReporter(interval=0.0)


def test_progress_helper_wraps_an_iterable(caplog):
    with caplog.at_level(logging.INFO, logger="atomlab.progress"):
        out = list(progress(range(5), label="items", interval=1e6))
    assert out == [0, 1, 2, 3, 4]
    assert "done 5/5" in caplog.text
