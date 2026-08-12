"""Regression tests for experiment provenance and atomic evidence output."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments import common


def snapshot(commit="launch", digest="same", *, source_dirty=False):
    entries = ([{"status": " M", "path": "atomlab/example.py"}]
               if source_dirty else [])
    return {
        "captured_at": "2026-01-01T00:00:00+0000",
        "git_commit": commit,
        "git_branch": "test",
        "git_dirty": source_dirty,
        "source_dirty": source_dirty,
        "source_reconstructable": True,
        "head_tree": f"tree-{commit}",
        "code": {
            "dirty_paths": [entry["path"] for entry in entries],
            "dirty_entries": entries,
            "source_dirty_entries": entries,
            "unreconstructable_source_entries": [],
            "n_source_files": 1,
            "sha256": digest,
            "scope": ["atomlab", "experiments", "scripts", "protocols",
                      "pyproject.toml"],
        },
    }


def test_save_npz_is_atomic_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    ctx = common.ExperimentContext("atomic", {}, output_dir=tmp_path)
    target = ctx.save_npz("raw", x=np.arange(5), y=np.eye(2))
    assert target == tmp_path / "raw.npz"
    assert not (tmp_path / "raw.npz.tmp").exists()
    with np.load(target) as data:
        assert np.array_equal(data["x"], np.arange(5))
        assert np.array_equal(data["y"], np.eye(2))


def test_manifest_records_launch_source_not_finish_head(tmp_path, monkeypatch):
    launch = snapshot("launch", "same")
    finish = snapshot("finish", "same")
    finish["head_tree"] = launch["head_tree"]
    states = iter([launch, finish])
    monkeypatch.setattr(common, "_source_snapshot", lambda: next(states))
    ctx = common.ExperimentContext("provenance", {}, output_dir=tmp_path)
    manifest = json.loads(ctx.write_manifest().read_text())
    assert manifest["git_commit"] == "launch"
    assert manifest["source_start"]["git_commit"] == "launch"
    assert manifest["source_end"]["git_commit"] == "finish"
    assert manifest["source_code_stable"] is True
    assert manifest["provenance_valid"] is True


def test_manifest_invalidates_source_changed_during_run(tmp_path, monkeypatch):
    states = iter([snapshot("launch", "before"), snapshot("finish", "after")])
    monkeypatch.setattr(common, "_source_snapshot", lambda: next(states))
    ctx = common.ExperimentContext("changed", {}, output_dir=tmp_path)
    manifest = json.loads(ctx.write_manifest().read_text())
    assert manifest["source_code_stable"] is False
    assert manifest["provenance_valid"] is False
    assert manifest["status"] == "invalid_provenance"


def test_manifest_invalidates_head_tree_changed_during_run(tmp_path, monkeypatch):
    launch = snapshot("launch", "same")
    finish = snapshot("finish", "same")
    states = iter([launch, finish])
    monkeypatch.setattr(common, "_source_snapshot", lambda: next(states))
    ctx = common.ExperimentContext("head-changed", {}, output_dir=tmp_path)
    manifest = json.loads(ctx.write_manifest().read_text())
    assert manifest["source_code_stable"] is True
    assert manifest["provenance_valid"] is False
    assert manifest["status"] == "invalid_provenance"


def test_production_main_refuses_dirty_numerical_source(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot",
                        lambda: snapshot(source_dirty=True))
    monkeypatch.setattr(common.sys, "argv", ["run.py", "--output", str(tmp_path)])
    called = False

    def run(_ctx):
        nonlocal called
        called = True
        return {}

    with pytest.raises(RuntimeError, match="dirty numerical source"):
        common.main(run, default_config={}, name="dirty")
    assert called is False


def test_quick_run_is_explicitly_smoke_only_on_dirty_source(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot",
                        lambda: snapshot(source_dirty=True))
    monkeypatch.setattr(
        common.sys, "argv", ["run.py", "--quick", "--output", str(tmp_path)]
    )
    result = common.main(lambda _ctx: {"ok": True}, default_config={}, name="smoke")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert result == {"ok": True}
    assert manifest["status"] == "smoke_only"
    assert manifest["provenance_valid"] is False


def test_confirmatory_seed_rejects_legacy_and_unfrozen_values(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    ctx = common.ExperimentContext("confirmatory", {}, output_dir=tmp_path, seed=0)
    with pytest.raises(RuntimeError, match="legacy seed"):
        common.require_fresh_production_seed(ctx, expected=20260812)
    ctx.seed = 17
    with pytest.raises(RuntimeError, match="prospectively frozen"):
        common.require_fresh_production_seed(ctx, expected=20260812)
    ctx.seed = 20260812
    common.require_fresh_production_seed(ctx, expected=20260812)


def test_smoke_run_may_reuse_seed_but_cannot_be_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    ctx = common.ExperimentContext(
        "smoke-seed", {}, output_dir=tmp_path, seed=0, quick=True
    )
    common.require_fresh_production_seed(ctx, expected=20260812)


def write_protocol(path, executable_config):
    document = {
        "protocol": "test_protocol_v2",
        "protocol_version": 2,
        "master_seed": 20260812,
        "executable_config": executable_config,
    }
    path.write_text(json.dumps(document))
    return document


def test_frozen_protocol_is_independent_exact_artifact_and_is_manifested(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    frozen = {"analysis": {"bound": 0.5}, "replication": {"chains": 8}}
    protocol_path = tmp_path / "protocol.json"
    write_protocol(protocol_path, frozen)
    output = tmp_path / "output"
    ctx = common.ExperimentContext(
        "protocol", frozen, output_dir=output, seed=20260812
    )
    common.require_frozen_protocol(
        ctx, protocol_path=protocol_path,
        expected_protocol="test_protocol_v2", expected_version=2,
    )
    manifest = json.loads(ctx.write_manifest().read_text())
    attestation = manifest["frozen_protocol"]
    assert attestation["path"] == str(protocol_path)
    assert attestation["sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert attestation["config_matches"] is True
    assert attestation["seed_matches"] is True
    assert attestation["master_seed"] == 20260812


@pytest.mark.parametrize(
    "config, message",
    [
        ({"analysis": {"bound": 0.5}, "replication": {"chains": 8},
          "posthoc": True}, "unexpected key"),
        ({"analysis": {"bound": 0.5}}, "missing key"),
        ({"analysis": {"bound": 0.5}, "replication": {"chains": 8.0}},
         "expected int"),
    ],
)
def test_frozen_protocol_rejects_extra_missing_and_type_drift(
    tmp_path, monkeypatch, config, message,
):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    frozen = {"analysis": {"bound": 0.5}, "replication": {"chains": 8}}
    protocol_path = tmp_path / "protocol.json"
    write_protocol(protocol_path, frozen)
    ctx = common.ExperimentContext(
        "protocol-drift", config, output_dir=tmp_path / "output", seed=20260812
    )
    with pytest.raises(RuntimeError, match=message):
        common.require_frozen_protocol(ctx, protocol_path=protocol_path)


def test_protocol_artifact_does_not_follow_mutated_runtime_defaults(tmp_path, monkeypatch):
    """Regression: passing DEFAULTS itself used to make every mutation 'frozen'."""
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    original = {"analysis": {"bound": 0.5}}
    protocol_path = tmp_path / "protocol.json"
    write_protocol(protocol_path, original)
    mutable_defaults = json.loads(json.dumps(original))
    mutable_defaults["analysis"]["bound"] = 0.7
    ctx = common.ExperimentContext(
        "protocol-independent", mutable_defaults, output_dir=tmp_path / "output"
    )
    ctx.config["analysis"]["bound"] = 0.7
    with pytest.raises(RuntimeError, match="violates the frozen protocol artifact"):
        common.require_frozen_protocol(ctx, protocol_path=protocol_path)


def test_protocol_artifact_owns_the_production_master_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    protocol_path = tmp_path / "protocol.json"
    write_protocol(protocol_path, {"analysis": {"bound": 0.5}})
    ctx = common.ExperimentContext(
        "protocol-seed", {"analysis": {"bound": 0.5}},
        output_dir=tmp_path / "output", seed=17,
    )
    with pytest.raises(RuntimeError, match="master_seed"):
        common.require_frozen_protocol(ctx, protocol_path=protocol_path)


def test_quick_protocol_override_is_smoke_only_and_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    protocol_path = tmp_path / "protocol.json"
    write_protocol(protocol_path, {"analysis": {"bound": 0.5}})
    ctx = common.ExperimentContext(
        "protocol-smoke", {"analysis": {"bound": 9.0}},
        output_dir=tmp_path / "output", quick=True,
    )
    common.require_frozen_protocol(ctx, protocol_path=protocol_path)
    assert ctx._frozen_protocol["config_matches"] is False


@pytest.mark.parametrize(
    "run_path, protocol_path",
    [
        ("experiments/exp09_calibration_replication/run.py", "protocols/exp09_v2.json"),
        ("experiments/exp10_endtoend_consistency/run.py", "protocols/exp10_v2.json"),
        ("experiments/exp11_counterexample_replication/run.py", "protocols/exp11_v2.json"),
    ],
)
def test_code_defaults_have_not_drifted_from_canonical_protocol(run_path, protocol_path):
    tree = ast.parse((common.REPO_ROOT / run_path).read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEFAULTS"
                for target in node.targets)
    )
    defaults = ast.literal_eval(assignment.value)
    protocol = json.loads((common.REPO_ROOT / protocol_path).read_text())
    assert common._canonical_json(defaults) == common._canonical_json(
        protocol["executable_config"]
    )


def test_porcelain_parser_preserves_spaces_and_both_rename_paths(monkeypatch):
    payload = (
        " M experiments/a b.py\0"
        "R  experiments/new.py\0docs/old.py\0"
        "?? protocols/frozen plan.json\0"
    )

    def fake_git(*args, **_kwargs):
        assert "-z" in args
        return payload

    monkeypatch.setattr(common, "_git", fake_git)
    entries = common._status_entries()
    assert entries[0]["paths"] == ["experiments/a b.py"]
    assert entries[1]["paths"] == ["experiments/new.py", "docs/old.py"]
    assert entries[2]["paths"] == ["protocols/frozen plan.json"]
    assert common._entry_touches_numerical_source(entries[0])
    assert common._entry_touches_numerical_source(entries[1])
    assert common._entry_touches_numerical_source(entries[2])


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def initialise_source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "atomlab").mkdir(parents=True)
    (repo / "experiments").mkdir()
    (repo / "scripts").mkdir()
    (repo / "protocols").mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "provenance@example.invalid")
    git(repo, "config", "user.name", "Provenance Test")
    return repo


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_head_blob_comparison_catches_changes_hidden_from_porcelain(
    tmp_path, monkeypatch, index_flag,
):
    repo = initialise_source_repo(tmp_path)
    source = repo / "atomlab" / "hidden.py"
    source.write_text("VALUE = 1\n")
    git(repo, "add", "atomlab/hidden.py")
    git(repo, "commit", "-m", "baseline")
    git(repo, "update-index", index_flag, "atomlab/hidden.py")
    source.write_text("VALUE = 999\n")
    assert git(repo, "status", "--porcelain") == ""

    monkeypatch.setattr(common, "REPO_ROOT", repo)
    report = common._code_digest()
    mismatch = next(
        item for item in report["head_verification"]["mismatches"]
        if item["path"] == "atomlab/hidden.py"
    )
    assert mismatch["reason"] == "content_differs_from_head"
    assert mismatch["index_flags"] is not None
    assert report["source_dirty_entries"]
    assert report["unreconstructable_source_entries"]


def test_head_tree_comparison_checks_file_mode_even_when_index_hides_it(
    tmp_path, monkeypatch,
):
    repo = initialise_source_repo(tmp_path)
    source = repo / "experiments" / "mode.py"
    source.write_text("VALUE = 1\n")
    source.chmod(0o644)
    git(repo, "add", "experiments/mode.py")
    git(repo, "commit", "-m", "baseline")
    git(repo, "update-index", "--assume-unchanged", "experiments/mode.py")
    source.chmod(0o755)

    monkeypatch.setattr(common, "REPO_ROOT", repo)
    report = common._code_digest()
    mismatch = next(
        item for item in report["head_verification"]["mismatches"]
        if item["path"] == "experiments/mode.py"
    )
    assert mismatch["reason"] == "mode_or_file_type_differs_from_head"
    assert mismatch["head_mode"] == "100644"
    assert mismatch["worktree_mode"] == "100755"


def test_committed_symlink_is_reconstructable_but_hidden_target_change_is_not(
    tmp_path, monkeypatch,
):
    repo = initialise_source_repo(tmp_path)
    (repo / "atomlab" / "target_a.py").write_text("A = 1\n")
    (repo / "atomlab" / "target_b.py").write_text("B = 2\n")
    link = repo / "scripts" / "linked.py"
    link.symlink_to("../atomlab/target_a.py")
    git(repo, "add", "atomlab/target_a.py", "atomlab/target_b.py", "scripts/linked.py")
    git(repo, "commit", "-m", "baseline")
    monkeypatch.setattr(common, "REPO_ROOT", repo)
    clean = common._code_digest()
    assert clean["unreconstructable_source_entries"] == []

    git(repo, "update-index", "--skip-worktree", "scripts/linked.py")
    link.unlink()
    link.symlink_to("../atomlab/target_b.py")
    hidden = common._code_digest()
    mismatch = next(
        item for item in hidden["head_verification"]["mismatches"]
        if item["path"] == "scripts/linked.py"
    )
    assert mismatch["reason"] == "content_differs_from_head"
    assert hidden["source_dirty_entries"]
    assert hidden["unreconstructable_source_entries"]


def test_production_context_requires_fresh_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    (tmp_path / "stale.json").write_text("{}")
    with pytest.raises(RuntimeError, match="fresh and empty"):
        common.ExperimentContext("stale", {}, output_dir=tmp_path)


def test_run_lock_rejects_concurrent_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "_source_snapshot", lambda: snapshot())
    first = common.ExperimentContext("one", {}, output_dir=tmp_path, quick=True)
    second = common.ExperimentContext("two", {}, output_dir=tmp_path, quick=True)
    first.acquire_run_lock()
    try:
        with pytest.raises(RuntimeError, match="another run holds"):
            second.acquire_run_lock()
    finally:
        first.release_run_lock()


def test_manifest_inventories_only_current_context_writes(tmp_path, monkeypatch):
    states = iter([snapshot(), snapshot()])
    monkeypatch.setattr(common, "_source_snapshot", lambda: next(states))
    ctx = common.ExperimentContext("inventory", {}, output_dir=tmp_path)
    (tmp_path / "unregistered.json").write_text("{\"stale\": true}")
    ctx.save_json("current", {"ok": True})
    manifest = json.loads(ctx.write_manifest().read_text())
    assert [entry["path"] for entry in manifest["artifacts"]] == ["current.json"]
