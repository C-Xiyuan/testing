"""Keep the generated manuscript variant complete and cross-reference-safe."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_tex2md():
    path = Path(__file__).resolve().parents[1] / "paper" / "tex2md.py"
    spec = importlib.util.spec_from_file_location("paper_tex2md", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tex2md_emits_every_table_and_resolves_new_references(tmp_path):
    module = load_tex2md()
    module.MD = tmp_path / "main.md"
    module.main()
    rendered = module.MD.read_text()

    assert "Table tab:" not in rendered
    assert "### Table I" in rendered
    assert "Legacy fixed-cell construction clusters" in rendered
    assert "### Table II" in rendered
    assert "Historical warning-light row counts" in rendered
    assert "### Table X" in rendered
    assert "*(246 words.)*" in rendered
