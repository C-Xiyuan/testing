#!/usr/bin/env python3
"""Regenerate the descriptive exp07 figure from tracked legacy aggregates.

This script does not rerun sampling or upgrade provenance. It exists so the
review PNG/PDF/SVG is reproducible from the deposited records and the current
truthful rendering code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp07_designed_counterexamples.run import make_figure
from experiments.observables_lib import PairBinObservable


class FigureContext:
    def figure_path(self, stem: str) -> Path:
        return ROOT / "figures" / f"exp07_designed_counterexamples_{stem}"


def load(relative: str):
    return json.loads((ROOT / "results" / relative).read_text())


def main() -> None:
    records = load("exp07_designed_counterexamples/records.json")
    manifest = load("exp07_designed_counterexamples/manifest.json")
    config = manifest["config"]
    observable_config = config["observable"]
    observable = PairBinObservable(
        np.linspace(
            observable_config["r_min"], observable_config["r_max"],
            observable_config["n_bins"] + 1,
        ),
        cutoff=config["potential"]["cutoff"],
    )
    make_figure(
        FigureContext(), records, observable,
        target_bin=np.asarray(observable_config["target_bin"], dtype=float),
    )
    print("wrote figures/exp07_designed_counterexamples_counterexamples.{png,pdf,svg}")


if __name__ == "__main__":
    main()
