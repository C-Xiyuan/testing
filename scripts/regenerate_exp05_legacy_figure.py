#!/usr/bin/env python3
"""Regenerate the descriptive exp05 figure from tracked legacy aggregates.

This script does not rerun sampling or create a population-level estimand. It
re-renders the deposited fixed-zoo records and summary with the current,
explicit row-resampling-sensitivity labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp05_proxy_correlation.run import make_figure


class FigureContext:
    def figure_path(self, stem: str) -> Path:
        return ROOT / "figures" / f"exp05_proxy_correlation_{stem}"


def load(relative: str):
    return json.loads((ROOT / "results" / relative).read_text())


def main() -> None:
    records = load("exp05_proxy_correlation/records.json")
    summary = load("exp05_proxy_correlation/summary.json")
    make_figure(FigureContext(), records, summary)
    print("wrote figures/exp05_proxy_correlation_proxy_correlation.{png,pdf,svg}")


if __name__ == "__main__":
    main()
