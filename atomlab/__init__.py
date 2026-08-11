"""atomlab -- controlled study of how machine-learned potential errors reach physics.

See ``docs/design.md`` for the interface contract and ``docs/theory.md`` for the
linear-response derivation that motivates the whole package.
"""

from .types import Configuration, Dataset, Result, Trajectory
from .units import KB, MVV2E, beta

__version__ = "0.1.0"

__all__ = ["Configuration", "Result", "Trajectory", "Dataset", "KB", "MVV2E", "beta"]
