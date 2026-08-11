"""Analytic reference potentials, designed perturbations, and the potential algebra.

The analytic potentials here are the *ground truth* of the study: because their
functional form is known everywhere in configuration space, the error field
``dU = U_model - U_reference`` is exactly computable rather than estimated, which
is what makes the response analysis in :mod:`atomlab.analysis.response` a
measurement rather than an inference.
"""

from .base import (
    Potential,
    ScaledPotential,
    SumPotential,
    ZeroPotential,
    check_forces,
    check_virial,
)
from .eam import EAM
from .harmonic import EinsteinCrystal, HarmonicPair
from .lennard_jones import LennardJones
from .morse import Morse
from .stillinger_weber import StillingerWeber

__all__ = [
    "Potential",
    "SumPotential",
    "ScaledPotential",
    "ZeroPotential",
    "check_forces",
    "check_virial",
    "LennardJones",
    "Morse",
    "StillingerWeber",
    "EAM",
    "EinsteinCrystal",
    "HarmonicPair",
]
