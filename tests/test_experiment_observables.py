"""Load-bearing tests for the plain observables used by the experiments."""

from __future__ import annotations

import numpy as np
import pytest

from atomlab.types import Configuration
from experiments.observables_lib import PairBinObservable


def test_pair_bin_rejects_invalid_edges_and_truncated_cutoff():
    with pytest.raises(ValueError, match="1-D"):
        PairBinObservable(np.ones((2, 2)))
    with pytest.raises(ValueError, match="strictly increasing"):
        PairBinObservable(np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="inside the outermost"):
        PairBinObservable(np.array([0.0, 1.0, 2.0]), cutoff=1.9)


def test_pair_bin_counts_each_open_pair_once_and_pins_edge_convention():
    # Pair distances are exactly 1, 2 and 3 A. np.histogram assigns internal
    # edges to the bin on their right and includes the final right edge.
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        cell=np.eye(3) * 20.0,
        pbc=False,
        symbols=("Ar",),
    )
    observable = PairBinObservable(np.array([0.0, 1.0, 2.0, 3.0]), cutoff=3.0)
    assert np.array_equal(observable(cfg), np.array([0.0, 1.0, 2.0]))


def test_pair_bin_half_list_counts_periodic_self_images_once():
    # A one-atom simple-cubic cell has six nearest images at distance a.  A
    # half list contains the three unique pair interactions, which is the raw
    # pair-count convention used by every claim-bearing experiment.
    a = 2.0
    cfg = Configuration(
        positions=np.zeros((1, 3)), cell=np.eye(3) * a, pbc=True, symbols=("Ar",)
    )
    observable = PairBinObservable(np.array([1.9, 2.1]), cutoff=2.1)
    assert observable(cfg).tolist() == [3.0]


def test_pair_bin_trajectory_shape_and_gofr_normalisation():
    cfg = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        pbc=True,
        symbols=("Ar",),
    )
    observable = PairBinObservable(np.array([1.0, 2.0]), cutoff=2.0)
    counts = observable.evaluate_trajectory([cfg, cfg.copy()])
    assert counts.shape == (2, 1)
    assert np.array_equal(counts[:, 0], np.array([1.0, 1.0]))

    shell = (4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3)
    other_density = (cfg.n_atoms - 1) / cfg.volume
    expected = 1.0 / (0.5 * cfg.n_atoms * other_density * shell)
    assert observable.to_g_of_r(np.array([1.0]), cfg)[0] == pytest.approx(expected)


def test_pair_bin_gofr_rejects_nonperiodic_or_single_atom_normalisation():
    observable = PairBinObservable(np.array([1.0, 2.0]), cutoff=2.0)
    nonperiodic = Configuration(
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        pbc=False,
        symbols=("Ar",),
    )
    with pytest.raises(ValueError, match="finite positive cell volume"):
        observable.to_g_of_r(np.array([1.0]), nonperiodic)

    singleton = Configuration(
        positions=np.zeros((1, 3)), cell=np.eye(3) * 10.0,
        pbc=True, symbols=("Ar",),
    )
    with pytest.raises(ValueError, match="at least two atoms"):
        observable.to_g_of_r(np.array([0.0]), singleton)
