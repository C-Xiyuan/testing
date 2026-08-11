"""Behler-Parrinello neural network potential (PyTorch, CPU).

The model is the original Behler-Parrinello construction: every atom's
environment is mapped by an atom-centred symmetry-function descriptor
(:class:`~atomlab.models.descriptors.acsf.ACSF`) onto a fixed-length invariant
feature vector ``G_i``, a small multilayer perceptron -- one per **species** --
maps that vector to an atomic energy, and the total energy is the sum::

    E = sum_i [ shift[s_i] + scale * MLP_{s_i}( (G_i - mean[s_i]) / sd[s_i] ) ]

Because every ``G_i`` is invariant under translation, rotation, permutation of
identical atoms and the choice of periodic image, so is ``E``, exactly and by
construction rather than by training.

Where the derivatives come from, and why they come from there
------------------------------------------------------------

Forces need ``dE/dr``.  There are two routes to it and this module deliberately
uses exactly one of them:

1. **The route used here.**  ``torch.autograd`` differentiates the *network
   only*, giving ``dE/dG_id`` -- an ``(N, D)`` array, cheap, and exact to
   machine precision.  The descriptor then supplies ``dG_id/dr_j`` analytically
   and the chain rule is closed by
   :meth:`~atomlab.models.base.DescriptorOutput.forces_from_energy_gradient`.
2. **The route not used here.**  Re-implement the descriptor in torch and let
   autograd differentiate through the symmetry functions as well.

Route 2 is not merely redundant, it is dangerous.  The ACSF derivatives are
already exact and already validated against finite differences in
``tests/test_acsf.py``; differentiating a second, independent implementation of
the same map means two things must agree that nothing checks against each
other.  Worse, the tempting hybrid -- autograd through a torch descriptor
*plus* the analytic ``dG/dr`` term -- double-counts every neighbour
contribution, producing forces exactly twice too large in the pure-radial part
and some other factor in the angular part.  That bug fits an energy curve
perfectly well and only shows up as wrong dynamics, which is precisely the
class of silent failure this repository exists to study.  So: **autograd stops
at the descriptor boundary.**  Nothing in this module ever backpropagates
through ``ACSF.compute``.

The same discipline applies during training.  The force *loss* must be
differentiable with respect to the network weights, so ``dE/dG`` is taken with
``create_graph=True`` and the chain-rule contraction is then done with torch
ops on the *constant* descriptor-derivative tensor.  The descriptor derivatives
are data, not part of the graph -- they do not depend on the weights.

Virial
------

Assembled from the pair-resolved descriptor derivatives.  Under the strain
``r -> (1 + eps) r`` acting on positions *and* lattice vectors, every pair
displacement ``D_p = r_j + S_p @ cell - r_i`` transforms as
``D_p -> (1 + eps) D_p``, and the energy depends on the geometry only through
those displacements.  Hence

    dU/d(eps_ab) = sum_p (dU/dD_p)_a * D_p,b
    W_ab = -dU/d(eps_ab) = sum_p f_p,a * D_p,b ,   f_p = -dU/dD_p

which is the pair-outer-product form of ``docs/design.md`` §3 (there written as
``sum_{i<j} f_ij (x) r_ij`` with ``r_ij = r_i - r_j``; both signs flip together
here because ``D`` points from ``i`` to ``j``, so the product is the same).
``dU/dD_p`` is ``sum_d dE/dG[i_p,d] * dG[i_p,d]/dr[j_p]``, i.e. exactly the row
of ``DescriptorOutput.derivatives`` belonging to that pair.

One subtlety makes this exact rather than approximate.  ``ACSF`` sums
contributions reaching the same neighbour through *several* periodic images
into a single row, which would lose the per-image ``D_p``.  This module
therefore refuses (via :func:`~atomlab.cell.check_minimum_image`) any cell
narrower than ``2 * cutoff``, and under that condition a given ordered pair
``(i, j)`` can appear through at most one image: two images would differ by a
lattice vector of length ``< 2 * cutoff``, and every nonzero lattice vector is
at least as long as the smallest perpendicular cell width.  The pair
displacement is then unambiguously the minimum image of ``r_j - r_i``.  This is
the same restriction ``docs/design.md`` §4 already places on every potential in
the package, and it is enforced rather than assumed.

Sizing for four CPU cores
-------------------------

Two hidden layers of 64 units on 36 ACSF features is ~6.9k parameters per
species.  That is at the small end of published BPNN work (which routinely uses
3 x 50-100 on 100+ symmetry functions) and it is a deliberate choice: this
study needs every model in the zoo trainable to useful accuracy on a few
thousand 64-atom configurations in minutes on four shared CPU cores.  The
descriptor evaluation, not the network, dominates the cost, and it is done
once up front and cached for the whole fit.

Units
-----

Metal units throughout: positions in A, energies in eV, forces in eV/A, virial
in eV.  Descriptor features and standardised features are dimensionless.

Precision
---------

``dtype='float64'`` is the default and is what the derivative tests require:
float32 autograd noise on ``dE/dG`` is of order ``1e-4`` eV/A in the assembled
forces, which is fine for training and useless for validating a chain rule.
float32 roughly halves the training time and is available via ``dtype``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..cell import check_minimum_image, minimum_image
from ..types import Configuration, Dataset, Result
from .base import Descriptor, DescriptorOutput, FitReport, MLModel
from .descriptors.acsf import ACSF

__all__ = ["BPNN", "DEFAULT_TORCH_THREADS", "set_torch_threads"]


#: Intra-op thread count used by :meth:`BPNN.fit`.
#:
#: Two, and the choice is measured rather than conventional.  Fitting 160
#: 32-atom argon configurations for 30 epochs at batch 16, on this 4-core box:
#:
#: ===========  ==========  ============================
#: threads      fit time    val force RMSE
#: ===========  ==========  ============================
#: 1            5.8 s       0.00147 eV/A
#: 2            3.8 s       0.00147 eV/A
#: 4            297 s       0.00147 eV/A
#: ===========  ==========  ============================
#:
#: The 4-thread figure is not a typo and is not contention with the other work
#: on this machine: the tensors here are small (a batch is ~7k pair rows x 36
#: features), and torch's OpenMP fork/join on shapes that size costs far more
#: than the arithmetic it splits.  Two threads is the measured optimum and also
#: leaves half the machine to the other models being fitted alongside.
#:
#: **Single-point evaluation wants one thread, not two.**  A ``compute()`` call
#: on a 32-atom cell takes ~5-6 ms at 1 thread, ~10-17 ms at 2 and ~60-90 ms at
#: 4 across repeated runs on this (shared) box; the spread is machine load, the
#: ordering is not.  The ACSF descriptor accounts for ~4 ms of that and is
#: unaffected by the thread count, so the variation is all torch.  Anything
#: driving molecular dynamics with a fitted ``BPNN`` should therefore call
#: ``set_torch_threads(1)`` and get its parallelism from running independent
#: trajectories, not from inside torch.  The asymmetry is why this module never
#: sets the thread count at import time: there is no single right value for
#: both phases, so the choice is left where it can be made per phase.
DEFAULT_TORCH_THREADS = 2


def set_torch_threads(n_threads: int | None = DEFAULT_TORCH_THREADS) -> None:
    """Set torch's intra-op thread count (``None`` leaves it alone).

    This is *global* torch state shared with every other model in the process,
    so it is exposed as a named function and applied only inside
    :meth:`BPNN.fit` (which restores the previous value on the way out) rather
    than at import time.
    """
    if n_threads is not None:
        torch.set_num_threads(int(n_threads))


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def _activation(name: str) -> nn.Module:
    """Instantiate an activation by name; all choices are C-infinity.

    A ReLU network has a discontinuous second derivative, so its *forces* are
    piecewise constant in the worst places and a force-fitted loss is not
    differentiable.  Only smooth activations are offered.
    """
    try:
        return _ACTIVATIONS[str(name).lower()]()
    except KeyError:
        raise ValueError(
            f"unknown activation {name!r}; known: {sorted(_ACTIVATIONS)}"
        ) from None


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------


class _BPNNNet(nn.Module):
    """Per-species MLP stack plus the standardisation and energy scaling.

    The standardisation statistics and the energy shift/scale live here as
    registered buffers rather than as plain attributes, so ``state_dict()``
    carries them and a saved model cannot be reloaded with the weights of one
    fit and the feature statistics of another.

    Parameters
    ----------
    n_species : int
        Number of species type indices ``0 .. n_species-1``.
    n_features : int
        Descriptor dimension ``D``.
    hidden : sequence of int
        Hidden layer widths, applied to every species MLP.
    activation : str
        Key into :data:`_ACTIVATIONS`.
    """

    def __init__(
        self,
        n_species: int,
        n_features: int,
        hidden: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.n_species = int(n_species)
        self.n_features = int(n_features)
        self.hidden = tuple(int(h) for h in hidden)

        nets = []
        for _ in range(self.n_species):
            layers: list[nn.Module] = []
            width = self.n_features
            for h in self.hidden:
                layers.append(nn.Linear(width, h))
                layers.append(_activation(activation))
                width = h
            layers.append(nn.Linear(width, 1))
            nets.append(nn.Sequential(*layers))
        self.nets = nn.ModuleList(nets)

        self.register_buffer("feature_mean", torch.zeros(self.n_species, self.n_features))
        self.register_buffer("feature_scale", torch.ones(self.n_species, self.n_features))
        self.register_buffer("energy_shift", torch.zeros(self.n_species))
        self.register_buffer("energy_scale", torch.ones(()))

    def forward(self, features: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """Per-atom energies in eV.

        Parameters
        ----------
        features : Tensor, shape (N, D)
            Raw (unstandardised) descriptor values.  Standardisation happens
            *inside* the graph so that ``dE/dG`` obtained by differentiating
            with respect to this tensor is a derivative with respect to the
            **raw** feature -- which is what the descriptor's ``dG/dr`` is a
            derivative of.  Doing it outside would silently drop the ``1/sd``
            factor from every force.
        species : Tensor, shape (N,), int64

        Returns
        -------
        Tensor, shape (N,)
            Atomic energies in eV.
        """
        g = (features - self.feature_mean[species]) / self.feature_scale[species]
        if self.n_species == 1:
            raw = self.nets[0](g).squeeze(-1)
        else:
            raw = features.new_zeros(features.shape[0])
            for s in range(self.n_species):
                mask = species == s
                if bool(mask.any()):
                    # index_put_ on a freshly allocated tensor; differentiable
                    # and double-differentiable, which the force loss needs.
                    raw = raw.masked_scatter(mask, self.nets[s](g[mask]).squeeze(-1))
        return self.energy_shift[species] + self.energy_scale * raw


# --------------------------------------------------------------------------
# cached featurisation
# --------------------------------------------------------------------------


@dataclass
class _Cached:
    """One configuration, featurised once and reused for every epoch.

    Attributes
    ----------
    features : Tensor, shape (N, D)
    derivatives : Tensor, shape (P, D, 3)
        ``dG[pair_i[p], d] / dr[pair_j[p]]`` in 1/A.  Constant data, never part
        of the autograd graph.
    pair_i, pair_j : Tensor, shape (P,), int64
    species : Tensor, shape (N,), int64
    energy : float
        Reference total energy in eV.
    forces : Tensor, shape (N, 3)
        Reference forces in eV/A.
    n_atoms : int
    """

    features: torch.Tensor
    derivatives: torch.Tensor | None
    pair_i: torch.Tensor | None
    pair_j: torch.Tensor | None
    species: torch.Tensor
    energy: float
    forces: torch.Tensor | None
    n_atoms: int


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class BPNN(MLModel):
    """Behler-Parrinello neural network potential.

    Parameters
    ----------
    descriptor : Descriptor
        Usually an :class:`~atomlab.models.descriptors.acsf.ACSF`.  Must return
        derivatives; the model's ``cutoff`` is taken from it.
    hidden : sequence of int
        Hidden layer widths of every per-species MLP.  Default ``(64, 64)``.
    activation : {'silu', 'tanh', 'softplus'}
        Smooth activation; see :func:`_activation`.
    n_species : int, optional
        Number of species type indices the model covers.  Inferred from the
        descriptor when it exposes a ``species`` attribute.
    seed : int
        Seeds the weight initialisation **and** the minibatch shuffling.  Two
        ``BPNN``s that differ only in ``seed`` are independent ensemble
        members; this is the ensembling hook
        (:mod:`atomlab.models.ensemble` consumes it).
    dtype : {'float64', 'float32'}
        Working precision.  See the module docstring.
    allow_untrained : bool
        Permit :meth:`compute` before :meth:`fit`.  Needed by the derivative
        tests, which must validate the chain rule on a *randomly initialised*
        network -- a trained one can mask a derivative bug by having small
        gradients where the test looks.
    name : str
        Identifier used in tables and figures.

    Attributes
    ----------
    net : _BPNNNet
        The torch module holding weights, feature statistics and energy scaling.
    cutoff : float
        Descriptor cutoff in angstrom.

    Notes
    -----
    ``compute`` refuses any periodic cell narrower than ``2 * cutoff``; see the
    module docstring for why the virial assembly requires it.
    """

    def __init__(
        self,
        descriptor: Descriptor,
        *,
        hidden: Sequence[int] = (64, 64),
        activation: str = "silu",
        n_species: int | None = None,
        seed: int = 0,
        dtype: str = "float64",
        allow_untrained: bool = False,
        name: str = "bpnn",
    ) -> None:
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
        hidden = tuple(int(h) for h in hidden)
        if not hidden or any(h < 1 for h in hidden):
            raise ValueError(f"hidden must be a non-empty sequence of positive widths, got {hidden}")

        self.descriptor = descriptor
        self.cutoff = float(descriptor.cutoff)
        self.name = str(name)
        self.seed = int(seed)
        self.dtype = dtype
        self.torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        self.allow_untrained = bool(allow_untrained)
        self.is_fitted = False
        self.activation = str(activation)
        self.hidden = hidden

        if n_species is None:
            sp = getattr(descriptor, "species", None)
            if sp is None:
                raise ValueError(
                    "n_species could not be inferred from the descriptor; pass it explicitly"
                )
            n_species = int(np.max(np.asarray(sp))) + 1
        self.n_species = int(n_species)
        if self.n_species < 1:
            raise ValueError(f"n_species must be >= 1, got {self.n_species}")

        # Seed locally and restore, so constructing a model never perturbs the
        # global torch stream another module may be relying on.
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            self.net = _BPNNNet(
                n_species=self.n_species,
                n_features=int(descriptor.n_features),
                hidden=hidden,
                activation=self.activation,
            )
        finally:
            torch.random.set_rng_state(state)
        self.net.to(self.torch_dtype)
        self.net.eval()

    # -- introspection -----------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of trainable weights and biases, summed over species."""
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def set_scaling(self, shift, scale: float) -> None:
        """Set the per-species energy offset (eV/atom) and the output scale (eV).

        The MLP produces an O(1) number; the physical atomic energy is
        ``shift[s] + scale * raw``.  Fitting a raw network straight onto
        energies of several eV works badly because the output layer has to span
        the offset and the fluctuation at once, and the offset carries no
        information about the forces.
        """
        with torch.no_grad():
            sh = np.broadcast_to(np.asarray(shift, dtype=np.float64), (self.n_species,))
            self.net.energy_shift.copy_(torch.as_tensor(np.array(sh), dtype=self.torch_dtype))
            self.net.energy_scale.fill_(float(scale))

    def set_feature_statistics(self, mean: np.ndarray, scale: np.ndarray) -> None:
        """Store the per-species feature standardisation, shapes ``(S, D)``.

        ACSF components span many orders of magnitude (a narrow Gaussian and a
        broad one are not comparable), and an unstandardised input makes the
        first layer's effective learning rate differ per column by that same
        factor.  The statistics are fitted on the training set only and stored
        with the model, so evaluation is reproducible after reload.
        """
        d = self.net.n_features
        with torch.no_grad():
            self.net.feature_mean.copy_(
                torch.as_tensor(
                    np.broadcast_to(np.asarray(mean, dtype=np.float64), (self.n_species, d)).copy(),
                    dtype=self.torch_dtype,
                )
            )
            self.net.feature_scale.copy_(
                torch.as_tensor(
                    np.broadcast_to(np.asarray(scale, dtype=np.float64), (self.n_species, d)).copy(),
                    dtype=self.torch_dtype,
                )
            )

    # -- the Potential interface -------------------------------------------

    def _validate(self, configuration: Configuration) -> None:
        if configuration.species.size and int(configuration.species.max()) >= self.n_species:
            raise ValueError(
                f"configuration contains species index {int(configuration.species.max())} "
                f"but {self.name} was built for {self.n_species} species"
            )
        if np.asarray(configuration.pbc).any():
            # Required for the pair-resolved virial (see module docstring) and
            # by the package-wide contract in docs/design.md section 4.
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Evaluate energy, forces and virial.

        Parameters
        ----------
        configuration : Configuration
            Geometry in angstrom.  Not modified.
        forces, virial : bool
            When both are False the descriptor is evaluated without derivatives
            and the network runs under ``no_grad`` -- roughly 3x faster, which
            matters because :meth:`Potential.numerical_forces` takes that path
            ``6N`` times.

        Returns
        -------
        Result
            ``energy`` eV, ``forces`` ``(N, 3)`` eV/A, ``virial`` ``(3, 3)`` eV,
            ``energies`` ``(N,)`` eV.
        """
        if not self.allow_untrained:
            self._require_fitted()
        self._validate(configuration)

        want_deriv = bool(forces or virial)
        desc = self.descriptor.compute(configuration, derivatives=want_deriv)

        feat = torch.tensor(
            desc.features, dtype=self.torch_dtype, requires_grad=want_deriv
        )
        species = torch.as_tensor(configuration.species.astype(np.int64))

        if want_deriv:
            e_atom = self.net(feat, species)
            total = e_atom.sum()
            # Autograd stops here: dE/dG only.  The descriptor supplies dG/dr.
            (dE_dG,) = torch.autograd.grad(total, feat, create_graph=False)
            dE_dG_np = dE_dG.detach().to(torch.float64).numpy()
        else:
            with torch.no_grad():
                e_atom = self.net(feat, species)
                total = e_atom.sum()
            dE_dG_np = None

        f_out = np.zeros((configuration.n_atoms, 3))
        w_out = None
        if want_deriv:
            f_out = desc.forces_from_energy_gradient(dE_dG_np)
        if virial:
            w_out = self._virial_from_pairs(configuration, desc, dE_dG_np)

        return Result(
            energy=float(total.detach().to(torch.float64).item()),
            forces=f_out,
            virial=w_out,
            energies=e_atom.detach().to(torch.float64).numpy(),
        )

    def _virial_from_pairs(
        self,
        configuration: Configuration,
        desc: DescriptorOutput,
        dE_dG: np.ndarray,
    ) -> np.ndarray:
        """``W_ab = sum_p f_p,a D_p,b`` over the descriptor's (centre, mover) pairs.

        Parameters
        ----------
        configuration : Configuration
        desc : DescriptorOutput
            Must carry derivatives.
        dE_dG : ndarray, shape (N, D)
            ``dE/dG`` in eV (features are dimensionless).

        Returns
        -------
        ndarray, shape (3, 3)
            Virial in eV.

        Notes
        -----
        ``f_p = -dU/dD_p = -sum_d dE/dG[i_p,d] dG[i_p,d]/dr[j_p]`` and ``D_p``
        is the pair displacement pointing from the centre ``i_p`` to the mover
        ``j_p``.  Self rows (``i_p == j_p``) have ``D_p = 0`` and drop out, as
        they must: a self-derivative is not attached to any displacement.  The
        minimum image is the correct ``D_p`` only because ``_validate`` has
        already refused cells in which a pair could reach through two images.
        """
        pi = np.asarray(desc.pair_i)
        if pi.size == 0:
            return np.zeros((3, 3))
        pj = np.asarray(desc.pair_j)
        # (P, 3): dU/dD_p, contracted over the descriptor dimension.
        dU_dD = np.einsum("pd,pda->pa", dE_dG[pi], desc.derivatives, optimize=True)
        dr = configuration.positions[pj] - configuration.positions[pi]
        if np.asarray(configuration.pbc).any():
            dr = minimum_image(dr, configuration.cell, configuration.pbc)
        return -np.einsum("pa,pb->ab", dU_dD, dr, optimize=True)

    # -- featurisation cache -----------------------------------------------

    def _featurise(self, dataset, *, need_forces: bool) -> list[_Cached]:
        """Run the descriptor once per configuration and keep the result.

        This is the expensive part of a fit (the network is tiny by
        comparison), so it happens once up front rather than once per epoch.
        Memory is ``O(P * D * 3)`` doubles per configuration -- about 0.5 MB
        for a 32-atom argon cell with 36 features, so a few thousand
        configurations is a few GB and is the practical dataset ceiling on this
        machine.  ``docs/design.md`` asks for a few thousand 64-atom
        configurations; at 64 atoms and 36 features that is ~1 MB each.
        """
        out: list[_Cached] = []
        for cfg in dataset:
            if cfg.energy is None or (need_forces and cfg.forces is None):
                raise ValueError(
                    f"{self.name}.fit needs labelled configurations "
                    "(energy, and forces when weight_force > 0)"
                )
            self._validate(cfg)
            desc = self.descriptor.compute(cfg, derivatives=need_forces)
            out.append(
                _Cached(
                    features=torch.as_tensor(desc.features, dtype=self.torch_dtype),
                    derivatives=(
                        torch.as_tensor(desc.derivatives, dtype=self.torch_dtype)
                        if need_forces
                        else None
                    ),
                    pair_i=(
                        torch.as_tensor(np.asarray(desc.pair_i, dtype=np.int64))
                        if need_forces
                        else None
                    ),
                    pair_j=(
                        torch.as_tensor(np.asarray(desc.pair_j, dtype=np.int64))
                        if need_forces
                        else None
                    ),
                    species=torch.as_tensor(cfg.species.astype(np.int64)),
                    energy=float(cfg.energy),
                    forces=(
                        torch.as_tensor(cfg.forces, dtype=self.torch_dtype)
                        if cfg.forces is not None
                        else None
                    ),
                    n_atoms=int(cfg.n_atoms),
                )
            )
        return out

    def _batch_predict(self, items: Sequence[_Cached], *, need_forces: bool, create_graph: bool):
        """Predict energies (and forces) for a batch of cached configurations.

        Returns
        -------
        e_pred : Tensor, shape (B,)
            Total energies in eV.
        f_pred : Tensor, shape (sum_b N_b, 3) or None
            Forces in eV/A, stacked in batch order.
        n_atoms : Tensor, shape (B,)

        Notes
        -----
        The batch is one block-diagonal system: configurations share no atoms,
        so a single ``autograd.grad`` of the summed energy with respect to the
        concatenated feature matrix yields each configuration's ``dE/dG``
        without cross-talk.
        """
        feats = torch.cat([it.features for it in items], dim=0)
        species = torch.cat([it.species for it in items], dim=0)
        counts = torch.tensor([it.n_atoms for it in items], dtype=torch.int64)
        batch_index = torch.repeat_interleave(
            torch.arange(len(items), dtype=torch.int64), counts
        )

        feats = feats.detach().requires_grad_(need_forces)
        e_atom = self.net(feats, species)
        e_pred = torch.zeros(len(items), dtype=self.torch_dtype).index_add(
            0, batch_index, e_atom
        )
        if not need_forces:
            return e_pred, None, counts

        (dE_dG,) = torch.autograd.grad(e_atom.sum(), feats, create_graph=create_graph)

        # Offset each configuration's pair indices into the concatenated frame.
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)[:-1]]
        )
        pair_i = torch.cat([it.pair_i + off for it, off in zip(items, offsets)])
        pair_j = torch.cat([it.pair_j + off for it, off in zip(items, offsets)])
        derivs = torch.cat([it.derivatives for it in items], dim=0)

        # Identical algebra to DescriptorOutput.forces_from_energy_gradient,
        # written in torch so the force loss can be differentiated w.r.t. the
        # weights.  `derivs` is constant data and carries no graph.
        # einsum, not broadcast-multiply-then-sum: the latter materialises a
        # (P, D, 3) intermediate and, worse, a second one in the double
        # backward, which dominates the step cost for realistic P.
        contrib = torch.einsum("pd,pda->pa", dE_dG[pair_i], derivs)
        f_pred = torch.zeros(
            (int(counts.sum()), 3), dtype=self.torch_dtype
        ).index_add(0, pair_j, -contrib)
        return e_pred, f_pred, counts

    def _batch_loss(
        self,
        items: Sequence[_Cached],
        *,
        weight_energy: float,
        weight_force: float,
        create_graph: bool,
    ):
        """Joint energy+force loss for one batch, plus its squared-error sums.

        The energy term is per atom (``(E_pred - E_ref) / N``) because total
        energy error grows with system size and would otherwise weight large
        cells more heavily for no physical reason.  The force term is the mean
        over all ``3N`` components.  In these natural units the force term is
        typically ~1e4 larger for near-equilibrium data; that is deliberate and
        not a bug -- forces carry ``3N`` labels per configuration against the
        energy's one and are what molecular dynamics integrates.  Reweight with
        ``weight_energy`` if a study needs the energy to dominate.
        """
        need_forces = weight_force > 0.0
        e_pred, f_pred, counts = self._batch_predict(
            items, need_forces=need_forces, create_graph=create_graph
        )
        e_ref = torch.tensor([it.energy for it in items], dtype=self.torch_dtype)
        de = (e_pred - e_ref) / counts.to(self.torch_dtype)
        loss = weight_energy * (de**2).mean()
        e_sq = float((de**2).sum().detach())

        f_sq, n_comp = 0.0, 0
        if need_forces:
            f_ref = torch.cat([it.forces for it in items], dim=0)
            df = f_pred - f_ref
            loss = loss + weight_force * (df**2).mean()
            f_sq = float((df**2).sum().detach())
            n_comp = int(df.numel())
        return loss, e_sq, f_sq, len(items), n_comp

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        train: Dataset,
        *,
        val: Dataset | None = None,
        epochs: int = 200,
        batch_size: int = 8,
        learning_rate: float = 3e-3,
        weight_energy: float = 1.0,
        weight_force: float = 1.0,
        weight_decay: float = 0.0,
        lr_schedule: str = "cosine",
        min_lr_factor: float = 0.01,
        patience: int = 40,
        grad_clip: float = 5.0,
        torch_threads: int | None = DEFAULT_TORCH_THREADS,
        seed: int | None = None,
        verbose: bool = False,
    ) -> FitReport:
        """Fit energies and forces jointly with Adam.

        Parameters
        ----------
        train, val : Dataset
            Labelled configurations.  ``val`` drives early stopping and the
            plateau schedule; without it, early stopping watches the training
            loss (and says so in ``FitReport.notes``, because a training-loss
            stopping rule is not a generalisation criterion).
        epochs : int
            Maximum number of passes over ``train``.
        batch_size : int
            Configurations per optimiser step.
        learning_rate : float
            Initial Adam learning rate.
        weight_energy, weight_force : float
            Loss weights; see :meth:`_batch_loss` for the normalisation.
        weight_decay : float
            Adam L2 penalty.
        lr_schedule : {'cosine', 'plateau', 'none'}
            ``cosine`` anneals to ``min_lr_factor * learning_rate`` over
            ``epochs``; ``plateau`` halves on a stalled validation loss.
        min_lr_factor : float
            Floor of the cosine schedule, as a fraction of ``learning_rate``.
        patience : int
            Epochs without validation improvement before stopping.  The best
            state seen is always restored at the end, so a late divergence
            cannot be silently kept.
        grad_clip : float
            Max global gradient norm; ``0`` disables.  Force losses have heavy
            gradient tails early in training when a random network's ``dE/dG``
            happens to align with a large ``dG/dr``.
        torch_threads : int or None
            Intra-op threads for the duration of the fit; the previous global
            value is restored afterwards.  See :data:`DEFAULT_TORCH_THREADS`.
        seed : int, optional
            Overrides ``self.seed`` for the minibatch shuffling only (the
            weights were seeded at construction).
        verbose : bool
            Print a line per epoch.

        Returns
        -------
        FitReport
            Includes ``history`` with per-epoch ``train_loss``, ``val_loss``,
            ``lr``, and the running train/val energy and force RMSEs, so the
            training curve is recoverable after the fact.

        Notes
        -----
        ``converged`` is True only if early stopping fired, i.e. the validation
        (or training) loss genuinely stopped improving for ``patience`` epochs.
        Exhausting ``epochs`` is reported as **not** converged even when the
        errors look fine: several experiments here deliberately train
        under-resourced models, and "converged and inaccurate" means something
        different from "stopped early".
        """
        if lr_schedule not in ("cosine", "plateau", "none"):
            raise ValueError(
                f"lr_schedule must be 'cosine', 'plateau' or 'none', got {lr_schedule!r}"
            )
        if len(train) == 0:
            raise ValueError("training set is empty")
        if weight_energy < 0.0 or weight_force < 0.0:
            raise ValueError("loss weights must be non-negative")
        if weight_energy == 0.0 and weight_force == 0.0:
            raise ValueError("at least one of weight_energy / weight_force must be positive")

        t0 = time.perf_counter()
        prev_threads = torch.get_num_threads()
        set_torch_threads(torch_threads)
        notes: list[str] = []
        try:
            need_forces = weight_force > 0.0
            train_items = self._featurise(train, need_forces=need_forces)
            val_items = (
                self._featurise(val, need_forces=need_forces)
                if val is not None and len(val) > 0
                else None
            )
            if val is not None and len(val) == 0:
                notes.append("validation set was empty; early stopping watched the training loss")

            self._fit_statistics(train_items)

            rng = np.random.default_rng(self.seed if seed is None else int(seed))
            optimiser = torch.optim.Adam(
                self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            scheduler = None
            if lr_schedule == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=max(epochs, 1), eta_min=learning_rate * min_lr_factor
                )
            elif lr_schedule == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser, factor=0.5, patience=max(patience // 4, 1)
                )

            history: dict[str, list[float]] = {
                k: []
                for k in (
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_energy_rmse",
                    "train_force_rmse",
                    "val_loss",
                    "val_energy_rmse",
                    "val_force_rmse",
                )
            }
            best_loss = float("inf")
            best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            best_epoch = 0
            stalled = 0
            converged = False
            self.net.train()

            n = len(train_items)
            for epoch in range(1, int(epochs) + 1):
                order = rng.permutation(n)
                tot_loss, tot_e, tot_f, tot_cfg, tot_comp = 0.0, 0.0, 0.0, 0, 0
                for start in range(0, n, batch_size):
                    batch = [train_items[k] for k in order[start : start + batch_size]]
                    optimiser.zero_grad(set_to_none=True)
                    loss, e_sq, f_sq, n_cfg, n_comp = self._batch_loss(
                        batch,
                        weight_energy=weight_energy,
                        weight_force=weight_force,
                        create_graph=need_forces,
                    )
                    loss.backward()
                    if grad_clip and grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                    optimiser.step()
                    tot_loss += float(loss.detach()) * n_cfg
                    tot_e += e_sq
                    tot_f += f_sq
                    tot_cfg += n_cfg
                    tot_comp += n_comp

                train_loss = tot_loss / max(tot_cfg, 1)
                train_e_rmse = float(np.sqrt(tot_e / max(tot_cfg, 1)))
                train_f_rmse = float(np.sqrt(tot_f / tot_comp)) if tot_comp else float("nan")

                if val_items is not None:
                    val_loss, val_e_rmse, val_f_rmse = self._evaluate_cached(
                        val_items,
                        weight_energy=weight_energy,
                        weight_force=weight_force,
                        batch_size=batch_size,
                    )
                else:
                    val_loss, val_e_rmse, val_f_rmse = (
                        train_loss,
                        train_e_rmse,
                        train_f_rmse,
                    )

                lr_now = float(optimiser.param_groups[0]["lr"])
                for key, value in (
                    ("epoch", epoch),
                    ("lr", lr_now),
                    ("train_loss", train_loss),
                    ("train_energy_rmse", train_e_rmse),
                    ("train_force_rmse", train_f_rmse),
                    ("val_loss", val_loss),
                    ("val_energy_rmse", val_e_rmse),
                    ("val_force_rmse", val_f_rmse),
                ):
                    history[key].append(float(value))

                if verbose:  # pragma: no cover - diagnostic only
                    print(
                        f"epoch {epoch:4d}  lr {lr_now:.2e}  train {train_loss:.4e}  "
                        f"val {val_loss:.4e}  F {val_f_rmse:.4f} eV/A"
                    )

                if scheduler is not None:
                    if lr_schedule == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # Strictly-better test: a loss that merely stops getting worse
                # is not progress, and counting it as such would defeat the
                # patience counter entirely.
                if val_loss < best_loss * (1.0 - 1e-4):
                    best_loss = val_loss
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().clone() for k, v in self.net.state_dict().items()
                    }
                    stalled = 0
                else:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_state = {
                            k: v.detach().clone() for k, v in self.net.state_dict().items()
                        }
                    stalled += 1
                    if stalled >= patience:
                        converged = True
                        notes.append(
                            f"early stop at epoch {epoch}: no improvement for {patience} epochs"
                        )
                        break

            self.net.load_state_dict(best_state)
            self.net.eval()
            self.is_fitted = True

            if not converged:
                notes.append(
                    f"ran the full {epochs} epochs without triggering early stopping; "
                    "the optimiser had not met its own stopping criterion"
                )
            if val_items is None:
                notes.append(
                    "no validation set: early stopping and the reported 'val' metrics "
                    "are training-set quantities"
                )

            train_loss, train_e_rmse, train_f_rmse = self._evaluate_cached(
                train_items,
                weight_energy=weight_energy,
                weight_force=weight_force,
                batch_size=batch_size,
            )
            if val_items is not None:
                _, val_e_rmse, val_f_rmse = self._evaluate_cached(
                    val_items,
                    weight_energy=weight_energy,
                    weight_force=weight_force,
                    batch_size=batch_size,
                )
            else:
                val_e_rmse, val_f_rmse = train_e_rmse, train_f_rmse

            return FitReport(
                converged=converged,
                n_epochs=len(history["epoch"]),
                wall_seconds=time.perf_counter() - t0,
                n_parameters=self.n_parameters,
                n_train=len(train_items),
                n_val=0 if val_items is None else len(val_items),
                train_energy_rmse=train_e_rmse,
                train_force_rmse=train_f_rmse,
                val_energy_rmse=val_e_rmse,
                val_force_rmse=val_f_rmse,
                history={**history, "best_epoch": [best_epoch]},
                notes=notes,
            )
        finally:
            self.net.eval()
            torch.set_num_threads(prev_threads)

    def _fit_statistics(self, items: Sequence[_Cached]) -> None:
        """Fit and store feature standardisation and energy shift/scale.

        Feature statistics are per species: the same descriptor column means
        different things around different chemistries, and pooling them would
        centre neither.  Constant columns (an inner radial function that no
        configuration ever reaches, say) get scale 1 rather than ~0, exactly as
        :func:`~atomlab.models.base.standardize` does -- dividing by a numerical
        zero would manufacture noise where there is no information.
        """
        feats = torch.cat([it.features for it in items], 0).to(torch.float64).numpy()
        species = torch.cat([it.species for it in items], 0).numpy()
        d = self.net.n_features
        mean = np.zeros((self.n_species, d))
        scale = np.ones((self.n_species, d))
        for s in range(self.n_species):
            rows = feats[species == s]
            if rows.shape[0] == 0:
                continue
            mean[s] = rows.mean(axis=0)
            sd = rows.std(axis=0)
            scale[s] = np.where(sd > 1e-12, sd, 1.0)
        self.set_feature_statistics(mean, scale)

        # Energy shift: the mean per-atom energy, so the network only has to
        # learn the fluctuation.  Scale: the spread of that fluctuation, with a
        # floor for the degenerate case of a dataset of identical energies.
        per_atom = np.array([it.energy / it.n_atoms for it in items])
        spread = float(per_atom.std())
        self.set_scaling(np.full(self.n_species, float(per_atom.mean())),
                         spread if spread > 1e-9 else 1.0)

    def _evaluate_cached(
        self,
        items: Sequence[_Cached],
        *,
        weight_energy: float,
        weight_force: float,
        batch_size: int,
    ) -> tuple[float, float, float]:
        """Loss and RMSEs over cached configurations, without touching weights.

        Force prediction needs autograd through the network, so this cannot run
        under ``no_grad``; ``create_graph=False`` keeps it first-order and the
        parameter gradients are discarded by the caller's ``zero_grad``.
        """
        tot_loss, tot_e, tot_f, tot_cfg, tot_comp = 0.0, 0.0, 0.0, 0, 0
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            loss, e_sq, f_sq, n_cfg, n_comp = self._batch_loss(
                batch,
                weight_energy=weight_energy,
                weight_force=weight_force,
                create_graph=False,
            )
            tot_loss += float(loss.detach()) * n_cfg
            tot_e += e_sq
            tot_f += f_sq
            tot_cfg += n_cfg
            tot_comp += n_comp
        self.net.zero_grad(set_to_none=True)
        return (
            tot_loss / max(tot_cfg, 1),
            float(np.sqrt(tot_e / max(tot_cfg, 1))),
            float(np.sqrt(tot_f / tot_comp)) if tot_comp else float("nan"),
        )

    # -- serialisation -----------------------------------------------------

    def save(self, path) -> None:
        """Save weights, buffers and hyper-parameters via ``torch.save``.

        Overrides the pickled-object default of :class:`MLModel`.  What is
        written is a ``state_dict`` plus the constructor arguments, so a saved
        model survives edits to this class that a pickled instance would not.
        The descriptor is stored as its constructor arguments when it is an
        :class:`~atomlab.models.descriptors.acsf.ACSF`; any other descriptor is
        embedded as an object (``torch.save`` pickles it) and a note is written
        into the payload, because silently dropping it would be worse.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "atomlab.bpnn.v1",
            "state_dict": self.net.state_dict(),
            "hparams": {
                "hidden": list(self.hidden),
                "activation": self.activation,
                "n_species": self.n_species,
                "seed": self.seed,
                "dtype": self.dtype,
                "name": self.name,
                "allow_untrained": self.allow_untrained,
            },
            "is_fitted": bool(self.is_fitted),
            "descriptor_spec": self._descriptor_spec(),
        }
        torch.save(payload, path)

    def _descriptor_spec(self) -> dict:
        """Constructor arguments of the descriptor, or the object as a fallback."""
        d = self.descriptor
        if isinstance(d, ACSF):
            return {
                "kind": "acsf",
                "species": [int(s) for s in d.species],
                "cutoff": float(d.cutoff),
                "radial": np.asarray(d.radial, dtype=np.float64),
                "angular": np.asarray(d.angular, dtype=np.float64),
                "angular_kind": list(d.angular_kind),
                "cutoff_function": d.cutoff_kind,
                "name": d.name,
            }
        return {"kind": "object", "object": d}

    @staticmethod
    def _descriptor_from_spec(spec: dict) -> Descriptor:
        if spec.get("kind") == "acsf":
            return ACSF(
                spec["species"],
                spec["cutoff"],
                spec["radial"],
                spec["angular"],
                angular_kind=spec["angular_kind"] or "G4",
                cutoff_function=spec["cutoff_function"],
                name=spec["name"],
            )
        if spec.get("kind") == "object":
            return spec["object"]
        raise ValueError(f"unrecognised descriptor spec {spec.get('kind')!r}")

    @classmethod
    def load(cls, path) -> "BPNN":
        """Reconstruct a model saved by :meth:`save`.

        Returns
        -------
        BPNN
            Predictions are **bitwise identical** to the saved model's: the
            weights, the feature standardisation and the energy shift/scale are
            all state-dict entries restored exactly, and evaluation has no
            stochastic component.

        Notes
        -----
        ``weights_only=False`` is required because the payload carries the
        descriptor specification (numpy arrays, or an embedded descriptor
        object) alongside the tensors.  Only load files you wrote yourself --
        the same caveat the pickled default in :class:`MLModel` carries.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "atomlab.bpnn.v1":
            raise ValueError(
                f"{path} is not an atomlab BPNN checkpoint (format={payload.get('format')!r})"
            )
        hp = payload["hparams"]
        model = cls(
            cls._descriptor_from_spec(payload["descriptor_spec"]),
            hidden=hp["hidden"],
            activation=hp["activation"],
            n_species=hp["n_species"],
            seed=hp["seed"],
            dtype=hp["dtype"],
            allow_untrained=hp["allow_untrained"],
            name=hp["name"],
        )
        model.net.load_state_dict(payload["state_dict"])
        model.net.eval()
        model.is_fitted = bool(payload["is_fitted"])
        return model

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flag = "fitted" if self.is_fitted else "untrained"
        return (
            f"BPNN(name={self.name!r}, D={self.net.n_features}, hidden={self.hidden}, "
            f"species={self.n_species}, p={self.n_parameters}, seed={self.seed}, {flag})"
        )
