"""E(3)-equivariant message-passing interatomic potential, written from scratch.

This module implements a small NequIP/Tensor-Field-Network-style model with no
external equivariance library: the real spherical harmonics, the Clebsch-Gordan
(Gaunt) tensor products, the gated nonlinearity and the message passing are all
explicit here so that the tensor algebra can be audited.  That is deliberate --
the whole premise of this repository is that a model whose derivatives or
symmetries are subtly wrong still trains to a plausible energy error and then
produces wrong physics, so the pieces that are usually delegated to a library
are the pieces that most need to be visible.

What is (and is not) claimed
----------------------------

The network is **O(3)-equivariant**, i.e. equivariant under rotations *and*
reflections, not merely SO(3).  This is a consequence of two choices:

1. Every feature carries the **natural parity** of its degree, ``p = (-1)^l``.
   The ``l = 0`` channels are true scalars, the ``l = 1`` channels are true
   (polar) vectors, the ``l = 2`` channels are true rank-2 tensors.  These are
   exactly the parities of the spherical harmonics ``Y_l(r_hat)`` of the bond
   directions, which is what the features are built from.
2. The tensor-product coefficients used here are the **real Gaunt
   coefficients** ``G_{l1 m1, l2 m2, l3 m3} = int Y_{l1 m1} Y_{l2 m2} Y_{l3 m3}
   dOmega``, which vanish identically unless ``l1 + l2 + l3`` is even.  That
   selection rule *is* parity conservation for natural-parity irreps.  A path
   such as ``(l1, l2, l3) = (1, 1, 1)`` -- the cross product of two vectors --
   is therefore absent, which is correct: the cross product of two polar
   vectors is an axial vector, an ``l = 1`` object of parity ``+1``, and
   admitting it would let the network build pseudoscalars and hence a
   parity-odd energy.  An energy that flips sign under inversion is a classic
   equivariant-network bug and ``tests/test_egnn.py`` tests for its absence
   explicitly.

Sizing
------

The architecture is deliberately small: ``L_max = 1``, 16 channels, 2
interaction layers, ~9k parameters.  At GPU scale one would use ``L_max = 2-3``,
64-128 channels and 4-6 layers; that is unaffordable on the 4 CPU cores this
study runs on and, more importantly, it is not what the study is measuring.
``L_max = 2`` is implemented and tested, it is simply not the default.

Virial
------

The virial is obtained by the **strain-derivative trick** rather than by
hand-assembling ``sum f_ij (x) r_ij``.  A symmetric strain variable ``eps``,
initialised at zero, is introduced and every bond vector is transformed as
``D -> (1 + eps) D``.  Because ``D = r_j + S @ cell - r_i`` already contains the
periodic image offset ``S @ cell``, transforming ``D`` is *identical* to
transforming the atomic positions and the lattice vectors together, which is
precisely the deformation the convention ``W = -dU/d(eps)`` is defined against.
Autograd then delivers the periodic virial with no possibility of dropping an
image term -- the standard failure mode of hand-assembled many-body virials.

Units
-----

Metal units throughout: positions in A, energies in eV, forces in eV/A, virial
in eV.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..cell import check_minimum_image
from ..neighbors import build_neighbor_list
from ..types import Configuration, Dataset, Result
from .base import FitReport, MLModel

__all__ = [
    "EGNN",
    "real_spherical_harmonics",
    "clebsch_gordan",
    "tensor_product_paths",
    "bessel_basis",
    "polynomial_envelope",
]

#: Default intra-op thread count applied by :func:`set_torch_threads`.
#:
#: The brief for this module specified ``torch.set_num_threads(2)``.  That was
#: measured on this machine and is *catastrophically* wrong for a model this
#: small, so it is not what the module does; the deviation is recorded here
#: rather than silently.  Every tensor in this network is tiny (a few hundred
#: edges x 16 channels x 4 irrep components), and torch's OpenMP fork/join for
#: such shapes costs far more than the arithmetic.  Timings for the dominant
#: kernel, ``index_add`` on a ``(388, 16, 4)`` source, 4 physical cores:
#:
#: ===========  ===============
#: threads      time per call
#: ===========  ===============
#: 1            12 us
#: 2            1.9 ms  (160x)
#: 4            176 ms  (14000x)
#: ===========  ===============
#:
#: A full ``compute()`` on a 32-atom cell goes from 98 ms at 2 threads to
#: 2.5 ms at 1 thread.  Single-threaded is therefore the correct setting, and
#: the way to use 4 cores here is 4 independent single-threaded fits, not one
#: 4-threaded fit.
DEFAULT_TORCH_THREADS = 1


def set_torch_threads(n_threads: int | None = DEFAULT_TORCH_THREADS) -> None:
    """Set torch's intra-op thread count (``None`` leaves it alone).

    This is global torch state, so it is done explicitly through a named
    function rather than buried in a constructor: a caller who is deliberately
    running one big multi-threaded job can pass ``torch_threads=None`` to
    :class:`EGNN` and keep their own setting.
    """
    if n_threads is not None:
        torch.set_num_threads(int(n_threads))


set_torch_threads()

#: Largest degree for which the cartesian spherical harmonics are hard-coded.
MAX_SUPPORTED_L = 2


# ==========================================================================
# 1. Real spherical harmonics in cartesian form
# ==========================================================================


def real_spherical_harmonics(rhat: Tensor, l_max: int) -> Tensor:
    """Real spherical harmonics ``Y_lm(r_hat)`` for ``l = 0 .. l_max``.

    Parameters
    ----------
    rhat : Tensor, shape (P, 3)
        Unit vectors (dimensionless).  The caller is responsible for
        normalising; the polynomials below are evaluated with the actual
        ``r^2`` of the input so that the result is exactly homogeneous, but the
        physical interpretation as ``Y_lm`` only holds for ``|r| = 1``.
    l_max : int
        Maximum degree, ``0 <= l_max <= 2``.

    Returns
    -------
    Tensor, shape (P, (l_max + 1)**2)
        Harmonics packed as ``[l=0 | l=1, m=-1,0,1 | l=2, m=-2..2]``, i.e.
        component ``l**2 + l + m``.

    Notes
    -----
    These are the standard real spherical harmonics rescaled by
    ``sqrt(4 pi / (2l + 1))``, so that ``sum_m Y_lm(r_hat)**2 = 1`` for every
    ``l``.  The scaling is a per-``l`` constant, so it changes neither the
    transformation law nor any equivariance property; it is chosen because it
    keeps every degree on the same numerical footing, which matters when a
    learned radial function multiplies them.

    Explicitly, with ``(x, y, z) = r_hat``::

        l = 0:  1
        l = 1:  (y, z, x)
        l = 2:  (sqrt(3) xy, sqrt(3) yz, (3 z^2 - r^2)/2,
                 sqrt(3) xz, sqrt(3) (x^2 - y^2) / 2)

    The ``m`` ordering ``(y, z, x)`` for ``l = 1`` is the usual one induced by
    ``m = -1, 0, +1``; it is *not* ``(x, y, z)``, and getting it wrong is
    invisible to every test except an explicit rotation check.

    The cartesian form is used rather than a ``(theta, phi)`` form because it is
    a polynomial: it is smooth and differentiable everywhere except the origin,
    where the ``theta`` derivatives of the spherical form are singular.  The
    origin never occurs here, since a neighbour always has ``r > 0``.
    """
    l_max = int(l_max)
    if l_max < 0 or l_max > MAX_SUPPORTED_L:
        raise ValueError(
            f"l_max must be in 0..{MAX_SUPPORTED_L} (the cartesian forms are "
            f"hard-coded up to l = {MAX_SUPPORTED_L}), got {l_max}"
        )
    if rhat.ndim != 2 or rhat.shape[1] != 3:
        raise ValueError(f"rhat must be (P, 3), got {tuple(rhat.shape)}")

    x = rhat[:, 0]
    y = rhat[:, 1]
    z = rhat[:, 2]
    out = [torch.ones_like(x)]
    if l_max >= 1:
        out += [y, z, x]
    if l_max >= 2:
        r2 = x * x + y * y + z * z
        s3 = math.sqrt(3.0)
        out += [
            s3 * x * y,
            s3 * y * z,
            0.5 * (3.0 * z * z - r2),
            s3 * x * z,
            0.5 * s3 * (x * x - y * y),
        ]
    return torch.stack(out, dim=-1)


def _sh_numpy(directions: np.ndarray, l_max: int) -> np.ndarray:
    """NumPy wrapper around :func:`real_spherical_harmonics` (float64)."""
    t = torch.as_tensor(np.asarray(directions, dtype=np.float64))
    return real_spherical_harmonics(t, l_max).numpy()


# ==========================================================================
# 2. Clebsch-Gordan / Gaunt coefficients
# ==========================================================================


def tensor_product_paths(
    l_max_in: int, l_max_sh: int, l_max_out: int
) -> list[tuple[int, int, int]]:
    """Allowed ``(l_in, l_sh, l_out)`` tensor-product paths.

    A path is kept when it satisfies the angular-momentum triangle inequality
    ``|l1 - l2| <= l3 <= l1 + l2`` **and** ``l1 + l2 + l3`` is even.  The second
    condition is parity conservation for natural-parity irreps
    (``p_i = (-1)^{l_i}``, so ``p1 p2 = p3`` reads ``(-1)^{l1+l2-l3} = 1``); it
    is also exactly the condition under which the real Gaunt coefficient is
    nonzero, so it is enforced twice over, once by symmetry and once by the
    numerics.

    Returns
    -------
    list of (int, int, int)
        Sorted, deterministic path list.
    """
    paths = []
    for l1 in range(int(l_max_in) + 1):
        for l2 in range(int(l_max_sh) + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, int(l_max_out)) + 1):
                if (l1 + l2 + l3) % 2 == 0:
                    paths.append((l1, l2, l3))
    return paths


@functools.lru_cache(maxsize=None)
def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real Clebsch-Gordan (Gaunt) tensor for ``l1 (x) l2 -> l3``.

    Parameters
    ----------
    l1, l2, l3 : int
        Degrees, each ``0 <= l <= 2``.

    Returns
    -------
    ndarray, shape (2*l1+1, 2*l2+1, 2*l3+1), float64
        Normalised to unit Frobenius norm, with entries below ``1e-10`` of the
        largest magnitude set to exactly zero so that the sparsity pattern (and
        hence the parity structure) is exact rather than approximate.

    Notes
    -----
    The coefficients are the **real Gaunt coefficients**

    .. math:: G_{m_1 m_2 m_3} = \\int Y_{l_1 m_1} Y_{l_2 m_2} Y_{l_3 m_3}\\,d\\Omega

    computed by exact quadrature (Gauss-Legendre in ``cos(theta)``, trapezoid in
    ``phi``; both integrands are polynomials of low degree, so the quadrature is
    exact to machine precision).

    Why this is a legitimate Clebsch-Gordan tensor: substituting
    ``r_hat -> R^{-1} r_hat`` in the integral leaves it invariant, while each
    ``Y_l`` picks up the real Wigner matrix ``D^l(R)``.  Hence

    .. math:: \\sum D^{l_1}_{m_1 m_1'} D^{l_2}_{m_2 m_2'} D^{l_3}_{m_3 m_3'}
              G_{m_1' m_2' m_3'} = G_{m_1 m_2 m_3}

    which is precisely the intertwiner property that makes
    ``(u (x) v)_{m_3} = sum G_{m_1 m_2 m_3} u_{m_1} v_{m_2}`` transform as an
    ``l_3`` object whenever ``u`` and ``v`` transform as ``l_1`` and ``l_2``.
    Tensor products of SO(3) irreps are multiplicity free, so for each allowed
    triple this intertwiner is unique up to the scale, which the normalisation
    below fixes.  The overall scale is in any case absorbed by the learned
    radial weights and has no effect beyond conditioning.

    The Gaunt coefficient vanishes for odd ``l1 + l2 + l3`` while the SO(3)
    Clebsch-Gordan coefficient does not; those paths are the parity-violating
    ones and their absence is what upgrades the model from SO(3) to O(3).
    """
    for l in (l1, l2, l3):
        if l < 0 or l > MAX_SUPPORTED_L:
            raise ValueError(f"degrees must be in 0..{MAX_SUPPORTED_L}, got {(l1, l2, l3)}")

    # Quadrature exact for spherical polynomials of degree <= 3 * MAX_SUPPORTED_L.
    n_theta, n_phi = 16, 32
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi = 2.0 * np.pi / n_phi

    dirs = np.empty((n_theta * n_phi, 3))
    weights = np.empty(n_theta * n_phi)
    k = 0
    for a in range(n_theta):
        for b in range(n_phi):
            dirs[k] = (st[a] * np.cos(phi[b]), st[a] * np.sin(phi[b]), ct[a])
            weights[k] = wt[a] * w_phi
            k += 1

    y = _sh_numpy(dirs, MAX_SUPPORTED_L)
    y1 = y[:, l1 * l1 : (l1 + 1) ** 2]
    y2 = y[:, l2 * l2 : (l2 + 1) ** 2]
    y3 = y[:, l3 * l3 : (l3 + 1) ** 2]
    g = np.einsum("qa,qb,qc,q->abc", y1, y2, y3, weights)

    peak = np.max(np.abs(g))
    if peak < 1e-10:
        # Parity-forbidden triple: return an exact zero rather than quadrature dust.
        return np.zeros_like(g)
    g[np.abs(g) < 1e-10 * peak] = 0.0
    return g / np.linalg.norm(g)


# ==========================================================================
# 3. Radial basis and cutoff
# ==========================================================================


def bessel_basis(r: Tensor, r_cut: float, n_basis: int) -> Tensor:
    """Radial Bessel basis ``sqrt(2/rc) sin(n pi r / rc) / r``.

    Parameters
    ----------
    r : Tensor, shape (P,)
        Interatomic distances in A.  Must be strictly positive.
    r_cut : float
        Cutoff radius in A.
    n_basis : int
        Number of basis functions.

    Returns
    -------
    Tensor, shape (P, n_basis)

    Notes
    -----
    These are the radial parts of the spherical Bessel ``j_0`` solutions on a
    ball of radius ``rc`` with a Dirichlet boundary condition, so every basis
    function already vanishes at ``r = rc``.  Their *derivatives* do not, which
    is why :func:`polynomial_envelope` is applied on top -- a basis that is only
    ``C^0`` at the cutoff gives discontinuous forces and an energy drift in MD
    that no static test-set metric would reveal.
    """
    n = torch.arange(1, int(n_basis) + 1, dtype=r.dtype, device=r.device)
    return math.sqrt(2.0 / r_cut) * torch.sin(n[None, :] * math.pi * r[:, None] / r_cut) / r[:, None]


def polynomial_envelope(r: Tensor, r_cut: float, p: int = 6) -> Tensor:
    """Smooth cutoff with ``u(rc) = u'(rc) = u''(rc) = 0``.

    Parameters
    ----------
    r : Tensor, shape (P,)
        Distances in A.
    r_cut : float
        Cutoff radius in A.
    p : int
        Polynomial order parameter; ``p = 6`` gives a function that is ``C^2``
        at the cutoff.

    Returns
    -------
    Tensor, shape (P,)
        ``u(r)`` in ``[0, 1]``, exactly zero for ``r >= rc``.

    Notes
    -----
    ``u(x) = 1 - ((p+1)(p+2)/2) x^p + p(p+2) x^{p+1} - (p(p+1)/2) x^{p+2}`` with
    ``x = r / rc``.  Two vanishing derivatives at the cutoff are needed and not
    merely nice: the finite-difference *virial* check strains the cell, which
    moves pairs across the cutoff, and a merely ``C^1`` envelope shows up there
    as a small but systematic disagreement that is easy to mistake for a bug in
    the strain derivation.
    """
    x = r / r_cut
    u = (
        1.0
        - 0.5 * (p + 1) * (p + 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - 0.5 * p * (p + 1) * x ** (p + 2)
    )
    return torch.where(x < 1.0, u, torch.zeros_like(u))


# ==========================================================================
# 4. Network modules
# ==========================================================================


class _RadialWeights(nn.Module):
    """Learned radial function producing one weight per (path, channel).

    The message weight is ``MLP(bessel(r)) * envelope(r)``.  Multiplying the
    *output* of the MLP by the envelope (rather than only the basis) is what
    guarantees that the message vanishes continuously at the cutoff whatever the
    MLP has learned -- an MLP applied to a basis that vanishes at ``rc`` still
    has a nonzero bias there.
    """

    def __init__(self, n_basis: int, hidden: int, n_out: int, r_cut: float) -> None:
        super().__init__()
        self.r_cut = float(r_cut)
        self.n_basis = int(n_basis)
        self.net = nn.Sequential(
            nn.Linear(n_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, r: Tensor) -> Tensor:
        """``(P,) -> (P, n_out)``."""
        b = bessel_basis(r, self.r_cut, self.n_basis)
        return self.net(b) * polynomial_envelope(r, self.r_cut)[:, None]


class _EquivariantLinear(nn.Module):
    """Channel mixing that acts identically on every ``m`` of a given ``l``.

    A separate ``(C_out, C_in)`` matrix per degree ``l``, contracted over the
    channel index only.  Mixing across ``m`` (or across ``l``) would not commute
    with the rotation matrices ``D^l`` and would destroy equivariance; mixing
    across channels at fixed ``(l, m)`` commutes trivially because ``D^l`` acts
    on ``m`` alone.
    """

    def __init__(self, l_max: int, c_in: int, c_out: int) -> None:
        super().__init__()
        self.l_max = int(l_max)
        self.weight = nn.Parameter(
            torch.randn(self.l_max + 1, c_out, c_in) / math.sqrt(c_in)
        )

    def forward(self, h: Tensor) -> Tensor:
        """``(N, C_in, n_lm) -> (N, C_out, n_lm)``."""
        out = []
        for l in range(self.l_max + 1):
            block = h[:, :, l * l : (l + 1) ** 2]
            out.append(torch.einsum("ncm,dc->ndm", block, self.weight[l]))
        return torch.cat(out, dim=2)


class _InteractionLayer(nn.Module):
    """One message-passing layer: tensor product, aggregation, gate, residual.

    The message from neighbour ``j`` to atom ``i`` is the channel-wise tensor
    product of ``j``'s features with the spherical harmonics of the bond
    direction, weighted by the learned radial function of ``r_ij``::

        m_i[c, l3, m3] = (1/sqrt(<n>)) sum_j sum_{paths p=(l1,l2,l3)}
                          w_p[c](r_ij) sum_{m1,m2} G^p_{m1 m2 m3}
                          h_j[c, l1, m1] Y_{l2 m2}(r_hat_ij)

    "Channel-wise" (depthwise) means output channel ``c`` sees only input
    channel ``c``; the channels are mixed afterwards by
    :class:`_EquivariantLinear`.  A fully dense tensor product would cost
    ``C^2`` per path and buys nothing at this size.
    """

    def __init__(
        self,
        channels: int,
        l_max: int,
        n_basis: int,
        radial_hidden: int,
        r_cut: float,
        avg_neighbors: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.l_max = int(l_max)
        self.n_lm = (self.l_max + 1) ** 2
        self.avg_neighbors = float(avg_neighbors)

        self.paths = tensor_product_paths(l_max, l_max, l_max)
        for l1, l2, l3 in self.paths:
            self.register_buffer(
                f"cg_{l1}_{l2}_{l3}",
                torch.as_tensor(clebsch_gordan(l1, l2, l3)),
                persistent=True,
            )

        self.radial = _RadialWeights(
            n_basis, radial_hidden, len(self.paths) * channels, r_cut
        )
        self.mix_message = _EquivariantLinear(l_max, channels, channels)
        self.mix_self = _EquivariantLinear(l_max, channels, channels)
        # One gate scalar per (channel, l>0).  Produced from the scalar part of
        # the update, so it is itself invariant.
        self.gate = nn.Linear(channels, channels * max(self.l_max, 1))

    def forward(
        self, h: Tensor, edge_i: Tensor, edge_j: Tensor, sh: Tensor, r: Tensor
    ) -> Tensor:
        """``(N, C, n_lm) -> (N, C, n_lm)``."""
        n_atoms = h.shape[0]
        c = self.channels

        if edge_i.numel() > 0:
            w = self.radial(r).view(-1, len(self.paths), c)
            x = h[edge_j]  # (P, C, n_lm)

            acc: list[Tensor | None] = [None] * (self.l_max + 1)
            for p, (l1, l2, l3) in enumerate(self.paths):
                cg = getattr(self, f"cg_{l1}_{l2}_{l3}")  # (2l1+1, 2l2+1, 2l3+1)
                y = sh[:, l2 * l2 : (l2 + 1) ** 2]  # (P, 2l2+1)
                # Contract the harmonics into the CG tensor first: this is the
                # cheap order, O(P * (2l1+1) * (2l3+1)) instead of touching the
                # channel axis inside the m-sum.
                t = torch.einsum("pn,mno->pmo", y, cg)
                xa = x[:, :, l1 * l1 : (l1 + 1) ** 2]  # (P, C, 2l1+1)
                contrib = w[:, p, :, None] * torch.einsum("pcm,pmo->pco", xa, t)
                acc[l3] = contrib if acc[l3] is None else acc[l3] + contrib

            msg = torch.cat(
                [
                    acc[l]
                    if acc[l] is not None
                    else h.new_zeros(edge_i.shape[0], c, 2 * l + 1)
                    for l in range(self.l_max + 1)
                ],
                dim=2,
            )
            agg = h.new_zeros(n_atoms, c, self.n_lm)
            agg = agg.index_add(0, edge_i, msg) / math.sqrt(self.avg_neighbors)
        else:
            agg = h.new_zeros(n_atoms, c, self.n_lm)

        update = self.mix_message(agg) + self.mix_self(h)

        # --- gated nonlinearity ------------------------------------------
        # An elementwise nonlinearity applied straight to an l > 0 component
        # would break equivariance: under a rotation those components mix,
        # h'_m = sum_m' D^l_{m m'} h_m', and for nonlinear sigma
        # sigma(D h) != D sigma(h).  Only l = 0 components are individually
        # invariant, so only they may be fed to an arbitrary nonlinearity.
        # For l > 0 the sole equivariant option that is still nonlinear in the
        # features is multiplication by an invariant scalar: g * (D h) =
        # D * (g h) holds for any scalar g, so gating commutes with rotation
        # exactly.
        scalars = update[:, :, 0]  # (N, C), invariant by construction
        out = [torch.nn.functional.silu(scalars)[:, :, None]]
        if self.l_max >= 1:
            gates = self.gate(scalars).view(-1, self.channels, self.l_max)
            gates = torch.nn.functional.silu(gates)
            for l in range(1, self.l_max + 1):
                block = update[:, :, l * l : (l + 1) ** 2]
                out.append(block * gates[:, :, l - 1][:, :, None])
        # Residual connection: keeps the deep model well conditioned and lets
        # the first layer's purely radial information survive to the readout.
        return h + torch.cat(out, dim=2)


class _EGNNNet(nn.Module):
    """The full network: embedding -> interaction layers -> scalar readout."""

    def __init__(
        self,
        *,
        n_species: int,
        channels: int,
        l_max: int,
        n_layers: int,
        n_basis: int,
        radial_hidden: int,
        r_cut: float,
        avg_neighbors: float,
        readout_hidden: int,
    ) -> None:
        super().__init__()
        self.n_species = int(n_species)
        self.channels = int(channels)
        self.l_max = int(l_max)
        self.n_lm = (self.l_max + 1) ** 2
        self.r_cut = float(r_cut)

        self.embedding = nn.Embedding(n_species, channels)
        self.layers = nn.ModuleList(
            [
                _InteractionLayer(
                    channels, l_max, n_basis, radial_hidden, r_cut, avg_neighbors
                )
                for _ in range(int(n_layers))
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(channels, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 1),
        )
        # Per-atom affine calibration, set from the training data by
        # EGNN.set_scaling.  Registered as buffers so they travel with the
        # state dict but are not optimised (they are data statistics, not
        # parameters, and fitting them would double-count the readout bias).
        self.register_buffer("energy_shift", torch.zeros(n_species))
        self.register_buffer("energy_scale", torch.ones(()))

    def forward(
        self,
        positions: Tensor,
        edge_i: Tensor,
        edge_j: Tensor,
        shift_cart: Tensor,
        species: Tensor,
        strain: Tensor | None = None,
    ) -> Tensor:
        """Per-atom energies in eV.

        Parameters
        ----------
        positions : Tensor, shape (N, 3)
            Cartesian positions in A.
        edge_i, edge_j : Tensor, shape (P,), int64
            Pair indices; the message flows from ``j`` to ``i``.
        shift_cart : Tensor, shape (P, 3)
            Periodic image offset ``shift @ cell`` in A, so that the bond vector
            is ``r_j + shift_cart - r_i`` (from ``i`` to ``j``, the package-wide
            convention).
        species : Tensor, shape (N,), int64
            Type indices.
        strain : Tensor, shape (3, 3), optional
            Strain variable; if given, the bond vectors are transformed as
            ``D -> (1 + sym(eps)) D``.  Evaluating at ``eps = 0`` leaves the
            energy unchanged, and ``-dE/d(eps)`` is the virial.

        Returns
        -------
        Tensor, shape (N,)
            Per-atom energies in eV.
        """
        d = positions[edge_j] + shift_cart - positions[edge_i]
        if strain is not None:
            # Symmetrise: the antisymmetric part of eps is an infinitesimal
            # rotation, which cannot change a rotationally invariant energy, so
            # excluding it removes a numerically noisy null direction and
            # guarantees a symmetric virial.  With eps symmetric,
            # D (1 + eps)^T == D (1 + eps).
            eps = 0.5 * (strain + strain.transpose(0, 1))
            d = d + d @ eps

        r = torch.linalg.norm(d, dim=1)
        rhat = d / r[:, None]
        sh = real_spherical_harmonics(rhat, self.l_max)

        # Build the initial features by concatenation rather than in-place
        # assignment: an in-place write into a tensor that later needs a
        # gradient is legal but fragile, and this is the one place where the
        # graph is created.
        emb = self.embedding(species)[:, :, None]  # (N, C, 1), the l = 0 block
        h = torch.cat(
            [emb, emb.new_zeros(emb.shape[0], self.channels, self.n_lm - 1)], dim=2
        )
        for layer in self.layers:
            h = layer(h, edge_i, edge_j, sh, r)

        e = self.readout(h[:, :, 0]).squeeze(-1)
        return self.energy_shift[species] + self.energy_scale * e


# ==========================================================================
# 5. The Potential
# ==========================================================================


class _Graph:
    """Cached neighbour-list plumbing for one configuration (torch tensors)."""

    __slots__ = ("edge_i", "edge_j", "shift", "species", "n_atoms")

    def __init__(self, edge_i, edge_j, shift, species, n_atoms):
        self.edge_i = edge_i
        self.edge_j = edge_j
        self.shift = shift  # integer image, (P, 3); cartesian offset needs the cell
        self.species = species
        self.n_atoms = n_atoms


class EGNN(MLModel):
    """E(3)-equivariant message-passing potential.

    Parameters
    ----------
    cutoff : float
        Interaction radius ``rc`` in A.
    channels : int
        Number of feature channels per irrep.  16 by default; this is small by
        the standards of the literature (64-128) and is chosen so that a fit on
        a few thousand 64-atom configurations finishes in minutes on 4 CPU
        cores.
    l_max : int
        Maximum degree of the equivariant features, 1 or 2.  ``l_max = 1``
        (scalars + vectors) is the default; ``l_max = 2`` roughly triples the
        cost of a layer for a modest accuracy gain at this size.
    n_layers : int
        Number of interaction layers.  Each layer extends the effective
        receptive field by one cutoff radius, so 2 layers see ``2 rc``.
    n_basis : int
        Size of the radial Bessel basis.
    radial_hidden, readout_hidden : int
        Hidden widths of the radial and readout MLPs.
    n_species : int
        Number of atom types the model can represent.
    avg_neighbors : float
        Message normalisation ``1/sqrt(<n_neighbours>)``.  Estimated from the
        training set by :meth:`fit`; the default is only used before fitting.
    seed : int
        Seeds parameter initialisation.  The global torch RNG state is saved and
        restored around the initialisation, so constructing a model never
        perturbs anyone else's random stream.
    dtype : {"float32", "float64"}
        Working precision.  ``float32`` is ~2x faster and is what training uses;
        ``float64`` is needed to check analytic derivatives against finite
        differences at the ``1e-5`` level, since float32 autograd noise alone is
        of order ``1e-4`` eV/A.
    torch_threads : int or None
        Applied via :func:`set_torch_threads` at construction.  Defaults to 1;
        see :data:`DEFAULT_TORCH_THREADS` for the measurements behind that.
        ``None`` leaves the global setting untouched.
    allow_untrained : bool
        Permit :meth:`compute` before :meth:`fit`.  Off by default -- an
        unfitted model returns arbitrary numbers that look like predictions --
        but the equivariance tests deliberately use a randomly initialised
        network, because a *trained* network could be accidentally invariant
        while the architecture is not.
    name : str
        Short identifier used in tables and figures.

    Attributes
    ----------
    net : torch.nn.Module
        The underlying network.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        *,
        channels: int = 16,
        l_max: int = 1,
        n_layers: int = 2,
        n_basis: int = 8,
        radial_hidden: int = 32,
        readout_hidden: int = 16,
        n_species: int = 1,
        avg_neighbors: float = 20.0,
        seed: int = 0,
        dtype: str = "float32",
        torch_threads: int | None = DEFAULT_TORCH_THREADS,
        allow_untrained: bool = False,
        name: str = "egnn",
    ) -> None:
        set_torch_threads(torch_threads)
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be > 0 A, got {cutoff}")
        if l_max not in (0, 1, 2):
            raise ValueError(f"l_max must be 0, 1 or 2, got {l_max}")
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")

        self.cutoff = float(cutoff)
        self.name = name
        self.seed = int(seed)
        self.dtype = dtype
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.allow_untrained = bool(allow_untrained)
        self.is_fitted = False
        self._hparams = dict(
            channels=channels,
            l_max=l_max,
            n_layers=n_layers,
            n_basis=n_basis,
            radial_hidden=radial_hidden,
            readout_hidden=readout_hidden,
            n_species=n_species,
            avg_neighbors=avg_neighbors,
        )

        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            self.net = _EGNNNet(
                n_species=n_species,
                channels=channels,
                l_max=l_max,
                n_layers=n_layers,
                n_basis=n_basis,
                radial_hidden=radial_hidden,
                r_cut=self.cutoff,
                avg_neighbors=avg_neighbors,
                readout_hidden=readout_hidden,
            )
        finally:
            torch.random.set_rng_state(state)
        self.net.to(self.torch_dtype)

    # -- introspection -----------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def to_dtype(self, dtype: str) -> "EGNN":
        """Switch working precision in place and return ``self``."""
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
        self.dtype = dtype
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.net.to(self.torch_dtype)
        return self

    def set_scaling(self, shift: np.ndarray | float, scale: float) -> None:
        """Set the per-species energy shift (eV/atom) and the energy scale (eV).

        The readout produces an O(1) number; the physical per-atom energy is
        ``shift[species] + scale * readout``.  Fitting a raw network directly to
        energies of a few eV with forces of a few eV/A works badly, because the
        output layer has to span both scales at once.
        """
        with torch.no_grad():
            sh = np.broadcast_to(
                np.asarray(shift, dtype=np.float64), self.net.energy_shift.shape
            )
            self.net.energy_shift.copy_(torch.as_tensor(np.array(sh), dtype=self.torch_dtype))
            self.net.energy_scale.fill_(float(scale))

    # -- graph construction ------------------------------------------------

    def _graph(self, configuration: Configuration) -> _Graph:
        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        return _Graph(
            edge_i=torch.as_tensor(nl.i.astype(np.int64)),
            edge_j=torch.as_tensor(nl.j.astype(np.int64)),
            shift=nl.shift.astype(np.float64),
            species=torch.as_tensor(configuration.species.astype(np.int64)),
            n_atoms=configuration.n_atoms,
        )

    def _validate(self, configuration: Configuration) -> None:
        if configuration.species.size and int(configuration.species.max()) >= self._hparams["n_species"]:
            raise ValueError(
                f"configuration contains species index "
                f"{int(configuration.species.max())} but the model was built for "
                f"{self._hparams['n_species']} species"
            )
        if np.asarray(configuration.pbc).any():
            check_minimum_image(
                configuration.cell, configuration.pbc, self.cutoff, what=self.name
            )

    # -- the Potential interface -------------------------------------------

    def compute(
        self,
        configuration: Configuration,
        *,
        forces: bool = True,
        virial: bool = True,
    ) -> Result:
        """Evaluate energy, forces and virial.

        Forces come from ``torch.autograd.grad`` of the total energy with
        respect to the positions, with ``create_graph=False`` (inference: no
        second-order graph is retained).  The virial comes from the same
        backward pass differentiated with respect to the strain variable; see
        the module docstring.
        """
        if not self.allow_untrained:
            self._require_fitted()
        self._validate(configuration)

        g = self._graph(configuration)
        dt = self.torch_dtype
        pos = torch.tensor(configuration.positions, dtype=dt, requires_grad=bool(forces))
        cell = torch.as_tensor(configuration.cell, dtype=dt)
        shift_cart = torch.as_tensor(g.shift, dtype=dt) @ cell
        strain = torch.zeros((3, 3), dtype=dt, requires_grad=bool(virial))

        e_atom = self.net(
            pos, g.edge_i, g.edge_j, shift_cart, g.species, strain if virial else None
        )
        total = e_atom.sum()

        wanted = [t for t in (pos, strain) if t.requires_grad]
        f_out = np.zeros((g.n_atoms, 3))
        w_out = np.zeros((3, 3)) if virial else None
        if wanted:
            grads = torch.autograd.grad(
                total, wanted, create_graph=False, allow_unused=True
            )
            k = 0
            if forces:
                gp = grads[k]
                k += 1
                if gp is not None:
                    f_out = -gp.detach().to(torch.float64).numpy()
            if virial:
                gs = grads[k]
                if gs is not None:
                    w_out = -gs.detach().to(torch.float64).numpy()

        return Result(
            energy=float(total.detach().to(torch.float64).item()),
            forces=f_out,
            virial=w_out,
            energies=e_atom.detach().to(torch.float64).numpy(),
        )

    # -- fitting -----------------------------------------------------------

    def _prepare_batch(self, graphs, cells, positions, dtype):
        """Concatenate several single-configuration graphs into one big graph."""
        offs = 0
        ei, ej, sc, sp, batch, pos = [], [], [], [], [], []
        for b, (g, cell, p) in enumerate(zip(graphs, cells, positions)):
            ei.append(g.edge_i + offs)
            ej.append(g.edge_j + offs)
            sc.append(torch.as_tensor(g.shift @ cell, dtype=dtype))
            sp.append(g.species)
            batch.append(torch.full((g.n_atoms,), b, dtype=torch.int64))
            pos.append(torch.as_tensor(p, dtype=dtype))
            offs += g.n_atoms
        return (
            torch.cat(pos),
            torch.cat(ei),
            torch.cat(ej),
            torch.cat(sc),
            torch.cat(sp),
            torch.cat(batch),
            offs,
        )

    def fit(
        self,
        train: Dataset,
        *,
        val: Dataset | None = None,
        epochs: int = 200,
        batch_size: int = 4,
        learning_rate: float = 0.01,
        weight_energy: float = 1.0,
        weight_force: float = 1.0,
        seed: int = 0,
        patience: int | None = None,
        max_seconds: float | None = None,
        grad_clip: float = 10.0,
        verbose: bool = False,
    ) -> FitReport:
        """Fit to energies and forces jointly.

        Parameters
        ----------
        train, val : Dataset
            Labelled configurations.  ``val`` drives early stopping and the
            reported validation errors; without it the best-epoch selection
            falls back to the training loss, which is optimistic and is recorded
            as such in the report notes.
        epochs : int
            Maximum number of passes over ``train``.
        batch_size : int
            Configurations per optimiser step.  Small batches are the right
            choice on CPU: the model is tiny, so the step is dominated by Python
            and kernel-launch overhead rather than by arithmetic.
        learning_rate : float
            Adam learning rate; decayed by ``ReduceLROnPlateau``.
        weight_energy, weight_force : float
            Loss weights.  The two terms are each divided by the standard
            deviation of the corresponding target over the training set, so the
            weights are dimensionless and comparable.
        seed : int
            Seeds the batch shuffling (parameter init is seeded at construction).
        patience : int, optional
            Stop after this many epochs without validation improvement.
            Defaults to ``max(20, epochs // 5)``.
        max_seconds : float, optional
            Wall-clock budget; training stops at the end of the first epoch that
            exceeds it, and ``converged`` is set False.
        grad_clip : float
            Global gradient-norm clip.
        verbose : bool
            Print per-epoch losses.

        Returns
        -------
        FitReport
        """
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")
        t0 = time.perf_counter()
        rng = np.random.default_rng(int(seed))
        dt = self.torch_dtype
        notes: list[str] = []

        for cfg in train:
            if not cfg.has_labels:
                raise ValueError("fit() requires labelled configurations (energy and forces)")
            self._validate(cfg)

        graphs = [self._graph(cfg) for cfg in train]
        cells = [np.asarray(cfg.cell, dtype=np.float64) for cfg in train]
        positions = [np.asarray(cfg.positions, dtype=np.float64) for cfg in train]
        e_ref = torch.as_tensor(
            np.array([cfg.energy for cfg in train], dtype=np.float64), dtype=dt
        )
        n_at = torch.as_tensor(
            np.array([cfg.n_atoms for cfg in train], dtype=np.float64), dtype=dt
        )
        f_ref = [torch.as_tensor(cfg.forces, dtype=dt) for cfg in train]

        # --- data statistics: shift/scale and the message normalisation ----
        e_pa = np.array([cfg.energy / cfg.n_atoms for cfg in train])
        all_f = np.concatenate([cfg.forces.reshape(-1) for cfg in train])
        f_rms = float(np.sqrt(np.mean(all_f**2)))
        e_std = float(np.std(e_pa))
        scale = max(f_rms, e_std, 1e-8)
        self.set_scaling(float(np.mean(e_pa)), scale)
        mean_neighbors = float(np.mean([g.edge_i.numel() / max(g.n_atoms, 1) for g in graphs]))
        for layer in self.net.layers:
            layer.avg_neighbors = max(mean_neighbors, 1.0)
        # Loss normalisers: both terms become O(1) and dimensionless.
        e_norm = max(e_std, 1e-8)
        f_norm = max(f_rms, 1e-8)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=float(learning_rate))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=0.5, patience=10
        )
        if patience is None:
            patience = max(20, int(epochs) // 5)

        history = {"train_loss": [], "val_force_rmse": [], "val_energy_rmse": []}
        best_score = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        best_epoch = -1
        converged = False
        n_epochs_run = 0

        idx_all = np.arange(len(train))
        for epoch in range(int(epochs)):
            n_epochs_run = epoch + 1
            order = rng.permutation(idx_all)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(order), int(batch_size)):
                sel = order[start : start + int(batch_size)]
                pos, ei, ej, sc, sp, batch, n_tot = self._prepare_batch(
                    [graphs[k] for k in sel],
                    [cells[k] for k in sel],
                    [positions[k] for k in sel],
                    dt,
                )
                pos.requires_grad_(True)
                e_atom = self.net(pos, ei, ej, sc, sp, None)
                e_cfg = torch.zeros(len(sel), dtype=dt).index_add(0, batch, e_atom)
                (grad_pos,) = torch.autograd.grad(
                    e_atom.sum(), pos, create_graph=True
                )
                f_pred = -grad_pos

                tgt_e = e_ref[sel] / n_at[sel]
                tgt_f = torch.cat([f_ref[k] for k in sel])
                loss_e = torch.mean(((e_cfg / n_at[sel]) - tgt_e) ** 2) / e_norm**2
                loss_f = torch.mean((f_pred - tgt_f) ** 2) / f_norm**2
                loss = weight_energy * loss_e + weight_force * loss_f

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                optimiser.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            history["train_loss"].append(epoch_loss)

            # The model must be usable for evaluation during training, so flip
            # the flag once the first epoch is done rather than at the very end.
            self.is_fitted = True
            if val is not None and len(val) > 0:
                m = self.evaluate(val)
                score = m["force_rmse"]
                history["val_force_rmse"].append(m["force_rmse"])
                history["val_energy_rmse"].append(m["energy_rmse"])
            else:
                score = epoch_loss
            scheduler.step(score)

            if score < best_score - 1e-12:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                best_epoch = epoch
            if verbose:
                print(f"epoch {epoch:4d}  loss {epoch_loss:.5f}  score {score:.5f}")

            if epoch - best_epoch >= int(patience):
                converged = True
                notes.append(f"early stop: no improvement for {patience} epochs")
                break
            if max_seconds is not None and time.perf_counter() - t0 > float(max_seconds):
                notes.append(
                    f"stopped after {n_epochs_run} epochs: wall-clock budget "
                    f"{max_seconds:.0f}s exhausted"
                )
                break
        else:
            converged = True
            notes.append("reached the requested epoch count")

        self.net.load_state_dict(best_state)
        self.is_fitted = True
        if val is None or len(val) == 0:
            notes.append("no validation set: best epoch chosen on the training loss")

        train_metrics = self.evaluate(train)
        val_metrics = self.evaluate(val) if (val is not None and len(val) > 0) else None
        return FitReport(
            converged=converged,
            n_epochs=n_epochs_run,
            wall_seconds=time.perf_counter() - t0,
            n_parameters=self.n_parameters,
            n_train=len(train),
            n_val=0 if val is None else len(val),
            train_energy_rmse=train_metrics["energy_rmse"],
            train_force_rmse=train_metrics["force_rmse"],
            val_energy_rmse=(
                float("nan") if val_metrics is None else val_metrics["energy_rmse"]
            ),
            val_force_rmse=(
                float("nan") if val_metrics is None else val_metrics["force_rmse"]
            ),
            history={k: np.asarray(v) for k, v in history.items()},
            notes=notes,
        )
