"""Behler-Parrinello atom-centred symmetry functions (ACSF) with analytic derivatives.

The descriptor maps the environment of every atom onto a fixed-length vector of
rotationally, translationally and permutationally invariant numbers built from
the neighbour distances (``G2``) and the neighbour bond angles (``G4``/``G5``).
Together with a per-species multilayer perceptron this is the original
Behler-Parrinello neural network potential; here it is also the input to the
linear "ACE-lite" baseline.

Definitions (all lengths in angstrom, all features dimensionless)
----------------------------------------------------------------

Radial::

    G2_i^{s,m} = sum_{j in s} exp(-eta_m (r_ij - Rs_m)^2) fc(r_ij)

Angular, over unordered pairs of neighbours ``j < k``::

    G4_i^{st,m} = 2^(1-zeta_m) sum_{j<k}
                  (1 + lambda_m cos(theta_ijk))^zeta_m
                  exp(-eta_m (r_ij^2 + r_ik^2 + r_jk^2))
                  fc(r_ij) fc(r_ik) fc(r_jk)

    G5_i^{st,m} = 2^(1-zeta_m) sum_{j<k}
                  (1 + lambda_m cos(theta_ijk))^zeta_m
                  exp(-eta_m (r_ij^2 + r_ik^2))
                  fc(r_ij) fc(r_ik)

``G5`` drops the ``r_jk`` factors: it is cheaper (no third cutoff evaluation,
no third gradient) and, more importantly, it keeps contributions from
neighbours that are both inside ``Rc`` of the centre but further than ``Rc``
from each other, which ``G4`` discards.  Both are provided because that
difference is a genuine modelling choice, not an implementation detail.

Cutoff functions
----------------

Two are offered, and both satisfy ``fc(Rc) = 0`` **and** ``fc'(Rc) = 0``::

    cosine:  fc(r) = 0.5 (cos(pi r / Rc) + 1)
    tanh:    fc(r) = tanh^3(1 - r / Rc)

The vanishing *derivative* is what matters here.  A cutoff that only made the
value continuous would leave a step in ``dG/dr`` at ``Rc``, which enters the
forces directly and would show up in molecular dynamics as an energy drift with
no counterpart in any static test-set metric -- exactly the class of silent
failure this repository exists to study.  ``tests/test_acsf.py`` scans a
neighbour through ``Rc`` and checks both the feature and its derivative.

Species resolution
------------------

Radial features are resolved by neighbour species and angular features by the
**unordered** species pair ``{s, t}`` of the two neighbours.  Without this an
alloy would be featurised as if all its atoms were identical, and the descriptor
would not even be injective on two-component structures.  The dimension is
therefore ``S * n_radial + S(S+1)/2 * n_angular``.

Derivative layout
-----------------

:meth:`ACSF.compute` returns a :class:`~atomlab.models.base.DescriptorOutput`
whose ``derivatives`` array is ``(P, D, 3)`` with ``derivatives[p, d]`` equal to
``dG[pair_i[p], d] / dr[pair_j[p]]``.  One row is emitted per distinct
``(centre, mover)`` pair appearing in the neighbour list, plus one self row per
atom.  Contributions reaching the same physical neighbour through several
periodic images are summed into that neighbour's single row, so the layout is
compact and unambiguous.

Sizing
------

:meth:`ACSF.default` produces 36 features for a single species (12 radial + 24
angular) and 44 for two (20 radial + 24 angular).  That is deliberately at the
small end of what the literature uses.  Published sets of 100+ symmetry
functions are affordable on a GPU; here every model in the zoo has to be
trainable on a few thousand 64-atom configurations in minutes on four CPU
cores, and the angular part costs ``O(n_neighbours^2)`` per atom.  A 36-feature
set fits Lennard-Jones and Stillinger-Weber to well below their own physical
error scales, which is what the study needs.

Implementations
---------------

The inner loops exist twice: a ``numba``-jitted kernel used by default and a
vectorised pure-NumPy reference selected with ``implementation="numpy"``.  They
are checked against each other in the test suite; a fast wrong kernel is the
most expensive kind of bug in this package.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit

from ...neighbors import build_neighbor_list, pair_vectors
from ...types import Configuration
from ..base import Descriptor, DescriptorOutput

__all__ = ["ACSF", "cutoff_function", "CUTOFF_KINDS"]


#: Name -> integer code for the cutoff functions understood by the kernels.
CUTOFF_KINDS: dict[str, int] = {"cosine": 0, "tanh": 1}


# --------------------------------------------------------------------------
# cutoff functions
# --------------------------------------------------------------------------


def cutoff_function(r, cutoff: float, kind: str = "cosine"):
    """Cutoff function and its first derivative, vectorised.

    Parameters
    ----------
    r : array_like
        Distances in angstrom.
    cutoff : float
        ``Rc`` in angstrom.
    kind : {'cosine', 'tanh'}
        ``cosine`` is ``0.5 (cos(pi r / Rc) + 1)``; ``tanh`` is
        ``tanh^3(1 - r / Rc)``.

    Returns
    -------
    f : ndarray
        ``fc(r)``, dimensionless, exactly zero for ``r >= Rc``.
    df : ndarray
        ``dfc/dr`` in 1/angstrom, exactly zero for ``r >= Rc``.

    Notes
    -----
    Both variants have ``fc(Rc) = fc'(Rc) = 0``, so a neighbour entering or
    leaving the cutoff sphere changes neither the feature nor its gradient
    discontinuously.  Only the *second* derivative jumps, which is harmless for
    forces and matters only to a finite-difference Hessian evaluated exactly at
    ``Rc``.
    """
    code = _cutoff_code(kind)
    x = np.asarray(r, dtype=np.float64)
    inside = x < cutoff
    xs = np.where(inside, x, 0.0)
    if code == 0:
        arg = np.pi * xs / cutoff
        f = 0.5 * (np.cos(arg) + 1.0)
        df = -0.5 * np.pi / cutoff * np.sin(arg)
    else:
        t = np.tanh(1.0 - xs / cutoff)
        f = t**3
        df = -3.0 / cutoff * t * t * (1.0 - t * t)
    return np.where(inside, f, 0.0), np.where(inside, df, 0.0)


def _cutoff_code(kind: str) -> int:
    try:
        return CUTOFF_KINDS[str(kind).lower()]
    except KeyError:
        raise ValueError(
            f"unknown cutoff function {kind!r}; known: {sorted(CUTOFF_KINDS)}"
        ) from None


@njit(cache=True)
def _fc_scalar(r, rcut, kind):
    """Scalar ``(fc, dfc/dr)``; mirrors :func:`cutoff_function` exactly."""
    if r >= rcut:
        return 0.0, 0.0
    if kind == 0:
        x = math.pi * r / rcut
        return 0.5 * (math.cos(x) + 1.0), -0.5 * math.pi / rcut * math.sin(x)
    t = math.tanh(1.0 - r / rcut)
    return t * t * t, -3.0 / rcut * t * t * (1.0 - t * t)


# --------------------------------------------------------------------------
# jitted kernel
# --------------------------------------------------------------------------


@njit(cache=True)
def _acsf_kernel(
    disp,
    dist,
    nl_j,
    first,
    count,
    sp_local,
    n_species,
    pair_block,
    r_eta,
    r_rs,
    a_eta,
    a_zeta,
    a_lam,
    a_pref,
    a_is_g4,
    rcut,
    fc_kind,
    slot,
    self_row,
    features,
    derivs,
    want_deriv,
):
    """Accumulate features and ``dG/dr`` for every atom.

    Parameters are the flattened form of everything :meth:`ACSF.compute`
    prepares; see that method for shapes.  Writes into ``features`` ``(N, D)``
    and ``derivs`` ``(P, D, 3)`` in place.

    Derivative derivation (this is the part that is usually wrong)
    -------------------------------------------------------------
    With ``d_ij = r_j + shift@cell - r_i`` (pointing i -> j), ``a = |d_ij|``,
    ``b = |d_ik|``, ``c = |d_ik - d_ij| = |r_k - r_j|`` and unit vectors
    ``u = d/|d|``::

        grad_j a = u_ij        grad_i a = -u_ij       grad_k a = 0
        grad_k b = u_ik        grad_i b = -u_ik       grad_j b = 0
        grad_k c = u_jk        grad_j c = -u_jk       grad_i c = 0

    and for ``cos(theta) = (d_ij . d_ik) / (a b)``::

        grad_j cos = ( u_ik - cos u_ij ) / a
        grad_k cos = ( u_ij - cos u_ik ) / b
        grad_i cos = -(grad_j cos + grad_k cos)

    The last line is *not* an approximation: ``cos(theta)`` depends on positions
    only through ``d_ij`` and ``d_ik``, both of which are translation
    invariant, so the three gradients must sum to zero.  Expanding it by hand
    gives ``-u_ik/a - u_ij/b + cos(u_ij/a + u_ik/b)``, which is the same thing;
    the sum form is used here because it makes translational invariance exact in
    floating point rather than merely true in exact arithmetic.  The same
    reasoning is applied to the whole angular term, whose centre-atom gradient
    is accumulated as minus the sum of the two neighbour gradients.
    """
    n_atoms = features.shape[0]
    n_r = r_eta.shape[0]
    n_a = a_eta.shape[0]
    ang_offset = n_species * n_r

    for i in range(n_atoms):
        st = first[i]
        ct = count[i]
        if ct == 0:
            continue

        # Cache per-neighbour geometry once: it is reused O(ct) times by the
        # angular double loop, where recomputing it would dominate the cost.
        fcv = np.empty(ct)
        dfcv = np.empty(ct)
        ux = np.empty(ct)
        uy = np.empty(ct)
        uz = np.empty(ct)
        for t in range(ct):
            p = st + t
            rp = dist[p]
            f, df = _fc_scalar(rp, rcut, fc_kind)
            fcv[t] = f
            dfcv[t] = df
            inv = 1.0 / rp
            ux[t] = disp[p, 0] * inv
            uy[t] = disp[p, 1] * inv
            uz[t] = disp[p, 2] * inv

        srow = self_row[i]

        # ---- radial G2 -------------------------------------------------
        for t in range(ct):
            if fcv[t] == 0.0:
                continue  # r >= Rc: fc and fc' both vanish, nothing to add
            p = st + t
            rp = dist[p]
            base = sp_local[nl_j[p]] * n_r
            prow = slot[p]
            for m in range(n_r):
                d = rp - r_rs[m]
                e = math.exp(-r_eta[m] * d * d)
                features[i, base + m] += e * fcv[t]
                if want_deriv:
                    # d/dr [ exp(-eta (r-Rs)^2) fc(r) ]
                    dg = e * (-2.0 * r_eta[m] * d * fcv[t] + dfcv[t])
                    gx = dg * ux[t]
                    gy = dg * uy[t]
                    gz = dg * uz[t]
                    derivs[prow, base + m, 0] += gx
                    derivs[prow, base + m, 1] += gy
                    derivs[prow, base + m, 2] += gz
                    derivs[srow, base + m, 0] -= gx
                    derivs[srow, base + m, 1] -= gy
                    derivs[srow, base + m, 2] -= gz

        # ---- angular G4 / G5 -------------------------------------------
        if n_a == 0:
            continue
        for t in range(ct):
            if fcv[t] == 0.0:
                continue
            p = st + t
            a = dist[p]
            sj = sp_local[nl_j[p]]
            prow = slot[p]
            for u in range(t + 1, ct):
                if fcv[u] == 0.0:
                    continue
                q = st + u
                b = dist[q]
                sk = sp_local[nl_j[q]]
                qrow = slot[q]
                base = ang_offset + pair_block[sj, sk] * n_a

                cosv = ux[t] * ux[u] + uy[t] * uy[u] + uz[t] * uz[u]
                cx = disp[q, 0] - disp[p, 0]
                cy = disp[q, 1] - disp[p, 1]
                cz = disp[q, 2] - disp[p, 2]
                c = math.sqrt(cx * cx + cy * cy + cz * cz)
                fcc, dfcc = _fc_scalar(c, rcut, fc_kind)
                if c > 0.0:
                    ucx = cx / c
                    ucy = cy / c
                    ucz = cz / c
                else:  # two neighbours at the identical point: unreachable in
                    ucx = 0.0  # any physical configuration, guarded anyway
                    ucy = 0.0
                    ucz = 0.0

                inv_a = 1.0 / a
                inv_b = 1.0 / b
                gjx = (ux[u] - cosv * ux[t]) * inv_a
                gjy = (uy[u] - cosv * uy[t]) * inv_a
                gjz = (uz[u] - cosv * uz[t]) * inv_a
                gkx = (ux[t] - cosv * ux[u]) * inv_b
                gky = (uy[t] - cosv * uy[u]) * inv_b
                gkz = (uz[t] - cosv * uz[u]) * inv_b

                s2 = a * a + b * b
                s3 = s2 + c * c

                for m in range(n_a):
                    g4 = a_is_g4[m]
                    if g4 and fcc == 0.0:
                        continue  # r_jk outside Rc kills the G4 triple entirely
                    zeta = a_zeta[m]
                    lam = a_lam[m]
                    eta = a_eta[m]

                    amp = 1.0 + lam * cosv
                    if amp < 0.0:
                        # Mathematically amp lies in [0, 2]; round-off in cos
                        # can push it a few ulp negative and a fractional power
                        # of a negative number is a NaN, so clamp.
                        amp = 0.0

                    if g4:
                        expo = math.exp(-eta * s3)
                        fprod = fcv[t] * fcv[u] * fcc
                    else:
                        expo = math.exp(-eta * s2)
                        fprod = fcv[t] * fcv[u]

                    amp_z = amp**zeta
                    val = a_pref[m] * amp_z * expo * fprod
                    features[i, base + m] += val

                    if want_deriv:
                        common = a_pref[m] * amp_z * expo
                        # d/dcos of the angular factor
                        dcos = a_pref[m] * zeta * lam * amp ** (zeta - 1.0) * expo * fprod
                        # d/da, d/db, d/dc of exp(...) * prod fc(...)
                        if g4:
                            dta = common * fcv[u] * fcc * (dfcv[t] - 2.0 * eta * a * fcv[t])
                            dtb = common * fcv[t] * fcc * (dfcv[u] - 2.0 * eta * b * fcv[u])
                            dtc = common * fcv[t] * fcv[u] * (dfcc - 2.0 * eta * c * fcc)
                        else:
                            dta = common * fcv[u] * (dfcv[t] - 2.0 * eta * a * fcv[t])
                            dtb = common * fcv[t] * (dfcv[u] - 2.0 * eta * b * fcv[u])
                            dtc = 0.0

                        # gradient with respect to neighbour j
                        jx = dcos * gjx + dta * ux[t] - dtc * ucx
                        jy = dcos * gjy + dta * uy[t] - dtc * ucy
                        jz = dcos * gjz + dta * uz[t] - dtc * ucz
                        # gradient with respect to neighbour k
                        kx = dcos * gkx + dtb * ux[u] + dtc * ucx
                        ky = dcos * gky + dtb * uy[u] + dtc * ucy
                        kz = dcos * gkz + dtb * uz[u] + dtc * ucz

                        derivs[prow, base + m, 0] += jx
                        derivs[prow, base + m, 1] += jy
                        derivs[prow, base + m, 2] += jz
                        derivs[qrow, base + m, 0] += kx
                        derivs[qrow, base + m, 1] += ky
                        derivs[qrow, base + m, 2] += kz
                        # centre gradient = -(j + k), exactly (see docstring)
                        derivs[srow, base + m, 0] -= jx + kx
                        derivs[srow, base + m, 1] -= jy + ky
                        derivs[srow, base + m, 2] -= jz + kz


# --------------------------------------------------------------------------
# pure-NumPy reference kernel
# --------------------------------------------------------------------------


def _acsf_reference(
    disp,
    dist,
    nl_j,
    first,
    count,
    sp_local,
    n_species,
    pair_block,
    r_eta,
    r_rs,
    a_eta,
    a_zeta,
    a_lam,
    a_pref,
    a_is_g4,
    rcut,
    fc_kind,
    slot,
    self_row,
    features,
    derivs,
    want_deriv,
):
    """Vectorised NumPy reference for :func:`_acsf_kernel`.

    Same arguments, same in-place outputs.  Written independently of the jitted
    version -- it broadcasts over symmetry-function parameters and over the
    ``ct*(ct-1)/2`` neighbour pairs of each atom instead of looping -- so that
    agreement between the two is evidence about the maths and not merely about
    a shared transcription.
    """
    n_atoms = features.shape[0]
    n_r = r_eta.shape[0]
    n_a = a_eta.shape[0]
    ang_offset = n_species * n_r
    r_index = np.arange(n_r)
    a_index = np.arange(n_a)

    for i in range(n_atoms):
        st = int(first[i])
        ct = int(count[i])
        if ct == 0:
            continue
        sl = slice(st, st + ct)
        rr = dist[sl]
        dd = disp[sl]
        rows = slot[sl]
        srow = int(self_row[i])
        sp = sp_local[nl_j[sl]]

        fcv, dfcv = cutoff_function(rr, rcut, "cosine" if fc_kind == 0 else "tanh")
        unit = dd / rr[:, None]

        # ---- radial ----------------------------------------------------
        delta = rr[:, None] - r_rs[None, :]                       # (ct, n_r)
        expo = np.exp(-r_eta[None, :] * delta * delta)
        g2 = expo * fcv[:, None]
        cols = sp[:, None] * n_r + r_index[None, :]               # (ct, n_r)
        np.add.at(features[i], cols.ravel(), g2.ravel())

        if want_deriv:
            dg2 = expo * (-2.0 * r_eta[None, :] * delta * fcv[:, None] + dfcv[:, None])
            contrib = dg2[:, :, None] * unit[:, None, :]          # (ct, n_r, 3)
            np.add.at(derivs, (rows[:, None], cols), contrib)
            np.add.at(derivs, (np.full_like(cols, srow), cols), -contrib)

        # ---- angular ---------------------------------------------------
        if n_a == 0 or ct < 2:
            continue
        tt, uu = np.triu_indices(ct, k=1)
        a = rr[tt]
        b = rr[uu]
        ua = unit[tt]
        ub = unit[uu]
        cosv = np.einsum("ij,ij->i", ua, ub)
        cvec = dd[uu] - dd[tt]
        c = np.linalg.norm(cvec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            uc = np.where(c[:, None] > 0.0, cvec / np.maximum(c, 1e-300)[:, None], 0.0)

        fca, dfca = fcv[tt], dfcv[tt]
        fcb, dfcb = fcv[uu], dfcv[uu]
        fcc, dfcc = cutoff_function(c, rcut, "cosine" if fc_kind == 0 else "tanh")

        blk = pair_block[sp[tt], sp[uu]]
        acols = ang_offset + blk[:, None] * n_a + a_index[None, :]   # (T, n_a)

        s2 = a * a + b * b
        s3 = s2 + c * c
        g4 = a_is_g4[None, :]
        expo_a = np.where(g4, np.exp(-a_eta[None, :] * s3[:, None]),
                          np.exp(-a_eta[None, :] * s2[:, None]))       # (T, n_a)
        fprod = np.where(g4, (fca * fcb * fcc)[:, None], (fca * fcb)[:, None])

        amp = np.clip(1.0 + a_lam[None, :] * cosv[:, None], 0.0, None)
        amp_z = amp ** a_zeta[None, :]
        val = a_pref[None, :] * amp_z * expo_a * fprod
        np.add.at(features[i], acols.ravel(), val.ravel())

        if not want_deriv:
            continue

        common = a_pref[None, :] * amp_z * expo_a
        dcos = a_pref[None, :] * a_zeta[None, :] * a_lam[None, :] * (
            amp ** (a_zeta[None, :] - 1.0)
        ) * expo_a * fprod
        eta = a_eta[None, :]
        dta = np.where(
            g4,
            common * (fcb * fcc)[:, None] * (dfca[:, None] - 2.0 * eta * (a * fca)[:, None]),
            common * fcb[:, None] * (dfca[:, None] - 2.0 * eta * (a * fca)[:, None]),
        )
        dtb = np.where(
            g4,
            common * (fca * fcc)[:, None] * (dfcb[:, None] - 2.0 * eta * (b * fcb)[:, None]),
            common * fca[:, None] * (dfcb[:, None] - 2.0 * eta * (b * fcb)[:, None]),
        )
        dtc = np.where(
            g4,
            common * (fca * fcb)[:, None] * (dfcc[:, None] - 2.0 * eta * (c * fcc)[:, None]),
            0.0,
        )
        # G4 triples with r_jk >= Rc contribute nothing at all
        alive = np.where(g4, (fcc > 0.0)[:, None], True)
        val = np.where(alive, val, 0.0)
        dcos = np.where(alive, dcos, 0.0)
        dta = np.where(alive, dta, 0.0)
        dtb = np.where(alive, dtb, 0.0)
        dtc = np.where(alive, dtc, 0.0)

        inv_a = (1.0 / a)[:, None]
        inv_b = (1.0 / b)[:, None]
        gj = (ub[:, None, :] - cosv[:, None, None] * ua[:, None, :]) * inv_a[:, :, None]
        gk = (ua[:, None, :] - cosv[:, None, None] * ub[:, None, :]) * inv_b[:, :, None]

        jcontrib = (
            dcos[:, :, None] * gj
            + dta[:, :, None] * ua[:, None, :]
            - dtc[:, :, None] * uc[:, None, :]
        )
        kcontrib = (
            dcos[:, :, None] * gk
            + dtb[:, :, None] * ub[:, None, :]
            + dtc[:, :, None] * uc[:, None, :]
        )
        np.add.at(derivs, (rows[tt][:, None], acols), jcontrib)
        np.add.at(derivs, (rows[uu][:, None], acols), kcontrib)
        np.add.at(derivs, (np.full_like(acols, srow), acols), -(jcontrib + kcontrib))

    # `val` recomputation above already folded the alive mask in for the
    # features of the *derivative* branch only; redo it for the value branch by
    # construction instead (G4 with fcc == 0 has fprod == 0), so nothing to fix.
    return features, derivs


# --------------------------------------------------------------------------
# the descriptor
# --------------------------------------------------------------------------


class ACSF(Descriptor):
    """Behler-Parrinello symmetry functions with exact analytic derivatives.

    Parameters
    ----------
    species : sequence of int
        The species type indices the descriptor is defined for, e.g. ``(0,)``
        or ``(0, 1)``.  A configuration containing any other index is rejected
        rather than silently featurised into the wrong block.
    cutoff : float
        ``Rc`` in angstrom.  The same cutoff is used for the radial functions,
        for both legs of the angular functions and (for ``G4``) for the
        neighbour-neighbour distance.
    radial : array_like, shape (n_radial, 2)
        Rows ``(eta, Rs)`` with ``eta`` in 1/A^2 and ``Rs`` in A.
    angular : array_like, shape (n_angular, 3)
        Rows ``(eta, zeta, lambda)`` with ``eta`` in 1/A^2, ``zeta >= 1``
        dimensionless and ``lambda`` exactly ``+1`` or ``-1``.
    angular_kind : {'G4', 'G5'} or sequence
        Which angular form each row of ``angular`` uses.  A single string
        applies to all rows; a sequence gives one entry per row, so a single
        descriptor may mix the two.
    cutoff_function : {'cosine', 'tanh'}
        See :func:`cutoff_function`.
    name : str
        Identifier used in figures and tables.

    Attributes
    ----------
    n_features : int
        ``S * n_radial + S(S+1)//2 * n_angular``.

    Notes
    -----
    The descriptor is invariant under translation, rotation, permutation of
    identical atoms and the choice of periodic image *by construction*: every
    feature is a sum over neighbours of a function of interatomic distances and
    angles only, and the neighbour list enumerates periodic images explicitly
    rather than applying a minimum-image convention.  The tests check this to
    ~1e-12 anyway, because "by construction" is exactly the kind of claim that
    survives a subtle indexing bug.
    """

    def __init__(
        self,
        species: Sequence[int],
        cutoff: float,
        radial,
        angular=None,
        *,
        angular_kind: str | Sequence[str] = "G4",
        cutoff_function: str = "cosine",
        name: str = "acsf",
    ) -> None:
        sp = np.asarray(list(species), dtype=np.int64)
        if sp.ndim != 1 or sp.size == 0:
            raise ValueError("species must be a non-empty 1-D sequence of type indices")
        if np.any(sp < 0):
            raise ValueError(f"species indices must be >= 0, got {sp}")
        if len(np.unique(sp)) != sp.size:
            raise ValueError(f"species indices must be unique, got {sp}")
        self.species = np.sort(sp)

        self.cutoff = float(cutoff)
        if not self.cutoff > 0.0 or not np.isfinite(self.cutoff):
            raise ValueError(f"cutoff must be a finite positive length, got {cutoff}")

        rad = np.atleast_2d(np.asarray(radial, dtype=np.float64))
        if rad.ndim != 2 or rad.shape[1] != 2:
            raise ValueError(f"radial must be (n_radial, 2) of (eta, Rs), got {rad.shape}")
        if np.any(rad[:, 0] < 0.0):
            raise ValueError("radial eta must be >= 0 (1/A^2)")
        if np.any(rad[:, 1] < 0.0):
            raise ValueError("radial Rs must be >= 0 (A)")
        self.radial = rad

        if angular is None or len(angular) == 0:
            ang = np.zeros((0, 3), dtype=np.float64)
        else:
            ang = np.atleast_2d(np.asarray(angular, dtype=np.float64))
        if ang.ndim != 2 or ang.shape[1] != 3:
            raise ValueError(
                f"angular must be (n_angular, 3) of (eta, zeta, lambda), got {ang.shape}"
            )
        if ang.size:
            if np.any(ang[:, 0] < 0.0):
                raise ValueError("angular eta must be >= 0 (1/A^2)")
            if np.any(ang[:, 1] < 1.0):
                # zeta < 1 makes d/dcos ~ amp^(zeta-1) singular where the
                # angular factor vanishes, i.e. at theta = +-pi.
                raise ValueError("angular zeta must be >= 1")
            if not np.all(np.isin(ang[:, 2], (-1.0, 1.0))):
                raise ValueError("angular lambda must be exactly +1 or -1")
        self.angular = ang

        n_a = ang.shape[0]
        if isinstance(angular_kind, str):
            kinds = [angular_kind] * n_a
        else:
            kinds = [str(k) for k in angular_kind]
            if len(kinds) != n_a:
                raise ValueError(
                    f"angular_kind has {len(kinds)} entries for {n_a} angular functions"
                )
        norm = []
        for k in kinds:
            key = str(k).upper()
            if key not in ("G4", "G5"):
                raise ValueError(f"angular_kind must be 'G4' or 'G5', got {k!r}")
            norm.append(key)
        self.angular_kind = tuple(norm)
        self._is_g4 = np.array([k == "G4" for k in norm], dtype=np.bool_)

        self.cutoff_kind = str(cutoff_function).lower()
        self._fc_code = _cutoff_code(self.cutoff_kind)
        self.name = str(name)

        # Derived tables -------------------------------------------------
        self._n_species = int(self.species.size)
        self._sp_map = np.full(int(self.species.max()) + 1, -1, dtype=np.int64)
        self._sp_map[self.species] = np.arange(self._n_species)

        s = self._n_species
        block = np.zeros((s, s), dtype=np.int64)
        idx = 0
        for x in range(s):
            for y in range(x, s):
                block[x, y] = block[y, x] = idx
                idx += 1
        self._pair_block = block
        self._n_species_pairs = idx

        self._r_eta = np.ascontiguousarray(rad[:, 0])
        self._r_rs = np.ascontiguousarray(rad[:, 1])
        self._a_eta = np.ascontiguousarray(ang[:, 0]) if n_a else np.zeros(0)
        self._a_zeta = np.ascontiguousarray(ang[:, 1]) if n_a else np.zeros(0)
        self._a_lam = np.ascontiguousarray(ang[:, 2]) if n_a else np.zeros(0)
        self._a_pref = 2.0 ** (1.0 - self._a_zeta)

    # -- factory -----------------------------------------------------------

    @classmethod
    def default(
        cls,
        species: Sequence[int],
        cutoff: float,
        *,
        angular_kind: str = "G4",
        cutoff_function: str = "cosine",
        name: str = "acsf",
    ) -> "ACSF":
        """A general-purpose parameter set sized for four CPU cores.

        Parameters
        ----------
        species : sequence of int
            Type indices present in the systems to be featurised.
        cutoff : float
            ``Rc`` in angstrom.  Every parameter below is expressed relative to
            it, so the set transfers between the ~8.5 A cutoff used for
            Lennard-Jones argon and the ~3.8 A one used for Stillinger-Weber
            silicon without retuning.
        angular_kind : {'G4', 'G5'}
        cutoff_function : {'cosine', 'tanh'}

        Returns
        -------
        ACSF
            36 features for one species, 44 for two, 78 for three.

        Notes
        -----
        *Radial.*  ``n_radial`` Gaussians with centres ``Rs`` uniformly spaced
        on ``(0, Rc)`` at the cell centres ``(m + 1/2) Rc / n_radial`` and a
        common width equal to that spacing, ``eta = n_radial^2 / (2 Rc^2)``.
        Uniform coverage rather than a hand-tuned set: the innermost one or two
        functions are essentially zero in an equilibrium crystal, but they are
        what carries the repulsive wall in the compressed and high-temperature
        configurations that ``training/generate.py`` deliberately includes, and
        a constant column costs nothing after
        :func:`~atomlab.models.base.standardize`.

        *Angular.*  ``zeta`` spans 1 to 16 so the angular resolution ranges from
        very broad to sharply peaked, both ``lambda`` signs are present (they
        select maxima at ``theta = 0`` and ``theta = pi``), and the radial decay
        ``eta`` takes three values scaled as ``c / Rc^2``.

        *Sizing.*  Multi-component sets shrink the per-block grids, because the
        dimension grows as ``S`` (radial) and ``S(S+1)/2`` (angular) and the
        study needs every model trainable on a few thousand 64-atom
        configurations in minutes.  A GPU-scale study would use 3-5x more
        symmetry functions; the accuracy cost here is small compared with the
        physical error scales the experiments resolve.
        """
        rc = float(cutoff)
        n_species = len(list(species))
        n_radial = 12 if n_species == 1 else 10
        centres = (np.arange(n_radial) + 0.5) * rc / n_radial
        width = rc / n_radial
        eta_r = 1.0 / (2.0 * width * width)
        radial = np.column_stack([np.full(n_radial, eta_r), centres])

        if n_species == 1:
            zetas = (1.0, 2.0, 4.0, 16.0)
            etas = np.array([0.05, 0.5, 2.0]) / (rc * rc)
        else:
            zetas = (1.0, 4.0)
            etas = np.array([0.1, 1.0]) / (rc * rc)

        rows = []
        for eta in etas:
            for zeta in zetas:
                for lam in (1.0, -1.0):
                    rows.append((eta, zeta, lam))
        angular = np.array(rows, dtype=np.float64)

        return cls(
            species,
            rc,
            radial,
            angular,
            angular_kind=angular_kind,
            cutoff_function=cutoff_function,
            name=name,
        )

    # -- interface ---------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Descriptor dimension per atom."""
        return int(
            self._n_species * self.radial.shape[0]
            + self._n_species_pairs * self.angular.shape[0]
        )

    @property
    def n_radial(self) -> int:
        """Number of ``(eta, Rs)`` radial parameter rows (per neighbour species)."""
        return int(self.radial.shape[0])

    @property
    def n_angular(self) -> int:
        """Number of angular parameter rows (per unordered neighbour-species pair)."""
        return int(self.angular.shape[0])

    def feature_labels(self) -> list[str]:
        """Human-readable name of every feature, in column order."""
        labels = []
        for s in self.species:
            for eta, rs in self.radial:
                labels.append(f"G2[{s}] eta={eta:g} Rs={rs:g}")
        for a in range(self._n_species):
            for b in range(a, self._n_species):
                sa, sb = int(self.species[a]), int(self.species[b])
                for k, (eta, zeta, lam) in zip(self.angular_kind, self.angular):
                    labels.append(
                        f"{k}[{sa},{sb}] eta={eta:g} zeta={zeta:g} lam={lam:+g}"
                    )
        return labels

    def compute(
        self,
        configuration: Configuration,
        *,
        derivatives: bool = True,
        implementation: str = "numba",
    ) -> DescriptorOutput:
        """Featurise every atom of ``configuration``.

        Parameters
        ----------
        configuration : Configuration
            Geometry in angstrom.  Not modified.
        derivatives : bool
            If False, ``DescriptorOutput.derivatives`` is ``None`` and the
            angular gradient work is skipped (roughly a factor 3 faster).
        implementation : {'numba', 'numpy'}
            Which kernel to run.  ``'numpy'`` is the reference used to validate
            the jitted one; it is 10-50x slower and exists only for testing.

        Returns
        -------
        DescriptorOutput
            ``features`` ``(N, D)``, dimensionless.  ``derivatives`` ``(P, D,
            3)`` in 1/A with ``derivatives[p, d] = dG[pair_i[p], d] /
            dr[pair_j[p]]``; one row per distinct ``(centre, neighbour)`` pair
            plus one self row ``(i, i)`` per atom.
        """
        if implementation not in ("numba", "numpy"):
            raise ValueError(
                f"implementation must be 'numba' or 'numpy', got {implementation!r}"
            )

        n_atoms = configuration.n_atoms
        n_features = self.n_features
        features = np.zeros((n_atoms, n_features), dtype=np.float64)

        sp_local = self._local_species(configuration)

        nl = build_neighbor_list(configuration, self.cutoff, half=False)
        disp, dist = pair_vectors(configuration, nl)

        # `build_neighbor_list` returns pairs lexsorted by (i, j, shift), so
        # each centre's pairs are contiguous and its neighbours are grouped.
        counts = np.bincount(nl.i, minlength=n_atoms).astype(np.int64)
        first = np.zeros(n_atoms, dtype=np.int64)
        np.cumsum(counts[:-1], out=first[1:])

        n_pairs = nl.n_pairs
        if n_pairs:
            new_group = np.empty(n_pairs, dtype=bool)
            new_group[0] = True
            new_group[1:] = (nl.i[1:] != nl.i[:-1]) | (nl.j[1:] != nl.j[:-1])
            slot = (np.cumsum(new_group) - 1).astype(np.int64)
            head = np.flatnonzero(new_group)
            group_i = nl.i[head].astype(np.int64)
            group_j = nl.j[head].astype(np.int64)
            n_groups = int(head.size)
        else:
            slot = np.zeros(0, dtype=np.int64)
            group_i = np.zeros(0, dtype=np.int64)
            group_j = np.zeros(0, dtype=np.int64)
            n_groups = 0

        self_row = n_groups + np.arange(n_atoms, dtype=np.int64)
        n_rows = n_groups + n_atoms

        want_deriv = bool(derivatives)
        derivs = (
            np.zeros((n_rows, n_features, 3), dtype=np.float64)
            if want_deriv
            else np.zeros((1, 1, 3), dtype=np.float64)
        )

        args = (
            np.ascontiguousarray(disp),
            np.ascontiguousarray(dist),
            nl.j.astype(np.int64),
            first,
            counts,
            sp_local,
            self._n_species,
            self._pair_block,
            self._r_eta,
            self._r_rs,
            self._a_eta,
            self._a_zeta,
            self._a_lam,
            self._a_pref,
            self._is_g4,
            self.cutoff,
            self._fc_code,
            slot,
            self_row,
            features,
            derivs,
            want_deriv,
        )
        if implementation == "numba":
            _acsf_kernel(*args)
        else:
            _acsf_reference(*args)

        if not want_deriv:
            return DescriptorOutput(features=features)

        return DescriptorOutput(
            features=features,
            derivatives=derivs,
            pair_i=np.concatenate([group_i, np.arange(n_atoms, dtype=np.int64)]),
            pair_j=np.concatenate([group_j, np.arange(n_atoms, dtype=np.int64)]),
        )

    # -- helpers -----------------------------------------------------------

    def _local_species(self, configuration: Configuration) -> np.ndarray:
        """Map ``configuration.species`` onto ``0 .. S-1``; raise on unknowns."""
        sp = np.asarray(configuration.species, dtype=np.int64)
        if sp.size and (sp.min() < 0 or sp.max() >= self._sp_map.size):
            raise ValueError(
                "configuration contains species indices outside "
                f"{[int(s) for s in self.species]}"
            )
        local = self._sp_map[sp] if sp.size else np.zeros(0, dtype=np.int64)
        if np.any(local < 0):
            bad = sorted(set(sp[local < 0].tolist()))
            raise ValueError(
                f"{self.name} was built for species {[int(s) for s in self.species]} but the "
                f"configuration contains {bad}; featurising it would put those "
                f"atoms in the wrong species block"
            )
        return np.ascontiguousarray(local)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kinds = "/".join(sorted(set(self.angular_kind))) if self.angular_kind else "none"
        return (
            f"ACSF(species={[int(s) for s in self.species]}, cutoff={self.cutoff:g} A, "
            f"n_radial={self.n_radial}, n_angular={self.n_angular} [{kinds}], "
            f"n_features={self.n_features})"
        )
