"""Structure builders: crystals, gases, defects, strained cells.

Every system this project simulates enters through this module, so the lattices
have to be exactly right -- a wrong basis here would silently contaminate every
cohesive energy, phonon spectrum and elastic constant downstream.

Conventions
-----------

*Cell.*  As everywhere in :mod:`atomlab`, ``cell`` is ``(3, 3)`` with the
lattice vectors as **rows**, so cartesian and fractional coordinates are related
by ``r = s @ cell``.  A builder therefore places its basis as
``positions = basis_fractional @ cell``.

*Conventional cells.*  ``sc``/``bcc``/``fcc``/``diamond`` return the
**conventional cubic** cell (1/2/4/8 atoms), not the primitive one.  This is the
convention the rest of the repository assumes when it says "an ``n x n x n``
supercell of fcc has ``4 n^3`` atoms", and it keeps the cell orthogonal, which
makes minimum-image reasoning and slab construction straightforward.  ``hcp``
has no cubic conventional cell, so it returns the 2-atom primitive hexagonal
cell.

*Strain sign.*  Strain helpers apply ``r -> (1 + eps) r`` to atoms *and* lattice
vectors via :meth:`Configuration.strained`, which is exactly the deformation the
virial ``W = -dU/d(eps)`` is defined against.  Keeping one convention on both
sides is what makes the elastic-constant fits in
``atomlab/observables/thermo.py`` come out with the right sign.

*Determinism.*  Every stochastic entry point takes an explicit ``seed`` (an int
or a :class:`numpy.random.Generator`).  There is no module-level RNG, and a
routine that needs randomness but is given no seed raises rather than inventing
one.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .types import Configuration
from .units import ATOMIC_MASSES

__all__ = [
    "sc",
    "bcc",
    "fcc",
    "hcp",
    "diamond",
    "lattice",
    "random_gas",
    "rattle",
    "scale_cell",
    "scale_to_density",
    "vacancy",
    "interstitial",
    "substitute",
    "surface_slab",
    "shear",
    "apply_strain",
    "uniaxial_strain",
    "hydrostatic_strain",
    "voigt_to_strain",
    "strain_to_voigt",
    "strain_scan",
    "volume_scan",
    "IDEAL_C_OVER_A",
    "LATTICE_BASES",
]

#: ``c/a`` of the ideal (hard-sphere close-packed) hcp lattice, ``sqrt(8/3)``.
#: At this ratio every one of the 12 nearest neighbours sits at exactly ``a``.
IDEAL_C_OVER_A = math.sqrt(8.0 / 3.0)

#: Fractional basis vectors of the conventional cubic cells.
LATTICE_BASES: dict[str, np.ndarray] = {
    "sc": np.array([[0.0, 0.0, 0.0]]),
    "bcc": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    "fcc": np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ),
    # Diamond is fcc with a second fcc sublattice displaced by (1/4, 1/4, 1/4)
    # along the body diagonal; that displacement of a*sqrt(3)/4 is the bond.
    "diamond": np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    ),
}


# --------------------------------------------------------------------------
# small internal helpers
# --------------------------------------------------------------------------


def _rng(seed) -> np.random.Generator:
    """Coerce ``seed`` (int, ``None``-free) or a Generator into a Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        raise ValueError(
            "a seed (int or numpy.random.Generator) is required: stochastic "
            "builders in atomlab are deterministic by contract"
        )
    return np.random.default_rng(seed)


def _mass_of(symbol: str) -> float:
    """Atomic mass in amu, from :data:`atomlab.units.ATOMIC_MASSES`."""
    try:
        return float(ATOMIC_MASSES[symbol])
    except KeyError:
        raise KeyError(
            f"unknown chemical symbol {symbol!r}; atomlab.units.ATOMIC_MASSES "
            f"knows {sorted(ATOMIC_MASSES)}"
        ) from None


def _as_reps(reps) -> tuple[int, int, int]:
    if np.isscalar(reps):
        reps = (reps, reps, reps)
    r = tuple(int(x) for x in reps)
    if len(r) != 3:
        raise ValueError(f"reps must be a scalar or length-3, got {reps!r}")
    if min(r) < 1:
        raise ValueError(f"repetitions must be >= 1, got {r}")
    return r  # type: ignore[return-value]


def _positive(value: float, name: str) -> float:
    v = float(value)
    if not v > 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _lattice_configuration(
    cell: np.ndarray,
    basis: np.ndarray,
    symbol: str,
    reps,
    info: dict,
) -> Configuration:
    """Assemble a single-species crystal and replicate it."""
    mass = _mass_of(symbol)
    positions = np.asarray(basis, dtype=float) @ cell
    n = positions.shape[0]
    cfg = Configuration(
        positions=positions,
        cell=cell,
        pbc=True,
        species=np.zeros(n, dtype=np.int32),
        symbols=(symbol,),
        masses=np.full(n, mass),
        info=info,
    )
    return cfg.repeated(_as_reps(reps))


# --------------------------------------------------------------------------
# crystal builders
# --------------------------------------------------------------------------


def lattice(
    structure: str,
    a: float,
    symbol: str,
    reps: Sequence[int] | int = (1, 1, 1),
    *,
    c: float | None = None,
) -> Configuration:
    """Dispatch to a named lattice builder.

    Parameters
    ----------
    structure:
        One of ``"sc"``, ``"bcc"``, ``"fcc"``, ``"diamond"``, ``"hcp"``.
    a:
        Lattice constant in angstrom.
    symbol:
        Chemical symbol; must be a key of
        :data:`atomlab.units.ATOMIC_MASSES`.
    reps:
        Supercell repetitions ``(nx, ny, nz)`` (or a single int).
    c:
        Only for ``"hcp"``: the ``c`` axis in angstrom.  Defaults to the ideal
        ``sqrt(8/3) * a``.

    Returns
    -------
    Configuration
        Periodic configuration; positions ``(N, 3)`` in angstrom.
    """
    key = structure.lower()
    if key == "hcp":
        return hcp(a, symbol, c=c, reps=reps)
    if c is not None:
        raise ValueError(f"structure {structure!r} takes no c axis")
    builders = {"sc": sc, "bcc": bcc, "fcc": fcc, "diamond": diamond}
    if key not in builders:
        raise ValueError(f"unknown structure {structure!r}; known: {sorted(builders)} + 'hcp'")
    return builders[key](a, symbol, reps)


def sc(a: float, symbol: str, reps: Sequence[int] | int = (1, 1, 1)) -> Configuration:
    """Simple-cubic crystal, 1 atom per conventional cell.

    Parameters
    ----------
    a:
        Lattice constant in angstrom.
    symbol:
        Chemical symbol; mass is taken from
        :data:`atomlab.units.ATOMIC_MASSES`.
    reps:
        ``(nx, ny, nz)`` supercell repetitions.

    Returns
    -------
    Configuration
        ``N = nx*ny*nz`` atoms, volume per atom ``a**3``, 6 nearest neighbours
        at ``a``.
    """
    a = _positive(a, "a")
    return _lattice_configuration(
        a * np.eye(3), LATTICE_BASES["sc"], symbol, reps, {"lattice": "sc", "a": a}
    )


def bcc(a: float, symbol: str, reps: Sequence[int] | int = (1, 1, 1)) -> Configuration:
    """Body-centred-cubic crystal, 2 atoms per conventional cell.

    Parameters
    ----------
    a:
        Cubic lattice constant in angstrom.
    symbol, reps:
        As in :func:`sc`.

    Returns
    -------
    Configuration
        ``N = 2 nx ny nz`` atoms, volume per atom ``a**3 / 2``, 8 nearest
        neighbours at ``a*sqrt(3)/2`` and 6 second neighbours at ``a``.
    """
    a = _positive(a, "a")
    return _lattice_configuration(
        a * np.eye(3), LATTICE_BASES["bcc"], symbol, reps, {"lattice": "bcc", "a": a}
    )


def fcc(a: float, symbol: str, reps: Sequence[int] | int = (1, 1, 1)) -> Configuration:
    """Face-centred-cubic crystal, 4 atoms per conventional cell.

    Parameters
    ----------
    a:
        Cubic lattice constant in angstrom.
    symbol, reps:
        As in :func:`sc`.

    Returns
    -------
    Configuration
        ``N = 4 nx ny nz`` atoms, volume per atom ``a**3 / 4``, 12 nearest
        neighbours at ``a/sqrt(2)``.
    """
    a = _positive(a, "a")
    return _lattice_configuration(
        a * np.eye(3), LATTICE_BASES["fcc"], symbol, reps, {"lattice": "fcc", "a": a}
    )


def diamond(a: float, symbol: str, reps: Sequence[int] | int = (1, 1, 1)) -> Configuration:
    """Diamond-cubic crystal, 8 atoms per conventional cell.

    Parameters
    ----------
    a:
        Cubic lattice constant in angstrom (5.431 A for silicon).
    symbol, reps:
        As in :func:`sc`.

    Returns
    -------
    Configuration
        ``N = 8 nx ny nz`` atoms, volume per atom ``a**3 / 8``, 4 tetrahedral
        nearest neighbours at the bond length ``a*sqrt(3)/4``.
    """
    a = _positive(a, "a")
    return _lattice_configuration(
        a * np.eye(3), LATTICE_BASES["diamond"], symbol, reps, {"lattice": "diamond", "a": a}
    )


def hcp(
    a: float,
    symbol: str,
    c: float | None = None,
    reps: Sequence[int] | int = (1, 1, 1),
) -> Configuration:
    """Hexagonal-close-packed crystal, 2 atoms in the primitive hexagonal cell.

    The cell is ``a1 = (a, 0, 0)``, ``a2 = (-a/2, a*sqrt(3)/2, 0)``,
    ``a3 = (0, 0, c)`` and the basis is ``(0, 0, 0)`` and ``(1/3, 2/3, 1/2)``.
    The second atom therefore sits over the centroid of a triangle of the first
    sublattice, one half-period up: that is the ABAB stacking.  At the ideal
    ratio ``c/a = sqrt(8/3)`` the three in-plane and the 3+3 out-of-plane
    neighbours coincide in distance, giving 12 neighbours at exactly ``a``;
    away from it the shell splits into 6 at ``a`` and 6 at
    ``sqrt(a^2/3 + c^2/4)``.

    Parameters
    ----------
    a:
        Basal lattice constant in angstrom.
    symbol:
        Chemical symbol; mass from :data:`atomlab.units.ATOMIC_MASSES`.
    c:
        ``c`` axis in angstrom.  ``None`` selects the ideal
        ``sqrt(8/3) * a``.
    reps:
        ``(nx, ny, nz)`` repetitions of the primitive cell.

    Returns
    -------
    Configuration
        ``N = 2 nx ny nz`` atoms, volume per atom
        ``sqrt(3) * a**2 * c / 4``.

    Notes
    -----
    The positional form ``hcp(a, c, symbol, reps)`` is also accepted, since the
    lattice needs two constants and both orders read naturally.
    """
    if not isinstance(symbol, str):
        # Called as hcp(a, c, symbol, reps): shuffle into the canonical order.
        if not isinstance(c, str):
            raise TypeError("hcp() needs a chemical symbol (str) for `symbol`")
        symbol, c = c, float(symbol)

    a = _positive(a, "a")
    c_axis = _positive(a * IDEAL_C_OVER_A if c is None else c, "c")
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c_axis],
        ]
    )
    basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    return _lattice_configuration(
        cell, basis, symbol, reps, {"lattice": "hcp", "a": a, "c": c_axis}
    )


# --------------------------------------------------------------------------
# disordered structures
# --------------------------------------------------------------------------


def random_gas(
    n_atoms: int,
    density: float,
    symbol: str,
    *,
    seed,
    min_distance: float = 0.0,
    max_attempts: int = 10_000,
) -> Configuration:
    """Random positions in a cubic periodic box at a fixed number density.

    Placements closer than ``min_distance`` to an already-placed atom (under the
    minimum image convention) are rejected and re-drawn.  Without that
    rejection a uniform sample puts pairs at arbitrarily small separation, and
    for any steeply repulsive potential the configuration then has an energy so
    large that it is useless both as an MD start and as a training point.

    Parameters
    ----------
    n_atoms:
        Number of atoms to place.
    density:
        Number density in atoms / A^3.  The cubic box edge is
        ``(n_atoms / density) ** (1/3)``.
    symbol:
        Chemical symbol; mass from :data:`atomlab.units.ATOMIC_MASSES`.
    seed:
        Int seed or :class:`numpy.random.Generator`.  Required.
    min_distance:
        Minimum allowed pair separation in angstrom.  ``0`` disables rejection.
    max_attempts:
        Draws allowed per atom before giving up.  Exceeding it raises rather
        than returning a configuration that quietly violates ``min_distance``.

    Returns
    -------
    Configuration
        ``n_atoms`` atoms in a cubic cell with full periodicity.
    """
    if int(n_atoms) < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
    n_atoms = int(n_atoms)
    density = _positive(density, "density")
    min_distance = float(min_distance)
    if min_distance < 0.0:
        raise ValueError(f"min_distance must be >= 0, got {min_distance}")
    rng = _rng(seed)
    mass = _mass_of(symbol)

    box = (n_atoms / density) ** (1.0 / 3.0)
    cell = box * np.eye(3)
    # Beyond L/2 the "minimum" image is ambiguous, so the constraint would not
    # mean what the caller thinks it means.
    if min_distance >= 0.5 * box:
        raise ValueError(
            f"min_distance {min_distance:.3f} A must be < half the box edge "
            f"({0.5 * box:.3f} A) for the minimum image convention to apply"
        )

    positions = np.empty((n_atoms, 3))
    n_placed = 0
    while n_placed < n_atoms:
        for attempt in range(max_attempts):
            trial = rng.random(3) * box
            if min_distance == 0.0 or n_placed == 0:
                break
            d = trial[None, :] - positions[:n_placed]
            d -= box * np.round(d / box)  # cubic cell: minimum image is exact here
            if np.min(np.einsum("ij,ij->i", d, d)) >= min_distance * min_distance:
                break
        else:
            raise RuntimeError(
                f"could not place atom {n_placed + 1}/{n_atoms} with "
                f"min_distance={min_distance} A at density {density} A^-3 in "
                f"{max_attempts} attempts; the box is too crowded for "
                f"rejection sampling -- lower the density or min_distance"
            )
        positions[n_placed] = trial
        n_placed += 1

    return Configuration(
        positions=positions,
        cell=cell,
        pbc=True,
        species=np.zeros(n_atoms, dtype=np.int32),
        symbols=(symbol,),
        masses=np.full(n_atoms, mass),
        info={"builder": "random_gas", "density": density, "min_distance": min_distance},
    )


def rattle(configuration: Configuration, sigma: float, *, seed) -> Configuration:
    """Displace every atom by an isotropic Gaussian of width ``sigma``.

    Each cartesian component is drawn independently from ``N(0, sigma^2)``, so
    the mean squared displacement per atom is ``3 sigma^2`` and the radial
    displacement follows a Maxwell (chi with 3 dof) distribution with mean
    ``2 sqrt(2/pi) sigma``.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    sigma:
        Standard deviation per cartesian component, in angstrom.
    seed:
        Int seed or :class:`numpy.random.Generator`.  Required.

    Returns
    -------
    Configuration
        Copy with displaced positions.  Reference labels are dropped: an
        energy/force label of the undisplaced geometry would be wrong for this
        one, and a silently stale label is the worst kind of dataset bug.
    """
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng = _rng(seed)
    out = configuration.stripped()
    out.positions = out.positions + rng.normal(0.0, sigma, size=out.positions.shape)
    out.info = dict(out.info)
    out.info["rattle_sigma"] = sigma
    return out


# --------------------------------------------------------------------------
# affine rescaling
# --------------------------------------------------------------------------


def scale_cell(configuration: Configuration, factor) -> Configuration:
    """Affinely rescale cell and positions by ``factor``.

    Fractional coordinates are exactly preserved, so a perfect crystal stays a
    perfect crystal.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    factor:
        Scalar linear scale factor, or a length-3 array of per-axis factors
        applied to lattice vectors ``a1, a2, a3`` (and correspondingly to the
        fractional coordinates).

    Returns
    -------
    Configuration
        Copy with ``cell -> diag(factor) @ cell`` and positions rescaled.
        Labels are dropped (they belong to the old geometry).
    """
    f = np.atleast_1d(np.asarray(factor, dtype=float))
    if f.size == 1:
        f = np.repeat(f, 3)
    if f.shape != (3,):
        raise ValueError(f"factor must be scalar or length-3, got {np.shape(factor)}")
    if np.any(f <= 0.0):
        raise ValueError(f"scale factors must be > 0, got {f}")
    if not configuration.pbc.any():
        raise ValueError("cannot rescale the cell of a fully non-periodic configuration")

    s = configuration.scaled_positions()
    out = configuration.stripped()
    out.cell = configuration.cell * f[:, None]
    out.positions = s @ out.cell
    return out


def scale_to_density(configuration: Configuration, density: float) -> Configuration:
    """Isotropically rescale the cell to a target number density.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    density:
        Target density in atoms / A^3.

    Returns
    -------
    Configuration
        Copy whose ``density`` equals ``density`` to round-off.
    """
    density = _positive(density, "density")
    target_volume = configuration.n_atoms / density
    factor = (target_volume / configuration.volume) ** (1.0 / 3.0)
    out = scale_cell(configuration, factor)
    out.info = dict(out.info)
    out.info["density"] = density
    return out


# --------------------------------------------------------------------------
# point defects
# --------------------------------------------------------------------------


def _species_index(configuration: Configuration, symbol: str) -> tuple[tuple[str, ...], int]:
    """Index of ``symbol`` in the configuration's symbol table, extending it."""
    symbols = tuple(configuration.symbols)
    if symbol in symbols:
        return symbols, symbols.index(symbol)
    _mass_of(symbol)  # fail early on an unknown element
    return symbols + (symbol,), len(symbols)


def vacancy(
    configuration: Configuration, index: int | None = None, *, seed=None
) -> Configuration:
    """Remove one atom.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    index:
        Index of the atom to remove.  If ``None`` an atom is chosen uniformly
        at random, which then requires ``seed``.
    seed:
        Int seed or :class:`numpy.random.Generator`; only used (and only
        required) when ``index`` is ``None``.

    Returns
    -------
    Configuration
        Copy with ``N - 1`` atoms.  The removed index is recorded in
        ``info["vacancy_index"]``.
    """
    n = configuration.n_atoms
    if n < 2:
        raise ValueError("cannot remove an atom from a configuration with < 2 atoms")
    if index is None:
        index = int(_rng(seed).integers(n))
    index = int(index)
    if not -n <= index < n:
        raise IndexError(f"vacancy index {index} out of range for {n} atoms")
    index %= n

    keep = np.ones(n, dtype=bool)
    keep[index] = False
    out = configuration.stripped()
    out.positions = out.positions[keep]
    out.species = out.species[keep]
    out.masses = out.masses[keep]
    out.info = dict(out.info)
    out.info["vacancy_index"] = index
    return out


def interstitial(
    configuration: Configuration, position, symbol: str | None = None
) -> Configuration:
    """Insert an extra atom at a given cartesian position.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    position:
        ``(3,)`` cartesian position in angstrom.  It is used as given (not
        wrapped), so the caller keeps control of which image it names.
    symbol:
        Chemical symbol of the inserted atom.  Defaults to the symbol of
        species 0, i.e. a self-interstitial.  A new symbol extends the
        configuration's symbol table.

    Returns
    -------
    Configuration
        Copy with ``N + 1`` atoms; the new atom is last.
    """
    pos = np.asarray(position, dtype=float).reshape(3)
    out = configuration.stripped()

    if symbol is None:
        if not configuration.symbols:
            raise ValueError("configuration has no symbols; pass `symbol` explicitly")
        symbols, sp = configuration.symbols, int(configuration.species[0])
        mass = float(configuration.masses[0])
    else:
        symbols, sp = _species_index(configuration, symbol)
        mass = _mass_of(symbol)

    out.symbols = symbols
    out.positions = np.vstack([out.positions, pos[None, :]])
    out.species = np.concatenate([out.species, np.array([sp], dtype=np.int32)])
    out.masses = np.concatenate([out.masses, np.array([mass])])
    out.info = dict(out.info)
    out.info["interstitial_index"] = out.positions.shape[0] - 1
    return out


def substitute(configuration: Configuration, index: int, species: str | int) -> Configuration:
    """Change the chemical identity of one atom.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    index:
        Index of the atom to substitute.
    species:
        Either a chemical symbol (added to the symbol table if new) or an
        existing integer type index.

    Returns
    -------
    Configuration
        Copy with the atom's species index and mass updated.
    """
    n = configuration.n_atoms
    index = int(index)
    if not -n <= index < n:
        raise IndexError(f"substitution index {index} out of range for {n} atoms")
    index %= n

    out = configuration.stripped()
    if isinstance(species, str):
        symbols, sp = _species_index(configuration, species)
        out.symbols = symbols
        mass = _mass_of(species)
    else:
        sp = int(species)
        if configuration.symbols and not 0 <= sp < len(configuration.symbols):
            raise ValueError(
                f"species index {sp} is outside the symbol table {configuration.symbols}"
            )
        mass = _mass_of(configuration.symbols[sp]) if configuration.symbols else float(
            configuration.masses[index]
        )
    out.species[index] = sp
    out.masses[index] = mass
    return out


# --------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------


def surface_slab(configuration: Configuration, axis: int, vacuum: float) -> Configuration:
    """Open one direction by inserting a vacuum gap.

    The lattice vector ``axis`` is lengthened by ``vacuum`` **along its own
    direction** (so any cell tilt is preserved), the slab is then shifted by
    ``vacuum/2`` along that direction so the added gap is split evenly between
    the two faces, and periodicity is switched off along ``axis``.

    Turning off ``pbc`` matters as much as the gap: with ``pbc`` still true a
    potential would keep interacting through the gap whenever the cutoff
    exceeded it, and a large-enough vacuum would then merely hide the error.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.  Atoms are wrapped into the cell
        along ``axis`` first, so that a slab built from a supercell is compact.
    axis:
        0, 1 or 2 -- which lattice vector to open.
    vacuum:
        Gap thickness in angstrom, measured along the lattice vector.

    Returns
    -------
    Configuration
        Copy with the enlarged cell and ``pbc[axis] = False``.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    vacuum = float(vacuum)
    if vacuum < 0.0:
        raise ValueError(f"vacuum must be >= 0, got {vacuum}")
    if not configuration.pbc[axis]:
        raise ValueError(f"axis {axis} is already non-periodic")

    # Wrap only along `axis`, so the slab is a compact block before we open it.
    s = configuration.scaled_positions()
    s[:, axis] = np.mod(s[:, axis] + 1e-12, 1.0)
    positions = s @ configuration.cell

    vec = configuration.cell[axis]
    length = float(np.linalg.norm(vec))
    unit = vec / length

    out = configuration.stripped()
    out.cell = configuration.cell.copy()
    out.cell[axis] = unit * (length + vacuum)
    out.positions = positions + 0.5 * vacuum * unit
    out.pbc = configuration.pbc.copy()
    out.pbc[axis] = False
    out.info = dict(out.info)
    out.info["vacuum"] = vacuum
    out.info["surface_axis"] = axis
    return out


# --------------------------------------------------------------------------
# strained cells (elastic constants)
# --------------------------------------------------------------------------

# Voigt index -> (row, col) pairs of the symmetric strain tensor.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_VOIGT_NAMES = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "zy": 3, "xz": 4, "zx": 4, "xy": 5, "yx": 5}


def voigt_to_strain(voigt) -> np.ndarray:
    """Convert a 6-vector of Voigt strains to the ``(3, 3)`` strain tensor.

    The **engineering** convention is used for the shear components, i.e.
    ``e4 = 2*eps_yz``, ``e5 = 2*eps_xz``, ``e6 = 2*eps_xy``, so that the
    elastic energy density is ``0.5 * e^T C e`` with the usual ``C44``.  The
    resulting tensor is symmetric, hence rotation-free: a pure shear applied
    antisymmetrically would rotate the cell and contaminate the stress with a
    spurious antisymmetric part.

    Parameters
    ----------
    voigt:
        ``(6,)`` array ``[e1, e2, e3, e4, e5, e6]`` (dimensionless).

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` symmetric strain tensor.
    """
    v = np.asarray(voigt, dtype=float).reshape(6)
    eps = np.zeros((3, 3))
    eps[0, 0], eps[1, 1], eps[2, 2] = v[0], v[1], v[2]
    eps[1, 2] = eps[2, 1] = 0.5 * v[3]
    eps[0, 2] = eps[2, 0] = 0.5 * v[4]
    eps[0, 1] = eps[1, 0] = 0.5 * v[5]
    return eps


def strain_to_voigt(strain) -> np.ndarray:
    """Inverse of :func:`voigt_to_strain` (symmetric part only).

    Parameters
    ----------
    strain:
        ``(3, 3)`` strain tensor.

    Returns
    -------
    numpy.ndarray
        ``(6,)`` Voigt strains, engineering shear convention.
    """
    e = np.asarray(strain, dtype=float).reshape(3, 3)
    sym = 0.5 * (e + e.T)
    return np.array(
        [sym[0, 0], sym[1, 1], sym[2, 2], 2 * sym[1, 2], 2 * sym[0, 2], 2 * sym[0, 1]]
    )


def apply_strain(configuration: Configuration, strain) -> Configuration:
    """Apply ``r -> (1 + eps) r`` to atoms and lattice vectors.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    strain:
        Either a ``(3, 3)`` tensor or a ``(6,)`` Voigt vector (engineering
        shear convention, see :func:`voigt_to_strain`).

    Returns
    -------
    Configuration
        Strained copy, labels dropped.
    """
    arr = np.asarray(strain, dtype=float)
    eps = voigt_to_strain(arr) if arr.shape == (6,) else arr.reshape(3, 3)
    return configuration.strained(eps)


def uniaxial_strain(configuration: Configuration, epsilon: float, axis: int = 0) -> Configuration:
    """Strain ``epsilon`` along one cartesian axis, the others held fixed.

    This is the deformation whose stress response gives ``C11`` (the stress
    component along ``axis``) and ``C12`` (the transverse components).

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    epsilon:
        Dimensionless engineering strain.
    axis:
        0, 1 or 2 (cartesian x, y, z).

    Returns
    -------
    Configuration
        Strained copy.
    """
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    eps = np.zeros((3, 3))
    eps[axis, axis] = float(epsilon)
    return configuration.strained(eps)


def hydrostatic_strain(configuration: Configuration, epsilon: float) -> Configuration:
    """Isotropic strain ``eps * I``; volume changes by ``(1+eps)^3``.

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    epsilon:
        Dimensionless linear strain.

    Returns
    -------
    Configuration
        Strained copy.  Equivalent to :func:`scale_cell` with factor
        ``1 + epsilon``.
    """
    return configuration.strained(float(epsilon) * np.eye(3))


def shear(configuration: Configuration, gamma: float, plane: str | Sequence[int]) -> Configuration:
    """Apply an engineering shear strain ``gamma`` in a cartesian plane.

    The strain tensor is symmetric with ``eps_ab = eps_ba = gamma/2``, so the
    engineering shear strain is ``gamma`` and the deformation carries no
    rigid rotation.  The stress response ``sigma_ab / gamma`` is then ``C44``
    directly (for a cubic crystal in its standard orientation).

    Parameters
    ----------
    configuration:
        Input configuration; not modified.
    gamma:
        Engineering shear strain (dimensionless).
    plane:
        ``"xy"``, ``"yz"``, ``"xz"`` (order-insensitive) or a pair of distinct
        cartesian axis indices.

    Returns
    -------
    Configuration
        Sheared copy.  To second order in ``gamma`` the volume changes by a
        factor ``1 - gamma^2/4``; the deformation is volume-preserving only to
        first order, which is exactly the accuracy at which linear elasticity
        is defined.
    """
    if isinstance(plane, str):
        key = plane.lower().replace("-", "").replace(",", "")
        if key not in _VOIGT_NAMES or _VOIGT_NAMES[key] < 3:
            raise ValueError(f"plane must name a shear plane (xy, yz, xz), got {plane!r}")
        a, b = _VOIGT_PAIRS[_VOIGT_NAMES[key]]
    else:
        a, b = (int(x) for x in plane)
        if a == b or not {a, b} <= {0, 1, 2}:
            raise ValueError(f"plane must be two distinct axes in 0..2, got {plane!r}")

    eps = np.zeros((3, 3))
    eps[a, b] = eps[b, a] = 0.5 * float(gamma)
    return configuration.strained(eps)


def strain_scan(
    configuration: Configuration, component: int | str, magnitudes: Sequence[float]
) -> list[Configuration]:
    """Configurations strained along one Voigt component, for stress fitting.

    Parameters
    ----------
    configuration:
        Reference (unstrained) configuration.
    component:
        Voigt index ``0..5`` or a name such as ``"xx"``/``"yz"``.
    magnitudes:
        Sequence of strain magnitudes (dimensionless).  For elastic constants
        use a symmetric set such as ``[-0.01, -0.005, 0.005, 0.01]``: the odd
        part of the stress response isolates the linear coefficient and cancels
        the leading anharmonic contribution.

    Returns
    -------
    list of Configuration
        One configuration per magnitude, in the given order.  Each carries
        ``info["strain_voigt"]`` for provenance.
    """
    if isinstance(component, str):
        key = component.lower()
        if key not in _VOIGT_NAMES:
            raise ValueError(f"unknown strain component {component!r}")
        idx = _VOIGT_NAMES[key]
    else:
        idx = int(component)
        if not 0 <= idx < 6:
            raise ValueError(f"Voigt component must be in 0..5, got {idx}")

    out = []
    for m in magnitudes:
        v = np.zeros(6)
        v[idx] = float(m)
        cfg = configuration.strained(voigt_to_strain(v))
        cfg.info = dict(cfg.info)
        cfg.info["strain_voigt"] = v
        out.append(cfg)
    return out


def volume_scan(
    configuration: Configuration, volume_ratios: Sequence[float]
) -> list[Configuration]:
    """Isotropically scaled copies at prescribed volume ratios ``V/V0``.

    This is the input to the Birch-Murnaghan equation-of-state fit in
    ``atomlab/observables/thermo.py``.

    Parameters
    ----------
    configuration:
        Reference configuration of volume ``V0``.
    volume_ratios:
        Sequence of ``V/V0`` values, all strictly positive.

    Returns
    -------
    list of Configuration
        Copies with cell scaled by ``(V/V0)**(1/3)``; each carries
        ``info["volume_ratio"]``.
    """
    out = []
    for ratio in volume_ratios:
        r = _positive(ratio, "volume ratio")
        cfg = scale_cell(configuration, r ** (1.0 / 3.0))
        cfg.info = dict(cfg.info)
        cfg.info["volume_ratio"] = r
        out.append(cfg)
    return out
