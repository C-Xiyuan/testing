"""Vibrational density of states from the velocity autocorrelation function.

The classical VDOS is the power spectrum of the mass-weighted velocity
autocorrelation,

``g(nu) = A int_{-inf}^{inf} sum_i m_i <v_i(0).v_i(t)> e^{-2 pi i nu t} dt``

with ``A`` fixed by the normalisation convention (below).  Mass weighting is not
cosmetic: it is what makes each mode contribute spectral weight proportional to
the number of degrees of freedom it carries rather than to its amplitude, so
that the integral counts modes.  For a single-species system it drops out.

Why this and not the phonon route?  :mod:`atomlab.observables.phonons` computes
the harmonic spectrum from force constants at a stationary point.  This module
computes the *anharmonic, finite-temperature* spectrum from real dynamics, so
the two disagree exactly to the extent that the system is anharmonic -- and both
are needed, because a surrogate potential can reproduce one and not the other.

Windowing: the tradeoff, stated plainly
---------------------------------------
The correlation function is known only up to a maximum lag ``L``.  Truncating it
there is itself a windowing operation -- multiplication by a rectangle -- and the
Fourier transform of a rectangle is a sinc, whose first sidelobe is only 13 dB
down and whose sidelobes decay as ``1/f``.  A sharp vibrational mode therefore
smears ringing across the whole spectrum, and because the sinc goes negative,
the computed VDOS goes negative too, which is unphysical.  For a solid with a
narrow optical peak sitting on a broad acoustic band, this leakage can easily
dominate the band.

A tapered window (Hann, Hamming, Blackman) suppresses the sidelobes -- Hann to
-31 dB with a ``1/f^3`` decay -- at the cost of a main lobe roughly twice as
wide.  So the choice is resolution against leakage, and there is no setting that
wins both:

=============  =================  ==============
window         main lobe FWHM     first sidelobe
=============  =================  ==============
rectangular    1.21 bins          -13 dB
hamming        1.82 bins          -43 dB
hann           2.00 bins          -31 dB
blackman       2.30 bins          -58 dB
=============  =================  ==============

(One "bin" is ``1 / (M dt)`` with ``M = 2L+1`` the length of the symmetric
correlation function; the FWHM quoted is that of the transform's *amplitude*,
which is what a VDOS peak inherits, i.e. the 6 dB power bandwidth.)

Hann is the default: it removes the negative excursions that make a spectrum
hard to interpret while costing less resolution than Blackman.  The measured
main-lobe width of whatever window was used is returned in
``VDOSResult.window_fwhm_thz``, so a peak width can always be compared against
the instrumental resolution instead of being over-interpreted.

Zero padding does **not** add resolution -- it interpolates the same
band-limited spectrum onto a finer grid.  It is still worth doing (4x by
default), because with a coarse grid a narrow peak can fall between samples and
be reported at the wrong position with the wrong height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..units import CLIGHT
from .base import ObservableResult
from .dynamics import VACFResult, _as_observable_result

__all__ = [
    "VDOSResult",
    "vibrational_dos",
    "spectral_window",
    "window_bandwidth",
    "dominant_peak",
    "THZ_TO_CM1_CORRECT",
]

#: Converts THz to wavenumbers in cm^-1.
#:
#: Derived from :data:`atomlab.units.CLIGHT` (2.99792458e6 A/ps) rather than
#: taken from ``units.THZ_TO_CM1``: the speed of light in cm/s is
#: ``CLIGHT * 1e-8 cm/A * 1e12 ps/s = CLIGHT * 1e4``, so 1 THz corresponds to
#: ``1e12 / (CLIGHT * 1e4) = 1e8 / CLIGHT = 33.356 cm^-1``.  ``units.THZ_TO_CM1``
#: currently evaluates to 3335.6, a factor of 100 too large; see the note in the
#: module tests.  Only the definitional cm/A and ps/s factors are written out
#: here -- the physical constant still comes from :mod:`atomlab.units`.
THZ_TO_CM1_CORRECT = 1.0e8 / CLIGHT


_WINDOWS = {
    "rectangular": np.ones,
    "none": np.ones,
    "bartlett": np.bartlett,
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
}


def spectral_window(name: str, n_samples: int) -> np.ndarray:
    """Symmetric window of ``n_samples`` points, 1 at the centre, ~0 at the ends.

    Parameters
    ----------
    name : {"rectangular", "none", "bartlett", "hann", "hamming", "blackman"}
    n_samples : int
        Length of the **symmetric** correlation function, i.e. ``2L+1`` for a
        one-sided VACF of maximum lag ``L``.

    Returns
    -------
    ndarray, shape (n_samples,)
        Dimensionless window weights.
    """
    try:
        maker = _WINDOWS[name]
    except KeyError:
        raise ValueError(
            f"unknown window {name!r}; expected one of {sorted(_WINDOWS)}"
        ) from None
    return np.asarray(maker(int(n_samples)), dtype=float)


def _cosine_transform(half: np.ndarray, n_fft: int) -> np.ndarray:
    """Zero-phase real transform of an even function given by its half support.

    ``half[m]`` is the value at lag ``m`` (``m = 0..L``) of a function that is
    even in the lag.  The DFT input is laid out with lag ``m`` at index ``m`` and
    lag ``-m`` at index ``n_fft - m`` so that the padding lands *between* the
    largest positive and the most negative lag, where the function is (after
    windowing) zero.  Padding at the end of a naively ordered array would instead
    insert zeros between lag -1 and lag 0 and produce a spectrum that is neither
    real nor correct.
    """
    lag_max = half.size - 1
    buf = np.zeros(int(n_fft))
    buf[: lag_max + 1] = half
    buf[n_fft - lag_max :] = half[:0:-1]
    return np.fft.rfft(buf).real


def window_bandwidth(name: str, n_samples: int, dt: float, *, zero_padding: int = 16) -> float:
    """Full width at half maximum of the window's own spectral peak, in THz.

    This is the instrumental resolution of the VDOS: a perfectly monochromatic
    oscillator produces exactly this peak, shifted to its frequency.  It is
    measured by transforming the window itself rather than taken from a table,
    so it stays correct if the window set is extended.

    Parameters
    ----------
    name : str
        Window name, see :func:`spectral_window`.
    n_samples : int
        Symmetric window length ``2L+1``.
    dt : float
        Frame spacing in ps (so that the answer is in THz).
    zero_padding : int
        Interpolation factor used for the width measurement only; the half-power
        points generally fall between DFT samples, and a coarse grid would
        quantise the answer.

    Returns
    -------
    float
        FWHM in THz.
    """
    n_samples = int(n_samples)
    lag_max = (n_samples - 1) // 2
    w = spectral_window(name, 2 * lag_max + 1)[lag_max:]
    n_fft = 1 << int(zero_padding * (2 * lag_max + 1) - 1).bit_length()
    spec = _cosine_transform(w, n_fft)
    freq = np.fft.rfftfreq(n_fft, d=dt)
    # The window's own peak sits at zero frequency, so the one-sided spectrum
    # holds only its right half; the full width is twice the half width.  (A
    # peak at a finite frequency, which is what a real mode gives, is handled by
    # _fwhm directly.)
    half = 0.5 * spec[0]
    k = int(np.argmax(spec < half))
    if k == 0:
        return float("nan")
    crossing = freq[k - 1] + (half - spec[k - 1]) * (freq[k] - freq[k - 1]) / (spec[k] - spec[k - 1])
    return 2.0 * float(crossing)


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """FWHM of the dominant peak of ``y(x)``, by linear interpolation."""
    peak = int(np.argmax(y))
    half = 0.5 * y[peak]
    left = peak
    while left > 0 and y[left] > half:
        left -= 1
    right = peak
    while right < y.size - 1 and y[right] > half:
        right += 1
    if y[left] > half or y[right] > half:
        return float("nan")  # peak not resolved within the range

    def cross(i: int, j: int) -> float:
        if y[j] == y[i]:
            return float(x[i])
        return float(x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i]))

    return cross(right, right - 1) - cross(left, left + 1)


@dataclass
class VDOSResult:
    """Vibrational density of states.

    Attributes
    ----------
    frequency_thz : ndarray, shape (F,)
        Frequency grid in THz (i.e. 1/ps), from 0 to the Nyquist frequency
        ``1/(2 dt)``.
    frequency_cm1 : ndarray, shape (F,)
        The same grid in wavenumbers, cm^-1.
    dos : ndarray, shape (F,)
        Density of states in states/THz, normalised so that
        ``int dos d(nu) = n_dof`` (or 1, see ``normalization``).
    normalization : {"n_dof", "unit"}
        Which convention ``dos`` obeys.
    n_dof : int
        Number of vibrational degrees of freedom the spectrum accounts for,
        ``3N`` by default.
    window : str
        Window applied to the correlation function.
    zero_padding : int
        Interpolation factor used for the transform.
    resolution_thz : float
        Frequency spacing of the returned grid, in THz.  Note this is *not* the
        resolution -- zero padding makes it finer than the true resolution,
        which is ``window_fwhm_thz``.
    window_fwhm_thz : float
        Instrumental FWHM: the width of the peak a single sharp mode produces.
    negative_fraction : float
        Fraction of the spectral weight sitting in negative excursions,
        ``sum(|dos| where dos<0) / sum(|dos|)``.  Exactly zero is unattainable
        with a truncated correlation function; a value above a per cent means
        the window is not controlling leakage and the spectrum should not be
        interpreted quantitatively.
    meta : dict
    """

    frequency_thz: np.ndarray
    frequency_cm1: np.ndarray
    dos: np.ndarray
    normalization: str
    n_dof: int
    window: str
    zero_padding: int
    resolution_thz: float
    window_fwhm_thz: float
    negative_fraction: float
    meta: dict = field(default_factory=dict)

    @property
    def integral(self) -> float:
        """``int dos d(nu)`` -- ``n_dof`` or 1 by construction; a self-check."""
        return float(np.trapezoid(self.dos, self.frequency_thz))

    def mean_frequency_thz(self) -> float:
        """First moment ``int nu g(nu) dnu / int g(nu) dnu`` in THz."""
        weight = np.trapezoid(self.dos, self.frequency_thz)
        return float(np.trapezoid(self.dos * self.frequency_thz, self.frequency_thz) / weight)

    def to_observable_result(self) -> ObservableResult:
        return _as_observable_result(
            "vibrational_dos",
            self.dos,
            np.full_like(self.dos, np.nan),
            "states/THz",
            {
                "frequency_thz": self.frequency_thz,
                "frequency_cm1": self.frequency_cm1,
                "window": self.window,
                "window_fwhm_thz": self.window_fwhm_thz,
            },
        )


def vibrational_dos(
    vacf_result: VACFResult,
    *,
    window: str = "hann",
    zero_padding: int = 4,
    normalization: Literal["n_dof", "unit"] = "n_dof",
    mass_weighted: bool = True,
    n_dof: int | None = None,
) -> VDOSResult:
    """Vibrational density of states from a velocity autocorrelation function.

    Parameters
    ----------
    vacf_result : VACFResult
        From :func:`atomlab.observables.dynamics.velocity_autocorrelation`.
        Its per-atom curves carry the masses needed for the weighting.
    window : str
        Taper applied to the correlation function before transforming; see the
        module docstring for the resolution/leakage tradeoff and
        :func:`spectral_window` for the available names.
    zero_padding : int
        Transform length as a multiple of the symmetric correlation length
        (rounded up to a power of two).  Interpolation only, not resolution.
    normalization : {"n_dof", "unit"}
        ``"n_dof"`` scales the spectrum so that ``int g dnu = n_dof = 3N``, the
        convention in which ``g`` counts modes.  ``"unit"`` normalises the
        integral to 1, which is convenient when comparing the *shape* of two
        spectra from systems of different size.
    mass_weighted : bool
        Use ``sum_i m_i <v_i(0).v_i(t)>`` rather than the plain atom average.
        Only matters for multi-species systems, where the unweighted spectrum
        over-counts light atoms.
    n_dof : int, optional
        Override the number of degrees of freedom.  Defaults to ``3N``.  (The
        centre-of-mass constraint removes 3, but the standard VDOS convention
        counts ``3N``; the difference is ``1/N`` and the choice is explicit here
        rather than hidden.)

    Returns
    -------
    VDOSResult
        Spectrum on a grid running from 0 to the Nyquist frequency
        ``1 / (2 dt)`` THz.  Anything in the system that vibrates faster than
        Nyquist is aliased back into the band -- that is a property of the frame
        stride the trajectory was written with, not of this routine, and
        ``meta["nyquist_thz"]`` is reported so it can be checked.

    Raises
    ------
    ValueError
        For an unknown window, a non-positive ``zero_padding``, or a VACF that
        is too short to transform.
    """
    if int(zero_padding) < 1:
        raise ValueError(f"zero_padding must be >= 1, got {zero_padding}")

    correlation = vacf_result.mass_weighted() if mass_weighted else vacf_result.vacf
    lag_max = correlation.size - 1
    if lag_max < 2:
        raise ValueError(f"need at least 3 lag points to transform, got {correlation.size}")

    dt = vacf_result.dt
    n_symmetric = 2 * lag_max + 1
    taper = spectral_window(window, n_symmetric)[lag_max:]  # right half, w(0) = 1
    n_fft = 1 << int(int(zero_padding) * n_symmetric - 1).bit_length()

    spectrum = _cosine_transform(correlation * taper, n_fft)
    freq = np.fft.rfftfreq(n_fft, d=dt)  # 1/ps == THz

    target = 1.0 if normalization == "unit" else float(
        3 * vacf_result.n_atoms if n_dof is None else n_dof
    )
    if normalization not in ("unit", "n_dof"):
        raise ValueError(f"unknown normalization {normalization!r}")
    area = float(np.trapezoid(spectrum, freq))
    if area <= 0.0:
        raise ValueError(
            "the transformed correlation function has non-positive total weight; "
            "the VACF is not a valid autocorrelation function"
        )
    # Normalised numerically rather than analytically: the analytic factor and
    # the discrete integral differ by a discretisation error that would leave
    # the reported integral only approximately equal to 3N.  Forcing it makes
    # `VDOSResult.integral` a genuine check on the caller's grid rather than an
    # approximation with an unstated error.
    dos = spectrum * (target / area)

    negative = float(np.abs(dos[dos < 0]).sum() / np.abs(dos).sum()) if np.any(dos < 0) else 0.0

    return VDOSResult(
        frequency_thz=freq,
        frequency_cm1=freq * THZ_TO_CM1_CORRECT,
        dos=dos,
        normalization=normalization,
        n_dof=int(target) if normalization == "n_dof" else 3 * vacf_result.n_atoms,
        window=window,
        zero_padding=int(zero_padding),
        resolution_thz=float(freq[1] - freq[0]),
        window_fwhm_thz=window_bandwidth(window, n_symmetric, dt),
        negative_fraction=negative,
        meta={
            "n_fft": int(n_fft),
            "lag_max": int(lag_max),
            "nyquist_thz": 0.5 / dt,
            "mass_weighted": bool(mass_weighted),
            "raw_area": area,
            "bin_thz": 1.0 / (n_symmetric * dt),
        },
    )


def dominant_peak(result: VDOSResult) -> tuple[float, float]:
    """Position and FWHM of the largest peak, both in THz.

    Used to compare a measured spectrum against an analytic expectation: for a
    set of independent harmonic oscillators the VDOS is a delta function at the
    oscillator frequency, so the peak position must be the frequency and the
    width must be the window's own bandwidth (:func:`window_bandwidth`) and
    nothing more.
    """
    peak = int(np.argmax(result.dos))
    return float(result.frequency_thz[peak]), _fwhm(result.frequency_thz, result.dos)
