# -*- coding: utf-8 -*-

from typing import Optional, Union

import numpy as np
from scipy.fft import irfft, rfft


def group_speed_from_kr(freq: float, kr: float, c: float) -> float:
    omega = 2 * np.pi * freq
    return (c**2) * kr / omega


def propagate_signal(
    signal: np.ndarray,
    greens_function: np.ndarray,
    freq: np.ndarray,
    t_offset: float = 0.0,
    nfft: Optional[int] = None,
) -> np.ndarray:
    """Propagate a signal through a medium.

    `freq` and `freq_inds` are the frequencies of interest and their corresponding
    indices, respectively. The signal is propagated by multiplying the signal's
    Fourier transform by the Green's function and the phase offset. Typically,
    not all frequencies are of interest, so only frequencies within a band
    are considered.

    Parameters
    ----------
    signal : np.ndarray
        Signal to propagate.
    greens_function : np.ndarray
        Green's function.
    freq : np.ndarray
        Array of frequencies of interest.
    freq_inds : np.ndarray
        Indices of the frequencies of interest.
    t_offset : float, optional
        Time offset, by default 0.0.
    nfft : Optional[int], optional
        Number of points for the FFT, by default None.

    Returns
    -------
    np.ndarray
        Propagated signal.
    """
    X = rfft(signal, axis=-1)
    phase_offset = np.exp(1j * 2 * np.pi * freq * t_offset)
    Y = X * greens_function * phase_offset
    return irfft(Y, n=nfft, axis=-1)


def range_ind_pressure_from_modes(
    phi_src: np.ndarray,
    phi_rec: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    r_offsets: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """Calculates pressure field given range-independent vertical mode
    functions.

    Uses the asymptotic approximation to a Hankel function.

    Parameters
    ----------
    phi_src : array
        Complex-valued mode shape function at the source depth.
    phi_rec : array
        Complex-valued mode shape function for specified depths (e.g.,
        at receiver depths).
    k : array
        Complex-valued vertical wavenumber.
    r : array
        A point or vector of ranges.

    Returns
    -------
    p : array
        Complex pressure field with dimension (depth x range).

    Notes
    -----
    This function implements equation 5.13 from [1]. NOTE: This implementation
    is in contrast to the KRAKEN MATLAB implementation, which normalizes the
    output by a factor of (1 / (4 * pi)).

    [1] Finn B. Jensen, William A. Kuperman, Michael B. Porter, and
    Henrik Schmidt. 2011. Computational Ocean Acoustics (2nd. ed.).
    Springer Publishing Company, Incorporated.
    """
    if r_offsets is not None:
        M = phi_rec.shape[0]
        N = len(r)
        p = np.zeros((M, N), dtype=complex)
        for zz in range(M):
            range_dep = np.outer(k, r + r_offsets[zz])
            hankel = np.exp(1j * range_dep.conj()) / np.sqrt(np.real(range_dep))
            p[zz] = (phi_src * phi_rec[zz]).dot(hankel)
    else:
        range_dep = np.outer(k, r)
        hankel = np.exp(1j * range_dep.conj()) / np.sqrt(np.real(range_dep))
        p = (phi_src * phi_rec).dot(hankel)
    p *= -np.exp(1j * np.pi / 4)
    p /= np.sqrt(8 * np.pi)
    p = p.conj()
    return p
