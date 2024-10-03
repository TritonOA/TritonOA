# -*- coding: utf-8 -*-

from typing import Sequence

import numpy as np
from scipy.fft import rfftfreq


def get_freqs_from_band(
    n: int, freq_band: Sequence[float], fs: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get frequencies within a given band.

    Parameters
    ----------
    n : int
        Number of samples.
    freq_band : Sequence[float]
        Frequency band.
    fs : float, optional
        Sampling rate (1/s), by default 1.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies within the band and their corresponding indices.
    """
    f = rfftfreq(n, d=1 / fs)
    inds = np.argwhere((f > freq_band[0]) & (f < freq_band[1])).squeeze()
    return f[inds], inds


def normalize_pressure(p: float | np.ndarray, log: bool = False) -> np.ndarray:
    """Takes complex pressure and returns normalized pressure.

    Parameters
    ----------
    p : array
        Array of complex pressure values.
    log : bool, default=False
        Scale normalized pressure logarithmically (e.g., for dB scale)

    Returns
    -------
    pn : array
        Normalized pressure.
    """

    pn = np.abs(p)
    pn /= np.max(pn)
    if log:
        pn = 10 * np.log10(pn)
    return pn
