

from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.fft as fft


def double_to_single_sided_fft(
    freqs: npt.NDArray[np.float64],
    fft_values: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """
    Convert a double-sided FFT to a properly scaled single-sided FFT.
    Efficiently handles both single-channel and multichannel FFT data.

    Parameters:
    -----------
    freqs : array of float
        The corresponding frequencies for the FFT result (positive and negative)
    fft_values : array of complex
        The complex FFT result (double-sided)
        Can be 1D array [n_fft_points] for single-channel data
        or 2D array [n_channels, n_fft_points] for multichannel data

    Returns:
    --------
    positive_freqs : array of float
        The corresponding positive frequencies
    single_sided_fft : array of complex
        The single-sided FFT with proper scaling
        Maintains the same dimensionality as input fft_values
    """
    # Store original shape and dimensionality
    original_ndim = fft_values.ndim

    # Get number of FFT points (last dimension)
    n = fft_values.shape[-1]

    # Find indices where frequencies are >= 0
    positive_indices = freqs >= 0
    positive_freqs = freqs[positive_indices]

    # Create scaling factors array
    scale = np.ones(len(positive_freqs), dtype=float)

    # Set scaling for all frequencies except DC (and Nyquist if present)
    if n % 2 == 0:  # Even number of points
        # Scale all but first (DC) and last (Nyquist) elements
        scale[1:-1] = 2.0
    else:  # Odd number of points
        # Scale all but first (DC) element
        scale[1:] = 2.0

    # Handle indexing into the FFT values differently based on dimensionality
    # and apply scaling
    if original_ndim == 1:
        single_sided_fft = fft_values[positive_indices].copy() * scale
    else:
        # For multichannel data, we need to index the last dimension
        single_sided_fft = (
            fft_values[..., positive_indices].copy() * scale[np.newaxis, :]
        )

    return positive_freqs, single_sided_fft


def double_to_single_sided_ifft(
    time_values: npt.NDArray[np.float64],
    ifft_values: npt.NDArray[np.complex128] | npt.NDArray[np.complexfloating],
    shift: bool = False,
) -> tuple[
    npt.NDArray[np.complex128] | npt.NDArray[np.complexfloating], npt.NDArray[np.float64]
]:
    """
    Extract the first half of the IFFT result for display or analysis purposes.
    Works with both single-channel and multi-channel data.

    Parameters:
    -----------
    time_values : array of float
        The time values corresponding to each IFFT sample
    ifft_values : array of complex or float
        The result of an inverse FFT (typically time-domain)
        Can be 1D array [n_time_points] for single-channel data
        or 2D array [n_channels, n_time_points] for multichannel data

    Returns:
    --------
    corresponding_times : array of float
        The time values corresponding to the first half
    first_half : array
        The first half of the IFFT result, with same dimensionality as input
    """
    # Get the total number of points (last dimension)
    n = ifft_values.shape[-1]

    if shift:
        ifft_values = fft.ifftshift(ifft_values, axes=-1)

    # Calculate the midpoint index
    # For even length, midpoint is n/2
    # For odd length, midpoint is (n+1)/2
    midpoint = (n + 1) // 2

    # Extract the first half of the data based on dimensionality
    if ifft_values.ndim == 1:
        first_half = ifft_values[:midpoint].copy()
    else:
        # For multi-channel data, keep all channels but take first half of time points
        first_half = ifft_values[..., :midpoint].copy()

    # Get corresponding time values
    corresponding_times = time_values[:midpoint]

    return corresponding_times, first_half


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
    f = fft.rfftfreq(n, d=1 / fs)
    inds = np.argwhere((f >= freq_band[0]) & (f <= freq_band[1])).squeeze()
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
