# TODO: Implement cupy support
# try:
# import cupyx.scipy.signal as signal
# except ImportError:
# import scipy.signal as signal

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import warnings

import numpy as np
import numpy.typing as npt
import scipy.signal as sp


class TaperLengthWarning(Warning):
    pass


@dataclass
class SignalParams:
    """Specifications for one or more hydrophones.

    Args:
        adc_vref: ADC reference voltage [V]. Defaults to 1.0 (no effect).
        gain: Gain [dB], converts counts to V. Defaults to 0.0 (no effect).
        sensitivity: Sensitivity [dB], converts V to uPa. Defaults to 0.0 (no effect).
    """

    adc_vref: list[float] = field(default_factory=lambda: [1.0])
    gain: list[float] = field(default_factory=lambda: [0.0])
    sensitivity: list[float] = field(default_factory=lambda: [0.0])

    def check_dimensions(self, num_channels: int):
        """Checks if the dimensions of the parameters match the channels.

        Args:
            channels (list): List of channels.
        """
        self.fill_like_channels(num_channels)
        if (
            len(self.adc_vref) != num_channels
            or len(self.gain) != num_channels
            or len(self.sensitivity) != num_channels
        ):
            # TODO: Implement specific exception for this.
            raise ValueError(
                "The number of gains and sensitivities must match the number of channels."
            )

    def fill_like_channels(self, num_channels: int):
        """Fills the parameters to match the number of channels.

        Args:
            num_channels (int): Number of channels.
        """
        if not isinstance(self.adc_vref, list):
            self.adc_vref = [self.adc_vref] * num_channels
        if not isinstance(self.gain, list):
            self.gain = [self.gain] * num_channels
        if not isinstance(self.sensitivity, list):
            self.sensitivity = [self.sensitivity] * num_channels


def db_to_linear(dbgain: float | list[float]) -> float | list[float]:
    """Converts a gain in dB to a linear gain factor.

    This function is adapted from the ObsPy library:
    https://docs.obspy.org/index.html

    Args:
        dbgain (float): Gain in dB.

    Returns:
        float: Linear gain factor.

    Examples:
    >>> dbgain_to_lineargain(6)
    2.0
    >>> dbgain_to_lineargain(20)
    10.0
    """
    if isinstance(dbgain, list):
        return [10.0 ** (gain / 20.0) for gain in dbgain]
    return 10.0 ** (dbgain / 20.0)


def get_filter(filt_type: str) -> Callable:
    FILTER_REGISTRY = {
        "bandpass": bandpass,
        "bandstop": bandstop,
        "highpass": highpass,
        "lowpass": lowpass,
    }
    return FILTER_REGISTRY[filt_type]


def bandpass(
    data: np.ndarray,
    freq: Iterable[float],
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    freqmin, freqmax = freq
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    if high - 1.0 > -1e-6:
        warnings.warn(
            (
                f"Selected high corner frequency ({freqmax}) of bandpass is at or "
                f"above Nyquist ({fe}). Applying a high-pass instead."
            )
        )
        return highpass(data, freq=freqmin, fs=fs, corners=corners, zerophase=zerophase)
    if low > 1:
        raise ValueError(
            f"Selected low corner frequency ({freqmin}) of bandpass is above Nyquist ({fe})."
        )

    z, p, k = sp.iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    sos = sp.zpk2sos(z, p, k)
    if zerophase:
        firstpass = sp.sosfilt(sos, data, axis=1)
        return sp.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return sp.sosfilt(sos, data, axis=1)


def bandstop(
    data: np.ndarray,
    freq: Iterable[float],
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    freqmin, freqmax = freq
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    if high - 1.0 >= -1e-6:
        high = 1.0 - 1e-6
        warnings.warn(
            (
                f"Selected high corner frequency ({freqmax}) is above "
                f"Nyquist ({fe}). Setting Nyquist as high corner."
            )
        )
    if low > 1.0:
        raise ValueError(
            f"Selected low corner frequency ({freqmin}) is above Nyquist ({fe})."
        )
    z, p, k = sp.iirfilter(
        corners, [low, high], btype="bandstop", ftype="butter", output="zpk"
    )
    sos = sp.zpk2sos(z, p, k)
    if zerophase:
        firstpass = sp.sosfilt(sos, data, axis=1)
        return sp.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return sp.sosfilt(sos, data, axis=1)


def convert_voltage_to_pressure(
    voltage: np.ndarray, linear_sensitivity: list[float]
) -> tuple[np.ndarray, str]:
    """Convert voltage data to pressure.

    Args:
        voltage (np.ndarray): Voltage data.

    Returns:
        tuple[np.ndarray, str]: Converted data and units.
    """
    return voltage / np.array(linear_sensitivity)[:, np.newaxis], "uPa"


def convert_counts_to_voltage(
    data: np.ndarray,
    linear_fixed_gain: list[float],
    adc_vref: list[float],
    adc_maxvalue: int,
) -> tuple[np.ndarray, str]:
    """Convert 24-bit data to voltage.

    Args:
        data (np.ndarray): 24-bit data.
        fixed_linear_gain (list[float]): Fixed linear gain [V/count].

    Returns:
        tuple[np.ndarray, str]: Converted data and units.
    """
    norm_factor = np.array(adc_vref) / adc_maxvalue / np.array(linear_fixed_gain)
    return data * norm_factor[:, np.newaxis], "V"


def highpass(
    data: np.ndarray,
    freq: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    f = freq / fe
    if f > 1:
        raise ValueError(f"Selected corner frequency ({freq}) is above Nyquist ({fe}).")

    z, p, k = sp.iirfilter(corners, f, btype="highpass", ftype="butter", output="zpk")
    sos = sp.zpk2sos(z, p, k)
    if zerophase:
        firstpass = sp.sosfilt(sos, data, axis=1)
        return sp.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return sp.sosfilt(sos, data, axis=1)


def lowpass(
    data: np.ndarray,
    freq: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
    fe = 0.5 * fs
    f = freq / fe
    if f > 1:
        f = 1.0
        warnings.warn(
            (
                f"Selected high corner frequency ({freq}) is above "
                f"Nyquist ({fe}). Setting Nyquist as high corner."
            )
        )

    z, p, k = sp.iirfilter(corners, f, btype="lowpass", ftype="butter", output="zpk")
    sos = sp.zpk2sos(z, p, k)
    if zerophase:
        firstpass = sp.sosfilt(sos, data, axis=1)
        return sp.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return sp.sosfilt(sos, data, axis=1)


def pulse_compression(
    transmitted_signal: np.ndarray,
    received_signal: np.ndarray,
    mode: str = "same",
) -> np.ndarray:
    """Applies pulse compression to the data using the given pulse.

    Args:
        transmitted_signal (np.ndarray): The pulse to use for compression.
        received_signal (np.ndarray): The data to be pulse compressed.
    Returns:
        np.ndarray: The pulse-compressed data.
    """
    transmitted_signal = np.squeeze(transmitted_signal)
    received_signal = np.atleast_2d(received_signal)
    M, _ = received_signal.shape
    matched_filter = np.tile(np.conj(transmitted_signal[::-1]), (M, 1))
    return sp.fftconvolve(received_signal, matched_filter, mode=mode, axes=1)


def taper(
    npts: int,
    max_percentage: float = 0.05,
    max_length: float | None = None,
    window_type: str = "hann",
    side: str = "both",
    sampling_rate: float = 1.0,
    **kwargs,
) -> npt.NDArray[np.float64]:
    if side not in ["left", "right", "both"]:
        raise ValueError("side must be 'left', 'right', or 'both'.")
    max_half_lengths = []
    if max_percentage is not None:
        max_half_lengths.append(int(max_percentage * npts))
    if max_length is not None:
        max_half_lengths.append(int(max_length * sampling_rate))
    if np.all([2 * mhl > npts for mhl in max_half_lengths]):
        warnings.warn(
            (
                "The requested taper is longer than the trace. "
                "The taper will be shortened to trace length."
            ),
            TaperLengthWarning,
        )
    max_half_lengths.append(int(npts / 2))
    window_length = min(max_half_lengths)

    if window_type == "cosine":
        kwargs["p"] = 1.0
    if 2 * window_length == npts:
        taper_sides = sp.get_window((window_type, *kwargs), 2 * window_length)
    else:
        taper_sides = sp.get_window((window_type, *kwargs), 2 * window_length + 1)

    if side == "left":
        taper = np.hstack((taper_sides[:window_length], np.ones(npts - window_length)))
    elif side == "right":
        taper = np.hstack(
            (
                np.ones(npts - window_length),
                taper_sides[len(taper_sides) - window_length :],
            )
        )
    else:
        taper = np.hstack(
            (
                taper_sides[:window_length],
                np.ones(npts - 2 * window_length),
                taper_sides[len(taper_sides) - window_length :],
            )
        )
    return taper
