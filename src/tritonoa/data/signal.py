# -*- coding: utf-8 -*-

# TODO: Implement cupy support
# try:
# import cupyx.scipy.signal as signal
# except ImportError:
# import scipy.signal as signal

from dataclasses import dataclass
import warnings

import numpy as np
import scipy.signal as signal


@dataclass
class SignalParams:
    """Specifications for one or more hydrophones.

    Args:
        gain (list[float] | optional): Gain [dB], converts counts to V. Defaults to 0.0 (no effect).
        sensitivity (list[float] | optional): Sensitivity [dB], converts V to uPa. Defaults to 0.0 (no effect).
    """

    gain: float | list[float] = 0.0
    sensitivity: float | list[float] = 0.0

    def check_dimensions(self, num_channels: int):
        """Checks if the dimensions of the parameters match the channels.

        Args:
            channels (list): List of channels.
        """
        if not isinstance(self.gain, list):
            self.gain = [self.gain] * num_channels
        if not isinstance(self.sensitivity, list):
            self.sensitivity = [self.sensitivity] * num_channels
        if len(self.gain) != num_channels or len(self.sensitivity) != num_channels:
            # TODO: Implement specific exception for this.
            raise ValueError(
                "The number of gains and sensitivities must match the number of channels."
            )


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


def get_filter(filt_type: str) -> callable:
    FILTER_REGISTRY = {
        "bandpass": bandpass,
        "bandstop": bandstop,
        "highpass": highpass,
        "lowpass": lowpass,
    }
    return FILTER_REGISTRY[filt_type]


def bandpass(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
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

    z, p, k = signal.iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data, axis=1)
        return signal.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return signal.sosfilt(sos, data, axis=1)


def bandstop(
    data: np.ndarray,
    freqmin: float,
    freqmax: float,
    fs: float,
    corners: int = 4,
    zerophase: bool = False,
) -> np.ndarray:
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
    z, p, k = signal.iirfilter(
        corners, [low, high], btype="bandstop", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data, axis=1)
        return signal.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return signal.sosfilt(sos, data, axis=1)


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

    z, p, k = signal.iirfilter(
        corners, f, btype="highpass", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data, axis=1)
        return signal.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return signal.sosfilt(sos, data, axis=1)


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

    z, p, k = signal.iirfilter(
        corners, f, btype="lowpass", ftype="butter", output="zpk"
    )
    sos = signal.zpk2sos(z, p, k)
    if zerophase:
        firstpass = signal.sosfilt(sos, data, axis=1)
        return signal.sosfilt(sos, firstpass[:, :-1], axis=1)[:, :-1]
    return signal.sosfilt(sos, data, axis=1)
