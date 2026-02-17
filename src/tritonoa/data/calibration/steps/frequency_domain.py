"""Frequency-domain calibration steps for signal conditioning.

This module provides frequency-domain calibration steps that apply
frequency-dependent corrections such as magnitude and phase responses.
These are essential for correcting non-flat hydrophone responses.
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, signal

from tritonoa.data.calibration.base import FrequencyDomainStep


class FrequencyResponse(FrequencyDomainStep):
    """Apply frequency-dependent magnitude and phase correction.

    This step corrects for frequency-dependent hydrophone response by
    applying the inverse of the measured frequency response in the
    frequency domain.

    Args:
        frequencies: Frequency points in Hz where response is known.
        magnitude_db: Magnitude response in dB re 1V/µPa at each frequency.
        phase_deg: Phase response in degrees at each frequency (default: None).
        interpolation: Interpolation method ('linear', 'cubic', 'pchip').
        extrapolation: How to handle frequencies outside range:
            - 'constant': Use edge values
            - 'linear': Linear extrapolation
            - 'raise': Raise error
        apply_inverse: If True, applies inverse response (default: True).

    Example:
        >>> step = FrequencyResponse(
        ...     frequencies=[10, 100, 1000, 10000],
        ...     magnitude_db=[-170, -168, -165, -170],
        ...     phase_deg=[0, 2, 5, 15]
        ... )
        >>> corrected = step.apply(data, sampling_rate=48000)
    """

    def __init__(
        self,
        frequencies: list[float] | NDArray[np.float64],
        magnitude_db: list[float] | NDArray[np.float64],
        phase_deg: list[float] | NDArray[np.float64] | None = None,
        interpolation: Literal["linear", "cubic", "pchip"] = "cubic",
        extrapolation: Literal["constant", "linear", "raise"] = "constant",
        apply_inverse: bool = True,
    ):
        self.frequencies = np.array(frequencies)
        self.magnitude_db = np.array(magnitude_db)
        self.phase_deg = np.array(phase_deg) if phase_deg is not None else None
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        self.apply_inverse = apply_inverse

        # Validate inputs
        if len(self.frequencies) != len(self.magnitude_db):
            raise ValueError(
                f"frequencies length ({len(self.frequencies)}) must match "
                f"magnitude_db length ({len(self.magnitude_db)})"
            )

        if self.phase_deg is not None and len(self.phase_deg) != len(self.frequencies):
            raise ValueError(
                f"phase_deg length ({len(self.phase_deg)}) must match "
                f"frequencies length ({len(self.frequencies)})"
            )

        # Check frequencies are sorted
        if not np.all(np.diff(self.frequencies) > 0):
            raise ValueError("frequencies must be strictly increasing")

        # Create interpolators
        self._setup_interpolators()

    def _setup_interpolators(self) -> None:
        """Setup interpolation functions for magnitude and phase."""
        fill_value = "extrapolate" if self.extrapolation == "linear" else np.nan

        if self.interpolation == "linear":
            self._mag_interp = interpolate.interp1d(
                self.frequencies,
                self.magnitude_db,
                kind="linear",
                fill_value=fill_value,
                bounds_error=(self.extrapolation == "raise"),
            )
            if self.phase_deg is not None:
                self._phase_interp = interpolate.interp1d(
                    self.frequencies,
                    self.phase_deg,
                    kind="linear",
                    fill_value=fill_value,
                    bounds_error=(self.extrapolation == "raise"),
                )
        elif self.interpolation == "cubic":
            self._mag_interp = interpolate.interp1d(
                self.frequencies,
                self.magnitude_db,
                kind="cubic",
                fill_value=fill_value,
                bounds_error=(self.extrapolation == "raise"),
            )
            if self.phase_deg is not None:
                self._phase_interp = interpolate.interp1d(
                    self.frequencies,
                    self.phase_deg,
                    kind="cubic",
                    fill_value=fill_value,
                    bounds_error=(self.extrapolation == "raise"),
                )
        elif self.interpolation == "pchip":
            self._mag_interp = interpolate.PchipInterpolator(
                self.frequencies, self.magnitude_db, extrapolate=False
            )
            if self.phase_deg is not None:
                self._phase_interp = interpolate.PchipInterpolator(
                    self.frequencies, self.phase_deg, extrapolate=False
                )
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

    def get_frequency_response(
        self, frequencies: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get interpolated frequency response.

        Args:
            frequencies: Frequency array in Hz.

        Returns:
            Tuple of (magnitude_linear, phase_radians).
        """
        # Interpolate magnitude
        mag_db_interp = self._mag_interp(frequencies)

        # Handle extrapolation for constant mode
        if self.extrapolation == "constant":
            mag_db_interp[frequencies < self.frequencies[0]] = self.magnitude_db[0]
            mag_db_interp[frequencies > self.frequencies[-1]] = self.magnitude_db[-1]

        # Convert dB to linear
        magnitude = 10.0 ** (mag_db_interp / 20.0)

        # Interpolate phase if available
        if self.phase_deg is not None:
            phase_deg_interp = self._phase_interp(frequencies)
            if self.extrapolation == "constant":
                phase_deg_interp[frequencies < self.frequencies[0]] = self.phase_deg[0]
                phase_deg_interp[frequencies > self.frequencies[-1]] = self.phase_deg[
                    -1
                ]
            phase = np.deg2rad(phase_deg_interp)
        else:
            phase = np.zeros_like(frequencies)

        return magnitude, phase

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply frequency response correction.

        Args:
            data: Input data (channels, samples).
            sampling_rate: Sampling rate in Hz.
            **kwargs: Additional parameters (unused).

        Returns:
            Frequency-corrected data (channels, samples).
        """
        self.validate_data_shape(data)

        # Handle single-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels, num_samples = data.shape

        # Apply to each channel
        corrected = np.zeros_like(data)
        for ch in range(num_channels):
            corrected[ch, :] = self._apply_channel(data[ch, :], sampling_rate)

        return corrected

    def _apply_channel(
        self, data: NDArray[np.float64], sampling_rate: float
    ) -> NDArray[np.float64]:
        """Apply frequency response correction to a single channel.

        Args:
            data: Single-channel data (samples,).
            sampling_rate: Sampling rate in Hz.

        Returns:
            Corrected single-channel data.
        """
        # Compute FFT
        spectrum = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), d=1.0 / sampling_rate)

        # Get frequency response
        magnitude, phase = self.get_frequency_response(freqs)

        # Create complex correction filter
        if self.apply_inverse:
            # Apply inverse: divide by magnitude, subtract phase
            correction = (1.0 / magnitude) * np.exp(-1j * phase)
        else:
            # Apply forward: multiply by magnitude, add phase
            correction = magnitude * np.exp(1j * phase)

        # Apply correction
        corrected_spectrum = spectrum * correction

        # Transform back to time domain
        corrected = np.fft.irfft(corrected_spectrum, n=len(data))

        return corrected

    def get_name(self) -> str:
        return "Frequency Response Correction"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "frequencies": self.frequencies.tolist(),
            "magnitude_db": self.magnitude_db.tolist(),
            "phase_deg": self.phase_deg.tolist() if self.phase_deg is not None else None,
            "interpolation": self.interpolation,
            "extrapolation": self.extrapolation,
            "apply_inverse": self.apply_inverse,
        }


class HighPassFilter(FrequencyDomainStep):
    """Apply high-pass filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order (default: 4).
        filter_type: Filter type ('butterworth', 'chebyshev1', 'chebyshev2', 'elliptic').

    Example:
        >>> step = HighPassFilter(cutoff_freq=10.0, order=4)
        >>> filtered = step.apply(data, sampling_rate=48000)
    """

    def __init__(
        self,
        cutoff_freq: float,
        order: int = 4,
        filter_type: Literal[
            "butterworth", "chebyshev1", "chebyshev2", "elliptic"
        ] = "butterworth",
    ):
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.filter_type = filter_type

    def get_frequency_response(
        self, frequencies: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get filter frequency response.

        Args:
            frequencies: Frequency array in Hz.

        Returns:
            Tuple of (magnitude, phase).
        """
        # This is a placeholder - actual implementation would compute
        # the filter response at given frequencies
        raise NotImplementedError(
            "get_frequency_response not implemented for filters"
        )

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply high-pass filter.

        Args:
            data: Input data (channels, samples).
            sampling_rate: Sampling rate in Hz.
            **kwargs: Additional parameters.

        Returns:
            Filtered data.
        """
        self.validate_data_shape(data)

        # Handle single-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # Design filter
        nyquist = sampling_rate / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist

        if self.filter_type == "butterworth":
            sos = signal.butter(
                self.order, normalized_cutoff, btype="highpass", output="sos"
            )
        else:
            raise NotImplementedError(f"Filter type {self.filter_type} not implemented")

        # Apply filter to each channel
        filtered = signal.sosfiltfilt(sos, data, axis=1)

        return filtered

    def get_name(self) -> str:
        return "High-Pass Filter"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "cutoff_freq": self.cutoff_freq,
            "order": self.order,
            "filter_type": self.filter_type,
        }


class LowPassFilter(FrequencyDomainStep):
    """Apply low-pass filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz.
        order: Filter order (default: 4).
        filter_type: Filter type ('butterworth', 'chebyshev1', 'chebyshev2', 'elliptic').

    Example:
        >>> step = LowPassFilter(cutoff_freq=20000.0, order=4)
        >>> filtered = step.apply(data, sampling_rate=48000)
    """

    def __init__(
        self,
        cutoff_freq: float,
        order: int = 4,
        filter_type: Literal[
            "butterworth", "chebyshev1", "chebyshev2", "elliptic"
        ] = "butterworth",
    ):
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.filter_type = filter_type

    def get_frequency_response(
        self, frequencies: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get filter frequency response."""
        raise NotImplementedError(
            "get_frequency_response not implemented for filters"
        )

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply low-pass filter."""
        self.validate_data_shape(data)

        # Handle single-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # Design filter
        nyquist = sampling_rate / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist

        if self.filter_type == "butterworth":
            sos = signal.butter(
                self.order, normalized_cutoff, btype="lowpass", output="sos"
            )
        else:
            raise NotImplementedError(f"Filter type {self.filter_type} not implemented")

        # Apply filter to each channel
        filtered = signal.sosfiltfilt(sos, data, axis=1)

        return filtered

    def get_name(self) -> str:
        return "Low-Pass Filter"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "cutoff_freq": self.cutoff_freq,
            "order": self.order,
            "filter_type": self.filter_type,
        }
