"""Time-domain calibration steps for signal conditioning.

This module provides time-domain calibration steps that apply
sample-by-sample transformations such as ADC conversion, gain,
and sensitivity corrections.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tritonoa.data.calibration.base import TimeDomainStep


class ADCConversion(TimeDomainStep):
    """Convert ADC counts to voltage.

    This step converts raw ADC counts to voltage based on the
    ADC reference voltage and bit depth.

    Args:
        adc_vref: ADC reference voltage(s) in volts. Can be:
            - Single float: Same reference for all channels
            - List of floats: Per-channel references
        adc_bits: ADC bit depth (e.g., 16, 24, 32).
        signed: Whether ADC uses signed representation (default: True).

    Example:
        >>> step = ADCConversion(adc_vref=2.5, adc_bits=24)
        >>> voltage = step.apply(raw_counts, sampling_rate=48000)
    """

    def __init__(
        self,
        adc_vref: float | list[float],
        adc_bits: int,
        signed: bool = True,
    ):
        self.adc_vref = adc_vref
        self.adc_bits = adc_bits
        self.signed = signed

        # Calculate maximum ADC value
        if signed:
            self.adc_max = 2 ** (adc_bits - 1)
        else:
            self.adc_max = 2**adc_bits - 1

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply ADC conversion.

        Args:
            data: Raw ADC counts (channels, samples).
            sampling_rate: Sampling rate in Hz (unused but required by interface).
            **kwargs: Additional parameters (unused).

        Returns:
            Voltage data (channels, samples) in volts.
        """
        self.validate_data_shape(data)

        # Handle single-channel or multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels = data.shape[0]

        # Convert adc_vref to array
        if isinstance(self.adc_vref, (int, float)):
            vref = np.full(num_channels, self.adc_vref)
        else:
            vref = np.array(self.adc_vref)
            if len(vref) != num_channels:
                raise ValueError(
                    f"adc_vref length ({len(vref)}) does not match "
                    f"number of channels ({num_channels})"
                )

        # Convert counts to voltage
        conversion_factor = vref / self.adc_max
        voltage = data * conversion_factor[:, np.newaxis]

        return voltage

    def get_name(self) -> str:
        return "ADC Conversion"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "adc_vref": self.adc_vref,
            "adc_bits": self.adc_bits,
            "signed": self.signed,
            "adc_max": self.adc_max,
        }


class Gain(TimeDomainStep):
    """Apply gain correction.

    This step applies a gain correction to convert between voltage levels
    or correct for preamplifier gain.

    Args:
        gain_db: Gain in dB. Can be:
            - Single float: Same gain for all channels
            - List of floats: Per-channel gains
        invert: If True, applies inverse gain (default: False).

    Example:
        >>> step = Gain(gain_db=[20.0, 20.0, 15.0, 15.0])
        >>> corrected = step.apply(voltage_data, sampling_rate=48000)
    """

    def __init__(self, gain_db: float | list[float], invert: bool = False):
        self.gain_db = gain_db
        self.invert = invert

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply gain correction.

        Args:
            data: Input data (channels, samples).
            sampling_rate: Sampling rate in Hz (unused but required by interface).
            **kwargs: Additional parameters (unused).

        Returns:
            Gain-corrected data (channels, samples).
        """
        self.validate_data_shape(data)

        # Handle single-channel or multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels = data.shape[0]

        # Convert gain_db to array
        if isinstance(self.gain_db, (int, float)):
            gain_db = np.full(num_channels, self.gain_db)
        else:
            gain_db = np.array(self.gain_db)
            if len(gain_db) != num_channels:
                raise ValueError(
                    f"gain_db length ({len(gain_db)}) does not match "
                    f"number of channels ({num_channels})"
                )

        # Convert dB to linear
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Apply gain (invert if needed)
        if self.invert:
            corrected = data * gain_linear[:, np.newaxis]
        else:
            corrected = data / gain_linear[:, np.newaxis]

        return corrected

    def get_name(self) -> str:
        return "Gain Correction"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "gain_db": self.gain_db,
            "invert": self.invert,
        }


class Sensitivity(TimeDomainStep):
    """Apply hydrophone sensitivity correction.

    This step converts voltage to pressure (micropascals) using
    hydrophone sensitivity specifications.

    Args:
        sensitivity_db: Sensitivity in dB re 1V/µPa. Can be:
            - Single float: Same sensitivity for all channels
            - List of floats: Per-channel sensitivities
        reference_pressure: Reference pressure in µPa (default: 1.0).

    Example:
        >>> step = Sensitivity(sensitivity_db=-170.0)
        >>> pressure = step.apply(voltage_data, sampling_rate=48000)
    """

    def __init__(
        self,
        sensitivity_db: float | list[float],
        reference_pressure: float = 1.0,
    ):
        self.sensitivity_db = sensitivity_db
        self.reference_pressure = reference_pressure

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply sensitivity correction.

        Args:
            data: Voltage data (channels, samples) in volts.
            sampling_rate: Sampling rate in Hz (unused but required by interface).
            **kwargs: Additional parameters (unused).

        Returns:
            Pressure data (channels, samples) in micropascals.
        """
        self.validate_data_shape(data)

        # Handle single-channel or multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels = data.shape[0]

        # Convert sensitivity_db to array
        if isinstance(self.sensitivity_db, (int, float)):
            sens_db = np.full(num_channels, self.sensitivity_db)
        else:
            sens_db = np.array(self.sensitivity_db)
            if len(sens_db) != num_channels:
                raise ValueError(
                    f"sensitivity_db length ({len(sens_db)}) does not match "
                    f"number of channels ({num_channels})"
                )

        # Convert dB to linear (V/µPa)
        sens_linear = 10.0 ** (sens_db / 20.0)

        # Convert voltage to pressure: P = V / S
        # where S is in V/µPa, so P is in µPa
        pressure = data / sens_linear[:, np.newaxis]

        # Apply reference pressure scaling
        if self.reference_pressure != 1.0:
            pressure *= self.reference_pressure

        return pressure

    def get_name(self) -> str:
        return "Sensitivity Correction"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "sensitivity_db": self.sensitivity_db,
            "reference_pressure": self.reference_pressure,
        }


class ScalarMultiply(TimeDomainStep):
    """Apply scalar multiplication.

    This is a generic step for applying any scalar correction factor.

    Args:
        factor: Multiplication factor(s). Can be:
            - Single float: Same factor for all channels
            - List of floats: Per-channel factors

    Example:
        >>> step = ScalarMultiply(factor=2.0)
        >>> scaled = step.apply(data, sampling_rate=48000)
    """

    def __init__(self, factor: float | list[float]):
        self.factor = factor

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply scalar multiplication.

        Args:
            data: Input data (channels, samples).
            sampling_rate: Sampling rate in Hz (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Scaled data (channels, samples).
        """
        self.validate_data_shape(data)

        # Handle single-channel or multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels = data.shape[0]

        # Convert factor to array
        if isinstance(self.factor, (int, float)):
            factors = np.full(num_channels, self.factor)
        else:
            factors = np.array(self.factor)
            if len(factors) != num_channels:
                raise ValueError(
                    f"factor length ({len(factors)}) does not match "
                    f"number of channels ({num_channels})"
                )

        return data * factors[:, np.newaxis]

    def get_name(self) -> str:
        return "Scalar Multiplication"

    def get_parameters(self) -> dict[str, Any]:
        return {"factor": self.factor}


class Offset(TimeDomainStep):
    """Apply offset correction.

    This step adds or subtracts a DC offset.

    Args:
        offset: Offset value(s). Can be:
            - Single float: Same offset for all channels
            - List of floats: Per-channel offsets

    Example:
        >>> step = Offset(offset=-0.5)
        >>> corrected = step.apply(data, sampling_rate=48000)
    """

    def __init__(self, offset: float | list[float]):
        self.offset = offset

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply offset correction.

        Args:
            data: Input data (channels, samples).
            sampling_rate: Sampling rate in Hz (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Offset-corrected data (channels, samples).
        """
        self.validate_data_shape(data)

        # Handle single-channel or multi-channel
        if data.ndim == 1:
            data = data[np.newaxis, :]

        num_channels = data.shape[0]

        # Convert offset to array
        if isinstance(self.offset, (int, float)):
            offsets = np.full(num_channels, self.offset)
        else:
            offsets = np.array(self.offset)
            if len(offsets) != num_channels:
                raise ValueError(
                    f"offset length ({len(offsets)}) does not match "
                    f"number of channels ({num_channels})"
                )

        return data + offsets[:, np.newaxis]

    def get_name(self) -> str:
        return "Offset Correction"

    def get_parameters(self) -> dict[str, Any]:
        return {"offset": self.offset}
