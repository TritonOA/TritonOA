"""Base classes for modular signal calibration system.

This module defines the abstract interfaces for the calibration system,
following the Strategy pattern for extensibility and the Chain of Responsibility
pattern for composability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CalibrationMetadata:
    """Metadata tracking calibration provenance and parameters.

    Attributes:
        version: Calibration version string.
        date: Date of calibration.
        valid_until: Expiration date of calibration (if applicable).
        manufacturer: Sensor manufacturer.
        model: Sensor model.
        serial_number: Sensor serial number.
        parameters: Dictionary of calibration parameters applied.
        steps_applied: List of calibration step names applied.
    """

    version: str | None = None
    date: str | None = None
    valid_until: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    steps_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "version": self.version,
            "date": self.date,
            "valid_until": self.valid_until,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "parameters": self.parameters,
            "steps_applied": self.steps_applied,
        }


@dataclass
class CalibratedData:
    """Container for calibrated data with metadata.

    Attributes:
        data: Calibrated data array (channels, samples).
        units: Output units (always 'uPa' for this system).
        sampling_rate: Sampling rate in Hz.
        metadata: Calibration metadata.
    """

    data: NDArray[np.float64]
    units: str
    sampling_rate: float
    metadata: CalibrationMetadata = field(default_factory=CalibrationMetadata)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of data array."""
        return self.data.shape

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self.data.shape[0] if self.data.ndim > 1 else 1

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        return self.data.shape[1] if self.data.ndim > 1 else self.data.shape[0]


class CalibrationStep(ABC):
    """Abstract base class for calibration steps.

    Each calibration step represents a single transformation in the
    calibration chain. Steps can operate in time domain or frequency domain.
    """

    @abstractmethod
    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> NDArray[np.float64]:
        """Apply calibration step to data.

        Args:
            data: Input data array (channels, samples).
            sampling_rate: Sampling rate in Hz.
            **kwargs: Additional parameters specific to the step.

        Returns:
            Calibrated data array with same shape as input.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this calibration step.

        Returns:
            Name of the calibration step.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Get parameters used by this calibration step.

        Returns:
            Dictionary of parameter names and values.
        """
        ...

    def validate_data_shape(self, data: NDArray[np.float64]) -> None:
        """Validate that data has expected shape.

        Args:
            data: Data array to validate.

        Raises:
            ValueError: If data shape is invalid.
        """
        if data.ndim not in [1, 2]:
            raise ValueError(
                f"Data must be 1D or 2D array, got {data.ndim}D"
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.get_parameters()})"


class TimeDomainStep(CalibrationStep):
    """Base class for time-domain calibration steps.

    Time-domain steps apply sample-by-sample transformations
    without requiring frequency domain conversion.
    """

    pass


class FrequencyDomainStep(CalibrationStep):
    """Base class for frequency-domain calibration steps.

    Frequency-domain steps apply corrections in the frequency domain,
    requiring FFT/IFFT operations.
    """

    @abstractmethod
    def get_frequency_response(
        self,
        frequencies: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get frequency response (magnitude and phase).

        Args:
            frequencies: Frequency array in Hz.

        Returns:
            Tuple of (magnitude, phase) where:
                - magnitude: Linear magnitude response
                - phase: Phase response in radians
        """
        ...
