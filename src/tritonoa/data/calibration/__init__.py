"""Modular signal calibration system for underwater acoustics.

This package provides a flexible, extensible system for calibrating
acoustic sensor data. It supports:

- Time-domain corrections (ADC, gain, sensitivity)
- Frequency-domain corrections (magnitude and phase responses)
- Composable calibration chains
- Configuration-driven calibration from YAML/JSON
- Sensor calibration registry

Example:
    >>> from tritonoa.data.calibration import CalibrationFactory
    >>> # Load calibration from file
    >>> chain = CalibrationFactory.from_yaml("configs/sensors/my_sensor.yaml")
    >>> # Apply to data
    >>> result = chain.apply(raw_data, sampling_rate=48000)
    >>> print(result.units)  # 'uPa'
"""

from tritonoa.data.calibration.base import (
    CalibrationMetadata,
    CalibrationStep,
    CalibratedData,
    FrequencyDomainStep,
    TimeDomainStep,
)
from tritonoa.data.calibration.chain import CalibrationChain
from tritonoa.data.calibration.factory import CalibrationFactory
from tritonoa.data.calibration.registry import CalibrationRegistry

__all__ = [
    # Base classes
    "CalibrationStep",
    "TimeDomainStep",
    "FrequencyDomainStep",
    "CalibrationMetadata",
    "CalibratedData",
    # Main classes
    "CalibrationChain",
    "CalibrationFactory",
    "CalibrationRegistry",
]

__version__ = "1.0.0"
