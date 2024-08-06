# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union


@dataclass
class Hydrophone:
    """Hydrophone.

    Args:
        serial_number (int, optional): Serial number. Defaults to 0.
        fixed_gain (float, optional): Fixed gain [dB]. Defaults to 0.0.
        sensitivity (float, optional): Sensitivity [dB]. Defaults to 0.0.
    """

    serial_number: int = 0
    fixed_gain: float = 0.0
    sensitivity: float = 0.0


@dataclass
class HydrophoneSpecs:
    """Specifications for one or more hydrophones.

    Args:
        fixed_gain (Optional[list[float]], optional): Fixed gain [dB]. Defaults to 0.0.
        sensitivity (Optional[list[float]], optional): Sensitivity [dB]. Defaults to 0.0.
        serial_number (Optional[list[int]], optional): Serial number. Defaults to 0.
    """

    fixed_gain: Union[float, list[float]] = 0.0
    sensitivity: Union[float, list[float]] = 0.0
    serial_number: Union[float, list[int]] = 0


@dataclass
class CalibrationData:
    ...