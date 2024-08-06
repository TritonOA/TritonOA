# -*- coding: utf-8 -*-

from dataclasses import dataclass


# @dataclass
# class Hydrophone:
#     """Hydrophone.

#     Args:
#         serial_number (int, optional): Serial number. Defaults to 0.
#         fixed_gain (float, optional): Fixed gain [dB]. Defaults to 0.0.
#         sensitivity (float, optional): Sensitivity [dB]. Defaults to 0.0.
#     """

#     serial_number: int = 0
#     fixed_gain: float = 0.0
#     sensitivity: float = 0.0


@dataclass
class HydrophoneSpecs:
    """Specifications for one or more hydrophones.

    Args:
        gain (list[float] | optional): Gain [dB], converts counts to V. Defaults to 0.0 (no effect).
        sensitivity (list[float] | optional): Sensitivity [dB], converts V to uPa. Defaults to 0.0 (no effect).
    """

    gain: float | list[float] = 0.0
    sensitivity: float | list[float] = 0.0