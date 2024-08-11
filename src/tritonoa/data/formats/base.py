# -*- coding: utf-8 -*-

# TODO: Clean up base class argument specs and returned types.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import polars as pl


class ChannelMismatchError(Exception):
    pass


@dataclass
class DataRecord:
    """Data record object.

    Attributes:
        filename (Path): File name.
        record_number (int): Record number.
        file_format (FileFormat): File format.
        timestamp_orig (Union[np.datetime64, pl.Datetime]): Original timestamp.
        timestamp (Union[np.datetime64, pl.Datetime]): Timestamp.
        sampling_rate_orig (float): Original sampling rate.
        sampling_rate (float): Sampling rate.
        npts (Optional[int], optional): Number of points in the record. Defaults to None.
    """

    filename: Path
    record_number: int
    file_format: str
    timestamp_orig: np.datetime64 | pl.Datetime
    timestamp: np.datetime64 | pl.Datetime
    sampling_rate_orig: float
    sampling_rate: float
    npts: Optional[int] = None
    nch: Optional[int] = None
    gain: float = 0.0
    sensitivity: float = 0.0


class FileFormatCheckerMixin:
    @classmethod
    def is_format(cls, value: str) -> bool:
        normalized_value = value.lower()
        return any(normalized_value == item.value.lower() for item in cls)


@dataclass
class Header(Protocol):
    """Protocol for file-format--specific header objects."""

    ...


class BaseReader(ABC):
    @abstractmethod
    def read(self, file_path: Path, **kwargs) -> None: ...

    @abstractmethod
    def read_headers(self, filename: Path) -> list[Header]: ...

    @abstractmethod
    def read_raw_data() -> None: ...


class BaseRecordFormatter(ABC):
    @abstractmethod
    def callback(self, file_format: str) -> None: ...

    @abstractmethod
    def format_record(self, **kwargs) -> None: ...


def validate_channels(
    nch: int, channels: Optional[int | list[int]] = None
) -> list[int]:
    if channels is None:
        return list(range(nch))

    channels = [channels] if not isinstance(channels, list) else channels
    if any((i > nch - 1) for i in channels):
        raise ChannelMismatchError(
            f"Channel {len(channels)} requested but only got {nch} from header."
        )
    return channels


# def _validate_channels(channels: list[int], num_channels: int) -> list[int]:
#     if len(channels) == 0:
#         channels = list(range(num_channels))  # 	fetch all channels
#     if len([x for x in channels if (x < 0) or (x > (num_channels - 1))]) != 0:
#         raise SIOReadError(
#             "Channel #s must be within range 0 to " + str(num_channels - 1)
#         )
#     return channels
