from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from numpy import datetime64
from numpy.typing import ArrayLike
from polars import Datetime


class ChannelMismatchError(Exception):
    """Raised when requested channel indices exceed available channels."""

    pass


@dataclass
class DataRecord:
    """Data record object.

    Attributes:
        filename: File name.
        record_number: Record number.
        file_format: File format.
        timestamp_orig: Original timestamp.
        timestamp: Timestamp.
        sampling_rate_orig: Original sampling rate.
        sampling_rate: Sampling rate.
        npts: Number of points in the record.
        nch: Number of channels.
        adc_vref: ADC voltage reference.
        gain: Gain value.
        sensitivity: Sensitivity value.
    """

    filename: Path
    record_number: int
    file_format: str
    timestamp_orig: datetime64 | Datetime | None = None
    timestamp: datetime64 | Datetime | None = None
    sampling_rate_orig: float = 1.0
    sampling_rate: float = 1.0
    npts: int | None = None
    nch: int | None = None
    adc_vref: float | list[float] = 1.0
    gain: float | list[float] = 0.0
    sensitivity: float | list[float] = 0.0


class FileFormatCheckerMixin:
    """Mixin class for checking file format values against enum members."""

    @classmethod
    def is_format(cls, value: str) -> bool:
        """Check if a value matches any enum member value.

        Args:
            value: The format string to check.

        Returns:
            True if the value matches any enum member, False otherwise.
        """
        normalized_value = value.lower()
        return any(normalized_value == item.value.lower() for item in cls)


@dataclass
class Header(Protocol):
    """Protocol for file-format-specific header objects."""

    ...


class BaseReader(ABC):
    """Abstract base class for file format readers.

    This class defines the interface that all file format readers must implement.
    Subclasses should provide concrete implementations for reading and processing
    data files in specific formats.
    """

    @abstractmethod
    def condition_data(self, raw_data: ArrayLike, *args, **kwargs) -> None:
        """Condition raw data by applying transformations.

        Args:
            raw_data: Raw data array to be conditioned.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        ...

    @abstractmethod
    def read(self, file_path: Path, **kwargs) -> None:
        """Read data from a file.

        Args:
            file_path: Path to the file to read.
            **kwargs: Additional keyword arguments.
        """
        ...

    @abstractmethod
    def read_headers(self, filename: Path) -> list[Header]:
        """Read headers from a file.

        Args:
            filename: Path to the file to read headers from.

        Returns:
            List of header objects parsed from the file.
        """
        ...

    @abstractmethod
    def read_raw_data() -> None:
        """Read raw data from a file.

        This method should be implemented by subclasses to read raw data
        in the specific file format.
        """
        ...


class BaseRecordFormatter(ABC):
    """Abstract base class for record formatters.

    This class defines the interface for formatting data records from
    various file formats into a standardized format.
    """

    @abstractmethod
    def callback(self, file_format: str) -> None:
        """Execute callback for a specific file format.

        Args:
            file_format: The file format identifier.
        """
        ...

    @abstractmethod
    def format_record(self, **kwargs) -> None:
        """Format a data record.

        Args:
            **kwargs: Keyword arguments containing record data to format.
        """
        ...


def validate_channels(nch: int, channels: int | list[int] | None = None) -> list[int]:
    """Validate and normalize channel indices.

    Args:
        nch: Total number of available channels.
        channels: Channel index or list of channel indices to validate.
            If None, returns all channels.

    Returns:
        List of validated channel indices.

    Raises:
        ChannelMismatchError: If any requested channel index exceeds the
            available number of channels.
    """
    if channels is None:
        return list(range(nch))

    channels = [channels] if not isinstance(channels, list) else channels
    if any((i > nch - 1) for i in channels):
        raise ChannelMismatchError(
            f"Channel {len(channels)} requested but only got {nch} from header."
        )
    return channels
