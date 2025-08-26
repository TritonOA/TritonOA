from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO
from wave import Wave_read

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.io.wavfile import read as wavread

from tritonoa.data.formats.base import (
    BaseReader,
    BaseRecordFormatter,
    DataRecord,
    FileFormatCheckerMixin,
    validate_channels,
)
from tritonoa.data.signal import SignalParams
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import (
    ClockParameters,
    convert_filename_to_datetime64,
    correct_clock_drift,
    correct_sampling_rate,
)


class WAVFileFormat(FileFormatCheckerMixin, Enum):
    FORMAT = "WAV"
    WAV = ".WAV"


@dataclass
class WAVHeader:
    sample_rate: int
    num_channels: int
    bytes_per_sample: int
    num_samples: int
    compression_type: str = "NONE"
    compression_name: str = "not compressed"


class WAVReader(BaseReader):

    def read(
        self,
        file_path: Path,
        channels: int | list[int] | None = None,
        clock: ClockParameters = ClockParameters(),
        conditioner: SignalParams = SignalParams(),
        time_init: float | DTypeLike = 0.0,
        time_end: float | DTypeLike | None = None,
        units: str = "counts",
        metadata: dict | None = None,
    ) -> DataStream:
        channels = [channels] if isinstance(channels, int) else channels
        raw_data, header = self.read_raw_data(file_path, channels=channels)
        return DataStream(
            stats=DataStreamStats(
                channels=[i for i in range(header.num_channels)],
                time_init=time_init,
                time_end=time_end,
                sampling_rate=header.sample_rate,
                units=units,
                metadata=metadata,
            ),
            data=raw_data,
        )

    def read_headers(self, filename: Path) -> list[WAVHeader]:
        with open(filename, "rb") as f:
            return [self._read_header(f)]

    @staticmethod
    def _read_header(fid: BinaryIO) -> WAVHeader:
        with Wave_read(fid) as wav:
            return WAVHeader(
                sample_rate=wav.getframerate(),
                num_channels=wav.getnchannels(),
                bytes_per_sample=wav.getsampwidth(),
                num_samples=wav.getnframes(),
                compression_type=wav.getcomptype(),
                compression_name=wav.getcompname(),
            )

    def read_raw_data(
        self, filename: Path, records: int = 1, channels: list[int] | None = None
    ) -> tuple[NDArray[np.float64], WAVHeader]:
        header = self._read_header(filename.open("rb"))

        channels = validate_channels(header.num_channels, channels)

        _, data = wavread(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if channels is not None:
            data = data[channels, :]
            header.num_channels = len(channels)

        return convert_wav_int_to_float(data), header

    def condition_data(
        self, raw_data: ArrayLike, *args, **kwargs
    ) -> tuple[NDArray[np.float64], None]:
        return raw_data, None


class WAVRecordFormatter(BaseRecordFormatter):
    file_format = "WAV"

    @staticmethod
    def callback(records: list[DataRecord]) -> list[DataRecord]:
        """Format WAV records.

        Args:
            records: List of DataRecord objects.

        Returns:
            List of formatted DataRecord objects.
        """
        return records

    def format_record(
        self,
        filename: Path,
        record_number: int,
        header: WAVHeader,
        clock: ClockParameters,
        conditioner: SignalParams,
    ) -> DataRecord:
        conditioner.fill_like_channels(header.num_channels)
        ts = 0.0
        fs = header.sample_rate
        return DataRecord(
            filename=filename,
            record_number=record_number,
            file_format=self.file_format,
            timestamp_orig=ts,
            timestamp=correct_clock_drift(ts, clock),
            sampling_rate_orig=fs,
            sampling_rate=correct_sampling_rate(fs, clock.drift_rate),
            npts=header.num_samples,
            nch=header.num_channels,
            gain=conditioner.gain,
            sensitivity=conditioner.sensitivity,
        )


def callback(
    records: list[DataRecord],
    hour_indices: tuple[int, int],
    minute_indices: tuple[int, int],
    month_indices: tuple[int, int] | None = None,
    day_indices: tuple[int, int] | None = None,
    doy_indices: tuple[int, int] | None = None,
    year_indices: tuple[int, int] | None = None,
    year: int | None = None,
    seconds_indices: tuple[int, int] | None = None,
) -> list[DataRecord]:
    """Format WAV records.

    Args:
        records: List of records.
        doy_indices: (start_index, length) for day-of-year.
        hour_indices: (start_index, length) for hours.
        minute_indices: (start_index, length) for minutes.
        year_indices: (start_index, length) for year.
        year: Direct year specification if not in filename.
        seconds_indices: (start_index, length) for seconds.
    """
    for record in records:
        timestamp = convert_filename_to_datetime64(
            record.filename.name,
            hour_indices,
            minute_indices,
            month_indices,
            day_indices,
            doy_indices,
            year_indices,
            year,
            seconds_indices,
        )
        record.timestamp_orig = timestamp
        record.timestamp = timestamp

    return records


def convert_wav_int_to_float(data: NDArray) -> NDArray[np.float64]:
    """Convert integer data to double precision data.

    Args:
        data: Raw audio data (from .wav)

    Returns:
        array of 32 bit float data
    """
    if data.dtype == "float32":
        return data.astype(np.float64)
    if data.dtype == "int32":
        nbits = 32
    if data.dtype == "int24":
        nbits = 24
    if data.dtype == "int16":
        nbits = 16

    max_nbits = float(2 ** (nbits - 1))

    return (data / (max_nbits + 1)).astype(np.float64)
