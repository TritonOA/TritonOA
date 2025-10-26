from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO
import warnings
import struct

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.io import wavfile

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
from tritonoa.data.signal import (
    convert_counts_to_voltage,
    convert_voltage_to_pressure,
    db_to_linear,
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

    @property
    def bit_depth(self) -> int:
        return int(self.bytes_per_sample * 8)


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

        data, units = self.condition_data(raw_data, conditioner=conditioner)

        return DataStream(
            stats=DataStreamStats(
                channels=[i for i in range(header.num_channels)],
                time_init=time_init,
                time_end=time_end,
                sampling_rate=header.sample_rate,
                units=units,
                metadata=metadata,
            ),
            data=data,
        )

    def read_headers(self, filename: Path) -> list[WAVHeader]:
        with open(filename, "rb") as f:
            return [self._read_header(f)]

    @staticmethod
    def _read_header(fid: BinaryIO) -> WAVHeader:
        """Read WAV file header for both PCM and floating-point formats.

        Manually parses the RIFF/WAVE file structure to extract header information
        without loading the audio data into memory.
        """
        # Read RIFF header
        riff_header = fid.read(12)
        if len(riff_header) < 12:
            raise ValueError("Invalid WAV file: too short")

        riff_id, _, wave_id = struct.unpack("<4sI4s", riff_header)
        if riff_id != b"RIFF" or wave_id != b"WAVE":
            raise ValueError("Invalid WAV file: not a RIFF/WAVE file")

        # Find and read fmt chunk
        fmt_found = False
        data_found = False
        num_samples = 0

        while not fmt_found or not data_found:
            chunk_header = fid.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)

            if chunk_id == b"fmt ":
                fmt_found = True
                fmt_data = fid.read(chunk_size)

                # Parse fmt chunk (minimum 16 bytes)
                format_code, num_channels, sample_rate, _, _, bits_per_sample = (
                    struct.unpack("<HHIIHH", fmt_data[:16])
                )
                bytes_per_sample = bits_per_sample // 8

                # 1 = PCM, 3 = IEEE float
                if format_code == 1:
                    compression_type = "NONE"
                    compression_name = "not compressed"
                elif format_code == 3:
                    compression_type = "FLOAT"
                    compression_name = "IEEE float"
                else:
                    compression_type = f"TYPE_{format_code}"
                    compression_name = f"format code {format_code}"

            elif chunk_id == b"data":
                data_found = True
                num_samples = chunk_size // (num_channels * bytes_per_sample)
                fid.seek(chunk_size, 1)
            else:
                fid.seek(chunk_size, 1)

        if not fmt_found:
            raise ValueError("Invalid WAV file: no fmt chunk found")
        if not data_found:
            raise ValueError("Invalid WAV file: no data chunk found")

        return WAVHeader(
            sample_rate=sample_rate,
            num_channels=num_channels,
            bytes_per_sample=bytes_per_sample,
            num_samples=num_samples,
            compression_type=compression_type,
            compression_name=compression_name,
        )

    def read_raw_data(
        self, filename: Path, records: int = 1, channels: list[int] | None = None
    ) -> tuple[NDArray[np.float64], WAVHeader]:
        header = self._read_header(filename.open("rb"))

        channels = validate_channels(header.num_channels, channels)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            _, data = wavfile.read(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif channels is not None:
            data = data[:, channels].T
            header.num_channels = len(channels)
        
        self.bit_depth = header.bit_depth
        self.compression_type = header.compression_type
        return data, header

    def condition_data(
        self, raw_data: ArrayLike, conditioner: SignalParams
    ) -> tuple[NDArray[np.float64], None]:
        if self.compression_type == "FLOAT":
            return raw_data.astype(np.float64), None
        try:
            conditioner.check_dimensions(raw_data.shape[0])
        except ValueError as e:
            warnings.warn(
                f"Incorrect number of gain or sensitivity values set: {e} "
                f"len(adc_vref)={len(conditioner.adc_vref)}, "
                f"len(gain)={len(conditioner.gain)}, "
                f"len(sensitivity)={len(conditioner.sensitivity)}, "
                f"len(channels)={raw_data.shape[0]}. "
                f"Returning the data unconditioned."
            )
            return raw_data, "counts"

        linear_fixed_gain = db_to_linear(conditioner.gain)
        linear_sensitivity = db_to_linear(conditioner.sensitivity)
        ADC_max = 2 ** (self.bit_depth - 1) - 1

        voltage, _ = convert_counts_to_voltage(
            raw_data, linear_fixed_gain, conditioner.adc_vref, ADC_max
        )
        return convert_voltage_to_pressure(voltage, linear_sensitivity)


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
        *,
        filename: Path,
        record_number: int,
        header: WAVHeader,
        conditioner: SignalParams,
        **kwargs,
    ) -> DataRecord:
        conditioner.fill_like_channels(header.num_channels)
        return DataRecord(
            filename=filename,
            record_number=record_number,
            file_format=self.file_format,
            timestamp_orig=0.0,
            timestamp=0.0,
            sampling_rate_orig=header.sample_rate,
            sampling_rate=header.sample_rate,
            npts=header.num_samples,
            nch=header.num_channels,
            adc_vref=conditioner.adc_vref,
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
    clock: ClockParameters = ClockParameters(),
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
        record.timestamp = correct_clock_drift(timestamp, clock)
        record.sampling_rate = correct_sampling_rate(
            record.sampling_rate_orig, clock.drift_rate
        )

    return records
