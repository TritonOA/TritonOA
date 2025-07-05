from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from tritonoa.data.formats.base import (
    BaseReader,
    BaseRecordFormatter,
    DataRecord,
    FileFormatCheckerMixin,
)
from tritonoa.data.signal import SignalParams
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import (
    TIME_PRECISION,
    ClockParameters,
    correct_clock_drift,
    correct_sampling_rate,
)


class WHOI3DVHAFileFormat(FileFormatCheckerMixin, Enum):
    FORMAT = "WHOI3DVHA"
    BIN = ".BIN"


@dataclass(frozen=True)
class WHOI3DVHAHeader:
    """Data class representing a 3DAT binary data file record header."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    channels: int
    sampling_rate: float
    bytes_per_channel: int
    record_length_secs: float
    records_per_file: int
    record_number: int
    record_data_bytes: int
    dataMask: int
    comp2Mask: int
    bit_shift: int
    step: np.ndarray
    offset: np.ndarray

    def datetime(self, precision: str | None = TIME_PRECISION) -> np.datetime64:
        """
        Convert the date and time fields from a RecordHeader to a np.datetime64 object
        with microsecond precision.

        Args:
            header: RecordHeader object containing year, month, day, hour, minute, second fields

        Returns:
            np.datetime64: DateTime with microsecond precision
        """
        dt = datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
        )
        return np.datetime64(dt, precision)

    @property
    def num_samples(self) -> int:
        """
        Calculate the number of samples in the record.

        Returns:
            int: Number of samples in the record.
        """
        return int(self.sampling_rate * self.record_length_secs)


class WHOI3DVHAReader(BaseReader):
    def read(
        self,
        file_path: Path,
        channels: int | list[int] | None = None,
        metadata: dict | None = None,
    ) -> DataStream:
        channels = [channels] if isinstance(channels, int) else channels
        raw_data, header = self.read_raw_data(file_path, channels=channels)
        return DataStream(
            stats=DataStreamStats(
                channels=[i for i in range(header.channels)],
                sampling_rate=header.sampling_rate,
                time_init=header.datetime(),
                metadata=metadata,
            ),
            data=raw_data,
        )

    def read_raw_data(
        self, filename: Path, records: int = 1, channels: list[int] = None
    ) -> tuple[npt.NDArray[np.float64], WHOI3DVHAHeader]:
        with open(filename, "rb") as fid:
            fid.seek(0, 0)
            header = self._read_header(fid)
            scans = int(header.sampling_rate * header.record_length_secs)
            data_array = self._read_raw_data(fid, scans, header.channels)
            if channels is not None:
                return data_array[channels, :], header
            return data_array, header

    def _read_raw_data(
        self, fid: BinaryIO, scans: int, channels: int
    ) -> npt.NDArray[np.float64]:
        total_values = int(channels * scans)
        data_array = data_array = np.fromfile(fid, dtype=np.float64, count=total_values)
        if len(data_array) < total_values:
            raise RuntimeError("Error reading [scans x channels] worth of data")
        return data_array.reshape((channels, scans), order="F")

    def read_headers(self, filename: Path) -> list[WHOI3DVHAHeader]:
        """Read data record header from an SIO file.

        SIO files contain only one header per file; however, to maintain
        consistenum_channelsy with the API, this method returns a list containing
        the header.

        Args:
            filename (Path): File to read.

        Returns:
            list[SIOHeader]: Data record headers.
        """
        with open(filename, "rb") as f:
            return [self._read_header(f)]

    @staticmethod
    def _read_header(fid: BinaryIO) -> WHOI3DVHAHeader:
        """Extract the record header from an open 3DVHA binary data file.

        Args:
            file_obj: An open file object in binary read mode

        Returns:
            WHOI3DVHAHeader: A data class containing all header information

        Raises:
            RuntimeError: If any part of the header cannot be read correctly
        """

        # Define a helper function to read a single field
        def read_field(format_str, field_name, count=1):
            try:
                data = fid.read(struct.calcsize(format_str) * count)
                if len(data) < struct.calcsize(format_str) * count:
                    raise RuntimeError(
                        f"Failed to read {field_name}: unexpected end of file"
                    )

                if count == 1:
                    return struct.unpack(format_str, data)[0]
                else:
                    return np.array(struct.unpack(f"{count}{format_str[0]}", data))
            except Exception as e:
                raise RuntimeError(f"Error reading {field_name} in header: {str(e)}")

        header_fields = [
            ("I", "year"),
            ("I", "month"),
            ("I", "day"),
            ("I", "hour"),
            ("I", "minute"),
            ("I", "second"),
            ("I", "channels"),
            ("I", "sampling_rate"),  # Will be converted to float later
            ("I", "bytes_per_channel"),
            ("I", "record_length_secs"),  # Will be converted to float later
            ("I", "records_per_file"),
            ("I", "record_number"),
            ("I", "record_data_bytes"),
            ("I", "dataMask"),
            ("I", "comp2Mask"),
            ("i", "bit_shift"),  # Note: int32 (lowercase i)
        ]

        header_data = {name: read_field(fmt, name) for fmt, name in header_fields}

        header_data["sampling_rate"] = float(header_data["sampling_rate"])
        header_data["record_length_secs"] = float(header_data["record_length_secs"])

        header_data["step"] = read_field("d", "step", count=8)
        header_data["offset"] = read_field("d", "offset", count=8)

        return WHOI3DVHAHeader(**header_data)

    def condition_data(
        self, raw_data: npt.NDArray, *args, **kwargs
    ) -> tuple[npt.NDArray[np.float64], None]:
        return raw_data, None


class WHOI3DVHARecordFormatter(BaseRecordFormatter):

    file_format = "WHOI3DVHA"

    @staticmethod
    def callback(records: list[DataRecord]) -> list[DataRecord]:
        """Format SHRU records.

        Args:
            records (list[Record]): List of records.

        Returns:
            list[Record]: List of records.
        """
        if len(records) == 1:
            return records
        return records

    def format_record(
        self,
        filename: Path,
        record_number: int,
        header: WHOI3DVHAHeader,
        clock: ClockParameters,
        conditioner: SignalParams,
    ):
        conditioner.fill_like_channels(header.channels)
        ts = header.datetime(precision=TIME_PRECISION)
        fs = header.sampling_rate
        return DataRecord(
            filename=filename,
            record_number=record_number,
            file_format=self.file_format,
            timestamp_orig=ts,
            timestamp=correct_clock_drift(ts, clock),
            sampling_rate_orig=fs,
            sampling_rate=correct_sampling_rate(fs, clock.drift_rate),
            npts=header.num_samples,
            nch=header.channels,
            gain=conditioner.gain,
            sensitivity=conditioner.sensitivity,
        )
