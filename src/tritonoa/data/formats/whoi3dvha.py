from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import struct

import numpy as np

import tritonoa.data.formats.base as base
from tritonoa.data.signal import SignalParams
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import (
    TIME_CONVERSION_FACTOR,
    TIME_PRECISION,
    ClockParameters,
    convert_to_datetime,
    correct_clock_drift,
    correct_sampling_rate,
)
from tritonoa.data.signal import db_to_linear


class WHOI3DVHAFileFormat(base.FileFormatCheckerMixin, Enum):
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


class WHOI3DVHAReader(base.BaseReader):
    def read(self, file_path: Path):

        with open(file_path, "rb") as fid:
            fid.seek(0, 0)
            header = self._read_header(fid)
            raw_data = self.read_raw_data(
                fid,
                int(header.sampling_rate * header.record_length_secs),
                header.channels,
            )

        return DataStream(
            stats=DataStreamStats(
                channels=[i for i in range(header.channels)],
                sampling_rate=header.sampling_rate,
                time_init=header.datetime(),
            ),
            data=raw_data,
        )

    def read_raw_data(self, fid, scans: int, channels: int):
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
    def _read_header(file_obj):
        """
        Extract the next record header from an open 3DAT binary data file.

        Args:
            file_obj: An open file object in binary read mode

        Returns:
            WHOI3DVHAHeader: A data class containing all header information

        Raises:
            RuntimeError: If any part of the header cannot be read correctly
        """
        try:
            year = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading year in header")

        try:
            month = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading month in header")

        try:
            day = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading day in header")

        try:
            hour = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading hour in header")

        try:
            minute = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading minute in header")

        try:
            second = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading second in header")

        try:
            channels = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading channels in header")

        try:
            sampling_rate = float(struct.unpack("I", file_obj.read(4))[0])
        except:
            raise RuntimeError("Error reading sampling_rate in header")

        try:
            bytes_per_channel = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading bytes_per_channel in header")

        try:
            record_length_secs = float(struct.unpack("I", file_obj.read(4))[0])
        except:
            raise RuntimeError("Error reading record_length_secs in header")

        try:
            records_per_file = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading records_per_file in header")

        try:
            record_number = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading record_number in header")

        try:
            record_data_bytes = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading record_data_bytes in header")

        try:
            dataMask = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading dataMask in header")

        try:
            comp2Mask = struct.unpack("I", file_obj.read(4))[0]
        except:
            raise RuntimeError("Error reading comp2Mask in header")

        try:
            bit_shift = struct.unpack("i", file_obj.read(4))[
                0
            ]  # Note: int32 (lowercase i)
        except:
            raise RuntimeError("Error reading bit_shift in header")

        # Read arrays of doubles
        try:
            step_data = file_obj.read(8 * 8)  # 8 doubles, 8 bytes each
            step = np.array(struct.unpack("8d", step_data))
        except:
            raise RuntimeError("Error reading steps in header")

        try:
            offset_data = file_obj.read(8 * 8)  # 8 doubles, 8 bytes each
            offset = np.array(struct.unpack("8d", offset_data))
        except:
            raise RuntimeError("Error reading offsets in header")

        # Create and return the RecordHeader data class instance
        return WHOI3DVHAHeader(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            channels=channels,
            sampling_rate=sampling_rate,
            bytes_per_channel=bytes_per_channel,
            record_length_secs=record_length_secs,
            records_per_file=records_per_file,
            record_number=record_number,
            record_data_bytes=record_data_bytes,
            dataMask=dataMask,
            comp2Mask=comp2Mask,
            bit_shift=bit_shift,
            step=step,
            offset=offset,
        )

    def condition_data(): ...


class WHOI3DVHARecordFormatter(base.BaseRecordFormatter):

    file_format = "WHOI3DVHA"

    @staticmethod
    def callback(records: list[base.DataRecord]) -> list[base.DataRecord]:
        """Format SHRU records.

        The time stamp of the first record in the 4-channel SHRU files seem
        to be rounded to seconds. This function corrects the time stamp of
        the first record by calculating the time offset between the first
        and second records using the number of points in the record and
        the sampling rate.

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
        conditioner.fill_like_channels(header.ch)
        ts = header.datetime(precision=TIME_PRECISION)
        fs = header.sampling_rate
        return base.DataRecord(
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
