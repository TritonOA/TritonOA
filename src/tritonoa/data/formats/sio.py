# TODO: Place acknowledgement/license for SIOREAD here.

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np

from tritonoa.data.formats.base import (
    BaseReader,
    BaseRecordFormatter,
    DataRecord,
    FileFormatCheckerMixin,
    validate_channels,
)
from tritonoa.data.signal import SignalParams
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import ClockParameters, convert_filename_to_datetime64


class SIOReadError(Exception):
    pass


class SIOReadWarning(Warning):
    pass


class SIODataFormat(Enum):
    F = "f"
    H = "h"


class SIOFileFormat(FileFormatCheckerMixin, Enum):
    FORMAT = "SIO"
    D23 = ".SIO"


@dataclass
class SIOHeader:
    ID: int | None = None
    num_records: int | None = None
    bytes_per_record: int | None = None
    num_channels: int | None = None
    bytes_per_sample: int | None = None
    tfReal: int | None = None
    samples_per_channel: int | None = None
    bs: int | None = None
    fname: str | None = None
    comment: str | None = None
    dtype: str | None = None

    # def __post_init__(self):
    # TODO: Check that dtype is a member of SIOFormat enum.
    #     if self.dtype is not None and self.dtype not in SIOFormat:
    #         raise ValueError("Incorrect format.")

    @property
    def _description(self) -> str:
        return """
            ID = ID Number
            num_records = # of Records in File
            bytes_per_record = # of Bytes per Record
            num_channels = # of channels in File
            bytes_per_sample = # of Bytes per Sample
            tfReal = 0 - integer, 1 - real
            samples_per_channel = # of Samples per Channel
            fname = File name
            comment = Comment String
            bs = Endian check value, should be 32677
            """

    @property
    def records_per_channel(self) -> int:
        """Records per channel."""
        return ceil(self.num_records / self.num_channels)

    @property
    def samples_per_record(self) -> int:
        """Samples per record."""
        return int(self.bytes_per_record / self.bytes_per_sample)


class SIOReader(BaseReader):

    def read(
        self,
        file_path: Path,
        channels: int | list[int] | None = None,
        time_init: float | np.datetime64 = 0.0,
        time_end: float | np.datetime64 | None = None,
        sampling_rate: float | None = None,
        units: str | None = None,
        metadata: dict | None = None,
        # clock: ClockParameters = ClockParameters(),
        # conditioner: SignalParams = SignalParams(),
    ) -> DataStream:
        channels = [channels] if isinstance(channels, int) else channels
        raw_data, header = self.read_raw_data(file_path, channels=channels)
        return DataStream(
            stats=DataStreamStats(
                channels=[i for i in range(header.num_channels)],
                time_init=time_init,
                time_end=time_end,
                sampling_rate=sampling_rate,
                units=units,
                metadata=metadata,
            ),
            data=raw_data,
        )

    @staticmethod
    def _endian_check(f: BinaryIO) -> str:
        for endian in [">", "<"]:
            f.seek(28)
            bs = struct.unpack(endian + "I", f.read(4))[0]  # should be 32677
            if bs == 32677:
                return endian
        raise SIOReadError(f"Problem with byte swap constant: {bs}")

    def read_headers(self, filename: Path) -> list[SIOHeader]:
        """Read data record header from an SIO file.

        SIO files contain only one header per file; however, to maintain
        consistenum_channelsy with the API, this method returns a list containing
        the header.

        Args:
            filename (Path): File to read.

        Returns:
            list[SIOHeader]: Data record headers.
        """
        with open(filename, "rb") as fid:
            header, _ = self._read_header(fid)
        return [header]

    def _read_header(self, fid: BinaryIO) -> tuple[SIOHeader, str]:
        endian = self._endian_check(fid)
        fid.seek(0)
        ID = int(struct.unpack(endian + "I", fid.read(4))[0])
        num_records = int(struct.unpack(endian + "I", fid.read(4))[0])
        bytes_per_record = int(struct.unpack(endian + "I", fid.read(4))[0])
        num_channels = int(struct.unpack(endian + "I", fid.read(4))[0])
        bytes_per_sample = int(struct.unpack(endian + "I", fid.read(4))[0])
        dtype = "h" if bytes_per_sample == 2 else "f"
        tfReal = struct.unpack(endian + "I", fid.read(4))[0]  # 0 = integer, 1 = real
        samples_per_channel = struct.unpack(endian + "I", fid.read(4))[0]
        bs = struct.unpack(endian + "I", fid.read(4))[0]  # should be 32677
        fname = struct.unpack("24s", fid.read(24))[0].decode().strip()
        comment = (
            struct.unpack("72s", fid.read(72))[0].decode().replace("\x00", "").strip()
        )
        return (
            SIOHeader(
                ID=ID,
                num_records=num_records,
                bytes_per_record=bytes_per_record,
                num_channels=num_channels,
                bytes_per_sample=bytes_per_sample,
                tfReal=tfReal,
                samples_per_channel=samples_per_channel,
                bs=bs,
                fname=fname,
                comment=comment,
                dtype=dtype,
            ),
            endian,
        )

    def read_raw_data(
        self,
        filename: Path,
        records: int = 1,
        # s_start: int = 0,
        channels: int | list[int] | None = None,
        # Ns: int = -1,
    ) -> tuple[np.ndarray, SIOHeader]:
        """Translation of Jit Sarkar's sioread.m to Python (which was a
        modification of Aaron Thode's with contributions from Geoff Edelman,
        James Murray, and Dave Ensberg).
        Translated by Hunter Akins, 4 June 2019
        Modified by William Jenkins, 6 April 2022

        Parameters
        ----------
        fname : str
            Name/path to .sio data file to be read.
        s_start : int, default=1
            Sample number from which to begin reading. Must be an integer
            multiple of the record number. To get the record number you can
            run the script with s_start = 1 and then check the header for
            the info.
        Ns : int, default=-1
            Total samples to read
        channels : list (int), default=[]
            Which channels to read (default all) (indexes at 0).
            Channel -1 returns header only, X is empty.=

        Returns
        -------
        X : array (Ns, num_channels)
            Data output matrix.
        header : dict
            Descriptors found in file header.
        """
        Ns = -1
        s_start = 0 if records == 1 else (records - 1) * Ns
        print(channels)
        with open(filename, "rb") as fid:
            header, endian = self._read_header(fid)

            samples_per_record = header.samples_per_record
            samples_per_channel = header.samples_per_channel
            num_channels = header.num_channels

            channels = validate_channels(num_channels, channels=channels)
            # If either channel or # of samples is 0, then return just header
            if (Ns == 0) or ((len(channels) == 1) and (channels[0] == -1)):
                data = []
                return data, header

            # Recheck parameters against header info
            Ns_max = samples_per_channel - s_start + 1
            if Ns == -1:
                Ns = Ns_max  # 	fetch all samples from start point
            if Ns > Ns_max:
                SIOReadWarning(
                    f"More samples requested than present in data file. Returning max num samples: {Ns_max}"
                )
                Ns = Ns_max

            # Check validity of channel list

            # Calculate file offsets
            # Header is size of 1 Record at beginning of file
            r_hoffset = 1
            # Starting and total records needed from file
            r_start = int(
                floor(s_start / samples_per_record) * num_channels + r_hoffset
            )
            r_total = int(ceil(Ns / samples_per_record) * num_channels)

            # # Aggregate loading
            # if inMem:
            # Move to starting location
            fid.seek(r_start * header.bytes_per_record)

            # Read in all records into single column
            factor = 4 if header.dtype == "f" else 2
            raw_data = np.array(
                struct.unpack(
                    endian + header.dtype * r_total * samples_per_record,
                    fid.read(r_total * samples_per_record * factor),
                )
            )
            if len(raw_data) != r_total * samples_per_record:
                raise SIOReadError("Not enough samples read from file")

            # Reshape data into a matrix of records
            raw_data = np.reshape(raw_data, (r_total, samples_per_record))

            # 	Select each requested channel and stack associated records
            m = len(channels)
            n = int(r_total / num_channels * samples_per_record)
            data = np.zeros((m, n))
            # return
            for i in range(len(channels)):
                chan = channels[i]
                blocks = np.arange(chan, r_total, num_channels, dtype="int")
                data[i] = raw_data[blocks].T.reshape(
                    n,
                )

            # Trim unneeded samples from start and end of matrix
            trim_start = int(s_start % samples_per_record)
            if trim_start != 0:
                data = data[trim_start:, :]
            if n > Ns:
                data = data[: int(Ns), :]
            if n < Ns:
                raise SIOReadError(
                    f"Requested # of samples not returned. Check that s_start ({s_start}) is multiple of rec_num: {samples_per_record}"
                )
            return data, header


class SIORecordFormatter(BaseRecordFormatter):

    file_format = "SIO"

    @staticmethod
    def callback(records: list[DataRecord]) -> list[DataRecord]:
        """Format SIO records.

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
        header: SIOHeader,
        conditioner: SignalParams,
        *args,
        **kwargs,
    ):
        conditioner.fill_like_channels(header.num_channels)
        return DataRecord(
            filename=filename,
            record_number=record_number,
            file_format=self.file_format,
            npts=header.samples_per_channel,
            nch=header.num_channels,
            gain=conditioner.gain,
            sensitivity=conditioner.sensitivity,
        )


def callback(
    records: list[DataRecord],
    sampling_rate: float,
    doy_indices: tuple[int, int],
    hour_indices: tuple[int, int],
    minute_indices: tuple[int, int],
    year_indices: tuple[int, int] | None = None,
    year: int | None = None,
    seconds_indices: tuple[int, int] | None = None,
) -> list[DataRecord]:
    """Format SIO records.

    Args:
        records: List of records.
        sampling_rate: Sampling rate of the data (Hz).
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
            doy_indices,
            hour_indices,
            minute_indices,
            year_indices,
            year,
            seconds_indices,
        )
        record.timestamp_orig = timestamp
        record.timestamp = timestamp
        record.sampling_rate_orig = sampling_rate
        record.sampling_rate = sampling_rate

    return records
