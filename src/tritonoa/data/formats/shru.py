# TODO: Implement method to take channel input specification and return only that
# channel's conditioning values.

from array import array
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import struct
from typing import BinaryIO
import warnings

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

ADC_HALFSCALE = 2.5  # ADC half scale volts (+/- half scale is ADC i/p range)
ADC_MAXVALUE = 2**23  # ADC maximum halfscale o/p value, half the 2's complement range
BYTES_HDR = 1024
BYTES_PER_SAMPLE = 3


class SHRUFileFormat(base.FileFormatCheckerMixin, Enum):
    FORMAT = "SHRU"
    D23 = ".D23"


@dataclass(frozen=True)
class SHRUHeader:
    """Data record header for SHRU files

    Attributes:
        rhkey (str): Record header key (first).
        date (array): Date of record.
        time (array): Time of record.
        microsec (int): Microseconds.
        rec (int): Record number.
        ch (int): SHRU channels.
        npts (int): Number of samples per channel.
        rhfs (float): Sample rate (Hz).
        unused (array): Unused.
        rectime (int): Record time.
        rhlat (str): Latitude.
        rhlng (str): Longitude.
        nav120 (array): Navigation data.
        nav115 (array): Navigation data.
        nav110 (array): Navigation data.
        POS (str): Position.
        unused2 (array): Unused.
        nav_day (int): Navigation day.
        nav_hour (int): Navigation hour.
        nav_min (int): Navigation minute.
        nav_sec (int): Navigation second.
        lblnav_flag (int): Navigation flag.
        unused3 (array): Unused.
        reclen (int): Record length.
        acq_day (int): Acquisition day.
        acq_hour (int): Acquisition hour.
        acq_min (int): Acquisition minute.
        acq_sec (int): Acquisition second.
        acq_recnum (int): Acquisition record number.
        ADC_tagbyte (int): ADC tag byte.
        glitchcode (int): Glitch code.
        bootflag (int): Boot flag.
        internal_temp (float): Internal temperature (C).
        bat_voltage (float): Battery voltage (V).
        bat_current (float): Battery current (A).
        drh_status (float): DRH status.
        proj (str): Project.
        shru_num (str): SHRU number.
        vla (str): VLA.
        hla (str): HLA.
        filename (str): Filename.
        record (str): Total records.
        adate (str): Date.
        atime (str): Time.
        file_length (int): File length.
        total_records (int): Total records.
        unused4 (array): Unused.
        adc_mode (int): ADC mode.
        adc_clk_code (int): ADC clock code.
        unused5 (array): Unused.
        timebase (int): Timebase.
        unused6 (array): Unused.
        unused7 (array): Unused.
        rhkeyl (str): Record header key (last).
    """

    rhkey: str
    date: array
    time: array
    microsec: int
    rec: int
    ch: int
    npts: int
    rhfs: float
    unused: array
    rectime: int
    rhlat: str
    rhlng: str
    nav120: array
    nav115: array
    nav110: array
    POS: str
    unused2: array
    nav_day: int
    nav_hour: int
    nav_min: int
    nav_sec: int
    lblnav_flag: int
    unused3: array
    reclen: int
    acq_day: int
    acq_hour: int
    acq_min: int
    acq_sec: int
    acq_recnum: int
    ADC_tagbyte: int
    glitchcode: int
    bootflag: int
    internal_temp: float
    bat_voltage: float
    bat_current: float
    drh_status: float
    proj: str
    shru_num: str
    vla: str
    hla: str
    filename: str
    record: str
    adate: str
    atime: str
    file_length: int
    total_records: int
    unused4: array
    adc_mode: int
    adc_clk_code: int
    unused5: array
    timebase: int
    unused6: array
    unused7: array
    rhkeyl: str


class SHRUReader(base.BaseReader):

    def read(
        self,
        file_path: Path,
        records: int | list[int] = None,
        channels: int | list[int] = None,
        clock: ClockParameters = ClockParameters(),
        conditioner: SignalParams = SignalParams(),
    ) -> DataStream:

        channels = [channels] if isinstance(channels, int) else channels

        # Read data and headers
        raw_data, header = self.read_raw_data(
            file_path,
            records=records,
            channels=channels,
        )

        # Condition 24-bit data into pressure (uPa)
        data, units = self.condition_data(raw_data, conditioner=conditioner)

        try:
            ts_orig = _get_timestamp(header)
            ts = correct_clock_drift(ts_orig, clock)
            fs = correct_sampling_rate(header.rhfs, clock.drift_rate)
        except:
            warnings.warn(
                "Unable to find timestamp; setting to 0; no clock drift applied."
            )
            ts = 0.0
            fs = header.rhfs

        return DataStream(
            stats=DataStreamStats(
                channels=channels,
                time_init=ts,
                sampling_rate=fs,
                units=units,
            ),
            data=data,
        )

    def read_raw_data(
        self,
        file_path: Path,
        records: int | list[int] = None,
        channels: int | list[int] = None,
    ) -> tuple[np.ndarray, SHRUHeader]:
        """Read 24-bit data from a SHRU file.

        Args:
            filename (Path): File to read.
            records (int | list[int]): Record number(s) to read.
            channels (int | list[int]): Channel number(s) to read.
            fixed_gain (float | list[float], optional): Fixed gain [dB]. Defaults to 20.0.
            drhs (Optional[list[SHRUHeader]], optional): Data record headers. Defaults to None.

        Returns:
            tuple[np.ndarray, SHRUHeader]: Data and header.

        Raises:
            ValueError: If the number of channels requested exceeds the number of channels in the file.
            ValueError: If the length of fixed_gain does not match the length of channels.
            ValueError: If the record is corrupted.
        """
        headers = self.read_headers(file_path)

        if records is None:
            records = list(range(len(headers)))
        elif not isinstance(records, list):
            records = [records]

        nch = self.get_num_channels(headers)
        channels = base.validate_channels(nch, channels=channels)

        header1 = headers[records[0]]

        # Calculate the number of bytes to skip to start reading data
        skip_bytes = sum([hdr.reclen for hdr in headers[0 : records[0]]])

        with open(file_path, "rb") as fid:
            fid.seek(skip_bytes, 0)  # Skip to selected record
            # Read the data
            spts = headers[records[0]].npts
            data = np.nan * np.ones((len(channels), spts * len(records)))

            for i, record_idx in enumerate(records):
                if spts != headers[record_idx].npts:
                    raise ValueError(
                        f"Record {record_idx} corrupted in file '{file_path.name}'."
                    )

                # Skip record header
                fid.seek(BYTES_HDR, 1)
                # Read data
                data_block, count = self.get_data_record(fid, nch, spts)
                # data_block, count = read_bit24_to_int(fid, nch, spts)
                if count != nch * spts:
                    raise ValueError(
                        f"Record {record_idx} corrupted in file '{file_path.name}'."
                    )

                # Keep only selected channels
                data_block = data_block[:, channels]

                # Store data
                data[:, i * spts : (i + 1) * spts] = data_block.T

        return data, header1

    @staticmethod
    def _read_header(fid: BinaryIO) -> tuple[SHRUHeader, bool]:
        """Read a SHRU data record header.

        Args:
            fid (BinaryIO): File to read.

        Returns:
            tuple[SHRUHeader, int]: Header and status.
        """
        end_of_file = False

        rhkey = fid.read(4).decode("utf-8")
        # Check if end of file:
        if rhkey == "":
            end_of_file = True
            return None, end_of_file

        date = array(
            "i",
            [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]],
        )
        time = array(
            "i",
            [struct.unpack(">H", fid.read(2))[0], struct.unpack(">H", fid.read(2))[0]],
        )
        microsec = struct.unpack(">H", fid.read(2))[0]
        rec = struct.unpack(">H", fid.read(2))[0]
        ch = struct.unpack(">H", fid.read(2))[0]
        npts = struct.unpack(">i", fid.read(4))[0]
        rhfs = struct.unpack(">f", fid.read(4))[0]
        unused = array("i", struct.unpack("BB", fid.read(2)))
        rectime = struct.unpack(">I", fid.read(4))[0]
        rhlat = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        rhlng = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        nav120 = array("f", struct.unpack("28I", fid.read(28 * 4)))
        nav115 = array("f", struct.unpack("28I", fid.read(28 * 4)))
        nav110 = array("f", struct.unpack("28I", fid.read(28 * 4)))
        POS = fid.read(128).decode("utf-8").replace("\x00", "").strip()
        unused2 = array("b", fid.read(208))
        nav_day = struct.unpack(">h", fid.read(2))[0]
        nav_hour = struct.unpack(">h", fid.read(2))[0]
        nav_min = struct.unpack(">h", fid.read(2))[0]
        nav_sec = struct.unpack(">h", fid.read(2))[0]
        lblnav_flag = struct.unpack(">h", fid.read(2))[0]
        unused3 = array("b", fid.read(2))
        reclen = struct.unpack(">I", fid.read(4))[0]
        acq_day = struct.unpack(">h", fid.read(2))[0]
        acq_hour = struct.unpack(">h", fid.read(2))[0]
        acq_min = struct.unpack(">h", fid.read(2))[0]
        acq_sec = struct.unpack(">h", fid.read(2))[0]
        acq_recnum = struct.unpack(">h", fid.read(2))[0]
        ADC_tagbyte = struct.unpack(">h", fid.read(2))[0]
        glitchcode = struct.unpack(">h", fid.read(2))[0]
        bootflag = struct.unpack(">h", fid.read(2))[0]
        internal_temp = float(fid.read(16).decode("utf-8").replace("\x00", "").strip())
        bat_voltage = float(fid.read(16).decode("utf-8").replace("\x00", "").strip())
        bat_current = float(fid.read(16).decode("utf-8").replace("\x00", "").strip())
        drh_status = float(fid.read(16).decode("utf-8").replace("\x00", "").strip())
        proj = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        shru_num = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        vla = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        hla = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        filename = fid.read(32).decode("utf-8").replace("\x00", "").strip()
        record = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        adate = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        atime = fid.read(16).decode("utf-8").replace("\x00", "").strip()
        file_length = struct.unpack(">I", fid.read(4))[0]
        total_records = struct.unpack(">I", fid.read(4))[0]
        unused4 = array("b", fid.read(2))
        adc_mode = struct.unpack(">h", fid.read(2))[0]
        adc_clk_code = struct.unpack(">h", fid.read(2))[0]
        unused5 = array("b", fid.read(2))
        timebase = struct.unpack(">i", fid.read(4))[0]
        unused6 = array("b", fid.read(12))
        unused7 = array("b", fid.read(12))
        rhkeyl = fid.read(4).decode("utf-8")

        if rhkey == "DATA" and rhkeyl == "ADAT":
            raise ValueError("Record header keys do not match!")

        return (
            SHRUHeader(
                rhkey=rhkey,
                date=date,
                time=time,
                microsec=microsec,
                rec=rec,
                ch=ch,
                npts=npts,
                rhfs=rhfs,
                unused=unused,
                rectime=rectime,
                rhlat=rhlat,
                rhlng=rhlng,
                nav120=nav120,
                nav115=nav115,
                nav110=nav110,
                POS=POS,
                unused2=unused2,
                nav_day=nav_day,
                nav_hour=nav_hour,
                nav_min=nav_min,
                nav_sec=nav_sec,
                lblnav_flag=lblnav_flag,
                unused3=unused3,
                reclen=reclen,
                acq_day=acq_day,
                acq_hour=acq_hour,
                acq_min=acq_min,
                acq_sec=acq_sec,
                acq_recnum=acq_recnum,
                ADC_tagbyte=ADC_tagbyte,
                glitchcode=glitchcode,
                bootflag=bootflag,
                internal_temp=internal_temp,
                bat_voltage=bat_voltage,
                bat_current=bat_current,
                drh_status=drh_status,
                proj=proj,
                shru_num=shru_num,
                vla=vla,
                hla=hla,
                filename=filename,
                record=record,
                adate=adate,
                atime=atime,
                file_length=file_length,
                total_records=total_records,
                unused4=unused4,
                adc_mode=adc_mode,
                adc_clk_code=adc_clk_code,
                unused5=unused5,
                timebase=timebase,
                unused6=unused6,
                unused7=unused7,
                rhkeyl=rhkeyl,
            ),
            end_of_file,
        )

    def read_headers(self, filename: Path) -> list[SHRUHeader]:
        """Read all data record headers from a SHRU file.

        Args:
            filename (Path): File to read.

        Returns:
            list[SHRUHeader]: Data record headers.
        """
        with open(filename, "rb") as fid:
            fid.seek(0, 0)  # Go to the beginning of the file
            headers = []
            record_counter = 0

            while True:
                header, end_of_file = self._read_header(fid)

                if end_of_file:
                    logging.debug(
                        f"End of file reached; found {len(headers)} record(s)."
                    )
                    return headers

                if header.rhkey != "DATA":
                    logging.warning(f"Bad record found at #{record_counter}")
                    return headers

                # Skip over data
                bytes_rec = header.ch * header.npts * 3
                fid.seek(bytes_rec, 1)

                record_counter += 1
                headers.append(header)

    def condition_data(
        self, data: np.ndarray, conditioner: SignalParams
    ) -> tuple[np.ndarray, str]:
        """Condition 24-bit data to pressure.

        Args:
            data (np.ndarray): 24-bit data.
            fixed_gain (list[float]): Fixed gain [dB].
            sensitivity (list[float]): Sensitivity [dB].

        Returns:
            tuple[np.ndarray, str]: Converted data and units.
        """
        try:
            conditioner.check_dimensions(data.shape[0])
        except ValueError as e:  # TODO: Implement specific exception for this.
            warnings.warn(
                f"Incorrect number of gain or sensitivity values set: {e}"
                f"len(gain)={len(conditioner.gain)}, "
                f"len(sensitivity)={len(conditioner.sensitivity)}, "
                f"len(channels)={data.shape[0]}."
                f"Returning the data unconditioned."
            )
            return data, "counts"

        linear_fixed_gain = db_to_linear(conditioner.gain)
        linear_sensitivity = db_to_linear(conditioner.sensitivity)
        data, _ = self._convert_to_voltage(data, linear_fixed_gain)
        return self._convert_to_pressure(data, linear_sensitivity)

    @staticmethod
    def _convert_24bit_to_int(
        raw_data: bytes, nch: int, spts: int
    ) -> tuple[np.ndarray, int]:
        """Convert 24-bit data to 32-bit integers.

        Args:
            raw_data (bytes): 24-bit data.
            nch (int): Number of channels.
            spts (int): Number of samples per channel.

        Returns:
            tuple[np.ndarray, int]: Converted data and count.
        """
        # Convert the byte data to 24-bit integers (bit manipulation may be required)
        # Ensure we have a suitable type for bit manipulation
        data = (
            np.frombuffer(raw_data, dtype=np.uint8)
            .reshape(spts, nch, BYTES_PER_SAMPLE)
            .astype(np.int32)
        )
        # Combine bytes to form 24-bit integers
        data_24bit = (data[:, :, 0] << 16) | (data[:, :, 1] << 8) | data[:, :, 2]
        # Adjust for sign if necessary (assuming two's complement representation)
        data_24bit[data_24bit >= 2**23] -= 2**24
        return data_24bit.reshape(spts, nch), len(raw_data) // BYTES_PER_SAMPLE

    @staticmethod
    def _convert_to_pressure(
        data: np.ndarray, linear_sensitivity: list[float]
    ) -> tuple[np.ndarray, str]:
        """Convert voltage data to pressure.

        Args:
            data (np.ndarray): Voltage data.

        Returns:
            tuple[np.ndarray, str]: Converted data and units.
        """
        return data * np.array(linear_sensitivity)[:, np.newaxis], "uPa"

    @staticmethod
    def _convert_to_voltage(
        data: np.ndarray, linear_fixed_gain: list[float]
    ) -> tuple[np.ndarray, str]:
        """Convert 24-bit data to voltage.

        Args:
            data (np.ndarray): 24-bit data.
            fixed_linear_gain (list[float]): Fixed linear gain [V/count].

        Returns:
            tuple[np.ndarray, str]: Converted data and units.
        """
        norm_factor = ADC_HALFSCALE / ADC_MAXVALUE / np.array(linear_fixed_gain)
        return data * norm_factor[:, np.newaxis], "V"

    def get_data_record(
        self, fid: BinaryIO, nch: int, spts: int
    ) -> tuple[np.ndarray, int]:
        total_bytes = nch * spts * BYTES_PER_SAMPLE
        data_bytes = fid.read(total_bytes)
        if len(data_bytes) != total_bytes:
            raise ValueError("Failed to read all data bytes.")
        return self._convert_24bit_to_int(data_bytes, nch, spts)

    @staticmethod
    def get_num_channels(headers: list[SHRUHeader]) -> int:
        for record in headers:
            if record.ch != headers[0].ch:
                warnings.warn("Number of channels varies across records.")
        return headers[0].ch


class SHRURecordFormatter(base.BaseRecordFormatter):

    file_format = "SHRU"

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
        offset_for_first_record = np.timedelta64(
            int(
                TIME_CONVERSION_FACTOR * records[0].npts / records[0].sampling_rate_orig
            ),
            TIME_PRECISION,
        )
        records[0].timestamp = records[1].timestamp - offset_for_first_record
        records[0].timestamp_orig = records[1].timestamp_orig - offset_for_first_record
        return records

    def format_record(
        self,
        filename: Path,
        record_number: int,
        header: SHRUHeader,
        clock: ClockParameters,
        conditioner: SignalParams,
    ):
        conditioner.fill_like_channels(header.ch)
        ts = _get_timestamp(header)
        fs = header.rhfs
        return base.DataRecord(
            filename=filename,
            record_number=record_number,
            file_format=self.file_format,
            timestamp_orig=ts,
            timestamp=correct_clock_drift(ts, clock),
            sampling_rate_orig=fs,
            sampling_rate=correct_sampling_rate(fs, clock.drift_rate),
            npts=header.npts,
            nch=header.ch,
            gain=conditioner.gain,
            sensitivity=conditioner.sensitivity,
        )


def _get_timestamp(header: SHRUHeader) -> np.datetime64:
    """Return the timestamp of a data record header.

    Args:
        header (SHRUHeader): Data record header.

    Returns:
        np.datetime64: Timestamp.
    """
    year = header.date[0]
    yd = header.date[1]
    minute = header.time[0]
    millisec = header.time[1]
    microsec = header.microsec
    return convert_to_datetime(year, yd, minute, millisec, microsec)
