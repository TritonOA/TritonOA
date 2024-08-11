# -*- coding: utf-8 -*-

# TODO: Place acknowledgement/license for SIOREAD here.

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor
from pathlib import Path
import struct
from typing import BinaryIO, Optional

import numpy as np

from tritonoa.data.formats import base
from tritonoa.data.stream import DataStream, DataStreamStats


class SIOReadError(Exception):
    pass


class SIOReadWarning(Warning):
    pass


class SIODataFormat(Enum):
    F = "f"
    H = "h"


class SIOFileFormat(base.FileFormatCheckerMixin, Enum):
    FORMAT = "SIO"
    D23 = ".SIO"


@dataclass
class SIOHeader:
    ID: Optional[int] = None
    num_records: Optional[int] = None
    bytes_per_record: Optional[int] = None
    num_channels: Optional[int] = None
    bytes_per_sample: Optional[int] = None
    tfReal: Optional[int] = None
    samples_per_channel: Optional[int] = None
    bs: Optional[int] = None
    fname: Optional[str] = None
    comment: Optional[str] = None
    dtype: Optional[str] = None

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


class SIOReader:

    def read(
        self,
        filename: Path,
        channels: Optional[int | list[int]] = None,
        sampling_rate: Optional[float] = 1.0,
        units: Optional[str] = "normalized",
    ):
        data, header = self.read_raw_data(filename, channels=channels)
        return DataStream(
            stats=DataStreamStats(
                channels=channels,
                time_init=0.0,
                sampling_rate=sampling_rate,
                units=units,
            )
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
        with open(filename, "rb") as f:
            header, _, _ = self._read_header(filename)
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
        file_path: Path,
        s_start: int = 0,
        channels: Optional[int | list[int]] = None,
        Ns: int = -1,
    ) -> tuple[np.ndarray, dict]:
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

        with open(file_path, "rb") as fid:
            header, endian = self._read_header(fid)

            samples_per_record = header.samples_per_record
            samples_per_channel = header.samples_per_channel
            num_channels = header.num_channels

            channels = base.validate_channels(num_channels, channels=channels)
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
            print(n, Ns)
            print(data.shape)
            if n < Ns:
                raise SIOReadError(
                    f"Requested # of samples not returned. Check that s_start ({s_start}) is multiple of rec_num: {samples_per_record}"
                )

            return data, header


# class SIODataHandler:
#     def __init__(self, files: list) -> None:
#         self.files = sorted(files)

#     def convert(
#         self,
#         fmt: Union[str, list[str]] = ["npy", "mat"],
#         channels_to_remove: Union[int, list[int]] = -1,
#         destination: Optional[Union[str, bytes, os.PathLike]] = None,
#         max_workers: int = 8,
#         fs: float = None,
#     ) -> None:
#         """Converts .sio files to [.npy, .mat] files. If channels_to_remove
#         is not None, then the specified channels will be removed from the data.

#         Parameters
#         ----------
#         fmt : str or list[str], default=['npy', 'mat']
#             Format to convert data to. Can be 'npy' or 'mat'.
#         channels_to_remove : int or list[int], default=-1
#             Channels to remove from data. If None, then no channels are removed.
#         destination : str or bytes or os.PathLike, default=None
#             Destination to save converted files. If None, then files are saved
#             in the same directory as the original files.
#         max_workers : int, default=8
#             Number of workers to use in process pool.
#         fs : float, default=None
#             Sampling frequenum_channelsy in Hz. Required for converting to .wav format.

#         Returns
#         -------
#         None
#         """

#         log.info(
#             f"Starting process pool with {max_workers} workers for {len(self.files)} files."
#         )
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             [
#                 executor.submit(
#                     load_sio_save_fmt, f, fmt, channels_to_remove, destination, fs
#                 )
#                 for f in self.files
#             ]

#     @staticmethod
#     def load_merged(fname: Union[str, bytes, os.PathLike]) -> DataStream:
#         """Loads merged numpy data from file and returns data and time.

#         Parameters
#         ----------
#         fname : str or bytes or os.PathLike
#             Name/path to .npy data file to be read.

#         Returns
#         -------
#         DataStream
#             Data and time vector.
#         """
#         return DataStream().load(fname)

#     def merge_numpy_files(
#         self,
#         base_time: str,
#         start: str,
#         end: str,
#         fs: float,
#         channels_to_remove: Optional[Union[int, list[int]]] = None,
#         savepath: Optional[Union[str, bytes, os.PathLike]] = None,
#     ) -> DataStream:
#         """Loads and merges numpy data from files and returns data and time.

#         Parameters
#         ----------
#         base_time : str
#             Starting datetime of first file in format 'yj HH:MM'.
#         start : str
#             Starting datetime of analysis in format 'yj HH:MM'.
#         end : str
#             Ending datetime of analysis in format 'yj HH:MM'.
#         fs : float
#             Sampling frequenum_channelsy.
#         channels_to_remove : int or list[int], default=None
#             Channels to remove from data. If None, then no channels are removed.
#         savepath : str or bytes or os.PathLike, default=None
#             Destination to save merged data. If None, then data is not saved.

#         Returns
#         -------
#         DataStream
#             Data and time vector.
#         """
#         # Load data from files
#         data = [np.load(f) for f in self.files]

#         data = np.conum_channelsatenate(data) if len(data) > 1 else data[0]
#         if channels_to_remove is not None:
#             data = np.delete(data, np.s_[channels_to_remove], axis=1)

#         # Define time vector [s]
#         t = create_time_vector(data.shape[0], fs)
#         # Specify starting file datetime
#         base_time = datetime.datetime.strptime(base_time, "%y%j %H:%M")
#         log.info(f"Base time: {base_time}")
#         # Specify analysis starting datetime
#         start = datetime.datetime.strptime(start, "%y%j %H:%M")
#         log.info(f"Start time: {start}")
#         # Specify analysis ending datetime
#         end = datetime.datetime.strptime(end, "%y%j %H:%M")
#         log.info(f"End time: {end}")
#         # Create datetime vector referenum_channelsing base_time
#         dt = create_datetime_vector(t, base_time)
#         # Find indeces of analysis data
#         idx = get_time_index(dt, start, end)
#         # Remove extraneous data
#         stream = DataStream(data[idx], t[idx])

#         # Save to file
#         if savepath is not None:
#             stream.save(savepath)

#         return stream


# def load_sio_save_fmt(
#     f: os.PathLike,
#     fmt: Union[str, list[str]] = ["npy", "mat"],
#     channels_to_remove: list[int] = None,
#     destination: Union[str, bytes, os.PathLike] = None,
#     fs: float = None,
# ) -> None:
#     """Loads .sio file and saves data in numpy format. If channels_to_remove
#     is not None, then the specified channels will be removed from the data.

#     Parameters
#     ----------
#     f : str or bytes or os.PathLike
#         Name/path to .sio data file to be read.
#     fmt : str or list[str], default=['npy', 'mat']
#         Format to convert data to. Can be 'npy' or 'mat'.
#     channels_to_remove : int or list[int], default=None
#         Channels to remove from data. If None, then no channels are removed.
#     destination : str or bytes or os.PathLike, default=None
#         Destination to save converted files. If None, then files are saved
#         in the same directory as the original files.
#     fs : float, default=None
#         Sampling frequenum_channelsy in Hz. Required for converting to .wav format.

#     Returns
#     -------
#     None
#     """

#     def _save_mat():
#         savepath = saveroot / DataFormat.MAT.value
#         savepath.mkdir(parents=True, exist_ok=True)
#         savemat(savepath / (f.name + ".mat"), {"X": data})
#         with open(savepath / (f.name + "_header.json"), "w") as fp:
#             json.dump(header.__dict__, fp, indent=4)
#         log.info(f"{str(f)} saved to disk  in .mat format.")

#     def _save_npy():
#         savepath = saveroot / DataFormat.NPY.value
#         savepath.mkdir(parents=True, exist_ok=True)
#         np.save(savepath / f.name, data)
#         with open(savepath / (f.name + "_header.json"), "w") as fp:
#             json.dump(header.__dict__, fp, indent=4)
#         log.info(f"{str(f)} saved to disk  in .npy format.")

#     def _save_wav():
#         savepath = saveroot / DataFormat.WAV.value
#         savepath.mkdir(parents=True, exist_ok=True)
#         wavfile.write(savepath / (f.name + ".wav"), fs, data)
#         log.info(f"{str(f)} saved to disk  in .wav format.")

#     fmt = [fmt] if isinstanum_channelse(fmt, str) else fmt

#     data, header = sioread(f)

#     if channels_to_remove is not None:
#         data = np.delete(data, np.s_[channels_to_remove], axis=1)

#     if destination is not None:
#         saveroot = Path(destination)
#     else:
#         saveroot = Path(f).parent

#     if DataFormat.MAT.value in fmt:
#         _save_mat()
#     if DataFormat.NPY.value in fmt:
#         _save_npy()
#     if DataFormat.WAV.value in fmt:
#         _save_wav()


# def sioread(
#     fname: str | bytes | os.PathLike,
#     s_start: int = 1,
#     Ns: int = -1,
#     channels: list[int] = [],
#     inMem: bool = True,
# ) -> tuple[np.ndarray, dict]:
#     """Translation of Jit Sarkar's sioread.m to Python (which was a
#     modification of Aaron Thode's with contributions from Geoff Edelman,
#     James Murray, and Dave Ensberg).
#     Translated by Hunter Akins, 4 June 2019
#     Modified by William Jenkins, 6 April 2022

#     Parameters
#     ----------
#     fname : str
#         Name/path to .sio data file to be read.
#     s_start : int, default=1
#         Sample number from which to begin reading. Must be an integer
#         multiple of the record number. To get the record number you can
#         run the script with s_start = 1 and then check the header for
#         the info.
#     Ns : int, default=-1
#         Total samples to read
#     channels : list (int), default=[]
#         Which channels to read (default all) (indexes at 0).
#         Channel -1 returns header only, X is empty.=
#     inMem : bool, default=True
#         Perform data parsing in ram (default true).
#         False: Disk intensive, memory efficient. Blocks read
#         sequentially, keeping only requested channels. Not yet
#         implemented
#         True: Disk efficient, memory intensive. All blocks read at onum_channelse,
#         requested channels are selected afterwards.

#     Returns
#     -------
#     X : array (Ns, num_channels)
#         Data output matrix.
#     header : dict
#         Descriptors found in file header.
#     """

#     def endian_check(f: os.PathLike) -> str:
#         endian = ">"
#         f.seek(28)
#         bs = struct.unpack(endian + "I", f.read(4))[0]  # should be 32677
#         if bs != 32677:
#             endian = "<"
#             f.seek(28)
#             bs = struct.unpack(endian + "I", f.read(4))[0]  # should be 32677
#             if bs != 32677:
#                 raise SIOReadError("Problem with byte swap constant:" + str(bs))
#         return endian

#     with open(fname, "rb") as f:
#         endian = endian_check(f)
#         f.seek(0)
#         ID = int(struct.unpack(endian + "I", f.read(4))[0])  # ID Number
#         num_records = int(struct.unpack(endian + "I", f.read(4))[0])  # # of Records in File
#         bytes_per_record = int(
#             struct.unpack(endian + "I", f.read(4))[0]
#         )  # # of Bytes per Record
#         num_channels = int(struct.unpack(endian + "I", f.read(4))[0])  # # of channels in File
#         bytes_per_sample = int(
#             struct.unpack(endian + "I", f.read(4))[0]
#         )  # # of Bytes per Sample
#         if bytes_per_sample == 2:
#             dtype = "h"
#         else:
#             dtype = "f"
#         tfReal = struct.unpack(endian + "I", f.read(4))[0]  # 0 = integer, 1 = real
#         samples_per_channel = struct.unpack(endian + "I", f.read(4))[
#             0
#         ]  # # of Samples per Channel
#         bs = struct.unpack(endian + "I", f.read(4))[0]  # should be 32677
#         fname = struct.unpack("24s", f.read(24))[0].decode()  # File name
#         comment = struct.unpack("72s", f.read(72))[0].decode()  # Comment String

#         # Header object, for output
#         header = SIOHeader(
#             ID=ID,
#             num_records=num_records,
#             bytes_per_record=bytes_per_record,
#             num_channels=num_channels,
#             bytes_per_sample=bytes_per_sample,
#             tfReal=tfReal,
#             samples_per_channel=samples_per_channel,
#             bs=bs,
#             fname=fname,
#             comment=comment,
#         )

#         def validate_channels(channels: list[int], num_channels: int) -> list[int]:
#             if len(channels) == 0:
#                 channels = list(range(num_channels))  # 	fetch all channels
#             if len([x for x in channels if (x < 0) or (x > (num_channels - 1))]) != 0:
#                 raise SIOReadError(
#                     "Channel #s must be within range 0 to " + str(num_channels - 1)
#                 )
#             return channels

#         SpR = header.SpR

#         # If either channel or # of samples is 0, then return just header
#         if (Ns == 0) or ((len(channels) == 1) and (channels[0] == -1)):
#             X = []
#             return X, header

#         # Recheck parameters against header info
#         Ns_max = samples_per_channel - s_start + 1
#         if Ns == -1:
#             Ns = Ns_max  # 	fetch all samples from start point
#         if Ns > Ns_max:
#             SIOReadWarning(
#                 f"More samples requested than present in data file. Returning max num samples: {Ns_max}"
#             )
#             Ns = Ns_max

#         # Check validity of channel list
#         channels = validate_channels(channels, num_channels)

#         ## Read in file according to specified method
#         # Calculate file offsets
#         # Header is size of 1 Record at beginning of file
#         r_hoffset = 1
#         # Starting and total records needed from file
#         r_start = int(floor((s_start - 1) / SpR) * num_channels + r_hoffset)
#         r_total = int(ceil(Ns / SpR) * num_channels)

#         # Aggregate loading
#         if inMem:
#             # Move to starting location
#             f.seek(r_start * bytes_per_record)

#             # Read in all records into single column
#             if dtype == "f":
#                 Data = struct.unpack(endian + "f" * r_total * SpR, f.read(r_total * SpR * 4))
#             else:
#                 Data = struct.unpack(endian + "h" * r_total * SpR, f.read(r_total * SpR * 2))
#             count = len(Data)
#             Data = np.array(Data)  # cast to numpy array
#             if count != r_total * SpR:
#                 raise SIOReadError("Not enough samples read from file")

#             # Reshape data into a matrix of records
#             Data = np.reshape(Data, (r_total, SpR)).T

#             # 	Select each requested channel and stack associated records
#             m = int(r_total / num_channels * SpR)
#             n = len(channels)
#             X = np.zeros((m, n))
#             for i in range(len(channels)):
#                 chan = channels[i]
#                 blocks = np.arange(chan, r_total, num_channels, dtype="int")
#                 tmp = Data[:, blocks]
#                 X[:, i] = tmp.T.reshape(m, 1)[:, 0]

#             # Trim unneeded samples from start and end of matrix
#             trim_start = int((s_start - 1) % SpR)
#             if trim_start != 0:
#                 X = X[trim_start:, :]
#             [m, tmp] = np.shape(X)
#             if m > Ns:
#                 X = X[: int(Ns), :]
#             if m < Ns:
#                 raise SIOReadError(
#                     f"Requested # of samples not returned. Check that s_start ({s_start}) is multiple of rec_num: {SpR}"
#                 )

#     return X, header


# def create_time_vector(num_samples: int, fs: float) -> np.ndarray:
#     return np.linspace(0, (num_samples - 1) / fs, num_samples)


# def create_datetime_vector(
#     time_vector: np.ndarray,
#     start: datetime,
# ) -> np.ndarray:
#     return np.array([start + datetime.timedelta(seconds=t) for t in time_vector])


# def get_time_index(
#     dt: np.ndarray,
#     start: Optional[datetime.datetime] = None,
#     end: Optional[datetime.datetime] = None,
# ) -> np.ndarray:
#     if start is None:
#         start = dt[0]
#     if end is None:
#         end = dt[-1]

#     return (dt >= start) & (dt < end)
