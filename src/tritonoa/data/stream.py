from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import json
import locale
import math
from pathlib import Path
import warnings

import h5py
import numpy as np
import scipy
import scipy.signal as sp

from tritonoa.data.signal import get_filter, pulse_compression
from tritonoa.data.time import (
    TIME_CONVERSION_FACTOR,
    TIME_PRECISION,
    datetime_linspace,
    convert_datetime64_to_iso,
)
from tritonoa.data.util import create_empty_data_chunk, round_away

locale.setlocale(locale.LC_ALL, "")


class NoDataWarning(Warning):
    pass


class DataFormat(Enum):
    """Data format."""

    CSV = "csv"
    MAT = "mat"
    NPY = "npy"
    NPZ = "npz"
    WAV = "wav"


@dataclass
class DataStreamStats:
    channels: int | list[int] | None = None
    time_init: float | np.datetime64 | None = None
    time_end: float | np.datetime64 | None = None
    sampling_rate: float | None = None
    units: str | None = None
    metadata: dict | None = None


class DataStream:
    """Contains acoustic data and data statistics.

    Args:
        stats (DataStreamStats): Data statistics.
        data (Optional[np.ndarray], optional): Acoustic data. Defaults to None.

    Returns:
        DataStream: Data stream object.

    Raises:
        NoDataWarning: If no data is found in the object.
    """

    def __init__(
        self, stats: DataStreamStats | None = None, data: np.ndarray | None = None
    ) -> None:
        self.stats = stats
        self.data = data
        self._post_init()

    def __getitem__(self, index: int | tuple) -> np.ndarray:
        """Returns data and time vector sliced by time index."""
        orig_timevec = self.time_vector
        stats = deepcopy(self.stats)
        if isinstance(index, tuple):
            stats.channels = self.stats.channels[index[0]]
            new_timevec = orig_timevec[index[1]]
            stats.time_init = new_timevec[0]
            stats.time_end = new_timevec[-1]
            data = self.data[index[0], index[1]]
        else:
            # np.atleast_2d preserves behavior of slicing method.
            stats.channels = self.stats.channels[index]
            data = np.atleast_2d(self.data[index])
        return DataStream(stats=stats, data=data)

    def _post_init(self):
        """Initializes data and time vector."""
        # TODO: Need to improve this - does not handle edge cases properly.
        # Set time_init to 0 if not provided
        if self.stats.time_init is None:
            self.stats.time_init = np.timedelta64(0, "us")

        # Compute sampling rate if time_init and time_end are provided
        if self.stats.time_end is not None and self.stats.sampling_rate is None:
            self.stats.sampling_rate = (
                self.stats.time_end - self.stats.time_init
            ) / self.num_samples

        # Set time_end if time_init and sampling rate are provided
        if self.stats.time_end is None and self.stats.sampling_rate is not None:
            self.stats.time_end = self.stats.time_init + np.timedelta64(
                int(
                    TIME_CONVERSION_FACTOR * self.num_samples / self.stats.sampling_rate
                ),
                TIME_PRECISION,
            )

    def __len__(self) -> int:
        """Returns length of data."""
        return self.num_samples

    def __repr__(self) -> str:
        """Returns string representation of the object."""
        return (
            f"DataStream(data={self.data}\n"
            f"channels={self.stats.channels}\n"
            f"num_channels={self.num_channels}\n"
            f"num_samples={self.num_samples}\n"
            f"time_init={self.stats.time_init}\n"
            f"time_end={self.stats.time_end}\n"
            f"sampling_rate={self.stats.sampling_rate}\n"
            f"units={self.stats.units}\n"
            f"metadata={self.stats.metadata})"
        )

    @property
    def num_channels(self) -> int:
        """Returns number of channels in data.

        Returns:
            int: Number of channels.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.data is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        if len(self.data.shape) == 1:
            return 1
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        """Returns number of samples in data.

        Returns:
            int: Number of samples.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.data is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        if len(self.data.shape) == 1:
            return self.data.shape[0]
        return self.data.shape[1]

    @property
    def time_vector(self) -> np.ndarray:
        """Returns time vector.

        Returns:
            np.ndarray: Time vector.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.data is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        return datetime_linspace(
            start=self.stats.time_init, end=self.stats.time_end, num=self.num_samples
        )

    @property
    def seconds(self) -> np.ndarray:
        """Returns time vector in seconds.

        Returns:
            np.ndarray: Time vector in seconds.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.data is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        time_int = self.time_vector.astype("int64") / TIME_CONVERSION_FACTOR
        return time_int - time_int[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Returns shape of data.

        Returns:
            tuple[int, int]: Shape of data.

        Raises:
            NoDataWarning: If no data is found in the object.
        """
        if self.data is None:
            warnings.warn("No data in variable 'X'.", NoDataWarning)
            return None
        return self.data.shape

    def copy(self) -> DataStream:
        return deepcopy(self)

    def slice(
        self,
        starttime: int | float | np.datetime64 | None = None,
        endtime: int | float | np.datetime64 | None = None,
        nearest_sample: bool = True,
    ) -> DataStream:
        """Slices data by time."""
        ds = copy(self)
        ds.stats = deepcopy(self.stats)
        return ds.trim(
            starttime=starttime, endtime=endtime, nearest_sample=nearest_sample
        )

    def write(self, path: Path) -> None:
        """Writes data to file."""
        np.savez(path, stats=self.stats, data=self.data)

    def write_hdf5(self, path: Path) -> None:
        """Writes data to HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=self.data)
            f.create_dataset("time", data=convert_datetime64_to_iso(self.time_vector))
            for key, value in asdict(self.stats).items():
                match value:
                    case np.datetime64():
                        f.attrs[key] = convert_datetime64_to_iso(value)
                    case int() | float() | str() | bool():
                        f.attrs[key] = value
                    case _:
                        f.attrs[key] = json.dumps(value)

    def write_wav(self, path: Path) -> None:
        """Writes data to WAV file."""
        scipy.io.wavfile.write(
            path, int(self.stats.sampling_rate), self.data.astype(np.int32)
        )

    def decimate(
        self,
        factor: int,
        n: int | None = None,
        ftype: str = "iir",
        zero_phase: bool = True,
    ) -> DataStream:
        """Decimates data.

        Args:
            factor (int): Decimation factor.
            n (int, optional): The order of the filter. Defaults to None.
            ftype (str, optional): The type of the filter. Defaults to 'iir'.
            zero_phase (bool, optional): Prevent phase shift by filtering forward
                and backward. Defaults to True.

        Returns:
            DataStream: Decimated data stream.
        """

        self.data = sp.decimate(
            self.data, factor, n=n, ftype=ftype, axis=1, zero_phase=zero_phase
        )
        self.stats.sampling_rate = self.stats.sampling_rate / float(factor)
        return self

    def filter(self, filt_type: str, **kwargs) -> DataStream:
        """Filters data.

        Args:
            filt_type (str): Filter type.
            **kwargs: Additional keyword arguments. Frequency parameters are
                expected for bandpass, bandstop, highpass, and lowpass filters.

        Returns:
            DataStream: Filtered data stream.
        """
        func = get_filter(filt_type)
        self.data = func(data=self.data, fs=self.stats.sampling_rate, **kwargs)
        return self

    def max(self) -> np.ndarray:
        """Returns maximum value of data.

        Returns:
            np.ndarray: Maximum value of data.
        """
        _max = np.atleast_1d(self.data.max(axis=1))
        _min = np.atleast_1d(self.data.min(axis=1))

        for i in range(self.num_channels):
            if abs(_max[i]) < abs(_min[i]):
                _max[i] = _min[i]

        return _max.squeeze()

    def pulse_compression(
        self,
        transmitted_signal: np.ndarray,
        mode: str = "same",
    ) -> DataStream:
        """Trims data by time.

        NOTE: This function and its derivatives modify the object in place.
        received_signal = self.data

        Args:
            transmitted_signal (np.ndarray): The pulse to use for compression.
        Returns:
            np.ndarray: The pulse-compressed data.
        """
        self.data = pulse_compression(transmitted_signal, self.data, mode=mode)
        return self

    def trim(
        self,
        starttime: int | float | np.datetime64 | None = None,
        endtime: int | float | np.datetime64 | None = None,
        pad: bool = False,
        nearest_sample: bool = True,
        fill_value=None,
    ) -> DataStream:
        """Trims data by time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            starttime (Union[int, float, np.datetime64]): Start time.
            endtime (Union[int, float, np.datetime64]): End time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            ValueError: If starttime is greater than endtime.
        """
        if starttime is not None and endtime is not None and starttime > endtime:
            raise ValueError("starttime must be less than endtime.")
        if starttime:
            self._ltrim(starttime, pad, nearest_sample, fill_value)
        if endtime:
            self._rtrim(endtime, pad, nearest_sample, fill_value)
        return self

    def _ltrim(
        self,
        starttime: int | float | np.datetime64,
        pad=False,
        nearest_sample=True,
        fill_value=None,
    ) -> DataStream:
        """Trims all data of this object's data to given start time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            starttime (Union[int, float, np.datetime64]): Start time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            TypeError: If starttime is not of type float, int, or np.datetime64.
            Exception: If time offset between starttime and time_init is too large.
        """
        dtype = self.data.dtype

        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = self.stats.time_init + np.timedelta64(
                int(starttime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )
        if not isinstance(starttime, np.datetime64):
            raise TypeError("starttime must be of type float, int, or np.datetime64.")

        if nearest_sample:
            delta = round_away(
                (starttime - self.stats.time_init)
                / np.timedelta64(1, "s")
                * self.stats.sampling_rate
            )
            if delta < 0 and pad:
                npts = abs(delta) + 10
                newstarttime = self.stats.time_init - np.timedelta64(
                    int(npts / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                    TIME_PRECISION,
                )
                newdelta = round_away(
                    (starttime - newstarttime)
                    / np.timedelta64(1, "s")
                    * self.stats.sampling_rate
                )
                delta = newdelta - npts
        else:
            delta = (
                int(
                    math.floor(
                        round(
                            (self.stats.time_init - starttime)
                            / np.timedelta64(1, "s")
                            * self.stats.sampling_rate,
                            7,
                        )
                    )
                )
                * -1
            )

        if delta > 0 or pad:
            self.stats.time_init += np.timedelta64(
                int(delta / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                TIME_PRECISION,
            )
        if delta == 0 or (delta < 0 and not pad):
            return self
        if delta < 0 and pad:
            try:
                gap = create_empty_data_chunk(abs(delta), self.data.dtype, fill_value)
            except ValueError:
                raise Exception(
                    "Time offset between starttime and time_init too large."
                )
            self.data = np.ma.concatenate([gap, self.data], axis=1)
            return self
        if starttime > self.stats.time_end:
            self.data = np.empty(0, dtype=dtype)
            return self
        if delta > 0:
            try:
                self.data = self.data[:, delta:]
            except IndexError:
                self.data = np.empty(0, dtype=dtype)
        return self

    def _rtrim(
        self,
        endtime: int | float | np.datetime64,
        pad=False,
        nearest_sample=True,
        fill_value=None,
    ) -> DataStream:
        """Trims all data of this object's data to given end time.

        NOTE: This function and its derivatives modify the object in place.

        This function is adapted from the ObsPy library:
        https://docs.obspy.org/index.html

        Args:
            endtime (Union[int, float, np.datetime64]): End time.
            pad (bool): If True, pads data with fill_value.
            nearest_sample (bool): If True, trims to nearest sample.
            fill_value: Fill value for padding.

        Returns:
            DataStream: Trimmed data stream.

        Raises:
            TypeError: If endtime is not of type float, int, or np.datetime64.
            Exception: If time offset between endtime and time_start is too large.
        """
        dtype = self.data.dtype

        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = self.stats.time_end - np.timedelta64(
                int(endtime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )
        if not isinstance(endtime, np.datetime64):
            raise TypeError("`endtime` must be of type float, int, or np.datetime64.")

        if nearest_sample:
            delta = round_away(
                (endtime - self.stats.time_init)
                / np.timedelta64(1, "s")
                * self.stats.sampling_rate
                - self.num_samples
                + 1
            )
        else:
            delta = (
                int(
                    math.floor(
                        round(
                            (endtime - self.stats.time_end)
                            / np.timedelta64(1, "s")
                            * self.stats.sampling_rate,
                            7,
                        )
                    )
                )
                * -1
            )

        if delta == 0 or (delta > 0 and not pad):
            return self
        if delta > 0 and pad:
            try:
                gap = create_empty_data_chunk(delta, self.data.dtype, fill_value)
            except ValueError:
                raise Exception(
                    "Time offset between starttime and time_start too large."
                )
            self.data = np.ma.concatenate([self.data, gap], axis=0)
            return self
        if endtime < self.stats.time_init:
            self.stats.time_init = self.stats.time_end + np.timedelta64(
                int(delta / self.stats.sampling_rate * TIME_CONVERSION_FACTOR),
                TIME_PRECISION,
            )
            self.data = np.empty(0, dtype=dtype)
            return self
        delta = abs(delta)
        total = self.num_samples - delta
        if endtime == self.stats.time_init:
            total = 1
        self.data = self.data[:, :total]
        self.stats.time_end = self.stats.time_init + np.timedelta64(
            int(TIME_CONVERSION_FACTOR * self.num_samples / self.stats.sampling_rate),
            TIME_PRECISION,
        )
        return self
