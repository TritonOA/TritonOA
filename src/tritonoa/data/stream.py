from __future__ import annotations

from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import json
import locale
import math
from pathlib import Path
import warnings

from h5py import File, Group
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


class TaperLengthWarning(Warning):
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

    def copy(self) -> DataStream:
        return deepcopy(self)

    def create_hdf5_dataset(self, f: File | Group) -> None:
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

    def detrend(
        self, axis: int = -1, type: str = "linear", bp: int | Sequence[int] = 0
    ) -> DataStream:
        """Detrend data.

        Args:
            axis (int, optional): Axis along which to detrend. Defaults to -1.
            type (str, optional): Type of detrending. Defaults to 'linear'.
            bp (int | Sequence[int], optional): Breakpoints for detrending. Defaults to 0.

        Returns:
            DataStream: Detrended data stream.
        """
        sp.detrend(self.data, axis=axis, type=type, bp=bp, overwrite_data=True)
        return self

    def filter(
        self, filt_type: str, freq: float | Sequence[float], **kwargs
    ) -> DataStream:
        """Filters data.

        Args:
            filt_type (str): Filter type.
            **kwargs: Additional keyword arguments. Frequency parameters are
                expected for bandpass, bandstop, highpass, and lowpass filters.

        Returns:
            DataStream: Filtered data stream.
        """
        func = get_filter(filt_type)
        self.data = func(self.data, freq, self.stats.sampling_rate, **kwargs)
        return self

    def hilbert(self) -> DataStream:
        """Applies Hilbert transform to data.

        Returns:
            DataStream: Data stream with Hilbert transform applied.
        """
        self.data = sp.hilbert(self.data)
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

    def resample(self, num: int, **kwargs) -> DataStream:
        """Resamples data to a new number of samples.

        Returns:
            DataStream: Resampled data stream.
        """
        self.data = sp.resample(self.data, num, axis=1, **kwargs)
        self.stats.sampling_rate = (
            self.stats.sampling_rate * num / self.num_samples
            if self.num_samples > 0
            else None
        )
        return self

    def resample_poly(self, up: int, down: int, **kwargs) -> DataStream:
        """Resamples data using polyphase filtering.

        Returns:
            DataStream: Resampled data stream.
        """
        self.data = sp.resample_poly(self.data, up, down, axis=1, **kwargs)
        self.stats.sampling_rate = self.stats.sampling_rate * up / down
        return self

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

    def taper(
        self,
        max_percentage: float = 0.05,
        max_length: float | None = None,
        window_type: str = "hann",
        side: str = "both",
        **kwargs,
    ) -> DataStream:
        # TODO: Refactor into `signal.util` module.
        if side not in ["left", "right", "both"]:
            raise ValueError("side must be 'left', 'right', or 'both'.")
        npts = self.num_samples
        max_half_lengths = []
        if max_percentage is not None:
            max_half_lengths.append(int(max_percentage * npts))
        if max_length is not None:
            max_half_lengths.append(int(max_length * self.stats.sampling_rate))
        if np.all([2 * mhl > npts for mhl in max_half_lengths]):
            warnings.warn(
                (
                    "The requested taper is longer than the trace. "
                    "The taper will be shortened to trace length."
                ),
                TaperLengthWarning,
            )
        max_half_lengths.append(int(npts / 2))
        window_length = min(max_half_lengths)

        if window_type == "cosine":
            kwargs["p"] = 1.0
        if 2 * window_length == npts:
            taper_sides = sp.get_window((window_type, *kwargs), 2 * window_length)
        else:
            taper_sides = sp.get_window((window_type, *kwargs), 2 * window_length + 1)

        if side == "left":
            taper = np.hstack(
                (taper_sides[:window_length], np.ones(npts - window_length))
            )
        elif side == "right":
            taper = np.hstack(
                (
                    np.ones(npts - window_length),
                    taper_sides[len(taper_sides) - window_length :],
                )
            )
        else:
            taper = np.hstack(
                (
                    taper_sides[:window_length],
                    np.ones(npts - 2 * window_length),
                    taper_sides[len(taper_sides) - window_length :],
                )
            )
        self.data = self.data * taper
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

    def write(self, path: Path) -> None:
        """Writes data to file."""
        np.savez(path, stats=self.stats, data=self.data)

    def write_hdf5(self, path: Path) -> None:
        """Writes data to HDF5 file."""
        with File(path, "w") as f:
            self.create_hdf5_dataset(f)

    def write_wav(self, path: Path) -> None:
        """Writes data to WAV file."""
        scipy.io.wavfile.write(path, int(self.stats.sampling_rate), self.data.T)

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
            if isinstance(self.stats.time_init, float):
                self.stats.time_init = np.timedelta64(
                    int(self.stats.time_init * TIME_CONVERSION_FACTOR), TIME_PRECISION
                )
            self.stats.time_end = self.stats.time_init + np.timedelta64(
                int(
                    TIME_CONVERSION_FACTOR * self.num_samples / self.stats.sampling_rate
                ),
                TIME_PRECISION,
            )

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


def pipeline(
    ds: DataStream,
    detrend: bool = True,
    taper_pc: float | None = None,
    taper_duration: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    detrend_kwargs: dict = {},
) -> DataStream:
    """Applies a signal processing pipeline to a DataStream object.

    Args:
        ds (DataStream): The DataStream object to process.
        detrend (bool, optional): Whether to detrend the data. Defaults to True.
        taper_pc (float | None, optional): Percentage of taper to apply. Defaults to None.
        taper_duration (float | None, optional): Duration of taper in seconds. Defaults to None.
        dec_factor (int | None, optional): Decimation factor. Defaults to None.
        filt_type (str | None, optional): Type of filter to apply. Defaults to None.
        filt_freq (float | Sequence[float] | None, optional): Frequency or frequencies
            for filtering. Defaults to None.
        detrend_kwargs (dict, optional): Additional keyword arguments for detrending.

    Returns:
        DataStream: The processed DataStream object.
    """
    if taper_pc and taper_duration:
        raise ValueError("Only one of taper_pc or taper_duration can be specified.")

    if detrend:
        ds.detrend(**detrend_kwargs)
    if taper_pc:
        ds.taper(max_percentage=taper_pc)
    if taper_duration:
        ds.taper(max_length=taper_duration * ds.stats.sampling_rate)
    if dec_factor:
        ds.decimate(dec_factor)
    if filt_type and filt_freq:
        ds.filter(filt_type, filt_freq)
    return ds
