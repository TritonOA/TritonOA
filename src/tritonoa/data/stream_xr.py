"""Acoustic data handling using xarray with custom accessor.

This module provides an xarray accessor-based approach for working with multi-channel
acoustic data. Instead of wrapping xarray.DataArray in a custom class, we extend it
using the @xr.register_dataarray_accessor decorator, which preserves compatibility
with the xarray ecosystem while adding acoustic-specific functionality.

Interface Specification:
-----------------------
Acoustic DataArrays must conform to this structure:

1. **Dimensions**: ('channel', 'time') - in that order
2. **Coordinates**:
   - channel: Channel identifiers (array of integers)
   - time: Timestamps as np.datetime64[us]
3. **Attributes** (stored in .attrs):
   - sampling_rate: float (Hz) - optional (defaults to 1.0)
   - units: str (e.g., 'Pa', 'V') - optional
   - metadata: dict - optional additional metadata

Usage:
------
    # Create properly structured acoustic data
    from tritonoa.data.stream_xr import create_acoustic_dataarray

    data = create_acoustic_dataarray(
        np.random.randn(2, 1000),
        sampling_rate=48000,
        units='Pa'
    )

    # Access metadata via accessor properties
    data.acoustic.sampling_rate  # 48000
    data.acoustic.num_channels   # 2

    # Or directly via attrs
    data.attrs['sampling_rate']  # 48000

    # Use acoustic-specific methods
    filtered = data.acoustic.filter('bandpass', [100, 1000])

    # Mix with native xarray operations
    subset = filtered.sel(channel=[0])
    subset.acoustic.decimate(4)  # Accessor preserved!

    # Validate existing DataArrays
    from tritonoa.data.stream_xr import validate_acoustic_data

    validate_acoustic_data(data)  # Raises ValidationError if invalid

Type Checking:
--------------
Use AcousticMetadata and AcousticDataArray for type hints:

    from tritonoa.data.stream_xr import AcousticMetadata, AcousticDataArray

    def process(data: xr.DataArray) -> xr.DataArray:
        # Runtime check
        if not isinstance(data, AcousticDataArray):
            raise TypeError("Expected acoustic data")
        ...
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import json
import locale
from typing import Protocol, TypedDict, runtime_checkable
import warnings
from pathlib import Path

from h5py import File, Group
import numpy as np
import scipy
import scipy.signal as sp
import xarray as xr

from tritonoa.data.signal import get_filter, pulse_compression
from tritonoa.data.time import (
    TIME_CONVERSION_FACTOR,
    TIME_PRECISION,
    datetime_linspace,
    convert_datetime64_to_iso,
)

locale.setlocale(locale.LC_ALL, "")


class NoDataWarning(Warning):
    pass


class TaperLengthWarning(Warning):
    pass


class ValidationError(Exception):
    """Raised when acoustic data validation fails."""

    pass


class DataFormat(Enum):
    """Data format."""

    CSV = "csv"
    MAT = "mat"
    NPY = "npy"
    NPZ = "npz"
    WAV = "wav"


# ============================================================================
# Type definitions for acoustic data interface
# ============================================================================


class AcousticMetadata(TypedDict, total=False):
    """Type definition for acoustic metadata stored in DataArray.attrs.

    Optional fields (with defaults):
        sampling_rate: Sampling rate in Hz (defaults to 1.0)
        units: Physical units of the data (e.g., 'Pa', 'V')
        metadata: Additional metadata dictionary
    """

    sampling_rate: float
    units: str
    metadata: dict


@runtime_checkable
class AcousticDataArray(Protocol):
    """Protocol defining the interface for acoustic DataArrays.

    An acoustic DataArray must have:
    - Dimensions: ('channel', 'time') in that order
    - Coordinates:
        - channel: Channel identifiers (array of integers)
        - time: Timestamps as np.datetime64[us]
    - Attributes:
        - sampling_rate: float (defaults to 1.0)
        - units: str (optional)
        - metadata: dict (optional)

    This protocol can be used for type checking with isinstance():
        >>> if isinstance(data, AcousticDataArray):
        ...     # data has the required structure
    """

    dims: tuple[str, ...]
    coords: dict
    attrs: dict
    values: np.ndarray

    def sel(self, **kwargs) -> xr.DataArray: ...
    def isel(self, **kwargs) -> xr.DataArray: ...


def validate_acoustic_data(
    data: xr.DataArray, strict: bool = True, raise_errors: bool = True
) -> bool:
    """Validate that a DataArray conforms to acoustic data interface.

    Args:
        data: DataArray to validate.
        strict: If True, require sampling_rate attribute. If False, warn instead.
        raise_errors: If True, raise ValidationError on failure. If False, return False.

    Returns:
        bool: True if valid.

    Raises:
        ValidationError: If validation fails and raise_errors=True.

    Example:
        >>> data = xr.DataArray(np.random.randn(2, 1000), dims=['channel', 'time'])
        >>> validate_acoustic_data(data)  # Raises ValidationError
        >>> validate_acoustic_data(data, raise_errors=False)  # Returns False
    """
    errors = []

    # Check dimensions exist and are in correct order
    if not hasattr(data, "dims"):
        errors.append("DataArray missing 'dims' attribute")
    elif data.dims != ("channel", "time"):
        errors.append(
            f"Expected dimensions ('channel', 'time'), got {data.dims}. "
            f"Use data.transpose('channel', 'time') to reorder."
        )

    # Check coordinates
    if not hasattr(data, "coords"):
        errors.append("DataArray missing 'coords' attribute")
    else:
        if "channel" not in data.coords:
            errors.append("Missing 'channel' coordinate")
        if "time" not in data.coords:
            errors.append("Missing 'time' coordinate")
        elif not np.issubdtype(data.time.dtype, np.datetime64):
            errors.append(f"Time coordinate must be datetime64, got {data.time.dtype}")

    # Check attributes
    if not hasattr(data, "attrs"):
        errors.append("DataArray missing 'attrs' attribute")
    else:
        if "sampling_rate" not in data.attrs:
            warnings.warn(
                "Missing 'sampling_rate' attribute, will default to 1.0 Hz",
                UserWarning,
            )

    # Handle errors
    if errors:
        if raise_errors:
            raise ValidationError(
                "Acoustic data validation failed:\n  - " + "\n  - ".join(errors)
            )
        return False

    return True


@xr.register_dataarray_accessor("acoustic")
class AcousticAccessor:
    """Accessor for acoustic data operations on xarray DataArrays.

    This accessor extends xarray.DataArray with acoustic-specific methods.
    Access via: data.acoustic.method_name()

    The DataArray should have:
    - Dimensions: (channel, time)
    - Coordinates:
        - channel: Channel identifiers (integers or list)
        - time: np.datetime64 timestamps for each sample
    - Attributes: sampling_rate, units, metadata

    Example:
        >>> data = create_acoustic_dataarray(np.random.randn(2, 1000), sampling_rate=48000)
        >>> data.acoustic.sampling_rate
        48000
        >>> filtered = data.acoustic.bandpass_filter('bandpass', [100, 1000])
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._validate()

    def _validate(self, strict: bool = False):
        """Validate that the DataArray has required structure.

        Args:
            strict: If True, require sampling_rate attribute. If False, warn instead.

        Raises:
            ValidationError: If validation fails.
        """
        validate_acoustic_data(self._obj, strict=strict, raise_errors=True)

    # ============================================================================
    # Properties - convenient access to metadata and coordinates
    # ============================================================================

    @property
    def sampling_rate(self) -> float:
        """Returns sampling rate in Hz (defaults to 1.0 if not set)."""
        return self._obj.attrs.get("sampling_rate", 1.0)

    @property
    def units(self) -> str:
        """Returns units of the data."""
        return self._obj.attrs.get("units", None)

    @property
    def metadata(self) -> dict:
        """Returns metadata dictionary."""
        return self._obj.attrs.get("metadata", None)

    @property
    def channels(self) -> list[int]:
        """Returns list of channel identifiers."""
        return self._obj.channel.values.tolist()

    @property
    def num_channels(self) -> int:
        """Returns number of channels."""
        return len(self._obj.channel)

    @property
    def num_samples(self) -> int:
        """Returns number of samples."""
        return len(self._obj.time)

    @property
    def time_vector(self) -> np.ndarray:
        """Returns time vector as np.datetime64 array."""
        return self._obj.time.values

    @property
    def time_init(self) -> np.datetime64:
        """Returns initial time."""
        return self._obj.time.values[0]

    @property
    def time_end(self) -> np.datetime64:
        """Returns end time."""
        return self._obj.time.values[-1]

    @property
    def seconds(self) -> np.ndarray:
        """Returns time vector in seconds relative to start."""
        time_int = self._obj.time.values.astype("int64") / TIME_CONVERSION_FACTOR
        return time_int - time_int[0]

    # ============================================================================
    # Signal processing methods - return new DataArrays
    # ============================================================================

    def decimate(
        self,
        factor: int,
        n: int | None = None,
        ftype: str = "iir",
        zero_phase: bool = True,
    ) -> xr.DataArray:
        """Decimates data.

        Args:
            factor: Decimation factor.
            n: The order of the filter. Defaults to None.
            ftype: The type of the filter. Defaults to 'iir'.
            zero_phase: Prevent phase shift by filtering forward and backward. Defaults to True.

        Returns:
            xr.DataArray: Decimated data array with updated sampling rate and time coordinates.
        """
        decimated = sp.decimate(
            self._obj.values, factor, n=n, ftype=ftype, axis=1, zero_phase=zero_phase
        )

        # Update sampling rate
        new_sr = self.sampling_rate / float(factor)

        # Decimate time coordinate
        new_time = self._obj.time.values[::factor]

        # Create new DataArray with decimated data
        return xr.DataArray(
            decimated,
            coords={"channel": self._obj.channel, "time": new_time},
            dims=["channel", "time"],
            attrs={**self._obj.attrs, "sampling_rate": new_sr},
        )

    def detrend(
        self, axis: int = -1, type: str = "linear", bp: int | Sequence[int] = 0
    ) -> xr.DataArray:
        """Detrend data.

        Args:
            axis: Axis along which to detrend. Defaults to -1.
            type: Type of detrending. Defaults to 'linear'.
            bp: Breakpoints for detrending. Defaults to 0.

        Returns:
            xr.DataArray: Detrended data array.
        """
        detrended = sp.detrend(self._obj.values, axis=axis, type=type, bp=bp)
        return self._obj.copy(data=detrended)

    def filter(
        self, filt_type: str, freq: float | Sequence[float], **kwargs
    ) -> xr.DataArray:
        """Filters data.

        Args:
            filt_type: Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop').
            freq: Frequency or frequencies for filtering.
            **kwargs: Additional keyword arguments.

        Returns:
            xr.DataArray: Filtered data array.
        """
        func = get_filter(filt_type)
        filtered = func(self._obj.values, freq, self.sampling_rate, **kwargs)
        return self._obj.copy(data=filtered)

    def hilbert(self) -> xr.DataArray:
        """Applies Hilbert transform to data.

        Returns:
            xr.DataArray: Data array with Hilbert transform applied.
        """
        transformed = sp.hilbert(self._obj.values)
        return self._obj.copy(data=transformed)

    def max(self) -> np.ndarray:
        """Returns maximum absolute value of data per channel.

        Returns:
            np.ndarray: Maximum absolute value per channel.
        """
        _max = self._obj.max(dim="time").values
        _min = self._obj.min(dim="time").values

        for i in range(self.num_channels):
            if abs(_max[i]) < abs(_min[i]):
                _max[i] = _min[i]

        return _max.squeeze()

    def pulse_compression(
        self,
        transmitted_signal: np.ndarray,
        mode: str = "same",
    ) -> xr.DataArray:
        """Apply pulse compression to data.

        Args:
            transmitted_signal: The pulse to use for compression.
            mode: Convolution mode ('same', 'full', 'valid').

        Returns:
            xr.DataArray: Pulse-compressed data array.
        """
        compressed = pulse_compression(transmitted_signal, self._obj.values, mode=mode)

        if mode == "same":
            return self._obj.copy(data=compressed)
        else:
            # Need to update time coordinate if size changes
            new_time = datetime_linspace(
                start=self.time_init,
                end=self.time_end,
                num=compressed.shape[1],
            )
            return xr.DataArray(
                compressed,
                coords={"channel": self._obj.channel, "time": new_time},
                dims=["channel", "time"],
                attrs=self._obj.attrs,
            )

    def resample(self, num: int, **kwargs) -> xr.DataArray:
        """Resamples data to a new number of samples.

        Args:
            num: New number of samples.
            **kwargs: Additional keyword arguments for scipy.signal.resample.

        Returns:
            xr.DataArray: Resampled data array with updated sampling rate.
        """
        resampled = sp.resample(self._obj.values, num, axis=1, **kwargs)

        # Update sampling rate
        new_sr = (
            self.sampling_rate * num / self.num_samples
            if self.num_samples > 0
            else None
        )

        # Generate new time coordinate
        new_time = datetime_linspace(start=self.time_init, end=self.time_end, num=num)

        return xr.DataArray(
            resampled,
            coords={"channel": self._obj.channel, "time": new_time},
            dims=["channel", "time"],
            attrs={**self._obj.attrs, "sampling_rate": new_sr},
        )

    def resample_poly(self, up: int, down: int, **kwargs) -> xr.DataArray:
        """Resamples data using polyphase filtering.

        Args:
            up: Upsampling factor.
            down: Downsampling factor.
            **kwargs: Additional keyword arguments for scipy.signal.resample_poly.

        Returns:
            xr.DataArray: Resampled data array with updated sampling rate.
        """
        resampled = sp.resample_poly(self._obj.values, up, down, axis=1, **kwargs)

        # Update sampling rate
        new_sr = self.sampling_rate * up / down

        # Calculate new number of samples
        num = resampled.shape[1]

        # Generate new time coordinate
        new_time = datetime_linspace(start=self.time_init, end=self.time_end, num=num)

        return xr.DataArray(
            resampled,
            coords={"channel": self._obj.channel, "time": new_time},
            dims=["channel", "time"],
            attrs={**self._obj.attrs, "sampling_rate": new_sr},
        )

    def taper(
        self,
        max_percentage: float = 0.05,
        max_length: float | None = None,
        window_type: str = "hann",
        side: str = "both",
        **kwargs,
    ) -> xr.DataArray:
        """Apply a taper to the data.

        Args:
            max_percentage: Maximum percentage of data to taper.
            max_length: Maximum length of taper in seconds.
            window_type: Type of window ('hann', 'hamming', 'cosine', etc.).
            side: Which side to taper ('left', 'right', 'both').
            **kwargs: Additional keyword arguments for window function.

        Returns:
            xr.DataArray: Tapered data array.
        """
        if side not in ["left", "right", "both"]:
            raise ValueError("side must be 'left', 'right', or 'both'.")

        npts = self.num_samples
        max_half_lengths = []

        if max_percentage is not None:
            max_half_lengths.append(int(max_percentage * npts))
        if max_length is not None:
            max_half_lengths.append(int(max_length * self.sampling_rate))
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
            taper_sides = sp.get_window(
                (window_type, *kwargs.values()), 2 * window_length
            )
        else:
            taper_sides = sp.get_window(
                (window_type, *kwargs.values()), 2 * window_length + 1
            )

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

        return self._obj.copy(data=self._obj.values * taper)

    def trim(
        self,
        starttime: int | float | np.datetime64 | None = None,
        endtime: int | float | np.datetime64 | None = None,
        nearest_sample: bool = True,
    ) -> xr.DataArray:
        """Trims data by time.

        Args:
            starttime: Start time (seconds from beginning, or np.datetime64).
            endtime: End time (seconds from beginning, or np.datetime64).
            nearest_sample: If True, trims to nearest sample.

        Returns:
            xr.DataArray: Trimmed data array.

        Raises:
            ValueError: If starttime is greater than endtime.
        """
        if starttime is not None and endtime is not None and starttime > endtime:
            raise ValueError("starttime must be less than endtime.")

        # Convert float/int times to np.datetime64
        if isinstance(starttime, (float, int)):
            starttime = self.time_init + np.timedelta64(
                int(starttime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )
        if isinstance(endtime, (float, int)):
            endtime = self.time_init + np.timedelta64(
                int(endtime * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )

        # Use xarray's selection capabilities for trimming
        if starttime is not None or endtime is not None:
            time_slice = slice(starttime, endtime)
            method = "nearest" if nearest_sample else None

            try:
                return self._obj.sel(time=time_slice, method=method)
            except (KeyError, ValueError):
                # Handle cases where selection is out of bounds
                # Create empty array
                return xr.DataArray(
                    np.empty((self.num_channels, 0), dtype=self._obj.dtype),
                    coords={
                        "channel": self._obj.channel,
                        "time": np.array([], dtype="datetime64[us]"),
                    },
                    dims=["channel", "time"],
                    attrs=self._obj.attrs,
                )

        return self._obj

    # ============================================================================
    # I/O methods
    # ============================================================================

    def to_hdf5(self, f: File | Group) -> None:
        """Write data to HDF5 file/group.

        Args:
            f: HDF5 file or group object.
        """
        f.create_dataset("data", data=self._obj.values)
        f.create_dataset("channels", data=self._obj.channel.values)
        f.create_dataset("time", data=self._obj.time.values.astype("int64"))

        # Store attributes
        for key, value in self._obj.attrs.items():
            match value:
                case np.datetime64():
                    f.attrs[key] = convert_datetime64_to_iso(value)
                case int() | float() | str() | bool():
                    f.attrs[key] = value
                case None:
                    pass  # Skip None values
                case _:
                    f.attrs[key] = json.dumps(value)

    def to_npz(self, path: Path) -> None:
        """Writes data to npz file.

        Args:
            path: Path to save npz file.
        """
        np.savez(
            path,
            data=self._obj.values,
            channels=self._obj.channel.values,
            time=self._obj.time.values,
            sampling_rate=self.sampling_rate,
            units=self.units,
            metadata=self.metadata,
        )

    def to_wav(self, path: Path) -> None:
        """Writes data to WAV file.

        Args:
            path: Path to save WAV file.
        """
        scipy.io.wavfile.write(path, int(self.sampling_rate), self._obj.values.T)


# ============================================================================
# Factory functions for creating acoustic DataArrays
# ============================================================================


def create_acoustic_dataarray(
    data: np.ndarray,
    channels: int | list[int] | None = None,
    time: np.ndarray | None = None,
    time_init: float | np.datetime64 | None = None,
    time_end: float | np.datetime64 | None = None,
    sampling_rate: float = 1.0,
    units: str | None = None,
    metadata: dict | None = None,
    validate: bool = True,
) -> xr.DataArray:
    """Creates an xarray DataArray with acoustic data structure.

    This factory function ensures the DataArray conforms to the acoustic data interface
    with proper dimensions, coordinates, and attributes.

    Args:
        data: Numpy array of acoustic data (channel, time).
        channels: Channel identifiers. If None, uses range(num_channels).
        time: Time vector as np.datetime64. If None, generated from time_init/time_end.
        time_init: Initial time. Defaults to 0.
        time_end: End time. Computed from sampling_rate if not provided.
        sampling_rate: Sampling rate in Hz (defaults to 1.0).
        units: Units of the data (e.g., 'Pa', 'V').
        metadata: Additional metadata dictionary.
        validate: If True, validate the created DataArray structure (default: True).

    Returns:
        xr.DataArray: DataArray with proper coordinates and attributes for acoustic data.

    Raises:
        ValidationError: If validate=True and the DataArray structure is invalid.

    Example:
        >>> # Create acoustic data
        >>> data = create_acoustic_dataarray(
        ...     np.random.randn(2, 1000),
        ...     sampling_rate=48000,
        ...     units='Pa',
        ...     metadata={'instrument': 'hydrophone'}
        ... )
        >>> data.acoustic.sampling_rate
        48000
        >>>
        >>> # sampling_rate is optional and defaults to 1.0
        >>> data = create_acoustic_dataarray(np.random.randn(2, 1000))
        >>> data.acoustic.sampling_rate
        1.0
    """
    # Ensure data is at least 2D
    if data.ndim == 1:
        data = np.atleast_2d(data)

    num_channels, num_samples = data.shape

    # Set up channel coordinate
    if channels is None:
        channels = list(range(num_channels))
    elif isinstance(channels, int):
        channels = list(range(channels))
    elif not isinstance(channels, list):
        channels = list(channels)

    # Set up time coordinate
    if time is not None:
        time_coord = time
    else:
        # Generate time coordinate from time_init, time_end, sampling_rate
        if time_init is None:
            time_init = np.datetime64(0, "us")
        elif isinstance(time_init, (float, int)):
            time_init = np.datetime64(0, "us") + np.timedelta64(
                int(time_init * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )

        if time_end is None and sampling_rate is not None:
            time_end = time_init + np.timedelta64(
                int(TIME_CONVERSION_FACTOR * num_samples / sampling_rate),
                TIME_PRECISION,
            )
        elif time_end is not None and sampling_rate is None:
            # Compute sampling rate from time range
            delta = (time_end - time_init) / np.timedelta64(1, "s")
            sampling_rate = (num_samples - 1) / delta if num_samples > 1 else 1.0

        if isinstance(time_end, (float, int)):
            time_end = time_init + np.timedelta64(
                int(time_end * TIME_CONVERSION_FACTOR), TIME_PRECISION
            )

        time_coord = datetime_linspace(start=time_init, end=time_end, num=num_samples)

    # Create DataArray
    data_array = xr.DataArray(
        data,
        coords={"channel": channels, "time": time_coord},
        dims=["channel", "time"],
        attrs={
            "sampling_rate": sampling_rate,
            "units": units,
            "metadata": metadata,
        },
    )

    # Validate structure if requested
    if validate:
        validate_acoustic_data(data_array, strict=True, raise_errors=True)

    return data_array


# ============================================================================
# Pipeline function
# ============================================================================


def pipeline(
    data: xr.DataArray,
    detrend: bool = True,
    taper_pc: float | None = None,
    taper_duration: float | None = None,
    dec_factor: int | None = None,
    filt_type: str | None = None,
    filt_freq: float | Sequence[float] | None = None,
    detrend_kwargs: dict = {},
) -> xr.DataArray:
    """Applies a signal processing pipeline to acoustic data.

    Args:
        data: The acoustic DataArray to process.
        detrend: Whether to detrend the data. Defaults to True.
        taper_pc: Percentage of taper to apply. Defaults to None.
        taper_duration: Duration of taper in seconds. Defaults to None.
        dec_factor: Decimation factor. Defaults to None.
        filt_type: Type of filter to apply. Defaults to None.
        filt_freq: Frequency or frequencies for filtering. Defaults to None.
        detrend_kwargs: Additional keyword arguments for detrending.

    Returns:
        xr.DataArray: The processed acoustic DataArray.

    Example:
        >>> data = create_acoustic_dataarray(np.random.randn(2, 1000), sampling_rate=48000)
        >>> processed = pipeline(data, detrend=True, filt_type='bandpass', filt_freq=[100, 1000])
    """
    if taper_pc and taper_duration:
        raise ValueError("Only one of taper_pc or taper_duration can be specified.")

    if detrend:
        data = data.acoustic.detrend(**detrend_kwargs)
    if taper_pc:
        data = data.acoustic.taper(max_percentage=taper_pc)
    if taper_duration:
        data = data.acoustic.taper(max_length=taper_duration)
    if dec_factor:
        data = data.acoustic.decimate(dec_factor)
    if filt_type and filt_freq:
        data = data.acoustic.filter(filt_type, filt_freq)
    return data
