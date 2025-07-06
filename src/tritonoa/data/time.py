from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional

import numpy as np
import polars as pl

TIME_PRECISION = "us"
TIME_CONVERSION_FACTOR = 1e6


@dataclass
class ClockParameters:
    time_check_0: Optional[np.datetime64] = np.datetime64("NaT")
    time_check_1: Optional[np.datetime64] = np.datetime64("NaT")
    offset_0: float = 0.0
    offset_1: float = 0.0

    @property
    def drift_rate(self) -> float:
        """Computes clock drift rate in s/day.

        Returns:
            float: Clock drift rate [s/day].
        """
        if np.isnat(self.time_check_0) or np.isnat(self.time_check_1):
            return 0.0
        days_diff = (self.time_check_1 - self.time_check_0) / np.timedelta64(1, "D")
        offset_diff = self.offset_1 - self.offset_0
        return offset_diff / days_diff if days_diff != 0 else 0.0


def convert_datetime64_to_pldatetime(np_datetime64: np.datetime64) -> str:
    """Convert numpy.datetime64 to Polars datetime."""
    np_datetime64_us = np_datetime64.astype(f"datetime64[{TIME_PRECISION}]")
    int64_us = np.int64(np_datetime64_us.view("int64"))
    return pl.lit(int64_us).cast(pl.Datetime(TIME_PRECISION))


def convert_datetime64_to_iso(dt64: np.datetime64) -> str:
    """Convert numpy.datetime64 to string in 'YYYY-MM-DDTHH:MM:SS.f' format"""
    return np.datetime_as_string(dt64, unit="us").tolist()


def convert_datetime64_to_string(dt64: np.datetime64, readable: bool = False) -> str:
    """Convert numpy.datetime64 to string in 'YYYYMMDDTHHMMSS' format"""
    dt = dt64.astype(datetime)
    if readable:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y%m%dT%H%M%S")


def convert_dtarray_to_ydarray(
    list_of_datetimes: list[list[np.datetime64]],
) -> np.ndarray:
    """Convert list of datetimes to 3D array.

    This function is required to convert to the .MAT 'FileInfo' structure.

    Args:
        list_of_datetimes (list[list[np.datetime64]]): List of datetimes.

    Returns:
        np.ndarray: 3D array of datetimes with dimensions (2 x M x N),
        where the first axis elements are the year and year-day, M is
        the number of records in each file, and N is the number of files.
    """
    L = 2
    M = max(len(dt) for dt in list_of_datetimes)
    N = len(list_of_datetimes)
    arr = np.zeros((L, M, N), dtype=np.float64)

    for i, dt in enumerate(list_of_datetimes):
        for j, d in enumerate(dt):
            year, yd_decimal = convert_timestamp_to_yyd(d)
            arr[0, j, i] = year
            arr[1, j, i] = yd_decimal

    return arr


def convert_filename_to_datetime64(
    filename: str,
    doy_indices: tuple[int, int],
    hour_indices: tuple[int, int],
    minute_indices: tuple[int, int],
    year_indices: tuple[int, int] | None = None,
    year: int | None = None,
    seconds_indices: tuple[int, int] | None = None,
):
    """Convert a filename with flexible format to np.datetime64 with microsecond
    precision.

    Args:
        filename: The filename containing date/time components.
        doy_indices: (start_index, length) for day-of-year.
        hour_indices: (start_index, length) for hours.
        minute_indices: (start_index, length) for minutes.
        year_indices: (start_index, length) for year.
        year: Direct year specification if not in filename.
        seconds_indices: (start_index, length) for seconds.

    Returns:
        Datetime with microsecond precision.
    """
    # Extract components
    doy = int(filename[doy_indices[0] : doy_indices[0] + doy_indices[1]])
    hour = int(filename[hour_indices[0] : hour_indices[0] + hour_indices[1]])
    minute = int(filename[minute_indices[0] : minute_indices[0] + minute_indices[1]])
    # Handle seconds if provided
    second = 0
    logging.debug(
        f"Extracted components from filename '{filename}': "
        f"doy={doy}, hour={hour}, minute={minute}"
    )
    if seconds_indices:
        # print(seconds_indices[0], seconds_indices[1])
        second = int(
            filename[seconds_indices[0] : seconds_indices[0] + seconds_indices[1]]
        )

    logging.debug(
        f"Extracted seconds from filename '{filename}': second={second if second else 'N/A'}"
    )

    # Handle year
    if year_indices:
        # Extract year from filename
        year_str = filename[year_indices[0] : year_indices[0] + year_indices[1]]

        # Handle YY format
        if len(year_str) == 2:
            # Assume 20XX for years less than 50, 19XX otherwise
            prefix = "20" if int(year_str) < 50 else "19"
            year_str = prefix + year_str

        year = int(year_str)
    elif year is None:
        # Default to current year if not specified
        year = datetime.now().year

    logging.debug(f"Final year extracted: {year}")

    date = datetime.strptime(f"{year}-{doy}", "%Y-%j").replace(
        hour=hour, minute=minute, second=second
    )
    logging.debug(f"Constructed datetime: {date}")
    return np.datetime64(date, TIME_PRECISION)


def convert_ints_to_datetime(
    year: int, yd: int, minute: int, millisec: int, microsec: int
) -> np.datetime64:
    """Converts year, year-day, minute, millisecond, and microsecond to a timestamp.

    Args:
        year (int): Year.
        yd (int): Year-day.
        minute (int): Minute.
        millisec (int): Millisecond.
        microsec (int): Microsecond.

    Returns:
        np.datetime64: Timestamp.
    """
    base_date = np.datetime64(f"{year}-01-01")
    return (
        base_date
        + np.timedelta64(yd - 1, "D")
        + np.timedelta64(minute, "m")
        + np.timedelta64(millisec, "ms")
        + np.timedelta64(microsec, "us")
    )


def convert_pldatetime_to_datetime64(pl_datetime: pl.Series) -> list[np.datetime64]:
    int64_us = pl_datetime.cast(pl.Int64)
    return np.datetime64(int64_us, TIME_PRECISION)


def convert_timestamp_to_yyd(timestamp: np.datetime64) -> tuple[int, float]:
    """Converts a timestamp to year and year-day fraction.

    Args:
        timestamp (np.datetime64): Timestamp to convert.

    Returns:
        tuple[int, float]: Year and year-day fraction.
    """
    Y, rmndr = [timestamp.astype(f"datetime64[{t}]") for t in ["Y", TIME_PRECISION]]
    year = Y.astype(int) + 1970
    yd = (rmndr - np.datetime64(f"{year - 1}-12-31")) / np.timedelta64(1, "D")
    return year, yd


def convert_yydfrac_to_timestamp(year: int, yd: float) -> np.datetime64:
    """Converts year and year-day fraction to a timestamp.

    Args:
        year (int): Year.
        yd (float): Year-day fraction.

    Returns:
        np.datetime64: Timestamp.
    """
    base_date = np.datetime64(f"{int(year)}-01-01")
    day = int(yd)
    fraction = yd - day
    rmndr = int(fraction * 24 * 60 * 60 * TIME_CONVERSION_FACTOR)
    return base_date + np.timedelta64(day, "D") + np.timedelta64(rmndr, TIME_PRECISION)


def correct_sampling_rate(fs: float, drift_rate: float) -> float:
    """Corrects sampling rate for clock drift.

    Args:
        fs (float): Sampling rate (Hz).
        drift_rate (float): Clock drift rate (s/s).

    Returns:
        float: Corrected sampling rate.
    """
    return fs / (1 + drift_rate / (24 * 60 * 60))


def correct_clock_drift(
    timestamp: np.datetime64, clock: ClockParameters
) -> np.datetime64:
    """Correct clock drift in a timestamp.

    Args:
        timestamp (np.datetime64): Timestamp to correct.
        clock (ClockParameters): Clock parameters.

    Returns:
        np.datetime64: Corrected timestamp.
    """
    if np.isnat(clock.time_check_0) or np.isnat(clock.time_check_1):
        return timestamp

    days_diff = (timestamp - clock.time_check_0) / np.timedelta64(1, "D")
    drift = clock.drift_rate * days_diff
    return timestamp + np.timedelta64(
        int(TIME_CONVERSION_FACTOR * drift), TIME_PRECISION
    )


def datetime_linspace(start: np.datetime64, end: np.datetime64, num: int) -> np.ndarray:
    start_int = start.astype("int64")
    end_int = end.astype("int64")
    return np.linspace(start_int, end_int, num, dtype="int64").astype(
        f"datetime64[{TIME_PRECISION}]"
    )
