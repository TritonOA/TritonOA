# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from tritonoa.data.catalog import Catalog
import tritonoa.data.formats.factory as factory
from tritonoa.data.formats.shru import SHRUReader
from tritonoa.data.formats.sio import SIOReader
from tritonoa.data.formats.wav import WAVReader
from tritonoa.data.stream import DataStream, DataStreamStats

MAX_BUFFER = int(2e9)

READER_REGISTRY = {
    "SHRU": SHRUReader,
    "SIO": SIOReader,
    "WAV": WAVReader,
}


def read_data(file_path: Path, data_type: Optional[str] = None, **kwargs) -> DataStream:
    file_format = factory.get_file_format(file_path.suffix, data_type)
    if file_format not in READER_REGISTRY:
        raise ValueError(f"Unsupported file format: {file_format}")
    return READER_REGISTRY[file_format]().read(file_path, **kwargs)


# def read_catalogue(query: CatalogueQuery, max_buffer: int = MAX_BUFFER) -> DataStream:
def read_catalogue(file_path: Path, channels: Optional[int | list[int]] = None, max_buffer: int = MAX_BUFFER) -> DataStream:
    """Reads data from catalogue using the query parameters.

    Args:
        query (CatalogueQuery): Query parameters.
        max_buffer (int): Maximum buffer length in samples.

    Returns:
        DataStream: Data stream object.

    Raises:
        NoDataError: If no data is found for the given query parameters.
        ValueError: If multiple sampling rates are found in the catalogue.
        BufferExceededWarning: If buffer length is less than expected samples.
    """

    def _enforce_sorted_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(by="timestamp").replace_column(
            0, pl.Series("row_nr", list(range(len(df))))
        )

    catalog = Catalog().load(file_path)
    print(catalog.columns)

    num_channels = len(channels)
    return
    df = select_records_by_time(
        _enforce_sorted_df(catalogue.df), query.time_start, query.time_end
    )
    if len(df) == 0:
        raise NoDataError("No data found for the given query parameters.")
    logging.debug(f"Reading {len(df)} records.")

    filenames = [
        Path(f) for f in sorted(df.unique(subset=["filename"])["filename"].to_list())
    ]
    timestamps = sorted(df.unique(subset=["filename"])["timestamp"].to_numpy())
    fixed_gains = df.unique(subset=["filename"])["fixed_gain"].to_list()
    sensitivities = df.unique(subset=["filename"])["hydrophone_sensitivity"].to_list()
    sampling_rates = (
        df.unique(subset=["filename"])
        .unique(subset=["sampling_rate"])["sampling_rate"]
        .to_list()
    )
    if len(set(sampling_rates)) > 1:
        raise ValueError(
            "Multiple sampling rates found in the catalogue; unable to proceed."
        )
    sampling_rate = sampling_rates[0]

    report_buffer(max_buffer, num_channels, sampling_rate)
    expected_buffer = compute_expected_buffer(df)
    if expected_buffer > max_buffer:
        warnings.warn(
            f"Buffer length {max_buffer} is less than expected samples {expected_buffer}.",
            BufferExceededWarning,
        )

    logging.debug(f"Initializing buffer...")
    waveform = -2009.0 * np.ones((expected_buffer, num_channels))
    logging.debug(f"Buffer initialized.")

    marker = 0
    time_init = None
    time_end = None
    for filename, timestamp, fixed_gain, sensitivity in zip(
        filenames, timestamps, fixed_gains, sensitivities
    ):
        logging.debug(f"Reading {filename} at {timestamp}.")
        # Define 'time_init' for waveform:
        if time_init is None:
            time_init = timestamp
        # Check time gap between files and stop if files are not continuous:
        if (time_end is not None) and (
            abs(timestamp - time_end) / np.timedelta64(1, "s") > 1 / sampling_rate
        ):
            warnings.warn(
                "Files are not continuous; time gap between files is greater than 1/sampling_rate.\n"
                f"    Stopping at {filename} at {time_end}.",
                DataDiscontinuityWarning,
            )
            break

        # Get record numbers for the file:
        rec_ind = df.filter(pl.col("filename") == str(filename))[
            "record_number"
        ].to_list()
        logging.debug(f"Reading records {rec_ind} from {filename}.")

        # Read data from file; header is not used here:
        raw_data, _ = _get_reader(file_format=validate_file_format(filename.suffix))(
            filename=filename,
            records=rec_ind,
            channels=query.channels,
            fixed_gain=fixed_gain,
        )

        # Condition data and get units:
        data, units = _get_conditioner(
            file_format=validate_file_format(filename.suffix)
        )(raw_data, fixed_gain, sensitivity)

        # Store data in waveform & advance marker by data length:
        waveform[marker : marker + data.shape[0]] = data
        marker += data.shape[0]

        # Compute time of last point in waveform:
        time_end = timestamp + np.timedelta64(
            int(TIME_CONVERSION_FACTOR * data.shape[0] / sampling_rate), TIME_PRECISION
        )

    if marker < expected_buffer:
        waveform = waveform[:marker]

    ds = DataStream(
        stats=DataStreamStats(
            channels=query.channels,
            time_init=time_init,
            time_end=time_end,
            sampling_rate=sampling_rate,
            units=units,
        ),
        waveform=waveform,
    )
    return ds.trim(starttime=query.time_start, endtime=query.time_end)


def select_records_by_time(
    df: pl.DataFrame, time_start: np.datetime64, time_end: np.datetime64
) -> pl.DataFrame:
    """Select files by time."""
    logging.debug(f"Selecting records by time: {time_start} to {time_end}.")
    if time_start > time_end:
        raise ValueError("time_start must be less than time_end.")
    if np.isnat(time_start) and np.isnat(time_end):
        logging.debug("No times provided; returning all rows.")
        return df
    if time_start is not None and np.isnat(time_end):
        logging.debug("Only start time provided; returning all rows after.")
        row_numbers = df.filter(pl.col("timestamp") >= time_start)["row_nr"].to_list()
        row_numbers.insert(0, row_numbers[0] - 1)
        mask = pl.col("row_nr").is_in(row_numbers)
        return df.filter(mask)
    if np.isnat(time_start) and time_end is not None:
        logging.debug("Only end time provided; returning all rows before.")
        return df.filter(pl.col("timestamp") <= time_end)

    logging.debug("Start and end times provided; returning rows between.")
    last_row_before_start = df.filter(pl.col("timestamp") < time_start)["row_nr"].max()
    row_numbers = df.filter(
        (pl.col("timestamp") >= time_start) & (pl.col("timestamp") <= time_end)
    )["row_nr"].to_list()
    row_numbers.insert(0, last_row_before_start)
    mask = pl.col("row_nr").is_in(row_numbers)
    return df.filter(mask)


def report_buffer(buffer: int, num_channels: int, sampling_rate: float) -> None:
    """Logs buffer size.

    Args:
        buffer (int): Buffer length in samples.
        num_channels (int): Number of channels.
        sampling_rate (float): Sampling rate.

    Returns:
        None
    """
    logging.debug(
        f"MAX BUFFER ---- length: {buffer:e} samples | "
        f"length per channel: {int(buffer / num_channels)} samples | "
        f"size: {8 * buffer / 1e6:n} MB | "
        f"duration: {buffer / sampling_rate / num_channels / 60:.3f} min with {num_channels} channels."
    )


def compute_expected_buffer(df: pl.DataFrame) -> int:
    """Computes expected samples.

    Args:
        df (pl.DataFrame): Data frame.

    Returns:
        int: Expected samples.
    """
    expected_samples = 0
    for filename in sorted(df.unique(subset=["filename"])["filename"].to_list()):
        rec_ind = df.filter(pl.col("filename") == filename)["record_number"].to_list()
        expected_samples += df.filter(
            (pl.col("filename") == filename) & (pl.col("record_number").is_in(rec_ind))
        )["npts"].sum()
    logging.debug(f"Expected samples: {expected_samples}")
    return expected_samples
