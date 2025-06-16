import logging
from pathlib import Path
import warnings

import numpy as np
import polars as pl

from tritonoa.data.inventory import Inventory
import tritonoa.data.formats.factory as factory
from tritonoa.data.signal import SignalParams
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import TIME_CONVERSION_FACTOR, TIME_PRECISION

MAX_BUFFER = int(2e9)


class BufferExceededWarning(Warning):
    pass


class DataDiscontinuityWarning(Warning):
    pass


class NoDataError(Exception):
    pass


def read_inventory(
    file_path: Path,
    time_start: np.datetime64 | None = None,
    time_end: np.datetime64 | None = None,
    channels: int | list[int] = None,
    metadata: dict | None = None,
    max_buffer: int = MAX_BUFFER,
) -> DataStream:
    """Reads data from inventory using the query parameters.

    Args:
        query (InventoryQuery): Query parameters.
        max_buffer (int): Maximum buffer length in samples.

    Returns:
        DataStream: Data stream object.

    Raises:
        NoDataError: If no data is found for the given query parameters.
        ValueError: If multiple sampling rates are found in the inventory.
        BufferExceededWarning: If buffer length is less than expected samples.
    """

    def _enforce_sorted_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(by="timestamp").replace_column(
            0, pl.Series("row_nr", list(range(len(df))))
        )

    def _get_gains_and_sensitivities(
        df: pl.DataFrame,
        num_timestamps: int,
    ) -> tuple[list[float], list[float]]:
        gains = df.unique(subset=["filename"])["gain"].to_list()[0]
        sensitivities = df.unique(subset=["filename"])["sensitivity"].to_list()[0]
        if channels is not None:
            gains = [gains[i] for i in channels]
            sensitivities = [sensitivities[i] for i in channels]
        return [gains] * num_timestamps, [sensitivities] * num_timestamps

    def _get_sampling_rate(df: pl.DataFrame) -> float:
        sampling_rates = (
            df.unique(subset=["filename"])
            .unique(subset=["sampling_rate"])["sampling_rate"]
            .to_list()
        )
        if len(set(sampling_rates)) > 1:
            raise ValueError(
                "Multiple sampling rates found in the inventory; unable to proceed."
            )
        return sampling_rates[0]

    catalog = Inventory().load(file_path)
    df = _select_records_by_time(_enforce_sorted_df(catalog), time_start, time_end)

    if channels is None:
        num_channels = _get_nchannels(df)
    else:
        channels = [channels] if isinstance(channels, int) else channels
        num_channels = len(channels)

    if len(df) == 0:
        raise NoDataError("No data found for the given query parameters.")
    logging.debug(f"Reading {len(df)} records.")

    filenames = [
        Path(f) for f in sorted(df.unique(subset=["filename"])["filename"].to_list())
    ]
    timestamps = sorted(df.unique(subset=["filename"])["timestamp"].to_numpy())
    gains, sensitivities = _get_gains_and_sensitivities(df, len(timestamps))
    sampling_rate = _get_sampling_rate(df)

    expected_buffer = _check_buffer(max_buffer, num_channels, sampling_rate, df)
    logging.debug(f"Initializing buffer...")
    waveform = -2009.0 * np.ones((num_channels, expected_buffer), dtype=np.float64)
    logging.debug(f"Buffer initialized: {waveform.shape}.")

    marker = 0
    time_init = None
    time_stop = None
    max_time_gap = 1 / sampling_rate

    for filename, timestamp, gain, sensitivity in zip(
        filenames, timestamps, gains, sensitivities
    ):
        logging.debug(f"Reading {filename} at {timestamp}.")
        # Define 'time_init' for waveform:
        if time_init is None:
            time_init = timestamp
        # Check time gap between files and stop if files are not continuous:
        if time_stop is not None and _exceeds_max_time_gap(
            timestamp, time_stop, max_time_gap
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
        logging.debug("Reading raw data.")
        reader = factory.get_reader(filename.suffix)
        raw_data, _ = reader.read_raw_data(filename, records=rec_ind, channels=channels)
        logging.debug(f"Raw data shape: {raw_data.shape}.")
        # Condition data and get units:
        data, units = reader.condition_data(raw_data, SignalParams(gain, sensitivity))
        logging.debug(f"Conditioned data shape: {raw_data.shape}.")
        # Store data in waveform & advance marker by data length:
        waveform[:, marker : marker + data.shape[1]] = data
        logging.debug(f"Old marker at {marker}.")
        marker += data.shape[1]
        logging.debug(f"New marker at {marker}.")
        # Compute time of last point in waveform:
        time_stop = timestamp + np.timedelta64(
            int(TIME_CONVERSION_FACTOR * data.shape[1] / sampling_rate), TIME_PRECISION
        )

    if marker < expected_buffer:
        waveform = waveform[:, :marker]

    return DataStream(
        stats=DataStreamStats(
            channels=channels,
            time_init=time_init,
            time_end=time_stop,
            sampling_rate=sampling_rate,
            units=units,
            metadata=metadata,
        ),
        data=waveform,
    ).trim(starttime=time_start, endtime=time_end)


def read_numpy(file_path: Path) -> DataStream:
    dsdata = np.load(file_path, allow_pickle=True)
    return DataStream(
        stats=dsdata["stats"].item(),
        data=dsdata["data"],
    )


def read_data(file_path: Path, data_type: str | None = None, **kwargs) -> DataStream:
    return factory.get_reader(file_path.suffix, data_type).read(file_path, **kwargs)


def _check_buffer(
    max_buffer: int, num_channels: int, sampling_rate: float, df: pl.DataFrame
) -> int:
    _report_buffer(max_buffer, num_channels, sampling_rate)
    expected_buffer = _compute_expected_buffer(df)
    if expected_buffer > max_buffer:
        warnings.warn(
            f"Buffer length {max_buffer} is less than expected samples {expected_buffer}.",
            BufferExceededWarning,
        )

    return expected_buffer


def _compute_expected_buffer(df: pl.DataFrame) -> int:
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


def _exceeds_max_time_gap(
    timestamp: np.datetime64,
    time_end: np.datetime64,
    max_time_gap: float,
) -> bool:
    time_gap = abs(timestamp - time_end) / np.timedelta64(1, "s")
    if time_gap > max_time_gap:
        return True
    return False


def _get_nchannels(df: pl.DataFrame) -> int:
    return len(df.select(pl.first("gain")).item())


def _report_buffer(buffer: int, num_channels: int, sampling_rate: float) -> None:
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


def _select_records_by_time(
    df: pl.DataFrame,
    time_start: np.datetime64 | None = None,
    time_end: np.datetime64 | None = None,
) -> pl.DataFrame:
    """Select files by time."""
    logging.debug(f"Selecting records by time: {time_start} to {time_end}.")
    if time_start is None and time_end is None:
        logging.debug("No times provided; returning all rows.")
        return df
    if time_start is not None and time_end is None:
        logging.debug("Only start time provided; returning all rows after.")
        row_numbers = df.filter(pl.col("timestamp") >= time_start)["row_nr"].to_list()
        row_numbers.insert(0, row_numbers[0] - 1)
        mask = pl.col("row_nr").is_in(row_numbers)
        return df.filter(mask)
    if time_start is None and time_end is not None:
        logging.debug("Only end time provided; returning all rows before.")
        return df.filter(pl.col("timestamp") <= time_end)
    if time_start > time_end:
        raise ValueError("time_start must be less than time_end.")

    logging.debug("Start and end times provided; returning rows between.")
    last_row_before_start = df.filter(pl.col("timestamp") < time_start)["row_nr"].max()
    row_numbers = df.filter(
        (pl.col("timestamp") >= time_start) & (pl.col("timestamp") <= time_end)
    )["row_nr"].to_list()
    row_numbers.insert(0, last_row_before_start)
    mask = pl.col("row_nr").is_in(row_numbers)
    return df.filter(mask)
