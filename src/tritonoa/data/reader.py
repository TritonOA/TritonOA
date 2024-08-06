# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional

import tritonoa.data.formats.factory as factory
from tritonoa.data.formats.shru import SHRUReader
from tritonoa.data.formats.sio import SIOReader
from tritonoa.data.formats.wav import WAVReader
from tritonoa.data.stream import DataStream


class DataReader:
    def __init__(self):
        self._readers = {
            "SHRU": SHRUReader,
            "SIO": SIOReader,
            "WAV": WAVReader,
        }
    
    def read(self, file_path: Path, data_type: Optional[str] = None, **kwargs) -> DataStream:
        file_format = factory.validate_file_format(file_path.suffix, data_type)
        print(file_format)
        reader = self._readers[file_format.value]()
        return reader.read(file_path, **kwargs)
        # TODO: Return a DataStream object with the data and header information.
        # This interface should be conducted with the format-specific reader 
        # implementations since each format will have its own unique considerations.        
    

class DatasetReader:
    def read(self, file_path: Path, data_type: Optional[str] = None, **kwargs) -> DataStream:
        return DataReader()

# class CatalogueReader:
#     def read(query: CatalogueQuery, max_buffer: int = MAX_BUFFER) -> DataStream:
#         """Reads data from catalogue using the query parameters.

#         Args:
#             query (CatalogueQuery): Query parameters.
#             max_buffer (int): Maximum buffer length in samples.

#         Returns:
#             DataStream: Data stream object.

#         Raises:
#             NoDataError: If no data is found for the given query parameters.
#             ValueError: If multiple sampling rates are found in the catalogue.
#             BufferExceededWarning: If buffer length is less than expected samples.
#         """

#         def _enforce_sorted_df(df: pl.DataFrame) -> pl.DataFrame:
#             return df.sort(by="timestamp").replace_column(
#                 0, pl.Series("row_nr", list(range(len(df))))
#             )

#         catalogue = DatasetCatalogue().load(query.catalogue)

#         num_channels = len(query.channels)
#         df = select_records_by_time(
#             _enforce_sorted_df(catalogue.df), query.time_start, query.time_end
#         )
#         if len(df) == 0:
#             raise NoDataError("No data found for the given query parameters.")
#         logging.debug(f"Reading {len(df)} records.")

#         filenames = [
#             Path(f) for f in sorted(df.unique(subset=["filename"])["filename"].to_list())
#         ]
#         timestamps = sorted(df.unique(subset=["filename"])["timestamp"].to_numpy())
#         fixed_gains = df.unique(subset=["filename"])["fixed_gain"].to_list()
#         sensitivities = df.unique(subset=["filename"])["hydrophone_sensitivity"].to_list()
#         sampling_rates = (
#             df.unique(subset=["filename"])
#             .unique(subset=["sampling_rate"])["sampling_rate"]
#             .to_list()
#         )
#         if len(set(sampling_rates)) > 1:
#             raise ValueError(
#                 "Multiple sampling rates found in the catalogue; unable to proceed."
#             )
#         sampling_rate = sampling_rates[0]

#         report_buffer(max_buffer, num_channels, sampling_rate)
#         expected_buffer = compute_expected_buffer(df)
#         if expected_buffer > max_buffer:
#             warnings.warn(
#                 f"Buffer length {max_buffer} is less than expected samples {expected_buffer}.",
#                 BufferExceededWarning,
#             )

#         logging.debug(f"Initializing buffer...")
#         waveform = -2009.0 * np.ones((expected_buffer, num_channels))
#         logging.debug(f"Buffer initialized.")

#         marker = 0
#         time_init = None
#         time_end = None
#         for filename, timestamp, fixed_gain, sensitivity in zip(
#             filenames, timestamps, fixed_gains, sensitivities
#         ):
#             logging.debug(f"Reading {filename} at {timestamp}.")
#             # Define 'time_init' for waveform:
#             if time_init is None:
#                 time_init = timestamp
#             # Check time gap between files and stop if files are not continuous:
#             if (time_end is not None) and (
#                 abs(timestamp - time_end) / np.timedelta64(1, "s") > 1 / sampling_rate
#             ):
#                 warnings.warn(
#                     "Files are not continuous; time gap between files is greater than 1/sampling_rate.\n"
#                     f"    Stopping at {filename} at {time_end}.",
#                     DataDiscontinuityWarning,
#                 )
#                 break

#             # Get record numbers for the file:
#             rec_ind = df.filter(pl.col("filename") == str(filename))[
#                 "record_number"
#             ].to_list()
#             logging.debug(f"Reading records {rec_ind} from {filename}.")

#             # Read data from file; header is not used here:
#             raw_data, _ = _get_reader(file_format=validate_file_format(filename.suffix))(
#                 filename=filename,
#                 records=rec_ind,
#                 channels=query.channels,
#                 fixed_gain=fixed_gain,
#             )

#             # Condition data and get units:
#             data, units = _get_conditioner(
#                 file_format=validate_file_format(filename.suffix)
#             )(raw_data, fixed_gain, sensitivity)

#             # Store data in waveform & advance marker by data length:
#             waveform[marker : marker + data.shape[0]] = data
#             marker += data.shape[0]

#             # Compute time of last point in waveform:
#             time_end = timestamp + np.timedelta64(
#                 int(TIME_CONVERSION_FACTOR * data.shape[0] / sampling_rate), TIME_PRECISION
#             )

#         if marker < expected_buffer:
#             waveform = waveform[:marker]

#         ds = DataStream(
#             stats=DataStreamStats(
#                 channels=query.channels,
#                 time_init=time_init,
#                 time_end=time_end,
#                 sampling_rate=sampling_rate,
#                 units=units,
#             ),
#             waveform=waveform,
#         )
#         return ds.trim(starttime=query.time_start, endtime=query.time_end)