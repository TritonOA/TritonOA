# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import logging
from pathlib import Path
import tomllib
from typing import Optional, Protocol, Union

import numpy as np
import polars as pl
import scipy

from tritonoa.data.formats.formats import FileFormat, validate_file_format
from tritonoa.data.formats.shru import read_shru_headers
from tritonoa.data.formats.sio import read_sio_headers
from tritonoa.data.formats.wav import read_wav_headers
from tritonoa.data.hydrophone import HydrophoneSpecs
from tritonoa.data.time import (
    TIME_PRECISION,
    TIME_CONVERSION_FACTOR,
    ClockParameters,
    convert_timestamp_to_yyd,
    convert_to_datetime,
    convert_yydfrac_to_timestamp,
    correct_clock_drift,
)

MAT_KEYS = ["__header__", "__version__", "__globals__"]


class FileCallbackHandler:
    """Callback handler for file-format--specific functions."""

    def __init__(self, file_format: FileFormat):
        self.file_format = file_format

    def format_records(self, records: list[DataRecord]) -> list[DataRecord]:
        if self.file_format == FileFormat.SHRU:
            return self.format_shru_records(records)
        return

    @staticmethod
    def format_shru_records(records: list[DataRecord]) -> list[DataRecord]:
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

    @staticmethod
    def format_sio_records(records: list[DataRecord]) -> list[DataRecord]:
        """Format SIO records.

        Nothing implemented.

        Args:
            records (list[Record]): List of records.

        Returns:
            list[Record]: List of records.
        """
        return records

    @staticmethod
    def format_wav_records(records: list[DataRecord]) -> list[DataRecord]:
        """Format WAV records.

        Nothing implemented.

        Args:
            records (list[Record]): List of records.

        Returns:
            list[Record]: List of records.
        """
        return records


class Header(Protocol):
    """Protocol for file-format--specific header objects."""

    ...


class DatasetCatalogueFileFormat(Enum):
    """File formats for record catalogues."""

    CSV = "csv"
    JSON = "json"
    MAT = "mat"


@dataclass
class DataRecord:
    """Data record object.

    Args:
        filename (Path): File name.
        record_number (int): Record number.
        file_format (FileFormat): File format.
        timestamp (Union[np.datetime64, pl.Datetime]): Timestamp.
        timestamp_orig (Union[np.datetime64, pl.Datetime]): Original timestamp.
        sampling_rate (float): Sampling rate.
        sampling_rate_orig (float): Original sampling rate.
        fixed_gain (float): Fixed gain.
        hydrophone_sensitivity (float): Hydrophone sensitivity.
        hydrophone_SN (int): Hydrophone serial number.
        npts (Optional[int], optional): Number of points in the record. Defaults to None.

    Returns:
        Record: Data record object.
    """

    filename: Path
    record_number: int
    file_format: FileFormat
    timestamp: np.datetime64 | pl.Datetime
    timestamp_orig: np.datetime64 | pl.Datetime
    sampling_rate: float
    sampling_rate_orig: float
    fixed_gain: float
    hydrophone_sensitivity: float
    hydrophone_SN: int
    npts: Optional[int] = None


class DataRecordConstructor(Protocol):
    """Protocol for file-format--specific data record constructors."""

    def __call__(self, filename: Path, record_number: int, header: Header) -> DataRecord:
        ...


class DatasetCatalogue:
    def __init__(self, records: Optional[list[DataRecord]] = None):
        self.records = records
        if records is not None:
            self.df = self._records_to_df()

    def __repr__(self):
        return f"DatasetCatalogue(records={self.records}, df={self.df})"

    def build(self, query: FileInfoQuery) -> DatasetCatalogue:
        """Build the record catalogue from file info query.

        Args:
            query (FileInfoQuery): File info query.

        Returns:
            DatasetCatalogue: Record catalogue.

        Raises:
            FileNotFoundError: If no SHRU files are found in the directory.
        """
        files = list(Path(query.data.directory).glob(query.data.glob_pattern))
        files.sort(key=lambda x: x.name)

        if len(files) == 0:
            logging.error("No SHRU files found in directory.")
            raise FileNotFoundError("No SHRU files found in directory.")
        logging.debug(f"{len(files)} files found for record catalogue.")

        records = []
        for j, f in enumerate(files):
            headers, file_format = read_headers(f, file_format=query.data.file_format)
            records_from_file = []
            callback = FileCallbackHandler(file_format)

            for i, header in enumerate(headers):
                ts_orig = get_timestamp(header)
                ts = correct_clock_drift(ts_orig, query.clock)
                records_from_file.append(
                    DataRecord(
                        filename=f,
                        record_number=i,
                        file_format=file_format,
                        npts=header.npts,
                        timestamp=ts,
                        timestamp_orig=ts_orig,
                        sampling_rate_orig=get_sampling_rate(file_format, headers),
                        sampling_rate=get_sampling_rate(file_format, headers)
                        / (1 + query.clock.drift_rate / 24 / 3600),
                        fixed_gain=query.hydrophones.fixed_gain,
                        hydrophone_sensitivity=query.hydrophones.sensitivity,
                        hydrophone_SN=query.hydrophones.serial_number,
                    )
                )
            corrected_records = callback.format_records(records_from_file)
            records.extend(corrected_records)
            logging.debug(
                f"{len(records) + 1} records | {j}/{len(files)} files processed."
            )

        self.records = records
        self.df = self._records_to_df()
        return self

    def _format_df_for_csv(self, df: pl.DataFrame) -> pl.DataFrame:
        """Format Polars DataFrame for CSV output.

        Args:
            df (pl.DataFrame): Polars DataFrame.

        Returns:
            pl.DataFrame: Polars DataFrame.
        """

        def _to_list(lst: list):
            return ",".join([str(i) for i in lst])

        return df.with_columns(
            pl.col("fixed_gain").map_elements(
                _to_list, return_dtype=pl.List(pl.Float64)
            ),
            pl.col("hydrophone_sensitivity").map_elements(
                _to_list, return_dtype=pl.List(pl.Float64)
            ),
            pl.col("hydrophone_SN").map_elements(
                _to_list, return_dtype=pl.List(pl.Int32)
            ),
        )

    def load(self, filepath: Path) -> DatasetCatalogue:
        """Load the record catalogue from file.

        Args:
            filepath (Path): Path to the catalogue file.

        Returns:
            DatasetCatalogue: Record catalogue.

        Raises:
            ValueError: If file format is not recognized.
        """
        extension = filepath.suffix[1:].lower()

        if extension not in DatasetCatalogueFileFormat:
            raise ValueError(f"File format '{extension}' is not recognized.")

        if extension == DatasetCatalogueFileFormat.CSV.value:
            self._read_csv(filepath)
        if extension == DatasetCatalogueFileFormat.JSON.value:
            self._read_json(filepath)
        if extension == DatasetCatalogueFileFormat.MAT.value:
            self._read_mat(filepath)
        return self

    def _read_csv(self, filepath: Path) -> None:
        """Read the record catalogue from CSV file.

        Args:
            filepath (Path): Path to the catalogue file.

        Returns:
            None
        """

        def _str_to_list(s: str, dtype=float) -> list:
            if s == "nan":
                return []
            return [dtype(i) for i in s.split(",") if i]

        self.df = pl.read_csv(filepath).with_columns(
            pl.col("timestamp").cast(pl.Datetime(TIME_PRECISION)),
            pl.col("timestamp_orig").cast(pl.Datetime(TIME_PRECISION)),
            pl.col("fixed_gain").map_elements(
                partial(_str_to_list, dtype=float), return_dtype=pl.List(float)
            ),
            pl.col("hydrophone_sensitivity").map_elements(
                partial(_str_to_list, dtype=float), return_dtype=pl.List(float)
            ),
            pl.col("hydrophone_SN").map_elements(
                partial(_str_to_list, dtype=int), return_dtype=pl.List(int)
            ),
        )

    def _read_json(self, filepath: Path) -> None:
        """Read the record catalogue from JSON file.

        Args:
            filepath (Path): Path to the catalogue file.

        Returns:
            None
        """
        self.df = pl.read_json(filepath)

    def _read_mat(self, filepath: Path) -> None:
        """Read the record catalogue from MAT file.

        Args:
            filepath (Path): Path to the catalogue file.

        Returns:
            None
        """
        mdict = scipy.io.loadmat(filepath)
        self.records = self._mdict_to_records(mdict)
        self.df = self._records_to_df()

    def _mdict_to_records(self, mdict: dict) -> list[DataRecord]:
        """Convert MAT file dictionary to records.

        Args:
            mdict (dict): MAT file dictionary.

        Returns:
            list[Record]: List of records.
        """
        [mdict.pop(k, None) for k in MAT_KEYS]
        cat_name = list(mdict.keys()).pop()
        cat_data = mdict[cat_name]

        filenames = cat_data["filenames"][0][0].astype(str).tolist()
        timestamp_data = cat_data["timestamps"][0][0].transpose(2, 1, 0)
        timestamp_orig_data = cat_data["timestamps_orig"][0][0].transpose(2, 1, 0)
        rhfs_orig = float(cat_data["rhfs_orig"][0][0].squeeze())
        rhfs = float(cat_data["rhfs"][0][0].squeeze())
        fixed_gain = cat_data["fixed_gain"][0][0].squeeze().astype(float).tolist()
        hydrophone_sensitivity = (
            cat_data["hydrophone_sensitivity"][0][0].squeeze().astype(float).tolist()
        )
        hydrophone_SN = cat_data["hydrophone_SN"][0][0].squeeze().astype(int).tolist()

        records = []
        for i, f in enumerate(filenames):
            n_records = cat_data["timestamps"][0][0].transpose(2, 1, 0).shape[1]
            for j in range(n_records):
                records.append(
                    DataRecord(
                        filename=Path(f),
                        record_number=j,
                        file_format=validate_file_format(Path(f).suffix),
                        timestamp=convert_yydfrac_to_timestamp(*timestamp_data[i][j]),
                        timestamp_orig=convert_yydfrac_to_timestamp(
                            *timestamp_orig_data[i][j]
                        ),
                        sampling_rate=rhfs,
                        sampling_rate_orig=rhfs_orig,
                        fixed_gain=fixed_gain,
                        hydrophone_sensitivity=hydrophone_sensitivity,
                        hydrophone_SN=hydrophone_SN,
                    )
                )

        return records

    def _records_to_dfdict(self) -> dict:
        """Convert records to dictionary.

        The dictionary is used to construct the object's DataFrame.

        Returns:
            dict: Dictionary of records.
        """
        return {
            "filename": [str(record.filename) for record in self.records],
            "record_number": [record.record_number for record in self.records],
            "file_format": [record.file_format.name for record in self.records],
            "npts": [record.npts for record in self.records],
            "timestamp": [record.timestamp.astype("int64") for record in self.records],
            "timestamp_orig": [
                record.timestamp_orig.astype("int64") for record in self.records
            ],
            "sampling_rate": [record.sampling_rate for record in self.records],
            "sampling_rate_orig": [
                record.sampling_rate_orig for record in self.records
            ],
            "fixed_gain": [record.fixed_gain for record in self.records],
            "hydrophone_sensitivity": [
                record.hydrophone_sensitivity for record in self.records
            ],
            "hydrophone_SN": [record.hydrophone_SN for record in self.records],
        }

    def _records_to_mdict(self) -> dict:
        """Convert records to dictionary.

        The dictionary is used to construct the .MAT file.

        Returns:
            dict: Dictionary of records.
        """
        filenames = list(set([str(record.filename) for record in self.records]))
        timestamps = []
        timestamps_orig = []
        for filename in filenames:
            timestamps.append(
                (
                    self.df.filter(pl.col("filename") == filename)
                    .select("timestamp")
                    .to_series()
                    .cast(pl.Int64)
                )
                .to_numpy()
                .astype("datetime64[us]")
            )
            timestamps_orig.append(
                (
                    self.df.filter(pl.col("filename") == filename)
                    .select("timestamp_orig")
                    .to_series()
                    .cast(pl.Int64)
                )
                .to_numpy()
                .astype("datetime64[us]")
            )

        return {
            self.records[0].file_format.name: {
                "filenames": filenames,
                "timestamps": to_ydarray(timestamps),
                "timestamps_orig": to_ydarray(timestamps_orig),
                "rhfs_orig": self.records[0].sampling_rate_orig,
                "rhfs": self.records[0].sampling_rate,
                "fixed_gain": self.records[0].fixed_gain,
                "hydrophone_sensitivity": self.records[0].hydrophone_sensitivity,
                "hydrophone_SN": self.records[0].hydrophone_SN,
            }
        }

    def _records_to_df(self) -> pl.DataFrame:
        """Convert records to Polars DataFrame.

        Returns:
            pl.DataFrame: Polars DataFrame.
        """
        return (
            pl.DataFrame(self._records_to_dfdict())
            .with_columns(pl.col("timestamp").cast(pl.Datetime(TIME_PRECISION)))
            .with_columns(pl.col("timestamp_orig").cast(pl.Datetime(TIME_PRECISION)))
            .sort(by="timestamp")
            .with_row_count()
        )

    def save(self, savepath: Path, fmt: Union[str, list[str]] = "csv"):
        """Save the catalogue to file.

        Args:
            savepath (Path): Path to save the catalogue file.
            fmt (Union[str, list[str]], optional): File format(s) to save the catalogue. Defaults to "csv".

        Raises:
            ValueError: If file format is not recognized.
        """
        if isinstance(fmt, str):
            fmt = [fmt]
        if not all(f in DatasetCatalogueFileFormat for f in fmt):
            raise ValueError(f"File format {fmt} is not recognized.")

        savepath.parent.mkdir(parents=True, exist_ok=True)

        if DatasetCatalogueFileFormat.CSV.name.lower() in fmt:
            self.write_csv(savepath.parent / (savepath.stem + ".csv"))
        if DatasetCatalogueFileFormat.JSON.name.lower() in fmt:
            self.write_json(savepath.parent / (savepath.stem + ".json"))
        if DatasetCatalogueFileFormat.MAT.name.lower() in fmt:
            self.write_mat(savepath.parent / (savepath.stem + ".mat"))

    def write_csv(self, savepath: Path):
        """Write the catalogue to CSV file.

        Args:
            savepath (Path): Path to save the catalogue file.

        Returns:
            None
        """
        df_out = self._format_df_for_csv(self.df)
        df_out.write_csv(savepath)

    def write_json(self, savepath: Path):
        """Write the catalogue to JSON file.

        Args:
            savepath (Path): Path to save the catalogue file.

        Returns:
            None
        """
        self.df.write_json(savepath, pretty=True)

    def write_mat(self, savepath: Path):
        """Write the catalogue to MAT file.

        Args:
            savepath (Path): Path to save the catalogue file.

        Returns:
            None
        """
        mdict = self._records_to_mdict()
        scipy.io.savemat(savepath, mdict)


def build_catalogues(queries: list[FileInfoQuery]) -> list[DatasetCatalogue]:
    """Build catalogues from file info queries.

    Args:
        queries (list[FileInfoQuery]): List of file info queries.

    Returns:
        list[DatasetCatalogue]: List of record catalogues.
    """
    return [DatasetCatalogue().build(q) for q in queries]


def build_and_save_catalogues(
    queries: list[FileInfoQuery], fmt: Union[str, list[str]] = "csv"
) -> None:
    """Build and save catalogues from file info queries.

    Args:
        queries (list[FileInfoQuery]): List of file info queries.
        fmt (Union[str, list[str]], optional): File format(s) to save the catalogue. Defaults to "csv".

    Raises:
        ValueError: If file format is not recognized.
    """
    if isinstance(fmt, str):
        fmt = [fmt]
    if not all(f in DatasetCatalogueFileFormat for f in fmt):
        raise ValueError(f"File format {fmt} is not recognized.")
    [
        cat.save(savepath=Path(q.data.destination) / f"{q.serial}_FileInfo", fmt=fmt)
        for q, cat in zip(queries, build_catalogues(queries))
    ]


def get_headers_reader(
    suffix: Optional[str] = None, file_format: Optional[str] = None
) -> tuple[callable, FileFormat]:
    """Factory to get header reader for file format.

    Args:
        suffix (Optional[str], optional): File suffix. Defaults to None.
        file_format (Optional[str], optional): File format. Defaults to None.

    Returns:
        tuple[callable, FileFormat]: Header reader and file format.

    Raises:
        ValueError: If file format is not recognized.
    """
    file_format = validate_file_format(suffix, file_format)
    if file_format == FileFormat.SHRU:
        return read_shru_headers, file_format
    if file_format == FileFormat.SIO:
        return read_sio_headers, file_format
    if file_format == FileFormat.WAV:
        return read_wav_headers, file_format
    raise ValueError(f"File format {file_format} is not recognized.")


def get_sampling_rate(file_format: FileFormat, headers: list[Header]) -> float:
    """Get sampling rate from headers.

    Args:
        file_format (FileFormat): File format.
        headers (list[Header]): List of headers.

    Returns:
        float: Sampling rate.
    """
    if file_format == FileFormat.SHRU:
        return headers[0].rhfs
    if file_format == FileFormat.SIO:
        return headers[0].rhfs
    if file_format == FileFormat.WAV:
        return headers[0].framerate


def get_timestamp(header: Header) -> np.datetime64:
    """Return the timestamp of a data record header.

    Args:
        header (Header): Data record header.

    Returns:
        np.datetime64: Timestamp.
    """
    year = header.date[0]
    yd = header.date[1]
    minute = header.time[0]
    millisec = header.time[1]
    microsec = header.microsec
    return convert_to_datetime(year, yd, minute, millisec, microsec)


def read_headers(
    filename: Path, file_format: str = None
) -> tuple[list[Header], FileFormat]:
    """Read headers from file.

    Args:
        filename (Path): File name.
        file_format (str, optional): File format. Defaults to None.

    Returns:
        tuple[list[Header], FileFormat]: List of headers and file format.

    Raises:
        ValueError: If file format is not recognized.
    """
    reader, file_format = get_headers_reader(
        suffix=filename.suffix, file_format=file_format
    )
    return reader(filename), file_format


def to_ydarray(list_of_datetimes: list[list[np.datetime64]]) -> np.ndarray:
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




@dataclass
class CatalogueQuery:
    catalogue: Path
    destination: Path
    time_start: Optional[Union[float, np.datetime64]] = None
    time_end: Optional[Union[float, np.datetime64]] = None
    channels: Optional[Union[int, list[int]]] = None

    def __repr__(self):
        return (
            f"CatalogueQuery(catalogue={self.catalogue},"
            f"destination={self.destination}, "
            f"time_start={self.time_start}, "
            f"time_end={self.time_end}, "
            f"channels={self.channels})"
        )


@dataclass
class DataSelection:
    directory: str
    glob_pattern: str
    destination: str = "."
    file_format: Optional[str] = None


@dataclass
class FileInfoQuery:
    serial: str
    data: DataSelection
    clock: ClockParameters = field(default_factory=ClockParameters)
    hydrophones: HydrophoneSpecs = field(default_factory=HydrophoneSpecs)


def load_catalogue_query(filename: Path) -> list[CatalogueQuery]:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    queries = []
    for params in config.values():
        queries.append(
            CatalogueQuery(
                catalogue=Path(params.get("catalogue")),
                destination=params.get("destination", Path.cwd()),
                time_start=np.datetime64(params.get("time_start", None)),
                time_end=np.datetime64(params.get("time_end", None)),
                channels=params.get("channels", None),
            )
        )

    return queries


def load_file_query(filename: Path) -> list[FileInfoQuery]:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    queries = []
    for serial, params in config.items():
        data = DataSelection(**params["data"])
        clock_params = params.get("clock", {})
        clock = ClockParameters(
            time_check_0=np.datetime64(clock_params.get("time_check_0", "NaT")),
            time_check_1=np.datetime64(clock_params.get("time_check_1", "NaT")),
            offset_0=clock_params.get("offset_0", 0.0),
            offset_1=clock_params.get("offset_1", 0.0),
        )
        hydrophone = HydrophoneSpecs(**params.get("hydrophone", {}))
        queries.append(FileInfoQuery(serial, data, clock, hydrophone))

    return queries
