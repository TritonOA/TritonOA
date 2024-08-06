# -*- coding: utf-8 -*-

# TODO: Replace hydrophone specs w/ signal params

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, Protocol

import polars as pl
import scipy.io

from tritonoa.data.formats import factory
from tritonoa.data.hydrophone import HydrophoneSpecs
from tritonoa.data.time import ClockParameters, TIME_PRECISION, to_ydarray


@dataclass
class Header(Protocol):
    """Protocol for file-format--specific header objects."""

    ...


@dataclass
class DataRecord(Protocol):
    """Data record object."""

    ...


class CatalogueFileFormat(Enum):
    """File formats for record catalogues."""

    CSV = "csv"
    JSON = "json"
    MAT = "mat"


class Catalog:
    def __init__(self, records: Optional[list[DataRecord]] = None):
        self.records = records
        if records is not None:
            self.df = self._records_to_df()

    def build(
        self,
        dataset_path: Path,
        glob_pattern: str = "*",
        clock_params: ClockParameters = ClockParameters(),
        hydrophone_params: Optional[HydrophoneSpecs] = HydrophoneSpecs(),  # TODO: Implement using kwargs
        record_fmt_callback: Optional[callable] = None,
    ) -> pl.DataFrame:
        files = self._get_files(dataset_path, glob_pattern)
        records = []
        for j, file in enumerate(files):
            reader = factory.get_reader(file.suffix)
            formatter = factory.get_formatter(file.suffix)
            headers = reader.read_headers(file)

            # if record_fmt_callback is None:

            records_from_file = []
            for i, header in enumerate(headers):
                records_from_file.append(
                    formatter.format_record(
                        filename=file,
                        record_number=i,
                        header=header,
                        clock=clock_params,
                        hydrophones=hydrophone_params,  # TODO: Implement using kwargs
                    )
                )
            
            corrected_records = formatter.callback(records_from_file)
            records.extend(corrected_records)
            logging.debug(
                f"{len(records) + 1} records | {j}/{len(files)} files processed."
            )

        self.records = records
        self.df = self._records_to_df()
        print(self.df)
        return self.df

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

    @staticmethod
    def _get_files(dataset_path: Path, glob_pattern: str) -> list[Path]:
        files = list(Path(dataset_path).glob(glob_pattern))
        files.sort(key=lambda x: x.name)
        if len(files) == 0:
            err_msg = f"No files found in directory {dataset_path} with pattern '{glob_pattern}'."
            logging.error(err_msg)
            raise FileNotFoundError(err_msg)
        logging.debug(f"{len(files)} files found for record catalogue.")
        return files

    def load(self, filepath: Path) -> pl.DataFrame:
        """Load the record catalogue from file.

        Args:
            filepath (Path): Path to the catalogue file.

        Returns:
            DatasetCatalogue: Record catalogue.

        Raises:
            ValueError: If file format is not recognized.
        """
        extension = filepath.suffix[1:].lower()

        if extension not in CatalogueFileFormat:
            raise ValueError(f"File format '{extension}' is not recognized.")

        if extension == CatalogueFileFormat.CSV.value:
            self._read_csv(filepath)
        if extension == CatalogueFileFormat.JSON.value:
            self._read_json(filepath)
        if extension == CatalogueFileFormat.MAT.value:
            self._read_mat(filepath)
        return self.df

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
    
    def _records_to_dfdict(self) -> dict:
        """Convert records to dictionary.

        The dictionary is used to construct the object's DataFrame.

        Returns:
            dict: Dictionary of records.
        """
        return {
            "filename": [str(record.filename) for record in self.records],
            "record_number": [record.record_number for record in self.records],
            "file_format": [record.file_format for record in self.records],
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
    
    def save(self, savepath: Path, fmt: str | list[str] = "csv"):
        """Save the catalogue to file.

        Args:
            savepath (Path): Path to save the catalogue file.
            fmt (str | list[str], optional): File format(s) to save the catalogue. Defaults to "csv".

        Raises:
            ValueError: If file format is not recognized.
        """
        if isinstance(fmt, str):
            fmt = [fmt]
        if not all(f in CatalogueFileFormat for f in fmt):
            raise ValueError(f"File format {fmt} is not recognized.")

        savepath.parent.mkdir(parents=True, exist_ok=True)

        if CatalogueFileFormat.CSV.name.lower() in fmt:
            self.write_csv(savepath.parent / (savepath.stem + ".csv"))
        if CatalogueFileFormat.JSON.name.lower() in fmt:
            self.write_json(savepath.parent / (savepath.stem + ".json"))
        if CatalogueFileFormat.MAT.name.lower() in fmt:
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
