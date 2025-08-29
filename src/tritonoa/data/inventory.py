# TODO: Replace hydrophone specs w/ signal params

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import logging
from pathlib import Path
from typing import Callable, Protocol

from polars import DataFrame
import polars as pl
from tqdm import tqdm
import scipy.io

from tritonoa.data.formats.base import DataRecord
from tritonoa.data.formats import factory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import (
    TIME_PRECISION,
    ClockParameters,
    convert_yydfrac_to_timestamp,
    convert_dtarray_to_ydarray,
)


MAT_KEYS = ["__header__", "__version__", "__globals__"]


@dataclass
class Header(Protocol):
    """Protocol for file-format--specific header objects."""

    ...


class InventoryFileFormat(StrEnum):
    """File formats for record inventorys."""

    BIN = "bin"
    CSV = "csv"
    JSON = "json"
    MAT = "mat"

    def __contains__(cls, item) -> bool:
        print([member.value for member in cls])
        return item in (member.value for member in cls)


class Inventory:
    def __init__(self, records: list[DataRecord] | None = None):
        self.records = records
        if records is not None:
            self.df = self._records_to_df()

    def build(
        self,
        dataset_path: Path,
        glob_pattern: str = "*",
        clock_params: ClockParameters = ClockParameters(),
        conditioner: SignalParams = SignalParams(),
        file_format: str | None = None,
        record_fmt_callback: Callable | None = None,
    ) -> DataFrame:
        def _process_file(file):
            if file_format is None:
                reader = factory.get_reader(file.suffix)
                formatter = factory.get_formatter(file.suffix)
            else:
                reader = factory.get_reader(file_format)
                formatter = factory.get_formatter(file_format)

            headers = reader.read_headers(file)

            records_from_file = []
            for i, header in enumerate(headers):
                records_from_file.append(
                    formatter.format_record(
                        filename=file,
                        record_number=i,
                        header=header,
                        clock=clock_params,
                        conditioner=conditioner,
                    )
                )

            if record_fmt_callback is not None:
                corrected_records = record_fmt_callback(records_from_file)
            else:
                corrected_records = formatter.callback(records_from_file)
            logging.debug(f"{len(corrected_records)} records processed from {file}.")
            return corrected_records

        files = self._get_files(dataset_path, glob_pattern)

        with ThreadPoolExecutor(max_workers=len(files)) as executor:
            results = list(
                tqdm(
                    executor.map(_process_file, files),
                    total=len(files),
                    desc="Processing files",
                )
            )

        records = []
        [records.extend(result) for result in results]
        self.records = records
        self.df = self._records_to_df()
        return self.df

    def _format_df_for_csv(self, df: DataFrame) -> DataFrame:
        """Format Polars DataFrame for CSV output.

        Args:
            df (DataFrame): Polars DataFrame.

        Returns:
            DataFrame: Polars DataFrame.
        """
        def _to_list(lst: float | list) -> str:
            return ",".join([str(i) for i in lst])

        return df.with_columns(
            pl.col("adc_vref").map_elements(
                _to_list,
                return_dtype=pl.String,
            ),
            pl.col("gain").map_elements(
                _to_list,
                return_dtype=pl.String,
            ),
            pl.col("sensitivity").map_elements(
                _to_list,
                return_dtype=pl.String,
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
        logging.debug(f"{len(files)} files found for record inventory.")
        return files

    def load(self, filepath: Path) -> DataFrame:
        """Load the record inventory from file.

        Args:
            filepath (Path): Path to the inventory file.

        Returns:
            DatasetInventory: Record inventory.

        Raises:
            ValueError: If file format is not recognized.
        """
        extension = filepath.suffix[1:].lower()

        if extension not in InventoryFileFormat or extension == "":
            raise ValueError(f"File format '{extension}' is not recognized.")

        if extension == InventoryFileFormat.BIN.value:
            self._read_bin(filepath)
        if extension == InventoryFileFormat.CSV.value:
            self._read_csv(filepath)
        if extension == InventoryFileFormat.JSON.value:
            self._read_json(filepath)
        if extension == InventoryFileFormat.MAT.value:
            self._read_mat(filepath)
        return self.df

    def _read_bin(self, filepath: Path) -> None:
        """Read the record inventory from binary file.

        Args:
            filepath (Path): Path to the inventory file.

        Returns:
            None
        """
        self.df = DataFrame.deserialize(filepath, format="binary")

    def _read_csv(self, filepath: Path) -> None:
        """Read the record inventory from CSV file.

        Args:
            filepath (Path): Path to the inventory file.

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
            pl.col("adc_vref").map_elements(
                partial(_str_to_list, dtype=float), return_dtype=pl.List(float)
            ),
            pl.col("gain").map_elements(
                partial(_str_to_list, dtype=float), return_dtype=pl.List(float)
            ),
            pl.col("sensitivity").map_elements(
                partial(_str_to_list, dtype=float), return_dtype=pl.List(float)
            ),
        )

    def _read_json(self, filepath: Path) -> None:
        """Read the record inventory from JSON file.

        Args:
            filepath (Path): Path to the inventory file.

        Returns:
            None
        """
        self.df = DataFrame.deserialize(filepath, format="json")

    def _read_mat(self, filepath: Path) -> None:
        """Read the record inventory from MAT file.

        Args:
            filepath (Path): Path to the inventory file.

        Returns:
            None
        """
        mdict = scipy.io.loadmat(filepath)
        self.records = self._mdict_to_records(mdict)
        self.df = self._records_to_df()

    def _records_to_df(self) -> DataFrame:
        """Convert records to Polars DataFrame.

        Returns:
            DataFrame: Polars DataFrame.
        """
        return (
            DataFrame(self._records_to_dfdict())
            .with_columns(pl.col("timestamp").cast(pl.Datetime(TIME_PRECISION)))
            .with_columns(pl.col("timestamp_orig").cast(pl.Datetime(TIME_PRECISION)))
            .sort(by="timestamp")
            .with_row_count()
        )

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

        filenames = [
            i.strip() for i in cat_data["filenames"][0][0].astype(str).tolist()
        ]
        timestamp_data = cat_data["timestamps"][0][0].transpose(2, 1, 0)
        timestamp_orig_data = cat_data["timestamps_orig"][0][0].transpose(2, 1, 0)
        rhfs_orig = float(cat_data["rhfs_orig"][0][0].squeeze())
        rhfs = float(cat_data["rhfs"][0][0].squeeze())
        adc_vref = cat_data["adc_vref"][0][0].squeeze().astype(float).tolist()
        fixed_gain = cat_data["gain"][0][0].squeeze().astype(float).tolist()
        hydrophone_sensitivity = (
            cat_data["sensitivity"][0][0].squeeze().astype(float).tolist()
        )

        records = []
        for i, f in enumerate(filenames):
            n_records = cat_data["timestamps"][0][0].transpose(2, 1, 0).shape[1]
            filename = Path(f) if not isinstance(f, Path) else f
            file_format = factory.validate_file_format(filename.suffix)

            for j in range(n_records):
                timestamp = convert_yydfrac_to_timestamp(*timestamp_data[i][j])
                timestamp_orig = convert_yydfrac_to_timestamp(
                    *timestamp_orig_data[i][j]
                )
                records.append(
                    DataRecord(
                        filename=filename,
                        record_number=j,
                        file_format=file_format,
                        timestamp=timestamp,
                        timestamp_orig=timestamp_orig,
                        sampling_rate=rhfs,
                        sampling_rate_orig=rhfs_orig,
                        adc_vref=adc_vref,
                        gain=fixed_gain,
                        sensitivity=hydrophone_sensitivity,
                    )
                )

        return records

    def _records_to_dfdict(self) -> dict:
        """Convert records to dictionary.

        The dictionary is used to construct the object's DataFrame.

        Returns:
            dict: Dictionary of records.
        """
        filename = []
        record_number = []
        file_format = []
        npts = []
        nch = []
        timestamp = []
        timestamp_orig = []
        sampling_rate = []
        sampling_rate_orig = []
        adc_vref = []
        gain = []
        sensitivity = []

        for record in self.records:
            filename.append(str(record.filename))
            record_number.append(record.record_number)
            file_format.append(record.file_format)
            npts.append(record.npts)
            nch.append(record.nch)
            if not isinstance(record.timestamp, float):
                timestamp.append(record.timestamp.astype("int64"))
            else:
                timestamp.append(record.timestamp)
            if not isinstance(record.timestamp_orig, float):
                timestamp_orig.append(record.timestamp_orig.astype("int64"))
            else:
                timestamp_orig.append(record.timestamp_orig)
            sampling_rate.append(record.sampling_rate)
            sampling_rate_orig.append(record.sampling_rate_orig)
            adc_vref.append(record.adc_vref)
            gain.append(record.gain)
            sensitivity.append(record.sensitivity)
        
        return {
            "filename": filename,
            "record_number": record_number,
            "file_format": file_format,
            "npts": npts,
            "nch": nch,
            "timestamp": timestamp,
            "timestamp_orig": timestamp_orig,
            "sampling_rate": sampling_rate,
            "sampling_rate_orig": sampling_rate_orig,
            "adc_vref": adc_vref,
            "gain": gain,
            "sensitivity": sensitivity,
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
            self.records[0].file_format: {
                "filenames": filenames,
                "timestamps": convert_dtarray_to_ydarray(timestamps),
                "timestamps_orig": convert_dtarray_to_ydarray(timestamps_orig),
                "rhfs_orig": self.records[0].sampling_rate_orig,
                "rhfs": self.records[0].sampling_rate,
                "adc_vref": self.records[0].adc_vref,
                "gain": self.records[0].gain,
                "sensitivity": self.records[0].sensitivity,
            }
        }

    def save(self, path: Path | str):
        """Save the inventory to file.

        Args:
            savepath (Path): Path to save the inventory file.
            fmt (str | list[str], optional): File format(s) to save the inventory. Defaults to "csv".

        Raises:
            ValueError: If file format is not recognized.
        """
        savepath = Path(path)
        extension = savepath.suffix[1:].lower()
        if extension is None or extension == "":
            extension = InventoryFileFormat.CSV.value
        if extension not in InventoryFileFormat:
            raise ValueError(f"File format '{extension}' is not recognized.")

        savepath.parent.mkdir(parents=True, exist_ok=True)

        if extension == InventoryFileFormat.BIN.name.lower():
            self.write_binary(savepath.parent / (savepath.stem + ".bin"))
        if extension == InventoryFileFormat.CSV.name.lower():
            self.write_csv(savepath.parent / (savepath.stem + ".csv"))
        if extension == InventoryFileFormat.JSON.name.lower():
            self.write_json(savepath.parent / (savepath.stem + ".json"))
        if extension == InventoryFileFormat.MAT.name.lower():
            self.write_mat(savepath.parent / (savepath.stem + ".mat"))

    def write_binary(self, savepath: Path):
        """Write the inventory to binary file.

        Args:
            savepath (Path): Path to save the inventory file.

        Returns:
            None
        """
        self.df.serialize(savepath, format="binary")

    def write_csv(self, savepath: Path):
        """Write the inventory to CSV file.

        Args:
            savepath (Path): Path to save the inventory file.

        Returns:
            None
        """
        df_out = self._format_df_for_csv(self.df)
        df_out.write_csv(savepath)

    def write_json(self, savepath: Path):
        """Write the inventory to JSON file.

        Args:
            savepath (Path): Path to save the inventory file.

        Returns:
            None
        """
        self.df.serialize(savepath, format="json")

    def write_mat(self, savepath: Path):
        """Write the inventory to MAT file.

        Args:
            savepath (Path): Path to save the inventory file.

        Returns:
            None
        """
        mdict = self._records_to_mdict()
        scipy.io.savemat(savepath, mdict)
