# Underwater Acoustic Data Utilities

This package contains general reading and formatting utilities for common underwater acoustic data formats.

> [!WARNING]
> This package is currently under development. The API is subject to change.

## Data Formats

### SHRU

Woods Hole Oceanographic Institution's (WHOI) Single-Hydrophone Recording Unit (SHRU) binary data format. All code has been adapted from Y.T. Lin's original MATLAB code.

> [!NOTE]
> `SHRU` data capability is currently implemented.

### SIO

Scripps Institution of Oceanography's (SIO) binary data format. Code has been adapted from Hunter Akins and Aaron Thode, with contributions from Geoff Edelman, James Murray, and Dave Ensberg.

> [!NOTE]
> `SIO` data capability is not yet implemented.

### WAV

The common `.wav` audio file format. This is a standard format for storing audio data, and is widely supported by many software packages.

> [!NOTE]
> `WAV` data capability is not yet implemented.

## Installation

To install the package, run the following command:

```bash
pip install git+https://github.com/NeptuneProjects/uwac-data-utils.git
```

## Usage

The API is designed to efficiently query large amounts of acoustic data.
This is done by constructing a inventory of the data files, which includes metadata such as the file path, start time, and sampling rate.
The inventory can then be queried to retrieve data in a specific time range.
Two main classes are provided for this purpose: `datautils.query.FileInfoQuery` and `datautils.query.InventoryQuery`.
Each has its own convenient constructor functions which read query data from TOML files.

### Data organization

Data organization is flexible due to the reliance on glob patterns to match files.
However, data are generally assumed to be organized into distinct categories such as arrays, events, or sensors.
For example, data might be organized into a directory structure according to multiple sensor arrays:

```
/data
├── ARRAY001
│   ├── 2009-05-22
│   │   ├── 11-00-00.D23
│   │   ├── 11-00-01.D23
│   │   ├── 11-00-02.D23
│   │   ├── ...
│   │   ├── 15-00-00.D23
│   │   ├── 15-00-01.D23
│   │   └── 15-00-02.D23
│   ├── 2009-05-23
│   │   ├── 11-00-00.D23
│   │   ├── 11-00-01.D23
│   │   ├── 11-00-02.D23
│   │   ├── ...
│   │   ├── 15-00-00.D23
│   │   ├── 15-00-01.D23
│   │   └── 15-00-02.D23
│   └── ...
├── ARRAY002
│   ├── ...
└── ...
```

### Inventoryuing data files with `DatasetInventory`

#### Building a inventory

To build a inventory, information about the files must first be gathered.
This is done by constructing a `datautils.query.FileInfoQuery` object, which is then used to build the inventory.
The `FileInfoQuery` object can be constructed direclty in a Python script, or more conveniently using a TOML file:

```toml
[ARRAY001]

[ARRAY001.data]
directory = "/data"
glob_pattern = "ARRAY001*/*.D23"
destination = "/data/cat"
file_format = "SHRU" # Optional

[ARRAY001.clock] # Optional
time_check_0 = 2009-05-22T11:00:00
time_check_1 = 2009-05-22T15:00:00
offset_0 = 200e-6
offset_1 = -1.080e-3

[ARRAY001.hydrophone] # Optional
fixed_gain = [3.0, 3.0, 3.0, 3.0] # dB
sensitivity = [200.0, 200.0, 200.0, 200.0] # dB
serial_number = [0, 1, 2, 3]

# [ARRAY002]
# ...
```
The above example specifies the following:
- `[ARRAY001]`: A unique identifier for a specific sensor or array.
- `[ARRAY001.data]`: Data file and inventory information.
  - `directory`: The directory where the data files are located.
  - `glob_pattern`: A glob pattern to match the data files.
  - `destination`: The directory where the inventory will be saved.
  - `file_format`: The format of the data files. This is optional, and will be inferred from the file extension if not provided.
- `[ARRAY001.clock]`: Optional metadata about the clock used to timestamp the data. This section is used to correct for clock drift. If not provided, clock drift is assumed to be zero.
  - `time_check_0` and `time_check_1`: The start and end times of the data to be included in the inventory. This is optional, and will include all data if not provided.
  - `offset_0` and `offset_1`: The time offsets for the start and end times. This is optional, and will default to 0 if not provided.
- `[ARRAY001.hydrophone]`: Optional metadata about the hydrophone. This section is used to correct for hydrophone gain and sensitivity. If not provided, no corrections are applied.

Multiple arrays can be specified in the same file, as denoted by the additional `[ARRAY002]` section.
The inventory is contained within the `datautils.inventory.DatasetInventory` object as a `polars.DataFrame`.
To build a inventorys for each of the arrays specified in the above TOML file, use the following code:

```python
from datautils.inventory import DatasetInventory
from datautils.query import load_file_query

queries = load_file_query("path/to/query.toml")
inventorys = []
for query in queries:
    inventorys.append(DatasetInventory.build_inventory(query))

[print(inventory.df) for inventory in inventorys]
```

#### Saving a inventory

The `DatasetInventory` object has a `save` method save the inventory to disk in `.csv`, `.json`, or `.MAT` formats.

```python
# Save the inventory to a CSV file (default)
inventory.save("path/to/inventory", fmt="csv")

# Save the inventory to a JSON file
inventory.save("path/to/inventory", fmt="json")

# Save the inventory to a MAT file
inventory.save("path/to/inventory", fmt="mat")

# Save the inventory to all three formats
inventory.save("path/to/inventory", fmt=["csv", "json", "mat"])
```

#### Building and saving multiple inventorys

The `datautils.query.build_and_save_inventorys` function can be used to build and save multiple inventorys at once.

```python
from datautils.inventory import build_and_save_inventorys
from datautils.query import load_file_query

queries = load_file_query("path/to/query.toml")
build_and_save_inventorys(queries, fmt=["csv", "json", "mat"])
```

#### Loading a inventory

A inventory can be loaded from disk using the `datautils.inventory.load_inventory` function.

```python
from datautils.inventory import DatasetInventory

inventory = DatasetInventory().load_inventory("path/to/inventory.csv")
```


### Reading data using a `InventoryQuery`

Once a inventory has been built, data can be read from disk by querying the inventory and returning data.

#### Constructing a inventory query

Here again, the query can be constructed directly in a Python script, or more conveniently using a TOML file:

```toml
[ARRAY001]
inventory = "path/to/inventory.csv"
destination = "." # Not currently used
time_start = 2009-05-22T12:00:00
time_end = 2009-05-22T13:00:00
channels = [0, 1, 2, 3]

# [ARRAY002]
# ...
```
The above example specifies the following:
- `[ARRAY001]`: A unique identifier for a specific sensor or array.
  - `inventory`: The path to the inventory file.
  - `destination`: The directory where the data will be saved. This is not currently used.
  - `time_start` and `time_end`: The start and end times of the data to be read.
  - `channels`: The channels to be read.

Multiple arrays can be specified in the same file, as denoted by the additional `[ARRAY002]` section.

#### Reading data

To read data for each of the arrays specified in the above TOML file, use the following code:

```python
from datautils.data import read
from datautils.query import load_inventory_query

queries = load_inventory_query("path/to/query.toml")
for query in queries:
    cat_query = datautils.query.InventoryQuery(
          inventory=query.inventory,
          destination=query.destination,
          time_start=query.time_start,
          time_end=query.time_end,
          channels=query.channels,
      )
      datastream = read(cat_query)

      # Do something with the datastream
      print(datastream.waveform)
```
The returned `datastream` is a `datautils.data.DataStream` object, which contains the following attributes:
- `waveform`: The data as a `numpy.ndarray` with shape `(n_samples, n_channels)`.
- `stats`: An object containing relevant metadata about the data.
- `time`: The time of the data as a `numpy.ndarray` with shape `(n_samples,)`.

## Acknowledgements

Package written by William Jenkins, with code derived from:
- Y.T. Lin
- Hunter Akins, Aaron Thode, Geoff Edelman, James Murray, and Dave Ensberg
- The ObsPy Development Team
