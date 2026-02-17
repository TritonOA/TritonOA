# Calibration System Usage Examples

This document provides comprehensive examples of using the new modular calibration system.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Integration with Existing Readers](#integration-with-existing-readers)
3. [Migration Guide](#migration-guide)
4. [Advanced Examples](#advanced-examples)

## Basic Usage

### Example 1: Simple Time-Domain Calibration

```python
from tritonoa.data.calibration import CalibrationChain
from tritonoa.data.calibration.steps import ADCConversion, Gain, Sensitivity
import numpy as np

# Create calibration chain
chain = CalibrationChain([
    ADCConversion(adc_vref=2.5, adc_bits=24),
    Gain(gain_db=20.0),
    Sensitivity(sensitivity_db=-170.0)
])

# Apply to raw data
raw_data = np.random.randint(-2**23, 2**23, size=(4, 48000))
result = chain.apply(raw_data, sampling_rate=48000)

print(f"Output units: {result.units}")  # 'uPa'
print(f"Output shape: {result.shape}")
print(f"Steps applied: {result.metadata.steps_applied}")
```

### Example 2: Loading from Configuration File

```python
from tritonoa.data.calibration import CalibrationFactory

# Load calibration from YAML
chain = CalibrationFactory.from_yaml(
    "src/tritonoa/data/calibration/configs/sensors/icListen_HF_example.yaml"
)

# Print summary
print(chain.summary())

# Apply calibration
result = chain.apply(raw_data, sampling_rate=192000)

# Access metadata
metadata = result.metadata.to_dict()
print(f"Calibration version: {metadata['version']}")
print(f"Calibration date: {metadata['date']}")
```

### Example 3: Frequency-Dependent Calibration

```python
from tritonoa.data.calibration import CalibrationChain
from tritonoa.data.calibration.steps import (
    ADCConversion,
    Gain,
    FrequencyResponse
)

# Define frequency-dependent sensitivity
chain = CalibrationChain([
    ADCConversion(adc_vref=2.5, adc_bits=24),
    Gain(gain_db=26.0),
    FrequencyResponse(
        frequencies=[10, 100, 1000, 10000, 100000],
        magnitude_db=[-170, -168, -165, -168, -175],
        phase_deg=[0, 2, 5, 15, 30],
        interpolation="cubic"
    )
])

result = chain.apply(raw_data, sampling_rate=192000)
```

## Integration with Existing Readers

### Option 1: Modify `condition_data` Method

Update existing reader classes to optionally use the new calibration system:

```python
from pathlib import Path
import numpy as np
from tritonoa.data.formats.base import BaseReader
from tritonoa.data.calibration import CalibrationChain, CalibrationFactory

class SHRUReader(BaseReader):
    """Updated SHRU reader with optional new calibration system."""

    def condition_data(
        self,
        data: np.ndarray,
        conditioner=None,
        calibration_chain: CalibrationChain | None = None,
        calibration_file: Path | str | None = None,
    ) -> tuple[np.ndarray, str]:
        """Condition data using either legacy or new calibration system.

        Args:
            data: Raw data array.
            conditioner: Legacy SignalParams (for backward compatibility).
            calibration_chain: CalibrationChain instance (new system).
            calibration_file: Path to calibration config file (new system).

        Returns:
            Tuple of (calibrated_data, units).
        """
        # New calibration system takes precedence
        if calibration_chain is not None:
            result = calibration_chain.apply(data, sampling_rate=self.sampling_rate)
            return result.data, result.units

        if calibration_file is not None:
            chain = CalibrationFactory.from_yaml(calibration_file)
            result = chain.apply(data, sampling_rate=self.sampling_rate)
            return result.data, result.units

        # Fall back to legacy system
        if conditioner is not None:
            # ... existing legacy calibration code ...
            pass

        return data, "counts"
```

### Option 2: Use Calibration in Reader Wrapper

Create a wrapper that applies calibration after reading:

```python
from tritonoa.data.formats.shru import SHRUReader
from tritonoa.data.calibration import CalibrationFactory

def read_shru_calibrated(
    file_path: Path,
    calibration_config: Path | str,
    **read_kwargs
):
    """Read SHRU file with new calibration system.

    Args:
        file_path: Path to SHRU data file.
        calibration_config: Path to calibration config file.
        **read_kwargs: Additional arguments passed to SHRUReader.read()

    Returns:
        Calibrated DataStream.
    """
    # Read raw data using existing reader
    reader = SHRUReader()
    raw_data, header = reader.read_raw_data(file_path, **read_kwargs)

    # Load calibration
    chain = CalibrationFactory.from_yaml(calibration_config)

    # Apply calibration
    result = chain.apply(raw_data, sampling_rate=header.rhfs)

    # Create DataStream with calibrated data
    from tritonoa.data.stream import DataStream, DataStreamStats
    return DataStream(
        stats=DataStreamStats(
            channels=list(range(result.num_channels)),
            time_init=_get_timestamp(header),
            sampling_rate=header.rhfs,
            units=result.units,
            metadata=result.metadata.to_dict(),
        ),
        data=result.data,
    )

# Usage
ds = read_shru_calibrated(
    "data.D23",
    "configs/sensors/shru_simple_example.yaml"
)
```

### Option 3: Registry-Based Lookup

Use the registry to automatically find calibration based on sensor metadata:

```python
from tritonoa.data.calibration import CalibrationRegistry

# Initialize registry (do once at startup)
registry = CalibrationRegistry()
registry.scan_directory("src/tritonoa/data/calibration/configs/sensors/")

def read_with_auto_calibration(
    file_path: Path,
    manufacturer: str,
    model: str,
    serial_number: str | None = None,
):
    """Read data with automatic calibration lookup."""
    # Read raw data
    reader = SHRUReader()
    raw_data, header = reader.read_raw_data(file_path)

    # Look up calibration
    try:
        chain = registry.get(
            manufacturer=manufacturer,
            model=model,
            serial_number=serial_number
        )
    except KeyError:
        print(f"No calibration found for {manufacturer} {model}")
        # Fall back to uncalibrated or legacy system
        return raw_data, "counts"

    # Apply calibration
    result = chain.apply(raw_data, sampling_rate=header.rhfs)
    return result.data, result.units

# Usage
data, units = read_with_auto_calibration(
    Path("data.D23"),
    manufacturer="WHOI",
    model="SHRU",
    serial_number="SHRU001"
)
```

## Migration Guide

### Phase 1: Create Calibration Configurations

1. For each sensor/deployment, create a calibration config file:

```bash
# Create config for your SHRU deployment
cat > configs/sensors/shru_deployment_2024.yaml << EOF
sensor:
  manufacturer: "WHOI"
  model: "SHRU"
  serial_number: "SHRU001"

calibration:
  version: "1.0"
  date: "2024-01-15"
  chain:
    - type: "adc_conversion"
      adc_vref: 2.5
      adc_bits: 24
      signed: true
    - type: "gain"
      gain_db: [20.0, 20.0, 20.0, 20.0]
    - type: "sensitivity"
      sensitivity_db: [-165.0, -165.0, -165.0, -165.0]
EOF
```

2. Test the configuration:

```python
from tritonoa.data.calibration import CalibrationFactory

chain = CalibrationFactory.from_yaml("configs/sensors/shru_deployment_2024.yaml")
print(chain.summary())

# Test with sample data
import numpy as np
test_data = np.random.randint(-2**23, 2**23, size=(4, 1000))
result = chain.apply(test_data, sampling_rate=48000)
assert result.units == "uPa"
```

### Phase 2: Dual System (Backward Compatible)

Modify your data processing scripts to support both systems:

```python
def process_data(
    file_path: Path,
    use_new_calibration: bool = True,
    calibration_config: Path | None = None,
    legacy_params: SignalParams | None = None,
):
    """Process data with optional new calibration system."""
    reader = SHRUReader()

    if use_new_calibration:
        raw_data, header = reader.read_raw_data(file_path)
        chain = CalibrationFactory.from_yaml(calibration_config)
        result = chain.apply(raw_data, sampling_rate=header.rhfs)
        return result.data, result.units
    else:
        # Use legacy system
        return reader.read(file_path, conditioner=legacy_params)
```

### Phase 3: Full Migration

Once confident in the new system:

1. Update `BaseReader.condition_data()` signature to accept `CalibrationChain`
2. Deprecate `SignalParams` in favor of calibration configs
3. Update all format readers to use new system
4. Remove legacy calibration code

## Advanced Examples

### Example 4: Multi-Channel with Different Calibrations

```python
from tritonoa.data.calibration.steps import FrequencyResponse

# Channel 0-1: One hydrophone type
chain_hydro1 = CalibrationChain([
    ADCConversion(adc_vref=[2.5, 2.5], adc_bits=24),
    FrequencyResponse(
        frequencies=[10, 1000, 10000],
        magnitude_db=[-170, -165, -170],
    )
])

# Channel 2-3: Different hydrophone type
chain_hydro2 = CalibrationChain([
    ADCConversion(adc_vref=[2.5, 2.5], adc_bits=24),
    FrequencyResponse(
        frequencies=[10, 1000, 10000],
        magnitude_db=[-175, -172, -176],
    )
])

# Apply to respective channels
result1 = chain_hydro1.apply(raw_data[0:2, :], sampling_rate=fs)
result2 = chain_hydro2.apply(raw_data[2:4, :], sampling_rate=fs)

# Combine
calibrated_data = np.vstack([result1.data, result2.data])
```

### Example 5: Dynamic Calibration Selection

```python
def get_calibration_for_deployment(
    deployment_id: str,
    timestamp: np.datetime64
) -> CalibrationChain:
    """Select appropriate calibration based on deployment and time."""
    registry = CalibrationRegistry()
    registry.scan_directory("configs/sensors/")

    # Get all calibrations for this deployment
    calibrations = [
        cal for cal in registry.list_calibrations()
        if cal['metadata'].get('deployment_id') == deployment_id
    ]

    # Select based on timestamp and validity
    for cal_info in calibrations:
        cal_date = np.datetime64(cal_info['date'])
        if 'valid_until' in cal_info and cal_info['valid_until']:
            valid_until = np.datetime64(cal_info['valid_until'])
            if cal_date <= timestamp < valid_until:
                return registry.get(identifier=cal_info['identifier'])

    raise ValueError(f"No valid calibration found for {deployment_id} at {timestamp}")
```

### Example 6: Batch Processing with Registry

```python
from pathlib import Path
from tritonoa.data.calibration import CalibrationRegistry

def batch_process_deployment(
    data_dir: Path,
    output_dir: Path,
    calibration_dir: Path,
):
    """Batch process all files in a deployment."""
    # Load calibrations
    registry = CalibrationRegistry()
    registry.scan_directory(calibration_dir)

    # Process each file
    for data_file in data_dir.glob("*.D23"):
        print(f"Processing {data_file.name}")

        # Read raw data
        reader = SHRUReader()
        raw_data, header = reader.read_raw_data(data_file)

        # Get calibration (assumes metadata in header)
        chain = registry.get(
            manufacturer="WHOI",
            model="SHRU",
            # Could extract serial from filename or header
        )

        # Apply calibration
        result = chain.apply(raw_data, sampling_rate=header.rhfs)

        # Save calibrated data
        output_file = output_dir / f"{data_file.stem}_calibrated.npy"
        np.save(output_file, result.data)

        # Save metadata
        metadata_file = output_dir / f"{data_file.stem}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(result.metadata.to_dict(), f, indent=2)
```

### Example 7: Creating Calibration from Existing SignalParams

```python
from tritonoa.data.signal import SignalParams
from tritonoa.data.calibration import CalibrationChain
from tritonoa.data.calibration.steps import ADCConversion, Gain, Sensitivity

def convert_legacy_to_new_calibration(
    params: SignalParams,
    adc_bits: int = 24,
) -> CalibrationChain:
    """Convert legacy SignalParams to new CalibrationChain."""
    return CalibrationChain([
        ADCConversion(
            adc_vref=params.adc_vref,
            adc_bits=adc_bits,
            signed=True
        ),
        Gain(gain_db=params.gain, invert=False),
        Sensitivity(sensitivity_db=params.sensitivity)
    ])

# Usage
legacy_params = SignalParams(
    adc_vref=[2.5, 2.5],
    gain=[20.0, 20.0],
    sensitivity=[-165.0, -165.0]
)

new_chain = convert_legacy_to_new_calibration(legacy_params)

# Optionally save to config file
from tritonoa.data.calibration import CalibrationFactory
CalibrationFactory.to_yaml(new_chain, "converted_calibration.yaml")
```

## Testing and Validation

### Verify Calibration Accuracy

```python
import matplotlib.pyplot as plt

def validate_calibration(
    chain: CalibrationChain,
    reference_signal: np.ndarray,
    reference_level_dB: float,
    sampling_rate: float,
):
    """Validate calibration against known reference signal."""
    # Apply calibration
    result = chain.apply(reference_signal, sampling_rate)

    # Calculate RMS level
    rms = np.sqrt(np.mean(result.data**2))
    level_dB = 20 * np.log10(rms / 1.0)  # dB re 1 µPa

    # Compare to expected
    error_dB = level_dB - reference_level_dB

    print(f"Expected: {reference_level_dB:.2f} dB re 1 µPa")
    print(f"Measured: {level_dB:.2f} dB re 1 µPa")
    print(f"Error: {error_dB:.2f} dB")

    return abs(error_dB) < 1.0  # Within 1 dB tolerance
```

### Compare Legacy vs New System

```python
def compare_calibration_methods(
    raw_data: np.ndarray,
    sampling_rate: float,
    legacy_params: SignalParams,
    calibration_file: Path,
):
    """Compare output from legacy and new calibration systems."""
    # Legacy system
    from tritonoa.data.signal import (
        convert_counts_to_voltage,
        convert_voltage_to_pressure,
        db_to_linear
    )

    linear_gain = db_to_linear(legacy_params.gain)
    linear_sens = db_to_linear(legacy_params.sensitivity)

    legacy_voltage, _ = convert_counts_to_voltage(
        raw_data, linear_gain, legacy_params.adc_vref, 2**23
    )
    legacy_pressure, _ = convert_voltage_to_pressure(legacy_voltage, linear_sens)

    # New system
    chain = CalibrationFactory.from_yaml(calibration_file)
    result = chain.apply(raw_data, sampling_rate)

    # Compare
    diff = np.abs(legacy_pressure - result.data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff:.6e} µPa")
    print(f"Mean difference: {mean_diff:.6e} µPa")
    print(f"Relative error: {mean_diff / np.mean(np.abs(legacy_pressure)) * 100:.3f}%")

    return np.allclose(legacy_pressure, result.data, rtol=1e-6)
```
