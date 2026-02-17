# Modular Signal Calibration System

A flexible, extensible calibration system for underwater acoustic sensors that supports both time-domain and frequency-domain corrections.

## Overview

This calibration system addresses the key challenges in hydrophone signal conditioning:

- ✅ **Modular Design**: Composable calibration steps following design patterns
- ✅ **Configuration-Driven**: Define calibrations in YAML/JSON files
- ✅ **Frequency-Dependent**: Support for non-linear magnitude and phase corrections
- ✅ **Manufacturer-Agnostic**: Easy to add new sensors and calibration procedures
- ✅ **Traceable**: Full metadata tracking of calibration provenance
- ✅ **Extensible**: Plugin architecture for custom calibration steps
- ✅ **Backward Compatible**: Works alongside existing code

## Key Features

### 1. Separation of Concerns
- Calibration logic is independent of file format readers
- Each calibration step has a single, well-defined responsibility
- Easy to test and validate individual components

### 2. Frequency-Domain Calibration
```python
# Correct for frequency-dependent hydrophone response
FrequencyResponse(
    frequencies=[10, 100, 1000, 10000],
    magnitude_db=[-170, -168, -165, -170],
    phase_deg=[0, 2, 5, 15],
    interpolation="cubic"
)
```

### 3. Chain of Responsibility Pattern
```python
# Compose multiple calibration steps
chain = CalibrationChain([
    ADCConversion(adc_vref=2.5, adc_bits=24),
    Gain(gain_db=20.0),
    FrequencyResponse(...),
    Sensitivity(sensitivity_db=-170.0)
])
```

### 4. Configuration Files
```yaml
# Define calibrations in YAML
sensor:
  manufacturer: "Ocean Sonics"
  model: "icListen HF"
  serial_number: "1234"

calibration:
  version: "2.1"
  date: "2024-01-15"
  chain:
    - type: "adc_conversion"
      adc_vref: 2.5
      adc_bits: 24
    - type: "frequency_response"
      frequencies: [10, 100, 1000, 10000]
      magnitude_db: [-170, -168, -165, -170]
      phase_deg: [0, 2, 5, 15]
```

### 5. Registry System
```python
# Automatic calibration lookup
registry = CalibrationRegistry()
registry.scan_directory("configs/sensors/")

chain = registry.get(
    manufacturer="Ocean Sonics",
    model="icListen HF"
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CalibrationConfig                      │
│              (YAML/JSON configuration)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CalibrationFactory                          │
│     (Creates calibration chains from config)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CalibrationChain                            │
│         (Executes sequence of steps)                     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬───────────────┐
         ▼                       ▼               ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ TimeDomainStep   │  │FrequencyDomain   │  │  CustomStep      │
│ - ADC Conversion │  │  Step            │  │ - User-defined   │
│ - Gain           │  │ - Phase corr.    │  │   corrections    │
│ - Sensitivity    │  │ - Magnitude corr.│  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Quick Start

### Installation

The calibration system is part of the `tritonoa` package:

```python
from tritonoa.data.calibration import (
    CalibrationChain,
    CalibrationFactory,
    CalibrationRegistry,
)
```

### Basic Usage

```python
from tritonoa.data.calibration import CalibrationFactory
import numpy as np

# Load calibration from config file
chain = CalibrationFactory.from_yaml("configs/sensors/icListen_HF_example.yaml")

# Apply to raw data
raw_data = np.random.randint(-2**23, 2**23, size=(4, 48000))
result = chain.apply(raw_data, sampling_rate=48000)

print(f"Units: {result.units}")  # Always 'uPa'
print(f"Shape: {result.shape}")
print(f"Steps: {result.metadata.steps_applied}")
```

### Creating Calibrations Programmatically

```python
from tritonoa.data.calibration import CalibrationChain
from tritonoa.data.calibration.steps import (
    ADCConversion,
    Gain,
    FrequencyResponse,
)

chain = CalibrationChain([
    ADCConversion(adc_vref=2.5, adc_bits=24),
    Gain(gain_db=20.0),
    FrequencyResponse(
        frequencies=[10, 100, 1000, 10000, 100000],
        magnitude_db=[-170, -168, -165, -168, -175],
        phase_deg=[0, 2, 5, 15, 30],
        interpolation="cubic"
    )
])

result = chain.apply(raw_data, sampling_rate=192000)
```

## Available Calibration Steps

### Time-Domain Steps

| Step | Description | Parameters |
|------|-------------|------------|
| `ADCConversion` | Convert counts to voltage | `adc_vref`, `adc_bits`, `signed` |
| `Gain` | Apply gain correction | `gain_db`, `invert` |
| `Sensitivity` | Convert voltage to pressure | `sensitivity_db`, `reference_pressure` |
| `ScalarMultiply` | Generic scalar multiplication | `factor` |
| `Offset` | Add/subtract DC offset | `offset` |

### Frequency-Domain Steps

| Step | Description | Parameters |
|------|-------------|------------|
| `FrequencyResponse` | Magnitude & phase correction | `frequencies`, `magnitude_db`, `phase_deg`, `interpolation` |
| `HighPassFilter` | High-pass filter | `cutoff_freq`, `order`, `filter_type` |
| `LowPassFilter` | Low-pass filter | `cutoff_freq`, `order`, `filter_type` |

## Documentation

- [Configuration File Format](configs/README.md) - How to write calibration configs
- [Usage Examples](USAGE_EXAMPLES.md) - Comprehensive usage examples
- [API Documentation](../../README.md) - Full API reference

## Design Patterns Used

1. **Strategy Pattern**: Different calibration strategies for different sensors
2. **Chain of Responsibility**: Composable calibration steps
3. **Factory Pattern**: Create calibrations from config files
4. **Registry Pattern**: Look up calibrations by sensor metadata
5. **Template Method**: Abstract base class for calibration steps

## Benefits

### For Users
- Define calibrations in simple YAML files
- Automatically correct for frequency-dependent responses
- Track calibration provenance and versioning
- Easy to update when manufacturer specs change

### For Developers
- Add new calibration types without modifying existing code
- Test calibration steps independently
- Reuse calibration steps across different sensors
- Clear separation between data reading and calibration

## Examples

### Example 1: Simple Time-Domain Calibration

```python
# configs/sensors/simple_hydrophone.yaml
sensor:
  manufacturer: "WHOI"
  model: "Simple Hydrophone"

calibration:
  version: "1.0"
  chain:
    - type: "adc_conversion"
      adc_vref: 2.5
      adc_bits: 24
    - type: "gain"
      gain_db: 20.0
    - type: "sensitivity"
      sensitivity_db: -165.0
```

### Example 2: Frequency-Dependent Calibration

```python
# configs/sensors/advanced_hydrophone.yaml
sensor:
  manufacturer: "Ocean Sonics"
  model: "icListen HF"

calibration:
  version: "2.1"
  chain:
    - type: "adc_conversion"
      adc_vref: 2.5
      adc_bits: 24

    - type: "gain"
      gain_db: 26.0

    - type: "frequency_response"
      frequencies: [10, 100, 1000, 10000, 100000]
      magnitude_db: [-170, -168, -165, -168, -175]
      phase_deg: [0, 2, 5, 15, 30]
      interpolation: "cubic"
      apply_inverse: true
```

### Example 3: Multi-Step Complex Calibration

```python
# configs/sensors/research_array.yaml
sensor:
  manufacturer: "High Tech Inc"
  model: "HTI-94-SSQ"

calibration:
  version: "3.2"
  chain:
    # Convert ADC counts to voltage
    - type: "adc_conversion"
      adc_vref: 5.0
      adc_bits: 16

    # Remove DC offset
    - type: "offset"
      offset: -0.001

    # Apply preamp gain
    - type: "gain"
      gain_db: 26.0

    # High-pass filter
    - type: "highpass_filter"
      cutoff_freq: 5.0
      order: 4

    # Frequency response correction
    - type: "frequency_response"
      frequencies: [1, 10, 100, 1000, 10000, 100000]
      magnitude_db: [-180, -172, -168, -167, -170, -178]
      phase_deg: [0, 2, 5, 10, 20, 45]
      interpolation: "pchip"

    # Anti-aliasing filter
    - type: "lowpass_filter"
      cutoff_freq: 95000.0
      order: 8
```

## Integration with Existing Code

The calibration system is designed to work alongside existing code. See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed migration strategies.

### Gradual Migration

1. **Phase 1**: Create calibration config files for your sensors
2. **Phase 2**: Use alongside existing code (dual system)
3. **Phase 3**: Fully migrate to new system

### Backward Compatibility

```python
# Option 1: Use new system with existing readers
from tritonoa.data.formats.shru import SHRUReader
from tritonoa.data.calibration import CalibrationFactory

reader = SHRUReader()
raw_data, header = reader.read_raw_data("data.D23")

chain = CalibrationFactory.from_yaml("configs/sensors/shru.yaml")
result = chain.apply(raw_data, sampling_rate=header.rhfs)

# Option 2: Convert legacy SignalParams
from tritonoa.data.signal import SignalParams

def convert_to_calibration_chain(params: SignalParams):
    return CalibrationChain([
        ADCConversion(adc_vref=params.adc_vref, adc_bits=24),
        Gain(gain_db=params.gain),
        Sensitivity(sensitivity_db=params.sensitivity)
    ])
```

## Extending the System

### Add Custom Calibration Steps

```python
from tritonoa.data.calibration.base import TimeDomainStep
from tritonoa.data.calibration import CalibrationFactory

class MyCustomStep(TimeDomainStep):
    def __init__(self, my_param: float):
        self.my_param = my_param

    def apply(self, data, sampling_rate, **kwargs):
        # Custom calibration logic
        return data * self.my_param

    def get_name(self):
        return "My Custom Calibration"

    def get_parameters(self):
        return {"my_param": self.my_param}

# Register it
CalibrationFactory.register_step("my_custom", MyCustomStep)

# Now usable in YAML configs:
# - type: "my_custom"
#   my_param: 1.5
```

## Testing

Run tests to verify calibration accuracy:

```python
from tritonoa.data.calibration import CalibrationFactory
import numpy as np

# Load calibration
chain = CalibrationFactory.from_yaml("configs/sensors/my_sensor.yaml")

# Test with known reference signal
reference_signal = np.sin(2 * np.pi * 1000 * np.arange(0, 1, 1/48000))
result = chain.apply(reference_signal, sampling_rate=48000)

assert result.units == "uPa"
assert result.shape == reference_signal.shape
print(chain.summary())
```

## Performance Considerations

- **Time-domain steps**: O(N) complexity, very fast
- **Frequency-domain steps**: O(N log N) due to FFT
- **Caching**: Consider caching calibration chains for repeated use
- **Memory**: Frequency-domain steps require 2x memory during FFT

## Future Enhancements

Potential future additions:

- [ ] Support for time-varying calibrations
- [ ] Automatic calibration validation against reference signals
- [ ] GPU acceleration for frequency-domain steps
- [ ] Distributed processing for large datasets
- [ ] Graphical calibration editor
- [ ] Automatic config generation from manufacturer datasheets

## Contributing

To add new calibration step types:

1. Inherit from `TimeDomainStep` or `FrequencyDomainStep`
2. Implement `apply()`, `get_name()`, and `get_parameters()`
3. Register with `CalibrationFactory.register_step()`
4. Add tests and documentation
5. Submit PR

## License

Same as parent TritonOA project.

## Support

For questions or issues:
1. Check [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
2. Review example configs in `configs/sensors/`
3. Open an issue on the project repository
