# Sensor Calibration Configurations

This directory contains calibration configurations for various underwater acoustic sensors. Each configuration file defines the signal conditioning chain needed to convert raw sensor data to calibrated pressure values in micropascals (µPa).

## Directory Structure

```
configs/
├── sensors/           # Sensor-specific calibration files
│   ├── *.yaml        # YAML configuration files
│   └── *.json        # JSON configuration files
└── schemas/          # JSON schemas for validation
```

## Configuration File Format

Calibration configurations can be written in YAML or JSON format. Each file contains:

### Required Sections

1. **sensor** - Sensor metadata (optional but recommended)
   - `manufacturer`: Sensor manufacturer name
   - `model`: Sensor model identifier
   - `serial_number`: Sensor serial number

2. **calibration** - Calibration specification
   - `version`: Calibration version string
   - `date`: Calibration date (ISO format recommended)
   - `valid_until`: Expiration date (optional)
   - `chain`: List of calibration steps (required)

### Available Calibration Steps

#### Time-Domain Steps

1. **adc_conversion** - Convert ADC counts to voltage
   ```yaml
   - type: "adc_conversion"
     adc_vref: 2.5       # ADC reference voltage (V)
     adc_bits: 24        # ADC bit depth
     signed: true        # Signed representation
   ```

2. **gain** - Apply gain correction
   ```yaml
   - type: "gain"
     gain_db: 20.0       # Gain in dB (scalar or per-channel list)
     invert: false       # If true, divides instead of multiplies
   ```

3. **sensitivity** - Convert voltage to pressure
   ```yaml
   - type: "sensitivity"
     sensitivity_db: -170.0           # Sensitivity in dB re 1V/µPa
     reference_pressure: 1.0          # Reference pressure (µPa)
   ```

4. **scalar_multiply** - Generic scalar multiplication
   ```yaml
   - type: "scalar_multiply"
     factor: 1.5         # Multiplication factor
   ```

5. **offset** - Add/subtract DC offset
   ```yaml
   - type: "offset"
     offset: -0.001      # Offset value
   ```

#### Frequency-Domain Steps

1. **frequency_response** - Frequency-dependent magnitude and phase correction
   ```yaml
   - type: "frequency_response"
     frequencies: [10, 100, 1000, 10000]              # Hz
     magnitude_db: [-170, -168, -165, -170]           # dB re 1V/µPa
     phase_deg: [0, 2, 5, 15]                         # degrees
     interpolation: "cubic"                            # linear, cubic, pchip
     extrapolation: "constant"                         # constant, linear, raise
     apply_inverse: true                               # Apply inverse response
   ```

2. **highpass_filter** - High-pass filter
   ```yaml
   - type: "highpass_filter"
     cutoff_freq: 10.0   # Cutoff frequency (Hz)
     order: 4            # Filter order
     filter_type: "butterworth"
   ```

3. **lowpass_filter** - Low-pass filter
   ```yaml
   - type: "lowpass_filter"
     cutoff_freq: 20000.0
     order: 4
     filter_type: "butterworth"
   ```

## Usage Examples

### Loading a Calibration

```python
from tritonoa.data.calibration import CalibrationFactory

# From YAML file
chain = CalibrationFactory.from_yaml("configs/sensors/icListen_HF.yaml")

# From JSON file
chain = CalibrationFactory.from_json("configs/sensors/my_sensor.json")

# Apply to data
result = chain.apply(raw_data, sampling_rate=48000)
print(result.units)  # Always 'uPa'
```

### Using the Registry

```python
from tritonoa.data.calibration import CalibrationRegistry

# Create registry and scan directory
registry = CalibrationRegistry()
registry.scan_directory("configs/sensors/")

# Look up by sensor metadata
chain = registry.get(
    manufacturer="Ocean Sonics",
    model="icListen HF"
)

# Or by serial number
chain = registry.get(serial_number="1234")
```

### Creating Calibrations Programmatically

```python
from tritonoa.data.calibration import CalibrationChain
from tritonoa.data.calibration.steps import ADCConversion, Gain, Sensitivity

chain = CalibrationChain([
    ADCConversion(adc_vref=2.5, adc_bits=24),
    Gain(gain_db=[20.0, 20.0, 15.0, 15.0]),  # Per-channel
    Sensitivity(sensitivity_db=-170.0)
])

# Save to file
from tritonoa.data.calibration import CalibrationFactory
CalibrationFactory.to_yaml(chain, "my_calibration.yaml")
```

## Best Practices

1. **Version your calibrations** - Always include version and date information
2. **Document expiration** - Use `valid_until` for time-limited calibrations
3. **Per-channel parameters** - Use lists for multi-channel sensors
4. **Frequency response** - Include phase data when available for best accuracy
5. **Validate configurations** - Test calibrations with known reference signals
6. **Organize by sensor** - Use clear, descriptive filenames (manufacturer_model_serial.yaml)

## Adding Custom Steps

You can register custom calibration steps:

```python
from tritonoa.data.calibration import CalibrationFactory
from tritonoa.data.calibration.base import TimeDomainStep

class MyCustomStep(TimeDomainStep):
    def __init__(self, my_param: float):
        self.my_param = my_param

    def apply(self, data, sampling_rate, **kwargs):
        # Custom calibration logic
        return data * self.my_param

    def get_name(self):
        return "My Custom Step"

    def get_parameters(self):
        return {"my_param": self.my_param}

# Register it
CalibrationFactory.register_step("my_custom_step", MyCustomStep)

# Now usable in config files:
# - type: "my_custom_step"
#   my_param: 1.5
```

## Troubleshooting

### Common Issues

1. **Wrong number of channels**: Ensure per-channel parameters match your data
2. **Frequency response extrapolation**: Set appropriate `extrapolation` mode
3. **Phase wrapping**: Phase values should be continuous (unwrapped)
4. **Filter instability**: Reduce filter order if experiencing numerical issues

### Validation

Print calibration chain summary:
```python
chain = CalibrationFactory.from_yaml("my_sensor.yaml")
print(chain.summary())
```

Check metadata:
```python
result = chain.apply(data, sampling_rate=fs)
print(result.metadata.to_dict())
```
