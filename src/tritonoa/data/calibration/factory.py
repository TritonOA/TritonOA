"""Factory for creating calibration chains from configuration files.

This module implements the Factory pattern for creating calibration
chains from YAML or JSON configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from tritonoa.data.calibration.base import CalibrationMetadata, CalibrationStep
from tritonoa.data.calibration.chain import CalibrationChain
from tritonoa.data.calibration.steps import (
    ADCConversion,
    FrequencyResponse,
    Gain,
    HighPassFilter,
    LowPassFilter,
    Offset,
    ScalarMultiply,
    Sensitivity,
)

logger = logging.getLogger(__name__)


class CalibrationFactory:
    """Factory for creating calibration chains from configuration.

    The factory pattern allows creating calibration chains from
    configuration files (YAML/JSON) or dictionaries, making it easy
    to define sensor calibrations externally.

    Example:
        >>> # From YAML file
        >>> chain = CalibrationFactory.from_yaml("configs/sensors/icListen_HF.yaml")
        >>>
        >>> # From dictionary
        >>> config = {
        ...     "calibration": {
        ...         "chain": [
        ...             {"type": "adc_conversion", "adc_vref": 2.5, "adc_bits": 24},
        ...             {"type": "gain", "gain_db": 20.0},
        ...             {"type": "sensitivity", "sensitivity_db": -170.0}
        ...         ]
        ...     }
        ... }
        >>> chain = CalibrationFactory.from_dict(config)
    """

    # Registry mapping step type names to classes
    _STEP_REGISTRY: dict[str, type[CalibrationStep]] = {
        "adc_conversion": ADCConversion,
        "gain": Gain,
        "sensitivity": Sensitivity,
        "scalar_multiply": ScalarMultiply,
        "offset": Offset,
        "frequency_response": FrequencyResponse,
        "highpass_filter": HighPassFilter,
        "lowpass_filter": LowPassFilter,
    }

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> CalibrationChain:
        """Create calibration chain from YAML file.

        Args:
            file_path: Path to YAML configuration file.

        Returns:
            CalibrationChain created from the configuration.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If configuration is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        logger.info(f"Loading calibration from YAML: {file_path}")

        with open(file_path, "r") as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_json(cls, file_path: str | Path) -> CalibrationChain:
        """Create calibration chain from JSON file.

        Args:
            file_path: Path to JSON configuration file.

        Returns:
            CalibrationChain created from the configuration.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If configuration is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        logger.info(f"Loading calibration from JSON: {file_path}")

        with open(file_path, "r") as f:
            config = json.load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CalibrationChain:
        """Create calibration chain from configuration dictionary.

        Args:
            config: Configuration dictionary with the following structure:
                {
                    "sensor": {
                        "manufacturer": "...",
                        "model": "...",
                        "serial_number": "..."
                    },
                    "calibration": {
                        "version": "...",
                        "date": "...",
                        "valid_until": "...",
                        "chain": [
                            {"type": "step_type", "param1": value1, ...},
                            ...
                        ]
                    }
                }

        Returns:
            CalibrationChain created from the configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Extract metadata
        metadata = cls._extract_metadata(config)

        # Extract and create calibration steps
        if "calibration" not in config:
            raise ValueError("Configuration must contain 'calibration' section")

        calibration_config = config["calibration"]

        if "chain" not in calibration_config:
            raise ValueError("Calibration configuration must contain 'chain' list")

        steps = cls._create_steps(calibration_config["chain"])

        logger.info(f"Created calibration chain with {len(steps)} steps")

        return CalibrationChain(steps=steps, metadata=metadata)

    @classmethod
    def _extract_metadata(cls, config: dict[str, Any]) -> CalibrationMetadata:
        """Extract metadata from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            CalibrationMetadata object.
        """
        metadata = CalibrationMetadata()

        # Extract sensor metadata
        if "sensor" in config:
            sensor = config["sensor"]
            metadata.manufacturer = sensor.get("manufacturer")
            metadata.model = sensor.get("model")
            metadata.serial_number = sensor.get("serial_number")

        # Extract calibration metadata
        if "calibration" in config:
            calibration = config["calibration"]
            metadata.version = calibration.get("version")
            metadata.date = calibration.get("date")
            metadata.valid_until = calibration.get("valid_until")

        return metadata

    @classmethod
    def _create_steps(cls, steps_config: list[dict[str, Any]]) -> list[CalibrationStep]:
        """Create calibration steps from configuration.

        Args:
            steps_config: List of step configurations.

        Returns:
            List of CalibrationStep objects.

        Raises:
            ValueError: If step configuration is invalid.
        """
        steps = []

        for i, step_config in enumerate(steps_config):
            if "type" not in step_config:
                raise ValueError(f"Step {i} missing 'type' field")

            step_type = step_config["type"]

            if step_type not in cls._STEP_REGISTRY:
                raise ValueError(
                    f"Unknown calibration step type: {step_type}. "
                    f"Available types: {list(cls._STEP_REGISTRY.keys())}"
                )

            # Get step class
            step_class = cls._STEP_REGISTRY[step_type]

            # Extract parameters (all fields except 'type')
            params = {k: v for k, v in step_config.items() if k != "type"}

            # Create step instance
            try:
                step = step_class(**params)
                steps.append(step)
                logger.debug(f"Created step {i + 1}: {step.get_name()} with params {params}")
            except TypeError as e:
                raise ValueError(
                    f"Invalid parameters for step type '{step_type}': {e}"
                ) from e

        return steps

    @classmethod
    def register_step(cls, step_type: str, step_class: type[CalibrationStep]) -> None:
        """Register a custom calibration step type.

        This allows users to register their own custom calibration steps
        that can be used in configuration files.

        Args:
            step_type: Type name to use in configuration files.
            step_class: CalibrationStep subclass.

        Example:
            >>> class MyCustomStep(CalibrationStep):
            ...     def apply(self, data, sampling_rate, **kwargs):
            ...         # Custom logic
            ...         return data
            ...
            >>> CalibrationFactory.register_step("my_custom_step", MyCustomStep)
        """
        if not issubclass(step_class, CalibrationStep):
            raise ValueError(
                f"step_class must be a subclass of CalibrationStep, got {step_class}"
            )

        cls._STEP_REGISTRY[step_type] = step_class
        logger.info(f"Registered custom calibration step: {step_type}")

    @classmethod
    def get_registered_steps(cls) -> list[str]:
        """Get list of registered step types.

        Returns:
            List of registered step type names.
        """
        return list(cls._STEP_REGISTRY.keys())

    @classmethod
    def to_yaml(cls, chain: CalibrationChain, file_path: str | Path) -> None:
        """Save calibration chain to YAML file.

        Args:
            chain: CalibrationChain to save.
            file_path: Path to output YAML file.
        """
        file_path = Path(file_path)
        config = cls.to_dict(chain)

        logger.info(f"Saving calibration to YAML: {file_path}")

        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def to_json(cls, chain: CalibrationChain, file_path: str | Path) -> None:
        """Save calibration chain to JSON file.

        Args:
            chain: CalibrationChain to save.
            file_path: Path to output JSON file.
        """
        file_path = Path(file_path)
        config = cls.to_dict(chain)

        logger.info(f"Saving calibration to JSON: {file_path}")

        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def to_dict(cls, chain: CalibrationChain) -> dict[str, Any]:
        """Convert calibration chain to configuration dictionary.

        Args:
            chain: CalibrationChain to convert.

        Returns:
            Configuration dictionary.
        """
        config: dict[str, Any] = {}

        # Add sensor metadata
        metadata = chain.metadata
        if any([metadata.manufacturer, metadata.model, metadata.serial_number]):
            config["sensor"] = {}
            if metadata.manufacturer:
                config["sensor"]["manufacturer"] = metadata.manufacturer
            if metadata.model:
                config["sensor"]["model"] = metadata.model
            if metadata.serial_number:
                config["sensor"]["serial_number"] = metadata.serial_number

        # Add calibration metadata and steps
        config["calibration"] = {}
        if metadata.version:
            config["calibration"]["version"] = metadata.version
        if metadata.date:
            config["calibration"]["date"] = metadata.date
        if metadata.valid_until:
            config["calibration"]["valid_until"] = metadata.valid_until

        # Add steps
        config["calibration"]["chain"] = []
        for step in chain.steps:
            # Find step type name
            step_type = None
            for type_name, step_class in cls._STEP_REGISTRY.items():
                if isinstance(step, step_class):
                    step_type = type_name
                    break

            if step_type is None:
                logger.warning(
                    f"Step {step.__class__.__name__} not in registry, using class name"
                )
                step_type = step.__class__.__name__.lower()

            step_config = {"type": step_type}
            step_config.update(step.get_parameters())
            config["calibration"]["chain"].append(step_config)

        return config
