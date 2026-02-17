"""Calibration chain for composing multiple calibration steps.

This module implements the Chain of Responsibility pattern for
applying a sequence of calibration steps to acoustic data.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tritonoa.data.calibration.base import (
    CalibrationMetadata,
    CalibrationStep,
    CalibratedData,
)

logger = logging.getLogger(__name__)


class CalibrationChain:
    """Chain of calibration steps applied in sequence.

    The CalibrationChain applies multiple calibration steps in order,
    passing the output of each step as input to the next. It tracks
    metadata about the calibration process.

    Args:
        steps: List of calibration steps to apply in order.
        metadata: Optional metadata about the calibration.

    Example:
        >>> from tritonoa.data.calibration.steps import ADCConversion, Gain, Sensitivity
        >>> chain = CalibrationChain([
        ...     ADCConversion(adc_vref=2.5, adc_bits=24),
        ...     Gain(gain_db=20.0),
        ...     Sensitivity(sensitivity_db=-170.0)
        ... ])
        >>> result = chain.apply(raw_data, sampling_rate=48000)
        >>> print(result.units)  # 'uPa'
    """

    def __init__(
        self,
        steps: list[CalibrationStep],
        metadata: CalibrationMetadata | None = None,
    ):
        self.steps = steps
        self.metadata = metadata or CalibrationMetadata()

        # Log the calibration chain
        logger.info(f"Created calibration chain with {len(steps)} steps")
        for i, step in enumerate(steps):
            logger.debug(f"  Step {i + 1}: {step.get_name()}")

    def apply(
        self,
        data: NDArray[np.float64],
        sampling_rate: float,
        **kwargs,
    ) -> CalibratedData:
        """Apply calibration chain to data.

        Args:
            data: Raw input data (channels, samples).
            sampling_rate: Sampling rate in Hz.
            **kwargs: Additional parameters passed to each step.

        Returns:
            CalibratedData object with calibrated data and metadata.

        Raises:
            ValueError: If any calibration step fails.
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False

        logger.info(
            f"Applying calibration chain to data with shape {data.shape} "
            f"at {sampling_rate} Hz"
        )

        # Apply each step in sequence
        calibrated = data.copy()
        for i, step in enumerate(self.steps):
            step_name = step.get_name()
            logger.debug(f"Applying step {i + 1}/{len(self.steps)}: {step_name}")

            try:
                calibrated = step.apply(calibrated, sampling_rate, **kwargs)

                # Record step in metadata
                self.metadata.steps_applied.append(step_name)
                self.metadata.parameters[step_name] = step.get_parameters()

            except Exception as e:
                logger.error(f"Error in calibration step '{step_name}': {e}")
                raise ValueError(
                    f"Calibration failed at step '{step_name}': {e}"
                ) from e

        # Squeeze output if input was 1D
        if squeeze_output:
            calibrated = calibrated.squeeze()

        logger.info("Calibration chain applied successfully")

        return CalibratedData(
            data=calibrated,
            units="uPa",  # Always output in micropascals
            sampling_rate=sampling_rate,
            metadata=self.metadata,
        )

    def add_step(self, step: CalibrationStep) -> None:
        """Add a calibration step to the end of the chain.

        Args:
            step: Calibration step to add.
        """
        self.steps.append(step)
        logger.debug(f"Added calibration step: {step.get_name()}")

    def insert_step(self, index: int, step: CalibrationStep) -> None:
        """Insert a calibration step at a specific position.

        Args:
            index: Position to insert the step.
            step: Calibration step to insert.
        """
        self.steps.insert(index, step)
        logger.debug(f"Inserted calibration step at position {index}: {step.get_name()}")

    def remove_step(self, index: int) -> CalibrationStep:
        """Remove a calibration step at a specific position.

        Args:
            index: Position of step to remove.

        Returns:
            The removed calibration step.
        """
        step = self.steps.pop(index)
        logger.debug(f"Removed calibration step at position {index}: {step.get_name()}")
        return step

    def get_step(self, index: int) -> CalibrationStep:
        """Get calibration step at a specific position.

        Args:
            index: Position of step to retrieve.

        Returns:
            The calibration step at the given index.
        """
        return self.steps[index]

    def get_steps(self) -> list[CalibrationStep]:
        """Get all calibration steps.

        Returns:
            List of all calibration steps in the chain.
        """
        return self.steps.copy()

    def clear_steps(self) -> None:
        """Remove all calibration steps from the chain."""
        self.steps.clear()
        logger.debug("Cleared all calibration steps")

    def __len__(self) -> int:
        """Number of steps in the chain."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation."""
        step_names = [step.get_name() for step in self.steps]
        return f"CalibrationChain({step_names})"

    def summary(self) -> str:
        """Get a human-readable summary of the calibration chain.

        Returns:
            Multi-line string describing the calibration chain.
        """
        lines = ["Calibration Chain Summary"]
        lines.append("=" * 50)
        lines.append(f"Number of steps: {len(self.steps)}")
        lines.append("")

        if self.metadata.manufacturer:
            lines.append(f"Manufacturer: {self.metadata.manufacturer}")
        if self.metadata.model:
            lines.append(f"Model: {self.metadata.model}")
        if self.metadata.serial_number:
            lines.append(f"Serial Number: {self.metadata.serial_number}")
        if self.metadata.version:
            lines.append(f"Calibration Version: {self.metadata.version}")
        if self.metadata.date:
            lines.append(f"Calibration Date: {self.metadata.date}")

        if any(
            [
                self.metadata.manufacturer,
                self.metadata.model,
                self.metadata.serial_number,
                self.metadata.version,
                self.metadata.date,
            ]
        ):
            lines.append("")

        lines.append("Calibration Steps:")
        lines.append("-" * 50)
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step.get_name()}")
            params = step.get_parameters()
            for key, value in params.items():
                # Truncate long lists
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 5:
                        value_str = f"[{value[0]}, {value[1]}, ..., {value[-1]}] (length={len(value)})"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                lines.append(f"   - {key}: {value_str}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration chain to dictionary.

        Returns:
            Dictionary representation of the calibration chain.
        """
        return {
            "steps": [
                {
                    "name": step.get_name(),
                    "type": step.__class__.__name__,
                    "parameters": step.get_parameters(),
                }
                for step in self.steps
            ],
            "metadata": self.metadata.to_dict(),
        }
