"""Registry for looking up sensor calibrations.

This module provides a registry system for managing and retrieving
sensor calibration configurations based on sensor metadata.
"""

import logging
from pathlib import Path
from typing import Any

from tritonoa.data.calibration.chain import CalibrationChain
from tritonoa.data.calibration.factory import CalibrationFactory

logger = logging.getLogger(__name__)


class CalibrationRegistry:
    """Registry for managing sensor calibrations.

    The registry allows looking up calibration configurations by
    sensor manufacturer, model, serial number, or custom identifiers.

    Example:
        >>> registry = CalibrationRegistry()
        >>> registry.scan_directory("configs/sensors/")
        >>> chain = registry.get(manufacturer="Ocean Sonics", model="icListen HF")
    """

    def __init__(self):
        """Initialize empty calibration registry."""
        self._calibrations: dict[str, CalibrationChain] = {}
        self._metadata_index: dict[str, list[str]] = {}

    def register(
        self,
        calibration: CalibrationChain,
        identifier: str | None = None,
    ) -> str:
        """Register a calibration chain.

        Args:
            calibration: CalibrationChain to register.
            identifier: Optional custom identifier. If None, generates
                one from metadata.

        Returns:
            The identifier used to register the calibration.

        Example:
            >>> chain = CalibrationFactory.from_yaml("sensor.yaml")
            >>> registry.register(chain, identifier="my_sensor")
        """
        if identifier is None:
            identifier = self._generate_identifier(calibration)

        self._calibrations[identifier] = calibration

        # Index by metadata
        self._index_metadata(identifier, calibration)

        logger.info(f"Registered calibration: {identifier}")
        return identifier

    def register_from_file(
        self,
        file_path: str | Path,
        identifier: str | None = None,
    ) -> str:
        """Register a calibration from a file.

        Args:
            file_path: Path to YAML or JSON calibration file.
            identifier: Optional custom identifier.

        Returns:
            The identifier used to register the calibration.

        Example:
            >>> registry.register_from_file("configs/sensors/icListen.yaml")
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() in [".yaml", ".yml"]:
            calibration = CalibrationFactory.from_yaml(file_path)
        elif file_path.suffix.lower() == ".json":
            calibration = CalibrationFactory.from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return self.register(calibration, identifier)

    def scan_directory(
        self,
        directory: str | Path,
        pattern: str = "*.yaml",
        recursive: bool = True,
    ) -> int:
        """Scan directory for calibration files and register them.

        Args:
            directory: Directory to scan.
            pattern: File pattern to match (default: "*.yaml").
            recursive: Whether to scan subdirectories (default: True).

        Returns:
            Number of calibrations registered.

        Example:
            >>> registry.scan_directory("configs/sensors/", pattern="*.yaml")
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find calibration files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Also scan for JSON files if pattern is YAML
        if pattern in ["*.yaml", "*.yml"]:
            if recursive:
                files.extend(list(directory.rglob("*.json")))
            else:
                files.extend(list(directory.glob("*.json")))

        logger.info(f"Scanning {directory} for calibration files ({len(files)} found)")

        count = 0
        for file_path in files:
            try:
                self.register_from_file(file_path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {file_path}: {e}")

        logger.info(f"Registered {count} calibrations from {directory}")
        return count

    def get(
        self,
        identifier: str | None = None,
        manufacturer: str | None = None,
        model: str | None = None,
        serial_number: str | None = None,
    ) -> CalibrationChain:
        """Get calibration by identifier or metadata.

        Args:
            identifier: Direct calibration identifier.
            manufacturer: Sensor manufacturer.
            model: Sensor model.
            serial_number: Sensor serial number.

        Returns:
            CalibrationChain matching the criteria.

        Raises:
            KeyError: If no matching calibration is found.
            ValueError: If multiple matching calibrations are found.

        Example:
            >>> chain = registry.get(manufacturer="Ocean Sonics", model="icListen HF")
        """
        if identifier is not None:
            if identifier not in self._calibrations:
                raise KeyError(f"No calibration found with identifier: {identifier}")
            return self._calibrations[identifier]

        # Search by metadata
        matches = self._search_by_metadata(manufacturer, model, serial_number)

        if len(matches) == 0:
            raise KeyError(
                f"No calibration found for manufacturer={manufacturer}, "
                f"model={model}, serial_number={serial_number}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple calibrations found ({len(matches)}) for "
                f"manufacturer={manufacturer}, model={model}, serial_number={serial_number}. "
                f"Use more specific criteria or identifier."
            )

        return self._calibrations[matches[0]]

    def list_calibrations(self) -> list[dict[str, Any]]:
        """List all registered calibrations.

        Returns:
            List of dictionaries containing calibration metadata.

        Example:
            >>> for cal in registry.list_calibrations():
            ...     print(cal["identifier"], cal["manufacturer"], cal["model"])
        """
        calibrations = []

        for identifier, chain in self._calibrations.items():
            metadata = chain.metadata
            calibrations.append(
                {
                    "identifier": identifier,
                    "manufacturer": metadata.manufacturer,
                    "model": metadata.model,
                    "serial_number": metadata.serial_number,
                    "version": metadata.version,
                    "date": metadata.date,
                    "num_steps": len(chain.steps),
                }
            )

        return calibrations

    def _generate_identifier(self, calibration: CalibrationChain) -> str:
        """Generate identifier from calibration metadata.

        Args:
            calibration: CalibrationChain.

        Returns:
            Generated identifier string.
        """
        metadata = calibration.metadata
        parts = []

        if metadata.manufacturer:
            parts.append(metadata.manufacturer.replace(" ", "_"))
        if metadata.model:
            parts.append(metadata.model.replace(" ", "_"))
        if metadata.serial_number:
            parts.append(f"SN{metadata.serial_number}")

        if not parts:
            # Fallback to unique ID based on number of registered calibrations
            parts.append(f"calibration_{len(self._calibrations)}")

        return "_".join(parts)

    def _index_metadata(self, identifier: str, calibration: CalibrationChain) -> None:
        """Index calibration by metadata fields.

        Args:
            identifier: Calibration identifier.
            calibration: CalibrationChain.
        """
        metadata = calibration.metadata

        # Index by manufacturer
        if metadata.manufacturer:
            key = f"manufacturer:{metadata.manufacturer}"
            self._metadata_index.setdefault(key, []).append(identifier)

        # Index by model
        if metadata.model:
            key = f"model:{metadata.model}"
            self._metadata_index.setdefault(key, []).append(identifier)

        # Index by serial number
        if metadata.serial_number:
            key = f"serial:{metadata.serial_number}"
            self._metadata_index.setdefault(key, []).append(identifier)

        # Index by manufacturer+model
        if metadata.manufacturer and metadata.model:
            key = f"mfg_model:{metadata.manufacturer}:{metadata.model}"
            self._metadata_index.setdefault(key, []).append(identifier)

    def _search_by_metadata(
        self,
        manufacturer: str | None,
        model: str | None,
        serial_number: str | None,
    ) -> list[str]:
        """Search for calibrations by metadata.

        Args:
            manufacturer: Sensor manufacturer.
            model: Sensor model.
            serial_number: Sensor serial number.

        Returns:
            List of matching calibration identifiers.
        """
        # Start with all calibrations
        candidates = set(self._calibrations.keys())

        # Filter by serial number first (most specific)
        if serial_number:
            key = f"serial:{serial_number}"
            if key in self._metadata_index:
                candidates &= set(self._metadata_index[key])
            else:
                return []

        # Filter by manufacturer+model (next most specific)
        if manufacturer and model:
            key = f"mfg_model:{manufacturer}:{model}"
            if key in self._metadata_index:
                candidates &= set(self._metadata_index[key])
            else:
                # Try manufacturer and model separately
                if manufacturer:
                    key = f"manufacturer:{manufacturer}"
                    if key in self._metadata_index:
                        candidates &= set(self._metadata_index[key])
                    else:
                        return []

                if model:
                    key = f"model:{model}"
                    if key in self._metadata_index:
                        candidates &= set(self._metadata_index[key])
                    else:
                        return []
        else:
            # Filter by manufacturer only
            if manufacturer:
                key = f"manufacturer:{manufacturer}"
                if key in self._metadata_index:
                    candidates &= set(self._metadata_index[key])
                else:
                    return []

            # Filter by model only
            if model:
                key = f"model:{model}"
                if key in self._metadata_index:
                    candidates &= set(self._metadata_index[key])
                else:
                    return []

        return list(candidates)

    def __len__(self) -> int:
        """Number of registered calibrations."""
        return len(self._calibrations)

    def __contains__(self, identifier: str) -> bool:
        """Check if identifier is registered."""
        return identifier in self._calibrations

    def __repr__(self) -> str:
        """String representation."""
        return f"CalibrationRegistry({len(self._calibrations)} calibrations)"
