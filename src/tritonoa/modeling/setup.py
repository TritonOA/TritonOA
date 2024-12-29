# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np
from scipy.interpolate import interp1d, griddata


@dataclass
class Layer:
    depth: Sequence[float]
    range: Optional[float | Sequence[float]] = None
    transverse: Optional[Sequence[float]] = None
    compressional_speed: float | Sequence[float] = 1500.0
    shear_speed: float | Sequence[float] = 0.0
    compressional_attenuation: float | Sequence[float] = 0.0
    shear_attenuation: float | Sequence[float] = 0.0
    density: float | Sequence[float] = 1.0
    bottom_depth: Optional[float | Sequence[float]] = None
    reflection_coefficient: Optional[float | Sequence[float]] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        WATER_ATTRIBS = [
            "compressional_speed",
            "shear_speed",
            "compressional_attenuation",
            "shear_attenuation",
            "density",
        ]
        BOTTOM_ATTRIBS = ["bottom_depth", "reflection_coefficient"]
        if self.bottom_depth is None:
            self.bottom_depth = self.depth[-1]
        [
            setattr(self, attrib, self._to_array(getattr(self, attrib)))
            for attrib in self.__dataclass_fields__
        ]
        # Check if any attribute is range-dependent
        self.range_dependent = any(
            [
                self._check_bottom_range_dependence(getattr(self, attrib))
                for attrib in BOTTOM_ATTRIBS
            ]
            + [
                self._check_profile_range_dependence(getattr(self, attrib))
                for attrib in WATER_ATTRIBS
            ]
        )
        if self.range_dependent:
            # Create a 2-D mesh for water attributes with depth and range.
            [
                setattr(self, attrib, self._create_2d_mesh(getattr(self, attrib)))
                for attrib in WATER_ATTRIBS
            ]
            # Create a 1-D mesh for bottom attributes with range.
            [
                setattr(
                    self,
                    attrib,
                    self._create_1d_mesh(getattr(self, attrib), self.n_ranges),
                )
                for attrib in BOTTOM_ATTRIBS
            ]
            # Verify dimensions of water attributes in depth and range
            [
                self._verify_dims(getattr(self, attrib), (self.n_depths, self.n_ranges))
                for attrib in WATER_ATTRIBS
            ]
            # Verify dimensions of bottom attributes in range
            [
                self._verify_dims(getattr(self, attrib), (self.n_ranges,))
                for attrib in BOTTOM_ATTRIBS
            ]
        else:
            # Create a 1-D mesh for water attributes with depth.
            [
                setattr(
                    self,
                    attrib,
                    self._create_1d_mesh(getattr(self, attrib), self.n_depths),
                )
                for attrib in WATER_ATTRIBS
            ]
            # Verify dimensions of water attributes in depth
            [
                self._verify_dims(getattr(self, attrib), (self.n_depths,))
                for attrib in WATER_ATTRIBS
            ]
            # Collapse vector of bottom attributes to a single value.
            [
                setattr(self, attrib, np.unique(getattr(self, attrib)))
                for attrib in BOTTOM_ATTRIBS
            ]

        # Ensure bottom depth array has dimensions as range
        # self.bottom_depth = self._create_1d_mesh(self.bottom_depth)

        # Check if bottom depth is within the depth range
        if self.bottom_depth.min() > self.depth.max():
            raise ValueError(
                "Bottom depth should be less than or equal to maximum depth"
            )

    @property
    def n_depths(self) -> int:
        return len(self.depth)

    @property
    def n_ranges(self) -> int:
        return len(self.range)

    def interpolate(self) -> None: ...

    def _check_bottom_range_dependence(self, attrib: np.ndarray) -> bool:
        # Get shape of the attribute
        shape = attrib.shape

        # Check if the attribute is range-dependent
        # Return error for unimplemented dimensions
        if len(shape) > 1:
            raise ValueError("Invalid shape for attribute")

        # Check if elements are the same
        if np.any(np.diff(attrib)):
            return True

        return False

    def _check_profile_range_dependence(self, attrib: np.ndarray) -> bool:
        # Get shape of the attribute
        shape = attrib.shape

        # Check if the attribute is range-dependent
        # Return error for unimplemented dimensions
        if len(shape) > 2:
            raise ValueError("Invalid shape for attribute")

        # Case I: 2-D attribute - Check if horizontal elements change with each row of matrix
        if len(shape) == 2 and np.any(np.diff(attrib, axis=1)):
            return True

        # All other cases: 1-D attribute - Not range-dependent
        return False

    def _create_1d_mesh(self, attrib: Sequence[float], n_points: int) -> np.ndarray:
        if attrib.size == 1:
            return np.full(n_points, attrib.item(), dtype=float)
        if attrib.size == n_points:
            return attrib
        if attrib.ndim == 2 and attrib.shape[0] == n_points:
            # Collapses the 2-D array to 1-D in depth.
            return attrib[:, 0]
        raise ValueError("Invalid shape for attribute")

    def _create_2d_mesh(self, attrib: Sequence[float]) -> np.ndarray:
        # Create a 2-D mesh with depth and range
        # Case I: Single-element
        if attrib.size == 1:
            return np.full(
                (len(self.depth), len(self.range)), attrib.item(), dtype=float
            )
        # Case II: 1-D array
        if len(attrib.shape) == 1:
            return attrib[:, np.newaxis] * np.ones(len(self.range), dtype=float)
        # Case III: 2-D array with same number of rows as depth, but not range columns
        if attrib.shape[0] == len(self.depth) and attrib.shape[1] != len(self.range):
            return attrib[:, np.newaxis] * np.ones(len(self.range), dtype=float)
        # Case IV: 2-D array with same dimensions as (depth, range)
        if (attrib.shape[0] == len(self.depth)) and (
            attrib.shape[1] == len(self.range)
        ):
            return attrib
        raise ValueError("Invalid shape for attribute")

    def _create_3d_mesh(self, attrib: Sequence[float]) -> np.ndarray:
        # TODO: Extend this function to create 3-D mesh
        raise NotImplementedError

    def _to_array(self, attrib: float | Sequence[float]) -> np.ndarray:
        # Convert a non-array sequence to an array:
        if not isinstance(attrib, np.ndarray):
            attrib = np.atleast_1d(attrib)
        return attrib

    def _verify_dims(self, attrib: np.ndarray, dims: tuple[int]) -> None:
        for i, dim in enumerate(dims):
            print(attrib.shape)
            print(attrib.shape[i], dim)
            if attrib.shape[i] != dim:
                raise ValueError("Invalid shape for attribute")


@dataclass
class Source:
    depth: float
    frequency: float



class Environment:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def resample_range(self, range: np.ndarray) -> None:
        for layer in self.layers:
            pass
        return


class Receiver:
    def __init__(self, depth: float) -> None:
        self.depth = depth



class Configuration:
    def __init__(self, source: Source, receiver: Receiver, environment: Environment) -> None:
        self.source = source
        self.receiver = receiver
        self.environment = environment