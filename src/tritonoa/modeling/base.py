# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np
from scipy.interpolate import interp1d, griddata


@dataclass
class Layer:
    depth: Sequence[float]
    range: float | Sequence[float]
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
        if (attrib.shape[0] == len(self.depth)) and (
            attrib.shape[1] == len(self.range)
        ):
            return attrib
        if attrib.size == 1:
            return np.full(
                (len(self.depth), len(self.range)), attrib.item(), dtype=float
            )
        if attrib.shape[0] == len(self.depth) and attrib.shape[1] != len(self.range):
            return attrib[:, np.newaxis] * np.ones(len(self.range), dtype=float)
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


# @dataclass
# class Halfspace:
#     halfspace_depth: Sequence[float]


# @dataclass
# class DataHolder:
#     data: Dict[str, np.ndarray] = field(default_factory=dict)
#     coordinates: Union[np.ndarray, None] = field(default=None)
#     interpolation_methods: List[str] = field(default_factory=lambda: ["linear"])

#     def __post_init__(self):
#         # Ensure data and coordinates are numpy arrays
#         for key in self.data:
#             self.data[key] = np.array(self.data[key])
#         if self.coordinates is not None:
#             self.coordinates = np.array(self.coordinates)

#         if self.coordinates is not None and len(self.coordinates.shape) != 2:
#             raise ValueError(
#                 "Coordinates should be in 2-D format: (n_points, n_dimensions)"
#             )

#         for key in self.data:
#             if len(self.coordinates) != len(self.data[key]):
#                 raise ValueError(
#                     f"Data and coordinates should have matching first dimensions for dataset '{key}'"
#                 )

#     def to_cylindrical(self):
#         """
#         Converts Cartesian coordinates to cylindrical coordinates (r, theta, z).
#         Works for 1-D, 2-D, and 3-D coordinates.
#         """
#         if self.coordinates is None:
#             raise ValueError("Coordinates are required for cylindrical conversion")

#         # Extract coordinates
#         x = self.coordinates[:, 0]
#         y = (
#             self.coordinates[:, 1]
#             if self.coordinates.shape[1] > 1
#             else np.zeros_like(x)
#         )
#         z = (
#             self.coordinates[:, 2]
#             if self.coordinates.shape[1] > 2
#             else np.zeros_like(x)
#         )

#         # Calculate cylindrical coordinates
#         r = np.sqrt(x**2 + y**2)
#         theta = np.arctan2(y, x)

#         # Update coordinates to cylindrical format
#         self.coordinates = np.column_stack((r, theta, z))

#     def interpolate(
#         self, new_coordinates: np.ndarray, methods: Optional[List[str]] = None
#     ) -> Dict[str, np.ndarray]:
#         """
#         Interpolates the data at the given new coordinates.
#         Allows specifying different interpolation methods for each dimension.
#         """
#         if not self.data or self.coordinates is None:
#             raise ValueError("Both data and coordinates are required for interpolation")

#         methods = methods or self.interpolation_methods
#         interpolated_data = {}

#         for key, values in self.data.items():
#             if len(values.shape) == 1:  # 1-D data
#                 interpolator = interp1d(
#                     self.coordinates[:, 0],
#                     values,
#                     kind=methods[0],
#                     fill_value="extrapolate",
#                 )
#                 interpolated_data[key] = interpolator(new_coordinates[:, 0])
#             else:  # Multi-dimensional data (2-D or 3-D)
#                 interpolated_values = []
#                 for i in range(self.coordinates.shape[1]):
#                     interpolator = interp1d(
#                         self.coordinates[:, i],
#                         values,
#                         kind=methods[i % len(methods)],
#                         fill_value="extrapolate",
#                         axis=0,
#                     )
#                     interpolated_values.append(interpolator(new_coordinates[:, i]))
#                 interpolated_data[key] = np.mean(interpolated_values, axis=0)

#         return interpolated_data

#     def resample(
#         self, new_grid: np.ndarray, methods: Optional[List[str]] = None
#     ) -> Dict[str, np.ndarray]:
#         """
#         Resamples data on a new grid, which can be in 1-D, 2-D, or 3-D.
#         """
#         return self.interpolate(new_grid, methods)

#     def extrapolate(
#         self, new_coordinates: np.ndarray, num_points: int = 5
#     ) -> Dict[str, np.ndarray]:
#         """
#         Extrapolates data using a linear fit based on the average of the last `num_points`.
#         """
#         if not self.data or self.coordinates is None:
#             raise ValueError("Both data and coordinates are required for extrapolation")

#         extrapolated_data = {}
#         for key, values in self.data.items():
#             if len(values.shape) == 1:  # 1-D extrapolation
#                 x = self.coordinates[:, 0]
#                 y = values
#                 slope = (y[-1] - y[-num_points]) / (x[-1] - x[-num_points])
#                 intercept = y[-1] - slope * x[-1]
#                 extrapolated_values = slope * new_coordinates[:, 0] + intercept
#                 extrapolated_data[key] = extrapolated_values
#             else:
#                 # For multi-dimensional extrapolation, using nearest method or linear approximation
#                 extrapolated_data[key] = self.interpolate(
#                     new_coordinates, methods=["nearest"]
#                 )[key]

#         return extrapolated_data


# # Example Usage:
# data = {"set1": np.array([1, 2, 3, 4]), "set2": np.array([4, 3, 2, 1])}
# coords = np.array([[0], [1], [2], [3]])
# data_holder = DataHolder(data=data, coordinates=coords)
# new_coords = np.array([[0.5], [1.5], [2.5]])
# interpolated_data = data_holder.interpolate(new_coords)
# print(interpolated_data)

# # Converting from Cartesian to cylindrical
# coords_3d = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
# data_holder_3d = DataHolder(data={"set1": np.array([1, 2, 3])}, coordinates=coords_3d)
# data_holder_3d.to_cylindrical()
# print(data_holder_3d.coordinates)
