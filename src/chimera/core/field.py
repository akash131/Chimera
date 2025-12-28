"""
Field representation for Chimera.

Fields are the primary unknown quantities in PDEs. This module provides
a unified field representation that works across discretization methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Dict, Any
from enum import Enum
import numpy as np


class FieldType(Enum):
    """Type of field quantity."""
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


@dataclass
class Field:
    """
    Physical field (unknown or known quantity).

    A Field represents a quantity defined over a domain. It can be:
    - An unknown to be solved for (primary variable)
    - A known quantity (material property, source term)
    - A derived quantity (computed from other fields)

    Attributes:
        name: Field identifier (e.g., 'u', 'temperature', 'velocity')
        field_type: Scalar, vector, or tensor
        dim: Spatial dimension of the domain
        components: Number of components (1 for scalar, dim for vector, etc.)
        values: Discretized values (shape depends on discretization)
        interpolant: Function for evaluating field at arbitrary points
    """
    name: str
    field_type: FieldType = FieldType.SCALAR
    dim: int = 2
    values: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None  # Points where values are defined
    _interpolant: Optional[Callable] = None

    def __post_init__(self):
        if self.field_type == FieldType.SCALAR:
            self.components = 1
        elif self.field_type == FieldType.VECTOR:
            self.components = self.dim
        elif self.field_type == FieldType.TENSOR:
            self.components = self.dim * self.dim

    @classmethod
    def scalar(cls, name: str, dim: int = 2) -> Field:
        """Create a scalar field."""
        return cls(name=name, field_type=FieldType.SCALAR, dim=dim)

    @classmethod
    def vector(cls, name: str, dim: int = 2) -> Field:
        """Create a vector field."""
        return cls(name=name, field_type=FieldType.VECTOR, dim=dim)

    @classmethod
    def from_function(cls, name: str, func: Callable[[np.ndarray], np.ndarray],
                      points: np.ndarray, field_type: FieldType = FieldType.SCALAR) -> Field:
        """Create a field from a function evaluated at given points."""
        values = func(points)
        return cls(
            name=name,
            field_type=field_type,
            dim=points.shape[1],
            values=values,
            points=points
        )

    def set_values(self, values: np.ndarray, points: Optional[np.ndarray] = None):
        """Set field values at discretization points."""
        self.values = values
        if points is not None:
            self.points = points
        self._interpolant = None  # Invalidate cache

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate field at arbitrary points using interpolation."""
        if self.values is None:
            raise ValueError(f"Field '{self.name}' has no values set")

        if self._interpolant is None:
            self._build_interpolant()

        return self._interpolant(x)

    def _build_interpolant(self):
        """Build interpolation function from discrete values."""
        from scipy.interpolate import RBFInterpolator

        if self.points is None:
            raise ValueError("Cannot build interpolant without point locations")

        # Use RBF interpolation for scattered data
        if self.field_type == FieldType.SCALAR:
            self._interpolant = RBFInterpolator(
                self.points, self.values.flatten(), kernel='thin_plate_spline'
            )
        else:
            # For vector/tensor fields, interpolate each component
            interpolants = []
            for i in range(self.components):
                interp = RBFInterpolator(
                    self.points, self.values[:, i], kernel='thin_plate_spline'
                )
                interpolants.append(interp)

            def vector_interpolant(x):
                return np.column_stack([interp(x) for interp in interpolants])

            self._interpolant = vector_interpolant

    def gradient(self) -> Field:
        """Compute gradient field (scalar -> vector, vector -> tensor)."""
        if self.values is None or self.points is None:
            raise ValueError("Cannot compute gradient without values and points")

        from scipy.interpolate import RBFInterpolator

        if self.field_type == FieldType.SCALAR:
            # Scalar gradient -> vector
            interp = RBFInterpolator(
                self.points, self.values.flatten(), kernel='thin_plate_spline'
            )
            grad_values = interp(self.points, derivative=1)

            return Field(
                name=f"grad_{self.name}",
                field_type=FieldType.VECTOR,
                dim=self.dim,
                values=grad_values,
                points=self.points
            )
        else:
            raise NotImplementedError("Gradient of vector/tensor fields")

    def divergence(self) -> Field:
        """Compute divergence (vector -> scalar)."""
        if self.field_type != FieldType.VECTOR:
            raise ValueError("Divergence is only defined for vector fields")

        if self.values is None or self.points is None:
            raise ValueError("Cannot compute divergence without values and points")

        from scipy.interpolate import RBFInterpolator

        div_values = np.zeros(len(self.points))
        for i in range(self.dim):
            interp = RBFInterpolator(
                self.points, self.values[:, i], kernel='thin_plate_spline'
            )
            # Get i-th derivative of i-th component
            deriv = np.zeros(self.dim, dtype=int)
            deriv[i] = 1
            div_values += interp(self.points, derivative=tuple(deriv)).flatten()

        return Field(
            name=f"div_{self.name}",
            field_type=FieldType.SCALAR,
            dim=self.dim,
            values=div_values,
            points=self.points
        )

    def norm(self, ord: int = 2) -> float:
        """Compute norm of field values."""
        if self.values is None:
            raise ValueError("Cannot compute norm without values")
        return np.linalg.norm(self.values.flatten(), ord=ord)

    def max(self) -> float:
        """Maximum value."""
        if self.values is None:
            raise ValueError("Cannot compute max without values")
        return float(np.max(self.values))

    def min(self) -> float:
        """Minimum value."""
        if self.values is None:
            raise ValueError("Cannot compute min without values")
        return float(np.min(self.values))

    def copy(self) -> Field:
        """Create a deep copy."""
        return Field(
            name=self.name,
            field_type=self.field_type,
            dim=self.dim,
            values=self.values.copy() if self.values is not None else None,
            points=self.points.copy() if self.points is not None else None
        )

    def __repr__(self) -> str:
        shape = self.values.shape if self.values is not None else None
        return f"Field('{self.name}', type={self.field_type.value}, shape={shape})"

    # Arithmetic operations
    def __add__(self, other: Union[Field, float, np.ndarray]) -> Field:
        result = self.copy()
        if isinstance(other, Field):
            result.values = self.values + other.values
        else:
            result.values = self.values + other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[Field, float, np.ndarray]) -> Field:
        result = self.copy()
        if isinstance(other, Field):
            result.values = self.values - other.values
        else:
            result.values = self.values - other
        return result

    def __mul__(self, other: Union[float, np.ndarray]) -> Field:
        result = self.copy()
        result.values = self.values * other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, np.ndarray]) -> Field:
        result = self.copy()
        result.values = self.values / other
        return result
