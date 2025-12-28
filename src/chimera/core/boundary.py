"""
Boundary condition definitions for Chimera.

Supports various boundary condition types that work across
discretization methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Tuple
import numpy as np


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions.

    All BC types must implement evaluation at boundary points.
    """

    @abstractmethod
    def evaluate(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 time: float = 0.0) -> np.ndarray:
        """Evaluate BC value at given boundary points."""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Get BC type identifier."""
        pass


@dataclass
class DirichletBC(BoundaryCondition):
    """
    Dirichlet (essential) boundary condition: u = g on boundary.

    Attributes:
        value: Constant value, function(x), or function(x, t)
        region: Callable to identify boundary region (returns bool mask)
        field_name: Name of field this BC applies to
        component: For vector fields, which component (None = all)
    """
    value: Union[float, np.ndarray, Callable]
    region: Optional[Callable[[np.ndarray], np.ndarray]] = None
    field_name: str = "u"
    component: Optional[int] = None
    name: str = "dirichlet"

    def evaluate(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 time: float = 0.0) -> np.ndarray:
        """Evaluate Dirichlet value at boundary points."""
        points = np.atleast_2d(points)
        n_points = len(points)

        if callable(self.value):
            # Check if function accepts time argument
            import inspect
            sig = inspect.signature(self.value)
            if len(sig.parameters) > 1:
                result = self.value(points, time)
            else:
                result = self.value(points)
        elif isinstance(self.value, np.ndarray):
            result = np.tile(self.value, (n_points, 1))
        else:
            result = np.full(n_points, self.value)

        return np.atleast_1d(result)

    def applies_to(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask of points where this BC applies."""
        if self.region is None:
            return np.ones(len(points), dtype=bool)
        return self.region(points)

    def get_type(self) -> str:
        return "dirichlet"

    @classmethod
    def constant(cls, value: float, field_name: str = "u",
                 region: Optional[Callable] = None) -> DirichletBC:
        """Create constant Dirichlet BC."""
        return cls(value=value, field_name=field_name, region=region)

    @classmethod
    def zero(cls, field_name: str = "u",
             region: Optional[Callable] = None) -> DirichletBC:
        """Create zero Dirichlet BC (homogeneous)."""
        return cls(value=0.0, field_name=field_name, region=region, name="zero_dirichlet")

    def __repr__(self) -> str:
        if callable(self.value):
            val_str = "f(x)"
        else:
            val_str = str(self.value)
        return f"DirichletBC({self.field_name} = {val_str})"


@dataclass
class NeumannBC(BoundaryCondition):
    """
    Neumann (natural) boundary condition: n . grad(u) = g on boundary.

    Attributes:
        value: Normal derivative value (constant or function)
        region: Callable to identify boundary region
        field_name: Name of field this BC applies to
    """
    value: Union[float, np.ndarray, Callable]
    region: Optional[Callable[[np.ndarray], np.ndarray]] = None
    field_name: str = "u"
    name: str = "neumann"

    def evaluate(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 time: float = 0.0) -> np.ndarray:
        """Evaluate Neumann value at boundary points."""
        points = np.atleast_2d(points)
        n_points = len(points)

        if callable(self.value):
            import inspect
            sig = inspect.signature(self.value)
            if len(sig.parameters) > 1:
                result = self.value(points, time)
            else:
                result = self.value(points)
        elif isinstance(self.value, np.ndarray):
            result = np.tile(self.value, (n_points, 1))
        else:
            result = np.full(n_points, self.value)

        return np.atleast_1d(result)

    def applies_to(self, points: np.ndarray) -> np.ndarray:
        """Return boolean mask of points where this BC applies."""
        if self.region is None:
            return np.ones(len(points), dtype=bool)
        return self.region(points)

    def get_type(self) -> str:
        return "neumann"

    @classmethod
    def zero_flux(cls, field_name: str = "u",
                  region: Optional[Callable] = None) -> NeumannBC:
        """Create zero flux (insulated) BC."""
        return cls(value=0.0, field_name=field_name, region=region, name="zero_flux")

    def __repr__(self) -> str:
        if callable(self.value):
            val_str = "f(x)"
        else:
            val_str = str(self.value)
        return f"NeumannBC(n.grad({self.field_name}) = {val_str})"


@dataclass
class RobinBC(BoundaryCondition):
    """
    Robin (mixed) boundary condition: alpha*u + beta*n.grad(u) = g.

    Useful for convective heat transfer, radiation, etc.
    """
    alpha: Union[float, Callable]
    beta: Union[float, Callable]
    value: Union[float, Callable]
    region: Optional[Callable[[np.ndarray], np.ndarray]] = None
    field_name: str = "u"
    name: str = "robin"

    def evaluate(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Robin coefficients at boundary points."""
        points = np.atleast_2d(points)
        n_points = len(points)

        # Evaluate alpha
        if callable(self.alpha):
            alpha_vals = self.alpha(points)
        else:
            alpha_vals = np.full(n_points, self.alpha)

        # Evaluate beta
        if callable(self.beta):
            beta_vals = self.beta(points)
        else:
            beta_vals = np.full(n_points, self.beta)

        # Evaluate g
        if callable(self.value):
            g_vals = self.value(points)
        else:
            g_vals = np.full(n_points, self.value)

        return alpha_vals, beta_vals, g_vals

    def applies_to(self, points: np.ndarray) -> np.ndarray:
        if self.region is None:
            return np.ones(len(points), dtype=bool)
        return self.region(points)

    def get_type(self) -> str:
        return "robin"

    @classmethod
    def convection(cls, h: float, T_inf: float,
                   field_name: str = "T",
                   region: Optional[Callable] = None) -> RobinBC:
        """Create convective BC: h*(T - T_inf) = -k*dT/dn."""
        # Rearranged: h*T + k*dT/dn = h*T_inf
        return cls(
            alpha=h,
            beta=1.0,  # Will be multiplied by k in solver
            value=h * T_inf,
            field_name=field_name,
            region=region,
            name="convection"
        )

    def __repr__(self) -> str:
        return f"RobinBC({self.alpha}*{self.field_name} + {self.beta}*n.grad({self.field_name}) = {self.value})"


@dataclass
class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition: u(x_left) = u(x_right).

    Attributes:
        direction: Axis along which periodicity applies (0, 1, or 2)
        field_name: Name of field this BC applies to
    """
    direction: int
    field_name: str = "u"
    name: str = "periodic"

    def evaluate(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                 time: float = 0.0) -> np.ndarray:
        """Periodic BCs don't have explicit values - handled by solver."""
        raise NotImplementedError("Periodic BCs handled specially by solver")

    def get_type(self) -> str:
        return "periodic"

    def __repr__(self) -> str:
        axis = ['x', 'y', 'z'][self.direction]
        return f"PeriodicBC({self.field_name}, direction={axis})"


@dataclass
class BoundaryConditionSet:
    """
    Collection of boundary conditions for a problem.

    Manages multiple BCs and their application to boundary points.
    """
    conditions: List[BoundaryCondition] = field(default_factory=list)

    def add(self, bc: BoundaryCondition) -> BoundaryConditionSet:
        """Add a boundary condition."""
        self.conditions.append(bc)
        return self

    def get_dirichlet(self) -> List[DirichletBC]:
        """Get all Dirichlet BCs."""
        return [bc for bc in self.conditions if isinstance(bc, DirichletBC)]

    def get_neumann(self) -> List[NeumannBC]:
        """Get all Neumann BCs."""
        return [bc for bc in self.conditions if isinstance(bc, NeumannBC)]

    def get_robin(self) -> List[RobinBC]:
        """Get all Robin BCs."""
        return [bc for bc in self.conditions if isinstance(bc, RobinBC)]

    def get_for_field(self, field_name: str) -> List[BoundaryCondition]:
        """Get all BCs for a specific field."""
        return [bc for bc in self.conditions if bc.field_name == field_name]

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self):
        return iter(self.conditions)

    def __repr__(self) -> str:
        return f"BoundaryConditionSet({len(self.conditions)} conditions)"
