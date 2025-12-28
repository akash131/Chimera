"""
Symbolic equation representation for Chimera.

This module provides a symbolic layer for defining PDEs that can be
compiled to different discretization schemes (FEM, FVM, meshless).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, List, Dict, Any, Tuple
from enum import Enum
import numpy as np
import sympy as sp
from sympy import Symbol, Function, Derivative, Expr
from sympy.vector import CoordSys3D


class OperatorType(Enum):
    """Types of differential operators."""
    IDENTITY = "identity"
    GRADIENT = "gradient"
    DIVERGENCE = "divergence"
    LAPLACIAN = "laplacian"
    CURL = "curl"
    ADVECTION = "advection"
    TIME_DERIVATIVE = "time_derivative"


@dataclass
class DifferentialOperator:
    """
    Representation of a differential operator.

    Supports composition and symbolic manipulation.
    """
    op_type: OperatorType
    coefficient: Union[float, Expr] = 1.0
    velocity_field: Optional[str] = None  # For advection

    def __mul__(self, other: Union[float, DifferentialOperator]) -> DifferentialOperator:
        if isinstance(other, (int, float)):
            return DifferentialOperator(
                self.op_type,
                self.coefficient * other,
                self.velocity_field
            )
        raise NotImplementedError("Operator composition")

    def __rmul__(self, other):
        return self.__mul__(other)


# Convenient operator constructors
def grad() -> DifferentialOperator:
    """Gradient operator."""
    return DifferentialOperator(OperatorType.GRADIENT)

def div() -> DifferentialOperator:
    """Divergence operator."""
    return DifferentialOperator(OperatorType.DIVERGENCE)

def laplacian() -> DifferentialOperator:
    """Laplacian operator (div(grad))."""
    return DifferentialOperator(OperatorType.LAPLACIAN)

def curl() -> DifferentialOperator:
    """Curl operator."""
    return DifferentialOperator(OperatorType.CURL)

def advect(velocity: str) -> DifferentialOperator:
    """Advection operator u . grad."""
    return DifferentialOperator(OperatorType.ADVECTION, velocity_field=velocity)

def ddt() -> DifferentialOperator:
    """Time derivative operator."""
    return DifferentialOperator(OperatorType.TIME_DERIVATIVE)


@dataclass
class Term:
    """
    A single term in a PDE.

    Represents coefficient * operator(field)
    """
    field_name: str
    operator: DifferentialOperator
    coefficient: Union[float, Callable, str] = 1.0

    def __repr__(self) -> str:
        return f"Term({self.coefficient} * {self.operator.op_type.value}({self.field_name}))"


@dataclass
class Equation:
    """
    Symbolic PDE equation.

    Represents: sum(terms) = rhs

    Example:
        # Heat equation: -div(k*grad(T)) = f
        eq = Equation(
            terms=[Term('T', laplacian(), coefficient=-k)],
            rhs='f',
            name='heat'
        )
    """
    terms: List[Term]
    rhs: Union[float, str, Callable] = 0.0
    name: str = "equation"

    # Symbolic representation
    _sympy_lhs: Optional[Expr] = field(default=None, repr=False)
    _sympy_rhs: Optional[Expr] = field(default=None, repr=False)

    def __post_init__(self):
        self._build_symbolic()

    def _build_symbolic(self):
        """Build SymPy symbolic representation."""
        x, y, z, t = sp.symbols('x y z t')

        # Create function symbols for each field
        fields = {}
        for term in self.terms:
            if term.field_name not in fields:
                fields[term.field_name] = sp.Function(term.field_name)(x, y, z, t)

        # Build LHS
        lhs_expr = sp.Integer(0)
        for term in self.terms:
            u = fields[term.field_name]
            coeff = term.coefficient if isinstance(term.coefficient, (int, float)) else sp.Symbol(str(term.coefficient))

            if term.operator.op_type == OperatorType.IDENTITY:
                lhs_expr += coeff * u
            elif term.operator.op_type == OperatorType.LAPLACIAN:
                lhs_expr += coeff * (sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2))
            elif term.operator.op_type == OperatorType.TIME_DERIVATIVE:
                lhs_expr += coeff * sp.diff(u, t)
            elif term.operator.op_type == OperatorType.GRADIENT:
                # Returns vector - store symbolically
                lhs_expr += coeff * sp.Matrix([sp.diff(u, x), sp.diff(u, y), sp.diff(u, z)])
            elif term.operator.op_type == OperatorType.DIVERGENCE:
                # Assumes u is vector-valued
                lhs_expr += coeff * (sp.diff(u, x) + sp.diff(u, y) + sp.diff(u, z))

        self._sympy_lhs = lhs_expr

        # Build RHS
        if isinstance(self.rhs, (int, float)):
            self._sympy_rhs = sp.Float(self.rhs)
        elif isinstance(self.rhs, str):
            self._sympy_rhs = sp.Function(self.rhs)(x, y, z, t)
        else:
            self._sympy_rhs = sp.Symbol('rhs')

    @classmethod
    def poisson(cls, field: str = 'u', source: str = 'f') -> Equation:
        """Create Poisson equation: -laplacian(u) = f."""
        return cls(
            terms=[Term(field, laplacian(), coefficient=-1.0)],
            rhs=source,
            name='poisson'
        )

    @classmethod
    def heat_steady(cls, field: str = 'T', conductivity: float = 1.0,
                    source: str = 'Q') -> Equation:
        """Create steady heat equation: -div(k*grad(T)) = Q."""
        return cls(
            terms=[Term(field, laplacian(), coefficient=-conductivity)],
            rhs=source,
            name='heat_steady'
        )

    @classmethod
    def heat_transient(cls, field: str = 'T', conductivity: float = 1.0,
                       density: float = 1.0, specific_heat: float = 1.0,
                       source: str = 'Q') -> Equation:
        """Create transient heat equation: rho*cp*dT/dt - div(k*grad(T)) = Q."""
        rho_cp = density * specific_heat
        return cls(
            terms=[
                Term(field, ddt(), coefficient=rho_cp),
                Term(field, laplacian(), coefficient=-conductivity)
            ],
            rhs=source,
            name='heat_transient'
        )

    @classmethod
    def advection_diffusion(cls, field: str = 'c', velocity: str = 'u',
                            diffusivity: float = 0.01, source: str = 's') -> Equation:
        """Create advection-diffusion: dc/dt + u.grad(c) - D*laplacian(c) = s."""
        return cls(
            terms=[
                Term(field, ddt(), coefficient=1.0),
                Term(field, advect(velocity), coefficient=1.0),
                Term(field, laplacian(), coefficient=-diffusivity)
            ],
            rhs=source,
            name='advection_diffusion'
        )

    def get_field_names(self) -> List[str]:
        """Get all field names in the equation."""
        names = set()
        for term in self.terms:
            names.add(term.field_name)
        if isinstance(self.rhs, str):
            names.add(self.rhs)
        return list(names)

    def is_time_dependent(self) -> bool:
        """Check if equation has time derivatives."""
        return any(
            term.operator.op_type == OperatorType.TIME_DERIVATIVE
            for term in self.terms
        )

    def get_order(self) -> int:
        """Get highest derivative order."""
        order = 0
        for term in self.terms:
            if term.operator.op_type == OperatorType.LAPLACIAN:
                order = max(order, 2)
            elif term.operator.op_type in [OperatorType.GRADIENT, OperatorType.DIVERGENCE]:
                order = max(order, 1)
        return order

    def __repr__(self) -> str:
        terms_str = " + ".join(str(t) for t in self.terms)
        return f"Equation({terms_str} = {self.rhs})"


@dataclass
class PDESystem:
    """
    System of coupled PDEs.

    Supports multiphysics coupling through shared fields.
    """
    equations: List[Equation]
    name: str = "system"

    def __post_init__(self):
        self._validate_coupling()

    def _validate_coupling(self):
        """Validate that coupled fields are consistent."""
        all_fields = set()
        for eq in self.equations:
            all_fields.update(eq.get_field_names())

    @classmethod
    def thermal_stress(cls, temperature_field: str = 'T',
                       displacement_field: str = 'u') -> PDESystem:
        """Create coupled thermal-structural system."""
        heat = Equation.heat_steady(field=temperature_field)
        # Structural equation would be added here
        return cls(equations=[heat], name='thermal_stress')

    def get_primary_fields(self) -> List[str]:
        """Get fields that are solved for (appear with derivatives)."""
        fields = set()
        for eq in self.equations:
            for term in eq.terms:
                fields.add(term.field_name)
        return list(fields)

    def get_secondary_fields(self) -> List[str]:
        """Get fields that are known (only in RHS)."""
        primary = set(self.get_primary_fields())
        all_fields = set()
        for eq in self.equations:
            all_fields.update(eq.get_field_names())
        return list(all_fields - primary)

    def is_coupled(self) -> bool:
        """Check if equations share fields."""
        if len(self.equations) < 2:
            return False
        field_sets = [set(eq.get_field_names()) for eq in self.equations]
        for i in range(len(field_sets)):
            for j in range(i + 1, len(field_sets)):
                if field_sets[i] & field_sets[j]:
                    return True
        return False

    def __len__(self) -> int:
        return len(self.equations)

    def __iter__(self):
        return iter(self.equations)

    def __repr__(self) -> str:
        return f"PDESystem({self.name}, {len(self.equations)} equations)"
