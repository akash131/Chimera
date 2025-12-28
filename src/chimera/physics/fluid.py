"""
Fluid mechanics physics module.

Incompressible and compressible flow equations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

from chimera.core.equation import Equation, Term, PDESystem, laplacian, ddt, grad, div, advect
from chimera.core.boundary import DirichletBC, NeumannBC
from chimera.core.field import Field, FieldType


@dataclass
class FluidProperties:
    """Fluid material properties."""
    rho: float = 1000.0  # Density (kg/m³)
    mu: float = 0.001  # Dynamic viscosity (Pa·s)
    name: str = "water"

    @property
    def nu(self) -> float:
        """Kinematic viscosity."""
        return self.mu / self.rho

    @classmethod
    def water(cls) -> FluidProperties:
        return cls(rho=1000, mu=0.001, name="water")

    @classmethod
    def air(cls) -> FluidProperties:
        return cls(rho=1.2, mu=1.8e-5, name="air")

    @classmethod
    def oil(cls) -> FluidProperties:
        return cls(rho=900, mu=0.1, name="oil")

    @classmethod
    def glycerin(cls) -> FluidProperties:
        return cls(rho=1260, mu=1.5, name="glycerin")


class Stokes:
    """
    Stokes equations for creeping flow (Re << 1).

    -μ∇²u + ∇p = f
    ∇·u = 0
    """

    def __init__(self, fluid: Optional[FluidProperties] = None,
                 dim: int = 2):
        self.fluid = fluid or FluidProperties()
        self.dim = dim
        self._body_force: Optional[np.ndarray] = None

    def set_body_force(self, force: np.ndarray):
        """Set body force (e.g., gravity)."""
        self._body_force = force

    def get_system(self) -> PDESystem:
        """Build the equation system."""
        # Momentum equations (one per dimension)
        momentum_eqs = []
        for i in range(self.dim):
            eq = Equation(
                terms=[
                    Term(f'u{i}', laplacian(), coefficient=-self.fluid.mu),
                    # Pressure gradient would be Term('p', grad())
                ],
                rhs=f'f{i}' if self._body_force is None else self._body_force[i],
                name=f'momentum_{i}'
            )
            momentum_eqs.append(eq)

        # Continuity equation: ∇·u = 0
        # Would need divergence of velocity field

        return PDESystem(equations=momentum_eqs, name='stokes')

    # Convenience BC constructors
    @staticmethod
    def no_slip(region: Optional[Callable] = None) -> DirichletBC:
        """No-slip wall BC (u = 0)."""
        return DirichletBC(value=0.0, field_name='u', region=region, name='no_slip')

    @staticmethod
    def inlet_velocity(velocity: np.ndarray,
                       region: Optional[Callable] = None) -> DirichletBC:
        """Inlet velocity BC."""
        return DirichletBC(value=velocity, field_name='u', region=region, name='inlet')

    @staticmethod
    def outlet_pressure(p: float = 0.0,
                        region: Optional[Callable] = None) -> DirichletBC:
        """Outlet pressure BC."""
        return DirichletBC(value=p, field_name='p', region=region, name='outlet')

    @staticmethod
    def symmetry(region: Optional[Callable] = None) -> NeumannBC:
        """Symmetry BC (zero normal velocity, zero tangential stress)."""
        return NeumannBC(value=0.0, field_name='u', region=region, name='symmetry')


class NavierStokes(Stokes):
    """
    Navier-Stokes equations for viscous flow.

    ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f
    ∇·u = 0
    """

    def __init__(self, fluid: Optional[FluidProperties] = None,
                 dim: int = 2, transient: bool = True):
        super().__init__(fluid, dim)
        self.transient = transient

    def reynolds_number(self, L: float, U: float) -> float:
        """Compute Reynolds number."""
        return self.fluid.rho * U * L / self.fluid.mu

    def get_system(self) -> PDESystem:
        """Build the equation system."""
        momentum_eqs = []

        for i in range(self.dim):
            terms = []

            if self.transient:
                terms.append(Term(f'u{i}', ddt(), coefficient=self.fluid.rho))

            # Advection term (nonlinear)
            terms.append(Term(f'u{i}', advect('u'), coefficient=self.fluid.rho))

            # Diffusion term
            terms.append(Term(f'u{i}', laplacian(), coefficient=-self.fluid.mu))

            eq = Equation(
                terms=terms,
                rhs=f'f{i}' if self._body_force is None else self._body_force[i],
                name=f'momentum_{i}'
            )
            momentum_eqs.append(eq)

        return PDESystem(equations=momentum_eqs, name='navier_stokes')


class Euler:
    """
    Euler equations for inviscid compressible flow.

    ∂ρ/∂t + ∇·(ρu) = 0
    ∂(ρu)/∂t + ∇·(ρu⊗u) + ∇p = 0
    ∂E/∂t + ∇·((E+p)u) = 0
    """

    def __init__(self, gamma: float = 1.4, dim: int = 2):
        self.gamma = gamma  # Specific heat ratio
        self.dim = dim

    def pressure(self, rho: np.ndarray, rho_u: np.ndarray,
                 E: np.ndarray) -> np.ndarray:
        """Compute pressure from conservative variables."""
        u_squared = np.sum(rho_u**2, axis=-1) / rho**2
        return (self.gamma - 1) * (E - 0.5 * rho * u_squared)

    def sound_speed(self, rho: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute sound speed."""
        return np.sqrt(self.gamma * p / rho)

    def max_eigenvalue(self, rho: np.ndarray, rho_u: np.ndarray,
                       p: np.ndarray) -> float:
        """Maximum wave speed for CFL condition."""
        u_mag = np.sqrt(np.sum(rho_u**2, axis=-1)) / rho
        c = self.sound_speed(rho, p)
        return float(np.max(u_mag + c))

    def flux(self, U: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute flux in given direction.

        Args:
            U: Conservative variables [rho, rho*u, rho*v, E]
            direction: 0 for x, 1 for y

        Returns:
            Flux vector
        """
        rho = U[..., 0]
        rho_u = U[..., 1:1 + self.dim]
        E = U[..., -1]

        u = rho_u / rho[..., np.newaxis]
        p = self.pressure(rho, rho_u, E)

        F = np.zeros_like(U)

        # Mass flux
        F[..., 0] = rho_u[..., direction]

        # Momentum flux
        for i in range(self.dim):
            F[..., 1 + i] = rho_u[..., i] * u[..., direction]
            if i == direction:
                F[..., 1 + i] += p

        # Energy flux
        F[..., -1] = (E + p) * u[..., direction]

        return F


class PotentialFlow:
    """
    Potential flow (irrotational, inviscid).

    ∇²φ = 0
    u = ∇φ
    """

    def __init__(self, dim: int = 2):
        self.dim = dim

    def get_equation(self) -> Equation:
        """Get Laplace equation for velocity potential."""
        return Equation.poisson(field='phi', source='0')

    def velocity_from_potential(self, phi: Field) -> Field:
        """Compute velocity from potential."""
        return phi.gradient()

    def pressure_coefficient(self, u: np.ndarray, U_inf: float) -> np.ndarray:
        """
        Compute pressure coefficient.

        Cp = 1 - (u/U_inf)²
        """
        u_mag = np.linalg.norm(u, axis=-1)
        return 1 - (u_mag / U_inf)**2

    @staticmethod
    def uniform_flow(U: float, direction: np.ndarray) -> Callable:
        """Uniform flow potential."""
        direction = direction / np.linalg.norm(direction)
        return lambda x: U * np.dot(x, direction)

    @staticmethod
    def source(strength: float, location: np.ndarray) -> Callable:
        """Point source potential."""
        def potential(x):
            r = np.linalg.norm(x - location, axis=-1)
            return strength / (2 * np.pi) * np.log(r + 1e-12)
        return potential

    @staticmethod
    def vortex(strength: float, location: np.ndarray) -> Callable:
        """Point vortex potential (stream function)."""
        def potential(x):
            dx = x - location
            theta = np.arctan2(dx[..., 1], dx[..., 0])
            return strength / (2 * np.pi) * theta
        return potential


def compute_vorticity(velocity: Field) -> Field:
    """Compute vorticity from velocity field."""
    if velocity.dim == 2:
        # 2D: ω = ∂v/∂x - ∂u/∂y
        # Would need proper derivative computation
        pass
    else:
        # 3D: ω = ∇ × u
        pass
    raise NotImplementedError("Vorticity computation")


def compute_stream_function(velocity: Field) -> Field:
    """Compute stream function from 2D velocity field."""
    if velocity.dim != 2:
        raise ValueError("Stream function only defined for 2D")
    # Would solve ∇²ψ = -ω
    raise NotImplementedError("Stream function computation")
