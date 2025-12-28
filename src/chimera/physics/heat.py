"""
Heat transfer physics module.

Provides ready-to-use heat transfer equations and material models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Union
import numpy as np

from chimera.core.equation import Equation, Term, laplacian, ddt, OperatorType
from chimera.core.boundary import DirichletBC, NeumannBC, RobinBC
from chimera.core.field import Field


@dataclass
class ThermalMaterial:
    """Thermal material properties."""
    conductivity: float = 1.0  # W/(m·K)
    density: float = 1000.0  # kg/m³
    specific_heat: float = 1000.0  # J/(kg·K)
    name: str = "generic"

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity α = k/(ρ·cp)."""
        return self.conductivity / (self.density * self.specific_heat)

    @classmethod
    def aluminum(cls) -> ThermalMaterial:
        return cls(conductivity=205, density=2700, specific_heat=900, name="aluminum")

    @classmethod
    def steel(cls) -> ThermalMaterial:
        return cls(conductivity=50, density=7850, specific_heat=500, name="steel")

    @classmethod
    def copper(cls) -> ThermalMaterial:
        return cls(conductivity=400, density=8960, specific_heat=385, name="copper")

    @classmethod
    def air(cls) -> ThermalMaterial:
        return cls(conductivity=0.026, density=1.2, specific_heat=1005, name="air")

    @classmethod
    def water(cls) -> ThermalMaterial:
        return cls(conductivity=0.6, density=1000, specific_heat=4186, name="water")


class HeatEquation:
    """
    Heat conduction equation.

    Steady: -∇·(k∇T) = Q
    Transient: ρcp ∂T/∂t - ∇·(k∇T) = Q
    """

    def __init__(self, material: Optional[ThermalMaterial] = None,
                 transient: bool = False):
        self.material = material or ThermalMaterial()
        self.transient = transient
        self._source: Optional[Callable] = None

    def set_source(self, source: Union[float, Callable[[np.ndarray], np.ndarray]]):
        """Set volumetric heat source Q (W/m³)."""
        if isinstance(source, (int, float)):
            self._source = lambda x: np.full(len(x), source)
        else:
            self._source = source

    def get_equation(self) -> Equation:
        """Build the equation object."""
        terms = []

        if self.transient:
            rho_cp = self.material.density * self.material.specific_heat
            terms.append(Term('T', ddt(), coefficient=rho_cp))

        terms.append(Term('T', laplacian(), coefficient=-self.material.conductivity))

        return Equation(
            terms=terms,
            rhs='Q' if self._source is None else 0.0,
            name='heat'
        )

    def get_source(self) -> Optional[Callable]:
        """Get source function."""
        return self._source

    # Convenience BC constructors
    @staticmethod
    def fixed_temperature(value: float, region: Optional[Callable] = None) -> DirichletBC:
        """Fixed temperature BC."""
        return DirichletBC(value=value, field_name='T', region=region, name='fixed_temp')

    @staticmethod
    def heat_flux(flux: float, region: Optional[Callable] = None) -> NeumannBC:
        """Heat flux BC (positive = into domain)."""
        return NeumannBC(value=flux, field_name='T', region=region, name='heat_flux')

    @staticmethod
    def insulated(region: Optional[Callable] = None) -> NeumannBC:
        """Insulated (adiabatic) BC."""
        return NeumannBC.zero_flux(field_name='T', region=region)

    @staticmethod
    def convection(h: float, T_ambient: float,
                   region: Optional[Callable] = None) -> RobinBC:
        """Convective heat transfer BC."""
        return RobinBC.convection(h=h, T_inf=T_ambient, field_name='T', region=region)


class StefanBoltzmann:
    """
    Radiative heat transfer using Stefan-Boltzmann law.

    q = εσ(T⁴ - T_ambient⁴)
    """
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant W/(m²·K⁴)

    def __init__(self, emissivity: float = 1.0, T_ambient: float = 300.0):
        self.emissivity = emissivity
        self.T_ambient = T_ambient

    def flux(self, T: np.ndarray) -> np.ndarray:
        """Compute radiative heat flux."""
        return self.emissivity * self.SIGMA * (T**4 - self.T_ambient**4)

    def linearized_coefficient(self, T: np.ndarray) -> np.ndarray:
        """Linearized radiation coefficient for Robin BC."""
        # q ≈ h_rad * (T - T_ambient) where h_rad = 4εσT³
        T_avg = 0.5 * (T + self.T_ambient)
        return 4 * self.emissivity * self.SIGMA * T_avg**3

    def get_robin_bc(self, T_estimate: float,
                     region: Optional[Callable] = None) -> RobinBC:
        """Create linearized radiation BC."""
        h_rad = 4 * self.emissivity * self.SIGMA * T_estimate**3
        return RobinBC(
            alpha=h_rad,
            beta=1.0,
            value=h_rad * self.T_ambient,
            field_name='T',
            region=region,
            name='radiation'
        )


class ThermalContact:
    """Thermal contact resistance between two regions."""

    def __init__(self, resistance: float = 1e-4):
        """
        Args:
            resistance: Thermal contact resistance (m²·K/W)
        """
        self.resistance = resistance

    @property
    def conductance(self) -> float:
        """Contact conductance (W/(m²·K))."""
        return 1.0 / self.resistance

    def flux(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """Heat flux from region 1 to region 2."""
        return self.conductance * (T1 - T2)
