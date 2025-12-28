"""
Structural mechanics physics module.

Linear and nonlinear elasticity for structural analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

from chimera.core.equation import Equation, Term, PDESystem
from chimera.core.boundary import DirichletBC, NeumannBC
from chimera.core.field import Field, FieldType


@dataclass
class ElasticMaterial:
    """Elastic material properties."""
    E: float = 200e9  # Young's modulus (Pa)
    nu: float = 0.3  # Poisson's ratio
    rho: float = 7850  # Density (kg/m³)
    name: str = "steel"

    @property
    def lambda_(self) -> float:
        """First Lamé parameter."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        """Shear modulus (second Lamé parameter)."""
        return self.E / (2 * (1 + self.nu))

    @property
    def K(self) -> float:
        """Bulk modulus."""
        return self.E / (3 * (1 - 2 * self.nu))

    def constitutive_matrix_3d(self) -> np.ndarray:
        """3D constitutive matrix (6x6 Voigt notation)."""
        E, nu = self.E, self.nu
        factor = E / ((1 + nu) * (1 - 2 * nu))

        C = factor * np.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ])
        return C

    def constitutive_matrix_plane_stress(self) -> np.ndarray:
        """Plane stress constitutive matrix (3x3)."""
        E, nu = self.E, self.nu
        factor = E / (1 - nu**2)

        C = factor * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2],
        ])
        return C

    def constitutive_matrix_plane_strain(self) -> np.ndarray:
        """Plane strain constitutive matrix (3x3)."""
        E, nu = self.E, self.nu
        factor = E / ((1 + nu) * (1 - 2 * nu))

        C = factor * np.array([
            [1 - nu, nu, 0],
            [nu, 1 - nu, 0],
            [0, 0, (1 - 2 * nu) / 2],
        ])
        return C

    @classmethod
    def aluminum(cls) -> ElasticMaterial:
        return cls(E=70e9, nu=0.33, rho=2700, name="aluminum")

    @classmethod
    def steel(cls) -> ElasticMaterial:
        return cls(E=200e9, nu=0.3, rho=7850, name="steel")

    @classmethod
    def titanium(cls) -> ElasticMaterial:
        return cls(E=116e9, nu=0.34, rho=4500, name="titanium")

    @classmethod
    def rubber(cls) -> ElasticMaterial:
        return cls(E=0.01e9, nu=0.49, rho=1100, name="rubber")


class LinearElasticity:
    """
    Linear elasticity equations.

    Equilibrium: -∇·σ = f
    Constitutive: σ = C : ε
    Kinematics: ε = (∇u + ∇u^T) / 2
    """

    def __init__(self, material: Optional[ElasticMaterial] = None,
                 dim: int = 2):
        self.material = material or ElasticMaterial()
        self.dim = dim
        self._body_force: Optional[np.ndarray] = None

    def set_body_force(self, force: np.ndarray):
        """Set body force (e.g., gravity)."""
        self._body_force = force

    def gravity(self, g: float = 9.81, direction: int = -1):
        """Set gravity body force."""
        self._body_force = np.zeros(self.dim)
        axis = abs(direction) - 1 if direction < 0 else direction
        self._body_force[axis] = np.sign(direction) * self.material.rho * g

    def get_stiffness_matrix_element(self, coords: np.ndarray,
                                     B: np.ndarray) -> np.ndarray:
        """
        Compute element stiffness matrix.

        K_e = ∫ B^T C B dV

        Args:
            coords: Element node coordinates
            B: Strain-displacement matrix

        Returns:
            Element stiffness matrix
        """
        if self.dim == 2:
            C = self.material.constitutive_matrix_plane_stress()
        else:
            C = self.material.constitutive_matrix_3d()

        return B.T @ C @ B

    # Convenience BC constructors
    @staticmethod
    def fixed(region: Optional[Callable] = None) -> DirichletBC:
        """Fully fixed (zero displacement) BC."""
        return DirichletBC(value=0.0, field_name='u', region=region, name='fixed')

    @staticmethod
    def prescribed_displacement(value: np.ndarray,
                                 region: Optional[Callable] = None) -> DirichletBC:
        """Prescribed displacement BC."""
        return DirichletBC(value=value, field_name='u', region=region, name='displacement')

    @staticmethod
    def traction(force: np.ndarray,
                 region: Optional[Callable] = None) -> NeumannBC:
        """Surface traction BC."""
        return NeumannBC(value=force, field_name='u', region=region, name='traction')

    @staticmethod
    def pressure(p: float, region: Optional[Callable] = None) -> NeumannBC:
        """Pressure load (normal to surface)."""
        # Would need normal vector information
        return NeumannBC(value=-p, field_name='u', region=region, name='pressure')


class PlaneStress(LinearElasticity):
    """Plane stress formulation (thin plates)."""

    def __init__(self, material: Optional[ElasticMaterial] = None,
                 thickness: float = 1.0):
        super().__init__(material, dim=2)
        self.thickness = thickness


class PlaneStrain(LinearElasticity):
    """Plane strain formulation (thick sections)."""

    def __init__(self, material: Optional[ElasticMaterial] = None):
        super().__init__(material, dim=2)
        self._use_plane_strain = True


def compute_von_mises(stress: np.ndarray) -> np.ndarray:
    """
    Compute von Mises equivalent stress.

    Args:
        stress: Stress components [σxx, σyy, σzz, τxy, τyz, τxz] or [σxx, σyy, τxy]

    Returns:
        von Mises stress
    """
    if len(stress.shape) == 1:
        stress = stress.reshape(1, -1)

    if stress.shape[1] == 3:
        # 2D (plane stress/strain)
        sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
        vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    else:
        # 3D
        sxx, syy, szz = stress[:, 0], stress[:, 1], stress[:, 2]
        sxy, syz, sxz = stress[:, 3], stress[:, 4], stress[:, 5]

        vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) +
                     3 * (sxy**2 + syz**2 + sxz**2))

    return vm


def compute_principal_stresses(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal stresses and directions.

    Returns:
        principal: Principal stress values (sorted)
        directions: Principal directions (eigenvectors)
    """
    if stress.shape[-1] == 3:
        # 2D
        sxx, syy, sxy = stress[..., 0], stress[..., 1], stress[..., 2]

        s_avg = 0.5 * (sxx + syy)
        R = np.sqrt(0.25 * (sxx - syy)**2 + sxy**2)

        s1 = s_avg + R
        s2 = s_avg - R

        theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)

        return np.stack([s1, s2], axis=-1), theta

    else:
        # 3D - compute eigenvalues
        stress_tensor = np.array([
            [stress[0], stress[3], stress[5]],
            [stress[3], stress[1], stress[4]],
            [stress[5], stress[4], stress[2]]
        ])
        eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)
        # Sort by magnitude
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]
