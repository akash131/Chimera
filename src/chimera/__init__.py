"""
Chimera - Hybrid Computational Physics Platform

A novel framework combining FEM, FVM, meshless methods, and neural operators
into a unified solver architecture with adaptive discretization switching.
"""

__version__ = "0.1.0"

from chimera.core.domain import Domain
from chimera.core.field import Field
from chimera.core.equation import Equation, PDESystem
from chimera.core.boundary import BoundaryCondition, DirichletBC, NeumannBC
from chimera.mesh.mesh import Mesh
from chimera.solvers.hybrid import HybridSolver
from chimera.physics.heat import HeatEquation
from chimera.physics.elasticity import LinearElasticity
from chimera.physics.fluid import NavierStokes, Stokes

__all__ = [
    # Core
    "Domain",
    "Field",
    "Equation",
    "PDESystem",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    # Mesh
    "Mesh",
    # Solvers
    "HybridSolver",
    # Physics
    "HeatEquation",
    "LinearElasticity",
    "NavierStokes",
    "Stokes",
]
