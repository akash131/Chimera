"""
Chimera - Hybrid Computational Physics Platform

A novel framework combining FEM, FVM, meshless methods, and neural operators
into a unified solver architecture with adaptive discretization switching.

Includes advanced capabilities:
- Differentiable solvers with automatic adjoint computation
- Multi-fidelity methods for cheap/expensive model fusion
- Symbolic JIT compilation for equations
- Conservation-preserving neural architectures
- Active learning for parameter space exploration
- Uncertainty quantification (Monte Carlo, PCE, MCMC)
- Geometric deep learning on meshes
"""

__version__ = "0.2.0"

from chimera.core.domain import Domain
from chimera.core.field import Field
from chimera.core.equation import Equation, PDESystem
from chimera.core.boundary import BoundaryCondition, DirichletBC, NeumannBC
from chimera.mesh.mesh import Mesh
from chimera.solvers.hybrid import HybridSolver
from chimera.physics.heat import HeatEquation
from chimera.physics.elasticity import LinearElasticity
from chimera.physics.fluid import NavierStokes, Stokes

# Advanced modules
from chimera import autodiff
from chimera import multifidelity
from chimera import compiler
from chimera import active
from chimera import uq
from chimera import geometric

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
    # Advanced modules
    "autodiff",
    "multifidelity",
    "compiler",
    "active",
    "uq",
    "geometric",
]
