"""Solver backends for Chimera."""

from chimera.solvers.base import Solver, SolverResult
from chimera.solvers.fem import FEMSolver
from chimera.solvers.fvm import FVMSolver
from chimera.solvers.meshless import MeshlessSolver
from chimera.solvers.hybrid import HybridSolver

__all__ = [
    "Solver",
    "SolverResult",
    "FEMSolver",
    "FVMSolver",
    "MeshlessSolver",
    "HybridSolver",
]
