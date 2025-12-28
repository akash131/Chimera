"""
Automatic differentiation for Chimera.

Makes the entire solver stack differentiable for gradient-based optimization.
"""

from chimera.autodiff.tape import Tape, Variable, gradient
from chimera.autodiff.differentiable_solver import DifferentiableFEM, DifferentiableAssembly
from chimera.autodiff.adjoint import AdjointSolver, compute_sensitivity

__all__ = [
    "Tape",
    "Variable",
    "gradient",
    "DifferentiableFEM",
    "DifferentiableAssembly",
    "AdjointSolver",
    "compute_sensitivity",
]
