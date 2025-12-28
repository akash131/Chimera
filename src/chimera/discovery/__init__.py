"""
Equation Discovery from Data.

Symbolic regression and SINDy for discovering governing equations.
"""

from chimera.discovery.sindy import (
    SINDy,
    SINDyPI,
    WeakSINDy,
    EnsembleSINDy,
)
from chimera.discovery.symbolic import (
    SymbolicRegression,
    GeneticProgramming,
    NeuralSymbolic,
)
from chimera.discovery.pde_find import (
    PDEFind,
    PDENet,
    PhysicsInformedDiscovery,
)

__all__ = [
    # SINDy
    "SINDy",
    "SINDyPI",
    "WeakSINDy",
    "EnsembleSINDy",
    # Symbolic
    "SymbolicRegression",
    "GeneticProgramming",
    "NeuralSymbolic",
    # PDE Discovery
    "PDEFind",
    "PDENet",
    "PhysicsInformedDiscovery",
]
