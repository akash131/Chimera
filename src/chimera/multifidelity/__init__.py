"""
Multi-fidelity methods for Chimera.

Intelligently combine cheap low-fidelity and expensive high-fidelity solvers.
"""

from chimera.multifidelity.fusion import (
    MultiFidelityFusion,
    ControlVariate,
    MFMC,
)
from chimera.multifidelity.transfer import (
    FidelityTransfer,
    CoarseToFine,
    PhysicsTransfer,
)
from chimera.multifidelity.adaptive import (
    AdaptiveFidelity,
    FidelitySelector,
)

__all__ = [
    "MultiFidelityFusion",
    "ControlVariate",
    "MFMC",
    "FidelityTransfer",
    "CoarseToFine",
    "PhysicsTransfer",
    "AdaptiveFidelity",
    "FidelitySelector",
]
