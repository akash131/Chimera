"""
Reduced Order Models (ROM).

Fast surrogate models for real-time simulation.
"""

from chimera.rom.pod import (
    POD,
    PODGalerkin,
    PODPetrovGalerkin,
    AdaptivePOD,
)
from chimera.rom.dmd import (
    DMD,
    ExtendedDMD,
    KernelDMD,
    StreamingDMD,
    OptimizedDMD,
    BayesianDMD,
)
from chimera.rom.autoencoder import (
    LinearAutoencoder,
    ConvAutoencoder,
    VariationalAutoencoder,
    PhysicsAutoencoder,
)
from chimera.rom.interpolation import (
    RBFInterpolator,
    GrassmannInterpolator,
    ManifoldInterpolator,
)
from chimera.rom.hyper_reduction import (
    DEIM,
    QDEIM,
    GNAT,
    EmpiricalCubature,
)

__all__ = [
    # POD
    "POD",
    "PODGalerkin",
    "PODPetrovGalerkin",
    "AdaptivePOD",
    # DMD
    "DMD",
    "ExtendedDMD",
    "KernelDMD",
    "StreamingDMD",
    "OptimizedDMD",
    "BayesianDMD",
    # Autoencoders
    "LinearAutoencoder",
    "ConvAutoencoder",
    "VariationalAutoencoder",
    "PhysicsAutoencoder",
    # Interpolation
    "RBFInterpolator",
    "GrassmannInterpolator",
    "ManifoldInterpolator",
    # Hyper-reduction
    "DEIM",
    "QDEIM",
    "GNAT",
    "EmpiricalCubature",
]
