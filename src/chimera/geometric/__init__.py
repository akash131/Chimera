"""
Geometric Deep Learning on Meshes.

Apply graph neural networks and differential geometry to simulations.
"""

from chimera.geometric.mesh_conv import (
    MeshConv,
    GeodesicConv,
    AnisotropicConv,
)
from chimera.geometric.spectral import (
    SpectralConv,
    ChebyshevConv,
    LaplacianEigenmaps,
)
from chimera.geometric.pooling import (
    MeshPooling,
    MeshUnpooling,
    MultiscaleMesh,
)
from chimera.geometric.equivariant import (
    GaugeEquivariantConv,
    FrameBundle,
    ParallelTransport,
)

__all__ = [
    "MeshConv",
    "GeodesicConv",
    "AnisotropicConv",
    "SpectralConv",
    "ChebyshevConv",
    "LaplacianEigenmaps",
    "MeshPooling",
    "MeshUnpooling",
    "MultiscaleMesh",
    "GaugeEquivariantConv",
    "FrameBundle",
    "ParallelTransport",
]
