"""
Domain Decomposition Methods.

Parallel solvers for large-scale problems.
"""

from chimera.decomposition.schwarz import (
    AdditiveSchwarz,
    MultiplicativeSchwarz,
    RestrictedAdditiveSchwarz,
    OptimizedSchwarz,
)
from chimera.decomposition.feti import (
    FETI,
    FETIDP,
    TFETI,
)
from chimera.decomposition.partitioning import (
    DomainPartitioner,
    GraphPartitioner,
    GeometricPartitioner,
)
from chimera.decomposition.coarse import (
    CoarseSpace,
    NicolaidesCoarse,
    GenEOCoarse,
    SpectralCoarse,
)

__all__ = [
    # Schwarz
    "AdditiveSchwarz",
    "MultiplicativeSchwarz",
    "RestrictedAdditiveSchwarz",
    "OptimizedSchwarz",
    # FETI
    "FETI",
    "FETIDP",
    "TFETI",
    # Partitioning
    "DomainPartitioner",
    "GraphPartitioner",
    "GeometricPartitioner",
    # Coarse spaces
    "CoarseSpace",
    "NicolaidesCoarse",
    "GenEOCoarse",
    "SpectralCoarse",
]
