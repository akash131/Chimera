"""
Koopman Operator Methods.

Linear embedding of nonlinear dynamics.
"""

from chimera.koopman.koopman import (
    KoopmanOperator,
    ExtendedDMDKoopman,
    DeepKoopman,
    KoopmanMPC,
)

__all__ = [
    "KoopmanOperator",
    "ExtendedDMDKoopman",
    "DeepKoopman",
    "KoopmanMPC",
]
