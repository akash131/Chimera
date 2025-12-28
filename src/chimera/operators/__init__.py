"""Neural operators and learned surrogates for Chimera."""

from chimera.operators.neural import (
    NeuralOperator,
    FourierNeuralOperator,
    DeepONet,
)
from chimera.operators.hybrid import (
    PhysicsInformedOperator,
    CorrectorNetwork,
)

__all__ = [
    "NeuralOperator",
    "FourierNeuralOperator",
    "DeepONet",
    "PhysicsInformedOperator",
    "CorrectorNetwork",
]
