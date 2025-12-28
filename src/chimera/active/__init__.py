"""
Active learning for simulation.

Intelligently explore parameter space to minimize total simulation cost.
"""

from chimera.active.sampling import (
    ActiveSampler,
    UncertaintySampling,
    ExpectedImprovement,
    QueryByCommittee,
)
from chimera.active.surrogate import (
    GaussianProcess,
    BayesianNeuralNetwork,
)

__all__ = [
    "ActiveSampler",
    "UncertaintySampling",
    "ExpectedImprovement",
    "QueryByCommittee",
    "GaussianProcess",
    "BayesianNeuralNetwork",
]
