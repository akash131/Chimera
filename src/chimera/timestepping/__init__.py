"""
Neural and adaptive time integration schemes.

Learn optimal time steppers for differential equations.
"""

from chimera.timestepping.neural_ode import (
    NeuralODE,
    AugmentedNeuralODE,
    LatentODE,
)
from chimera.timestepping.learned_integrators import (
    LearnedIntegrator,
    SymplecticNetwork,
    VolumePreservingNetwork,
)
from chimera.timestepping.adaptive import (
    AdaptiveTimestepper,
    ErrorEstimator,
    StepSizeController,
)
from chimera.timestepping.imex import (
    IMEXScheme,
    NeuralIMEX,
    AdaptiveIMEX,
)

__all__ = [
    # Neural ODE
    "NeuralODE",
    "AugmentedNeuralODE",
    "LatentODE",
    # Learned integrators
    "LearnedIntegrator",
    "SymplecticNetwork",
    "VolumePreservingNetwork",
    # Adaptive
    "AdaptiveTimestepper",
    "ErrorEstimator",
    "StepSizeController",
    # IMEX
    "IMEXScheme",
    "NeuralIMEX",
    "AdaptiveIMEX",
]
