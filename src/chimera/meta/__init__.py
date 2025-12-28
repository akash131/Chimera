"""
Meta-Learning for PDE Families.

Learn to solve families of PDEs with few-shot adaptation.
"""

from chimera.meta.meta_solver import (
    MetaSolver,
    MAML,
    Reptile,
    ProtoNet,
)
from chimera.meta.few_shot import (
    FewShotSolver,
    ParameterTransfer,
    TaskEmbedding,
)

__all__ = [
    "MetaSolver",
    "MAML",
    "Reptile",
    "ProtoNet",
    "FewShotSolver",
    "ParameterTransfer",
    "TaskEmbedding",
]
