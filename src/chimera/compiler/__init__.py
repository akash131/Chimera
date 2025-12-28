"""
Symbolic JIT compiler for Chimera.

Compile equations to optimized numerical code at runtime.
"""

from chimera.compiler.symbolic import (
    SymbolicExpression,
    SymbolicPDE,
    SymbolicCompiler,
)
from chimera.compiler.codegen import (
    CodeGenerator,
    NumpyBackend,
    NumbaBackend,
)
from chimera.compiler.optimize import (
    ExpressionOptimizer,
    CommonSubexpressionElimination,
    LoopFusion,
)

__all__ = [
    "SymbolicExpression",
    "SymbolicPDE",
    "SymbolicCompiler",
    "CodeGenerator",
    "NumpyBackend",
    "NumbaBackend",
    "ExpressionOptimizer",
    "CommonSubexpressionElimination",
    "LoopFusion",
]
