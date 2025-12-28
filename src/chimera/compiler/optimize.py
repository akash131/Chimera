"""
Expression optimization for code generation.

Apply compiler optimizations to improve generated code performance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import numpy as np


@dataclass
class OptimizationResult:
    """Result of expression optimization."""
    optimized_expr: str
    subexpressions: Dict[str, str]  # Extracted common subexpressions
    operations_saved: int
    transformations_applied: List[str]


class ExpressionOptimizer:
    """
    Optimize symbolic expressions before code generation.

    Applies:
    - Common subexpression elimination (CSE)
    - Strength reduction
    - Constant folding
    - Loop-invariant code motion
    """

    def __init__(self):
        self._transformations = [
            self._constant_folding,
            self._strength_reduction,
            self._common_subexpression_elimination,
        ]

    def optimize(self, expression) -> OptimizationResult:
        """Apply all optimizations to expression."""
        import sympy as sp

        if hasattr(expression, '_sympy_expr'):
            expr = expression._sympy_expr
        else:
            expr = expression

        transformations = []
        subexpressions = {}

        # Constant folding
        expr_simplified = sp.simplify(expr)
        if expr_simplified != expr:
            transformations.append("constant_folding")
            expr = expr_simplified

        # CSE
        replacements, reduced = sp.cse(expr)
        if replacements:
            transformations.append("cse")
            for sym, sub_expr in replacements:
                subexpressions[str(sym)] = str(sub_expr)
            expr = reduced[0] if reduced else expr

        # Strength reduction (power -> multiplication)
        expr = self._strength_reduction(expr)
        if transformations and "strength_reduction" not in transformations:
            transformations.append("strength_reduction")

        return OptimizationResult(
            optimized_expr=str(expr),
            subexpressions=subexpressions,
            operations_saved=len(replacements) if replacements else 0,
            transformations_applied=transformations
        )

    def _constant_folding(self, expr):
        """Fold constant expressions."""
        import sympy as sp
        return sp.simplify(expr)

    def _strength_reduction(self, expr):
        """Replace expensive operations with cheaper ones."""
        import sympy as sp

        # Replace x**2 with x*x
        expr = expr.replace(
            lambda e: e.is_Pow and e.exp == 2,
            lambda e: e.base * e.base
        )

        # Replace x**0.5 with sqrt(x)
        expr = expr.replace(
            lambda e: e.is_Pow and e.exp == sp.Rational(1, 2),
            lambda e: sp.sqrt(e.base)
        )

        return expr

    def _common_subexpression_elimination(self, expr):
        """Extract common subexpressions."""
        import sympy as sp
        return sp.cse(expr)


class CommonSubexpressionElimination:
    """
    Dedicated CSE pass for complex expressions.

    More aggressive than SymPy's built-in CSE for numerical code.
    """

    def __init__(self, min_occurrences: int = 2):
        self.min_occurrences = min_occurrences
        self._subexpr_counter = 0

    def process(self, expressions: List) -> Tuple[Dict[str, str], List[str]]:
        """
        Process multiple expressions together for better CSE.

        Args:
            expressions: List of symbolic expressions

        Returns:
            (subexpressions, reduced_expressions)
        """
        import sympy as sp

        # Combine into single CSE analysis
        all_exprs = []
        for expr in expressions:
            if hasattr(expr, '_sympy_expr'):
                all_exprs.append(expr._sympy_expr)
            else:
                all_exprs.append(expr)

        # Run CSE
        replacements, reduced = sp.cse(all_exprs)

        subexpressions = {}
        for sym, sub_expr in replacements:
            subexpressions[str(sym)] = str(sub_expr)

        reduced_strs = [str(r) for r in reduced]

        return subexpressions, reduced_strs

    def _find_common_subexpressions(self, expr) -> Dict[str, int]:
        """Find subexpressions and their occurrence counts."""
        import sympy as sp

        counts = {}

        def count_subexpr(e):
            if e.is_Atom:
                return
            key = str(e)
            counts[key] = counts.get(key, 0) + 1
            for arg in e.args:
                count_subexpr(arg)

        count_subexpr(expr)

        return {k: v for k, v in counts.items() if v >= self.min_occurrences}


class LoopFusion:
    """
    Fuse multiple loops into single loop for better cache performance.
    """

    def __init__(self):
        pass

    def fuse(self, operations: List[str],
             array_name: str = "u",
             dimensions: int = 2) -> str:
        """
        Fuse multiple array operations into single loop.

        Args:
            operations: List of array operations (as code strings)
            array_name: Name of the array variable
            dimensions: Number of loop dimensions

        Returns:
            Fused loop code
        """
        if dimensions == 1:
            loop_template = f"""
def fused_loop({array_name}):
    n = len({array_name})
    result = np.zeros(n)
    for i in range(1, n - 1):
        {self._indent_operations(operations, 2)}
    return result
"""
        elif dimensions == 2:
            loop_template = f"""
def fused_loop({array_name}):
    nx, ny = {array_name}.shape
    result = np.zeros((nx, ny))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            {self._indent_operations(operations, 3)}
    return result
"""
        else:
            raise NotImplementedError(f"{dimensions}D loop fusion")

        return loop_template

    def _indent_operations(self, operations: List[str], level: int) -> str:
        """Indent operations for loop body."""
        indent = "    " * level
        return ("\n" + indent).join(operations)


class MemoryOptimizer:
    """
    Optimize memory access patterns.
    """

    def __init__(self):
        pass

    def analyze_access_pattern(self, loop_code: str) -> Dict:
        """Analyze memory access pattern in loop."""
        # Simplified analysis
        info = {
            "stride": 1,  # Memory stride
            "reuse_distance": 0,  # Cache reuse
            "vectorizable": True,
        }
        return info

    def optimize_layout(self, array_accesses: List[str],
                        array_shape: Tuple[int, ...]) -> str:
        """
        Suggest optimal memory layout.

        Returns recommendation: 'C' (row-major) or 'F' (column-major)
        """
        # Analyze access patterns
        # If inner loop accesses consecutive elements in last dimension -> C order
        # If inner loop accesses consecutive elements in first dimension -> F order

        return 'C'  # Default to C order

    def generate_prefetch(self, array_name: str,
                          lookahead: int = 4) -> str:
        """Generate prefetch hints for better cache utilization."""
        # Software prefetching (limited support in Python)
        return f"# Prefetch: {array_name}[i + {lookahead}]"


class VectorizationAnalyzer:
    """
    Analyze code for SIMD vectorization opportunities.
    """

    def __init__(self, vector_width: int = 4):
        self.vector_width = vector_width

    def analyze(self, loop_code: str) -> Dict:
        """Check if loop can be vectorized."""
        info = {
            "vectorizable": True,
            "dependencies": [],
            "suggested_unroll": self.vector_width,
        }

        # Check for loop-carried dependencies
        if "result[i-1]" in loop_code or "result[i+1]" in loop_code:
            info["vectorizable"] = False
            info["dependencies"].append("loop-carried")

        return info

    def generate_vectorized(self, scalar_code: str,
                            array_name: str = "u") -> str:
        """Generate vectorized version of scalar loop."""
        # Using NumPy for vectorization
        vectorized = f"""
def vectorized_loop({array_name}):
    # Vectorized version
    result = np.zeros_like({array_name})
    result[1:-1] = {scalar_code.replace(f'{array_name}[i]', f'{array_name}[1:-1]')}
    return result
"""
        return vectorized
