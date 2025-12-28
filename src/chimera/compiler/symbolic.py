"""
Symbolic representation for JIT compilation.

Parse mathematical expressions and convert to optimized code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Union, Set
from enum import Enum
import numpy as np
import sympy as sp
from sympy import Symbol, Function, Derivative, Expr, lambdify
from sympy.parsing.sympy_parser import parse_expr


class ExprType(Enum):
    """Expression node types."""
    CONSTANT = "constant"
    VARIABLE = "variable"
    FIELD = "field"
    OPERATOR = "operator"
    FUNCTION = "function"
    DERIVATIVE = "derivative"


@dataclass
class ExprNode:
    """Node in expression tree."""
    type: ExprType
    value: Union[float, str, None] = None
    children: List['ExprNode'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        if self.type == ExprType.CONSTANT:
            return str(self.value)
        elif self.type == ExprType.VARIABLE:
            return self.value
        elif self.type == ExprType.OPERATOR:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0]})"
            else:
                return f"({self.children[0]} {self.value} {self.children[1]})"
        return f"{self.type.value}:{self.value}"


class SymbolicExpression:
    """
    Symbolic expression with compilation support.

    Provides a middle layer between SymPy symbolic math
    and numerical evaluation.
    """

    def __init__(self, expr: Union[str, Expr]):
        if isinstance(expr, str):
            self._sympy_expr = parse_expr(expr)
        else:
            self._sympy_expr = expr

        self._compiled: Optional[Callable] = None
        self._variables: Set[str] = set()
        self._extract_variables()

    def _extract_variables(self):
        """Extract free symbols from expression."""
        for sym in self._sympy_expr.free_symbols:
            self._variables.add(str(sym))

    @property
    def variables(self) -> Set[str]:
        return self._variables

    def differentiate(self, var: str) -> 'SymbolicExpression':
        """Differentiate with respect to variable."""
        sym = sp.Symbol(var)
        deriv = sp.diff(self._sympy_expr, sym)
        return SymbolicExpression(deriv)

    def simplify(self) -> 'SymbolicExpression':
        """Simplify expression."""
        simplified = sp.simplify(self._sympy_expr)
        return SymbolicExpression(simplified)

    def compile(self, backend: str = "numpy") -> Callable:
        """
        Compile to numerical function.

        Args:
            backend: 'numpy' or 'numba'

        Returns:
            Callable that evaluates the expression
        """
        if self._compiled is not None:
            return self._compiled

        # Sort variables for consistent ordering
        var_names = sorted(self._variables)
        var_symbols = [sp.Symbol(v) for v in var_names]

        if backend == "numpy":
            self._compiled = lambdify(var_symbols, self._sympy_expr, modules=['numpy'])
        elif backend == "numba":
            # First compile with numpy, then wrap with numba
            numpy_func = lambdify(var_symbols, self._sympy_expr, modules=['numpy'])
            try:
                import numba
                self._compiled = numba.jit(nopython=True)(numpy_func)
            except ImportError:
                self._compiled = numpy_func
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return self._compiled

    def evaluate(self, **kwargs) -> np.ndarray:
        """Evaluate expression with given variable values."""
        if self._compiled is None:
            self.compile()

        # Get values in correct order
        var_names = sorted(self._variables)
        args = [kwargs[v] for v in var_names]

        return self._compiled(*args)

    def to_code(self, language: str = "python") -> str:
        """Generate code string."""
        if language == "python":
            return str(self._sympy_expr)
        elif language == "c":
            from sympy.printing.ccode import ccode
            return ccode(self._sympy_expr)
        elif language == "fortran":
            from sympy.printing.fcode import fcode
            return fcode(self._sympy_expr)
        else:
            raise ValueError(f"Unknown language: {language}")

    def __add__(self, other: 'SymbolicExpression') -> 'SymbolicExpression':
        return SymbolicExpression(self._sympy_expr + other._sympy_expr)

    def __mul__(self, other: Union['SymbolicExpression', float]) -> 'SymbolicExpression':
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self._sympy_expr * other._sympy_expr)
        return SymbolicExpression(self._sympy_expr * other)

    def __repr__(self):
        return f"SymbolicExpression({self._sympy_expr})"


class SymbolicPDE:
    """
    Symbolic PDE representation for JIT compilation.

    Parses PDEs and generates optimized evaluation code.
    """

    def __init__(self, equation_str: str, field_name: str = "u",
                 spatial_vars: Tuple[str, ...] = ("x", "y")):
        self.equation_str = equation_str
        self.field_name = field_name
        self.spatial_vars = spatial_vars
        self._parse_equation()

    def _parse_equation(self):
        """Parse PDE string into symbolic form."""
        # Create field as function of spatial variables
        spatial_syms = [sp.Symbol(v) for v in self.spatial_vars]
        self.field = sp.Function(self.field_name)(*spatial_syms)

        # Parse equation
        # Support notation: laplacian(u), grad(u), div(u)
        expr_str = self.equation_str

        # Replace differential operators with SymPy equivalents
        expr_str = self._expand_operators(expr_str)

        self._sympy_expr = parse_expr(expr_str)

    def _expand_operators(self, expr_str: str) -> str:
        """Expand differential operators to explicit derivatives."""
        dim = len(self.spatial_vars)
        u = self.field_name

        # Laplacian: laplacian(u) -> d²u/dx² + d²u/dy² + ...
        if "laplacian" in expr_str:
            laplacian_terms = []
            for var in self.spatial_vars:
                laplacian_terms.append(f"Derivative({u}({','.join(self.spatial_vars)}), {var}, 2)")
            laplacian_str = " + ".join(laplacian_terms)
            expr_str = expr_str.replace(f"laplacian({u})", f"({laplacian_str})")

        # Gradient components: grad_x(u), grad_y(u), ...
        for i, var in enumerate(self.spatial_vars):
            grad_str = f"grad_{var}({u})"
            if grad_str in expr_str:
                expr_str = expr_str.replace(
                    grad_str,
                    f"Derivative({u}({','.join(self.spatial_vars)}), {var})"
                )

        return expr_str

    def get_stencil(self, order: int = 2) -> Dict[str, np.ndarray]:
        """
        Generate finite difference stencil coefficients.

        Args:
            order: Accuracy order of finite differences

        Returns:
            Dictionary mapping derivative types to stencil coefficients
        """
        stencils = {}

        # Second derivative stencil (centered)
        if order == 2:
            stencils["d2"] = np.array([1, -2, 1])
        elif order == 4:
            stencils["d2"] = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])

        # First derivative stencil (centered)
        if order == 2:
            stencils["d1"] = np.array([-0.5, 0, 0.5])
        elif order == 4:
            stencils["d1"] = np.array([1/12, -2/3, 0, 2/3, -1/12])

        return stencils

    def compile_residual(self, mesh_shape: Tuple[int, ...],
                         dx: Tuple[float, ...]) -> Callable:
        """
        Compile PDE residual evaluation.

        Generates optimized code for evaluating R(u) = 0.

        Args:
            mesh_shape: Shape of solution array
            dx: Grid spacing in each direction

        Returns:
            Compiled function (u_array) -> residual_array
        """
        dim = len(self.spatial_vars)
        stencils = self.get_stencil(order=2)

        def residual(u: np.ndarray) -> np.ndarray:
            """Evaluate PDE residual."""
            res = np.zeros_like(u)

            if dim == 1:
                dx0 = dx[0]
                for i in range(1, u.shape[0] - 1):
                    # Laplacian term
                    d2u = (u[i-1] - 2*u[i] + u[i+1]) / dx0**2
                    res[i] = -d2u  # For Poisson: -Δu = f

            elif dim == 2:
                dx0, dx1 = dx
                for i in range(1, u.shape[0] - 1):
                    for j in range(1, u.shape[1] - 1):
                        d2u_x = (u[i-1, j] - 2*u[i, j] + u[i+1, j]) / dx0**2
                        d2u_y = (u[i, j-1] - 2*u[i, j] + u[i, j+1]) / dx1**2
                        res[i, j] = -(d2u_x + d2u_y)

            return res

        # Try to compile with numba for speed
        try:
            import numba
            return numba.jit(nopython=True)(residual)
        except ImportError:
            return residual

    def compile_jacobian(self, mesh_shape: Tuple[int, ...],
                         dx: Tuple[float, ...]) -> Callable:
        """
        Compile Jacobian matrix assembly.

        For linear PDEs, returns the system matrix.
        """
        dim = len(self.spatial_vars)

        def jacobian(shape: Tuple[int, ...]) -> np.ndarray:
            """Assemble Jacobian matrix."""
            n = np.prod(shape)
            J = np.zeros((n, n))

            if dim == 1:
                dx0 = dx[0]
                coeff = 1.0 / dx0**2
                for i in range(1, shape[0] - 1):
                    J[i, i-1] = -coeff
                    J[i, i] = 2 * coeff
                    J[i, i+1] = -coeff

            elif dim == 2:
                dx0, dx1 = dx
                cx = 1.0 / dx0**2
                cy = 1.0 / dx1**2

                for i in range(1, shape[0] - 1):
                    for j in range(1, shape[1] - 1):
                        idx = i * shape[1] + j
                        J[idx, idx] = 2 * (cx + cy)
                        J[idx, idx - shape[1]] = -cx
                        J[idx, idx + shape[1]] = -cx
                        J[idx, idx - 1] = -cy
                        J[idx, idx + 1] = -cy

            return J

        return jacobian


class SymbolicCompiler:
    """
    Main compiler interface.

    Converts symbolic PDEs to optimized numerical code.
    """

    def __init__(self):
        self._cache: Dict[str, Callable] = {}

    def compile(self, pde: SymbolicPDE,
                mesh_shape: Tuple[int, ...],
                dx: Tuple[float, ...],
                backend: str = "numpy") -> Dict[str, Callable]:
        """
        Compile PDE to numerical functions.

        Args:
            pde: Symbolic PDE
            mesh_shape: Grid shape
            dx: Grid spacing
            backend: 'numpy' or 'numba'

        Returns:
            Dictionary with compiled functions:
            - 'residual': Evaluate PDE residual
            - 'jacobian': Assemble Jacobian matrix
            - 'apply': Apply operator (matrix-free)
        """
        cache_key = f"{pde.equation_str}_{mesh_shape}_{dx}_{backend}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        compiled = {
            'residual': pde.compile_residual(mesh_shape, dx),
            'jacobian': pde.compile_jacobian(mesh_shape, dx),
            'apply': self._compile_matrix_free(pde, mesh_shape, dx),
        }

        self._cache[cache_key] = compiled
        return compiled

    def _compile_matrix_free(self, pde: SymbolicPDE,
                             mesh_shape: Tuple[int, ...],
                             dx: Tuple[float, ...]) -> Callable:
        """Compile matrix-free operator application."""
        # For iterative solvers: apply A*v without forming A

        dim = len(pde.spatial_vars)

        def apply_operator(v: np.ndarray) -> np.ndarray:
            """Apply PDE operator matrix-free."""
            result = np.zeros_like(v)

            if dim == 2:
                dx0, dx1 = dx
                cx = 1.0 / dx0**2
                cy = 1.0 / dx1**2

                for i in range(1, v.shape[0] - 1):
                    for j in range(1, v.shape[1] - 1):
                        result[i, j] = (
                            2 * (cx + cy) * v[i, j]
                            - cx * v[i-1, j] - cx * v[i+1, j]
                            - cy * v[i, j-1] - cy * v[i, j+1]
                        )

            return result

        try:
            import numba
            return numba.jit(nopython=True)(apply_operator)
        except ImportError:
            return apply_operator

    def clear_cache(self):
        """Clear compilation cache."""
        self._cache.clear()
