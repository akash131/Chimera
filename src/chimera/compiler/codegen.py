"""
Code generation backends for Chimera compiler.

Generate optimized numerical code from symbolic expressions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Tuple
import numpy as np


@dataclass
class GeneratedCode:
    """Container for generated code."""
    source: str
    language: str
    function: Optional[Callable] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CodeGenerator(ABC):
    """Abstract base for code generation backends."""

    @abstractmethod
    def generate(self, expression, variables: List[str]) -> GeneratedCode:
        """Generate code for expression."""
        pass

    @abstractmethod
    def compile(self, code: GeneratedCode) -> Callable:
        """Compile generated code to callable."""
        pass


class NumpyBackend(CodeGenerator):
    """
    NumPy code generation backend.

    Generates vectorized NumPy code for fast evaluation.
    """

    def __init__(self):
        self._imports = "import numpy as np\n"

    def generate(self, expression, variables: List[str]) -> GeneratedCode:
        """Generate NumPy-compatible Python code."""
        # Convert SymPy expression to NumPy code
        from sympy import lambdify
        from sympy.printing.numpy import NumPyPrinter

        if hasattr(expression, '_sympy_expr'):
            expr = expression._sympy_expr
        else:
            expr = expression

        printer = NumPyPrinter()
        expr_code = printer.doprint(expr)

        # Generate function code
        arg_str = ", ".join(variables)
        func_code = f"""
def generated_func({arg_str}):
    return {expr_code}
"""
        source = self._imports + func_code

        return GeneratedCode(
            source=source,
            language="python",
            metadata={"backend": "numpy", "variables": variables}
        )

    def compile(self, code: GeneratedCode) -> Callable:
        """Compile to Python function."""
        namespace = {"np": np}
        exec(code.source, namespace)
        code.function = namespace['generated_func']
        return code.function

    def generate_loop(self, body_expr, loop_vars: List[str],
                      array_shape: Tuple[int, ...]) -> GeneratedCode:
        """
        Generate explicit loop code (for debugging or non-vectorizable ops).
        """
        dim = len(loop_vars)

        if dim == 1:
            loop_code = f"""
def loop_func(u):
    result = np.zeros_like(u)
    for i in range(1, u.shape[0] - 1):
        result[i] = {body_expr}
    return result
"""
        elif dim == 2:
            loop_code = f"""
def loop_func(u):
    result = np.zeros_like(u)
    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            result[i, j] = {body_expr}
    return result
"""
        else:
            raise NotImplementedError(f"{dim}D loops")

        source = self._imports + loop_code

        return GeneratedCode(
            source=source,
            language="python",
            metadata={"backend": "numpy", "loop_dims": dim}
        )


class NumbaBackend(CodeGenerator):
    """
    Numba JIT compilation backend.

    Generates Numba-compatible code for maximum performance.
    """

    def __init__(self, parallel: bool = True, fastmath: bool = True):
        self.parallel = parallel
        self.fastmath = fastmath
        self._imports = """
import numpy as np
import numba
from numba import jit, prange
"""

    def generate(self, expression, variables: List[str]) -> GeneratedCode:
        """Generate Numba-compatible code."""
        if hasattr(expression, '_sympy_expr'):
            from sympy.printing.numpy import NumPyPrinter
            printer = NumPyPrinter()
            expr_code = printer.doprint(expression._sympy_expr)
        else:
            expr_code = str(expression)

        arg_str = ", ".join(variables)

        # Numba decorator options
        jit_opts = []
        if self.parallel:
            jit_opts.append("parallel=True")
        if self.fastmath:
            jit_opts.append("fastmath=True")
        jit_str = f"@jit(nopython=True, {', '.join(jit_opts)})" if jit_opts else "@jit(nopython=True)"

        func_code = f"""
{jit_str}
def generated_func({arg_str}):
    return {expr_code}
"""
        source = self._imports + func_code

        return GeneratedCode(
            source=source,
            language="python+numba",
            metadata={"backend": "numba", "parallel": self.parallel}
        )

    def compile(self, code: GeneratedCode) -> Callable:
        """Compile with Numba JIT."""
        try:
            import numba
            from numba import jit, prange

            namespace = {"np": np, "numba": numba, "jit": jit, "prange": prange}
            exec(code.source, namespace)
            code.function = namespace['generated_func']
            return code.function
        except ImportError:
            # Fallback to NumPy
            numpy_backend = NumpyBackend()
            numpy_code = numpy_backend.generate(code.metadata.get('expression'), code.metadata.get('variables', []))
            return numpy_backend.compile(numpy_code)

    def generate_stencil(self, stencil_weights: np.ndarray,
                         dimensions: int) -> GeneratedCode:
        """
        Generate optimized stencil application code.

        Uses Numba's stencil decorator for cache-friendly access patterns.
        """
        if dimensions == 1:
            code = f"""
@numba.stencil
def stencil_1d(u):
    return {' + '.join(f'{w}*u[{i-len(stencil_weights)//2}]' for i, w in enumerate(stencil_weights))}

@jit(nopython=True, parallel=True)
def apply_stencil(u):
    return stencil_1d(u)
"""
        elif dimensions == 2:
            # 5-point Laplacian stencil
            code = """
@numba.stencil
def laplacian_2d(u):
    return u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4*u[0, 0]

@jit(nopython=True, parallel=True)
def apply_stencil(u):
    return laplacian_2d(u)
"""
        else:
            raise NotImplementedError(f"{dimensions}D stencils")

        source = self._imports + code

        return GeneratedCode(
            source=source,
            language="python+numba",
            metadata={"backend": "numba", "stencil_dims": dimensions}
        )

    def generate_parallel_loop(self, body_expr: str,
                               loop_vars: List[str],
                               array_name: str = "u") -> GeneratedCode:
        """
        Generate parallel loop with Numba prange.
        """
        dim = len(loop_vars)

        if dim == 2:
            code = f"""
@jit(nopython=True, parallel=True, fastmath=True)
def parallel_loop({array_name}):
    result = np.zeros_like({array_name})
    for i in prange(1, {array_name}.shape[0] - 1):
        for j in range(1, {array_name}.shape[1] - 1):
            result[i, j] = {body_expr}
    return result
"""
        else:
            raise NotImplementedError(f"{dim}D parallel loops")

        source = self._imports + code

        return GeneratedCode(
            source=source,
            language="python+numba",
            metadata={"backend": "numba", "parallel": True}
        )


class CUDABackend(CodeGenerator):
    """
    CUDA code generation for GPU acceleration.

    Generates CUDA kernels for massively parallel execution.
    """

    def __init__(self, block_size: Tuple[int, int] = (16, 16)):
        self.block_size = block_size

    def generate(self, expression, variables: List[str]) -> GeneratedCode:
        """Generate CUDA kernel code."""
        # Simplified CUDA kernel generation
        kernel_code = f"""
__global__ void compute_kernel(float* u, float* result, int nx, int ny) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {{
        int idx = i * ny + j;
        // Laplacian
        result[idx] = u[idx - ny] + u[idx + ny] + u[idx - 1] + u[idx + 1] - 4.0f * u[idx];
    }}
}}
"""
        return GeneratedCode(
            source=kernel_code,
            language="cuda",
            metadata={"backend": "cuda", "block_size": self.block_size}
        )

    def compile(self, code: GeneratedCode) -> Callable:
        """Compile CUDA kernel (requires CuPy or PyCUDA)."""
        try:
            import cupy as cp

            kernel = cp.RawKernel(code.source, 'compute_kernel')

            def cuda_func(u: np.ndarray) -> np.ndarray:
                u_gpu = cp.asarray(u, dtype=cp.float32)
                result_gpu = cp.zeros_like(u_gpu)
                nx, ny = u.shape

                block = self.block_size
                grid = ((nx + block[0] - 1) // block[0],
                        (ny + block[1] - 1) // block[1])

                kernel(grid, block, (u_gpu, result_gpu, nx, ny))
                return cp.asnumpy(result_gpu)

            code.function = cuda_func
            return cuda_func

        except ImportError:
            raise RuntimeError("CUDA backend requires CuPy")
