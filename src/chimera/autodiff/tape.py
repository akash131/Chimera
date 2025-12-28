"""
Reverse-mode automatic differentiation tape.

Custom implementation that works with sparse matrices and linear solvers.
This is the foundation for differentiating through the entire solver stack.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Union, Dict, Any
from enum import Enum
import numpy as np
from scipy.sparse import issparse, csr_matrix


class OpType(Enum):
    """Operation types for the tape."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MATMUL = "matmul"
    DOT = "dot"
    SUM = "sum"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    EXP = "exp"
    LOG = "log"
    SIN = "sin"
    COS = "cos"
    POWER = "power"
    SQRT = "sqrt"
    ABS = "abs"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    LINEAR_SOLVE = "linear_solve"  # Key: differentiate through Ax = b
    EIGENSOLVE = "eigensolve"
    CUSTOM = "custom"


@dataclass
class TapeEntry:
    """Single entry in the computation tape."""
    op: OpType
    inputs: List[int]  # Indices of input variables
    output: int  # Index of output variable
    data: Dict[str, Any] = field(default_factory=dict)  # Op-specific data


class Tape:
    """
    Computation tape for reverse-mode autodiff.

    Records operations and enables gradient computation via backpropagation.
    Key innovation: handles sparse matrices and linear system solves.
    """

    _active_tape: Optional['Tape'] = None

    def __init__(self):
        self.entries: List[TapeEntry] = []
        self.values: List[np.ndarray] = []
        self.gradients: List[Optional[np.ndarray]] = []
        self.requires_grad: List[bool] = []
        self._var_counter = 0

    def __enter__(self) -> 'Tape':
        Tape._active_tape = self
        return self

    def __exit__(self, *args):
        Tape._active_tape = None

    @classmethod
    def get_active(cls) -> Optional['Tape']:
        return cls._active_tape

    def new_variable(self, value: np.ndarray, requires_grad: bool = True) -> int:
        """Register a new variable."""
        idx = self._var_counter
        self._var_counter += 1
        self.values.append(np.asarray(value))
        self.gradients.append(None)
        self.requires_grad.append(requires_grad)
        return idx

    def record(self, op: OpType, inputs: List[int], output: int,
               data: Optional[Dict] = None):
        """Record an operation."""
        self.entries.append(TapeEntry(
            op=op,
            inputs=inputs,
            output=output,
            data=data or {}
        ))

    def backward(self, output_idx: int, seed: Optional[np.ndarray] = None):
        """
        Compute gradients via reverse-mode autodiff.

        Args:
            output_idx: Index of output variable to differentiate
            seed: Initial gradient (default: ones)
        """
        if seed is None:
            seed = np.ones_like(self.values[output_idx])

        self.gradients[output_idx] = seed

        # Reverse pass
        for entry in reversed(self.entries):
            if self.gradients[entry.output] is None:
                continue

            grad_out = self.gradients[entry.output]
            grads_in = self._backward_op(entry, grad_out)

            for inp_idx, grad_in in zip(entry.inputs, grads_in):
                if grad_in is None:
                    continue
                if self.gradients[inp_idx] is None:
                    self.gradients[inp_idx] = grad_in
                else:
                    self.gradients[inp_idx] = self.gradients[inp_idx] + grad_in

    def _backward_op(self, entry: TapeEntry,
                     grad_out: np.ndarray) -> List[Optional[np.ndarray]]:
        """Compute gradients for a single operation."""
        inputs = [self.values[i] for i in entry.inputs]

        if entry.op == OpType.ADD:
            return [grad_out, grad_out]

        elif entry.op == OpType.SUB:
            return [grad_out, -grad_out]

        elif entry.op == OpType.MUL:
            a, b = inputs
            return [grad_out * b, grad_out * a]

        elif entry.op == OpType.DIV:
            a, b = inputs
            return [grad_out / b, -grad_out * a / (b ** 2)]

        elif entry.op == OpType.MATMUL:
            A, x = inputs
            # d(Ax)/dA = grad @ x^T, d(Ax)/dx = A^T @ grad
            if issparse(A):
                return [None, A.T @ grad_out]  # Skip A gradient for efficiency
            else:
                return [np.outer(grad_out, x), A.T @ grad_out]

        elif entry.op == OpType.DOT:
            a, b = inputs
            return [grad_out * b, grad_out * a]

        elif entry.op == OpType.SUM:
            return [np.full_like(inputs[0], grad_out)]

        elif entry.op == OpType.EXP:
            return [grad_out * np.exp(inputs[0])]

        elif entry.op == OpType.LOG:
            return [grad_out / inputs[0]]

        elif entry.op == OpType.SIN:
            return [grad_out * np.cos(inputs[0])]

        elif entry.op == OpType.COS:
            return [grad_out * (-np.sin(inputs[0]))]

        elif entry.op == OpType.POWER:
            base, exp = inputs
            return [
                grad_out * exp * (base ** (exp - 1)),
                grad_out * (base ** exp) * np.log(base + 1e-12)
            ]

        elif entry.op == OpType.SQRT:
            return [grad_out * 0.5 / np.sqrt(inputs[0] + 1e-12)]

        elif entry.op == OpType.LINEAR_SOLVE:
            # Key innovation: gradient through linear solve
            # If Ax = b, then:
            # dx/dA = -A^{-T} @ (grad @ x^T)
            # dx/db = A^{-T} @ grad
            A = entry.data['matrix']
            x = self.values[entry.output]

            if issparse(A):
                from scipy.sparse.linalg import spsolve
                # Solve adjoint system A^T @ lambda = grad
                adjoint = spsolve(A.T.tocsr(), grad_out)
            else:
                adjoint = np.linalg.solve(A.T, grad_out)

            # Gradient w.r.t. RHS
            grad_b = adjoint

            # Gradient w.r.t. matrix (optional, expensive)
            # grad_A = -np.outer(adjoint, x)

            return [None, grad_b]  # Skip matrix gradient for now

        elif entry.op == OpType.CUSTOM:
            # User-provided backward function
            backward_fn = entry.data.get('backward')
            if backward_fn:
                return backward_fn(grad_out, inputs, entry.data)
            return [None] * len(inputs)

        return [None] * len(inputs)


class Variable:
    """
    Differentiable variable wrapper.

    Overloads arithmetic operations to record on tape.
    """

    def __init__(self, value: np.ndarray, tape: Optional[Tape] = None,
                 requires_grad: bool = True):
        self.tape = tape or Tape.get_active()
        if self.tape is None:
            raise RuntimeError("No active tape. Use 'with Tape() as tape:'")

        self._value = np.asarray(value)
        self._idx = self.tape.new_variable(self._value, requires_grad)
        self.requires_grad = requires_grad

    @property
    def value(self) -> np.ndarray:
        return self.tape.values[self._idx]

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self.tape.gradients[self._idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._value.shape

    def backward(self, seed: Optional[np.ndarray] = None):
        """Compute gradients."""
        self.tape.backward(self._idx, seed)

    def _make_var(self, value: np.ndarray, op: OpType,
                  inputs: List['Variable'], data: Optional[Dict] = None) -> 'Variable':
        """Create new variable from operation."""
        result = Variable.__new__(Variable)
        result.tape = self.tape
        result._value = value
        result._idx = self.tape.new_variable(value, True)
        result.requires_grad = True

        self.tape.record(op, [v._idx for v in inputs], result._idx, data)
        return result

    def __add__(self, other: Union['Variable', np.ndarray, float]) -> 'Variable':
        if isinstance(other, Variable):
            return self._make_var(self.value + other.value, OpType.ADD, [self, other])
        else:
            other_var = Variable(np.asarray(other), self.tape, False)
            return self._make_var(self.value + other, OpType.ADD, [self, other_var])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union['Variable', np.ndarray, float]) -> 'Variable':
        if isinstance(other, Variable):
            return self._make_var(self.value - other.value, OpType.SUB, [self, other])
        else:
            other_var = Variable(np.asarray(other), self.tape, False)
            return self._make_var(self.value - other, OpType.SUB, [self, other_var])

    def __rsub__(self, other):
        other_var = Variable(np.asarray(other), self.tape, False)
        return other_var.__sub__(self)

    def __mul__(self, other: Union['Variable', np.ndarray, float]) -> 'Variable':
        if isinstance(other, Variable):
            return self._make_var(self.value * other.value, OpType.MUL, [self, other])
        else:
            other_var = Variable(np.asarray(other), self.tape, False)
            return self._make_var(self.value * other, OpType.MUL, [self, other_var])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union['Variable', np.ndarray, float]) -> 'Variable':
        if isinstance(other, Variable):
            return self._make_var(self.value / other.value, OpType.DIV, [self, other])
        else:
            other_var = Variable(np.asarray(other), self.tape, False)
            return self._make_var(self.value / other, OpType.DIV, [self, other_var])

    def __pow__(self, other: Union['Variable', float]) -> 'Variable':
        if isinstance(other, Variable):
            return self._make_var(self.value ** other.value, OpType.POWER, [self, other])
        else:
            other_var = Variable(np.asarray(other), self.tape, False)
            return self._make_var(self.value ** other, OpType.POWER, [self, other_var])

    def __neg__(self) -> 'Variable':
        return self * (-1)

    def sum(self) -> 'Variable':
        result = float(np.sum(self.value))
        return self._make_var(np.array(result), OpType.SUM, [self])

    def dot(self, other: 'Variable') -> 'Variable':
        result = np.dot(self.value, other.value)
        return self._make_var(np.array(result), OpType.DOT, [self, other])

    def sqrt(self) -> 'Variable':
        return self._make_var(np.sqrt(self.value), OpType.SQRT, [self])

    def exp(self) -> 'Variable':
        return self._make_var(np.exp(self.value), OpType.EXP, [self])

    def log(self) -> 'Variable':
        return self._make_var(np.log(self.value), OpType.LOG, [self])

    def __repr__(self) -> str:
        return f"Variable(shape={self.shape}, grad={'computed' if self.grad is not None else 'None'})"


def gradient(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0):
    """
    Decorator to compute gradients of a function.

    Example:
        @gradient
        def loss(x, y):
            return ((x - y) ** 2).sum()

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.1, 2.1, 3.1])
        grad_x = loss(x, y)
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def wrapper(*args):
        with Tape() as tape:
            # Wrap specified arguments as Variables
            var_args = list(args)
            variables = []
            for i in argnums:
                var_args[i] = Variable(args[i], tape)
                variables.append(var_args[i])

            # Forward pass
            result = func(*var_args)

            # Backward pass
            if isinstance(result, Variable):
                result.backward()

            # Return gradients
            if len(variables) == 1:
                return variables[0].grad
            return tuple(v.grad for v in variables)

    return wrapper


def linear_solve_differentiable(A: np.ndarray, b: 'Variable',
                                tape: Optional[Tape] = None) -> 'Variable':
    """
    Solve Ax = b with gradient support.

    The gradient through a linear solve is computed via the adjoint equation.
    """
    tape = tape or Tape.get_active()
    if tape is None:
        raise RuntimeError("No active tape")

    # Forward solve
    if issparse(A):
        from scipy.sparse.linalg import spsolve
        x = spsolve(A, b.value)
    else:
        x = np.linalg.solve(A, b.value)

    # Create result variable
    result = Variable.__new__(Variable)
    result.tape = tape
    result._value = x
    result._idx = tape.new_variable(x, True)
    result.requires_grad = True

    # Record operation with matrix stored for backward pass
    tape.record(
        OpType.LINEAR_SOLVE,
        [b._idx],
        result._idx,
        {'matrix': A}
    )

    return result
