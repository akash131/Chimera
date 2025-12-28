"""
Base solver interface for Chimera.

All discretization methods (FEM, FVM, meshless) implement this interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import numpy as np
import time

from chimera.core.domain import Domain
from chimera.core.field import Field
from chimera.core.equation import Equation, PDESystem
from chimera.core.boundary import BoundaryCondition, BoundaryConditionSet
from chimera.mesh.mesh import Mesh


class SolverStatus(Enum):
    """Status of solver execution."""
    SUCCESS = "success"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    FAILED = "failed"


@dataclass
class SolverResult:
    """
    Result of a solver execution.

    Contains solution fields, convergence info, and diagnostics.
    """
    status: SolverStatus
    fields: Dict[str, Field]
    residual: float = 0.0
    iterations: int = 0
    solve_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in [SolverStatus.SUCCESS, SolverStatus.CONVERGED]

    def get_field(self, name: str) -> Field:
        """Get solution field by name."""
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not in solution")
        return self.fields[name]

    def max_value(self, field_name: str) -> float:
        """Get maximum value of a field."""
        return self.fields[field_name].max()

    def min_value(self, field_name: str) -> float:
        """Get minimum value of a field."""
        return self.fields[field_name].min()

    def __repr__(self) -> str:
        return (f"SolverResult(status={self.status.value}, "
                f"residual={self.residual:.2e}, "
                f"iterations={self.iterations}, "
                f"time={self.solve_time:.3f}s)")


@dataclass
class SolverConfig:
    """Configuration for solvers."""
    # Linear solver
    linear_solver: str = "direct"  # 'direct', 'cg', 'gmres', 'bicgstab'
    preconditioner: str = "ilu"  # 'none', 'jacobi', 'ilu', 'amg'

    # Nonlinear solver
    nonlinear_solver: str = "newton"  # 'newton', 'picard'
    max_iterations: int = 100
    tolerance: float = 1e-8
    relaxation: float = 1.0

    # Time stepping
    time_scheme: str = "backward_euler"  # 'backward_euler', 'crank_nicolson', 'bdf2'
    dt: float = 0.01
    t_final: float = 1.0

    # Output
    verbose: bool = False
    store_history: bool = False


class Solver(ABC):
    """
    Abstract base class for all solvers.

    Defines the interface that FEM, FVM, and meshless solvers implement.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.mesh: Optional[Mesh] = None
        self.equation: Optional[Equation] = None
        self.bcs: BoundaryConditionSet = BoundaryConditionSet()
        self._assembled = False
        self._history: List[Dict] = []

    @abstractmethod
    def set_mesh(self, mesh: Mesh):
        """Set the computational mesh."""
        pass

    @abstractmethod
    def set_equation(self, equation: Equation):
        """Set the governing equation."""
        pass

    def add_bc(self, bc: BoundaryCondition):
        """Add a boundary condition."""
        self.bcs.add(bc)
        self._assembled = False

    def set_bcs(self, bcs: BoundaryConditionSet):
        """Set all boundary conditions."""
        self.bcs = bcs
        self._assembled = False

    @abstractmethod
    def assemble(self):
        """Assemble the discrete system."""
        pass

    @abstractmethod
    def solve(self) -> SolverResult:
        """Solve the assembled system."""
        pass

    def solve_transient(self, initial: Field,
                        callback: Optional[Callable[[float, Field], None]] = None
                        ) -> List[SolverResult]:
        """
        Solve transient problem.

        Args:
            initial: Initial condition field
            callback: Called after each time step with (t, solution)

        Returns:
            List of results at each time step
        """
        if not self.equation.is_time_dependent():
            raise ValueError("Equation is not time-dependent")

        results = []
        t = 0.0
        solution = initial.copy()

        while t < self.config.t_final:
            t += self.config.dt

            # Solve for this time step
            result = self._step_transient(solution, self.config.dt)
            results.append(result)

            if not result.success:
                break

            solution = result.get_field(initial.name)

            if callback:
                callback(t, solution)

            if self.config.verbose:
                print(f"t = {t:.4f}, residual = {result.residual:.2e}")

        return results

    def _step_transient(self, previous: Field, dt: float) -> SolverResult:
        """Single time step (to be overridden by specific solvers)."""
        raise NotImplementedError("Transient solving not implemented")

    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        """Get the assembled system matrix."""
        pass

    @abstractmethod
    def get_rhs(self) -> np.ndarray:
        """Get the assembled right-hand side."""
        pass

    def estimate_error(self, solution: Field) -> np.ndarray:
        """
        Estimate local error for adaptivity.

        Returns per-cell error estimates.
        """
        # Default: gradient recovery error estimator
        if self.mesh is None:
            raise ValueError("No mesh set")

        grad = solution.gradient()
        # Compute jumps across elements
        # This is a simplified estimator
        return np.ones(self.mesh.n_cells)

    def adapt_mesh(self, solution: Field, threshold: float = 0.1) -> Mesh:
        """
        Adapt mesh based on error estimates.

        Args:
            solution: Current solution field
            threshold: Refinement threshold

        Returns:
            Refined mesh
        """
        errors = self.estimate_error(solution)
        max_error = errors.max()

        # Mark cells for refinement
        refine_mask = errors > threshold * max_error

        if not np.any(refine_mask):
            return self.mesh

        # Refine marked cells (simplified - uniform refinement)
        return self.mesh.refine()

    def _solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b."""
        from scipy.sparse import issparse
        from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab
        from scipy.linalg import solve

        if self.config.linear_solver == "direct":
            if issparse(A):
                return spsolve(A, b)
            else:
                return solve(A, b)

        elif self.config.linear_solver == "cg":
            x, info = cg(A, b, tol=self.config.tolerance)
            return x

        elif self.config.linear_solver == "gmres":
            x, info = gmres(A, b, tol=self.config.tolerance)
            return x

        elif self.config.linear_solver == "bicgstab":
            x, info = bicgstab(A, b, tol=self.config.tolerance)
            return x

        else:
            raise ValueError(f"Unknown linear solver: {self.config.linear_solver}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mesh={self.mesh}, equation={self.equation})"
