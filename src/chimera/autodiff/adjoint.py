"""
Adjoint-based sensitivity analysis.

Efficient gradient computation for PDE-constrained optimization.
"""

from __future__ import annotations
from typing import Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    gradient: np.ndarray
    adjoint_solution: np.ndarray
    objective_value: float
    iterations: int = 1


class AdjointSolver:
    """
    Adjoint method for PDE-constrained optimization.

    For problem:
        min J(u, p)
        s.t. R(u, p) = 0

    Computes dJ/dp efficiently using adjoint equation:
        (dR/du)^T * lambda = -dJ/du
        dJ/dp = dJ_partial/dp + lambda^T * dR/dp

    This avoids solving N forward problems for N parameters.
    """

    def __init__(self):
        self._forward_residual: Optional[Callable] = None
        self._objective: Optional[Callable] = None
        self._last_adjoint: Optional[np.ndarray] = None

    def set_problem(self,
                    residual: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    objective: Callable[[np.ndarray, np.ndarray], float],
                    jacobian_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    jacobian_p: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    obj_grad_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    obj_grad_p: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Set up the optimization problem.

        Args:
            residual: R(u, p) - forward residual
            objective: J(u, p) - objective function
            jacobian_u: dR/du - Jacobian w.r.t. state
            jacobian_p: dR/dp - Jacobian w.r.t. parameters
            obj_grad_u: dJ/du - objective gradient w.r.t. state
            obj_grad_p: dJ/dp - objective gradient w.r.t. parameters
        """
        self._residual = residual
        self._objective = objective
        self._jacobian_u = jacobian_u
        self._jacobian_p = jacobian_p
        self._obj_grad_u = obj_grad_u
        self._obj_grad_p = obj_grad_p

    def compute_sensitivity(self, u: np.ndarray, p: np.ndarray) -> SensitivityResult:
        """
        Compute sensitivity dJ/dp using adjoint method.

        Args:
            u: State solution (from forward solve)
            p: Parameters

        Returns:
            SensitivityResult with gradient and adjoint solution
        """
        # Compute objective
        J = self._objective(u, p)

        # Compute Jacobians
        dR_du = self._jacobian_u(u, p)
        dR_dp = self._jacobian_p(u, p)
        dJ_du = self._obj_grad_u(u, p)
        dJ_dp = self._obj_grad_p(u, p)

        # Solve adjoint equation: (dR/du)^T * lambda = -dJ/du
        if isinstance(dR_du, csr_matrix):
            adjoint = spsolve(dR_du.T, -dJ_du)
        else:
            adjoint = np.linalg.solve(dR_du.T, -dJ_du)

        self._last_adjoint = adjoint

        # Compute total gradient
        # dJ/dp = dJ_partial/dp + lambda^T * dR/dp
        if isinstance(dR_dp, csr_matrix):
            gradient = dJ_dp + dR_dp.T @ adjoint
        else:
            gradient = dJ_dp + dR_dp.T @ adjoint

        return SensitivityResult(
            gradient=gradient,
            adjoint_solution=adjoint,
            objective_value=J
        )


def compute_sensitivity(
    forward_solve: Callable[[np.ndarray], np.ndarray],
    objective: Callable[[np.ndarray, np.ndarray], float],
    parameters: np.ndarray,
    method: str = "adjoint"
) -> np.ndarray:
    """
    Convenience function for sensitivity computation.

    Args:
        forward_solve: Function p -> u (solves forward problem)
        objective: Function (u, p) -> J (objective)
        parameters: Current parameter values
        method: 'adjoint' or 'finite_difference'

    Returns:
        Gradient dJ/dp
    """
    if method == "finite_difference":
        return _finite_difference_gradient(forward_solve, objective, parameters)
    elif method == "adjoint":
        raise NotImplementedError("Use AdjointSolver class for full adjoint")
    else:
        raise ValueError(f"Unknown method: {method}")


def _finite_difference_gradient(
    forward_solve: Callable,
    objective: Callable,
    p: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Finite difference gradient (for validation)."""
    u0 = forward_solve(p)
    J0 = objective(u0, p)

    grad = np.zeros_like(p)
    for i in range(len(p)):
        p_pert = p.copy()
        p_pert[i] += eps
        u_pert = forward_solve(p_pert)
        J_pert = objective(u_pert, p_pert)
        grad[i] = (J_pert - J0) / eps

    return grad


class ShapeOptimization:
    """
    Shape optimization using mesh deformation and adjoint sensitivities.

    Optimizes node positions to minimize objective.
    """

    def __init__(self, mesh, solver):
        self.mesh = mesh
        self.solver = solver
        self._design_nodes: Optional[np.ndarray] = None

    def set_design_nodes(self, node_indices: np.ndarray):
        """Set which nodes can move."""
        self._design_nodes = node_indices

    def compute_shape_gradient(self, u: np.ndarray,
                               objective: Callable) -> np.ndarray:
        """
        Compute shape gradient dJ/dX.

        Uses material derivative approach:
        dJ/dX = integral_boundary (stress * displacement_sensitivity) dS
        """
        n_design = len(self._design_nodes)
        n_dim = self.mesh.dim

        gradient = np.zeros((n_design, n_dim))

        # Finite difference on mesh positions
        eps = 1e-6
        J0 = objective(u, self.mesh.points)

        for i, node in enumerate(self._design_nodes):
            for d in range(n_dim):
                # Perturb mesh
                self.mesh.points[node, d] += eps

                # Re-solve
                result = self.solver.solve()
                u_pert = list(result.fields.values())[0].values

                # Evaluate objective
                J_pert = objective(u_pert, self.mesh.points)

                gradient[i, d] = (J_pert - J0) / eps

                # Restore mesh
                self.mesh.points[node, d] -= eps

        return gradient
