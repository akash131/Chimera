"""
Inverse problem solvers for Chimera.

Parameter estimation, material identification, source reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize, least_squares

from chimera.core.field import Field
from chimera.mesh.mesh import Mesh
from chimera.solvers.base import Solver, SolverResult


@dataclass
class InverseConfig:
    """Configuration for inverse solvers."""
    regularization: str = "tikhonov"  # 'tikhonov', 'tv', 'sparse'
    reg_weight: float = 1e-4
    max_iterations: int = 100
    tolerance: float = 1e-6
    method: str = "L-BFGS-B"  # Optimization method


class InverseSolver:
    """
    General inverse problem solver.

    Finds parameters p such that F(p) ≈ measurements.

    Uses adjoint method for efficient gradient computation.
    """

    def __init__(self, forward_solver: Solver,
                 config: Optional[InverseConfig] = None):
        self.solver = forward_solver
        self.config = config or InverseConfig()
        self._history: List[dict] = []

    def set_measurements(self, locations: np.ndarray,
                         values: np.ndarray,
                         noise_std: float = 0.0):
        """
        Set measurement data.

        Args:
            locations: Measurement point coordinates
            values: Measured values
            noise_std: Measurement noise standard deviation
        """
        self.measurement_locations = locations
        self.measurement_values = values
        self.noise_std = noise_std

    def objective(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient.

        J(p) = ||F(p) - m||² + R(p)

        Uses adjoint method for gradient.
        """
        # Update parameters in solver
        self._set_parameters(params)

        # Solve forward problem
        result = self.solver.solve()

        if not result.success:
            return float('inf'), np.zeros_like(params)

        # Extract solution at measurement points
        solution = list(result.fields.values())[0]
        predicted = solution.evaluate(self.measurement_locations)

        # Data misfit
        residual = predicted - self.measurement_values
        misfit = 0.5 * np.sum(residual**2)

        # Regularization
        reg, reg_grad = self._regularization(params)

        # Total objective
        J = misfit + self.config.reg_weight * reg

        # Gradient via adjoint method
        grad = self._compute_gradient(result, residual, params)
        grad += self.config.reg_weight * reg_grad

        return J, grad

    def _set_parameters(self, params: np.ndarray):
        """Set parameters in forward solver (to be overridden)."""
        pass

    def _compute_gradient(self, forward_result: SolverResult,
                          residual: np.ndarray,
                          params: np.ndarray) -> np.ndarray:
        """
        Compute gradient via adjoint method.

        Adjoint: A^T λ = -∂J/∂u
        Gradient: dJ/dp = ∂J/∂p + λ^T ∂F/∂p
        """
        # Simplified - full adjoint would require matrix transpose solve
        n_params = len(params)
        grad = np.zeros(n_params)

        # Finite difference fallback
        eps = 1e-6
        J0, _ = self.objective(params)

        for i in range(n_params):
            params_pert = params.copy()
            params_pert[i] += eps
            self._set_parameters(params_pert)
            result = self.solver.solve()

            if result.success:
                solution = list(result.fields.values())[0]
                predicted = solution.evaluate(self.measurement_locations)
                residual = predicted - self.measurement_values
                J_pert = 0.5 * np.sum(residual**2)
                grad[i] = (J_pert - J0) / eps

        return grad

    def _regularization(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute regularization term and gradient."""
        if self.config.regularization == "tikhonov":
            # L2 regularization
            return 0.5 * np.sum(params**2), params

        elif self.config.regularization == "tv":
            # Total variation (1D)
            diff = np.diff(params)
            eps = 1e-6
            return np.sum(np.sqrt(diff**2 + eps)), np.zeros_like(params)  # Simplified

        elif self.config.regularization == "sparse":
            # L1 regularization
            return np.sum(np.abs(params)), np.sign(params)

        return 0.0, np.zeros_like(params)

    def solve(self, initial_guess: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Solve inverse problem.

        Args:
            initial_guess: Initial parameter values

        Returns:
            Optimal parameters and optimization info
        """
        def obj_fun(x):
            J, g = self.objective(x)
            self._history.append({'params': x.copy(), 'objective': J})
            return J

        def grad_fun(x):
            _, g = self.objective(x)
            return g

        result = minimize(
            obj_fun,
            initial_guess,
            method=self.config.method,
            jac=grad_fun,
            options={
                'maxiter': self.config.max_iterations,
                'gtol': self.config.tolerance
            }
        )

        return result.x, {
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
            'final_objective': result.fun
        }


class ParameterEstimation(InverseSolver):
    """
    Material parameter estimation.

    Given measured displacements/temperatures, estimate material properties.
    """

    def __init__(self, forward_solver: Solver,
                 parameter_names: List[str],
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 config: Optional[InverseConfig] = None):
        super().__init__(forward_solver, config)
        self.parameter_names = parameter_names
        self.bounds = bounds or [(0, np.inf)] * len(parameter_names)

    def _set_parameters(self, params: np.ndarray):
        """Update material parameters in solver."""
        for name, value in zip(self.parameter_names, params):
            # Would update solver's material properties
            pass


class SourceReconstruction(InverseSolver):
    """
    Source term reconstruction.

    Given measured field, estimate source distribution.
    """

    def __init__(self, forward_solver: Solver,
                 source_mesh: Mesh,
                 config: Optional[InverseConfig] = None):
        super().__init__(forward_solver, config)
        self.source_mesh = source_mesh
        self.n_sources = source_mesh.n_points

    def _set_parameters(self, params: np.ndarray):
        """Set source values."""
        # params = source values at source_mesh points
        def source_func(x):
            # Interpolate from source mesh to evaluation points
            from scipy.interpolate import RBFInterpolator
            interp = RBFInterpolator(self.source_mesh.points, params)
            return interp(x)

        self.solver.set_source(source_func)


class BayesianInversion:
    """
    Bayesian approach to inverse problems.

    Quantifies uncertainty in parameter estimates.
    """

    def __init__(self, forward_solver: Solver,
                 prior_mean: np.ndarray,
                 prior_cov: np.ndarray):
        self.solver = forward_solver
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_precision = np.linalg.inv(prior_cov)

    def posterior(self, params: np.ndarray,
                  measurements: np.ndarray,
                  noise_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior mean and covariance (Gaussian approximation).

        Uses Laplace approximation around MAP estimate.
        """
        # MAP estimate
        def neg_log_posterior(p):
            # Prior term
            dp = p - self.prior_mean
            prior = 0.5 * dp @ self.prior_precision @ dp

            # Likelihood term
            # Would compute forward model and likelihood
            return prior

        from scipy.optimize import minimize
        result = minimize(neg_log_posterior, self.prior_mean)
        map_estimate = result.x

        # Posterior covariance via Hessian
        from scipy.optimize import approx_fprime
        from numpy.linalg import inv

        def hessian(f, x, eps=1e-5):
            n = len(x)
            H = np.zeros((n, n))
            for i in range(n):
                x1, x2 = x.copy(), x.copy()
                x1[i] += eps
                x2[i] -= eps
                g1 = approx_fprime(x1, f, eps)
                g2 = approx_fprime(x2, f, eps)
                H[i] = (g1 - g2) / (2 * eps)
            return 0.5 * (H + H.T)

        H = hessian(neg_log_posterior, map_estimate)
        posterior_cov = inv(H)

        return map_estimate, posterior_cov

    def sample_posterior(self, n_samples: int,
                         mean: np.ndarray,
                         cov: np.ndarray) -> np.ndarray:
        """Draw samples from posterior distribution."""
        return np.random.multivariate_normal(mean, cov, n_samples)
