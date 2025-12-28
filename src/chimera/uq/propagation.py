"""
Uncertainty propagation methods.

Propagate input uncertainty through computational models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class UQResult:
    """Result of uncertainty quantification."""
    mean: np.ndarray
    std: np.ndarray
    samples: Optional[np.ndarray] = None
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    pdf: Optional[Callable] = None


class UQMethod(ABC):
    """Abstract base for UQ methods."""

    @abstractmethod
    def propagate(self, model: Callable, input_dist) -> UQResult:
        """Propagate uncertainty through model."""
        pass


class MonteCarlo(UQMethod):
    """
    Monte Carlo uncertainty propagation.

    Simple but robust: sample inputs, run model, compute statistics.
    """

    def __init__(self, n_samples: int = 1000, seed: Optional[int] = None):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def propagate(self, model: Callable,
                  input_dist: List[stats.rv_continuous]) -> UQResult:
        """
        Propagate uncertainty via Monte Carlo sampling.

        Args:
            model: Function (parameters) -> output
            input_dist: List of scipy distributions for each input

        Returns:
            UQResult with statistics
        """
        # Sample inputs
        n_inputs = len(input_dist)
        samples = np.zeros((self.n_samples, n_inputs))

        for i, dist in enumerate(input_dist):
            samples[:, i] = dist.rvs(self.n_samples, random_state=self.rng)

        # Evaluate model
        outputs = []
        for sample in samples:
            out = model(sample)
            outputs.append(out)

        outputs = np.array(outputs)

        # Compute statistics
        mean = np.mean(outputs, axis=0)
        std = np.std(outputs, axis=0)

        # Confidence interval (95%)
        ci_low = np.percentile(outputs, 2.5, axis=0)
        ci_high = np.percentile(outputs, 97.5, axis=0)

        return UQResult(
            mean=mean,
            std=std,
            samples=outputs,
            confidence_interval=(ci_low, ci_high)
        )

    def convergence_check(self, outputs: np.ndarray) -> dict:
        """Check Monte Carlo convergence."""
        n = len(outputs)

        # Running mean and std
        running_mean = np.cumsum(outputs, axis=0) / np.arange(1, n + 1)[:, np.newaxis]
        running_var = np.zeros_like(running_mean)

        for i in range(1, n):
            running_var[i] = np.var(outputs[:i + 1], axis=0)

        # Standard error
        se = np.sqrt(running_var[-1] / n)

        return {
            "n_samples": n,
            "final_mean": running_mean[-1],
            "final_std": np.sqrt(running_var[-1]),
            "standard_error": se,
            "relative_error": se / (np.abs(running_mean[-1]) + 1e-10)
        }


class PolynomialChaos(UQMethod):
    """
    Polynomial Chaos Expansion (PCE).

    Represent output as polynomial of input uncertainties.
    More efficient than Monte Carlo for smooth functions.
    """

    def __init__(self, order: int = 3, n_quad: int = 10):
        self.order = order
        self.n_quad = n_quad
        self._coefficients: Optional[np.ndarray] = None

    def propagate(self, model: Callable,
                  input_dist: List[stats.rv_continuous]) -> UQResult:
        """
        Build PCE and compute statistics.

        Uses non-intrusive spectral projection.
        """
        n_inputs = len(input_dist)

        # Get quadrature points and weights
        quad_points, quad_weights = self._get_quadrature(input_dist)

        # Evaluate model at quadrature points
        n_quad_total = len(quad_points)
        outputs = []
        for point in quad_points:
            outputs.append(model(point))
        outputs = np.array(outputs)

        # Build polynomial basis
        basis_values = self._evaluate_basis(quad_points, input_dist)

        # Compute PCE coefficients via projection
        # c_k = E[f * P_k] / E[P_k^2]
        n_terms = basis_values.shape[1]
        output_dim = outputs.shape[1] if len(outputs.shape) > 1 else 1
        outputs = outputs.reshape(n_quad_total, output_dim)

        coefficients = np.zeros((n_terms, output_dim))
        for k in range(n_terms):
            numerator = np.sum(quad_weights[:, np.newaxis] * outputs * basis_values[:, k:k+1], axis=0)
            denominator = np.sum(quad_weights * basis_values[:, k] ** 2)
            coefficients[k] = numerator / (denominator + 1e-10)

        self._coefficients = coefficients

        # Compute statistics from PCE
        # Mean = c_0 (coefficient of constant term)
        mean = coefficients[0]

        # Variance = sum(c_k^2) for k > 0
        variance = np.sum(coefficients[1:] ** 2, axis=0)
        std = np.sqrt(variance)

        return UQResult(mean=mean, std=std)

    def _get_quadrature(self, input_dist) -> Tuple[np.ndarray, np.ndarray]:
        """Get Gaussian quadrature points and weights."""
        n_inputs = len(input_dist)

        # 1D quadrature for each input
        quad_1d = []
        for dist in input_dist:
            # Use Gauss-Hermite for normal, Gauss-Legendre for uniform
            if isinstance(dist, stats.rv_continuous):
                points, weights = np.polynomial.legendre.leggauss(self.n_quad)
                # Scale to distribution support
                loc, scale = dist.mean(), dist.std()
                points = points * scale * np.sqrt(2) + loc
                weights = weights / np.sqrt(np.pi)
            else:
                points, weights = np.polynomial.legendre.leggauss(self.n_quad)

            quad_1d.append((points, weights))

        # Tensor product
        if n_inputs == 1:
            return quad_1d[0][0].reshape(-1, 1), quad_1d[0][1]

        from itertools import product
        indices = list(product(*[range(self.n_quad) for _ in range(n_inputs)]))

        quad_points = np.array([[quad_1d[d][0][i[d]] for d in range(n_inputs)]
                                for i in indices])
        quad_weights = np.array([np.prod([quad_1d[d][1][i[d]] for d in range(n_inputs)])
                                 for i in indices])

        return quad_points, quad_weights

    def _evaluate_basis(self, points: np.ndarray,
                        input_dist: List) -> np.ndarray:
        """Evaluate polynomial basis at given points."""
        n_points = len(points)
        n_inputs = points.shape[1]

        # Generate multi-indices up to total degree
        from itertools import product
        indices = [idx for idx in product(range(self.order + 1), repeat=n_inputs)
                   if sum(idx) <= self.order]
        n_terms = len(indices)

        basis = np.ones((n_points, n_terms))

        for k, idx in enumerate(indices):
            for d in range(n_inputs):
                if idx[d] > 0:
                    # Legendre polynomial (for uniform)
                    leg = np.polynomial.legendre.Legendre.basis(idx[d])
                    # Scale points to [-1, 1]
                    dist = input_dist[d]
                    x_scaled = (points[:, d] - dist.mean()) / (dist.std() * np.sqrt(2))
                    basis[:, k] *= leg(x_scaled)

        return basis


class StochasticCollocation(UQMethod):
    """
    Stochastic Collocation method.

    Interpolate output in parameter space, then integrate for statistics.
    """

    def __init__(self, n_collocation: int = 5):
        self.n_collocation = n_collocation

    def propagate(self, model: Callable,
                  input_dist: List[stats.rv_continuous]) -> UQResult:
        """
        Stochastic collocation with tensor product quadrature.
        """
        n_inputs = len(input_dist)

        # Collocation points (Clenshaw-Curtis or Gauss-Legendre)
        colloc_points = self._get_collocation_points(input_dist)
        n_points = len(colloc_points)

        # Evaluate model
        outputs = np.array([model(p) for p in colloc_points])
        output_dim = outputs.shape[1] if len(outputs.shape) > 1 else 1
        outputs = outputs.reshape(n_points, output_dim)

        # Build interpolant and integrate for statistics
        # Simplified: use quadrature weights directly
        weights = np.ones(n_points) / n_points

        mean = np.sum(weights[:, np.newaxis] * outputs, axis=0)
        variance = np.sum(weights[:, np.newaxis] * (outputs - mean) ** 2, axis=0)
        std = np.sqrt(variance)

        return UQResult(mean=mean, std=std)

    def _get_collocation_points(self, input_dist) -> np.ndarray:
        """Generate collocation points."""
        n_inputs = len(input_dist)

        # Chebyshev nodes for each dimension
        nodes_1d = []
        for dist in input_dist:
            # Chebyshev nodes in [0, 1]
            k = np.arange(self.n_collocation)
            nodes = 0.5 * (1 - np.cos(np.pi * (2 * k + 1) / (2 * self.n_collocation)))
            # Scale to distribution
            nodes = dist.ppf(nodes)
            nodes_1d.append(nodes)

        # Tensor product
        from itertools import product
        if n_inputs == 1:
            return nodes_1d[0].reshape(-1, 1)

        return np.array(list(product(*nodes_1d)))
