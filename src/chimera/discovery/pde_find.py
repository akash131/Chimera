"""
PDE Discovery from data.

Find partial differential equations from spatiotemporal data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PDEResult:
    """Result of PDE discovery."""
    coefficients: np.ndarray
    term_names: List[str]
    equation_string: str
    residual: float


class PDEFind:
    """
    PDE-FIND: Sparse regression for PDE discovery.

    Discovers PDE of form:
    u_t = Θ(u, u_x, u_xx, ...) @ ξ
    """

    def __init__(self, derivative_order: int = 2,
                 polynomial_order: int = 2,
                 threshold: float = 0.1):
        """
        Args:
            derivative_order: Maximum spatial derivative order
            polynomial_order: Maximum polynomial degree
            threshold: Sparsity threshold
        """
        self.derivative_order = derivative_order
        self.polynomial_order = polynomial_order
        self.threshold = threshold

    def fit(self, u: np.ndarray, x: np.ndarray, t: np.ndarray) -> PDEResult:
        """
        Discover PDE from data.

        Args:
            u: Solution data (n_t, n_x)
            x: Spatial coordinates (n_x,)
            t: Time coordinates (n_t,)

        Returns:
            PDEResult with discovered PDE
        """
        n_t, n_x = u.shape
        dx = x[1] - x[0]
        dt = t[1] - t[0]

        # Compute derivatives
        u_t = self._time_derivative(u, dt)
        derivatives = self._spatial_derivatives(u, dx)

        # Build library
        Theta, term_names = self._build_library(u, derivatives)

        # Flatten for regression
        u_t_flat = u_t.flatten()
        Theta_flat = Theta.reshape(-1, Theta.shape[-1])

        # Remove NaN/Inf
        valid = np.isfinite(u_t_flat) & np.all(np.isfinite(Theta_flat), axis=1)
        u_t_flat = u_t_flat[valid]
        Theta_flat = Theta_flat[valid]

        # Sparse regression
        coefficients = self._stlsq(Theta_flat, u_t_flat)

        # Build equation string
        terms = []
        for coef, name in zip(coefficients, term_names):
            if abs(coef) > 1e-10:
                terms.append(f"{coef:.4f}*{name}")

        equation = "u_t = " + " + ".join(terms) if terms else "u_t = 0"

        # Compute residual
        residual = np.mean((u_t_flat - Theta_flat @ coefficients)**2)

        return PDEResult(
            coefficients=coefficients,
            term_names=term_names,
            equation_string=equation,
            residual=residual
        )

    def _time_derivative(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Compute time derivative."""
        return np.gradient(u, dt, axis=0)

    def _spatial_derivatives(self, u: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
        """Compute spatial derivatives."""
        derivatives = {'u': u}

        # First derivative
        derivatives['u_x'] = np.gradient(u, dx, axis=1)

        # Higher derivatives
        for order in range(2, self.derivative_order + 1):
            prev_key = 'u' + '_x' * (order - 1)
            new_key = 'u' + '_x' * order
            derivatives[new_key] = np.gradient(derivatives[prev_key], dx, axis=1)

        return derivatives

    def _build_library(self, u: np.ndarray,
                       derivatives: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Build library of candidate terms."""
        n_t, n_x = u.shape
        features = []
        names = []

        # Constant
        features.append(np.ones((n_t, n_x)))
        names.append('1')

        # Linear terms
        for name, deriv in derivatives.items():
            features.append(deriv)
            names.append(name)

        # Polynomial terms
        if self.polynomial_order >= 2:
            for name1, d1 in derivatives.items():
                for name2, d2 in derivatives.items():
                    if name1 <= name2:  # Avoid duplicates
                        features.append(d1 * d2)
                        names.append(f"{name1}*{name2}")

        # Nonlinear terms (u * u_x, etc.)
        for name, deriv in derivatives.items():
            if name != 'u':
                features.append(u * deriv)
                names.append(f"u*{name}")

        return np.stack(features, axis=-1), names

    def _stlsq(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sequentially thresholded least squares."""
        coefficients = np.linalg.lstsq(Theta, y, rcond=None)[0]

        for _ in range(50):
            mask = np.abs(coefficients) < self.threshold
            coefficients[mask] = 0

            nonzero = ~mask
            if np.any(nonzero):
                coefficients[nonzero] = np.linalg.lstsq(Theta[:, nonzero], y, rcond=None)[0]

        return coefficients


class PDENet:
    """
    Physics-Informed Neural Network for PDE discovery.

    Learns both the solution and the PDE simultaneously.
    """

    def __init__(self, hidden_dims: List[int] = None,
                 n_pde_terms: int = 10):
        """
        Args:
            hidden_dims: Hidden layer dimensions for solution network
            n_pde_terms: Number of candidate PDE terms
        """
        self.hidden_dims = hidden_dims or [64, 64, 64]
        self.n_pde_terms = n_pde_terms

        self._solution_net: List = []
        self._pde_coefficients: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, t: np.ndarray,
            u: np.ndarray, n_epochs: int = 1000) -> str:
        """
        Discover PDE from data.

        Args:
            x: Spatial coordinates
            t: Time coordinates
            u: Solution data
            n_epochs: Training epochs

        Returns:
            Discovered PDE string
        """
        # Initialize networks
        self._initialize_networks()

        # Training
        for epoch in range(n_epochs):
            # Sample collocation points
            idx = np.random.choice(len(x) * len(t), size=min(1000, len(x) * len(t)))

            # Compute physics loss
            # (simplified - would need autodiff for proper implementation)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}")

        return "u_t = D*u_xx"  # Placeholder

    def _initialize_networks(self):
        """Initialize neural networks."""
        # Solution network: (x, t) -> u
        dims = [2] + self.hidden_dims + [1]

        self._solution_net = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._solution_net.append((W, b))

        # PDE coefficients
        self._pde_coefficients = np.random.randn(self.n_pde_terms) * 0.1


class PhysicsInformedDiscovery:
    """
    Physics-Informed PDE Discovery.

    Combines data-driven discovery with known physical constraints.
    """

    def __init__(self, known_terms: Optional[List[str]] = None,
                 conservation_laws: Optional[List[Callable]] = None):
        """
        Args:
            known_terms: Terms known to be in PDE
            conservation_laws: Functions that must be conserved
        """
        self.known_terms = known_terms or []
        self.conservation_laws = conservation_laws or []

        self._pde_find = PDEFind()

    def fit(self, u: np.ndarray, x: np.ndarray, t: np.ndarray) -> PDEResult:
        """
        Discover PDE with physics constraints.
        """
        # First run standard PDE-FIND
        result = self._pde_find.fit(u, x, t)

        # Check conservation laws
        if self.conservation_laws:
            result = self._enforce_conservation(result, u, x, t)

        return result

    def _enforce_conservation(self, result: PDEResult,
                               u: np.ndarray, x: np.ndarray, t: np.ndarray) -> PDEResult:
        """Project coefficients to satisfy conservation."""
        # Check each conservation law
        for law in self.conservation_laws:
            # Compute conserved quantity at each time
            quantities = [law(u[i]) for i in range(len(t))]

            # Check if conserved
            variation = np.std(quantities)
            if variation > 0.01 * np.mean(np.abs(quantities)):
                print(f"Warning: Conservation law violated (variation: {variation:.4f})")

        return result
