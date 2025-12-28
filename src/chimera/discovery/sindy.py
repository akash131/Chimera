"""
Sparse Identification of Nonlinear Dynamics (SINDy).

Discover governing equations from time series data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SINDyResult:
    """Result of SINDy regression."""
    coefficients: np.ndarray  # (n_features, n_outputs)
    feature_names: List[str]
    equation_strings: List[str]
    complexity: int  # Number of non-zero terms
    r2_score: float


class SINDy:
    """
    Sparse Identification of Nonlinear Dynamics.

    Finds sparse representation:
    dx/dt = Θ(x) @ ξ

    where Θ is library of candidate functions and ξ are sparse coefficients.
    """

    def __init__(self, threshold: float = 0.1,
                 max_iter: int = 100,
                 alpha: float = 0.05,
                 optimizer: str = "stlsq"):
        """
        Args:
            threshold: Sparsity threshold for STLSQ
            max_iter: Maximum iterations
            alpha: Regularization parameter
            optimizer: "stlsq", "lasso", or "sr3"
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.optimizer = optimizer

        self._library: Optional[Callable] = None
        self._feature_names: List[str] = []
        self._coefficients: Optional[np.ndarray] = None

    def add_polynomial_library(self, degree: int = 3,
                                include_interaction: bool = True):
        """Add polynomial library."""
        def library(X):
            n_samples, n_features = X.shape
            features = [np.ones(n_samples)]
            names = ["1"]

            # Linear
            for i in range(n_features):
                features.append(X[:, i])
                names.append(f"x{i}")

            if degree >= 2:
                # Quadratic
                for i in range(n_features):
                    features.append(X[:, i]**2)
                    names.append(f"x{i}^2")

                if include_interaction:
                    for i in range(n_features):
                        for j in range(i+1, n_features):
                            features.append(X[:, i] * X[:, j])
                            names.append(f"x{i}*x{j}")

            if degree >= 3:
                # Cubic
                for i in range(n_features):
                    features.append(X[:, i]**3)
                    names.append(f"x{i}^3")

            return np.column_stack(features), names

        self._library = library

    def add_fourier_library(self, n_frequencies: int = 3):
        """Add Fourier library."""
        def library(X):
            n_samples, n_features = X.shape
            features = [np.ones(n_samples)]
            names = ["1"]

            for i in range(n_features):
                for k in range(1, n_frequencies + 1):
                    features.append(np.sin(k * X[:, i]))
                    names.append(f"sin({k}*x{i})")
                    features.append(np.cos(k * X[:, i]))
                    names.append(f"cos({k}*x{i})")

            return np.column_stack(features), names

        self._library = library

    def add_custom_library(self, functions: List[Callable],
                            names: List[str]):
        """Add custom function library."""
        def library(X):
            n_samples = X.shape[0]
            features = [np.ones(n_samples)]
            feature_names = ["1"]

            for func, name in zip(functions, names):
                features.append(func(X))
                feature_names.append(name)

            return np.column_stack(features), feature_names

        self._library = library

    def fit(self, X: np.ndarray, X_dot: np.ndarray) -> SINDyResult:
        """
        Fit SINDy model.

        Args:
            X: State data (n_samples, n_features)
            X_dot: Derivative data (n_samples, n_features)

        Returns:
            SINDyResult with discovered equations
        """
        if self._library is None:
            self.add_polynomial_library()

        # Build library
        Theta, self._feature_names = self._library(X)

        # Solve for coefficients
        if self.optimizer == "stlsq":
            coefficients = self._stlsq(Theta, X_dot)
        elif self.optimizer == "lasso":
            coefficients = self._lasso(Theta, X_dot)
        elif self.optimizer == "sr3":
            coefficients = self._sr3(Theta, X_dot)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        self._coefficients = coefficients

        # Build equation strings
        equation_strings = []
        for i in range(X_dot.shape[1]):
            terms = []
            for j, (coef, name) in enumerate(zip(coefficients[:, i], self._feature_names)):
                if abs(coef) > 1e-10:
                    if coef >= 0:
                        terms.append(f"+ {coef:.4f}*{name}")
                    else:
                        terms.append(f"- {abs(coef):.4f}*{name}")

            eq = f"dx{i}/dt = " + " ".join(terms) if terms else f"dx{i}/dt = 0"
            equation_strings.append(eq)

        # Compute R²
        prediction = Theta @ coefficients
        ss_res = np.sum((X_dot - prediction)**2)
        ss_tot = np.sum((X_dot - np.mean(X_dot, axis=0))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        complexity = np.sum(np.abs(coefficients) > 1e-10)

        return SINDyResult(
            coefficients=coefficients,
            feature_names=self._feature_names,
            equation_strings=equation_strings,
            complexity=complexity,
            r2_score=r2
        )

    def _stlsq(self, Theta: np.ndarray, X_dot: np.ndarray) -> np.ndarray:
        """
        Sequentially Thresholded Least Squares.

        Iteratively solve least squares and threshold small coefficients.
        """
        n_features = Theta.shape[1]
        n_outputs = X_dot.shape[1]

        # Initialize with least squares
        coefficients = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]

        for _ in range(self.max_iter):
            prev_coefficients = coefficients.copy()

            # Threshold small coefficients
            small_mask = np.abs(coefficients) < self.threshold
            coefficients[small_mask] = 0

            # Re-solve for non-zero coefficients
            for i in range(n_outputs):
                nonzero = ~small_mask[:, i]
                if np.any(nonzero):
                    Theta_reduced = Theta[:, nonzero]
                    coef_reduced = np.linalg.lstsq(Theta_reduced, X_dot[:, i], rcond=None)[0]
                    coefficients[nonzero, i] = coef_reduced

            # Check convergence
            if np.allclose(coefficients, prev_coefficients):
                break

        return coefficients

    def _lasso(self, Theta: np.ndarray, X_dot: np.ndarray) -> np.ndarray:
        """LASSO regression with coordinate descent."""
        n_features = Theta.shape[1]
        n_outputs = X_dot.shape[1]

        coefficients = np.zeros((n_features, n_outputs))

        # Precompute
        TT = Theta.T @ Theta
        Ty = Theta.T @ X_dot

        for output in range(n_outputs):
            coef = np.zeros(n_features)

            for _ in range(self.max_iter):
                for j in range(n_features):
                    # Partial residual
                    r = Ty[j, output] - TT[j, :] @ coef + TT[j, j] * coef[j]

                    # Soft thresholding
                    if r > self.alpha:
                        coef[j] = (r - self.alpha) / TT[j, j]
                    elif r < -self.alpha:
                        coef[j] = (r + self.alpha) / TT[j, j]
                    else:
                        coef[j] = 0

            coefficients[:, output] = coef

        return coefficients

    def _sr3(self, Theta: np.ndarray, X_dot: np.ndarray) -> np.ndarray:
        """
        Sparse Relaxed Regularized Regression (SR3).

        More robust to noisy data than STLSQ.
        """
        n_features = Theta.shape[1]
        n_outputs = X_dot.shape[1]

        nu = 1.0  # Relaxation parameter

        # Initialize
        coefficients = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]
        W = coefficients.copy()

        TT = Theta.T @ Theta
        Ty = Theta.T @ X_dot

        for _ in range(self.max_iter):
            # Update coefficients
            coefficients = np.linalg.solve(TT + nu * np.eye(n_features), Ty + nu * W)

            # Update W with thresholding
            W = coefficients.copy()
            W[np.abs(W) < self.threshold] = 0

            # Check convergence
            if np.linalg.norm(coefficients - W) < 1e-6:
                break

        return W

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict derivatives from state."""
        if self._coefficients is None:
            raise ValueError("Call fit() first")

        Theta, _ = self._library(X)
        return Theta @ self._coefficients

    def simulate(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Simulate discovered system forward."""
        if self._coefficients is None:
            raise ValueError("Call fit() first")

        def f(x):
            Theta, _ = self._library(x.reshape(1, -1))
            return (Theta @ self._coefficients).flatten()

        # Forward Euler
        trajectory = np.zeros((len(t), len(x0)))
        trajectory[0] = x0

        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            trajectory[i] = trajectory[i-1] + dt * f(trajectory[i-1])

        return trajectory


class SINDyPI:
    """
    SINDy with Physics-Informed constraints.

    Incorporates known physical constraints (conservation laws, symmetries).
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self._sindy = SINDy(threshold=threshold)
        self._constraints: List[Callable] = []

    def add_conservation_constraint(self, quantities: List[int]):
        """Add conservation law: sum of quantities is constant."""
        def constraint(coefficients):
            # Sum of relevant rows should be zero
            return np.sum(coefficients[quantities, :], axis=0)

        self._constraints.append(constraint)

    def add_symmetry_constraint(self, var1: int, var2: int):
        """Add symmetry: equations for var1 and var2 should be related."""
        def constraint(coefficients):
            return coefficients[:, var1] - coefficients[:, var2]

        self._constraints.append(constraint)

    def fit(self, X: np.ndarray, X_dot: np.ndarray) -> SINDyResult:
        """Fit with constraints."""
        # First get unconstrained solution
        result = self._sindy.fit(X, X_dot)

        # Project onto constraint manifold
        coefficients = result.coefficients.copy()

        for _ in range(100):
            for constraint in self._constraints:
                violation = constraint(coefficients)
                # Gradient projection (simplified)
                coefficients -= 0.1 * violation.reshape(1, -1)

            # Re-threshold
            coefficients[np.abs(coefficients) < self.threshold] = 0

        result.coefficients = coefficients
        return result


class WeakSINDy:
    """
    Weak-form SINDy for noisy data.

    Uses integration instead of differentiation for noise robustness.
    """

    def __init__(self, threshold: float = 0.1,
                 test_function: str = "polynomial"):
        """
        Args:
            threshold: Sparsity threshold
            test_function: "polynomial", "fourier", or "gaussian"
        """
        self.threshold = threshold
        self.test_function = test_function

    def fit(self, X: np.ndarray, t: np.ndarray) -> SINDyResult:
        """
        Fit using weak formulation.

        Integrates both sides against test functions.
        """
        n_samples, n_features = X.shape
        n_test = min(20, n_samples // 2)

        # Generate test functions
        test_funcs = self._generate_test_functions(t, n_test)

        # Build weak form matrices
        # integral(phi * dx/dt) = -integral(dphi/dt * x) for compactly supported phi
        # integral(phi * Theta(x) @ xi) = integral(phi * dx/dt)

        # Compute library
        Theta, feature_names = self._polynomial_library(X)

        # Weak form matrices
        A = np.zeros((n_test * n_features, Theta.shape[1] * n_features))
        b = np.zeros(n_test * n_features)

        for k in range(n_test):
            phi = test_funcs[k]
            dphi = np.gradient(phi, t)

            for i in range(n_features):
                row = k * n_features + i

                # RHS: -integral(dphi/dt * x_i)
                b[row] = -np.trapz(dphi * X[:, i], t)

                # LHS: integral(phi * theta_j) for each library term
                for j in range(Theta.shape[1]):
                    col = j * n_features + i
                    A[row, col] = np.trapz(phi * Theta[:, j], t)

        # Solve sparse regression
        coefficients = self._stlsq(A, b)
        coefficients = coefficients.reshape(Theta.shape[1], n_features)

        # Build equations
        equation_strings = []
        for i in range(n_features):
            terms = []
            for j, (coef, name) in enumerate(zip(coefficients[:, i], feature_names)):
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:.4f}*{name}")
            eq = f"dx{i}/dt = " + " + ".join(terms) if terms else f"dx{i}/dt = 0"
            equation_strings.append(eq)

        return SINDyResult(
            coefficients=coefficients,
            feature_names=feature_names,
            equation_strings=equation_strings,
            complexity=np.sum(np.abs(coefficients) > 1e-10),
            r2_score=0.0  # Would need to compute differently
        )

    def _generate_test_functions(self, t: np.ndarray, n_test: int) -> List[np.ndarray]:
        """Generate compactly supported test functions."""
        test_funcs = []
        T = t[-1] - t[0]

        for k in range(n_test):
            center = t[0] + (k + 0.5) * T / n_test
            width = T / n_test / 2

            if self.test_function == "polynomial":
                # Bump function
                phi = np.maximum(0, 1 - ((t - center) / width)**2)**2
            elif self.test_function == "gaussian":
                phi = np.exp(-((t - center) / width)**2)
            else:
                phi = np.sin(2 * np.pi * k * t / T)

            test_funcs.append(phi)

        return test_funcs

    def _polynomial_library(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Simple polynomial library."""
        n_samples, n_features = X.shape
        features = [np.ones(n_samples)]
        names = ["1"]

        for i in range(n_features):
            features.append(X[:, i])
            names.append(f"x{i}")

        for i in range(n_features):
            features.append(X[:, i]**2)
            names.append(f"x{i}^2")

        for i in range(n_features):
            for j in range(i+1, n_features):
                features.append(X[:, i] * X[:, j])
                names.append(f"x{i}*x{j}")

        return np.column_stack(features), names

    def _stlsq(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """STLSQ for weak form."""
        coefficients = np.linalg.lstsq(A, b, rcond=None)[0]

        for _ in range(50):
            mask = np.abs(coefficients) < self.threshold
            coefficients[mask] = 0

            nonzero = ~mask
            if np.any(nonzero):
                coefficients[nonzero] = np.linalg.lstsq(A[:, nonzero], b, rcond=None)[0]

        return coefficients


class EnsembleSINDy:
    """
    Ensemble SINDy for uncertainty quantification.

    Trains multiple SINDy models with bootstrap/bagging.
    """

    def __init__(self, n_models: int = 100,
                 threshold_range: Tuple[float, float] = (0.05, 0.2),
                 subsample_fraction: float = 0.8):
        """
        Args:
            n_models: Number of ensemble members
            threshold_range: Range of thresholds to sample
            subsample_fraction: Fraction of data for each model
        """
        self.n_models = n_models
        self.threshold_range = threshold_range
        self.subsample_fraction = subsample_fraction

        self._models: List[SINDy] = []
        self._results: List[SINDyResult] = []

    def fit(self, X: np.ndarray, X_dot: np.ndarray) -> Dict:
        """
        Fit ensemble of SINDy models.

        Returns:
            Dictionary with mean coefficients, std, and inclusion probabilities
        """
        n_samples = X.shape[0]
        subsample_size = int(n_samples * self.subsample_fraction)

        all_coefficients = []

        for i in range(self.n_models):
            # Random threshold
            threshold = np.random.uniform(*self.threshold_range)

            # Bootstrap sample
            indices = np.random.choice(n_samples, size=subsample_size, replace=True)
            X_sub = X[indices]
            X_dot_sub = X_dot[indices]

            # Fit model
            model = SINDy(threshold=threshold)
            model.add_polynomial_library()

            try:
                result = model.fit(X_sub, X_dot_sub)
                self._models.append(model)
                self._results.append(result)
                all_coefficients.append(result.coefficients)
            except:
                continue

        all_coefficients = np.array(all_coefficients)

        # Statistics
        mean_coef = np.mean(all_coefficients, axis=0)
        std_coef = np.std(all_coefficients, axis=0)
        inclusion_prob = np.mean(np.abs(all_coefficients) > 1e-10, axis=0)

        # Consensus equation (terms that appear in > 50% of models)
        consensus = mean_coef.copy()
        consensus[inclusion_prob < 0.5] = 0

        return {
            "mean_coefficients": mean_coef,
            "std_coefficients": std_coef,
            "inclusion_probabilities": inclusion_prob,
            "consensus_coefficients": consensus,
            "n_successful_models": len(self._results),
            "feature_names": self._results[0].feature_names if self._results else []
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Returns:
            (mean_prediction, std_prediction)
        """
        predictions = []

        for model in self._models:
            predictions.append(model.predict(X))

        predictions = np.array(predictions)

        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
