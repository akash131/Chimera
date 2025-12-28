"""
Surrogate models for active learning.

Probabilistic models that provide uncertainty estimates.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


class GaussianProcess:
    """
    Gaussian Process surrogate model.

    Provides mean prediction and uncertainty estimates.
    """

    def __init__(self, n_dims: int, length_scale: float = 1.0,
                 noise: float = 1e-6):
        self.n_dims = n_dims
        self.length_scale = length_scale
        self.noise = noise
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared exponential (RBF) kernel."""
        # Compute pairwise distances
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T

        return np.exp(-0.5 * dist_sq / self.length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self._X_train = X
        self._y_train = y

        # Compute kernel matrix
        K = self._kernel(X, X)
        K += self.noise * np.eye(len(X))

        # Precompute for prediction
        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation.

        Args:
            X: Test points (n_test, n_dims)

        Returns:
            (mean, std) predictions
        """
        if self._X_train is None:
            # No training data - return prior
            return np.zeros(len(X)), np.ones(len(X))

        # Kernel between test and training
        K_star = self._kernel(X, self._X_train)

        # Mean prediction
        mean = K_star @ self._alpha

        # Variance prediction
        K_star_star = self._kernel(X, X)
        var = np.diag(K_star_star) - np.sum(K_star @ self._K_inv * K_star, axis=1)
        var = np.maximum(var, 1e-10)  # Ensure positive

        return mean, np.sqrt(var)

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood for hyperparameter optimization."""
        if self._X_train is None:
            return -np.inf

        n = len(self._X_train)
        K = self._kernel(self._X_train, self._X_train)
        K += self.noise * np.eye(n)

        # Log determinant
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            return -np.inf

        # Data fit term
        data_fit = -0.5 * self._y_train @ self._K_inv @ self._y_train

        return data_fit - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)


class BayesianNeuralNetwork:
    """
    Bayesian Neural Network surrogate.

    Uses dropout for uncertainty estimation (MC Dropout).
    """

    def __init__(self, input_dim: int, hidden_dims: list = [64, 64],
                 dropout_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self._build_network()

    def _build_network(self):
        """Initialize network weights."""
        dims = [self.input_dim] + self.hidden_dims + [1]
        self._weights = []
        self._biases = []

        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            self._weights.append(w)
            self._biases.append(b)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            lr: float = 0.01):
        """Train network."""
        n = len(X)
        y = y.reshape(-1, 1)

        for epoch in range(epochs):
            # Forward pass with dropout
            h = X
            activations = [h]

            for i, (w, b) in enumerate(zip(self._weights, self._biases)):
                # Dropout
                mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (1 - self.dropout_rate)
                h = h * mask

                h = h @ w + b

                if i < len(self._weights) - 1:
                    h = np.maximum(0, h)  # ReLU

                activations.append(h)

            # Loss
            loss = np.mean((h - y) ** 2)

            # Backward pass
            grad = 2 * (h - y) / n

            for i in range(len(self._weights) - 1, -1, -1):
                # Gradient w.r.t. weights and biases
                grad_w = activations[i].T @ grad
                grad_b = np.sum(grad, axis=0)

                self._weights[i] -= lr * grad_w
                self._biases[i] -= lr * grad_b

                if i > 0:
                    grad = grad @ self._weights[i].T
                    grad = grad * (activations[i] > 0)  # ReLU derivative

    def predict(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty via MC Dropout.

        Args:
            X: Test points
            n_samples: Number of forward passes

        Returns:
            (mean, std) predictions
        """
        predictions = []

        for _ in range(n_samples):
            h = X

            for i, (w, b) in enumerate(zip(self._weights, self._biases)):
                # Dropout at test time
                mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (1 - self.dropout_rate)
                h = h * mask

                h = h @ w + b

                if i < len(self._weights) - 1:
                    h = np.maximum(0, h)

            predictions.append(h.flatten())

        predictions = np.array(predictions)

        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std


class EnsembleSurrogate:
    """
    Ensemble of diverse surrogate models.

    Combines predictions and uncertainties from multiple models.
    """

    def __init__(self, models: list):
        self.models = models

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models."""
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble.

        Returns:
            (mean, std) where std includes both model uncertainty and ensemble disagreement
        """
        means = []
        stds = []

        for model in self.models:
            m, s = model.predict(X)
            means.append(m)
            stds.append(s)

        means = np.array(means)
        stds = np.array(stds)

        # Ensemble mean
        ensemble_mean = np.mean(means, axis=0)

        # Total uncertainty: model uncertainty + epistemic uncertainty
        model_var = np.mean(stds ** 2, axis=0)
        epistemic_var = np.var(means, axis=0)
        total_std = np.sqrt(model_var + epistemic_var)

        return ensemble_mean, total_std
