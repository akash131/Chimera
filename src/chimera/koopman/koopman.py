"""
Koopman operator methods for nonlinear dynamics.

Koopman theory: any nonlinear system can be represented as
an infinite-dimensional linear system in observable space.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import linalg


@dataclass
class KoopmanResult:
    """Result from Koopman analysis."""
    K: np.ndarray  # Koopman matrix
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    modes: np.ndarray
    reconstruction_error: float


class KoopmanOperator:
    """
    Finite-dimensional approximation of Koopman operator.

    The Koopman operator K acts on observables g(x):
    (K g)(x) = g(F(x))

    where F is the dynamics.
    """

    def __init__(self, observables: Optional[List[Callable]] = None,
                 rank: Optional[int] = None):
        """
        Args:
            observables: List of observable functions
            rank: Truncation rank
        """
        self.observables = observables
        self.rank = rank

        self._K: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None

    def set_polynomial_observables(self, n_vars: int, degree: int = 2):
        """Set polynomial observables."""
        observables = []

        # Linear
        for i in range(n_vars):
            observables.append(lambda x, i=i: x[i])

        # Quadratic
        if degree >= 2:
            for i in range(n_vars):
                observables.append(lambda x, i=i: x[i]**2)
                for j in range(i+1, n_vars):
                    observables.append(lambda x, i=i, j=j: x[i] * x[j])

        # Cubic
        if degree >= 3:
            for i in range(n_vars):
                observables.append(lambda x, i=i: x[i]**3)

        self.observables = observables

    def lift(self, X: np.ndarray) -> np.ndarray:
        """Lift state to observable space."""
        if self.observables is None:
            return X

        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        X = X.reshape(n_samples, -1)

        lifted = []
        for x in X:
            obs_values = [g(x) for g in self.observables]
            lifted.append(obs_values)

        return np.array(lifted)

    def fit(self, X: np.ndarray, X_next: np.ndarray) -> KoopmanResult:
        """
        Fit Koopman operator from snapshot pairs.

        Args:
            X: States at time t (n_samples, n_features)
            X_next: States at time t+dt

        Returns:
            KoopmanResult
        """
        # Lift to observable space
        G = self.lift(X)
        G_next = self.lift(X_next)

        # Solve for Koopman matrix: G_next ≈ G @ K^T
        # Least squares: K = (G^T G)^{-1} G^T G_next
        self._K = linalg.lstsq(G, G_next)[0].T

        # Truncate if specified
        if self.rank is not None:
            U, s, Vt = linalg.svd(self._K)
            self._K = U[:, :self.rank] @ np.diag(s[:self.rank]) @ Vt[:self.rank, :]

        # Eigendecomposition
        self._eigenvalues, self._eigenvectors = linalg.eig(self._K)

        # Sort by magnitude
        idx = np.argsort(np.abs(self._eigenvalues))[::-1]
        self._eigenvalues = self._eigenvalues[idx]
        self._eigenvectors = self._eigenvectors[:, idx]

        # Compute modes
        modes = G.T @ self._eigenvectors

        # Reconstruction error
        G_pred = G @ self._K.T
        error = np.linalg.norm(G_next - G_pred) / np.linalg.norm(G_next)

        return KoopmanResult(
            K=self._K,
            eigenvalues=self._eigenvalues,
            eigenfunctions=self._eigenvectors,
            modes=modes,
            reconstruction_error=error
        )

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict trajectory using Koopman operator."""
        if self._K is None:
            raise ValueError("Call fit() first")

        g = self.lift(x0.reshape(1, -1))[0]

        trajectory = [x0.copy()]

        for _ in range(n_steps):
            g = self._K @ g

            # Project back to state space (first n_vars observables are linear)
            n_vars = len(x0)
            x = g[:n_vars]
            trajectory.append(x.copy())

        return np.array(trajectory)

    def spectral_analysis(self) -> Dict:
        """Analyze Koopman spectrum."""
        if self._eigenvalues is None:
            raise ValueError("Call fit() first")

        # Continuous-time eigenvalues
        dt = 1.0  # Assume unit time step
        continuous = np.log(self._eigenvalues + 1e-10) / dt

        return {
            "discrete_eigenvalues": self._eigenvalues,
            "continuous_eigenvalues": continuous,
            "frequencies": np.imag(continuous) / (2 * np.pi),
            "growth_rates": np.real(continuous),
            "stable_modes": np.sum(np.abs(self._eigenvalues) < 1),
            "unstable_modes": np.sum(np.abs(self._eigenvalues) > 1)
        }


class ExtendedDMDKoopman(KoopmanOperator):
    """
    Extended DMD for Koopman approximation.

    Uses dictionary of observables for better approximation.
    """

    def __init__(self, dictionary: str = "polynomial",
                 degree: int = 3,
                 n_rbf: int = 100):
        """
        Args:
            dictionary: "polynomial", "rbf", or "fourier"
            degree: Polynomial degree
            n_rbf: Number of RBF centers
        """
        super().__init__()
        self.dictionary = dictionary
        self.degree = degree
        self.n_rbf = n_rbf

    def fit(self, X: np.ndarray, X_next: np.ndarray) -> KoopmanResult:
        """Fit using extended dictionary."""
        n_vars = X.shape[1]

        # Build dictionary
        if self.dictionary == "polynomial":
            self.set_polynomial_observables(n_vars, self.degree)
        elif self.dictionary == "rbf":
            self._set_rbf_observables(X)
        elif self.dictionary == "fourier":
            self._set_fourier_observables(n_vars)

        return super().fit(X, X_next)

    def _set_rbf_observables(self, X: np.ndarray):
        """Set RBF observables."""
        # Choose centers from data
        indices = np.random.choice(len(X), size=min(self.n_rbf, len(X)), replace=False)
        centers = X[indices]
        sigma = np.std(X)

        self.observables = []
        for center in centers:
            self.observables.append(
                lambda x, c=center, s=sigma: np.exp(-np.linalg.norm(x - c)**2 / (2 * s**2))
            )

    def _set_fourier_observables(self, n_vars: int):
        """Set Fourier observables."""
        self.observables = []

        for k in range(1, 5):
            for i in range(n_vars):
                self.observables.append(lambda x, k=k, i=i: np.sin(k * x[i]))
                self.observables.append(lambda x, k=k, i=i: np.cos(k * x[i]))


class DeepKoopman:
    """
    Deep Learning for Koopman operator.

    Learn observables using neural network autoencoders.
    """

    def __init__(self, latent_dim: int = 32,
                 encoder_dims: List[int] = None,
                 decoder_dims: List[int] = None):
        """
        Args:
            latent_dim: Dimension of learned observable space
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
        """
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims or [64, 64]
        self.decoder_dims = decoder_dims or [64, 64]

        self._encoder: List = []
        self._decoder: List = []
        self._K: Optional[np.ndarray] = None

    def _initialize(self, input_dim: int):
        """Initialize networks."""
        # Encoder
        dims = [input_dim] + self.encoder_dims + [self.latent_dim]
        self._encoder = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._encoder.append((W, b))

        # Decoder
        dims = [self.latent_dim] + self.decoder_dims + [input_dim]
        self._decoder = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._decoder.append((W, b))

        # Koopman matrix
        self._K = np.eye(self.latent_dim) + 0.01 * np.random.randn(self.latent_dim, self.latent_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode to latent space."""
        h = x
        for i, (W, b) in enumerate(self._encoder):
            h = h @ W + b
            if i < len(self._encoder) - 1:
                h = np.tanh(h)
        return h

    def decode(self, g: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        h = g
        for i, (W, b) in enumerate(self._decoder):
            h = h @ W + b
            if i < len(self._decoder) - 1:
                h = np.tanh(h)
        return h

    def fit(self, X: np.ndarray, X_next: np.ndarray,
            n_epochs: int = 1000, lr: float = 0.001) -> Dict:
        """
        Train deep Koopman model.

        Loss = reconstruction + prediction + linearity
        """
        self._initialize(X.shape[1])

        for epoch in range(n_epochs):
            # Forward pass
            g = self.encode(X)
            g_next = self.encode(X_next)

            # Predicted next observable
            g_pred = g @ self._K.T

            # Reconstructions
            x_recon = self.decode(g)
            x_next_recon = self.decode(g_pred)

            # Losses
            recon_loss = np.mean((X - x_recon)**2)
            pred_loss = np.mean((g_next - g_pred)**2)
            future_recon_loss = np.mean((X_next - x_next_recon)**2)

            total_loss = recon_loss + pred_loss + future_recon_loss

            # Simplified gradient update
            # Update K
            self._K -= lr * 2 * (g_pred - g_next).T @ g / len(X)

            # Update networks (numerical gradient)
            for W, b in self._encoder + self._decoder:
                W -= lr * 0.01 * np.random.randn(*W.shape) * total_loss

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}")

        return {
            "K": self._K,
            "eigenvalues": linalg.eigvals(self._K),
            "final_loss": total_loss
        }

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict trajectory."""
        g = self.encode(x0.reshape(1, -1))

        trajectory = [x0.copy()]

        for _ in range(n_steps):
            g = g @ self._K.T
            x = self.decode(g)[0]
            trajectory.append(x.copy())

        return np.array(trajectory)


class KoopmanMPC:
    """
    Model Predictive Control using Koopman operator.

    Linear MPC in observable space for nonlinear systems.
    """

    def __init__(self, koopman: KoopmanOperator,
                 horizon: int = 10,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None):
        """
        Args:
            koopman: Fitted Koopman operator
            horizon: Prediction horizon
            Q: State cost matrix
            R: Control cost matrix
        """
        self.koopman = koopman
        self.horizon = horizon
        self.Q = Q
        self.R = R

    def solve(self, x_current: np.ndarray, x_target: np.ndarray,
              B: np.ndarray) -> np.ndarray:
        """
        Solve MPC problem.

        Args:
            x_current: Current state
            x_target: Target state
            B: Control matrix in observable space

        Returns:
            Optimal control sequence
        """
        n_obs = self.koopman._K.shape[0]
        n_control = B.shape[1]

        if self.Q is None:
            self.Q = np.eye(n_obs)
        if self.R is None:
            self.R = 0.1 * np.eye(n_control)

        K = self.koopman._K

        # Lift states
        g_current = self.koopman.lift(x_current.reshape(1, -1))[0]
        g_target = self.koopman.lift(x_target.reshape(1, -1))[0]

        # Build prediction matrices
        # g_{k+N} = K^N g_0 + sum K^i B u_{N-1-i}
        n = self.horizon * n_control

        # Simplified: use iterative approach
        u_opt = np.zeros(n)

        for _ in range(50):  # Iterations
            # Compute trajectory
            g = g_current.copy()
            grad = np.zeros(n)

            for k in range(self.horizon):
                u_k = u_opt[k*n_control:(k+1)*n_control]
                g = K @ g + B @ u_k

                # State cost gradient
                error = g - g_target
                grad[k*n_control:(k+1)*n_control] += B.T @ (self.Q @ error)

                # Control cost
                grad[k*n_control:(k+1)*n_control] += self.R @ u_k

            u_opt -= 0.01 * grad

        return u_opt.reshape(self.horizon, n_control)
