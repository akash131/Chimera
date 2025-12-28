"""
Dynamic Mode Decomposition (DMD) and variants.

Extract spatiotemporal modes and predict dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg


@dataclass
class DMDResult:
    """Results from DMD computation."""
    modes: np.ndarray  # (n_dofs, n_modes) - spatial modes (complex)
    eigenvalues: np.ndarray  # (n_modes,) - discrete-time eigenvalues
    continuous_eigenvalues: np.ndarray  # (n_modes,) - continuous-time eigenvalues
    amplitudes: np.ndarray  # (n_modes,) - mode amplitudes
    frequencies: np.ndarray  # (n_modes,) - oscillation frequencies
    growth_rates: np.ndarray  # (n_modes,) - growth/decay rates
    dt: float


class DMD:
    """
    Standard Dynamic Mode Decomposition.

    Approximates Koopman operator from time series data.
    Finds linear operator A such that X' ≈ A X.
    """

    def __init__(self, rank: Optional[int] = None,
                 svd_threshold: float = 1e-10,
                 exact: bool = True):
        """
        Initialize DMD.

        Args:
            rank: Truncation rank (None = full rank)
            svd_threshold: Threshold for singular values
            exact: Use exact DMD (project onto high-dimensional modes)
        """
        self.rank = rank
        self.svd_threshold = svd_threshold
        self.exact = exact
        self._result: Optional[DMDResult] = None

    def fit(self, X: np.ndarray, dt: float = 1.0) -> DMDResult:
        """
        Fit DMD to snapshot data.

        Args:
            X: Snapshot matrix (n_dofs, n_snapshots) - time-ordered
            dt: Time step between snapshots

        Returns:
            DMDResult with modes, eigenvalues, etc.
        """
        if X.shape[0] > X.shape[1]:
            # Assume (n_dofs, n_snapshots)
            pass
        else:
            X = X.T

        n_dofs, n_snapshots = X.shape

        # Split into X and X'
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # SVD of X1
        U, s, Vh = linalg.svd(X1, full_matrices=False)

        # Truncate
        if self.rank is not None:
            r = min(self.rank, len(s))
        else:
            r = np.sum(s > self.svd_threshold * s[0])

        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]

        # Reduced operator A_tilde = U^T X' V S^{-1}
        A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1.0 / s_r)

        # Eigendecomposition
        eigenvalues, W = linalg.eig(A_tilde)

        # DMD modes
        if self.exact:
            # Exact modes: Phi = X' V S^{-1} W
            modes = X2 @ Vh_r.T @ np.diag(1.0 / s_r) @ W
        else:
            # Projected modes: Phi = U W
            modes = U_r @ W

        # Compute amplitudes (initial condition projection)
        x0 = X[:, 0]
        amplitudes = linalg.lstsq(modes, x0)[0]

        # Continuous-time eigenvalues
        continuous_eigenvalues = np.log(eigenvalues) / dt

        # Frequencies and growth rates
        frequencies = np.imag(continuous_eigenvalues) / (2 * np.pi)
        growth_rates = np.real(continuous_eigenvalues)

        self._result = DMDResult(
            modes=modes,
            eigenvalues=eigenvalues,
            continuous_eigenvalues=continuous_eigenvalues,
            amplitudes=amplitudes,
            frequencies=frequencies,
            growth_rates=growth_rates,
            dt=dt
        )

        return self._result

    def predict(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Predict state at time t.

        Args:
            t: Time or array of times

        Returns:
            Predicted state(s)
        """
        if self._result is None:
            raise ValueError("Call fit() first")

        t = np.atleast_1d(t)

        # x(t) = sum_i b_i * exp(omega_i * t) * phi_i
        omega = self._result.continuous_eigenvalues
        b = self._result.amplitudes
        Phi = self._result.modes

        predictions = np.zeros((len(t), Phi.shape[0]), dtype=complex)

        for i, ti in enumerate(t):
            dynamics = b * np.exp(omega * ti)
            predictions[i] = Phi @ dynamics

        return np.real(predictions.T)

    def reconstruct(self, n_steps: int) -> np.ndarray:
        """Reconstruct time series."""
        t = np.arange(n_steps) * self._result.dt
        return self.predict(t)


class ExtendedDMD:
    """
    Extended DMD with dictionary of observables.

    Lifts data to feature space for better Koopman approximation.
    """

    def __init__(self, observables: Optional[List[Callable]] = None,
                 rank: Optional[int] = None):
        """
        Args:
            observables: List of observable functions g(x) -> scalar/vector
            rank: Truncation rank for DMD
        """
        self.observables = observables or []
        self.rank = rank
        self._dmd = DMD(rank=rank)

    def add_polynomial_observables(self, degree: int = 2, n_vars: int = None):
        """Add polynomial observables up to given degree."""
        from itertools import combinations_with_replacement

        def make_poly(indices):
            def f(x):
                result = 1.0
                for i in indices:
                    result *= x[i] if i < len(x) else 1.0
                return result
            return f

        if n_vars is None:
            n_vars = 10  # Default

        for d in range(1, degree + 1):
            for indices in combinations_with_replacement(range(n_vars), d):
                self.observables.append(make_poly(indices))

    def add_rbf_observables(self, centers: np.ndarray, epsilon: float = 1.0):
        """Add RBF observables."""
        for center in centers:
            def f(x, c=center, eps=epsilon):
                return np.exp(-eps * np.linalg.norm(x - c)**2)
            self.observables.append(f)

    def lift(self, X: np.ndarray) -> np.ndarray:
        """Lift data to observable space."""
        n_dofs, n_snapshots = X.shape

        # Evaluate all observables
        lifted = [X]  # Include original state

        for g in self.observables:
            g_values = np.zeros(n_snapshots)
            for i in range(n_snapshots):
                val = g(X[:, i])
                if np.isscalar(val):
                    g_values[i] = val
                else:
                    g_values[i] = val[0] if len(val) > 0 else 0
            lifted.append(g_values.reshape(1, -1))

        return np.vstack(lifted)

    def fit(self, X: np.ndarray, dt: float = 1.0) -> DMDResult:
        """Fit extended DMD."""
        # Lift to observable space
        X_lifted = self.lift(X)

        # Standard DMD on lifted data
        return self._dmd.fit(X_lifted, dt)

    def predict(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Predict in lifted space."""
        return self._dmd.predict(t)


class KernelDMD:
    """
    Kernel DMD for nonlinear Koopman approximation.

    Uses kernel trick to work in infinite-dimensional feature space.
    """

    def __init__(self, kernel: str = "rbf",
                 gamma: float = 1.0,
                 rank: Optional[int] = None):
        """
        Args:
            kernel: "rbf", "polynomial", or "linear"
            gamma: Kernel parameter
            rank: Truncation rank
        """
        self.kernel = kernel
        self.gamma = gamma
        self.rank = rank

        self._X1: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K[i,j] = k(x_i, y_j)."""
        if self.kernel == "linear":
            return X.T @ Y
        elif self.kernel == "polynomial":
            return (1 + X.T @ Y) ** int(self.gamma)
        elif self.kernel == "rbf":
            X_sq = np.sum(X**2, axis=0)
            Y_sq = np.sum(Y**2, axis=0)
            dist_sq = X_sq[:, np.newaxis] + Y_sq[np.newaxis, :] - 2 * X.T @ Y
            return np.exp(-self.gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, dt: float = 1.0):
        """Fit kernel DMD."""
        if X.shape[0] > X.shape[1]:
            pass
        else:
            X = X.T

        X1 = X[:, :-1]
        X2 = X[:, 1:]
        self._X1 = X1

        m = X1.shape[1]

        # Kernel matrices
        K1 = self._kernel_matrix(X1, X1)
        K2 = self._kernel_matrix(X1, X2)

        # Regularized pseudo-inverse
        reg = 1e-6 * np.eye(m)
        K1_inv = np.linalg.solve(K1 + reg, np.eye(m))

        # Koopman matrix in feature space
        A_hat = K1_inv @ K2

        # Eigendecomposition
        eigenvalues, W = linalg.eig(A_hat)

        # Truncate
        if self.rank is not None:
            idx = np.argsort(np.abs(eigenvalues))[::-1][:self.rank]
            eigenvalues = eigenvalues[idx]
            W = W[:, idx]

        self._eigenvalues = eigenvalues
        self._alpha = W
        self._dt = dt

        return eigenvalues

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict trajectory from initial condition."""
        if self._X1 is None:
            raise ValueError("Call fit() first")

        n_dofs = x0.shape[0]
        trajectory = np.zeros((n_dofs, n_steps))

        x = x0.copy()
        for i in range(n_steps):
            trajectory[:, i] = x

            # Kernel evaluation
            k = self._kernel_matrix(self._X1, x.reshape(-1, 1)).flatten()

            # Predict next state using eigenfunctions
            x_next = np.zeros(n_dofs)
            for j, (lam, alpha) in enumerate(zip(self._eigenvalues, self._alpha.T)):
                psi = k @ alpha  # Eigenfunction value
                x_next += np.real(lam * psi) * self._X1[:, np.argmax(np.abs(alpha))]

            x = x_next / (np.linalg.norm(x_next) + 1e-10) * np.linalg.norm(x)

        return trajectory


class StreamingDMD:
    """
    Online/streaming DMD for real-time applications.

    Updates DMD incrementally without storing all snapshots.
    """

    def __init__(self, rank: int = 10,
                 forgetting_factor: float = 1.0):
        """
        Args:
            rank: Fixed rank for online DMD
            forgetting_factor: Weight decay for old data (< 1 for time-varying)
        """
        self.rank = rank
        self.forgetting_factor = forgetting_factor

        self._A: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._Q: Optional[np.ndarray] = None
        self._n_updates = 0

    def initialize(self, X_init: np.ndarray, dt: float = 1.0):
        """Initialize with batch of snapshots."""
        self._dt = dt

        X1 = X_init[:, :-1]
        X2 = X_init[:, 1:]

        # Initial DMD fit
        U, s, Vh = linalg.svd(X1, full_matrices=False)
        r = min(self.rank, len(s))

        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]

        # Initial A in reduced space
        A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1.0 / s_r)

        # Store for updates
        self._A = A_tilde
        self._P = U_r.T @ X1
        self._Q = X1

        # For rank-1 updates
        self._Px = np.eye(r)
        self._n_updates = X1.shape[1]

    def update(self, x_old: np.ndarray, x_new: np.ndarray):
        """
        Rank-1 update with new snapshot pair.

        Args:
            x_old: Previous state
            x_new: New state (x_new ≈ A @ x_old)
        """
        if self._A is None:
            raise ValueError("Call initialize() first")

        rho = self.forgetting_factor
        r = self.rank

        # Project to POD space (using stored basis)
        # This is simplified; full implementation would update POD basis too

        # Rank-1 update of A
        # A_new = A_old + (y - A_old @ x) @ x^T @ P / (1 + x^T @ P @ x)
        # where P is inverse covariance

        x_proj = self._P @ x_old[:self._P.shape[1]] if self._P.shape[1] <= len(x_old) else np.zeros(r)
        y_proj = self._P @ x_new[:self._P.shape[1]] if self._P.shape[1] <= len(x_new) else np.zeros(r)

        # Update inverse covariance
        Px = self._Px @ x_proj
        denom = rho + x_proj @ Px
        self._Px = (self._Px - np.outer(Px, Px) / denom) / rho

        # Update A
        innovation = y_proj - self._A @ x_proj
        self._A += np.outer(innovation, Px) / denom

        self._n_updates += 1

    def get_eigenvalues(self) -> np.ndarray:
        """Get current eigenvalues."""
        if self._A is None:
            raise ValueError("Not initialized")
        return linalg.eigvals(self._A)

    def get_frequencies(self) -> np.ndarray:
        """Get current frequencies."""
        eigenvalues = self.get_eigenvalues()
        continuous = np.log(eigenvalues) / self._dt
        return np.imag(continuous) / (2 * np.pi)


class OptimizedDMD:
    """
    Optimized DMD that jointly optimizes modes, amplitudes, and eigenvalues.

    Better for noisy data than standard DMD.
    """

    def __init__(self, rank: int = 10,
                 max_iter: int = 100,
                 tol: float = 1e-6):
        """
        Args:
            rank: Number of modes
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, dt: float = 1.0) -> DMDResult:
        """
        Fit optimized DMD.

        Solves: min_{Phi, b, omega} ||X - Phi @ diag(b) @ Vander(exp(omega*dt))||^2
        """
        if X.shape[0] > X.shape[1]:
            pass
        else:
            X = X.T

        n_dofs, n_snapshots = X.shape
        r = self.rank

        # Initialize with standard DMD
        dmd = DMD(rank=r)
        result = dmd.fit(X, dt)

        Phi = result.modes
        b = result.amplitudes
        omega = result.continuous_eigenvalues

        times = np.arange(n_snapshots) * dt

        for iteration in range(self.max_iter):
            # Fix omega, b: solve for Phi
            Vander = np.vander(np.exp(omega * dt), n_snapshots, increasing=True).T
            B = np.diag(b) @ Vander
            Phi_new = X @ linalg.pinv(B)

            # Fix Phi, omega: solve for b
            for j in range(r):
                exp_omega_t = np.exp(omega[j] * times)
                b[j] = np.real(Phi[:, j].conj() @ X @ exp_omega_t.conj()) / \
                       (np.linalg.norm(Phi[:, j])**2 * n_snapshots)

            # Fix Phi, b: solve for omega (gradient descent)
            for j in range(r):
                grad = 0
                for k in range(n_snapshots):
                    residual = X[:, k] - Phi @ (b * np.exp(omega * times[k]))
                    grad += -2 * np.real(
                        b[j].conj() * Phi[:, j].conj() @ residual * times[k] * np.exp(omega[j].conj() * times[k])
                    )
                omega[j] -= 0.01 * grad / n_snapshots  # Small learning rate

            # Check convergence
            reconstruction = Phi @ np.diag(b) @ np.vander(np.exp(omega * dt), n_snapshots, increasing=True).T
            error = np.linalg.norm(X - reconstruction) / np.linalg.norm(X)

            if error < self.tol:
                break

            Phi = Phi_new

        # Compute discrete eigenvalues
        eigenvalues = np.exp(omega * dt)

        return DMDResult(
            modes=Phi,
            eigenvalues=eigenvalues,
            continuous_eigenvalues=omega,
            amplitudes=b,
            frequencies=np.imag(omega) / (2 * np.pi),
            growth_rates=np.real(omega),
            dt=dt
        )


class BayesianDMD:
    """
    Bayesian DMD with uncertainty quantification.

    Provides posterior distributions over modes and eigenvalues.
    """

    def __init__(self, rank: int = 10,
                 n_samples: int = 1000,
                 noise_std: float = 0.01):
        """
        Args:
            rank: Number of modes
            n_samples: Number of posterior samples
            noise_std: Assumed observation noise
        """
        self.rank = rank
        self.n_samples = n_samples
        self.noise_std = noise_std

    def fit(self, X: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Fit Bayesian DMD.

        Returns dictionary with posterior samples.
        """
        if X.shape[0] > X.shape[1]:
            pass
        else:
            X = X.T

        n_dofs, n_snapshots = X.shape
        r = self.rank

        # Point estimate from standard DMD
        dmd = DMD(rank=r)
        result = dmd.fit(X, dt)

        # Posterior sampling via Laplace approximation
        # Approximate posterior as Gaussian around MAP estimate

        eigenvalue_samples = []
        mode_samples = []
        amplitude_samples = []

        # Estimate uncertainty from residuals
        reconstruction = dmd.reconstruct(n_snapshots)
        residual_std = np.std(X - reconstruction)

        for _ in range(self.n_samples):
            # Add noise to eigenvalues
            noise_real = np.random.randn(r) * residual_std * 0.1
            noise_imag = np.random.randn(r) * residual_std * 0.1
            omega_sample = result.continuous_eigenvalues + noise_real + 1j * noise_imag

            # Add noise to modes
            mode_noise = np.random.randn(*result.modes.shape) * residual_std * 0.1
            mode_sample = result.modes + mode_noise

            # Add noise to amplitudes
            amp_noise = np.random.randn(r) * residual_std * 0.1
            amp_sample = result.amplitudes + amp_noise

            eigenvalue_samples.append(np.exp(omega_sample * dt))
            mode_samples.append(mode_sample)
            amplitude_samples.append(amp_sample)

        return {
            "eigenvalue_samples": np.array(eigenvalue_samples),
            "mode_samples": np.array(mode_samples),
            "amplitude_samples": np.array(amplitude_samples),
            "map_estimate": result,
            "eigenvalue_mean": np.mean(eigenvalue_samples, axis=0),
            "eigenvalue_std": np.std(eigenvalue_samples, axis=0),
            "frequency_mean": np.mean([np.imag(np.log(e) / dt) / (2*np.pi)
                                       for e in eigenvalue_samples], axis=0),
            "frequency_std": np.std([np.imag(np.log(e) / dt) / (2*np.pi)
                                     for e in eigenvalue_samples], axis=0),
        }

    def predict_with_uncertainty(self, posterior: Dict,
                                  t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty bounds.

        Returns (mean_prediction, std_prediction).
        """
        predictions = []

        for i in range(len(posterior["eigenvalue_samples"])):
            eigenvalues = posterior["eigenvalue_samples"][i]
            modes = posterior["mode_samples"][i]
            amplitudes = posterior["amplitude_samples"][i]

            omega = np.log(eigenvalues) / posterior["map_estimate"].dt

            pred = np.zeros((len(t), modes.shape[0]), dtype=complex)
            for j, tj in enumerate(t):
                dynamics = amplitudes * np.exp(omega * tj)
                pred[j] = modes @ dynamics

            predictions.append(np.real(pred))

        predictions = np.array(predictions)

        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


class MultiresolutionDMD:
    """
    Multi-resolution DMD for systems with multiple time scales.

    Separates slow and fast dynamics.
    """

    def __init__(self, n_levels: int = 3,
                 rank_per_level: Optional[List[int]] = None):
        """
        Args:
            n_levels: Number of resolution levels
            rank_per_level: Rank at each level
        """
        self.n_levels = n_levels
        self.rank_per_level = rank_per_level or [10] * n_levels

        self._results: List[DMDResult] = []

    def fit(self, X: np.ndarray, dt: float = 1.0) -> List[DMDResult]:
        """
        Fit multi-resolution DMD.

        Decomposes into slow and fast components at each level.
        """
        if X.shape[0] > X.shape[1]:
            pass
        else:
            X = X.T

        residual = X.copy()
        current_dt = dt

        for level in range(self.n_levels):
            # DMD at current resolution
            rank = self.rank_per_level[level]
            dmd = DMD(rank=rank)
            result = dmd.fit(residual, current_dt)
            self._results.append(result)

            # Reconstruct and subtract
            reconstruction = dmd.reconstruct(residual.shape[1])
            residual = residual - reconstruction

            # Downsample for next level (coarser time scale)
            if level < self.n_levels - 1:
                residual = residual[:, ::2]
                current_dt *= 2

        return self._results

    def reconstruct(self, n_steps: int) -> np.ndarray:
        """Reconstruct from all levels."""
        n_dofs = self._results[0].modes.shape[0]
        reconstruction = np.zeros((n_dofs, n_steps))

        for level, result in enumerate(self._results):
            # Number of steps at this level
            level_steps = n_steps // (2 ** level)
            if level_steps == 0:
                continue

            # DMD reconstruction at this level
            dmd = DMD()
            dmd._result = result
            level_recon = dmd.reconstruct(level_steps)

            # Upsample to original resolution
            for i in range(level_steps):
                start = i * (2 ** level)
                end = min((i + 1) * (2 ** level), n_steps)
                reconstruction[:, start:end] += level_recon[:, i:i+1]

        return reconstruction
