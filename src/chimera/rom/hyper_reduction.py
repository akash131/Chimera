"""
Hyper-reduction methods for efficient nonlinear ROM evaluation.

Reduces computational cost of nonlinear terms.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import linalg


@dataclass
class HyperReductionResult:
    """Result of hyper-reduction."""
    sample_indices: np.ndarray  # Selected sample points
    interpolation_matrix: np.ndarray  # For reconstructing full residual
    n_samples: int
    error_estimate: float


class DEIM:
    """
    Discrete Empirical Interpolation Method.

    Approximates nonlinear function by interpolating at few sample points.
    f(u) ≈ U (P^T U)^{-1} P^T f(u) = U c

    where P selects sample points and U are nonlinear basis modes.
    """

    def __init__(self, n_samples: Optional[int] = None,
                 energy_threshold: float = 0.99):
        """
        Args:
            n_samples: Number of sample points
            energy_threshold: Energy threshold for selecting modes
        """
        self.n_samples = n_samples
        self.energy_threshold = energy_threshold

        self._basis: Optional[np.ndarray] = None
        self._sample_indices: Optional[np.ndarray] = None
        self._interpolation_matrix: Optional[np.ndarray] = None

    def fit(self, nonlinear_snapshots: np.ndarray) -> HyperReductionResult:
        """
        Compute DEIM basis and sample points.

        Args:
            nonlinear_snapshots: Snapshots of nonlinear term f(u)
                                 Shape: (n_dofs, n_snapshots)
        """
        if nonlinear_snapshots.shape[0] < nonlinear_snapshots.shape[1]:
            nonlinear_snapshots = nonlinear_snapshots.T

        n_dofs, n_snapshots = nonlinear_snapshots.shape

        # POD of nonlinear snapshots
        U, s, _ = linalg.svd(nonlinear_snapshots, full_matrices=False)

        # Determine number of modes
        total_energy = np.sum(s**2)
        cumulative = np.cumsum(s**2) / total_energy

        if self.n_samples is not None:
            n_modes = min(self.n_samples, len(s), n_dofs)
        else:
            n_modes = np.searchsorted(cumulative, self.energy_threshold) + 1
            n_modes = min(n_modes, len(s), n_dofs)

        self._basis = U[:, :n_modes]

        # DEIM point selection
        sample_indices = self._select_deim_points(self._basis)
        self._sample_indices = sample_indices

        # Interpolation matrix: U (P^T U)^{-1}
        P = np.zeros((n_dofs, n_modes))
        P[sample_indices, np.arange(n_modes)] = 1

        PTU = P.T @ self._basis
        self._interpolation_matrix = self._basis @ linalg.inv(PTU)

        # Error estimate
        residual = nonlinear_snapshots - self._interpolation_matrix @ (
            nonlinear_snapshots[sample_indices, :]
        )
        error = np.linalg.norm(residual) / np.linalg.norm(nonlinear_snapshots)

        return HyperReductionResult(
            sample_indices=sample_indices,
            interpolation_matrix=self._interpolation_matrix,
            n_samples=n_modes,
            error_estimate=error
        )

    def _select_deim_points(self, U: np.ndarray) -> np.ndarray:
        """
        Greedy DEIM point selection algorithm.
        """
        n_dofs, n_modes = U.shape
        sample_indices = np.zeros(n_modes, dtype=int)

        # First point: maximum of first mode
        sample_indices[0] = np.argmax(np.abs(U[:, 0]))

        for j in range(1, n_modes):
            # Solve for interpolation coefficients
            U_j = U[:, :j]
            P_j = np.zeros((n_dofs, j))
            P_j[sample_indices[:j], np.arange(j)] = 1

            c = linalg.solve(P_j.T @ U_j, P_j.T @ U[:, j])

            # Residual
            r = U[:, j] - U_j @ c

            # New point: maximum residual
            sample_indices[j] = np.argmax(np.abs(r))

        return sample_indices

    def approximate(self, f_samples: np.ndarray) -> np.ndarray:
        """
        Approximate full nonlinear vector from samples.

        Args:
            f_samples: Values at sample points (n_samples,)

        Returns:
            Approximation of full vector (n_dofs,)
        """
        return self._interpolation_matrix @ f_samples


class QDEIM:
    """
    Q-DEIM: DEIM with QR factorization for improved stability.

    Uses pivoted QR for more stable point selection.
    """

    def __init__(self, n_samples: Optional[int] = None,
                 oversampling: int = 10):
        """
        Args:
            n_samples: Number of sample points
            oversampling: Extra samples for stability
        """
        self.n_samples = n_samples
        self.oversampling = oversampling

        self._basis: Optional[np.ndarray] = None
        self._sample_indices: Optional[np.ndarray] = None
        self._interpolation_matrix: Optional[np.ndarray] = None

    def fit(self, nonlinear_snapshots: np.ndarray) -> HyperReductionResult:
        """Compute Q-DEIM approximation."""
        if nonlinear_snapshots.shape[0] < nonlinear_snapshots.shape[1]:
            nonlinear_snapshots = nonlinear_snapshots.T

        n_dofs, n_snapshots = nonlinear_snapshots.shape

        # POD
        U, s, _ = linalg.svd(nonlinear_snapshots, full_matrices=False)

        if self.n_samples is not None:
            n_modes = min(self.n_samples, len(s), n_dofs)
        else:
            n_modes = min(50, len(s), n_dofs)

        self._basis = U[:, :n_modes]

        # Q-DEIM: Use pivoted QR
        n_total = min(n_modes + self.oversampling, n_dofs)
        sample_indices = self._qr_point_selection(self._basis, n_total)
        self._sample_indices = sample_indices[:n_modes]

        # Least squares interpolation
        U_sampled = self._basis[self._sample_indices, :]
        self._interpolation_matrix = self._basis @ linalg.pinv(U_sampled)

        # Error estimate
        residual = nonlinear_snapshots - self._interpolation_matrix @ (
            nonlinear_snapshots[self._sample_indices, :]
        )
        error = np.linalg.norm(residual) / np.linalg.norm(nonlinear_snapshots)

        return HyperReductionResult(
            sample_indices=self._sample_indices,
            interpolation_matrix=self._interpolation_matrix,
            n_samples=n_modes,
            error_estimate=error
        )

    def _qr_point_selection(self, U: np.ndarray, n_points: int) -> np.ndarray:
        """Select points using column-pivoted QR of U^T."""
        # QR with column pivoting
        _, _, perm = linalg.qr(U.T, pivoting=True)
        return perm[:n_points]

    def approximate(self, f_samples: np.ndarray) -> np.ndarray:
        """Approximate full vector from samples."""
        return self._interpolation_matrix @ f_samples


class GNAT:
    """
    Gauss-Newton with Approximated Tensors.

    Hyper-reduction for implicit time stepping.
    """

    def __init__(self, n_samples: int,
                 n_basis_modes: int):
        """
        Args:
            n_samples: Number of sample points
            n_basis_modes: Number of POD modes for state
        """
        self.n_samples = n_samples
        self.n_basis_modes = n_basis_modes

        self._sample_indices: Optional[np.ndarray] = None
        self._Phi: Optional[np.ndarray] = None  # State basis
        self._Theta: Optional[np.ndarray] = None  # Residual basis

    def fit(self, state_snapshots: np.ndarray,
            residual_snapshots: np.ndarray) -> HyperReductionResult:
        """
        Compute GNAT approximation.

        Args:
            state_snapshots: State snapshots (n_dofs, n_snapshots)
            residual_snapshots: Residual R(u) snapshots (n_dofs, n_snapshots)
        """
        if state_snapshots.shape[0] < state_snapshots.shape[1]:
            state_snapshots = state_snapshots.T
        if residual_snapshots.shape[0] < residual_snapshots.shape[1]:
            residual_snapshots = residual_snapshots.T

        n_dofs = state_snapshots.shape[0]

        # POD of states
        U_state, _, _ = linalg.svd(state_snapshots, full_matrices=False)
        self._Phi = U_state[:, :self.n_basis_modes]

        # POD of residuals
        U_res, _, _ = linalg.svd(residual_snapshots, full_matrices=False)
        n_res_modes = min(self.n_samples, U_res.shape[1])
        self._Theta = U_res[:, :n_res_modes]

        # Sample point selection
        self._sample_indices = self._select_sample_points()

        # Compute reduced matrices
        Phi_s = self._Phi[self._sample_indices, :]
        Theta_s = self._Theta[self._sample_indices, :]

        # Pseudo-inverse for reconstruction
        self._Theta_pinv = linalg.pinv(Theta_s)

        # Error estimate from training data
        residual_approx = self._Theta @ (
            self._Theta_pinv @ residual_snapshots[self._sample_indices, :]
        )
        error = np.linalg.norm(residual_snapshots - residual_approx) / \
                np.linalg.norm(residual_snapshots)

        return HyperReductionResult(
            sample_indices=self._sample_indices,
            interpolation_matrix=self._Theta @ self._Theta_pinv,
            n_samples=len(self._sample_indices),
            error_estimate=error
        )

    def _select_sample_points(self) -> np.ndarray:
        """Select sample points based on both Phi and Theta."""
        # Combined matrix
        combined = np.hstack([self._Phi, self._Theta])

        # Use Q-DEIM style selection
        _, _, perm = linalg.qr(combined.T, pivoting=True)

        return perm[:self.n_samples]

    def compute_reduced_residual(self, residual_samples: np.ndarray) -> np.ndarray:
        """
        Compute reduced residual from sample evaluations.
        """
        # Approximate full residual
        residual_full = self._Theta @ (self._Theta_pinv @ residual_samples)

        # Project onto state basis
        return self._Phi.T @ residual_full

    def compute_reduced_jacobian(self, jacobian_samples: np.ndarray) -> np.ndarray:
        """
        Compute reduced Jacobian from sample rows.

        Args:
            jacobian_samples: J[sample_indices, :] @ Phi (n_samples, n_basis)
        """
        return self._Theta_pinv.T @ self._Theta.T @ self._Phi @ jacobian_samples


class EmpiricalCubature:
    """
    Empirical Cubature Method (ECM).

    Finds optimal quadrature points and weights for integration.
    """

    def __init__(self, n_points: int,
                 positive_weights: bool = True):
        """
        Args:
            n_points: Number of cubature points
            positive_weights: Enforce positive weights
        """
        self.n_points = n_points
        self.positive_weights = positive_weights

        self._points: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def fit(self, integration_snapshots: np.ndarray,
            element_volumes: np.ndarray) -> HyperReductionResult:
        """
        Compute cubature points and weights.

        Args:
            integration_snapshots: Integrands at all elements
                                   (n_elements, n_snapshots)
            element_volumes: Volume of each element
        """
        if integration_snapshots.shape[0] < integration_snapshots.shape[1]:
            integration_snapshots = integration_snapshots.T

        n_elements, n_snapshots = integration_snapshots.shape

        # Weight integrands by element volumes
        weighted = integration_snapshots * element_volumes[:, np.newaxis]

        # Target: full integration
        target = np.sum(weighted, axis=0)

        # Find subset that approximates integration
        if self.positive_weights:
            points, weights = self._nonnegative_cubature(
                weighted, target
            )
        else:
            points, weights = self._least_squares_cubature(
                weighted, target
            )

        self._points = points
        self._weights = weights

        # Error estimate
        approx = np.sum(weighted[points, :] * weights[:, np.newaxis], axis=0)
        error = np.linalg.norm(target - approx) / np.linalg.norm(target)

        return HyperReductionResult(
            sample_indices=points,
            interpolation_matrix=weights.reshape(-1, 1),
            n_samples=len(points),
            error_estimate=error
        )

    def _least_squares_cubature(self, weighted: np.ndarray,
                                 target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find cubature by least squares."""
        n_elements = weighted.shape[0]

        # Greedy selection
        points = []
        remaining = list(range(n_elements))

        for _ in range(self.n_points):
            best_point = None
            best_error = np.inf

            for p in remaining:
                test_points = points + [p]
                W = weighted[test_points, :].T
                w, _, _, _ = linalg.lstsq(W, target)

                error = np.linalg.norm(W @ w - target)
                if error < best_error:
                    best_error = error
                    best_point = p

            points.append(best_point)
            remaining.remove(best_point)

        # Final weights
        W = weighted[points, :].T
        weights, _, _, _ = linalg.lstsq(W, target)

        return np.array(points), weights

    def _nonnegative_cubature(self, weighted: np.ndarray,
                               target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find cubature with non-negative weights."""
        from scipy.optimize import nnls

        n_elements = weighted.shape[0]

        # Use NNLS to find sparse non-negative weights
        W = weighted.T  # (n_snapshots, n_elements)

        # Solve for all elements first
        weights_all, _ = nnls(W, target)

        # Select points with significant weights
        threshold = 1e-6 * np.max(weights_all)
        significant = np.where(weights_all > threshold)[0]

        if len(significant) > self.n_points:
            # Select top weights
            idx = np.argsort(weights_all)[::-1][:self.n_points]
            significant = idx

        # Recompute weights for selected points
        W_selected = W[:, significant]
        weights, _ = nnls(W_selected, target)

        return significant, weights

    def integrate(self, integrand: np.ndarray) -> float:
        """
        Approximate integral using cubature.

        Args:
            integrand: Values at all elements
        """
        return np.sum(integrand[self._points] * self._weights)


class MissingPointEstimation:
    """
    Missing Point Estimation for hyper-reduction.

    Alternative to DEIM that uses least-squares fitting.
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

        self._sample_indices: Optional[np.ndarray] = None
        self._reconstruction_matrix: Optional[np.ndarray] = None

    def fit(self, snapshots: np.ndarray) -> HyperReductionResult:
        """Compute MPE sample points."""
        if snapshots.shape[0] < snapshots.shape[1]:
            snapshots = snapshots.T

        n_dofs, n_snapshots = snapshots.shape

        # Use leverage scores for sampling
        U, _, _ = linalg.svd(snapshots, full_matrices=False)
        k = min(self.n_samples, U.shape[1])
        U_k = U[:, :k]

        # Leverage scores
        leverage = np.sum(U_k ** 2, axis=1)

        # Sample proportional to leverage
        probs = leverage / np.sum(leverage)
        sample_indices = np.random.choice(
            n_dofs, size=self.n_samples, replace=False, p=probs
        )
        sample_indices = np.sort(sample_indices)

        self._sample_indices = sample_indices

        # Reconstruction matrix
        A_sampled = snapshots[sample_indices, :]
        self._reconstruction_matrix = linalg.lstsq(A_sampled.T, snapshots.T)[0].T

        # Error estimate
        reconstructed = self._reconstruction_matrix @ A_sampled
        error = np.linalg.norm(snapshots - reconstructed) / np.linalg.norm(snapshots)

        return HyperReductionResult(
            sample_indices=sample_indices,
            interpolation_matrix=self._reconstruction_matrix,
            n_samples=self.n_samples,
            error_estimate=error
        )

    def reconstruct(self, samples: np.ndarray) -> np.ndarray:
        """Reconstruct full vector from samples."""
        return self._reconstruction_matrix @ samples
