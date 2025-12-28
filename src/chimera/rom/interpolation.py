"""
Interpolation methods for parametric reduced-order models.

Interpolate between parameter-specific ROMs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg
from scipy.interpolate import RBFInterpolator as ScipyRBF


@dataclass
class ParametricBasis:
    """Basis at a specific parameter value."""
    parameter: np.ndarray
    modes: np.ndarray
    mean: np.ndarray
    n_modes: int


class RBFInterpolator:
    """
    Radial Basis Function interpolation for parametric ROMs.

    Simple but effective for low-dimensional parameter spaces.
    """

    def __init__(self, kernel: str = "thin_plate_spline",
                 smoothing: float = 0.0):
        """
        Args:
            kernel: RBF kernel type
            smoothing: Regularization parameter
        """
        self.kernel = kernel
        self.smoothing = smoothing

        self._parameters: List[np.ndarray] = []
        self._bases: List[ParametricBasis] = []
        self._rbf: Optional[ScipyRBF] = None

    def add_basis(self, parameter: np.ndarray, modes: np.ndarray,
                  mean: np.ndarray):
        """Add basis for a parameter value."""
        self._parameters.append(parameter)
        self._bases.append(ParametricBasis(
            parameter=parameter,
            modes=modes,
            mean=mean,
            n_modes=modes.shape[1]
        ))

    def fit(self):
        """Fit RBF interpolator to stored bases."""
        if len(self._bases) < 2:
            return

        params = np.array(self._parameters)
        n_dofs = self._bases[0].modes.shape[0]
        n_modes = self._bases[0].n_modes

        # Flatten modes for interpolation
        modes_flat = np.array([b.modes.flatten() for b in self._bases])
        means = np.array([b.mean for b in self._bases])

        # Fit RBF for modes
        self._rbf_modes = ScipyRBF(params, modes_flat,
                                    kernel=self.kernel,
                                    smoothing=self.smoothing)

        # Fit RBF for means
        self._rbf_mean = ScipyRBF(params, means,
                                   kernel=self.kernel,
                                   smoothing=self.smoothing)

        self._n_dofs = n_dofs
        self._n_modes = n_modes

    def interpolate(self, parameter: np.ndarray) -> ParametricBasis:
        """Interpolate basis at new parameter value."""
        parameter = np.atleast_2d(parameter)

        # Interpolate modes
        modes_flat = self._rbf_modes(parameter)[0]
        modes = modes_flat.reshape(self._n_dofs, self._n_modes)

        # Orthonormalize
        modes, _ = np.linalg.qr(modes)

        # Interpolate mean
        mean = self._rbf_mean(parameter)[0]

        return ParametricBasis(
            parameter=parameter.flatten(),
            modes=modes,
            mean=mean,
            n_modes=self._n_modes
        )


class GrassmannInterpolator:
    """
    Interpolation on Grassmann manifold.

    Properly handles the geometry of subspaces.
    """

    def __init__(self, method: str = "geodesic"):
        """
        Args:
            method: "geodesic" or "karcher" (Karcher mean based)
        """
        self.method = method
        self._parameters: List[np.ndarray] = []
        self._bases: List[ParametricBasis] = []

    def add_basis(self, parameter: np.ndarray, modes: np.ndarray,
                  mean: np.ndarray):
        """Add basis for a parameter value."""
        self._parameters.append(parameter)
        self._bases.append(ParametricBasis(
            parameter=parameter,
            modes=modes,
            mean=mean,
            n_modes=modes.shape[1]
        ))

    def log_map(self, Phi_base: np.ndarray, Phi: np.ndarray) -> np.ndarray:
        """
        Logarithmic map from Phi_base to Phi.

        Maps point on Grassmann manifold to tangent space.
        """
        # Phi_base^T Phi = U S V^T
        M = Phi_base.T @ Phi
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Clamp singular values
        S = np.clip(S, -1, 1)
        angles = np.arccos(S)

        # Tangent vector
        # Gamma = (I - Phi_base Phi_base^T) Phi V S^{-1} arccos(S)
        perpendicular = Phi - Phi_base @ M
        tangent = perpendicular @ Vt.T @ np.diag(angles / (np.sin(angles) + 1e-10))

        return tangent

    def exp_map(self, Phi_base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """
        Exponential map from Phi_base along tangent.

        Maps tangent vector back to Grassmann manifold.
        """
        # SVD of tangent
        U, S, Vt = np.linalg.svd(tangent, full_matrices=False)

        # Exponential: Phi_base V cos(S) V^T + U sin(S) V^T
        Phi_new = Phi_base @ Vt.T @ np.diag(np.cos(S)) @ Vt + U @ np.diag(np.sin(S)) @ Vt

        # Orthonormalize for numerical stability
        Phi_new, _ = np.linalg.qr(Phi_new)

        return Phi_new

    def geodesic_distance(self, Phi1: np.ndarray, Phi2: np.ndarray) -> float:
        """Geodesic distance on Grassmann manifold."""
        M = Phi1.T @ Phi2
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        S = np.clip(S, -1, 1)
        angles = np.arccos(S)
        return np.sqrt(np.sum(angles**2))

    def interpolate(self, parameter: np.ndarray) -> ParametricBasis:
        """
        Interpolate basis at new parameter value.
        """
        parameter = np.atleast_1d(parameter)

        if len(self._bases) == 1:
            return self._bases[0]

        # Find closest basis as reference
        distances = np.array([np.linalg.norm(b.parameter - parameter)
                              for b in self._bases])
        ref_idx = np.argmin(distances)
        Phi_ref = self._bases[ref_idx].modes

        # Compute weights (inverse distance)
        weights = 1.0 / (distances + 1e-8)
        weights[ref_idx] = 0  # Don't include reference in weighted average
        weights /= np.sum(weights)

        # Weighted average in tangent space
        tangent_avg = np.zeros_like(Phi_ref)

        for i, basis in enumerate(self._bases):
            if i == ref_idx:
                continue

            # Log map to tangent space
            tangent = self.log_map(Phi_ref, basis.modes)
            tangent_avg += weights[i] * tangent

        # Exp map back to manifold
        Phi_interp = self.exp_map(Phi_ref, tangent_avg)

        # Interpolate mean using standard RBF
        params = np.array(self._parameters)
        means = np.array([b.mean for b in self._bases])

        weights_mean = 1.0 / (distances + 1e-8)
        weights_mean /= np.sum(weights_mean)
        mean_interp = np.sum(means * weights_mean[:, np.newaxis], axis=0)

        return ParametricBasis(
            parameter=parameter,
            modes=Phi_interp,
            mean=mean_interp,
            n_modes=Phi_interp.shape[1]
        )

    def karcher_mean(self, bases: List[np.ndarray],
                     weights: Optional[np.ndarray] = None,
                     max_iter: int = 20,
                     tol: float = 1e-6) -> np.ndarray:
        """
        Compute Karcher mean of multiple bases.

        Weighted Frechet mean on Grassmann manifold.
        """
        n_bases = len(bases)

        if weights is None:
            weights = np.ones(n_bases) / n_bases

        # Initialize with first basis
        mean = bases[0].copy()

        for iteration in range(max_iter):
            # Compute weighted sum of tangent vectors
            tangent_sum = np.zeros_like(mean)

            for i, basis in enumerate(bases):
                tangent = self.log_map(mean, basis)
                tangent_sum += weights[i] * tangent

            # Check convergence
            norm = np.linalg.norm(tangent_sum)
            if norm < tol:
                break

            # Update mean
            mean = self.exp_map(mean, tangent_sum)

        return mean


class ManifoldInterpolator:
    """
    General manifold interpolation using learned coordinates.

    Uses autoencoder to learn manifold structure.
    """

    def __init__(self, latent_dim: int = 5):
        """
        Args:
            latent_dim: Dimension of learned manifold
        """
        self.latent_dim = latent_dim

        self._parameters: List[np.ndarray] = []
        self._bases: List[ParametricBasis] = []
        self._encoder_W: Optional[np.ndarray] = None
        self._decoder_W: Optional[np.ndarray] = None

    def add_basis(self, parameter: np.ndarray, modes: np.ndarray,
                  mean: np.ndarray):
        """Add basis for a parameter value."""
        self._parameters.append(parameter)
        self._bases.append(ParametricBasis(
            parameter=parameter,
            modes=modes,
            mean=mean,
            n_modes=modes.shape[1]
        ))

    def fit(self, n_epochs: int = 100, lr: float = 0.01):
        """
        Learn manifold coordinates via autoencoder.
        """
        n_bases = len(self._bases)
        n_dofs = self._bases[0].modes.shape[0]
        n_modes = self._bases[0].n_modes

        # Stack all modes
        modes_flat = np.array([b.modes.flatten() for b in self._bases])
        input_dim = modes_flat.shape[1]

        # Simple linear autoencoder
        self._encoder_W = np.random.randn(input_dim, self.latent_dim) * 0.1
        self._decoder_W = np.random.randn(self.latent_dim, input_dim) * 0.1

        for epoch in range(n_epochs):
            # Forward
            latent = modes_flat @ self._encoder_W
            reconstructed = latent @ self._decoder_W

            # Loss
            loss = np.mean((modes_flat - reconstructed) ** 2)

            # Backward
            error = reconstructed - modes_flat
            grad_decoder = latent.T @ error / n_bases
            grad_encoder = modes_flat.T @ (error @ self._decoder_W.T) / n_bases

            # Update
            self._encoder_W -= lr * grad_encoder
            self._decoder_W -= lr * grad_decoder

        self._n_dofs = n_dofs
        self._n_modes = n_modes

        # Compute latent coordinates
        self._latent_coords = modes_flat @ self._encoder_W

    def interpolate(self, parameter: np.ndarray) -> ParametricBasis:
        """Interpolate in learned manifold space."""
        parameter = np.atleast_1d(parameter)

        params = np.array(self._parameters)

        # Inverse distance weighting in parameter space
        distances = np.linalg.norm(params - parameter, axis=1)
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights)

        # Interpolate in latent space
        latent_interp = np.sum(self._latent_coords * weights[:, np.newaxis], axis=0)

        # Decode
        modes_flat = latent_interp @ self._decoder_W
        modes = modes_flat.reshape(self._n_dofs, self._n_modes)

        # Orthonormalize
        modes, _ = np.linalg.qr(modes)

        # Interpolate mean
        means = np.array([b.mean for b in self._bases])
        mean = np.sum(means * weights[:, np.newaxis], axis=0)

        return ParametricBasis(
            parameter=parameter,
            modes=modes,
            mean=mean,
            n_modes=self._n_modes
        )


class SplineInterpolator:
    """
    Spline interpolation for 1D parameter spaces.

    Provides smooth interpolation with continuity guarantees.
    """

    def __init__(self, order: int = 3):
        """
        Args:
            order: Spline order (3 = cubic)
        """
        self.order = order
        self._parameters: List[float] = []
        self._bases: List[ParametricBasis] = []

    def add_basis(self, parameter: float, modes: np.ndarray,
                  mean: np.ndarray):
        """Add basis for a parameter value."""
        self._parameters.append(parameter)
        self._bases.append(ParametricBasis(
            parameter=np.array([parameter]),
            modes=modes,
            mean=mean,
            n_modes=modes.shape[1]
        ))

    def fit(self):
        """Fit splines to mode coefficients."""
        from scipy.interpolate import make_interp_spline

        # Sort by parameter
        idx = np.argsort(self._parameters)
        params = np.array(self._parameters)[idx]

        n_dofs = self._bases[0].modes.shape[0]
        n_modes = self._bases[0].n_modes

        # Splines for each mode coefficient
        modes_array = np.array([self._bases[i].modes for i in idx])

        self._mode_splines = []
        for i in range(n_dofs):
            for j in range(n_modes):
                values = modes_array[:, i, j]
                if len(params) > self.order:
                    spline = make_interp_spline(params, values, k=self.order)
                else:
                    spline = make_interp_spline(params, values, k=len(params)-1)
                self._mode_splines.append(spline)

        # Spline for mean
        means = np.array([self._bases[i].mean for i in idx])
        self._mean_splines = []
        for i in range(n_dofs):
            if len(params) > self.order:
                spline = make_interp_spline(params, means[:, i], k=self.order)
            else:
                spline = make_interp_spline(params, means[:, i], k=len(params)-1)
            self._mean_splines.append(spline)

        self._n_dofs = n_dofs
        self._n_modes = n_modes
        self._param_range = (params.min(), params.max())

    def interpolate(self, parameter: float) -> ParametricBasis:
        """Interpolate at parameter value."""
        # Clamp to range
        parameter = np.clip(parameter, *self._param_range)

        # Evaluate splines
        modes = np.zeros((self._n_dofs, self._n_modes))
        k = 0
        for i in range(self._n_dofs):
            for j in range(self._n_modes):
                modes[i, j] = self._mode_splines[k](parameter)
                k += 1

        mean = np.array([s(parameter) for s in self._mean_splines])

        # Orthonormalize
        modes, _ = np.linalg.qr(modes)

        return ParametricBasis(
            parameter=np.array([parameter]),
            modes=modes,
            mean=mean,
            n_modes=self._n_modes
        )


class NeuralInterpolator:
    """
    Neural network interpolation for high-dimensional parameters.

    Learns mapping from parameters to basis coefficients.
    """

    def __init__(self, param_dim: int,
                 hidden_dims: List[int] = None):
        """
        Args:
            param_dim: Dimension of parameter space
            hidden_dims: Hidden layer dimensions
        """
        self.param_dim = param_dim
        self.hidden_dims = hidden_dims or [64, 64]

        self._parameters: List[np.ndarray] = []
        self._bases: List[ParametricBasis] = []
        self._layers: List = []

    def add_basis(self, parameter: np.ndarray, modes: np.ndarray,
                  mean: np.ndarray):
        """Add basis for a parameter value."""
        self._parameters.append(parameter)
        self._bases.append(ParametricBasis(
            parameter=parameter,
            modes=modes,
            mean=mean,
            n_modes=modes.shape[1]
        ))

    def fit(self, n_epochs: int = 1000, lr: float = 0.001):
        """Train neural network interpolator."""
        n_bases = len(self._bases)
        n_dofs = self._bases[0].modes.shape[0]
        n_modes = self._bases[0].n_modes

        params = np.array(self._parameters)
        modes_flat = np.array([b.modes.flatten() for b in self._bases])
        means = np.array([b.mean for b in self._bases])

        output_dim = modes_flat.shape[1] + n_dofs  # modes + mean

        # Build network
        dims = [self.param_dim] + self.hidden_dims + [output_dim]

        self._weights = []
        self._biases = []

        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._weights.append(W)
            self._biases.append(b)

        # Training
        for epoch in range(n_epochs):
            # Forward
            h = params
            activations = [h]

            for i, (W, b) in enumerate(zip(self._weights, self._biases)):
                h = h @ W + b
                if i < len(self._weights) - 1:
                    h = np.maximum(0, h)  # ReLU
                activations.append(h)

            # Split output
            pred_modes = h[:, :modes_flat.shape[1]]
            pred_means = h[:, modes_flat.shape[1]:]

            # Loss
            loss = np.mean((modes_flat - pred_modes)**2) + np.mean((means - pred_means)**2)

            # Backward
            grad = np.hstack([2 * (pred_modes - modes_flat) / n_bases,
                              2 * (pred_means - means) / n_bases])

            for i in range(len(self._weights) - 1, -1, -1):
                # Gradient w.r.t. weights
                grad_W = activations[i].T @ grad
                grad_b = np.mean(grad, axis=0)

                # Gradient w.r.t. input
                grad = grad @ self._weights[i].T

                # ReLU gradient
                if i > 0:
                    grad = grad * (activations[i] > 0)

                # Update
                self._weights[i] -= lr * grad_W
                self._biases[i] -= lr * grad_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        self._n_dofs = n_dofs
        self._n_modes = n_modes
        self._modes_dim = modes_flat.shape[1]

    def interpolate(self, parameter: np.ndarray) -> ParametricBasis:
        """Interpolate at new parameter value."""
        parameter = np.atleast_2d(parameter)

        # Forward pass
        h = parameter
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            h = h @ W + b
            if i < len(self._weights) - 1:
                h = np.maximum(0, h)

        h = h[0]

        # Split output
        modes_flat = h[:self._modes_dim]
        mean = h[self._modes_dim:]

        # Reshape and orthonormalize
        modes = modes_flat.reshape(self._n_dofs, self._n_modes)
        modes, _ = np.linalg.qr(modes)

        return ParametricBasis(
            parameter=parameter.flatten(),
            modes=modes,
            mean=mean,
            n_modes=self._n_modes
        )
