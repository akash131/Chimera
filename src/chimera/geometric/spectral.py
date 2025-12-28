"""
Spectral methods for geometric deep learning.

Use mesh Laplacian eigenvectors as basis for convolutions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass
class SpectralBasis:
    """Spectral basis from mesh Laplacian."""
    eigenvalues: np.ndarray  # (n_modes,)
    eigenvectors: np.ndarray  # (n_nodes, n_modes)
    laplacian: sparse.csr_matrix


class LaplacianEigenmaps:
    """
    Compute Laplacian eigenmaps for mesh.

    Eigenvectors of graph Laplacian provide smooth basis functions
    on the mesh, analogous to Fourier basis on regular grids.
    """

    def __init__(self, n_modes: int = 50):
        self.n_modes = n_modes
        self._basis: Optional[SpectralBasis] = None

    def compute(self, nodes: np.ndarray, edges: np.ndarray,
                edge_weights: Optional[np.ndarray] = None) -> SpectralBasis:
        """
        Compute spectral basis.

        Args:
            nodes: Node positions (n_nodes, dim)
            edges: Edge list (2, n_edges)
            edge_weights: Optional edge weights

        Returns:
            SpectralBasis with eigenvalues and eigenvectors
        """
        n_nodes = len(nodes)

        # Build adjacency matrix
        if edge_weights is None:
            # Use inverse distance weighting
            edge_lengths = np.linalg.norm(
                nodes[edges[1]] - nodes[edges[0]], axis=1
            )
            edge_weights = 1.0 / (edge_lengths + 1e-8)

        # Symmetric adjacency
        W = sparse.csr_matrix(
            (edge_weights, (edges[0], edges[1])),
            shape=(n_nodes, n_nodes)
        )
        W = W + W.T

        # Degree matrix
        D = sparse.diags(np.array(W.sum(axis=1)).flatten())

        # Graph Laplacian: L = D - W
        L = D - W

        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.array(D.diagonal()) + 1e-8))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Compute smallest eigenvalues (smoothest modes)
        n_modes = min(self.n_modes, n_nodes - 2)
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_modes, which='SM')

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self._basis = SpectralBasis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            laplacian=L_norm
        )

        return self._basis

    def project(self, signal: np.ndarray) -> np.ndarray:
        """Project signal onto spectral basis."""
        if self._basis is None:
            raise ValueError("Call compute() first")
        return self._basis.eigenvectors.T @ signal

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct signal from spectral coefficients."""
        if self._basis is None:
            raise ValueError("Call compute() first")
        return self._basis.eigenvectors @ coefficients


class SpectralConv:
    """
    Spectral convolution on mesh.

    Perform convolution in spectral domain:
    1. Transform to spectral domain (eigenvector projection)
    2. Multiply by learnable filter
    3. Transform back to spatial domain

    Analogous to FFT convolution on regular grids.
    """

    def __init__(self, in_features: int, out_features: int, n_modes: int = 50):
        self.in_features = in_features
        self.out_features = out_features
        self.n_modes = n_modes

        # Spectral filter (learnable)
        self.filter_weights = np.random.randn(n_modes, in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

        self._eigenmaps: Optional[LaplacianEigenmaps] = None
        self._basis: Optional[SpectralBasis] = None

    def setup(self, nodes: np.ndarray, edges: np.ndarray):
        """Precompute spectral basis for mesh."""
        self._eigenmaps = LaplacianEigenmaps(self.n_modes)
        self._basis = self._eigenmaps.compute(nodes, edges)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            features: Node features (n_nodes, in_features)

        Returns:
            Filtered features (n_nodes, out_features)
        """
        if self._basis is None:
            raise ValueError("Call setup() with mesh first")

        n_nodes = len(features)

        # Project to spectral domain
        spectral = self._basis.eigenvectors.T @ features  # (n_modes, in_features)

        # Apply spectral filter
        filtered = np.zeros((self.n_modes, self.out_features))
        for k in range(self.n_modes):
            filtered[k] = spectral[k] @ self.filter_weights[k]

        # Back to spatial domain
        output = self._basis.eigenvectors @ filtered  # (n_nodes, out_features)

        return output + self.bias


class ChebyshevConv:
    """
    Chebyshev spectral convolution.

    Approximate spectral filter with Chebyshev polynomials.
    Avoids explicit eigendecomposition - more scalable.

    Filter: g(L) ≈ sum_k θ_k T_k(L̃)
    where T_k are Chebyshev polynomials
    """

    def __init__(self, in_features: int, out_features: int, K: int = 3):
        self.in_features = in_features
        self.out_features = out_features
        self.K = K  # Polynomial order

        # Chebyshev coefficients (learnable)
        self.theta = np.random.randn(K, in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

        self._L_scaled: Optional[sparse.csr_matrix] = None
        self._lambda_max: float = 2.0

    def setup(self, nodes: np.ndarray, edges: np.ndarray,
              edge_weights: Optional[np.ndarray] = None):
        """Precompute scaled Laplacian."""
        n_nodes = len(nodes)

        # Build adjacency
        if edge_weights is None:
            edge_lengths = np.linalg.norm(
                nodes[edges[1]] - nodes[edges[0]], axis=1
            )
            edge_weights = 1.0 / (edge_lengths + 1e-8)

        W = sparse.csr_matrix(
            (edge_weights, (edges[0], edges[1])),
            shape=(n_nodes, n_nodes)
        )
        W = W + W.T

        D = sparse.diags(np.array(W.sum(axis=1)).flatten())
        L = D - W

        # Normalized Laplacian
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.array(D.diagonal()) + 1e-8))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        # Estimate largest eigenvalue
        try:
            lambda_max = eigsh(L_norm, k=1, which='LM', return_eigenvectors=False)[0]
        except:
            lambda_max = 2.0

        self._lambda_max = lambda_max

        # Scale Laplacian to [-1, 1]: L̃ = 2L/λ_max - I
        I = sparse.eye(n_nodes)
        self._L_scaled = 2 * L_norm / lambda_max - I

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass using Chebyshev recursion.
        """
        if self._L_scaled is None:
            raise ValueError("Call setup() with mesh first")

        n_nodes = len(features)

        # Chebyshev recursion: T_0(x) = 1, T_1(x) = x, T_k(x) = 2x T_{k-1} - T_{k-2}
        T = [None] * self.K
        T[0] = features  # T_0 * x = x

        if self.K > 1:
            T[1] = self._L_scaled @ features  # T_1 * x = L̃ * x

        for k in range(2, self.K):
            T[k] = 2 * self._L_scaled @ T[k-1] - T[k-2]

        # Apply filter: sum_k θ_k T_k(L̃) x
        output = np.zeros((n_nodes, self.out_features))
        for k in range(self.K):
            output += T[k] @ self.theta[k]

        return output + self.bias


class WaveletConv:
    """
    Wavelet convolution on mesh.

    Use spectral graph wavelets for multi-scale analysis.
    Wavelets are localized in both space and frequency.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_scales: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_scales = n_scales

        # Scale-specific weights
        self.scale_weights = np.random.randn(
            n_scales, in_features, out_features
        ) * 0.1
        self.bias = np.zeros(out_features)

        self._basis: Optional[SpectralBasis] = None

    def setup(self, nodes: np.ndarray, edges: np.ndarray):
        """Compute spectral basis."""
        eigenmaps = LaplacianEigenmaps(n_modes=min(100, len(nodes) - 2))
        self._basis = eigenmaps.compute(nodes, edges)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass with wavelet decomposition.
        """
        if self._basis is None:
            raise ValueError("Call setup() with mesh first")

        n_nodes = len(features)
        eigenvalues = self._basis.eigenvalues
        eigenvectors = self._basis.eigenvectors

        output = np.zeros((n_nodes, self.out_features))

        # Mexican hat wavelet in spectral domain
        for s, scale in enumerate(np.logspace(-1, 1, self.n_scales)):
            # Wavelet kernel: g(λ) = λ * exp(-λ * scale)
            wavelet_kernel = eigenvalues * np.exp(-eigenvalues * scale)

            # Apply wavelet transform
            spectral = eigenvectors.T @ features  # (n_modes, in_features)
            wavelet_coeffs = wavelet_kernel[:, np.newaxis] * spectral

            # Back to spatial
            spatial = eigenvectors @ wavelet_coeffs  # (n_nodes, in_features)

            # Apply scale-specific weights
            output += spatial @ self.scale_weights[s]

        return output + self.bias


class HeatKernel:
    """
    Heat kernel for diffusion on mesh.

    Heat kernel K(t) = exp(-t L) provides smooth diffusion.
    Can be used for feature propagation and smoothing.
    """

    def __init__(self, n_modes: int = 50):
        self.n_modes = n_modes
        self._basis: Optional[SpectralBasis] = None

    def setup(self, nodes: np.ndarray, edges: np.ndarray):
        """Compute spectral basis."""
        eigenmaps = LaplacianEigenmaps(self.n_modes)
        self._basis = eigenmaps.compute(nodes, edges)

    def diffuse(self, signal: np.ndarray, t: float) -> np.ndarray:
        """
        Apply heat diffusion for time t.

        Args:
            signal: Initial signal (n_nodes,) or (n_nodes, n_features)
            t: Diffusion time

        Returns:
            Diffused signal
        """
        if self._basis is None:
            raise ValueError("Call setup() with mesh first")

        eigenvalues = self._basis.eigenvalues
        eigenvectors = self._basis.eigenvectors

        # Project to spectral domain
        spectral = eigenvectors.T @ signal

        # Apply heat kernel: exp(-t λ)
        heat_kernel = np.exp(-t * eigenvalues)

        if len(signal.shape) == 1:
            filtered = heat_kernel * spectral
        else:
            filtered = heat_kernel[:, np.newaxis] * spectral

        # Back to spatial
        return eigenvectors @ filtered

    def multiscale_signatures(self, n_times: int = 10) -> np.ndarray:
        """
        Compute heat kernel signatures (HKS).

        HKS provides intrinsic shape descriptor invariant to
        isometric deformations.
        """
        if self._basis is None:
            raise ValueError("Call setup() with mesh first")

        n_nodes = self._basis.eigenvectors.shape[0]
        eigenvalues = self._basis.eigenvalues
        eigenvectors = self._basis.eigenvectors

        # Time scales
        t_min = 4 * np.log(10) / eigenvalues[-1]
        t_max = 4 * np.log(10) / max(eigenvalues[1], 1e-8)
        times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

        # HKS: diagonal of heat kernel
        hks = np.zeros((n_nodes, n_times))
        for i, t in enumerate(times):
            heat_kernel = np.exp(-t * eigenvalues)
            hks[:, i] = np.sum(eigenvectors**2 * heat_kernel, axis=1)

        return hks
