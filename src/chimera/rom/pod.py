"""
Proper Orthogonal Decomposition (POD).

Extract dominant modes from simulation data for model reduction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds, eigsh


@dataclass
class PODBasis:
    """POD basis and associated information."""
    modes: np.ndarray  # (n_dofs, n_modes) - spatial modes
    singular_values: np.ndarray  # (n_modes,)
    temporal_coefficients: np.ndarray  # (n_snapshots, n_modes)
    mean: np.ndarray  # (n_dofs,) - mean subtracted before POD
    energy_fraction: np.ndarray  # Cumulative energy per mode
    n_modes: int


class POD:
    """
    Proper Orthogonal Decomposition.

    Computes optimal linear basis for representing snapshot data.

    Methods:
    - SVD: Standard singular value decomposition
    - Snapshot: More efficient when n_snapshots << n_dofs
    - Randomized: Fast approximate SVD for large problems
    - Incremental: Online/streaming POD
    """

    def __init__(self, n_modes: Optional[int] = None,
                 energy_threshold: float = 0.9999,
                 method: str = "auto",
                 center: bool = True):
        """
        Initialize POD.

        Args:
            n_modes: Number of modes to retain (overrides energy_threshold)
            energy_threshold: Retain modes capturing this fraction of energy
            method: "svd", "snapshot", "randomized", or "auto"
            center: Whether to subtract mean before decomposition
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.method = method
        self.center = center
        self._basis: Optional[PODBasis] = None

    def fit(self, snapshots: np.ndarray,
            weights: Optional[np.ndarray] = None) -> PODBasis:
        """
        Compute POD basis from snapshot matrix.

        Args:
            snapshots: Data matrix (n_dofs, n_snapshots) or (n_snapshots, n_dofs)
            weights: Optional inner product weights (n_dofs,)

        Returns:
            PODBasis with computed modes
        """
        # Ensure snapshots is (n_dofs, n_snapshots)
        if snapshots.shape[0] < snapshots.shape[1]:
            snapshots = snapshots.T

        n_dofs, n_snapshots = snapshots.shape

        # Center data
        if self.center:
            mean = np.mean(snapshots, axis=1)
            X = snapshots - mean[:, np.newaxis]
        else:
            mean = np.zeros(n_dofs)
            X = snapshots

        # Apply weights if provided
        if weights is not None:
            W_sqrt = np.sqrt(weights)
            X_weighted = X * W_sqrt[:, np.newaxis]
        else:
            X_weighted = X
            W_sqrt = None

        # Select method
        if self.method == "auto":
            if n_snapshots < n_dofs / 10:
                method = "snapshot"
            elif n_dofs > 10000:
                method = "randomized"
            else:
                method = "svd"
        else:
            method = self.method

        # Compute SVD
        if method == "svd":
            U, s, Vt = linalg.svd(X_weighted, full_matrices=False)
        elif method == "snapshot":
            U, s, Vt = self._snapshot_method(X_weighted)
        elif method == "randomized":
            U, s, Vt = self._randomized_svd(X_weighted)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Undo weighting
        if W_sqrt is not None:
            U = U / W_sqrt[:, np.newaxis]

        # Determine number of modes
        total_energy = np.sum(s**2)
        cumulative_energy = np.cumsum(s**2) / total_energy

        if self.n_modes is not None:
            n_modes = min(self.n_modes, len(s))
        else:
            n_modes = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
            n_modes = min(n_modes, len(s))

        # Truncate
        modes = U[:, :n_modes]
        singular_values = s[:n_modes]
        temporal_coefficients = Vt[:n_modes, :].T * singular_values

        self._basis = PODBasis(
            modes=modes,
            singular_values=singular_values,
            temporal_coefficients=temporal_coefficients,
            mean=mean,
            energy_fraction=cumulative_energy[:n_modes],
            n_modes=n_modes
        )

        return self._basis

    def _snapshot_method(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Snapshot method for when n_snapshots << n_dofs.

        Compute eigendecomposition of X^T X instead of X X^T.
        """
        n_dofs, n_snapshots = X.shape

        # Correlation matrix
        C = X.T @ X / n_snapshots

        # Eigendecomposition
        eigvals, V = linalg.eigh(C)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        V = V[:, idx]

        # Remove numerical noise
        eigvals = np.maximum(eigvals, 0)

        # Singular values
        s = np.sqrt(eigvals * n_snapshots)

        # Left singular vectors
        mask = s > 1e-10
        U = X @ V[:, mask] / s[mask]

        return U, s[mask], V[:, mask].T

    def _randomized_svd(self, X: np.ndarray,
                        oversampling: int = 10,
                        n_power_iter: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomized SVD for large matrices.

        Uses random projection to find approximate range.
        """
        n_dofs, n_snapshots = X.shape

        # Target rank
        if self.n_modes is not None:
            k = self.n_modes + oversampling
        else:
            k = min(100, min(n_dofs, n_snapshots))

        k = min(k, min(n_dofs, n_snapshots))

        # Random projection
        Omega = np.random.randn(n_snapshots, k)
        Y = X @ Omega

        # Power iteration for better accuracy
        for _ in range(n_power_iter):
            Y = X @ (X.T @ Y)

        # Orthonormalize
        Q, _ = linalg.qr(Y, mode='economic')

        # Project and compute SVD
        B = Q.T @ X
        U_tilde, s, Vt = linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde

        return U, s, Vt

    def project(self, u: np.ndarray) -> np.ndarray:
        """Project full-order state onto reduced basis."""
        if self._basis is None:
            raise ValueError("Call fit() first")

        u_centered = u - self._basis.mean
        return self._basis.modes.T @ u_centered

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Reconstruct full-order state from reduced coordinates."""
        if self._basis is None:
            raise ValueError("Call fit() first")

        return self._basis.modes @ a + self._basis.mean

    def reconstruction_error(self, snapshots: np.ndarray) -> Dict[str, float]:
        """Compute reconstruction error metrics."""
        if self._basis is None:
            raise ValueError("Call fit() first")

        if snapshots.shape[0] < snapshots.shape[1]:
            snapshots = snapshots.T

        # Project and reconstruct
        projected = self.project(snapshots.T).T
        reconstructed = self.reconstruct(projected.T).T

        # Errors
        error = snapshots - reconstructed
        rel_error = np.linalg.norm(error) / np.linalg.norm(snapshots)
        max_error = np.max(np.abs(error))

        return {
            "relative_l2": rel_error,
            "max_absolute": max_error,
            "energy_captured": self._basis.energy_fraction[-1]
        }


class PODGalerkin:
    """
    POD-Galerkin projection for reduced-order modeling.

    Projects governing equations onto POD basis to create
    a reduced dynamical system.
    """

    def __init__(self, pod: POD):
        self.pod = pod
        self._A_r: Optional[np.ndarray] = None
        self._B_r: Optional[np.ndarray] = None
        self._f_r: Optional[Callable] = None

    def project_linear_system(self, A: np.ndarray,
                               B: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project linear system du/dt = Au + Bu onto reduced basis.

        Returns reduced matrices A_r, B_r such that:
        da/dt = A_r @ a + B_r @ u_input
        """
        if self.pod._basis is None:
            raise ValueError("POD basis not computed")

        Phi = self.pod._basis.modes

        # A_r = Phi^T A Phi
        if issparse(A):
            A_Phi = A @ Phi
        else:
            A_Phi = A @ Phi
        self._A_r = Phi.T @ A_Phi

        # B_r = Phi^T B (if B is provided)
        if B is not None:
            self._B_r = Phi.T @ B

        return self._A_r, self._B_r

    def project_nonlinear_term(self, f: Callable[[np.ndarray], np.ndarray]) -> Callable:
        """
        Project nonlinear term f(u) onto reduced basis.

        Returns f_r(a) = Phi^T f(Phi @ a + mean)

        Note: This is expensive without hyper-reduction!
        """
        Phi = self.pod._basis.modes
        mean = self.pod._basis.mean

        def f_reduced(a: np.ndarray) -> np.ndarray:
            u = Phi @ a + mean
            return Phi.T @ f(u)

        self._f_r = f_reduced
        return f_reduced

    def simulate(self, a0: np.ndarray, t_span: Tuple[float, float],
                 dt: float, input_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate reduced-order model.

        Args:
            a0: Initial reduced state
            t_span: (t_start, t_end)
            dt: Time step
            input_func: Optional input u(t)

        Returns:
            (times, states) where states is (n_steps, n_modes)
        """
        if self._A_r is None:
            raise ValueError("Project system first")

        t_start, t_end = t_span
        times = np.arange(t_start, t_end + dt, dt)
        n_steps = len(times)
        n_modes = len(a0)

        states = np.zeros((n_steps, n_modes))
        states[0] = a0

        a = a0.copy()

        for i in range(1, n_steps):
            t = times[i-1]

            # RHS
            rhs = self._A_r @ a

            if self._B_r is not None and input_func is not None:
                u_in = input_func(t)
                rhs += self._B_r @ u_in

            if self._f_r is not None:
                rhs += self._f_r(a)

            # Forward Euler (simple, replace with better integrator)
            a = a + dt * rhs
            states[i] = a

        return times, states

    def reconstruct_trajectory(self, states: np.ndarray) -> np.ndarray:
        """Reconstruct full-order trajectory from reduced states."""
        n_steps = len(states)
        n_dofs = self.pod._basis.modes.shape[0]

        trajectory = np.zeros((n_steps, n_dofs))
        for i in range(n_steps):
            trajectory[i] = self.pod.reconstruct(states[i])

        return trajectory


class PODPetrovGalerkin:
    """
    Petrov-Galerkin projection with different test and trial spaces.

    Uses LSPG (Least-Squares Petrov-Galerkin) for improved stability.
    """

    def __init__(self, pod: POD, use_lspg: bool = True):
        self.pod = pod
        self.use_lspg = use_lspg

    def project_residual(self, residual: Callable[[np.ndarray], np.ndarray],
                         jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Set up LSPG projection of residual R(u) = 0.

        LSPG minimizes ||R(Phi @ a + mean)||^2
        """
        self._residual = residual
        self._jacobian = jacobian

        Phi = self.pod._basis.modes
        mean = self.pod._basis.mean

        def reduced_residual(a: np.ndarray) -> np.ndarray:
            u = Phi @ a + mean
            R = residual(u)

            if self.use_lspg:
                # LSPG: Phi^T J^T R where J = dR/du
                if jacobian is not None:
                    J = jacobian(u)
                    return Phi.T @ J.T @ R
                else:
                    return Phi.T @ R
            else:
                # Standard Galerkin
                return Phi.T @ R

        return reduced_residual

    def solve_steady(self, a0: np.ndarray,
                     tol: float = 1e-8,
                     max_iter: int = 100) -> np.ndarray:
        """Solve steady reduced problem."""
        from scipy.optimize import fsolve

        reduced_residual = self.project_residual(self._residual, self._jacobian)
        a_solution, info, ier, msg = fsolve(reduced_residual, a0, full_output=True)

        if ier != 1:
            print(f"Warning: solver did not converge: {msg}")

        return a_solution


class AdaptivePOD:
    """
    Adaptive POD that updates basis during simulation.

    Adds new modes when projection error exceeds threshold.
    """

    def __init__(self, n_initial_modes: int = 10,
                 error_threshold: float = 0.1,
                 max_modes: int = 100):
        self.n_initial_modes = n_initial_modes
        self.error_threshold = error_threshold
        self.max_modes = max_modes

        self._modes: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None

    def initialize(self, snapshots: np.ndarray):
        """Initialize with initial snapshots."""
        pod = POD(n_modes=self.n_initial_modes, center=True)
        basis = pod.fit(snapshots)

        self._modes = basis.modes.copy()
        self._mean = basis.mean.copy()

    def update(self, new_snapshot: np.ndarray) -> bool:
        """
        Update basis with new snapshot if needed.

        Returns True if basis was updated.
        """
        if self._modes is None:
            raise ValueError("Call initialize() first")

        # Check projection error
        u_centered = new_snapshot - self._mean
        a = self._modes.T @ u_centered
        u_reconstructed = self._modes @ a

        error = np.linalg.norm(u_centered - u_reconstructed)
        rel_error = error / (np.linalg.norm(u_centered) + 1e-10)

        if rel_error > self.error_threshold and self._modes.shape[1] < self.max_modes:
            # Add new mode from residual
            residual = u_centered - u_reconstructed
            new_mode = residual / (np.linalg.norm(residual) + 1e-10)

            # Orthogonalize against existing modes
            for i in range(self._modes.shape[1]):
                new_mode -= np.dot(new_mode, self._modes[:, i]) * self._modes[:, i]

            new_mode /= (np.linalg.norm(new_mode) + 1e-10)

            self._modes = np.column_stack([self._modes, new_mode])

            # Update mean
            alpha = 1.0 / (self._modes.shape[1] + 1)
            self._mean = (1 - alpha) * self._mean + alpha * new_snapshot

            return True

        return False

    def project(self, u: np.ndarray) -> np.ndarray:
        """Project onto current basis."""
        u_centered = u - self._mean
        return self._modes.T @ u_centered

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Reconstruct from reduced coordinates."""
        return self._modes @ a + self._mean

    @property
    def n_modes(self) -> int:
        """Current number of modes."""
        return self._modes.shape[1] if self._modes is not None else 0


class WeightedPOD(POD):
    """
    POD with custom inner product weights.

    Useful for:
    - Mass-weighted POD (finite element)
    - Energy-based POD
    - Region-of-interest weighting
    """

    def __init__(self, weights: np.ndarray, **kwargs):
        """
        Initialize with weight matrix.

        Args:
            weights: Diagonal weights (n_dofs,) or sparse matrix (n_dofs, n_dofs)
        """
        super().__init__(**kwargs)

        if len(weights.shape) == 1:
            self._weights = weights
            self._weights_matrix = None
        else:
            self._weights = None
            self._weights_matrix = weights

    def fit(self, snapshots: np.ndarray) -> PODBasis:
        """Fit with weighted inner product."""
        if self._weights is not None:
            return super().fit(snapshots, weights=self._weights)
        else:
            # Full weight matrix case
            if snapshots.shape[0] < snapshots.shape[1]:
                snapshots = snapshots.T

            n_dofs, n_snapshots = snapshots.shape

            if self.center:
                mean = np.mean(snapshots, axis=1)
                X = snapshots - mean[:, np.newaxis]
            else:
                mean = np.zeros(n_dofs)
                X = snapshots

            # Weighted SVD via generalized eigenvalue problem
            # Find modes such that Phi^T W Phi = I
            if issparse(self._weights_matrix):
                WX = self._weights_matrix @ X
            else:
                WX = self._weights_matrix @ X

            C = X.T @ WX

            eigvals, V = linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            V = V[:, idx]

            s = np.sqrt(np.maximum(eigvals, 0))
            mask = s > 1e-10

            U = X @ V[:, mask] / s[mask]

            # Determine number of modes
            total_energy = np.sum(s**2)
            cumulative_energy = np.cumsum(s**2) / total_energy

            if self.n_modes is not None:
                n_modes = min(self.n_modes, np.sum(mask))
            else:
                n_modes = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
                n_modes = min(n_modes, np.sum(mask))

            self._basis = PODBasis(
                modes=U[:, :n_modes],
                singular_values=s[:n_modes],
                temporal_coefficients=V[:, :n_modes] * s[:n_modes],
                mean=mean,
                energy_fraction=cumulative_energy[:n_modes],
                n_modes=n_modes
            )

            return self._basis


class MultiFieldPOD:
    """
    POD for multi-field problems (e.g., velocity + pressure).

    Options:
    - Concatenated: Stack fields, single POD
    - Separate: Independent POD per field
    - Coupled: Capture correlations between fields
    """

    def __init__(self, field_sizes: List[int],
                 n_modes_per_field: Optional[List[int]] = None,
                 coupled: bool = True):
        """
        Args:
            field_sizes: Size of each field
            n_modes_per_field: Modes per field (for separate POD)
            coupled: Use coupled POD (captures cross-field correlations)
        """
        self.field_sizes = field_sizes
        self.n_fields = len(field_sizes)
        self.n_modes_per_field = n_modes_per_field
        self.coupled = coupled

        self._bases: List[Optional[PODBasis]] = [None] * self.n_fields
        self._coupled_basis: Optional[PODBasis] = None

    def fit(self, snapshots_per_field: List[np.ndarray]) -> Union[PODBasis, List[PODBasis]]:
        """
        Compute POD for multi-field data.

        Args:
            snapshots_per_field: List of snapshot matrices, one per field
        """
        if self.coupled:
            # Stack all fields
            stacked = np.vstack(snapshots_per_field)
            pod = POD(n_modes=sum(self.n_modes_per_field) if self.n_modes_per_field else None)
            self._coupled_basis = pod.fit(stacked)
            return self._coupled_basis
        else:
            # Separate POD per field
            for i, snapshots in enumerate(snapshots_per_field):
                n_modes = self.n_modes_per_field[i] if self.n_modes_per_field else None
                pod = POD(n_modes=n_modes)
                self._bases[i] = pod.fit(snapshots)
            return self._bases

    def project(self, fields: List[np.ndarray]) -> Union[np.ndarray, List[np.ndarray]]:
        """Project multi-field state."""
        if self.coupled:
            stacked = np.concatenate(fields)
            stacked_centered = stacked - self._coupled_basis.mean
            return self._coupled_basis.modes.T @ stacked_centered
        else:
            return [
                self._bases[i].modes.T @ (fields[i] - self._bases[i].mean)
                for i in range(self.n_fields)
            ]

    def reconstruct(self, reduced: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """Reconstruct multi-field state."""
        if self.coupled:
            stacked = self._coupled_basis.modes @ reduced + self._coupled_basis.mean
            # Split back into fields
            fields = []
            offset = 0
            for size in self.field_sizes:
                fields.append(stacked[offset:offset+size])
                offset += size
            return fields
        else:
            return [
                self._bases[i].modes @ reduced[i] + self._bases[i].mean
                for i in range(self.n_fields)
            ]


class ParametricPOD:
    """
    POD for parametric problems.

    Interpolates between parameter-specific bases.
    """

    def __init__(self, interpolation: str = "rbf"):
        """
        Args:
            interpolation: "rbf", "grassmann", or "manifold"
        """
        self.interpolation = interpolation
        self._parameter_bases: Dict[tuple, PODBasis] = {}
        self._parameters: List[np.ndarray] = []

    def add_basis(self, parameter: np.ndarray, snapshots: np.ndarray,
                  n_modes: int = 10):
        """Add POD basis for a specific parameter value."""
        pod = POD(n_modes=n_modes)
        basis = pod.fit(snapshots)

        key = tuple(parameter.flatten())
        self._parameter_bases[key] = basis
        self._parameters.append(parameter)

    def interpolate_basis(self, parameter: np.ndarray) -> np.ndarray:
        """
        Interpolate basis at new parameter value.

        Uses Grassmann interpolation for proper handling of
        orthonormal matrices.
        """
        if len(self._parameters) == 0:
            raise ValueError("No bases added")

        if len(self._parameters) == 1:
            return list(self._parameter_bases.values())[0].modes

        if self.interpolation == "rbf":
            return self._rbf_interpolate(parameter)
        elif self.interpolation == "grassmann":
            return self._grassmann_interpolate(parameter)
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

    def _rbf_interpolate(self, parameter: np.ndarray) -> np.ndarray:
        """Simple RBF interpolation of modes (not Grassmann-aware)."""
        from scipy.interpolate import RBFInterpolator

        params = np.array(self._parameters)
        n_modes = list(self._parameter_bases.values())[0].n_modes
        n_dofs = list(self._parameter_bases.values())[0].modes.shape[0]

        # Stack all modes
        all_modes = np.zeros((len(self._parameters), n_dofs * n_modes))
        for i, key in enumerate(self._parameter_bases.keys()):
            all_modes[i] = self._parameter_bases[key].modes.flatten()

        # Interpolate
        rbf = RBFInterpolator(params, all_modes)
        interpolated_flat = rbf(parameter.reshape(1, -1))[0]

        # Reshape and orthonormalize
        modes = interpolated_flat.reshape(n_dofs, n_modes)
        modes, _ = np.linalg.qr(modes)

        return modes

    def _grassmann_interpolate(self, parameter: np.ndarray) -> np.ndarray:
        """
        Grassmann manifold interpolation.

        Properly handles the geometry of subspaces.
        """
        params = np.array(self._parameters)

        # Find closest basis as reference
        distances = np.linalg.norm(params - parameter, axis=1)
        ref_idx = np.argmin(distances)
        ref_key = list(self._parameter_bases.keys())[ref_idx]
        Phi_ref = self._parameter_bases[ref_key].modes

        # Compute weights (inverse distance)
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights)

        # Interpolate in tangent space
        tangent_sum = np.zeros_like(Phi_ref)

        for i, key in enumerate(self._parameter_bases.keys()):
            if i == ref_idx:
                continue

            Phi_i = self._parameter_bases[key].modes

            # Log map: Phi_ref^T Phi_i = U S V^T, tangent = V U^T
            M = Phi_ref.T @ Phi_i
            U, S, Vt = np.linalg.svd(M, full_matrices=False)

            # Clamp singular values
            S = np.clip(S, -1, 1)
            angles = np.arccos(S)

            # Tangent vector
            tangent = Phi_i @ Vt.T - Phi_ref @ U @ np.diag(np.cos(angles)) @ Vt
            tangent = tangent @ np.diag(angles / (np.sin(angles) + 1e-10)) @ Vt.T @ U.T

            tangent_sum += weights[i] * tangent

        # Exponential map back to Grassmann
        U, S, Vt = np.linalg.svd(tangent_sum, full_matrices=False)
        Phi_new = Phi_ref @ Vt.T @ np.diag(np.cos(S)) @ Vt + U @ np.diag(np.sin(S)) @ Vt

        # Orthonormalize
        Phi_new, _ = np.linalg.qr(Phi_new)

        return Phi_new
