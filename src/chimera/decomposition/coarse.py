"""
Coarse space construction for two-level methods.

Provides global coupling for scalable domain decomposition.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass
class CoarseSpaceResult:
    """Coarse space basis and operators."""
    basis: np.ndarray  # (n_global, n_coarse)
    restriction: np.ndarray  # (n_coarse, n_global) = basis.T
    prolongation: np.ndarray  # (n_global, n_coarse) = basis
    coarse_matrix: np.ndarray  # basis.T @ A @ basis
    n_coarse: int


class CoarseSpace:
    """
    Base class for coarse space construction.
    """

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> CoarseSpaceResult:
        """
        Build coarse space.

        Args:
            A: Global stiffness matrix
            partition: Domain partition

        Returns:
            CoarseSpaceResult
        """
        raise NotImplementedError


class NicolaidesCoarse(CoarseSpace):
    """
    Nicolaides coarse space.

    One basis function per subdomain (constant on each subdomain).
    Simple but may not be sufficient for heterogeneous problems.
    """

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> CoarseSpaceResult:
        """Build Nicolaides coarse space."""
        n_global = A.shape[0]
        n_coarse = len(partition)

        # One constant per subdomain
        basis = np.zeros((n_global, n_coarse))

        for i, nodes in enumerate(partition):
            basis[nodes, i] = 1.0
            # Normalize
            basis[nodes, i] /= np.sqrt(len(nodes))

        # Coarse operators
        restriction = basis.T
        prolongation = basis
        coarse_matrix = restriction @ (A @ prolongation)

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=prolongation,
            coarse_matrix=coarse_matrix.toarray() if sparse.issparse(coarse_matrix) else coarse_matrix,
            n_coarse=n_coarse
        )


class GenEOCoarse(CoarseSpace):
    """
    GenEO (Generalized Eigenproblems in the Overlaps) coarse space.

    Robust coarse space based on local eigenproblems.
    Provides bounds on condition number independent of coefficient jumps.
    """

    def __init__(self, threshold: float = 0.1,
                 max_modes_per_domain: int = 10):
        """
        Args:
            threshold: Eigenvalue threshold for mode selection
            max_modes_per_domain: Maximum coarse modes per subdomain
        """
        self.threshold = threshold
        self.max_modes_per_domain = max_modes_per_domain

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray],
              overlap_info: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> CoarseSpaceResult:
        """
        Build GenEO coarse space.

        Args:
            A: Global matrix
            partition: Domain partition
            overlap_info: Overlap regions for each subdomain
        """
        n_global = A.shape[0]
        n_domains = len(partition)

        coarse_vectors = []

        for i, nodes in enumerate(partition):
            # Extract local matrix
            n_local = len(nodes)
            R = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), nodes)),
                shape=(n_local, n_global)
            )

            A_local = R @ A @ R.T

            # Build partition of unity D_i
            # (simplified: diagonal with 1 on interior, 0.5 on overlap)
            D_local = np.ones(n_local)

            if overlap_info is not None and i < len(overlap_info):
                interior, overlap = overlap_info[i]
                interior_local = np.searchsorted(nodes, interior)
                overlap_local = np.searchsorted(nodes, overlap)

                valid_overlap = overlap_local[overlap_local < n_local]
                D_local[valid_overlap] = 0.5

            D = sparse.diags(D_local)

            # Generalized eigenproblem: D A_local D v = λ A_local v
            DAD = D @ A_local @ D

            try:
                # Compute smallest eigenvalues
                n_eigs = min(self.max_modes_per_domain, n_local - 1)
                eigvals, eigvecs = eigsh(DAD.tocsr(), k=n_eigs,
                                         M=A_local.tocsr(),
                                         which='SM', sigma=0)

                # Select modes below threshold
                selected = eigvals < self.threshold

                if np.any(selected):
                    local_modes = eigvecs[:, selected]

                    # Extend to global
                    for j in range(local_modes.shape[1]):
                        global_mode = np.zeros(n_global)
                        global_mode[nodes] = D_local * local_modes[:, j]
                        coarse_vectors.append(global_mode)
                else:
                    # At least add constant
                    global_mode = np.zeros(n_global)
                    global_mode[nodes] = D_local / np.sqrt(len(nodes))
                    coarse_vectors.append(global_mode)

            except Exception:
                # Fallback to constant
                global_mode = np.zeros(n_global)
                global_mode[nodes] = D_local / np.sqrt(len(nodes))
                coarse_vectors.append(global_mode)

        # Assemble coarse basis
        basis = np.column_stack(coarse_vectors)
        n_coarse = basis.shape[1]

        # Orthonormalize
        basis, _ = np.linalg.qr(basis)

        # Coarse operators
        restriction = basis.T
        prolongation = basis
        coarse_matrix = restriction @ A.toarray() @ prolongation

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=prolongation,
            coarse_matrix=coarse_matrix,
            n_coarse=n_coarse
        )


class SpectralCoarse(CoarseSpace):
    """
    Spectral coarse space from global eigenmodes.

    Uses low-frequency modes of global operator.
    """

    def __init__(self, n_modes: int = 20):
        """
        Args:
            n_modes: Number of global modes
        """
        self.n_modes = n_modes

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> CoarseSpaceResult:
        """Build spectral coarse space."""
        n_global = A.shape[0]
        n_modes = min(self.n_modes, n_global - 1)

        # Compute lowest eigenmodes
        try:
            eigvals, eigvecs = eigsh(A, k=n_modes, which='SM', sigma=0)
            basis = eigvecs
        except Exception:
            # Fallback to random
            basis = np.random.randn(n_global, n_modes)
            basis, _ = np.linalg.qr(basis)

        # Coarse operators
        restriction = basis.T
        prolongation = basis
        coarse_matrix = restriction @ A.toarray() @ prolongation

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=prolongation,
            coarse_matrix=coarse_matrix,
            n_coarse=n_modes
        )


class MultiscaleCoarse(CoarseSpace):
    """
    Multiscale coarse space.

    Uses local multiscale basis functions that capture fine-scale effects.
    """

    def __init__(self, n_local_modes: int = 4):
        """
        Args:
            n_local_modes: Modes per coarse element
        """
        self.n_local_modes = n_local_modes

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> CoarseSpaceResult:
        """Build multiscale coarse space."""
        n_global = A.shape[0]
        n_domains = len(partition)

        coarse_vectors = []

        for i, nodes in enumerate(partition):
            n_local = len(nodes)

            if n_local <= 1:
                global_mode = np.zeros(n_global)
                global_mode[nodes] = 1.0
                coarse_vectors.append(global_mode)
                continue

            # Extract local matrix
            R = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), nodes)),
                shape=(n_local, n_global)
            )

            A_local = R @ A @ R.T

            # Local eigendecomposition
            n_modes = min(self.n_local_modes, n_local - 1)

            try:
                eigvals, eigvecs = eigsh(A_local.tocsr(), k=n_modes, which='SM', sigma=0)

                for j in range(n_modes):
                    global_mode = np.zeros(n_global)
                    global_mode[nodes] = eigvecs[:, j]
                    coarse_vectors.append(global_mode)

            except Exception:
                # Fallback
                global_mode = np.zeros(n_global)
                global_mode[nodes] = 1.0 / np.sqrt(n_local)
                coarse_vectors.append(global_mode)

        # Assemble and orthonormalize
        basis = np.column_stack(coarse_vectors)
        basis, _ = np.linalg.qr(basis)

        # Coarse operators
        restriction = basis.T
        prolongation = basis
        coarse_matrix = restriction @ A.toarray() @ prolongation

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=prolongation,
            coarse_matrix=coarse_matrix,
            n_coarse=basis.shape[1]
        )


class AdaptiveCoarse(CoarseSpace):
    """
    Adaptive coarse space that grows during iteration.

    Adds modes based on residual information.
    """

    def __init__(self, initial_modes: int = 5,
                 max_modes: int = 50,
                 tolerance: float = 0.1):
        """
        Args:
            initial_modes: Starting number of modes
            max_modes: Maximum modes
            tolerance: Convergence tolerance for adding modes
        """
        self.initial_modes = initial_modes
        self.max_modes = max_modes
        self.tolerance = tolerance

        self._basis: Optional[np.ndarray] = None

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> CoarseSpaceResult:
        """Build initial coarse space."""
        # Start with Nicolaides
        nicolaides = NicolaidesCoarse()
        return nicolaides.build(A, partition)

    def enrich(self, A: sparse.csr_matrix,
               residual: np.ndarray,
               current_result: CoarseSpaceResult) -> CoarseSpaceResult:
        """
        Enrich coarse space based on residual.

        Args:
            A: Global matrix
            residual: Current residual
            current_result: Current coarse space

        Returns:
            Enriched coarse space
        """
        n_global = A.shape[0]

        if current_result.n_coarse >= self.max_modes:
            return current_result

        # Project residual out of current coarse space
        r_perp = residual - current_result.prolongation @ (
            current_result.restriction @ residual
        )

        # Check if enrichment needed
        if np.linalg.norm(r_perp) < self.tolerance * np.linalg.norm(residual):
            return current_result

        # Add new mode from residual
        new_mode = r_perp / np.linalg.norm(r_perp)

        # Orthogonalize against existing basis
        for i in range(current_result.n_coarse):
            new_mode -= np.dot(new_mode, current_result.basis[:, i]) * current_result.basis[:, i]

        new_mode /= (np.linalg.norm(new_mode) + 1e-10)

        # Extend basis
        basis = np.column_stack([current_result.basis, new_mode])

        # Update operators
        restriction = basis.T
        prolongation = basis
        coarse_matrix = restriction @ A.toarray() @ prolongation

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=prolongation,
            coarse_matrix=coarse_matrix,
            n_coarse=basis.shape[1]
        )


class HierarchicalCoarse(CoarseSpace):
    """
    Hierarchical coarse space using mesh hierarchy.

    Uses geometric multigrid-style coarsening.
    """

    def __init__(self, levels: int = 3):
        """
        Args:
            levels: Number of coarsening levels
        """
        self.levels = levels

    def build(self, A: sparse.csr_matrix,
              partition: List[np.ndarray],
              coordinates: Optional[np.ndarray] = None) -> CoarseSpaceResult:
        """Build hierarchical coarse space."""
        n_global = A.shape[0]

        if coordinates is None:
            # Use graph-based coarsening
            return self._graph_coarsen(A, partition)
        else:
            return self._geometric_coarsen(A, coordinates)

    def _graph_coarsen(self, A: sparse.csr_matrix,
                       partition: List[np.ndarray]) -> CoarseSpaceResult:
        """Coarsen using graph matching."""
        n_global = A.shape[0]

        # Heavy edge matching
        matched = np.zeros(n_global, dtype=bool)
        coarse_id = np.zeros(n_global, dtype=int)
        n_coarse = 0

        order = np.random.permutation(n_global)

        for i in order:
            if matched[i]:
                continue

            # Find best unmatched neighbor
            row = A.getrow(i)
            best_j = -1
            best_w = 0

            for j, w in zip(row.indices, row.data):
                if not matched[j] and w > best_w:
                    best_w = w
                    best_j = j

            coarse_id[i] = n_coarse
            matched[i] = True

            if best_j >= 0:
                coarse_id[best_j] = n_coarse
                matched[best_j] = True

            n_coarse += 1

        # Build prolongation
        prolongation = sparse.csr_matrix(
            (np.ones(n_global),
             (np.arange(n_global), coarse_id)),
            shape=(n_global, n_coarse)
        )

        # Normalize
        col_sums = np.array(prolongation.sum(axis=0)).flatten()
        prolongation = prolongation @ sparse.diags(1.0 / (col_sums + 1e-10))

        basis = prolongation.toarray()

        # Coarse operators
        restriction = basis.T
        coarse_matrix = restriction @ A.toarray() @ basis

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=basis,
            coarse_matrix=coarse_matrix,
            n_coarse=n_coarse
        )

    def _geometric_coarsen(self, A: sparse.csr_matrix,
                           coordinates: np.ndarray) -> CoarseSpaceResult:
        """Coarsen using geometric clustering."""
        n_global = A.shape[0]
        dim = coordinates.shape[1]

        # Coarsen by factor of 2 in each direction
        bbox_min = coordinates.min(axis=0)
        bbox_max = coordinates.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Coarse grid spacing
        n_coarse_per_dim = max(2, int(n_global ** (1/dim) / 2))
        h_coarse = bbox_size / n_coarse_per_dim

        # Assign to coarse cells
        cell_coords = ((coordinates - bbox_min) / (h_coarse + 1e-10)).astype(int)
        cell_coords = np.clip(cell_coords, 0, n_coarse_per_dim - 1)

        # Linear index
        if dim == 2:
            cell_ids = cell_coords[:, 0] + cell_coords[:, 1] * n_coarse_per_dim
        else:
            cell_ids = (cell_coords[:, 0] +
                       cell_coords[:, 1] * n_coarse_per_dim +
                       cell_coords[:, 2] * n_coarse_per_dim**2)

        n_coarse = n_coarse_per_dim ** dim

        # Build prolongation
        prolongation = sparse.csr_matrix(
            (np.ones(n_global),
             (np.arange(n_global), cell_ids)),
            shape=(n_global, n_coarse)
        )

        # Remove empty coarse nodes
        col_sums = np.array(prolongation.sum(axis=0)).flatten()
        non_empty = col_sums > 0
        prolongation = prolongation[:, non_empty]
        n_coarse = prolongation.shape[1]

        # Normalize
        col_sums = np.array(prolongation.sum(axis=0)).flatten()
        prolongation = prolongation @ sparse.diags(1.0 / (col_sums + 1e-10))

        basis = prolongation.toarray()

        # Coarse operators
        restriction = basis.T
        coarse_matrix = restriction @ A.toarray() @ basis

        return CoarseSpaceResult(
            basis=basis,
            restriction=restriction,
            prolongation=basis,
            coarse_matrix=coarse_matrix,
            n_coarse=n_coarse
        )
