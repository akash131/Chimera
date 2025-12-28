"""
Schwarz domain decomposition methods.

Overlapping domain decomposition with various variants.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, gmres


@dataclass
class DomainInfo:
    """Information about a subdomain."""
    indices: np.ndarray  # Global indices of subdomain nodes
    interior_indices: np.ndarray  # Non-overlap indices
    overlap_indices: np.ndarray  # Overlap indices
    local_matrix: sparse.csr_matrix  # Local stiffness matrix
    restriction: sparse.csr_matrix  # R_i: global -> local
    prolongation: sparse.csr_matrix  # P_i: local -> global


class AdditiveSchwarz:
    """
    Additive Schwarz preconditioner.

    P^{-1} = sum_i R_i^T A_i^{-1} R_i

    Each subdomain solve can be done in parallel.
    """

    def __init__(self, overlap: int = 1):
        """
        Args:
            overlap: Number of overlap layers
        """
        self.overlap = overlap
        self._domains: List[DomainInfo] = []
        self._n_global: int = 0

    def setup(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> None:
        """
        Set up Schwarz preconditioner.

        Args:
            A: Global stiffness matrix
            partition: List of node indices for each subdomain
        """
        self._n_global = A.shape[0]
        n_domains = len(partition)

        # Build adjacency for overlap extension
        adjacency = self._build_adjacency(A)

        for domain_id, core_nodes in enumerate(partition):
            # Extend domain with overlap
            domain_nodes = self._extend_domain(core_nodes, adjacency)

            # Identify interior vs overlap
            interior = core_nodes
            overlap_mask = ~np.isin(domain_nodes, core_nodes)
            overlap_nodes = domain_nodes[overlap_mask]

            # Build restriction operator R_i
            n_local = len(domain_nodes)
            restriction = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), domain_nodes)),
                shape=(n_local, self._n_global)
            )

            # Prolongation is transpose
            prolongation = restriction.T

            # Extract local matrix
            local_matrix = restriction @ A @ prolongation

            self._domains.append(DomainInfo(
                indices=domain_nodes,
                interior_indices=interior,
                overlap_indices=overlap_nodes,
                local_matrix=local_matrix.tocsr(),
                restriction=restriction,
                prolongation=prolongation
            ))

    def _build_adjacency(self, A: sparse.csr_matrix) -> Dict[int, List[int]]:
        """Build adjacency list from matrix sparsity."""
        adjacency = {}
        A_coo = A.tocoo()

        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                if i not in adjacency:
                    adjacency[i] = []
                adjacency[i].append(j)

        return adjacency

    def _extend_domain(self, core: np.ndarray,
                       adjacency: Dict[int, List[int]]) -> np.ndarray:
        """Extend domain by overlap layers."""
        current = set(core)

        for _ in range(self.overlap):
            extension = set()
            for node in current:
                if node in adjacency:
                    extension.update(adjacency[node])
            current.update(extension)

        return np.array(sorted(current))

    def apply(self, r: np.ndarray) -> np.ndarray:
        """
        Apply preconditioner P^{-1} r.
        """
        result = np.zeros(self._n_global)

        # Sum of local solves (can be parallelized)
        for domain in self._domains:
            # Restrict residual
            r_local = domain.restriction @ r

            # Local solve
            x_local = spsolve(domain.local_matrix, r_local)

            # Prolongate and add
            result += domain.prolongation @ x_local

        return result

    def solve(self, A: sparse.csr_matrix, b: np.ndarray,
              tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Solve Ax = b using preconditioned GMRES.
        """
        # Preconditioner as LinearOperator
        def precond(x):
            return self.apply(x)

        M = sparse.linalg.LinearOperator(
            shape=(self._n_global, self._n_global),
            matvec=precond
        )

        # Solve
        x, info = gmres(A, b, M=M, tol=tol, maxiter=max_iter)

        return x, {"converged": info == 0, "iterations": info if info > 0 else max_iter}


class MultiplicativeSchwarz:
    """
    Multiplicative Schwarz method.

    Solves subdomains sequentially, using updated solution.
    Better convergence but sequential.
    """

    def __init__(self, overlap: int = 1):
        self.overlap = overlap
        self._domains: List[DomainInfo] = []
        self._n_global: int = 0

    def setup(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> None:
        """Set up multiplicative Schwarz."""
        # Same setup as additive
        self._n_global = A.shape[0]
        self._A = A

        adjacency = {}
        A_coo = A.tocoo()
        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                if i not in adjacency:
                    adjacency[i] = []
                adjacency[i].append(j)

        for core_nodes in partition:
            current = set(core_nodes)
            for _ in range(self.overlap):
                extension = set()
                for node in current:
                    if node in adjacency:
                        extension.update(adjacency[node])
                current.update(extension)
            domain_nodes = np.array(sorted(current))

            n_local = len(domain_nodes)
            restriction = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), domain_nodes)),
                shape=(n_local, self._n_global)
            )
            prolongation = restriction.T

            local_matrix = restriction @ A @ prolongation

            self._domains.append(DomainInfo(
                indices=domain_nodes,
                interior_indices=core_nodes,
                overlap_indices=np.array([]),
                local_matrix=local_matrix.tocsr(),
                restriction=restriction,
                prolongation=prolongation
            ))

    def apply(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        One multiplicative Schwarz iteration.
        """
        for domain in self._domains:
            # Compute residual
            r = b - self._A @ x

            # Restrict
            r_local = domain.restriction @ r
            x_local = domain.restriction @ x

            # Local solve for correction
            correction = spsolve(domain.local_matrix, r_local)

            # Update solution
            x = x + domain.prolongation @ (correction - x_local)

        return x

    def solve(self, A: sparse.csr_matrix, b: np.ndarray,
              tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, Dict]:
        """Solve using multiplicative iterations."""
        x = np.zeros(self._n_global)

        for i in range(max_iter):
            x_old = x.copy()
            x = self.apply(x, b)

            # Check convergence
            residual = np.linalg.norm(b - A @ x)
            if residual < tol:
                return x, {"converged": True, "iterations": i + 1}

            # Check stagnation
            if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x):
                return x, {"converged": True, "iterations": i + 1}

        return x, {"converged": False, "iterations": max_iter}


class RestrictedAdditiveSchwarz:
    """
    Restricted Additive Schwarz (RAS).

    Only updates interior nodes, avoiding overlap addition.
    Better scalability than standard additive Schwarz.
    """

    def __init__(self, overlap: int = 1):
        self.overlap = overlap
        self._domains: List[DomainInfo] = []
        self._n_global: int = 0

    def setup(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> None:
        """Set up RAS preconditioner."""
        self._n_global = A.shape[0]

        adjacency = {}
        A_coo = A.tocoo()
        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                if i not in adjacency:
                    adjacency[i] = []
                adjacency[i].append(j)

        for core_nodes in partition:
            # Extend for overlap
            current = set(core_nodes)
            for _ in range(self.overlap):
                extension = set()
                for node in current:
                    if node in adjacency:
                        extension.update(adjacency[node])
                current.update(extension)
            domain_nodes = np.array(sorted(current))

            n_local = len(domain_nodes)
            n_interior = len(core_nodes)

            # Restriction (to full domain)
            restriction = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), domain_nodes)),
                shape=(n_local, self._n_global)
            )

            # Restricted prolongation (only interior nodes)
            # Map local interior indices to global
            local_interior = np.searchsorted(domain_nodes, core_nodes)
            prolongation = sparse.csr_matrix(
                (np.ones(n_interior),
                 (core_nodes, local_interior)),
                shape=(self._n_global, n_local)
            )

            local_matrix = restriction @ A @ restriction.T

            self._domains.append(DomainInfo(
                indices=domain_nodes,
                interior_indices=core_nodes,
                overlap_indices=np.setdiff1d(domain_nodes, core_nodes),
                local_matrix=local_matrix.tocsr(),
                restriction=restriction,
                prolongation=prolongation
            ))

    def apply(self, r: np.ndarray) -> np.ndarray:
        """Apply RAS preconditioner."""
        result = np.zeros(self._n_global)

        for domain in self._domains:
            r_local = domain.restriction @ r
            x_local = spsolve(domain.local_matrix, r_local)
            result += domain.prolongation @ x_local

        return result


class OptimizedSchwarz:
    """
    Optimized Schwarz Method (OSM).

    Uses Robin transmission conditions instead of Dirichlet.
    Better for wave propagation and convection problems.
    """

    def __init__(self, overlap: int = 0,
                 robin_param: float = 1.0):
        """
        Args:
            overlap: Overlap layers (can be 0 for OSM)
            robin_param: Robin parameter for transmission
        """
        self.overlap = overlap
        self.robin_param = robin_param
        self._domains: List[DomainInfo] = []
        self._n_global: int = 0
        self._interface_pairs: List[Tuple[int, int, np.ndarray]] = []

    def setup(self, A: sparse.csr_matrix,
              partition: List[np.ndarray]) -> None:
        """Set up optimized Schwarz method."""
        self._n_global = A.shape[0]
        self._A = A

        # Build adjacency
        adjacency = {}
        A_coo = A.tocoo()
        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                if i not in adjacency:
                    adjacency[i] = []
                adjacency[i].append(j)

        # Set up domains
        for domain_id, core_nodes in enumerate(partition):
            current = set(core_nodes)
            for _ in range(self.overlap):
                extension = set()
                for node in current:
                    if node in adjacency:
                        extension.update(adjacency[node])
                current.update(extension)
            domain_nodes = np.array(sorted(current))

            n_local = len(domain_nodes)
            restriction = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), domain_nodes)),
                shape=(n_local, self._n_global)
            )

            local_matrix = restriction @ A @ restriction.T

            self._domains.append(DomainInfo(
                indices=domain_nodes,
                interior_indices=core_nodes,
                overlap_indices=np.setdiff1d(domain_nodes, core_nodes),
                local_matrix=local_matrix.tocsr(),
                restriction=restriction,
                prolongation=restriction.T
            ))

        # Find interface pairs
        self._find_interfaces(partition, adjacency)

    def _find_interfaces(self, partition: List[np.ndarray],
                         adjacency: Dict[int, List[int]]) -> None:
        """Find interface nodes between domains."""
        n_domains = len(partition)
        domain_of_node = {}

        for d, nodes in enumerate(partition):
            for n in nodes:
                domain_of_node[n] = d

        for d1 in range(n_domains):
            for d2 in range(d1 + 1, n_domains):
                interface = []

                for node in partition[d1]:
                    if node in adjacency:
                        for neighbor in adjacency[node]:
                            if domain_of_node.get(neighbor, -1) == d2:
                                interface.append(node)
                                break

                if interface:
                    self._interface_pairs.append(
                        (d1, d2, np.array(interface))
                    )

    def apply(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        One OSM iteration with Robin transmission.
        """
        result = np.zeros(self._n_global)

        for domain in self._domains:
            # Modified local problem with Robin BC on interfaces
            r_local = domain.restriction @ (b - self._A @ x)

            # Add Robin contribution from neighbors
            robin_rhs = np.zeros(len(domain.indices))

            for d1, d2, interface in self._interface_pairs:
                if d1 == self._domains.index(domain) or d2 == self._domains.index(domain):
                    local_interface = np.searchsorted(domain.indices, interface)
                    valid = local_interface < len(domain.indices)
                    local_interface = local_interface[valid]

                    robin_rhs[local_interface] += self.robin_param * x[interface[valid]]

            # Solve modified local problem
            modified_matrix = domain.local_matrix.copy()
            # Add Robin penalty to diagonal at interface
            for d1, d2, interface in self._interface_pairs:
                if d1 == self._domains.index(domain) or d2 == self._domains.index(domain):
                    local_interface = np.searchsorted(domain.indices, interface)
                    valid = local_interface < len(domain.indices)
                    for li in local_interface[valid]:
                        modified_matrix[li, li] += self.robin_param

            x_local = spsolve(modified_matrix.tocsr(), r_local + robin_rhs)

            result += domain.prolongation @ x_local

        return result

    def solve(self, A: sparse.csr_matrix, b: np.ndarray,
              tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, Dict]:
        """Solve using OSM iterations."""
        x = np.zeros(self._n_global)

        for i in range(max_iter):
            x = self.apply(x, b)

            residual = np.linalg.norm(b - A @ x)
            if residual < tol:
                return x, {"converged": True, "iterations": i + 1}

        return x, {"converged": False, "iterations": max_iter}


class TwoLevelAdditiveSchwarz:
    """
    Two-level additive Schwarz with coarse space.

    Adds coarse problem for global coupling.
    Essential for scalability.
    """

    def __init__(self, overlap: int = 1):
        self.overlap = overlap
        self._domains: List[DomainInfo] = []
        self._n_global: int = 0

        # Coarse space
        self._coarse_basis: Optional[np.ndarray] = None
        self._coarse_matrix: Optional[np.ndarray] = None

    def setup(self, A: sparse.csr_matrix,
              partition: List[np.ndarray],
              coarse_basis: Optional[np.ndarray] = None) -> None:
        """
        Set up two-level Schwarz.

        Args:
            A: Global matrix
            partition: Domain partition
            coarse_basis: Coarse space basis (n_global, n_coarse)
        """
        self._n_global = A.shape[0]
        self._A = A

        # Set up subdomain solves (same as additive Schwarz)
        adjacency = {}
        A_coo = A.tocoo()
        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                if i not in adjacency:
                    adjacency[i] = []
                adjacency[i].append(j)

        for core_nodes in partition:
            current = set(core_nodes)
            for _ in range(self.overlap):
                extension = set()
                for node in current:
                    if node in adjacency:
                        extension.update(adjacency[node])
                current.update(extension)
            domain_nodes = np.array(sorted(current))

            n_local = len(domain_nodes)
            restriction = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), domain_nodes)),
                shape=(n_local, self._n_global)
            )

            local_matrix = restriction @ A @ restriction.T

            self._domains.append(DomainInfo(
                indices=domain_nodes,
                interior_indices=core_nodes,
                overlap_indices=np.array([]),
                local_matrix=local_matrix.tocsr(),
                restriction=restriction,
                prolongation=restriction.T
            ))

        # Set up coarse space
        if coarse_basis is None:
            # Default: one constant per subdomain
            n_coarse = len(partition)
            coarse_basis = np.zeros((self._n_global, n_coarse))
            for i, nodes in enumerate(partition):
                coarse_basis[nodes, i] = 1.0

        self._coarse_basis = coarse_basis
        self._coarse_matrix = coarse_basis.T @ (A @ coarse_basis)

    def apply(self, r: np.ndarray) -> np.ndarray:
        """Apply two-level preconditioner."""
        result = np.zeros(self._n_global)

        # Fine level corrections
        for domain in self._domains:
            r_local = domain.restriction @ r
            x_local = spsolve(domain.local_matrix, r_local)
            result += domain.prolongation @ x_local

        # Coarse correction
        r_coarse = self._coarse_basis.T @ r
        x_coarse = np.linalg.solve(self._coarse_matrix, r_coarse)
        result += self._coarse_basis @ x_coarse

        return result

    def solve(self, A: sparse.csr_matrix, b: np.ndarray,
              tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """Solve using preconditioned GMRES."""
        def precond(x):
            return self.apply(x)

        M = sparse.linalg.LinearOperator(
            shape=(self._n_global, self._n_global),
            matvec=precond
        )

        x, info = gmres(A, b, M=M, tol=tol, maxiter=max_iter)

        return x, {"converged": info == 0}
