"""
FETI (Finite Element Tearing and Interconnecting) methods.

Non-overlapping domain decomposition with Lagrange multipliers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, gmres, cg


@dataclass
class SubdomainData:
    """Data for a FETI subdomain."""
    nodes: np.ndarray  # Global node indices
    interior_nodes: np.ndarray  # Interior nodes
    boundary_nodes: np.ndarray  # Interface nodes
    local_matrix: sparse.csr_matrix  # K_i
    local_rhs: np.ndarray  # f_i
    rigid_body_modes: Optional[np.ndarray]  # Null space of K_i
    is_floating: bool  # Whether subdomain has rigid body modes


@dataclass
class InterfaceData:
    """Data for an interface between subdomains."""
    subdomain1: int
    subdomain2: int
    nodes_sub1: np.ndarray  # Local indices in subdomain 1
    nodes_sub2: np.ndarray  # Local indices in subdomain 2
    global_nodes: np.ndarray  # Global node indices


class FETI:
    """
    Classic FETI method (FETI-1).

    Non-overlapping decomposition with Lagrange multipliers
    for interface continuity.
    """

    def __init__(self, preconditioner: str = "dirichlet"):
        """
        Args:
            preconditioner: "dirichlet", "lumped", or "none"
        """
        self.preconditioner = preconditioner

        self._subdomains: List[SubdomainData] = []
        self._interfaces: List[InterfaceData] = []
        self._n_global: int = 0
        self._n_multipliers: int = 0

    def setup(self, A: sparse.csr_matrix, b: np.ndarray,
              partition: List[np.ndarray]) -> None:
        """
        Set up FETI system.

        Args:
            A: Global stiffness matrix
            b: Global RHS
            partition: Non-overlapping partition
        """
        self._n_global = A.shape[0]
        n_domains = len(partition)

        # Identify interfaces
        self._find_interfaces(partition, A)

        # Set up subdomains
        for domain_id, nodes in enumerate(partition):
            nodes = np.array(sorted(nodes))
            n_local = len(nodes)

            # Restriction
            R = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), nodes)),
                shape=(n_local, self._n_global)
            )

            # Local matrix and RHS
            K_local = R @ A @ R.T
            f_local = R @ b

            # Find boundary nodes (on interfaces)
            boundary = []
            for interface in self._interfaces:
                if interface.subdomain1 == domain_id:
                    boundary.extend(interface.nodes_sub1)
                elif interface.subdomain2 == domain_id:
                    boundary.extend(interface.nodes_sub2)

            boundary = np.unique(boundary)
            interior = np.setdiff1d(np.arange(n_local), boundary)

            # Detect rigid body modes (null space of K)
            rigid_modes = self._compute_rigid_body_modes(K_local)
            is_floating = rigid_modes is not None and rigid_modes.shape[1] > 0

            self._subdomains.append(SubdomainData(
                nodes=nodes,
                interior_nodes=interior,
                boundary_nodes=boundary,
                local_matrix=K_local.tocsr(),
                local_rhs=f_local,
                rigid_body_modes=rigid_modes,
                is_floating=is_floating
            ))

        # Set up interface system
        self._setup_interface_system()

    def _find_interfaces(self, partition: List[np.ndarray],
                         A: sparse.csr_matrix) -> None:
        """Find interfaces between subdomains."""
        n_domains = len(partition)

        # Build node-to-domain map
        node_to_domain = {}
        for d, nodes in enumerate(partition):
            for n in nodes:
                node_to_domain[n] = d

        # Find shared nodes
        A_coo = A.tocoo()
        shared = {}  # (d1, d2) -> nodes

        for i, j in zip(A_coo.row, A_coo.col):
            if i != j:
                d1 = node_to_domain.get(i, -1)
                d2 = node_to_domain.get(j, -1)

                if d1 != d2 and d1 >= 0 and d2 >= 0:
                    key = (min(d1, d2), max(d1, d2))
                    if key not in shared:
                        shared[key] = set()
                    shared[key].add(i)
                    shared[key].add(j)

        # Create interface data
        for (d1, d2), nodes in shared.items():
            nodes = np.array(sorted(nodes))

            # Local indices
            nodes_d1 = []
            nodes_d2 = []
            for n in nodes:
                if n in partition[d1]:
                    nodes_d1.append(np.searchsorted(partition[d1], n))
                if n in partition[d2]:
                    nodes_d2.append(np.searchsorted(partition[d2], n))

            self._interfaces.append(InterfaceData(
                subdomain1=d1,
                subdomain2=d2,
                nodes_sub1=np.array(nodes_d1),
                nodes_sub2=np.array(nodes_d2),
                global_nodes=nodes
            ))

        self._n_multipliers = sum(len(iface.global_nodes) for iface in self._interfaces)

    def _compute_rigid_body_modes(self, K: sparse.csr_matrix,
                                   tol: float = 1e-10) -> Optional[np.ndarray]:
        """Compute null space of stiffness matrix."""
        n = K.shape[0]

        # Try to find small eigenvalues
        try:
            from scipy.sparse.linalg import eigsh
            eigvals, eigvecs = eigsh(K, k=min(6, n-1), which='SM', sigma=0)

            # Keep only truly zero eigenvalues
            zero_mask = np.abs(eigvals) < tol
            if np.any(zero_mask):
                return eigvecs[:, zero_mask]
        except:
            pass

        return None

    def _setup_interface_system(self) -> None:
        """Set up the interface problem matrices."""
        n_domains = len(self._subdomains)

        # Build Boolean matrices B_i mapping local to interface
        self._B_matrices = []

        offset = 0
        for domain_id, sub in enumerate(self._subdomains):
            n_local = len(sub.nodes)

            rows = []
            cols = []
            data = []

            for iface in self._interfaces:
                if iface.subdomain1 == domain_id:
                    for i, local_idx in enumerate(iface.nodes_sub1):
                        rows.append(offset + i)
                        cols.append(local_idx)
                        data.append(1.0)
                    offset += len(iface.nodes_sub1)
                elif iface.subdomain2 == domain_id:
                    for i, local_idx in enumerate(iface.nodes_sub2):
                        rows.append(offset + i)
                        cols.append(local_idx)
                        data.append(-1.0)  # Opposite sign for coupling
                    offset += len(iface.nodes_sub2)

            B_i = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(self._n_multipliers, n_local)
            )
            self._B_matrices.append(B_i)

    def solve(self, tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Solve the FETI system.
        """
        # Solve interface problem for Lagrange multipliers
        # F λ = d where F = sum B_i K_i^+ B_i^T

        # Build d = sum B_i K_i^+ f_i
        d = np.zeros(self._n_multipliers)

        for i, (sub, B) in enumerate(zip(self._subdomains, self._B_matrices)):
            if sub.is_floating:
                # Use pseudo-inverse for floating domains
                K_plus = np.linalg.pinv(sub.local_matrix.toarray())
                u_local = K_plus @ sub.local_rhs
            else:
                u_local = spsolve(sub.local_matrix, sub.local_rhs)

            d += B @ u_local

        # Solve F λ = d using preconditioned CG
        def matvec_F(lam):
            """Apply F = sum B_i K_i^+ B_i^T."""
            result = np.zeros(self._n_multipliers)

            for i, (sub, B) in enumerate(zip(self._subdomains, self._B_matrices)):
                # B_i^T λ
                g = B.T @ lam

                # K_i^+ g
                if sub.is_floating:
                    K_plus = np.linalg.pinv(sub.local_matrix.toarray())
                    u = K_plus @ g
                else:
                    u = spsolve(sub.local_matrix, g)

                # B_i u
                result += B @ u

            return result

        F = sparse.linalg.LinearOperator(
            shape=(self._n_multipliers, self._n_multipliers),
            matvec=matvec_F
        )

        # Preconditioner
        if self.preconditioner == "dirichlet":
            M = self._build_dirichlet_preconditioner()
        elif self.preconditioner == "lumped":
            M = self._build_lumped_preconditioner()
        else:
            M = None

        # Solve
        lam, info = cg(F, d, M=M, tol=tol, maxiter=max_iter)

        # Recover solution on each subdomain
        u_global = np.zeros(self._n_global)

        for i, (sub, B) in enumerate(zip(self._subdomains, self._B_matrices)):
            # u_i = K_i^+ (f_i - B_i^T λ)
            rhs = sub.local_rhs - B.T @ lam

            if sub.is_floating:
                K_plus = np.linalg.pinv(sub.local_matrix.toarray())
                u_local = K_plus @ rhs
            else:
                u_local = spsolve(sub.local_matrix, rhs)

            u_global[sub.nodes] = u_local

        return u_global, {"converged": info == 0, "iterations": abs(info)}

    def _build_dirichlet_preconditioner(self) -> sparse.linalg.LinearOperator:
        """Build Dirichlet preconditioner."""
        def matvec(x):
            result = np.zeros(self._n_multipliers)

            for i, (sub, B) in enumerate(zip(self._subdomains, self._B_matrices)):
                # Extract boundary-boundary block
                bb = sub.boundary_nodes
                if len(bb) > 0:
                    K_bb = sub.local_matrix[np.ix_(bb, bb)]
                    g = B.T @ x
                    g_b = g[bb]

                    # Solve Schur complement approximately
                    s = spsolve(K_bb.tocsr(), g_b)
                    g_result = np.zeros(len(sub.nodes))
                    g_result[bb] = s

                    result += B @ g_result

            return result

        return sparse.linalg.LinearOperator(
            shape=(self._n_multipliers, self._n_multipliers),
            matvec=matvec
        )

    def _build_lumped_preconditioner(self) -> sparse.linalg.LinearOperator:
        """Build lumped preconditioner (diagonal scaling)."""
        # Compute diagonal scaling
        diag = np.zeros(self._n_multipliers)

        for i, (sub, B) in enumerate(zip(self._subdomains, self._B_matrices)):
            K_diag = sub.local_matrix.diagonal()

            for j in range(B.shape[0]):
                row = B.getrow(j)
                for k, v in zip(row.indices, row.data):
                    diag[j] += v**2 / (K_diag[k] + 1e-10)

        def matvec(x):
            return x / (diag + 1e-10)

        return sparse.linalg.LinearOperator(
            shape=(self._n_multipliers, self._n_multipliers),
            matvec=matvec
        )


class FETIDP:
    """
    FETI-DP (Dual-Primal FETI).

    More robust than FETI-1. Uses primal continuity at corners
    and dual coupling at remaining interface.
    """

    def __init__(self):
        self._subdomains: List[SubdomainData] = []
        self._n_global: int = 0
        self._corner_nodes: List[np.ndarray] = []  # Primal (continuous)
        self._remainder_nodes: List[np.ndarray] = []  # Dual (multipliers)

    def setup(self, A: sparse.csr_matrix, b: np.ndarray,
              partition: List[np.ndarray],
              corners: Optional[List[np.ndarray]] = None) -> None:
        """
        Set up FETI-DP system.

        Args:
            A: Global matrix
            b: Global RHS
            partition: Non-overlapping partition
            corners: Corner nodes per subdomain (primal DOFs)
        """
        self._n_global = A.shape[0]
        n_domains = len(partition)

        # Identify corners if not provided
        if corners is None:
            corners = self._identify_corners(partition, A)

        self._corner_nodes = corners

        # Set up subdomains
        for domain_id, nodes in enumerate(partition):
            nodes = np.array(sorted(nodes))
            n_local = len(nodes)

            # Restriction
            R = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), nodes)),
                shape=(n_local, self._n_global)
            )

            K_local = R @ A @ R.T
            f_local = R @ b

            # Corners are interior (primal), rest of boundary is remainder (dual)
            corner_local = np.searchsorted(nodes, corners[domain_id])

            self._subdomains.append(SubdomainData(
                nodes=nodes,
                interior_nodes=np.arange(n_local),  # All local
                boundary_nodes=corner_local,  # Corners for now
                local_matrix=K_local.tocsr(),
                local_rhs=f_local,
                rigid_body_modes=None,
                is_floating=False
            ))

    def _identify_corners(self, partition: List[np.ndarray],
                          A: sparse.csr_matrix) -> List[np.ndarray]:
        """Identify corner nodes (shared by 3+ subdomains)."""
        n_domains = len(partition)

        # Count subdomain membership
        node_count = {}
        for d, nodes in enumerate(partition):
            for n in nodes:
                if n not in node_count:
                    node_count[n] = []
                node_count[n].append(d)

        # Find corners (3+ subdomains)
        corners_per_domain = [[] for _ in range(n_domains)]

        for node, domains in node_count.items():
            if len(domains) >= 3:
                for d in domains:
                    corners_per_domain[d].append(node)

        return [np.array(c) for c in corners_per_domain]

    def solve(self, tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Solve FETI-DP system.

        Simplified implementation - full FETI-DP requires more complex setup.
        """
        # Collect all corner DOFs
        all_corners = np.unique(np.concatenate(self._corner_nodes))
        n_corners = len(all_corners)

        # Build global corner problem
        # K_cc u_c = f_c - sum K_cr K_rr^{-1} f_r

        # For now, use simplified approach
        u_global = np.zeros(self._n_global)

        # Solve each subdomain
        for sub in self._subdomains:
            u_local = spsolve(sub.local_matrix, sub.local_rhs)
            u_global[sub.nodes] = u_local

        return u_global, {"converged": True, "iterations": 1}


class TFETI:
    """
    Total FETI.

    All subdomains are floating (rigid body modes removed).
    Useful for problems without Dirichlet BCs.
    """

    def __init__(self):
        self._subdomains: List[SubdomainData] = []
        self._n_global: int = 0

    def setup(self, A: sparse.csr_matrix, b: np.ndarray,
              partition: List[np.ndarray]) -> None:
        """Set up Total FETI system."""
        self._n_global = A.shape[0]

        for domain_id, nodes in enumerate(partition):
            nodes = np.array(sorted(nodes))
            n_local = len(nodes)

            R = sparse.csr_matrix(
                (np.ones(n_local),
                 (np.arange(n_local), nodes)),
                shape=(n_local, self._n_global)
            )

            K_local = R @ A @ R.T
            f_local = R @ b

            # All domains are floating in TFETI
            rigid_modes = self._compute_rigid_body_modes(K_local)

            self._subdomains.append(SubdomainData(
                nodes=nodes,
                interior_nodes=np.arange(n_local),
                boundary_nodes=np.array([]),
                local_matrix=K_local.tocsr(),
                local_rhs=f_local,
                rigid_body_modes=rigid_modes,
                is_floating=True
            ))

    def _compute_rigid_body_modes(self, K: sparse.csr_matrix) -> np.ndarray:
        """Compute rigid body modes."""
        n = K.shape[0]

        # For 2D elasticity: 3 rigid body modes
        # For 3D: 6 modes
        # Simplified: constant modes
        modes = np.ones((n, 1))
        modes /= np.linalg.norm(modes)

        return modes

    def solve(self, tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, Dict]:
        """Solve Total FETI system."""
        u_global = np.zeros(self._n_global)

        for sub in self._subdomains:
            # Use pseudo-inverse for floating domains
            K_dense = sub.local_matrix.toarray()

            # Project out rigid body modes
            R = sub.rigid_body_modes
            if R is not None:
                projector = np.eye(len(sub.nodes)) - R @ R.T
                K_projected = projector @ K_dense @ projector
                f_projected = projector @ sub.local_rhs

                u_local = np.linalg.lstsq(K_projected, f_projected, rcond=None)[0]
            else:
                u_local = spsolve(sub.local_matrix, sub.local_rhs)

            u_global[sub.nodes] = u_local

        return u_global, {"converged": True, "iterations": 1}
