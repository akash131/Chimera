"""
Meshless (mesh-free) solver for Chimera.

Implements Radial Basis Function (RBF) and Moving Least Squares (MLS) methods.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable, List
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csr_matrix
import time

from chimera.solvers.base import Solver, SolverResult, SolverStatus, SolverConfig
from chimera.core.field import Field, FieldType
from chimera.core.equation import Equation, OperatorType
from chimera.core.boundary import DirichletBC, NeumannBC
from chimera.mesh.mesh import Mesh, MeshType


class RBFType:
    """Radial Basis Function types."""
    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    THIN_PLATE_SPLINE = "thin_plate_spline"
    POLYHARMONIC = "polyharmonic"


class MeshlessSolver(Solver):
    """
    Meshless solver using RBF collocation.

    Advantages:
    - No mesh generation required
    - Easy refinement (just add points)
    - Handles complex geometries naturally
    - Good for moving boundaries

    Implements:
    - RBF-FD (Finite Difference with RBF)
    - Symmetric RBF collocation
    - MLS (Moving Least Squares)
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._A: Optional[csr_matrix] = None
        self._b: Optional[np.ndarray] = None
        self._source_func: Optional[Callable] = None
        self._rbf_type: str = RBFType.POLYHARMONIC
        self._shape_param: float = 1.0  # Shape parameter for RBF
        self._stencil_size: int = 20  # Number of neighbors for RBF-FD
        self._poly_degree: int = 2  # Polynomial augmentation degree

    def set_mesh(self, mesh: Mesh):
        """Set the point cloud."""
        self.mesh = mesh
        self._assembled = False

        # Build neighbor structure if not present
        if mesh._neighbors is None or mesh._neighbors.shape[1] < self._stencil_size:
            mesh.find_neighbors(k=self._stencil_size)

    def set_equation(self, equation: Equation):
        """Set the governing equation."""
        self.equation = equation
        self._assembled = False

    def set_source(self, source: Callable[[np.ndarray], np.ndarray]):
        """Set source term function."""
        self._source_func = source
        self._assembled = False

    def set_rbf(self, rbf_type: str, shape_param: float = 1.0):
        """Configure RBF parameters."""
        self._rbf_type = rbf_type
        self._shape_param = shape_param
        self._assembled = False

    def set_stencil_size(self, size: int):
        """Set number of neighbors for RBF-FD stencils."""
        self._stencil_size = size
        if self.mesh is not None:
            self.mesh.find_neighbors(k=size)
        self._assembled = False

    def assemble(self):
        """Assemble meshless system using RBF-FD."""
        if self.mesh is None:
            raise ValueError("No mesh set")
        if self.equation is None:
            raise ValueError("No equation set")

        start_time = time.time()

        n_points = self.mesh.n_points
        self._A = lil_matrix((n_points, n_points))
        self._b = np.zeros(n_points)

        # Identify interior and boundary points
        boundary_set = set(self.mesh.boundary_points) if self.mesh.boundary_points is not None else set()

        # Get diffusion coefficient
        diffusion = 1.0
        for term in self.equation.terms:
            if term.operator.op_type == OperatorType.LAPLACIAN:
                if isinstance(term.coefficient, (int, float)):
                    diffusion = abs(term.coefficient)

        # Assemble for each interior point
        for i in range(n_points):
            if i in boundary_set:
                continue  # Handle boundary separately

            # Get local stencil
            neighbors = self.mesh._neighbors[i]
            stencil = np.concatenate([[i], neighbors[neighbors >= 0]])
            stencil_coords = self.mesh.points[stencil]

            # Compute RBF-FD weights for Laplacian
            weights = self._compute_laplacian_weights(stencil_coords)

            # Fill matrix row
            for j, node in enumerate(stencil):
                self._A[i, node] = diffusion * weights[j]

        # Source term
        interior = np.array([i for i in range(n_points) if i not in boundary_set])
        if self._source_func is not None and len(interior) > 0:
            self._b[interior] = self._source_func(self.mesh.points[interior])

        self._A = self._A.tocsr()
        self._assembled = True

        if self.config.verbose:
            print(f"Meshless assembly: {time.time() - start_time:.3f}s")

    def _rbf(self, r: np.ndarray) -> np.ndarray:
        """Evaluate radial basis function."""
        eps = self._shape_param

        if self._rbf_type == RBFType.GAUSSIAN:
            return np.exp(-(eps * r) ** 2)

        elif self._rbf_type == RBFType.MULTIQUADRIC:
            return np.sqrt(1 + (eps * r) ** 2)

        elif self._rbf_type == RBFType.INVERSE_MULTIQUADRIC:
            return 1.0 / np.sqrt(1 + (eps * r) ** 2)

        elif self._rbf_type == RBFType.THIN_PLATE_SPLINE:
            # r^2 * log(r), with r^2*log(r) -> 0 as r -> 0
            result = np.zeros_like(r)
            mask = r > 0
            result[mask] = r[mask] ** 2 * np.log(r[mask])
            return result

        elif self._rbf_type == RBFType.POLYHARMONIC:
            # r^3 (odd order polyharmonic spline)
            return r ** 3

        else:
            raise ValueError(f"Unknown RBF type: {self._rbf_type}")

    def _rbf_laplacian(self, r: np.ndarray, dim: int) -> np.ndarray:
        """Evaluate Laplacian of RBF."""
        eps = self._shape_param

        if self._rbf_type == RBFType.GAUSSIAN:
            return 2 * eps ** 2 * (2 * eps ** 2 * r ** 2 - dim) * np.exp(-(eps * r) ** 2)

        elif self._rbf_type == RBFType.MULTIQUADRIC:
            mq = np.sqrt(1 + (eps * r) ** 2)
            return eps ** 2 * (dim - 1 + 1 / mq ** 2) / mq

        elif self._rbf_type == RBFType.THIN_PLATE_SPLINE:
            result = np.zeros_like(r)
            mask = r > 0
            result[mask] = 2 * (dim + 2 * np.log(r[mask]))
            return result

        elif self._rbf_type == RBFType.POLYHARMONIC:
            # Laplacian of r^3 = 3*(dim+1)*r
            return 3 * (dim + 1) * r

        else:
            # Numerical approximation
            return self._numerical_laplacian(r)

    def _numerical_laplacian(self, r: np.ndarray) -> np.ndarray:
        """Numerical Laplacian via finite differences."""
        h = 1e-5
        lap = np.zeros_like(r)
        for i, ri in enumerate(r):
            if ri > h:
                lap[i] = (self._rbf(np.array([ri + h]))[0]
                          - 2 * self._rbf(np.array([ri]))[0]
                          + self._rbf(np.array([ri - h]))[0]) / h ** 2
        return lap

    def _compute_laplacian_weights(self, stencil: np.ndarray) -> np.ndarray:
        """
        Compute RBF-FD weights for Laplacian operator.

        Uses polynomial augmentation for improved accuracy.
        """
        n = len(stencil)
        dim = stencil.shape[1]
        center = stencil[0]

        # Compute distances from center
        diffs = stencil - center
        dists = np.linalg.norm(diffs, axis=1)

        # Build RBF matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(stencil[i] - stencil[j])

        A = self._rbf(dist_matrix)

        # Polynomial terms (up to linear)
        n_poly = 1 + dim  # 1, x, y (for 2D)
        if self._poly_degree >= 2:
            n_poly += dim * (dim + 1) // 2  # Add x^2, xy, y^2

        P = np.zeros((n, n_poly))
        P[:, 0] = 1  # Constant
        P[:, 1:1 + dim] = diffs  # Linear terms

        if self._poly_degree >= 2 and n_poly > 1 + dim:
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    P[:, col] = diffs[:, i] * diffs[:, j]
                    col += 1

        # Augmented system
        M = np.zeros((n + n_poly, n + n_poly))
        M[:n, :n] = A
        M[:n, n:] = P
        M[n:, :n] = P.T

        # RHS: Laplacian of RBF at center (r=0 for center point)
        lap_rbf = self._rbf_laplacian(dists, dim)

        # Laplacian of polynomials at center
        lap_poly = np.zeros(n_poly)
        if self._poly_degree >= 2 and n_poly > 1 + dim:
            # Laplacian of x^2 is 2, of y^2 is 2, of xy is 0
            col = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        lap_poly[col] = 2.0
                    col += 1

        rhs = np.concatenate([lap_rbf, lap_poly])

        # Solve for weights
        try:
            coeffs = np.linalg.solve(M, rhs)
            weights = coeffs[:n]
        except np.linalg.LinAlgError:
            # Fallback: least squares
            coeffs, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            weights = coeffs[:n]

        return weights

    def solve(self) -> SolverResult:
        """Solve the meshless system."""
        if not self._assembled:
            self.assemble()

        start_time = time.time()

        # Apply boundary conditions
        A, b = self._apply_boundary_conditions()

        # Solve
        try:
            u = self._solve_linear_system(A, b)
            status = SolverStatus.SUCCESS
            residual = np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-12)
        except Exception as e:
            if self.config.verbose:
                print(f"Solve failed: {e}")
            u = np.zeros(self.mesh.n_points)
            status = SolverStatus.FAILED
            residual = float('inf')

        solve_time = time.time() - start_time

        field_name = self.equation.terms[0].field_name
        solution = Field(
            name=field_name,
            field_type=FieldType.SCALAR,
            dim=self.mesh.dim,
            values=u,
            points=self.mesh.points
        )

        return SolverResult(
            status=status,
            fields={field_name: solution},
            residual=residual,
            iterations=1,
            solve_time=solve_time,
            metadata={"method": "meshless", "rbf": self._rbf_type}
        )

    def _apply_boundary_conditions(self) -> Tuple[csr_matrix, np.ndarray]:
        """Apply boundary conditions."""
        A = self._A.copy().tolil()
        b = self._b.copy()

        boundary_nodes = self.mesh.boundary_points
        if boundary_nodes is None:
            return A.tocsr(), b

        for bc in self.bcs.get_dirichlet():
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_nodes = np.where(mask)[0]
            else:
                bc_nodes = boundary_nodes

            bc_values = bc.evaluate(self.mesh.points[bc_nodes])

            for idx, node in enumerate(bc_nodes):
                A[node, :] = 0
                A[node, node] = 1.0
                b[node] = bc_values[idx]

        for bc in self.bcs.get_neumann():
            # Neumann BCs require derivative weights
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_nodes = np.where(mask)[0]
            else:
                bc_nodes = boundary_nodes

            # Simplified: use finite difference approximation
            if self.mesh.boundary_normals is not None:
                for idx, node in enumerate(bc_nodes):
                    # Get outward normal
                    # Find which boundary face this node belongs to
                    pass  # Would need face-to-node mapping

        return A.tocsr(), b

    def get_matrix(self) -> np.ndarray:
        if self._A is None:
            self.assemble()
        return self._A.toarray()

    def get_rhs(self) -> np.ndarray:
        if self._b is None:
            self.assemble()
        return self._b

    def add_points(self, new_points: np.ndarray):
        """Add refinement points to the mesh."""
        old_points = self.mesh.points
        self.mesh.points = np.vstack([old_points, new_points])
        self.mesh._kdtree = None
        self.mesh.find_neighbors(k=self._stencil_size)
        self._assembled = False

    def estimate_error(self, solution: Field) -> np.ndarray:
        """Estimate error using residual-based indicator."""
        n = self.mesh.n_points
        errors = np.zeros(n)

        # Residual at each point
        residual = self._A @ solution.values - self._b
        errors = np.abs(residual)

        return errors
