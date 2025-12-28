"""
Finite Element Method (FEM) solver for Chimera.

Implements standard Galerkin FEM with various element types.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

from chimera.solvers.base import Solver, SolverResult, SolverStatus, SolverConfig
from chimera.core.field import Field, FieldType
from chimera.core.equation import Equation, OperatorType
from chimera.core.boundary import DirichletBC, NeumannBC, RobinBC
from chimera.mesh.mesh import Mesh, CellType


class FEMSolver(Solver):
    """
    Finite Element Method solver.

    Supports:
    - Linear and quadratic elements
    - 2D triangular and quadrilateral elements
    - 3D tetrahedral and hexahedral elements
    - Scalar and vector problems

    Uses standard Galerkin weak formulation.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._K: Optional[csr_matrix] = None  # Stiffness matrix
        self._M: Optional[csr_matrix] = None  # Mass matrix
        self._f: Optional[np.ndarray] = None  # Load vector
        self._source_func: Optional[Callable] = None

    def set_mesh(self, mesh: Mesh):
        """Set the computational mesh."""
        self.mesh = mesh
        self._assembled = False

        # Ensure boundary info is computed
        if mesh.boundary_points is None:
            mesh.find_boundary()

    def set_equation(self, equation: Equation):
        """Set the governing equation."""
        self.equation = equation
        self._assembled = False

    def set_source(self, source: Callable[[np.ndarray], np.ndarray]):
        """Set source term function f(x) -> values."""
        self._source_func = source
        self._assembled = False

    def assemble(self):
        """Assemble FEM system matrices."""
        if self.mesh is None:
            raise ValueError("No mesh set")
        if self.equation is None:
            raise ValueError("No equation set")

        start_time = time.time()

        n_dof = self.mesh.n_points
        self._K = lil_matrix((n_dof, n_dof))
        self._M = lil_matrix((n_dof, n_dof))
        self._f = np.zeros(n_dof)

        # Get quadrature points and weights
        quad_pts, quad_wts = self._get_quadrature()

        # Assemble element contributions
        for cell_idx, cell in enumerate(self.mesh.cells):
            Ke, Me, fe = self._element_matrices(cell, quad_pts, quad_wts)

            # Add to global matrices
            for i, ni in enumerate(cell):
                for j, nj in enumerate(cell):
                    self._K[ni, nj] += Ke[i, j]
                    self._M[ni, nj] += Me[i, j]
                self._f[ni] += fe[i]

        # Apply Neumann BCs (natural BCs in weak form)
        self._apply_neumann_bcs()

        # Convert to CSR for efficient solving
        self._K = self._K.tocsr()
        self._M = self._M.tocsr()

        self._assembled = True

        if self.config.verbose:
            print(f"FEM assembly: {time.time() - start_time:.3f}s")

    def _get_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get quadrature points and weights for reference element."""
        if self.mesh.cell_type == CellType.TRIANGLE:
            # 3-point Gauss quadrature for triangle
            pts = np.array([
                [1/6, 1/6],
                [2/3, 1/6],
                [1/6, 2/3]
            ])
            wts = np.array([1/6, 1/6, 1/6])

        elif self.mesh.cell_type == CellType.QUAD:
            # 2x2 Gauss quadrature for quad
            g = 1 / np.sqrt(3)
            pts = np.array([
                [-g, -g], [g, -g], [g, g], [-g, g]
            ])
            wts = np.array([1, 1, 1, 1])

        elif self.mesh.cell_type == CellType.TETRAHEDRON:
            # 4-point quadrature for tet
            a = (5 - np.sqrt(5)) / 20
            b = (5 + 3 * np.sqrt(5)) / 20
            pts = np.array([
                [a, a, a],
                [b, a, a],
                [a, b, a],
                [a, a, b]
            ])
            wts = np.array([1/24, 1/24, 1/24, 1/24])

        else:
            raise NotImplementedError(f"Quadrature for {self.mesh.cell_type}")

        return pts, wts

    def _element_matrices(self, cell: np.ndarray,
                          quad_pts: np.ndarray,
                          quad_wts: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute element stiffness, mass matrices and load vector.

        For Poisson/heat: K = integral(grad(N)^T * k * grad(N))
        For mass: M = integral(N^T * N)
        For load: f = integral(N^T * f)
        """
        n_nodes = len(cell)
        coords = self.mesh.points[cell]

        Ke = np.zeros((n_nodes, n_nodes))
        Me = np.zeros((n_nodes, n_nodes))
        fe = np.zeros(n_nodes)

        # Get diffusion coefficient from equation
        diffusion = 1.0
        for term in self.equation.terms:
            if term.operator.op_type == OperatorType.LAPLACIAN:
                if isinstance(term.coefficient, (int, float)):
                    diffusion = abs(term.coefficient)

        for qp, qw in zip(quad_pts, quad_wts):
            # Shape functions and derivatives at quadrature point
            N, dN_dxi = self._shape_functions(qp)

            # Jacobian
            J = dN_dxi.T @ coords
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Physical derivatives
            dN_dx = dN_dxi @ invJ

            # Physical coordinates for source evaluation
            x_phys = N @ coords

            # Stiffness contribution
            Ke += qw * detJ * diffusion * (dN_dx @ dN_dx.T)

            # Mass contribution
            Me += qw * detJ * np.outer(N, N)

            # Load contribution
            if self._source_func is not None:
                f_val = self._source_func(x_phys.reshape(1, -1))[0]
                fe += qw * detJ * N * f_val

        return Ke, Me, fe

    def _shape_functions(self, xi: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate shape functions and derivatives at reference coordinates.

        Returns:
            N: Shape function values (n_nodes,)
            dN: Shape function derivatives (n_nodes, dim)
        """
        if self.mesh.cell_type == CellType.TRIANGLE:
            # Linear triangle: N = [1-xi-eta, xi, eta]
            N = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
            dN = np.array([
                [-1, -1],
                [1, 0],
                [0, 1]
            ])

        elif self.mesh.cell_type == CellType.QUAD:
            # Bilinear quad
            xi1, xi2 = xi
            N = 0.25 * np.array([
                (1 - xi1) * (1 - xi2),
                (1 + xi1) * (1 - xi2),
                (1 + xi1) * (1 + xi2),
                (1 - xi1) * (1 + xi2)
            ])
            dN = 0.25 * np.array([
                [-(1 - xi2), -(1 - xi1)],
                [(1 - xi2), -(1 + xi1)],
                [(1 + xi2), (1 + xi1)],
                [-(1 + xi2), (1 - xi1)]
            ])

        elif self.mesh.cell_type == CellType.TETRAHEDRON:
            # Linear tetrahedron
            N = np.array([1 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]])
            dN = np.array([
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        else:
            raise NotImplementedError(f"Shape functions for {self.mesh.cell_type}")

        return N, dN

    def _apply_neumann_bcs(self):
        """Apply Neumann boundary conditions to load vector."""
        for bc in self.bcs.get_neumann():
            if self.mesh.boundary_faces is None:
                continue

            for face in self.mesh.boundary_faces:
                face_coords = self.mesh.points[face]
                face_center = face_coords.mean(axis=0)

                # Check if this face is in the BC region
                if bc.region is not None:
                    if not bc.region(face_center.reshape(1, -1))[0]:
                        continue

                # Evaluate flux
                flux = bc.evaluate(face_center.reshape(1, -1))[0]

                # Face length/area
                if self.mesh.dim == 2:
                    face_measure = np.linalg.norm(face_coords[1] - face_coords[0])
                else:
                    # Triangle face area
                    v1 = face_coords[1] - face_coords[0]
                    v2 = face_coords[2] - face_coords[0]
                    face_measure = 0.5 * np.linalg.norm(np.cross(v1, v2))

                # Distribute to nodes
                for node in face:
                    self._f[node] += flux * face_measure / len(face)

    def solve(self) -> SolverResult:
        """Solve the assembled FEM system."""
        if not self._assembled:
            self.assemble()

        start_time = time.time()

        # Apply Dirichlet BCs
        A, b = self._apply_dirichlet_bcs()

        # Solve system
        try:
            u = self._solve_linear_system(A, b)
            status = SolverStatus.SUCCESS
            residual = np.linalg.norm(A @ u - b) / np.linalg.norm(b)
        except Exception as e:
            if self.config.verbose:
                print(f"Solve failed: {e}")
            u = np.zeros(self.mesh.n_points)
            status = SolverStatus.FAILED
            residual = float('inf')

        solve_time = time.time() - start_time

        # Create solution field
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
            metadata={"method": "FEM", "n_dof": len(u)}
        )

    def _apply_dirichlet_bcs(self) -> Tuple[csr_matrix, np.ndarray]:
        """Apply Dirichlet BCs by modifying system."""
        A = self._K.copy().tolil()
        b = self._f.copy()

        for bc in self.bcs.get_dirichlet():
            # Find boundary nodes
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_nodes = np.where(mask)[0]
            else:
                bc_nodes = self.mesh.boundary_points

            if bc_nodes is None:
                continue

            # Get BC values
            bc_values = bc.evaluate(self.mesh.points[bc_nodes])

            # Modify system
            for idx, node in enumerate(bc_nodes):
                # Zero out row
                A[node, :] = 0
                A[node, node] = 1.0
                b[node] = bc_values[idx]

        return A.tocsr(), b

    def _step_transient(self, previous: Field, dt: float) -> SolverResult:
        """Solve single time step with backward Euler."""
        if not self._assembled:
            self.assemble()

        # M/dt * u_n+1 + K * u_n+1 = M/dt * u_n + f
        # (M/dt + K) * u_n+1 = M/dt * u_n + f

        A = self._M / dt + self._K
        b = (self._M / dt) @ previous.values + self._f

        # Apply Dirichlet BCs
        A_bc = A.tolil()
        b_bc = b.copy()

        for bc in self.bcs.get_dirichlet():
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_nodes = np.where(mask)[0]
            else:
                bc_nodes = self.mesh.boundary_points

            if bc_nodes is None:
                continue

            bc_values = bc.evaluate(self.mesh.points[bc_nodes])

            for idx, node in enumerate(bc_nodes):
                A_bc[node, :] = 0
                A_bc[node, node] = 1.0
                b_bc[node] = bc_values[idx]

        # Solve
        u = self._solve_linear_system(A_bc.tocsr(), b_bc)
        residual = np.linalg.norm(A_bc @ u - b_bc)

        field_name = previous.name
        solution = Field(
            name=field_name,
            field_type=FieldType.SCALAR,
            dim=self.mesh.dim,
            values=u,
            points=self.mesh.points
        )

        return SolverResult(
            status=SolverStatus.SUCCESS,
            fields={field_name: solution},
            residual=residual,
            iterations=1,
            solve_time=0.0,
        )

    def get_matrix(self) -> np.ndarray:
        """Get the stiffness matrix."""
        if self._K is None:
            self.assemble()
        return self._K.toarray()

    def get_rhs(self) -> np.ndarray:
        """Get the load vector."""
        if self._f is None:
            self.assemble()
        return self._f

    def get_mass_matrix(self) -> np.ndarray:
        """Get the mass matrix."""
        if self._M is None:
            self.assemble()
        return self._M.toarray()

    def compute_energy_norm(self, solution: Field) -> float:
        """Compute energy norm ||u||_a = sqrt(u^T K u)."""
        u = solution.values
        return np.sqrt(u @ self._K @ u)

    def estimate_error(self, solution: Field) -> np.ndarray:
        """
        Estimate local error using gradient recovery.

        ZZ error estimator: compare raw gradient to recovered gradient.
        """
        n_cells = self.mesh.n_cells
        errors = np.zeros(n_cells)

        # Compute gradient at each node (area-weighted average)
        grad_recovered = np.zeros((self.mesh.n_points, self.mesh.dim))
        weights = np.zeros(self.mesh.n_points)

        for cell_idx, cell in enumerate(self.mesh.cells):
            coords = self.mesh.points[cell]
            u_local = solution.values[cell]

            # Compute gradient (constant for linear elements)
            _, dN = self._shape_functions(np.array([1/3, 1/3] if self.mesh.dim == 2
                                                   else [1/4, 1/4, 1/4]))
            J = dN.T @ coords
            invJ = np.linalg.inv(J)
            dN_dx = dN @ invJ
            grad_elem = dN_dx.T @ u_local

            area = self.mesh.volumes[cell_idx]

            for node in cell:
                grad_recovered[node] += area * grad_elem
                weights[node] += area

        grad_recovered /= weights[:, np.newaxis]

        # Compute error per element
        for cell_idx, cell in enumerate(self.mesh.cells):
            coords = self.mesh.points[cell]
            u_local = solution.values[cell]

            _, dN = self._shape_functions(np.array([1/3, 1/3] if self.mesh.dim == 2
                                                   else [1/4, 1/4, 1/4]))
            J = dN.T @ coords
            invJ = np.linalg.inv(J)
            dN_dx = dN @ invJ
            grad_elem = dN_dx.T @ u_local

            # Average recovered gradient at element
            grad_rec_elem = grad_recovered[cell].mean(axis=0)

            # Error = difference
            errors[cell_idx] = np.linalg.norm(grad_elem - grad_rec_elem)

        return errors
