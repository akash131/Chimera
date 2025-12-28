"""
Finite Volume Method (FVM) solver for Chimera.

Implements cell-centered FVM for conservation laws.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable, List
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time

from chimera.solvers.base import Solver, SolverResult, SolverStatus, SolverConfig
from chimera.core.field import Field, FieldType
from chimera.core.equation import Equation, OperatorType
from chimera.core.boundary import DirichletBC, NeumannBC
from chimera.mesh.mesh import Mesh, MeshType


class FVMSolver(Solver):
    """
    Finite Volume Method solver.

    Implements cell-centered FVM with:
    - Various flux schemes (central, upwind, TVD)
    - Structured and unstructured meshes
    - Explicit and implicit time integration

    Best for:
    - Conservation laws
    - Advection-dominated problems
    - Shock capturing
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._A: Optional[csr_matrix] = None
        self._b: Optional[np.ndarray] = None
        self._source_func: Optional[Callable] = None
        self._diffusivity: float = 1.0
        self._velocity: Optional[np.ndarray] = None
        self._flux_scheme: str = "central"  # 'central', 'upwind', 'tvd'

    def set_mesh(self, mesh: Mesh):
        """Set the computational mesh."""
        if mesh.mesh_type not in [MeshType.CARTESIAN, MeshType.UNSTRUCTURED]:
            raise ValueError("FVM requires Cartesian or unstructured mesh")

        self.mesh = mesh
        self._assembled = False

        # Ensure neighbor info is available
        if mesh._neighbors is None:
            mesh.find_neighbors(k=2 * mesh.dim)

    def set_equation(self, equation: Equation):
        """Set the governing equation."""
        self.equation = equation
        self._assembled = False

        # Extract coefficients
        for term in equation.terms:
            if term.operator.op_type == OperatorType.LAPLACIAN:
                if isinstance(term.coefficient, (int, float)):
                    self._diffusivity = abs(term.coefficient)

    def set_source(self, source: Callable[[np.ndarray], np.ndarray]):
        """Set source term function."""
        self._source_func = source
        self._assembled = False

    def set_velocity(self, velocity: np.ndarray):
        """Set velocity field for advection."""
        self._velocity = velocity

    def set_flux_scheme(self, scheme: str):
        """Set flux discretization scheme."""
        if scheme not in ['central', 'upwind', 'tvd']:
            raise ValueError(f"Unknown flux scheme: {scheme}")
        self._flux_scheme = scheme
        self._assembled = False

    def assemble(self):
        """Assemble FVM system."""
        if self.mesh is None:
            raise ValueError("No mesh set")
        if self.equation is None:
            raise ValueError("No equation set")

        start_time = time.time()

        n_cells = self.mesh.n_points  # Cell-centered: cells = points
        self._A = lil_matrix((n_cells, n_cells))
        self._b = np.zeros(n_cells)

        # Get mesh spacing (for Cartesian)
        if hasattr(self.mesh, '_resolution'):
            dx = [(b[1] - b[0]) / r
                  for b, r in zip([(0, 1), (0, 1), (0, 1)][:self.mesh.dim],
                                  self.mesh._resolution)]
        else:
            # Estimate from neighbor distances
            dx = [0.1] * self.mesh.dim  # Fallback

        # Assemble diffusion operator
        self._assemble_diffusion(dx)

        # Assemble advection operator if velocity field present
        if self._velocity is not None:
            self._assemble_advection(dx)

        # Add source term
        if self._source_func is not None:
            self._b += self._source_func(self.mesh.points)

        self._A = self._A.tocsr()
        self._assembled = True

        if self.config.verbose:
            print(f"FVM assembly: {time.time() - start_time:.3f}s")

    def _assemble_diffusion(self, dx: List[float]):
        """Assemble diffusion operator using finite differences."""
        n_cells = self.mesh.n_points
        neighbors = self.mesh._neighbors
        dim = self.mesh.dim

        for i in range(n_cells):
            diag = 0.0

            for d in range(dim):
                # Face area and distance (simplified for Cartesian)
                area = 1.0
                for d2 in range(dim):
                    if d2 != d:
                        area *= dx[d2]

                dist = dx[d]
                diff_coeff = self._diffusivity * area / dist

                # Left neighbor (index 2*d)
                left = neighbors[i, 2 * d]
                if left >= 0:
                    self._A[i, left] -= diff_coeff
                    diag += diff_coeff
                # else: boundary - handled separately

                # Right neighbor (index 2*d + 1)
                right = neighbors[i, 2 * d + 1]
                if right >= 0:
                    self._A[i, right] -= diff_coeff
                    diag += diff_coeff

            self._A[i, i] = diag

    def _assemble_advection(self, dx: List[float]):
        """Assemble advection operator."""
        n_cells = self.mesh.n_points
        neighbors = self.mesh._neighbors
        dim = self.mesh.dim

        for i in range(n_cells):
            vel = self._velocity[i] if len(self._velocity.shape) > 1 else self._velocity

            for d in range(dim):
                # Face area
                area = 1.0
                for d2 in range(dim):
                    if d2 != d:
                        area *= dx[d2]

                v_d = vel[d] if hasattr(vel, '__len__') else vel

                # Upwind scheme
                if self._flux_scheme == 'upwind':
                    left = neighbors[i, 2 * d]
                    right = neighbors[i, 2 * d + 1]

                    if v_d > 0:
                        # Flow from left
                        if left >= 0:
                            self._A[i, left] -= v_d * area
                            self._A[i, i] += v_d * area
                    else:
                        # Flow from right
                        if right >= 0:
                            self._A[i, right] += v_d * area
                            self._A[i, i] -= v_d * area

                elif self._flux_scheme == 'central':
                    # Central difference
                    left = neighbors[i, 2 * d]
                    right = neighbors[i, 2 * d + 1]

                    if left >= 0 and right >= 0:
                        self._A[i, right] += 0.5 * v_d * area
                        self._A[i, left] -= 0.5 * v_d * area

    def solve(self) -> SolverResult:
        """Solve the assembled FVM system."""
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
            metadata={"method": "FVM", "flux_scheme": self._flux_scheme}
        )

    def _apply_boundary_conditions(self) -> Tuple[csr_matrix, np.ndarray]:
        """Apply boundary conditions."""
        A = self._A.copy().tolil()
        b = self._b.copy()

        boundary_cells = self.mesh.boundary_points
        if boundary_cells is None:
            return A.tocsr(), b

        for bc in self.bcs.get_dirichlet():
            # Find cells in this BC region
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_cells = np.where(mask)[0]
            else:
                bc_cells = boundary_cells

            bc_values = bc.evaluate(self.mesh.points[bc_cells])

            for idx, cell in enumerate(bc_cells):
                # Ghost cell approach: set diagonal to 1, RHS to BC value
                A[cell, :] = 0
                A[cell, cell] = 1.0
                b[cell] = bc_values[idx]

        for bc in self.bcs.get_neumann():
            # Neumann: modify flux at boundary
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_cells = np.where(mask)[0]
            else:
                bc_cells = boundary_cells

            flux_values = bc.evaluate(self.mesh.points[bc_cells])

            # Find which face is at boundary and add flux
            neighbors = self.mesh._neighbors
            for idx, cell in enumerate(bc_cells):
                for d in range(2 * self.mesh.dim):
                    if neighbors[cell, d] < 0:
                        # This is a boundary face
                        b[cell] += flux_values[idx]
                        break

        return A.tocsr(), b

    def _step_transient(self, previous: Field, dt: float) -> SolverResult:
        """Solve single time step."""
        if not self._assembled:
            self.assemble()

        n_cells = self.mesh.n_points

        # Explicit Euler: u_n+1 = u_n + dt * (b - A*u_n) / V
        # where V is cell volume
        volumes = np.ones(n_cells)  # Simplified

        rhs = self._b - self._A @ previous.values
        u_new = previous.values + dt * rhs / volumes

        # Apply Dirichlet BCs
        for bc in self.bcs.get_dirichlet():
            if bc.region is not None:
                mask = bc.applies_to(self.mesh.points)
                bc_cells = np.where(mask)[0]
            else:
                bc_cells = self.mesh.boundary_points

            if bc_cells is not None:
                u_new[bc_cells] = bc.evaluate(self.mesh.points[bc_cells])

        field_name = previous.name
        solution = Field(
            name=field_name,
            field_type=FieldType.SCALAR,
            dim=self.mesh.dim,
            values=u_new,
            points=self.mesh.points
        )

        return SolverResult(
            status=SolverStatus.SUCCESS,
            fields={field_name: solution},
            residual=0.0,
            iterations=1,
            solve_time=0.0,
        )

    def get_matrix(self) -> np.ndarray:
        if self._A is None:
            self.assemble()
        return self._A.toarray()

    def get_rhs(self) -> np.ndarray:
        if self._b is None:
            self.assemble()
        return self._b

    def compute_flux(self, solution: Field, face: int) -> float:
        """Compute flux through a face."""
        # Simplified flux computation
        raise NotImplementedError("Per-face flux computation")

    def check_conservation(self, solution: Field) -> float:
        """Check global conservation (sum of fluxes should be zero)."""
        if self._A is None:
            return float('inf')

        # Conservation check: residual should be small
        residual = self._A @ solution.values - self._b
        return np.sum(np.abs(residual))
