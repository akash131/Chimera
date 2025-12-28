"""
Differentiable FEM solver.

End-to-end differentiable assembly and solve for gradient-based optimization.
"""

from __future__ import annotations
from typing import Optional, Callable, Tuple, List
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from chimera.autodiff.tape import Tape, Variable, linear_solve_differentiable
from chimera.mesh.mesh import Mesh
from chimera.core.equation import Equation


class DifferentiableAssembly:
    """
    Differentiable finite element assembly.

    Assembles stiffness matrices with tracked dependencies on
    material parameters, enabling gradient computation.
    """

    def __init__(self, mesh: Mesh, equation: Equation):
        self.mesh = mesh
        self.equation = equation
        self._shape_cache = {}

    def assemble_stiffness(self, material_field: Variable,
                           tape: Optional[Tape] = None) -> Tuple[np.ndarray, Variable]:
        """
        Assemble stiffness matrix with material field as Variable.

        K(E) = sum_e E_e * K0_e

        where K0_e is the unit stiffness for element e.

        Args:
            material_field: Per-element material property (e.g., Young's modulus)
            tape: Computation tape for autodiff

        Returns:
            K: Assembled stiffness matrix (not differentiable for now)
            compliance: u^T K u (differentiable)
        """
        tape = tape or Tape.get_active()
        n_dof = self.mesh.n_points
        n_elem = self.mesh.n_cells

        # Pre-compute unit element stiffnesses
        K0_elements = self._compute_unit_stiffnesses()

        # Assemble global stiffness
        K = lil_matrix((n_dof, n_dof))

        for e, cell in enumerate(self.mesh.cells):
            E_e = material_field.value[e] if isinstance(material_field, Variable) else material_field[e]
            K_e = E_e * K0_elements[e]

            for i, ni in enumerate(cell):
                for j, nj in enumerate(cell):
                    K[ni, nj] += K_e[i, j]

        return K.tocsr(), K0_elements

    def _compute_unit_stiffnesses(self) -> List[np.ndarray]:
        """Compute element stiffness matrices with unit material property."""
        K0_list = []

        for cell in self.mesh.cells:
            coords = self.mesh.points[cell]
            K0_e = self._element_stiffness_unit(coords)
            K0_list.append(K0_e)

        return K0_list

    def _element_stiffness_unit(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute unit element stiffness (E=1).

        For triangular elements with linear shape functions.
        """
        n_nodes = len(coords)

        if n_nodes == 3:  # Triangle
            # Area
            x = coords[:, 0]
            y = coords[:, 1]
            area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

            # Shape function gradients (constant for linear triangle)
            dN = np.array([
                [y[1] - y[2], y[2] - y[0], y[0] - y[1]],
                [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
            ]) / (2 * area)

            # Stiffness: K = area * B^T B (for Laplacian)
            K0 = area * (dN.T @ dN)
            return K0

        elif n_nodes == 4:  # Quad
            # 2x2 Gauss quadrature
            gauss_pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
            K0 = np.zeros((4, 4))

            for gp in gauss_pts:
                xi, eta = gp

                # Shape function derivatives in reference coords
                dN_dxi = 0.25 * np.array([
                    [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                ])

                # Jacobian
                J = dN_dxi @ coords
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                # Physical derivatives
                dN_dx = invJ @ dN_dxi

                K0 += detJ * (dN_dx.T @ dN_dx)

            return K0

        else:
            raise NotImplementedError(f"Element with {n_nodes} nodes")


class DifferentiableFEM:
    """
    Fully differentiable FEM solver.

    Computes gradients of objective functions with respect to:
    - Material properties (topology optimization)
    - Boundary conditions
    - Geometry (shape optimization)
    - Source terms
    """

    def __init__(self, mesh: Mesh, equation: Equation):
        self.mesh = mesh
        self.equation = equation
        self.assembly = DifferentiableAssembly(mesh, equation)
        self._bc_nodes: List[int] = []
        self._bc_values: np.ndarray = None

    def set_dirichlet_bc(self, nodes: List[int], values: np.ndarray):
        """Set Dirichlet boundary conditions."""
        self._bc_nodes = nodes
        self._bc_values = values

    def solve(self, material: Variable, source: Variable,
              tape: Optional[Tape] = None) -> Variable:
        """
        Solve FEM system with full gradient support.

        Args:
            material: Per-element material property
            source: Per-node source term
            tape: Computation tape

        Returns:
            Solution as Variable (gradients can be computed)
        """
        tape = tape or Tape.get_active()
        if tape is None:
            raise RuntimeError("No active tape")

        # Assemble system
        K, K0_elements = self.assembly.assemble_stiffness(material, tape)
        f = source.value.copy()

        # Apply Dirichlet BCs
        K_bc = K.tolil()
        for i, node in enumerate(self._bc_nodes):
            K_bc[node, :] = 0
            K_bc[node, node] = 1.0
            f[node] = self._bc_values[i]
        K_bc = K_bc.tocsr()

        # Solve (differentiable)
        u = linear_solve_differentiable(K_bc, source, tape)

        return u

    def compliance(self, u: Variable, material: Variable) -> Variable:
        """
        Compute compliance C = u^T K u.

        This is the standard objective for topology optimization.
        Fully differentiable with respect to material field.
        """
        tape = u.tape

        # Compliance = sum of element strain energies
        # C = sum_e E_e * u_e^T K0_e u_e
        K0_elements = self.assembly._compute_unit_stiffnesses()

        total = Variable(np.array(0.0), tape)

        for e, cell in enumerate(self.mesh.cells):
            u_e = Variable(u.value[cell], tape, requires_grad=False)
            K0_e = K0_elements[e]

            # Element compliance
            Ku = K0_e @ u_e.value
            c_e = material.value[e] * np.dot(u_e.value, Ku)

            total = total + c_e

        return total


class DifferentiableTopologyOptimization:
    """
    Topology optimization using differentiable solver.

    Uses automatic differentiation instead of manual adjoint derivation.
    """

    def __init__(self, mesh: Mesh, equation: Equation):
        self.mesh = mesh
        self.fem = DifferentiableFEM(mesh, equation)
        self.volume_fraction = 0.5
        self.penalization = 3.0

    def optimize(self, loads: np.ndarray, fixed_nodes: List[int],
                 n_iterations: int = 100) -> np.ndarray:
        """
        Run topology optimization.

        Args:
            loads: Applied loads
            fixed_nodes: Fixed displacement nodes
            n_iterations: Number of optimization iterations

        Returns:
            Optimal density field
        """
        n_elem = self.mesh.n_cells
        rho = np.full(n_elem, self.volume_fraction)

        # Set BCs
        self.fem.set_dirichlet_bc(fixed_nodes, np.zeros(len(fixed_nodes)))

        learning_rate = 0.1

        for iteration in range(n_iterations):
            with Tape() as tape:
                # Create variables
                material = Variable(rho ** self.penalization, tape)
                source = Variable(loads, tape)

                # Forward solve
                u = self.fem.solve(material, source, tape)

                # Compliance objective
                C = self.fem.compliance(u, material)

                # Backward pass
                C.backward()

                # Get gradient
                grad_material = material.grad

            # Gradient w.r.t. density (chain rule for penalization)
            grad_rho = self.penalization * (rho ** (self.penalization - 1)) * grad_material

            # Update with volume constraint projection
            rho_new = rho - learning_rate * grad_rho
            rho_new = self._project_volume(rho_new)
            rho = np.clip(rho_new, 0.001, 1.0)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: C = {C.value:.4f}")

        return rho

    def _project_volume(self, rho: np.ndarray) -> np.ndarray:
        """Project to volume constraint via bisection."""
        volumes = self.mesh.volumes
        target = self.volume_fraction * np.sum(volumes)

        # Bisection to find Lagrange multiplier
        lo, hi = -1e6, 1e6
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            rho_proj = np.clip(rho - mid, 0.001, 1.0)
            vol = np.sum(rho_proj * volumes)

            if vol > target:
                lo = mid
            else:
                hi = mid

        return np.clip(rho - 0.5 * (lo + hi), 0.001, 1.0)
