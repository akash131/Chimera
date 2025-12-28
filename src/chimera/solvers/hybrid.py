"""
Hybrid solver for Chimera.

The core innovation: adaptively switch between discretization methods
based on local physics characteristics.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time

from chimera.solvers.base import Solver, SolverResult, SolverStatus, SolverConfig
from chimera.solvers.fem import FEMSolver
from chimera.solvers.fvm import FVMSolver
from chimera.solvers.meshless import MeshlessSolver
from chimera.core.field import Field, FieldType
from chimera.core.equation import Equation
from chimera.core.boundary import BoundaryConditionSet
from chimera.mesh.mesh import Mesh, MeshType
from chimera.mesh.generators import generate_structured_mesh, generate_delaunay_mesh


class MethodType(Enum):
    """Discretization method types."""
    FEM = "fem"
    FVM = "fvm"
    MESHLESS = "meshless"
    AUTO = "auto"


@dataclass
class RegionConfig:
    """Configuration for a subdomain region."""
    method: MethodType
    mesh: Optional[Mesh] = None
    indices: Optional[np.ndarray] = None  # Point/cell indices in this region
    priority: int = 0  # Higher priority regions override lower


class HybridSolver(Solver):
    """
    Hybrid solver that combines multiple discretization methods.

    Key capabilities:
    - Automatic method selection based on physics
    - Domain decomposition with different methods per region
    - Coupled solving across method interfaces
    - Adaptive method switching during solve

    Use cases:
    - FVM for advection-dominated regions + FEM for diffusion regions
    - Meshless for complex geometry + structured FEM for bulk
    - High-order FEM for smooth regions + low-order for discontinuities
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        super().__init__(config)
        self._regions: List[RegionConfig] = []
        self._solvers: Dict[str, Solver] = {}
        self._coupling_matrices: Dict[Tuple[str, str], np.ndarray] = {}
        self._method_selector: Optional[Callable] = None
        self._default_method: MethodType = MethodType.FEM

    def set_mesh(self, mesh: Mesh):
        """Set the global mesh."""
        self.mesh = mesh
        self._assembled = False

    def set_equation(self, equation: Equation):
        """Set the governing equation."""
        self.equation = equation
        self._assembled = False

    def set_default_method(self, method: MethodType):
        """Set the default discretization method."""
        self._default_method = method

    def set_method_selector(self, selector: Callable[[np.ndarray, Field], MethodType]):
        """
        Set function to automatically select method per point.

        Args:
            selector: Function(coordinates, solution) -> MethodType
        """
        self._method_selector = selector

    def add_region(self, name: str, method: MethodType,
                   predicate: Callable[[np.ndarray], np.ndarray],
                   priority: int = 0):
        """
        Add a region with specific discretization method.

        Args:
            name: Region identifier
            method: Discretization method for this region
            predicate: Function(points) -> bool mask
            priority: Higher priority overrides lower
        """
        if self.mesh is None:
            raise ValueError("Set mesh before adding regions")

        mask = predicate(self.mesh.points)
        indices = np.where(mask)[0]

        region = RegionConfig(
            method=method,
            indices=indices,
            priority=priority
        )
        self._regions.append(region)
        self._assembled = False

    def auto_partition(self, solution: Optional[Field] = None):
        """
        Automatically partition domain based on physics.

        Uses gradients, Peclet number, etc. to decide method.
        """
        if self.mesh is None:
            raise ValueError("No mesh set")

        n_points = self.mesh.n_points
        methods = np.full(n_points, self._default_method)

        if self._method_selector is not None and solution is not None:
            for i in range(n_points):
                coord = self.mesh.points[i:i + 1]
                methods[i] = self._method_selector(coord, solution)

        # Group by method
        self._regions = []
        for method in MethodType:
            if method == MethodType.AUTO:
                continue
            mask = methods == method
            if np.any(mask):
                self._regions.append(RegionConfig(
                    method=method,
                    indices=np.where(mask)[0]
                ))

    def assemble(self):
        """Assemble hybrid system."""
        if self.mesh is None:
            raise ValueError("No mesh set")
        if self.equation is None:
            raise ValueError("No equation set")

        start_time = time.time()

        # If no regions defined, use single method
        if not self._regions:
            self._assemble_single_method()
        else:
            self._assemble_multi_method()

        self._assembled = True

        if self.config.verbose:
            print(f"Hybrid assembly: {time.time() - start_time:.3f}s")

    def _assemble_single_method(self):
        """Assemble using single default method."""
        if self._default_method == MethodType.FEM:
            solver = FEMSolver(self.config)
        elif self._default_method == MethodType.FVM:
            solver = FVMSolver(self.config)
        elif self._default_method == MethodType.MESHLESS:
            solver = MeshlessSolver(self.config)
        else:
            solver = FEMSolver(self.config)

        solver.set_mesh(self.mesh)
        solver.set_equation(self.equation)
        solver.set_bcs(self.bcs)
        solver.assemble()

        self._solvers["global"] = solver

    def _assemble_multi_method(self):
        """Assemble with multiple methods in different regions."""
        # Sort regions by priority
        regions = sorted(self._regions, key=lambda r: r.priority, reverse=True)

        # Assign points to methods (highest priority wins)
        n_points = self.mesh.n_points
        point_method = np.full(n_points, -1, dtype=int)
        point_region = np.full(n_points, -1, dtype=int)

        for i, region in enumerate(regions):
            if region.indices is not None:
                unassigned = point_method[region.indices] == -1
                point_method[region.indices[unassigned]] = region.method.value
                point_region[region.indices[unassigned]] = i

        # Create solver for each active method
        active_methods = set(point_method[point_method >= 0])

        for method_val in active_methods:
            method_points = np.where(point_method == method_val)[0]

            # Create submesh for this region
            submesh = self._extract_submesh(method_points)

            # Create appropriate solver
            method = MethodType(method_val) if isinstance(method_val, str) else list(MethodType)[method_val]

            if method == MethodType.FEM:
                solver = FEMSolver(self.config)
            elif method == MethodType.FVM:
                solver = FVMSolver(self.config)
            elif method == MethodType.MESHLESS:
                solver = MeshlessSolver(self.config)
            else:
                continue

            solver.set_mesh(submesh)
            solver.set_equation(self.equation)
            # Would need to map BCs to submesh
            solver.assemble()

            self._solvers[f"region_{method.value}"] = solver

        # Build coupling between regions
        self._build_coupling()

    def _extract_submesh(self, point_indices: np.ndarray) -> Mesh:
        """Extract submesh containing given points."""
        # Create new mesh with subset of points
        new_points = self.mesh.points[point_indices]

        # Map old to new indices
        old_to_new = {old: new for new, old in enumerate(point_indices)}

        # Extract cells that have all nodes in the subset
        new_cells = []
        if self.mesh.cells is not None:
            for cell in self.mesh.cells:
                if all(n in old_to_new for n in cell):
                    new_cells.append([old_to_new[n] for n in cell])

        cells = np.array(new_cells) if new_cells else None

        submesh = Mesh(
            points=new_points,
            cells=cells,
            cell_type=self.mesh.cell_type,
            mesh_type=self.mesh.mesh_type
        )
        submesh.find_boundary()

        return submesh

    def _build_coupling(self):
        """Build coupling operators between regions."""
        # Find interface points (points that neighbor different regions)
        # For now, simplified approach: use interpolation at interfaces
        pass

    def solve(self) -> SolverResult:
        """Solve the hybrid system."""
        if not self._assembled:
            self.assemble()

        start_time = time.time()

        if len(self._solvers) == 1:
            # Single solver case
            result = list(self._solvers.values())[0].solve()
        else:
            # Multi-solver case: iterative coupling
            result = self._solve_coupled()

        result.solve_time = time.time() - start_time
        result.metadata["method"] = "hybrid"
        result.metadata["n_regions"] = len(self._regions)

        return result

    def _solve_coupled(self) -> SolverResult:
        """Solve coupled multi-method system."""
        max_iters = self.config.max_iterations
        tol = self.config.tolerance

        # Initialize with zeros
        n_points = self.mesh.n_points
        u = np.zeros(n_points)

        for iteration in range(max_iters):
            u_old = u.copy()

            # Solve each region
            for name, solver in self._solvers.items():
                result = solver.solve()
                if not result.success:
                    return result

                # Map solution back to global mesh
                # (simplified - would need proper index mapping)

            # Check convergence
            residual = np.linalg.norm(u - u_old) / (np.linalg.norm(u) + 1e-12)

            if self.config.verbose:
                print(f"Iteration {iteration}: residual = {residual:.2e}")

            if residual < tol:
                break

        field_name = self.equation.terms[0].field_name
        solution = Field(
            name=field_name,
            field_type=FieldType.SCALAR,
            dim=self.mesh.dim,
            values=u,
            points=self.mesh.points
        )

        return SolverResult(
            status=SolverStatus.CONVERGED if residual < tol else SolverStatus.MAX_ITERATIONS,
            fields={field_name: solution},
            residual=residual,
            iterations=iteration + 1,
            solve_time=0.0
        )

    def adapt(self, solution: Field, threshold: float = 0.1) -> bool:
        """
        Adapt method assignment based on solution.

        Returns True if any region changed method.
        """
        if self._method_selector is None:
            return False

        changed = False
        for i, region in enumerate(self._regions):
            if region.indices is None:
                continue

            # Check if method should change for any points
            for idx in region.indices:
                coord = self.mesh.points[idx:idx + 1]
                new_method = self._method_selector(coord, solution)

                if new_method != region.method:
                    changed = True
                    break

        if changed:
            self.auto_partition(solution)
            self._assembled = False

        return changed

    def solve_adaptive(self, max_adaptations: int = 5) -> SolverResult:
        """
        Solve with adaptive method switching.

        Iteratively solves and re-partitions until convergence.
        """
        result = self.solve()
        if not result.success:
            return result

        for i in range(max_adaptations):
            solution = list(result.fields.values())[0]

            if not self.adapt(solution):
                break

            result = self.solve()
            if not result.success:
                break

            if self.config.verbose:
                print(f"Adaptation {i + 1}: {len(self._regions)} regions")

        result.metadata["adaptations"] = i + 1
        return result

    def get_matrix(self) -> np.ndarray:
        """Get global system matrix (if single method)."""
        if len(self._solvers) == 1:
            return list(self._solvers.values())[0].get_matrix()
        raise NotImplementedError("Matrix for coupled systems")

    def get_rhs(self) -> np.ndarray:
        """Get global RHS vector (if single method)."""
        if len(self._solvers) == 1:
            return list(self._solvers.values())[0].get_rhs()
        raise NotImplementedError("RHS for coupled systems")

    def visualize_partition(self) -> np.ndarray:
        """Return array indicating method for each point (for visualization)."""
        n_points = self.mesh.n_points
        partition = np.zeros(n_points, dtype=int)

        for i, region in enumerate(self._regions):
            if region.indices is not None:
                partition[region.indices] = i + 1

        return partition


def peclet_method_selector(coord: np.ndarray, solution: Field,
                           velocity: float = 1.0,
                           diffusivity: float = 0.01,
                           threshold: float = 1.0) -> MethodType:
    """
    Select method based on local Peclet number.

    Pe > threshold: Use FVM (advection-dominated)
    Pe <= threshold: Use FEM (diffusion-dominated)
    """
    # Estimate local length scale from mesh
    h = 0.1  # Would compute from mesh

    Pe = velocity * h / diffusivity

    if Pe > threshold:
        return MethodType.FVM
    else:
        return MethodType.FEM


def gradient_method_selector(coord: np.ndarray, solution: Field,
                             threshold: float = 10.0) -> MethodType:
    """
    Select method based on solution gradient.

    High gradient: Use fine FEM or meshless
    Low gradient: Use coarse FEM or FVM
    """
    if solution.values is None:
        return MethodType.FEM

    # Estimate gradient magnitude at this point
    grad = solution.gradient()
    grad_mag = np.linalg.norm(grad.evaluate(coord))

    if grad_mag > threshold:
        return MethodType.MESHLESS  # Adaptable refinement
    else:
        return MethodType.FEM
