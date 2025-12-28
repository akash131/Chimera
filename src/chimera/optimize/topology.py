"""
Topology optimization for Chimera.

Solver-in-the-loop optimization for structural design.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List
import numpy as np

from chimera.core.field import Field
from chimera.mesh.mesh import Mesh
from chimera.solvers.base import Solver, SolverResult


@dataclass
class TopologyConfig:
    """Configuration for topology optimization."""
    volume_fraction: float = 0.5  # Target volume fraction
    penalization: float = 3.0  # SIMP penalization power
    filter_radius: float = 0.05  # Density filter radius
    min_density: float = 1e-3  # Minimum density (avoid singularity)
    max_iterations: int = 100
    convergence_tol: float = 1e-3
    move_limit: float = 0.2  # OC move limit


class SIMPMethod:
    """
    Solid Isotropic Material with Penalization (SIMP).

    Classic topology optimization method with density-based design variables.
    """

    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()

    def penalize(self, rho: np.ndarray) -> np.ndarray:
        """Apply SIMP penalization: E(rho) = rho^p * E0."""
        return np.maximum(rho, self.config.min_density) ** self.config.penalization

    def sensitivity(self, rho: np.ndarray, strain_energy: np.ndarray) -> np.ndarray:
        """
        Compute design sensitivity.

        dC/drho = -p * rho^(p-1) * u^T K0 u
        """
        p = self.config.penalization
        return -p * rho ** (p - 1) * strain_energy


class TopologyOptimizer:
    """
    Topology optimization with actual solver in the loop.

    This is the "solver-in-the-loop" paradigm - using real physics
    evaluations rather than surrogate models during optimization.
    """

    def __init__(self, solver: Solver, config: Optional[TopologyConfig] = None):
        self.solver = solver
        self.config = config or TopologyConfig()
        self.simp = SIMPMethod(self.config)
        self._history: List[dict] = []
        self._filter_weights: Optional[np.ndarray] = None

    def setup(self, mesh: Mesh, loads: np.ndarray,
              fixed_nodes: np.ndarray):
        """
        Setup optimization problem.

        Args:
            mesh: Design domain mesh
            loads: Applied loads
            fixed_nodes: Nodes with fixed displacement
        """
        self.mesh = mesh
        self.loads = loads
        self.fixed_nodes = fixed_nodes

        # Initialize design variables (uniform density)
        self.n_elements = mesh.n_cells if mesh.cells is not None else mesh.n_points
        self.rho = np.full(self.n_elements, self.config.volume_fraction)

        # Build density filter
        self._build_filter()

    def _build_filter(self):
        """Build density filter for minimum length scale."""
        centroids = self.mesh.centroids
        n = len(centroids)
        r = self.config.filter_radius

        # Weight matrix for filtering
        self._filter_weights = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < r:
                    self._filter_weights[i, j] = r - dist

        # Normalize rows
        row_sums = self._filter_weights.sum(axis=1, keepdims=True)
        self._filter_weights /= np.maximum(row_sums, 1e-12)

    def filter_density(self, rho: np.ndarray) -> np.ndarray:
        """Apply density filter."""
        return self._filter_weights @ rho

    def filter_sensitivity(self, rho: np.ndarray,
                           dc: np.ndarray) -> np.ndarray:
        """Apply chain rule for filtered sensitivity."""
        return self._filter_weights.T @ (dc * rho) / np.maximum(rho, 1e-12)

    def objective(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate compliance objective and gradient.

        This is where the solver is called - actual physics evaluation.
        """
        # Filter density
        rho_filtered = self.filter_density(rho)

        # Apply SIMP penalization
        E_factor = self.simp.penalize(rho_filtered)

        # Update material properties in solver
        # (simplified - would modify stiffness matrix)

        # Solve forward problem
        result = self.solver.solve()

        if not result.success:
            return float('inf'), np.zeros_like(rho)

        # Get displacement field
        u = list(result.fields.values())[0].values

        # Compute compliance: C = f^T u = u^T K u
        compliance = np.dot(self.loads.flatten(), u.flatten())

        # Compute element strain energies for sensitivity
        strain_energy = self._compute_element_strain_energy(u)

        # Sensitivity
        dc = self.simp.sensitivity(rho_filtered, strain_energy)
        dc = self.filter_sensitivity(rho_filtered, dc)

        return compliance, dc

    def _compute_element_strain_energy(self, u: np.ndarray) -> np.ndarray:
        """Compute strain energy per element."""
        # Simplified - would compute u_e^T K_e u_e for each element
        n = self.n_elements
        return np.ones(n)  # Placeholder

    def volume_constraint(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """Volume constraint: sum(rho*v) / sum(v) - vf = 0."""
        volumes = self.mesh.volumes if hasattr(self.mesh, 'volumes') else np.ones(len(rho))
        vol_current = np.sum(rho * volumes) / np.sum(volumes)
        g = vol_current - self.config.volume_fraction
        dg = volumes / np.sum(volumes)
        return g, dg

    def optimize(self) -> np.ndarray:
        """
        Run topology optimization.

        Uses Optimality Criteria (OC) method for efficiency.
        """
        rho = self.rho.copy()

        for iteration in range(self.config.max_iterations):
            # Evaluate objective and constraint
            c, dc = self.objective(rho)
            g, dg = self.volume_constraint(rho)

            # Store history
            self._history.append({
                'iteration': iteration,
                'compliance': c,
                'volume': g + self.config.volume_fraction
            })

            # Check convergence
            if iteration > 0:
                change = np.max(np.abs(rho - rho_old))
                if change < self.config.convergence_tol:
                    print(f"Converged at iteration {iteration}")
                    break

            rho_old = rho.copy()

            # OC update
            rho = self._oc_update(rho, dc, dg)

        self.rho = rho
        return rho

    def _oc_update(self, rho: np.ndarray, dc: np.ndarray,
                   dg: np.ndarray) -> np.ndarray:
        """Optimality Criteria update."""
        move = self.config.move_limit
        eta = 0.5  # Damping coefficient

        # Bisection for Lagrange multiplier
        l1, l2 = 1e-9, 1e9

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l1 + l2)

            # OC formula
            B = -dc / (lmid * dg + 1e-12)
            rho_new = rho * B ** eta

            # Apply bounds and move limits
            rho_new = np.maximum(self.config.min_density,
                                 np.maximum(rho - move,
                                            np.minimum(1.0,
                                                       np.minimum(rho + move, rho_new))))

            # Check volume constraint
            g, _ = self.volume_constraint(rho_new)

            if g > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new

    def get_optimal_design(self) -> Field:
        """Return optimal density field."""
        return Field(
            name='density',
            dim=self.mesh.dim,
            values=self.rho,
            points=self.mesh.centroids
        )


class ComplianceMinimization(TopologyOptimizer):
    """Standard compliance minimization problem."""
    pass


class StressConstrainedOptimizer(TopologyOptimizer):
    """
    Topology optimization with stress constraints.

    More challenging: stress is local, creates many constraints.
    """

    def __init__(self, solver: Solver, stress_limit: float,
                 config: Optional[TopologyConfig] = None):
        super().__init__(solver, config)
        self.stress_limit = stress_limit

    def stress_constraint(self, rho: np.ndarray,
                          stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stress constraints: sigma_vm <= sigma_max for all elements.

        Uses p-norm aggregation to reduce constraint count.
        """
        # P-norm aggregation
        p = 8  # Aggregation parameter
        sigma_normalized = stress / self.stress_limit
        pn = (np.sum(sigma_normalized ** p)) ** (1 / p)

        # Gradient
        dpn = (sigma_normalized ** (p - 1)) / (pn ** (p - 1) + 1e-12)

        return pn - 1, dpn


class MultiMaterialOptimizer:
    """
    Multi-material topology optimization.

    Optimizes material distribution across multiple materials
    with different properties.
    """

    def __init__(self, solver: Solver, materials: List[dict],
                 config: Optional[TopologyConfig] = None):
        self.solver = solver
        self.materials = materials  # List of {E, nu, rho, cost}
        self.config = config or TopologyConfig()
        self.n_materials = len(materials)

    def setup(self, mesh: Mesh, loads: np.ndarray,
              fixed_nodes: np.ndarray):
        """Setup multi-material problem."""
        self.mesh = mesh
        self.loads = loads
        self.fixed_nodes = fixed_nodes

        n = mesh.n_cells if mesh.cells is not None else mesh.n_points

        # Design variables: one density per material per element
        # Constraint: sum over materials = 1
        self.rho = np.ones((n, self.n_materials)) / self.n_materials

    def optimize(self) -> np.ndarray:
        """Optimize multi-material distribution."""
        # Would use alternating direction method or
        # generalized SIMP for multiple materials
        raise NotImplementedError("Multi-material optimization")
