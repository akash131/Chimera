"""
Example 3: Hybrid Discretization

Demonstrates Chimera's unique capability: using different
discretization methods in different regions of the domain.

Problem: Advection-diffusion with sharp gradient
- FVM in advection-dominated region
- FEM in diffusion-dominated region
"""

import numpy as np
from chimera import Domain, Equation
from chimera.mesh import generate_delaunay_mesh
from chimera.solvers import HybridSolver
from chimera.solvers.hybrid import MethodType, peclet_method_selector
from chimera.core.boundary import DirichletBC


def main():
    # 1. Define domain
    domain = Domain.box((0, 2), (0, 1), name="channel")

    # 2. Generate mesh
    mesh = generate_delaunay_mesh(domain, n_points=500, seed=42)
    print(f"Mesh: {mesh.n_points} points")

    # 3. Define advection-diffusion equation
    # ∂c/∂t + u·∇c - D∇²c = 0
    equation = Equation.advection_diffusion(
        field='c',
        velocity='u',
        diffusivity=0.01,  # Low diffusivity -> advection dominated
        source='0'
    )

    # 4. Setup hybrid solver with automatic method selection
    solver = HybridSolver()
    solver.set_mesh(mesh)
    solver.set_equation(equation)

    # 5. Define method selection based on local Peclet number
    # High Pe -> FVM (upwind stable for advection)
    # Low Pe -> FEM (accurate for diffusion)
    velocity = 1.0
    diffusivity = 0.01

    solver.set_method_selector(
        lambda coord, sol: peclet_method_selector(
            coord, sol,
            velocity=velocity,
            diffusivity=diffusivity,
            threshold=10.0
        )
    )

    # 6. Partition domain automatically
    solver.auto_partition()

    # 7. Visualize partition
    partition = solver.visualize_partition()
    n_fem = np.sum(partition == MethodType.FEM.value)
    n_fvm = np.sum(partition == MethodType.FVM.value)

    print(f"\nDomain partition:")
    print(f"  FEM regions: {n_fem} points")
    print(f"  FVM regions: {n_fvm} points")

    # 8. Apply boundary conditions
    inlet_bc = DirichletBC(
        value=1.0,
        region=lambda p: np.abs(p[:, 0]) < 1e-6,
        field_name='c'
    )
    outlet_bc = DirichletBC(
        value=0.0,
        region=lambda p: np.abs(p[:, 0] - 2.0) < 1e-6,
        field_name='c'
    )

    solver.add_bc(inlet_bc)
    solver.add_bc(outlet_bc)

    # 9. Solve
    result = solver.solve()
    print(f"\nSolve result: {result.status.value}")
    print(f"Method: {result.metadata.get('method', 'unknown')}")


if __name__ == '__main__':
    main()
