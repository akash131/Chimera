"""
Example 1: Steady Heat Conduction

Solves the 2D heat equation on a unit square:
    -∇·(k∇T) = Q

With boundary conditions:
    T = 0 on left edge
    T = 1 on right edge
    ∂T/∂n = 0 on top and bottom (insulated)
"""

import numpy as np
from chimera import Domain, Mesh, HeatEquation, HybridSolver
from chimera.mesh import generate_delaunay_mesh
from chimera.core.boundary import DirichletBC
from chimera.physics.heat import ThermalMaterial


def main():
    # 1. Define domain
    domain = Domain.unit_square()

    # 2. Generate mesh
    mesh = generate_delaunay_mesh(domain, n_points=200, seed=42)
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    # 3. Define physics
    material = ThermalMaterial.aluminum()
    heat_eq = HeatEquation(material=material, transient=False)

    # 4. Setup solver
    solver = HybridSolver()
    solver.set_mesh(mesh)
    solver.set_equation(heat_eq.get_equation())

    # 5. Apply boundary conditions
    # Left edge: T = 0
    left_bc = DirichletBC(
        value=0.0,
        region=lambda p: np.abs(p[:, 0]) < 1e-6,
        field_name='T'
    )

    # Right edge: T = 100
    right_bc = DirichletBC(
        value=100.0,
        region=lambda p: np.abs(p[:, 0] - 1.0) < 1e-6,
        field_name='T'
    )

    solver.add_bc(left_bc)
    solver.add_bc(right_bc)

    # 6. Solve
    result = solver.solve()
    print(f"Solve: {result}")

    # 7. Analyze results
    T = result.get_field('T')
    print(f"Temperature range: {T.min():.2f} to {T.max():.2f}")

    # 8. Visualize (if matplotlib available)
    try:
        from chimera.utils.visualization import plot_field
        plot_field(T, mesh, title="Temperature Distribution", show=True)
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == '__main__':
    main()
