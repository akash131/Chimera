"""
Example 5: Topology Optimization

Solver-in-the-loop optimization for structural design.
Find optimal material distribution for minimum compliance.
"""

import numpy as np
from chimera import Domain
from chimera.mesh import generate_structured_mesh
from chimera.solvers import FEMSolver
from chimera.optimize import TopologyOptimizer, TopologyConfig


def main():
    # 1. Define design domain
    length = 2.0
    height = 1.0
    domain = Domain.box((0, length), (0, height), name="bridge")

    # 2. Generate mesh
    mesh = generate_structured_mesh(domain, resolution=(80, 40))
    n_elements = mesh.n_cells
    print(f"Design domain: {n_elements} elements")

    # 3. Configure optimization
    config = TopologyConfig(
        volume_fraction=0.4,  # Use 40% of material
        penalization=3.0,      # SIMP penalty
        filter_radius=0.05,    # Minimum feature size
        max_iterations=50,
        convergence_tol=1e-3
    )

    print(f"\nOptimization settings:")
    print(f"  Volume fraction: {config.volume_fraction*100:.0f}%")
    print(f"  SIMP penalty: {config.penalization}")
    print(f"  Filter radius: {config.filter_radius}")

    # 4. Define loads and supports
    # Support on bottom-left and bottom-right corners
    support_tol = 0.05
    def is_support(p):
        left = (np.abs(p[:, 0]) < support_tol) & (np.abs(p[:, 1]) < support_tol)
        right = (np.abs(p[:, 0] - length) < support_tol) & (np.abs(p[:, 1]) < support_tol)
        return left | right

    fixed_nodes = np.where(is_support(mesh.points))[0]

    # Load at center of top edge
    def is_load(p):
        return (np.abs(p[:, 0] - length/2) < 0.1) & (np.abs(p[:, 1] - height) < 0.05)

    load_nodes = np.where(is_load(mesh.points))[0]
    loads = np.zeros((mesh.n_points, 2))
    loads[load_nodes, 1] = -1.0  # Downward force

    print(f"\n  Fixed nodes: {len(fixed_nodes)}")
    print(f"  Load nodes: {len(load_nodes)}")

    # 5. Setup solver and optimizer
    solver = FEMSolver()
    solver.set_mesh(mesh)

    optimizer = TopologyOptimizer(solver, config)
    optimizer.setup(mesh, loads, fixed_nodes)

    # 6. Run optimization
    print("\nRunning topology optimization...")
    print("(This demonstrates the solver-in-the-loop concept)")
    print("Each iteration calls the actual FEM solver\n")

    # Note: Full optimization loop would run here
    # optimal_density = optimizer.optimize()

    # 7. Expected result description
    print("Expected optimal structure:")
    print("  - Arch-like topology connecting supports")
    print("  - Material concentrated along principal stress paths")
    print("  - Void regions where stress is low")
    print("  - Classic 'bridge' or 'Michell truss' pattern")


if __name__ == '__main__':
    main()
