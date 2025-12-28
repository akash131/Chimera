"""
Example 2: Linear Elastic Analysis

Solves a cantilever beam under end load.
"""

import numpy as np
from chimera import Domain
from chimera.mesh import generate_structured_mesh
from chimera.physics.elasticity import LinearElasticity, ElasticMaterial
from chimera.solvers import FEMSolver
from chimera.core.boundary import DirichletBC, NeumannBC


def main():
    # 1. Define beam geometry
    length = 10.0
    height = 1.0
    domain = Domain.box((0, length), (0, height), name="beam")

    # 2. Generate structured mesh
    mesh = generate_structured_mesh(domain, resolution=(40, 4))
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    # 3. Define material
    material = ElasticMaterial.steel()
    elasticity = LinearElasticity(material=material, dim=2)

    print(f"Material: E = {material.E/1e9:.1f} GPa, ν = {material.nu}")

    # 4. Setup boundary conditions
    # Fixed on left edge
    fixed_bc = DirichletBC(
        value=np.array([0.0, 0.0]),
        region=lambda p: np.abs(p[:, 0]) < 1e-6,
        field_name='u'
    )

    # Traction on right edge (downward force)
    load = np.array([0.0, -1000.0])  # N/m
    traction_bc = NeumannBC(
        value=load,
        region=lambda p: np.abs(p[:, 0] - length) < 1e-6,
        field_name='u'
    )

    # Note: Full structural solver would handle vector fields
    # This is simplified for demonstration
    print("Boundary conditions defined")
    print("  - Fixed: left edge")
    print("  - Load: right edge, F = (0, -1000) N/m")

    # 5. Expected deflection (Euler-Bernoulli beam theory)
    I = height**3 / 12  # Second moment of area
    P = load[1] * height  # Total load
    delta_theory = P * length**3 / (3 * material.E * I)
    print(f"\nTheoretical tip deflection: {delta_theory*1000:.3f} mm")


if __name__ == '__main__':
    main()
