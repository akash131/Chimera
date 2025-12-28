"""
Example 4: Neural Operator for Fast Surrogate

Train a Fourier Neural Operator (FNO) on heat equation solutions,
then use it for rapid parameter sweeps.
"""

import numpy as np
from chimera import Domain, Equation
from chimera.mesh import generate_structured_mesh
from chimera.solvers import FEMSolver
from chimera.operators import FourierNeuralOperator, OperatorConfig
from chimera.core.field import Field
from chimera.core.boundary import DirichletBC


def generate_training_data(n_samples: int = 100):
    """Generate training data: source -> solution pairs."""
    domain = Domain.unit_square()
    mesh = generate_structured_mesh(domain, resolution=(32, 32))

    equation = Equation.poisson('u', 'f')

    inputs = []
    outputs = []

    print(f"Generating {n_samples} training samples...")

    for i in range(n_samples):
        # Random source term (sum of Gaussians)
        n_sources = np.random.randint(1, 4)
        centers = np.random.rand(n_sources, 2)
        amplitudes = np.random.randn(n_sources) * 10

        def source_func(x, c=centers, a=amplitudes):
            result = np.zeros(len(x))
            for center, amp in zip(c, a):
                r2 = np.sum((x - center)**2, axis=1)
                result += amp * np.exp(-20 * r2)
            return result

        # Solve
        solver = FEMSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.set_source(source_func)

        # Zero Dirichlet BC on all boundaries
        solver.add_bc(DirichletBC(value=0.0, field_name='u'))

        result = solver.solve()

        if result.success:
            # Store source and solution
            f_values = source_func(mesh.points)
            u_values = result.get_field('u').values

            inputs.append(Field(name='f', values=f_values, points=mesh.points))
            outputs.append(Field(name='u', values=u_values, points=mesh.points))

    print(f"Generated {len(inputs)} valid samples")
    return inputs, outputs, mesh


def main():
    # 1. Generate training data
    inputs, outputs, mesh = generate_training_data(n_samples=50)

    # 2. Build neural operator
    config = OperatorConfig(
        hidden_dims=[32, 32, 32],
        n_modes=8,
        epochs=50
    )
    operator = FourierNeuralOperator(config)
    operator.build(input_dim=1, output_dim=1)

    print("\nNeural operator architecture:")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Fourier modes: {config.n_modes}")

    # 3. Training would happen here
    # operator.train(inputs, outputs)
    print("\n[Training skipped - would train on source->solution pairs]")

    # 4. Demonstrate usage
    print("\nUsage pattern:")
    print("  1. Train on FEM solutions: operator.train(sources, solutions)")
    print("  2. Fast prediction: operator.predict(new_source, mesh)")
    print("  3. Speedup: ~100-1000x faster than FEM solve")

    # 5. Potential applications
    print("\nApplications:")
    print("  - Parameter sweeps")
    print("  - Optimization inner loops")
    print("  - Real-time simulation")
    print("  - Uncertainty quantification (many samples)")


if __name__ == '__main__':
    main()
