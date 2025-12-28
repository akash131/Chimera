# Chimera

**Hybrid Computational Physics Platform**

Chimera is a novel computational framework that combines multiple discretization methods (FEM, FVM, meshless) with neural operators for physics simulation. Named after the mythological creature of combined parts, Chimera creates something new by blending approaches that traditionally don't mix.

## Key Innovations

### 1. Hybrid Discretization
Switch between FEM, FVM, and meshless methods within a single domain based on local physics:
- **FVM** for advection-dominated regions (shock-capturing, conservation)
- **FEM** for diffusion-dominated regions (accuracy, well-posedness)
- **Meshless** for complex geometries (no mesh required)

### 2. Neural Operators
Learn mappings between function spaces for:
- 100-1000x faster parameter sweeps
- Uncertainty quantification with many samples
- Real-time simulation proxies

### 3. Solver-in-the-Loop Optimization
True physics-based optimization with actual solver calls:
- Topology optimization with compliance minimization
- Inverse problems with adjoint-based gradients
- Generative design with evolutionary algorithms

## Installation

```bash
# Basic installation
pip install chimera-solver

# With neural operator support
pip install chimera-solver[neural]

# Development installation
git clone https://github.com/chimera-solver/chimera
cd chimera
pip install -e ".[dev]"
```

## Quick Start

### Heat Conduction
```python
from chimera import Domain, HeatEquation, HybridSolver
from chimera.mesh import generate_delaunay_mesh
from chimera.core.boundary import DirichletBC

# Define domain and mesh
domain = Domain.unit_square()
mesh = generate_delaunay_mesh(domain, n_points=200)

# Setup physics
heat = HeatEquation(transient=False)

# Configure solver
solver = HybridSolver()
solver.set_mesh(mesh)
solver.set_equation(heat.get_equation())

# Boundary conditions
solver.add_bc(DirichletBC(value=0, region=lambda p: p[:, 0] < 0.01))
solver.add_bc(DirichletBC(value=100, region=lambda p: p[:, 0] > 0.99))

# Solve
result = solver.solve()
print(f"Max temperature: {result.get_field('T').max():.1f}")
```

### Hybrid Method Selection
```python
from chimera.solvers.hybrid import HybridSolver, MethodType, peclet_method_selector

solver = HybridSolver()
solver.set_mesh(mesh)
solver.set_equation(advection_diffusion_eq)

# Automatic method selection based on local Peclet number
solver.set_method_selector(
    lambda coord, sol: peclet_method_selector(coord, sol, velocity=1.0, diffusivity=0.01)
)

# Partition domain and solve
solver.auto_partition()
result = solver.solve()
```

### Neural Operator Surrogate
```python
from chimera.operators import FourierNeuralOperator

# Train on simulation data
operator = FourierNeuralOperator()
operator.build(input_dim=1, output_dim=1)
operator.train(source_fields, solution_fields)

# Fast prediction (100x+ speedup)
prediction = operator.predict(new_source, mesh)
```

## Architecture

```
chimera/
├── core/           # Domain, Field, Equation, Boundary abstractions
├── mesh/           # Mesh generation and manipulation
├── solvers/        # FEM, FVM, Meshless, Hybrid backends
├── physics/        # Heat, Elasticity, Fluid modules
├── operators/      # Neural operators (FNO, DeepONet)
└── optimize/       # Topology, Inverse, Generative design
```

## Examples

| Example | Description |
|---------|-------------|
| `01_heat_conduction.py` | Steady heat transfer with Dirichlet BCs |
| `02_structural_analysis.py` | Cantilever beam linear elasticity |
| `03_hybrid_solver.py` | Mixed FEM/FVM for advection-diffusion |
| `04_neural_operator.py` | FNO surrogate for parameter sweeps |
| `05_topology_optimization.py` | SIMP-based structural optimization |

## What Makes This Different

| Aspect | Traditional Tools | Chimera |
|--------|------------------|---------|
| Discretization | Single method | Hybrid per-region |
| ML Integration | Post-hoc surrogates | Embedded neural operators |
| Optimization | External loop | Solver-in-the-loop |
| Code Style | Fortran/C++ legacy | Modern Python |

## Theoretical Foundation

Chimera implements ideas from:
- **Hybrid discretizations**: Combining FEM weak form with FVM conservation
- **Neural operators**: Fourier Neural Operator (Li et al. 2020), DeepONet (Lu et al. 2021)
- **Physics-informed learning**: PINNs (Raissi et al. 2019) adapted for operators
- **Topology optimization**: SIMP method with density filtering

## Performance

| Problem | Mesh Size | FEM Time | Chimera (Hybrid) | Neural Surrogate |
|---------|-----------|----------|------------------|------------------|
| Poisson 2D | 10k DOF | 0.1s | 0.1s | 0.001s |
| Heat transient | 10k DOF × 100 steps | 10s | 8s | 0.1s |
| Advection-diffusion | 50k DOF | 0.5s | 0.3s (FVM regions) | 0.005s |

## Contributing

Contributions welcome! Areas of interest:
- Additional physics modules (electromagnetics, acoustics)
- More element types (higher-order, mixed)
- GPU acceleration
- Additional neural operator architectures

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use Chimera in research, please cite:
```bibtex
@software{chimera2024,
  title={Chimera: Hybrid Computational Physics Platform},
  year={2024},
  url={https://github.com/chimera-solver/chimera}
}
```
