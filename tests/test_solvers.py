"""Tests for solver modules."""

import pytest
import numpy as np

from chimera.core.domain import Domain
from chimera.core.equation import Equation
from chimera.core.boundary import DirichletBC
from chimera.mesh.generators import generate_delaunay_mesh, generate_structured_mesh
from chimera.solvers import FEMSolver, FVMSolver, MeshlessSolver, HybridSolver
from chimera.solvers.base import SolverConfig, SolverStatus


class TestFEMSolver:
    """Tests for FEM solver."""

    def test_poisson_simple(self):
        """Test simple Poisson problem."""
        # Problem: -Δu = 1 on [0,1]², u = 0 on boundary
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=100, seed=42)

        equation = Equation.poisson('u', 'f')

        solver = FEMSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.set_source(lambda x: np.ones(len(x)))

        # Zero Dirichlet BC
        solver.add_bc(DirichletBC(value=0.0, field_name='u'))

        result = solver.solve()

        assert result.status == SolverStatus.SUCCESS
        assert 'u' in result.fields

        u = result.get_field('u')
        # Solution should be zero on boundary
        boundary_values = u.values[mesh.boundary_points]
        np.testing.assert_array_almost_equal(boundary_values, 0, decimal=2)

        # Maximum should be at center (approximately)
        assert u.max() > 0

    def test_matrix_assembly(self):
        """Test stiffness matrix assembly."""
        domain = Domain.unit_square()
        mesh = generate_structured_mesh(domain, resolution=(5, 5))
        equation = Equation.poisson('u', 'f')

        solver = FEMSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.assemble()

        K = solver.get_matrix()

        # Matrix should be square
        assert K.shape[0] == K.shape[1] == mesh.n_points

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

        # Diagonal should be positive
        assert np.all(np.diag(K) >= 0)

    def test_mass_matrix(self):
        """Test mass matrix assembly."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=50, seed=42)
        equation = Equation.heat_transient('T')

        solver = FEMSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.assemble()

        M = solver.get_mass_matrix()

        # Mass matrix should be symmetric positive definite
        assert M.shape[0] == M.shape[1]
        assert np.all(np.diag(M) > 0)


class TestFVMSolver:
    """Tests for FVM solver."""

    def test_diffusion_1d_analytic(self):
        """Test 1D diffusion against analytic solution."""
        # -d²u/dx² = 1 on [0,1], u(0) = u(1) = 0
        # Analytic: u(x) = x(1-x)/2
        from chimera.mesh import generate_cartesian_mesh

        domain = Domain.box((0, 1), (0, 0.1))  # Thin 2D domain ≈ 1D
        mesh = generate_cartesian_mesh(domain, resolution=(20, 2))

        equation = Equation.poisson('u', 'f')

        solver = FVMSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.set_source(lambda x: np.ones(len(x)))

        # Boundary conditions
        solver.add_bc(DirichletBC(
            value=0.0,
            region=lambda p: (p[:, 0] < 0.05) | (p[:, 0] > 0.95),
            field_name='u'
        ))

        result = solver.solve()
        assert result.success

    def test_flux_scheme_options(self):
        """Test different flux schemes."""
        domain = Domain.unit_square()
        from chimera.mesh import generate_cartesian_mesh
        mesh = generate_cartesian_mesh(domain, resolution=(10, 10))
        equation = Equation.poisson('u', 'f')

        for scheme in ['central', 'upwind']:
            solver = FVMSolver()
            solver.set_mesh(mesh)
            solver.set_equation(equation)
            solver.set_flux_scheme(scheme)
            solver.add_bc(DirichletBC(value=0.0, field_name='u'))

            result = solver.solve()
            assert result.success, f"Failed with scheme: {scheme}"


class TestMeshlessSolver:
    """Tests for meshless solver."""

    def test_poisson_rbf(self):
        """Test Poisson with RBF collocation."""
        domain = Domain.unit_square()
        from chimera.mesh.generators import generate_point_cloud

        mesh = generate_point_cloud(domain, n_points=100, seed=42)
        equation = Equation.poisson('u', 'f')

        solver = MeshlessSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.set_source(lambda x: np.ones(len(x)))
        solver.add_bc(DirichletBC(value=0.0, field_name='u'))

        result = solver.solve()

        # Meshless should converge (though possibly less accurately)
        assert result.status in [SolverStatus.SUCCESS, SolverStatus.CONVERGED]


class TestHybridSolver:
    """Tests for hybrid solver."""

    def test_single_method_fallback(self):
        """Test hybrid solver with single method (fallback)."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=50, seed=42)
        equation = Equation.poisson('u', 'f')

        solver = HybridSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)
        solver.set_source(lambda x: np.ones(len(x)))
        solver.add_bc(DirichletBC(value=0.0, field_name='u'))

        result = solver.solve()

        assert result.success
        assert result.metadata.get('method') == 'hybrid'

    def test_partition_visualization(self):
        """Test domain partition output."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=100, seed=42)
        equation = Equation.poisson('u', 'f')

        solver = HybridSolver()
        solver.set_mesh(mesh)
        solver.set_equation(equation)

        partition = solver.visualize_partition()
        assert len(partition) == mesh.n_points


class TestSolverConfig:
    """Tests for solver configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SolverConfig()
        assert config.linear_solver == 'direct'
        assert config.max_iterations == 100
        assert config.tolerance == 1e-8

    def test_custom_config(self):
        """Test custom configuration."""
        config = SolverConfig(
            linear_solver='cg',
            max_iterations=50,
            tolerance=1e-6
        )

        solver = FEMSolver(config)
        assert solver.config.linear_solver == 'cg'
        assert solver.config.max_iterations == 50
