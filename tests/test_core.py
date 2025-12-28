"""Tests for core modules."""

import pytest
import numpy as np

from chimera.core.domain import Domain
from chimera.core.field import Field, FieldType
from chimera.core.equation import Equation, Term, laplacian, ddt
from chimera.core.boundary import DirichletBC, NeumannBC, RobinBC


class TestDomain:
    """Tests for Domain class."""

    def test_box_creation(self):
        """Test box domain creation."""
        domain = Domain.box((0, 1), (0, 2))
        assert domain.dim == 2
        assert domain.bounds == [(0, 1), (0, 2)]

    def test_unit_square(self):
        """Test unit square convenience method."""
        domain = Domain.unit_square()
        assert domain.dim == 2
        assert domain.bounds == [(0.0, 1.0), (0.0, 1.0)]

    def test_unit_cube(self):
        """Test unit cube convenience method."""
        domain = Domain.unit_cube()
        assert domain.dim == 3
        assert len(domain.bounds) == 3

    def test_contains(self):
        """Test point containment."""
        domain = Domain.unit_square()
        points = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.5]])
        result = domain.contains(points)
        assert result[0] == True
        assert result[1] == False
        assert result[2] == False

    def test_sample_points(self):
        """Test random point sampling."""
        domain = Domain.unit_square()
        points = domain.sample_points(100)
        assert len(points) == 100
        assert np.all(domain.contains(points))

    def test_circle_domain(self):
        """Test circular domain."""
        domain = Domain.circle(center=(0, 0), radius=1)
        assert domain.dim == 2
        assert domain.geometry is not None

        # Center should be inside
        assert domain.contains(np.array([[0, 0]]))[0]
        # Point outside should be outside
        assert not domain.contains(np.array([[2, 0]]))[0]


class TestField:
    """Tests for Field class."""

    def test_scalar_field(self):
        """Test scalar field creation."""
        field = Field.scalar('temperature')
        assert field.field_type == FieldType.SCALAR
        assert field.components == 1

    def test_vector_field(self):
        """Test vector field creation."""
        field = Field.vector('velocity', dim=3)
        assert field.field_type == FieldType.VECTOR
        assert field.components == 3

    def test_field_from_function(self):
        """Test field creation from function."""
        points = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        func = lambda x: x[:, 0] + x[:, 1]

        field = Field.from_function('u', func, points)
        assert len(field.values) == 3
        np.testing.assert_array_almost_equal(field.values, [0, 1, 1])

    def test_field_arithmetic(self):
        """Test field arithmetic operations."""
        f1 = Field(name='a', values=np.array([1, 2, 3]))
        f2 = Field(name='b', values=np.array([4, 5, 6]))

        sum_field = f1 + f2
        np.testing.assert_array_equal(sum_field.values, [5, 7, 9])

        diff_field = f2 - f1
        np.testing.assert_array_equal(diff_field.values, [3, 3, 3])

        scaled = f1 * 2
        np.testing.assert_array_equal(scaled.values, [2, 4, 6])


class TestEquation:
    """Tests for Equation class."""

    def test_poisson_equation(self):
        """Test Poisson equation creation."""
        eq = Equation.poisson('u', 'f')
        assert eq.name == 'poisson'
        assert len(eq.terms) == 1
        assert eq.rhs == 'f'

    def test_heat_steady(self):
        """Test steady heat equation."""
        eq = Equation.heat_steady('T', conductivity=2.0)
        assert eq.name == 'heat_steady'

    def test_heat_transient(self):
        """Test transient heat equation."""
        eq = Equation.heat_transient('T')
        assert eq.is_time_dependent()
        assert len(eq.terms) == 2  # dT/dt and laplacian

    def test_get_field_names(self):
        """Test field name extraction."""
        eq = Equation.poisson('u', 'f')
        names = eq.get_field_names()
        assert 'u' in names
        assert 'f' in names


class TestBoundaryConditions:
    """Tests for boundary condition classes."""

    def test_dirichlet_constant(self):
        """Test constant Dirichlet BC."""
        bc = DirichletBC.constant(value=100.0, field_name='T')
        points = np.array([[0, 0], [1, 1]])
        values = bc.evaluate(points)
        np.testing.assert_array_equal(values, [100.0, 100.0])

    def test_dirichlet_function(self):
        """Test function-valued Dirichlet BC."""
        bc = DirichletBC(value=lambda x: x[:, 0]**2, field_name='u')
        points = np.array([[0, 0], [2, 0], [3, 0]])
        values = bc.evaluate(points)
        np.testing.assert_array_almost_equal(values, [0, 4, 9])

    def test_neumann_zero_flux(self):
        """Test zero flux Neumann BC."""
        bc = NeumannBC.zero_flux('T')
        points = np.array([[0, 0]])
        values = bc.evaluate(points)
        assert values[0] == 0.0

    def test_robin_convection(self):
        """Test convective Robin BC."""
        h = 10.0
        T_inf = 25.0
        bc = RobinBC.convection(h=h, T_inf=T_inf, field_name='T')
        assert bc.alpha == h
        assert bc.value == h * T_inf

    def test_bc_region_filter(self):
        """Test BC region filtering."""
        bc = DirichletBC(
            value=1.0,
            region=lambda p: p[:, 0] < 0.5
        )
        points = np.array([[0.1, 0], [0.6, 0]])
        mask = bc.applies_to(points)
        assert mask[0] == True
        assert mask[1] == False
