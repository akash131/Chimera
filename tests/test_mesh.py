"""Tests for mesh module."""

import pytest
import numpy as np

from chimera.core.domain import Domain
from chimera.mesh.mesh import Mesh, MeshType, CellType
from chimera.mesh.generators import (
    generate_structured_mesh,
    generate_delaunay_mesh,
    generate_cartesian_mesh,
    generate_point_cloud,
)


class TestMeshGeneration:
    """Tests for mesh generation."""

    def test_structured_2d(self):
        """Test 2D structured mesh generation."""
        domain = Domain.unit_square()
        mesh = generate_structured_mesh(domain, resolution=(10, 10))

        assert mesh.n_points == 11 * 11
        assert mesh.n_cells == 10 * 10
        assert mesh.cell_type == CellType.QUAD
        assert mesh.mesh_type == MeshType.STRUCTURED

    def test_structured_3d(self):
        """Test 3D structured mesh generation."""
        domain = Domain.unit_cube()
        mesh = generate_structured_mesh(domain, resolution=(5, 5, 5))

        assert mesh.n_points == 6 * 6 * 6
        assert mesh.n_cells == 5 * 5 * 5
        assert mesh.cell_type == CellType.HEXAHEDRON

    def test_delaunay_2d(self):
        """Test 2D Delaunay mesh generation."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=100, seed=42)

        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        assert mesh.cell_type == CellType.TRIANGLE

    def test_cartesian_mesh(self):
        """Test Cartesian mesh for FVM."""
        domain = Domain.unit_square()
        mesh = generate_cartesian_mesh(domain, resolution=(10, 10))

        assert mesh.n_points == 10 * 10  # Cell centers
        assert mesh.mesh_type == MeshType.CARTESIAN
        assert mesh._neighbors is not None

    def test_point_cloud(self):
        """Test meshless point cloud generation."""
        domain = Domain.unit_square()
        mesh = generate_point_cloud(domain, n_points=100, method='random')

        assert mesh.n_points == 100
        assert mesh.cells is None
        assert mesh.mesh_type == MeshType.MESHLESS


class TestMeshOperations:
    """Tests for mesh operations."""

    def test_boundary_detection_2d(self):
        """Test boundary detection in 2D."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=50, seed=42)
        mesh.find_boundary()

        assert mesh.boundary_points is not None
        assert len(mesh.boundary_points) > 0

    def test_boundary_tagging(self):
        """Test boundary tagging."""
        domain = Domain.unit_square()
        mesh = generate_structured_mesh(domain, resolution=(5, 5))
        mesh.find_boundary()

        # Tag left boundary
        mesh.tag_boundary('left', lambda p: np.abs(p[:, 0]) < 1e-6)

        assert 'left' in mesh.boundary_tags
        left_nodes = mesh.get_boundary('left')
        assert len(left_nodes) > 0

        # All left nodes should have x ≈ 0
        left_coords = mesh.points[left_nodes]
        np.testing.assert_array_almost_equal(left_coords[:, 0], 0, decimal=5)

    def test_neighbor_finding(self):
        """Test nearest neighbor finding."""
        domain = Domain.unit_square()
        mesh = generate_point_cloud(domain, n_points=100)
        neighbors = mesh.find_neighbors(k=10)

        assert neighbors.shape == (100, 10)
        # Neighbors should not include self
        for i in range(100):
            assert i not in neighbors[i]

    def test_volume_computation(self):
        """Test cell volume computation."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=50, seed=42)

        volumes = mesh.volumes
        assert len(volumes) == mesh.n_cells
        assert np.all(volumes > 0)

        # Total volume should be approximately 1 (unit square)
        np.testing.assert_almost_equal(np.sum(volumes), 1.0, decimal=1)

    def test_centroid_computation(self):
        """Test cell centroid computation."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=50, seed=42)

        centroids = mesh.centroids
        assert centroids.shape == (mesh.n_cells, 2)

        # All centroids should be inside domain
        assert np.all(centroids >= 0)
        assert np.all(centroids <= 1)

    def test_mesh_refinement(self):
        """Test mesh refinement."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=20, seed=42)
        original_cells = mesh.n_cells

        refined = mesh.refine()

        # Refinement should increase cell count
        assert refined.n_cells > original_cells
        # Each triangle split into 4
        assert refined.n_cells == 4 * original_cells

    def test_quality_metrics(self):
        """Test mesh quality metrics."""
        domain = Domain.unit_square()
        mesh = generate_delaunay_mesh(domain, n_points=100, seed=42)

        metrics = mesh.quality_metrics()

        assert 'n_points' in metrics
        assert 'n_cells' in metrics
        assert 'min_volume' in metrics
        assert 'max_volume' in metrics
        assert metrics['max_volume'] > metrics['min_volume']

    def test_kdtree(self):
        """Test KD-tree for point queries."""
        domain = Domain.unit_square()
        mesh = generate_point_cloud(domain, n_points=100)

        tree = mesh.kdtree
        assert tree is not None

        # Query nearest point to center
        dist, idx = tree.query([0.5, 0.5], k=1)
        assert 0 <= idx < mesh.n_points
