"""
Unified mesh representation for Chimera.

This module provides a mesh abstraction that works across discretization
methods (FEM, FVM, meshless) while maintaining method-specific data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
import numpy as np
from scipy.spatial import Delaunay, KDTree


class MeshType(Enum):
    """Type of mesh/discretization."""
    STRUCTURED = "structured"  # Regular grid
    UNSTRUCTURED = "unstructured"  # Triangle/tet mesh
    CARTESIAN = "cartesian"  # Axis-aligned cells (for FVM)
    MESHLESS = "meshless"  # Point cloud with neighbors


class CellType(Enum):
    """Cell/element types."""
    POINT = 0
    LINE = 1
    TRIANGLE = 2
    QUAD = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 5
    WEDGE = 6
    PYRAMID = 7


@dataclass
class Mesh:
    """
    Unified mesh representation.

    Stores:
    - Node coordinates
    - Cell connectivity
    - Boundary information
    - Derived quantities (areas, volumes, normals, neighbors)

    Can be created from different sources and adapted for different
    discretization methods.
    """
    # Core data
    points: np.ndarray  # (n_points, dim)
    cells: Optional[np.ndarray] = None  # (n_cells, nodes_per_cell)
    cell_type: CellType = CellType.TRIANGLE

    # Mesh metadata
    mesh_type: MeshType = MeshType.UNSTRUCTURED
    dim: int = 2

    # Boundary data
    boundary_points: Optional[np.ndarray] = None  # Indices of boundary points
    boundary_faces: Optional[np.ndarray] = None  # Boundary face connectivity
    boundary_normals: Optional[np.ndarray] = None  # Outward normals
    boundary_tags: Optional[Dict[str, np.ndarray]] = None  # Named boundaries

    # Derived quantities (computed lazily)
    _volumes: Optional[np.ndarray] = field(default=None, repr=False)
    _centroids: Optional[np.ndarray] = field(default=None, repr=False)
    _face_areas: Optional[np.ndarray] = field(default=None, repr=False)
    _neighbors: Optional[np.ndarray] = field(default=None, repr=False)
    _kdtree: Optional[KDTree] = field(default=None, repr=False)

    def __post_init__(self):
        self.dim = self.points.shape[1]
        if self.boundary_tags is None:
            self.boundary_tags = {}

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells) if self.cells is not None else 0

    @property
    def n_boundary_points(self) -> int:
        return len(self.boundary_points) if self.boundary_points is not None else 0

    @property
    def volumes(self) -> np.ndarray:
        """Cell volumes/areas (computed on first access)."""
        if self._volumes is None:
            self._compute_volumes()
        return self._volumes

    @property
    def centroids(self) -> np.ndarray:
        """Cell centroids (computed on first access)."""
        if self._centroids is None:
            self._compute_centroids()
        return self._centroids

    @property
    def kdtree(self) -> KDTree:
        """KD-tree for point queries (computed on first access)."""
        if self._kdtree is None:
            self._kdtree = KDTree(self.points)
        return self._kdtree

    def _compute_volumes(self):
        """Compute cell volumes/areas."""
        if self.cells is None:
            self._volumes = np.array([])
            return

        n_cells = len(self.cells)
        self._volumes = np.zeros(n_cells)

        if self.cell_type == CellType.TRIANGLE:
            for i, cell in enumerate(self.cells):
                p0, p1, p2 = self.points[cell]
                # Cross product for triangle area
                v1 = p1 - p0
                v2 = p2 - p0
                if self.dim == 2:
                    self._volumes[i] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                else:
                    self._volumes[i] = 0.5 * np.linalg.norm(np.cross(v1, v2))

        elif self.cell_type == CellType.QUAD:
            for i, cell in enumerate(self.cells):
                # Split into two triangles
                p0, p1, p2, p3 = self.points[cell]
                v1 = p1 - p0
                v2 = p2 - p0
                v3 = p3 - p0
                a1 = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                a2 = 0.5 * abs(v2[0] * v3[1] - v2[1] * v3[0])
                self._volumes[i] = a1 + a2

        elif self.cell_type == CellType.TETRAHEDRON:
            for i, cell in enumerate(self.cells):
                p0, p1, p2, p3 = self.points[cell]
                v1 = p1 - p0
                v2 = p2 - p0
                v3 = p3 - p0
                self._volumes[i] = abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    def _compute_centroids(self):
        """Compute cell centroids."""
        if self.cells is None:
            self._centroids = np.array([]).reshape(0, self.dim)
            return

        self._centroids = np.zeros((len(self.cells), self.dim))
        for i, cell in enumerate(self.cells):
            self._centroids[i] = self.points[cell].mean(axis=0)

    def find_boundary(self):
        """Identify boundary points and faces."""
        if self.cells is None:
            # Meshless - use convex hull or alpha shape
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.points)
            self.boundary_points = np.unique(hull.simplices)
            return

        if self.dim == 2:
            self._find_boundary_2d()
        else:
            self._find_boundary_3d()

    def _find_boundary_2d(self):
        """Find boundary edges in 2D mesh."""
        edge_count = {}

        for cell in self.cells:
            n = len(cell)
            for i in range(n):
                edge = tuple(sorted([cell[i], cell[(i + 1) % n]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear exactly once
        boundary_edges = [e for e, count in edge_count.items() if count == 1]

        self.boundary_points = np.unique(np.array(boundary_edges).flatten())
        self.boundary_faces = np.array(boundary_edges)

        # Compute normals
        self.boundary_normals = np.zeros((len(boundary_edges), 2))
        for i, (n1, n2) in enumerate(boundary_edges):
            p1, p2 = self.points[n1], self.points[n2]
            tangent = p2 - p1
            # Outward normal (rotate tangent 90 degrees)
            normal = np.array([tangent[1], -tangent[0]])
            self.boundary_normals[i] = normal / np.linalg.norm(normal)

    def _find_boundary_3d(self):
        """Find boundary faces in 3D mesh."""
        face_count = {}

        for cell in self.cells:
            if self.cell_type == CellType.TETRAHEDRON:
                faces = [
                    tuple(sorted([cell[0], cell[1], cell[2]])),
                    tuple(sorted([cell[0], cell[1], cell[3]])),
                    tuple(sorted([cell[0], cell[2], cell[3]])),
                    tuple(sorted([cell[1], cell[2], cell[3]])),
                ]
            else:
                continue  # Handle other cell types as needed

            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1

        boundary_faces = [f for f, count in face_count.items() if count == 1]
        self.boundary_points = np.unique(np.array(boundary_faces).flatten())
        self.boundary_faces = np.array(boundary_faces)

    def find_neighbors(self, k: int = 10) -> np.ndarray:
        """Find k nearest neighbors for each point."""
        if self._neighbors is not None and self._neighbors.shape[1] == k:
            return self._neighbors

        distances, self._neighbors = self.kdtree.query(self.points, k=k + 1)
        # Exclude self (first neighbor)
        self._neighbors = self._neighbors[:, 1:]
        return self._neighbors

    def tag_boundary(self, name: str, predicate: callable):
        """
        Tag boundary points matching a predicate.

        Args:
            name: Tag name (e.g., 'inlet', 'wall', 'outlet')
            predicate: Function(points) -> bool mask
        """
        if self.boundary_points is None:
            self.find_boundary()

        boundary_coords = self.points[self.boundary_points]
        mask = predicate(boundary_coords)
        self.boundary_tags[name] = self.boundary_points[mask]

    def get_boundary(self, name: str) -> np.ndarray:
        """Get point indices for a named boundary."""
        if name not in self.boundary_tags:
            raise KeyError(f"Boundary '{name}' not found")
        return self.boundary_tags[name]

    def refine(self, factor: int = 2) -> Mesh:
        """Uniformly refine the mesh."""
        if self.mesh_type == MeshType.STRUCTURED:
            return self._refine_structured(factor)
        elif self.cell_type == CellType.TRIANGLE:
            return self._refine_triangular()
        else:
            raise NotImplementedError(f"Refinement for {self.mesh_type}")

    def _refine_structured(self, factor: int) -> Mesh:
        """Refine structured mesh by subdivision."""
        # Implement uniform refinement
        raise NotImplementedError("Structured mesh refinement")

    def _refine_triangular(self) -> Mesh:
        """Refine triangular mesh by edge bisection."""
        # Midpoint refinement
        new_points = list(self.points)
        new_cells = []
        edge_to_mid = {}

        for cell in self.cells:
            mids = []
            for i in range(3):
                edge = tuple(sorted([cell[i], cell[(i + 1) % 3]]))
                if edge not in edge_to_mid:
                    mid_pt = (self.points[edge[0]] + self.points[edge[1]]) / 2
                    edge_to_mid[edge] = len(new_points)
                    new_points.append(mid_pt)
                mids.append(edge_to_mid[edge])

            # Create 4 new triangles
            n0, n1, n2 = cell
            m0, m1, m2 = mids
            new_cells.extend([
                [n0, m0, m2],
                [m0, n1, m1],
                [m2, m1, n2],
                [m0, m1, m2],
            ])

        return Mesh(
            points=np.array(new_points),
            cells=np.array(new_cells),
            cell_type=CellType.TRIANGLE,
            mesh_type=self.mesh_type,
        )

    def to_meshless(self, n_neighbors: int = 20) -> Mesh:
        """Convert to meshless representation."""
        new_mesh = Mesh(
            points=self.points.copy(),
            cells=None,
            mesh_type=MeshType.MESHLESS,
            boundary_points=self.boundary_points.copy() if self.boundary_points is not None else None,
        )
        new_mesh.find_neighbors(n_neighbors)
        return new_mesh

    def quality_metrics(self) -> Dict[str, float]:
        """Compute mesh quality metrics."""
        if self.cells is None:
            return {"type": "meshless", "n_points": self.n_points}

        metrics = {
            "n_points": self.n_points,
            "n_cells": self.n_cells,
            "min_volume": float(self.volumes.min()),
            "max_volume": float(self.volumes.max()),
            "volume_ratio": float(self.volumes.max() / self.volumes.min()),
        }

        if self.cell_type == CellType.TRIANGLE:
            # Compute aspect ratios
            aspects = []
            for cell in self.cells:
                pts = self.points[cell]
                edges = [
                    np.linalg.norm(pts[1] - pts[0]),
                    np.linalg.norm(pts[2] - pts[1]),
                    np.linalg.norm(pts[0] - pts[2]),
                ]
                aspects.append(max(edges) / min(edges))

            metrics["min_aspect"] = float(np.min(aspects))
            metrics["max_aspect"] = float(np.max(aspects))
            metrics["mean_aspect"] = float(np.mean(aspects))

        return metrics

    def save(self, filename: str):
        """Save mesh to file (VTK, meshio formats)."""
        import meshio

        if self.cells is None:
            # Point cloud
            meshio.write(
                filename,
                meshio.Mesh(self.points, []),
            )
        else:
            cell_type_map = {
                CellType.TRIANGLE: "triangle",
                CellType.QUAD: "quad",
                CellType.TETRAHEDRON: "tetra",
                CellType.HEXAHEDRON: "hexahedron",
            }
            meshio.write(
                filename,
                meshio.Mesh(
                    self.points,
                    [(cell_type_map[self.cell_type], self.cells)]
                ),
            )

    @classmethod
    def load(cls, filename: str) -> Mesh:
        """Load mesh from file."""
        import meshio

        mesh = meshio.read(filename)

        cells = None
        cell_type = CellType.TRIANGLE

        if mesh.cells:
            cell_block = mesh.cells[0]
            cells = cell_block.data

            type_map = {
                "triangle": CellType.TRIANGLE,
                "quad": CellType.QUAD,
                "tetra": CellType.TETRAHEDRON,
                "hexahedron": CellType.HEXAHEDRON,
            }
            cell_type = type_map.get(cell_block.type, CellType.TRIANGLE)

        return cls(
            points=mesh.points[:, :mesh.points.shape[1]],
            cells=cells,
            cell_type=cell_type,
        )

    def __repr__(self) -> str:
        return (f"Mesh(type={self.mesh_type.value}, dim={self.dim}, "
                f"points={self.n_points}, cells={self.n_cells})")
