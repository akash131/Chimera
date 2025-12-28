"""
Mesh generation utilities for Chimera.

Provides functions to generate various mesh types from domain definitions.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial import Delaunay

from chimera.mesh.mesh import Mesh, MeshType, CellType
from chimera.core.domain import Domain


def generate_structured_mesh(domain: Domain,
                             resolution: Tuple[int, ...]) -> Mesh:
    """
    Generate a structured mesh on a box domain.

    Args:
        domain: Domain to mesh
        resolution: Number of cells in each direction (nx, ny, ...)

    Returns:
        Mesh with structured grid
    """
    if domain.geometry is not None:
        raise ValueError("Structured mesh requires box domain (no implicit geometry)")

    dim = domain.dim
    if len(resolution) != dim:
        raise ValueError(f"Resolution must have {dim} components")

    # Generate grid points
    linspaces = []
    for i, (lo, hi) in enumerate(domain.bounds):
        linspaces.append(np.linspace(lo, hi, resolution[i] + 1))

    if dim == 1:
        points = linspaces[0].reshape(-1, 1)
        # 1D cells are line segments
        n_cells = resolution[0]
        cells = np.zeros((n_cells, 2), dtype=int)
        for i in range(n_cells):
            cells[i] = [i, i + 1]
        cell_type = CellType.LINE

    elif dim == 2:
        xx, yy = np.meshgrid(linspaces[0], linspaces[1], indexing='ij')
        points = np.column_stack([xx.flatten(), yy.flatten()])

        # Create quad cells
        nx, ny = resolution
        n_cells = nx * ny
        cells = np.zeros((n_cells, 4), dtype=int)

        for i in range(nx):
            for j in range(ny):
                cell_idx = i * ny + j
                n0 = i * (ny + 1) + j
                n1 = (i + 1) * (ny + 1) + j
                n2 = (i + 1) * (ny + 1) + (j + 1)
                n3 = i * (ny + 1) + (j + 1)
                cells[cell_idx] = [n0, n1, n2, n3]

        cell_type = CellType.QUAD

    elif dim == 3:
        xx, yy, zz = np.meshgrid(linspaces[0], linspaces[1], linspaces[2], indexing='ij')
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

        # Create hex cells
        nx, ny, nz = resolution
        n_cells = nx * ny * nz
        cells = np.zeros((n_cells, 8), dtype=int)

        def node_index(i, j, k):
            return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

        cell_idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cells[cell_idx] = [
                        node_index(i, j, k),
                        node_index(i + 1, j, k),
                        node_index(i + 1, j + 1, k),
                        node_index(i, j + 1, k),
                        node_index(i, j, k + 1),
                        node_index(i + 1, j, k + 1),
                        node_index(i + 1, j + 1, k + 1),
                        node_index(i, j + 1, k + 1),
                    ]
                    cell_idx += 1

        cell_type = CellType.HEXAHEDRON

    mesh = Mesh(
        points=points,
        cells=cells,
        cell_type=cell_type,
        mesh_type=MeshType.STRUCTURED,
    )
    mesh.find_boundary()

    # Tag boundaries
    _tag_box_boundaries(mesh, domain)

    return mesh


def generate_delaunay_mesh(domain: Domain,
                           n_points: int = 100,
                           seed: Optional[int] = None) -> Mesh:
    """
    Generate an unstructured triangular/tetrahedral mesh using Delaunay.

    Args:
        domain: Domain to mesh
        n_points: Approximate number of interior points
        seed: Random seed for reproducibility

    Returns:
        Mesh with Delaunay triangulation
    """
    if domain.dim not in [2, 3]:
        raise ValueError("Delaunay mesh requires 2D or 3D domain")

    rng = np.random.default_rng(seed)

    # Generate interior points
    interior = domain.sample_points(n_points, seed=seed)

    # Add boundary points for box domains
    if domain.geometry is None:
        boundary = domain.boundary_points(n_per_face=int(np.sqrt(n_points)))
        points = np.vstack([interior, boundary])
    else:
        points = interior

    # Delaunay triangulation
    tri = Delaunay(points)

    # Filter cells outside domain
    if domain.geometry is not None:
        centroids = points[tri.simplices].mean(axis=1)
        inside = domain.contains(centroids)
        cells = tri.simplices[inside]
    else:
        cells = tri.simplices

    cell_type = CellType.TRIANGLE if domain.dim == 2 else CellType.TETRAHEDRON

    mesh = Mesh(
        points=points,
        cells=cells,
        cell_type=cell_type,
        mesh_type=MeshType.UNSTRUCTURED,
    )
    mesh.find_boundary()

    if domain.geometry is None:
        _tag_box_boundaries(mesh, domain)

    return mesh


def generate_cartesian_mesh(domain: Domain,
                            resolution: Tuple[int, ...]) -> Mesh:
    """
    Generate a Cartesian mesh suitable for FVM (cell-centered).

    Returns cell centers instead of vertices.

    Args:
        domain: Domain to mesh
        resolution: Number of cells in each direction

    Returns:
        Mesh with cell centers and FVM-compatible structure
    """
    if domain.geometry is not None:
        raise ValueError("Cartesian mesh requires box domain")

    dim = domain.dim
    if len(resolution) != dim:
        raise ValueError(f"Resolution must have {dim} components")

    # Generate cell centers
    centers = []
    for i, (lo, hi) in enumerate(domain.bounds):
        dx = (hi - lo) / resolution[i]
        centers.append(np.linspace(lo + dx / 2, hi - dx / 2, resolution[i]))

    if dim == 1:
        points = centers[0].reshape(-1, 1)
    elif dim == 2:
        xx, yy = np.meshgrid(centers[0], centers[1], indexing='ij')
        points = np.column_stack([xx.flatten(), yy.flatten()])
    elif dim == 3:
        xx, yy, zz = np.meshgrid(centers[0], centers[1], centers[2], indexing='ij')
        points = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

    # For FVM, we store cells as point clouds with neighbor info
    mesh = Mesh(
        points=points,
        cells=None,  # FVM uses cell-centered approach
        mesh_type=MeshType.CARTESIAN,
    )

    # Build neighbor structure for FVM
    n_neighbors = 2 * dim
    neighbors = np.full((len(points), n_neighbors), -1, dtype=int)

    if dim == 1:
        for i in range(resolution[0]):
            if i > 0:
                neighbors[i, 0] = i - 1
            if i < resolution[0] - 1:
                neighbors[i, 1] = i + 1

    elif dim == 2:
        nx, ny = resolution
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                if i > 0:
                    neighbors[idx, 0] = (i - 1) * ny + j  # left
                if i < nx - 1:
                    neighbors[idx, 1] = (i + 1) * ny + j  # right
                if j > 0:
                    neighbors[idx, 2] = i * ny + (j - 1)  # bottom
                if j < ny - 1:
                    neighbors[idx, 3] = i * ny + (j + 1)  # top

    elif dim == 3:
        nx, ny, nz = resolution
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * ny * nz + j * nz + k
                    if i > 0:
                        neighbors[idx, 0] = (i - 1) * ny * nz + j * nz + k
                    if i < nx - 1:
                        neighbors[idx, 1] = (i + 1) * ny * nz + j * nz + k
                    if j > 0:
                        neighbors[idx, 2] = i * ny * nz + (j - 1) * nz + k
                    if j < ny - 1:
                        neighbors[idx, 3] = i * ny * nz + (j + 1) * nz + k
                    if k > 0:
                        neighbors[idx, 4] = i * ny * nz + j * nz + (k - 1)
                    if k < nz - 1:
                        neighbors[idx, 5] = i * ny * nz + j * nz + (k + 1)

    mesh._neighbors = neighbors

    # Identify boundary cells (those with -1 in neighbors)
    boundary_mask = np.any(neighbors == -1, axis=1)
    mesh.boundary_points = np.where(boundary_mask)[0]

    # Store resolution for later use
    mesh._resolution = resolution

    return mesh


def generate_point_cloud(domain: Domain,
                         n_points: int,
                         method: str = "random",
                         seed: Optional[int] = None) -> Mesh:
    """
    Generate a meshless point cloud.

    Args:
        domain: Domain for points
        n_points: Number of points
        method: 'random', 'halton', or 'poisson'
        seed: Random seed

    Returns:
        Meshless mesh with neighbor information
    """
    if method == "random":
        points = domain.sample_points(n_points, seed=seed)
    elif method == "halton":
        points = _halton_sequence(domain, n_points)
    elif method == "poisson":
        points = _poisson_disk_sampling(domain, n_points, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    mesh = Mesh(
        points=points,
        cells=None,
        mesh_type=MeshType.MESHLESS,
    )

    # Build neighbor structure
    mesh.find_neighbors(k=20)
    mesh.find_boundary()

    return mesh


def _halton_sequence(domain: Domain, n: int) -> np.ndarray:
    """Generate Halton sequence points in domain."""

    def halton(index: int, base: int) -> float:
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    primes = [2, 3, 5, 7, 11][:domain.dim]
    points = np.zeros((n, domain.dim))

    for i in range(n):
        for d in range(domain.dim):
            lo, hi = domain.bounds[d]
            points[i, d] = lo + halton(i + 1, primes[d]) * (hi - lo)

    # Filter points outside complex domains
    if domain.geometry is not None:
        mask = domain.contains(points)
        points = points[mask]
        # Resample if needed
        while len(points) < n:
            extra = domain.sample_points(n - len(points))
            points = np.vstack([points, extra])
        points = points[:n]

    return points


def _poisson_disk_sampling(domain: Domain, n: int,
                           seed: Optional[int] = None) -> np.ndarray:
    """Generate Poisson disk sampled points."""
    rng = np.random.default_rng(seed)

    # Estimate radius from target count and domain volume
    volume = 1.0
    for lo, hi in domain.bounds:
        volume *= (hi - lo)

    if domain.dim == 2:
        r = np.sqrt(volume / (n * np.pi))
    else:
        r = (3 * volume / (4 * n * np.pi)) ** (1 / 3)

    r *= 0.8  # Slightly smaller for denser sampling

    points = [domain.sample_points(1, seed=seed)[0]]
    active = [0]

    k = 30  # Candidates per point

    while active and len(points) < n * 2:  # Oversample
        idx = rng.choice(active)
        center = points[idx]

        found = False
        for _ in range(k):
            # Random point in annulus
            theta = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(r, 2 * r)

            if domain.dim == 2:
                candidate = center + rad * np.array([np.cos(theta), np.sin(theta)])
            else:
                phi = rng.uniform(0, np.pi)
                candidate = center + rad * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])

            # Check if in domain
            if not domain.contains(candidate.reshape(1, -1))[0]:
                continue

            # Check distance to existing points
            dists = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.all(dists >= r):
                points.append(candidate)
                active.append(len(points) - 1)
                found = True
                break

        if not found:
            active.remove(idx)

    # Return requested count
    points = np.array(points)
    if len(points) > n:
        indices = rng.choice(len(points), n, replace=False)
        points = points[indices]

    return points


def _tag_box_boundaries(mesh: Mesh, domain: Domain):
    """Tag boundaries of a box domain."""
    if mesh.boundary_points is None:
        return

    boundary_coords = mesh.points[mesh.boundary_points]
    tol = 1e-10

    if domain.dim == 1:
        x0, x1 = domain.bounds[0]
        mesh.tag_boundary('left', lambda p: np.abs(p[:, 0] - x0) < tol)
        mesh.tag_boundary('right', lambda p: np.abs(p[:, 0] - x1) < tol)

    elif domain.dim == 2:
        (x0, x1), (y0, y1) = domain.bounds
        mesh.tag_boundary('left', lambda p: np.abs(p[:, 0] - x0) < tol)
        mesh.tag_boundary('right', lambda p: np.abs(p[:, 0] - x1) < tol)
        mesh.tag_boundary('bottom', lambda p: np.abs(p[:, 1] - y0) < tol)
        mesh.tag_boundary('top', lambda p: np.abs(p[:, 1] - y1) < tol)

    elif domain.dim == 3:
        (x0, x1), (y0, y1), (z0, z1) = domain.bounds
        mesh.tag_boundary('left', lambda p: np.abs(p[:, 0] - x0) < tol)
        mesh.tag_boundary('right', lambda p: np.abs(p[:, 0] - x1) < tol)
        mesh.tag_boundary('front', lambda p: np.abs(p[:, 1] - y0) < tol)
        mesh.tag_boundary('back', lambda p: np.abs(p[:, 1] - y1) < tol)
        mesh.tag_boundary('bottom', lambda p: np.abs(p[:, 2] - z0) < tol)
        mesh.tag_boundary('top', lambda p: np.abs(p[:, 2] - z1) < tol)
