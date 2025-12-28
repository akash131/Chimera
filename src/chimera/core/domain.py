"""
Domain representation for Chimera.

The Domain class provides a unified interface for defining computational
domains regardless of the underlying discretization method.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Tuple, List
import numpy as np


@dataclass
class Domain:
    """
    Computational domain definition.

    Supports multiple domain types:
    - Box domains (1D, 2D, 3D)
    - Implicit domains (level-set functions)
    - Mesh-based domains

    Attributes:
        dim: Spatial dimension (1, 2, or 3)
        bounds: Axis-aligned bounding box [(xmin, xmax), (ymin, ymax), ...]
        geometry: Optional callable f(x) < 0 inside, > 0 outside (level-set)
        name: Optional domain identifier
    """
    dim: int
    bounds: List[Tuple[float, float]]
    geometry: Optional[Callable[[np.ndarray], np.ndarray]] = None
    name: str = "domain"

    def __post_init__(self):
        if len(self.bounds) != self.dim:
            raise ValueError(f"bounds must have {self.dim} entries for {self.dim}D domain")
        for i, (lo, hi) in enumerate(self.bounds):
            if lo >= hi:
                raise ValueError(f"Invalid bounds on axis {i}: {lo} >= {hi}")

    @classmethod
    def box(cls, *bounds: Tuple[float, float], name: str = "box") -> Domain:
        """Create a box domain from bounds."""
        return cls(dim=len(bounds), bounds=list(bounds), name=name)

    @classmethod
    def unit_square(cls) -> Domain:
        """Create the unit square [0,1]^2."""
        return cls.box((0.0, 1.0), (0.0, 1.0), name="unit_square")

    @classmethod
    def unit_cube(cls) -> Domain:
        """Create the unit cube [0,1]^3."""
        return cls.box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), name="unit_cube")

    @classmethod
    def circle(cls, center: Tuple[float, float] = (0.0, 0.0),
               radius: float = 1.0) -> Domain:
        """Create a circular domain."""
        cx, cy = center
        def level_set(x: np.ndarray) -> np.ndarray:
            return (x[..., 0] - cx)**2 + (x[..., 1] - cy)**2 - radius**2

        bounds = [(cx - radius, cx + radius), (cy - radius, cy + radius)]
        return cls(dim=2, bounds=bounds, geometry=level_set, name="circle")

    @classmethod
    def sphere(cls, center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
               radius: float = 1.0) -> Domain:
        """Create a spherical domain."""
        cx, cy, cz = center
        def level_set(x: np.ndarray) -> np.ndarray:
            return ((x[..., 0] - cx)**2 + (x[..., 1] - cy)**2 +
                    (x[..., 2] - cz)**2 - radius**2)

        bounds = [(cx - radius, cx + radius),
                  (cy - radius, cy + radius),
                  (cz - radius, cz + radius)]
        return cls(dim=3, bounds=bounds, geometry=level_set, name="sphere")

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside the domain."""
        points = np.atleast_2d(points)

        # Check bounding box
        in_bounds = np.ones(len(points), dtype=bool)
        for i, (lo, hi) in enumerate(self.bounds):
            in_bounds &= (points[:, i] >= lo) & (points[:, i] <= hi)

        # Check geometry if present
        if self.geometry is not None:
            in_geom = self.geometry(points) <= 0
            return in_bounds & in_geom

        return in_bounds

    def sample_points(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside the domain."""
        rng = np.random.default_rng(seed)

        # Oversample and reject
        points = []
        while len(points) < n:
            batch_size = max(n * 2, 1000)
            candidates = np.zeros((batch_size, self.dim))
            for i, (lo, hi) in enumerate(self.bounds):
                candidates[:, i] = rng.uniform(lo, hi, batch_size)

            mask = self.contains(candidates)
            points.extend(candidates[mask])

        return np.array(points[:n])

    def boundary_points(self, n_per_face: int = 10) -> np.ndarray:
        """Generate points on the boundary (for box domains)."""
        if self.geometry is not None:
            raise NotImplementedError(
                "boundary_points for implicit geometry requires marching cubes"
            )

        points = []

        if self.dim == 1:
            points = [[self.bounds[0][0]], [self.bounds[0][1]]]

        elif self.dim == 2:
            (x0, x1), (y0, y1) = self.bounds
            # Four edges
            t = np.linspace(0, 1, n_per_face)
            points.extend([[x0 + t_i * (x1 - x0), y0] for t_i in t])  # bottom
            points.extend([[x0 + t_i * (x1 - x0), y1] for t_i in t])  # top
            points.extend([[x0, y0 + t_i * (y1 - y0)] for t_i in t])  # left
            points.extend([[x1, y0 + t_i * (y1 - y0)] for t_i in t])  # right

        elif self.dim == 3:
            (x0, x1), (y0, y1), (z0, z1) = self.bounds
            # Six faces - sample grid on each
            t = np.linspace(0, 1, n_per_face)
            tt, ss = np.meshgrid(t, t)
            tt, ss = tt.flatten(), ss.flatten()

            # x-faces
            for x in [x0, x1]:
                for ti, si in zip(tt, ss):
                    points.append([x, y0 + ti * (y1 - y0), z0 + si * (z1 - z0)])
            # y-faces
            for y in [y0, y1]:
                for ti, si in zip(tt, ss):
                    points.append([x0 + ti * (x1 - x0), y, z0 + si * (z1 - z0)])
            # z-faces
            for z in [z0, z1]:
                for ti, si in zip(tt, ss):
                    points.append([x0 + ti * (x1 - x0), y0 + si * (y1 - y0), z])

        return np.array(points)

    def __repr__(self) -> str:
        return f"Domain('{self.name}', dim={self.dim}, bounds={self.bounds})"
