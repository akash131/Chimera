"""
Visualization utilities for Chimera.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def plot_field(field, mesh=None, title: str = "",
               cmap: str = "viridis", show: bool = True,
               save_path: Optional[str] = None):
    """
    Plot a scalar field on a mesh.

    Args:
        field: Field object or array of values
        mesh: Mesh object (uses field.points if not provided)
        title: Plot title
        cmap: Colormap name
        show: Whether to display plot
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    # Extract data
    if hasattr(field, 'values'):
        values = field.values
        points = field.points if mesh is None else mesh.points
    else:
        values = field
        points = mesh.points

    dim = points.shape[1]

    if dim == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(points[:, 0], values, 'b-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(title)
        plt.grid(True)

    elif dim == 2:
        plt.figure(figsize=(10, 8))

        if mesh is not None and mesh.cells is not None:
            # Use triangulation
            tri = Triangulation(points[:, 0], points[:, 1], mesh.cells)
            plt.tripcolor(tri, values.flatten(), cmap=cmap, shading='gouraud')
        else:
            # Scatter plot
            plt.scatter(points[:, 0], points[:, 1], c=values, cmap=cmap, s=10)

        plt.colorbar(label='Value')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.axis('equal')

    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=values, cmap=cmap, s=10)
        plt.colorbar(scatter)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_mesh(mesh, field=None, show_boundaries: bool = True,
              title: str = "", show: bool = True):
    """
    Plot mesh with optional field overlay.

    Args:
        mesh: Mesh object
        field: Optional field to overlay
        show_boundaries: Highlight boundary points
        title: Plot title
        show: Whether to display plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    points = mesh.points
    dim = mesh.dim

    if dim == 1:
        plt.figure(figsize=(10, 2))
        plt.scatter(points[:, 0], np.zeros(len(points)), s=50)
        plt.title(title)

    elif dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        if mesh.cells is not None:
            # Draw cells
            polygons = []
            for cell in mesh.cells:
                poly = points[cell]
                polygons.append(poly)

            if field is not None:
                values = field.values if hasattr(field, 'values') else field
                colors = values[mesh.cells].mean(axis=1) if len(values) == len(points) else values
                pc = PolyCollection(polygons, array=colors, cmap='viridis',
                                    edgecolors='k', linewidths=0.5)
                ax.add_collection(pc)
                plt.colorbar(pc)
            else:
                pc = PolyCollection(polygons, facecolors='lightblue',
                                    edgecolors='k', linewidths=0.5)
                ax.add_collection(pc)

        else:
            # Point cloud
            if field is not None:
                values = field.values if hasattr(field, 'values') else field
                plt.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', s=10)
            else:
                plt.scatter(points[:, 0], points[:, 1], s=10)

        if show_boundaries and mesh.boundary_points is not None:
            bp = points[mesh.boundary_points]
            plt.scatter(bp[:, 0], bp[:, 1], c='red', s=30, marker='s',
                        label='Boundary', zorder=5)
            plt.legend()

        ax.set_xlim(points[:, 0].min() - 0.1, points[:, 0].max() + 0.1)
        ax.set_ylim(points[:, 1].min() - 0.1, points[:, 1].max() + 0.1)
        ax.set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence(history: list, quantities: list = ['residual'],
                     log_scale: bool = True, show: bool = True):
    """
    Plot convergence history.

    Args:
        history: List of dicts with iteration data
        quantities: Which quantities to plot
        log_scale: Use log scale for y-axis
        show: Whether to display plot
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for qty in quantities:
        values = [h.get(qty, np.nan) for h in history]
        iters = range(len(values))
        ax.plot(iters, values, 'o-', label=qty)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    plt.title('Convergence History')

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(field1, field2, mesh, labels: Tuple[str, str] = ('Field 1', 'Field 2'),
                    show: bool = True):
    """
    Plot two fields side by side for comparison.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    points = mesh.points

    if mesh.cells is not None:
        tri = Triangulation(points[:, 0], points[:, 1], mesh.cells)

        v1 = field1.values if hasattr(field1, 'values') else field1
        v2 = field2.values if hasattr(field2, 'values') else field2

        vmin = min(v1.min(), v2.min())
        vmax = max(v1.max(), v2.max())

        im1 = ax1.tripcolor(tri, v1.flatten(), cmap='viridis',
                            shading='gouraud', vmin=vmin, vmax=vmax)
        ax1.set_title(labels[0])
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.tripcolor(tri, v2.flatten(), cmap='viridis',
                            shading='gouraud', vmin=vmin, vmax=vmax)
        ax2.set_title(labels[1])
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()
