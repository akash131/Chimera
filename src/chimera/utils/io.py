"""
I/O utilities for Chimera.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json
import numpy as np
from pathlib import Path


def save_results(filepath: str, result, mesh=None, metadata: Optional[Dict] = None):
    """
    Save solver results to file.

    Args:
        filepath: Output path (.npz for numpy, .vtk for visualization)
        result: SolverResult object
        mesh: Optional mesh to save alongside
        metadata: Additional metadata
    """
    path = Path(filepath)

    if path.suffix == '.npz':
        _save_npz(path, result, mesh, metadata)
    elif path.suffix in ['.vtk', '.vtu']:
        _save_vtk(path, result, mesh)
    elif path.suffix == '.h5':
        _save_hdf5(path, result, mesh, metadata)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def _save_npz(path: Path, result, mesh, metadata):
    """Save to numpy archive."""
    data = {
        'status': result.status.value,
        'residual': result.residual,
        'iterations': result.iterations,
        'solve_time': result.solve_time,
    }

    # Save fields
    for name, field in result.fields.items():
        data[f'field_{name}_values'] = field.values
        if field.points is not None:
            data[f'field_{name}_points'] = field.points

    # Save mesh
    if mesh is not None:
        data['mesh_points'] = mesh.points
        if mesh.cells is not None:
            data['mesh_cells'] = mesh.cells

    # Save metadata
    if metadata:
        data['metadata'] = json.dumps(metadata)

    np.savez(path, **data)


def _save_vtk(path: Path, result, mesh):
    """Save to VTK format for visualization."""
    import meshio

    if mesh is None:
        raise ValueError("Mesh required for VTK output")

    point_data = {}
    for name, field in result.fields.items():
        point_data[name] = field.values

    if mesh.cells is not None:
        cell_type_map = {
            'triangle': 'triangle',
            'quad': 'quad',
            'tetrahedron': 'tetra',
        }
        from chimera.mesh.mesh import CellType
        type_name = mesh.cell_type.name.lower()
        vtk_type = cell_type_map.get(type_name, 'triangle')

        meshio_mesh = meshio.Mesh(
            mesh.points,
            [(vtk_type, mesh.cells)],
            point_data=point_data
        )
    else:
        meshio_mesh = meshio.Mesh(mesh.points, [], point_data=point_data)

    meshio_mesh.write(path)


def _save_hdf5(path: Path, result, mesh, metadata):
    """Save to HDF5 format."""
    import h5py

    with h5py.File(path, 'w') as f:
        # Result info
        f.attrs['status'] = result.status.value
        f.attrs['residual'] = result.residual
        f.attrs['iterations'] = result.iterations
        f.attrs['solve_time'] = result.solve_time

        # Fields
        fields_grp = f.create_group('fields')
        for name, field in result.fields.items():
            field_grp = fields_grp.create_group(name)
            field_grp.create_dataset('values', data=field.values)
            if field.points is not None:
                field_grp.create_dataset('points', data=field.points)

        # Mesh
        if mesh is not None:
            mesh_grp = f.create_group('mesh')
            mesh_grp.create_dataset('points', data=mesh.points)
            if mesh.cells is not None:
                mesh_grp.create_dataset('cells', data=mesh.cells)

        # Metadata
        if metadata:
            f.attrs['metadata'] = json.dumps(metadata)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load saved results.

    Args:
        filepath: Path to saved results

    Returns:
        Dictionary with loaded data
    """
    path = Path(filepath)

    if path.suffix == '.npz':
        return _load_npz(path)
    elif path.suffix == '.h5':
        return _load_hdf5(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def _load_npz(path: Path) -> Dict[str, Any]:
    """Load from numpy archive."""
    data = dict(np.load(path, allow_pickle=True))

    result = {
        'status': data['status'],
        'residual': float(data['residual']),
        'iterations': int(data['iterations']),
        'solve_time': float(data['solve_time']),
        'fields': {},
    }

    # Extract fields
    for key in data:
        if key.startswith('field_') and key.endswith('_values'):
            name = key[6:-7]  # Remove 'field_' prefix and '_values' suffix
            result['fields'][name] = {
                'values': data[key],
                'points': data.get(f'field_{name}_points')
            }

    # Mesh
    if 'mesh_points' in data:
        result['mesh'] = {
            'points': data['mesh_points'],
            'cells': data.get('mesh_cells')
        }

    # Metadata
    if 'metadata' in data:
        result['metadata'] = json.loads(str(data['metadata']))

    return result


def _load_hdf5(path: Path) -> Dict[str, Any]:
    """Load from HDF5."""
    import h5py

    with h5py.File(path, 'r') as f:
        result = {
            'status': f.attrs['status'],
            'residual': f.attrs['residual'],
            'iterations': f.attrs['iterations'],
            'solve_time': f.attrs['solve_time'],
            'fields': {},
        }

        # Fields
        if 'fields' in f:
            for name in f['fields']:
                grp = f['fields'][name]
                result['fields'][name] = {
                    'values': grp['values'][()],
                    'points': grp['points'][()] if 'points' in grp else None
                }

        # Mesh
        if 'mesh' in f:
            result['mesh'] = {
                'points': f['mesh']['points'][()],
                'cells': f['mesh']['cells'][()] if 'cells' in f['mesh'] else None
            }

        # Metadata
        if 'metadata' in f.attrs:
            result['metadata'] = json.loads(f.attrs['metadata'])

    return result
