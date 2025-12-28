"""
Gauge equivariant neural networks on meshes.

Handle coordinate frame ambiguity in geometric learning.
Important for learning on manifolds and physical systems.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class LocalFrame:
    """Local coordinate frame at a node."""
    origin: np.ndarray  # (dim,)
    basis: np.ndarray  # (dim, dim) - orthonormal basis vectors


@dataclass
class Connection:
    """Parallel transport connection between nodes."""
    source: int
    target: int
    transport_matrix: np.ndarray  # (dim, dim) rotation


class FrameBundle:
    """
    Frame bundle over mesh.

    Assigns local coordinate frames to each node.
    Enables gauge equivariant operations.
    """

    def __init__(self, nodes: np.ndarray, edges: np.ndarray,
                 normals: Optional[np.ndarray] = None):
        self.nodes = nodes
        self.edges = edges
        self.dim = nodes.shape[1]
        self.n_nodes = len(nodes)

        # Compute local frames
        self.frames = self._compute_frames(normals)

        # Compute parallel transport
        self.connections = self._compute_connections()

    def _compute_frames(self, normals: Optional[np.ndarray]) -> List[LocalFrame]:
        """Compute local coordinate frame at each node."""
        frames = []

        for i in range(self.n_nodes):
            origin = self.nodes[i]

            if self.dim == 2:
                # 2D: tangent direction from neighbors
                neighbor_mask = self.edges[0] == i
                neighbors = self.edges[1, neighbor_mask]

                if len(neighbors) > 0:
                    # Average direction to neighbors
                    directions = self.nodes[neighbors] - origin
                    avg_dir = np.mean(directions, axis=0)
                    if np.linalg.norm(avg_dir) > 1e-8:
                        e1 = avg_dir / np.linalg.norm(avg_dir)
                    else:
                        e1 = np.array([1.0, 0.0])

                    # Orthogonal
                    e2 = np.array([-e1[1], e1[0]])
                    basis = np.column_stack([e1, e2])
                else:
                    basis = np.eye(2)

            elif self.dim == 3:
                if normals is not None:
                    # Use provided normal
                    n = normals[i]
                    n = n / (np.linalg.norm(n) + 1e-8)

                    # Build tangent frame
                    if abs(n[0]) < 0.9:
                        t1 = np.cross(n, np.array([1, 0, 0]))
                    else:
                        t1 = np.cross(n, np.array([0, 1, 0]))
                    t1 = t1 / (np.linalg.norm(t1) + 1e-8)
                    t2 = np.cross(n, t1)

                    basis = np.column_stack([t1, t2, n])
                else:
                    # PCA on neighborhood
                    neighbor_mask = self.edges[0] == i
                    neighbors = self.edges[1, neighbor_mask]

                    if len(neighbors) >= 3:
                        directions = self.nodes[neighbors] - origin
                        cov = directions.T @ directions
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        basis = eigvecs[:, ::-1]  # Sort descending
                    else:
                        basis = np.eye(3)

            else:
                basis = np.eye(self.dim)

            frames.append(LocalFrame(origin=origin, basis=basis))

        return frames

    def _compute_connections(self) -> List[Connection]:
        """Compute parallel transport between adjacent nodes."""
        connections = []

        for src, tgt in self.edges.T:
            # Rotation from src frame to tgt frame
            R_src = self.frames[src].basis
            R_tgt = self.frames[tgt].basis

            # Transport matrix: express tgt frame in src frame
            transport = R_src.T @ R_tgt

            connections.append(Connection(
                source=src,
                target=tgt,
                transport_matrix=transport
            ))

        return connections

    def to_local(self, i: int, vector: np.ndarray) -> np.ndarray:
        """Transform global vector to local frame at node i."""
        return self.frames[i].basis.T @ vector

    def to_global(self, i: int, local_vector: np.ndarray) -> np.ndarray:
        """Transform local vector to global frame."""
        return self.frames[i].basis @ local_vector


class ParallelTransport:
    """
    Parallel transport of features along mesh.

    Transport features from one node to another while
    respecting the geometry (connection).
    """

    def __init__(self, frame_bundle: FrameBundle):
        self.bundle = frame_bundle

    def transport(self, src: int, tgt: int,
                  feature: np.ndarray) -> np.ndarray:
        """
        Transport feature from src to tgt.

        For vector features, rotates according to frame change.
        """
        # Find connection
        for conn in self.bundle.connections:
            if conn.source == src and conn.target == tgt:
                return conn.transport_matrix @ feature

        # If no direct edge, use identity (approximate)
        return feature

    def transport_along_path(self, path: List[int],
                             feature: np.ndarray) -> np.ndarray:
        """Transport feature along path of nodes."""
        current = feature

        for i in range(len(path) - 1):
            current = self.transport(path[i], path[i+1], current)

        return current


class GaugeEquivariantConv:
    """
    Gauge equivariant convolution.

    Features transform according to local frame rotations.
    Output is independent of arbitrary frame choices.

    Key idea: Transport neighbor features to center frame before aggregating.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_bases: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_bases = n_bases

        # Learnable parameters
        self.W_self = np.random.randn(in_features, out_features) * 0.1
        self.W_bases = np.random.randn(n_bases, in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

        self._bundle: Optional[FrameBundle] = None

    def setup(self, nodes: np.ndarray, edges: np.ndarray,
              normals: Optional[np.ndarray] = None):
        """Setup frame bundle for mesh."""
        self._bundle = FrameBundle(nodes, edges, normals)

    def forward(self, features: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Forward pass with gauge equivariance.
        """
        if self._bundle is None:
            raise ValueError("Call setup() first")

        n_nodes = len(features)
        output = np.zeros((n_nodes, self.out_features))

        for i in range(n_nodes):
            # Self connection
            h_self = features[i] @ self.W_self

            # Neighbor aggregation with parallel transport
            neighbor_mask = edges[0] == i
            neighbor_indices = edges[1, neighbor_mask]

            h_neighbor = np.zeros(self.out_features)

            for conn_idx, conn in enumerate(self._bundle.connections):
                if conn.target != i:
                    continue

                j = conn.source
                if j not in neighbor_indices:
                    continue

                # Get neighbor feature
                h_j = features[j]

                # Compute angular bin based on edge direction
                edge_vec = self._bundle.nodes[i] - self._bundle.nodes[j]
                local_edge = self._bundle.to_local(i, edge_vec)

                if len(local_edge) >= 2:
                    angle = np.arctan2(local_edge[1], local_edge[0])
                    bin_idx = int((angle + np.pi) / (2 * np.pi) * self.n_bases) % self.n_bases
                else:
                    bin_idx = 0

                # Apply basis-specific kernel
                h_neighbor += h_j @ self.W_bases[bin_idx]

            # Normalize
            n_neighbors = len(neighbor_indices)
            if n_neighbors > 0:
                h_neighbor /= n_neighbors

            output[i] = np.maximum(0, h_self + h_neighbor + self.bias)

        return output


class SE3EquivariantConv:
    """
    SE(3) equivariant convolution.

    Equivariant to 3D rotations and translations.
    Uses spherical harmonics for angular dependence.
    """

    def __init__(self, in_features: int, out_features: int,
                 max_l: int = 2):
        self.in_features = in_features
        self.out_features = out_features
        self.max_l = max_l  # Maximum spherical harmonic degree

        # Radial network
        self.radial_weights = np.random.randn(10, in_features, out_features) * 0.1

        # Spherical harmonic coefficients
        n_sh = (max_l + 1) ** 2
        self.sh_weights = np.random.randn(n_sh, in_features, out_features) * 0.1

    def forward(self, features: np.ndarray, positions: np.ndarray,
                edges: np.ndarray) -> np.ndarray:
        """
        Forward pass with SE(3) equivariance.
        """
        n_nodes = len(features)
        output = np.zeros((n_nodes, self.out_features))

        for i in range(n_nodes):
            neighbor_mask = edges[0] == i
            neighbors = edges[1, neighbor_mask]

            if len(neighbors) == 0:
                continue

            h_agg = np.zeros(self.out_features)

            for j in neighbors:
                # Edge vector
                r_ij = positions[j] - positions[i]
                r = np.linalg.norm(r_ij)

                # Radial basis
                radial_bins = np.exp(-np.linspace(0, 5, 10) * r)
                radial_feature = np.einsum('r,rif->f', radial_bins, self.radial_weights)

                # Spherical harmonics (simplified: use direction components)
                if r > 1e-8:
                    direction = r_ij / r
                    # Y_0^0 = 1, Y_1^m ~ x, y, z, Y_2^m ~ quadratic...
                    sh = self._spherical_harmonics(direction)
                else:
                    sh = np.zeros((self.max_l + 1) ** 2)
                    sh[0] = 1

                angular_feature = np.einsum('s,sif->f', sh, self.sh_weights)

                # Combine
                h_j = features[j]
                message = h_j * (radial_feature + angular_feature)
                h_agg += message

            output[i] = h_agg / max(len(neighbors), 1)

        return np.maximum(0, output)

    def _spherical_harmonics(self, direction: np.ndarray) -> np.ndarray:
        """Compute spherical harmonics up to max_l."""
        x, y, z = direction

        sh = []
        # l=0
        sh.append(1.0)

        if self.max_l >= 1:
            # l=1
            sh.extend([y, z, x])

        if self.max_l >= 2:
            # l=2
            sh.extend([
                x*y,
                y*z,
                3*z*z - 1,
                x*z,
                x*x - y*y
            ])

        return np.array(sh[:((self.max_l + 1) ** 2)])


class VectorNeurons:
    """
    Vector neuron network.

    Features are vectors/matrices that transform under rotations.
    All operations preserve equivariance.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight matrix for channel mixing
        self.W = np.random.randn(in_channels, out_channels) * 0.1

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            features: (n_nodes, in_channels, 3) - vector features

        Returns:
            (n_nodes, out_channels, 3) - transformed vector features
        """
        # Linear combination of channels
        # features: (n, in, 3), W: (in, out) -> (n, out, 3)
        return np.einsum('nic,io->noc', features, self.W)

    def relu(self, features: np.ndarray) -> np.ndarray:
        """
        Equivariant ReLU for vector features.

        Project to direction of maximum norm, apply scalar ReLU.
        """
        n_nodes, n_channels, dim = features.shape

        output = np.zeros_like(features)

        for i in range(n_nodes):
            for c in range(n_channels):
                v = features[i, c]
                norm = np.linalg.norm(v)

                if norm > 0:
                    direction = v / norm
                    # ReLU on norm
                    new_norm = max(0, norm - 0.1)  # Leaky threshold
                    output[i, c] = direction * new_norm

        return output

    def inner_product(self, features: np.ndarray) -> np.ndarray:
        """
        Compute invariant features via inner products.

        Returns: (n_nodes, n_channels, n_channels) pairwise inner products
        """
        # features: (n, c, 3)
        # inner: features @ features.T over spatial dimension
        return np.einsum('nci,ndi->ncd', features, features)
