"""
Graph convolutions on computational meshes.

Treat mesh as graph and apply message passing neural networks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy import sparse


@dataclass
class MeshGraph:
    """Graph representation of a mesh."""
    nodes: np.ndarray  # (n_nodes, n_features)
    edges: np.ndarray  # (2, n_edges) - source, target indices
    edge_attr: Optional[np.ndarray] = None  # (n_edges, n_edge_features)
    node_pos: Optional[np.ndarray] = None  # (n_nodes, dim)


class MeshConv:
    """
    Basic message passing convolution on mesh.

    Each node aggregates features from neighbors:
    h_i' = UPDATE(h_i, AGGREGATE({h_j : j in N(i)}))
    """

    def __init__(self, in_features: int, out_features: int,
                 aggregation: str = "mean"):
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation

        # Learnable parameters
        self.W_self = np.random.randn(in_features, out_features) * 0.1
        self.W_neighbor = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, graph: MeshGraph) -> np.ndarray:
        """
        Forward pass.

        Args:
            graph: MeshGraph with node features

        Returns:
            Updated node features (n_nodes, out_features)
        """
        n_nodes = len(graph.nodes)

        # Self transformation
        h_self = graph.nodes @ self.W_self

        # Neighbor aggregation
        h_neighbor = np.zeros((n_nodes, self.out_features))
        neighbor_count = np.zeros(n_nodes)

        for src, tgt in graph.edges.T:
            # Message from src to tgt
            message = graph.nodes[src] @ self.W_neighbor
            h_neighbor[tgt] += message
            neighbor_count[tgt] += 1

        # Normalize by neighbor count for mean aggregation
        if self.aggregation == "mean":
            h_neighbor /= np.maximum(neighbor_count[:, np.newaxis], 1)
        elif self.aggregation == "sum":
            pass  # Already summed
        elif self.aggregation == "max":
            # Would need different accumulation logic
            pass

        # Combine and activate
        output = h_self + h_neighbor + self.bias
        return np.maximum(0, output)  # ReLU

    def build_adjacency(self, graph: MeshGraph) -> sparse.csr_matrix:
        """Build sparse adjacency matrix from edge list."""
        n_nodes = len(graph.nodes)
        data = np.ones(graph.edges.shape[1])
        adj = sparse.csr_matrix(
            (data, (graph.edges[0], graph.edges[1])),
            shape=(n_nodes, n_nodes)
        )
        return adj


class GeodesicConv:
    """
    Geodesic convolution respecting mesh geometry.

    Uses geodesic distances instead of hop distances.
    Better for curved surfaces where Euclidean distance is misleading.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_hops: int = 2, use_geodesic: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.n_hops = n_hops
        self.use_geodesic = use_geodesic

        # Distance-weighted kernel
        self.kernel_weights = np.random.randn(n_hops + 1, in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, graph: MeshGraph) -> np.ndarray:
        """
        Forward pass with geodesic weighting.
        """
        n_nodes = len(graph.nodes)

        # Compute geodesic distances
        if self.use_geodesic and graph.node_pos is not None:
            distances = self._compute_geodesic_distances(graph)
        else:
            distances = self._compute_hop_distances(graph)

        # Aggregate by distance bands
        output = np.zeros((n_nodes, self.out_features))

        for k in range(self.n_hops + 1):
            # Find nodes at hop distance k
            if k == 0:
                # Self connection
                output += graph.nodes @ self.kernel_weights[k]
            else:
                # k-hop neighbors
                for i in range(n_nodes):
                    neighbors = np.where(distances[i] == k)[0]
                    if len(neighbors) > 0:
                        agg = np.mean(graph.nodes[neighbors], axis=0)
                        output[i] += agg @ self.kernel_weights[k]

        return np.maximum(0, output + self.bias)

    def _compute_hop_distances(self, graph: MeshGraph) -> np.ndarray:
        """Compute shortest path distances in hops."""
        n_nodes = len(graph.nodes)
        adj = self._build_adjacency(graph)

        # BFS from each node (simplified)
        distances = np.full((n_nodes, n_nodes), self.n_hops + 1)
        np.fill_diagonal(distances, 0)

        current = adj.toarray()
        for k in range(1, self.n_hops + 1):
            reachable = current > 0
            distances[reachable & (distances > k)] = k
            current = current @ adj.toarray()

        return distances

    def _compute_geodesic_distances(self, graph: MeshGraph) -> np.ndarray:
        """Approximate geodesic distances using Dijkstra on mesh edges."""
        n_nodes = len(graph.nodes)
        pos = graph.node_pos

        # Build weighted adjacency (edge weights = Euclidean distances)
        n_edges = graph.edges.shape[1]
        edge_lengths = np.linalg.norm(
            pos[graph.edges[1]] - pos[graph.edges[0]], axis=1
        )

        # Use scipy's shortest path
        adj = sparse.csr_matrix(
            (edge_lengths, (graph.edges[0], graph.edges[1])),
            shape=(n_nodes, n_nodes)
        )
        adj = adj + adj.T  # Make symmetric

        from scipy.sparse.csgraph import dijkstra
        distances = dijkstra(adj, directed=False, limit=self.n_hops * np.max(edge_lengths))

        # Discretize to hop-like bands
        max_dist = np.max(distances[distances < np.inf])
        band_width = max_dist / self.n_hops
        hop_distances = np.minimum(np.floor(distances / band_width).astype(int), self.n_hops)

        return hop_distances

    def _build_adjacency(self, graph: MeshGraph) -> sparse.csr_matrix:
        """Build sparse adjacency matrix."""
        n_nodes = len(graph.nodes)
        data = np.ones(graph.edges.shape[1])
        adj = sparse.csr_matrix(
            (data, (graph.edges[0], graph.edges[1])),
            shape=(n_nodes, n_nodes)
        )
        return adj + adj.T


class AnisotropicConv:
    """
    Anisotropic convolution with directional awareness.

    Uses edge directions to apply directional filters.
    Important for physics where gradients have direction.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_directions: int = 8):
        self.in_features = in_features
        self.out_features = out_features
        self.n_directions = n_directions

        # Direction-specific kernels
        self.direction_kernels = np.random.randn(
            n_directions, in_features, out_features
        ) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, graph: MeshGraph) -> np.ndarray:
        """
        Forward pass with directional filtering.
        """
        if graph.node_pos is None:
            raise ValueError("AnisotropicConv requires node positions")

        n_nodes = len(graph.nodes)
        dim = graph.node_pos.shape[1]

        # Compute edge directions
        edge_vectors = graph.node_pos[graph.edges[1]] - graph.node_pos[graph.edges[0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        edge_dirs = edge_vectors / (edge_lengths + 1e-8)

        # Quantize directions
        direction_bins = self._quantize_directions(edge_dirs, dim)

        # Aggregate by direction
        output = np.zeros((n_nodes, self.out_features))

        for d in range(self.n_directions):
            mask = direction_bins == d
            edges_in_dir = graph.edges[:, mask]

            for src, tgt in edges_in_dir.T:
                message = graph.nodes[src] @ self.direction_kernels[d]
                output[tgt] += message

        # Normalize
        degree = np.bincount(graph.edges[1], minlength=n_nodes).astype(float)
        output /= np.maximum(degree[:, np.newaxis], 1)

        return np.maximum(0, output + self.bias)

    def _quantize_directions(self, directions: np.ndarray, dim: int) -> np.ndarray:
        """Quantize continuous directions to discrete bins."""
        if dim == 2:
            # Angle-based quantization
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            bins = ((angles + np.pi) / (2 * np.pi) * self.n_directions).astype(int)
            return bins % self.n_directions
        else:
            # 3D: use spherical coordinates or clustering
            # Simplified: project to 2D and use angle
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            bins = ((angles + np.pi) / (2 * np.pi) * self.n_directions).astype(int)
            return bins % self.n_directions


class EdgeConv:
    """
    Edge convolution (from DGCNN).

    Dynamically computes graph based on feature similarity.
    Learns edge features explicitly.
    """

    def __init__(self, in_features: int, out_features: int, k: int = 20):
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # Edge MLP
        self.W1 = np.random.randn(2 * in_features, out_features) * 0.1
        self.W2 = np.random.randn(out_features, out_features) * 0.1
        self.bias1 = np.zeros(out_features)
        self.bias2 = np.zeros(out_features)

    def forward(self, features: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Forward pass with dynamic graph construction.

        Args:
            features: Node features (n_nodes, in_features)
            positions: Node positions for kNN (n_nodes, dim)

        Returns:
            Updated features (n_nodes, out_features)
        """
        n_nodes = len(features)

        # Build kNN graph
        edges = self._build_knn_graph(positions)

        # Compute edge features
        output = np.zeros((n_nodes, self.out_features))

        for i in range(n_nodes):
            neighbors = edges[i]
            if len(neighbors) == 0:
                continue

            # Edge features: [h_i, h_j - h_i]
            h_i = features[i]
            h_j = features[neighbors]
            edge_features = np.hstack([
                np.tile(h_i, (len(neighbors), 1)),
                h_j - h_i
            ])

            # Edge MLP
            e = edge_features @ self.W1 + self.bias1
            e = np.maximum(0, e)
            e = e @ self.W2 + self.bias2

            # Max aggregation
            output[i] = np.max(e, axis=0)

        return np.maximum(0, output)

    def _build_knn_graph(self, positions: np.ndarray) -> List[np.ndarray]:
        """Build k-nearest neighbor graph."""
        n_nodes = len(positions)
        k = min(self.k, n_nodes - 1)

        # Compute all pairwise distances
        diff = positions[:, np.newaxis] - positions[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)

        # Find k nearest neighbors for each node
        edges = []
        for i in range(n_nodes):
            # Exclude self
            dists[i, i] = np.inf
            neighbors = np.argsort(dists[i])[:k]
            edges.append(neighbors)

        return edges


class AttentionConv:
    """
    Graph attention convolution on mesh.

    Learns attention weights for neighbor aggregation.
    Useful when some neighbors are more relevant than others.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        # Attention parameters
        self.W_query = np.random.randn(in_features, out_features) * 0.1
        self.W_key = np.random.randn(in_features, out_features) * 0.1
        self.W_value = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, graph: MeshGraph) -> np.ndarray:
        """
        Forward pass with attention.
        """
        n_nodes = len(graph.nodes)

        # Compute Q, K, V
        Q = graph.nodes @ self.W_query
        K = graph.nodes @ self.W_key
        V = graph.nodes @ self.W_value

        # Reshape for multi-head attention
        Q = Q.reshape(n_nodes, self.n_heads, self.head_dim)
        K = K.reshape(n_nodes, self.n_heads, self.head_dim)
        V = V.reshape(n_nodes, self.n_heads, self.head_dim)

        output = np.zeros((n_nodes, self.n_heads, self.head_dim))

        # Compute attention for each node
        for i in range(n_nodes):
            # Find neighbors
            neighbor_mask = graph.edges[1] == i
            neighbors = graph.edges[0, neighbor_mask]

            if len(neighbors) == 0:
                output[i] = V[i]
                continue

            # Include self
            all_nodes = np.concatenate([[i], neighbors])

            # Attention scores
            scores = np.einsum('hd,nhd->nh', Q[i], K[all_nodes])
            scores /= np.sqrt(self.head_dim)

            # Softmax
            scores_exp = np.exp(scores - np.max(scores, axis=0, keepdims=True))
            attention = scores_exp / np.sum(scores_exp, axis=0, keepdims=True)

            # Weighted sum
            output[i] = np.einsum('nh,nhd->hd', attention, V[all_nodes])

        # Reshape back
        output = output.reshape(n_nodes, self.out_features)

        return np.maximum(0, output + self.bias)
