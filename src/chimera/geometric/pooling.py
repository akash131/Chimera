"""
Mesh pooling and unpooling for multi-scale learning.

Coarsen and refine meshes while preserving geometric structure.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import sparse


@dataclass
class PoolingInfo:
    """Information for pooling/unpooling."""
    cluster_assignment: np.ndarray  # (n_nodes,) - which cluster each node belongs to
    n_clusters: int
    pooling_matrix: sparse.csr_matrix  # (n_clusters, n_nodes)


@dataclass
class MeshHierarchy:
    """Hierarchical mesh representation."""
    levels: List[np.ndarray]  # Node positions at each level
    edges_per_level: List[np.ndarray]  # Edges at each level
    pooling_ops: List[PoolingInfo]  # Pooling between levels


class MeshPooling:
    """
    Pool mesh nodes using graph clustering.

    Supports multiple methods:
    - Graclus: edge-weighted matching
    - k-means: geometric clustering
    - Edge contraction: topology-aware
    """

    def __init__(self, method: str = "graclus", ratio: float = 0.5):
        self.method = method
        self.ratio = ratio  # Target size ratio

    def pool(self, nodes: np.ndarray, edges: np.ndarray,
             features: np.ndarray,
             edge_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PoolingInfo]:
        """
        Pool mesh to coarser level.

        Args:
            nodes: Node positions (n_nodes, dim)
            edges: Edge list (2, n_edges)
            features: Node features (n_nodes, n_features)
            edge_weights: Optional edge weights

        Returns:
            (coarse_nodes, coarse_edges, coarse_features, pooling_info)
        """
        if self.method == "graclus":
            return self._graclus_pool(nodes, edges, features, edge_weights)
        elif self.method == "kmeans":
            return self._kmeans_pool(nodes, edges, features)
        elif self.method == "voxel":
            return self._voxel_pool(nodes, edges, features)
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")

    def _graclus_pool(self, nodes, edges, features, edge_weights):
        """
        Graclus-style weighted matching.

        Greedy matching that maximizes edge weights.
        """
        n_nodes = len(nodes)
        n_clusters = int(n_nodes * self.ratio)

        # Build adjacency
        if edge_weights is None:
            edge_weights = np.ones(edges.shape[1])

        # Greedy matching
        matched = np.full(n_nodes, -1)
        cluster_id = 0

        # Process nodes in random order
        order = np.random.permutation(n_nodes)

        for i in order:
            if matched[i] >= 0:
                continue

            # Find best unmatched neighbor
            neighbor_mask = edges[0] == i
            neighbors = edges[1, neighbor_mask]
            weights = edge_weights[neighbor_mask]

            best_neighbor = None
            best_weight = -np.inf

            for j, w in zip(neighbors, weights):
                if matched[j] < 0 and w > best_weight:
                    best_weight = w
                    best_neighbor = j

            # Match
            matched[i] = cluster_id
            if best_neighbor is not None:
                matched[best_neighbor] = cluster_id
            cluster_id += 1

        # Handle unmatched nodes
        for i in range(n_nodes):
            if matched[i] < 0:
                matched[i] = cluster_id
                cluster_id += 1

        n_clusters = cluster_id

        # Build pooling matrix
        pooling_matrix = sparse.csr_matrix(
            (np.ones(n_nodes), (matched, np.arange(n_nodes))),
            shape=(n_clusters, n_nodes)
        )

        # Normalize rows
        row_sums = np.array(pooling_matrix.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (row_sums + 1e-8))
        pooling_matrix = D_inv @ pooling_matrix

        # Compute coarse quantities
        coarse_nodes = pooling_matrix @ nodes
        coarse_features = pooling_matrix @ features

        # Compute coarse edges
        coarse_edges = self._compute_coarse_edges(edges, matched, n_clusters)

        pooling_info = PoolingInfo(
            cluster_assignment=matched,
            n_clusters=n_clusters,
            pooling_matrix=pooling_matrix
        )

        return coarse_nodes, coarse_edges, coarse_features, pooling_info

    def _kmeans_pool(self, nodes, edges, features):
        """K-means clustering based on positions."""
        n_nodes = len(nodes)
        n_clusters = int(n_nodes * self.ratio)

        # Simple k-means
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(nodes, n_clusters, minit='++')

        # Build pooling matrix
        pooling_matrix = sparse.csr_matrix(
            (np.ones(n_nodes), (labels, np.arange(n_nodes))),
            shape=(n_clusters, n_nodes)
        )

        row_sums = np.array(pooling_matrix.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (row_sums + 1e-8))
        pooling_matrix = D_inv @ pooling_matrix

        coarse_nodes = centroids
        coarse_features = pooling_matrix @ features
        coarse_edges = self._compute_coarse_edges(edges, labels, n_clusters)

        pooling_info = PoolingInfo(
            cluster_assignment=labels,
            n_clusters=n_clusters,
            pooling_matrix=pooling_matrix
        )

        return coarse_nodes, coarse_edges, coarse_features, pooling_info

    def _voxel_pool(self, nodes, edges, features):
        """Voxel-based pooling (regular grid)."""
        n_nodes = len(nodes)

        # Compute voxel size based on ratio
        bbox_min = nodes.min(axis=0)
        bbox_max = nodes.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Estimate voxel count
        target_voxels = int(n_nodes * self.ratio)
        dim = nodes.shape[1]
        voxel_size = (np.prod(bbox_size) / target_voxels) ** (1/dim)

        # Assign to voxels
        voxel_coords = ((nodes - bbox_min) / voxel_size).astype(int)
        voxel_ids = {}
        labels = np.zeros(n_nodes, dtype=int)

        for i, coord in enumerate(voxel_coords):
            key = tuple(coord)
            if key not in voxel_ids:
                voxel_ids[key] = len(voxel_ids)
            labels[i] = voxel_ids[key]

        n_clusters = len(voxel_ids)

        # Build pooling matrix
        pooling_matrix = sparse.csr_matrix(
            (np.ones(n_nodes), (labels, np.arange(n_nodes))),
            shape=(n_clusters, n_nodes)
        )

        row_sums = np.array(pooling_matrix.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (row_sums + 1e-8))
        pooling_matrix = D_inv @ pooling_matrix

        coarse_nodes = pooling_matrix @ nodes
        coarse_features = pooling_matrix @ features
        coarse_edges = self._compute_coarse_edges(edges, labels, n_clusters)

        pooling_info = PoolingInfo(
            cluster_assignment=labels,
            n_clusters=n_clusters,
            pooling_matrix=pooling_matrix
        )

        return coarse_nodes, coarse_edges, coarse_features, pooling_info

    def _compute_coarse_edges(self, edges: np.ndarray, labels: np.ndarray,
                               n_clusters: int) -> np.ndarray:
        """Compute edges in coarsened graph."""
        coarse_edges_set = set()

        for src, tgt in edges.T:
            c_src = labels[src]
            c_tgt = labels[tgt]
            if c_src != c_tgt:
                edge = (min(c_src, c_tgt), max(c_src, c_tgt))
                coarse_edges_set.add(edge)

        if not coarse_edges_set:
            return np.zeros((2, 0), dtype=int)

        coarse_edges = np.array(list(coarse_edges_set)).T
        # Make bidirectional
        coarse_edges = np.hstack([coarse_edges, coarse_edges[::-1]])

        return coarse_edges


class MeshUnpooling:
    """
    Unpool features from coarse to fine mesh.

    Uses pooling information to interpolate features.
    """

    def __init__(self, method: str = "interp"):
        self.method = method

    def unpool(self, coarse_features: np.ndarray,
               pooling_info: PoolingInfo) -> np.ndarray:
        """
        Unpool features.

        Args:
            coarse_features: Features at coarse level (n_clusters, n_features)
            pooling_info: Pooling information from MeshPooling

        Returns:
            Features at fine level (n_nodes, n_features)
        """
        # Simple: each fine node gets features of its cluster
        return coarse_features[pooling_info.cluster_assignment]


class MultiscaleMesh:
    """
    Multi-scale mesh representation.

    Build hierarchy of coarsened meshes for multi-scale processing.
    Useful for capturing both local and global features.
    """

    def __init__(self, n_levels: int = 4, pooling_ratio: float = 0.5):
        self.n_levels = n_levels
        self.pooling_ratio = pooling_ratio
        self._hierarchy: Optional[MeshHierarchy] = None
        self._pooling_ops: List[PoolingInfo] = []

    def build_hierarchy(self, nodes: np.ndarray,
                        edges: np.ndarray) -> MeshHierarchy:
        """
        Build mesh hierarchy.

        Args:
            nodes: Original node positions
            edges: Original edge list

        Returns:
            MeshHierarchy with multiple levels
        """
        levels = [nodes]
        edges_per_level = [edges]
        pooling_ops = []

        current_nodes = nodes
        current_edges = edges
        current_features = nodes  # Use positions as initial features

        pooler = MeshPooling(method="graclus", ratio=self.pooling_ratio)

        for level in range(1, self.n_levels):
            if len(current_nodes) < 10:
                break

            coarse_nodes, coarse_edges, _, pool_info = pooler.pool(
                current_nodes, current_edges, current_features
            )

            levels.append(coarse_nodes)
            edges_per_level.append(coarse_edges)
            pooling_ops.append(pool_info)

            current_nodes = coarse_nodes
            current_edges = coarse_edges
            current_features = coarse_nodes

        self._hierarchy = MeshHierarchy(
            levels=levels,
            edges_per_level=edges_per_level,
            pooling_ops=pooling_ops
        )
        self._pooling_ops = pooling_ops

        return self._hierarchy

    def encode(self, features: np.ndarray,
               encoders: List) -> List[np.ndarray]:
        """
        Encode features at each level of hierarchy.

        Args:
            features: Input features at finest level
            encoders: List of encoder networks (one per level)

        Returns:
            Encoded features at each level
        """
        if self._hierarchy is None:
            raise ValueError("Call build_hierarchy() first")

        encoded = [encoders[0](features)]

        for level in range(1, len(self._hierarchy.levels)):
            # Pool to this level
            pooled = self._pooling_ops[level-1].pooling_matrix @ encoded[-1]
            # Encode
            encoded.append(encoders[level](pooled))

        return encoded

    def decode(self, features_per_level: List[np.ndarray],
               decoders: List) -> np.ndarray:
        """
        Decode features from coarse to fine.

        Uses skip connections from encoding.
        """
        if self._hierarchy is None:
            raise ValueError("Call build_hierarchy() first")

        n_levels = len(features_per_level)
        unpooler = MeshUnpooling()

        current = features_per_level[-1]

        for level in range(n_levels - 2, -1, -1):
            # Unpool
            unpooled = unpooler.unpool(current, self._pooling_ops[level])

            # Skip connection
            skip = features_per_level[level]
            combined = np.concatenate([unpooled, skip], axis=1)

            # Decode
            current = decoders[level](combined)

        return current


class EdgePooling:
    """
    Edge pooling based on edge contraction.

    Learn which edges to contract based on features.
    More topology-aware than node clustering.
    """

    def __init__(self, score_func=None):
        self.score_func = score_func

    def pool(self, nodes: np.ndarray, edges: np.ndarray,
             features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PoolingInfo]:
        """
        Pool by contracting top-scored edges.
        """
        n_nodes = len(nodes)
        n_edges = edges.shape[1]

        # Score edges
        if self.score_func is None:
            # Default: prefer short edges
            edge_lengths = np.linalg.norm(
                nodes[edges[1]] - nodes[edges[0]], axis=1
            )
            scores = 1.0 / (edge_lengths + 1e-8)
        else:
            scores = self.score_func(features, edges)

        # Sort by score
        order = np.argsort(-scores)

        # Greedily contract edges
        contracted = set()
        merges = {}  # node -> representative

        for idx in order:
            i, j = edges[:, idx]

            # Find representatives
            while i in merges:
                i = merges[i]
            while j in merges:
                j = merges[j]

            if i == j:
                continue
            if i in contracted or j in contracted:
                continue

            # Contract j into i
            merges[j] = i
            contracted.add(j)

        # Build cluster assignment
        def find_rep(x):
            while x in merges:
                x = merges[x]
            return x

        reps = set()
        for i in range(n_nodes):
            reps.add(find_rep(i))

        rep_to_cluster = {rep: idx for idx, rep in enumerate(sorted(reps))}
        n_clusters = len(reps)

        labels = np.array([rep_to_cluster[find_rep(i)] for i in range(n_nodes)])

        # Build pooling matrix
        pooling_matrix = sparse.csr_matrix(
            (np.ones(n_nodes), (labels, np.arange(n_nodes))),
            shape=(n_clusters, n_nodes)
        )

        row_sums = np.array(pooling_matrix.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (row_sums + 1e-8))
        pooling_matrix = D_inv @ pooling_matrix

        coarse_nodes = pooling_matrix @ nodes
        coarse_features = pooling_matrix @ features

        # Compute coarse edges
        coarse_edges_set = set()
        for src, tgt in edges.T:
            c_src = labels[src]
            c_tgt = labels[tgt]
            if c_src != c_tgt:
                coarse_edges_set.add((c_src, c_tgt))
                coarse_edges_set.add((c_tgt, c_src))

        coarse_edges = np.array(list(coarse_edges_set)).T if coarse_edges_set else np.zeros((2, 0), dtype=int)

        pooling_info = PoolingInfo(
            cluster_assignment=labels,
            n_clusters=n_clusters,
            pooling_matrix=pooling_matrix
        )

        return coarse_nodes, coarse_edges, coarse_features, pooling_info
