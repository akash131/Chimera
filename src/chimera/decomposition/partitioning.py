"""
Domain partitioning algorithms.

Divide mesh into subdomains for parallel computation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse


@dataclass
class PartitionResult:
    """Result of domain partitioning."""
    partition: List[np.ndarray]  # List of node indices per subdomain
    n_subdomains: int
    edge_cut: int  # Number of cut edges
    balance: float  # Load balance ratio (max/average size)


class DomainPartitioner:
    """
    Base class for domain partitioners.
    """

    def partition(self, n_parts: int,
                  nodes: np.ndarray,
                  adjacency: sparse.csr_matrix) -> PartitionResult:
        """
        Partition domain.

        Args:
            n_parts: Number of partitions
            nodes: Node coordinates (n_nodes, dim)
            adjacency: Adjacency matrix

        Returns:
            PartitionResult
        """
        raise NotImplementedError


class GraphPartitioner(DomainPartitioner):
    """
    Graph-based partitioning.

    Uses recursive bisection or multilevel methods.
    """

    def __init__(self, method: str = "recursive_bisection"):
        """
        Args:
            method: "recursive_bisection", "spectral", or "greedy"
        """
        self.method = method

    def partition(self, n_parts: int,
                  nodes: np.ndarray,
                  adjacency: sparse.csr_matrix) -> PartitionResult:
        """Partition using graph methods."""
        n_nodes = nodes.shape[0]

        if self.method == "recursive_bisection":
            labels = self._recursive_bisection(adjacency, n_parts)
        elif self.method == "spectral":
            labels = self._spectral_partition(adjacency, n_parts)
        elif self.method == "greedy":
            labels = self._greedy_partition(adjacency, n_parts)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert labels to partition
        partition = [np.where(labels == i)[0] for i in range(n_parts)]

        # Compute metrics
        edge_cut = self._compute_edge_cut(adjacency, labels)
        sizes = [len(p) for p in partition]
        balance = max(sizes) / (sum(sizes) / len(sizes)) if sizes else 1.0

        return PartitionResult(
            partition=partition,
            n_subdomains=n_parts,
            edge_cut=edge_cut,
            balance=balance
        )

    def _recursive_bisection(self, adj: sparse.csr_matrix,
                              n_parts: int) -> np.ndarray:
        """Recursive graph bisection."""
        n_nodes = adj.shape[0]
        labels = np.zeros(n_nodes, dtype=int)

        if n_parts <= 1:
            return labels

        # Initial bisection
        labels = self._spectral_bisection(adj)

        if n_parts <= 2:
            return labels

        # Recursively partition each half
        for part in [0, 1]:
            mask = labels == part
            indices = np.where(mask)[0]

            if len(indices) > 1:
                # Extract subgraph
                sub_adj = adj[np.ix_(indices, indices)]

                # Recursive call
                sub_parts = n_parts // 2 if part == 0 else n_parts - n_parts // 2
                sub_labels = self._recursive_bisection(sub_adj, sub_parts)

                # Map back
                offset = 0 if part == 0 else n_parts // 2
                labels[indices] = sub_labels + offset

        return labels

    def _spectral_bisection(self, adj: sparse.csr_matrix) -> np.ndarray:
        """Bisect using Fiedler vector."""
        n = adj.shape[0]

        if n <= 2:
            return np.arange(n)

        # Graph Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()
        D = sparse.diags(degree)
        L = D - adj

        # Fiedler vector (second smallest eigenvector)
        from scipy.sparse.linalg import eigsh

        try:
            _, eigvecs = eigsh(L, k=2, which='SM', sigma=0)
            fiedler = eigvecs[:, 1]
        except:
            # Fallback to random
            fiedler = np.random.randn(n)

        # Partition by sign
        labels = (fiedler > np.median(fiedler)).astype(int)

        return labels

    def _spectral_partition(self, adj: sparse.csr_matrix,
                            n_parts: int) -> np.ndarray:
        """Partition using multiple eigenvectors."""
        n = adj.shape[0]

        # Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()
        D = sparse.diags(degree)
        L = D - adj

        # Compute embedding
        from scipy.sparse.linalg import eigsh

        try:
            k = min(n_parts, n - 1)
            _, eigvecs = eigsh(L, k=k+1, which='SM', sigma=0)
            embedding = eigvecs[:, 1:k+1]  # Skip constant eigenvector
        except:
            embedding = np.random.randn(n, n_parts)

        # K-means clustering
        labels = self._kmeans(embedding, n_parts)

        return labels

    def _greedy_partition(self, adj: sparse.csr_matrix,
                          n_parts: int) -> np.ndarray:
        """Greedy graph growing partition."""
        n = adj.shape[0]
        labels = np.full(n, -1)
        target_size = n // n_parts

        for part in range(n_parts):
            if np.all(labels >= 0):
                break

            # Find seed (unassigned node)
            unassigned = np.where(labels < 0)[0]
            if len(unassigned) == 0:
                break

            seed = unassigned[0]
            frontier = [seed]
            labels[seed] = part
            count = 1

            # Grow region
            while frontier and count < target_size:
                node = frontier.pop(0)

                # Find unassigned neighbors
                neighbors = adj.getrow(node).indices
                for nbr in neighbors:
                    if labels[nbr] < 0:
                        labels[nbr] = part
                        frontier.append(nbr)
                        count += 1
                        if count >= target_size:
                            break

        # Assign remaining nodes
        remaining = labels < 0
        if np.any(remaining):
            labels[remaining] = n_parts - 1

        return labels

    def _kmeans(self, X: np.ndarray, n_clusters: int,
                max_iter: int = 100) -> np.ndarray:
        """Simple k-means clustering."""
        n = X.shape[0]

        # Initialize centroids
        indices = np.random.choice(n, n_clusters, replace=False)
        centroids = X[indices].copy()

        for _ in range(max_iter):
            # Assign to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = X[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels

    def _compute_edge_cut(self, adj: sparse.csr_matrix,
                          labels: np.ndarray) -> int:
        """Count edges between partitions."""
        adj_coo = adj.tocoo()
        cut = 0
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i < j and labels[i] != labels[j]:
                cut += 1
        return cut


class GeometricPartitioner(DomainPartitioner):
    """
    Geometry-based partitioning.

    Uses spatial coordinates directly.
    """

    def __init__(self, method: str = "kdtree"):
        """
        Args:
            method: "kdtree", "rcp" (recursive coordinate bisection), or "sfc"
        """
        self.method = method

    def partition(self, n_parts: int,
                  nodes: np.ndarray,
                  adjacency: sparse.csr_matrix) -> PartitionResult:
        """Partition using geometry."""
        if self.method == "kdtree":
            labels = self._kdtree_partition(nodes, n_parts)
        elif self.method == "rcb":
            labels = self._recursive_coordinate_bisection(nodes, n_parts)
        elif self.method == "sfc":
            labels = self._space_filling_curve(nodes, n_parts)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        partition = [np.where(labels == i)[0] for i in range(n_parts)]

        edge_cut = self._compute_edge_cut(adjacency, labels)
        sizes = [len(p) for p in partition]
        balance = max(sizes) / (sum(sizes) / len(sizes)) if sizes else 1.0

        return PartitionResult(
            partition=partition,
            n_subdomains=n_parts,
            edge_cut=edge_cut,
            balance=balance
        )

    def _kdtree_partition(self, nodes: np.ndarray, n_parts: int) -> np.ndarray:
        """Partition using k-d tree."""
        n = len(nodes)
        labels = np.zeros(n, dtype=int)
        indices = np.arange(n)

        self._kdtree_recurse(nodes, indices, labels, 0, n_parts, 0)

        return labels

    def _kdtree_recurse(self, nodes: np.ndarray, indices: np.ndarray,
                        labels: np.ndarray, part_offset: int,
                        n_parts: int, depth: int) -> None:
        """Recursive k-d tree partitioning."""
        if n_parts <= 1 or len(indices) == 0:
            labels[indices] = part_offset
            return

        # Split along dimension
        dim = depth % nodes.shape[1]
        coords = nodes[indices, dim]
        median = np.median(coords)

        left_mask = coords <= median
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        # Recurse
        left_parts = n_parts // 2
        right_parts = n_parts - left_parts

        self._kdtree_recurse(nodes, left_indices, labels, part_offset,
                             left_parts, depth + 1)
        self._kdtree_recurse(nodes, right_indices, labels, part_offset + left_parts,
                             right_parts, depth + 1)

    def _recursive_coordinate_bisection(self, nodes: np.ndarray,
                                         n_parts: int) -> np.ndarray:
        """Recursive coordinate bisection (RCB)."""
        n = len(nodes)
        labels = np.zeros(n, dtype=int)

        def rcb(indices, offset, parts, depth):
            if parts <= 1 or len(indices) <= 1:
                labels[indices] = offset
                return

            # Find longest dimension
            coords = nodes[indices]
            ranges = coords.max(axis=0) - coords.min(axis=0)
            dim = np.argmax(ranges)

            # Split at median
            values = coords[:, dim]
            median = np.median(values)

            left = indices[values <= median]
            right = indices[values > median]

            left_parts = parts // 2
            right_parts = parts - left_parts

            rcb(left, offset, left_parts, depth + 1)
            rcb(right, offset + left_parts, right_parts, depth + 1)

        rcb(np.arange(n), 0, n_parts, 0)
        return labels

    def _space_filling_curve(self, nodes: np.ndarray,
                              n_parts: int) -> np.ndarray:
        """Partition using space-filling curve (Hilbert/Morton)."""
        n = len(nodes)
        dim = nodes.shape[1]

        # Normalize to [0, 1]
        nodes_norm = nodes - nodes.min(axis=0)
        nodes_norm /= (nodes_norm.max(axis=0) + 1e-10)

        # Compute Morton code (Z-order curve)
        resolution = 1 << 16  # 16 bits per dimension
        indices = (nodes_norm * resolution).astype(np.int64)
        indices = np.clip(indices, 0, resolution - 1)

        # Interleave bits
        if dim == 2:
            codes = self._interleave_2d(indices[:, 0], indices[:, 1])
        elif dim == 3:
            codes = self._interleave_3d(indices[:, 0], indices[:, 1], indices[:, 2])
        else:
            # Fallback: simple ordering
            codes = np.sum(indices * (resolution ** np.arange(dim)), axis=1)

        # Sort by code
        order = np.argsort(codes)

        # Partition into equal parts
        labels = np.zeros(n, dtype=int)
        part_size = n // n_parts

        for i, idx in enumerate(order):
            labels[idx] = min(i // part_size, n_parts - 1)

        return labels

    def _interleave_2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interleave bits of x and y for Morton code."""
        def spread(v):
            v = v & 0xFFFF
            v = (v | (v << 8)) & 0x00FF00FF
            v = (v | (v << 4)) & 0x0F0F0F0F
            v = (v | (v << 2)) & 0x33333333
            v = (v | (v << 1)) & 0x55555555
            return v

        return spread(x) | (spread(y) << 1)

    def _interleave_3d(self, x: np.ndarray, y: np.ndarray,
                       z: np.ndarray) -> np.ndarray:
        """Interleave bits for 3D Morton code."""
        def spread(v):
            v = v & 0x1FFFFF
            v = (v | (v << 32)) & 0x1F00000000FFFF
            v = (v | (v << 16)) & 0x1F0000FF0000FF
            v = (v | (v << 8)) & 0x100F00F00F00F00F
            v = (v | (v << 4)) & 0x10C30C30C30C30C3
            v = (v | (v << 2)) & 0x1249249249249249
            return v

        return spread(x) | (spread(y) << 1) | (spread(z) << 2)

    def _compute_edge_cut(self, adj: sparse.csr_matrix,
                          labels: np.ndarray) -> int:
        """Count edges between partitions."""
        adj_coo = adj.tocoo()
        cut = 0
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i < j and labels[i] != labels[j]:
                cut += 1
        return cut


class MetisPartitioner(DomainPartitioner):
    """
    METIS-style multilevel partitioning.

    Simplified version of METIS algorithm.
    """

    def __init__(self, matching: str = "heavy_edge"):
        """
        Args:
            matching: Coarsening method ("heavy_edge" or "random")
        """
        self.matching = matching

    def partition(self, n_parts: int,
                  nodes: np.ndarray,
                  adjacency: sparse.csr_matrix) -> PartitionResult:
        """METIS-style partition."""
        # 1. Coarsening phase
        coarse_levels = self._coarsen(adjacency)

        # 2. Initial partition on coarsest
        coarsest_adj = coarse_levels[-1]["adjacency"]
        labels = GraphPartitioner("spectral").partition(
            n_parts, nodes[:coarsest_adj.shape[0]], coarsest_adj
        ).partition

        labels_flat = np.zeros(coarsest_adj.shape[0], dtype=int)
        for i, part in enumerate(labels):
            labels_flat[part] = i

        # 3. Uncoarsening with refinement
        for level in reversed(coarse_levels[:-1]):
            labels_flat = self._uncoarsen(labels_flat, level)
            labels_flat = self._refine(labels_flat, level["adjacency"], n_parts)

        partition = [np.where(labels_flat == i)[0] for i in range(n_parts)]

        edge_cut = self._compute_edge_cut(adjacency, labels_flat)
        sizes = [len(p) for p in partition]
        balance = max(sizes) / (sum(sizes) / len(sizes)) if sizes else 1.0

        return PartitionResult(
            partition=partition,
            n_subdomains=n_parts,
            edge_cut=edge_cut,
            balance=balance
        )

    def _coarsen(self, adj: sparse.csr_matrix,
                 min_size: int = 100) -> List[Dict]:
        """Coarsen graph through matching."""
        levels = [{"adjacency": adj, "mapping": np.arange(adj.shape[0])}]

        current = adj
        while current.shape[0] > min_size:
            # Heavy edge matching
            matching = self._heavy_edge_matching(current)

            # Contract graph
            n_coarse = len(set(matching))
            mapping = np.zeros(current.shape[0], dtype=int)

            cluster_id = 0
            visited = set()
            for i, j in enumerate(matching):
                if i not in visited:
                    mapping[i] = cluster_id
                    if j != i:
                        mapping[j] = cluster_id
                        visited.add(j)
                    visited.add(i)
                    cluster_id += 1

            # Build coarse adjacency
            coarse_adj = self._contract_graph(current, mapping)

            levels.append({
                "adjacency": coarse_adj,
                "mapping": mapping
            })

            if coarse_adj.shape[0] >= current.shape[0] * 0.9:
                break  # Not enough coarsening

            current = coarse_adj

        return levels

    def _heavy_edge_matching(self, adj: sparse.csr_matrix) -> np.ndarray:
        """Find heavy edge matching."""
        n = adj.shape[0]
        matching = np.arange(n)
        matched = np.zeros(n, dtype=bool)

        # Random order
        order = np.random.permutation(n)

        for i in order:
            if matched[i]:
                continue

            # Find heaviest unmatched neighbor
            row = adj.getrow(i)
            best_j = i
            best_weight = 0

            for j, w in zip(row.indices, row.data):
                if not matched[j] and w > best_weight:
                    best_weight = w
                    best_j = j

            matching[i] = best_j
            matching[best_j] = i
            matched[i] = True
            matched[best_j] = True

        return matching

    def _contract_graph(self, adj: sparse.csr_matrix,
                        mapping: np.ndarray) -> sparse.csr_matrix:
        """Contract graph according to mapping."""
        n_coarse = len(np.unique(mapping))

        rows = []
        cols = []
        data = []

        adj_coo = adj.tocoo()
        for i, j, w in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            ci = mapping[i]
            cj = mapping[j]
            if ci != cj:
                rows.append(ci)
                cols.append(cj)
                data.append(w)

        coarse_adj = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_coarse, n_coarse)
        )

        return coarse_adj

    def _uncoarsen(self, labels: np.ndarray, level: Dict) -> np.ndarray:
        """Project partition to finer level."""
        mapping = level["mapping"]
        n_fine = len(mapping)

        fine_labels = np.zeros(n_fine, dtype=int)
        for i, c in enumerate(mapping):
            fine_labels[i] = labels[c]

        return fine_labels

    def _refine(self, labels: np.ndarray, adj: sparse.csr_matrix,
                n_parts: int, max_iter: int = 10) -> np.ndarray:
        """Kernighan-Lin style refinement."""
        n = adj.shape[0]

        for _ in range(max_iter):
            improved = False

            for i in range(n):
                current_part = labels[i]

                # Compute gain for moving to each part
                gains = np.zeros(n_parts)
                row = adj.getrow(i)

                for j, w in zip(row.indices, row.data):
                    part_j = labels[j]
                    if part_j == current_part:
                        gains -= w  # Would lose internal edge
                        gains[current_part] += w
                    else:
                        gains[part_j] += w  # Would gain edge

                # Find best move (considering balance)
                sizes = np.bincount(labels, minlength=n_parts)
                max_size = n / n_parts * 1.1  # 10% imbalance allowed

                best_gain = 0
                best_part = current_part

                for p in range(n_parts):
                    if p != current_part and gains[p] > best_gain:
                        if sizes[p] < max_size:
                            best_gain = gains[p]
                            best_part = p

                if best_part != current_part:
                    labels[i] = best_part
                    improved = True

            if not improved:
                break

        return labels

    def _compute_edge_cut(self, adj: sparse.csr_matrix,
                          labels: np.ndarray) -> int:
        """Count edges between partitions."""
        adj_coo = adj.tocoo()
        cut = 0
        for i, j in zip(adj_coo.row, adj_coo.col):
            if i < j and labels[i] != labels[j]:
                cut += 1
        return cut
