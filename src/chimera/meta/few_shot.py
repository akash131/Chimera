"""
Few-Shot Learning for PDE Solvers.

Adapt solvers with minimal data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class FewShotSolver:
    """
    Few-shot adaptation of pre-trained PDE solver.
    """

    def __init__(self, base_solver: Callable,
                 adaptation_method: str = "fine_tune",
                 n_adaptation_steps: int = 10):
        """
        Args:
            base_solver: Pre-trained solver function
            adaptation_method: "fine_tune", "last_layer", or "feature_reuse"
            n_adaptation_steps: Steps for adaptation
        """
        self.base_solver = base_solver
        self.adaptation_method = adaptation_method
        self.n_adaptation_steps = n_adaptation_steps

        self._adapted_params: Optional[Dict] = None

    def adapt(self, support_points: np.ndarray,
              support_values: np.ndarray,
              lr: float = 0.01) -> None:
        """
        Adapt to new task using support set.

        Args:
            support_points: Few example input points
            support_values: Corresponding solution values
            lr: Learning rate for adaptation
        """
        # Placeholder: would adapt neural network parameters
        self._support_points = support_points
        self._support_values = support_values

    def predict(self, query_points: np.ndarray) -> np.ndarray:
        """Predict at query points."""
        # Combine base solver with local correction
        base_pred = self.base_solver(query_points)

        # Local correction using support set (RBF interpolation)
        correction = self._local_correction(query_points)

        return base_pred + correction

    def _local_correction(self, query_points: np.ndarray) -> np.ndarray:
        """Compute local correction from support set."""
        if self._support_points is None:
            return np.zeros(len(query_points))

        # Compute residuals at support points
        base_at_support = self.base_solver(self._support_points)
        residuals = self._support_values - base_at_support.flatten()

        # RBF interpolation of residuals
        epsilon = 1.0
        correction = np.zeros(len(query_points))

        for i, q in enumerate(query_points):
            weights = np.exp(-epsilon * np.sum((self._support_points - q)**2, axis=1))
            weights /= np.sum(weights) + 1e-10
            correction[i] = np.sum(weights * residuals)

        return correction


class ParameterTransfer:
    """
    Transfer learning between related PDE problems.

    Transfer parameters from source to target problem.
    """

    def __init__(self, transfer_layers: str = "all"):
        """
        Args:
            transfer_layers: "all", "encoder", or "decoder"
        """
        self.transfer_layers = transfer_layers

        self._source_params: Dict = {}

    def learn_source(self, source_solver: 'MetaSolver',
                     source_data: Dict):
        """Learn from source problem."""
        self._source_params = {
            'layers': source_solver.get_params(),
            'data_stats': {
                'mean': np.mean(source_data['values']),
                'std': np.std(source_data['values'])
            }
        }

    def transfer_to_target(self, target_solver: 'MetaSolver',
                           target_data: Dict,
                           freeze_layers: int = 2) -> 'MetaSolver':
        """
        Transfer to target problem.

        Args:
            target_solver: Target solver to initialize
            target_data: Target problem data
            freeze_layers: Number of layers to freeze
        """
        if not self._source_params:
            return target_solver

        source_layers = self._source_params['layers']

        # Copy source parameters
        for i, (W_src, b_src) in enumerate(source_layers):
            if i < len(target_solver._layers):
                target_solver._layers[i][0] = W_src.copy()
                target_solver._layers[i][1] = b_src.copy()

        # Fine-tune unfrozen layers
        lr = 0.01
        for epoch in range(100):
            pred = target_solver.forward(target_data['points'])
            error = pred.flatten() - target_data['values']
            loss = np.mean(error**2)

            # Only update unfrozen layers
            for i in range(freeze_layers, len(target_solver._layers)):
                W, b = target_solver._layers[i]
                W -= lr * 0.01 * np.random.randn(*W.shape) * loss

        return target_solver


class TaskEmbedding:
    """
    Learn embeddings of PDE tasks for similarity-based solving.
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of task embeddings
        """
        self.embedding_dim = embedding_dim

        self._encoder: List = []
        self._task_database: List[Tuple[np.ndarray, Callable]] = []

    def _initialize_encoder(self, input_dim: int):
        """Initialize embedding encoder."""
        dims = [input_dim, 128, 64, self.embedding_dim]

        self._encoder = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._encoder.append((W, b))

    def encode_task(self, task_description: Dict) -> np.ndarray:
        """
        Encode task to embedding vector.

        Args:
            task_description: Dictionary describing the PDE task

        Returns:
            Embedding vector
        """
        # Create feature vector from task description
        features = []

        # PDE parameters
        if 'parameters' in task_description:
            for val in task_description['parameters'].values():
                features.append(float(val))

        # Domain information
        if 'domain' in task_description:
            features.extend(task_description['domain'])

        # Data statistics
        if 'data' in task_description:
            data = task_description['data']
            features.append(np.mean(data))
            features.append(np.std(data))
            features.append(np.min(data))
            features.append(np.max(data))

        features = np.array(features)

        # Pad to fixed size
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]

        return features

    def add_task(self, task_description: Dict, solver: Callable):
        """Add solved task to database."""
        embedding = self.encode_task(task_description)
        self._task_database.append((embedding, solver))

    def find_similar(self, task_description: Dict, k: int = 5) -> List[Tuple[float, Callable]]:
        """
        Find k most similar tasks.

        Returns:
            List of (similarity, solver) pairs
        """
        embedding = self.encode_task(task_description)

        similarities = []
        for stored_emb, solver in self._task_database:
            sim = 1.0 / (1.0 + np.linalg.norm(embedding - stored_emb))
            similarities.append((sim, solver))

        similarities.sort(key=lambda x: -x[0])
        return similarities[:k]

    def ensemble_solve(self, task_description: Dict,
                       query_points: np.ndarray) -> np.ndarray:
        """
        Solve using ensemble of similar tasks.
        """
        similar = self.find_similar(task_description, k=5)

        if not similar:
            raise ValueError("No similar tasks in database")

        # Weighted ensemble
        predictions = []
        weights = []

        for sim, solver in similar:
            pred = solver(query_points)
            predictions.append(pred)
            weights.append(sim)

        weights = np.array(weights)
        weights /= np.sum(weights)

        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        return ensemble_pred


class MultiTaskLearning:
    """
    Multi-task learning for related PDE problems.

    Share representations across related PDEs.
    """

    def __init__(self, shared_dims: List[int] = None,
                 task_dims: List[int] = None):
        """
        Args:
            shared_dims: Shared encoder dimensions
            task_dims: Task-specific head dimensions
        """
        self.shared_dims = shared_dims or [64, 64]
        self.task_dims = task_dims or [32]

        self._shared_layers: List = []
        self._task_heads: Dict[str, List] = {}

    def _initialize(self, input_dim: int, n_tasks: int):
        """Initialize shared encoder and task heads."""
        # Shared encoder
        dims = [input_dim] + self.shared_dims
        self._shared_layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._shared_layers.append((W, b))

        # Task-specific heads
        for task_id in range(n_tasks):
            head_dims = [self.shared_dims[-1]] + self.task_dims + [1]
            head = []
            for i in range(len(head_dims) - 1):
                W = np.random.randn(head_dims[i], head_dims[i+1]) * np.sqrt(2.0 / head_dims[i])
                b = np.zeros(head_dims[i+1])
                head.append((W, b))
            self._task_heads[f"task_{task_id}"] = head

    def forward(self, x: np.ndarray, task_id: str) -> np.ndarray:
        """Forward pass for specific task."""
        # Shared encoder
        h = x
        for W, b in self._shared_layers:
            h = np.tanh(h @ W + b)

        # Task-specific head
        if task_id not in self._task_heads:
            raise ValueError(f"Unknown task: {task_id}")

        for i, (W, b) in enumerate(self._task_heads[task_id]):
            h = h @ W + b
            if i < len(self._task_heads[task_id]) - 1:
                h = np.tanh(h)

        return h

    def train(self, task_data: Dict[str, Dict],
              n_epochs: int = 1000, lr: float = 0.01):
        """
        Train on multiple tasks simultaneously.

        Args:
            task_data: {task_id: {'points': array, 'values': array}}
        """
        n_tasks = len(task_data)
        input_dim = list(task_data.values())[0]['points'].shape[1]

        self._initialize(input_dim, n_tasks)

        # Rename task IDs to standard format
        task_mapping = {old: f"task_{i}" for i, old in enumerate(task_data.keys())}

        for epoch in range(n_epochs):
            total_loss = 0

            for old_id, data in task_data.items():
                task_id = task_mapping[old_id]

                pred = self.forward(data['points'], task_id)
                error = pred.flatten() - data['values']
                loss = np.mean(error**2)
                total_loss += loss

                # Update (simplified)
                for W, b in self._shared_layers:
                    W -= lr * 0.01 * np.random.randn(*W.shape) * loss

                for W, b in self._task_heads[task_id]:
                    W -= lr * 0.01 * np.random.randn(*W.shape) * loss

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Total loss = {total_loss / n_tasks:.6f}")
