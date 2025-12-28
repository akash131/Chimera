"""
Meta-Learning Algorithms for PDE Solvers.

Learn to quickly adapt to new PDE instances.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from copy import deepcopy


@dataclass
class Task:
    """A PDE task for meta-learning."""
    parameters: Dict[str, float]  # PDE parameters
    train_points: np.ndarray  # (n_train, dim+1) - spatial/temporal coords
    train_values: np.ndarray  # (n_train,) - solution values
    test_points: np.ndarray
    test_values: np.ndarray
    pde_residual: Optional[Callable] = None  # Physics residual function


class MetaSolver:
    """
    Base class for meta-learning PDE solvers.
    """

    def __init__(self, hidden_dims: List[int] = None,
                 activation: str = "tanh"):
        """
        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        self.hidden_dims = hidden_dims or [64, 64, 64]
        self.activation = activation

        self._layers: List = []

    def _initialize(self, input_dim: int, output_dim: int = 1):
        """Initialize network."""
        dims = [input_dim] + self.hidden_dims + [output_dim]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append([W, b])  # Mutable list

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = x
        for i, (W, b) in enumerate(self._layers):
            h = h @ W + b
            if i < len(self._layers) - 1:
                if self.activation == "tanh":
                    h = np.tanh(h)
                elif self.activation == "relu":
                    h = np.maximum(0, h)
        return h

    def get_params(self) -> List:
        """Get current parameters."""
        return deepcopy(self._layers)

    def set_params(self, params: List):
        """Set parameters."""
        self._layers = deepcopy(params)

    def compute_loss(self, task: Task, params: Optional[List] = None) -> float:
        """Compute loss on task."""
        if params is not None:
            old_params = self.get_params()
            self.set_params(params)

        pred = self.forward(task.train_points)
        loss = np.mean((pred.flatten() - task.train_values)**2)

        if params is not None:
            self.set_params(old_params)

        return loss


class MAML(MetaSolver):
    """
    Model-Agnostic Meta-Learning for PDEs.

    Learn initialization that adapts quickly to new tasks.
    """

    def __init__(self, inner_lr: float = 0.01,
                 inner_steps: int = 5,
                 meta_lr: float = 0.001,
                 **kwargs):
        """
        Args:
            inner_lr: Learning rate for task adaptation
            inner_steps: Gradient steps for adaptation
            meta_lr: Meta-learning rate
        """
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr

    def train(self, tasks: List[Task], n_epochs: int = 1000):
        """
        Train meta-learner on task distribution.
        """
        if not tasks:
            return

        input_dim = tasks[0].train_points.shape[1]
        self._initialize(input_dim)

        for epoch in range(n_epochs):
            # Sample batch of tasks
            batch_size = min(4, len(tasks))
            batch_tasks = np.random.choice(tasks, size=batch_size, replace=False)

            meta_grads = [np.zeros_like(W) for W, b in self._layers] + \
                         [np.zeros_like(b) for W, b in self._layers]

            total_loss = 0

            for task in batch_tasks:
                # Inner loop: adapt to task
                adapted_params = self._adapt(task)

                # Compute loss on test set with adapted params
                self.set_params(adapted_params)
                test_pred = self.forward(task.test_points)
                test_loss = np.mean((test_pred.flatten() - task.test_values)**2)
                total_loss += test_loss

                # Compute meta-gradient (simplified)
                # Would need second-order gradients for full MAML
                for i, ((W, b), (W_adapted, b_adapted)) in enumerate(zip(self._layers, adapted_params)):
                    meta_grads[i] += (W_adapted - W) / len(batch_tasks)
                    meta_grads[len(self._layers) + i] += (b_adapted - b) / len(batch_tasks)

            # Meta update
            for i, (W, b) in enumerate(self._layers):
                W -= self.meta_lr * meta_grads[i]
                b -= self.meta_lr * meta_grads[len(self._layers) + i]

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Meta-loss = {total_loss / len(batch_tasks):.6f}")

    def _adapt(self, task: Task) -> List:
        """Adapt to single task."""
        params = self.get_params()

        for _ in range(self.inner_steps):
            # Compute gradient
            grads = self._compute_gradients(task, params)

            # Update
            for i, (W, b) in enumerate(params):
                W -= self.inner_lr * grads[i][0]
                b -= self.inner_lr * grads[i][1]

        return params

    def _compute_gradients(self, task: Task, params: List) -> List:
        """Compute gradients numerically."""
        eps = 1e-5
        grads = []

        base_loss = self.compute_loss(task, params)

        for i, (W, b) in enumerate(params):
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)

            # Subsample for efficiency
            n_samples = min(10, W.size)
            indices = np.random.choice(W.size, size=n_samples, replace=False)

            for idx in indices:
                flat_idx = np.unravel_index(idx, W.shape)
                W[flat_idx] += eps
                loss_plus = self.compute_loss(task, params)
                W[flat_idx] -= 2 * eps
                loss_minus = self.compute_loss(task, params)
                W[flat_idx] += eps
                grad_W[flat_idx] = (loss_plus - loss_minus) / (2 * eps)

            grads.append((grad_W, grad_b))

        return grads

    def solve(self, task: Task, n_adapt_steps: int = 10) -> np.ndarray:
        """Solve new task with few-shot adaptation."""
        # Adapt to task
        old_params = self.get_params()

        for _ in range(n_adapt_steps):
            grads = self._compute_gradients(task, self._layers)
            for i, (W, b) in enumerate(self._layers):
                W -= self.inner_lr * grads[i][0]
                b -= self.inner_lr * grads[i][1]

        # Predict
        pred = self.forward(task.test_points)

        # Restore
        self.set_params(old_params)

        return pred


class Reptile(MetaSolver):
    """
    Reptile meta-learning algorithm.

    Simpler than MAML, doesn't require second-order gradients.
    """

    def __init__(self, inner_lr: float = 0.01,
                 inner_steps: int = 10,
                 meta_lr: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr

    def train(self, tasks: List[Task], n_epochs: int = 1000):
        """Train using Reptile."""
        if not tasks:
            return

        input_dim = tasks[0].train_points.shape[1]
        self._initialize(input_dim)

        for epoch in range(n_epochs):
            # Sample task
            task = np.random.choice(tasks)

            # Store initial params
            init_params = self.get_params()

            # Train on task
            for _ in range(self.inner_steps):
                grads = self._compute_gradients(task, self._layers)
                for i, (W, b) in enumerate(self._layers):
                    W -= self.inner_lr * grads[i][0]
                    b -= self.inner_lr * grads[i][1]

            # Reptile update: move toward adapted params
            for i, ((W_init, b_init), (W, b)) in enumerate(zip(init_params, self._layers)):
                self._layers[i][0] = W_init + self.meta_lr * (W - W_init)
                self._layers[i][1] = b_init + self.meta_lr * (b - b_init)

            if epoch % 100 == 0:
                loss = self.compute_loss(task)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def _compute_gradients(self, task: Task, params: List) -> List:
        """Compute gradients."""
        eps = 1e-5
        grads = []

        for i, (W, b) in enumerate(params):
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)

            # Numerical gradient (simplified)
            base_loss = self.compute_loss(task, params)

            for j in range(min(5, W.shape[0])):
                for k in range(min(5, W.shape[1])):
                    W[j, k] += eps
                    loss_plus = self.compute_loss(task, params)
                    W[j, k] -= eps
                    grad_W[j, k] = (loss_plus - base_loss) / eps

            grads.append((grad_W, grad_b))

        return grads


class ProtoNet(MetaSolver):
    """
    Prototypical Networks for PDE tasks.

    Learn task embeddings and solve by similarity to prototypes.
    """

    def __init__(self, embedding_dim: int = 32, **kwargs):
        """
        Args:
            embedding_dim: Dimension of task embeddings
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

        self._prototypes: Dict[str, np.ndarray] = {}
        self._prototype_solutions: Dict[str, Callable] = {}

    def _compute_embedding(self, task: Task) -> np.ndarray:
        """Compute task embedding from support set."""
        # Use features of the data as embedding
        features = []

        # Statistics of data
        features.append(np.mean(task.train_values))
        features.append(np.std(task.train_values))
        features.append(np.mean(task.train_points, axis=0))
        features.append(np.std(task.train_points, axis=0))

        # PDE parameters if available
        if task.parameters:
            for val in task.parameters.values():
                features.append(val)

        embedding = np.concatenate([np.atleast_1d(f).flatten() for f in features])

        # Pad or truncate to fixed size
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]

        return embedding

    def train(self, tasks: List[Task], n_epochs: int = 100):
        """Build prototypes from training tasks."""
        if not tasks:
            return

        # Cluster tasks by similarity
        embeddings = [self._compute_embedding(task) for task in tasks]

        # Simple: use each task as its own prototype
        for i, (task, emb) in enumerate(zip(tasks, embeddings)):
            key = f"proto_{i}"
            self._prototypes[key] = emb

            # Store solver for this prototype
            self._prototype_solutions[key] = self._fit_solver(task)

    def _fit_solver(self, task: Task) -> Callable:
        """Fit solver for single task."""
        input_dim = task.train_points.shape[1]
        self._initialize(input_dim)

        # Simple training
        lr = 0.01
        for _ in range(100):
            pred = self.forward(task.train_points)
            error = pred.flatten() - task.train_values

            # Update (simplified)
            for W, b in self._layers:
                W -= lr * 0.01 * np.random.randn(*W.shape) * np.mean(error**2)

        params = self.get_params()

        def solver(x):
            self.set_params(params)
            return self.forward(x)

        return solver

    def solve(self, task: Task) -> np.ndarray:
        """Solve new task using nearest prototype."""
        embedding = self._compute_embedding(task)

        # Find nearest prototype
        min_dist = np.inf
        nearest_key = None

        for key, proto in self._prototypes.items():
            dist = np.linalg.norm(embedding - proto)
            if dist < min_dist:
                min_dist = dist
                nearest_key = key

        if nearest_key is None:
            raise ValueError("No prototypes available")

        # Use prototype's solver
        solver = self._prototype_solutions[nearest_key]
        return solver(task.test_points)
