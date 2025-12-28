"""
Neural operator implementations for Chimera.

These learn mappings between function spaces - genuinely novel
approach to accelerating simulations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np

from chimera.core.field import Field
from chimera.mesh.mesh import Mesh


@dataclass
class OperatorConfig:
    """Configuration for neural operators."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "gelu"
    n_modes: int = 12  # For Fourier operators
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100


class NeuralOperator(ABC):
    """
    Base class for neural operators.

    Neural operators learn mappings G: A -> U where A and U are
    function spaces. Unlike standard neural networks that learn
    point-to-point mappings, operators learn function-to-function
    mappings that generalize across discretizations.

    Key innovation: resolution-invariant learned surrogates.
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        self.config = config or OperatorConfig()
        self._model = None
        self._trained = False

    @abstractmethod
    def build(self, input_dim: int, output_dim: int):
        """Build the neural network architecture."""
        pass

    @abstractmethod
    def forward(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate operator at query points.

        Args:
            a: Input function values at input points
            x: Query points where to evaluate output

        Returns:
            Output function values at query points
        """
        pass

    def train(self, inputs: List[Field], outputs: List[Field],
              validation_split: float = 0.1):
        """
        Train the operator on input-output function pairs.

        Args:
            inputs: List of input fields
            outputs: List of corresponding output fields
            validation_split: Fraction for validation
        """
        # Convert to training data
        X_train, y_train = self._prepare_training_data(inputs, outputs)

        # Training loop (simplified - would use PyTorch)
        n_samples = len(X_train)
        n_val = int(n_samples * validation_split)

        for epoch in range(self.config.epochs):
            # Shuffle
            idx = np.random.permutation(n_samples - n_val)

            # Mini-batch training
            for i in range(0, len(idx), self.config.batch_size):
                batch_idx = idx[i:i + self.config.batch_size]
                # Forward pass, loss, backward pass
                pass

        self._trained = True

    def _prepare_training_data(self, inputs: List[Field],
                               outputs: List[Field]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert fields to training arrays."""
        X = np.array([f.values for f in inputs])
        y = np.array([f.values for f in outputs])
        return X, y

    def predict(self, input_field: Field, query_mesh: Mesh) -> Field:
        """
        Predict output field from input field.

        Args:
            input_field: Input function
            query_mesh: Mesh where to evaluate output

        Returns:
            Predicted output field
        """
        if not self._trained:
            raise RuntimeError("Operator not trained")

        output_values = self.forward(input_field.values, query_mesh.points)

        return Field(
            name=f"predicted_{input_field.name}",
            field_type=input_field.field_type,
            dim=query_mesh.dim,
            values=output_values,
            points=query_mesh.points
        )

    def save(self, path: str):
        """Save trained model."""
        raise NotImplementedError()

    def load(self, path: str):
        """Load trained model."""
        raise NotImplementedError()


class FourierNeuralOperator(NeuralOperator):
    """
    Fourier Neural Operator (FNO).

    Key idea: Learn operator in Fourier space where convolutions
    become multiplications. Resolution-invariant by design.

    Architecture:
    1. Lift input to higher dimension
    2. Apply Fourier layers (spectral convolution + local linear)
    3. Project to output dimension

    Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2020)
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._fourier_weights = None
        self._linear_weights = None
        self._lift_weights = None
        self._project_weights = None

    def build(self, input_dim: int, output_dim: int):
        """Build FNO architecture."""
        n_modes = self.config.n_modes
        hidden = self.config.hidden_dims[0]

        # Initialize weights (simplified - would be PyTorch tensors)
        # Lifting layer: input_dim -> hidden
        self._lift_weights = np.random.randn(input_dim, hidden) * 0.02

        # Fourier layers
        n_layers = len(self.config.hidden_dims)
        self._fourier_weights = []
        self._linear_weights = []

        for i in range(n_layers):
            # Spectral weights (complex)
            w_fourier = (np.random.randn(n_modes, hidden, hidden) +
                         1j * np.random.randn(n_modes, hidden, hidden)) * 0.02
            self._fourier_weights.append(w_fourier)

            # Local linear weights
            w_linear = np.random.randn(hidden, hidden) * 0.02
            self._linear_weights.append(w_linear)

        # Projection layer: hidden -> output_dim
        self._project_weights = np.random.randn(hidden, output_dim) * 0.02

    def forward(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through FNO.

        Args:
            a: Input function values (n_points, input_dim)
            x: Query points (n_query, dim) - not used, FNO is grid-based

        Returns:
            Output function values (n_points, output_dim)
        """
        if self._lift_weights is None:
            raise RuntimeError("Model not built")

        # Lift
        h = a @ self._lift_weights
        h = self._activation(h)

        # Fourier layers
        for w_fourier, w_linear in zip(self._fourier_weights, self._linear_weights):
            # FFT
            h_fft = np.fft.rfft(h, axis=0)

            # Spectral convolution (multiply in frequency space)
            n_modes = w_fourier.shape[0]
            h_fft_out = np.zeros_like(h_fft)
            h_fft_out[:n_modes] = np.einsum('mij,mj->mi', w_fourier, h_fft[:n_modes])

            # Inverse FFT
            h_spectral = np.fft.irfft(h_fft_out, n=len(h), axis=0)

            # Local linear transform
            h_local = h @ w_linear

            # Combine and activate
            h = self._activation(h_spectral + h_local)

        # Project
        return h @ self._project_weights

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "gelu":
            return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.config.activation == "relu":
            return np.maximum(0, x)
        else:
            return x


class DeepONet(NeuralOperator):
    """
    Deep Operator Network (DeepONet).

    Architecture with two sub-networks:
    - Branch net: encodes input function
    - Trunk net: encodes query locations

    Output: sum of products of branch and trunk outputs

    u(y) = sum_k b_k(a) * t_k(y)

    Reference: Lu et al., "Learning nonlinear operators" (2021)
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._branch_weights = None
        self._trunk_weights = None
        self._n_basis = 64  # Number of basis functions

    def build(self, input_dim: int, output_dim: int, query_dim: int = 2):
        """Build DeepONet architecture."""
        hidden = self.config.hidden_dims

        # Branch network: input function -> basis coefficients
        self._branch_weights = []
        dims = [input_dim] + hidden + [self._n_basis]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            self._branch_weights.append(w)

        # Trunk network: query location -> basis functions
        self._trunk_weights = []
        dims = [query_dim] + hidden + [self._n_basis * output_dim]
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            self._trunk_weights.append(w)

        self._output_dim = output_dim

    def forward(self, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through DeepONet.

        Args:
            a: Input function values (flattened)
            x: Query points (n_query, dim)

        Returns:
            Output function values (n_query, output_dim)
        """
        # Branch network
        b = a.flatten()
        for i, w in enumerate(self._branch_weights):
            b = b @ w
            if i < len(self._branch_weights) - 1:
                b = self._activation(b)
        # b shape: (n_basis,)

        # Trunk network for each query point
        outputs = np.zeros((len(x), self._output_dim))

        for i, xi in enumerate(x):
            t = xi
            for j, w in enumerate(self._trunk_weights):
                t = t @ w
                if j < len(self._trunk_weights) - 1:
                    t = self._activation(t)
            # t shape: (n_basis * output_dim,)
            t = t.reshape(self._n_basis, self._output_dim)

            # Combine: sum over basis
            outputs[i] = b @ t

        return outputs

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        return np.tanh(x)


class MessagePassingOperator(NeuralOperator):
    """
    Graph Neural Operator using message passing.

    Works on arbitrary meshes without regular grid structure.
    Key innovation: mesh-independent operator learning.
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._edge_mlp_weights = None
        self._node_mlp_weights = None
        self._n_layers = 4

    def build(self, input_dim: int, output_dim: int):
        """Build message passing architecture."""
        hidden = self.config.hidden_dims[0]

        # Edge MLP weights
        self._edge_mlp_weights = []
        for _ in range(self._n_layers):
            w1 = np.random.randn(2 * hidden + 3, hidden) * 0.02  # node features + edge features
            w2 = np.random.randn(hidden, hidden) * 0.02
            self._edge_mlp_weights.append((w1, w2))

        # Node MLP weights
        self._node_mlp_weights = []
        for _ in range(self._n_layers):
            w1 = np.random.randn(2 * hidden, hidden) * 0.02
            w2 = np.random.randn(hidden, hidden) * 0.02
            self._node_mlp_weights.append((w1, w2))

        # Input/output projections
        self._input_proj = np.random.randn(input_dim, hidden) * 0.02
        self._output_proj = np.random.randn(hidden, output_dim) * 0.02

    def forward(self, a: np.ndarray, x: np.ndarray,
                edge_index: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with message passing.

        Args:
            a: Node features (n_nodes, input_dim)
            x: Node positions (n_nodes, dim)
            edge_index: Edge connectivity (2, n_edges)

        Returns:
            Output node features (n_nodes, output_dim)
        """
        if edge_index is None:
            # Build edges from k-nearest neighbors
            from scipy.spatial import KDTree
            tree = KDTree(x)
            k = min(10, len(x) - 1)
            _, neighbors = tree.query(x, k=k + 1)
            neighbors = neighbors[:, 1:]  # Exclude self

            # Build edge index
            sources = np.repeat(np.arange(len(x)), k)
            targets = neighbors.flatten()
            edge_index = np.stack([sources, targets])

        n_nodes = len(a)
        n_edges = edge_index.shape[1]

        # Project input
        h = a @ self._input_proj

        # Message passing layers
        for (edge_w1, edge_w2), (node_w1, node_w2) in zip(
                self._edge_mlp_weights, self._node_mlp_weights):

            # Compute edge features
            src, tgt = edge_index
            edge_features = np.concatenate([
                h[src],
                h[tgt],
                x[tgt] - x[src]  # Relative position
            ], axis=1)

            # Edge MLP
            edge_msg = self._activation(edge_features @ edge_w1)
            edge_msg = edge_msg @ edge_w2

            # Aggregate messages (sum)
            aggregated = np.zeros((n_nodes, edge_msg.shape[1]))
            np.add.at(aggregated, tgt, edge_msg)

            # Node update
            node_input = np.concatenate([h, aggregated], axis=1)
            h_new = self._activation(node_input @ node_w1)
            h_new = h_new @ node_w2

            # Residual connection
            h = h + h_new

        # Project output
        return h @ self._output_proj

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)  # ReLU
