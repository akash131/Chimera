"""
Neural network autoencoders for reduced-order modeling.

Nonlinear dimensionality reduction for complex dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AutoencoderConfig:
    """Configuration for autoencoder."""
    input_dim: int
    latent_dim: int
    hidden_dims: List[int] = None
    activation: str = "relu"
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 100


class LinearAutoencoder:
    """
    Linear autoencoder (equivalent to PCA/POD).

    Useful as baseline and for understanding nonlinear extensions.
    """

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim
        self._encoder: Optional[np.ndarray] = None
        self._decoder: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, n_epochs: int = 100, lr: float = 0.01):
        """
        Fit linear autoencoder via gradient descent.

        For linear case, this converges to PCA solution.
        """
        if X.shape[0] < X.shape[1]:
            X = X.T

        n_samples, n_features = X.shape

        # Center data
        self._mean = np.mean(X, axis=0)
        X_centered = X - self._mean

        # Initialize weights
        self._encoder = np.random.randn(n_features, self.latent_dim) * 0.1
        self._decoder = np.random.randn(self.latent_dim, n_features) * 0.1

        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            latent = X_centered @ self._encoder
            reconstructed = latent @ self._decoder

            # Loss
            loss = np.mean((X_centered - reconstructed) ** 2)

            # Gradients
            error = reconstructed - X_centered
            grad_decoder = latent.T @ error / n_samples
            grad_encoder = X_centered.T @ (error @ self._decoder.T) / n_samples

            # Update
            self._encoder -= lr * grad_encoder
            self._decoder -= lr * grad_decoder

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        # Orthogonalize encoder for stability
        self._encoder, _ = np.linalg.qr(self._encoder)
        self._decoder = self._encoder.T

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode to latent space."""
        return (X - self._mean) @ self._encoder

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        return z @ self._decoder + self._mean


class MLPLayer:
    """Simple MLP layer with activation."""

    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.activation = activation

        # For backprop
        self._input: Optional[np.ndarray] = None
        self._pre_activation: Optional[np.ndarray] = None

        # Gradients
        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._pre_activation = x @ self.W + self.b

        if self.activation == "relu":
            return np.maximum(0, self._pre_activation)
        elif self.activation == "tanh":
            return np.tanh(self._pre_activation)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-self._pre_activation))
        elif self.activation == "linear" or self.activation is None:
            return self._pre_activation
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Activation gradient
        if self.activation == "relu":
            grad_activation = (self._pre_activation > 0).astype(float)
        elif self.activation == "tanh":
            grad_activation = 1 - np.tanh(self._pre_activation) ** 2
        elif self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-self._pre_activation))
            grad_activation = s * (1 - s)
        else:
            grad_activation = np.ones_like(self._pre_activation)

        grad_pre = grad_output * grad_activation

        # Parameter gradients
        self.grad_W = self._input.T @ grad_pre / len(grad_output)
        self.grad_b = np.mean(grad_pre, axis=0)

        # Input gradient
        return grad_pre @ self.W.T


class ConvAutoencoder:
    """
    Convolutional autoencoder for structured data.

    Works with 2D spatial fields.
    """

    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.latent_dim = config.latent_dim

        # Simplified: use MLP with reshaping
        # Real implementation would use conv layers
        hidden = config.hidden_dims or [256, 128]

        # Encoder layers
        self.encoder_layers = []
        dims = [config.input_dim] + hidden + [config.latent_dim]
        for i in range(len(dims) - 1):
            activation = config.activation if i < len(dims) - 2 else "linear"
            self.encoder_layers.append(MLPLayer(dims[i], dims[i+1], activation))

        # Decoder layers (mirror)
        self.decoder_layers = []
        dims_dec = [config.latent_dim] + hidden[::-1] + [config.input_dim]
        for i in range(len(dims_dec) - 1):
            activation = config.activation if i < len(dims_dec) - 2 else "linear"
            self.decoder_layers.append(MLPLayer(dims_dec[i], dims_dec[i+1], activation))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode through encoder layers."""
        h = X
        for layer in self.encoder_layers:
            h = layer.forward(h)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode through decoder layers."""
        h = z
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full forward pass."""
        z = self.encode(X)
        x_recon = self.decode(z)
        return x_recon, z

    def fit(self, X: np.ndarray, n_epochs: Optional[int] = None,
            lr: Optional[float] = None, verbose: bool = True):
        """Train autoencoder."""
        if X.shape[0] < X.shape[1]:
            X = X.T

        n_samples = len(X)
        n_epochs = n_epochs or self.config.n_epochs
        lr = lr or self.config.learning_rate
        batch_size = self.config.batch_size

        for epoch in range(n_epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]

                # Forward
                x_recon, z = self.forward(batch)
                loss = np.mean((batch - x_recon) ** 2)
                total_loss += loss
                n_batches += 1

                # Backward
                grad = 2 * (x_recon - batch) / len(batch)

                # Decoder backward
                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)

                # Encoder backward
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # Update
                for layer in self.encoder_layers + self.decoder_layers:
                    layer.W -= lr * layer.grad_W
                    layer.b -= lr * layer.grad_b

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / n_batches:.6f}")


class VariationalAutoencoder:
    """
    Variational Autoencoder (VAE).

    Learns probabilistic latent representation.
    Enables sampling and uncertainty quantification.
    """

    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.latent_dim = config.latent_dim

        hidden = config.hidden_dims or [256, 128]

        # Encoder: outputs mean and log_var
        self.encoder_layers = []
        dims = [config.input_dim] + hidden
        for i in range(len(dims) - 1):
            self.encoder_layers.append(MLPLayer(dims[i], dims[i+1], config.activation))

        self.fc_mean = MLPLayer(hidden[-1], config.latent_dim, "linear")
        self.fc_log_var = MLPLayer(hidden[-1], config.latent_dim, "linear")

        # Decoder
        self.decoder_layers = []
        dims_dec = [config.latent_dim] + hidden[::-1] + [config.input_dim]
        for i in range(len(dims_dec) - 1):
            activation = config.activation if i < len(dims_dec) - 2 else "linear"
            self.decoder_layers.append(MLPLayer(dims_dec[i], dims_dec[i+1], activation))

    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to latent distribution parameters."""
        h = X
        for layer in self.encoder_layers:
            h = layer.forward(h)

        mean = self.fc_mean.forward(h)
        log_var = self.fc_log_var.forward(h)

        return mean, log_var

    def reparameterize(self, mean: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """Sample from latent distribution using reparameterization trick."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mean.shape)
        return mean + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        h = z
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass."""
        mean, log_var = self.encode(X)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decode(z)
        return x_recon, mean, log_var

    def loss(self, X: np.ndarray, x_recon: np.ndarray,
             mean: np.ndarray, log_var: np.ndarray) -> Tuple[float, float, float]:
        """
        VAE loss = Reconstruction + KL divergence.
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean((X - x_recon) ** 2)

        # KL divergence: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * np.mean(1 + log_var - mean**2 - np.exp(log_var))

        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss

    def fit(self, X: np.ndarray, n_epochs: Optional[int] = None,
            lr: Optional[float] = None, beta: float = 1.0):
        """
        Train VAE.

        Args:
            beta: Weight on KL term (beta-VAE)
        """
        if X.shape[0] < X.shape[1]:
            X = X.T

        n_samples = len(X)
        n_epochs = n_epochs or self.config.n_epochs
        lr = lr or self.config.learning_rate
        batch_size = self.config.batch_size

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]

                # Forward
                x_recon, mean, log_var = self.forward(batch)
                loss, recon, kl = self.loss(batch, x_recon, mean, log_var)
                total_loss += loss
                n_batches += 1

                # Simplified backward pass
                # Full implementation would compute gradients properly
                grad_recon = 2 * (x_recon - batch) / len(batch)

                # Update decoder
                grad = grad_recon
                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)

                # Update encoder (simplified)
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # Apply updates
                all_layers = (self.encoder_layers + [self.fc_mean, self.fc_log_var] +
                              self.decoder_layers)
                for layer in all_layers:
                    if layer.grad_W is not None:
                        layer.W -= lr * layer.grad_W
                        layer.b -= lr * layer.grad_b

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / n_batches:.6f}")

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from prior and decode."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)


class PhysicsAutoencoder:
    """
    Physics-constrained autoencoder.

    Incorporates physical constraints:
    - Conservation laws
    - Symmetries
    - Governing equation residuals
    """

    def __init__(self, config: AutoencoderConfig,
                 physics_loss: Optional[Callable] = None,
                 constraint_func: Optional[Callable] = None):
        """
        Args:
            config: Autoencoder configuration
            physics_loss: Function computing physics residual loss
            constraint_func: Hard constraint function
        """
        self.config = config

        # Base autoencoder
        hidden = config.hidden_dims or [256, 128]

        self.encoder_layers = []
        dims = [config.input_dim] + hidden + [config.latent_dim]
        for i in range(len(dims) - 1):
            activation = config.activation if i < len(dims) - 2 else "linear"
            self.encoder_layers.append(MLPLayer(dims[i], dims[i+1], activation))

        self.decoder_layers = []
        dims_dec = [config.latent_dim] + hidden[::-1] + [config.input_dim]
        for i in range(len(dims_dec) - 1):
            activation = config.activation if i < len(dims_dec) - 2 else "linear"
            self.decoder_layers.append(MLPLayer(dims_dec[i], dims_dec[i+1], activation))

        self.physics_loss = physics_loss
        self.constraint_func = constraint_func

    def encode(self, X: np.ndarray) -> np.ndarray:
        h = X
        for layer in self.encoder_layers:
            h = layer.forward(h)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = z
        for layer in self.decoder_layers:
            h = layer.forward(h)

        # Apply hard constraints if provided
        if self.constraint_func is not None:
            h = self.constraint_func(h)

        return h

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.encode(X)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_loss(self, X: np.ndarray, x_recon: np.ndarray,
                     z: np.ndarray,
                     physics_weight: float = 1.0) -> Tuple[float, Dict]:
        """Compute total loss with physics term."""
        # Reconstruction loss
        recon_loss = np.mean((X - x_recon) ** 2)

        # Physics loss
        if self.physics_loss is not None:
            phys_loss = self.physics_loss(x_recon, z)
        else:
            phys_loss = 0.0

        total_loss = recon_loss + physics_weight * phys_loss

        return total_loss, {
            "reconstruction": recon_loss,
            "physics": phys_loss,
            "total": total_loss
        }

    def fit(self, X: np.ndarray, n_epochs: Optional[int] = None,
            lr: Optional[float] = None, physics_weight: float = 1.0):
        """Train physics-constrained autoencoder."""
        if X.shape[0] < X.shape[1]:
            X = X.T

        n_samples = len(X)
        n_epochs = n_epochs or self.config.n_epochs
        lr = lr or self.config.learning_rate
        batch_size = self.config.batch_size

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]

                # Forward
                x_recon, z = self.forward(batch)
                loss, loss_dict = self.compute_loss(batch, x_recon, z, physics_weight)
                total_loss += loss
                n_batches += 1

                # Backward (simplified)
                grad = 2 * (x_recon - batch) / len(batch)

                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # Update
                for layer in self.encoder_layers + self.decoder_layers:
                    layer.W -= lr * layer.grad_W
                    layer.b -= lr * layer.grad_b

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / n_batches:.6f}")


class SymmetryAutoencoder:
    """
    Autoencoder that respects symmetries of the physical system.

    Learns equivariant representations.
    """

    def __init__(self, config: AutoencoderConfig,
                 symmetry_group: str = "rotation"):
        """
        Args:
            config: Autoencoder configuration
            symmetry_group: Type of symmetry ("rotation", "reflection", "translation")
        """
        self.config = config
        self.symmetry_group = symmetry_group

        hidden = config.hidden_dims or [256, 128]

        # Standard encoder/decoder
        self.encoder_layers = []
        dims = [config.input_dim] + hidden + [config.latent_dim]
        for i in range(len(dims) - 1):
            activation = config.activation if i < len(dims) - 2 else "linear"
            self.encoder_layers.append(MLPLayer(dims[i], dims[i+1], activation))

        self.decoder_layers = []
        dims_dec = [config.latent_dim] + hidden[::-1] + [config.input_dim]
        for i in range(len(dims_dec) - 1):
            activation = config.activation if i < len(dims_dec) - 2 else "linear"
            self.decoder_layers.append(MLPLayer(dims_dec[i], dims_dec[i+1], activation))

    def augment(self, X: np.ndarray) -> List[np.ndarray]:
        """Apply symmetry transformations for data augmentation."""
        augmented = [X]

        if self.symmetry_group == "rotation":
            # 90-degree rotations for 2D grid data
            # Simplified: random shuffle
            augmented.append(X[:, ::-1])
        elif self.symmetry_group == "reflection":
            augmented.append(-X)
        elif self.symmetry_group == "translation":
            shift = np.random.randint(1, min(10, len(X[0])))
            augmented.append(np.roll(X, shift, axis=1))

        return augmented

    def equivariance_loss(self, X: np.ndarray, z: np.ndarray) -> float:
        """Compute loss encouraging equivariant latent space."""
        # Apply transformation to input
        X_transformed = self.augment(X)[1] if len(self.augment(X)) > 1 else X

        # Encode transformed input
        z_transformed = self.encode(X_transformed)

        # Latent representations should be related by same transformation
        # For now, use simple consistency loss
        return np.mean((z - z_transformed) ** 2)

    def encode(self, X: np.ndarray) -> np.ndarray:
        h = X
        for layer in self.encoder_layers:
            h = layer.forward(h)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = z
        for layer in self.decoder_layers:
            h = layer.forward(h)
        return h

    def fit(self, X: np.ndarray, n_epochs: Optional[int] = None,
            lr: Optional[float] = None, equivariance_weight: float = 0.1):
        """Train with equivariance regularization."""
        if X.shape[0] < X.shape[1]:
            X = X.T

        n_samples = len(X)
        n_epochs = n_epochs or self.config.n_epochs
        lr = lr or self.config.learning_rate
        batch_size = self.config.batch_size

        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]

            total_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]

                # Forward
                z = self.encode(batch)
                x_recon = self.decode(z)

                # Losses
                recon_loss = np.mean((batch - x_recon) ** 2)
                equiv_loss = self.equivariance_loss(batch, z)
                loss = recon_loss + equivariance_weight * equiv_loss

                total_loss += loss
                n_batches += 1

                # Backward (simplified)
                grad = 2 * (x_recon - batch) / len(batch)

                for layer in reversed(self.decoder_layers):
                    grad = layer.backward(grad)
                for layer in reversed(self.encoder_layers):
                    grad = layer.backward(grad)

                # Update
                for layer in self.encoder_layers + self.decoder_layers:
                    layer.W -= lr * layer.grad_W
                    layer.b -= lr * layer.grad_b

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / n_batches:.6f}")
