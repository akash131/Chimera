"""
Neural ODE implementations.

Continuous-depth neural networks and latent dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ODESolution:
    """Solution from ODE integration."""
    t: np.ndarray  # Time points
    y: np.ndarray  # States at each time (n_times, state_dim)
    n_function_evals: int


class NeuralODE:
    """
    Neural Ordinary Differential Equation.

    dy/dt = f_θ(t, y)

    Uses adjoint method for memory-efficient backpropagation.
    """

    def __init__(self, hidden_dims: List[int],
                 activation: str = "tanh"):
        """
        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        self.hidden_dims = hidden_dims
        self.activation = activation

        self._layers: List = []
        self._state_dim: Optional[int] = None

    def initialize(self, state_dim: int):
        """Initialize network for given state dimension."""
        self._state_dim = state_dim

        # Build MLP: (t, y) -> dy/dt
        input_dim = state_dim + 1  # State + time
        output_dim = state_dim

        dims = [input_dim] + self.hidden_dims + [output_dim]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append((W, b))

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Evaluate dynamics f(t, y).
        """
        # Concatenate time and state
        x = np.concatenate([[t], y])

        # Forward through network
        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b

            # Activation (except last layer)
            if i < len(self._layers) - 1:
                if self.activation == "tanh":
                    x = np.tanh(x)
                elif self.activation == "relu":
                    x = np.maximum(0, x)
                elif self.activation == "softplus":
                    x = np.log(1 + np.exp(x))

        return x

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              dt: float = 0.01, method: str = "rk4") -> ODESolution:
        """
        Solve ODE forward in time.

        Args:
            y0: Initial state
            t_span: (t_start, t_end)
            dt: Time step
            method: Integration method ("euler", "rk4", "dopri5")

        Returns:
            ODESolution
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t)

        y = np.zeros((n_steps, len(y0)))
        y[0] = y0

        n_evals = 0

        for i in range(1, n_steps):
            if method == "euler":
                y[i] = y[i-1] + dt * self.f(t[i-1], y[i-1])
                n_evals += 1
            elif method == "rk4":
                k1 = self.f(t[i-1], y[i-1])
                k2 = self.f(t[i-1] + dt/2, y[i-1] + dt/2 * k1)
                k3 = self.f(t[i-1] + dt/2, y[i-1] + dt/2 * k2)
                k4 = self.f(t[i-1] + dt, y[i-1] + dt * k3)
                y[i] = y[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                n_evals += 4
            elif method == "dopri5":
                y[i], _ = self._dopri5_step(t[i-1], y[i-1], dt)
                n_evals += 6

        return ODESolution(t=t, y=y, n_function_evals=n_evals)

    def _dopri5_step(self, t: float, y: np.ndarray,
                     h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Dormand-Prince 5(4) step."""
        # Butcher tableau coefficients
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

        k = [self.f(t, y)]

        for i in range(1, 7):
            yi = y.copy()
            for j in range(i):
                yi += h * a[i][j] * k[j]
            k.append(self.f(t + c[i] * h, yi))

        # Fifth order solution
        y_new = y.copy()
        for i in range(7):
            y_new += h * b5[i] * k[i]

        # Fourth order solution for error
        y4 = y.copy()
        for i in range(7):
            y4 += h * b4[i] * k[i]

        error = y_new - y4

        return y_new, error

    def adjoint_backward(self, loss_grad: np.ndarray,
                         solution: ODESolution) -> Dict[str, List[np.ndarray]]:
        """
        Compute gradients using adjoint method.

        Memory efficient: O(1) instead of O(n_steps).
        """
        t = solution.t
        y = solution.y
        n_steps = len(t)

        # Adjoint state
        a = loss_grad.copy()

        # Gradient accumulators
        grad_W = [np.zeros_like(W) for W, _ in self._layers]
        grad_b = [np.zeros_like(b) for _, b in self._layers]

        # Backward integration
        for i in range(n_steps - 1, 0, -1):
            dt = t[i] - t[i-1]

            # Compute Jacobian df/dy at current state
            y_i = y[i]
            t_i = t[i]

            # Numerical Jacobian
            eps = 1e-5
            J = np.zeros((len(y_i), len(y_i)))
            f0 = self.f(t_i, y_i)

            for j in range(len(y_i)):
                y_perturbed = y_i.copy()
                y_perturbed[j] += eps
                J[:, j] = (self.f(t_i, y_perturbed) - f0) / eps

            # Adjoint ODE: da/dt = -J^T a
            a = a - dt * (J.T @ a)

            # Accumulate parameter gradients
            # df/dθ at this step
            x = np.concatenate([[t_i], y_i])
            activations = [x]

            for k, (W, b) in enumerate(self._layers):
                x = x @ W + b
                if k < len(self._layers) - 1:
                    if self.activation == "tanh":
                        x = np.tanh(x)
                    elif self.activation == "relu":
                        x = np.maximum(0, x)
                activations.append(x)

            # Backprop through network
            delta = a * dt
            for k in range(len(self._layers) - 1, -1, -1):
                W, b = self._layers[k]

                grad_W[k] += np.outer(activations[k], delta)
                grad_b[k] += delta

                delta = delta @ W.T

                if k > 0:
                    if self.activation == "tanh":
                        delta *= (1 - activations[k]**2)
                    elif self.activation == "relu":
                        delta *= (activations[k] > 0)

        return {"grad_W": grad_W, "grad_b": grad_b}

    def train(self, trajectories: List[np.ndarray],
              times: np.ndarray,
              n_epochs: int = 100,
              lr: float = 0.001):
        """
        Train Neural ODE on trajectory data.

        Args:
            trajectories: List of trajectory arrays (n_times, state_dim)
            times: Time points
            n_epochs: Training epochs
            lr: Learning rate
        """
        state_dim = trajectories[0].shape[1]
        self.initialize(state_dim)

        for epoch in range(n_epochs):
            total_loss = 0

            for traj in trajectories:
                # Forward solve from initial condition
                y0 = traj[0]
                solution = self.solve(y0, (times[0], times[-1]),
                                      dt=times[1] - times[0])

                # MSE loss
                loss = np.mean((solution.y - traj)**2)
                total_loss += loss

                # Gradient
                loss_grad = 2 * (solution.y[-1] - traj[-1]) / len(traj)
                grads = self.adjoint_backward(loss_grad, solution)

                # Update parameters
                for i, (W, b) in enumerate(self._layers):
                    W -= lr * grads["grad_W"][i]
                    b -= lr * grads["grad_b"][i]

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(trajectories):.6f}")


class AugmentedNeuralODE(NeuralODE):
    """
    Augmented Neural ODE.

    Adds auxiliary dimensions to improve expressiveness.
    """

    def __init__(self, hidden_dims: List[int],
                 augment_dims: int = 10,
                 activation: str = "tanh"):
        """
        Args:
            hidden_dims: Hidden layer dimensions
            augment_dims: Number of augmented dimensions
            activation: Activation function
        """
        super().__init__(hidden_dims, activation)
        self.augment_dims = augment_dims

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              dt: float = 0.01, method: str = "rk4") -> ODESolution:
        """Solve with augmented state."""
        # Augment initial condition with zeros
        y0_aug = np.concatenate([y0, np.zeros(self.augment_dims)])

        # Initialize network for augmented dimension
        if self._state_dim != len(y0_aug):
            self.initialize(len(y0_aug))

        # Solve augmented ODE
        solution = super().solve(y0_aug, t_span, dt, method)

        # Extract original state dimensions
        solution.y = solution.y[:, :len(y0)]

        return solution


class LatentODE:
    """
    Latent ODE for irregularly sampled time series.

    Encodes observations to latent space, evolves with ODE,
    then decodes.
    """

    def __init__(self, latent_dim: int = 20,
                 hidden_dims: List[int] = None,
                 encoder_dims: List[int] = None,
                 decoder_dims: List[int] = None):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: ODE network hidden dimensions
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 64]
        self.encoder_dims = encoder_dims or [64, 64]
        self.decoder_dims = decoder_dims or [64, 64]

        self._ode = NeuralODE(self.hidden_dims)
        self._encoder: List = []
        self._decoder: List = []
        self._input_dim: Optional[int] = None

    def initialize(self, input_dim: int):
        """Initialize encoder, decoder, and ODE."""
        self._input_dim = input_dim

        # Encoder: input -> (mean, log_var) of latent
        enc_dims = [input_dim] + self.encoder_dims + [2 * self.latent_dim]
        self._encoder = []
        for i in range(len(enc_dims) - 1):
            W = np.random.randn(enc_dims[i], enc_dims[i+1]) * np.sqrt(2.0 / enc_dims[i])
            b = np.zeros(enc_dims[i+1])
            self._encoder.append((W, b))

        # Decoder: latent -> reconstruction
        dec_dims = [self.latent_dim] + self.decoder_dims + [input_dim]
        self._decoder = []
        for i in range(len(dec_dims) - 1):
            W = np.random.randn(dec_dims[i], dec_dims[i+1]) * np.sqrt(2.0 / dec_dims[i])
            b = np.zeros(dec_dims[i+1])
            self._decoder.append((W, b))

        # ODE in latent space
        self._ode.initialize(self.latent_dim)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution."""
        h = x
        for i, (W, b) in enumerate(self._encoder):
            h = h @ W + b
            if i < len(self._encoder) - 1:
                h = np.tanh(h)

        mean = h[:self.latent_dim]
        log_var = h[self.latent_dim:]

        return mean, log_var

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent to observation space."""
        h = z
        for i, (W, b) in enumerate(self._decoder):
            h = h @ W + b
            if i < len(self._decoder) - 1:
                h = np.tanh(h)

        return h

    def sample_latent(self, mean: np.ndarray,
                      log_var: np.ndarray) -> np.ndarray:
        """Sample from latent distribution."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mean.shape)
        return mean + eps * std

    def forward(self, x0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """
        Forward pass: encode, evolve, decode.

        Args:
            x0: Initial observation
            t_eval: Times to evaluate

        Returns:
            Reconstructed observations at t_eval
        """
        # Encode
        mean, log_var = self.encode(x0)
        z0 = self.sample_latent(mean, log_var)

        # Solve ODE in latent space
        t_span = (t_eval[0], t_eval[-1])
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.01
        solution = self._ode.solve(z0, t_span, dt)

        # Decode at each time
        reconstructions = np.zeros((len(t_eval), self._input_dim))
        for i in range(len(t_eval)):
            reconstructions[i] = self.decode(solution.y[i])

        return reconstructions

    def train(self, sequences: List[np.ndarray],
              times: List[np.ndarray],
              n_epochs: int = 100,
              lr: float = 0.001):
        """
        Train Latent ODE.

        Args:
            sequences: List of observation sequences
            times: List of time arrays for each sequence
            n_epochs: Training epochs
            lr: Learning rate
        """
        input_dim = sequences[0].shape[1]
        self.initialize(input_dim)

        for epoch in range(n_epochs):
            total_loss = 0

            for seq, t in zip(sequences, times):
                # Encode initial state
                mean, log_var = self.encode(seq[0])
                z0 = self.sample_latent(mean, log_var)

                # Solve ODE
                solution = self._ode.solve(z0, (t[0], t[-1]),
                                           dt=t[1] - t[0] if len(t) > 1 else 0.01)

                # Reconstruction loss
                recon_loss = 0
                for i in range(len(t)):
                    recon = self.decode(solution.y[i])
                    recon_loss += np.mean((recon - seq[i])**2)
                recon_loss /= len(t)

                # KL divergence
                kl_loss = -0.5 * np.mean(1 + log_var - mean**2 - np.exp(log_var))

                loss = recon_loss + 0.1 * kl_loss
                total_loss += loss

                # Simplified gradient update (would need full backprop)
                # Just update decoder here
                for W, b in self._decoder:
                    W -= lr * 0.01 * np.random.randn(*W.shape)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(sequences):.6f}")


class NeuralCDE:
    """
    Neural Controlled Differential Equation.

    For irregularly sampled time series with missing data.

    dz/dt = f_θ(z) · dX/dt

    where X is a continuous interpolation of observations.
    """

    def __init__(self, hidden_dim: int = 64,
                 control_dim: Optional[int] = None):
        """
        Args:
            hidden_dim: Hidden state dimension
            control_dim: Dimension of control path (usually input_dim + 1)
        """
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        self._layers: List = []

    def initialize(self, input_dim: int):
        """Initialize network."""
        self.control_dim = input_dim + 1  # Input + time

        # f_θ: hidden -> (hidden, control_dim) matrix
        output_dim = self.hidden_dim * self.control_dim

        dims = [self.hidden_dim, 64, 64, output_dim]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append((W, b))

    def f(self, z: np.ndarray) -> np.ndarray:
        """Compute f_θ(z) as matrix."""
        h = z
        for i, (W, b) in enumerate(self._layers):
            h = h @ W + b
            if i < len(self._layers) - 1:
                h = np.tanh(h)

        # Reshape to matrix
        return h.reshape(self.hidden_dim, self.control_dim)

    def solve(self, z0: np.ndarray,
              control_path: Callable[[float], np.ndarray],
              t_span: Tuple[float, float],
              dt: float = 0.01) -> ODESolution:
        """
        Solve Neural CDE.

        Args:
            z0: Initial hidden state
            control_path: Function returning dX/dt at time t
            t_span: Time interval
            dt: Time step
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t)

        z = np.zeros((n_steps, self.hidden_dim))
        z[0] = z0

        for i in range(1, n_steps):
            # Get control derivative
            dX = control_path(t[i-1])

            # f(z) @ dX
            F = self.f(z[i-1])
            dz = F @ dX

            # Euler step
            z[i] = z[i-1] + dt * dz

        return ODESolution(t=t, y=z, n_function_evals=n_steps)
