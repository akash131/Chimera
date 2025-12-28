"""
Learned time integration schemes.

Neural networks that learn to step forward in time
while respecting physical constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class LearnedIntegrator:
    """
    Neural network that learns optimal time stepping.

    Trained to match high-accuracy reference solutions
    while using larger time steps.
    """

    def __init__(self, hidden_dims: List[int] = None,
                 use_residual: bool = True):
        """
        Args:
            hidden_dims: Hidden layer dimensions
            use_residual: Add residual connection (y_new = y + network(y, dt))
        """
        self.hidden_dims = hidden_dims or [128, 128, 128]
        self.use_residual = use_residual

        self._layers: List = []
        self._state_dim: Optional[int] = None

    def initialize(self, state_dim: int):
        """Initialize network."""
        self._state_dim = state_dim

        # Input: (y, dt) -> y_new
        input_dim = state_dim + 1  # State + dt
        output_dim = state_dim

        dims = [input_dim] + self.hidden_dims + [output_dim]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append((W, b))

    def step(self, y: np.ndarray, dt: float) -> np.ndarray:
        """
        Take one time step.

        Args:
            y: Current state
            dt: Time step

        Returns:
            Next state
        """
        x = np.concatenate([y, [dt]])

        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:
                x = np.tanh(x)

        if self.use_residual:
            return y + dt * x
        else:
            return x

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve forward in time.

        Returns:
            (times, states)
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            y[i] = self.step(y[i-1], dt)

        return t, y

    def train(self, f: Callable[[np.ndarray], np.ndarray],
              y_samples: np.ndarray,
              dt_range: Tuple[float, float] = (0.01, 0.1),
              n_epochs: int = 1000,
              lr: float = 0.001):
        """
        Train integrator to match RK4.

        Args:
            f: RHS function dy/dt = f(y)
            y_samples: Sample states for training
            dt_range: Range of time steps
            n_epochs: Training epochs
            lr: Learning rate
        """
        self.initialize(y_samples.shape[1])

        for epoch in range(n_epochs):
            total_loss = 0

            for y in y_samples:
                # Random dt
                dt = np.random.uniform(*dt_range)

                # Reference: RK4
                k1 = f(y)
                k2 = f(y + dt/2 * k1)
                k3 = f(y + dt/2 * k2)
                k4 = f(y + dt * k3)
                y_ref = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

                # Learned step
                y_pred = self.step(y, dt)

                # Loss
                loss = np.mean((y_pred - y_ref)**2)
                total_loss += loss

                # Gradient (simplified numerical)
                eps = 1e-5
                for layer_idx, (W, b) in enumerate(self._layers):
                    grad_W = np.zeros_like(W)
                    grad_b = np.zeros_like(b)

                    for i in range(W.shape[0]):
                        for j in range(W.shape[1]):
                            W[i, j] += eps
                            loss_plus = np.mean((self.step(y, dt) - y_ref)**2)
                            W[i, j] -= 2 * eps
                            loss_minus = np.mean((self.step(y, dt) - y_ref)**2)
                            W[i, j] += eps
                            grad_W[i, j] = (loss_plus - loss_minus) / (2 * eps)

                    W -= lr * grad_W

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(y_samples):.6f}")


class SymplecticNetwork:
    """
    Symplectic neural network for Hamiltonian systems.

    Preserves symplectic structure: preserves phase space volume.
    Essential for long-time stability in Hamiltonian dynamics.
    """

    def __init__(self, n_layers: int = 3, hidden_dim: int = 64):
        """
        Args:
            n_layers: Number of symplectic layers
            hidden_dim: Hidden dimension for internal networks
        """
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self._K_layers: List = []  # Kinetic-like updates
        self._V_layers: List = []  # Potential-like updates

    def initialize(self, dim: int):
        """Initialize for system with dim degrees of freedom."""
        self._dim = dim

        # Each symplectic layer alternates between:
        # q <- q + grad_p K(p)
        # p <- p - grad_q V(q)

        for _ in range(self.n_layers):
            # K network: p -> scalar
            K_net = []
            dims = [dim, self.hidden_dim, self.hidden_dim, 1]
            for i in range(len(dims) - 1):
                W = np.random.randn(dims[i], dims[i+1]) * 0.1
                b = np.zeros(dims[i+1])
                K_net.append((W, b))
            self._K_layers.append(K_net)

            # V network: q -> scalar
            V_net = []
            for i in range(len(dims) - 1):
                W = np.random.randn(dims[i], dims[i+1]) * 0.1
                b = np.zeros(dims[i+1])
                V_net.append((W, b))
            self._V_layers.append(V_net)

    def _eval_network(self, x: np.ndarray, layers: List) -> float:
        """Evaluate scalar network."""
        h = x
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.tanh(h)
        return float(h)

    def _grad_network(self, x: np.ndarray, layers: List) -> np.ndarray:
        """Compute gradient of scalar network."""
        eps = 1e-5
        grad = np.zeros_like(x)
        f0 = self._eval_network(x, layers)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self._eval_network(x_plus, layers) - f0) / eps

        return grad

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        One symplectic time step.

        Uses splitting method with learned potentials.
        """
        for K_net, V_net in zip(self._K_layers, self._V_layers):
            # Update q using gradient of K(p)
            grad_K = self._grad_network(p, K_net)
            q = q + dt * grad_K

            # Update p using gradient of V(q)
            grad_V = self._grad_network(q, V_net)
            p = p - dt * grad_V

        return q, p

    def solve(self, q0: np.ndarray, p0: np.ndarray,
              t_span: Tuple[float, float], dt: float) -> Dict[str, np.ndarray]:
        """Solve Hamiltonian system."""
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)

        q = np.zeros((len(t), len(q0)))
        p = np.zeros((len(t), len(p0)))
        q[0] = q0
        p[0] = p0

        for i in range(1, len(t)):
            q[i], p[i] = self.step(q[i-1], p[i-1], dt)

        return {"t": t, "q": q, "p": p}


class VolumePreservingNetwork:
    """
    Volume-preserving neural network.

    Uses shear transformations that preserve phase space volume.
    More general than symplectic (works for non-Hamiltonian systems).
    """

    def __init__(self, n_layers: int = 4):
        """
        Args:
            n_layers: Number of volume-preserving layers
        """
        self.n_layers = n_layers
        self._shear_nets: List = []

    def initialize(self, dim: int):
        """Initialize shear networks."""
        self._dim = dim

        for i in range(self.n_layers):
            # Each layer updates one coordinate using others
            target_dim = i % dim

            # Network: all dimensions except target -> update for target
            net = []
            dims = [dim - 1, 32, 32, 1]
            for j in range(len(dims) - 1):
                W = np.random.randn(dims[j], dims[j+1]) * 0.1
                b = np.zeros(dims[j+1])
                net.append((W, b))

            self._shear_nets.append((target_dim, net))

    def _eval_shear(self, x: np.ndarray, target_dim: int, net: List) -> float:
        """Evaluate shear amount for target dimension."""
        # Extract non-target dimensions
        other_dims = [i for i in range(len(x)) if i != target_dim]
        x_other = x[other_dims]

        h = x_other
        for i, (W, b) in enumerate(net):
            h = h @ W + b
            if i < len(net) - 1:
                h = np.tanh(h)

        return float(h)

    def step(self, y: np.ndarray, dt: float) -> np.ndarray:
        """
        One volume-preserving step.

        Each shear is of form:
        x_i <- x_i + dt * f(x_1, ..., x_{i-1}, x_{i+1}, ..., x_n)

        This preserves volume because Jacobian has unit determinant.
        """
        y_new = y.copy()

        for target_dim, net in self._shear_nets:
            shear = self._eval_shear(y_new, target_dim, net)
            y_new[target_dim] += dt * shear

        return y_new

    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve forward."""
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            y[i] = self.step(y[i-1], dt)

        return t, y


class HamiltonianNeuralNetwork:
    """
    Hamiltonian Neural Network (HNN).

    Learns Hamiltonian H(q, p) from data, then uses symplectic integrator.
    """

    def __init__(self, hidden_dims: List[int] = None):
        """
        Args:
            hidden_dims: Hidden layer dimensions
        """
        self.hidden_dims = hidden_dims or [64, 64]
        self._layers: List = []

    def initialize(self, state_dim: int):
        """Initialize Hamiltonian network."""
        # H: (q, p) -> scalar
        dims = [state_dim] + self.hidden_dims + [1]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append((W, b))

    def H(self, qp: np.ndarray) -> float:
        """Evaluate Hamiltonian."""
        h = qp
        for i, (W, b) in enumerate(self._layers):
            h = h @ W + b
            if i < len(self._layers) - 1:
                h = np.tanh(h)
        return float(h)

    def grad_H(self, qp: np.ndarray) -> np.ndarray:
        """Gradient of Hamiltonian."""
        eps = 1e-5
        grad = np.zeros_like(qp)
        H0 = self.H(qp)

        for i in range(len(qp)):
            qp_plus = qp.copy()
            qp_plus[i] += eps
            grad[i] = (self.H(qp_plus) - H0) / eps

        return grad

    def equations_of_motion(self, qp: np.ndarray) -> np.ndarray:
        """
        Hamilton's equations:
        dq/dt = dH/dp
        dp/dt = -dH/dq
        """
        n = len(qp) // 2
        grad = self.grad_H(qp)

        dq = grad[n:]  # dH/dp
        dp = -grad[:n]  # -dH/dq

        return np.concatenate([dq, dp])

    def step_symplectic(self, q: np.ndarray, p: np.ndarray,
                        dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Symplectic Euler step using learned Hamiltonian."""
        n = len(q)
        qp = np.concatenate([q, p])

        # Symplectic Euler: first update p, then q
        grad = self.grad_H(qp)
        p_new = p - dt * grad[:n]

        qp_new = np.concatenate([q, p_new])
        grad_new = self.grad_H(qp_new)
        q_new = q + dt * grad_new[n:]

        return q_new, p_new

    def train(self, trajectories: List[np.ndarray],
              dtrajectories: List[np.ndarray],
              n_epochs: int = 1000, lr: float = 0.001):
        """
        Train HNN to match trajectory derivatives.

        Args:
            trajectories: List of (q, p) trajectories
            dtrajectories: List of (dq/dt, dp/dt) trajectories
        """
        state_dim = trajectories[0].shape[1]
        self.initialize(state_dim)

        for epoch in range(n_epochs):
            total_loss = 0

            for traj, dtraj in zip(trajectories, dtrajectories):
                for i in range(len(traj)):
                    qp = traj[i]
                    dqp_true = dtraj[i]

                    # Predicted derivatives from Hamiltonian
                    dqp_pred = self.equations_of_motion(qp)

                    loss = np.mean((dqp_pred - dqp_true)**2)
                    total_loss += loss

                    # Numerical gradient update
                    eps = 1e-5
                    for W, b in self._layers:
                        for i in range(min(5, W.shape[0])):  # Subsample
                            for j in range(min(5, W.shape[1])):
                                W[i, j] += eps
                                loss_plus = np.mean((self.equations_of_motion(qp) - dqp_true)**2)
                                W[i, j] -= 2 * eps
                                loss_minus = np.mean((self.equations_of_motion(qp) - dqp_true)**2)
                                W[i, j] += eps

                                grad = (loss_plus - loss_minus) / (2 * eps)
                                W[i, j] -= lr * grad

            if epoch % 100 == 0:
                avg_loss = total_loss / sum(len(t) for t in trajectories)
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
