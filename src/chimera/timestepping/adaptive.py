"""
Adaptive time stepping methods.

Automatically adjust step size based on error estimates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AdaptiveResult:
    """Result of adaptive integration."""
    t: np.ndarray
    y: np.ndarray
    n_steps: int
    n_rejected: int
    n_function_evals: int
    step_sizes: np.ndarray


class ErrorEstimator:
    """
    Error estimation strategies.
    """

    @staticmethod
    def embedded_rk(y_high: np.ndarray, y_low: np.ndarray) -> float:
        """Error from embedded RK pair."""
        return np.linalg.norm(y_high - y_low)

    @staticmethod
    def richardson(y_h: np.ndarray, y_h2: np.ndarray, order: int) -> float:
        """Richardson extrapolation error estimate."""
        factor = 2**order - 1
        return np.linalg.norm(y_h - y_h2) / factor

    @staticmethod
    def defect(f: Callable, t: float, y: np.ndarray,
               y_interp: Callable[[float], np.ndarray],
               dt: float) -> float:
        """Defect-based error estimate."""
        # Sample residual at midpoint
        t_mid = t + dt / 2
        y_mid = y_interp(t_mid)
        f_mid = f(t_mid, y_mid)

        # Approximate derivative from interpolant
        eps = dt * 1e-4
        dy = (y_interp(t_mid + eps) - y_interp(t_mid - eps)) / (2 * eps)

        return np.linalg.norm(dy - f_mid) * dt


class StepSizeController:
    """
    Step size control algorithms.
    """

    def __init__(self, method: str = "pi",
                 safety: float = 0.9,
                 min_factor: float = 0.2,
                 max_factor: float = 5.0):
        """
        Args:
            method: "standard", "pi", or "pid"
            safety: Safety factor
            min_factor: Minimum step size change factor
            max_factor: Maximum step size change factor
        """
        self.method = method
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor

        # Controller memory
        self._prev_error: float = 1.0
        self._prev_prev_error: float = 1.0

    def compute_factor(self, error: float, tolerance: float,
                       order: int) -> float:
        """
        Compute step size change factor.

        Args:
            error: Current error estimate
            tolerance: Target tolerance
            order: Method order

        Returns:
            Factor to multiply step size by
        """
        if error < 1e-15:
            return self.max_factor

        ratio = tolerance / error

        if self.method == "standard":
            factor = self.safety * ratio ** (1.0 / (order + 1))

        elif self.method == "pi":
            # PI controller
            k_I = 0.7 / (order + 1)
            k_P = 0.4 / (order + 1)

            factor = (self.safety * ratio ** k_I *
                      (self._prev_error / tolerance) ** k_P)
            self._prev_error = error

        elif self.method == "pid":
            # PID controller
            k_I = 0.6 / (order + 1)
            k_P = 0.3 / (order + 1)
            k_D = 0.1 / (order + 1)

            factor = (self.safety *
                      ratio ** k_I *
                      (self._prev_error / tolerance) ** k_P *
                      (self._prev_prev_error / tolerance) ** (-k_D))

            self._prev_prev_error = self._prev_error
            self._prev_error = error

        else:
            factor = self.safety * ratio ** (1.0 / (order + 1))

        return np.clip(factor, self.min_factor, self.max_factor)


class AdaptiveTimestepper:
    """
    Adaptive time stepping with embedded RK methods.
    """

    def __init__(self, method: str = "dopri5",
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 controller: str = "pi"):
        """
        Args:
            method: "dopri5", "rkf45", or "cash_karp"
            rtol: Relative tolerance
            atol: Absolute tolerance
            controller: Step size controller type
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.controller = StepSizeController(method=controller)

        # Method order
        self._order = {"dopri5": 5, "rkf45": 4, "cash_karp": 4}[method]

    def solve(self, f: Callable[[float, np.ndarray], np.ndarray],
              y0: np.ndarray,
              t_span: Tuple[float, float],
              max_step: float = np.inf,
              initial_step: Optional[float] = None) -> AdaptiveResult:
        """
        Solve ODE with adaptive stepping.

        Args:
            f: RHS function f(t, y)
            y0: Initial condition
            t_span: (t_start, t_end)
            max_step: Maximum step size
            initial_step: Initial step size (estimated if None)
        """
        t_start, t_end = t_span

        # Estimate initial step
        if initial_step is None:
            h = self._estimate_initial_step(f, t_start, y0)
        else:
            h = initial_step

        h = min(h, max_step, t_end - t_start)

        # Storage
        t_list = [t_start]
        y_list = [y0.copy()]
        h_list = []

        t = t_start
        y = y0.copy()
        n_steps = 0
        n_rejected = 0
        n_evals = 0

        while t < t_end:
            # Don't overshoot
            if t + h > t_end:
                h = t_end - t

            # Try step
            if self.method == "dopri5":
                y_new, y_err, n_f = self._dopri5_step(f, t, y, h)
            elif self.method == "rkf45":
                y_new, y_err, n_f = self._rkf45_step(f, t, y, h)
            elif self.method == "cash_karp":
                y_new, y_err, n_f = self._cash_karp_step(f, t, y, h)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            n_evals += n_f

            # Error estimate
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            error = np.linalg.norm(y_err / scale) / np.sqrt(len(y))

            # Accept or reject
            if error <= 1.0:
                # Accept step
                t += h
                y = y_new
                t_list.append(t)
                y_list.append(y.copy())
                h_list.append(h)
                n_steps += 1
            else:
                n_rejected += 1

            # Update step size
            factor = self.controller.compute_factor(error, 1.0, self._order)
            h = min(h * factor, max_step)

            # Minimum step size
            h = max(h, 1e-15)

        return AdaptiveResult(
            t=np.array(t_list),
            y=np.array(y_list),
            n_steps=n_steps,
            n_rejected=n_rejected,
            n_function_evals=n_evals,
            step_sizes=np.array(h_list)
        )

    def _estimate_initial_step(self, f: Callable, t0: float,
                                y0: np.ndarray) -> float:
        """Estimate reasonable initial step size."""
        f0 = f(t0, y0)
        d0 = np.linalg.norm(y0)
        d1 = np.linalg.norm(f0)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        # Estimate second derivative
        y1 = y0 + h0 * f0
        f1 = f(t0 + h0, y1)
        d2 = np.linalg.norm(f1 - f0) / h0

        if max(d1, d2) < 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (self._order + 1))

        return min(100 * h0, h1)

    def _dopri5_step(self, f: Callable, t: float, y: np.ndarray,
                     h: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """Dormand-Prince 5(4) step."""
        # Butcher tableau
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
        ]
        b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

        k = [f(t, y)]
        for i in range(1, 7):
            yi = y.copy()
            for j in range(i):
                yi += h * a[i][j] * k[j]
            k.append(f(t + c[i] * h, yi))

        # Fifth order
        y5 = y.copy()
        for i in range(7):
            y5 += h * b5[i] * k[i]

        # Fourth order
        y4 = y.copy()
        for i in range(7):
            y4 += h * b4[i] * k[i]

        return y5, y5 - y4, 7

    def _rkf45_step(self, f: Callable, t: float, y: np.ndarray,
                    h: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """Runge-Kutta-Fehlberg 4(5) step."""
        k1 = f(t, y)
        k2 = f(t + h/4, y + h/4 * k1)
        k3 = f(t + 3*h/8, y + h*(3/32*k1 + 9/32*k2))
        k4 = f(t + 12*h/13, y + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
        k5 = f(t + h, y + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
        k6 = f(t + h/2, y + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))

        y5 = y + h*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        y4 = y + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)

        return y5, y5 - y4, 6

    def _cash_karp_step(self, f: Callable, t: float, y: np.ndarray,
                        h: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """Cash-Karp 5(4) step."""
        k1 = f(t, y)
        k2 = f(t + h/5, y + h/5*k1)
        k3 = f(t + 3*h/10, y + h*(3/40*k1 + 9/40*k2))
        k4 = f(t + 3*h/5, y + h*(3/10*k1 - 9/10*k2 + 6/5*k3))
        k5 = f(t + h, y + h*(-11/54*k1 + 5/2*k2 - 70/27*k3 + 35/27*k4))
        k6 = f(t + 7*h/8, y + h*(1631/55296*k1 + 175/512*k2 + 575/13824*k3 + 44275/110592*k4 + 253/4096*k5))

        y5 = y + h*(37/378*k1 + 250/621*k3 + 125/594*k4 + 512/1771*k6)
        y4 = y + h*(2825/27648*k1 + 18575/48384*k3 + 13525/55296*k4 + 277/14336*k5 + 1/4*k6)

        return y5, y5 - y4, 6


class LearnedStepController:
    """
    Neural network for step size control.

    Learns optimal step sizes from simulation data.
    """

    def __init__(self, hidden_dims: List[int] = None):
        """
        Args:
            hidden_dims: Hidden layer dimensions
        """
        self.hidden_dims = hidden_dims or [32, 32]
        self._layers: List = []

    def initialize(self, state_dim: int):
        """Initialize network."""
        # Input: (y, f(y), h_prev, error_prev) -> h_new
        input_dim = state_dim * 2 + 2
        output_dim = 1

        dims = [input_dim] + self.hidden_dims + [output_dim]

        self._layers = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._layers.append((W, b))

    def predict_step(self, y: np.ndarray, f_y: np.ndarray,
                     h_prev: float, error_prev: float) -> float:
        """Predict optimal step size."""
        x = np.concatenate([y, f_y, [np.log(h_prev + 1e-10), np.log(error_prev + 1e-10)]])

        for i, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if i < len(self._layers) - 1:
                x = np.tanh(x)

        # Output is log(h) for positivity
        return np.exp(float(x))

    def train(self, trajectories: List[Dict],
              n_epochs: int = 100, lr: float = 0.001):
        """
        Train on trajectory data.

        Args:
            trajectories: List of dicts with 't', 'y', 'optimal_h'
        """
        if not trajectories:
            return

        state_dim = trajectories[0]['y'].shape[1]
        self.initialize(state_dim)

        for epoch in range(n_epochs):
            total_loss = 0

            for traj in trajectories:
                t = traj['t']
                y = traj['y']
                h_opt = traj['optimal_h']

                for i in range(1, len(t) - 1):
                    # Features
                    f_y = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
                    h_prev = t[i] - t[i-1]
                    error_prev = np.linalg.norm(y[i] - y[i-1]) / h_prev  # Rough estimate

                    # Predict
                    h_pred = self.predict_step(y[i], f_y, h_prev, error_prev)

                    # Loss
                    loss = (np.log(h_pred) - np.log(h_opt[i]))**2
                    total_loss += loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}")
