"""
IMEX (Implicit-Explicit) time stepping schemes.

Handle stiff (implicit) and non-stiff (explicit) terms separately.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import solve


@dataclass
class IMEXResult:
    """Result of IMEX integration."""
    t: np.ndarray
    y: np.ndarray
    n_linear_solves: int
    n_function_evals: int


class IMEXScheme:
    """
    IMEX Runge-Kutta schemes.

    dy/dt = f_E(y) + f_I(y)

    where f_E is non-stiff (explicit) and f_I is stiff (implicit).
    """

    def __init__(self, scheme: str = "imex_euler"):
        """
        Args:
            scheme: "imex_euler", "imex_midpoint", "ars343", or "imex_bdf2"
        """
        self.scheme = scheme

    def solve(self, f_explicit: Callable[[np.ndarray], np.ndarray],
              f_implicit: Callable[[np.ndarray], np.ndarray],
              jacobian_implicit: Optional[Callable[[np.ndarray], np.ndarray]],
              y0: np.ndarray,
              t_span: Tuple[float, float],
              dt: float) -> IMEXResult:
        """
        Solve IMEX system.

        Args:
            f_explicit: Non-stiff RHS
            f_implicit: Stiff RHS
            jacobian_implicit: Jacobian of f_implicit (for Newton)
            y0: Initial condition
            t_span: Time interval
            dt: Time step

        Returns:
            IMEXResult
        """
        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        n_solves = 0
        n_evals = 0

        if self.scheme == "imex_euler":
            # Forward Euler for explicit, Backward Euler for implicit
            for i in range(1, len(t)):
                # Explicit part
                f_e = f_explicit(y[i-1])
                n_evals += 1

                # Implicit solve: y_new = y_old + dt*(f_e + f_i(y_new))
                y[i] = self._newton_solve(
                    lambda x: x - y[i-1] - dt * (f_e + f_implicit(x)),
                    lambda x: np.eye(len(y0)) - dt * jacobian_implicit(x) if jacobian_implicit else None,
                    y[i-1],
                    f_implicit
                )
                n_solves += 1

        elif self.scheme == "imex_midpoint":
            # Explicit midpoint for f_E, implicit midpoint for f_I
            for i in range(1, len(t)):
                # Predictor (explicit Euler)
                f_e = f_explicit(y[i-1])
                y_pred = y[i-1] + dt * f_e
                n_evals += 1

                # Corrector with implicit midpoint
                y_mid = 0.5 * (y[i-1] + y_pred)
                f_e_mid = f_explicit(y_mid)
                n_evals += 1

                # Implicit solve at midpoint
                def residual(x):
                    y_new = y[i-1] + dt * (f_e_mid + f_implicit(0.5 * (y[i-1] + x)))
                    return x - y_new

                y[i] = self._newton_solve(residual, None, y_pred, f_implicit)
                n_solves += 1

        elif self.scheme == "ars343":
            # Third-order IMEX RK (Ascher-Ruuth-Spiteri)
            for i in range(1, len(t)):
                y[i] = self._ars343_step(f_explicit, f_implicit, jacobian_implicit,
                                          y[i-1], dt)
                n_solves += 3  # 3 implicit stages
                n_evals += 3

        elif self.scheme == "imex_bdf2":
            # IMEX BDF2 (needs startup)
            if len(t) < 2:
                return IMEXResult(t=t, y=y, n_linear_solves=0, n_function_evals=0)

            # First step: IMEX Euler
            f_e = f_explicit(y[0])
            y[1] = self._newton_solve(
                lambda x: x - y[0] - dt * (f_e + f_implicit(x)),
                None, y[0], f_implicit
            )
            n_solves += 1
            n_evals += 1

            # Subsequent steps: BDF2
            for i in range(2, len(t)):
                f_e = f_explicit(y[i-1])
                n_evals += 1

                # BDF2: 3y_n - 4y_{n-1} + y_{n-2} = 2*dt*(f_e + f_i(y_n))
                def residual(x):
                    return 3*x - 4*y[i-1] + y[i-2] - 2*dt*(f_e + f_implicit(x))

                y[i] = self._newton_solve(residual, None, y[i-1], f_implicit)
                n_solves += 1

        return IMEXResult(
            t=t,
            y=y,
            n_linear_solves=n_solves,
            n_function_evals=n_evals
        )

    def _newton_solve(self, residual: Callable, jacobian: Optional[Callable],
                      y0: np.ndarray, f_impl: Callable,
                      tol: float = 1e-10, max_iter: int = 20) -> np.ndarray:
        """Newton's method for implicit solve."""
        y = y0.copy()

        for _ in range(max_iter):
            r = residual(y)

            if np.linalg.norm(r) < tol:
                break

            if jacobian is not None:
                J = jacobian(y)
                dy = solve(J, -r)
            else:
                # Numerical Jacobian
                n = len(y)
                J = np.zeros((n, n))
                eps = 1e-7

                for j in range(n):
                    y_plus = y.copy()
                    y_plus[j] += eps
                    J[:, j] = (residual(y_plus) - r) / eps

                dy = solve(J + 1e-10 * np.eye(n), -r)

            y = y + dy

        return y

    def _ars343_step(self, f_e: Callable, f_i: Callable, jac: Optional[Callable],
                     y: np.ndarray, dt: float) -> np.ndarray:
        """ARS(3,4,3) IMEX step."""
        # Coefficients
        gamma = 0.4358665215
        a21 = 0.5
        a31 = 0.0
        a32 = 0.5

        # Stage 1
        k_e1 = f_e(y)
        y1 = self._newton_solve(
            lambda x: x - y - dt * gamma * f_i(x),
            lambda x: np.eye(len(y)) - dt * gamma * jac(x) if jac else None,
            y, f_i
        )
        k_i1 = f_i(y1)

        # Stage 2
        y2_exp = y + dt * a21 * k_e1
        y2 = self._newton_solve(
            lambda x: x - y2_exp - dt * gamma * f_i(x),
            None, y2_exp, f_i
        )
        k_e2 = f_e(y2)
        k_i2 = f_i(y2)

        # Stage 3
        y3_exp = y + dt * (a31 * k_e1 + a32 * k_e2)
        y3 = self._newton_solve(
            lambda x: x - y3_exp - dt * gamma * f_i(x),
            None, y3_exp, f_i
        )
        k_e3 = f_e(y3)
        k_i3 = f_i(y3)

        # Combine
        b = [0.0, 0.5, 0.5]
        y_new = y + dt * (b[0] * (k_e1 + k_i1) + b[1] * (k_e2 + k_i2) + b[2] * (k_e3 + k_i3))

        return y_new


class NeuralIMEX:
    """
    Neural network-enhanced IMEX scheme.

    Learns optimal splitting of RHS into stiff/non-stiff parts.
    """

    def __init__(self, hidden_dims: List[int] = None):
        """
        Args:
            hidden_dims: Hidden layer dimensions
        """
        self.hidden_dims = hidden_dims or [64, 64]
        self._splitting_net: List = []

    def initialize(self, state_dim: int):
        """Initialize splitting network."""
        # Network outputs fraction of f to treat implicitly
        dims = [state_dim] + self.hidden_dims + [state_dim]

        self._splitting_net = []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self._splitting_net.append((W, b))

        self._state_dim = state_dim

    def compute_splitting(self, y: np.ndarray) -> np.ndarray:
        """
        Compute splitting coefficients.

        Returns alpha in [0, 1]^n where:
        f_I = alpha * f
        f_E = (1 - alpha) * f
        """
        h = y
        for i, (W, b) in enumerate(self._splitting_net):
            h = h @ W + b
            if i < len(self._splitting_net) - 1:
                h = np.tanh(h)

        # Sigmoid for [0, 1]
        return 1.0 / (1.0 + np.exp(-h))

    def solve(self, f: Callable[[np.ndarray], np.ndarray],
              jacobian: Optional[Callable],
              y0: np.ndarray,
              t_span: Tuple[float, float],
              dt: float) -> IMEXResult:
        """
        Solve with learned splitting.
        """
        self.initialize(len(y0))

        t_start, t_end = t_span
        t = np.arange(t_start, t_end + dt, dt)

        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        n_solves = 0

        for i in range(1, len(t)):
            # Compute adaptive splitting
            alpha = self.compute_splitting(y[i-1])

            # Split RHS
            f_full = f(y[i-1])
            f_e = (1 - alpha) * f_full
            f_i_func = lambda x: alpha * f(x)

            # IMEX Euler step
            def residual(x):
                return x - y[i-1] - dt * (f_e + f_i_func(x))

            y[i] = self._newton_solve(residual, y[i-1], f_i_func)
            n_solves += 1

        return IMEXResult(
            t=t, y=y, n_linear_solves=n_solves, n_function_evals=len(t) * 2
        )

    def _newton_solve(self, residual: Callable, y0: np.ndarray,
                      f_i: Callable, tol: float = 1e-10, max_iter: int = 10) -> np.ndarray:
        """Simplified Newton solve."""
        y = y0.copy()

        for _ in range(max_iter):
            r = residual(y)
            if np.linalg.norm(r) < tol:
                break

            # Numerical Jacobian
            n = len(y)
            J = np.zeros((n, n))
            eps = 1e-7
            for j in range(n):
                y_plus = y.copy()
                y_plus[j] += eps
                J[:, j] = (residual(y_plus) - r) / eps

            dy = np.linalg.solve(J + 1e-10 * np.eye(n), -r)
            y = y + dy

        return y

    def train(self, f: Callable,
              y_samples: np.ndarray,
              dt_range: Tuple[float, float],
              n_epochs: int = 100,
              lr: float = 0.001):
        """
        Train splitting network to minimize implicit solve iterations.
        """
        self.initialize(y_samples.shape[1])

        for epoch in range(n_epochs):
            total_cost = 0

            for y in y_samples:
                dt = np.random.uniform(*dt_range)

                # Try current splitting
                alpha = self.compute_splitting(y)

                f_full = f(y)
                f_e = (1 - alpha) * f_full

                # Count Newton iterations
                n_iters = 0
                y_next = y.copy()

                for _ in range(20):
                    r = y_next - y - dt * (f_e + alpha * f(y_next))
                    if np.linalg.norm(r) < 1e-8:
                        break
                    n_iters += 1
                    y_next = y_next - 0.1 * r  # Simplified

                total_cost += n_iters

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg iterations = {total_cost / len(y_samples):.2f}")


class AdaptiveIMEX:
    """
    IMEX with adaptive time stepping.

    Adjusts step size based on error estimates.
    """

    def __init__(self, base_scheme: str = "ars343",
                 rtol: float = 1e-4,
                 atol: float = 1e-6):
        """
        Args:
            base_scheme: Base IMEX scheme
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.imex = IMEXScheme(base_scheme)
        self.rtol = rtol
        self.atol = atol

    def solve(self, f_explicit: Callable,
              f_implicit: Callable,
              jacobian: Optional[Callable],
              y0: np.ndarray,
              t_span: Tuple[float, float],
              dt_init: float = 0.01) -> IMEXResult:
        """
        Solve with adaptive stepping.
        """
        t_start, t_end = t_span
        dt = dt_init

        t_list = [t_start]
        y_list = [y0.copy()]

        t = t_start
        y = y0.copy()

        n_solves = 0
        n_rejected = 0

        while t < t_end:
            if t + dt > t_end:
                dt = t_end - t

            # Take step with current dt
            result1 = self.imex.solve(
                f_explicit, f_implicit, jacobian,
                y, (t, t + dt), dt
            )
            y_full = result1.y[-1]

            # Take two half steps for error estimate
            result2a = self.imex.solve(
                f_explicit, f_implicit, jacobian,
                y, (t, t + dt/2), dt/2
            )
            y_half = result2a.y[-1]

            result2b = self.imex.solve(
                f_explicit, f_implicit, jacobian,
                y_half, (t + dt/2, t + dt), dt/2
            )
            y_double = result2b.y[-1]

            # Error estimate
            error = np.linalg.norm(y_full - y_double)
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_full))
            error_norm = error / np.linalg.norm(scale)

            if error_norm <= 1.0:
                # Accept step with Richardson extrapolation
                y = y_double + (y_double - y_full) / 3  # Improved solution
                t += dt
                t_list.append(t)
                y_list.append(y.copy())
                n_solves += result1.n_linear_solves + result2a.n_linear_solves + result2b.n_linear_solves

                # Increase step size
                factor = min(2.0, 0.9 * (1.0 / error_norm) ** 0.25)
                dt *= factor
            else:
                # Reject and reduce step size
                n_rejected += 1
                dt *= 0.5

            dt = max(dt, 1e-10)

        return IMEXResult(
            t=np.array(t_list),
            y=np.array(y_list),
            n_linear_solves=n_solves,
            n_function_evals=0
        )
