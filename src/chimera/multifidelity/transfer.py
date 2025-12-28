"""
Cross-fidelity transfer learning.

Learn mappings between different fidelity levels for acceleration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np


@dataclass
class TransferModel:
    """Learned transfer between fidelity levels."""
    source_fidelity: str
    target_fidelity: str
    transform: Callable[[np.ndarray], np.ndarray]
    inverse: Optional[Callable[[np.ndarray], np.ndarray]] = None
    error_estimate: float = 0.0


class FidelityTransfer:
    """
    Learn transfer operators between fidelity levels.

    Key insight: the difference between fidelities is often
    more structured than either fidelity alone.
    """

    def __init__(self):
        self._correction_weights = None
        self._trained = False

    def train(self, low_fidelity_data: np.ndarray,
              high_fidelity_data: np.ndarray,
              method: str = "linear"):
        """
        Train transfer model.

        Args:
            low_fidelity_data: Low-fidelity outputs (n_samples, n_outputs)
            high_fidelity_data: Corresponding high-fidelity outputs
            method: 'linear', 'polynomial', or 'neural'
        """
        if method == "linear":
            self._train_linear(low_fidelity_data, high_fidelity_data)
        elif method == "polynomial":
            self._train_polynomial(low_fidelity_data, high_fidelity_data)
        elif method == "neural":
            self._train_neural(low_fidelity_data, high_fidelity_data)
        else:
            raise ValueError(f"Unknown method: {method}")

        self._trained = True

    def _train_linear(self, lf: np.ndarray, hf: np.ndarray):
        """Linear correction: hf = A @ lf + b."""
        # Least squares fit
        n = len(lf)
        lf_aug = np.column_stack([lf, np.ones(n)])

        # Solve normal equations
        self._correction_weights = np.linalg.lstsq(lf_aug, hf, rcond=None)[0]

    def _train_polynomial(self, lf: np.ndarray, hf: np.ndarray, degree: int = 2):
        """Polynomial correction."""
        n, d = lf.shape if len(lf.shape) > 1 else (len(lf), 1)
        lf = lf.reshape(n, -1)

        # Build polynomial features
        features = [np.ones(n)]
        for i in range(d):
            for p in range(1, degree + 1):
                features.append(lf[:, i] ** p)

        # Cross terms for degree 2
        if degree >= 2 and d > 1:
            for i in range(d):
                for j in range(i + 1, d):
                    features.append(lf[:, i] * lf[:, j])

        X = np.column_stack(features)
        self._correction_weights = np.linalg.lstsq(X, hf, rcond=None)[0]
        self._poly_degree = degree

    def _train_neural(self, lf: np.ndarray, hf: np.ndarray):
        """Neural network correction."""
        n = len(lf)
        d_in = lf.shape[1] if len(lf.shape) > 1 else 1
        d_out = hf.shape[1] if len(hf.shape) > 1 else 1

        # Simple 2-layer network
        hidden = 64
        self._w1 = np.random.randn(d_in, hidden) * 0.1
        self._b1 = np.zeros(hidden)
        self._w2 = np.random.randn(hidden, d_out) * 0.1
        self._b2 = np.zeros(d_out)

        # Training
        lr = 0.01
        for epoch in range(1000):
            # Forward
            h = np.tanh(lf @ self._w1 + self._b1)
            pred = h @ self._w2 + self._b2

            # Loss
            loss = np.mean((pred - hf) ** 2)

            # Backward
            grad_pred = 2 * (pred - hf) / n
            grad_w2 = h.T @ grad_pred
            grad_b2 = np.sum(grad_pred, axis=0)

            grad_h = grad_pred @ self._w2.T * (1 - h ** 2)
            grad_w1 = lf.T @ grad_h
            grad_b1 = np.sum(grad_h, axis=0)

            # Update
            self._w1 -= lr * grad_w1
            self._b1 -= lr * grad_b1
            self._w2 -= lr * grad_w2
            self._b2 -= lr * grad_b2

    def transfer(self, low_fidelity: np.ndarray) -> np.ndarray:
        """Apply learned transfer to get high-fidelity estimate."""
        if not self._trained:
            raise ValueError("Model not trained")

        if hasattr(self, '_w1'):
            # Neural network
            h = np.tanh(low_fidelity @ self._w1 + self._b1)
            return h @ self._w2 + self._b2
        else:
            # Linear/polynomial
            lf = low_fidelity.reshape(-1, 1) if len(low_fidelity.shape) == 1 else low_fidelity
            lf_aug = np.column_stack([lf, np.ones(len(lf))])
            return lf_aug @ self._correction_weights


class CoarseToFine:
    """
    Coarse-to-fine transfer for mesh refinement.

    Learn correction from coarse mesh solution to fine mesh solution.
    """

    def __init__(self, coarse_mesh, fine_mesh):
        self.coarse_mesh = coarse_mesh
        self.fine_mesh = fine_mesh
        self._prolongation = None
        self._correction = None

    def build_prolongation(self):
        """Build interpolation operator from coarse to fine."""
        from scipy.interpolate import RBFInterpolator

        # Simple: RBF interpolation from coarse to fine points
        n_fine = self.fine_mesh.n_points
        n_coarse = self.coarse_mesh.n_points

        # Prolongation matrix P: fine = P @ coarse
        P = np.zeros((n_fine, n_coarse))

        for i in range(n_fine):
            fine_pt = self.fine_mesh.points[i]

            # Find nearest coarse points
            dists = np.linalg.norm(self.coarse_mesh.points - fine_pt, axis=1)
            nearest = np.argsort(dists)[:4]  # Use 4 nearest

            # Inverse distance weighting
            weights = 1.0 / (dists[nearest] + 1e-10)
            weights /= np.sum(weights)

            P[i, nearest] = weights

        self._prolongation = P

    def train_correction(self, coarse_solutions: List[np.ndarray],
                         fine_solutions: List[np.ndarray]):
        """
        Learn correction operator.

        correction = fine - prolongate(coarse)
        """
        if self._prolongation is None:
            self.build_prolongation()

        # Collect corrections
        corrections = []
        prolongated = []

        for coarse, fine in zip(coarse_solutions, fine_solutions):
            prol = self._prolongation @ coarse
            corr = fine - prol
            corrections.append(corr)
            prolongated.append(prol)

        corrections = np.array(corrections)
        prolongated = np.array(prolongated)

        # Learn correction as function of prolongated solution
        # Simple: linear regression
        self._correction = FidelityTransfer()
        self._correction.train(prolongated, corrections, method='linear')

    def transfer(self, coarse_solution: np.ndarray) -> np.ndarray:
        """Apply coarse-to-fine transfer with learned correction."""
        # Prolongate
        fine_estimate = self._prolongation @ coarse_solution

        # Add correction
        if self._correction is not None:
            correction = self._correction.transfer(fine_estimate.reshape(1, -1))
            fine_estimate = fine_estimate + correction.flatten()

        return fine_estimate


class PhysicsTransfer:
    """
    Transfer between different physics.

    Learn to predict expensive physics from cheap physics.
    Example: Predict CFD from potential flow.
    """

    def __init__(self):
        self._transfer_model = None

    def train(self, cheap_physics: List[np.ndarray],
              expensive_physics: List[np.ndarray],
              parameters: np.ndarray):
        """
        Train physics transfer model.

        Args:
            cheap_physics: Solutions from cheap physics model
            expensive_physics: Corresponding expensive physics solutions
            parameters: Problem parameters for each sample
        """
        # Combine cheap physics output with parameters
        n_samples = len(cheap_physics)
        inputs = []
        outputs = []

        for cheap, expensive, param in zip(cheap_physics, expensive_physics, parameters):
            # Flatten and concatenate
            inp = np.concatenate([cheap.flatten(), param.flatten()])
            out = expensive.flatten()
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        # Train transfer model
        self._transfer_model = FidelityTransfer()
        self._transfer_model.train(inputs, outputs, method='neural')

    def predict(self, cheap_solution: np.ndarray,
                parameters: np.ndarray) -> np.ndarray:
        """Predict expensive physics from cheap physics."""
        inp = np.concatenate([cheap_solution.flatten(), parameters.flatten()])
        return self._transfer_model.transfer(inp.reshape(1, -1)).flatten()
