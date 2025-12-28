"""
Adaptive fidelity selection.

Automatically choose when to use cheap vs expensive models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict
import numpy as np


@dataclass
class FidelityDecision:
    """Result of fidelity selection."""
    chosen_fidelity: int
    confidence: float
    estimated_error: float
    cost_saved: float


class FidelitySelector:
    """
    Learn when to use which fidelity level.

    Key insight: some regions of parameter space are well-approximated
    by cheap models, others require expensive models.
    """

    def __init__(self, fidelities: List[Callable], costs: List[float]):
        self.fidelities = fidelities
        self.costs = np.array(costs)
        self.n_fidelities = len(fidelities)
        self._classifier = None
        self._error_predictor = None

    def train(self, parameters: np.ndarray,
              outputs: List[List[np.ndarray]],
              error_threshold: float = 0.1):
        """
        Train fidelity selector.

        Args:
            parameters: Parameter samples
            outputs: outputs[i][j] = output of fidelity j at parameter i
            error_threshold: Acceptable error for using lower fidelity
        """
        n_samples = len(parameters)

        # Compute errors relative to highest fidelity
        errors = np.zeros((n_samples, self.n_fidelities))
        for i in range(n_samples):
            hf_output = outputs[i][-1]  # Highest fidelity
            for j in range(self.n_fidelities):
                rel_error = np.linalg.norm(outputs[i][j] - hf_output) / (
                    np.linalg.norm(hf_output) + 1e-10
                )
                errors[i, j] = rel_error

        # For each sample, find cheapest acceptable fidelity
        acceptable = errors <= error_threshold
        cheapest_acceptable = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            valid = np.where(acceptable[i])[0]
            if len(valid) > 0:
                # Find cheapest among acceptable
                cheapest_idx = valid[np.argmin(self.costs[valid])]
                cheapest_acceptable[i] = cheapest_idx
            else:
                # Use highest fidelity
                cheapest_acceptable[i] = self.n_fidelities - 1

        # Train classifier: parameters -> optimal fidelity
        self._train_classifier(parameters, cheapest_acceptable)

        # Train error predictor: (parameters, fidelity) -> expected error
        self._train_error_predictor(parameters, errors)

    def _train_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train fidelity selection classifier."""
        # Simple k-NN classifier
        self._classifier_data = (X, y)

    def _train_error_predictor(self, parameters: np.ndarray,
                               errors: np.ndarray):
        """Train error prediction model."""
        # Flatten: (param, fidelity) -> error
        n, K = errors.shape
        X = []
        y = []

        for i in range(n):
            for j in range(K):
                X.append(np.concatenate([parameters[i], [j]]))
                y.append(errors[i, j])

        X = np.array(X)
        y = np.array(y)

        # Simple linear regression for error prediction
        X_aug = np.column_stack([X, np.ones(len(X))])
        self._error_weights = np.linalg.lstsq(X_aug, np.log(y + 1e-10), rcond=None)[0]

    def select(self, parameter: np.ndarray,
               error_threshold: float = 0.1) -> FidelityDecision:
        """
        Select appropriate fidelity for given parameter.

        Args:
            parameter: Problem parameters
            error_threshold: Maximum acceptable error

        Returns:
            FidelityDecision with chosen fidelity and confidence
        """
        if self._classifier_data is None:
            raise ValueError("Model not trained")

        # Predict errors for all fidelities
        predicted_errors = np.zeros(self.n_fidelities)
        for j in range(self.n_fidelities):
            x = np.concatenate([parameter, [j], [1.0]])
            log_err = np.dot(x, self._error_weights)
            predicted_errors[j] = np.exp(log_err)

        # Find cheapest fidelity with acceptable error
        acceptable = predicted_errors <= error_threshold
        if np.any(acceptable):
            valid = np.where(acceptable)[0]
            chosen = valid[np.argmin(self.costs[valid])]
        else:
            chosen = self.n_fidelities - 1

        # Confidence based on margin
        confidence = 1.0 - predicted_errors[chosen] / error_threshold

        # Cost savings
        cost_saved = (self.costs[-1] - self.costs[chosen]) / self.costs[-1]

        return FidelityDecision(
            chosen_fidelity=chosen,
            confidence=float(confidence),
            estimated_error=float(predicted_errors[chosen]),
            cost_saved=float(cost_saved)
        )


class AdaptiveFidelity:
    """
    Adaptive fidelity during optimization.

    Start with cheap models, switch to expensive when needed.
    """

    def __init__(self, fidelities: List[Callable], costs: List[float]):
        self.fidelities = fidelities
        self.costs = np.array(costs)
        self._selector = FidelitySelector(fidelities, costs)
        self._history: List[Dict] = []

    def evaluate(self, parameter: np.ndarray,
                 mode: str = "adaptive") -> Tuple[np.ndarray, int]:
        """
        Evaluate at parameter with adaptive fidelity.

        Args:
            parameter: Problem parameters
            mode: 'adaptive', 'cheapest', or 'highest'

        Returns:
            (output, fidelity_used)
        """
        if mode == "cheapest":
            fidelity = 0
        elif mode == "highest":
            fidelity = len(self.fidelities) - 1
        else:
            decision = self._selector.select(parameter)
            fidelity = decision.chosen_fidelity

        output = self.fidelities[fidelity](parameter)

        self._history.append({
            'parameter': parameter.copy(),
            'fidelity': fidelity,
            'cost': self.costs[fidelity]
        })

        return output, fidelity

    def get_cost_statistics(self) -> Dict:
        """Get statistics on fidelity usage and cost savings."""
        if not self._history:
            return {}

        fidelities = [h['fidelity'] for h in self._history]
        costs = [h['cost'] for h in self._history]

        highest_cost = len(self._history) * self.costs[-1]
        actual_cost = sum(costs)

        return {
            'n_evaluations': len(self._history),
            'fidelity_counts': np.bincount(fidelities, minlength=len(self.fidelities)).tolist(),
            'total_cost': actual_cost,
            'cost_if_highest': highest_cost,
            'savings_fraction': 1 - actual_cost / highest_cost
        }


class TrustRegionFidelity:
    """
    Trust-region approach to fidelity selection.

    Use cheap model within trust region, expensive model to validate.
    """

    def __init__(self, cheap: Callable, expensive: Callable,
                 cheap_cost: float = 1.0, expensive_cost: float = 100.0):
        self.cheap = cheap
        self.expensive = expensive
        self.cheap_cost = cheap_cost
        self.expensive_cost = expensive_cost
        self._trust_radius = 1.0
        self._center = None
        self._center_value = None

    def evaluate(self, parameter: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Evaluate with trust-region based fidelity selection.

        Returns:
            (output, used_expensive)
        """
        if self._center is None:
            # First evaluation: use expensive
            self._center = parameter.copy()
            self._center_value = self.expensive(parameter)
            return self._center_value, True

        # Check if within trust region
        dist = np.linalg.norm(parameter - self._center)

        if dist <= self._trust_radius:
            # Use cheap model
            return self.cheap(parameter), False
        else:
            # Outside trust region: update and use expensive
            cheap_pred = self.cheap(parameter)
            expensive_val = self.expensive(parameter)

            # Update trust region based on prediction quality
            pred_error = np.linalg.norm(cheap_pred - expensive_val) / (
                np.linalg.norm(expensive_val) + 1e-10
            )

            if pred_error < 0.1:
                # Good prediction: expand trust region
                self._trust_radius *= 1.5
            else:
                # Poor prediction: shrink trust region
                self._trust_radius *= 0.5

            # Update center
            self._center = parameter.copy()
            self._center_value = expensive_val

            return expensive_val, True
