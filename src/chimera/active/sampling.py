"""
Active sampling strategies for simulation.

Choose where to run expensive simulations for maximum information gain.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np


@dataclass
class Sample:
    """A single sample point."""
    parameters: np.ndarray
    value: Optional[np.ndarray] = None
    uncertainty: float = float('inf')


class ActiveSampler(ABC):
    """Base class for active sampling strategies."""

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        """
        Args:
            bounds: Parameter bounds (n_params, 2) - [[low, high], ...]
            seed: Random seed
        """
        self.bounds = bounds
        self.n_params = len(bounds)
        self.rng = np.random.default_rng(seed)
        self._samples: List[Sample] = []
        self._surrogate = None

    @abstractmethod
    def next_sample(self) -> np.ndarray:
        """Suggest next parameter point to evaluate."""
        pass

    def add_observation(self, parameters: np.ndarray, value: np.ndarray):
        """Add observation from simulation."""
        self._samples.append(Sample(parameters=parameters, value=value))
        self._update_surrogate()

    def _update_surrogate(self):
        """Update surrogate model with new data."""
        pass

    def get_samples(self) -> List[Sample]:
        """Return all samples."""
        return self._samples


class UncertaintySampling(ActiveSampler):
    """
    Sample where uncertainty is highest.

    Classic active learning: query the point where model is least confident.
    """

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        super().__init__(bounds, seed)
        from chimera.active.surrogate import GaussianProcess
        self._surrogate = GaussianProcess(n_dims=self.n_params)

    def next_sample(self) -> np.ndarray:
        """Find point with maximum uncertainty."""
        if len(self._samples) < 5:
            # Initial random samples
            return self._random_sample()

        # Evaluate uncertainty on grid
        n_candidates = 1000
        candidates = self._generate_candidates(n_candidates)

        # Predict uncertainty
        _, uncertainties = self._surrogate.predict(candidates)

        # Return highest uncertainty point
        best_idx = np.argmax(uncertainties)
        return candidates[best_idx]

    def _random_sample(self) -> np.ndarray:
        """Generate random sample within bounds."""
        sample = np.zeros(self.n_params)
        for i, (low, high) in enumerate(self.bounds):
            sample[i] = self.rng.uniform(low, high)
        return sample

    def _generate_candidates(self, n: int) -> np.ndarray:
        """Generate candidate points."""
        candidates = np.zeros((n, self.n_params))
        for i, (low, high) in enumerate(self.bounds):
            candidates[:, i] = self.rng.uniform(low, high, n)
        return candidates

    def _update_surrogate(self):
        """Update GP with new observations."""
        if len(self._samples) < 2:
            return

        X = np.array([s.parameters for s in self._samples])
        y = np.array([s.value.flatten()[0] if s.value is not None else 0
                      for s in self._samples])

        self._surrogate.fit(X, y)


class ExpectedImprovement(ActiveSampler):
    """
    Expected Improvement acquisition function.

    Balances exploration (high uncertainty) and exploitation (promising regions).
    Standard for Bayesian optimization.
    """

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None,
                 xi: float = 0.01):
        """
        Args:
            bounds: Parameter bounds
            seed: Random seed
            xi: Exploration-exploitation tradeoff (higher = more exploration)
        """
        super().__init__(bounds, seed)
        self.xi = xi
        from chimera.active.surrogate import GaussianProcess
        self._surrogate = GaussianProcess(n_dims=self.n_params)

    def next_sample(self) -> np.ndarray:
        """Find point with maximum expected improvement."""
        if len(self._samples) < 5:
            return self._random_sample()

        # Find current best
        values = [s.value.flatten()[0] for s in self._samples if s.value is not None]
        if not values:
            return self._random_sample()
        y_best = np.min(values)

        # Evaluate EI on candidates
        n_candidates = 1000
        candidates = self._generate_candidates(n_candidates)
        ei_values = self._expected_improvement(candidates, y_best)

        best_idx = np.argmax(ei_values)
        return candidates[best_idx]

    def _expected_improvement(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """Compute Expected Improvement."""
        mean, std = self._surrogate.predict(X)
        std = np.maximum(std, 1e-9)

        # Standardized improvement
        z = (y_best - mean - self.xi) / std

        # EI = std * (z * Phi(z) + phi(z))
        from scipy.stats import norm
        ei = std * (z * norm.cdf(z) + norm.pdf(z))

        return ei

    def _random_sample(self) -> np.ndarray:
        sample = np.zeros(self.n_params)
        for i, (low, high) in enumerate(self.bounds):
            sample[i] = self.rng.uniform(low, high)
        return sample

    def _generate_candidates(self, n: int) -> np.ndarray:
        candidates = np.zeros((n, self.n_params))
        for i, (low, high) in enumerate(self.bounds):
            candidates[:, i] = self.rng.uniform(low, high, n)
        return candidates

    def _update_surrogate(self):
        if len(self._samples) < 2:
            return

        X = np.array([s.parameters for s in self._samples])
        y = np.array([s.value.flatten()[0] if s.value is not None else 0
                      for s in self._samples])

        self._surrogate.fit(X, y)


class QueryByCommittee(ActiveSampler):
    """
    Query-by-Committee active learning.

    Train ensemble of models, query where they disagree most.
    """

    def __init__(self, bounds: np.ndarray, n_models: int = 5,
                 seed: Optional[int] = None):
        super().__init__(bounds, seed)
        self.n_models = n_models
        self._committee = []
        self._initialize_committee()

    def _initialize_committee(self):
        """Create committee of diverse models."""
        from chimera.active.surrogate import GaussianProcess

        for i in range(self.n_models):
            # Different length scales for diversity
            gp = GaussianProcess(n_dims=self.n_params, length_scale=0.5 + 0.2 * i)
            self._committee.append(gp)

    def next_sample(self) -> np.ndarray:
        """Find point with maximum committee disagreement."""
        if len(self._samples) < 5:
            return self._random_sample()

        n_candidates = 1000
        candidates = self._generate_candidates(n_candidates)

        # Get predictions from all committee members
        predictions = []
        for model in self._committee:
            mean, _ = model.predict(candidates)
            predictions.append(mean)

        predictions = np.array(predictions)

        # Disagreement = variance across committee
        disagreement = np.var(predictions, axis=0)

        best_idx = np.argmax(disagreement)
        return candidates[best_idx]

    def _random_sample(self) -> np.ndarray:
        sample = np.zeros(self.n_params)
        for i, (low, high) in enumerate(self.bounds):
            sample[i] = self.rng.uniform(low, high)
        return sample

    def _generate_candidates(self, n: int) -> np.ndarray:
        candidates = np.zeros((n, self.n_params))
        for i, (low, high) in enumerate(self.bounds):
            candidates[:, i] = self.rng.uniform(low, high, n)
        return candidates

    def _update_surrogate(self):
        if len(self._samples) < 2:
            return

        X = np.array([s.parameters for s in self._samples])
        y = np.array([s.value.flatten()[0] if s.value is not None else 0
                      for s in self._samples])

        # Train each committee member on bootstrap sample
        n = len(X)
        for model in self._committee:
            indices = self.rng.choice(n, n, replace=True)
            model.fit(X[indices], y[indices])


class BatchActiveSampler(ActiveSampler):
    """
    Batch active learning for parallel simulations.

    Select multiple diverse points to evaluate simultaneously.
    """

    def __init__(self, bounds: np.ndarray, batch_size: int = 10,
                 seed: Optional[int] = None):
        super().__init__(bounds, seed)
        self.batch_size = batch_size
        from chimera.active.surrogate import GaussianProcess
        self._surrogate = GaussianProcess(n_dims=self.n_params)

    def next_batch(self) -> np.ndarray:
        """Suggest batch of points for parallel evaluation."""
        if len(self._samples) < 5:
            return np.array([self._random_sample() for _ in range(self.batch_size)])

        batch = []
        candidates = self._generate_candidates(1000)

        for _ in range(self.batch_size):
            # Find point with max uncertainty that's diverse from batch
            _, uncertainties = self._surrogate.predict(candidates)

            if batch:
                # Penalize points close to already selected
                batch_array = np.array(batch)
                for i, c in enumerate(candidates):
                    min_dist = np.min(np.linalg.norm(batch_array - c, axis=1))
                    uncertainties[i] *= (1 - np.exp(-min_dist))

            best_idx = np.argmax(uncertainties)
            batch.append(candidates[best_idx])

            # Remove selected point from candidates
            candidates = np.delete(candidates, best_idx, axis=0)
            uncertainties = np.delete(uncertainties, best_idx)

        return np.array(batch)

    def next_sample(self) -> np.ndarray:
        return self.next_batch()[0]

    def _random_sample(self) -> np.ndarray:
        sample = np.zeros(self.n_params)
        for i, (low, high) in enumerate(self.bounds):
            sample[i] = self.rng.uniform(low, high)
        return sample

    def _generate_candidates(self, n: int) -> np.ndarray:
        candidates = np.zeros((n, self.n_params))
        for i, (low, high) in enumerate(self.bounds):
            candidates[:, i] = self.rng.uniform(low, high, n)
        return candidates

    def _update_surrogate(self):
        if len(self._samples) < 2:
            return

        X = np.array([s.parameters for s in self._samples])
        y = np.array([s.value.flatten()[0] if s.value is not None else 0
                      for s in self._samples])

        self._surrogate.fit(X, y)
