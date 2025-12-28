"""
Multi-fidelity fusion methods.

Combine outputs from multiple fidelity levels optimally.
Key insight: use cheap models to reduce variance of expensive model estimates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass
class FidelityLevel:
    """A single fidelity level."""
    name: str
    solver: Callable[[np.ndarray], np.ndarray]
    cost: float  # Relative computational cost
    correlation: float = 0.0  # Correlation with highest fidelity


@dataclass
class MultiFidelityEstimate:
    """Result of multi-fidelity estimation."""
    mean: np.ndarray
    variance: np.ndarray
    n_samples: List[int]
    total_cost: float
    variance_reduction: float


class MultiFidelityFusion:
    """
    Optimal fusion of multiple fidelity levels.

    Given a hierarchy of models f_0, f_1, ..., f_K with increasing
    accuracy and cost, find optimal sample allocation to minimize
    variance for a given budget.
    """

    def __init__(self, fidelity_levels: List[FidelityLevel]):
        self.levels = sorted(fidelity_levels, key=lambda x: -x.cost)
        self.n_levels = len(self.levels)
        self._correlations: Optional[np.ndarray] = None
        self._costs = np.array([l.cost for l in self.levels])

    def estimate_correlations(self, param_samples: np.ndarray,
                              n_pilot: int = 50) -> np.ndarray:
        """
        Estimate correlations between fidelity levels.

        Args:
            param_samples: Parameter samples for pilot runs
            n_pilot: Number of pilot samples

        Returns:
            Correlation matrix between levels
        """
        # Run pilot samples on all levels
        outputs = []
        indices = np.random.choice(len(param_samples), min(n_pilot, len(param_samples)), replace=False)
        pilot_params = param_samples[indices]

        for level in self.levels:
            level_outputs = []
            for p in pilot_params:
                out = level.solver(p)
                level_outputs.append(out.flatten()[0] if out.size > 1 else float(out))
            outputs.append(np.array(level_outputs))

        # Compute correlation matrix
        outputs = np.array(outputs)
        corr = np.corrcoef(outputs)

        self._correlations = corr
        return corr

    def optimal_allocation(self, budget: float) -> np.ndarray:
        """
        Find optimal sample allocation across fidelity levels.

        Minimizes variance subject to budget constraint.
        Uses MFMC optimal allocation formula.

        Args:
            budget: Total computational budget

        Returns:
            Optimal number of samples per level
        """
        if self._correlations is None:
            raise ValueError("Run estimate_correlations first")

        K = self.n_levels
        costs = self._costs
        rho = self._correlations[0, :]  # Correlation with highest fidelity

        # MFMC optimal allocation formula
        # n_k = n_0 * sqrt((rho_k^2 - rho_{k+1}^2) / c_k) * sqrt(c_0 / (1 - rho_1^2))
        rho_sq = rho ** 2
        rho_sq_diff = np.zeros(K)
        rho_sq_diff[:-1] = rho_sq[:-1] - rho_sq[1:]
        rho_sq_diff[-1] = rho_sq[-1]

        # Avoid negative values
        rho_sq_diff = np.maximum(rho_sq_diff, 1e-10)

        # Relative sample sizes
        r = np.sqrt(rho_sq_diff / costs) * np.sqrt(costs[0] / (1 - rho_sq[1] + 1e-10))

        # Scale to budget
        total_cost_per_n0 = np.sum(r * costs)
        n0 = budget / total_cost_per_n0

        n_samples = (r * n0).astype(int)
        n_samples = np.maximum(n_samples, 1)

        return n_samples

    def estimate(self, param: np.ndarray, n_samples: np.ndarray) -> MultiFidelityEstimate:
        """
        Compute multi-fidelity estimate.

        Uses control variate formulation for optimal variance reduction.

        Args:
            param: Parameter to evaluate
            n_samples: Number of samples per level

        Returns:
            Multi-fidelity estimate with variance
        """
        K = self.n_levels
        outputs = []

        # Evaluate all levels
        for k, level in enumerate(self.levels):
            level_out = level.solver(param)
            outputs.append(level_out)

        # MFMC estimator
        # Y_mf = Y_0 + sum_k alpha_k (Y_k - Y_{k-1})
        # where Y_k is the level-k estimate

        # Optimal alpha_k
        if self._correlations is not None:
            rho = self._correlations[0, :]
            alpha = rho[1:] * np.sqrt(
                (rho[:-1]**2 - rho[1:]**2) /
                (1 - rho[1:]**2 + 1e-10)
            )
        else:
            alpha = np.ones(K - 1)

        # Compute estimate
        Y_mf = outputs[0].copy()
        for k in range(1, K):
            correction = alpha[k - 1] * (outputs[k] - outputs[k - 1])
            Y_mf = Y_mf + correction

        # Variance estimation (simplified)
        var_hf = np.var(outputs[0]) if outputs[0].size > 1 else 0.0
        var_reduction = 1.0  # Would compute actual reduction

        total_cost = float(np.sum(n_samples * self._costs))

        return MultiFidelityEstimate(
            mean=Y_mf,
            variance=np.array([var_hf / n_samples[0]]),
            n_samples=list(n_samples),
            total_cost=total_cost,
            variance_reduction=var_reduction
        )


class ControlVariate:
    """
    Control variate method for variance reduction.

    Use cheap low-fidelity model as control variate for
    expensive high-fidelity model.

    Y_cv = Y_hf - alpha * (Y_lf - E[Y_lf])

    Reduces variance by factor (1 - rho^2) where rho is
    correlation between Y_hf and Y_lf.
    """

    def __init__(self, high_fidelity: Callable, low_fidelity: Callable):
        self.hf = high_fidelity
        self.lf = low_fidelity
        self._alpha: Optional[float] = None
        self._lf_mean: Optional[float] = None
        self._correlation: float = 0.0

    def calibrate(self, pilot_params: np.ndarray):
        """
        Calibrate control variate on pilot samples.

        Estimates optimal alpha and low-fidelity mean.
        """
        hf_outputs = []
        lf_outputs = []

        for p in pilot_params:
            hf_outputs.append(float(self.hf(p)))
            lf_outputs.append(float(self.lf(p)))

        hf_outputs = np.array(hf_outputs)
        lf_outputs = np.array(lf_outputs)

        # Estimate correlation
        self._correlation = np.corrcoef(hf_outputs, lf_outputs)[0, 1]

        # Optimal alpha = Cov(hf, lf) / Var(lf)
        cov = np.cov(hf_outputs, lf_outputs)[0, 1]
        var_lf = np.var(lf_outputs)
        self._alpha = cov / (var_lf + 1e-12)

        # Low-fidelity mean (could use more samples cheaply)
        self._lf_mean = np.mean(lf_outputs)

    def estimate(self, param: np.ndarray) -> Tuple[float, float]:
        """
        Compute control variate estimate.

        Returns:
            (estimate, variance_reduction_factor)
        """
        if self._alpha is None:
            raise ValueError("Run calibrate first")

        y_hf = float(self.hf(param))
        y_lf = float(self.lf(param))

        # Control variate estimate
        y_cv = y_hf - self._alpha * (y_lf - self._lf_mean)

        # Variance reduction factor
        var_reduction = 1 - self._correlation ** 2

        return y_cv, var_reduction


class MFMC:
    """
    Multi-Fidelity Monte Carlo (MFMC).

    Optimal allocation across many fidelity levels for
    Monte Carlo estimation with budget constraints.

    Reference: Peherstorfer et al., "Optimal Model Management for
    Multifidelity Monte Carlo Estimation" (2016)
    """

    def __init__(self, models: List[Callable], costs: List[float]):
        self.models = models
        self.costs = np.array(costs)
        self.n_models = len(models)
        self._covariance: Optional[np.ndarray] = None

    def estimate_statistics(self, samples: np.ndarray, n_pilot: int = 100):
        """Estimate covariance structure from pilot samples."""
        pilot_idx = np.random.choice(len(samples), min(n_pilot, len(samples)), replace=False)
        pilot = samples[pilot_idx]

        outputs = np.zeros((self.n_models, len(pilot)))
        for i, model in enumerate(self.models):
            for j, p in enumerate(pilot):
                outputs[i, j] = float(model(p))

        self._covariance = np.cov(outputs)
        self._means = np.mean(outputs, axis=1)
        self._variances = np.var(outputs, axis=1)

    def optimal_allocation(self, budget: float) -> np.ndarray:
        """Compute optimal sample allocation."""
        if self._covariance is None:
            raise ValueError("Run estimate_statistics first")

        # Correlation with highest fidelity
        corr = self._covariance[0, :] / np.sqrt(
            self._covariance[0, 0] * np.diag(self._covariance) + 1e-12
        )

        # Optimal ratios (Peherstorfer formula)
        K = self.n_models
        r = np.ones(K)

        for k in range(1, K):
            rho_k = corr[k]
            rho_k1 = corr[k - 1] if k > 1 else 1.0
            c_k = self.costs[k]
            c_0 = self.costs[0]

            r[k] = np.sqrt(
                (rho_k**2 - (corr[k + 1]**2 if k + 1 < K else 0)) * c_0 /
                ((1 - corr[1]**2 + 1e-10) * c_k)
            )

        # Scale to budget
        cost_per_n0 = np.sum(r * self.costs)
        n0 = int(budget / cost_per_n0)

        n_samples = (r * n0).astype(int)
        n_samples = np.maximum(n_samples, 1)

        return n_samples

    def estimate_mean(self, n_samples: np.ndarray,
                      sampler: Callable[[], np.ndarray]) -> Tuple[float, float]:
        """
        Compute MFMC mean estimate.

        Args:
            n_samples: Number of samples per model
            sampler: Function to generate parameter samples

        Returns:
            (mean_estimate, variance_estimate)
        """
        K = self.n_models

        # Generate nested samples
        all_samples = [sampler() for _ in range(n_samples[0])]

        # Evaluate models
        outputs = []
        for k in range(K):
            n_k = n_samples[k]
            model_out = [float(self.models[k](s)) for s in all_samples[:n_k]]
            outputs.append(np.array(model_out))

        # Compute optimal weights
        corr = self._covariance[0, :] / np.sqrt(
            self._covariance[0, 0] * np.diag(self._covariance) + 1e-12
        )

        # MFMC estimator
        Y = np.mean(outputs[0])

        for k in range(1, K):
            n_k = n_samples[k]
            n_k1 = n_samples[k - 1]

            # Control variate correction
            alpha_k = corr[k]
            Y_k_shared = np.mean(outputs[k][:n_k1])
            Y_k_all = np.mean(outputs[k])

            Y += alpha_k * (Y_k_shared - Y_k_all)

        # Variance estimate
        var = self._variances[0] / n_samples[0]
        for k in range(1, K):
            var *= (1 - corr[k]**2)

        return Y, var
