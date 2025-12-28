"""
Global sensitivity analysis methods.

Identify which inputs most affect output uncertainty.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    first_order: np.ndarray  # Main effects
    total_order: np.ndarray  # Total effects (including interactions)
    parameter_names: Optional[List[str]] = None


class SobolIndices:
    """
    Sobol sensitivity indices.

    Variance-based global sensitivity analysis.
    First-order: effect of single parameter
    Total-order: effect including all interactions
    """

    def __init__(self, n_samples: int = 1000, seed: Optional[int] = None):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def analyze(self, model: Callable,
                bounds: np.ndarray,
                parameter_names: Optional[List[str]] = None) -> SensitivityResult:
        """
        Compute Sobol indices.

        Args:
            model: Function (n_params,) -> scalar
            bounds: Parameter bounds (n_params, 2)
            parameter_names: Optional names for parameters

        Returns:
            SensitivityResult with first and total order indices
        """
        n_params = len(bounds)
        n = self.n_samples

        # Generate two independent sample matrices A and B
        A = self._sample_uniform(bounds, n)
        B = self._sample_uniform(bounds, n)

        # Evaluate model on A and B
        f_A = np.array([model(a) for a in A])
        f_B = np.array([model(b) for b in B])

        # Estimate total variance
        f_all = np.concatenate([f_A, f_B])
        var_total = np.var(f_all)

        if var_total < 1e-10:
            # No variance - all indices are zero
            return SensitivityResult(
                first_order=np.zeros(n_params),
                total_order=np.zeros(n_params),
                parameter_names=parameter_names
            )

        # Compute first-order and total-order indices
        first_order = np.zeros(n_params)
        total_order = np.zeros(n_params)

        for i in range(n_params):
            # A_B_i: A with i-th column from B
            A_B = A.copy()
            A_B[:, i] = B[:, i]
            f_AB = np.array([model(ab) for ab in A_B])

            # First-order index (Saltelli estimator)
            first_order[i] = np.mean(f_B * (f_AB - f_A)) / var_total

            # Total-order index
            total_order[i] = 0.5 * np.mean((f_A - f_AB) ** 2) / var_total

        # Clip to valid range
        first_order = np.clip(first_order, 0, 1)
        total_order = np.clip(total_order, 0, 1)

        return SensitivityResult(
            first_order=first_order,
            total_order=total_order,
            parameter_names=parameter_names
        )

    def _sample_uniform(self, bounds: np.ndarray, n: int) -> np.ndarray:
        """Sample uniformly within bounds."""
        samples = np.zeros((n, len(bounds)))
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = self.rng.uniform(low, high, n)
        return samples


class MorrisScreening:
    """
    Morris method for screening.

    Efficient method to identify influential parameters.
    Good for high-dimensional problems.
    """

    def __init__(self, n_trajectories: int = 10, n_levels: int = 4):
        self.n_trajectories = n_trajectories
        self.n_levels = n_levels

    def analyze(self, model: Callable,
                bounds: np.ndarray,
                parameter_names: Optional[List[str]] = None) -> dict:
        """
        Perform Morris screening.

        Returns:
            Dictionary with mu (mean effect), sigma (std of effect)
        """
        n_params = len(bounds)

        # Generate trajectories
        trajectories = self._generate_trajectories(bounds)

        # Compute elementary effects
        effects = [[] for _ in range(n_params)]

        for traj in trajectories:
            for i in range(n_params):
                # Find step in parameter i
                for j in range(len(traj) - 1):
                    if np.abs(traj[j + 1, i] - traj[j, i]) > 1e-10:
                        f1 = model(traj[j])
                        f2 = model(traj[j + 1])
                        delta = traj[j + 1, i] - traj[j, i]
                        effect = (f2 - f1) / delta
                        effects[i].append(effect)
                        break

        # Compute statistics
        mu = np.array([np.mean(np.abs(e)) if e else 0 for e in effects])
        sigma = np.array([np.std(e) if len(e) > 1 else 0 for e in effects])

        return {
            "mu": mu,
            "sigma": sigma,
            "mu_star": mu,  # mu* = mean of |effect|
            "parameter_names": parameter_names
        }

    def _generate_trajectories(self, bounds: np.ndarray) -> List[np.ndarray]:
        """Generate Morris trajectories."""
        n_params = len(bounds)
        trajectories = []

        for _ in range(self.n_trajectories):
            # Start point
            levels = np.arange(self.n_levels) / (self.n_levels - 1)
            x0 = np.array([np.random.choice(levels) for _ in range(n_params)])

            # Scale to bounds
            x0_scaled = bounds[:, 0] + x0 * (bounds[:, 1] - bounds[:, 0])

            # Generate trajectory
            traj = [x0_scaled.copy()]

            # Random order of parameters
            order = np.random.permutation(n_params)

            for i in order:
                delta = (bounds[i, 1] - bounds[i, 0]) / (self.n_levels - 1)
                x_new = traj[-1].copy()

                # Step up or down
                if np.random.rand() > 0.5:
                    x_new[i] = min(x_new[i] + delta, bounds[i, 1])
                else:
                    x_new[i] = max(x_new[i] - delta, bounds[i, 0])

                traj.append(x_new)

            trajectories.append(np.array(traj))

        return trajectories


class VarianceDecomposition:
    """
    Functional ANOVA decomposition.

    Decompose output variance into contributions from each input and their interactions.
    """

    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples

    def decompose(self, model: Callable,
                  bounds: np.ndarray) -> dict:
        """
        Perform variance decomposition.

        Returns components of variance for each input and interaction.
        """
        n_params = len(bounds)
        n = self.n_samples

        # Sample
        samples = np.zeros((n, n_params))
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = np.random.uniform(low, high, n)

        # Evaluate
        outputs = np.array([model(s) for s in samples])
        total_var = np.var(outputs)

        # Main effects
        main_effects = np.zeros(n_params)

        for i in range(n_params):
            # Marginal variance of i-th input
            # E[Var[Y | X_i]]
            n_bins = 10
            bins = np.linspace(bounds[i, 0], bounds[i, 1], n_bins + 1)
            bin_means = []

            for j in range(n_bins):
                mask = (samples[:, i] >= bins[j]) & (samples[:, i] < bins[j + 1])
                if np.sum(mask) > 0:
                    bin_means.append(np.mean(outputs[mask]))

            if bin_means:
                main_effects[i] = np.var(bin_means)

        # Normalize
        main_effects /= (total_var + 1e-10)

        return {
            "total_variance": total_var,
            "main_effects": main_effects,
            "main_effects_fraction": main_effects,
            "interaction_fraction": 1 - np.sum(main_effects)
        }
