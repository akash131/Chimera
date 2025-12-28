"""
Bayesian inference methods.

Infer distributions over parameters given observations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


@dataclass
class InferenceResult:
    """Result of Bayesian inference."""
    samples: np.ndarray
    map_estimate: np.ndarray
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    acceptance_rate: Optional[float] = None
    log_evidence: Optional[float] = None


class BayesianInference:
    """
    Base class for Bayesian inference.

    Infers posterior p(theta | data) given likelihood and prior.
    """

    def __init__(self, prior_mean: np.ndarray, prior_std: np.ndarray):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.n_params = len(prior_mean)

    def log_prior(self, theta: np.ndarray) -> float:
        """Evaluate log prior."""
        z = (theta - self.prior_mean) / self.prior_std
        return -0.5 * np.sum(z ** 2)

    def log_likelihood(self, theta: np.ndarray, data: np.ndarray,
                       model: Callable, noise_std: float) -> float:
        """Evaluate log likelihood."""
        prediction = model(theta)
        residual = data - prediction
        return -0.5 * np.sum((residual / noise_std) ** 2)

    def log_posterior(self, theta: np.ndarray, data: np.ndarray,
                      model: Callable, noise_std: float) -> float:
        """Evaluate log posterior (up to normalizing constant)."""
        return self.log_prior(theta) + self.log_likelihood(theta, data, model, noise_std)


class MCMC(BayesianInference):
    """
    Markov Chain Monte Carlo sampling.

    Sample from posterior using Metropolis-Hastings.
    """

    def __init__(self, prior_mean: np.ndarray, prior_std: np.ndarray,
                 proposal_std: Optional[np.ndarray] = None):
        super().__init__(prior_mean, prior_std)
        self.proposal_std = proposal_std if proposal_std is not None else prior_std * 0.1

    def sample(self, data: np.ndarray, model: Callable,
               noise_std: float, n_samples: int = 10000,
               n_burnin: int = 1000) -> InferenceResult:
        """
        Run MCMC sampling.

        Args:
            data: Observed data
            model: Forward model
            noise_std: Observation noise standard deviation
            n_samples: Number of samples after burn-in
            n_burnin: Number of burn-in samples

        Returns:
            InferenceResult with posterior samples
        """
        samples = []
        accepted = 0

        # Initialize at prior mean
        theta = self.prior_mean.copy()
        log_p = self.log_posterior(theta, data, model, noise_std)

        for i in range(n_samples + n_burnin):
            # Propose
            theta_prop = theta + np.random.randn(self.n_params) * self.proposal_std
            log_p_prop = self.log_posterior(theta_prop, data, model, noise_std)

            # Accept/reject
            if np.log(np.random.rand()) < log_p_prop - log_p:
                theta = theta_prop
                log_p = log_p_prop
                if i >= n_burnin:
                    accepted += 1

            if i >= n_burnin:
                samples.append(theta.copy())

        samples = np.array(samples)

        return InferenceResult(
            samples=samples,
            map_estimate=samples[np.argmax([self.log_posterior(s, data, model, noise_std)
                                            for s in samples])],
            posterior_mean=np.mean(samples, axis=0),
            posterior_std=np.std(samples, axis=0),
            acceptance_rate=accepted / n_samples
        )


class AdaptiveMCMC(MCMC):
    """
    Adaptive MCMC with automatic proposal tuning.
    """

    def __init__(self, prior_mean: np.ndarray, prior_std: np.ndarray,
                 target_acceptance: float = 0.234):
        super().__init__(prior_mean, prior_std)
        self.target_acceptance = target_acceptance
        self._adaptation_rate = 0.1

    def sample(self, data: np.ndarray, model: Callable,
               noise_std: float, n_samples: int = 10000,
               n_burnin: int = 1000) -> InferenceResult:
        """Run adaptive MCMC."""
        samples = []
        accepted = 0
        acceptance_window = 100

        theta = self.prior_mean.copy()
        log_p = self.log_posterior(theta, data, model, noise_std)

        window_accepted = 0

        for i in range(n_samples + n_burnin):
            # Propose
            theta_prop = theta + np.random.randn(self.n_params) * self.proposal_std
            log_p_prop = self.log_posterior(theta_prop, data, model, noise_std)

            # Accept/reject
            if np.log(np.random.rand()) < log_p_prop - log_p:
                theta = theta_prop
                log_p = log_p_prop
                window_accepted += 1
                if i >= n_burnin:
                    accepted += 1

            # Adapt proposal during burn-in
            if i < n_burnin and (i + 1) % acceptance_window == 0:
                current_rate = window_accepted / acceptance_window
                if current_rate > self.target_acceptance:
                    self.proposal_std *= 1 + self._adaptation_rate
                else:
                    self.proposal_std *= 1 - self._adaptation_rate
                window_accepted = 0

            if i >= n_burnin:
                samples.append(theta.copy())

        samples = np.array(samples)

        return InferenceResult(
            samples=samples,
            map_estimate=samples[np.argmax([self.log_posterior(s, data, model, noise_std)
                                            for s in samples])],
            posterior_mean=np.mean(samples, axis=0),
            posterior_std=np.std(samples, axis=0),
            acceptance_rate=accepted / n_samples
        )


class VariationalInference(BayesianInference):
    """
    Variational Inference for approximate posterior.

    Approximate posterior with simpler distribution (e.g., Gaussian).
    Much faster than MCMC for high-dimensional problems.
    """

    def __init__(self, prior_mean: np.ndarray, prior_std: np.ndarray):
        super().__init__(prior_mean, prior_std)

    def fit(self, data: np.ndarray, model: Callable,
            noise_std: float, n_iterations: int = 1000,
            n_samples: int = 10) -> InferenceResult:
        """
        Fit variational approximation.

        Uses mean-field Gaussian approximation.
        """
        # Variational parameters
        mu = self.prior_mean.copy()
        log_sigma = np.log(self.prior_std)

        learning_rate = 0.01

        for iteration in range(n_iterations):
            # Sample from variational distribution
            eps = np.random.randn(n_samples, self.n_params)
            sigma = np.exp(log_sigma)
            samples = mu + eps * sigma

            # Estimate ELBO gradient
            grad_mu = np.zeros(self.n_params)
            grad_log_sigma = np.zeros(self.n_params)

            for sample, e in zip(samples, eps):
                log_p = self.log_posterior(sample, data, model, noise_std)

                # Score function estimator
                grad_mu += log_p * e / sigma
                grad_log_sigma += log_p * (e ** 2 - 1)

            grad_mu /= n_samples
            grad_log_sigma /= n_samples

            # Add entropy gradient (for variational distribution)
            grad_log_sigma += 1  # d/d(log_sigma) of log(sigma)

            # Update
            mu += learning_rate * grad_mu
            log_sigma += learning_rate * 0.1 * grad_log_sigma

        sigma = np.exp(log_sigma)

        # Generate samples from fitted distribution
        samples = mu + np.random.randn(1000, self.n_params) * sigma

        return InferenceResult(
            samples=samples,
            map_estimate=mu,
            posterior_mean=mu,
            posterior_std=sigma
        )


class EnsembleKalman(BayesianInference):
    """
    Ensemble Kalman Inversion.

    Efficient for high-dimensional inverse problems.
    """

    def __init__(self, prior_mean: np.ndarray, prior_std: np.ndarray,
                 n_ensemble: int = 100):
        super().__init__(prior_mean, prior_std)
        self.n_ensemble = n_ensemble

    def infer(self, data: np.ndarray, model: Callable,
              noise_std: float, n_iterations: int = 10) -> InferenceResult:
        """
        Run Ensemble Kalman Inversion.
        """
        # Initialize ensemble from prior
        ensemble = self.prior_mean + np.random.randn(self.n_ensemble, self.n_params) * self.prior_std

        for iteration in range(n_iterations):
            # Forward model evaluation
            predictions = np.array([model(e) for e in ensemble])
            pred_dim = predictions.shape[1] if len(predictions.shape) > 1 else 1
            predictions = predictions.reshape(self.n_ensemble, pred_dim)

            # Compute covariances
            ensemble_mean = np.mean(ensemble, axis=0)
            pred_mean = np.mean(predictions, axis=0)

            # Cross-covariance Cov(theta, G(theta))
            C_theta_G = np.zeros((self.n_params, pred_dim))
            for i in range(self.n_ensemble):
                C_theta_G += np.outer(ensemble[i] - ensemble_mean,
                                      predictions[i] - pred_mean)
            C_theta_G /= (self.n_ensemble - 1)

            # Prediction covariance + noise
            C_G = np.cov(predictions.T) if pred_dim > 1 else np.var(predictions)
            C_G = np.atleast_2d(C_G) + noise_std ** 2 * np.eye(pred_dim)

            # Kalman gain
            K = C_theta_G @ np.linalg.inv(C_G)

            # Update ensemble
            for i in range(self.n_ensemble):
                noise = np.random.randn(pred_dim) * noise_std
                innovation = data.flatten() - predictions[i] + noise
                ensemble[i] += K @ innovation

        return InferenceResult(
            samples=ensemble,
            map_estimate=np.mean(ensemble, axis=0),
            posterior_mean=np.mean(ensemble, axis=0),
            posterior_std=np.std(ensemble, axis=0)
        )
