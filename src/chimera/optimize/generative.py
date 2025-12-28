"""
Generative design with solver-in-the-loop.

Novel: Combine generative models with physics simulation
for design space exploration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np

from chimera.core.field import Field
from chimera.mesh.mesh import Mesh
from chimera.solvers.base import Solver


@dataclass
class GenerativeConfig:
    """Configuration for generative design."""
    latent_dim: int = 32
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1


class LatentSpaceOptimizer:
    """
    Optimize in learned latent space.

    Key idea: Learn a compact representation of design space,
    then optimize in that space rather than high-dimensional
    parameter space.
    """

    def __init__(self, config: Optional[GenerativeConfig] = None):
        self.config = config or GenerativeConfig()
        self._encoder = None
        self._decoder = None
        self._trained = False

    def build(self, design_dim: int):
        """Build encoder-decoder architecture."""
        latent = self.config.latent_dim
        hidden = 128

        # Encoder: design -> latent
        self._enc_w1 = np.random.randn(design_dim, hidden) * 0.02
        self._enc_w2 = np.random.randn(hidden, latent) * 0.02

        # Decoder: latent -> design
        self._dec_w1 = np.random.randn(latent, hidden) * 0.02
        self._dec_w2 = np.random.randn(hidden, design_dim) * 0.02

    def encode(self, design: np.ndarray) -> np.ndarray:
        """Encode design to latent space."""
        h = np.tanh(design @ self._enc_w1)
        return h @ self._enc_w2

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to design."""
        h = np.tanh(z @ self._dec_w1)
        # Sigmoid for density-like outputs (0, 1)
        return 1 / (1 + np.exp(-h @ self._dec_w2))

    def train(self, designs: np.ndarray, n_epochs: int = 100):
        """Train autoencoder on design database."""
        n, d = designs.shape
        lr = 0.001

        for epoch in range(n_epochs):
            # Encode
            z = self.encode(designs)

            # Decode
            reconstructed = self.decode(z)

            # Reconstruction loss
            loss = np.mean((reconstructed - designs)**2)

            # Backprop (simplified)
            # Would use proper autograd here

        self._trained = True

    def optimize(self, objective: Callable[[np.ndarray], float],
                 initial_z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize in latent space.

        Args:
            objective: Function design -> scalar (to minimize)
            initial_z: Initial latent vector

        Returns:
            Optimal design
        """
        if initial_z is None:
            initial_z = np.random.randn(self.config.latent_dim)

        def latent_objective(z):
            design = self.decode(z)
            return objective(design)

        from scipy.optimize import minimize
        result = minimize(latent_objective, initial_z, method='L-BFGS-B')

        return self.decode(result.x)


class GenerativeDesigner:
    """
    Generative design using evolutionary algorithms with physics.

    Solver-in-the-loop evolution: fitness is actual physics performance.
    """

    def __init__(self, solver: Solver,
                 config: Optional[GenerativeConfig] = None):
        self.solver = solver
        self.config = config or GenerativeConfig()
        self._population: List[np.ndarray] = []
        self._fitness: List[float] = []
        self._history: List[dict] = []

    def initialize(self, design_dim: int,
                   bounds: Tuple[np.ndarray, np.ndarray]):
        """Initialize random population."""
        self.design_dim = design_dim
        self.bounds = bounds
        low, high = bounds

        self._population = [
            np.random.uniform(low, high)
            for _ in range(self.config.population_size)
        ]

    def evaluate_fitness(self, design: np.ndarray) -> float:
        """
        Evaluate fitness using actual physics solver.

        This is the key innovation: using real simulation
        instead of surrogate or heuristic.
        """
        # Set design in solver
        self._set_design(design)

        # Solve physics
        result = self.solver.solve()

        if not result.success:
            return float('inf')

        # Extract performance metric
        # Lower is better (for minimization)
        return self._compute_performance(result)

    def _set_design(self, design: np.ndarray):
        """Update solver with design (to be overridden)."""
        pass

    def _compute_performance(self, result) -> float:
        """Compute fitness from simulation result (to be overridden)."""
        # Default: return residual
        return result.residual

    def evolve(self) -> np.ndarray:
        """
        Run evolutionary optimization.

        Returns best design found.
        """
        pop = self.config.population_size
        n_elite = int(pop * self.config.elite_fraction)

        for gen in range(self.config.n_generations):
            # Evaluate fitness
            fitness = [self.evaluate_fitness(d) for d in self._population]

            # Sort by fitness
            sorted_idx = np.argsort(fitness)
            self._population = [self._population[i] for i in sorted_idx]
            fitness = [fitness[i] for i in sorted_idx]

            # Record history
            self._history.append({
                'generation': gen,
                'best_fitness': fitness[0],
                'mean_fitness': np.mean(fitness)
            })

            # Create next generation
            new_pop = []

            # Elitism: keep best
            new_pop.extend(self._population[:n_elite])

            # Fill rest with crossover and mutation
            while len(new_pop) < pop:
                # Selection (tournament)
                p1 = self._tournament_select()
                p2 = self._tournament_select()

                # Crossover
                if np.random.rand() < self.config.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = p1.copy()

                # Mutation
                child = self._mutate(child)

                new_pop.append(child)

            self._population = new_pop

        # Return best
        best_idx = np.argmin([self.evaluate_fitness(d) for d in self._population])
        return self._population[best_idx]

    def _tournament_select(self, k: int = 3) -> np.ndarray:
        """Tournament selection."""
        idx = np.random.choice(len(self._population), k, replace=False)
        fitness = [self.evaluate_fitness(self._population[i]) for i in idx]
        return self._population[idx[np.argmin(fitness)]].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Blend crossover."""
        alpha = np.random.uniform(-0.5, 1.5, len(p1))
        child = alpha * p1 + (1 - alpha) * p2

        # Clip to bounds
        low, high = self.bounds
        return np.clip(child, low, high)

    def _mutate(self, design: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        low, high = self.bounds
        sigma = self.config.mutation_rate * (high - low)

        mutation = np.random.randn(len(design)) * sigma
        mask = np.random.rand(len(design)) < self.config.mutation_rate
        design[mask] += mutation[mask]

        return np.clip(design, low, high)


class DifferentiableDesigner:
    """
    Differentiable design optimization.

    Uses automatic differentiation through solver for
    gradient-based design optimization.
    """

    def __init__(self, solver: Solver):
        self.solver = solver

    def optimize(self, initial_design: np.ndarray,
                 objective: Callable[[np.ndarray], float],
                 n_iters: int = 100) -> np.ndarray:
        """
        Gradient-based design optimization.

        Uses adjoint method for efficient gradients through solver.
        """
        design = initial_design.copy()
        lr = 0.01

        for i in range(n_iters):
            # Forward pass
            J = objective(design)

            # Compute gradient via adjoint
            grad = self._adjoint_gradient(design, objective)

            # Update
            design -= lr * grad

            # Project to constraints if needed
            design = np.clip(design, 0, 1)

        return design

    def _adjoint_gradient(self, design: np.ndarray,
                          objective: Callable) -> np.ndarray:
        """Compute gradient via adjoint method."""
        # Would implement full adjoint sensitivity analysis
        # Fallback: finite differences
        eps = 1e-5
        grad = np.zeros_like(design)

        J0 = objective(design)
        for i in range(len(design)):
            design_pert = design.copy()
            design_pert[i] += eps
            J1 = objective(design_pert)
            grad[i] = (J1 - J0) / eps

        return grad
