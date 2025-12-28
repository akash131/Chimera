"""
Symbolic Regression for equation discovery.

Evolutionary and neural approaches to find symbolic expressions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from copy import deepcopy


@dataclass
class Expression:
    """Symbolic expression tree node."""
    operation: str  # 'const', 'var', 'add', 'mul', 'sin', 'cos', 'exp', 'pow'
    value: Optional[float] = None  # For constants
    var_index: Optional[int] = None  # For variables
    children: List['Expression'] = None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate expression on data."""
        if self.operation == 'const':
            return np.full(len(X), self.value)
        elif self.operation == 'var':
            return X[:, self.var_index]
        elif self.operation == 'add':
            return self.children[0].evaluate(X) + self.children[1].evaluate(X)
        elif self.operation == 'sub':
            return self.children[0].evaluate(X) - self.children[1].evaluate(X)
        elif self.operation == 'mul':
            return self.children[0].evaluate(X) * self.children[1].evaluate(X)
        elif self.operation == 'div':
            denom = self.children[1].evaluate(X)
            return self.children[0].evaluate(X) / (denom + 1e-10 * np.sign(denom + 1e-20))
        elif self.operation == 'sin':
            return np.sin(self.children[0].evaluate(X))
        elif self.operation == 'cos':
            return np.cos(self.children[0].evaluate(X))
        elif self.operation == 'exp':
            val = self.children[0].evaluate(X)
            return np.exp(np.clip(val, -20, 20))
        elif self.operation == 'pow':
            base = self.children[0].evaluate(X)
            return np.abs(base) ** self.value
        elif self.operation == 'neg':
            return -self.children[0].evaluate(X)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def to_string(self) -> str:
        """Convert to string representation."""
        if self.operation == 'const':
            return f"{self.value:.4f}"
        elif self.operation == 'var':
            return f"x{self.var_index}"
        elif self.operation == 'add':
            return f"({self.children[0].to_string()} + {self.children[1].to_string()})"
        elif self.operation == 'sub':
            return f"({self.children[0].to_string()} - {self.children[1].to_string()})"
        elif self.operation == 'mul':
            return f"({self.children[0].to_string()} * {self.children[1].to_string()})"
        elif self.operation == 'div':
            return f"({self.children[0].to_string()} / {self.children[1].to_string()})"
        elif self.operation == 'sin':
            return f"sin({self.children[0].to_string()})"
        elif self.operation == 'cos':
            return f"cos({self.children[0].to_string()})"
        elif self.operation == 'exp':
            return f"exp({self.children[0].to_string()})"
        elif self.operation == 'pow':
            return f"({self.children[0].to_string()})^{self.value}"
        elif self.operation == 'neg':
            return f"-{self.children[0].to_string()}"
        else:
            return "?"

    def complexity(self) -> int:
        """Count number of nodes."""
        if self.children is None:
            return 1
        return 1 + sum(c.complexity() for c in self.children)

    def copy(self) -> 'Expression':
        """Deep copy."""
        return deepcopy(self)


class GeneticProgramming:
    """
    Genetic Programming for symbolic regression.

    Evolves population of expression trees.
    """

    def __init__(self, population_size: int = 100,
                 max_depth: int = 5,
                 n_generations: int = 50,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 5,
                 parsimony_coefficient: float = 0.01):
        """
        Args:
            population_size: Number of individuals
            max_depth: Maximum tree depth
            n_generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Tournament selection size
            parsimony_coefficient: Penalty for complexity
        """
        self.population_size = population_size
        self.max_depth = max_depth
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient

        self._n_vars: int = 0
        self._population: List[Expression] = []
        self._best: Optional[Expression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Expression:
        """
        Find symbolic expression y = f(X).

        Args:
            X: Input data (n_samples, n_features)
            y: Target (n_samples,)

        Returns:
            Best expression found
        """
        self._n_vars = X.shape[1]

        # Initialize population
        self._population = [self._random_tree(self.max_depth) for _ in range(self.population_size)]

        best_fitness = np.inf
        self._best = None

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = []
            for expr in self._population:
                fitness = self._fitness(expr, X, y)
                fitnesses.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    self._best = expr.copy()

            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness:.6f}, "
                      f"Best expr = {self._best.to_string()}")

            # Create new population
            new_population = []

            # Elitism: keep best
            sorted_indices = np.argsort(fitnesses)
            for i in range(2):
                new_population.append(self._population[sorted_indices[i]].copy())

            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(fitnesses)
                parent2 = self._tournament_select(fitnesses)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            self._population = new_population[:self.population_size]

        return self._best

    def _random_tree(self, max_depth: int, current_depth: int = 0) -> Expression:
        """Generate random expression tree."""
        if current_depth >= max_depth or (current_depth > 0 and np.random.random() < 0.3):
            # Terminal
            if np.random.random() < 0.5:
                return Expression('const', value=np.random.uniform(-2, 2))
            else:
                return Expression('var', var_index=np.random.randint(self._n_vars))
        else:
            # Non-terminal
            ops = ['add', 'sub', 'mul', 'sin', 'cos']
            op = np.random.choice(ops)

            if op in ['sin', 'cos']:
                child = self._random_tree(max_depth, current_depth + 1)
                return Expression(op, children=[child])
            else:
                left = self._random_tree(max_depth, current_depth + 1)
                right = self._random_tree(max_depth, current_depth + 1)
                return Expression(op, children=[left, right])

    def _fitness(self, expr: Expression, X: np.ndarray, y: np.ndarray) -> float:
        """Compute fitness (lower is better)."""
        try:
            pred = expr.evaluate(X)
            mse = np.mean((pred - y)**2)

            # Parsimony pressure
            complexity_penalty = self.parsimony_coefficient * expr.complexity()

            return mse + complexity_penalty
        except:
            return np.inf

    def _tournament_select(self, fitnesses: List[float]) -> Expression:
        """Tournament selection."""
        indices = np.random.choice(len(fitnesses), size=self.tournament_size, replace=False)
        best_idx = indices[np.argmin([fitnesses[i] for i in indices])]
        return self._population[best_idx].copy()

    def _crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Expression, Expression]:
        """Subtree crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Find random subtree in each
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)

        if len(nodes1) > 1 and len(nodes2) > 1:
            node1 = nodes1[np.random.randint(1, len(nodes1))]
            node2 = nodes2[np.random.randint(1, len(nodes2))]

            # Swap
            node1.operation, node2.operation = node2.operation, node1.operation
            node1.value, node2.value = node2.value, node1.value
            node1.var_index, node2.var_index = node2.var_index, node1.var_index
            node1.children, node2.children = node2.children, node1.children

        return child1, child2

    def _mutate(self, expr: Expression) -> Expression:
        """Point mutation."""
        nodes = self._get_all_nodes(expr)
        if not nodes:
            return expr

        node = nodes[np.random.randint(len(nodes))]

        mutation_type = np.random.choice(['constant', 'operation', 'subtree'])

        if mutation_type == 'constant' and node.operation == 'const':
            node.value += np.random.normal(0, 0.5)
        elif mutation_type == 'operation' and node.children:
            if len(node.children) == 2:
                node.operation = np.random.choice(['add', 'sub', 'mul'])
            else:
                node.operation = np.random.choice(['sin', 'cos', 'neg'])
        elif mutation_type == 'subtree':
            new_subtree = self._random_tree(2)
            node.operation = new_subtree.operation
            node.value = new_subtree.value
            node.var_index = new_subtree.var_index
            node.children = new_subtree.children

        return expr

    def _get_all_nodes(self, expr: Expression) -> List[Expression]:
        """Get all nodes in tree."""
        nodes = [expr]
        if expr.children:
            for child in expr.children:
                nodes.extend(self._get_all_nodes(child))
        return nodes


class SymbolicRegression:
    """
    High-level symbolic regression interface.

    Combines multiple methods for robust discovery.
    """

    def __init__(self, method: str = "gp",
                 **kwargs):
        """
        Args:
            method: "gp" (genetic programming) or "neural"
            **kwargs: Method-specific arguments
        """
        self.method = method
        self.kwargs = kwargs

        if method == "gp":
            self._model = GeneticProgramming(**kwargs)
        else:
            self._model = NeuralSymbolic(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit symbolic regressor."""
        return self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using discovered expression."""
        if hasattr(self._model, '_best'):
            return self._model._best.evaluate(X)
        else:
            raise ValueError("Model not fitted")


class NeuralSymbolic:
    """
    Neural network-guided symbolic regression.

    Uses neural network to guide symbolic search.
    """

    def __init__(self, hidden_dims: List[int] = None,
                 max_expression_length: int = 20):
        """
        Args:
            hidden_dims: Hidden layer dimensions
            max_expression_length: Maximum tokens in expression
        """
        self.hidden_dims = hidden_dims or [64, 64]
        self.max_expression_length = max_expression_length

        self._vocabulary = ['x0', 'x1', 'x2', '+', '-', '*', '/', 'sin', 'cos', 'const', 'END']
        self._layers: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Discover symbolic expression.

        Uses neural network to predict token sequence.
        """
        n_features = X.shape[1]

        # For now, use simplified approach: enumerate and score candidates
        best_expr = None
        best_score = np.inf

        # Generate candidate expressions
        candidates = self._generate_candidates(n_features)

        for expr_str in candidates:
            try:
                score = self._evaluate_expression(expr_str, X, y)
                if score < best_score:
                    best_score = score
                    best_expr = expr_str
            except:
                continue

        print(f"Best expression: {best_expr} (score: {best_score:.6f})")
        return best_expr

    def _generate_candidates(self, n_features: int) -> List[str]:
        """Generate candidate expressions."""
        candidates = []

        # Linear
        for i in range(n_features):
            candidates.append(f"x{i}")
            candidates.append(f"-x{i}")

        # Polynomials
        for i in range(n_features):
            candidates.append(f"x{i}**2")
            for j in range(i, n_features):
                candidates.append(f"x{i}*x{j}")

        # Trigonometric
        for i in range(n_features):
            candidates.append(f"np.sin(x{i})")
            candidates.append(f"np.cos(x{i})")

        # Combinations
        for i in range(n_features):
            candidates.append(f"x{i} + x{i}**2")
            candidates.append(f"np.sin(x{i}) + np.cos(x{i})")

        return candidates

    def _evaluate_expression(self, expr_str: str, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate expression string."""
        # Create local variables
        local_vars = {f"x{i}": X[:, i] for i in range(X.shape[1])}
        local_vars['np'] = np

        try:
            pred = eval(expr_str, {"__builtins__": {}}, local_vars)
            mse = np.mean((pred - y)**2)
            return mse
        except:
            return np.inf
