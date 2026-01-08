"""
Benchmark Strategy Configurations for Panobbgo

This module defines different strategy configurations for benchmarking,
including various combinations of heuristics and bandit parameters.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from panobbgo.strategies import StrategyRoundRobin, StrategyRewarding
from panobbgo.heuristics import (
    Random, Nearby, Zero, LatinHypercube, Extremal, NelderMead, LBFGSB,
    QuadraticWlsModel, Center, WeightedAverage, GaussianProcessHeuristic
)


@dataclass
class BenchmarkStrategy:
    """Represents a strategy configuration for benchmarking."""
    name: str
    strategy_class: type
    heuristics: List[type]
    config_overrides: Dict[str, Any]
    description: str

    def create_strategy(self, problem):
        """Create a strategy instance with this configuration."""
        strategy = self.strategy_class(problem, parse_args=False)

        # Apply configuration overrides
        for key, value in self.config_overrides.items():
            if hasattr(strategy.config, key):
                setattr(strategy.config, key, value)

        # Add heuristics
        for heuristic_class in self.heuristics:
            strategy.add(heuristic_class)

        return strategy


# Comprehensive set of benchmark strategy configurations
BENCHMARK_STRATEGIES = [
    # Baseline: Round-robin with single heuristic
    BenchmarkStrategy(
        name="round_robin_random",
        strategy_class=StrategyRoundRobin,
        heuristics=[Random],
        config_overrides={},
        description="Round-robin with single Random heuristic (baseline)"
    ),

    # Single heuristic variants
    BenchmarkStrategy(
        name="round_robin_nearby",
        strategy_class=StrategyRoundRobin,
        heuristics=[Nearby],
        config_overrides={},
        description="Round-robin with single Nearby heuristic"
    ),

    BenchmarkStrategy(
        name="round_robin_latin",
        strategy_class=StrategyRoundRobin,
        heuristics=[LatinHypercube],
        config_overrides={},
        description="Round-robin with single LatinHypercube heuristic"
    ),

    BenchmarkStrategy(
        name="round_robin_nelder",
        strategy_class=StrategyRoundRobin,
        heuristics=[NelderMead],
        config_overrides={},
        description="Round-robin with single NelderMead heuristic"
    ),

    # Multi-heuristic round-robin
    BenchmarkStrategy(
        name="round_robin_multi",
        strategy_class=StrategyRoundRobin,
        heuristics=[Random, Nearby, Extremal, Center],
        config_overrides={},
        description="Round-robin with multiple basic heuristics"
    ),

    # Bandit strategies with different configurations
    BenchmarkStrategy(
        name="bandit_basic",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, Extremal, Center],
        config_overrides={},
        description="Basic bandit with 4 heuristics (default parameters)"
    ),

    BenchmarkStrategy(
        name="bandit_expensive",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, Extremal, NelderMead, LBFGSB],
        config_overrides={},
        description="Bandit with expensive heuristics (NelderMead, LBFGSB)"
    ),

    BenchmarkStrategy(
        name="bandit_model_based",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, QuadraticWlsModel, GaussianProcessHeuristic],
        config_overrides={},
        description="Bandit with model-based heuristics"
    ),

    # Bandit with different discount factors
    BenchmarkStrategy(
        name="bandit_high_discount",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, Extremal, Center],
        config_overrides={'discount': 0.99},
        description="Bandit with high discount factor (slower forgetting)"
    ),

    BenchmarkStrategy(
        name="bandit_low_discount",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, Extremal, Center],
        config_overrides={'discount': 0.8},
        description="Bandit with low discount factor (faster forgetting)"
    ),

    # Large heuristic pools
    BenchmarkStrategy(
        name="bandit_large_pool",
        strategy_class=StrategyRewarding,
        heuristics=[
            Random, Nearby, Zero, LatinHypercube, Extremal,
            Center, WeightedAverage, NelderMead, QuadraticWlsModel
        ],
        config_overrides={},
        description="Bandit with large pool of 9 diverse heuristics"
    ),

    # Specialized configurations
    BenchmarkStrategy(
        name="bandit_unimodal",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Nearby, NelderMead, LBFGSB, QuadraticWlsModel],
        config_overrides={},
        description="Bandit optimized for unimodal problems"
    ),

    BenchmarkStrategy(
        name="bandit_multimodal",
        strategy_class=StrategyRewarding,
        heuristics=[Random, Extremal, LatinHypercube, WeightedAverage],
        config_overrides={},
        description="Bandit optimized for multimodal problems"
    )
]


def get_benchmark_strategies(strategy_type: str = None) -> List[BenchmarkStrategy]:
    """Get filtered list of benchmark strategies."""
    strategies = BENCHMARK_STRATEGIES

    if strategy_type == "round_robin":
        strategies = [s for s in strategies if "round_robin" in s.name]
    elif strategy_type == "bandit":
        strategies = [s for s in strategies if "bandit" in s.name]

    return strategies


def create_strategy_comparison_matrix() -> Dict[str, List[str]]:
    """Create a matrix of strategies to compare across different problem types."""
    return {
        "easy_problems": ["round_robin_random", "bandit_basic", "bandit_large_pool"],
        "medium_problems": ["round_robin_multi", "bandit_basic", "bandit_expensive"],
        "hard_problems": ["bandit_large_pool", "bandit_model_based", "bandit_unimodal", "bandit_multimodal"],
        "high_dimensional": ["round_robin_random", "bandit_basic", "bandit_high_discount"],
        "shifted_problems": ["round_robin_nearby", "bandit_basic", "bandit_low_discount"]
    }