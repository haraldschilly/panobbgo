#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Bandit Strategy Comparison Benchmark
====================================

This script compares the performance of different bandit strategies on a set of
benchmark problems. It generates convergence plots and logs results.

Strategies compared:
- StrategyRoundRobin (Baseline)
- StrategyRewarding (Softmax Bandit)
- StrategyUCB (Upper Confidence Bound)
- StrategyThompsonSampling (Beta-Bernoulli Bandit)
- StrategyLinUCB (Contextual Bandit)

Usage:
    uv run python benchmarks/run_bandit_comparison.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Type, Dict, Any, Tuple

from panobbgo.lib.classic import Rosenbrock, Rastrigin, Griewank, RosenbrockConstraint
from panobbgo.heuristics import Random, LatinHypercube, NelderMead, LBFGSB, ConstraintGradient
from panobbgo.strategies.round_robin import StrategyRoundRobin
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.strategies.thompson import StrategyThompsonSampling
from panobbgo.strategies.contextual import StrategyLinUCB

# Configuration
OUTPUT_DIR = "benchmarks/results_bandits"
EVALUATIONS = 500  # Number of evaluations per run
REPETITIONS = 5  # Number of repetitions per problem/strategy pair
SEED = 42

PROBLEMS = [
    (Rosenbrock, {"dims": 5}, "Rosenbrock 5D"),
    (Rastrigin, {"dims": 5}, "Rastrigin 5D"),
    (Griewank, {"dims": 5}, "Griewank 5D"),
    (RosenbrockConstraint, {"dims": 2}, "RosenbrockConstraint 2D"),
]

STRATEGIES = [
    (StrategyRoundRobin, "RoundRobin"),
    (StrategyRewarding, "Rewarding"),
    (StrategyUCB, "UCB"),
    (StrategyThompsonSampling, "Thompson"),
    (StrategyLinUCB, "LinUCB"),
]


def setup_heuristics(strategy, problem_type):
    """Adds a standard set of heuristics to the strategy."""
    # Basic exploration
    strategy.add(Random)
    strategy.add(LatinHypercube, div=5)

    # Local search
    strategy.add(NelderMead)
    strategy.add(LBFGSB)

    # Constraint specific (only if constrained)
    if issubclass(problem_type, RosenbrockConstraint):  # Simple check
        strategy.add(ConstraintGradient)


def run_experiment(problem_class, problem_kwargs, problem_name, strategy_class, strategy_name, repetition):
    """Runs a single experiment."""

    # Initialize problem
    problem = problem_class(**problem_kwargs)

    # Initialize strategy
    # Use threaded evaluation for speed and stability in benchmark script
    strategy = strategy_class(problem, max_eval=EVALUATIONS, evaluation_method="threaded")

    # Add heuristics
    setup_heuristics(strategy, problem_class)

    # Run optimization
    start_time = time.time()
    try:
        strategy.start()
    except Exception as e:
        print(f"Error running {strategy_name} on {problem_name} (rep {repetition}): {e}")
        return None

    duration = time.time() - start_time

    # Collect results
    # We want the convergence trace (best fx vs evaluations)
    history = strategy.results.get_history()
    fx = history["fx"]
    cv = history["cv"]

    # Calculate best-so-far trace
    best_fx_trace = []
    current_best = float("inf")

    # Simple best-so-far logic (ignoring constraints for simple plot, or handling them)
    # For constrained problems, we should prioritize feasibility

    for i in range(len(fx)):
        val = fx[i]
        viol = cv[i] if i < len(cv) else 0.0

        # Penalize violation for plotting
        penalty = 1e9 if viol > 1e-9 else 0.0
        penalized_val = val + penalty

        if penalized_val < current_best:
            current_best = penalized_val

        best_fx_trace.append(current_best)

    return {
        "problem": problem_name,
        "strategy": strategy_name,
        "repetition": repetition,
        "evaluations": len(fx),
        "final_best": current_best,
        "duration": duration,
        "trace": best_fx_trace,
    }


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    np.random.seed(SEED)

    all_results = []

    print(f"Starting Bandit Benchmark ({EVALUATIONS} evals, {REPETITIONS} reps)")
    print("-" * 60)

    for prob_cls, prob_kwargs, prob_name in PROBLEMS:
        print(f"\nProblem: {prob_name}")

        # Prepare plot for this problem
        plt.figure(figsize=(10, 6))

        for strat_cls, strat_name in STRATEGIES:
            print(f"  Strategy: {strat_name} ... ", end="", flush=True)

            traces = []
            final_bests = []

            for rep in range(REPETITIONS):
                res = run_experiment(prob_cls, prob_kwargs, prob_name, strat_cls, strat_name, rep)
                if res:
                    all_results.append(res)
                    traces.append(res["trace"])
                    final_bests.append(res["final_best"])

            if traces:
                # Pad traces to same length if necessary
                max_len = max(len(t) for t in traces)
                padded_traces = []
                for t in traces:
                    if len(t) < max_len:
                        t = t + [t[-1]] * (max_len - len(t))
                    padded_traces.append(t)

                avg_trace = np.mean(padded_traces, axis=0)
                plt.plot(avg_trace, label=f"{strat_name} (Mean: {np.mean(final_bests):.4f})")
                print(f"Done. Avg Best: {np.mean(final_bests):.4f}")
            else:
                print("Failed.")

        plt.title(f"Convergence on {prob_name}")
        plt.xlabel("Evaluations")
        plt.ylabel("Best Value (Penalized)")
        plt.yscale("symlog")  # Good for values that might cross zero or have large range
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plot_path = os.path.join(OUTPUT_DIR, f"convergence_{prob_name.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        print(f"  Saved plot to {plot_path}")
        plt.close()

    # Save summary CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "trace"} for r in all_results])
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark complete. Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
