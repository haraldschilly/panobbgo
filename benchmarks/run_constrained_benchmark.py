#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Constrained Optimization Benchmark
==================================

This script compares the performance of different constraint handling methods
on a set of constrained benchmark problems.

Handlers compared:
- DefaultConstraintHandler (Lexicographic)
- PenaltyConstraintHandler (Static Penalty)
- AugmentedLagrangianConstraintHandler (ALM)
- FilterConstraintHandler (Multi-objective Filter)

Usage:
    uv run python benchmarks/run_constrained_benchmark.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Type, Dict, Any, Tuple

from panobbgo.lib.classic import RosenbrockConstraint, Simionescu, MishraBird, PressureVessel
from panobbgo.heuristics import Random, LatinHypercube, NelderMead, ConstraintGradient, LocalPenaltySearch
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    FilterConstraintHandler,
)

# Configuration
OUTPUT_DIR = "benchmarks/results_constrained"
EVALUATIONS = 300  # Number of evaluations per run
REPETITIONS = 5  # Number of repetitions per problem/strategy pair
SEED = 42

PROBLEMS = [
    (RosenbrockConstraint, {"dims": 2}, "RosenbrockConstraint 2D"),
    (Simionescu, {}, "Simionescu"),
    (MishraBird, {}, "MishraBird"),
    (PressureVessel, {}, "PressureVessel"),
]

HANDLERS = [
    ("Default", "DefaultConstraintHandler"),
    ("Penalty", "PenaltyConstraintHandler"),
    ("ALM", "AugmentedLagrangianConstraintHandler"),
    ("Filter", "FilterConstraintHandler"),
]


def setup_heuristics(strategy):
    """Adds a standard set of heuristics to the strategy."""
    strategy.add(Random)
    strategy.add(LatinHypercube, div=5)
    strategy.add(NelderMead)
    # Add constraint-specific heuristics
    strategy.add(ConstraintGradient)
    strategy.add(LocalPenaltySearch)


def run_experiment(problem_class, problem_kwargs, problem_name, handler_name, handler_class_str, repetition):
    """Runs a single experiment."""

    # Initialize problem
    problem = problem_class(**problem_kwargs)

    # Initialize strategy with specific constraint handler
    # We pass handler name as string to config, StrategyBase will instantiate it
    strategy = StrategyRewarding(
        problem, max_eval=EVALUATIONS, evaluation_method="threaded", constraint_handler=handler_class_str
    )

    # Add heuristics
    setup_heuristics(strategy)

    # Run optimization
    start_time = time.time()
    try:
        strategy.start()
    except Exception as e:
        print(f"Error running {handler_name} on {problem_name} (rep {repetition}): {e}")
        return None

    duration = time.time() - start_time

    # Collect results
    # We want the convergence trace (best fx vs evaluations)
    history = strategy.results.get_history()
    fx = history["fx"]
    cv = history["cv"]  # This is the norm of positive violations

    # Calculate best-feasible-so-far trace
    best_feasible_trace = []
    current_best_feasible = float("inf")

    # Also track best-infeasible (min cv) if no feasible point found yet
    min_cv_trace = []
    current_min_cv = float("inf")

    final_cv = float("inf")
    final_fx = float("inf")

    for i in range(len(fx)):
        val = fx[i]
        viol = cv[i] if i < len(cv) else 0.0

        if viol <= 1e-9:
            if val < current_best_feasible:
                current_best_feasible = val

        if viol < current_min_cv:
            current_min_cv = viol

        # For plotting: use best feasible if available, else use a large value + min_cv
        if current_best_feasible < float("inf"):
            best_feasible_trace.append(current_best_feasible)
        else:
            # If no feasible point, plot something indicating violation magnitude
            # Using a large offset + CV to show progress towards feasibility
            best_feasible_trace.append(1e5 + current_min_cv)

    if current_best_feasible < float("inf"):
        final_fx = current_best_feasible
        final_cv = 0.0
    else:
        # If didn't find feasible, report best CV found
        # Find index of min CV
        if len(cv) > 0:
            idx = np.argmin(cv)
            final_cv = cv[idx]
            final_fx = fx[idx]
        else:
            final_cv = float("inf")
            final_fx = float("inf")

    return {
        "problem": problem_name,
        "handler": handler_name,
        "repetition": repetition,
        "evaluations": len(fx),
        "final_fx": final_fx,
        "final_cv": final_cv,
        "duration": duration,
        "trace": best_feasible_trace,
    }


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    np.random.seed(SEED)

    all_results = []

    print(f"Starting Constrained Benchmark ({EVALUATIONS} evals, {REPETITIONS} reps)")
    print("-" * 60)

    for prob_cls, prob_kwargs, prob_name in PROBLEMS:
        print(f"\nProblem: {prob_name}")

        # Prepare plot for this problem
        plt.figure(figsize=(10, 6))

        for handler_name, handler_class_str in HANDLERS:
            print(f"  Handler: {handler_name} ... ", end="", flush=True)

            traces = []
            final_fxs = []
            final_cvs = []

            for rep in range(REPETITIONS):
                res = run_experiment(prob_cls, prob_kwargs, prob_name, handler_name, handler_class_str, rep)
                if res:
                    all_results.append(res)
                    traces.append(res["trace"])
                    final_fxs.append(res["final_fx"])
                    final_cvs.append(res["final_cv"])

            if traces:
                # Pad traces to same length if necessary
                max_len = max(len(t) for t in traces)
                padded_traces = []
                for t in traces:
                    if len(t) < max_len:
                        t = t + [t[-1]] * (max_len - len(t))
                    padded_traces.append(t)

                avg_trace = np.mean(padded_traces, axis=0)

                # Check feasibility success rate
                success_count = sum(1 for cv in final_cvs if cv <= 1e-9)
                success_rate = (success_count / REPETITIONS) * 100

                label = f"{handler_name} (Succ: {success_rate:.0f}%, AvgFx: {np.mean([f for f in final_fxs if f < 1e5]):.2f})"
                plt.plot(avg_trace, label=label)
                print(f"Done. Success: {success_rate:.0f}%")
            else:
                print("Failed.")

        plt.title(f"Performance on {prob_name}")
        plt.xlabel("Evaluations")
        plt.ylabel("Best Feasible Value (1e5+CV if infeasible)")
        plt.yscale("symlog")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plot_path = os.path.join(OUTPUT_DIR, f"constrained_{prob_name.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        print(f"  Saved plot to {plot_path}")
        plt.close()

    # Save summary CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "trace"} for r in all_results])
    csv_path = os.path.join(OUTPUT_DIR, "constrained_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark complete. Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
