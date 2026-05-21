#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Smoke test for the panobbgo -> IOH adapter.

Wraps a single MA-BBOB instance, runs the default ``Rewarding_Diverse``
strategy at the competition budget (2000 * d), and prints the AOCC.

Usage::

    uv run python scripts/ioh_smoke.py
    uv run python scripts/ioh_smoke.py --dim 5 --instance 0 --max-eval 500
"""

from __future__ import annotations

import argparse
import time

from panobbgo.ioh_runner import IOHTracker, _BudgetExhausted, aocc
from panobbgo.lib.ioh_wrapper import IOHProblem


def build_rewarding_strategy(problem):
    from panobbgo.strategies import StrategyRewarding
    from panobbgo.heuristics import Random, Nearby, NelderMead, Center, Sobol

    strategy = StrategyRewarding(problem, parse_args=False, testing_mode=True)
    strategy.add(Sobol, n=16, scramble=True)
    strategy.add(Random)
    strategy.add(Nearby, radius=0.1, axes="all", new=3)
    strategy.add(Center)
    strategy.add(NelderMead)
    return strategy


def build_random_strategy(problem):
    """Pure random baseline for sanity-checking AOCC magnitudes."""
    import numpy as np

    class _PureRandom:
        def __init__(self, p):
            self.problem = p
            self.config = type("c", (), {"max_eval": 0})()

        def start(self):
            try:
                while True:
                    x = self.problem.box[:, 0] + np.random.rand(self.problem.dim) * self.problem.ranges
                    self.problem.eval(x)
            except _BudgetExhausted:
                return

    return _PureRandom(problem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Override budget; default is 2000*d per MA-BBOB anytime rules.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run a pure random search instead of the Rewarding strategy.",
    )
    args = parser.parse_args()

    budget = args.max_eval if args.max_eval is not None else 2000 * args.dim

    wrapped = IOHProblem(kind="MA-BBOB", instance=args.instance, dim=args.dim)
    try:
        print(f"Problem: {wrapped}  f_opt={wrapped.optimum_y:.6f}  budget={budget}")

        tracker = IOHTracker(wrapped, budget=budget)
        factory = build_random_strategy if args.baseline else build_rewarding_strategy

        t0 = time.time()
        try:
            strategy = factory(wrapped)
            if hasattr(strategy, "config"):
                if hasattr(strategy.config, "max_eval"):
                    strategy.config.max_eval = budget
                if hasattr(strategy.config, "stop_on_convergence"):
                    strategy.config.stop_on_convergence = False
            try:
                strategy.start()
            except _BudgetExhausted:
                pass
        finally:
            tracker.restore()
        dt = time.time() - t0

        score = aocc(tracker.best_so_far, f_opt=wrapped.optimum_y, budget=budget)
        print(
            f"n_evals={tracker.n_evals}/{budget}  best_fx={tracker.best_fx:.6f}  "
            f"precision={tracker.best_fx - wrapped.optimum_y:.3e}  AOCC={score:.4f}  "
            f"elapsed={dt:.1f}s"
        )
        return 0
    finally:
        wrapped.close()


if __name__ == "__main__":
    raise SystemExit(main())
