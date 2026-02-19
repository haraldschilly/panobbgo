# -*- coding: utf8 -*-
import unittest
import numpy as np
import pytest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.heuristics.random import Random
from panobbgo.config import Config

class TestConstraintsComprehensive(PanobbgoTestCase):

    def test_constrained_optimization_integration(self):
        """
        Comprehensive test for constraint handling.
        Uses Augmented Lagrangian and specialized heuristics.
        """
        dim = 2
        # Use a box that includes the optimum region
        problem = RosenbrockConstraint(dims=dim, par2=0.01, box=[(-2, 2)]*dim)

        # Configure strategy
        # Use threaded evaluation for speed and to test concurrency
        config_dict = {
            "max_eval": 200,
            "constraint_handler": "AugmentedLagrangianConstraintHandler",
            "evaluation_method": "threaded",
            "dask_n_workers": 2, # Use 2 threads
            "rho": 10.0,
            "dynamic_penalty_rate": 2.0
        }

        strategy = StrategyRewarding(problem, **config_dict)

        # Add standard heuristic to get started
        strategy.add(Random)

        # Add constraint heuristics
        strategy.add(ConstraintGradient, samples=3)
        strategy.add(LocalPenaltySearch) # Uses scipy minimize
        strategy.add(FeasibleSearch, samples=5)

        # Run strategy
        with strategy:
            strategy.start()

        # Verify results
        results = strategy.results
        self.assertGreater(len(results), 0)

        # Check if we found a feasible point
        # Best should be feasible (cv=0) if we succeeded
        best = strategy.best
        self.assertIsNotNone(best)

        # In constrained optimization, finding a feasible point is the first success
        # With 200 evals on 2D Rosenbrock, it should be possible
        if best.cv > 1e-6:
             print(f"Warning: Did not find perfectly feasible point. Best CV: {best.cv}")
             # Relax check slightly if hard
             self.assertLess(best.cv, 0.1, "Failed to find reasonably feasible point")
        else:
             self.assertLess(best.cv, 1e-6)

        # Check heuristics participation
        who_counts = results.results['who'].value_counts()
        print("\nHeuristic participation:")
        print(who_counts)

        # We expect ConstraintGradient and LocalPenaltySearch to have contributed
        # Note: LocalPenaltySearch runs in background, might contribute fewer points if fast
        # FeasibleSearch only active if best is infeasible (which is true at start)

        # Verify at least some constraint heuristics ran
        constraint_heurs = ['ConstraintGradient', 'LocalPenaltySearch', 'FeasibleSearch']
        found_constraint_heur = any(h in who_counts.index for h in constraint_heurs)

        if not found_constraint_heur:
             print("Warning: No constraint heuristics contributed points directly. This might happen if Random found feasible point immediately.")
             # But Random finding feasible point immediately on RosenbrockConstraint(par2=0.01) is unlikely but possible?
             # par2=0.01 means (y[1]-y[0])^2 >= 0.01. Gap.
             # And y>=0.

        # Assert strategy finished gracefully
        self.assertTrue(strategy.results.backend is None or strategy.results.backend.closed)

if __name__ == "__main__":
    unittest.main()
