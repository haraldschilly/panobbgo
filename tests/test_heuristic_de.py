# -*- coding: utf8 -*-
import unittest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.heuristics.differential_evolution import DifferentialEvolution
from panobbgo.lib import Result, Point
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib.constraints import DefaultConstraintHandler
import numpy as np
from queue import Queue

class MockConfig:
    def get_logger(self, name):
        return MockLogger()
    @property
    def capacity(self):
        return 100

class MockLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass

class MockStrategy:
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.results = None
        self.eventbus = None
        self.best = None
        self.constraint_handler = DefaultConstraintHandler(self)

class TestDifferentialEvolution(PanobbgoTestCase):

    def test_de_initialization(self):
        problem = Rosenbrock(dim=2)
        strategy = MockStrategy(problem)
        de = DifferentialEvolution(strategy, NP=10)

        de.on_start()

        # Check initial population generation
        # It should emit NP points
        points = de.get_points(limit=100)
        self.assertEqual(len(points), 10)

        # Pending trials should be populated
        self.assertEqual(len(de.pending_trials), 10)

    def test_de_population_update(self):
        problem = Rosenbrock(dim=2)
        strategy = MockStrategy(problem)
        de = DifferentialEvolution(strategy, NP=4)
        de.on_start()

        # Clear output
        de.get_points(limit=100)

        # Simulate results for initialization
        # Create results matching pending trials
        results = []
        for req_id in list(de.pending_trials.keys()):
            # Create dummy result with correct ID
            x = np.array([0., 0.])
            who = f"{de.name}:{req_id}"
            p = Point(x, who)
            r = Result(p, 100.0)
            results.append(r)

        de.on_new_results(results)

        # Population should be full
        self.assertEqual(de.pop_size, 4)
        for i in range(4):
            self.assertIsNotNone(de.population[i])

        # It should have emitted new trials now that population is full
        new_points = de.get_points(limit=100)
        self.assertGreater(len(new_points), 0)

    def test_de_evolution(self):
        problem = Rosenbrock(dim=2)
        strategy = MockStrategy(problem)
        de = DifferentialEvolution(strategy, NP=4)

        # Initialize population manually to skip random part
        for i in range(4):
            x = np.ones(2) * i # [0,0], [1,1], [2,2], [3,3]
            p = Point(x, de.name)
            r = Result(p, float(i*10))
            de.population[i] = r
        de.pop_size = 4

        # Force pending trial for index 0
        target_idx = 0
        req_id = "test-id-123"
        de.pending_trials[req_id] = target_idx

        # Incoming result for trial, better than target (0.0 vs 0.0? No target is 0.0)
        # Target: [0,0], fx=0.0
        # Trial: [0.5, 0.5], fx=-10.0 (Better)
        who = f"{de.name}:{req_id}"
        p_trial = Point(np.array([0.5, 0.5]), who)
        r_trial = Result(p_trial, -10.0)

        de.on_new_results([r_trial])

        # Population[0] should be updated
        self.assertEqual(de.population[0].fx, -10.0)

        # Should have emitted new trial
        new_points = de.get_points(limit=1)
        self.assertEqual(len(new_points), 1)

if __name__ == "__main__":
    unittest.main()
