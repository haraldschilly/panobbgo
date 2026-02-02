# -*- coding: utf8 -*-
import unittest
from unittest.mock import MagicMock
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.strategies.thompson import StrategyThompsonSampling
from panobbgo.heuristics import Center, Random
from panobbgo.lib import Point, Result, Problem

class TestStrategyThompson(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy = StrategyThompsonSampling(self.problem, max_eval=100)
        # self.strategy.evaluators is a property, cannot set it.
        # It relies on self.strategy.pending.
        self.strategy.pending = {} # No pending tasks
        self.strategy.jobs_per_client = 1

    def test_init_heuristics(self):
        """Test that heuristics are initialized with Beta parameters"""
        self.strategy.add(Center)
        self.strategy.add(Random)

        # Manually trigger addition to internal list as start() would do
        h_center = Center(self.strategy)
        h_random = Random(self.strategy)
        self.strategy.add_heuristic(h_center)
        self.strategy.add_heuristic(h_random)

        self.assertTrue(hasattr(h_center, "ts_alpha"))
        self.assertEqual(h_center.ts_alpha, 1.0)
        self.assertEqual(h_center.ts_beta, 1.0)

    def test_reward_update(self):
        """Test that on_new_best updates Beta parameters correctly"""
        h = Center(self.strategy)
        self.strategy.add_heuristic(h)

        # Initial state
        self.assertEqual(h.ts_alpha, 1.0)
        self.assertEqual(h.ts_beta, 1.0)

        # Simulate improvement
        # First best (baseline)
        best1 = Result(Point(np.zeros(2), h.name), 10.0)
        self.strategy.on_new_best(best1)

        # First point reward is 1.0
        # alpha -> 1 + 1 = 2
        # beta -> 1 + 0 = 1
        self.assertEqual(h.ts_alpha, 2.0)
        self.assertEqual(h.ts_beta, 1.0)

        # Second best (small improvement)
        # Using DefaultConstraintHandler: improvement = max(0, old.fx - new.fx)
        # improvement = 10.0 - 9.0 = 1.0
        # reward = 1 - exp(-1.0) = 1 - 0.3678 = 0.632
        best2 = Result(Point(np.zeros(2), h.name), 9.0)
        self.strategy.on_new_best(best2)

        # alpha -> 2 + 0.632 = 2.632
        # beta -> 1 + (1 - 0.632) = 1.368
        self.assertAlmostEqual(h.ts_alpha, 2.63212, places=4)
        self.assertAlmostEqual(h.ts_beta, 1.36788, places=4)

    def test_execution_selection(self):
        """Test that execute() selects heuristics"""
        h1 = Center(self.strategy)
        h2 = Random(self.strategy)

        # Mock h1 to be very good (high alpha), h2 to be bad (high beta)
        h1.ts_alpha = 100.0
        h1.ts_beta = 1.0
        h2.ts_alpha = 1.0
        h2.ts_beta = 100.0

        # Mock get_points
        h1.get_points = MagicMock(return_value=[Point(np.zeros(2), h1.name)])
        h2.get_points = MagicMock(return_value=[Point(np.zeros(2), h2.name)])

        self.strategy.add_heuristic(h1)
        self.strategy.add_heuristic(h2)

        # Execute should prefer h1 (higher sampled theta)
        points = self.strategy.execute()

        self.assertTrue(len(points) > 0)
        self.assertEqual(points[0].who, h1.name)
        # Verify h1 was called
        h1.get_points.assert_called()
