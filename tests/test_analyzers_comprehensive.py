# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import unicode_literals

import unittest
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result, BoundingBox
from panobbgo.lib.constraints import DefaultConstraintHandler

class TestAnalyzersComprehensive(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint

        # Use a higher dimensional problem for better coverage
        self.problem = RosenbrockConstraint(3)
        self.strategy = self.init_strategy()
        # Ensure strategy has a constraint handler
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    # --- Best Analyzer Tests ---

    def test_best_analyzer_init(self):
        """Test Best analyzer initialization."""
        from panobbgo.analyzers.best import Best

        best = Best(self.strategy)
        self.assertIsNone(best._min)
        self.assertIsNone(best._cv)
        self.assertIsNone(best._pareto)
        self.assertEqual(best._pareto_front, [])
        self.assertEqual(best.pareto_front, [])
        self.assertIsNone(best.best)
        self.assertIsNone(best.cv)
        self.assertIsNone(best.min)

    def test_best_on_new_results_single_feasible(self):
        """Test Best analyzer processing a single feasible result."""
        from panobbgo.analyzers.best import Best

        best = Best(self.strategy)
        x = np.array([1.0, 1.0, 1.0])
        p = Point(x, "test")
        r = Result(p, 5.0, cv_vec=None)  # Feasible

        best.on_new_results([r])

        self.assertIsNotNone(best.min)
        self.assertEqual(best.min.fx, 5.0)

        # Verify Result behavior
        self.assertEqual(r.cv, 0.0)

        # best.cv tracks the result with minimal constraint violation.
        # Since r.cv is 0.0 (minimal possible), best.cv should track it.
        self.assertIsNotNone(best.cv)
        self.assertEqual(best.cv.fx, 5.0)
        self.assertEqual(best.cv.cv, 0.0)

        self.assertIsNotNone(best.best)
        self.assertEqual(best.best.fx, 5.0)

        self.assertEqual(len(best.pareto_front), 1)

    def test_best_on_new_results_infeasible_better_obj(self):
        """Test Best analyzer with an infeasible point having better objective."""
        from panobbgo.analyzers.best import Best

        best = Best(self.strategy)

        # Feasible point
        r1 = Result(Point(np.array([1.0, 1.0, 1.0]), "t1"), 10.0, cv_vec=None)
        best.on_new_results([r1])

        # Infeasible point with better objective
        r2 = Result(Point(np.array([2.0, 2.0, 2.0]), "t2"), 1.0, cv_vec=np.array([0.1]))
        best.on_new_results([r2])

        # min (objective only) should track r2
        self.assertEqual(best.min.fx, 1.0)

        # cv (constraint violation) should track r1 (cv=0 < cv=0.1)
        self.assertEqual(best.cv.fx, 10.0)
        self.assertEqual(best.cv.cv, 0.0)

        # best (pareto/decision) should still be r1 because r2 is infeasible
        self.assertEqual(best.best.fx, 10.0)

        # Pareto front might contain both depending on logic
        # r1: fx=10, cv=0
        # r2: fx=1, cv=0.1
        # Neither dominates the other -> both should be in front
        self.assertEqual(len(best.pareto_front), 2)

    def test_best_pareto_front_logic(self):
        """Test the pareto front update logic specifically."""
        from panobbgo.analyzers.best import Best

        best = Best(self.strategy)

        # P1: fx=10, cv=0 (Feasible, poor obj)
        r1 = Result(Point(np.array([0., 0., 0.]), "1"), 10.0, cv_vec=np.array([0.0]))
        # P2: fx=5, cv=1 (Infeasible, better obj)
        r2 = Result(Point(np.array([1., 1., 1.]), "2"), 5.0, cv_vec=np.array([1.0]))
        # P3: fx=1, cv=2 (Infeasible, best obj)
        r3 = Result(Point(np.array([2., 2., 2.]), "3"), 1.0, cv_vec=np.array([2.0]))
        # P4: fx=11, cv=0 (Dominated by P1)
        r4 = Result(Point(np.array([3., 3., 3.]), "4"), 11.0, cv_vec=np.array([0.0]))
        # P5: fx=5, cv=2 (Dominated by P2)
        r5 = Result(Point(np.array([4., 4., 4.]), "5"), 5.0, cv_vec=np.array([2.0]))

        best.on_new_results([r1, r2, r3, r4, r5])

        pf = best.pareto_front

        # Check that dominated points are removed
        # P1, P2, P3 should be in the front (trade-off between fx and cv)
        # P4 is dominated by P1 (worse fx, same cv)
        # P5 is dominated by P2 (same fx, worse cv)

        # Verify that r1, r2, r3 are present
        pf_points = [(r.fx, r.cv) for r in pf]
        self.assertIn((10.0, 0.0), pf_points) # P1
        self.assertIn((5.0, 1.0), pf_points)  # P2
        self.assertIn((1.0, 2.0), pf_points)  # P3

        # Verify that r4, r5 are NOT present
        self.assertNotIn((11.0, 0.0), pf_points) # P4
        self.assertNotIn((5.0, 2.0), pf_points)  # P5

        self.assertEqual(len(pf), 3)

    def test_best_empty_results(self):
        """Test with empty result list."""
        from panobbgo.analyzers.best import Best
        best = Best(self.strategy)
        best.on_new_results([])
        self.assertIsNone(best.best)

    # --- Grid Analyzer Tests ---

    def test_grid_init(self):
        """Test Grid analyzer initialization."""
        from panobbgo.analyzers.grid import Grid

        grid = Grid(self.strategy)
        # __start__ needs to be called manually or by strategy
        grid.__start__()

        self.assertIsNotNone(grid._grid)
        self.assertIsInstance(grid._grid, dict)
        self.assertEqual(grid._grid_div, 5.0)

        ranges = self.problem.ranges
        expected_lengths = ranges / 5.0
        np.testing.assert_array_equal(grid._grid_lengths, expected_lengths)

    def test_grid_add_and_retrieve(self):
        """Test adding points to grid and retrieving them."""
        from panobbgo.analyzers.grid import Grid

        grid = Grid(self.strategy)
        grid.__start__()

        # Point 1
        x1 = np.array([0.1, 0.1, 0.1])
        r1 = Result(Point(x1, "t1"), 1.0)

        # Point 2 close to Point 1
        x2 = np.array([0.15, 0.15, 0.15])
        r2 = Result(Point(x2, "t2"), 1.1)

        # Point 3 far away
        x3 = np.array([1.9, 1.9, 1.9])
        r3 = Result(Point(x3, "t3"), 2.0)

        grid.on_new_results([r1, r2, r3])

        # Check if they landed in correct boxes
        # _grid_mapping uses floor(x / l) * l

        # Map x1
        key1 = grid._grid_mapping(x1)
        self.assertIn(key1, grid._grid)
        self.assertEqual(len(grid._grid[key1]), 2) # r1 and r2 should be here
        self.assertIn(r1, grid._grid[key1])
        self.assertIn(r2, grid._grid[key1])

        # Map x3
        key3 = grid._grid_mapping(x3)
        self.assertIn(key3, grid._grid)
        self.assertEqual(len(grid._grid[key3]), 1)
        self.assertIn(r3, grid._grid[key3])

        # Test in_same_grid
        points_near_p1 = grid.in_same_grid(Point(x1, "q"))
        self.assertEqual(len(points_near_p1), 2)

        points_near_p3 = grid.in_same_grid(Point(x3, "q"))
        self.assertEqual(len(points_near_p3), 1)

    def test_grid_boundary_handling(self):
        """Test points exactly on grid boundaries."""
        from panobbgo.analyzers.grid import Grid

        grid = Grid(self.strategy)
        grid.__start__()

        l = grid._grid_lengths[0]

        # Point exactly on boundary
        x = np.array([l, l, l])
        r = Result(Point(x, "boundary"), 1.0)

        grid.on_new_results([r])

        key = grid._grid_mapping(x)
        self.assertIn(key, grid._grid)

        # Point slightly below
        x_below = np.array([l - 1e-10, l - 1e-10, l - 1e-10])
        r_below = Result(Point(x_below, "below"), 1.0)
        grid.on_new_results([r_below])

        key_below = grid._grid_mapping(x_below)
        self.assertNotEqual(key, key_below)
