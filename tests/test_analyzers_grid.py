# -*- coding: utf8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result
from panobbgo.analyzers.grid import Grid
from panobbgo.lib.classic import RosenbrockConstraint


class TestGridAnalyzer(PanobbgoTestCase):
    def setUp(self):
        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_grid_initialization(self):
        grid_analyzer = Grid(self.strategy)
        grid_analyzer.__start__()

        self.assertEqual(grid_analyzer._grid, {})
        self.assertEqual(grid_analyzer._grid_div, 5.0)
        np.testing.assert_array_equal(grid_analyzer._grid_lengths, self.problem.ranges / 5.0)

    def test_grid_mapping(self):
        grid_analyzer = Grid(self.strategy)
        grid_analyzer.__start__()

        # Test mapping logic
        # For RosenbrockConstraint(2), box is likely default or specified. We can use arbitrary points
        # to test the _grid_mapping logic since we know it uses floor(x / l) * l
        l = grid_analyzer._grid_lengths

        x = np.array([0.0, 0.0])
        expected_mapping = tuple(np.floor(x / l) * l)
        self.assertEqual(grid_analyzer._grid_mapping(x), expected_mapping)

        x2 = np.array([l[0] * 1.5, l[1] * -0.5])
        expected_mapping2 = tuple(np.floor(x2 / l) * l)
        self.assertEqual(grid_analyzer._grid_mapping(x2), expected_mapping2)

    def test_grid_add_and_same_grid(self):
        grid_analyzer = Grid(self.strategy)
        grid_analyzer.__start__()

        l = grid_analyzer._grid_lengths

        # Two points in the same grid cell
        x1 = np.array([l[0] * 1.1, l[1] * 1.1])
        x2 = np.array([l[0] * 1.9, l[1] * 1.9])

        # A point in a different grid cell
        x3 = np.array([l[0] * 2.5, l[1] * 2.5])

        p1 = Point(x1, "test")
        p2 = Point(x2, "test")
        p3 = Point(x3, "test")

        r1 = Result(p1, 1.0)
        r2 = Result(p2, 2.0)
        r3 = Result(p3, 3.0)

        grid_analyzer.on_new_results([r1, r2, r3])

        # Verify grouping
        key12 = grid_analyzer._grid_mapping(x1)
        key3 = grid_analyzer._grid_mapping(x3)

        self.assertEqual(len(grid_analyzer._grid), 2)
        self.assertEqual(grid_analyzer._grid[key12], [r1, r2])
        self.assertEqual(grid_analyzer._grid[key3], [r3])

        # Verify in_same_grid
        self.assertEqual(grid_analyzer.in_same_grid(p1), [r1, r2])
        self.assertEqual(grid_analyzer.in_same_grid(p2), [r1, r2])
        self.assertEqual(grid_analyzer.in_same_grid(p3), [r3])

    def test_grid_in_same_grid_empty(self):
        grid_analyzer = Grid(self.strategy)
        grid_analyzer.__start__()

        x = np.array([0.0, 0.0])
        p = Point(x, "test")

        self.assertEqual(grid_analyzer.in_same_grid(p), [])


if __name__ == "__main__":
    import unittest

    unittest.main()
