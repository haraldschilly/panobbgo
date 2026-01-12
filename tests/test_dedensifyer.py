from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result


class TestDedensifyer(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint
        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_dedensifyer_init(self):
        from panobbgo.analyzers.dedensifyer import Dedensifyer

        dedens = Dedensifyer(self.strategy, max_depth=5)
        assert dedens.max_depth == 5
        assert dedens.strategy == self.strategy
        # Check internal structure initialization
        assert isinstance(dedens.boxes, list)
        assert len(dedens.boxes) == 5
        for b in dedens.boxes:
            assert isinstance(b, dict)

    def test_dedensifyer_start(self):
        from panobbgo.analyzers.dedensifyer import Dedensifyer

        dedens = Dedensifyer(self.strategy, max_depth=3)
        dedens.__start__()

        # Verify box_dims are correctly calculated
        assert len(dedens.box_dims) == 3
        # Depth 0: ranges / 2**0 = ranges
        np.testing.assert_array_equal(dedens.box_dims[0], self.problem.ranges)
        # Depth 1: ranges / 2**1 = ranges / 2
        np.testing.assert_array_equal(dedens.box_dims[1], self.problem.ranges / 2.0)

    def test_dedensifyer_register(self):
        from panobbgo.analyzers.dedensifyer import Dedensifyer, Box

        dedens = Dedensifyer(self.strategy, max_depth=2)
        dedens.__start__()

        # Point at origin (should be index 0,0 at all depths if box starts at < 0)
        # RosenbrockConstraint has box [(-1.5, 1.5), (-1.5, 1.5)]? No, let's check problem ranges.
        # Classic RosenbrockConstraint usually [(-1.5, 1.5)]^2

        # Let's create a point exactly in the middle of the box
        center = self.problem.center
        p = Point(center, "test")
        res = Result(p, 1.0)

        dedens.register(res)

        # Should be registered at all depths
        for depth in range(2):
            key = dedens.gridkey(center, depth)
            assert key in dedens.boxes[depth]
            box = dedens.boxes[depth][key]
            assert isinstance(box, Box)
            assert box.count == 1
            assert box.min_fx == res
            assert box.max_fx == res

    def test_dedensifyer_grid_logic(self):
        from panobbgo.analyzers.dedensifyer import Dedensifyer

        dedens = Dedensifyer(self.strategy, max_depth=2)
        dedens.__start__()

        # Box min is [-1.5, -1.5], ranges [3.0, 3.0]
        # Depth 0: cell size [3.0, 3.0]. Grid is 1x1.
        # Depth 1: cell size [1.5, 1.5]. Grid is 2x2.

        # Point slightly above min -> (0,0)
        p1 = self.problem.box[:, 0] + 0.1
        key0 = dedens.gridkey(p1, 0)
        key1 = dedens.gridkey(p1, 1)
        assert key0 == (0, 0)
        assert key1 == (0, 0)

        # Point slightly below max -> (0,0) at depth 0, (1,1) at depth 1
        p2 = self.problem.box[:, 1] - 0.1
        key0_p2 = dedens.gridkey(p2, 0)
        key1_p2 = dedens.gridkey(p2, 1)
        assert key0_p2 == (0, 0) # Still in the big box
        assert key1_p2 == (1, 1) # In the top-right quadrant

    def test_dedensifyer_multiple_points(self):
        from panobbgo.analyzers.dedensifyer import Dedensifyer

        dedens = Dedensifyer(self.strategy, max_depth=2)
        dedens.__start__()

        # Two points in same grid cell at depth 1
        p1 = Point(self.problem.box[:, 0] + 0.1, "test") # Bottom-left
        p2 = Point(self.problem.box[:, 0] + 0.2, "test") # Bottom-left, close to p1

        res1 = Result(p1, 10.0)
        res2 = Result(p2, 5.0) # Better fx

        dedens.register(res1)
        dedens.register(res2)

        key = dedens.gridkey(p1.x, 1)
        box = dedens.boxes[1][key]

        assert box.count == 2
        assert box.min_fx.fx == 5.0
        assert box.max_fx.fx == 10.0
