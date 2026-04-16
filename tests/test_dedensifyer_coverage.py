import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result
from panobbgo.analyzers.dedensifyer import Dedensifyer, Box

class TestDedensifyerCoverage(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint
        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_dedensifyer_box_min_max(self):
        box = Box()
        # Initial register sets everything
        r1 = Result(Point(np.zeros(2), "p1"), 5.0, cv_vec=np.array([1.0]))
        box.register(r1)

        self.assertEqual(box.max_fx, r1)
        self.assertEqual(box.min_fx, r1)
        self.assertEqual(box.max_cv, r1)
        self.assertEqual(box.min_cv, r1)

        # Second register updates
        r2 = Result(Point(np.zeros(2), "p2"), 10.0, cv_vec=np.array([2.0]))
        box.register(r2)

        self.assertEqual(box.max_fx, r2)
        self.assertEqual(box.min_fx, r1)
        self.assertEqual(box.max_cv, r2)
        self.assertEqual(box.min_cv, r1)

        # Third register updates min
        r3 = Result(Point(np.zeros(2), "p3"), 1.0, cv_vec=np.array([0.0]))
        box.register(r3)

        self.assertEqual(box.max_fx, r2)
        self.assertEqual(box.min_fx, r3)
        self.assertEqual(box.max_cv, r2)
        self.assertEqual(box.min_cv, r3)

    def test_dedensifyer_get_box(self):
        dedens = Dedensifyer(self.strategy, max_depth=2)
        dedens.__start__()

        p1 = Point(self.problem.center, "p1")
        r1 = Result(p1, 5.0, cv_vec=np.array([1.0]))

        # Box should be None before registration
        self.assertIsNone(dedens.get_box(p1.x, 0))

        dedens.register(r1)

        # Box should exist after
        box = dedens.get_box(p1.x, 0)
        self.assertIsNotNone(box)
        self.assertEqual(box.count, 1)

    def test_dedensifyer_on_new_results(self):
        dedens = Dedensifyer(self.strategy, max_depth=2)
        dedens.__start__()

        p1 = Point(self.problem.center, "p1")
        r1 = Result(p1, 5.0, cv_vec=np.array([1.0]))

        p2 = Point(self.problem.box[:, 0] + 0.1, "p2")
        r2 = Result(p2, 10.0, cv_vec=np.array([2.0]))

        dedens.on_new_results([r1, r2])

        key1 = dedens.gridkey(p1.x, 0)
        key2 = dedens.gridkey(p2.x, 0)

        # Should be registered
        self.assertIn(key1, dedens.boxes[0])
        self.assertIn(key2, dedens.boxes[0])
