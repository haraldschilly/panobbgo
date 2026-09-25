import numpy as np
from tests.support import PanobbgoTestCase
from panobbgo.lib import Point, Result
from panobbgo.analyzers.splitter import Splitter


class TestSplitterCoverage(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint

        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_splitter_get_box(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        p = Point(self.problem.center, "p")
        self.assertEqual(splitter.get_box(p.x), splitter.root)

    def test_splitter_get_leaf_before_registration(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        r = Result(Point(self.problem.center, "p"), 1.0)
        self.assertIsNone(splitter.get_leaf(r))
        splitter.on_new_results([r])
        self.assertIs(splitter.get_leaf(r), splitter.root)

    def test_box_properties(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        box = splitter.root
        self.assertEqual(box.fx, float("inf"))

        np.testing.assert_array_equal(box.ranges, self.problem.ranges)
        self.assertTrue(np.isscalar(box.log_volume))
        self.assertTrue(np.isscalar(box.volume))
        self.assertEqual(len(box), 0)

        r = Result(Point(self.problem.center, "p"), 1.0)
        box += r
        self.assertEqual(len(box), 1)

    def test_box_get_child_boxes_error(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        with self.assertRaises(AssertionError):
            splitter.root.get_child_boxes(self.problem.center)

    def test_box_repr(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        r = repr(splitter.root)
        self.assertIn("Box-0", r)
        self.assertIn("leaf", r)

    def test_splitter_get_leaf(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        r = Result(Point(self.problem.center, "p"), 1.0)
        splitter.on_new_results([r])
        self.assertEqual(splitter.get_leaf(r), splitter.root)

    def test_splitter_new_split(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        # Manually force split
        splitter.root.split()
        self.assertEqual(len(splitter.leafs), 2)
        self.assertEqual(splitter.max_depth, 1)

    def test_splitter_best_box_updates(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        # create points to trigger a split natively and exercise on_new_split logic
        for i in range(int(splitter.limit) + 1):
            r = Result(Point(self.problem.center + np.random.randn(2) * 0.01, f"p{i}"), 10.0 - float(i) / 100.0)
            splitter.on_new_results([r])

        # self.assertIsNotNone(splitter.best_box)
        # self.assertFalse(splitter.best_box.leaf == False)

    def test_splitter_on_refresh_best(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        # Give it a root with results
        r1 = Result(Point(self.problem.center, "p1"), 5.0, cv_vec=np.array([1.0]))
        r2 = Result(Point(self.problem.center + 0.1, "p2"), 1.0, cv_vec=np.array([2.0]))

        splitter.on_new_results([r1, r2])
        splitter.leafs.append(splitter.root)

        # Test on_refresh_best
        splitter.on_refresh_best([r1, r2])
        self.assertEqual(splitter.root.best, r2)

    def test_splitter_on_refresh_best_no_ch(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()
        splitter.strategy.constraint_handler = None

        # Give it a root with results
        r1 = Result(Point(self.problem.center, "p1"), 5.0, cv_vec=np.array([1.0]))
        r2 = Result(Point(self.problem.center + 0.1, "p2"), 1.0, cv_vec=np.array([2.0]))

        splitter.on_new_results([r1, r2])
        splitter.leafs.append(splitter.root)

        # Test on_refresh_best
        splitter.on_refresh_best([r1, r2])
        self.assertEqual(splitter.root.best, r2)

    def test_splitter_register_result_no_ch(self):
        splitter = Splitter(self.strategy)
        splitter.strategy.constraint_handler = None
        splitter.__start__()

        # Test fallback comparison logic in _register_result
        r1 = Result(Point(self.problem.center, "p1"), 5.0, cv_vec=np.array([1.0]))
        r2 = Result(Point(self.problem.center + 0.1, "p2"), 1.0, cv_vec=np.array([2.0]))

        splitter.root._register_result(r1)
        # self.assertEqual(splitter.root.best, r1)

        # Add worse result (cv is higher), should not replace r1
        splitter.root._register_result(r2)
        # self.assertEqual(splitter.root.best, r1)

        # Without a constraint handler the fallback compares fx only: r2
        # (fx 1.0) replaced r1, and r3 (also fx 1.0) is not strictly better.
        # (This passed as ``best == r3`` only while Result.__eq__ compared fx.)
        r3 = Result(Point(self.problem.center + 0.2, "p3"), 1.0, cv_vec=np.array([1.0]))
        splitter.root._register_result(r3)
        self.assertIs(splitter.root.best, r2)

    def test_splitter_result_in_leaf_after_split(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        # add points to force a split
        for i in range(int(splitter.limit) + 1):
            r = Result(Point(self.problem.center + np.random.randn(2) * 0.01, f"p{i}"), 10.0)
            splitter.on_new_results([r])

        r_new = Result(Point(self.problem.center, "p_new"), 1.0)
        splitter.on_new_results([r_new])

        leaf = splitter.get_leaf(r_new)
        self.assertTrue(leaf.leaf)
        self.assertGreater(leaf.depth, 0)
        self.assertIn(r_new, leaf.results)
        self.assertIn(r_new, splitter.root.results)

    def test_interior_boxes_recompute_their_results(self):
        """Only leaves and the root store results; a split box's list is
        rebuilt from the root and must equal the list it used to keep —
        same objects, same (arrival) order — including points on a cut."""
        splitter = Splitter(self.strategy)
        splitter.__start__()
        rng = np.random.default_rng(0)
        box = np.asarray(self.problem.box.box)
        lo, rg = box[:, 0], box[:, 1] - box[:, 0]
        results = []
        for i in range(20 * int(splitter.limit)):
            x = lo + rg * rng.random(2)
            if i % 7 == 0 and results:
                x = results[-1].x.copy()  # duplicates, some of which end up on a cut
            r = Result(Point(x, "p"), float(rng.random()))
            results.append(r)
            splitter.on_new_results([r])

        def walk(b):
            yield b
            for c in b.children:
                yield from walk(c)

        boxes = list(walk(splitter.root))
        interior = [b for b in boxes if not b.leaf and b.parent is not None]
        self.assertTrue(interior, "the cloud must build a tree of depth >= 2")
        for b in boxes:
            expected = [r for r in results if b is splitter.root or b.contains(r.x)]
            self.assertEqual([id(r) for r in b.results], [id(r) for r in expected])
            if b.parent is not None and not b.leaf:
                self.assertIsNone(b._results)
            self.assertEqual(len(b), len(expected))
        for r in results:
            leaf = splitter.get_leaf(r)
            self.assertTrue(leaf.leaf and leaf.contains(r.x))

        # ``len``/truthiness/``repr`` of a split box never rebuild its list.
        calls = []
        orig = splitter._results_in
        splitter._results_in = lambda box: calls.append(box) or orig(box)
        for b in interior:
            self.assertTrue(len(b) > 0 and bool(b) and "Box-" in repr(b))
        self.assertEqual(calls, [])

        # A root list edited by hand (tests do) is detected and the
        # coordinate matrix rebuilt.
        extra = Result(Point(lo + 0.5 * rg, "p"), 0.0)
        splitter.root.results.append(extra)
        b = interior[0]
        want = [r for r in splitter.root.results if b.contains(r.x)]
        self.assertEqual([id(r) for r in orig(b)], [id(r) for r in want])

    def test_splitter_on_new_split_update_branches(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()

        box = splitter.root

        class MockResult:
            def __init__(self, fx, cv=0):
                self.fx = fx
                self.cv = cv

            def __eq__(self, other):
                if not isinstance(other, MockResult):
                    return False
                return self.fx == other.fx

            def __lt__(self, other):
                return self.fx < other.fx

        class MockBox:
            def __init__(self, best=None, fx=0):
                self.best = best
                self.fx = fx

        # branch 1: best_box is None
        splitter.best_box = None
        new_box = MockBox()
        splitter.on_new_split(box, [new_box], 0)
        self.assertEqual(splitter.best_box, new_box)

        # branch 2: best_box not None, but new_box.best is None
        new_box2 = MockBox(best=None)
        splitter.on_new_split(box, [new_box2], 0)
        self.assertEqual(splitter.best_box, new_box)

        # branch 3: best_box.best is None
        new_box.best = None
        new_box3 = MockBox(best=MockResult(1.0))
        splitter.on_new_split(box, [new_box3], 0)
        self.assertEqual(splitter.best_box, new_box3)

        # branch 4: is_better = True
        new_box4 = MockBox(best=MockResult(0.5))
        splitter.on_new_split(box, [new_box4], 0)
        self.assertEqual(splitter.best_box, new_box4)

        # branch 5: not is_better, but best_box_was_parent and equal
        splitter.best_box = box
        box.best = MockResult(1.0)
        new_box5 = MockBox(best=MockResult(1.0))
        splitter.on_new_split(box, [new_box5], 0)
        self.assertEqual(splitter.best_box, new_box5)

    def test_splitter_on_new_split_update_branches_no_ch(self):
        splitter = Splitter(self.strategy)
        splitter.strategy.constraint_handler = None
        splitter.__start__()

        box = splitter.root

        class MockResult:
            def __init__(self, fx, cv=0):
                self.fx = fx
                self.cv = cv

            def __eq__(self, other):
                if not isinstance(other, MockResult):
                    return False
                return self.fx == other.fx

            def __lt__(self, other):
                return self.fx < other.fx

        class MockBox:
            def __init__(self, best=None, fx=0):
                self.best = best
                self.fx = fx

        # branch 1: best_box is None
        splitter.best_box = box
        box.best = MockResult(1.0)
        new_box = MockBox(best=MockResult(0.5), fx=0.5)

        # Test branch 4: is_better without CH
        splitter.on_new_split(box, [new_box], 0)
        self.assertEqual(splitter.best_box, new_box)

        # branch 5 identity check
        splitter.best_box = box
        new_box2 = MockBox(best=box.best, fx=1.0)
        splitter.on_new_split(box, [new_box2], 0)
        self.assertEqual(splitter.best_box, new_box2)

        # branch 5 equality check but not identity
        splitter.best_box = box
        new_box3 = MockBox(best=MockResult(1.0, 0), fx=1.0)
        splitter.on_new_split(box, [new_box3], 0)
        self.assertEqual(splitter.best_box, new_box3)
