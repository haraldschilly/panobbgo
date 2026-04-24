import pytest
import numpy as np
from panobbgo.analyzers.splitter import Splitter
from panobbgo.lib import Problem, Point, Result, BoundingBox
from unittest import mock


class MockStrategy:
    def __init__(self, problem):
        self.problem = problem
        self.config = mock.MagicMock()
        self.config.max_eval = 100
        self.config.get_logger.return_value = mock.MagicMock()
        self.eventbus = mock.MagicMock()
        self.constraint_handler = mock.MagicMock()


class MockProblem(Problem):
    def __init__(self, dim):
        super().__init__([(-5, 5)] * dim)

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        return [0.0]


def make_result(x, fx, cv=0.0):
    r = Result(Point(np.array(x), "who"), fx=fx, cv_vec=np.array([cv]), error=0.0)
    return r


def test_splitter_get_box():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0)
    splitter.limit = 1
    splitter.root.add_result(r1)  # it will split automatically

    box = splitter.get_box(r1.x)
    assert box.leaf


def test_splitter_best_box_updates_identical_points():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)

    def is_better_mock(best, new):
        return new.fx < best.fx

    strategy.constraint_handler.is_better.side_effect = is_better_mock

    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0)
    r2 = make_result([-1.0, -1.0], 10.0)

    b1 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())
    b2 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())

    b1.best = r1
    b2.best = r2

    splitter.best_box = splitter.root
    splitter.best_box.best = r1

    splitter.on_new_split(splitter.root, [b1, b2], 0)


def test_splitter_best_box_updates_identical_points_same_fx_different_cv():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)

    def is_better_mock(best, new):
        return new.fx < best.fx

    strategy.constraint_handler.is_better.side_effect = is_better_mock

    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0, cv=0.0)
    r2 = make_result([1.0, 1.0], 10.0, cv=1.0)

    b1 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())
    b2 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())

    b1.best = r1
    b2.best = r2

    splitter.best_box = splitter.root
    splitter.best_box.best = r1

    splitter.on_new_split(splitter.root, [b1, b2], 0)
    # the second one fails the identity test, because identity test only covers the specific exact case where neither is better but they are identical (fx and cv). Here cv is different so they are not identical. In the actual code:
    # new_box.best.cv == self.best_box.best.cv is FALSE. So it will NOT update to b2. Which is correct, it will stay b1.
    assert splitter.best_box == b1


def test_splitter_best_box_updates_identical_points_same_fx_same_cv():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)

    def is_better_mock(best, new):
        return new.fx < best.fx

    strategy.constraint_handler.is_better.side_effect = is_better_mock

    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0, cv=0.0)
    r2 = make_result([1.0, 1.0], 10.0, cv=0.0)

    b1 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())
    b2 = Splitter.Box(splitter.root, splitter, splitter.root.box.copy())

    b1.best = r1
    b2.best = r2

    splitter.best_box = splitter.root
    splitter.best_box.best = r1

    # We pass b1 first so we get the update behavior covered correctly
    splitter.on_new_split(splitter.root, [b1, b2], 0)
    # The logic in splitter is:
    # It loops over b1, then b2.
    # For b1: new_box.best (r1) == best_box.best (r1). Yes. new_box.best IS best_box.best. Yes. best_box = b1.
    # For b2: new_box.best (r2) == best_box.best (r1). Yes. new_box.best IS NOT best_box.best. But new_box.best.cv == best_box.best.cv. Yes. best_box = b2.
    # So b2 becomes the best box!
    assert splitter.best_box == b2


def test_splitter_box_split_dim():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0)

    # We want to test when dim=None. Box.split should compute dim.
    splitter.limit = 1
    splitter.root.results.append(r1)

    # The box is [-5, 5] x [-5, 5], range is 10 for both dims.
    # split should choose dim=0 since both are 10.
    splitter.root.split(dim=None)

    assert not splitter.root.leaf
    assert len(splitter.root.children) == 2
    assert splitter.root.split_dim == 0


def test_splitter_box_add_result_recursive():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    splitter = Splitter(strategy)
    splitter.__start__()

    # It splits when length >= limit AFTER insertion.
    # So if limit = 1, appending 1 item makes len=1, so it splits immediately.
    splitter.limit = 1
    r1 = make_result([-1.0, -1.0], 10.0)

    # Manually split it to get not a leaf
    splitter.root.results.append(r1)
    splitter.root.split(0)

    assert not splitter.root.leaf

    # Add a new result, it should route to child.
    r2 = make_result([4.0, 4.0], 5.0)
    splitter.root.add_result(r2)

    # Get leaf should point to one of the children
    leaf = splitter.get_leaf(r2)
    assert leaf is not splitter.root
    assert leaf.contains(r2.x)


def test_splitter_get_leaf_wait():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    splitter = Splitter(strategy)
    splitter.__start__()

    r1 = make_result([-1.0, -1.0], 10.0)

    # Simulate another thread waiting for result
    import threading
    import time

    def wait_for_leaf():
        # This will block until r1 is in result2leaf
        leaf = splitter.get_leaf(r1)
        assert leaf == splitter.root

    t = threading.Thread(target=wait_for_leaf)
    t.start()

    time.sleep(0.1)  # let the thread block

    # Now simulate new result coming in
    splitter.on_new_results([r1])

    t.join(timeout=1.0)
    assert not t.is_alive()


def test_splitter_properties():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    splitter = Splitter(strategy)
    splitter.__start__()

    box = splitter.root

    # Test ranges
    np.testing.assert_array_equal(box.ranges, np.array([10.0, 10.0]))

    # Test log_volume
    # log(10) + log(10) = 2 * log(10)
    assert np.isclose(box.log_volume, 2 * np.log(10))

    # Test volume
    assert np.isclose(box.volume, 100.0)

    # Test repr
    s = repr(box)
    assert "Box" in s
