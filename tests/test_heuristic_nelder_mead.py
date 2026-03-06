import numpy as np
import threading
import time
from panobbgo.heuristics.nelder_mead import NelderMead
from panobbgo.lib import Point, Result

class MockProblem:
    def __init__(self, dim=2):
        self.dim = dim

class MockConstraintHandler:
    def is_better(self, a, b):
        return a.fx < b.fx

    def get_penalty_value(self, r):
        return r.fx

class MockStrategy:
    def __init__(self):
        self.problem = MockProblem()
        self.constraint_handler = MockConstraintHandler()
        self.name = "Mock"
        self.config = MockConfig()

class MockConfig:
    def __init__(self):
        self.capacity = 100

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockBox:
    def __init__(self, results, parent=None):
        self.results = results
        self.parent = parent

def test_nm_init():
    strategy = MockStrategy()
    nm = NelderMead(strategy)
    assert nm.name == "Nelder Mead"

def test_gram_schmidt_not_enough_points():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    r1 = Result(Point(np.array([0., 0.]), "test"), 1.0)
    res = nm.gram_schmidt(2, [r1])
    assert res is None

def test_gram_schmidt_success():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    r1 = Result(Point(np.array([1., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([0., 1.]), "test"), 2.0)
    r3 = Result(Point(np.array([2., 2.]), "test"), 3.0)

    # Note: sorting happens internally based on fx value using MockConstraintHandler.
    # Therefore, order may change, and the exact instance might be sorted.
    res = nm.gram_schmidt(2, [r1, r2, r3])

    # Should select a basis of size 2
    assert res is not None
    assert len(res) == 2
    # Ensure they are from our result set
    assert res[0] in [r1, r2, r3]
    assert res[1] in [r1, r2, r3]

def test_gram_schmidt_degenerate():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    # Points on the same line (degenerate)
    r1 = Result(Point(np.array([1., 1.]), "test"), 1.0)
    r2 = Result(Point(np.array([2., 2.]), "test"), 2.0)
    r3 = Result(Point(np.array([3., 3.]), "test"), 3.0)

    res = nm.gram_schmidt(2, [r1, r2, r3])

    # Should not be able to form a 2D basis
    assert res is None

def test_gram_schmidt_zero_vector():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    # Origin should not be skipped, but subsequent zero projections should be handled
    r1 = Result(Point(np.array([0., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([1., 0.]), "test"), 2.0)
    r3 = Result(Point(np.array([0., 1.]), "test"), 3.0)

    res = nm.gram_schmidt(2, [r1, r2, r3])
    assert res is not None
    assert len(res) == 2

def test_nelder_mead_init():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    r1 = Result(Point(np.array([0., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([1., 0.]), "test"), 2.0)
    r3 = Result(Point(np.array([0., 1.]), "test"), 3.0) # Worst

    worst, centroid = nm.nelder_mead_init([r1, r2, r3])

    # constraint handler logic sorts descending or ascending. Mock handles it.
    assert worst in [r1, r2, r3]
    assert centroid.shape == (2,)

def test_nelder_mead_init_equal_values():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    r1 = Result(Point(np.array([0., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([1., 0.]), "test"), 1.0)
    r3 = Result(Point(np.array([0., 1.]), "test"), 1.0)

    worst, centroid = nm.nelder_mead_init([r1, r2, r3])
    # When weights sum to 0, it falls back to standard average
    assert worst is not None
    assert centroid.shape == (2,)
    np.testing.assert_array_equal(centroid, np.array([0.5, 0.])) # Assuming r3 is picked as worst

def test_nelder_mead_sample():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    r1 = Result(Point(np.array([0., 1.]), "test"), 3.0) # worst
    centroid = np.array([0.5, 0.])

    # Should not throw
    np.random.seed(0)
    pt = nm.nelder_mead_sample(r1, centroid, scale=1, offset=0)
    assert pt.shape == (2,)

def test_on_start_loop_break_on_stop():
    strategy = MockStrategy()
    nm = NelderMead(strategy)
    nm._output = type('MockQueue', (), {'full': lambda self: False, 'put': lambda self, x: None})()

    # Mock clear_output to prevent errors
    nm.clear_output = lambda: None

    # Needs to be defined to avoid AttributeError
    nm.best_box = None

    def background():
        time.sleep(0.1)
        nm._stopped = True
        nm.got_bb.set()

    t = threading.Thread(target=background)
    t.start()

    # Run loop
    nm.on_start()
    t.join()
    # If it finishes, it didn't block forever

def test_on_start_with_box():
    strategy = MockStrategy()
    nm = NelderMead(strategy)

    class MockQueue:
        def __init__(self):
            self.items = []
        def full(self):
            return len(self.items) >= 2 # Fill after 2 items
        def put(self, item, timeout=None):
            self.items.append(item)
    nm._output = MockQueue()
    nm.clear_output = lambda: None
    # We mock emit because nelder_mead directly calls self.emit which puts in queue differently
    def mock_emit(x):
        nm._output.put(x)
    nm.emit = mock_emit

    r1 = Result(Point(np.array([1., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([0., 1.]), "test"), 2.0)
    r3 = Result(Point(np.array([2., 2.]), "test"), 3.0)

    box = MockBox([r1, r2, r3])
    nm.best_box = box
    nm.got_bb.set()

    def background():
        time.sleep(0.2) # Let it generate some points
        nm._stopped = True

    t = threading.Thread(target=background)
    t.start()

    try:
        nm.on_start()
    except Exception as e:
        # Ignore StopHeuristic which gets raised by core.py emit when _stopped is True
        if type(e).__name__ != "StopHeuristic":
            raise
    t.join()

    assert len(nm._output.items) > 0

def test_on_start_fallback_to_parent():
    strategy = MockStrategy()
    nm = NelderMead(strategy)
    nm._output = type('MockQueue', (), {'full': lambda self: False, 'put': lambda self, x: None})()
    nm.clear_output = lambda: None

    r1 = Result(Point(np.array([1., 0.]), "test"), 1.0)
    r2 = Result(Point(np.array([0., 1.]), "test"), 2.0)
    r3 = Result(Point(np.array([2., 2.]), "test"), 3.0)

    # Parent has points, child does not
    parent_box = MockBox([r1, r2, r3])
    child_box = MockBox([r1], parent=parent_box)

    nm.best_box = child_box
    nm.got_bb.set()

    def background():
        time.sleep(0.2)
        nm._stopped = True

    t = threading.Thread(target=background)
    t.start()

    try:
        nm.on_start()
    except Exception as e:
        if type(e).__name__ != "StopHeuristic":
            raise
    t.join()

    # It should have tried the parent box, found the basis, and entered the generation loop
    # If the logic works, got_bb will still be set during the generation loop, but _stopped breaks it.

def test_on_new_best_box():
    strategy = MockStrategy()
    nm = NelderMead(strategy)
    nm.clear_output = lambda: None

    box = MockBox([])
    nm.on_new_best_box(box)

    assert nm.best_box == box
    assert nm.got_bb.is_set()
