from tests.support import StrategyDouble
from panobbgo.analyzers.best import Best
from panobbgo.lib import Result, Point
import numpy as np


class MockConstraintHandler:
    def is_better(self, old_best, new_result):
        # Let's say new is better if fx < old_best.fx + 0.1
        return new_result.fx < old_best.fx


class MockStrategy(StrategyDouble):
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = MockEventBus()
        self.best = None
        self.results = []
        self.constraint_handler = MockConstraintHandler()


class MockStrategyNoCH(StrategyDouble):
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = MockEventBus()
        self.best = None
        self.results = []
        self.constraint_handler = None


class MockStrategyWithLogger(StrategyDouble):
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = MockEventBus()
        self.best = None
        self.results = []
        self.constraint_handler = MockConstraintHandler()
        self.panobbgo_logger = MockPanobbgoLogger()
        self._progress_updated = False

    def _update_progress_status(self):
        self._progress_updated = True


class MockStrategyWithFailingLogger(StrategyDouble):
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = MockEventBus()
        self.best = None
        self.results = []
        self.constraint_handler = MockConstraintHandler()
        self.panobbgo_logger = MockPanobbgoLogger()

    def _update_progress_status(self):
        raise ValueError("Simulated update failure")


class MockProgressReporter:
    def __init__(self):
        self.enabled = True


class MockPanobbgoLogger:
    def __init__(self):
        self.progress_reporter = MockProgressReporter()


class MockConfig:
    def get_logger(self, name):
        import logging

        return logging.getLogger(name)


class MockEventBus:
    def __init__(self):
        self.events = []

    def publish(self, key, **kwargs):
        self.events.append((key, kwargs))


def test_best_no_constraint_handler():
    strategy = MockStrategyNoCH()
    analyzer = Best(strategy)

    r1 = Result(Point(np.array([0.0]), "test"), 5.0, cv_vec=np.array([1.0]))
    analyzer.on_new_results([r1])
    assert analyzer.best == r1

    # Same cv, better fx
    r2 = Result(Point(np.array([0.0]), "test"), 4.0, cv_vec=np.array([1.0]))
    analyzer.on_new_results([r2])
    assert analyzer.best == r2

    # Worse cv, better fx -> not better
    r3 = Result(Point(np.array([0.0]), "test"), 1.0, cv_vec=np.array([2.0]))
    analyzer.on_new_results([r3])
    assert analyzer.best == r2


def test_best_on_refresh_best():
    strategy = MockStrategy()
    analyzer = Best(strategy)

    r1 = Result(Point(np.array([0.0]), "test"), 5.0, cv_vec=np.array([1.0]))
    analyzer.on_new_results([r1])
    assert analyzer.best == r1

    # Candidates where one is better
    candidates = [
        Result(Point(np.array([0.0]), "test"), 6.0, cv_vec=np.array([1.0])),
        Result(Point(np.array([0.0]), "test"), 4.0, cv_vec=np.array([1.0])),
    ]

    analyzer.on_refresh_best(candidates)

    assert analyzer.best == candidates[1]


def test_best_on_refresh_best_no_ch():
    strategy = MockStrategyNoCH()
    analyzer = Best(strategy)

    r1 = Result(Point(np.array([0.0]), "test"), 5.0, cv_vec=np.array([1.0]))
    analyzer.on_new_results([r1])
    assert analyzer.best == r1

    # Candidates where one is better (lower cv)
    candidates = [
        Result(Point(np.array([0.0]), "test"), 6.0, cv_vec=np.array([1.0])),
        Result(Point(np.array([0.0]), "test"), 4.0, cv_vec=np.array([0.0])),
    ]

    analyzer.on_refresh_best(candidates)

    assert analyzer.best == candidates[1]


def test_best_on_refresh_best_empty():
    strategy = MockStrategy()
    analyzer = Best(strategy)

    r1 = Result(Point(np.array([0.0]), "test"), 5.0, cv_vec=np.array([1.0]))
    analyzer.on_new_results([r1])

    # Empty candidates
    analyzer.on_refresh_best([])

    assert analyzer.best == r1


def test_report_progress_event_success():
    strategy = MockStrategyWithLogger()
    analyzer = Best(strategy)

    # Manually trigger on_new_min to make sure it runs the inner logging parts
    r1 = Result(Point(np.array([0.0]), "test"), 5.0)
    analyzer.on_new_min(r1)

    assert strategy._progress_updated


def test_report_progress_event_fail():
    strategy = MockStrategyWithFailingLogger()
    analyzer = Best(strategy)

    # Shouldn't raise
    r1 = Result(Point(np.array([0.0]), "test"), 5.0)
    analyzer.on_new_results([r1])


def test_on_new_pareto_front_reports_progress():
    strategy = MockStrategyWithLogger()
    analyzer = Best(strategy)

    analyzer.on_new_pareto_front([Result(Point(np.array([0.0]), "test"), 5.0)])

    assert strategy._progress_updated


def test_best_min_and_cv_recover_from_a_nan_objective():
    """A NaN objective ranks last; it does not pin ``min`` / ``cv`` forever.

    Every comparison with NaN is False, so an early NaN result used to stay
    ``min`` (and ``cv`` on a tie in cv) for the rest of the run.
    """
    strategy = MockStrategyNoCH()
    analyzer = Best(strategy)
    r_nan = Result(Point(np.array([0.0]), "test"), float("nan"))
    r1 = Result(Point(np.array([1.0]), "test"), 5.0)
    r2 = Result(Point(np.array([2.0]), "test"), 3.0)
    analyzer.on_new_results([r_nan])
    analyzer.on_new_results([r1, r2])
    assert analyzer.min is r2
    assert analyzer.cv is r2
    assert analyzer.best is r2
    assert analyzer.pareto_front == [r2]


def _reference_front(front, r):
    """The pre-2026-09 front rebuild: re-sort front + [r], keep decreasing cv."""
    import math

    pf = sorted(front + [r], key=lambda p: math.inf if math.isnan(p.fx) else p.fx)
    new = [pf[0]]
    for p in pf[1:]:
        if new[-1].cv > p.cv:
            new.append(p)
    return new


def test_incremental_pareto_front_matches_the_full_rebuild():
    """The incremental update gives the rebuild's front, incl. ties and NaN."""
    rng = np.random.default_rng(11)
    strategy = MockStrategyNoCH()
    analyzer = Best(strategy)
    front = []
    changes = 0
    for k in range(2000):
        fx = float(rng.integers(0, 30)) if k % 3 else float(rng.uniform(0, 30))
        if k % 97 == 0:
            fx = float("nan")
        cv = float(rng.integers(0, 10)) if k % 2 else float(rng.uniform(0, 10))
        r = Result(Point(np.array([float(k)]), "test"), fx, cv_vec=np.array([cv]))
        before = analyzer.pareto_front
        analyzer._update_pareto(r)
        front = _reference_front(front, r)
        after = analyzer.pareto_front
        assert [id(p) for p in after] == [id(p) for p in front]
        changes += [id(p) for p in before] != [id(p) for p in after]
    events = [e for e in strategy.eventbus.events if e[0] == "new_pareto_front"]
    assert len(events) == changes > 10
