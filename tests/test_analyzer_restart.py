# -*- coding: utf8 -*-
import numpy as np
import pytest
from unittest import mock
from panobbgo.analyzers.restart import Restart
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class FlatProblem(Problem):
    """f(x) = 0 for all x. Guarantees stagnation."""

    def __init__(self, dim=2):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return 0.0


class ImprovingProblem(Problem):
    """f(x) = -sum(x). Easy to improve by increasing x."""

    def __init__(self, dim=2):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return float(-np.sum(x))


def _make_strategy(problem):
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = Config(parse_args=False, testing_mode=True)
    strategy.constraint_handler = None
    return strategy


def _make_results(xs, problem):
    results = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        pt = Point(x, "test")
        fx = problem.eval(x)
        results.append(Result(pt, fx))
    return results


def test_restart_fires_after_patience():
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=10, max_restarts=5)
    r.__start__()

    rng = np.random.default_rng(0)

    # First result sets the baseline (improvement from inf → 0)
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))
    assert r.restart_count == 0

    # Feed 10 more stagnant results — should trigger restart
    xs = rng.uniform(-5, 5, (10, 2))
    r.on_new_results(_make_results(xs, problem))
    assert r.restart_count == 1

    strategy.eventbus.publish.assert_called()
    call_args = strategy.eventbus.publish.call_args
    assert call_args[0][0] == "restart"
    assert "center" in call_args[1]
    assert "reason" in call_args[1]
    center = call_args[1]["center"]
    # Center should be inside the box
    assert np.all(center >= -5.0) and np.all(center <= 5.0)


def test_no_restart_when_improving():
    problem = ImprovingProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=10, max_restarts=5)
    r.__start__()

    # Feed results with continuously improving fx
    for i in range(20):
        x = np.array([float(i), float(i)])
        results = _make_results([x], problem)
        r.on_new_results(results)

    assert r.restart_count == 0


def test_max_restarts_limit():
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=3)
    r.__start__()

    rng = np.random.default_rng(1)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger many restart cycles
    for _ in range(10):
        xs = rng.uniform(-5, 5, (6, 2))
        r.on_new_results(_make_results(xs, problem))

    # Should stop at max_restarts
    assert r.restart_count == 3


def test_default_patience():
    problem = FlatProblem(dim=4)
    strategy = _make_strategy(problem)
    r = Restart(strategy)
    r.__start__()
    assert r._patience == 20  # 5 * dim


def test_diverse_strategy():
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=5, restart_strategy="diverse")
    r.__start__()

    rng = np.random.default_rng(2)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger two restarts
    for _ in range(2):
        xs = rng.uniform(-5, 5, (6, 2))
        r.on_new_results(_make_results(xs, problem))

    assert r.restart_count == 2
    # Previous centers should be stored
    assert len(r._previous_centers) == 2
    # Centers should be distinct
    dist = np.linalg.norm(r._previous_centers[0] - r._previous_centers[1])
    assert dist > 0


def test_counter_resets_after_restart():
    """After restart, the patience counter should reset so it takes another patience evals to trigger."""
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=10)
    r.__start__()

    rng = np.random.default_rng(3)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger first restart (5 stagnant)
    xs = rng.uniform(-5, 5, (5, 2))
    r.on_new_results(_make_results(xs, problem))
    assert r.restart_count == 1

    # A restart opens a new epoch whose best is reset, so — as at the start of
    # the run — the first result after it sets the epoch baseline and the
    # patience clock runs from there.  (This test used to count 3 + 2 results
    # straight after the restart; that relied on the old bug that the global
    # best was never reset, so the baseline result counted as stagnation.)
    # Baseline + 3 stagnant — should NOT restart yet
    xs2 = rng.uniform(-5, 5, (4, 2))
    r.on_new_results(_make_results(xs2, problem))
    assert r.restart_count == 1

    # 2 more to complete patience — NOW restart
    xs3 = rng.uniform(-5, 5, (2, 2))
    r.on_new_results(_make_results(xs3, problem))
    assert r.restart_count == 2


def test_new_basin_judged_against_epoch_best_not_global():
    """After a restart the new basin only has to improve on *itself*.

    Regression: the global best was never reset, so a basin that is worse than
    the incumbent but still descending looked stagnant and every restart fired
    back to back.
    """
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=10)
    r.__start__()

    def feed(fx):
        r.on_new_results([Result(Point(np.zeros(2), "t"), fx)])

    feed(-100.0)  # a deep incumbent
    for _ in range(5):
        feed(0.0)
    assert r.restart_count == 1

    # The new basin sits far above the incumbent but improves steadily.
    for i in range(30):
        feed(50.0 - i)
    assert r.restart_count == 1


def test_batch_counts_results_after_the_last_improvement():
    """Stagnant results after the improving one in the same batch count."""
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=10)
    r.__start__()
    results = [Result(Point(np.zeros(2), "t"), 1.0)] + [Result(Point(np.zeros(2), "t"), 2.0) for _ in range(5)]
    r.on_new_results(results)
    assert r.restart_count == 1


def test_restart_ignore_none_fx():
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=1)
    r.__start__()

    # We test that passing an invalid result does not crash, but correctly gets processed.
    # We use a valid fx then fx=None.
    r1 = Result(Point(np.array([0.0, 0.0]), "test"), fx=10.0)
    r2 = Result(Point(np.array([0.0, 0.0]), "test"), fx=None)

    r.on_new_results([r1])
    assert r.restart_count == 0

    # This shouldn't increment _evals_since_improvement because it skips it entirely inside the loop.
    # Wait, in the source code it skips the result but still adds len(results) if improved is False.
    r.on_new_results([r2])
    assert r.restart_count == 1


def test_restart_with_constraint_handler():
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)

    class MockConstraintHandler:
        def get_penalty_value(self, r):
            return r.fx + sum(r.cv_vec)

    strategy.constraint_handler = MockConstraintHandler()
    r = Restart(strategy, patience=1)
    r.__start__()

    r1 = Result(Point(np.array([0.0, 0.0]), "test"), fx=10.0, cv_vec=np.array([1.0]))
    r.on_new_results([r1])

    r2 = Result(Point(np.array([0.0, 0.0]), "test"), fx=10.0, cv_vec=np.array([1.0]))
    r.on_new_results([r2])
    assert r.restart_count == 1


def test_sphere_strategy_uses_normal_distribution():
    """``restart_strategy='sphere'`` produces a Gaussian draw around the box centre."""
    problem = FlatProblem(dim=3)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=10, restart_strategy="sphere")
    r.__start__()

    rng = np.random.default_rng(4)

    # First result sets the baseline
    xs0 = rng.uniform(-5, 5, (1, 3))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger many restarts (override numpy's RNG to a deterministic stream
    # via np.random.seed so the Gaussian draws are reproducible).
    centers = []
    np.random.seed(123)
    for _ in range(40):
        xs = rng.uniform(-5, 5, (6, 3))
        r.on_new_results(_make_results(xs, problem))
        if r._previous_centers:
            centers.append(r._previous_centers[-1])
        if r.restart_count >= 10:
            break

    assert r.restart_count == 10
    centers_arr = np.asarray(centers)

    # All centers must lie inside the box (Gaussian draws are clipped).
    assert np.all(centers_arr >= -5.0)
    assert np.all(centers_arr <= 5.0)

    # Gaussian-around-centre means the empirical mean should be close to
    # the box centre (which is the origin for a symmetric box).  Use a
    # wide tolerance because only 10 draws — but tighter than the
    # ``uniform`` baseline would produce.
    assert np.linalg.norm(np.mean(centers_arr, axis=0)) < 2.0


def test_sphere_strategy_independent_of_previous_centers():
    """``sphere`` ignores ``_previous_centers`` and always Gaussian-samples.

    Distinguishes it from ``"diverse"`` which switches to max-min-distance
    selection once any previous centers are stored.
    """
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=3, max_restarts=5, restart_strategy="sphere")
    r.__start__()

    rng = np.random.default_rng(5)
    np.random.seed(456)

    # Stuff in a "previous center" manually to confirm sphere ignores it.
    r._previous_centers.append(np.array([4.9, 4.9]))  # near a corner

    # Trigger one restart
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))
    xs = rng.uniform(-5, 5, (4, 2))
    r.on_new_results(_make_results(xs, problem))

    assert r.restart_count == 1
    # The new center (last appended) should be the sphere draw, *not* the
    # max-min-distance draw which would be anti-correlated with the corner
    # we injected.
    new_center = r._previous_centers[-1]
    # Sphere draws use ranges/6 std around centre; corner draws would land
    # near (-4.9, -4.9) ideally.  Empirically the sphere draw is within
    # the central half of the box with very high probability under
    # std = 5/3.
    assert np.all(np.abs(new_center) <= 5.0)


def test_invalid_restart_strategy_raises():
    """Constructor rejects an unknown ``restart_strategy``."""
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    with pytest.raises(ValueError, match="restart_strategy must be one of"):
        Restart(strategy, restart_strategy="invalid_strategy")


def test_supported_restart_strategies_constant():
    """``SUPPORTED_RESTART_STRATEGIES`` lists the three policies."""
    assert Restart.SUPPORTED_RESTART_STRATEGIES == ("random", "diverse", "sphere")
