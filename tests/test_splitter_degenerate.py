# -*- coding: utf8 -*-
"""The Splitter must not deepen its tree when a cut cannot separate points.

``Box.contains`` includes both boundaries, so a split through a cluster of
identical points puts every one of them in *both* children.  Each child is
then an over-full leaf that splits again on the next result, and the tree
grows without bound — a live-lock that ate the rest of the evaluation
budget (CMA-ES on MA-BBOB d5 stalled at 408 of 1000 evaluations, 154 s).
"""

import time

import numpy as np
import pytest

from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


def _strategy(max_eval=400):
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=3), parse_args=False, testing_mode=True, seed=0)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    return s


def _splitter(strategy):
    from panobbgo.analyzers import Splitter

    sp = Splitter(strategy)
    sp.__start__()
    return sp


def test_identical_points_do_not_deepen_the_tree():
    strategy = _strategy()
    sp = _splitter(strategy)
    x = np.array([0.25, -0.5, 1.0])

    started = time.perf_counter()
    for _ in range(5 * int(sp.limit)):
        sp.root += Result(Point(x.copy(), "test"), 1.0, cv_vec=np.zeros(3))
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0, f"adding identical points took {elapsed:.1f}s — tree is still deepening"
    depths = [b.depth for b in sp.leafs]
    assert max(depths) <= sp.root.MAX_DEPTH


def test_distinct_points_still_split():
    strategy = _strategy()
    sp = _splitter(strategy)
    rng = np.random.default_rng(0)

    for _ in range(3 * int(sp.limit)):
        x = strategy.problem.random_point(rng=rng)
        sp.root += Result(Point(x, "test"), float(rng.random()), cv_vec=np.zeros(3))

    assert len(sp.leafs) > 1, "a spread-out point cloud must still be split"
    assert not sp.root.leaf


def test_split_picks_a_dimension_the_points_differ_in():
    strategy = _strategy()
    sp = _splitter(strategy)
    # Vary only dimension 2; dimension 0 is the widest but constant here.
    for i in range(int(sp.limit) + 1):
        x = np.array([0.0, 0.0, -1.0 + 2.0 * i / int(sp.limit)])
        sp.root += Result(Point(x, "test"), float(i), cv_vec=np.zeros(3))

    assert not sp.root.leaf, "the box should have been split"
    assert sp.root.split_dim == 2


@pytest.mark.parametrize("dim", [2, 5])
def test_run_with_a_collapsing_search_spends_its_budget(dim):
    """A heuristic that keeps proposing the same point must not stall the run."""
    from panobbgo.core import Heuristic

    class Collapsed(Heuristic):
        """Emits the same point over and over — the degenerate case."""

        def on_start(self):
            self.fill_queue(lambda: np.zeros(self.problem.dim))

        def on_new_results(self, results):
            self.fill_queue(lambda: np.zeros(self.problem.dim))

    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=0)
    s.config.max_eval = 300
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_heuristic(Collapsed(s))

    started = time.perf_counter()
    s.start()
    elapsed = time.perf_counter() - started

    assert len(s.results) >= 300, f"only {len(s.results)}/300 evaluations"
    assert elapsed < 30.0, f"run took {elapsed:.1f}s"
