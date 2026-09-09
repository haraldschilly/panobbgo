# -*- coding: utf8 -*-
"""Credit assignment modes of StrategyRewarding."""

import numpy as np
import pytest

from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


def _strategy(**kw):
    from panobbgo.strategies import StrategyRewarding

    return StrategyRewarding(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0, **kw)


def _result(who, fx, x=(0.0, 0.0)):
    return Result(Point(np.array(x, dtype=float), who), fx, cv_vec=np.zeros(2))


def test_default_mode_is_ema_and_kwarg_overrides():
    s = _strategy()
    assert s.credit == "ema"
    assert s.explore == pytest.approx(0.2)
    assert _strategy(credit="legacy").credit == "legacy"
    assert _strategy(explore=0.5).explore == pytest.approx(0.5)
    s.config.rewarding_credit = "legacy"
    assert _strategy().credit == "ema"  # config is per strategy
    with pytest.raises(ValueError):
        _strategy(credit="bogus")


def test_ema_rewards_per_evaluation():
    from panobbgo.heuristics import Random

    s = _strategy(credit="ema")
    a, b = Random(s, name="A"), Random(s, name="B")
    s.add_heuristic(a)
    s.add_heuristic(b)
    alpha = 1.0 - float(s.config.discount)

    s.on_new_results([_result("A", 10.0)])  # first result: reward 1
    assert a.performance == pytest.approx(1.0)
    assert a.n_evals == 1

    s.on_new_results([_result("B", 20.0), _result("B", 30.0)])  # no improvement: reward 0
    assert b.performance == pytest.approx((1 - alpha) ** 2)
    assert b.n_evals == 2

    s.on_new_results([_result("B", 1.0)])  # improvement by 9 -> reward ~1
    expected = (1 - alpha) * (1 - alpha) ** 2 + alpha * (1 - np.exp(-9.0))
    assert b.performance == pytest.approx(expected)
    assert s.last_best.fx == 1.0


def test_has_points_tracks_the_queue():
    from panobbgo.heuristics import Random

    h = Random(_strategy())
    assert not h.has_points
    h.emit([np.zeros(2)])
    assert h.has_points
    h.get_points()
    assert not h.has_points


def test_ema_selection_only_draws_from_heuristics_with_points():
    from panobbgo.heuristics import Random

    s = _strategy(credit="ema", explore=0.0)
    a, b = Random(s, name="A"), Random(s, name="B")
    s.add_heuristic(a)
    s.add_heuristic(b)
    a.performance, b.performance = 0.9, 0.1
    b.emit([np.zeros(2) for _ in range(5)])  # only B has points

    batch = s._select_ema([a, b], target=4)

    assert len(batch) == 4
    assert all(p.who == "B" for p in batch)


def test_ema_probability_matching_with_floor():
    from panobbgo.heuristics import Random

    s = _strategy(credit="ema", explore=0.2)
    a, b = Random(s, name="A"), Random(s, name="B")
    s.add_heuristic(a)
    s.add_heuristic(b)
    a.performance, b.performance = 0.8, 0.0
    a.emit([np.zeros(2) for _ in range(20)])
    b.emit([np.zeros(2) for _ in range(20)])

    batch = s._select_ema([a, b], target=10)
    who = [p.who for p in batch]
    # A: 0.8*1.0 + 0.1 = 0.9 -> 9 points, B: floor 0.1 -> 1 point
    assert who.count("A") == 9
    assert who.count("B") == 1


def test_ema_end_to_end_is_reproducible():
    from panobbgo.heuristics import JSO, Nearby, Random

    def run():
        s = _strategy(credit="ema")
        s.config.max_eval = 60
        s.config.sync_evaluation = True
        s.add(Random)
        s.add(Nearby, radius=0.1, axes="all", new=3)
        s.add(JSO, NP_init=8)
        s.start()
        df = s.results._results_df
        return df["fx"].to_numpy(dtype=float).ravel()

    fa, fb = run(), run()
    assert len(fa) >= 60
    np.testing.assert_array_equal(fa, fb)
