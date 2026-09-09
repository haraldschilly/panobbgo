# -*- coding: utf8 -*-
"""Behaviour a library user gets with the *default* configuration.

The benchmark harness overrides several config values by hand; these tests
pin down what happens without any of that (planning/DISCOVERY_2026-09-09.md §3).
"""

from panobbgo.lib.classic import Rosenbrock


def test_default_run_spends_the_whole_budget():
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.strategies import StrategyRewarding

    budget = 300
    strategy = StrategyRewarding(Rosenbrock(dim=5), parse_args=False, max_eval=budget, seed=1)
    assert strategy.config.stop_on_convergence is False
    strategy.add(Random)
    strategy.add(Nearby, radius=0.1, axes="all", new=3)
    strategy.start()

    assert len(strategy.results) >= budget
    assert strategy.best is not None
