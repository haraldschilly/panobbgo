#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Tests for the benchmark system.
"""

import pytest
from panobbgo.benchmark import ProblemSpec, StrategySpec
from panobbgo.lib.classic import Rosenbrock


class TestProblemSpec:
    """Test ProblemSpec functionality."""

    def test_create_problem(self):
        """Test creating a problem from spec."""
        spec = ProblemSpec(
            name="test_rosenbrock",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{"x": [1.0, 1.0], "fx": 0.0}],
            tolerance=1e-6,
        )

        problem = spec.create_problem()
        assert isinstance(problem, Rosenbrock)
        assert problem.dim == 2


class TestStrategySpecDimGate:
    """Dimension-gated heuristic activation (``gate_min_dim`` / ``gate_max_dim``)."""

    class _RecordingStrategy:
        """Stands in for a StrategyBase: records add() calls, builds nothing."""

        def __init__(self, problem, parse_args=False, seed=None):
            self.problem = problem
            self.config = type("Cfg", (), {})()
            self.seed = seed
            self.added = []

        def add(self, heur_class, **kwargs):
            self.added.append((heur_class, kwargs))

        def add_analyzer(self, analyzer):
            pass

    class _ArmA:
        pass

    class _ArmB:
        pass

    def _spec(self, heuristics):
        return StrategySpec(
            name="gated",
            strategy_class=self._RecordingStrategy,
            heuristics=heuristics,
        )

    def test_min_dim_gates_arm_below_threshold(self):
        spec = self._spec([(self._ArmA, {}), (self._ArmB, {"k": 3, "gate_min_dim": 5})])
        strategy = spec.create_strategy(Rosenbrock(dims=2))
        assert [c for c, _ in strategy.added] == [self._ArmA]

    def test_min_dim_admits_arm_at_threshold_and_strips_gate_key(self):
        spec = self._spec([(self._ArmA, {}), (self._ArmB, {"k": 3, "gate_min_dim": 5})])
        strategy = spec.create_strategy(Rosenbrock(dims=5))
        assert [c for c, _ in strategy.added] == [self._ArmA, self._ArmB]
        # The reserved key must never reach the heuristic constructor.
        assert strategy.added[1][1] == {"k": 3}

    def test_max_dim_gates_arm_above_threshold(self):
        spec = self._spec([(self._ArmA, {"gate_max_dim": 3})])
        assert spec.create_strategy(Rosenbrock(dims=5)).added == []
        assert [c for c, _ in spec.create_strategy(Rosenbrock(dims=3)).added] == [self._ArmA]

    def test_gate_does_not_mutate_spec_kwargs(self):
        kwargs = {"k": 3, "gate_min_dim": 5}
        spec = self._spec([(self._ArmB, kwargs)])
        spec.create_strategy(Rosenbrock(dims=5))
        assert kwargs == {"k": 3, "gate_min_dim": 5}


if __name__ == "__main__":
    pytest.main([__file__])


def test_create_strategy_accepts_a_factory_callable():
    """``strategy_class`` may be any callable, not only a StrategyBase subclass.

    Used to pre-bind arguments a spec cannot express (e.g. StrategyPhased's
    phase list).  ``issubclass`` raises TypeError on a non-class, so the
    check has to be guarded.
    """
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import Random
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    made = {}

    def factory(problem, parse_args=False, seed=None, **kwargs):
        made["seed"] = seed
        return StrategyRoundRobin(problem, parse_args=parse_args, seed=seed, **kwargs)

    spec = StrategySpec(
        name="factory",
        strategy_class=factory,
        heuristics=[(Random, {})],
        config_overrides={"max_eval": 13},
    )
    strategy = spec.create_strategy(Rosenbrock(dim=2), seed=5)
    assert made["seed"] == 5
    assert strategy.config.max_eval == 13


def test_create_strategy_max_eval_is_visible_to_heuristic_constructors():
    """``create_strategy(max_eval=...)`` must land on the config *before* ``add``.

    Regression test for the harnesses' old ordering (build the strategy, then
    assign ``strategy.config.max_eval = budget``): a heuristic that sizes
    itself from the horizon in its constructor — ``LSHADE(NP_init="auto")`` —
    read ``Config``'s default 1000 instead of the run's budget.
    """
    from types import SimpleNamespace

    from panobbgo.heuristics import LSHADE
    from panobbgo.heuristics.lshade import _resolve_auto_np_init
    from panobbgo.strategies import StrategyRoundRobin

    budget = 100
    # ``dim=5`` keeps the auto size above its floor of 6 at both the run's
    # budget and the config default, so the "they must disagree" check below
    # stays meaningful under the ``3*dim*(budget/(500*dim))**0.25`` rule.
    dim = 5

    def np_init_for(max_eval, NP_min):
        stub = SimpleNamespace(config=SimpleNamespace(max_eval=max_eval), problem=SimpleNamespace(dim=dim))
        return _resolve_auto_np_init(stub, NP_min)

    spec = StrategySpec(
        name="auto_np",
        strategy_class=StrategyRoundRobin,
        heuristics=[(LSHADE, {"NP_init": "auto"})],
    )
    strategy = spec.create_strategy(Rosenbrock(dim=dim), seed=1, max_eval=budget)

    assert strategy.config.max_eval == budget
    lshade = next(h for h in strategy._hs if isinstance(h, LSHADE))
    assert np_init_for(budget, lshade.NP_min) != np_init_for(1000, lshade.NP_min)
    assert lshade.NP_init == np_init_for(budget, lshade.NP_min)


def test_create_strategy_max_eval_overrides_the_spec_level_override():
    """The caller's budget wins, matching the harnesses' old post-assignment."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    spec = StrategySpec(
        name="override",
        strategy_class=StrategyRoundRobin,
        heuristics=[(Random, {})],
        config_overrides={"max_eval": 13},
    )
    assert spec.create_strategy(Rosenbrock(dim=2)).config.max_eval == 13
    assert spec.create_strategy(Rosenbrock(dim=2), max_eval=77).config.max_eval == 77


def test_strategy_spec_rng_identity_defaults_to_name():
    """``seed_name`` is opt-in; default behaviour is unchanged."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    spec = StrategySpec(name="Display", strategy_class=StrategyRoundRobin, heuristics=[(Random, {})])
    assert spec.seed_name is None
    assert spec.rng_identity == "Display"
    import dataclasses

    assert dataclasses.replace(spec, seed_name="arm").rng_identity == "arm"
