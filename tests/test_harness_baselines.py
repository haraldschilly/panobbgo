#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for ``panobbgo.harness_baselines`` – external reference strategies.

Covers:

1. Structural invariants of the adapter (`BaselineStrategy` surface matches
   what the harness expects: ``.config``, ``.start()``, ``.best``,
   ``.results.results`` MultiIndex DataFrame).
2. Hard budget enforcement — no baseline is allowed to overshoot
   ``max_eval`` under any circumstance.
3. Individual optimizer wrappers actually reduce the objective on simple
   problems (Random should beat nothing; DE/Anneal should find the sphere
   optimum well within tolerance).
4. End-to-end integration via the ``BenchmarkHarness`` with
   ``include_baselines=True``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panobbgo.harness import BenchmarkHarness, HarnessConfig
from panobbgo.harness_baselines import (
    BaselineStrategy,
    RandomSearchStrategy,
    SciPyAnnealStrategy,
    SciPyDEStrategy,
    _BudgetExhausted,
    _EvaluationLog,
    _make_objective,
    make_baseline_strategies,
)
from panobbgo.lib.classic import DeJong, Rosenbrock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dejong_2d():
    """Tiny smooth problem suitable for fast baseline convergence tests."""
    return DeJong(dims=2)


# ---------------------------------------------------------------------------
# 1. Budget guard
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    def test_objective_records_and_stops(self):
        problem = _make_dejong_2d()
        log = _EvaluationLog(who="test", max_eval=5)
        objective = _make_objective(problem, log)

        for _ in range(5):
            objective(np.array([0.1, 0.2]))

        assert len(log.fxs) == 5
        with pytest.raises(_BudgetExhausted):
            objective(np.array([0.1, 0.2]))

    def test_best_tracking_ignores_non_finite(self):
        log = _EvaluationLog(who="test", max_eval=10)
        log.record(np.array([1.0, 2.0]), float("inf"))
        log.record(np.array([0.0, 0.0]), 0.5)
        log.record(np.array([3.0, 4.0]), 10.0)

        assert log.best_fx == pytest.approx(0.5)
        assert log.best_x is not None
        assert np.allclose(log.best_x, [0.0, 0.0])

    def test_objective_projects_into_box(self):
        problem = _make_dejong_2d()
        log = _EvaluationLog(who="test", max_eval=5)
        objective = _make_objective(problem, log)

        # Force a point well outside the box — objective must still succeed
        # (projection clamps it) and log the projected coordinates.
        out_of_box = problem.box[:, 1] + 1e6
        objective(out_of_box)

        assert len(log.xs) == 1
        logged = log.xs[0]
        assert np.all(logged <= problem.box[:, 1])
        assert np.all(logged >= problem.box[:, 0])


# ---------------------------------------------------------------------------
# 2. Adapter surface
# ---------------------------------------------------------------------------


class TestBaselineStrategySurface:
    def test_config_defaults(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        assert hasattr(strat.config, "max_eval")
        assert hasattr(strat.config, "evaluation_method")

    def test_add_methods_noop(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        # Must not raise even though baselines do not consume heuristics.
        strat.add(object, foo="bar")
        strat.add_analyzer(object())

    def test_parse_args_kw_accepted(self):
        # StrategySpec.create_strategy() calls cls(problem, parse_args=False).
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem, parse_args=False)
        assert strat.problem is problem

    def test_abstract_optimize_raises(self):
        problem = _make_dejong_2d()
        strat = BaselineStrategy(problem)
        with pytest.raises(NotImplementedError):
            strat.start()

    def test_results_df_has_multiindex_columns(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        strat.config.max_eval = 8
        strat.start()

        df = strat.results.results
        assert df is not None
        assert isinstance(df.columns, pd.MultiIndex)
        # The harness relies on these exact top-level names.
        top = set(df.columns.get_level_values(0))
        assert {"x", "fx", "who"}.issubset(top)
        assert len(df) == 8

    def test_best_populated_after_start(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        strat.config.max_eval = 10
        strat.start()

        best = strat.best
        assert best is not None
        assert best.fx is not None and np.isfinite(best.fx)
        assert best.who == "Random"


# ---------------------------------------------------------------------------
# 3. Optimizer wrappers – do they actually optimize?
# ---------------------------------------------------------------------------


class TestRandomSearch:
    def test_uniform_in_box(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        strat.config.max_eval = 50
        np.random.seed(0)
        strat.start()

        df = strat.results.results
        assert df is not None
        xs = df["x"].values
        # Every sampled point must be inside the problem's box.
        assert np.all(xs >= problem.box[:, 0])
        assert np.all(xs <= problem.box[:, 1])

    def test_respects_hard_budget(self):
        problem = _make_dejong_2d()
        strat = RandomSearchStrategy(problem)
        strat.config.max_eval = 17  # odd number to catch off-by-one bugs
        np.random.seed(1)
        strat.start()

        assert strat.results.results is not None
        assert len(strat.results.results) == 17


class TestSciPyDE:
    def test_converges_on_sphere(self):
        """DE should get close to the origin on DeJong with modest budget."""
        problem = _make_dejong_2d()
        strat = SciPyDEStrategy(problem)
        strat.config.max_eval = 150
        np.random.seed(7)
        strat.start()

        best = strat.best
        assert best is not None
        # Loose bound — exact value depends on scipy version's DE defaults,
        # but "much better than random" is the floor we need to prove.
        assert best.fx is not None and best.fx < 0.5

    def test_hard_budget_never_exceeded(self):
        problem = _make_dejong_2d()
        strat = SciPyDEStrategy(problem)
        strat.config.max_eval = 40
        np.random.seed(11)
        strat.start()

        df = strat.results.results
        assert df is not None
        assert len(df) <= 40

    def test_who_label_propagates(self):
        problem = _make_dejong_2d()
        strat = SciPyDEStrategy(problem)
        strat.config.max_eval = 30
        np.random.seed(3)
        strat.start()

        df = strat.results.results
        assert df is not None
        who = df["who"].values.ravel()
        assert all(w == "SciPyDE" for w in who)


class TestSciPyAnneal:
    def test_converges_on_sphere(self):
        problem = _make_dejong_2d()
        strat = SciPyAnnealStrategy(problem)
        strat.config.max_eval = 200
        np.random.seed(19)
        strat.start()

        best = strat.best
        assert best is not None
        assert best.fx is not None and best.fx < 0.1

    def test_hard_budget_never_exceeded(self):
        problem = _make_dejong_2d()
        strat = SciPyAnnealStrategy(problem)
        strat.config.max_eval = 60
        np.random.seed(5)
        strat.start()

        df = strat.results.results
        assert df is not None
        assert len(df) <= 60


# ---------------------------------------------------------------------------
# 4. Harness integration
# ---------------------------------------------------------------------------


class TestHarnessIntegration:
    def test_make_baseline_strategies_count_and_names(self):
        specs = make_baseline_strategies()
        names = {s.name for s in specs}
        assert names == {
            "Baseline_Random",
            "Baseline_SciPyDE",
            "Baseline_SciPyAnneal",
        }
        # Baselines register zero heuristics — the spec carries the solver
        # class, not a portfolio composition.
        for s in specs:
            assert s.heuristics == []

    def test_include_baselines_appends_to_strategy_list(self):
        config = HarnessConfig(mode="quick", include_baselines=True)
        harness = BenchmarkHarness(config)
        strategies = harness.get_strategies()
        names = {s.name for s in strategies}
        assert {
            "Baseline_Random",
            "Baseline_SciPyDE",
            "Baseline_SciPyAnneal",
        }.issubset(names)

    def test_include_baselines_default_false(self):
        config = HarnessConfig(mode="quick")
        harness = BenchmarkHarness(config)
        names = {s.name for s in harness.get_strategies()}
        assert "Baseline_Random" not in names

    @pytest.mark.flaky(retries=2)
    def test_end_to_end_random_baseline_runs(self):
        """Run a tiny harness with only the Random baseline and confirm the
        pipeline produces a finite composite score."""
        config = HarnessConfig(
            mode="quick",
            budget=30,
            reps=1,
            seed=42,
            strategies=["Baseline_Random"],
            include_baselines=True,
            timeout_per_run=30.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        assert result.total_runs == len(harness.get_problems())
        assert 0.0 <= result.composite_score <= 1.0
        # No run should error out — Random is pure NumPy.
        for psr in result.problem_strategy_results:
            for run in psr.runs:
                assert run.error is None
                assert run.evaluations_used == 30

    @pytest.mark.flaky(retries=2)
    def test_baseline_strategy_filter_works(self):
        """Filtering by baseline name produces only that baseline's results."""
        config = HarnessConfig(
            mode="quick",
            budget=20,
            reps=1,
            seed=42,
            strategies=["Baseline_Random"],
            problems=["Rosenbrock_2D"],
            include_baselines=True,
            timeout_per_run=30.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        assert result.total_runs == 1
        assert result.problem_strategy_results[0].strategy_name == "Baseline_Random"


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_random_search_is_seed_reproducible(self):
        problem = Rosenbrock(dims=2)

        np.random.seed(123)
        s1 = RandomSearchStrategy(problem)
        s1.config.max_eval = 20
        s1.start()

        np.random.seed(123)
        s2 = RandomSearchStrategy(problem)
        s2.config.max_eval = 20
        s2.start()

        assert s1.best is not None and s2.best is not None
        assert s1.best.fx == pytest.approx(s2.best.fx)
        df1 = s1.results.results
        df2 = s2.results.results
        assert df1 is not None and df2 is not None
        np.testing.assert_allclose(df1["x"].values, df2["x"].values)
