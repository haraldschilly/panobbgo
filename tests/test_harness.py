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
Tests for panobbgo.harness – the agent-oriented benchmark harness.

The test suite is split into three layers:

1. **Unit tests** – metric helpers and data structures; no strategy execution.
2. **Smoke tests** – fast end-to-end run using only one tiny problem+strategy
   combination to verify the plumbing works.
3. **CLI tests** – test the benchmark_harness.py command-line interface.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
from typing import List

import numpy as np
import pytest

from panobbgo.harness import (
    BenchmarkHarness,
    ConvergencePoint,
    HarnessConfig,
    HarnessResult,
    ProblemStrategyResult,
    RunRecord,
    compare,
    compute_ert,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_run(
    success: bool,
    first_hit: int | None,
    budget: int = 100,
    func_distance: float | None = None,
    tolerance: float = 0.1,
) -> RunRecord:
    """Create a synthetic RunRecord for unit-testing metrics."""
    if func_distance is None:
        func_distance = 0.0 if success else 1.0

    convergence: List[ConvergencePoint] = []
    if first_hit is not None:
        convergence.append(
            ConvergencePoint(eval_idx=first_hit, fx=0.05, func_distance=0.05)
        )

    return RunRecord(
        problem_name="Fake",
        problem_dim=2,
        strategy_name="FakeStrategy",
        rep=0,
        seed=0,
        budget=budget,
        evaluations_used=budget,
        best_fx=0.05 if success else 999.0,
        f_opt=0.0,
        func_distance=func_distance,
        tolerance=tolerance,
        success=success,
        convergence=convergence,
        heuristic_counts={},
        duration=0.1,
    )


# ===========================================================================
# 1. Unit tests – ConvergencePoint & RunRecord
# ===========================================================================


class TestConvergencePoint:
    def test_fields(self):
        pt = ConvergencePoint(eval_idx=5, fx=2.3, func_distance=2.3)
        assert pt.eval_idx == 5
        assert pt.fx == pytest.approx(2.3)
        assert pt.func_distance == pytest.approx(2.3)


class TestRunRecord:
    def test_first_success_eval_found(self):
        run = _make_run(success=True, first_hit=40)
        assert run.first_success_eval == 40

    def test_first_success_eval_not_found(self):
        run = _make_run(success=False, first_hit=None)
        assert run.first_success_eval is None

    def test_first_success_eval_tolerance_check(self):
        """Only convergence points within tolerance are returned."""
        run = RunRecord(
            problem_name="T",
            problem_dim=2,
            strategy_name="S",
            rep=0,
            seed=0,
            budget=100,
            evaluations_used=100,
            best_fx=0.05,
            f_opt=0.0,
            func_distance=0.05,
            tolerance=0.1,
            success=True,
            convergence=[
                ConvergencePoint(
                    eval_idx=20, fx=0.5, func_distance=0.5
                ),  # still too far
                ConvergencePoint(
                    eval_idx=50, fx=0.05, func_distance=0.05
                ),  # within tol
            ],
            heuristic_counts={},
            duration=0.1,
        )
        assert run.first_success_eval == 50  # second entry, which is within tolerance


# ===========================================================================
# 2. Unit tests – ProblemStrategyResult.compute_metrics
# ===========================================================================


class TestProblemStrategyResult:
    """Test the metric computation on synthetic run records."""

    def _make_psr(self, runs: List[RunRecord]) -> ProblemStrategyResult:
        psr = ProblemStrategyResult(
            problem_name="Fake",
            problem_dim=2,
            strategy_name="FakeStrategy",
            f_opt=0.0,
            tolerance=0.1,
            budget=100,
            runs=runs,
        )
        psr.compute_metrics()
        return psr

    def test_all_success_early(self):
        """All runs succeed at eval 1 → score ≈ 1, ERT ≈ 1."""
        runs = [_make_run(True, first_hit=1, budget=100) for _ in range(5)]
        psr = self._make_psr(runs)
        assert psr.success_rate == pytest.approx(1.0)
        assert psr.ert == pytest.approx(1.0)
        assert psr.score == pytest.approx(1.0)

    def test_all_success_at_budget(self):
        """Runs succeed only at the last evaluation → small but positive score."""
        runs = [_make_run(True, first_hit=100, budget=100) for _ in range(5)]
        psr = self._make_psr(runs)
        assert psr.success_rate == pytest.approx(1.0)
        assert psr.ert == pytest.approx(100.0)
        # Score = 1 - (100-1)/100 = 0.01: small but strictly > 0 (not same as failure)
        assert psr.score == pytest.approx(0.01, abs=0.005)
        assert psr.score > 0.0

    def test_half_success(self):
        """Half succeed → success_rate = 0.5."""
        success_runs = [_make_run(True, first_hit=50, budget=100) for _ in range(3)]
        fail_runs = [_make_run(False, first_hit=None, budget=100) for _ in range(3)]
        psr = self._make_psr(success_runs + fail_runs)
        assert psr.success_rate == pytest.approx(0.5)
        assert psr.score > 0.0
        assert psr.score < 1.0

    def test_no_success(self):
        """No run succeeds → score = 0, ERT = inf."""
        runs = [_make_run(False, first_hit=None, budget=100) for _ in range(5)]
        psr = self._make_psr(runs)
        assert psr.success_rate == pytest.approx(0.0)
        assert math.isinf(psr.ert)
        assert psr.score == pytest.approx(0.0)

    def test_ert_calculation(self):
        """ERT = total / n_success for a known configuration."""
        # 2 success at eval 30, 1 failure → ERT = (30 + 30 + 100) / 2 = 80
        runs = [
            _make_run(True, first_hit=30, budget=100),
            _make_run(True, first_hit=30, budget=100),
            _make_run(False, first_hit=None, budget=100),
        ]
        psr = self._make_psr(runs)
        assert psr.ert == pytest.approx(80.0)

    def test_best_and_median_distances(self):
        runs = [
            _make_run(True, first_hit=10, func_distance=0.01, budget=100),
            _make_run(False, first_hit=None, func_distance=0.5, budget=100),
            _make_run(False, first_hit=None, func_distance=0.9, budget=100),
        ]
        psr = self._make_psr(runs)
        assert psr.best_func_distance == pytest.approx(0.01)
        assert psr.median_func_distance == pytest.approx(0.5)


# ===========================================================================
# 3. Unit tests – compute_ert helper
# ===========================================================================


class TestComputeErt:
    def test_all_success(self):
        runs = [_make_run(True, first_hit=50, budget=100) for _ in range(4)]
        ert = compute_ert(runs, budget=100)
        assert ert == pytest.approx(50.0)

    def test_no_success(self):
        runs = [_make_run(False, first_hit=None, budget=100) for _ in range(4)]
        ert = compute_ert(runs, budget=100)
        assert math.isinf(ert)

    def test_mixed(self):
        runs = [
            _make_run(True, first_hit=20, budget=100),
            _make_run(False, first_hit=None, budget=100),
        ]
        # ERT = (20 + 100) / 1 = 120
        ert = compute_ert(runs, budget=100)
        assert ert == pytest.approx(120.0)

    def test_tolerance_override(self):
        """A run with func_distance=0.05 should count as success for tol=0.1
        but not for tol=0.01."""
        run = _make_run(True, first_hit=40, func_distance=0.05, tolerance=0.1)
        assert compute_ert([run], tolerance=0.1, budget=100) == pytest.approx(40.0)
        # With tighter tolerance, this run doesn't count as success
        ert = compute_ert([run], tolerance=0.01, budget=100)
        assert math.isinf(ert)


# ===========================================================================
# 4. Unit tests – Serialisation (to_dict / save / load)
# ===========================================================================


class TestSerialization:
    def _make_result(self) -> HarnessResult:
        """Create a minimal synthetic HarnessResult."""
        run = _make_run(True, first_hit=25, budget=50)
        psr = ProblemStrategyResult(
            problem_name="Fake",
            problem_dim=2,
            strategy_name="FakeStrategy",
            f_opt=0.0,
            tolerance=0.1,
            budget=50,
            runs=[run],
        )
        psr.compute_metrics()
        return HarnessResult(
            config=HarnessConfig(mode="quick"),
            timestamp="2026-01-01T00:00:00+00:00",
            total_runs=1,
            total_duration=0.5,
            problem_strategy_results=[psr],
            composite_score=psr.score,
        )

    def test_to_dict_json_serialisable(self):
        result = self._make_result()
        d = result.to_dict()
        # Should not raise
        s = json.dumps(d)
        assert "composite_score" in s

    def test_to_dict_no_inf(self):
        """inf values should be replaced with None in the dict."""
        run = _make_run(False, first_hit=None, budget=50)
        psr = ProblemStrategyResult(
            problem_name="Fake",
            problem_dim=2,
            strategy_name="FakeStrategy",
            f_opt=0.0,
            tolerance=0.1,
            budget=50,
            runs=[run],
        )
        psr.compute_metrics()
        result = HarnessResult(
            config=HarnessConfig(mode="quick"),
            timestamp="2026-01-01T00:00:00+00:00",
            total_runs=1,
            total_duration=0.5,
            problem_strategy_results=[psr],
            composite_score=0.0,
        )
        d = result.to_dict()
        # ERT is inf for failed runs; should be serialised as None
        psr_d = d["problem_strategy_results"][0]
        assert psr_d["ert"] is None

    def test_save_and_load_round_trip(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(pathlib.Path(tmpdir) / "test_result.json")
            result.save(path)
            loaded = HarnessResult.load(path)

        assert loaded.composite_score == pytest.approx(result.composite_score, rel=1e-6)
        assert loaded.total_runs == result.total_runs
        assert loaded.config.mode == result.config.mode
        assert len(loaded.problem_strategy_results) == 1
        psr = loaded.problem_strategy_results[0]
        assert psr.problem_name == "Fake"
        assert psr.score == pytest.approx(
            result.problem_strategy_results[0].score, rel=1e-6
        )
        assert len(psr.runs) == 1
        assert psr.runs[0].success is True


# ===========================================================================
# 5. Unit tests – compare()
# ===========================================================================


class TestCompare:
    def _result(self, score_a: float, score_b: float) -> tuple:
        """Create two HarnessResults with given per-pair scores."""

        def _psr(prob: str, strat: str, score: float) -> ProblemStrategyResult:
            run = _make_run(
                score > 0,
                first_hit=max(1, int((1 - score) * 99)) if score > 0 else None,
                budget=100,
            )
            psr = ProblemStrategyResult(
                problem_name=prob,
                problem_dim=2,
                strategy_name=strat,
                f_opt=0.0,
                tolerance=0.1,
                budget=100,
                runs=[run],
                score=score,
            )
            return psr

        before = HarnessResult(
            config=HarnessConfig(mode="quick"),
            timestamp="",
            total_runs=1,
            total_duration=0.0,
            problem_strategy_results=[_psr("P1", "S1", score_a)],
            composite_score=score_a,
        )
        after = HarnessResult(
            config=HarnessConfig(mode="quick"),
            timestamp="",
            total_runs=1,
            total_duration=0.0,
            problem_strategy_results=[_psr("P1", "S1", score_b)],
            composite_score=score_b,
        )
        return before, after

    def test_improvement_detected(self):
        before, after = self._result(0.3, 0.5)
        cmp = compare(before, after, eps=0.01)
        assert cmp.delta == pytest.approx(0.2, abs=1e-9)
        assert len(cmp.improved) == 1
        assert len(cmp.degraded) == 0

    def test_degradation_detected(self):
        before, after = self._result(0.5, 0.3)
        cmp = compare(before, after, eps=0.01)
        assert cmp.delta == pytest.approx(-0.2, abs=1e-9)
        assert len(cmp.degraded) == 1
        assert len(cmp.improved) == 0

    def test_unchanged_within_eps(self):
        before, after = self._result(0.5, 0.505)
        cmp = compare(before, after, eps=0.01)
        assert len(cmp.unchanged) == 1
        assert len(cmp.improved) == 0
        assert len(cmp.degraded) == 0

    def test_mismatched_pairs_not_compared(self):
        """Pairs present only in one result should not cause false regressions."""

        def _psr(prob: str, strat: str, score: float) -> ProblemStrategyResult:
            run = _make_run(score > 0, first_hit=1 if score > 0 else None, budget=100)
            return ProblemStrategyResult(
                problem_name=prob, problem_dim=2, strategy_name=strat,
                f_opt=0.0, tolerance=0.1, budget=100, runs=[run], score=score,
            )

        before = HarnessResult(
            config=HarnessConfig(mode="quick"), timestamp="", total_runs=2,
            total_duration=0.0,
            problem_strategy_results=[_psr("P1", "S1", 0.5), _psr("P2", "S1", 0.4)],
            composite_score=0.45,
        )
        after = HarnessResult(
            config=HarnessConfig(mode="quick"), timestamp="", total_runs=2,
            total_duration=0.0,
            problem_strategy_results=[_psr("P1", "S1", 0.5), _psr("P3", "S1", 0.6)],
            composite_score=0.55,
        )
        cmp = compare(before, after, eps=0.01)
        # P1/S1 is in both → unchanged
        assert len(cmp.unchanged) == 1
        # P2/S1 only in before, P3/S1 only in after → not compared
        assert len(cmp.improved) == 0
        assert len(cmp.degraded) == 0
        assert len(cmp.only_before) == 1
        assert cmp.only_before[0][0] == "P2"
        assert len(cmp.only_after) == 1
        assert cmp.only_after[0][0] == "P3"


# ===========================================================================
# 6. Unit tests – HarnessConfig
# ===========================================================================


class TestHarnessConfig:
    def test_effective_budget_default(self):
        assert HarnessConfig(mode="quick").effective_budget() == 75
        assert HarnessConfig(mode="standard").effective_budget() == 200
        assert HarnessConfig(mode="full").effective_budget() == 500

    def test_effective_budget_override(self):
        assert HarnessConfig(mode="quick", budget=42).effective_budget() == 42

    def test_effective_reps_default(self):
        assert HarnessConfig(mode="quick").effective_reps() == 3
        assert HarnessConfig(mode="standard").effective_reps() == 5

    def test_effective_reps_override(self):
        assert HarnessConfig(mode="quick", reps=7).effective_reps() == 7


# ===========================================================================
# 7. Unit tests – BenchmarkHarness.get_problems / get_strategies
# ===========================================================================


class TestHarnessProblemsStrategies:
    def test_quick_problems(self):
        harness = BenchmarkHarness(HarnessConfig(mode="quick"))
        problems = harness.get_problems()
        assert len(problems) >= 3
        names = {p.name for p in problems}
        assert "Rosenbrock_2D" in names
        assert "Rastrigin_2D" in names

    def test_standard_has_more_than_quick(self):
        q = BenchmarkHarness(HarnessConfig(mode="quick")).get_problems()
        s = BenchmarkHarness(HarnessConfig(mode="standard")).get_problems()
        assert len(s) > len(q)

    def test_quick_strategies(self):
        harness = BenchmarkHarness(HarnessConfig(mode="quick"))
        strats = harness.get_strategies()
        assert len(strats) >= 2
        names = {s.name for s in strats}
        assert "RoundRobin_Random" in names
        assert "Rewarding_Diverse" in names

    def test_problem_filter(self):
        config = HarnessConfig(mode="quick", problems=["Rosenbrock_2D"])
        harness = BenchmarkHarness(config)
        problems = harness.get_problems()
        assert len(problems) == 1
        assert problems[0].name == "Rosenbrock_2D"

    def test_strategy_filter(self):
        config = HarnessConfig(mode="quick", strategies=["RoundRobin_Random"])
        harness = BenchmarkHarness(config)
        strats = harness.get_strategies()
        assert len(strats) == 1
        assert strats[0].name == "RoundRobin_Random"

    def test_budget_applied_to_problems(self):
        config = HarnessConfig(mode="quick", budget=33)
        harness = BenchmarkHarness(config)
        for prob in harness.get_problems():
            assert prob.max_evaluations == 33

    def test_invalid_mode(self):
        harness = BenchmarkHarness(HarnessConfig(mode="nonexistent"))
        with pytest.raises(ValueError, match="Unknown mode"):
            harness.get_problems()


# ===========================================================================
# 8. Unit tests – seed derivation
# ===========================================================================


class TestSeedDerivation:
    def test_different_problems_give_different_seeds(self):
        s1 = BenchmarkHarness._derive_seed(42, "ProbA", "Strat", 0)
        s2 = BenchmarkHarness._derive_seed(42, "ProbB", "Strat", 0)
        assert s1 != s2

    def test_different_reps_give_different_seeds(self):
        s1 = BenchmarkHarness._derive_seed(42, "ProbA", "Strat", 0)
        s2 = BenchmarkHarness._derive_seed(42, "ProbA", "Strat", 1)
        assert s1 != s2

    def test_different_base_seeds_give_different_seeds(self):
        s1 = BenchmarkHarness._derive_seed(42, "P", "S", 0)
        s2 = BenchmarkHarness._derive_seed(99, "P", "S", 0)
        assert s1 != s2

    def test_seeds_in_valid_range(self):
        for base in [0, 42, 12345, 2**31]:
            seed = BenchmarkHarness._derive_seed(base, "P", "S", 0)
            assert 0 <= seed < 2**32

    def test_deterministic(self):
        """Same inputs must always give the same seed."""
        s1 = BenchmarkHarness._derive_seed(42, "Rosenbrock_2D", "Rewarding_Diverse", 2)
        s2 = BenchmarkHarness._derive_seed(42, "Rosenbrock_2D", "Rewarding_Diverse", 2)
        assert s1 == s2


# ===========================================================================
# 9. Unit tests – convergence extraction (array-based, primary API)
# ===========================================================================


class TestBuildConvergenceFromArrays:
    """Test _build_convergence_from_arrays – the primary convergence method."""

    def test_empty_array(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(np.array([]), f_opt=0.0)
        assert trace == []

    def test_monotone_decrease(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(
            np.array([10.0, 7.0, 3.0, 1.0]), f_opt=0.0
        )
        assert len(trace) == 4
        assert trace[0].eval_idx == 1
        assert trace[0].fx == pytest.approx(10.0)
        assert trace[-1].fx == pytest.approx(1.0)

    def test_no_improvements_after_first(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(
            np.array([5.0, 8.0, 12.0]), f_opt=0.0
        )
        # Only the first value is an "improvement" from inf
        assert len(trace) == 1
        assert trace[0].fx == pytest.approx(5.0)

    def test_func_distance_computed(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(
            np.array([2.5]), f_opt=1.0
        )
        assert trace[0].func_distance == pytest.approx(1.5)

    def test_nan_skipped(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(
            np.array([float("nan"), 3.0]), f_opt=0.0
        )
        assert len(trace) == 1
        assert trace[0].eval_idx == 2

    def test_eval_idx_correct(self):
        trace = BenchmarkHarness._build_convergence_from_arrays(
            np.array([10.0, 10.0, 10.0, 5.0, 10.0, 2.0]), f_opt=0.0
        )
        # Improvements at positions 0 (first), 3, 5 → eval_idx 1, 4, 6
        assert trace[0].eval_idx == 1
        assert trace[1].eval_idx == 4
        assert trace[2].eval_idx == 6


# ===========================================================================
# 10. Unit tests – heuristic count extraction (array-based)
# ===========================================================================


class TestBuildHeuristicCountsFromArray:
    def test_counts_correct(self):
        who_arr = np.array(["H1", "H2", "H1", "H1"], dtype=object)
        counts = BenchmarkHarness._build_heuristic_counts_from_array(who_arr)
        assert counts["H1"] == 3
        assert counts["H2"] == 1

    def test_empty(self):
        assert BenchmarkHarness._build_heuristic_counts_from_array(np.array([])) == {}

    def test_none_skipped(self):
        who_arr = np.array(["H1", None, "H2"], dtype=object)
        counts = BenchmarkHarness._build_heuristic_counts_from_array(who_arr)
        assert "H1" in counts
        assert "H2" in counts
        assert None not in counts


# ===========================================================================
# 10b. Unit tests – legacy itertuples extraction helpers
# ===========================================================================


class TestExtractConvergenceLegacy:
    """Test the backward-compatible _extract_convergence that works on NamedTuples.

    The DataFrame MultiIndex flattens ("fx", 0) → "fx_0" in itertuples.
    The legacy helper also accepts plain "fx" for external callers.
    """

    def _row(self, fx: float, field: str = "fx"):
        from collections import namedtuple

        Row = namedtuple("Row", [field, "who"])
        return Row(**{field: fx, "who": "H1"})

    def test_plain_fx_field(self):
        """Accepts rows with plain 'fx' field (backward compat)."""
        rows = [self._row(fx, field="fx") for fx in [5.0, 3.0, 8.0]]
        trace = BenchmarkHarness._extract_convergence(rows, f_opt=0.0)
        assert len(trace) == 2  # 5.0 and 3.0 are improvements

    def test_fx_0_field(self):
        """Accepts rows with 'fx_0' field (pandas MultiIndex flattening)."""
        rows = [self._row(fx, field="fx_0") for fx in [5.0, 3.0, 8.0]]
        trace = BenchmarkHarness._extract_convergence(rows, f_opt=0.0)
        assert len(trace) == 2


class TestExtractHeuristicCountsLegacy:
    def _row(self, who: str, field: str = "who"):
        from collections import namedtuple

        Row = namedtuple("Row", ["fx", field])
        return Row(fx=0.5, **{field: who})

    def test_plain_who_field(self):
        rows = [self._row("H1"), self._row("H2"), self._row("H1")]
        counts = BenchmarkHarness._extract_heuristic_counts(rows)
        assert counts["H1"] == 2
        assert counts["H2"] == 1

    def test_who_0_field(self):
        rows = [self._row("H1", "who_0"), self._row("H2", "who_0")]
        counts = BenchmarkHarness._extract_heuristic_counts(rows)
        assert counts["H1"] == 1
        assert counts["H2"] == 1

    def test_empty(self):
        assert BenchmarkHarness._extract_heuristic_counts([]) == {}


# ===========================================================================
# 11. Smoke tests – end-to-end (fast, one problem × one strategy × 1 rep)
# ===========================================================================


@pytest.mark.flaky(retries=3)
class TestHarnessSmokeRun:
    """Minimal end-to-end runs to verify the plumbing without being slow."""

    def test_run_one_problem_one_strategy(self, tmp_path):
        """A single (problem, strategy, rep) should complete without error."""
        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=30,
            reps=1,
            seed=0,
            timeout_per_run=60.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        assert result.total_runs == 1
        assert 0.0 <= result.composite_score <= 1.0
        assert len(result.problem_strategy_results) == 1
        psr = result.problem_strategy_results[0]
        assert psr.problem_name == "DeJong_2D"
        assert psr.strategy_name == "RoundRobin_Random"
        assert len(psr.runs) == 1
        run = psr.runs[0]
        assert run.evaluations_used >= 0
        assert run.duration >= 0.0
        assert run.best_fx < float("inf")

    def test_save_load_round_trip(self, tmp_path):
        """Running and saving then loading should preserve the composite score."""
        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=25,
            reps=1,
            seed=1,
            timeout_per_run=60.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        path = str(tmp_path / "smoke_result.json")
        result.save(path)
        loaded = HarnessResult.load(path)

        assert loaded.composite_score == pytest.approx(result.composite_score, rel=1e-9)
        assert loaded.total_runs == result.total_runs

    def test_convergence_trace_non_empty(self, tmp_path):
        """After running we should have at least one convergence point per run."""
        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=30,
            reps=1,
            seed=2,
            timeout_per_run=60.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        for psr in result.problem_strategy_results:
            for run in psr.runs:
                if run.error is None:
                    # At least the first evaluation should appear in convergence
                    assert len(run.convergence) >= 1

    def test_heuristic_counts_populated(self, tmp_path):
        """After a run, heuristic_counts should have entries."""
        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=30,
            reps=1,
            seed=3,
            timeout_per_run=60.0,
        )
        harness = BenchmarkHarness(config)
        result = harness.run(verbose=False)

        for psr in result.problem_strategy_results:
            for run in psr.runs:
                if run.error is None and run.evaluations_used > 0:
                    assert len(run.heuristic_counts) > 0
                    total = sum(run.heuristic_counts.values())
                    assert total > 0


# ===========================================================================
# 12. CLI tests
# ===========================================================================


@pytest.mark.flaky(retries=3)
class TestCLI:
    """Tests for benchmark_harness.py CLI via direct main() invocation."""

    def test_run_quick(self, tmp_path):
        """--quick mode should produce a valid JSON file."""
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from benchmark_harness import main

        output = str(tmp_path / "cli_result.json")
        ret = main(
            [
                "run",
                "--quick",
                "--problems",
                "DeJong_2D",
                "--strategies",
                "RoundRobin_Random",
                "--budget",
                "20",
                "--reps",
                "1",
                "--output",
                output,
                "--quiet",
            ]
        )
        assert ret == 0
        assert pathlib.Path(output).exists()

        data = json.loads(pathlib.Path(output).read_text())
        assert "composite_score" in data
        assert 0.0 <= data["composite_score"] <= 1.0

    def test_score_command(self, tmp_path):
        """score command should succeed for a valid result file."""
        from benchmark_harness import main
        from panobbgo.harness import BenchmarkHarness, HarnessConfig

        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=20,
            reps=1,
            seed=10,
            timeout_per_run=60.0,
        )
        result = BenchmarkHarness(config).run(verbose=False)
        path = str(tmp_path / "score_test.json")
        result.save(path)

        ret = main(["score", path])
        assert ret == 0

    def test_compare_command_no_regression(self, tmp_path):
        """compare command returns 0 when no significant regression."""
        from benchmark_harness import main
        from panobbgo.harness import BenchmarkHarness, HarnessConfig

        def run_and_save(fname: str, seed: int) -> str:
            config = HarnessConfig(
                mode="quick",
                problems=["DeJong_2D"],
                strategies=["RoundRobin_Random"],
                budget=20,
                reps=1,
                seed=seed,
                timeout_per_run=60.0,
            )
            result = BenchmarkHarness(config).run(verbose=False)
            path = str(tmp_path / fname)
            result.save(path)
            return path

        before = run_and_save("before.json", 10)
        after = run_and_save("after.json", 11)

        ret = main(["compare", before, after])
        assert ret in (0, 2)  # 0 = no regression, 2 = regression detected

    def test_list_command(self, capsys):
        """list command should print available problems and strategies."""
        from benchmark_harness import main

        ret = main(["list", "--quick"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DeJong_2D" in captured.out
        assert "RoundRobin_Random" in captured.out

    def test_missing_file_returns_error(self, tmp_path):
        """score command on a nonexistent file should return 1."""
        from benchmark_harness import main

        ret = main(["score", str(tmp_path / "does_not_exist.json")])
        assert ret == 1

    def test_compare_missing_file_returns_error(self, tmp_path):
        """compare with a missing file should return 1."""
        from benchmark_harness import main

        ret = main(
            [
                "compare",
                str(tmp_path / "missing_before.json"),
                str(tmp_path / "missing_after.json"),
            ]
        )
        assert ret == 1


def test_statistical_compare(tmp_path):
    from benchmark_harness import main
    from panobbgo.harness import BenchmarkHarness, HarnessConfig

    def run_and_save(fname: str, seed: int) -> str:
        config = HarnessConfig(
            mode="quick",
            problems=["DeJong_2D"],
            strategies=["RoundRobin_Random"],
            budget=20,
            reps=3,
            seed=seed,
            timeout_per_run=60.0,
        )
        result = BenchmarkHarness(config).run(verbose=False)
        path = str(tmp_path / fname)
        result.save(path)
        return path

    before = run_and_save("before.json", 10)
    after = run_and_save("after.json", 42)

    # Run compare with statistical flag
    ret = main(["compare", before, after, "--statistical"])
    assert ret in (0, 2)
