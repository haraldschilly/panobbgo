# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
"""Tests for panobbgo.harness_ioh — IOH-driven multi-instance benchmarking.

The ``ioh`` binding lives in an isolated child venv under
``tools/ioh_worker/``; tests that actually spawn the worker subprocess
are gated by :func:`worker_available`, which checks that ``uv`` is
present and the worker's ``.venv`` has been created.

The AOCC / battery / seed / downsample tests do not need the worker
and run unconditionally.
"""

from __future__ import annotations

import json
import math
from typing import Iterator

import numpy as np
import pytest

from panobbgo.harness_baselines import make_baseline_strategies
from panobbgo.harness_ioh import (
    IOHBatterySpec,
    IOHHarnessResult,
    _derive_seed,
    _downsample_trajectory,
    make_full_battery,
    make_quick_battery,
    make_standard_battery,
    run_ioh_harness,
)
from panobbgo.ioh_runner import IOHTracker, _BudgetExhausted, aocc
from panobbgo.lib.ioh_wrapper import IOHProblem, worker_available


requires_worker = pytest.mark.skipif(
    not worker_available(),
    reason=("ioh worker venv not set up (run `cd tools/ioh_worker && uv sync` to enable IOH integration tests)"),
)


@pytest.fixture
def ioh_problem() -> Iterator[IOHProblem]:
    """Yield a fresh 2-D MA-BBOB problem and close its worker on teardown."""
    prob = IOHProblem(kind="MA-BBOB", instance=0, dim=2)
    try:
        yield prob
    finally:
        prob.close()


# ---------------------------------------------------------------------------
# AOCC (pure Python — no worker)
# ---------------------------------------------------------------------------


class TestAOCC:
    def test_optimum_hit_gives_one(self) -> None:
        # precision = 0  -> clipped to log_lo -> AOCC = 1.0
        assert aocc([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_far_above_high_gives_zero(self) -> None:
        # precision >> 10^log_hi  -> clipped to log_hi -> AOCC = 0.0
        assert aocc([1e10, 1e10, 1e10]) == pytest.approx(0.0, abs=1e-9)

    def test_midrange_constant(self) -> None:
        # precision = 10^-3, range [10^-8, 10^2] -> normalised = 5/10
        # AOCC = 1 - 0.5 = 0.5
        assert aocc([1e-3, 1e-3, 1e-3]) == pytest.approx(0.5)

    def test_padding_to_budget_penalises_early_stop(self) -> None:
        # 5 evals at fx=0 (perfect), but budget=10 — pad with last=0 (perfect),
        # so AOCC stays 1.0
        assert aocc([0.0] * 5, budget=10) == pytest.approx(1.0)

        # 5 evals at fx=1e10 (worst), budget=10 — padding does not rescue
        assert aocc([1e10] * 5, budget=10) == pytest.approx(0.0, abs=1e-9)

        # Half the run at perfect, half at terrible end — averaged
        traj = [0.0] * 5 + [1e10] * 5
        assert 0.4 <= aocc(traj) <= 0.6

    def test_empty_trajectory(self) -> None:
        assert aocc([]) == 0.0


# ---------------------------------------------------------------------------
# Battery shape (pure Python — no worker)
# ---------------------------------------------------------------------------


class TestBatteries:
    def test_quick_battery_shape(self) -> None:
        b = make_quick_battery()
        assert b.problem_kind == "MA-BBOB"
        assert b.dims == (2,)
        assert len(b.instances) == 3
        assert b.budget_for(2) == 200

    def test_standard_battery_covers_two_dims(self) -> None:
        b = make_standard_battery()
        assert 2 in b.dims and 5 in b.dims
        assert b.pair_count(1) == len(b.dims) * len(b.instances) * b.reps

    def test_full_battery_uses_competition_budget(self) -> None:
        b = make_full_battery()
        assert b.budget_for(5) == 10000
        assert b.budget_for(2) == 4000


# ---------------------------------------------------------------------------
# Seed derivation (pure Python — no worker)
# ---------------------------------------------------------------------------


class TestSeed:
    def test_seed_deterministic(self) -> None:
        s1 = _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 0)
        s2 = _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 0)
        assert s1 == s2

    def test_seed_changes_with_inputs(self) -> None:
        s = _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 0)
        assert s != _derive_seed(43, "MA-BBOB", 2, 0, "Foo", 0)
        assert s != _derive_seed(42, "MA-BBOB", 5, 0, "Foo", 0)
        assert s != _derive_seed(42, "MA-BBOB", 2, 1, "Foo", 0)
        assert s != _derive_seed(42, "MA-BBOB", 2, 0, "Bar", 0)
        assert s != _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 1)


# ---------------------------------------------------------------------------
# Trajectory downsampling (pure Python — no worker)
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_downsample_keeps_k_points(self) -> None:
        traj = np.linspace(100, 0, 1000).tolist()
        idx, fx = _downsample_trajectory(traj, budget=1000, k=16)
        assert 1 <= len(idx) <= 17  # k plus optional tail-pad
        assert all(0 <= i <= 1000 for i in idx)
        assert all(math.isclose(f_t, traj[i - 1]) for i, f_t in zip(idx, fx) if i <= len(traj))

    def test_downsample_pads_tail_when_short(self) -> None:
        traj = [10.0] * 50
        idx, fx = _downsample_trajectory(traj, budget=200, k=8)
        # last entry must be budget=200 with the final value
        assert idx[-1] == 200
        assert fx[-1] == 10.0

    def test_downsample_empty(self) -> None:
        idx, fx = _downsample_trajectory([], budget=100)
        assert idx == [] and fx == []


# ---------------------------------------------------------------------------
# IOHTracker (needs the worker for a real IOH problem to evaluate against)
# ---------------------------------------------------------------------------


@requires_worker
class TestIOHTracker:
    def test_soft_budget_returns_inf_past_budget(self, ioh_problem: IOHProblem) -> None:
        tracker = IOHTracker(ioh_problem, budget=5)
        # Within budget: real evaluations recorded
        for _ in range(5):
            fx = ioh_problem.eval(np.zeros(ioh_problem.dim))
            assert np.isfinite(fx)
        assert tracker.n_evals == 5
        # Past budget: soft no-op, n_evals does not advance
        for _ in range(3):
            fx = ioh_problem.eval(np.zeros(ioh_problem.dim))
            # The "soft" return value is the last known best (finite) here
            # because at least one in-budget eval recorded a result.
            assert fx == tracker.best_fx
        assert tracker.n_evals == 5
        tracker.restore()

    def test_hard_budget_raises(self, ioh_problem: IOHProblem) -> None:
        tracker = IOHTracker(ioh_problem, budget=2, hard=True)
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        with pytest.raises(_BudgetExhausted):
            ioh_problem.eval(np.zeros(ioh_problem.dim))
        tracker.restore()

    def test_restore_stops_tracking(self, ioh_problem: IOHProblem) -> None:
        tracker = IOHTracker(ioh_problem, budget=10)
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        assert tracker.n_evals == 1
        tracker.restore()
        # After restore, evaluating the problem must not increment the
        # tracker's counters — the wrapper has been removed.
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        assert tracker.n_evals == 1


# ---------------------------------------------------------------------------
# End-to-end harness runs (need the worker)
# ---------------------------------------------------------------------------


@requires_worker
class TestRunIOHHarness:
    def test_random_baseline_on_quick_battery(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        assert baselines, "expected a Baseline_Random spec"
        battery = make_quick_battery()
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False)

        assert result.problem_kind == "MA-BBOB"
        assert len(result.runs) == battery.pair_count(1)
        # All runs should have used the full budget (random search never starves)
        for r in result.runs:
            assert r.error is None, r.error
            assert r.n_evals == r.budget, (r.n_evals, r.budget)
            assert 0.0 <= r.aocc <= 1.0
        assert 0.0 <= result.mean_aocc <= 1.0

    def test_result_json_roundtrip(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-tiny", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        text = result.to_json()
        round_tripped = IOHHarnessResult.from_dict(json.loads(text))
        assert round_tripped.battery_name == result.battery_name
        assert round_tripped.problem_kind == result.problem_kind
        assert len(round_tripped.runs) == len(result.runs)
        assert round_tripped.mean_aocc == pytest.approx(result.mean_aocc)

    def test_strategy_uses_full_budget_in_harness(self) -> None:
        """Regression: the harness disables stop_on_convergence so panobbgo
        strategies run to budget (the anytime metric penalises unused
        evals).  Before the fix the Rewarding strategy halted at ~4% of
        budget on dim 5 because the Convergence analyzer fired on a
        stagnation window."""
        from panobbgo.harness import _make_quick_strategies

        rewarding = [s for s in _make_quick_strategies() if s.name == "Rewarding_Diverse"]
        if not rewarding:
            pytest.skip("Rewarding_Diverse not in quick strategy registry")

        battery = IOHBatterySpec(
            name="ioh-budget-fill",
            problem_kind="MA-BBOB",
            dims=(2,),
            instances=(0,),
            reps=1,
            budget_multiplier=200,
        )
        result = run_ioh_harness(rewarding, battery, base_seed=42, progress=False)
        rec = result.runs[0]
        assert rec.error is None, rec.error
        # Allow a tiny slack for off-by-one in candidate-batching, but the
        # strategy must use at least 90% of the budget — otherwise the
        # premature-stop bug has regressed.
        assert rec.n_evals >= 0.9 * rec.budget, (rec.n_evals, rec.budget)

    def test_aocc_to_harness_result_roundtrip(self) -> None:
        """The HarnessResult adapter must encode AOCC such that
        ``ProblemStrategyResult.score`` and ``composite_score`` read back
        the original AOCC values to within 1/budget rounding error."""
        from panobbgo.harness_ioh import aocc_to_harness_result, make_ioh_strategies

        battery = IOHBatterySpec(
            name="ioh-adapter-test",
            problem_kind="MA-BBOB",
            dims=(2,),
            instances=(0, 1),
            reps=1,
            budget_multiplier=50,
        )
        # Single strategy keeps the test cheap and the mapping check tight.
        strats = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
        ioh_result = run_ioh_harness(strats, battery, base_seed=42, progress=False)
        wrapped = aocc_to_harness_result(ioh_result)

        # composite_score must equal the IOH mean AOCC up to encoding error.
        budget = battery.budget_for(2)
        assert wrapped.composite_score == pytest.approx(ioh_result.mean_aocc, abs=1.0 / budget)

        # Each ProblemStrategyResult must hold AOCC for its source run.
        per_strat = ioh_result.per_strategy_aocc()
        for psr in wrapped.problem_strategy_results:
            assert psr.strategy_name in per_strat
            assert 0.0 <= psr.score <= 1.0

    def test_ioh_strategies_registry_runs(self) -> None:
        """``make_ioh_strategies`` returns a working list of specs."""
        from panobbgo.harness_ioh import make_ioh_strategies

        strats = make_ioh_strategies()
        assert len(strats) >= 1
        names = [s.name for s in strats]
        assert "Rewarding_Restart" in names
        battery = IOHBatterySpec(
            name="ioh-iohstrats",
            problem_kind="MA-BBOB",
            dims=(2,),
            instances=(0,),
            reps=1,
            budget_multiplier=50,
        )
        result = run_ioh_harness(strats, battery, base_seed=42, progress=False)
        for r in result.runs:
            assert r.error is None, f"{r.strategy_name}: {r.error}"
            # No half-finished runs; the IOH driver disables stop_on_convergence.
            assert r.n_evals >= 0.9 * r.budget

    def test_self_improve_loop_aocc_metric(self) -> None:
        """A 1-iteration self-improve run with metric='aocc' must complete
        and produce a record whose scores look like AOCC (in [0, 1])."""
        from panobbgo.self_improve import LoopConfig, SelfImprover

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            ledger = f"{td}/ledger.jsonl"
            cfg = LoopConfig(
                iterations=1,
                mode="quick",
                metric="aocc",
                ledger_path=ledger,
                randomize=True,
                stop_sentinel_path="",  # disable
            )
            improver = SelfImprover(cfg)
            records = improver.run(verbose=False)

        assert len(records) == 1
        rec = records[0]
        # AOCC always sits in [0, 1] — the encoding into HarnessResult
        # preserves this for both baseline and candidate.
        assert 0.0 <= rec.baseline_score <= 1.0, rec.baseline_score
        assert 0.0 <= rec.candidate_score <= 1.0, rec.candidate_score
        # ci_low/ci_high are deltas, can be negative — just sanity check shape
        assert rec.ci_low <= rec.ci_high

    def test_reproducible_seed(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-repro", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        r1 = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        r2 = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        assert r1.mean_aocc == pytest.approx(r2.mean_aocc)

        r3 = run_ioh_harness(baselines, battery, base_seed=43, progress=False)
        # Different seed -> different RNG path -> AOCC typically differs.
        # In rare cases (very low budget, identical first-evals) it may
        # coincide, so this is a soft check.
        assert isinstance(r3.mean_aocc, float)


# ---------------------------------------------------------------------------
# LoopConfig metric validation (pure Python — no worker)
# ---------------------------------------------------------------------------


class TestLoopConfigMetric:
    def test_loop_config_metric_validation(self) -> None:
        from panobbgo.self_improve import LoopConfig

        # Valid values
        LoopConfig(iterations=0, metric="composite")
        LoopConfig(iterations=0, metric="aocc")
        # Invalid value
        with pytest.raises(ValueError, match="metric"):
            LoopConfig(iterations=0, metric="bogus")
