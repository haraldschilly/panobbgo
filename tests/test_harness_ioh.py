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
from types import SimpleNamespace
from typing import ClassVar, Iterator, List

import numpy as np
import pytest

from panobbgo.benchmark import StrategySpec
from panobbgo.harness_baselines import make_baseline_strategies
from panobbgo.harness_ioh import (
    ALL_BBOB_FIDS,
    BBOB_CLASS_OF_FID,
    BBOB_CLASS_ORDER,
    DEFAULT_DECISION_SEEDS,
    IOHBatterySpec,
    IOHHarnessResult,
    IOHMultiSeedResult,
    IOHRunRecord,
    _derive_noise_seed,
    _derive_seed,
    _downsample_trajectory,
    bbob_class_of,
    make_bbob_battery,
    make_full_battery,
    make_quick_battery,
    make_standard_battery,
    paired_seed_stats,
    run_ioh_harness,
    run_ioh_harness_multi_seed,
)
from panobbgo.heuristics import LSHADE
from panobbgo.heuristics.lshade import _resolve_auto_np_init
from panobbgo.ioh_runner import IOHTracker, _BudgetExhausted, aocc
from panobbgo.strategies import StrategyRoundRobin
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

    def test_trace_longer_than_budget_is_truncated(self) -> None:
        # Evaluations past the budget are never scored: a late perfect hit
        # at eval 11 on a budget of 10 must not lift the score.
        assert aocc([1e10] * 10 + [0.0], budget=10) == pytest.approx(0.0, abs=1e-9)
        assert aocc([0.0] * 10 + [1e10] * 5, budget=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# IOHTracker thread safety (fake problem — no worker)
# ---------------------------------------------------------------------------


class _SlowSphere:
    """Stand-in for an IOH problem: ``eval`` yields the GIL mid-call."""

    dim = 2

    def __init__(self) -> None:
        import threading

        self.calls = 0
        self._calls_lock = threading.Lock()

    def eval(self, x: np.ndarray) -> float:
        import time

        with self._calls_lock:
            self.calls += 1
        time.sleep(1e-4)
        return float(np.sum(np.asarray(x) ** 2))


class TestIOHTrackerThreads:
    def test_concurrent_evals_respect_budget_and_stay_monotone(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        prob = _SlowSphere()
        tracker = IOHTracker(prob, budget=200)
        rng = np.random.default_rng(0)
        xs = [rng.uniform(-5, 5, size=2) for _ in range(400)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(prob.eval, xs))
        tracker.restore()
        # Exactly the budget reached the objective and was recorded.
        assert prob.calls == 200
        assert tracker.n_evals == 200
        assert len(tracker.best_so_far) == 200
        trace = np.asarray(tracker.best_so_far)
        assert np.all(np.diff(trace) <= 0)
        assert trace[-1] == tracker.best_fx

    def test_on_timeout_runs_outside_the_lock(self) -> None:
        prob = _SlowSphere()
        tracker = IOHTracker(prob, budget=10, timeout_s=0.0)
        seen = []
        # A callback that re-enters the tracker would deadlock under the lock.
        tracker.on_timeout = lambda: seen.append(prob.eval(np.zeros(2)))
        prob.eval(np.zeros(2))
        tracker.restore()
        assert tracker.timed_out and len(seen) == 1 and tracker.n_evals == 0

    def test_failed_eval_releases_its_slot(self) -> None:
        class Flaky(_SlowSphere):
            def eval(self, x: np.ndarray) -> float:
                if self.calls == 0:
                    self.calls += 1
                    raise RuntimeError("worker hiccup")
                return super().eval(x)

        prob = Flaky()
        tracker = IOHTracker(prob, budget=3)
        with pytest.raises(RuntimeError):
            prob.eval(np.zeros(2))
        for _ in range(5):
            prob.eval(np.ones(2))
        tracker.restore()
        assert tracker.n_evals == 3


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

    def test_deadline_stops_counting(self, ioh_problem: IOHProblem) -> None:
        tracker = IOHTracker(ioh_problem, budget=10, timeout_s=0.0)
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        assert tracker.timed_out
        assert tracker.n_evals == 0
        tracker.restore()

    def test_no_deadline_by_default(self, ioh_problem: IOHProblem) -> None:
        tracker = IOHTracker(ioh_problem, budget=10)
        ioh_problem.eval(np.zeros(ioh_problem.dim))
        assert not tracker.timed_out
        assert tracker.n_evals == 1
        tracker.restore()


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

    def test_ioh_strategies_registry_runs(self) -> None:
        """``make_ioh_strategies`` returns a working list of specs."""
        from panobbgo.harness_ioh import make_ioh_strategies

        strats = make_ioh_strategies()
        assert len(strats) >= 1
        names = [s.name for s in strats]
        # The competition candidate, the pure-random floor and the
        # portfolio control (see make_ioh_strategies).
        assert "RoundRobin_CMAES" in names
        assert "RoundRobin_Random" in names
        assert "Blocks_warm_CMAES_JSO" in names
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


# ---------------------------------------------------------------------------
# Multi-seed batteries & paired decision stats (pure Python — no worker)
# ---------------------------------------------------------------------------


def _mk_seed_result(per_strategy_aocc: dict, battery: str = "ioh-ms") -> IOHHarnessResult:
    """Build a synthetic single-seed result with one run per strategy."""
    runs = [
        IOHRunRecord(
            problem_kind="MA-BBOB",
            dim=2,
            instance=0,
            strategy_name=name,
            rep=0,
            budget=100,
            n_evals=100,
            best_fx=0.5,
            f_opt=0.0,
            aocc=score,
            elapsed_s=0.01,
            seed=1,
        )
        for name, score in per_strategy_aocc.items()
    ]
    return IOHHarnessResult(battery_name=battery, problem_kind="MA-BBOB", log_lo=-8.0, log_hi=2.0, runs=runs)


def _mk_multi(seed_to_scores: dict, battery: str = "ioh-ms") -> IOHMultiSeedResult:
    """Build a synthetic multi-seed result from ``{seed: {strategy: aocc}}``."""
    seeds = list(seed_to_scores)
    return IOHMultiSeedResult(
        battery_name=battery,
        problem_kind="MA-BBOB",
        log_lo=-8.0,
        log_hi=2.0,
        base_seeds=seeds,
        results=[_mk_seed_result(seed_to_scores[s], battery=battery) for s in seeds],
    )


class TestMultiSeedResult:
    def test_mean_and_per_strategy_matrix(self) -> None:
        ms = _mk_multi({42: {"A": 0.30, "B": 0.20}, 7: {"A": 0.40, "B": 0.10}})
        assert ms.mean_aocc == pytest.approx(0.25)
        matrix = ms.per_strategy_seed_aocc()
        assert matrix == {"A": [0.30, 0.40], "B": [0.20, 0.10]}
        assert ms.per_strategy_aocc() == {"A": pytest.approx(0.35), "B": pytest.approx(0.15)}

    def test_ragged_strategy_dropped(self) -> None:
        # "B" only ran at seed 42 → cannot be paired, dropped from the matrix.
        ms = _mk_multi({42: {"A": 0.30, "B": 0.20}, 7: {"A": 0.40}})
        assert set(ms.per_strategy_seed_aocc()) == {"A"}

    def test_json_roundtrip_and_discriminator(self) -> None:
        ms = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}, 1234: {"A": 0.35}})
        d = json.loads(ms.to_json())
        assert d["multi_seed"] is True
        assert d["base_seeds"] == [42, 7, 1234]
        rt = IOHMultiSeedResult.from_dict(d)
        assert rt.base_seeds == ms.base_seeds
        assert rt.mean_aocc == pytest.approx(ms.mean_aocc)
        assert rt.per_strategy_seed_aocc() == ms.per_strategy_seed_aocc()

    def test_default_decision_seeds_shape(self) -> None:
        # The canonical 12-seed roster from the 2026-08-03 protocol.
        assert len(DEFAULT_DECISION_SEEDS) == 12
        assert len(set(DEFAULT_DECISION_SEEDS)) == 12
        assert 42 in DEFAULT_DECISION_SEEDS


class TestErroredRunsCount:
    """Crashed and timed-out runs stay in the aggregates (as ``_screen.fold`` counts them)."""

    @staticmethod
    def _result() -> IOHHarnessResult:
        from dataclasses import replace

        res = _mk_seed_result({"A": 0.6})
        base = res.runs[0]
        res.runs += [
            # timed out: scored on the trace up to the deadline
            replace(base, dim=5, aocc=0.3, error="TimeoutError: stopped after 1s at 40/100 evals"),
            # crashed: nothing scored
            replace(base, dim=10, aocc=0.0, error="RuntimeError: boom"),
            # a wedged worker is a crash, not a (scored) timeout
            replace(base, dim=10, rep=1, aocc=0.0, error="TimeoutError: IOH worker did not answer in time"),
        ]
        return res

    def test_means_include_errored_runs(self) -> None:
        res = self._result()
        assert res.mean_aocc == pytest.approx(0.9 / 4)
        assert res.per_strategy_aocc() == {"A": pytest.approx(0.9 / 4)}
        assert res.per_strategy_per_dim_aocc() == {
            ("A", 2): pytest.approx(0.6),
            ("A", 5): pytest.approx(0.3),
            ("A", 10): pytest.approx(0.0),
        }
        assert res.per_strategy_counts() == {"A": {"n": 4, "crashed": 2, "timed_out": 1, "ended_early": 0}}

    def test_an_early_end_is_neither_a_crash_nor_a_timeout(self) -> None:
        from dataclasses import replace

        from panobbgo.harness_ioh import EARLY_END_ERROR_PREFIX

        res = self._result()
        res.runs.append(replace(res.runs[0], rep=2, aocc=0.2, error=f"{EARLY_END_ERROR_PREFIX} 8/100 evals"))
        assert res.per_strategy_counts()["A"] == {"n": 5, "crashed": 2, "timed_out": 1, "ended_early": 1}

    def test_summary_reports_counts(self, capsys) -> None:
        self._result().print_summary()
        out = capsys.readouterr().out
        assert "2 crashed (AOCC 0), 1 timed out" in out
        assert "(n=4, 2 crashed, 1 timed out)" in out


class TestPairedSeedStats:
    def test_constant_shift(self) -> None:
        before = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}, 1234: {"A": 0.35}})
        after = _mk_multi({42: {"A": 0.31}, 7: {"A": 0.41}, 1234: {"A": 0.36}})
        st = paired_seed_stats(before, after)["A"]
        assert st["n"] == 3
        assert st["mean_delta"] == pytest.approx(0.01)
        assert st["sd"] == pytest.approx(0.0, abs=1e-12)
        # Zero-variance deltas → CI collapses onto the mean, excludes 0.
        assert st["ci_low"] == pytest.approx(0.01)
        assert st["ci_high"] == pytest.approx(0.01)
        assert st["per_seed_delta"] == pytest.approx([0.01, 0.01, 0.01])

    def test_pairs_by_seed_value_not_position(self) -> None:
        before = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}})
        # after lists the seeds in the opposite order; pairing must
        # still match 42↔42 and 7↔7.
        after = _mk_multi({7: {"A": 0.42}, 42: {"A": 0.33}})
        st = paired_seed_stats(before, after)["A"]
        assert st["seeds"] == [42, 7]
        assert st["per_seed_delta"] == pytest.approx([0.03, 0.02])

    def test_noise_straddles_zero(self) -> None:
        before = _mk_multi({s: {"A": 0.30} for s in (1, 2, 3, 4)})
        after = _mk_multi({1: {"A": 0.32}, 2: {"A": 0.28}, 3: {"A": 0.31}, 4: {"A": 0.29}})
        st = paired_seed_stats(before, after)["A"]
        assert st["mean_delta"] == pytest.approx(0.0)
        assert st["ci_low"] < 0 < st["ci_high"]

    def test_single_common_seed_has_nan_ci(self) -> None:
        before = _mk_multi({42: {"A": 0.30}})
        after = _mk_multi({42: {"A": 0.35}})
        st = paired_seed_stats(before, after)["A"]
        assert st["n"] == 1
        assert st["mean_delta"] == pytest.approx(0.05)
        assert math.isnan(st["sd"]) and math.isnan(st["ci_low"]) and math.isnan(st["ci_high"])

    def test_no_common_seeds_raises(self) -> None:
        before = _mk_multi({42: {"A": 0.30}})
        after = _mk_multi({7: {"A": 0.35}})
        with pytest.raises(ValueError, match="no common base seeds"):
            paired_seed_stats(before, after)

    def test_partial_seed_overlap(self) -> None:
        before = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}, 99: {"A": 0.50}})
        after = _mk_multi({7: {"A": 0.45}, 42: {"A": 0.32}, 555: {"A": 0.60}})
        st = paired_seed_stats(before, after)["A"]
        assert st["seeds"] == [42, 7]
        assert st["n"] == 2

    def test_strategies_intersected(self) -> None:
        before = _mk_multi({42: {"A": 0.3, "B": 0.2}, 7: {"A": 0.4, "B": 0.3}})
        after = _mk_multi({42: {"A": 0.3, "C": 0.1}, 7: {"A": 0.4, "C": 0.2}})
        assert set(paired_seed_stats(before, after)) == {"A"}


@requires_worker
class TestRunIOHHarnessMultiSeed:
    def test_two_seed_run(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-ms-tiny", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        ms = run_ioh_harness_multi_seed(baselines, battery, [42, 7], progress=False)
        assert ms.base_seeds == [42, 7]
        assert len(ms.results) == 2
        matrix = ms.per_strategy_seed_aocc()
        assert "Baseline_Random" in matrix
        assert len(matrix["Baseline_Random"]) == 2
        # Null self-compare: identical results pair to exactly-zero deltas.
        st = paired_seed_stats(ms, ms)["Baseline_Random"]
        assert st["mean_delta"] == pytest.approx(0.0, abs=1e-12)

    def test_empty_seeds_raises(self) -> None:
        with pytest.raises(ValueError, match="base_seeds"):
            run_ioh_harness_multi_seed([], make_quick_battery(), [])


class TestIOHBenchmarkCompareCLI:
    """Format dispatch of ``scripts/ioh_benchmark.py compare`` (no worker)."""

    @staticmethod
    def _cli():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
        spec = importlib.util.spec_from_file_location("ioh_benchmark_cli", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_compare_multi_seed_files(self, tmp_path, capsys) -> None:
        cli = self._cli()
        before = _mk_multi({42: {"A": 0.30, "B": 0.20}, 7: {"A": 0.40, "B": 0.30}})
        after = _mk_multi({42: {"A": 0.32, "B": 0.20}, 7: {"A": 0.42, "B": 0.30}})
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        rc = cli.main(["compare", str(b_path), str(a_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Paired multi-seed compare" in out
        assert "2 common seed(s)" in out
        # A shifted by a constant +0.02 → improved; B untouched → flat control.
        assert "+ improved" in out
        assert "per-seed deltas" in out

    def test_compare_mixed_formats_errors(self, tmp_path, capsys) -> None:
        cli = self._cli()
        multi = _mk_multi({42: {"A": 0.30}})
        single = _mk_seed_result({"A": 0.30})
        m_path, s_path = tmp_path / "m.json", tmp_path / "s.json"
        m_path.write_text(multi.to_json())
        s_path.write_text(single.to_json())
        rc = cli.main(["compare", str(m_path), str(s_path)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Cannot compare" in err

    def test_compare_single_seed_files_unchanged(self, tmp_path, capsys) -> None:
        cli = self._cli()
        before = _mk_seed_result({"A": 0.30})
        after = _mk_seed_result({"A": 0.35})
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        rc = cli.main(["compare", str(b_path), str(a_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mean AOCC:  before=0.3000  after=0.3500  delta=+0.0500" in out

    def test_compare_multi_fail_on_regression(self, tmp_path) -> None:
        cli = self._cli()
        before = _mk_multi({42: {"A": 0.40}, 7: {"A": 0.40}})
        after = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.30}})
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        assert cli.main(["compare", str(b_path), str(a_path), "--fail-on-regression"]) == 2

    def _gate(self, tmp_path, before, after) -> int:
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        return self._cli().main(["compare", str(b_path), str(a_path), "--fail-on-regression"])

    def test_single_gate_ignores_one_sided_strategies(self, tmp_path) -> None:
        # Baselines only on the before side: A improved, the gate must pass.
        before = _mk_seed_result({"A": 0.30, "Baseline_Random": 0.90})
        after = _mk_seed_result({"A": 0.31})
        assert self._gate(tmp_path, before, after) == 0
        assert self._gate(tmp_path, before, _mk_seed_result({"A": 0.29})) == 2

    def test_multi_gate_ignores_one_sided_strategies(self, tmp_path) -> None:
        before = _mk_multi({42: {"A": 0.30, "B": 0.90}, 7: {"A": 0.40, "B": 0.90}})
        after = _mk_multi({42: {"A": 0.31}, 7: {"A": 0.41}})
        assert self._gate(tmp_path, before, after) == 0

    def test_multi_gate_uses_the_ci_not_the_raw_mean(self, tmp_path) -> None:
        # Mean delta -0.005 but the per-seed deltas straddle zero widely: noise, not a regression.
        before = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}, 3: {"A": 0.35}})
        after = _mk_multi({42: {"A": 0.35}, 7: {"A": 0.34}, 3: {"A": 0.345}})
        assert self._gate(tmp_path, before, after) == 0

    def test_families_default_to_sync_eval(self, monkeypatch) -> None:
        cli = self._cli()
        seen: list = []

        def fake_run_family_harness(*args, **kwargs):
            seen.append(kwargs["sync_eval"])
            return SimpleNamespace(print_summary=lambda: None)

        monkeypatch.setattr(cli, "run_family_harness", fake_run_family_harness)
        assert cli.main(["run", "--families-quick", "--quiet"]) == 0
        assert cli.main(["run", "--families-quick", "--quiet", "--no-sync-eval"]) == 0
        assert seen == [True, False]

    def test_ioh_batteries_default_to_sync_eval(self, monkeypatch) -> None:
        # Every battery measures synchronously unless --no-sync-eval opts out.
        cli = self._cli()
        seen: list = []

        def fake_run(*args, **kwargs):
            seen.append(kwargs["sync_eval"])
            return SimpleNamespace(print_summary=lambda: None)

        monkeypatch.setattr(cli, "run_ioh_harness", fake_run)
        monkeypatch.setattr(cli, "run_ioh_harness_multi_seed", fake_run)
        assert cli.main(["run", "--quick", "--quiet"]) == 0
        assert cli.main(["run", "--quick", "--quiet", "--no-sync-eval"]) == 0
        assert cli.main(["run", "--quick", "--quiet", "--seeds", "1", "2"]) == 0
        assert seen == [True, False, True]


# ---------------------------------------------------------------------------
# Synchronous-harvest evaluation mode (--sync-eval / config.sync_evaluation)
# ---------------------------------------------------------------------------


class TestSyncEvalPlumbing:
    """Serialization + CLI plumbing of the sync_eval flag (no worker)."""

    def test_single_seed_result_roundtrip(self) -> None:
        res = _mk_seed_result({"A": 0.30})
        assert res.sync_eval is False  # default off
        res.sync_eval = True
        rt = IOHHarnessResult.from_dict(json.loads(res.to_json()))
        assert rt.sync_eval is True

    def test_legacy_dict_defaults_to_off(self) -> None:
        # Result files written before the flag existed have no key.
        d = json.loads(_mk_seed_result({"A": 0.30}).to_json())
        del d["sync_eval"]
        assert IOHHarnessResult.from_dict(d).sync_eval is False
        md = json.loads(_mk_multi({42: {"A": 0.30}}).to_json())
        del md["sync_eval"]
        assert IOHMultiSeedResult.from_dict(md).sync_eval is False

    def test_multi_seed_result_roundtrip(self) -> None:
        ms = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}})
        ms.sync_eval = True
        rt = IOHMultiSeedResult.from_dict(json.loads(ms.to_json()))
        assert rt.sync_eval is True

    def test_compare_warns_on_mode_mismatch(self, tmp_path, capsys) -> None:
        cli = TestIOHBenchmarkCompareCLI._cli()
        before = _mk_multi({42: {"A": 0.30}, 7: {"A": 0.40}})
        after = _mk_multi({42: {"A": 0.31}, 7: {"A": 0.41}})
        after.sync_eval = True
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        rc = cli.main(["compare", str(b_path), str(a_path)])
        captured = capsys.readouterr()
        assert rc == 0  # warning, not an error — the compare still runs
        assert "evaluation-mode mismatch" in captured.err

    def test_compare_silent_when_modes_match(self, tmp_path, capsys) -> None:
        cli = TestIOHBenchmarkCompareCLI._cli()
        before = _mk_multi({42: {"A": 0.30}})
        after = _mk_multi({42: {"A": 0.31}})
        before.sync_eval = True
        after.sync_eval = True
        b_path, a_path = tmp_path / "b.json", tmp_path / "a.json"
        b_path.write_text(before.to_json())
        a_path.write_text(after.to_json())
        cli.main(["compare", str(b_path), str(a_path)])
        assert "evaluation-mode mismatch" not in capsys.readouterr().err


@requires_worker
class TestSyncEvalHarness:
    """End-to-end sync-eval battery runs (worker required)."""

    def test_sync_eval_run_completes_and_tags_result(self) -> None:
        from panobbgo.harness_ioh import make_ioh_strategies

        specs = [s for s in make_ioh_strategies() if s.name == "Blocks_warm_CMAES_JSO"]
        assert specs, "expected the Blocks_warm_CMAES_JSO spec"
        battery = IOHBatterySpec(
            name="ioh-sync-tiny", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
        assert result.sync_eval is True
        rec = result.runs[0]
        assert rec.error is None, rec.error
        # The strategy must still run to (near) budget under sync harvest.
        assert rec.n_evals >= 0.9 * rec.budget, (rec.n_evals, rec.budget)

    def test_default_run_is_sync(self) -> None:
        # Measurements default to sync_eval (reproducibility) since 2026-09-25.
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-sync-default", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        assert result.sync_eval is True

    def test_timeout_marks_the_run(self) -> None:
        """``timeout_s`` reaches the run: a zero deadline cuts it off and says so."""
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-timeout", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False, timeout_s=0.0)
        rec = result.runs[0]
        assert rec.error is not None and rec.error.startswith("TimeoutError"), rec.error
        assert rec.n_evals < rec.budget


class TestCompetitionCandidate:
    """The competition candidate is one population method with the whole budget."""

    def test_candidate_is_a_single_population_method(self) -> None:
        from panobbgo.harness_ioh import make_ioh_strategies
        from panobbgo.heuristics import CMAES

        spec = next(s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES")
        assert [cls for cls, _ in spec.heuristics] == [CMAES]

    def test_candidate_has_no_restart_analyzer(self) -> None:
        """CMA-ES restarts itself; the external analyzer halved it (0.663 -> 0.301)."""
        from panobbgo.analyzers import Restart
        from panobbgo.harness_ioh import make_ioh_strategies

        spec = next(s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES")
        assert Restart not in [cls for cls, _ in spec.analyzers]


# ---------------------------------------------------------------------------
# The evaluation budget must reach heuristic constructors
# ---------------------------------------------------------------------------


class _RecordingRoundRobin(StrategyRoundRobin):
    """Round-robin strategy that keeps every instance built, for inspection."""

    built: ClassVar[List["_RecordingRoundRobin"]] = []

    def __init__(self, problem, **kwargs) -> None:
        super().__init__(problem, **kwargs)
        type(self).built.append(self)


def _np_init_for(budget: float, dim: int, NP_min: int) -> int:
    """``_resolve_auto_np_init`` for a (budget, dim) pair, via a stub strategy."""
    stub = SimpleNamespace(config=SimpleNamespace(max_eval=budget), problem=SimpleNamespace(dim=dim))
    return _resolve_auto_np_init(stub, NP_min)


@requires_worker
class TestBudgetReachesHeuristicConstructors:
    """``NP_init="auto"`` must size itself from the battery budget, not the default.

    Regression test: ``_run_one`` used to build the strategy first and assign
    ``config.max_eval = budget`` afterwards, so ``_resolve_auto_np_init`` —
    which runs inside ``LSHADE.__init__`` — read ``Config``'s default 1000
    instead of ``budget_multiplier * dim``.
    """

    def test_auto_np_init_sizes_from_the_battery_budget(self) -> None:
        battery = IOHBatterySpec(
            name="ioh-auto-np",
            problem_kind="MA-BBOB",
            dims=(5,),
            instances=(0,),
            reps=1,
            # budget = 250 evals, half the reference 500*dim; ``dim=5`` keeps
            # the auto size above its floor of 6 at both budgets, so the
            # "expected != stale" check below is not vacuous.
            budget_multiplier=50,
        )
        dim = battery.dims[0]
        budget = battery.budget_for(dim)
        spec = StrategySpec(
            name="RoundRobin_LSHADE_auto",
            strategy_class=_RecordingRoundRobin,
            heuristics=[(LSHADE, {"NP_init": "auto"})],
        )

        _RecordingRoundRobin.built.clear()
        try:
            result = run_ioh_harness([spec], battery, base_seed=42, progress=False, sync_eval=True)
            assert [r.error for r in result.runs] == [None]
            assert len(_RecordingRoundRobin.built) == 1
            strategy = _RecordingRoundRobin.built[0]
        finally:
            _RecordingRoundRobin.built.clear()

        assert strategy.config.max_eval == budget
        lshade = next(h for h in strategy._hs if isinstance(h, LSHADE))

        expected = _np_init_for(budget, dim, lshade.NP_min)
        stale = _np_init_for(1000, dim, lshade.NP_min)  # what the default max_eval gives
        assert expected != stale, "test is vacuous unless the two budgets disagree"
        assert lshade.NP_init == expected


# ---------------------------------------------------------------------------
# seed_name: variants of one arm can share the RNG stream
# ---------------------------------------------------------------------------


def _tiny_battery(name: str) -> IOHBatterySpec:
    return IOHBatterySpec(name=name, problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50)


def _solo_lshade(name: str, seed_name: str | None = None) -> StrategySpec:
    return StrategySpec(
        name=name,
        strategy_class=StrategyRoundRobin,
        heuristics=[(LSHADE, {"NP_init": 8})],
        seed_name=seed_name,
    )


def _outcome(rec: IOHRunRecord) -> tuple:
    return (rec.aocc, tuple(rec.trace_evals), tuple(rec.trace_fx))


class TestSeedName:
    def test_rng_identity_defaults_to_the_display_name(self) -> None:
        spec = _solo_lshade("Variant_A")
        assert spec.seed_name is None
        assert spec.rng_identity == "Variant_A"
        assert _solo_lshade("Variant_A", seed_name="arm").rng_identity == "arm"

    def test_harness_seed_is_unchanged_without_seed_name(self) -> None:
        """Default behaviour must stay byte-identical to hashing ``spec.name``."""
        spec = _solo_lshade("Variant_A")
        assert _derive_seed(42, "MA-BBOB", 2, 0, spec.rng_identity, 0) == _derive_seed(
            42, "MA-BBOB", 2, 0, "Variant_A", 0
        )

    @requires_worker
    def test_same_seed_name_gives_identical_runs(self) -> None:
        specs = [_solo_lshade("Variant_A", seed_name="arm"), _solo_lshade("Variant_B", seed_name="arm")]
        result = run_ioh_harness(specs, _tiny_battery("ioh-seedname-same"), base_seed=7, progress=False, sync_eval=True)
        by_name = {r.strategy_name: r for r in result.runs}
        assert set(by_name) == {"Variant_A", "Variant_B"}
        assert [r.error for r in result.runs] == [None, None]
        assert by_name["Variant_A"].seed == by_name["Variant_B"].seed
        assert _outcome(by_name["Variant_A"]) == _outcome(by_name["Variant_B"])

    @requires_worker
    def test_different_seed_name_gives_different_runs(self) -> None:
        specs = [_solo_lshade("Variant_A", seed_name="arm_a"), _solo_lshade("Variant_B", seed_name="arm_b")]
        result = run_ioh_harness(specs, _tiny_battery("ioh-seedname-diff"), base_seed=7, progress=False, sync_eval=True)
        by_name = {r.strategy_name: r for r in result.runs}
        assert [r.error for r in result.runs] == [None, None]
        assert by_name["Variant_A"].seed != by_name["Variant_B"].seed
        assert _outcome(by_name["Variant_A"]) != _outcome(by_name["Variant_B"])


# ---------------------------------------------------------------------------
# The BBOB function axis (planning/DESIGN_suite_2026-09-14.md gap 1)
# ---------------------------------------------------------------------------


class TestBBOBClassMapping:
    def test_covers_1_to_24_with_no_gaps(self) -> None:
        assert sorted(BBOB_CLASS_OF_FID) == list(range(1, 25))
        assert ALL_BBOB_FIDS == tuple(range(1, 25))
        # Every class tag is one of the five COCO groups, and all five are used.
        assert set(BBOB_CLASS_OF_FID.values()) == set(BBOB_CLASS_ORDER)
        assert len(BBOB_CLASS_ORDER) == 5

    def test_group_boundaries_match_coco(self) -> None:
        assert bbob_class_of(1) == bbob_class_of(5) == "separable"
        assert bbob_class_of(6) == bbob_class_of(9) == "low-cond"
        assert bbob_class_of(10) == bbob_class_of(14) == "high-cond"
        assert bbob_class_of(15) == bbob_class_of(19) == "multimodal-global"
        assert bbob_class_of(20) == bbob_class_of(24) == "multimodal-weak"

    def test_rejects_out_of_range(self) -> None:
        for bad in (0, 25, -1, 100):
            with pytest.raises(ValueError):
                bbob_class_of(bad)


class TestFidAxisSpec:
    def test_empty_fids_is_todays_behaviour(self) -> None:
        b = make_standard_battery()
        assert b.fids == ()
        assert b.fid_axis == (None,)
        assert b.pair_count(2) == 2 * len(b.dims) * len(b.instances) * b.reps

    def test_fids_normalised_like_instances(self) -> None:
        b = IOHBatterySpec(name="x", problem_kind="BBOB", dims=(2,), instances=(0,), fids=range(1, 4))
        assert b.fids == (1, 2, 3)
        assert all(isinstance(f, int) for f in b.fids)
        # frozen + normalised -> hashable and comparable
        assert hash(b) == hash(IOHBatterySpec(name="x", problem_kind="BBOB", dims=(2,), instances=(0,), fids=(1, 2, 3)))

    def test_fid_axis_multiplies_the_cube(self) -> None:
        b = IOHBatterySpec(name="x", problem_kind="BBOB", dims=(2, 5), instances=(0, 1, 2), reps=2, fids=(1, 2, 3, 4))
        assert b.pair_count(3) == 3 * 4 * 2 * 3 * 2

    def test_rejects_non_bbob_kind(self) -> None:
        with pytest.raises(ValueError, match="BBOB problem kind"):
            IOHBatterySpec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(0,), fids=(1,))

    def test_rejects_ids_outside_1_24(self) -> None:
        for bad in ((0,), (25,), (1, 30)):
            with pytest.raises(ValueError, match="1..24"):
                IOHBatterySpec(name="x", problem_kind="BBOB", dims=(2,), instances=(0,), fids=bad)

    def test_rejects_fid_in_extra_builder_kwargs_too(self) -> None:
        with pytest.raises(ValueError, match="extra_builder_kwargs"):
            IOHBatterySpec(
                name="x",
                problem_kind="BBOB",
                dims=(2,),
                instances=(0,),
                fids=(1,),
                extra_builder_kwargs=(("fid", 2),),
            )

    def test_preset_carries_all_24(self) -> None:
        b = make_bbob_battery()
        assert b.problem_kind == "BBOB"
        assert b.fids == tuple(range(1, 25))
        assert b.dims == (2, 5) and b.instances == (0, 1, 2)
        assert b.budget_for(5) == 1000  # 200 * dim
        assert b.pair_count(1) == 24 * 2 * 3


class TestFidSeedDerivation:
    #: Byte-compatibility pins.  These two numbers were produced by the
    #: pre-2026-09-14 payloads (``base|kind|dim|inst|strategy|rep`` and
    #: ``noise|base|kind|dim|inst|rep``).  The fid axis appends its segment
    #: only when a fid is present, so every historical battery must still
    #: hash to exactly these values — if either changes, every number in
    #: planning/DISCOVERY_2026-09-09.md became unreproducible.
    PIN_SEED: ClassVar[int] = 3371302379
    PIN_NOISE_SEED: ClassVar[int] = 1915730949

    def test_no_fid_is_byte_identical_to_the_pin(self) -> None:
        assert _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 0) == self.PIN_SEED
        # Explicit None must not change the payload either.
        assert _derive_seed(42, "MA-BBOB", 2, 0, "Foo", 0, None, None) == self.PIN_SEED
        assert _derive_noise_seed(42, "MA-BBOB-noisy-gauss", 2, 0, 0) == self.PIN_NOISE_SEED
        assert _derive_noise_seed(42, "MA-BBOB-noisy-gauss", 2, 0, 0, None) == self.PIN_NOISE_SEED

    def test_a_fid_changes_the_seed(self) -> None:
        base = _derive_seed(42, "BBOB", 2, 0, "Foo", 0)
        assert _derive_seed(42, "BBOB", 2, 0, "Foo", 0, None, 1) != base

    def test_two_fids_on_one_cell_get_different_seeds(self) -> None:
        # The bug the fid segment exists to prevent: without it every
        # function of a battery would share one RNG stream and the runs on
        # f1 and f2 would be the same draw.
        seeds = {f: _derive_seed(42, "BBOB", 2, 0, "Foo", 0, None, f) for f in range(1, 25)}
        assert len(set(seeds.values())) == 24

    def test_two_fids_get_different_noise_seeds(self) -> None:
        ns = {f: _derive_noise_seed(42, "MA-BBOB-noisy-gauss", 2, 0, 0, f) for f in range(1, 25)}
        assert len(set(ns.values())) == 24

    def test_run_records_and_json_carry_the_fid(self) -> None:
        rec = IOHRunRecord(
            problem_kind="BBOB",
            dim=2,
            instance=0,
            strategy_name="S",
            rep=0,
            budget=10,
            n_evals=10,
            best_fx=1.0,
            f_opt=0.0,
            aocc=0.5,
            elapsed_s=0.1,
            seed=1,
            fid=7,
        )
        res = IOHHarnessResult(battery_name="b", problem_kind="BBOB", log_lo=-8, log_hi=2, runs=[rec])
        back = IOHHarnessResult.from_dict(json.loads(res.to_json()))
        assert back.runs[0].fid == 7
        assert back.per_strategy_per_class_aocc() == {("S", "low-cond"): pytest.approx(0.5)}

    def test_old_rows_without_a_fid_still_load(self) -> None:
        row = {
            "problem_kind": "MA-BBOB",
            "dim": 2,
            "instance": 0,
            "strategy_name": "S",
            "rep": 0,
            "budget": 10,
            "n_evals": 10,
            "best_fx": 1.0,
            "f_opt": 0.0,
            "aocc": 0.5,
            "elapsed_s": 0.1,
            "seed": 1,
        }
        res = IOHHarnessResult.from_dict(
            {"battery_name": "b", "problem_kind": "MA-BBOB", "runs": [row]},
        )
        assert res.runs[0].fid is None
        assert res.per_strategy_per_class_aocc() == {}


@requires_worker
class TestFidReachesWorker:
    def test_different_fids_build_different_problems(self) -> None:
        p1 = IOHProblem(kind="BBOB", instance=0, dim=2, fid=1)
        p24 = IOHProblem(kind="BBOB", instance=0, dim=2, fid=24)
        try:
            assert p1.ioh_problem_id == 1
            assert p24.ioh_problem_id == 24
            assert p1.ioh_name != p24.ioh_name
        finally:
            p1.close()
            p24.close()

    def test_harness_runs_the_axis_and_records_the_fid(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-fid-axis",
            problem_kind="BBOB",
            dims=(2,),
            instances=(0,),
            reps=1,
            budget_multiplier=25,
            fids=(1, 24),
        )
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        assert len(result.runs) == battery.pair_count(1) == 2
        by_fid = {r.fid: r for r in result.runs}
        assert set(by_fid) == {1, 24}
        for r in result.runs:
            assert r.error is None, r.error
            assert r.n_evals == r.budget
        # The two functions are genuinely different RNG streams.
        assert by_fid[1].seed != by_fid[24].seed
        # f1 (sphere) and f24 (Lunacek bi-Rastrigin) have different optima.
        assert by_fid[1].f_opt != by_fid[24].f_opt
        assert result.per_strategy_per_class_aocc().keys() == {
            ("Baseline_Random", "separable"),
            ("Baseline_Random", "multimodal-weak"),
        }


@requires_worker
def test_ioh_smoke_script_is_seeded_and_uses_the_harness_path(capsys, monkeypatch):
    """``scripts/ioh_smoke.py`` runs through ``_run_one``: budget respected, same seed -> same line."""
    import runpy
    import sys
    from pathlib import Path

    path = str(Path(__file__).resolve().parent.parent / "scripts" / "ioh_smoke.py")
    lines = []
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", ["ioh_smoke.py", "--max-eval", "60", "--seed", "3"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(path, run_name="__main__")
        assert exc.value.code == 0
        out = capsys.readouterr().out.strip().splitlines()[-1]
        lines.append(out.split("elapsed=")[0])
    assert "n_evals=60/60" in lines[0]
    assert lines[0] == lines[1]
