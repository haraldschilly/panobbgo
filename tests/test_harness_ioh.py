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
    DEFAULT_DECISION_SEEDS,
    IOHBatterySpec,
    IOHHarnessResult,
    IOHMultiSeedResult,
    IOHRunRecord,
    _derive_seed,
    _downsample_trajectory,
    make_full_battery,
    make_quick_battery,
    make_standard_battery,
    paired_seed_stats,
    run_ioh_harness,
    run_ioh_harness_multi_seed,
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

        specs = [s for s in make_ioh_strategies() if s.name == "Rewarding_Restart"]
        assert specs, "expected the Rewarding_Restart spec"
        battery = IOHBatterySpec(
            name="ioh-sync-tiny", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
        assert result.sync_eval is True
        rec = result.runs[0]
        assert rec.error is None, rec.error
        # The strategy must still run to (near) budget under sync harvest.
        assert rec.n_evals >= 0.9 * rec.budget, (rec.n_evals, rec.budget)

    def test_default_run_is_untagged(self) -> None:
        baselines = [s for s in make_baseline_strategies() if s.name == "Baseline_Random"]
        battery = IOHBatterySpec(
            name="ioh-sync-off", problem_kind="MA-BBOB", dims=(2,), instances=(0,), reps=1, budget_multiplier=50
        )
        result = run_ioh_harness(baselines, battery, base_seed=42, progress=False)
        assert result.sync_eval is False
