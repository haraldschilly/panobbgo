# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Virtual-clock parallel evaluation (``evaluation.method = "virtual"``, panobbgo/virtual_clock.py)."""

import pickle
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from panobbgo.core import Analyzer
from panobbgo.ioh_runner import IOHTracker, aocc, aocc_virtual_time
from panobbgo.lib.classic import Rosenbrock
from panobbgo.virtual_clock import (
    CallableDuration,
    ConstantDuration,
    LogNormalDuration,
    VirtualSpec,
    make_duration_model,
)


class Recorder(Analyzer):
    """Keeps every delivered :class:`~panobbgo.lib.Result` object, in delivery order."""

    def __init__(self, strategy: Any) -> None:
        Analyzer.__init__(self, strategy, name="Recorder")
        self.seen: List[Any] = []
        self.batches: List[int] = []

    def on_new_results(self, results: List[Any]) -> None:
        self.seen.extend(results)
        self.batches.append(len(results))


class FlakyRosenbrock(Rosenbrock):
    """Rosenbrock that counts objective calls and raises on the left edge of the box."""

    def __init__(self, dim: int = 2, fail_below: float = -1.0) -> None:
        super().__init__(dim=dim)
        self.calls = 0
        self.fail_below = fail_below

    def eval(self, x: Any) -> float:
        self.calls += 1
        if float(x[0]) < self.fail_below:
            raise RuntimeError("objective crashed")
        return super().eval(x)


def x0_duration(x: np.ndarray) -> float:
    """x-dependent duration (module level: picklable)."""
    return 0.5 + abs(float(x[0]))


def slow_on_the_right(x: np.ndarray) -> float:
    """5 units right of x0 = 1 (Rosenbrock(2) box: [0, 2]^2), else 1."""
    return 5.0 if float(x[0]) > 1.0 else 1.0


def _strategy(
    method: str,
    *,
    seed: int = 3,
    max_eval: int = 60,
    strategy: str = "roundrobin",
    problem: Any = None,
    timeout: Any = None,
    observer: Any = None,
    **spec: Any,
) -> Any:
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.strategies import StrategyRewarding, StrategyRoundRobin

    cls = StrategyRoundRobin if strategy == "roundrobin" else StrategyRewarding
    # ``dask_n_workers=1`` at construction, where the pool is built: the
    # sync run's ``len(evaluators)`` (batch target), as q = 1 is the virtual one's.
    s = cls(problem or Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed, dask_n_workers=1)
    cfg = s.config
    cfg.max_eval = max_eval
    cfg.stop_on_convergence = False
    cfg.ui_show = False
    cfg.evaluation_timeout = timeout
    if method == "sync":
        cfg.sync_evaluation = True
        cfg.evaluation_method = "threaded"
    else:
        VirtualSpec(**spec).apply(s, observer=observer)
    s.add(Random)
    s.add(Nearby, radius=0.1, axes="all", new=3)
    rec = Recorder(s)
    s.add_analyzer(rec)
    return s, rec


def _run(method: str, **kw: Any) -> Any:
    s, rec = _strategy(method, **kw)
    s.start()
    return s, rec


def _xs(rec: Recorder) -> np.ndarray:
    return np.asarray([r.x for r in rec.seen])


def _fx(rec: Recorder) -> np.ndarray:
    return np.asarray([r.fx for r in rec.seen], dtype=float)


def _max_concurrency(seen: List[Any]) -> int:
    """Most calls running at once (a call completing at t has freed its worker at t)."""
    return max(sum(1 for r in seen if r.t_dispatch <= t < r.t_complete) for t in {r.t_dispatch for r in seen})


# ── duration models and settings ──


def test_duration_models():
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    assert ConstantDuration(2.5)(x, rng) == 2.5
    assert make_duration_model("constant").mean == 1.0
    assert make_duration_model("constant", mean=3.0)(x, rng) == 3.0
    assert make_duration_model(3)(x, rng) == 3.0
    assert make_duration_model("2.5")(x, rng) == 2.5  # a numeric string from YAML
    ln = make_duration_model("lognormal", sigma=0.8)
    assert isinstance(ln, LogNormalDuration) and ln.mean == 1.0
    draws = np.array([ln(x, rng) for _ in range(20000)])
    assert draws.min() > 0
    assert abs(draws.mean() - 1.0) < 0.05  # mean-1 parametrisation
    xdep = make_duration_model(x0_duration, mean=1.0)
    assert isinstance(xdep, CallableDuration)
    assert xdep(np.array([2.0, 0.0]), rng) == 2.5
    with pytest.raises(ValueError, match="explicit nominal mean"):
        make_duration_model(x0_duration)
    with pytest.raises(ValueError):
        make_duration_model("weibull")
    for bad in (0, -1.0, "0", float("nan")):
        with pytest.raises(ValueError):
            make_duration_model(bad)
    with pytest.raises(ValueError):
        VirtualSpec(duration=0)  # a zero time unit used to crash every run
    with pytest.raises(ValueError):
        VirtualSpec(workers=0)
    with pytest.raises(ValueError):
        VirtualSpec(policy="eager")
    spec = VirtualSpec(workers=2, duration=x0_duration, mean=1.5)
    assert spec.to_dict() == {"workers": 2, "policy": "async", "model": "callable", "mean": 1.5, "fn": "x0_duration"}
    pickle.loads(pickle.dumps(spec))  # a module-level duration function travels to jobs > 1 workers


def test_validate_setup_rejects_bad_virtual_settings():
    for key, value in (("virtual_workers", 0), ("virtual_policy", "eager"), ("virtual_duration", "weibull")):
        s, _ = _strategy("virtual", workers=2)
        setattr(s.config, key, value)
        with pytest.raises(ValueError, match="virtual"):
            s.start()


def test_quadratic_wls_sees_the_virtual_clock_as_synchronous():
    from panobbgo.heuristics.quadratic_wls import QuadraticWlsModel

    s, _ = _strategy("virtual", workers=2)
    s.config.sync_evaluation = False  # e.g. a harness that set it before apply(); virtual implies sync
    assert s._sync_mode
    fake = SimpleNamespace(strategy=s, config=s.config)
    assert QuadraticWlsModel._sync(fake) is True  # type: ignore[arg-type]
    s._cleanup()


def test_result_defaults_survive_old_pickles():
    from panobbgo.lib import Point, Result

    r = Result(Point(np.zeros(2), "t"), 1.0)
    del r.__dict__["t_dispatch"], r.__dict__["t_complete"]  # as pickled before the fields existed
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.t_dispatch is None and r2.t_complete is None


# ── the metric ──


def test_aocc_virtual_time_equals_aocc_at_q1_constant():
    vals = [5.0, 3.0, 4.0, 0.1, 0.01, 0.2]
    timeline = [(float(k + 1), v) for k, v in enumerate(vals)]
    best = np.minimum.accumulate(vals)
    assert aocc_virtual_time(timeline, budget=6, workers=1) == pytest.approx(aocc(best, budget=6), abs=1e-15)
    # A shorter run is held at its last best value, like aocc's padding.
    assert aocc_virtual_time(timeline[:4], budget=6, workers=1) == pytest.approx(aocc(best[:4], budget=6))


def test_aocc_virtual_time_grid_and_horizon():
    # q = 2, budget 4: grid t = 0.5, 1, 1.5, 2.  Nothing is known at 0.5 (worst gap = 1).
    timeline = [(1.0, 1e-8), (1.0, 1.0), (2.0, 1.0), (2.0, 1.0)]
    got = aocc_virtual_time(timeline, budget=4, workers=2)
    assert got == pytest.approx(aocc([np.inf, 1e-8, 1e-8, 1e-8], budget=4))
    assert got == pytest.approx(0.75)
    # A completion after the horizon (budget/q mean durations) is not scored.
    assert aocc_virtual_time([(2.5, 1e-8)], budget=4, workers=2) == pytest.approx(0.0)
    # Order of the timeline does not matter; NaN (a spent call) never improves the best.
    shuffled = list(reversed(timeline)) + [(0.1, float("nan"))]
    assert aocc_virtual_time(shuffled, budget=4, workers=2) == pytest.approx(got)


# ── the sync policy: a regression mode ──


@pytest.mark.parametrize("strategy,max_eval", [("roundrobin", 60), ("rewarding", 120)])
def test_sync_policy_q1_constant_reproduces_sync_eval_exactly(strategy, max_eval):
    """policy="sync", q = 1, constant duration, dask_n_workers = 1: the evaluation.sync run, batch for batch."""
    s_sync, sync = _run("sync", strategy=strategy, max_eval=max_eval)
    s_virt, virt = _run("virtual", strategy=strategy, max_eval=max_eval, workers=1, policy="sync")
    assert len(sync.seen) == len(virt.seen) == max_eval
    np.testing.assert_array_equal(_xs(sync), _xs(virt))
    np.testing.assert_array_equal(_fx(sync), _fx(virt))
    assert [r.who for r in sync.seen] == [r.who for r in virt.seen]
    assert sync.batches == virt.batches
    assert [r.t_dispatch for r in virt.seen] == [float(k) for k in range(max_eval)]
    assert [r.t_complete for r in virt.seen] == [float(k + 1) for k in range(max_eval)]
    assert all(r.t_dispatch is None for r in sync.seen)
    assert len(s_virt.evaluators) == len(s_sync.evaluators) == 1


# ── the async policy (default) ──


@pytest.mark.parametrize("strategy", ["roundrobin", "rewarding"])
def test_async_q1_is_one_at_a_time_with_fresh_decisions(strategy):
    """q = 1: every candidate is chosen after the previous result arrived; one result per decision."""
    s, rec = _strategy("virtual", strategy=strategy, max_eval=120, workers=1)
    decisions: List[Any] = []
    execute = s.execute

    def spy():
        # Every dispatched call has been delivered before the next decision.
        decisions.append((s._dispatched, len(s.results), len(s.pending)))
        return execute()

    s.execute = spy
    s.start()
    seen = rec.seen
    assert len(seen) == 120
    assert set(rec.batches) == {1}
    assert s.jobs_per_client == 1
    assert s._virtual_clock.n_trimmed == 0
    assert all(d == n and p == 0 for d, n, p in decisions)
    for prev, cur in zip(seen, seen[1:]):
        assert cur.t_dispatch == prev.t_complete
    assert [r.t_complete for r in seen] == [float(k + 1) for k in range(120)]


def test_async_never_queues_past_the_free_workers():
    """RoundRobin draws 10 at a time; with q = 4 the request cap limits it to the free workers."""
    q = 4
    s, rec = _strategy("virtual", workers=q, duration="lognormal", sigma=0.7, max_eval=80)
    returned: List[int] = []
    back = s._return_to_queues

    def spy(points):
        returned.append(len(points))
        back(points)

    s._return_to_queues = spy
    s.start()
    seen = rec.seen
    assert len(seen) == 80
    assert returned == [] and s._virtual_clock.n_trimmed == 0  # the cap, not admit(), keeps the queue empty
    assert _max_concurrency(seen) <= q
    assert sum(1 for r in seen if r.t_dispatch == 0.0) == q
    # Asynchronous: a worker is refilled at the instant it frees up.
    completions = {r.t_complete for r in seen}
    assert all(r.t_dispatch == 0.0 or r.t_dispatch in completions for r in seen)
    # Durations vary, so single results arrive at their own instants.
    assert rec.batches.count(1) > len(rec.batches) // 2


def test_virtual_run_is_deterministic():
    """Same seed, same q and duration model: same sequence, same timestamps."""
    kw = dict(workers=4, duration="lognormal", sigma=1.0, max_eval=80, strategy="rewarding")
    _, a = _run("virtual", **kw)
    _, b = _run("virtual", **kw)
    np.testing.assert_array_equal(_xs(a), _xs(b))
    np.testing.assert_array_equal(_fx(a), _fx(b))
    assert [(r.t_dispatch, r.t_complete) for r in a.seen] == [(r.t_dispatch, r.t_complete) for r in b.seen]
    assert len({r.t_complete - r.t_dispatch for r in a.seen}) > 10


@pytest.mark.filterwarnings("ignore:virtual clock. observed mean")
def test_event_order_and_worker_limit():
    """Results arrive in (completion time, dispatch) order; never more than q calls run at once."""
    q = 4
    s, rec = _run("virtual", workers=q, duration=x0_duration, mean=1.0, max_eval=80, strategy="rewarding")
    seen = rec.seen
    assert len(seen) == 80
    assert len(s.evaluators) == q
    completes = [r.t_complete for r in seen]
    assert completes == sorted(completes)
    for r in seen:
        assert r.t_complete - r.t_dispatch == pytest.approx(x0_duration(r.x))
    assert _max_concurrency(seen) <= q
    dispatched = [r.t_dispatch for r in seen]
    assert dispatched != sorted(dispatched)  # x-dependent durations reorder the deliveries


def test_nominal_mean_mismatch_warns():
    with pytest.warns(RuntimeWarning, match="nominal mean"):
        _run("virtual", workers=2, duration=x0_duration, mean=10.0, max_eval=30)


# ── timeouts, failures and the tracker ──


@pytest.mark.filterwarnings("ignore:virtual clock. observed mean")
def test_timeout_and_failure_are_spent_evaluations_in_completion_order():
    """A timed-out call is the NaN placeholder at dispatch + timeout (unevaluated); both it and a raising
    call count as spent, non-improving evaluations in the tracker, which records in completion order."""
    problem = FlakyRosenbrock(dim=2, fail_below=0.2)
    tracker = IOHTracker(problem, budget=50)
    s, rec = _run(
        "virtual",
        workers=3,
        duration=slow_on_the_right,
        mean=2.0,
        timeout=2.0,
        max_eval=50,
        problem=problem,
        observer=tracker,
    )
    tracker.restore()
    timed = [r for r in rec.seen if r.timed_out]
    ok = [r for r in rec.seen if not r.timed_out]
    assert timed and ok
    for r in timed:
        assert float(r.x[0]) > 1.0
        assert np.isnan(r.fx) and r.cv == float("inf")
        assert r.t_complete - r.t_dispatch == pytest.approx(2.0)
    for r in ok:
        assert r.t_complete - r.t_dispatch == pytest.approx(1.0)
    assert s.n_timed_out == len(timed)
    n_failed = s.n_finished - len(rec.seen)
    assert n_failed > 0  # the objective raised for x0 < 0.2
    assert problem.calls == s.n_finished - len(timed)  # timed-out calls never reach the objective
    # The tracker: every call of the budget, failures included, in completion order.
    assert tracker.n_evals == s.n_finished == 50
    assert len(tracker.best_so_far) == len(tracker.timeline) == 50
    times = [t for t, _ in tracker.timeline]
    assert times == sorted(times)
    spent = [v for _, v in tracker.timeline if np.isnan(v)]
    assert len(spent) == len(timed) + n_failed
    assert np.all(np.diff(tracker.best_so_far) <= 0)


def test_tracker_record_spent_respects_the_budget():
    tracker = IOHTracker(Rosenbrock(dim=2), budget=2)
    tracker.record_spent(1.0)
    tracker.record_spent(2.0)
    tracker.record_spent(3.0)  # past the budget: not counted
    tracker.restore()
    assert tracker.n_evals == 2 and tracker.best_so_far == [float("inf")] * 2
    assert [t for t, _ in tracker.timeline] == [1.0, 2.0]


# ── harness paths ──


def test_ioh_run_tracked_with_virtual_scores_both_metrics():
    """The IOH driver (``_run_tracked``) with a plain and a noisy problem; no worker venv needed."""
    import dataclasses

    from panobbgo.harness_ioh import _run_tracked, make_ioh_strategies
    from panobbgo.lib.noise import NoisyProblem, make_noise_model

    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]
    kw = dict(f_opt=0.0, budget=40, seed=4, sync_eval=True, log_lo=-8.0, log_hi=6.0, timeout_s=None)
    q1 = _run_tracked(spec, (p := Rosenbrock(dim=2)), IOHTracker(p, budget=40), virtual=VirtualSpec(workers=1), **kw)
    plain = _run_tracked(spec, (p := Rosenbrock(dim=2)), IOHTracker(p, budget=40), **kw)
    assert plain.aocc_time is None
    assert q1.n_evals == 40 and q1.aocc_time is not None
    assert q1.aocc_time == pytest.approx(q1.aocc)
    assert q1.aocc > 0.0

    noisy = NoisyProblem(Rosenbrock(dim=2), make_noise_model("gauss", dim=2), seed=1, f_opt=0.0)
    q4 = _run_tracked(
        dataclasses.replace(spec),
        noisy,
        IOHTracker(noisy, budget=40),
        virtual=VirtualSpec(workers=4, duration="lognormal"),
        **kw,
    )
    assert q4.error is None and q4.aocc_observed is not None
    assert q4.aocc_time is not None and 0.0 < q4.aocc_time <= 1.0


def test_family_harness_scores_aocc_over_virtual_time():
    from panobbgo.harness_families import make_families_battery, run_family_harness
    from panobbgo.harness_ioh import IOHHarnessResult, make_ioh_strategies

    instances = make_families_battery(dims=(2,), n_instances=1)[:1]
    specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
    # log_hi = 6: a nonzero score at this small budget.
    kw = dict(budget_multiplier=20, base_seed=5, progress=False, log_hi=6.0)
    plain = run_family_harness(specs, instances, **kw)
    q1 = run_family_harness(specs, instances, virtual=VirtualSpec(workers=1, policy="sync"), **kw)
    q4 = run_family_harness(specs, instances, virtual=VirtualSpec(workers=4, duration="lognormal"), **kw)

    assert plain.virtual is None and plain.runs[0].aocc_time is None and plain.mean_aocc_time is None
    r1 = q1.runs[0]
    assert r1.error is None and r1.n_evals == r1.budget
    assert r1.aocc == plain.runs[0].aocc > 0.0  # sync policy, q = 1: the sync trajectory
    assert r1.aocc_time == pytest.approx(r1.aocc)
    assert q1.sync_eval is True
    r4 = q4.runs[0]
    assert r4.error is None and r4.aocc_time is not None and 0.0 < r4.aocc_time <= 1.0
    assert q4.virtual == {"workers": 4, "policy": "async", "model": "lognormal", "mean": 1.0, "sigma": 0.5}
    back = IOHHarnessResult.from_dict(q4.to_dict())
    assert back.virtual == q4.virtual and back.runs[0].aocc_time == r4.aocc_time
    assert back.per_strategy_aocc_time() == q4.per_strategy_aocc_time()


def _record(name: str, aocc_value: float, aocc_time: Any, error: Any = None) -> Any:
    from panobbgo.harness_ioh import IOHRunRecord

    return IOHRunRecord(
        problem_kind="k",
        dim=2,
        instance=1,
        strategy_name=name,
        rep=0,
        budget=10,
        n_evals=10,
        best_fx=1.0,
        f_opt=0.0,
        aocc=aocc_value,
        elapsed_s=0.0,
        seed=1,
        error=error,
        aocc_time=aocc_time,
    )


def test_time_aggregates_leave_out_strategies_without_a_clock():
    from panobbgo.harness_ioh import IOHHarnessResult, run_record_from_dict, warn_missing_time_scores

    runs = [
        _record("A", 0.5, 0.4),
        _record("A", 0.3, 0.2),
        _record("A", 0.0, None, error="RuntimeError: boom"),  # a crash scores 0
        _record("Baseline", 0.6, None),  # not on the clock
    ]
    res = IOHHarnessResult("b", "k", -8.0, 2.0, runs, virtual={"workers": 4})
    assert res.per_strategy_aocc_time() == {"A": pytest.approx(0.2)}
    assert res.mean_aocc_time == pytest.approx(0.2)
    assert res.n_aocc_time == 3
    assert res.strategies_without_time_score() == ["Baseline"]
    with pytest.warns(RuntimeWarning, match="Baseline"):
        warn_missing_time_scores(res)
    # Newer files with unknown run keys still load.
    row = {**res.to_dict()["runs"][0], "some_future_field": 1}
    assert run_record_from_dict(row).aocc_time == 0.4


# ── CLI ──


def _cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli_virtual", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--duration", "lognormal"],
        ["run", "--duration-sigma", "0.3"],
        ["run", "--virtual-policy", "sync"],
        ["run", "--virtual-workers", "0"],
        ["run", "--virtual-workers", "two"],
        ["run", "--virtual-workers", "4", "--duration-sigma", "-1"],
    ],
)
def test_cli_rejects_bad_virtual_arguments(argv):
    with pytest.raises(SystemExit) as e:
        _cli().main(argv)
    assert e.value.code == 2


def test_cli_resolves_the_virtual_spec():
    import argparse

    cli = _cli()
    ns = argparse.Namespace(virtual_workers=16, duration="lognormal", duration_sigma=None, virtual_policy=None)
    assert cli._resolve_virtual(ns) == VirtualSpec(workers=16, duration="lognormal", sigma=0.5, policy="async")
    assert cli._resolve_virtual(argparse.Namespace(virtual_workers=None)) is None


def test_compare_guards_a_virtual_mismatch(tmp_path, capsys):
    from panobbgo.harness_ioh import IOHHarnessResult

    def result(virtual, t):
        return IOHHarnessResult("b", "k", -8.0, 2.0, [_record("A", 0.5, t)], sync_eval=True, virtual=virtual)

    cli = _cli()
    b, a, a4 = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "a4.json"
    b.write_text(result({"workers": 1}, 0.4).to_json())
    a.write_text(result({"workers": 1}, 0.45).to_json())
    a4.write_text(result({"workers": 4}, 0.3).to_json())

    assert cli.main(["compare", str(b), str(a), "--fail-on-regression"]) == 0
    out = capsys.readouterr().out
    assert "AOCC (time): before=0.4000  after=0.4500  delta=+0.0500" in out

    assert cli.main(["compare", str(b), str(a4)]) == 0
    assert "virtual-clock mismatch" in capsys.readouterr().err
    assert cli.main(["compare", str(b), str(a4), "--fail-on-regression"]) == 2

    # Multi-seed files: the setting of their first seed.
    from panobbgo.harness_ioh import IOHMultiSeedResult

    def multi(virtual):
        return IOHMultiSeedResult("b", "k", -8.0, 2.0, [1], [result(virtual, 0.4)], virtual=virtual)

    mb, ma = tmp_path / "mb.json", tmp_path / "ma.json"
    mb.write_text(multi({"workers": 1}).to_json())
    ma.write_text(multi({"workers": 16}).to_json())
    assert cli._virtual_of(IOHMultiSeedResult.from_dict(multi({"workers": 16}).to_dict()).to_dict()) == {"workers": 16}
    assert cli.main(["compare", str(mb), str(ma), "--fail-on-regression"]) == 2


# ── the request cap (async policy) ──


def test_constant_durations_complete_together_as_one_batch():
    """q = 4, constant duration: the four calls of each instant arrive as one new_results batch."""
    s, rec = _run("virtual", workers=4, max_eval=40, strategy="rewarding")
    seen = rec.seen
    assert len(seen) == 40
    by_instant: List[int] = []
    for r in seen:
        if by_instant and seen[sum(by_instant) - 1].t_complete == r.t_complete:
            by_instant[-1] += 1
        else:
            by_instant.append(1)
    assert rec.batches == by_instant
    assert set(rec.batches) == {4}
    assert [r.t_complete for r in seen[::4]] == [float(k + 1) for k in range(10)]


def test_a_short_strategy_is_re_asked_at_the_same_instant():
    """RoundRobin(size=1) yields one point per call: it is asked again until the free workers are full."""
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), size=1, parse_args=False, testing_mode=True, seed=2)
    s.config.max_eval = 40
    s.config.stop_on_convergence = False
    VirtualSpec(workers=4).apply(s)
    s.add(Random)
    s.add(Nearby, radius=0.1, axes="all", new=3)
    rec = Recorder(s)
    s.add_analyzer(rec)
    s.start()
    seen = rec.seen
    assert len(seen) == 40
    for t in sorted({r.t_dispatch for r in seen}):
        assert sum(1 for r in seen if r.t_dispatch == t) == 4  # every instant refills all four workers


@pytest.mark.filterwarnings("ignore:virtual clock. the strategy produced")
def test_admit_keeps_the_first_candidates():
    from panobbgo.lib import Point

    s, _ = _strategy("virtual", workers=2)
    s._ensure_cluster()
    pts = [Point(np.full(2, float(i)), "nobody") for i in range(5)]
    assert s._virtual_clock.admit(pts) == pts[:2]
    assert s._virtual_clock.admit(pts[:1]) == pts[:1]
    s._cleanup()


def test_request_cap_is_the_free_workers_within_the_budget():
    s, _ = _strategy("virtual", workers=4, max_eval=6)
    s._ensure_cluster()
    clock = s._virtual_clock
    assert clock.request_cap() == 4
    s._dispatched = 3
    assert clock.request_cap() == 3  # three evaluations of the budget left
    s.config.virtual_policy = "sync"
    clock.configure()
    assert clock.request_cap() is None
    s._cleanup()


def _no_trim(s: Any) -> List[int]:
    """Spy on the async safety net: the lengths it trimmed (should stay empty)."""
    trimmed: List[int] = []
    admit = s._virtual_clock.admit

    def spy(points):
        kept = admit(points)
        if len(kept) < len(points):
            trimmed.append(len(points) - len(kept))
        return kept

    s._virtual_clock.admit = spy
    return trimmed


@pytest.mark.parametrize("credit", ["ema", "legacy"])
@pytest.mark.parametrize("workers,duration", [(1, "constant"), (4, "lognormal")])
def test_rewarding_allocation_follows_the_bandit_not_list_order(credit, workers, duration):
    """Two identical arms under q = 1: the capped split is a draw on the bandit's probabilities.

    Before the request cap every round produced one point per arm and the
    first in list order won every contest (A: 120 of 120).
    """
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRewarding

    s = StrategyRewarding(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=11, credit=credit)
    s.config.max_eval = 120
    s.config.stop_on_convergence = False
    VirtualSpec(workers=workers, duration=duration).apply(s)
    s.add_heuristic(Random(s, name="A"))
    s.add_heuristic(Random(s, name="B"))
    rec = Recorder(s)
    s.add_analyzer(rec)
    s._ensure_cluster()
    trimmed = _no_trim(s)
    s.start()
    who = [r.who for r in rec.seen]
    assert len(who) == 120
    assert trimmed == [] and s._virtual_clock.n_trimmed == 0  # never produced past the free workers
    assert 0.2 < who.count("A") / len(who) < 0.8  # list order used to win every contest


def test_ema_select_splits_exactly_by_probability():
    from panobbgo.strategies._bandit import ema_select

    class Arm:
        def __init__(self, name, perf):
            self.name, self.performance, self.can_produce, self.n = name, perf, True, 0

        def produce(self, k):
            self.n += k
            return [self.name] * k

    arms = [Arm("a", 3.0), Arm("b", 1.0)]
    rng = np.random.default_rng(0)
    for _ in range(4000):
        assert len(ema_select(arms, 1, 0.0, rng=rng)) == 1
    assert arms[0].n / 4000 == pytest.approx(0.75, abs=0.03)


def test_block_length_counts_dispatched_evaluations():
    """StrategyBlockBandit under async q = 4: blocks close after ~block_evals real evaluations."""
    from panobbgo.heuristics import CMAES, JSO
    from panobbgo.strategies.blocks import StrategyBlockBandit

    s = StrategyBlockBandit(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=5, policy="uniform")
    s.config.max_eval = 200
    s.config.stop_on_convergence = False
    VirtualSpec(workers=4, duration="lognormal").apply(s)
    s.add(CMAES)
    s.add(JSO, NP_init=8)
    s._ensure_cluster()
    trimmed = _no_trim(s)
    s.start()
    assert len(s.results) == 200
    assert trimmed == [] and s._virtual_clock.n_trimmed == 0
    assert s._block_size > 1
    # Each closed block spent at least block_evals evaluations (or its arm starved).
    assert s._blocks_closed <= 200 / s._block_size + 1


# ── the tracker's spent path ──


def test_failed_call_with_several_measurements_is_charged_once():
    tracker = IOHTracker(Rosenbrock(dim=2), budget=3)
    tracker.begin_call(0)
    tracker.problem.eval(np.zeros(2))
    tracker.problem.eval(np.ones(2))  # a problem that evaluated twice, then failed
    tracker.end_call()
    tracker.complete_call(0, 1.0, ok=False)
    assert tracker.n_evals == 1 and tracker._reserved == 1
    tracker.record_spent(2.0)
    tracker.record_spent(3.0)
    tracker.record_spent(4.0)  # past the budget
    tracker.restore()
    assert tracker.n_evals == 3


def test_spent_path_checks_the_deadline():
    fired: List[bool] = []
    tracker = IOHTracker(Rosenbrock(dim=2), budget=10, timeout_s=0.0)
    tracker.on_timeout = lambda: fired.append(True)
    tracker.record_spent(1.0)
    tracker.restore()
    assert tracker.timed_out and fired == [True]
    assert tracker.n_evals == 0


def test_zero_duration_from_a_callable_is_rejected():
    s, _ = _strategy("virtual", workers=2, duration=lambda x: 0.0, mean=1.0)
    with pytest.raises(ValueError, match="durations must be finite and > 0"):
        s.start()


# ── external ask/tell baselines on the virtual clock ──


class _ToyAdapter:
    """Uniform ask/tell adapter in generations of ``gen`` (needs every tell before the next generation)."""

    def __init__(self, lo, hi, seed, gen):
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi, self.gen = lo, hi, gen
        self.key = 0
        self.left = 0  # candidates of the current generation not yet asked
        self.open = set()
        self.asks: List[int] = []
        self.told: List[Any] = []
        self.max_in_flight = 0

    def ask(self, n):
        self.asks.append(n)
        if self.left == 0 and not self.open:
            self.left = self.gen
        out = []
        while self.left and len(out) < n:
            out.append((self.key, self.rng.uniform(self.lo, self.hi)))
            self.open.add(self.key)
            self.key += 1
            self.left -= 1
        self.max_in_flight = max(self.max_in_flight, len(self.open))
        return out

    def tell(self, key, fx):
        self.open.remove(key)
        self.told.append((key, fx))

    def close(self):
        pass


def _toy_baseline(gen: int = 3):
    from panobbgo.harness_baselines import AskTellBaselineStrategy

    class ToyBaseline(AskTellBaselineStrategy):
        who = "Toy"

        def make_adapter(self, seed, budget, batch_size):
            box = np.asarray(self.problem.box.box)
            self.adapter = _ToyAdapter(box[:, 0], box[:, 1], seed, gen)
            return self.adapter

    return ToyBaseline


def test_asktell_baseline_on_the_virtual_clock():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_ioh import _run_tracked

    kw = dict(f_opt=0.0, budget=30, seed=4, sync_eval=True, log_lo=-8.0, log_hi=6.0, timeout_s=None)
    spec = StrategySpec(name="Toy", strategy_class=_toy_baseline(), heuristics=[])

    # q = 1, constant: sequential, aocc_time == aocc.
    tracker = IOHTracker((p := Rosenbrock(dim=2)), budget=30)
    q1 = _run_tracked(spec, p, tracker, virtual=VirtualSpec(workers=1), **kw)
    assert q1.error is None and q1.n_evals == 30
    assert [t for t, _ in tracker.timeline] == [float(k + 1) for k in range(30)]
    assert q1.aocc_time == pytest.approx(q1.aocc)

    # q = 4, lognormal: generations of 3 never fill 4 workers; completions in time order.
    tracker = IOHTracker((p := Rosenbrock(dim=2)), budget=30)
    q4 = _run_tracked(spec, p, tracker, virtual=VirtualSpec(workers=4, duration="lognormal"), **kw)
    assert q4.error is None and q4.n_evals == 30 and q4.aocc_time is not None
    times = [t for t, _ in tracker.timeline]
    assert times == sorted(times) and len(set(times)) > 20


def test_run_ask_tell_fills_free_workers_and_tells_in_completion_order():
    from panobbgo.virtual_clock import run_ask_tell

    adapter = _ToyAdapter(np.zeros(2), np.full(2, 2.0), 0, gen=8)
    evaluated: List[float] = []

    def evaluate(x):
        evaluated.append(float(x[0]))
        if x[0] < 0.2:
            raise RuntimeError("failed")
        return float(np.sum(x**2))

    run_ask_tell(
        adapter.ask,
        adapter.tell,
        evaluate,
        workers=4,
        model=CallableDuration(x0_duration, mean=1.5),
        rng=np.random.default_rng(0),
        budget=40,
        timeout=2.0,
    )
    assert len(adapter.told) == 40 and adapter.max_in_flight == 4  # never past the workers
    assert adapter.asks[0] == 4  # the free workers, not the generation
    # Failed (x0 < 0.2) and timed-out (duration > 2) calls are told as NaN.
    nan_told = sum(1 for _k, fx in adapter.told if np.isnan(fx))
    assert nan_told > 0
    assert len(evaluated) < 40  # timed-out calls were never evaluated


def test_collect_pulls_honours_the_request_cap():
    """cap 3 under a target of 8: at most 3 points, the selector asked for what is still allowed."""
    from panobbgo.core import StrategyBase
    from panobbgo.strategies._bandit import collect_pulls

    fake = SimpleNamespace(
        jobs_per_client=1,
        evaluators=SimpleNamespace(outstanding=[], __len__=None),
        request_cap=3,
        eventbus=SimpleNamespace(inflight=0),
        logger=SimpleNamespace(warning=lambda *a: None),
        name="fake",
    )

    class Evaluators(list):
        outstanding: List[Any] = []

    fake.evaluators = Evaluators(range(8))
    fake._collect_points_safely = lambda target, sel, until=None: StrategyBase._collect_points_safely(
        fake,  # type: ignore[arg-type]
        target,
        sel,
        until,
    )
    asked: List[int] = []

    def one_at_a_time(n):
        asked.append(n)
        return ["p"]

    assert collect_pulls(fake, one_at_a_time, count_outstanding=False) == ["p"] * 3
    assert asked == [3, 2, 1]

    asked.clear()

    def two_then_one(n):
        asked.append(n)
        return ["p"] * min(2, n)

    assert len(collect_pulls(fake, two_then_one)) == 3
    assert asked == [3, 1]

    fake.request_cap = None  # uncapped: the historical path, the selector sees the full target
    asked.clear()
    assert len(collect_pulls(fake, one_at_a_time)) == 8
    assert asked == [8] * 8


def test_run_ask_tell_re_asks_at_the_same_instant():
    """An adapter that returns one point per ask: all q workers are filled at t = 0."""
    from panobbgo.virtual_clock import run_ask_tell

    asks: List[int] = []
    open_at_first_tell: List[int] = []
    in_flight = [0]

    def ask(n):
        asks.append(n)
        in_flight[0] += 1
        return [(len(asks), np.zeros(2))]

    def tell(key, fx):
        if not open_at_first_tell:
            open_at_first_tell.append(in_flight[0])
        in_flight[0] -= 1

    run_ask_tell(ask, tell, lambda x: 0.0, workers=4, model=ConstantDuration(), rng=np.random.default_rng(0), budget=12)
    assert asks[:4] == [4, 3, 2, 1]
    assert open_at_first_tell == [4]


def test_run_ask_tell_rejects_an_over_asking_adapter():
    from panobbgo.virtual_clock import run_ask_tell

    with pytest.raises(RuntimeError, match="at most n"):
        run_ask_tell(
            lambda n: [(k, np.zeros(2)) for k in range(n + 1)],
            lambda key, fx: None,
            lambda x: 0.0,
            workers=2,
            model=ConstantDuration(),
            rng=np.random.default_rng(0),
            budget=10,
        )


def test_admit_counts_and_warns_on_a_trim():
    from panobbgo.lib import Point

    s, _ = _strategy("virtual", workers=2)
    s._ensure_cluster()
    pts = [Point(np.full(2, float(i)), "nobody") for i in range(5)]
    with pytest.warns(RuntimeWarning, match="request_cap"):
        s._virtual_clock.admit(pts)
    assert s._virtual_clock.n_trimmed == 3
    s._cleanup()


def test_phased_caps_the_request_at_the_phase_budget():
    """StrategyPhased under async q = 4: the phase budget caps the request, nothing is trimmed afterwards."""
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.strategies import StrategyPhased, StrategyRewarding, StrategyRoundRobin

    s = StrategyPhased(
        Rosenbrock(dim=2),
        phases=[
            {"pct": 30, "strategy": (StrategyRoundRobin, {"size": 10}), "heuristics": [(Random, {})]},
            {
                "strategy": (StrategyRewarding, {}),
                "heuristics": [(Random, {}), (Nearby, {"radius": 0.1, "axes": "all", "new": 3})],
            },
        ],
        parse_args=False,
        testing_mode=True,
        seed=3,
    )
    s.config.max_eval = 50
    s.config.stop_on_convergence = False
    # Constant durations: at t = 3 the first phase has 3 of its 15 evaluations
    # left while 4 workers are free, so the phase budget must cap the request.
    VirtualSpec(workers=4).apply(s)
    returned: List[int] = []
    back = s._return_to_queues

    def spy(points):
        returned.append(len(points))
        back(points)

    s._return_to_queues = spy
    s.start()
    assert len(s.results) == 50
    assert returned == [] and s._virtual_clock.n_trimmed == 0


# ── failure regions (families with crash / timeout) ──


def _failure_families():
    from panobbgo.harness_families import make_failure_battery

    inst = make_failure_battery(dims=(2,), n_instances=1)
    crash = next(p for _n, p in inst if p.failure is not None and p.failure.mode == "crash")
    timeout = next(p for _n, p in inst if p.failure is not None and p.failure.mode == "timeout")
    return crash, timeout


def test_deferred_failure_is_counted_once_at_completion():
    """#346 books an EvaluationFailed at once; on the clock it is deferred and counted once, in completion order."""
    from panobbgo.lib.lib import EvaluationCrashed

    class Crashy(Rosenbrock):
        def eval(self, x):
            if float(x[0]) > 1.0:
                raise EvaluationCrashed("crash region")
            return super().eval(x)

    tracker = IOHTracker(Crashy(dim=2), budget=5)
    for key, x in enumerate((np.full(2, 1.5), np.zeros(2), np.full(2, 1.8))):
        tracker.begin_call(key)
        try:
            tracker.problem.eval(x)
        except EvaluationCrashed:
            pass
        tracker.end_call()
    assert tracker.n_evals == 0 and tracker.best_so_far == []  # nothing booked at call time
    # Completion order differs from dispatch order.
    tracker.complete_call(1, 1.0, ok=True)
    tracker.complete_call(0, 2.0, ok=False)  # the evaluation path saw the exception
    tracker.complete_call(2, 3.0, ok=True)  # a driver that caught it (the baselines answer NaN)
    tracker.restore()
    assert tracker.n_evals == 3 and tracker._reserved == 3
    assert [t for t, _ in tracker.timeline] == [1.0, 2.0, 3.0]
    assert [np.isnan(v) for _, v in tracker.timeline] == [False, True, True]
    assert tracker.best_so_far == [tracker.best_so_far[0]] * 3


@pytest.mark.parametrize("mode", ["crash", "timeout"])
def test_failure_families_on_the_virtual_clock(mode):
    """A panobbgo strategy on a crash / timeout family: each failed call counted once, timeouts charged the cut."""
    crash, timeout = _failure_families()
    problem = crash if mode == "crash" else timeout
    calls = [0]
    orig = problem.eval

    def counting(x):
        calls[0] += 1
        return orig(x)

    problem.eval = counting
    try:
        tracker = IOHTracker(problem, budget=60)
        s, rec = _run(
            "virtual",
            workers=4,
            duration="lognormal",
            max_eval=60,
            timeout=3.0,
            problem=problem,
            observer=tracker,
            strategy="rewarding",
        )
        tracker.restore()
    finally:
        problem.eval = orig
    assert s.n_finished == 60
    assert tracker.n_evals == 60 and len(tracker.timeline) == 60
    times = [t for t, _ in tracker.timeline]
    assert times == sorted(times)
    spent = sum(1 for _, v in tracker.timeline if np.isnan(v))
    if mode == "crash":
        n_crashed = s.n_finished - len(rec.seen)
        assert n_crashed > 0 and spent == n_crashed + s.n_timed_out
        assert calls[0] == 60 - s.n_timed_out  # crashes are evaluated; simulated timeouts are not
    else:
        signalled = [r for r in rec.seen if r.timed_out]
        assert signalled and spent == len(signalled)
        for r in signalled:
            assert r.t_complete - r.t_dispatch == pytest.approx(3.0)  # charged evaluation.timeout
        assert calls[0] == 60 - len(signalled)  # a signalled timeout is known in advance, not evaluated
        assert s.n_timed_out == len(signalled)


def test_failure_families_through_the_harness_for_strategies_and_baselines():
    """run_family_harness on the failure battery at q = 4: panobbgo and an ask/tell baseline both get aocc_time."""
    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_families import make_failure_battery, run_family_harness
    from panobbgo.harness_ioh import make_ioh_strategies

    instances = make_failure_battery(dims=(2,), n_instances=1)
    specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"]
    specs.append(StrategySpec(name="Toy", strategy_class=_toy_baseline(), heuristics=[]))
    res = run_family_harness(
        specs,
        instances,
        budget_multiplier=20,
        base_seed=3,
        progress=False,
        log_hi=6.0,
        virtual=VirtualSpec(workers=4, duration="lognormal"),
    )
    assert len(res.runs) == 8
    for r in res.runs:
        assert r.error is None, r.error
        assert r.n_evals == r.budget
        assert r.aocc_time is not None and 0.0 <= r.aocc_time <= 1.0
    assert res.strategies_without_time_score() == []


def test_run_ask_tell_signalled_timeout_is_not_evaluated():
    from panobbgo.virtual_clock import failure_mode, run_ask_tell

    _crash, problem = _failure_families()
    tracker = IOHTracker(problem, budget=40)
    adapter = _ToyAdapter(problem.box.box[:, 0], problem.box.box[:, 1], 0, gen=8)
    evaluated = [0]

    def evaluate(x):
        evaluated[0] += 1
        return float(problem.eval(x))

    run_ask_tell(
        adapter.ask,
        adapter.tell,
        evaluate,
        workers=4,
        model=ConstantDuration(),
        rng=np.random.default_rng(0),
        budget=40,
        timeout=2.5,
        observer=tracker,
        failure_at=lambda x: failure_mode(problem, x),
    )
    tracker.restore()
    n_signalled = sum(1 for _k, fx in adapter.told if np.isnan(fx))
    assert n_signalled > 0 and evaluated[0] == 40 - n_signalled
    assert tracker.n_evals == 40
    assert sorted(t for t, v in tracker.timeline if np.isnan(v)) == [t for t, v in tracker.timeline if np.isnan(v)]
