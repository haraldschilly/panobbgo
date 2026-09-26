# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""The real asynchronous loop's pull-when-free policy (``evaluation.async_policy = "pull"``).

Asynchronous runs are not reproducible, so these tests check properties of a
short threaded run on a fake objective with random sleeps: never more in
flight than workers, no candidate dispatched twice, the exact budget, and
bandit bookkeeping that covers only dispatched points.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest

import panobbgo.strategies._bandit as bandit
from panobbgo.lib.classic import Rosenbrock

Q = 3
MAX_EVAL = 45


class SlowRosenbrock(Rosenbrock):
    """Rosenbrock whose every call sleeps 1-4 ms (a random duration) and counts itself."""

    def __init__(self, dim: int = 3, seed: int = 0) -> None:
        super().__init__(dim=dim)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self.calls = 0

    def eval(self, x: Any) -> float:
        with self._lock:
            self.calls += 1
            d = float(self._rng.uniform(1e-3, 4e-3))
        time.sleep(d)
        return super().eval(x)


def _strategy(name: str, policy: str = "pull", seed: int = 5, q: int = Q, max_eval: int = MAX_EVAL) -> Any:
    from panobbgo.heuristics import CMAES, Nearby, Random
    from panobbgo.strategies import StrategyPhased, StrategyRewarding, StrategyRoundRobin, StrategyUCB

    problem = SlowRosenbrock(seed=seed)
    kw: Dict[str, Any] = dict(
        parse_args=False,
        testing_mode=True,
        seed=seed,
        max_eval=max_eval,
        dask_n_workers=q,
        evaluation_method="threaded",
        sync_evaluation=False,
        async_policy=policy,
        stop_on_convergence=False,
    )
    if name == "roundrobin":
        s = StrategyRoundRobin(problem, size=10, **kw)
        s.add(Random)
        s.add(Nearby, radius=0.1, axes="all", new=3)
    elif name == "blocks":
        from panobbgo.harness_ioh import make_ioh_strategies

        spec = {sp.name: sp for sp in make_ioh_strategies()}["Blocks_warm_CMAES_JSO"]
        s = spec.strategy_class(problem, **kw, **spec.config_overrides)
        for h, hk in spec.heuristics:
            s.add(h, **hk)
        for a, ak in spec.analyzers:
            s.add_analyzer(a(s, **ak))
    elif name == "rewarding":
        s = StrategyRewarding(problem, **kw)
        s.add(Random)
        s.add(Nearby, radius=0.1, axes="all", new=3)
    elif name == "ucb":
        s = StrategyUCB(problem, **kw)
        s.add(Random)
        s.add(Nearby, radius=0.1, axes="all", new=3)
    elif name == "phased":
        s = StrategyPhased(
            problem,
            phases=[
                {"pct": 30, "strategy": (StrategyRoundRobin, {"size": 10}), "heuristics": [(Random, {})]},
                {
                    "strategy": (StrategyRewarding, {}),
                    "heuristics": [(CMAES, {}), (Nearby, {"radius": 0.1, "axes": "all", "new": 3})],
                },
            ],
            **kw,
        )
    else:
        raise KeyError(name)
    return s


def _instrument(s: Any) -> Dict[str, Any]:
    """Record every submission to the pool: the points, and the tasks outstanding right after it."""
    rec: Dict[str, Any] = {"points": [], "peak": 0, "caps": []}
    pool = s._pool
    submit = pool.submit

    def spy(task_id: str, point: Any) -> None:
        submit(task_id, point)
        rec["points"].append(point)
        rec["peak"] = max(rec["peak"], len(pool))  # queued, running, or finished but not harvested

    pool.submit = spy
    execute = s.execute

    def spy_execute() -> List[Any]:
        # The cap next to what is really free right now, computed here from
        # the raw counters rather than through the code under test.
        used = max(s._dispatched, len(s.results) + len(s.pending))
        rec["caps"].append((s.request_cap, Q - len(s.pending), s.config.max_eval - used))
        return execute()

    s.execute = spy_execute
    return rec


@pytest.mark.parametrize("name", ["roundrobin", "blocks", "rewarding", "ucb", "phased"])
def test_pull_when_free_properties(name):
    """In flight never exceeds the workers, nothing is dispatched twice, the budget is exact.

    Invariants of every schedule, not statistics: no retries.
    """
    s = _strategy(name)
    rec = _instrument(s)
    s.start()
    assert rec["peak"] <= Q
    ids = [id(p) for p in rec["points"]]
    assert len(ids) == len(set(ids)) == MAX_EVAL  # each candidate once, the whole budget
    assert len(s.results) == MAX_EVAL
    assert s._problem.calls == MAX_EVAL
    assert s.n_admit_trimmed == 0  # every strategy honours its request cap
    # Asked only while a worker was free, for exactly the free workers within the budget.
    assert rec["caps"]
    for cap, free, room in rec["caps"]:
        assert cap is not None and 1 <= cap == min(free, room), (cap, free, room)


@pytest.mark.flaky(retries=3)
def test_legacy_policy_still_floods_the_pool():
    """``async_policy = "legacy"`` keeps the old sizing: a fixed-size strategy queues past the workers."""
    s = _strategy("roundrobin", policy="legacy")
    rec = _instrument(s)
    s.start()
    assert len(s.results) == MAX_EVAL
    assert rec["peak"] > Q
    assert all(c is None for c, _, _ in rec["caps"])  # no request cap


def test_ucb_counts_only_dispatched_pulls():
    """Under the cap a bandit's pull counts are the dispatched evaluations, exactly."""
    s = _strategy("ucb")
    s.start()
    assert len(s.results) == MAX_EVAL
    assert s.total_selections == MAX_EVAL
    assert sum(h.ucb_count for h in s._heuristics.values()) == MAX_EVAL  # inactive arms too


@pytest.mark.flaky(retries=3)
def test_rewarding_allocation_follows_probabilities_under_the_cap(monkeypatch):
    """Capped selector rounds split at most the free workers, by a draw on the bandit's probabilities."""
    rounds: List[Any] = []
    draw = bandit.proportional_counts

    def spy(rng, probs, k):
        counts = draw(rng, probs, k)
        p = np.asarray(probs, dtype=float)
        rounds.append((p / p.sum(), int(k), np.asarray(counts)))
        return counts

    monkeypatch.setattr(bandit, "proportional_counts", spy)
    s = _strategy("rewarding", max_eval=150)
    s.start()
    assert len(s.results) == 150
    assert rounds and all(1 <= k <= Q for _, k, _ in rounds)
    assert all(c.sum() == k for _, k, c in rounds)
    expected = sum(k * p for p, k, _ in rounds)
    observed = sum(c for _, _, c in rounds)
    sd = np.sqrt(sum(k * p * (1 - p) for p, k, _ in rounds))
    assert np.all(np.abs(observed - expected) <= 4 * sd + 2), (observed, expected)


def test_async_policy_is_validated():
    s = _strategy("roundrobin", policy="eager")
    try:
        errors = s._validate_config()
    finally:
        s._cleanup()
    assert any("evaluation.async_policy" in e for e in errors)


def test_sync_and_virtual_ignore_the_async_policy():
    """``evaluation.sync`` and the virtual clock keep their own policies whatever ``async_policy`` says."""
    s = _strategy("roundrobin")
    try:
        assert s._pull_mode
        s.config.sync_evaluation = True
        assert not s._pull_mode
        s.config.sync_evaluation = False
        s.config.evaluation_method = "virtual"
        assert not s._pull_mode
    finally:
        s.config.evaluation_method = "threaded"
        s._cleanup()


def test_free_workers_within_the_budget():
    s = _strategy("roundrobin", max_eval=10)
    try:
        assert s._free_workers() == Q
        s.pending = {"a": 1, "b": 2}
        assert s._free_workers() == Q - 2
        s._dispatched = 9
        assert s._free_workers() == 1
        s._dispatched = 10
        assert s._free_workers() == 0
    finally:
        s.pending = {}
        s._cleanup()


def test_admit_free_returns_the_surplus():
    """The safety net for a strategy that ignores its cap: first ``cap`` points kept, the rest handed back once-warned."""
    from panobbgo.lib import Point

    s = _strategy("roundrobin")
    try:
        pts = [Point(np.full(3, float(i)), "nobody") for i in range(5)]
        with pytest.warns(RuntimeWarning, match="request_cap"):
            kept = s._admit_free(pts, 2)
        assert kept == pts[:2] and s.n_admit_trimmed == 3
        assert s._admit_free(pts[:2], 2) == pts[:2]
    finally:
        s._cleanup()


class _Clock:
    """Stands in for ``panobbgo.core.time_module``: counts the main loop's sleeps, forwards the rest."""

    def __init__(self) -> None:
        self.sleeps = 0

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        time.sleep(seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def test_pull_waits_on_the_pool_and_refills_at_once(monkeypatch):
    """Busy workers: block on the pool, not a 1 ms poll; a harvest: ask again without a pause.

    Removing either path shows up in the counts: polling every millisecond
    makes hundreds of sleeps over this ~0.4 s run, a pause after every
    harvest about one per evaluation.
    """
    import panobbgo.core as core

    clock = _Clock()
    monkeypatch.setattr(core, "time_module", clock)
    s = _strategy("roundrobin", q=2, max_eval=20)
    s._problem.eval = lambda x: (time.sleep(0.04), float(np.sum(np.asarray(x) ** 2)))[1]
    pauses: List[str] = []
    pause = s._pull_pause

    def spy(finished_moved: bool) -> str:
        kind = pause(finished_moved)
        pauses.append(kind)
        return kind

    s._pull_pause = spy
    s.start()
    assert len(s.results) == 20
    assert pauses.count("wait") >= 5 and pauses.count("none") >= 5
    assert clock.sleeps <= 5, (clock.sleeps, pauses.count("sleep"))
    assert s.loops < 150


def test_dask_without_workers_still_pulls(monkeypatch):
    """A dask cluster reporting zero workers: bandits still submit (one at a time), no backstop wait."""
    import panobbgo.dask_evaluation as dask_evaluation
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.strategies import StrategyUCB

    class Future:
        _keys = iter(range(10**6))

        def __init__(self, fn: Any, args: Any) -> None:
            self.key = "f%d" % next(self._keys)
            self.value = fn(*args)

        def done(self) -> bool:
            return True

        def result(self) -> Any:
            return self.value

        def cancel(self) -> None:
            pass

    submitted: List[int] = []

    def fake_setup(strategy: Any, problem: Any) -> None:
        class Client:
            def submit(self, fn: Any, *args: Any, pure: bool = False) -> Future:
                submitted.append(len(strategy.pending))  # in flight before this one
                return Future(fn, args)

            def scheduler_info(self) -> Dict[str, Any]:
                return {"workers": {}}

            def close(self) -> None:
                pass

        strategy._client = Client()
        strategy._problem_future = problem

    monkeypatch.setattr(dask_evaluation, "setup_cluster", fake_setup)
    s = StrategyUCB(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3, max_eval=12)
    s.config.evaluation_method = "dask"
    s.config.sync_evaluation = False
    s.config.stop_on_convergence = False
    s.config.deadlock_seconds = 5.0
    s.add(Random)
    s.add(Nearby, radius=0.1, axes="all", new=3)
    t0 = time.time()
    s.start()
    assert len(s.results) == 12 and len(submitted) == 12
    assert max(submitted) == 0  # one worker assumed: never two in flight
    assert time.time() - t0 < 4.0  # not the deadlock backstop
