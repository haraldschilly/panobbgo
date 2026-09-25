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

"""``evaluation_method="processes"``: a spawn-context process pool.

The old implementation launched ``python3 -c`` per point with ``panobbgo/``
itself first on ``sys.path`` (so ``panobbgo/logging`` shadowed the stdlib and
every evaluation failed), never removed failed tasks from ``pending`` and
killed every evaluation after a hard-coded 30 s.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from panobbgo.lib import Problem
from panobbgo.lib.classic import Rosenbrock


class _Failing(Problem):
    """Every evaluation raises (module level, so it pickles for spawn)."""

    def __init__(self):
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        raise RuntimeError("boom")


class _Slow(Problem):
    def __init__(self, delay):
        self.delay = delay
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        time.sleep(self.delay)
        return float(np.sum(x**2))


def _fx(s):
    return s.results.results["fx"].to_numpy(dtype=float).ravel()


def _run(problem, max_eval, sync, timeout=None, method="processes"):
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = method
    s.config.dask_n_workers = 2
    s.config.max_eval = max_eval
    s.config.sync_evaluation = sync
    s.config.evaluation_timeout = timeout
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    return s


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_processes_mode_evaluates_the_budget(sync):
    s = _run(Rosenbrock(dim=2), 12, sync)
    assert len(s.results) == 12
    assert np.all(np.isfinite(_fx(s)))


def test_processes_mode_matches_threaded_under_sync():
    a = _run(Rosenbrock(dim=2), 10, True)
    b = _run(Rosenbrock(dim=2), 10, True, method="threaded")
    assert np.array_equal(_fx(a), _fx(b))


def test_failed_evaluations_leave_pending():
    s = _run(_Failing(), 6, False)
    assert len(s.results) == 0
    assert not s.pending
    assert s._dispatched == 6


@pytest.mark.parametrize(
    "method,sync", [("threaded", True), ("threaded", False), ("processes", True)], ids=lambda v: str(v)
)
def test_failed_evaluations_are_published_with_their_points(method, sync):
    """Failures were only logged; a module waiting for its point's value never heard of it."""
    from panobbgo.core import Analyzer
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    got = []

    class Listener(Analyzer):
        def on_failed_evaluations(self, points):
            got.extend(points)

    s = StrategyRoundRobin(_Failing(), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = method
    s.config.dask_n_workers = 2
    s.config.max_eval = 5
    s.config.sync_evaluation = sync
    s.config.stop_on_convergence = False
    s.add(Random)
    s.add_analyzer(Listener(s))
    s.start()
    s.eventbus.wait_idle(timeout=5.0)
    assert len(s.results) == 0
    assert len(got) == 5
    assert {p.who for p in got} == {"Random"}


def test_unpicklable_problem_is_a_clear_error():
    p = Rosenbrock(dim=2)
    p.hook = lambda: None  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="picklable"):
        _run(p, 4, True)


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_timeout_is_configurable_and_abandons_the_evaluation(sync):
    t0 = time.time()
    s = _run(_Slow(1.5), 2, sync, timeout=0.3)
    assert len(s.results) == 0
    assert not s.pending
    assert time.time() - t0 < 15


class _PidSleeper(Problem):
    """Records its worker's pid, then sleeps ``delay`` seconds."""

    def __init__(self, delay, pid_dir):
        self.delay = delay
        self.pid_dir = str(pid_dir)
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import os

        open(os.path.join(self.pid_dir, str(os.getpid())), "w").close()
        time.sleep(self.delay)
        return float(np.sum(x**2))


class _CrashOnce(Problem):
    """The first evaluation (marker file absent) kills its worker process with ``os._exit``."""

    def __init__(self, marker):
        self.marker = str(marker)
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import os

        try:  # atomic: exactly one worker wins, even when two start together
            os.close(os.open(self.marker, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return float(np.sum(x**2))
        os._exit(3)
        return float(np.sum(x**2))


def _proc_alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("method", ["processes", "threaded"])
def test_timeout_counts_running_time_not_queue_wait(method):
    """0.4 s objective, 2 workers, 12 queued: waiting in the queue must not time a task out."""
    s = _run(_Slow(0.4), 12, False, timeout=1.0, method=method)
    assert len(s.results) == 12


@pytest.mark.skipif(not __import__("os").path.isdir("/proc"), reason="needs /proc")
@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_timed_out_worker_processes_are_killed(sync, tmp_path):
    t0 = time.time()
    s = _run(_PidSleeper(25.0, tmp_path), 3, sync, timeout=0.5)
    assert len(s.results) == 0
    assert time.time() - t0 < 15
    pids = [int(p.name) for p in tmp_path.iterdir()]
    assert pids
    deadline = time.time() + 5
    while any(_proc_alive(p) for p in pids) and time.time() < deadline:
        time.sleep(0.05)
    assert not any(_proc_alive(p) for p in pids)


def test_cleanup_kills_workers_still_running_at_the_deadline(tmp_path):
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(_PidSleeper(25.0, tmp_path), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "processes"
    s.config.dask_n_workers = 2
    s.config.max_eval = 2
    s.config.shutdown_grace_seconds = 0.5
    s.config.stop_on_convergence = False
    s.add(Random)
    s.initialize()
    from panobbgo.lib import Point

    for i in range(2):
        s._pool.submit(str(i), Point(np.zeros(2), "t"))
    deadline = time.time() + 10
    while len(list(tmp_path.iterdir())) < 2 and time.time() < deadline:
        time.sleep(0.05)
    t0 = time.time()
    s._cleanup()
    assert time.time() - t0 < 5
    pids = [int(p.name) for p in tmp_path.iterdir()]
    deadline = time.time() + 5
    while any(_proc_alive(p) for p in pids) and time.time() < deadline:
        time.sleep(0.05)
    assert not any(_proc_alive(p) for p in pids)


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_a_crashing_worker_fails_one_evaluation_and_the_run_goes_on(sync, tmp_path):
    s = _run(_CrashOnce(tmp_path / "crashed"), 8, sync)
    assert (tmp_path / "crashed").exists()
    assert len(s.results) == 7
    assert s._dispatched == 8


class _UnpicklableAnywhere(Problem):
    def __init__(self):
        super().__init__([(-1, 1), (-1, 1)])

    def __setstate__(self, state):
        raise RuntimeError("cannot load me")

    def eval(self, x):
        return 0.0


class _UnpicklableInWorkers(Problem):
    """Unpickles in the creating process only -- like a class defined in __main__."""

    def __init__(self):
        import os

        self.home_pid = os.getpid()
        super().__init__([(-1, 1), (-1, 1)])

    def __setstate__(self, state):
        import os

        if state["home_pid"] != os.getpid():
            raise RuntimeError("not importable here")
        self.__dict__.update(state)

    def eval(self, x):
        return 0.0


class _SigtermSelf(Problem):
    def __init__(self):
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import os
        import signal

        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        return 0.0


def test_a_problem_that_does_not_unpickle_is_refused_at_setup():
    with pytest.raises(TypeError, match="does not unpickle"):
        _run(_UnpicklableAnywhere(), 4, True)


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_workers_that_cannot_load_the_problem_raise_instead_of_rebuilding_forever(sync):
    from panobbgo.local_pool import WorkerInitError

    t0 = time.time()
    with pytest.raises(WorkerInitError, match="cannot initialise the problem"):
        _run(_UnpicklableInWorkers(), 4, sync)
    assert time.time() - t0 < 30


def test_an_evaluation_whose_worker_is_sigtermed_is_not_retried_forever():
    t0 = time.time()
    s = _run(_SigtermSelf(), 2, False)
    assert len(s.results) == 0
    assert not s.pending
    assert time.time() - t0 < 60


class _WedgeFirstTwo(Problem):
    """The first two evaluations hang for 3 s; the rest are instant."""

    def __init__(self):
        import itertools
        import threading

        self._calls = itertools.count()
        self._lock = threading.Lock()
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        with self._lock:
            n = next(self._calls)
        if n < 2:
            time.sleep(3.0)
        return float(np.sum(x**2))


def test_timed_out_threads_do_not_starve_the_queue():
    """Two wedged threads held both slots of a 2-worker executor; the rest waited behind them."""
    t0 = time.time()
    s = _run(_WedgeFirstTwo(), 10, False, timeout=0.3, method="threaded")
    assert len(s.results) == 8
    assert time.time() - t0 < 2.5


def test_sync_threaded_timeout_is_ignored_with_one_warning():
    from unittest import mock

    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 6
    s.config.sync_evaluation = True
    s.config.evaluation_timeout = 1.0
    s.config.stop_on_convergence = False
    s.add(Random)
    with mock.patch.object(s.logger, "warning") as warn:
        s.start()
    msgs = [c.args[0] for c in warn.call_args_list if "evaluation.timeout is ignored" in c.args[0]]
    assert len(msgs) == 1
    assert len(s.results) == 6


class _BlockFirst(Problem):
    """The first evaluation blocks on ``release`` (threads only: the event is not picklable)."""

    def __init__(self, release, wait_s):
        import itertools

        self.release = release
        self.wait_s = wait_s
        self.first_returned = False
        self._calls = itertools.count()
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        if next(self._calls) == 0:
            self.release.wait(self.wait_s)
            self.first_returned = True
        return float(np.sum(x**2))


def test_a_hung_threaded_evaluation_trips_the_deadlock_backstop():
    """A running-but-silent evaluation counted as progress, so a hang (no timeout) never ended the run."""
    import threading

    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    release = threading.Event()
    problem = _BlockFirst(release, 30.0)
    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 3
    s.config.sync_evaluation = False
    s.config.deadlock_seconds = 0.5
    s.config.shutdown_grace_seconds = 0.1
    s.config.stop_on_convergence = False
    s.add(Random)
    try:
        s.start()
        # The run ended while the evaluation still hung (it used to wait it out).
        assert not problem.first_returned
        assert len(s.results) == 2  # the other two evaluations of the budget
    finally:
        release.set()


def test_a_hung_sync_process_evaluation_trips_the_deadlock_backstop():
    """The sync processes harvest loop waited for the pool with no bound at all."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(_Slow(8.0), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "processes"
    s.config.dask_n_workers = 2
    s.config.max_eval = 2
    s.config.sync_evaluation = True
    s.config.deadlock_seconds = 1.0
    s.config.shutdown_grace_seconds = 0.0
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert len(s.results) == 0  # both evaluations were still hanging (they finish at 8 s)


class _WaitForLoops(Problem):
    """Returns once the strategy's main loop has made ``n`` passes (threads only)."""

    def __init__(self, n):
        self.n = n
        self.strategy = None
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import threading

        ev = threading.Event()
        deadline = time.time() + 20.0
        while self.strategy is not None and self.strategy.loops < self.n and time.time() < deadline:
            ev.wait(1e-3)
        return float(np.sum(x**2))


def test_idle_passes_do_not_end_a_run_with_a_slow_evaluation(monkeypatch):
    """A cap of max_eval * 10000 main-loop passes ended runs whose evaluations take ~10 s each.

    The loop's 1 ms sleep is patched out so the passes pile up fast; the
    evaluation returns only after 12000 passes (the old cap for max_eval=1
    was 10000, which ended the run with no result).
    """
    import panobbgo.core as core
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    problem = _WaitForLoops(12000)
    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    problem.strategy = s
    s.config.max_eval = 1
    s.config.sync_evaluation = False
    s.config.shutdown_grace_seconds = 0.1
    s.config.stop_on_convergence = False
    s.add(Random)
    monkeypatch.setattr(core.time_module, "sleep", lambda _s: None)
    s.start()
    assert len(s.results) == 1


def test_queued_tasks_with_nothing_running_are_not_progress():
    """The deadlock backstop must see a wedge: pending tasks that never start are not progress."""
    from panobbgo.strategies import StrategyRoundRobin

    class FakePool:
        events = 5
        processes = False

        def running(self):
            return 0

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    real = s._pool
    s._pool = FakePool()
    s.pending = {"queued": "queued"}
    assert not s._pool_progressed()
    assert not s._pool_progressed()
    FakePool.events = 6
    assert s._pool_progressed()
    # A running evaluation that neither starts nor finishes anything is not
    # progress either: a hung objective used to keep the backstop off forever.
    s._pool.running = lambda: 1
    assert not s._pool_progressed()
    # Processes: queued tasks with nothing started are workers spawning.
    FakePool.processes = True
    FakePool.__len__ = lambda self: 1
    s._pool.running = lambda: 0
    assert s._pool_progressed()
    s._pool.running = lambda: 1
    assert not s._pool_progressed()
    real.close(time.time())


class _CrashOnceAbove(Problem):
    """The first evaluation with x[0] > 0.5 kills its worker (atomic marker file)."""

    def __init__(self, marker):
        self.marker = str(marker)
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import os

        if x[0] > 0.5:
            try:
                os.close(os.open(self.marker, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                pass
            else:
                os._exit(3)
        return float(np.sum(x**2))


@pytest.mark.parametrize("seed", range(5))
def test_a_break_between_polls_does_not_escape_submit(seed, tmp_path):
    """A worker dying between polls made the next submit() raise BrokenProcessPool out of start()."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(_CrashOnceAbove(tmp_path / "m"), parse_args=False, testing_mode=True, seed=seed)
    s.config.evaluation_method = "processes"
    s.config.dask_n_workers = 2
    s.config.max_eval = 60
    s.config.sync_evaluation = False
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert (tmp_path / "m").exists()
    # The crashing task fails when its dead worker is identified (59); when
    # the exit code is lost it only takes a strike and its retry succeeds,
    # because the marker makes the crash one-off (60).  Either way the run
    # finishes inside the budget.
    assert len(s.results) in (59, 60)


def test_submit_to_a_broken_pool_returns_a_failed_future():
    from concurrent.futures.process import BrokenProcessPool

    from panobbgo.lib import Point
    from panobbgo.local_pool import LocalPool

    pool = LocalPool(Rosenbrock(dim=2), 1, processes=True)
    pool._pool._broken = "simulated"  # what the executor sets when a worker dies
    f = pool._submit_future("t", Point(np.zeros(2), "t"))
    assert isinstance(f.exception(), BrokenProcessPool)
    pool.close(time.time())


def test_processes_are_not_stopped_by_the_backstop_while_workers_spawn():
    """Queued tasks while workers spawn are progress for processes (deadlock_seconds 0.01 gave 0/20)."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "processes"
    s.config.dask_n_workers = 2
    s.config.max_eval = 20
    s.config.sync_evaluation = False
    s.config.deadlock_seconds = 0.01
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert len(s.results) == 20
