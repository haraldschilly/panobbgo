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


def _assert_timed_out(s, n):
    """``n`` evaluation.timeout placeholders: NaN results, marked, in the frame."""
    df = s.results.results
    assert int(df[("timed_out", 0)].astype(bool).sum()) == n
    assert np.isnan(df[("fx", 0)].to_numpy(dtype=float)).sum() == n
    assert s.n_timed_out == n


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_timeout_is_configurable_and_books_a_nan_result(sync):
    """A timed-out evaluation is a regular NaN result (owner decision 2026-09-25), not a failure."""
    t0 = time.time()
    s = _run(_Slow(1.5), 2, sync, timeout=0.3)
    assert len(s.results) == 2  # recorded ...
    assert s._dispatched == 2  # ... and charged once
    _assert_timed_out(s, 2)
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
    assert len(s.results) == 3  # NaN placeholders, one per killed evaluation
    _assert_timed_out(s, 3)
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
    """The first two evaluations hang until ``release`` is set; the rest are instant."""

    def __init__(self, release):
        import itertools
        import threading

        self.release = release
        self.wedged_returned = 0
        self._calls = itertools.count()
        self._lock = threading.Lock()
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        with self._lock:
            n = next(self._calls)
        if n < 2:
            self.release.wait(60.0)
            with self._lock:
                self.wedged_returned += 1
        return float(np.sum(x**2))


def test_timed_out_threads_do_not_starve_the_queue():
    """Two wedged threads held both slots of a 2-worker executor; the rest waited behind them.

    Asserted on ordering, not wall time: the run must finish all eight other
    evaluations while both wedged ones are still hanging.  (It used to
    require the run to end in 2.5 s against a 3 s wedge -- a 0.5 s margin
    that a loaded machine can eat.)
    """
    import threading

    release = threading.Event()
    problem = _WedgeFirstTwo(release)
    try:
        s = _run(problem, 10, False, timeout=0.3, method="threaded")
        assert len(s.results) == 10  # 8 evaluated + 2 NaN placeholders
        _assert_timed_out(s, 2)
        assert problem.wedged_returned == 0
    finally:
        release.set()


def test_sync_threaded_timeout_is_enforced():
    """It used to be ignored (inline evaluation); with a timeout the batch goes through the pool."""
    import threading

    release = threading.Event()
    problem = _BlockFirst(release, 10.0)
    try:
        s = _run(problem, 6, True, timeout=0.3, method="threaded")
        assert len(s.results) == 6
        _assert_timed_out(s, 1)
        # Booked in submission order: the placeholder is the first result.
        assert bool(s.results.results[("timed_out", 0)].iloc[0])
        assert not problem.first_returned
    finally:
        release.set()


def test_sync_threaded_with_a_timeout_matches_inline_when_nothing_times_out():
    a = _run(Rosenbrock(dim=2), 8, True, method="threaded")
    b = _run(Rosenbrock(dim=2), 8, True, timeout=30.0, method="threaded")
    assert np.array_equal(_fx(a), _fx(b))


class _ConstrainedSlow(Problem):
    """Two constraints; the first evaluation sleeps (threads)."""

    def __init__(self, release):
        import itertools

        self.release = release
        self._calls = itertools.count()
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        if next(self._calls) == 0:
            self.release.wait(10.0)
        return float(np.sum(x**2))

    def eval_constraints(self, x):
        return np.array([x[0] - 2.0, x[1] - 2.0])


def test_a_timed_out_constrained_evaluation_is_infeasible():
    """NaN violations: cv = inf (infeasible), cv_vec as long as the problem's."""
    import threading

    from panobbgo.core import Analyzer
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    seen = []

    class Listener(Analyzer):
        def on_new_results(self, results):
            seen.extend(r for r in results if r.timed_out)

    release = threading.Event()
    s = StrategyRoundRobin(_ConstrainedSlow(release), parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 4
    s.config.sync_evaluation = True
    s.config.evaluation_timeout = 0.3
    s.config.stop_on_convergence = False
    s.add(Random)
    s.add_analyzer(Listener(s))
    try:
        s.start()
        s.eventbus.wait_idle(timeout=5.0)
    finally:
        release.set()
    assert len(seen) == 1  # published through new_results like any result
    r = seen[0]
    assert np.isnan(r.fx) and r.cv == float("inf")
    assert r.cv_vec is not None and r.cv_vec.shape == (2,) and np.isnan(r.cv_vec).all()
    assert s.best is not None and not s.best.timed_out and np.isfinite(s.best.fx)


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


def test_a_long_threaded_evaluation_is_never_cut_by_the_backstop():
    """A running evaluation is waited for however long it takes (owner decision 2026-09-25).

    Only ``evaluation.timeout`` limits one evaluation; the deadlock backstop
    (here 0.5 s) must not end the run while the 3 s evaluation runs.
    """
    import threading

    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    release = threading.Event()
    problem = _BlockFirst(release, 3.0)
    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 3
    s.config.sync_evaluation = False
    s.config.deadlock_seconds = 0.5
    s.config.shutdown_grace_seconds = 0.1
    s.config.stop_on_convergence = False
    s.add(Random)
    try:
        s.start()
        assert problem.first_returned
        assert len(s.results) == 3
    finally:
        release.set()


def test_a_long_sync_process_evaluation_is_never_cut_by_the_backstop():
    """The sync processes harvest loop waits for running evaluations too."""
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(_Slow(3.0), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "processes"
    s.config.dask_n_workers = 2
    s.config.max_eval = 2
    s.config.sync_evaluation = True
    s.config.deadlock_seconds = 0.5
    s.config.shutdown_grace_seconds = 0.0
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert len(s.results) == 2  # both 3 s evaluations completed


class _FakeDaskFuture:
    """A dask-like future that is done ``delay`` seconds after submission."""

    _keys = iter(range(10**6))

    def __init__(self, fn, args, delay):
        self.key = "fake-%d" % next(self._keys)
        self._fn, self._args = fn, args
        self._ready_at = time.time() + delay

    def done(self):
        return time.time() >= self._ready_at

    def result(self):
        return self._fn(*self._args)

    def cancel(self):
        pass


def test_a_long_outstanding_dask_future_is_never_cut_by_the_backstop(monkeypatch):
    """Dask: any outstanding future is "waiting" (the client cannot tell queued from running)."""
    import panobbgo.dask_evaluation as dask_evaluation
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    def fake_setup(strategy, problem):
        class Client:
            def submit(self, fn, *args, pure=False):
                return _FakeDaskFuture(fn, args, delay=1.5)

            def close(self):
                pass

        strategy._client = Client()
        strategy._problem_future = problem

    monkeypatch.setattr(dask_evaluation, "setup_cluster", fake_setup)
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "dask"
    s.config.max_eval = 2
    s.config.deadlock_seconds = 0.3
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert len(s.results) == 2  # both futures resolved after 1.5 s > deadlock_seconds


def test_dask_timeout_books_nan_results_and_the_run_goes_on(monkeypatch):
    """evaluation.timeout applies to dask: a future that never finishes becomes a NaN result."""
    import panobbgo.dask_evaluation as dask_evaluation
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    def fake_setup(strategy, problem):
        calls = iter(range(10**6))

        class Client:
            def submit(self, fn, *args, pure=False):
                # The first task hangs on the "cluster"; the others take 50 ms.
                return _FakeDaskFuture(fn, args, delay=1e9 if next(calls) == 0 else 0.05)

            def close(self):
                pass

        strategy._client = Client()
        strategy._problem_future = problem

    monkeypatch.setattr(dask_evaluation, "setup_cluster", fake_setup)
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "dask"
    s.config.max_eval = 4
    s.config.evaluation_timeout = 0.3
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    assert len(s.results) == 4 and s._dispatched == 4
    _assert_timed_out(s, 1)


def test_a_claimed_point_that_never_comes_trips_the_backstop():
    """Nothing outstanding, yet the run reports itself alive: that is the deadlock."""
    from panobbgo.core import Heuristic
    from panobbgo.strategies import StrategyRoundRobin

    class Liar(Heuristic):
        @property
        def can_produce(self):
            return True  # claims a point, never emits one

        @property
        def active(self):
            return True

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 5
    s.config.sync_evaluation = False
    s.config.deadlock_seconds = 0.5
    s.config.stop_on_convergence = False
    s.add(Liar)
    t0 = time.time()
    s.start()
    assert len(s.results) == 0
    assert time.time() - t0 < 30


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


def test_outstanding_evaluations_are_progress_and_orphans_are_not():
    """What the deadlock backstop waits for: any outstanding evaluation, never an orphaned queue."""
    from panobbgo.strategies import StrategyRoundRobin

    class FakePool:
        events = 5
        is_waiting = False

        def waiting(self):
            return self.is_waiting

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    real = s._pool
    s._pool = FakePool()
    s.pending = {"queued": "queued"}
    # Queued tasks no worker can pick up: not progress, the backstop may fire.
    assert not s._pool_progressed()
    assert not s._pool_progressed()
    FakePool.events = 6  # a start or finish
    assert s._pool_progressed()
    assert not s._pool_progressed()
    # A running evaluation (or one queued for live workers) is progress,
    # however long it takes.
    FakePool.is_waiting = True
    assert s._pool_progressed()
    assert s._pool_progressed()
    # Dask: every outstanding future counts, queued or running.
    s.config.evaluation_method = "dask"
    assert s._pool_progressed()
    s.pending = {}
    assert not s._pool_progressed()
    s.config.evaluation_method = "threaded"
    real.close(time.time())


def test_local_pool_waiting():
    """LocalPool.waiting: running, or queued for live workers."""
    import threading

    from panobbgo.lib import Point
    from panobbgo.local_pool import LocalPool

    release = threading.Event()
    pool = LocalPool(_BlockFirst(release, 10.0), 1, processes=False)
    assert not pool.waiting()  # nothing outstanding
    try:
        pool.submit("a", Point(np.zeros(2), "t"))
        pool.submit("b", Point(np.zeros(2), "t"))
        deadline = time.time() + 5
        while pool.running() == 0 and time.time() < deadline:
            time.sleep(1e-3)
        assert pool.running() == 1 and pool.waiting()  # "a" runs, "b" queued behind it
        pool._tasks.pop("a")
        pool._started.pop("a", None)
        assert pool.waiting()  # "b" queued for a live executor
        pool._pool._shutdown = True  # an executor that can no longer run anything
        assert not pool.waiting()
        pool._pool._shutdown = False
    finally:
        release.set()
        pool.close(time.time() + 5)


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
