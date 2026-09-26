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
            return f.read().split(")")[-1].split()[0] not in ("Z", "X", "x")  # zombie, or dead mid-reap
    except (FileNotFoundError, ProcessLookupError):
        # reaped between open() and read(): procfs raises ESRCH
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
    # Unknown violations: no guessed vector, but cv = inf (infeasible); the
    # frame writes NaN into its cv_vec columns for the row.
    assert r.cv_vec is None and r.timed_out
    df = s.results.results
    assert sum(1 for c in df.columns if c[0] == "cv_vec") == 2
    assert np.isnan(df["cv_vec"].to_numpy(dtype=float)[0]).all()
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

            def scheduler_info(self):  # two workers: the pull-when-free cap
                return {"workers": {"a": {}, "b": {}}}

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


class _HangOnce(Problem):
    """The first evaluation (atomic marker file) hangs for ``hang`` seconds; the rest are instant."""

    def __init__(self, marker, hang=60.0):
        self.marker = str(marker)
        self.hang = hang
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import os

        try:
            os.close(os.open(self.marker, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return float(np.sum(x**2))
        time.sleep(self.hang)
        return float(np.sum(x**2))


class _ThreadFuture:
    """A dask-like future whose task runs on a thread — a stand-in dask worker."""

    _keys = iter(range(10**6))

    def __init__(self, fn, args):
        import threading

        self.key = "thr-%d" % next(self._keys)
        self._value = self._exc = None
        self._thread = threading.Thread(target=self._run, args=(fn, args), daemon=True)
        self._thread.start()

    def _run(self, fn, args):
        try:
            self._value = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            self._exc = exc

    def done(self):
        return not self._thread.is_alive()

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value

    def cancel(self):
        pass


def test_dask_timeout_is_enforced_on_the_worker_per_call(monkeypatch, tmp_path):
    """evaluation.timeout on dask: the worker kills the hung call and returns a NaN placeholder.

    The task really runs (here on a thread standing in for a dask worker):
    the hung call's child process is killed after the limit, measured from
    the call's start, and the worker is free again.
    """
    import panobbgo.dask_evaluation as dask_evaluation
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    def fake_setup(strategy, problem):
        class Client:
            def submit(self, fn, *args, pure=False):
                return _ThreadFuture(fn, args)

            def scheduler_info(self):  # two workers: the pull-when-free cap
                return {"workers": {"a": {}, "b": {}}}

            def close(self):
                pass

        strategy._client = Client()
        strategy._problem_future = problem

    monkeypatch.setattr(dask_evaluation, "setup_cluster", fake_setup)
    s = StrategyRoundRobin(_HangOnce(tmp_path / "m"), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "dask"
    s.config.max_eval = 4
    s.config.evaluation_timeout = 2.0
    s.config.stop_on_convergence = False
    s.add(Random)
    t0 = time.time()
    s.start()
    assert time.time() - t0 < 45  # the 60 s hang was cut
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
    # A worker runs one task at a time, so the crash is attributed exactly:
    # the crashing task fails, nothing else is lost or retried.
    assert len(s.results) == 59


def test_submit_to_a_dead_worker_does_not_raise():
    """A worker that died between polls: submit() re-queues, the next poll replaces the worker."""
    from panobbgo.lib import Point
    from panobbgo.local_pool import LocalPool, ProcessPool

    pool = LocalPool(Rosenbrock(dim=2), 1, processes=True)
    assert isinstance(pool, ProcessPool)
    try:
        pool.submit("a", Point(np.zeros(2), "t"))
        out = []
        deadline = time.time() + 30
        while not out and time.time() < deadline:
            pool.wait()
            out = pool.poll()
        assert [o.ok for o in out] == [True]
        [w] = pool._workers
        w.proc.kill()
        w.proc.join(5)
        pool.submit("b", Point(np.ones(2), "t"))  # must not raise
        out = []
        deadline = time.time() + 30
        while not out and time.time() < deadline:
            pool.wait()
            out = pool.poll()
        assert [(o.task_id, o.ok) for o in out] == [("b", True)]  # never started: re-queued, not failed
    finally:
        pool.close(time.time())


class _SleepPerPoint(Problem):
    """Sleeps ``x[0]`` seconds (as a pid-stamped file records), then returns."""

    def __init__(self, pid_dir):
        self.pid_dir = str(pid_dir)
        super().__init__([(0, 30), (0, 1)])

    def eval(self, x):
        import os

        open(os.path.join(self.pid_dir, "%d-%g" % (os.getpid(), x[0])), "w").close()
        time.sleep(float(x[0]))
        return float(x[0])


@pytest.mark.skipif(not __import__("os").path.isdir("/proc"), reason="needs /proc")
def test_a_timeout_kills_only_that_call(tmp_path):
    """Owner decision: the timed-out call dies, an evaluation in flight on another worker lives on.

    ``slow`` (1.8 s) starts ~1 s after ``hang``, so it is mid-flight when
    ``hang`` hits the 2 s limit; it must finish, evaluated exactly once.
    """
    from panobbgo.lib import Point
    from panobbgo.local_pool import ProcessPool

    pool = ProcessPool(_SleepPerPoint(tmp_path), 2)

    def harvest(n, until=60.0):
        got = {}
        deadline = time.time() + until
        while len(got) < n and time.time() < deadline:
            pool.wait()
            for o in pool.poll(timeout=2.0):
                got[o.task_id] = o
        return got

    try:
        pool.submit("warm1", Point(np.array([0.0, 0.0]), "t"))
        pool.submit("warm2", Point(np.array([0.0, 0.0]), "t"))
        assert len(harvest(2)) == 2  # both workers up
        pool.submit("hang", Point(np.array([25.0, 0.0]), "t"))
        deadline = time.time() + 10
        while pool.running() == 0 and time.time() < deadline:
            pool.wait()
            assert pool.poll(timeout=2.0) == []
        time.sleep(1.0)
        pool.submit("slow", Point(np.array([1.8, 0.0]), "t"))
        outcomes = harvest(2)
        assert outcomes["hang"].timed_out and not outcomes["hang"].ok
        assert outcomes["slow"].ok and outcomes["slow"].result.fx == 1.8
        stamps = sorted(p.name for p in tmp_path.iterdir())
        assert len([n for n in stamps if n.endswith("-1.8")]) == 1  # evaluated once, not restarted
        hang_pid = int(next(n for n in stamps if n.endswith("-25")).split("-")[0])
        slow_pid = int(next(n for n in stamps if n.endswith("-1.8")).split("-")[0])
        assert hang_pid != slow_pid and _proc_alive(slow_pid)
        deadline = time.time() + 5
        while _proc_alive(hang_pid) and time.time() < deadline:
            time.sleep(0.05)
        assert not _proc_alive(hang_pid)
    finally:
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


def test_placeholders_earn_no_first_point_credit_and_are_never_best():
    """A timed-out first result must not become the incumbent or earn the "first best" reward."""
    from types import SimpleNamespace

    from panobbgo.lib import Point, Result
    from panobbgo.strategies._bandit import ema_credit, linucb_observe

    ph = Result(Point(np.zeros(2), "A"), float("nan"), timed_out=True)
    real = Result(Point(np.ones(2), "B"), 3.0)
    a, b = SimpleNamespace(performance=1.0), SimpleNamespace(performance=1.0)
    best = ema_credit(None, {"A": a, "B": b}.__getitem__, None, [ph, real], alpha=0.5)
    assert best is real
    assert a.performance == 0.5 and b.performance == 1.0  # A: reward 0; B: the first real best
    assert linucb_observe(None, None, ph) == (0.0, None)


def test_best_skips_a_timed_out_first_result():
    from panobbgo.analyzers.best import Best
    from panobbgo.lib import Point, Result
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    try:
        best = Best(s)
        ph = Result(Point(np.zeros(2), "A"), float("nan"), timed_out=True)
        best.on_new_results([ph])
        assert best.best is None
        real = Result(Point(np.ones(2), "B"), 3.0)
        best.on_new_results([real])
        assert best.best is real
    finally:
        s._cleanup()


def test_n_timed_out_is_restored_from_storage(tmp_path):
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    uri = str(tmp_path / "t.db")

    def make():
        s = StrategyRoundRobin(_Slow(1.0), parse_args=False, testing_mode=True, seed=3)
        s.config.storage_backend = "sqlite"
        s.config.storage_uri = uri
        s.config.evaluation_timeout = 0.3
        s.config.sync_evaluation = True
        s.config.stop_on_convergence = False
        s.add(Random)
        return s

    s = make()
    s.config.max_eval = 2
    s.start()
    assert s.n_timed_out == 2
    t = make()
    t.config.max_eval = 2  # the restored results already fill the budget
    t.start()
    assert t.n_timed_out == 2 and len(t.results) == 2


def test_waiting_on_a_long_evaluation_is_not_silent():
    """A WARNING every deadlock_seconds while evaluations are outstanding and none finishes."""
    import threading
    from unittest import mock

    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    release = threading.Event()
    problem = _BlockFirst(release, 2.0)
    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = 1
    s.config.sync_evaluation = False
    s.config.deadlock_seconds = 0.4
    s.config.stop_on_convergence = False
    s.add(Random)
    try:
        with mock.patch.object(s.logger, "warning") as warn:
            s.start()
    finally:
        release.set()
    msgs = [c.args[0] for c in warn.call_args_list if c.args[0].startswith("Waiting:")]
    assert 2 <= len(msgs) <= 6, msgs  # ~every 0.4 s over a 2 s evaluation
    assert "1 evaluation(s) outstanding" in msgs[0] and "evaluation.timeout unset" in msgs[0]
    assert "oldest running started" in msgs[0]
    assert len(s.results) == 1


def test_waiting_on_dask_warns_about_a_cluster_without_workers(monkeypatch):
    from unittest import mock

    import panobbgo.dask_evaluation as dask_evaluation
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    def fake_setup(strategy, problem):
        class Client:
            def submit(self, fn, *args, pure=False):
                return _FakeDaskFuture(fn, args, delay=1.2)

            def scheduler_info(self):
                return {"workers": {}}

            def close(self):
                pass

        strategy._client = Client()
        strategy._problem_future = problem

    monkeypatch.setattr(dask_evaluation, "setup_cluster", fake_setup)
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = "dask"
    s.config.max_eval = 1
    s.config.deadlock_seconds = 0.3
    s.config.stop_on_convergence = False
    s.add(Random)
    with mock.patch.object(s.logger, "warning") as warn:
        s.start()
    msgs = [c.args[0] for c in warn.call_args_list]
    assert any(m.startswith("Waiting:") and "oldest queued/submitted" in m for m in msgs)
    assert any("zero workers" in m for m in msgs)
    assert len(s.results) == 1


def test_abandoned_timed_out_threads_are_counted_and_warned_about():
    """Threads cannot be killed: a timed-out call is abandoned, counted, and the user told to use processes."""
    import threading
    from unittest import mock

    from panobbgo.lib import Point
    from panobbgo.local_pool import LocalPool

    release = threading.Event()
    logger = mock.Mock()
    pool = LocalPool(_BlockFirst(release, 10.0), 1, processes=False, logger=logger)
    try:
        pool.submit("a", Point(np.zeros(2), "t"))
        deadline = time.time() + 5
        while pool.running() == 0 and time.time() < deadline:
            time.sleep(1e-3)
        time.sleep(0.15)
        [o] = pool.poll(timeout=0.1)
        assert o.timed_out and "abandoned" in o.error
        assert pool.abandoned == 1
        [msg] = [c.args[0] for c in logger.warning.call_args_list]
        assert "1 timed-out evaluation thread(s) still running" in msg and "'processes'" in msg
    finally:
        release.set()
        pool.close(time.time() + 5)
    deadline = time.time() + 5
    while pool.abandoned and time.time() < deadline:
        time.sleep(0.01)
    assert pool.abandoned == 0  # the call returned in the background; its result was discarded


class _SpawnsASleeper(Problem):
    """Starts ``sleep 60`` as a subprocess (records its pid), then hangs."""

    def __init__(self, pid_file):
        self.pid_file = str(pid_file)
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        import subprocess

        proc = subprocess.Popen(["sleep", "60"])
        with open(self.pid_file, "w") as f:
            f.write(str(proc.pid))
        time.sleep(60)
        return 0.0


@pytest.mark.skipif(not __import__("os").path.isdir("/proc"), reason="needs /proc")
def test_a_timeout_also_kills_the_objectives_own_subprocesses(tmp_path):
    """Workers run in their own process group; killing one kills what its objective started."""
    from panobbgo.lib import Point
    from panobbgo.local_pool import ProcessPool

    pid_file = tmp_path / "pid"
    pool = ProcessPool(_SpawnsASleeper(pid_file), 1)
    try:
        pool.submit("a", Point(np.zeros(2), "t"))
        out = []
        deadline = time.time() + 60
        while not out and time.time() < deadline:
            pool.wait()
            out = pool.poll(timeout=1.5)
        assert [o.timed_out for o in out] == [True]
        sleeper = int(pid_file.read_text())
        deadline = time.time() + 5
        while _proc_alive(sleeper) and time.time() < deadline:
            time.sleep(0.05)
        assert not _proc_alive(sleeper)
    finally:
        pool.close(time.time())
