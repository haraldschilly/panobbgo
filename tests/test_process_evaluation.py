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
