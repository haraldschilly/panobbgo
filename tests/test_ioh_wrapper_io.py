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

"""IOHProblem's worker I/O, against fake workers (no ``ioh`` venv needed).

* stderr is not a pipe nobody reads — a worker writing more than a pipe
  buffer (64 KB) to stderr used to block forever;
* a round-trip is bounded by :attr:`IOHProblem.deadline`, and a worker
  that misses it is killed;
* a dead worker's stderr tail is quoted in the error;
* a closed problem's healthy worker is reused by the next one, a broken
  one never is.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest

from panobbgo.lib.ioh_wrapper import IOHProblem, shutdown_idle_workers, worker_available

_FAKE = r"""
import json, sys, time
MODE = sys.argv[1]
for line in sys.stdin:
    req = json.loads(line)
    cmd = req["cmd"]
    if MODE == "chatty":
        sys.stderr.write("x" * 200_000 + "\n")
        sys.stderr.flush()
    if cmd == "create":
        resp = {"ok": True, "lb": [-5.0, -5.0], "ub": [5.0, 5.0], "optimum_y": 0.0,
                "problem_id": 1, "instance": 0, "name": "fake"}
    elif cmd == "eval":
        if MODE == "hang":
            time.sleep(600)
        if MODE == "die":
            sys.stderr.write("boom: worker crashed\n")
            sys.stderr.flush()
            sys.exit(3)
        resp = {"ok": True, "fx": float(sum(v * v for v in req["x"]))}
    elif cmd == "shutdown":
        print(json.dumps({"ok": True}), flush=True)
        break
    else:
        resp = {"ok": True}
    print(json.dumps(resp), flush=True)
"""


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    """Make IOHProblem spawn ``_FAKE`` in the given mode."""

    def use(mode: str) -> None:
        def spawn(self: IOHProblem) -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, "-c", _FAKE, mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                bufsize=0,
            )

        monkeypatch.setattr(IOHProblem, "_spawn_worker", spawn)

    shutdown_idle_workers()
    yield lambda mode: (use(mode), tmp_path)[1]
    shutdown_idle_workers()


def test_chatty_worker_does_not_block(fake_worker):
    wd = fake_worker("chatty")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        p.deadline = time.monotonic() + 30.0
        # 20 × 200 KB of stderr: a PIPE would have filled after the first.
        for _ in range(20):
            assert p.eval(np.array([1.0, 2.0])) == pytest.approx(5.0)
    finally:
        p.close()


def test_hung_worker_times_out_and_is_killed(fake_worker):
    wd = fake_worker("hang")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    proc = p._proc
    assert proc is not None
    p.deadline = time.monotonic() + 0.5
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        p.eval(np.zeros(2))
    assert time.monotonic() - t0 < 10.0
    assert proc.poll() is not None  # killed, not left to answer late
    with pytest.raises(RuntimeError, match="not running"):
        p.eval(np.zeros(2))
    p.close()


def test_call_timeout_bounds_a_round_trip_without_any_deadline(fake_worker):
    wd = fake_worker("hang")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd, call_timeout=0.5)
    assert p.deadline is None
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        p.eval(np.zeros(2))
    assert time.monotonic() - t0 < 10.0
    p.close()


def test_default_call_timeout_is_finite():
    from panobbgo.lib.ioh_wrapper import DEFAULT_CALL_TIMEOUT_S

    assert 0 < DEFAULT_CALL_TIMEOUT_S < float("inf")


def test_stderr_file_is_append_only_and_bounded(fake_worker, monkeypatch):
    import os

    import panobbgo.lib.ioh_wrapper as w

    monkeypatch.setattr(w, "STDERR_MAX_BYTES", 300_000)
    monkeypatch.setattr(w, "STDERR_KEEP_BYTES", 1_000)
    wd = fake_worker("chatty")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        fd = p._stderr.fileno()
        import fcntl

        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND
        for _ in range(5):
            p.eval(np.zeros(2))
            # Reading the tail never moves the child's write position.
            assert p._stderr_tail(10).strip("\n") == "x" * len(p._stderr_tail(10).strip("\n"))
        assert os.fstat(fd).st_size <= 300_000 + 200_001
    finally:
        p.close()


def test_dead_worker_reports_its_stderr(fake_worker):
    wd = fake_worker("die")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        with pytest.raises(RuntimeError, match="boom: worker crashed"):
            p.eval(np.zeros(2))
    finally:
        p.close()


@pytest.mark.skipif(not worker_available(), reason="ioh worker venv not set up (tools/ioh_worker)")
def test_real_worker_stderr_path():
    """The real ``_spawn_worker``: stderr is the O_APPEND temp file, evals work."""
    import fcntl
    import os

    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2)
    try:
        assert np.isfinite(p.eval(np.zeros(2)))
        assert fcntl.fcntl(p._stderr.fileno(), fcntl.F_GETFL) & os.O_APPEND
        assert isinstance(p._stderr_tail(), str)
    finally:
        p.close()


# -- worker reuse ------------------------------------------------------------


def test_closed_problem_hands_its_worker_to_the_next_one(fake_worker):
    wd = fake_worker("ok")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    proc = p._proc
    assert p.eval(np.array([1.0, 2.0])) == pytest.approx(5.0)
    p.close()
    assert proc is not None and proc.poll() is None  # parked, not stopped
    with pytest.raises(RuntimeError, match="not running"):
        p.eval(np.zeros(2))  # the closed problem cannot reach the parked worker

    q = IOHProblem(kind="MA-BBOB", instance=1, dim=2, worker_dir=wd)
    try:
        assert q._proc is proc
        assert q.eval(np.array([3.0, 0.0])) == pytest.approx(9.0)
    finally:
        q.close()


def test_worker_is_per_kind_and_opt_out_is_private(fake_worker):
    wd = fake_worker("ok")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    proc = p._proc
    p.close()
    other = IOHProblem(kind="BBOB", instance=0, dim=2, fid=1, worker_dir=wd)
    assert other._proc is not proc
    other.close()
    private = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd, reuse_worker=False)
    assert private._proc is not proc  # did not take the idle one
    private_proc = private._proc
    private.close()
    assert private_proc is not None and private_proc.wait(timeout=10) == 0  # shut down


def test_timed_out_worker_is_never_reused(fake_worker):
    wd = fake_worker("hang")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd, call_timeout=0.5)
    proc = p._proc
    with pytest.raises(TimeoutError):
        p.eval(np.zeros(2))
    p.close()
    q = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        assert q._proc is not proc
        assert proc is not None and proc.poll() is not None
    finally:
        q.close()


def test_idle_worker_that_died_is_replaced(fake_worker):
    wd = fake_worker("ok")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    proc = p._proc
    p.close()
    assert proc is not None
    proc.kill()
    proc.wait(timeout=5)
    q = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        assert q._proc is not proc
        assert q.eval(np.array([1.0, 1.0])) == pytest.approx(2.0)
    finally:
        q.close()


def test_release_while_the_pool_lock_is_held_does_not_deadlock(fake_worker):
    """GC can run ``__del__`` -> ``close`` -> the pool in a thread that already
    holds the pool lock; it must be re-entrant."""
    import threading

    import panobbgo.lib.ioh_wrapper as w

    wd = fake_worker("ok")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    finished = threading.Event()

    def release_under_lock():
        with w._IDLE_LOCK:
            p.close()
        finished.set()

    t = threading.Thread(target=release_under_lock, daemon=True)
    t.start()
    t.join(timeout=10)
    assert finished.is_set(), "close() deadlocked on the idle-pool lock"


@pytest.mark.skipif(not worker_available(), reason="ioh worker venv not set up (tools/ioh_worker)")
def test_real_reused_worker_evaluates_like_a_fresh_one():
    """Re-targeting with ``create`` leaves nothing of the previous problem behind."""
    shutdown_idle_workers()
    rng = np.random.default_rng(0)
    xs = [rng.uniform(-5, 5, 3) for _ in range(5)]
    cases = [("MA-BBOB", 0, None), ("BBOB", 1, 8), ("MA-BBOB", 2, None), ("BBOB", 0, 24)]

    def values(reuse):
        out = []
        for kind, inst, fid in cases:
            p = IOHProblem(kind=kind, instance=inst, dim=3, fid=fid, reuse_worker=reuse)
            try:
                out.append((p.optimum_y, [p.eval(x) for x in xs]))
            finally:
                p.close()
        return out

    try:
        assert values(True) == values(False)
    finally:
        shutdown_idle_workers()
