# -*- coding: utf8 -*-
"""The per-call timeout child (:mod:`panobbgo.timeout_call`) behind dask's ``evaluation.timeout``."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from panobbgo.timeout_call import _problem_bytes, call_with_timeout

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc") or os.name != "posix", reason="needs posix /proc")


def _alive(pid: int) -> bool:
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _wait_dead(pid: int, limit: float = 10.0) -> bool:
    deadline = time.time() + limit
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


class LeavesAThread:
    """Returns at once but leaves a non-daemon thread running for a minute."""

    def __call__(self, point):
        threading.Thread(target=time.sleep, args=(60,), daemon=False).start()
        return point * 2


class Forks:
    """Forks a grandchild that keeps every inherited fd open for 30 s, then returns."""

    def __init__(self, pid_file):
        self.pid_file = str(pid_file)

    def __call__(self, point):
        pid = os.fork()
        if pid == 0:  # the grandchild: holds the result pipe open
            time.sleep(30)
            os._exit(0)
        with open(self.pid_file, "w") as f:
            f.write(str(pid))
        return point + 1


class PidSleeper:
    """Records its pid, then sleeps ``point`` seconds."""

    def __init__(self, pid_file):
        self.pid_file = str(pid_file)

    def __call__(self, point):
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))
        time.sleep(point)
        return point


def test_a_leftover_non_daemon_thread_does_not_hang_the_call():
    t0 = time.time()
    call = call_with_timeout(LeavesAThread(), 21, timeout=30.0)
    assert call.result == 42 and not call.timed_out
    assert time.time() - t0 < 20


def test_a_forking_objective_is_not_booked_as_timed_out(tmp_path):
    """The result is length-framed: a grandchild holding the pipe open must not look like a hang."""
    pid_file = tmp_path / "gc"
    t0 = time.time()
    call = call_with_timeout(Forks(pid_file), 1, timeout=10.0)
    assert call.result == 2 and not call.timed_out and call.error is None
    assert time.time() - t0 < 9
    assert _wait_dead(int(pid_file.read_text()))  # the background grandchild is killed with the group


def test_a_timed_out_call_is_killed(tmp_path):
    pid_file = tmp_path / "pid"
    call = call_with_timeout(PidSleeper(pid_file), 60, timeout=1.0)
    assert call.timed_out and 1.0 <= call.seconds < 5.0
    assert _wait_dead(int(pid_file.read_text()))


def test_the_child_dies_with_its_parent(tmp_path):
    """A dask worker killed mid-call must not leave its evaluation running (watchdog pipe / PDEATHSIG)."""
    pid_file = tmp_path / "pid"
    code = (
        "from tests.test_timeout_call import PidSleeper\n"
        "from panobbgo.timeout_call import call_with_timeout\n"
        "call_with_timeout(PidSleeper(%r), 60, timeout=120.0)\n" % str(pid_file)
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    parent = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        deadline = time.time() + 60
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "the evaluation never started"
        time.sleep(0.2)
        evaluator = int(pid_file.read_text())
        assert _alive(evaluator)
        parent.send_signal(signal.SIGKILL)
        parent.wait(10)
        assert _wait_dead(evaluator), "the orphaned evaluation kept running"
    finally:
        if parent.poll() is None:
            parent.kill()


def test_the_problem_is_serialized_once_per_object():
    p = PidSleeper("/nonexistent")
    assert _problem_bytes(p) is _problem_bytes(p)
    assert _problem_bytes(PidSleeper("/other")) is not _problem_bytes(p)


def test_an_unserializable_problem_fails_at_setup():
    from panobbgo.dask_evaluation import check_timeout_problem

    class Holder:
        def __init__(self):
            self.lock = threading.Lock()

    with pytest.raises(TypeError, match="serializable"):
        check_timeout_problem(Holder())


def test_resampled_noise_is_warned_about():
    from unittest import mock

    from panobbgo.dask_evaluation import check_timeout_problem
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.lib.noise import NoisyProblem, make_noise_model

    logger = mock.Mock()
    noisy = NoisyProblem(Rosenbrock(dim=2), make_noise_model("gauss", dim=2), seed=1, resample=True)
    check_timeout_problem(noisy, logger)
    assert "resample=True" in logger.warning.call_args.args[0]
