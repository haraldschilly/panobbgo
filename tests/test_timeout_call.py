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

from panobbgo.lib import Problem
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


class SpawnsASleeper:
    """Starts ``sleep 300`` (records "evaluator-pid sleeper-pid"), then sleeps ``point`` seconds."""

    def __init__(self, pid_file):
        self.pid_file = str(pid_file)

    def __call__(self, point):
        proc = subprocess.Popen(["sleep", "300"])
        with open(self.pid_file, "w") as f:
            f.write("%d %d" % (os.getpid(), proc.pid))
        time.sleep(point)
        return point


def _run_and_kill_parent(code, pid_file, sig):
    """Run ``code`` in a parent process, wait for the pid file, kill the parent with ``sig``; return the pids."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    parent = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        deadline = time.time() + 60
        while not (pid_file.exists() and pid_file.read_text().strip()) and time.time() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "the evaluation never started"
        time.sleep(0.2)
        pids = [int(v) for v in pid_file.read_text().split()]
        assert all(_alive(p) for p in pids)
        parent.send_signal(sig)
        parent.wait(10)
        return pids
    finally:
        if parent.poll() is None:
            parent.kill()


def test_the_child_and_its_subprocesses_die_with_a_sigkilled_parent(tmp_path):
    """A dask worker killed mid-call must leave nothing running — not even the objective's subprocess.

    PDEATHSIG delivers SIGTERM, whose handler kills the whole group (a
    SIGKILL there killed the child only; its ``sleep 300`` survived).
    """
    pid_file = tmp_path / "pids"
    code = (
        "from tests.test_timeout_call import SpawnsASleeper\n"
        "from panobbgo.timeout_call import call_with_timeout\n"
        "call_with_timeout(SpawnsASleeper(%r), 60, timeout=120.0)\n" % str(pid_file)
    )
    pids = _run_and_kill_parent(code, pid_file, signal.SIGKILL)
    for pid in pids:
        assert _wait_dead(pid), "pid %d outlived its killed parent" % pid


def test_process_pool_workers_die_with_a_parent_that_skips_atexit(tmp_path):
    """processes mode: a SIGTERMed parent runs no atexit, and its setsid workers must still go."""
    pid_file = tmp_path / "pids"
    code = (
        "import time\n"
        "import numpy as np\n"
        "from tests.test_timeout_call import SleeperProblem\n"
        "from panobbgo.lib import Point\n"
        "from panobbgo.local_pool import ProcessPool\n"
        "if __name__ == '__main__':\n"
        "    pool = ProcessPool(SleeperProblem(%r), 1)\n"
        "    pool.submit('a', Point(np.zeros(2), 't'))\n"
        "    while True:\n"
        "        pool.wait()\n"
        "        pool.poll()\n" % str(pid_file)
    )
    pids = _run_and_kill_parent(code, pid_file, signal.SIGTERM)
    for pid in pids:
        assert _wait_dead(pid), "pid %d outlived its terminated parent" % pid


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


class SleeperProblem(Problem):
    """A :class:`Problem` whose evaluation starts ``sleep 300`` and hangs (for ProcessPool)."""

    def __init__(self, pid_file):
        self.pid_file = str(pid_file)
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        SpawnsASleeper(self.pid_file)(300)
        return 0.0


class _Wraps:
    def __init__(self, inner):
        self.problem = inner


def test_resampled_noise_is_found_behind_wrappers():
    from unittest import mock

    from panobbgo.dask_evaluation import check_timeout_problem
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.lib.noise import NoisyProblem, make_noise_model

    logger = mock.Mock()
    noisy = NoisyProblem(Rosenbrock(dim=2), make_noise_model("gauss", dim=2), seed=1, resample=True)
    check_timeout_problem(_Wraps(_Wraps(noisy)), logger)
    assert "resample=True" in logger.warning.call_args.args[0]
    logger.reset_mock()
    check_timeout_problem(_Wraps(Rosenbrock(dim=2)), logger)
    logger.warning.assert_not_called()
