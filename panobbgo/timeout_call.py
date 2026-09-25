# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""
One evaluation under a hard time limit, in a child interpreter
==============================================================

:func:`call_with_timeout` evaluates ``problem(point)`` in a fresh child
Python process and kills that child (and its process group) when the call
runs longer than ``timeout`` seconds.  It is how ``evaluation.timeout`` is
enforced **where the evaluation runs** on a dask worker
(:mod:`panobbgo.dask_evaluation`): the clock starts when the call starts in
the child — not at submission, so queue time on the cluster never counts —
and expiry kills exactly that call, so a hung objective never holds its
dask worker.

The child is started with :mod:`subprocess` (``python -m
panobbgo.timeout_call``), not :mod:`multiprocessing`: dask worker processes
are daemonic by default, and a daemonic process may not have
``multiprocessing`` children.  The problem and point travel pickled
(``cloudpickle`` when available) over the child's stdin; the child reports
"started" once it has loaded them, then the result, over a private copy of
its stdout — whatever the objective prints goes to stderr.
"""

from __future__ import annotations

import os
import pickle
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

#: How long the child may take to start and load the problem before the
#: call counts as failed (a problem that cannot load in a fresh interpreter).
STARTUP_LIMIT_S = 300.0

_STARTED = b"S"


@dataclass
class TimedCall:
    """The outcome of :func:`call_with_timeout`.

    Exactly one of: ``result`` (the objective returned), ``error`` (it
    raised, or the child could not start/load it), ``timed_out``.
    ``seconds`` is the running time of the call itself (child start-up
    excluded).
    """

    result: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    seconds: float = 0.0


def _dumps(obj: Any) -> bytes:
    try:
        import cloudpickle  # dask's own serializer: also handles interactively defined problems
    except ImportError:  # pragma: no cover - dask depends on it
        return pickle.dumps(obj)
    return cloudpickle.dumps(obj)


def _kill(proc: subprocess.Popen) -> None:
    """Kill the child and everything it started (its own process group)."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(5.0)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _read_until(fd: int, want: Optional[int], deadline: Optional[float]) -> Optional[bytes]:
    """Read from ``fd`` until ``want`` bytes (``None``: EOF) or ``deadline``; ``None`` on timeout."""
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    chunks = []
    n = 0
    try:
        while want is None or n < want:
            rest = None if deadline is None else deadline - time.monotonic()
            if rest is not None and rest <= 0:
                return None
            if not sel.select(rest):
                return None
            chunk = os.read(fd, 65536 if want is None else want - n)
            if not chunk:
                break  # EOF
            chunks.append(chunk)
            n += len(chunk)
    finally:
        sel.close()
    return b"".join(chunks)


def call_with_timeout(problem: Any, point: Any, timeout: float) -> TimedCall:
    """Evaluate ``problem(point)`` in a child process, killing it after ``timeout`` seconds of running time."""
    payload = _dumps((problem, point))
    # The child imports what this process can import (the problem's module
    # may live on a path added at run time, not in PYTHONPATH).
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [os.getcwd(), *sys.path] if p)
    proc = subprocess.Popen(
        [sys.executable, "-m", "panobbgo.timeout_call"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
        start_new_session=True,  # its own process group: a kill reaches its children too
    )
    assert proc.stdin is not None and proc.stdout is not None
    fd = proc.stdout.fileno()
    try:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except BrokenPipeError:
            pass  # the child died loading; reported below
        started = _read_until(fd, 1, time.monotonic() + STARTUP_LIMIT_S)
        if started != _STARTED:
            _kill(proc)
            return TimedCall(error="evaluation child did not start (exit code %s)" % proc.returncode)
        t0 = time.monotonic()
        data = _read_until(fd, None, t0 + float(timeout))
        seconds = time.monotonic() - t0
        if data is None:
            _kill(proc)
            return TimedCall(timed_out=True, seconds=seconds)
        proc.wait()
        try:
            kind, value = pickle.loads(data)
        except Exception:
            return TimedCall(error="evaluation child died (exit code %s)" % proc.returncode, seconds=seconds)
        if kind == "ok":
            return TimedCall(result=value, seconds=seconds)
        return TimedCall(error=str(value), seconds=seconds)
    finally:
        if proc.poll() is None:
            _kill(proc)
        proc.stdout.close()


def _child() -> None:
    """``python -m panobbgo.timeout_call``: evaluate one pickled ``(problem, point)`` from stdin."""
    out = os.dup(1)
    os.dup2(2, 1)  # the objective's prints go to stderr, not into the protocol
    problem, point = pickle.loads(sys.stdin.buffer.read())
    os.write(out, _STARTED)
    try:
        res: Any = ("ok", problem(point))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        res = ("err", repr(exc))
    try:
        data = pickle.dumps(res)
    except Exception as exc:
        data = pickle.dumps(("err", "result could not be pickled: %r" % exc))
    view = memoryview(data)
    while view:
        view = view[os.write(out, view) :]
    os.close(out)


if __name__ == "__main__":
    _child()
