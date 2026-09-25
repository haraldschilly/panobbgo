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
Python process and kills that child — with everything it started — when the
call runs longer than ``timeout`` seconds.  It is how ``evaluation.timeout``
is enforced **where the evaluation runs** on a dask worker
(:mod:`panobbgo.dask_evaluation`): the clock starts when the call starts in
the child — not at submission, so queue time on the cluster never counts —
and expiry kills exactly that call, so a hung objective never holds its
dask worker.

Mechanics:

* The child is started with :mod:`subprocess` (``python -P -m
  panobbgo.timeout_call``), not :mod:`multiprocessing`: dask worker
  processes are daemonic by default, and a daemonic process may not have
  ``multiprocessing`` children.  Its import path is exactly the parent's
  ``sys.path`` (``-P``: the child's cwd is not prepended).
* It runs in its own process group; the parent always ends the call with
  ``killpg`` — after a timeout, and also after a successful call, so no
  background grandchild of the objective outlives it.
* **Orphan protection.**  The child watches a dedicated pipe from the
  parent: when the parent dies (a dask worker killed by the nanny), the
  pipe hits EOF and the child kills its own process group.  On Linux the
  kernel also sends it ``SIGTERM`` when its parent dies
  (``PR_SET_PDEATHSIG``), whose handler kills the whole group — not just
  the child, which would leave the objective's subprocesses running
  (:func:`kill_group_with_parent`).
* The result travels as a length-prefixed frame (8-byte big-endian length
  + pickle) over a private copy of the child's stdout; the parent reads
  exactly that many bytes, so an objective that forks (keeping the pipe
  open) cannot make a finished call look timed out.  Whatever the objective
  prints goes to stderr.  The child ends with ``os._exit`` once the frame
  is written, so a non-daemon thread left by the objective cannot keep it
  alive.

**State caveat.**  Each call evaluates a *fresh copy* of the problem (the
worker's copy, serialized once per worker and cached).  State the problem
object accumulates while evaluating does not carry over from one call to
the next — counters, caches, a problem-held RNG (which would repeat its
draws).  In particular :class:`~panobbgo.lib.noise.NoisyProblem` with
``resample=True`` counts evaluations per point in the problem object, so
under dask with ``evaluation.timeout`` every re-evaluation of a point draws
the *same* noise (as with ``resample=False``);
:func:`panobbgo.dask_evaluation.check_timeout_problem` warns about it.
"""

from __future__ import annotations

import os
import pickle
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: How long the child may take to start and load the problem before the
#: call counts as failed (a problem that cannot load in a fresh interpreter).
STARTUP_LIMIT_S = 300.0

_STARTED = b"S"
_LEN = struct.Struct(">Q")


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


def dumps(obj: Any) -> bytes:
    """Serialize like dask does (``cloudpickle`` when available)."""
    try:
        import cloudpickle  # dask's own serializer: also handles interactively defined problems
    except ImportError:  # pragma: no cover - dask depends on it
        return pickle.dumps(obj)
    return cloudpickle.dumps(obj)


#: id(problem) -> (weakref or the object, serialized bytes): a worker's
#: problem copy is serialized once, not per call.  **Do not mutate the
#: problem object during a run**: a changed problem is not re-serialized
#: (on a dask worker nothing mutates it — the calls run in children).
_PAYLOADS: Dict[int, Tuple[Any, bytes]] = {}
_PAYLOADS_LOCK = threading.Lock()


def _problem_bytes(problem: Any) -> bytes:
    key = id(problem)
    with _PAYLOADS_LOCK:
        hit = _PAYLOADS.get(key)
        if hit is not None:
            ref, data = hit
            obj = ref() if isinstance(ref, weakref.ref) else ref
            if obj is problem:
                return data
    data = dumps(problem)
    try:
        ref: Any = weakref.ref(problem, lambda _r, k=key: _PAYLOADS.pop(k, None))
    except TypeError:
        return data  # not weak-referenceable: no cache
    with _PAYLOADS_LOCK:
        _PAYLOADS[key] = (ref, data)
    return data


def _killpg(proc: subprocess.Popen) -> None:
    """Kill the child and everything it started (its own process group), bounded."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(5.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - a process in uninterruptible sleep
        pass


def _read_exact(fd: int, n: int, deadline: Optional[float]) -> Optional[bytes]:
    """Read exactly ``n`` bytes from ``fd``; ``None`` on ``deadline``, short bytes on EOF."""
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    chunks = []
    got = 0
    try:
        while got < n:
            rest = None if deadline is None else deadline - time.monotonic()
            if rest is not None and rest <= 0:
                return None
            if not sel.select(rest):
                return None
            chunk = os.read(fd, min(n - got, 1 << 20))
            if not chunk:
                break  # EOF
            chunks.append(chunk)
            got += len(chunk)
    finally:
        sel.close()
    return b"".join(chunks)


def call_with_timeout(problem: Any, point: Any, timeout: float) -> TimedCall:
    """Evaluate ``problem(point)`` in a child process, killing it after ``timeout`` seconds of running time."""
    problem_bytes = _problem_bytes(problem)
    point_bytes = dumps(point)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    watch_r, watch_w = os.pipe()  # the child's lifeline: EOF means the parent is gone
    try:
        proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "panobbgo.timeout_call", str(watch_r), str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            pass_fds=(watch_r,),
            start_new_session=True,  # its own process group: a kill reaches its children too
        )
    finally:
        os.close(watch_r)
    assert proc.stdin is not None and proc.stdout is not None
    fd = proc.stdout.fileno()
    try:
        try:
            proc.stdin.write(_LEN.pack(len(problem_bytes)) + problem_bytes + point_bytes)
            proc.stdin.close()
        except BrokenPipeError:
            pass  # the child died loading; reported below
        started = _read_exact(fd, 1, time.monotonic() + STARTUP_LIMIT_S)
        if started != _STARTED:
            return TimedCall(error="evaluation child did not start (exit code %s)" % proc.poll())
        t0 = time.monotonic()
        deadline = t0 + float(timeout)
        head = _read_exact(fd, _LEN.size, deadline)
        if head is None:
            return TimedCall(timed_out=True, seconds=time.monotonic() - t0)
        if len(head) < _LEN.size:
            return TimedCall(error="evaluation child died (exit code %s)" % proc.poll(), seconds=time.monotonic() - t0)
        (n,) = _LEN.unpack(head)
        # The frame is being written: finish reading it even if that runs a
        # moment past the deadline — the call itself has returned.
        body = _read_exact(fd, n, None)
        seconds = time.monotonic() - t0
        if body is None or len(body) < n:
            return TimedCall(error="evaluation child died writing its result", seconds=seconds)
        kind, value = pickle.loads(body)
        if kind == "ok":
            return TimedCall(result=value, seconds=seconds)
        return TimedCall(error=str(value), seconds=seconds)
    finally:
        os.close(watch_w)
        _killpg(proc)  # always: a timed-out call, and background grandchildren of a finished one
        proc.stdout.close()


def _watchdog(fd: int) -> None:
    """Child side: the parent closed the lifeline (or died) — kill this process group."""
    try:
        while os.read(fd, 1):
            pass
    except OSError:
        pass
    os.killpg(0, signal.SIGKILL)


def _kill_own_group(signum: int = 0, frame: Any = None) -> None:
    """Kill this process group — the calling process, the objective and everything it started."""
    os.killpg(0, signal.SIGKILL)


def kill_group_with_parent(parent_pid: int) -> None:
    """Make the calling process group die with its parent (Linux; call it in a process-group leader).

    ``PR_SET_PDEATHSIG`` delivers ``SIGTERM`` when the parent dies, and the
    handler kills the *whole group* — a ``SIGKILL`` there would kill only this
    process and leave the objective's own subprocesses running.  The check of
    ``getppid`` covers a parent that died before the ``prctl``.  Must run in
    the main thread (it installs a signal handler).  Elsewhere than Linux it
    does nothing; the callers have other lifelines (a watchdog pipe, the
    pool's close/atexit).
    """
    if not sys.platform.startswith("linux"):
        return
    signal.signal(signal.SIGTERM, _kill_own_group)
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:  # pragma: no cover
        return
    if os.getppid() != parent_pid:  # the parent is already gone
        _kill_own_group()


def _child(watch_fd: int, parent_pid: int) -> None:
    """``python -m panobbgo.timeout_call FD PPID``: evaluate one pickled ``(problem, point)`` from stdin."""
    kill_group_with_parent(parent_pid)
    threading.Thread(target=_watchdog, args=(watch_fd,), daemon=True).start()
    out = os.dup(1)
    os.dup2(2, 1)  # the objective's prints go to stderr, not into the protocol
    data = sys.stdin.buffer.read()
    (n,) = _LEN.unpack(data[: _LEN.size])
    problem = pickle.loads(data[_LEN.size : _LEN.size + n])
    point = pickle.loads(data[_LEN.size + n :])
    os.write(out, _STARTED)
    try:
        res: Any = ("ok", problem(point))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        res = ("err", repr(exc))
    try:
        body = pickle.dumps(res)
    except Exception as exc:
        body = pickle.dumps(("err", "result could not be pickled: %r" % exc))
    view = memoryview(_LEN.pack(len(body)) + body)
    while view:
        view = view[os.write(out, view) :]
    os._exit(0)  # a non-daemon thread left by the objective must not keep this process alive


if __name__ == "__main__":
    _child(int(sys.argv[1]), int(sys.argv[2]))
