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
Local evaluation pool
=====================

The backend of ``evaluation_method = "threaded"`` and ``"processes"``:
evaluations are submitted by task id and harvested without blocking by
:meth:`LocalPool.poll`, which reports every task exactly once, as a result
or as a failure.

What it guarantees beyond a bare executor:

* **The timeout counts running time only.**  The clock of a task starts
  when a worker picks it up (a thread records it; a worker process reports
  it through a queue), so time spent queued behind other evaluations never
  times a task out.
* **Processes: a timed-out evaluation is killed, not abandoned.**  The
  pool's workers are killed and a fresh pool takes over; the other
  in-flight tasks are resubmitted (they lose their progress, not their
  place in the budget).  Threads cannot be interrupted: a timed-out thread
  evaluation is abandoned and runs to completion in the background.
* **Processes: a crashing evaluation does not end the run.**  A worker
  that dies (``os._exit``, a segfault, the OOM killer) breaks a
  ``ProcessPoolExecutor`` and fails every in-flight future with
  ``BrokenProcessPool``.  The task whose worker died is reported failed;
  the ones merely caught in the break are resubmitted to a fresh pool.
* **Closing is bounded and leaves nothing running** (:meth:`close`): queued
  tasks are cancelled, running ones get until the deadline, then worker
  processes are killed.

Processes mode evaluates *copies* of the problem: each worker unpickles its
own, so state the problem object accumulates while evaluating (counters,
traces, caches, loggers) lives in the workers and is not seen by the
caller's object.
"""

from __future__ import annotations

import os
import pickle
import signal
import threading
import time
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing

#: The problem a worker process evaluates and the queue it reports task
#: starts on; set once per worker by :func:`_process_worker_init`.
_WORKER_PROBLEM: Any = None
_WORKER_STARTS: Any = None


def _process_worker_init(payload: bytes, starts: Any) -> None:
    """Pool initializer: unpickle the problem once per worker process."""
    global _WORKER_PROBLEM, _WORKER_STARTS
    _WORKER_PROBLEM = pickle.loads(payload)
    _WORKER_STARTS = starts


def _process_worker_eval(task_id: str, point: Any) -> Any:
    """Report the start (task id, pid, wall time), then evaluate ``point``."""
    # SimpleQueue writes synchronously: the start is on the pipe before the
    # objective runs, even if the objective then kills the process.
    _WORKER_STARTS.put((task_id, os.getpid(), time.time()))
    return _WORKER_PROBLEM(point)


@dataclass
class _Task:
    point: Any
    future: Future
    strikes: int = 0


@dataclass
class Outcome:
    """What became of one task: ``ok`` with its ``result``, or failed with an ``error`` message."""

    task_id: str
    ok: bool
    result: Any = None
    error: str = ""
    started: Optional[float] = None
    finished: float = field(default_factory=time.time)


class LocalPool:
    """Threads or spawned worker processes evaluating ``problem`` (see the module docstring).

    Not thread-safe: one thread (the strategy's main loop) submits, polls
    and closes.
    """

    def __init__(self, problem: Any, n_workers: int, processes: bool, logger: Any = None) -> None:
        self.problem = problem
        self.n_workers = max(1, int(n_workers))
        self.processes = bool(processes)
        self.logger = logger
        self._tasks: Dict[str, _Task] = {}
        #: task id -> (worker pid or thread ident, start wall time)
        self._started: Dict[str, Tuple[int, float]] = {}
        self._procs: Dict[int, Any] = {}
        self._payload: Optional[bytes] = None
        self._starts: Any = None
        if self.processes:
            try:
                self._payload = pickle.dumps(problem)
            except Exception as exc:
                raise TypeError(
                    "evaluation_method='processes' needs a picklable problem (defined at module level, "
                    f"no lambdas/closures): {exc!r}"
                ) from exc
        self._pool: Any = self._make_pool()

    # -- pool lifecycle ------------------------------------------------------

    def _make_pool(self) -> Any:
        if not self.processes:
            return ThreadPoolExecutor(max_workers=self.n_workers)
        ctx = multiprocessing.get_context("spawn")
        # A fresh queue per pool: a worker killed mid-``put`` may leave the
        # old one's lock held.
        self._starts = ctx.SimpleQueue()
        payload: bytes = self._payload or b""
        initargs: Tuple[Any, ...] = (payload, self._starts)
        return ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_process_worker_init,
            initargs=initargs,
        )

    def _thread_eval(self, task_id: str, point: Any) -> Any:
        self._started[task_id] = (threading.get_ident(), time.time())
        return self.problem(point)

    def _submit_future(self, task_id: str, point: Any) -> Future:
        if self.processes:
            return self._pool.submit(_process_worker_eval, task_id, point)
        return self._pool.submit(self._thread_eval, task_id, point)

    def _replace_pool(self, kill: bool) -> None:
        """Retire the current process pool (killing its workers if asked) and resubmit every task."""
        old = self._pool
        if kill:
            old.kill_workers()
        else:
            old.shutdown(wait=False, cancel_futures=True)
        self._pool = self._make_pool()
        self._procs = {}
        for tid, task in self._tasks.items():
            self._started.pop(tid, None)
            task.future = self._submit_future(tid, task.point)

    # -- public API ----------------------------------------------------------

    def submit(self, task_id: str, point: Any) -> None:
        """Queue one evaluation."""
        self._tasks[task_id] = _Task(point, self._submit_future(task_id, point))
        self._snapshot_procs()

    def _snapshot_procs(self) -> None:
        # Keep our own handles on the worker processes: a broken executor
        # drops its table, and the exit codes are how a crash is attributed.
        procs = getattr(self._pool, "_processes", None)
        if procs:
            self._procs.update(procs)

    def __len__(self) -> int:
        """Tasks queued or running."""
        return len(self._tasks)

    def task_ids(self) -> List[str]:
        return list(self._tasks)

    def _drain_starts(self) -> None:
        if not self.processes or self._starts is None:
            return
        try:
            while not self._starts.empty():
                tid, pid, t = self._starts.get()
                if tid in self._tasks:
                    self._started[tid] = (pid, t)
        except (OSError, EOFError):  # pragma: no cover - queue torn down
            pass
        self._snapshot_procs()

    def poll(self, timeout: Optional[float] = None) -> List[Outcome]:
        """Harvest finished, failed and timed-out tasks without blocking.

        ``timeout`` is the per-evaluation limit in seconds of *running* time
        (``None``: no limit).
        """
        self._drain_starts()
        out: List[Outcome] = []
        now = time.time()
        broken = False
        timed_out: List[str] = []
        for tid, task in list(self._tasks.items()):
            f = task.future
            started = self._started.get(tid)
            if f.done():
                try:
                    exc = f.exception()
                except CancelledError as ce:
                    exc = ce
                if isinstance(exc, BrokenProcessPool):
                    broken = True
                    continue
                del self._tasks[tid]
                self._started.pop(tid, None)
                t0 = started[1] if started else None
                if exc is not None:
                    out.append(Outcome(tid, False, error=repr(exc), started=t0))
                else:
                    out.append(Outcome(tid, True, result=f.result(), started=t0))
            elif timeout is not None and started is not None and now - started[1] > timeout:
                timed_out.append(tid)
        for tid in timed_out:
            self._tasks.pop(tid).future.cancel()
            t0 = self._started.pop(tid)[1]
            action = "killed" if self.processes else "abandoned (a thread cannot be interrupted)"
            out.append(Outcome(tid, False, error="timed out after %.1fs; %s" % (timeout or 0.0, action), started=t0))
        if timed_out and self.processes and not broken:
            self._replace_pool(kill=True)
        if broken:
            out.extend(self._recover_broken())
        return out

    def _recover_broken(self) -> List[Outcome]:
        """The pool broke: report the task whose worker died, resubmit the rest to a fresh pool."""
        self._drain_starts()
        # Give the executor a moment to reap its processes so exit codes are set.
        for p in self._procs.values():
            try:
                p.join(1.0)
            except Exception:  # pragma: no cover
                pass
        out: List[Outcome] = []
        # Started and unfinished when the pool broke: one of them killed it.
        suspects = []
        for tid, task in self._tasks.items():
            f = task.future
            if f.done() and not f.cancelled() and f.exception() is None:
                continue  # finished just before the break: harvested next poll
            if tid in self._started:
                suspects.append(tid)
        codes = {tid: getattr(self._procs.get(self._started[tid][0]), "exitcode", None) for tid in suspects}
        for tid in suspects:
            code = codes[tid]
            if code is not None:
                # Died on its own, or terminated (SIGTERM) by the executor
                # after another worker died.
                culprit = code != -signal.SIGTERM
            elif len(suspects) == 1:
                culprit = True  # the only evaluation that was running
            else:
                task = self._tasks[tid]
                task.strikes += 1  # cannot tell: fail it on its second break
                culprit = task.strikes >= 2
            if culprit:
                del self._tasks[tid]
                t0 = self._started.pop(tid)[1]
                out.append(
                    Outcome(tid, False, error="worker process died (exit code %s) evaluating it" % code, started=t0)
                )
        if self.logger is not None:
            self.logger.error(
                "Process pool broke (a worker died); %d evaluation(s) failed, resubmitting %d."
                % (len(out), len(self._tasks))
            )
        self._replace_pool(kill=False)
        return out

    def wait(self, deadline: Optional[float] = None, poll_interval: float = 1e-3) -> None:
        """Block until a task can be harvested (or ``deadline``)."""
        pending = [t.future for t in self._tasks.values()]
        if not pending:
            return
        rest = None if deadline is None else max(0.0, deadline - time.time())
        rest = poll_interval * 50 if rest is None else min(rest, poll_interval * 50)
        futures_wait(pending, timeout=rest, return_when="FIRST_COMPLETED")

    def close(self, deadline: float) -> int:
        """Cancel queued tasks, give running ones until ``deadline``, then stop the workers.

        Returns the number of evaluations still running at the deadline
        (killed for processes, abandoned for threads).
        """
        for task in self._tasks.values():
            task.future.cancel()
        running = [t.future for t in self._tasks.values() if not t.future.done()]
        still: Any = []
        if running:
            _done, still = futures_wait(running, timeout=max(0.0, deadline - time.time()))
        self._tasks.clear()
        if self.processes:
            if still:
                self._pool.kill_workers()
            else:
                self._pool.shutdown(wait=True, cancel_futures=True)
        else:
            self._pool.shutdown(wait=False, cancel_futures=True)
        return len(still)
