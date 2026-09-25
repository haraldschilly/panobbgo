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

The backend of ``evaluation_method = "threaded"`` (:class:`LocalPool`) and
``"processes"`` (:class:`ProcessPool`): evaluations are submitted by task
id and harvested without blocking by :meth:`LocalPool.poll`, which reports
every task exactly once, as a result or as a failure.

What it guarantees beyond a bare executor:

* **A timed-out task is reported with** ``Outcome.timed_out``; the
  strategy books it as a ``NaN`` result (``Result.timed_out``), not a
  failure.  ``evaluation.timeout`` is a limit on *one call*.
* **The timeout counts running time only.**  The clock of a task starts
  when a worker picks it up (a thread records it; a worker process reports
  it over its pipe), so time spent queued behind other evaluations never
  times a task out.
* **Processes: only the timed-out call dies.**  :class:`ProcessPool` runs
  its own worker processes, one task at a time each, and kills exactly the
  worker whose call ran past the limit; a fresh worker replaces it.  Every
  other in-flight evaluation keeps running undisturbed.
* **Threads: a timed-out evaluation is abandoned** — a thread cannot be
  killed, so the call runs to completion in the background and its result
  is discarded.  So that it cannot hold a worker slot forever, the
  executor is retired (``shutdown(wait=False)``) and the tasks that had not
  started yet move to a fresh ``ThreadPoolExecutor``.  ``ThreadPoolExecutor``
  threads are not daemon threads: an abandoned evaluation that never
  returns keeps the interpreter from exiting.  Abandoned threads are
  counted and warned about.  **Use** ``"processes"`` **(or dask) for
  objectives that can hang.**
* **Processes: a crashing evaluation does not end the run.**  A worker
  that dies (``os._exit``, a segfault, the OOM killer, a signal) while
  evaluating fails exactly its own task — attribution is exact, because a
  worker runs one task at a time.  A task whose worker died before the
  call *started* (still loading the problem) is re-queued, not failed.
* **Processes: workers that cannot load the problem are an error.**  If
  :data:`MAX_IDLE_BREAKS` workers in a row die before any evaluation
  started (the problem does not unpickle in a fresh interpreter — a class
  defined in ``__main__`` or a notebook, a ``__setstate__`` that raises),
  respawning cannot help: :class:`WorkerInitError` is raised.
* **Closing is bounded and leaves nothing running** (:meth:`close`): queued
  tasks are cancelled, running ones get until the deadline, then worker
  processes are killed.

Processes mode evaluates *copies* of the problem: each worker unpickles its
own, so state the problem object accumulates while evaluating (counters,
traces, caches, loggers) lives in the workers and is not seen by the
caller's object.
"""

from __future__ import annotations

import collections
import multiprocessing
import pickle
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field
from multiprocessing.connection import wait as connection_wait
from typing import Any, Deque, Dict, List, Optional, Tuple

#: Consecutive worker deaths with no evaluation started before giving up.
MAX_IDLE_BREAKS = 2

#: Abandoned (timed-out) evaluation threads still running before every
#: further timeout is reported at ``ERROR`` level; see :class:`LocalPool`.
ABANDONED_THREADS_WARN = 4


class WorkerInitError(RuntimeError):
    """Worker processes die before evaluating anything: the problem cannot be loaded in them."""


@dataclass
class _Task:
    point: Any
    future: Any = None
    strikes: int = 0
    submitted: float = field(default_factory=time.time)


@dataclass
class Outcome:
    """What became of one task: ``ok`` with its ``result``, or failed with an ``error`` message.

    ``point`` is the evaluated point, so a failure can be reported to the
    module that asked for it (the ``failed_evaluations`` event).
    ``timed_out`` marks a failure by ``evaluation.timeout``: the caller books
    it as a ``NaN`` result rather than a failure.
    """

    task_id: str
    ok: bool
    result: Any = None
    error: str = ""
    started: Optional[float] = None
    finished: float = field(default_factory=time.time)
    point: Any = None
    timed_out: bool = False


class LocalPool:
    """Threads evaluating ``problem`` (see the module docstring).

    ``LocalPool(..., processes=True)`` returns a :class:`ProcessPool`.

    Not thread-safe: one thread (the strategy's main loop) submits, polls
    and closes.
    """

    def __new__(cls, problem: Any, n_workers: int, processes: bool = False, logger: Any = None):
        if processes and cls is LocalPool:
            return super().__new__(ProcessPool)
        return super().__new__(cls)

    def __init__(self, problem: Any, n_workers: int, processes: bool = False, logger: Any = None) -> None:
        self.problem = problem
        self.n_workers = max(1, int(n_workers))
        self.processes = False
        self.logger = logger
        self._tasks: Dict[str, _Task] = {}
        #: task id -> (thread ident, start wall time)
        self._started: Dict[str, Tuple[int, float]] = {}
        #: Starts and finishes seen: the caller's liveness check reads it.
        self.events = 0
        #: Futures of timed-out calls whose thread is still running.
        self._abandoned: List[Future] = []
        self._pool: Any = ThreadPoolExecutor(max_workers=self.n_workers)

    def _thread_eval(self, task_id: str, point: Any) -> Any:
        self._started[task_id] = (threading.get_ident(), time.time())
        self.events += 1
        return self.problem(point)

    def _replace_executor(self) -> None:
        """Retire the executor; tasks that had not started move to a fresh one."""
        old = self._pool
        old.shutdown(wait=False, cancel_futures=True)
        self._pool = ThreadPoolExecutor(max_workers=self.n_workers)
        for tid, task in self._tasks.items():
            if not task.future.cancelled():
                continue  # running (or done) in the old executor: leave it there
            self._started.pop(tid, None)
            task.future = self._pool.submit(self._thread_eval, tid, task.point)

    # -- public API ----------------------------------------------------------

    def submit(self, task_id: str, point: Any) -> None:
        """Queue one evaluation."""
        self._tasks[task_id] = _Task(point, self._pool.submit(self._thread_eval, task_id, point))

    def __len__(self) -> int:
        """Tasks queued or running."""
        return len(self._tasks)

    def running(self) -> int:
        """Tasks that a worker has picked up and that have not been harvested."""
        return sum(1 for tid in self._tasks if tid in self._started)

    def task_ids(self) -> List[str]:
        return list(self._tasks)

    def ages(self, now: Optional[float] = None) -> Tuple[int, Optional[float], Optional[float]]:
        """``(outstanding, oldest running age, oldest queued age)`` in seconds (``None``: none such)."""
        now = time.time() if now is None else now
        running = [now - self._started[tid][1] for tid in self._tasks if tid in self._started]
        queued = [now - task.submitted for tid, task in self._tasks.items() if tid not in self._started]
        return len(self._tasks), max(running, default=None), max(queued, default=None)

    def waiting(self) -> bool:
        """``True`` while an outstanding task is legitimately being waited for.

        A task is running (a worker picked it up and it has not been
        harvested), or it is queued on a pool whose workers can still pick it
        up.  How long a task runs is limited only by ``evaluation.timeout``;
        the caller's deadlock backstop reads this so that it never cuts a
        running evaluation.
        """
        if not self._tasks:
            return False
        if self.running() > 0:
            return True
        return self._workers_alive()

    def _workers_alive(self) -> bool:
        return not getattr(self._pool, "_shutdown", False)

    @property
    def abandoned(self) -> int:
        """Timed-out evaluation threads that are still running."""
        self._abandoned = [f for f in self._abandoned if not f.done()]
        return len(self._abandoned)

    def poll(self, timeout: Optional[float] = None) -> List[Outcome]:
        """Harvest finished, failed and timed-out tasks without blocking.

        ``timeout`` is the per-evaluation limit in seconds of *running* time
        (``None``: no limit).
        """
        out: List[Outcome] = []
        now = time.time()
        timed_out: List[str] = []
        for tid, task in list(self._tasks.items()):
            f = task.future
            started = self._started.get(tid)
            if f.done():
                try:
                    exc = f.exception()
                except CancelledError as ce:
                    exc = ce
                del self._tasks[tid]
                self._started.pop(tid, None)
                self.events += 1
                t0 = started[1] if started else None
                if exc is not None:
                    out.append(Outcome(tid, False, error=repr(exc), started=t0, point=task.point))
                else:
                    out.append(Outcome(tid, True, result=f.result(), started=t0, point=task.point))
            elif timeout is not None and started is not None and now - started[1] > timeout:
                timed_out.append(tid)
        for tid in timed_out:
            task = self._tasks.pop(tid)
            task.future.cancel()
            self._abandoned.append(task.future)
            t0 = self._started.pop(tid)[1]
            out.append(
                Outcome(
                    tid,
                    False,
                    error="timed out after %.1fs; abandoned (a thread cannot be interrupted)" % (timeout or 0.0),
                    started=t0,
                    point=task.point,
                    timed_out=True,
                )
            )
        if timed_out:
            # The abandoned thread keeps its slot, so move the waiting tasks
            # to a fresh executor rather than let n wedged threads starve them.
            self._replace_executor()
            self._warn_abandoned()
        return out

    def _warn_abandoned(self) -> None:
        n = self.abandoned
        if self.logger is None or n == 0:
            return
        msg = (
            "%d timed-out evaluation thread(s) still running in the background (a thread cannot be killed; "
            "their results are discarded). Use evaluation.method 'processes' or 'dask' for objectives that "
            "can hang." % n
        )
        if n >= ABANDONED_THREADS_WARN * self.n_workers:
            self.logger.error(msg)
        else:
            self.logger.warning(msg)

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
        (abandoned: a thread cannot be killed).
        """
        for task in self._tasks.values():
            task.future.cancel()
        running = [t.future for t in self._tasks.values() if not t.future.done()]
        still: Any = []
        if running:
            _done, still = futures_wait(running, timeout=max(0.0, deadline - time.time()))
        self._tasks.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
        return len(still)


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------


def _worker_main(conn: Any, payload: bytes) -> None:
    """A worker process: load the problem once, then evaluate one task at a time.

    Protocol (over ``conn``): receives ``(task_id, point)`` or ``None``
    (exit); sends ``("start", task_id, t)`` before each call — synchronously,
    so the start is on the pipe even if the objective then kills the
    process — and ``("ok", task_id, result)`` / ``("err", task_id, message)``
    after it.
    """
    problem = pickle.loads(payload)  # an exception here ends the process: a failed init
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        task_id, point = msg
        conn.send(("start", task_id, time.time()))
        try:
            result = problem(point)
        except BaseException as exc:  # noqa: BLE001 - reported to the parent, the worker lives on
            conn.send(("err", task_id, repr(exc)))
            continue
        try:
            conn.send(("ok", task_id, result))
        except Exception as exc:  # an unpicklable result
            conn.send(("err", task_id, "result could not be sent: %r" % exc))


class _Worker:
    """One worker process of a :class:`ProcessPool`."""

    def __init__(self, ctx: Any, payload: bytes) -> None:
        self.conn, child = ctx.Pipe(duplex=True)
        # Not a daemon: an objective may start processes of its own.  The
        # pool kills its workers on close; ``_kill_all_at_exit`` covers an
        # interpreter that exits without closing it.
        self.proc = ctx.Process(target=_worker_main, args=(child, payload), daemon=False)
        self.proc.start()
        child.close()
        self.task: Optional[str] = None
        #: The pipe broke on send: reap it at the next poll.
        self.broken = False
        #: Reported the start of at least one evaluation (it loaded the problem).
        self.started_any = False
        _LIVE_WORKERS.add(self)

    @property
    def pid(self) -> int:
        return int(self.proc.pid or 0)

    def kill(self) -> None:
        try:
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(1.0)
        except Exception:  # pragma: no cover
            pass
        try:
            self.conn.close()
        except Exception:  # pragma: no cover
            pass
        _LIVE_WORKERS.discard(self)


_LIVE_WORKERS: "set[_Worker]" = set()


def _kill_all_at_exit() -> None:  # pragma: no cover - interpreter shutdown
    for w in list(_LIVE_WORKERS):
        w.kill()


import atexit  # noqa: E402

atexit.register(_kill_all_at_exit)


class ProcessPool(LocalPool):
    """Spawned worker processes evaluating ``problem``, each one task at a time.

    The pool owns its workers (no ``ProcessPoolExecutor``), so a timed-out
    call is killed by killing exactly its worker, and a crash is attributed
    exactly to the task that worker was running.  Workers are spawned on
    demand, up to ``n_workers``, and replaced when they die.
    """

    def __init__(self, problem: Any, n_workers: int, processes: bool = True, logger: Any = None) -> None:
        self.problem = problem
        self.n_workers = max(1, int(n_workers))
        self.processes = True
        self.logger = logger
        self._tasks: Dict[str, _Task] = {}
        #: task id -> (worker pid, start wall time)
        self._started: Dict[str, Tuple[int, float]] = {}
        self.events = 0
        self._abandoned = []
        self._idle_breaks = 0
        self._queue: Deque[str] = collections.deque()
        self._workers: List[_Worker] = []
        #: Outcomes found outside :meth:`poll` (an unsendable point), reported by the next poll.
        self._pending_outcomes: List[Outcome] = []
        self._ctx = multiprocessing.get_context("spawn")
        try:
            self._payload: bytes = pickle.dumps(problem)
        except Exception as exc:
            raise TypeError(
                "evaluation_method='processes' needs a picklable problem (defined at module level, "
                f"no lambdas/closures): {exc!r}"
            ) from exc
        try:  # cheap probe; a spawned worker can still fail (e.g. a class from __main__)
            pickle.loads(self._payload)
        except Exception as exc:
            raise TypeError(f"evaluation_method='processes': the problem does not unpickle: {exc!r}") from exc
        self._pool = None  # no executor: see _workers

    # -- workers ---------------------------------------------------------------

    def _dispatch(self) -> None:
        """Hand queued tasks to idle workers, spawning workers up to ``n_workers``."""
        while self._queue:
            worker = next((w for w in self._workers if w.task is None and not w.broken), None)
            if worker is None:
                if len(self._workers) >= self.n_workers:
                    return
                worker = _Worker(self._ctx, self._payload)
                self._workers.append(worker)
            tid = self._queue.popleft()
            task = self._tasks[tid]
            try:
                worker.conn.send((tid, task.point))
            except (BrokenPipeError, EOFError, OSError):
                self._queue.appendleft(tid)  # the worker is gone: requeue, reap it in poll
                worker.broken = True
                continue
            except Exception as exc:  # an unpicklable point: fail the task, keep the worker
                del self._tasks[tid]
                self._pending_outcomes.append(
                    Outcome(tid, False, error="point could not be sent: %r" % exc, point=task.point)
                )
                continue
            worker.task = tid

    def _read(self, worker: _Worker, out: List[Outcome]) -> bool:
        """Drain ``worker``'s pipe into ``out``; ``False`` if the worker is gone."""
        try:
            while worker.conn.poll(0):
                kind, tid, payload = worker.conn.recv()
                if kind == "start":
                    worker.started_any = True
                    self._idle_breaks = 0  # an evaluation ran: the workers can load the problem
                    if tid in self._tasks:
                        self._started[tid] = (worker.pid, float(payload))
                        self.events += 1
                    continue
                worker.task = None
                task = self._tasks.pop(tid, None)
                started = self._started.pop(tid, None)
                if task is None:
                    continue
                self.events += 1
                t0 = started[1] if started else None
                if kind == "ok":
                    out.append(Outcome(tid, True, result=payload, started=t0, point=task.point))
                else:
                    out.append(Outcome(tid, False, error=str(payload), started=t0, point=task.point))
        except (EOFError, OSError, pickle.UnpicklingError, ValueError, TypeError):
            return False
        return worker.proc.is_alive()

    def _reap(self, worker: _Worker, out: List[Outcome]) -> None:
        """``worker`` died: fail the task it was evaluating, or re-queue one that never started."""
        worker.kill()
        code = worker.proc.exitcode
        self._workers.remove(worker)
        tid = worker.task
        if tid is not None and tid in self._tasks and tid in self._started:
            task = self._tasks.pop(tid)
            t0 = self._started.pop(tid)[1]
            self.events += 1
            out.append(
                Outcome(
                    tid,
                    False,
                    error="worker process died (exit code %s) evaluating it" % code,
                    started=t0,
                    point=task.point,
                )
            )
            if self.logger is not None:
                self.logger.error("A worker process died (exit code %s) evaluating %s; the run goes on." % (code, tid))
            return
        if tid is not None and tid in self._tasks:
            self._queue.appendleft(tid)  # never started: not its fault
        if worker.started_any:
            return  # an idle worker that had evaluated before: just replace it
        # Died before evaluating anything: loading the problem failed?
        self._idle_breaks += 1
        if self._idle_breaks >= MAX_IDLE_BREAKS:
            raise WorkerInitError(
                "Worker processes cannot initialise the problem: %d worker processes died before any evaluation "
                "started. Is the problem class importable in a fresh interpreter (not defined in __main__ or a "
                "notebook), and does it unpickle without errors?" % self._idle_breaks
            )

    # -- public API ----------------------------------------------------------

    def submit(self, task_id: str, point: Any) -> None:
        """Queue one evaluation."""
        self._tasks[task_id] = _Task(point)
        self._queue.append(task_id)
        self._dispatch()

    def waiting(self) -> bool:
        """Outstanding tasks are always waited for: the pool spawns the workers it needs."""
        return bool(self._tasks)

    def _workers_alive(self) -> bool:
        return True

    def poll(self, timeout: Optional[float] = None) -> List[Outcome]:
        """Harvest finished, failed and timed-out tasks without blocking.

        ``timeout`` is the per-evaluation limit in seconds of *running* time
        (``None``: no limit); a call past it is killed with its worker, and
        only that call.
        """
        out: List[Outcome] = self._pending_outcomes
        self._pending_outcomes = []
        for worker in list(self._workers):
            if not self._read(worker, out) or worker.broken:
                self._read(worker, out)  # the process is gone: whatever it wrote is complete now
                self._reap(worker, out)
        if timeout is not None:
            now = time.time()
            for worker in list(self._workers):
                tid = worker.task
                started = self._started.get(tid) if tid is not None else None
                if started is None or now - started[1] <= timeout:
                    continue
                task = self._tasks.pop(tid)  # type: ignore[arg-type]
                self._started.pop(tid)  # type: ignore[arg-type]
                worker.task = None
                worker.kill()
                self._workers.remove(worker)
                self.events += 1
                out.append(
                    Outcome(
                        tid,  # type: ignore[arg-type]
                        False,
                        error="timed out after %.1fs; its worker process was killed" % timeout,
                        started=started[1],
                        point=task.point,
                        timed_out=True,
                    )
                )
        self._dispatch()
        return out

    def wait(self, deadline: Optional[float] = None, poll_interval: float = 1e-3) -> None:
        """Block until a worker reports something or dies (or ``deadline``)."""
        if not self._tasks:
            return
        rest = None if deadline is None else max(0.0, deadline - time.time())
        rest = poll_interval * 50 if rest is None else min(rest, poll_interval * 50)
        handles: List[Any] = []
        for w in self._workers:
            handles.extend([w.conn, w.proc.sentinel])
        if handles:
            connection_wait(handles, timeout=rest)
        else:
            time.sleep(rest)

    def close(self, deadline: float) -> int:
        """Cancel queued tasks, give running ones until ``deadline``, then kill the workers.

        Returns the number of evaluations still running at the deadline.
        """
        for tid in list(self._queue):
            self._tasks.pop(tid, None)
        self._queue.clear()
        while any(w.task is not None for w in self._workers) and time.time() < deadline:
            sink: List[Outcome] = []
            for w in list(self._workers):
                if w.task is not None and not self._read(w, sink):
                    w.task = None
            self.wait(deadline)
        still = sum(1 for w in self._workers if w.task is not None)
        for w in self._workers:
            if w.task is None:
                try:
                    w.conn.send(None)  # a clean exit
                except Exception:
                    pass
        for w in self._workers:
            if w.task is None:
                w.proc.join(0.5)
            w.kill()
        self._workers = []
        self._tasks.clear()
        self._started.clear()
        return still
