# -*- coding: utf8 -*-
# Copyright 2012 - 2026 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resource hygiene for long local runs (benchmarks, self-improvement loops).

Optimizer progress is measured in objective evaluations, never in wall
time, so throttling the CPU never changes a result.  Every long-running
entry point therefore

* lowers its own scheduling priority (``nice``) so the machine stays
  usable while a battery runs, and
* refuses to start when the free memory is below a floor, instead of
  pushing the desktop into swap.

Use :func:`add_arguments` on the script's ``ArgumentParser`` and
:func:`apply` on the parsed namespace.

Independent runs (one per seed and battery cell) are deterministic under
``sync_eval``, so they can run in parallel worker processes without
changing a number: :class:`TaskPool` / :func:`shared_pool`, selected with
``--jobs N`` (:func:`add_jobs_argument`) or a screen's ``jobs=N``.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

DEFAULT_NICENESS = 15
DEFAULT_MIN_FREE_GB = 2.0


def be_nice(niceness: int = DEFAULT_NICENESS) -> int:
    """Raise the process niceness to at least ``niceness`` (never lower it).

    Child processes (evaluation workers, the IOH worker) inherit it.
    Returns the effective niceness afterwards.
    """
    try:
        current = os.nice(0)
        if current < niceness:
            return os.nice(niceness - current)
        return current
    except (AttributeError, OSError):  # not POSIX, or not permitted
        return 0


def available_memory_gb() -> Optional[float]:
    """``MemAvailable`` from ``/proc/meminfo`` in GiB, or ``None`` if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0**2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def check_free_memory(min_free_gb: float = DEFAULT_MIN_FREE_GB) -> None:
    """Abort with a clear message when less than ``min_free_gb`` GiB is available."""
    avail = available_memory_gb()
    if avail is not None and avail < min_free_gb:
        sys.exit(
            "refusing to start: %.1f GiB memory available, floor is %.1f GiB "
            "(override with --min-free-mem-gb)" % (avail, min_free_gb)
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--nice`` / ``--no-nice`` / ``--min-free-mem-gb`` to ``parser``.

    A parser with subcommands gets the flags on each subparser instead, so
    they can be given after the subcommand name.
    """
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if subparsers:
        for action in subparsers:
            for sub in action.choices.values():
                add_arguments(sub)
        return
    g = parser.add_argument_group("local-run hygiene")
    g.add_argument(
        "--nice",
        type=int,
        default=DEFAULT_NICENESS,
        metavar="N",
        help="run at this niceness so the machine stays responsive (default: %(default)s)",
    )
    g.add_argument("--no-nice", action="store_true", help="keep the normal scheduling priority")
    g.add_argument(
        "--min-free-mem-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        metavar="GB",
        help="refuse to start below this much available memory; 0 disables (default: %(default)s)",
    )


def apply(args: argparse.Namespace) -> None:
    """Apply the flags added by :func:`add_arguments`."""
    if args.min_free_mem_gb > 0:
        check_free_memory(args.min_free_mem_gb)
    if not args.no_nice:
        be_nice(args.nice)


# -- a process pool for independent runs -------------------------------------


def add_jobs_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--jobs N`` (independent runs in parallel worker processes) to ``parser``."""
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="run independent (seed, cell) runs in N worker processes; results do not depend on N "
        "(default: %(default)s, in-process)",
    )


def screen_jobs(opts: Mapping[str, str]) -> int:
    """``jobs=N`` from a screen's ``key=value`` options (default ``1``).

    With ``N > 1`` the memory floor is checked and this process niced
    before the pool starts (the workers nice themselves too).
    """
    jobs = max(int(opts.get("jobs") or 1), 1)
    if jobs > 1:
        check_free_memory(DEFAULT_MIN_FREE_GB)
        be_nice(DEFAULT_NICENESS)
    return jobs


def _worker_init(niceness: Optional[int]) -> None:
    if niceness is not None:
        be_nice(niceness)


@contextlib.contextmanager
def _main_not_reimported():
    """Keep spawned workers from re-running the caller's ``__main__`` script.

    A ``spawn`` child re-executes the parent's main *script* (as
    ``__mp_main__``) unless the main module's ``__spec__`` names a
    ``__main__`` module — the case ``python -m pkg`` relies on.  The
    benchmark screens are plain top-level scripts without a
    ``if __name__ == "__main__"`` guard, so re-running them in every worker
    would start the whole screen again.  The tasks this pool runs are
    functions of importable ``panobbgo`` modules, so a worker needs nothing
    from the script: the main module gets a ``__main__`` spec while a task is
    submitted, which is when the executor starts the workers it needs.
    """
    main = sys.modules.get("__main__")
    if main is None or getattr(main, "__spec__", None) is not None:
        yield
        return
    import importlib.machinery

    main.__spec__ = importlib.machinery.ModuleSpec("__main__", None)
    try:
        yield
    finally:
        main.__spec__ = None


class TaskPool:
    """A ``spawn`` process pool for independent, deterministic runs.

    ``map(fn, tasks)`` runs ``fn(**task)`` for every task and returns the
    results **in task order**, whatever order they finish in — so a caller
    that folds them gets the same answer for every ``jobs``.  ``fn`` and the
    task payloads must pickle by reference to importable modules (nothing
    defined in a script's ``__main__``).

    Workers lower their priority to ``niceness`` (``None`` leaves it) and a
    new task is not handed out while less than ``min_free_gb`` GiB of memory
    is available and another task is still running (it waits for one to
    finish instead of pushing the machine into swap).

    ``jobs <= 1`` runs everything in the calling process.
    """

    def __init__(
        self,
        jobs: int,
        *,
        niceness: Optional[int] = DEFAULT_NICENESS,
        min_free_gb: float = DEFAULT_MIN_FREE_GB,
    ) -> None:
        self.jobs = max(int(jobs), 1)
        self.niceness = niceness
        self.min_free_gb = float(min_free_gb)
        self._executor: Optional[ProcessPoolExecutor] = None
        if self.jobs > 1:
            ctx = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.jobs, mp_context=ctx, initializer=_worker_init, initargs=(niceness,)
            )

    def _memory_low(self) -> bool:
        if self.min_free_gb <= 0:
            return False
        avail = available_memory_gb()
        return avail is not None and avail < self.min_free_gb

    def map(
        self,
        fn: Callable[..., Any],
        tasks: Sequence[Dict[str, Any]],
        on_done: Optional[Callable[[int, Any], None]] = None,
    ) -> List[Any]:
        """``[fn(**t) for t in tasks]``, in parallel; ``on_done(i, result)`` as each finishes."""
        results: List[Any] = [None] * len(tasks)
        if self._executor is None:
            for i, task in enumerate(tasks):
                results[i] = fn(**task)
                if on_done is not None:
                    on_done(i, results[i])
            return results
        pending: Dict[Future, int] = {}
        todo = list(enumerate(tasks))
        todo.reverse()
        while todo or pending:
            while todo and len(pending) < self.jobs and not (pending and self._memory_low()):
                i, task = todo.pop()
                # ``submit`` is where the executor starts a worker it needs.
                with _main_not_reimported():
                    pending[self._executor.submit(_call, fn, task)] = i
            done, _ = futures_wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                i = pending.pop(fut)
                results[i] = fut.result()
                if on_done is not None:
                    on_done(i, results[i])
        return results

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "TaskPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _call(fn: Callable[..., Any], task: Dict[str, Any]) -> Any:
    return fn(**task)


_SHARED: Dict[int, TaskPool] = {}


def shared_pool(jobs: int) -> TaskPool:
    """A process-wide :class:`TaskPool` with ``jobs`` workers, created on first use.

    Callers that run many batches (one per seed) reuse the same workers
    instead of paying the interpreter start-up per batch; the pools are
    closed at exit.
    """
    jobs = max(int(jobs), 1)
    pool = _SHARED.get(jobs)
    if pool is None:
        pool = _SHARED[jobs] = TaskPool(jobs)
        if len(_SHARED) == 1:
            atexit.register(_close_shared)
    return pool


def _close_shared() -> None:
    for pool in _SHARED.values():
        pool.close()
    _SHARED.clear()
