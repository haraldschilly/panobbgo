# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
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
IOHprofiler Problem Wrapper (subprocess client)
================================================

Adapts an IOHprofiler problem (BBOB, MA-BBOB, ...) as a panobbgo
:class:`~panobbgo.lib.lib.Problem`.  The actual ``ioh`` C++ binding is
**not** imported in this process — it lives in an isolated child venv
under ``tools/ioh_worker/``, pinned to Python 3.12 (the latest version
with prebuilt cp-wheels for ``ioh``).  Each :class:`IOHProblem` instance
talks to one such child, sends commands over JSON-Lines on stdin, and
reads responses from stdout.  Children are reused: a closed problem hands
its healthy worker back to a small per-process pool, and the next
:class:`IOHProblem` of the same kind re-targets it with ``create`` instead
of paying ``uv run`` + interpreter + ``import ioh`` again (~0.2–0.4 s per
run, a fifth of a quick run).

This split lets the panobbgo core stay on the newest Python without
being held back by the ``ioh`` wheel coverage matrix.

Setup
-----

::

    cd tools/ioh_worker
    uv sync

The parent process discovers the worker directory by walking up from
this module until it sees ``tools/ioh_worker/pyproject.toml``.  Override
with the ``PANOBBGO_IOH_WORKER`` environment variable.

Protocol
--------

See ``tools/ioh_worker/README.md``.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import selectors
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.lib.lib import Problem

#: Default bound on one worker round-trip, independent of any run timeout:
#: a healthy evaluation answers in milliseconds, so this only ends a wedged
#: worker.  Per instance via ``IOHProblem(call_timeout=...)``.
DEFAULT_CALL_TIMEOUT_S: float = 300.0

#: The worker's stderr file is cut back to its last ``STDERR_KEEP_BYTES``
#: once it exceeds ``STDERR_MAX_BYTES``.
STDERR_MAX_BYTES: int = 1 << 20
STDERR_KEEP_BYTES: int = 64 << 10


_WORKER_DIR_CACHE: Optional[Path] = None

#: Idle workers kept per ``(worker_dir, kind)``; more are shut down.
MAX_IDLE_PER_KEY: int = 2


class _IdleWorker:
    """A live worker between two problems: the process, its selector and stderr file."""

    __slots__ = ("proc", "sel", "stderr", "pid")

    def __init__(self, proc: subprocess.Popen, sel: selectors.BaseSelector, stderr: Any) -> None:
        self.proc = proc
        self.sel = sel
        self.stderr = stderr
        #: The process that owns it (a forked child must not adopt it).
        self.pid = os.getpid()

    def shutdown(self) -> None:
        proc = self.proc
        try:
            if proc.poll() is None:
                assert proc.stdin is not None
                proc.stdin.write(b'{"cmd": "shutdown"}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        for closeable in (proc.stdin, proc.stdout, self.sel, self.stderr):
            try:
                if closeable is not None:
                    closeable.close()
            except Exception:
                pass


_IDLE: Dict[Tuple[str, str], List[_IdleWorker]] = {}
#: Re-entrant: garbage collection can run ``IOHProblem.__del__`` -> ``close``
#: -> :func:`_put_idle` in a thread that already holds it.  Only list
#: bookkeeping happens under it; stopping a worker (up to seconds) never does.
_IDLE_LOCK = threading.RLock()


def _take_idle(key: Tuple[str, str]) -> Optional[_IdleWorker]:
    """An idle, still-running worker for ``key`` owned by this process, or ``None``."""
    dead: List[_IdleWorker] = []
    found: Optional[_IdleWorker] = None
    with _IDLE_LOCK:
        stack = _IDLE.get(key)
        while stack:
            w = stack.pop()
            if w.pid != os.getpid():  # inherited through fork: not ours to use or stop
                continue
            if w.proc.poll() is None:
                found = w
                break
            dead.append(w)
    for w in dead:  # outside the lock: shutdown may block
        w.shutdown()
    return found


def _put_idle(key: Tuple[str, str], worker: _IdleWorker) -> None:
    with _IDLE_LOCK:
        stack = _IDLE.setdefault(key, [])
        stack.append(worker)
        evicted = stack[:-MAX_IDLE_PER_KEY] if len(stack) > MAX_IDLE_PER_KEY else []
        del stack[: len(evicted)]
    for w in evicted:
        w.shutdown()


def shutdown_idle_workers() -> None:
    """Stop every idle worker of this process (also run at interpreter exit)."""
    with _IDLE_LOCK:
        workers = [w for stack in _IDLE.values() for w in stack if w.pid == os.getpid()]
        _IDLE.clear()
    for w in workers:
        w.shutdown()


atexit.register(shutdown_idle_workers)


def _resolve_worker_dir() -> Path:
    """Locate the ``tools/ioh_worker`` uv project.

    Resolution order:

    1. ``PANOBBGO_IOH_WORKER`` env var (absolute path)
    2. Walk up from this file until we find ``tools/ioh_worker/pyproject.toml``

    Raises :class:`FileNotFoundError` if neither yields a directory.
    """
    global _WORKER_DIR_CACHE
    if _WORKER_DIR_CACHE is not None:
        return _WORKER_DIR_CACHE

    env = os.environ.get("PANOBBGO_IOH_WORKER")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "pyproject.toml").exists():
            raise FileNotFoundError(f"PANOBBGO_IOH_WORKER={env!r} does not contain a pyproject.toml")
        _WORKER_DIR_CACHE = p
        return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tools" / "ioh_worker"
        if (candidate / "pyproject.toml").exists():
            _WORKER_DIR_CACHE = candidate
            return candidate

    raise FileNotFoundError(
        "Cannot locate tools/ioh_worker/. Set PANOBBGO_IOH_WORKER or run from a panobbgo source checkout."
    )


def worker_available() -> bool:
    """Return ``True`` iff the ``ioh`` worker subprocess can be spawned.

    Checks that the worker directory exists and that ``uv`` is on ``PATH``.
    Does *not* try to spawn the worker (the spawn itself triggers
    ``uv sync`` which can be expensive); use this for cheap test-skip
    predicates.
    """
    if shutil.which("uv") is None:
        return False
    try:
        worker_dir = _resolve_worker_dir()
    except FileNotFoundError:
        return False
    venv = worker_dir / ".venv"
    return venv.exists()


class IOHProblem(Problem):
    """Panobbgo :class:`Problem` backed by an IOH problem running in a child venv.

    Parameters
    ----------
    kind
        Problem family.  Currently ``"MA-BBOB"`` (default IOH MA-BBOB
        anytime competition) or ``"BBOB"`` (single BBOB function;
        requires ``fid``).
    instance
        Integer instance id passed to the underlying IOH builder.
    dim
        Problem dimensionality.
    fid
        BBOB function id (1..24).  Required when ``kind == "BBOB"``,
        ignored otherwise.
    worker_dir
        Override the path to the worker uv project.  Defaults to the
        result of :func:`_resolve_worker_dir`.
    reuse_worker
        ``True`` (default): take an idle worker of the same
        ``(worker_dir, kind)`` if there is one, and hand this one back on
        :meth:`close`.  ``False``: a private worker, shut down on close.

    Notes
    -----
    The child process is acquired eagerly in ``__init__`` — an idle one
    re-targeted with ``create`` (the protocol replaces the worker's problem;
    ``tools/ioh_worker/README.md``), else a fresh spawn — and released by
    :meth:`close` (or :meth:`__del__` as a safety net).  Only a worker
    in a known-good state goes back to the pool: running, no unread output,
    never timed out or killed.  A crashed or timed-out worker is discarded
    as before, so a run cannot inherit a broken process.  Each
    :meth:`eval` does one synchronous JSON-Lines round-trip behind a
    per-instance lock; the underlying IOH problem object is not
    thread-safe in C++ so serialising is correct.

    The worker's stderr goes to an anonymous temp file, never a pipe: a
    pipe nobody drains blocks a chatty worker once it holds 64 KB.  The
    file is opened ``O_APPEND`` and read with :func:`os.pread`, so this
    process never moves the offset the child writes at; it is cut back to
    its last :data:`STDERR_KEEP_BYTES` once it exceeds
    :data:`STDERR_MAX_BYTES`.  Its tail is quoted in the error when the
    worker dies.

    Every round-trip is bounded: by ``call_timeout`` seconds (default
    :data:`DEFAULT_CALL_TIMEOUT_S`, ``None`` = unbounded) and by
    :attr:`deadline` (a :func:`time.monotonic` instant, ``None`` = none),
    whichever comes first.  A worker that has not answered by then is
    killed and the call raises :class:`TimeoutError`, instead of blocking
    the run forever.
    """

    def __init__(
        self,
        kind: str,
        instance: int,
        dim: int,
        *,
        fid: Optional[int] = None,
        worker_dir: Optional[Path] = None,
        call_timeout: Optional[float] = DEFAULT_CALL_TIMEOUT_S,
        reuse_worker: bool = True,
    ) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._sel: Optional[selectors.BaseSelector] = None
        self._lock = threading.Lock()
        self._closed = False
        self._rbuf = b""
        #: Absolute :func:`time.monotonic` deadline for worker round-trips.
        self.deadline: Optional[float] = None
        #: Seconds one round-trip may take (``None``: unbounded).
        self.call_timeout: Optional[float] = call_timeout
        self._worker_dir = Path(worker_dir).resolve() if worker_dir is not None else _resolve_worker_dir()
        self._pool_key: Optional[Tuple[str, str]] = (str(self._worker_dir), str(kind)) if reuse_worker else None

        idle = _take_idle(self._pool_key) if self._pool_key is not None else None
        if idle is not None:
            self._stderr = idle.stderr
            self._proc = idle.proc
            self._sel = idle.sel
        else:
            self._start_fresh_worker()

        create_kwargs: Dict[str, Any] = {
            "kind": str(kind),
            "instance": int(instance),
            "dim": int(dim),
        }
        if fid is not None:
            create_kwargs["fid"] = int(fid)

        try:
            meta = self._call("create", **create_kwargs)
        except (RuntimeError, OSError):
            # An adopted idle worker that died since it was parked (its
            # ``create`` found no process or a closed pipe): start a fresh
            # one.  A protocol error from a live worker is the caller's.
            if idle is None or (self._proc is not None and self._proc.poll() is None):
                raise
            self._kill()
            try:
                self._stderr.close()
            except Exception:
                pass
            self._rbuf = b""
            self._start_fresh_worker()
            meta = self._call("create", **create_kwargs)

        lb = np.asarray(meta["lb"], dtype=np.float64)
        ub = np.asarray(meta["ub"], dtype=np.float64)
        box = list(zip(lb.tolist(), ub.tolist()))
        super().__init__(box=box)

        self._optimum_y: float = float(meta["optimum_y"])
        self.ioh_problem_id: int = int(meta["problem_id"])
        self.ioh_instance: int = int(meta["instance"])
        self.ioh_name: str = str(meta["name"])
        self.kind: str = str(kind)

    # ------------------------------------------------------------------
    # panobbgo.Problem API
    # ------------------------------------------------------------------

    @property
    def optimum_y(self) -> float:
        """Known optimum f-value (IOH problems are shifted to a known minimum)."""
        return self._optimum_y

    def eval(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        resp = self._call("eval", x=x.tolist())
        return float(resp["fx"])

    def eval_constraints(self, x: np.ndarray):
        # MA-BBOB / BBOB are unconstrained.
        return None

    def reset(self) -> None:
        """Reset the wrapped IOH problem's evaluation counter."""
        self._call("reset")

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_fresh_worker(self) -> None:
        """A new stderr file and a newly spawned worker (not from the idle pool)."""
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        fd = self._stderr.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_APPEND)
        self._proc = self._spawn_worker()
        assert self._proc.stdout is not None
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._proc.stdout.fileno(), selectors.EVENT_READ)

    def _spawn_worker(self) -> subprocess.Popen:
        cmd = [
            "uv",
            "run",
            "--project",
            str(self._worker_dir),
            "python",
            "-m",
            "ioh_worker",
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            bufsize=0,  # raw pipes: stdout is read with os.read (see _readline)
        )

    def _stderr_tail(self, limit: int = 4000) -> str:
        """The last ``limit`` bytes the worker wrote to stderr (``pread``: no offset moves)."""
        try:
            fd = self._stderr.fileno()
            size = os.fstat(fd).st_size
            start = max(0, size - limit)
            return os.pread(fd, size - start, start).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _trim_stderr(self) -> None:
        """Keep the stderr file bounded: cut it back to its tail once it is large."""
        try:
            fd = self._stderr.fileno()
            size = os.fstat(fd).st_size
            if size <= STDERR_MAX_BYTES:
                return
            tail = os.pread(fd, STDERR_KEEP_BYTES, size - STDERR_KEEP_BYTES)
            os.ftruncate(fd, 0)
            os.write(fd, tail)  # O_APPEND: lands at the new end, as the child's writes do
        except Exception:
            pass

    def _readline(self) -> bytes:
        """One response line from the worker; ``b""`` on EOF.

        Reads the raw fd so the selector can bound the wait (a buffered
        text reader would hide data from it).  Raises :class:`TimeoutError`
        past ``call_timeout`` or :attr:`deadline`.
        """
        assert self._proc is not None and self._proc.stdout is not None and self._sel is not None
        fd = self._proc.stdout.fileno()
        limits = [] if self.deadline is None else [self.deadline]
        if self.call_timeout is not None:
            limits.append(time.monotonic() + float(self.call_timeout))
        until = min(limits) if limits else None
        while b"\n" not in self._rbuf:
            timeout = None if until is None else max(0.0, until - time.monotonic())
            if not self._sel.select(timeout):
                raise TimeoutError("IOH worker did not answer in time")
            chunk = os.read(fd, 65536)
            if not chunk:
                return b""
            self._rbuf += chunk
        line, _, self._rbuf = self._rbuf.partition(b"\n")
        return line + b"\n"

    def _close_selector(self) -> None:
        if self._sel is not None:
            try:
                self._sel.close()
            except Exception:
                pass
            self._sel = None

    def _kill(self) -> None:
        """Kill a worker whose protocol state is unknown (e.g. after a timeout)."""
        self._close_selector()
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
            for pipe in (proc.stdin, proc.stdout):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:
                    pass

    def _call(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("IOH worker is not running")
        msg = {"cmd": cmd, **kwargs}
        line = json.dumps(msg)
        with self._lock:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write((line + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            try:
                resp_line = self._readline()
            except TimeoutError:
                # A late answer would desynchronise every later call.
                self._kill()
                raise
            self._trim_stderr()
        if not resp_line:
            raise RuntimeError(
                f"IOH worker closed stdout while waiting for response to {cmd!r}. "
                f"stderr: {self._stderr_tail().strip()!r}"
            )
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise RuntimeError(f"IOH worker error on {cmd!r}: {resp.get('error', 'unknown')}")
        return resp

    def close(self) -> None:
        """Release the worker: back to the idle pool if healthy, else ``shutdown``."""
        if self._closed:
            return
        if self._proc is None:  # never spawned, or killed after a timeout
            self._closed = True
            self._close_selector()
            self._stderr.close()
            return
        self._closed = True
        if self._pool_key is not None and self._release_to_pool():
            return
        try:
            if self._proc.poll() is None:
                try:
                    with self._lock:
                        assert self._proc.stdin is not None and self._proc.stdout is not None
                        self._proc.stdin.write((json.dumps({"cmd": "shutdown"}) + "\n").encode("utf-8"))
                        self._proc.stdin.flush()
                        self.deadline = time.monotonic() + 5.0
                        self._readline()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        finally:
            self._proc = None
            self._close_selector()
            try:
                self._stderr.close()
            except Exception:
                pass

    def _release_to_pool(self) -> bool:
        """Hand a worker in a known-good state to the idle pool; ``True`` if it was taken.

        Known-good: still running, nothing unread (every request answered),
        and never timed out (a timeout kills it and clears ``_proc``).
        Under the call lock, so an evaluation racing with ``close`` either
        finishes first or finds the worker gone ("not running").
        """
        with self._lock:
            proc, sel = self._proc, self._sel
            if proc is None or sel is None or proc.poll() is not None or self._rbuf:
                return False
            try:
                if sel.select(0):  # unsolicited output: protocol state unknown
                    return False
            except Exception:
                return False
            self._proc = None
            self._sel = None
        assert self._pool_key is not None
        _put_idle(self._pool_key, _IdleWorker(proc, sel, self._stderr))
        return True

    def __del__(self) -> None:  # pragma: no cover — best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def fingerprint(self) -> str:
        """Storage identity: suite kind, function id, instance and dimension."""
        return f"IOHProblem(kind={self.kind!r}, id={self.ioh_problem_id}, instance={self.ioh_instance}, dim={self.dim})"

    def __repr__(self) -> str:
        return (
            f"IOHProblem(kind={self.kind!r}, id={self.ioh_problem_id}, "
            f"instance={self.ioh_instance}, dim={self.dim}, name={self.ioh_name!r})"
        )
