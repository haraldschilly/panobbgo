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
spawns one such child, sends commands over JSON-Lines on stdin, and
reads responses from stdout.

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
from typing import Any, Dict, Optional

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

    Notes
    -----
    The child process is spawned eagerly in ``__init__`` and torn down
    by :meth:`close` (or :meth:`__del__` as a safety net).  Each
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
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        fd = self._stderr.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_APPEND)

        self._worker_dir = Path(worker_dir).resolve() if worker_dir is not None else _resolve_worker_dir()
        self._proc = self._spawn_worker()
        assert self._proc.stdout is not None
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._proc.stdout.fileno(), selectors.EVENT_READ)

        create_kwargs: Dict[str, Any] = {
            "kind": str(kind),
            "instance": int(instance),
            "dim": int(dim),
        }
        if fid is not None:
            create_kwargs["fid"] = int(fid)

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
        """Send ``shutdown`` and wait for the child to exit."""
        if self._closed:
            return
        if self._proc is None:  # never spawned, or killed after a timeout
            self._closed = True
            self._close_selector()
            self._stderr.close()
            return
        self._closed = True
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
