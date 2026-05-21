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

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from panobbgo.lib.lib import Problem


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
    """

    def __init__(
        self,
        kind: str,
        instance: int,
        dim: int,
        *,
        fid: Optional[int] = None,
        worker_dir: Optional[Path] = None,
    ) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._closed = False

        self._worker_dir = Path(worker_dir).resolve() if worker_dir is not None else _resolve_worker_dir()
        self._proc = self._spawn_worker()

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
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

    def _call(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("IOH worker is not running")
        msg = {"cmd": cmd, **kwargs}
        line = json.dumps(msg)
        with self._lock:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = ""
            if self._proc.stderr is not None:
                try:
                    stderr = self._proc.stderr.read() or ""
                except Exception:
                    stderr = ""
            raise RuntimeError(
                f"IOH worker closed stdout while waiting for response to {cmd!r}. stderr: {stderr.strip()!r}"
            )
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise RuntimeError(f"IOH worker error on {cmd!r}: {resp.get('error', 'unknown')}")
        return resp

    def close(self) -> None:
        """Send ``shutdown`` and wait for the child to exit."""
        if self._closed or self._proc is None:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                try:
                    with self._lock:
                        assert self._proc.stdin is not None and self._proc.stdout is not None
                        self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                        self._proc.stdin.flush()
                        self._proc.stdout.readline()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        finally:
            self._proc = None

    def __del__(self) -> None:  # pragma: no cover — best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"IOHProblem(kind={self.kind!r}, id={self.ioh_problem_id}, "
            f"instance={self.ioh_instance}, dim={self.dim}, name={self.ioh_name!r})"
        )
