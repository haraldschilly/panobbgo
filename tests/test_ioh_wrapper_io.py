# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""IOHProblem's worker I/O, against fake workers (no ``ioh`` venv needed).

* stderr is not a pipe nobody reads — a worker writing more than a pipe
  buffer (64 KB) to stderr used to block forever;
* a round-trip is bounded by :attr:`IOHProblem.deadline`, and a worker
  that misses it is killed;
* a dead worker's stderr tail is quoted in the error.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest

from panobbgo.lib.ioh_wrapper import IOHProblem

_FAKE = r"""
import json, sys, time
MODE = sys.argv[1]
for line in sys.stdin:
    req = json.loads(line)
    cmd = req["cmd"]
    if MODE == "chatty":
        sys.stderr.write("x" * 200_000 + "\n")
        sys.stderr.flush()
    if cmd == "create":
        resp = {"ok": True, "lb": [-5.0, -5.0], "ub": [5.0, 5.0], "optimum_y": 0.0,
                "problem_id": 1, "instance": 0, "name": "fake"}
    elif cmd == "eval":
        if MODE == "hang":
            time.sleep(600)
        if MODE == "die":
            sys.stderr.write("boom: worker crashed\n")
            sys.stderr.flush()
            sys.exit(3)
        resp = {"ok": True, "fx": float(sum(v * v for v in req["x"]))}
    elif cmd == "shutdown":
        print(json.dumps({"ok": True}), flush=True)
        break
    else:
        resp = {"ok": True}
    print(json.dumps(resp), flush=True)
"""


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    """Make IOHProblem spawn ``_FAKE`` in the given mode."""

    def use(mode: str) -> None:
        def spawn(self: IOHProblem) -> subprocess.Popen:
            return subprocess.Popen(
                [sys.executable, "-c", _FAKE, mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                bufsize=0,
            )

        monkeypatch.setattr(IOHProblem, "_spawn_worker", spawn)

    return lambda mode: (use(mode), tmp_path)[1]


def test_chatty_worker_does_not_block(fake_worker):
    wd = fake_worker("chatty")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        p.deadline = time.monotonic() + 30.0
        # 20 × 200 KB of stderr: a PIPE would have filled after the first.
        for _ in range(20):
            assert p.eval(np.array([1.0, 2.0])) == pytest.approx(5.0)
    finally:
        p.close()


def test_hung_worker_times_out_and_is_killed(fake_worker):
    wd = fake_worker("hang")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    proc = p._proc
    assert proc is not None
    p.deadline = time.monotonic() + 0.5
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        p.eval(np.zeros(2))
    assert time.monotonic() - t0 < 10.0
    assert proc.poll() is not None  # killed, not left to answer late
    with pytest.raises(RuntimeError, match="not running"):
        p.eval(np.zeros(2))
    p.close()


def test_dead_worker_reports_its_stderr(fake_worker):
    wd = fake_worker("die")
    p = IOHProblem(kind="MA-BBOB", instance=0, dim=2, worker_dir=wd)
    try:
        with pytest.raises(RuntimeError, match="boom: worker crashed"):
            p.eval(np.zeros(2))
    finally:
        p.close()
