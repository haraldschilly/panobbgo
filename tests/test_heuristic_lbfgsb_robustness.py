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

"""Robustness tests for the multi-start L-BFGS-B heuristic.

Complements ``test_heuristic_lbfgsb.py`` with the failure-mode paths:
non-finite objective values, numerically broken descents, subprocess
start failures, and pipe teardown during restart.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from panobbgo.heuristics.lbfgsb import LBFGSB, _make_pipe_objective
from panobbgo.utils import PanobbgoTestCase


_BOUNDS = [(-1.0, 1.0), (-1.0, 1.0)]
_LB = np.array([-1.0, -1.0])
_UB = np.array([1.0, 1.0])


class _RecordingPipe:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    def send(self, x):
        self.sent.append(np.asarray(x, dtype=float))

    def recv(self):
        if not self._responses:
            raise EOFError
        return self._responses.pop(0)


class PipeObjectiveTests(PanobbgoTestCase):
    def test_non_finite_reply_becomes_inf(self):
        f = _make_pipe_objective(_RecordingPipe([float("nan")]))
        assert f(np.array([0.0, 0.0])) == float("inf")

    def test_none_reply_becomes_inf(self):
        f = _make_pipe_objective(_RecordingPipe([None]))
        assert f(np.array([0.0, 0.0])) == float("inf")

    def test_closed_pipe_raises_systemexit(self):
        f = _make_pipe_objective(_RecordingPipe([]))  # recv -> EOFError
        try:
            f(np.array([0.0, 0.0]))
            assert False, "expected SystemExit"
        except SystemExit:
            pass

    def test_finite_reply_passthrough(self):
        f = _make_pipe_objective(_RecordingPipe([2.5]))
        assert f(np.array([0.0, 0.0])) == 2.5


class _FlakyPipe:
    """Returns a non-finite value for the first descent, then a quadratic.

    Forces the first L-BFGS-B start to encounter a degenerate objective so
    the worker's per-start ``except`` path is exercised; the worker must
    still advance to the next restart rather than crash.
    """

    def __init__(self, switch_after):
        self.calls = 0
        self.switch_after = switch_after
        self.last = None

    def send(self, x):
        self.last = np.asarray(x, dtype=float)

    def recv(self):
        self.calls += 1
        if self.calls <= self.switch_after:
            return float("nan")
        return float(np.sum((self.last - 0.3) ** 2))


class WorkerRobustnessTests(PanobbgoTestCase):
    def test_worker_survives_degenerate_first_descent(self):
        pipe = _FlakyPipe(switch_after=3)
        output = mock.MagicMock()
        # Two starts: the first sees NaNs, the second a clean quadratic.
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=2,
            maxfun=100,
            epsilon=None,
        )
        output.send.assert_called_once_with({"done": 2})

    def test_on_start_handles_unexpected_exception(self):
        h = LBFGSB(self.strategy)
        h.out1 = mock.MagicMock()
        h.out1.poll.return_value = False
        h.p1 = mock.MagicMock()
        # A non-EOF error should be logged and break the loop, not propagate.
        h.p1.poll.side_effect = RuntimeError("boom")
        h._stopped = False
        h.logger = mock.MagicMock()
        h.on_start()
        assert h.logger.error.called


class StartFailureTests(PanobbgoTestCase):
    def test_start_subprocess_exception_wrapped(self):
        h = LBFGSB(self.strategy)
        with mock.patch("multiprocessing.get_context", side_effect=Exception("boom")):
            with self.assertRaises(RuntimeError):
                h.__start__()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_teardown_failure_is_swallowed(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        try:
            # A teardown that raises must not propagate; on_restart should
            # still spawn a fresh worker.
            with mock.patch.object(h.lbfgsb, "terminate", side_effect=OSError("nope")):
                old = h.lbfgsb
                h.on_restart(center=None, reason="test")
                assert h.lbfgsb is not old
                assert h.lbfgsb.is_alive()
        finally:
            h.__stop__()
