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

"""Restarts of the pipe-bridge arms (L-BFGS-B, COBYQA).

Regressions:

* ``on_restart`` respawned the worker and rebound ``p1`` on the event-bus
  thread while the main thread could be inside ``_bridge_next_point``;
* a point of the replaced worker still in flight answered the new worker's
  first ``f(x)``;
* ``_bridge_finished`` set ``_stopped``, so a converged COBYQA / an L-BFGS-B at
  ``max_starts`` ignored every later restart.
"""

from unittest import mock

import numpy as np

from panobbgo.heuristics.cobyqa import COBYQA
from panobbgo.heuristics.lbfgsb import LBFGSB
from panobbgo.lib import Point, Result
from panobbgo.utils import PanobbgoTestCase


def _live(h, attr):
    h.out1 = mock.MagicMock()
    h.out1.poll.return_value = False
    h.p1 = mock.MagicMock()
    proc = mock.MagicMock()
    proc.is_alive.return_value = True
    setattr(h, attr, proc)
    return h


ARMS = [(LBFGSB, "lbfgsb"), (COBYQA, "cobyqa")]


class BridgeRestartTests(PanobbgoTestCase):
    def _arm(self, cls, attr):
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        return _live(cls(self.strategy), attr)

    def test_on_restart_only_records(self):
        for cls, attr in ARMS:
            h = self._arm(cls, attr)
            h.p1.poll.return_value = True
            h.p1.recv.return_value = np.array([0.3, 0.3])
            p1, proc = h.p1, getattr(h, attr)
            with mock.patch.object(cls, "_bridge_respawn") as respawn:
                h.on_restart(center=np.zeros(2), reason="t")
                # Nothing touched on the bus thread ...
                respawn.assert_not_called()
                assert h.p1 is p1 and getattr(h, attr) is proc
                proc.terminate.assert_not_called()
                assert h.can_produce
                # ... the next produce (main thread) respawns.
                h.produce(1)
                respawn.assert_called_once()
                np.testing.assert_array_equal(respawn.call_args.args[0], np.zeros(2))

    def test_stale_result_does_not_answer_the_new_worker(self):
        for cls, attr in ARMS:
            h = self._arm(cls, attr)
            h.p1.poll.return_value = True
            h.p1.recv.return_value = np.array([0.1, 0.1])
            (old_pt,) = h.produce(1)  # the old worker's point, in flight
            with mock.patch.object(cls, "_bridge_respawn"):
                h.on_restart(center=None, reason="t")
                h.p1.recv.return_value = np.array([0.7, 0.7])
                (new_pt,) = h.produce(1)  # the new worker's first point
            np.testing.assert_allclose(new_pt.x, [0.7, 0.7])
            # The old point's value lands now: it must be ignored ...
            h.on_new_results([Result(old_pt, 111.0)])
            assert h._fx_inbox.empty()
            # ... and only the new point's value answers the new worker.
            h.on_new_results([Result(new_pt, 2.0)])
            assert h._fx_inbox.get_nowait() == 2.0

    def test_finished_worker_still_restarts(self):
        for cls, attr in ARMS:
            h = self._arm(cls, attr)
            getattr(h, attr).is_alive.return_value = False
            h.p1.poll.return_value = False
            assert h.produce(1) == []  # converged / max_starts reached
            assert h._bridge_done and not h._stopped
            assert not h.can_produce and not h.active

            def respawn(center, h=h, attr=attr):
                getattr(h, attr).is_alive.return_value = True

            with mock.patch.object(cls, "_bridge_respawn", side_effect=respawn) as rs:
                h.on_restart(center=np.zeros(2), reason="t")
                assert h.can_produce and h.active
                h.p1.poll.return_value = True
                h.p1.recv.return_value = np.array([0.2, 0.2])
                pts = h.produce(1)
                rs.assert_called_once()
            assert not h._bridge_done
            assert len(pts) == 1

    def test_restart_after_stop_is_ignored(self):
        for cls, attr in ARMS:
            h = self._arm(cls, attr)
            h._stopped = True
            with mock.patch.object(cls, "_bridge_respawn") as respawn:
                h.on_restart(center=np.zeros(2), reason="t")
                h.produce(1)
                respawn.assert_not_called()


class _FakeCtx:
    """Captures the worker ``Process`` arguments instead of spawning."""

    def __init__(self):
        self.seeds = []

    def Pipe(self, duplex=True):
        return mock.MagicMock(), mock.MagicMock()

    def Process(self, target, args, name):
        self.seeds.append(args[6])
        return mock.MagicMock()


class LBFGSBWorkerSeedTests(PanobbgoTestCase):
    def _seeds(self, seed):
        h = LBFGSB(self.init_strategy(), seed=seed)
        ctx = _FakeCtx()
        with mock.patch("panobbgo.heuristics.lbfgsb.multiprocessing.get_context", return_value=ctx):
            h.__start__()
            h._bridge_respawn(None)
            h._bridge_respawn(None)
        return h._worker_seed, ctx.seeds

    def test_respawn_draws_a_fresh_worker_seed(self):
        """Regression: the worker seed was fixed at construction, so every
        restart replayed the first worker's multi-start ``x0`` sequence."""
        for seed in (None, 7):
            first, seeds = self._seeds(seed)
            assert seeds[0] == first  # the first worker is unchanged
            assert len(set(seeds)) == 3  # every respawn gets its own seed
            assert self._seeds(seed)[1] == seeds  # ... reproducibly


def test_result_point_roundtrip():
    """``Result.x`` is the emitted (projected) point, which the bridge matches on."""
    p = Point(np.array([0.25, 0.5]), "LBFGSB")
    assert np.array_equal(Result(p, 1.0).x, p.x)
