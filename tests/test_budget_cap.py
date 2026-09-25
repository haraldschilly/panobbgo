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

"""``max_eval`` is a hard cap in every strategy and evaluation mode.

``StrategyBase._run`` used to dispatch the whole ``execute()`` batch, so a
run overshot its budget by up to one batch (37 → 38, 523 → 535 evaluations).
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.core import Heuristic
from panobbgo.lib.classic import Rosenbrock


def _strategies():
    from panobbgo.strategies import StrategyRewarding, StrategyRoundRobin, StrategyUCB

    return [StrategyRewarding, StrategyRoundRobin, StrategyUCB]


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize("max_eval", [37, 53])
@pytest.mark.parametrize("strategy_idx", [0, 1, 2], ids=["Rewarding", "RoundRobin", "UCB"])
def test_run_evaluates_exactly_max_eval(strategy_idx, max_eval, sync):
    from panobbgo.heuristics import Nearby, Random

    cls = _strategies()[strategy_idx]
    s = cls(Rosenbrock(dim=3), parse_args=False, testing_mode=True, seed=7)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = sync
    s.config.stop_on_convergence = False
    s.add(Random)
    s.add(Nearby)
    s.start()
    assert len(s.results) == max_eval


def test_clamp_counts_pending_and_dispatched_evaluations():
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 10
    s.pending = {"a": None, "b": None, "c": None}
    assert len(s._clamp_to_budget(list(range(20)))) == 7
    # Everything is dispatched: nothing more, whether or not results arrived.
    s.pending = {}
    assert s._clamp_to_budget([1, 2]) == []
    s._dispatched = 0
    assert s._clamp_to_budget([1, 2]) == [1, 2]
    assert s._dispatched == 2


def test_surplus_goes_back_to_its_heuristic_in_order():
    from panobbgo.lib import Point
    from panobbgo.strategies import StrategyRoundRobin

    class Holder(Heuristic):
        pass

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 2
    h = Holder(s)
    s.add_heuristic(h)
    h._put(Point(np.array([9.0, 9.0]), "Holder"))
    batch = [Point(np.array([float(i), 0.0]), "Holder") for i in range(5)]
    assert s._clamp_to_budget(batch) == batch[:2]
    back = h.get_points()
    assert [p.x[0] for p in back] == [2.0, 3.0, 4.0, 9.0]


class _Flood(Heuristic):
    """Emits far more points than any budget on start."""

    def on_start(self):
        self.emit([np.zeros(self.problem.dim)] * 500)


class _Failing(Rosenbrock):
    calls = 0

    def eval(self, x):
        type(self).calls += 1
        raise RuntimeError("objective failed")


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_failing_evaluations_are_charged_and_the_run_ends(sync):
    from panobbgo.strategies import StrategyRoundRobin

    _Failing.calls = 0
    s = StrategyRoundRobin(_Failing(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 23
    s.config.sync_evaluation = sync
    s.config.stop_on_convergence = False
    s.add(_Flood)
    s.start()
    assert len(s.results) == 0
    assert _Failing.calls == 23


def test_cleanup_cancels_queued_evaluations():
    import threading
    import time

    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.stop_on_convergence = False
    s.add(_Flood)
    s.initialize()
    gate = threading.Event()
    ran = []

    def slow(i):
        gate.wait(5.0)
        ran.append(i)

    s._futures = {i: s._thread_pool.submit(slow, i) for i in range(50)}
    time.sleep(0.05)
    threading.Timer(0.3, gate.set).start()  # release the running ones mid-cleanup
    s._cleanup()
    n = len(ran)
    time.sleep(0.2)
    # Cleanup waited for the evaluations already running, cancelled the
    # queued ones, and nothing runs after it returned.
    assert 0 < n == len(ran) < 50


def test_request_stop_ends_the_main_loop():
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 5000
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add(Random)
    s.request_stop()
    s.start()
    assert len(s.results) < 5000
