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

import pytest

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


def test_clamp_counts_pending_evaluations():
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 10
    s.pending = {"a": None, "b": None, "c": None}
    assert len(s._clamp_to_budget(list(range(20)))) == 7
    s.pending = {i: None for i in range(10)}
    assert s._clamp_to_budget(list(range(5))) == []
    s.pending = {}
    assert s._clamp_to_budget([1, 2]) == [1, 2]


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
