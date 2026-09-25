# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

"""Tagged-trial arms keep working when some of their evaluations fail.

A failed evaluation returns no result; before the ``failed_evaluations``
handlers, the trial stayed pending forever and froze its population slot
(LSHADE stopped after 6 of 400 evaluations, PSO froze every particle).
"""

import pytest

from panobbgo.heuristics import JSO, LSHADE, PSO
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyRoundRobin


class _FailsRight(Rosenbrock):
    def eval(self, x):
        if x[0] > 0.6:
            raise ValueError("objective failed")
        return super().eval(x)


@pytest.mark.parametrize("arm", [LSHADE, JSO, PSO])
def test_failed_trials_do_not_freeze_the_population(arm):
    s = StrategyRoundRobin(_FailsRight(3), parse_args=False, testing_mode=True, seed=1)
    s.config.max_eval = 300
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add(arm)
    s.start()
    assert s._dispatched == 300
    assert len(s.results) > 150  # most of the budget produced results
