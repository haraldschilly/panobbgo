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

"""One ordering per constraint handler, shared by every consumer that ranks.

Regression: ``DefaultConstraintHandler.is_better`` ranks by ``cv`` first, but
the rankers (``Archive``, ``Restart``, the ``Splitter`` value rule,
``RegionUCB``) used the inherited ``fx + rho * cv`` — so an infeasible
``fx = -1000, cv = 1`` outranked a feasible ``fx = 0`` everywhere except in
``Best``.
"""

from unittest import mock

import numpy as np
import pytest

from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import (
    AugmentedLagrangianConstraintHandler,
    DefaultConstraintHandler,
    DynamicPenaltyConstraintHandler,
    EpsilonConstraintHandler,
    FilterConstraintHandler,
    PenaltyConstraintHandler,
    rank_positions,
)


def _r(fx, cv, x=(0.0, 0.0)):
    return Result(Point(np.asarray(x, dtype=float), "t"), fx, cv_vec=np.array([cv]))


INFEASIBLE = (-1000.0, 1.0)
FEASIBLE = (0.0, 0.0)


def _strategy_with(handler_cls, **kw):
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.constraint_handler = handler_cls(strategy=s, **kw)
    return s


HANDLERS = [
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    DynamicPenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    EpsilonConstraintHandler,
    FilterConstraintHandler,
]


@pytest.mark.parametrize("cls", HANDLERS)
def test_improvement_positive_iff_better(cls):
    """``calculate_improvement`` never contradicts ``is_better`` (Filter's
    reward is the documented exception: it rewards any non-dominated point)."""
    strategy = mock.MagicMock()
    strategy.results = []
    h = cls(strategy=strategy)
    rng = np.random.default_rng(0)
    pts = [_r(float(f), float(c)) for f, c in zip(rng.normal(size=40), np.maximum(rng.normal(size=40), 0.0))]
    for a in pts:
        for b in pts:
            better = h.is_better(a, b)
            assert better == (h.rank_key(b) < h.rank_key(a))
            if cls is not FilterConstraintHandler:
                assert (h.calculate_improvement(a, b) > 0.0) == better


def test_default_rankers_agree_with_is_better():
    from panobbgo.analyzers import Archive

    s = _strategy_with(DefaultConstraintHandler)
    h = s.constraint_handler
    bad, good = _r(*INFEASIBLE, x=(1.0, 1.0)), _r(*FEASIBLE, x=(0.5, 0.5))
    assert h.is_better(bad, good)

    # Archive
    a = Archive(s, k=4)
    a.on_new_results([bad, good])
    assert a.top_k(1) == [good] and a.top_k(1)[0] is good
    assert a.results[0] is good

    # generic rank helper (Splitter value rule, RegionUCB)
    assert list(rank_positions(h, [bad, good])) == [1.0, 0.0]


def test_default_restart_counts_feasibility_as_improvement():
    from panobbgo.analyzers.restart import Restart

    s = _strategy_with(DefaultConstraintHandler)
    r = Restart(s, patience=2, max_restarts=5)
    r.__start__()
    r.on_new_results([_r(*INFEASIBLE)])
    r.on_new_results([_r(*FEASIBLE)])  # an improvement: feasible beats infeasible
    assert r._evals_since_improvement == 0


def test_splitter_value_rule_ranks_by_handler_order():
    from panobbgo.analyzers import Splitter

    s = _strategy_with(DefaultConstraintHandler)
    sp = Splitter(s, split_rule="value")
    sp.__start__()
    sp.root.results = [_r(*INFEASIBLE), _r(*FEASIBLE), _r(5.0, 0.0)]
    assert list(sp.root._penalties()) == [2.0, 0.0, 1.0]


def test_archive_reranks_under_a_time_varying_handler():
    """Dynamic penalty: rho grows with len(results), so the heap order changes."""
    from panobbgo.analyzers import Archive

    s = _strategy_with(DynamicPenaltyConstraintHandler, rho_start=1.0, rate=1.0, exponent=1.0)
    h = s.constraint_handler
    assert not h.time_invariant
    lo_fx_infeasible = _r(-10.0, 1.0)  # P = -10 + rho
    feasible = _r(0.0, 0.0)  # P = 0
    a = Archive(s, k=4)
    with mock.patch.object(type(h), "_get_current_rho", return_value=1.0):
        a.on_new_results([lo_fx_infeasible, feasible])
        assert a.results[0] is lo_fx_infeasible
    with mock.patch.object(type(h), "_get_current_rho", return_value=100.0):
        assert a.results[0] is feasible
        assert a.top_k(1)[0] is feasible
