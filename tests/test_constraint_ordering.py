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

from tests.support import attach_spawn_rng
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
    attach_spawn_rng(strategy)
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


def _dynamic_archive(k):
    from panobbgo.analyzers import Archive
    from panobbgo.lib.classic import Rosenbrock

    strategy = mock.MagicMock()
    attach_spawn_rng(strategy)
    strategy.problem = Rosenbrock(dim=2)
    strategy.results = []
    strategy.constraint_handler = DynamicPenaltyConstraintHandler(strategy, rho_start=1.0, rate=1.0, exponent=1.0)
    return strategy, Archive(strategy, k=k)


def test_archive_reranks_under_a_time_varying_handler():
    """Dynamic penalty: rho grows with len(results), so the heap order changes."""
    strategy, a = _dynamic_archive(4)
    assert not strategy.constraint_handler.time_invariant
    lo_fx_infeasible = _r(-10.0, 1.0)  # P = -10 + rho
    feasible = _r(0.0, 0.0)  # P = 0
    a.on_new_results([lo_fx_infeasible, feasible])  # rho = 1
    assert a.results[0] is lo_fx_infeasible
    strategy.results = list(range(99))  # rho = 100
    assert a.results[0] is feasible
    assert a.top_k(1)[0] is feasible


def test_archive_rerank_is_cached_per_result_count():
    strategy, a = _dynamic_archive(4)
    a.on_new_results([_r(1.0, 0.0), _r(2.0, 0.0)])
    a.top_k(1)  # re-ranks once for the current clock
    with mock.patch.object(type(a), "_key", wraps=a._key) as key:
        a.top_k(1)
        a.top_k(1)
        assert key.call_count == 0  # nothing moved since the batch's re-rank
        strategy.results = [0]
        a.top_k(1)
        assert key.call_count == 2


def test_archive_queries_do_not_lose_concurrent_admissions():
    """Regression: a query re-ranked by rebinding the heap while the bus thread
    pushed onto the old one, so admissions were lost."""
    import threading

    strategy, a = _dynamic_archive(100000)
    n = 400
    done = threading.Event()

    def bus():
        for i in range(n):
            strategy.results.append(i)  # the clock moves: every query re-ranks
            a.on_new_results([_r(float(i), 0.0)])
        done.set()

    t = threading.Thread(target=bus)
    t.start()
    while not done.is_set():
        a.top_k(3)
    t.join()
    assert len(a) == n


@pytest.mark.parametrize("cls", HANDLERS)
def test_missing_objective_ranks_last(cls):
    """``result_key`` of fx=None is padded to the key length: it used to be
    ``(inf,)``, which sorts before ``(inf, x)``."""
    from panobbgo.lib.constraints import result_key

    strategy = mock.MagicMock()
    attach_spawn_rng(strategy)
    strategy.results = []
    h = cls(strategy=strategy)
    missing = Result(Point(np.zeros(2), "t"), None)
    worst = _r(float("inf"), float("inf"))
    assert len(result_key(h, missing)) == len(h.rank_key(_r(0.0, 0.0))) == h.rank_key_size
    assert not result_key(h, missing) < result_key(h, worst)


def test_epsilon_nan_violation_is_not_feasible():
    strategy = mock.MagicMock()
    attach_spawn_rng(strategy)
    strategy.results = []
    h = EpsilonConstraintHandler(strategy=strategy, epsilon_start=0.0)
    # ``Result.cv`` maps a NaN entry of ``cv_vec`` to ``inf``; a NaN ``cv``
    # still reaches the handler from results that compute it otherwise.
    nan_cv = mock.MagicMock(fx=-100.0, cv=float("nan"))
    assert h._phi(nan_cv) == float("inf")
    assert h.is_better(nan_cv, _r(5.0, 0.0))


def test_result_cv_nan_entry_is_infeasible():
    """A NaN entry of ``cv_vec`` is an unknown violation: ``cv`` is ``inf``.

    ``cv_vec[cv_vec > 0.0]`` used to drop it, so a NaN constraint counted as
    satisfied and the result as feasible."""
    assert Result(Point(np.zeros(2), "t"), 1.0, cv_vec=np.array([np.nan, -1.0])).cv == float("inf")
    assert Result(Point(np.zeros(2), "t"), 1.0, cv_vec=np.array([np.nan, 2.0])).cv == float("inf")
    # no NaN: unchanged (norm of the positive entries); unconstrained: 0.0
    assert Result(Point(np.zeros(2), "t"), 1.0, cv_vec=np.array([3.0, -1.0, 4.0])).cv == 5.0
    assert Result(Point(np.zeros(2), "t"), 1.0).cv == 0.0
    h = DefaultConstraintHandler(strategy=mock.MagicMock(results=[]))
    feasible = _r(100.0, 0.0)
    nan_cv = Result(Point(np.zeros(2), "t"), -100.0, cv_vec=np.array([np.nan]))
    assert h.is_better(nan_cv, feasible)
    assert not h.is_better(feasible, nan_cv)


def test_alm_nan_constraint_is_infeasible():
    """ALM applies the same policy: a NaN constraint value makes the
    Lagrangian ``inf`` (it used to be ``nan_to_num``-ed to 0.0 = satisfied)."""
    strategy = mock.MagicMock()
    attach_spawn_rng(strategy)
    strategy.results = []
    h = AugmentedLagrangianConstraintHandler(strategy=strategy)
    h.lambdas = np.zeros(2)
    nan_r = Result(Point(np.zeros(2), "t"), -100.0, cv_vec=np.array([np.nan, -1.0]))
    ok_r = Result(Point(np.ones(2), "u"), 5.0, cv_vec=np.array([-1.0, -1.0]))
    assert h.get_penalty_value(nan_r) == float("inf")
    assert h.get_penalty_value(ok_r) == 5.0

    # The history scan must not crown the NaN row as the new best.
    strategy.results = [nan_r, ok_r]
    strategy.best = ok_r
    published = []
    strategy.eventbus.publish.side_effect = lambda *a, **kw: published.append(kw)
    h._scan_history_for_new_best()
    assert published, "scan did not publish a candidate"
    assert published[-1]["candidates"][0].fx == 5.0


def test_default_reward_is_finite_for_unknown_violation():
    """``Result.cv`` is ``inf`` for a NaN constraint; the default handler's reward
    stays finite and positive (``calculate_improvement`` maps an infinite
    magnitude to 1.0) — pinned by the cv audit of 2026-09."""
    h = DefaultConstraintHandler(strategy=mock.MagicMock(results=[]))
    unknown = Result(Point(np.zeros(2), "t"), 1.0, cv_vec=np.array([np.nan]))
    feasible = _r(5.0, 0.0)
    infeasible = _r(5.0, 2.0)
    assert h.is_better(unknown, feasible) and h.is_better(unknown, infeasible)
    for new in (feasible, infeasible):
        imp = h.calculate_improvement(unknown, new)
        assert np.isfinite(imp) and imp > 0


def test_time_invariance_flags():
    assert DefaultConstraintHandler.time_invariant
    assert PenaltyConstraintHandler.time_invariant
    assert FilterConstraintHandler.time_invariant  # static ordering; only the reward uses the filter
    assert not DynamicPenaltyConstraintHandler.time_invariant
    assert not EpsilonConstraintHandler.time_invariant
    assert not AugmentedLagrangianConstraintHandler.time_invariant
