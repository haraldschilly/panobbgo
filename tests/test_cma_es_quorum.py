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
"""CMA-ES's update quorum under asynchronous evaluation (DISCOVERY §64).

Results that arrive one at a time close a generation at ``min_results_fraction
· λ`` (μ).  With one worker that bought nothing and dropped λ − μ evaluated
offspring per generation — and with them the μ-of-λ selection and every rank
active CMA gives a negative weight.  With one worker the quorum is now the
whole generation; with more, ``late_results="fold"`` ranks the late
offspring with the next generation instead of dropping them.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.heuristics import CMAES
from panobbgo.lib import Result
from panobbgo.lib.classic import DeJong
from panobbgo.strategies import StrategyRoundRobin
from panobbgo.virtual_clock import VirtualSpec
from tests.support import PanobbgoTestCase


def _sphere(points):
    return [Result(p, float(np.sum(p.x**2))) for p in points]


class TestQuorum(PanobbgoTestCase):
    def _cma(self, workers: int, **kw) -> CMAES:
        from panobbgo.lib.constraints import DefaultConstraintHandler

        # A fresh stand-in per instance: the same RNG stream, so two instances sample the same offspring.
        self.strategy = self.init_strategy()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        self.strategy._n_evaluators = lambda: workers
        cma = CMAES(self.strategy, popsize=8, **kw)
        cma.on_start()
        return cma

    def test_one_worker_waits_for_the_whole_generation(self):
        cma = self._cma(1)
        points = cma.get_points(100)
        assert len(points) == 8
        cma.on_new_results(_sphere(points[:4]))
        assert cma.get_points(100) == []  # μ results: no update yet
        assert cma._counteval == 0
        cma.on_new_results(_sphere(points[4:]))
        assert cma._counteval == 8
        assert len(cma.get_points(100)) == 8

    def test_parallel_workers_keep_the_fractional_quorum(self):
        cma = self._cma(4)
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        assert cma._counteval == 8
        assert len(cma.get_points(100)) == 8

    def test_late_results_are_dropped_under_drop(self):
        cma = self._cma(4, late_results="drop")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_new_results(_sphere(points[4:]))
        assert cma.n_late_dropped == 4
        assert cma.n_late_folded == 0
        assert cma._late == {}

    def test_late_results_are_folded_into_the_next_update(self):
        cma = self._cma(4, late_results="fold")
        gen1 = cma.get_points(100)
        cma.on_new_results(_sphere(gen1[:4]))  # quorum: generation 1 closes, generation 2 goes out
        gen2 = cma.get_points(100)
        cma.on_new_results(_sphere(gen1[4:]))  # late
        assert cma.n_late_folded == 4
        assert cma.n_late_dropped == 0
        assert [len(v) for v in cma._late.values()] == [4]
        late = next(iter(cma._late.values()))
        assert all(e["injected"] and e["late"] for e in late)
        # not charged again, and they do not count toward generation 2's quorum
        assert cma._counteval == 8
        cma.on_new_results(_sphere(gen2[:3]))
        assert cma._counteval == 8

        seen = []
        update = cma._update

        def spy(collected, n_offspring):
            seen.append(list(collected))
            return update(collected, n_offspring)

        cma._update = spy
        cma.on_new_results(_sphere(gen2[3:4]))  # the fourth own result: quorum
        assert len(seen) == 1
        assert len(seen[0]) == 8  # 4 own + 4 folded
        assert sum(bool(e.get("late")) for e in seen[0]) == 4
        assert cma._late == {}
        assert cma._counteval == 16

    def test_a_late_failure_is_dropped_even_under_fold(self):
        cma = self._cma(4, late_results="fold")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_failed_evaluations(points[4:5])
        assert cma.n_late_dropped == 1
        assert cma.n_late_folded == 0

    def test_a_flush_clears_the_folded_results(self):
        cma = self._cma(4, late_results="fold")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_new_results(_sphere(points[4:]))
        assert cma._late
        cma._drop_generation()
        assert cma._late == {}

    def test_rejects_unknown_late_results(self):
        with pytest.raises(ValueError, match="late_results"):
            CMAES(self.init_strategy(), late_results="keep")

    def _cov_after_one_generation(self, workers: int, active: bool, arrive: int, **kw) -> np.ndarray:
        # Unguarded: one of the first generation's offspring is projected onto
        # Rosenbrock's box, and the guard would skip the negative update.
        cma = self._cma(workers, active=active, active_skip_repaired=False, **kw)
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:arrive]))
        assert cma._counteval == 8
        assert cma._C is not None
        return cma._C.copy()

    def test_active_weights_apply_with_one_worker(self):
        """With the whole generation ranked, active CMA changes the covariance update."""
        c_active = self._cov_after_one_generation(1, True, 8)
        c_positive = self._cov_after_one_generation(1, False, 8)
        assert not np.allclose(c_active, c_positive)

    def test_active_is_a_no_op_on_a_quorum_of_mu(self):
        """The §64 finding: a generation closed at μ results has no rank for a negative weight."""
        c_active = self._cov_after_one_generation(4, True, 4)
        c_positive = self._cov_after_one_generation(4, False, 4)
        np.testing.assert_array_equal(c_active, c_positive)

    def test_active_weights_apply_to_a_folded_generation(self):
        """With ``late_results="fold"`` a generation closed at μ ranks more than μ entries again."""

        def cov(active: bool) -> np.ndarray:
            cma = self._cma(4, late_results="fold", active=active, active_skip_repaired=False)
            gen1 = cma.get_points(100)
            cma.on_new_results(_sphere(gen1[:4]))
            gen2 = cma.get_points(100)
            cma.on_new_results(_sphere(gen1[4:]))
            cma.on_new_results(_sphere(gen2[:4]))
            assert cma._counteval == 16
            assert cma._C is not None
            return cma._C.copy()

        assert not np.allclose(cov(True), cov(False))


BOX = [(-2.0, 8.0)] * 3


def _run(virtual=None, max_eval=400, **kw):
    s = StrategyRoundRobin(
        DeJong(3, box=list(BOX)), max_evaluations=max_eval, seed=42, parse_args=False, testing_mode=True
    )
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    if virtual is not None:
        virtual.apply(s)
    h = CMAES(s, **kw)
    s.add_heuristic(h)
    s.start()
    df = s.results.results
    assert df is not None
    return h, np.stack(list(df["x"].to_numpy())).astype(float)


def test_one_async_worker_is_the_synchronous_trajectory():
    """Virtual q = 1, pulled one result at a time, samples exactly what the batch-synchronous run samples."""
    _, x_sync = _run()
    h, x_q1 = _run(VirtualSpec(workers=1, duration="lognormal"))
    np.testing.assert_array_equal(x_sync, x_q1)
    assert h.n_late_dropped == 0


@pytest.mark.parametrize("late", ["drop", "fold"])
def test_late_results_under_the_async_virtual_clock(late):
    h, x = _run(VirtualSpec(workers=4, duration="lognormal"), late_results=late)
    assert len(x) == 400
    if late == "drop":
        assert h.n_late_dropped > 100 and h.n_late_folded == 0
    else:
        assert h.n_late_folded > 100 and h.n_late_dropped < 10
