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

Results that arrive one at a time used to close a generation at
``min_results_fraction · λ`` (μ) arrivals: CMA-ES ranked only the first μ,
dropped the other λ − μ evaluated offspring, and active CMA's negative
weights had no rank to act on.  Now a generation may close early only once
all its offspring are dispatched (``quorum="dispatched"``), and the late
offspring of an early-closed generation join the next update
(``late_results="fold"``).
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
    def _cma(self, **kw) -> CMAES:
        from panobbgo.lib.constraints import DefaultConstraintHandler

        # A fresh stand-in per instance: the same RNG stream, so two instances sample the same offspring.
        self.strategy = self.init_strategy()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        cma = CMAES(self.strategy, popsize=8, **kw)
        cma.on_start()
        return cma

    def test_queued_offspring_hold_the_generation_open(self):
        cma = self._cma()
        first = cma.get_points(4)  # four dispatched, four still queued
        cma.on_new_results(_sphere(first))
        assert cma._counteval == 0  # μ results, but the generation is not all out
        rest = cma.get_points(100)
        assert len(rest) == 4
        cma.on_new_results(_sphere(rest[:1]))  # all dispatched and five in: the fraction applies
        assert cma._counteval == 8

    def test_one_at_a_time_ranks_the_whole_generation(self):
        """One worker: every result but the last arrives with offspring still queued."""
        cma = self._cma()
        seen = []
        update = cma._update

        def spy(collected, n_offspring):
            seen.append(len(collected))
            return update(collected, n_offspring)

        cma._update = spy
        for _ in range(8):
            cma.on_new_results(_sphere(cma.get_points(1)))
        assert seen == [8]
        assert cma.n_late_dropped == cma.n_late_folded == 0

    def test_a_dispatched_generation_closes_at_the_fraction(self):
        cma = self._cma()
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        assert cma._counteval == 8
        assert len(cma.get_points(100)) == 8

    def test_fraction_rule_closes_with_offspring_still_queued(self):
        """``quorum="fraction"``: the rule before §64."""
        cma = self._cma(quorum="fraction")
        cma.on_new_results(_sphere(cma.get_points(4)))
        assert cma._counteval == 8

    def test_rejects_unknown_quorum(self):
        with pytest.raises(ValueError, match="quorum"):
            CMAES(self.init_strategy(), quorum="all")

    def test_late_results_are_dropped_under_drop(self):
        cma = self._cma(late_results="drop")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_new_results(_sphere(points[4:]))
        assert cma.n_late_dropped == 4
        assert cma.n_late_folded == 0
        assert cma._late == {}

    def test_late_results_are_folded_into_the_next_update(self):
        cma = self._cma(late_results="fold")
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
        cma = self._cma(late_results="fold")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_failed_evaluations(points[4:5])
        assert cma.n_late_dropped == 1
        assert cma.n_late_folded == 0

    def test_a_flush_clears_the_folded_results(self):
        cma = self._cma(late_results="fold")
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:4]))
        cma.on_new_results(_sphere(points[4:]))
        assert cma._late
        cma._drop_generation()
        assert cma._late == {}

    def test_rejects_unknown_late_results(self):
        with pytest.raises(ValueError, match="late_results"):
            CMAES(self.init_strategy(), late_results="keep")

    def _cov_after_one_generation(self, active: bool, arrive: int, **kw) -> np.ndarray:
        # Unguarded: one of the first generation's offspring is projected onto
        # Rosenbrock's box, and the guard would skip the negative update.
        cma = self._cma(active=active, active_skip_repaired=False, **kw)
        points = cma.get_points(100)
        cma.on_new_results(_sphere(points[:arrive]))
        assert cma._counteval == 8
        assert cma._C is not None
        return cma._C.copy()

    def test_active_weights_apply_to_a_whole_generation(self):
        """With the whole generation ranked, active CMA changes the covariance update."""
        c_active = self._cov_after_one_generation(True, 8)
        c_positive = self._cov_after_one_generation(False, 8)
        assert not np.allclose(c_active, c_positive)

    def test_active_is_a_no_op_on_a_quorum_of_mu(self):
        """The §64 finding: a generation closed at μ results has no rank for a negative weight."""
        c_active = self._cov_after_one_generation(True, 4)
        c_positive = self._cov_after_one_generation(False, 4)
        np.testing.assert_array_equal(c_active, c_positive)

    def test_active_weights_apply_to_a_folded_generation(self):
        """With ``late_results="fold"`` a generation closed at μ ranks more than μ entries again."""

        def cov(active: bool) -> np.ndarray:
            cma = self._cma(late_results="fold", active=active, active_skip_repaired=False)
            gen1 = cma.get_points(100)
            cma.on_new_results(_sphere(gen1[:4]))
            gen2 = cma.get_points(100)
            cma.on_new_results(_sphere(gen1[4:]))
            cma.on_new_results(_sphere(gen2[:4]))
            assert cma._counteval == 16
            assert cma._C is not None
            return cma._C.copy()

        assert not np.allclose(cov(True), cov(False))

    def _close_first_generation(self, **kw):
        """Generation 1 closes at μ; returns ``(cma, late points, generation 2)``."""
        cma = self._cma(**kw)
        gen1 = cma.get_points(100)
        cma.on_new_results(_sphere(gen1[:4]))
        gen2 = cma.get_points(100)
        return cma, gen1[4:], gen2

    def test_capped_fold_keeps_the_best_quarter(self):
        cma, late, _ = self._close_first_generation(late_results="fold_capped")
        cma.on_new_results([Result(p, float(fx)) for p, fx in zip(late, (3.0, 1.0, 4.0, 2.0))])
        (bucket,) = cma._late.values()
        assert sorted(e["penalty"] for e in bucket) == [1.0, 2.0]  # max(1, λ // 4) = 2
        assert (cma.n_late_folded, cma.n_late_dropped) == (2, 2)

    def test_a_far_late_point_is_clipped_to_c_y(self):
        cma, late, _ = self._close_first_generation()
        assert cma._m is not None and cma._B is not None and cma._D is not None
        info = cma._pending[late[0].who]
        info["x_eval"] = cma._m + 50.0 * cma._sigma  # evaluated far from the current mean
        cma.on_new_results(_sphere(late[:1]))
        (entry,) = next(iter(cma._late.values()))
        n = cma.problem.dim
        c_y = np.sqrt(n) + 2.0 * n / (n + 2.0)
        mahal = np.linalg.norm((1.0 / cma._D) * (cma._B.T @ entry["y"]))
        assert mahal == pytest.approx(c_y)
        np.testing.assert_allclose(entry["x"], cma._m + cma._sigma * entry["y"])
        np.testing.assert_array_equal(entry["x_eval"], info["x_eval"])

    def test_a_repaired_late_offspring_folds_its_evaluated_point(self):
        cma, late, _ = self._close_first_generation()
        info = cma._pending[late[0].who]
        info["x"] = info["x_eval"] + 0.25  # the repaired (clipped) update position differs
        x_eval = np.array(info["x_eval"], dtype=float)
        cma.on_new_results(_sphere(late[:1]))
        (entry,) = next(iter(cma._late.values()))
        np.testing.assert_array_equal(entry["x_eval"], x_eval)
        n = cma.problem.dim
        c_y = np.sqrt(n) + 2.0 * n / (n + 2.0)
        assert cma._B is not None
        assert cma._D is not None
        expected = cma._clip_injected((x_eval - cma._m) / cma._sigma, cma._B, cma._D, c_y)
        np.testing.assert_allclose(entry["y"], expected)

    def _cov_with_worst_ranked_late(self, active: bool, guard: bool) -> np.ndarray:
        """Generation 2's update with its own μ best and the four late points at ranks μ … λ."""
        cma, late, gen2 = self._close_first_generation(active=active, active_skip_repaired=guard)
        for p in gen2:
            cma._pending[p.who]["repaired"] = False  # keep the repair guard out of the picture
        cma.on_new_results([Result(p, 1e6 + i) for i, p in enumerate(late)])  # ranked last
        cma.on_new_results(_sphere(gen2[:4]))
        assert cma._counteval == 16
        assert cma._C is not None
        return cma._C.copy()

    def test_late_points_get_no_negative_weight_under_the_guard(self):
        positive = self._cov_with_worst_ranked_late(active=False, guard=True)
        np.testing.assert_array_equal(self._cov_with_worst_ranked_late(active=True, guard=True), positive)
        # without the guard the same late points do get negative weights
        assert not np.allclose(self._cov_with_worst_ranked_late(active=True, guard=False), positive)


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
    """Virtual q = 1 samples exactly what the batch-synchronous run samples.

    Results arrive one at a time, but while a generation has queued offspring
    it cannot close, so every update ranks the whole generation, as the
    synchronous batches do (λ = 7 fits RoundRobin's 10-point request).
    """
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
