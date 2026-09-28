# -*- coding: utf8 -*-
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

"""The shared failure model (:mod:`panobbgo.analyzers.failure_model`) and its integration (DISCOVERY §71)."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.analyzers.failure_model import FailureModel
from panobbgo.lib import Point, Result
from tests.support import PanobbgoTestCase


def _box(model):
    return np.array(model.problem.box[:, :], dtype=float)


def _uniform(model, n, seed=1):
    box = _box(model)
    return np.random.default_rng(seed).uniform(box[:, 0], box[:, 1], size=(n, box.shape[0]))


class FailureModelTests(PanobbgoTestCase):
    """The model on its own, fed by hand (``Rosenbrock(2)``, box ``[-5, 5]^2`` or similar)."""

    def test_validation(self):
        kws: list[dict] = [{"threshold": 0.0}, {"prior": 0.0}, {"max_share": 0.0}, {"max_rejects": 0}]
        for kw in kws:
            with pytest.raises(ValueError):
                FailureModel(self.strategy, **kw)

    def test_no_failures_is_all_clear_and_does_no_work(self):
        m = FailureModel(self.strategy, filter=True)
        for x in _uniform(m, 50):
            m.add_success(x)
        X = _uniform(m, 20, seed=2)
        assert np.all(m.p_fail(X) == 0.0)
        assert not m.armed
        assert not np.any(m.in_poison(X))
        pts = [Point(x, "A:1") for x in X]
        assert m.reject(pts) == [False] * len(pts)
        assert m.n_rejected == 0

    def test_empty_model(self):
        m = FailureModel(self.strategy)
        x = _uniform(m, 1)[0]
        assert m.p_fail(x) == 0.0 and m.in_poison(x) is False
        assert m.n_fail == 0 and m.n_ok == 0

    def test_few_points_one_failure_is_not_a_zone_but_its_repeat_is(self):
        m = FailureModel(self.strategy)
        box = _box(m)
        centre = box.mean(axis=1)
        m.add_failure(centre)
        assert m.p_fail(centre) == 1.0  # a deterministic failure repeats
        near = centre + 1e-3 * (box[:, 1] - box[:, 0])
        assert 0.0 < m.p_fail(near) <= 1.0 / (1.0 + m.prior) + 1e-12
        assert not m.in_poison(near)
        far = box[:, 0] + 0.01 * (box[:, 1] - box[:, 0])
        assert m.p_fail(far) < m.threshold and not m.in_poison(far)  # unexplored is not poisoned

    def test_a_dense_cluster_of_failures_is_a_zone(self):
        m = FailureModel(self.strategy)
        box = _box(m)
        w = box[:, 1] - box[:, 0]
        c = box[:, 0] + 0.7 * w
        rng = np.random.default_rng(3)
        for _ in range(6):
            m.add_failure(c + 0.02 * w * rng.standard_normal(2))
        for x in _uniform(m, 30):
            if np.max(np.abs(x - c) / w) > 0.2:
                m.add_success(x)
        assert m.in_poison(c + 0.005 * w)
        assert m.armed
        assert not m.in_poison(box[:, 0] + 0.1 * w)

    def test_all_fail_never_excludes_the_whole_box(self):
        m = FailureModel(self.strategy, filter=True)
        X = _uniform(m, 300)
        for x in X:
            m.add_failure(x)
        assert m.poisoned_share() > m.max_share
        assert not m.armed
        assert not np.any(m.in_poison(_uniform(m, 50, seed=5)))
        assert m.reject([Point(x, "A:1") for x in _uniform(m, 5, seed=6)]) == [False] * 5

    def test_half_plane_recovery(self):
        """Uniform samples labelled by a half-plane: held-out points away from the boundary are classified right."""
        m = FailureModel(self.strategy)
        box = _box(m)
        w = box[:, 1] - box[:, 0]
        t = box[0, 0] + 0.7 * w[0]  # failing side: x_0 > t (30 % of the box)
        for x in _uniform(m, 300):
            (m.add_failure if x[0] > t else m.add_success)(x)
        assert m.armed
        test = _uniform(m, 2000, seed=9)
        margin = np.abs(test[:, 0] - t) / w[0]
        test = test[margin > 0.05]
        truth = test[:, 0] > t
        pred = m.in_poison(test)
        acc = float(np.mean(pred == truth))
        assert acc > 0.95, acc
        # conservative: almost no false alarm on the good side
        assert float(np.mean(pred[~truth])) < 0.02

    def test_labels_from_results_and_crashes(self):
        m = FailureModel(self.strategy)
        X = _uniform(m, 4)
        ok = Result(Point(X[0], "A"), 1.0)
        nan = Result(Point(X[1], "A"), float("nan"))
        tmo = Result(Point(X[2], "A"), float("nan"), timed_out=True)
        m.on_new_results([ok, nan, tmo])
        m.on_failed_evaluations([Point(X[3], "A")])
        assert m.n_ok == 1 and m.n_fail == 3
        assert m.kinds == {"nan": 1, "timeout": 1, "crash": 1}
        for x in X[1:]:
            assert m.p_fail(x) == 1.0

    def test_reject_streak_lets_a_candidate_through(self):
        m = FailureModel(self.strategy, filter=True, max_rejects=3)
        box = _box(m)
        w = box[:, 1] - box[:, 0]
        bad = box[:, 0] + 0.8 * w
        for _ in range(5):
            m.add_failure(bad)
        for x in _uniform(m, 40):
            if np.max(np.abs(x - bad) / w) > 0.3:
                m.add_success(x)
        assert m.in_poison(bad)
        flags = [m.reject([Point(bad, "A:%d" % i)])[0] for i in range(8)]
        assert flags == [True, True, True, False, True, True, True, False]
        # streaks are per heuristic: B starts fresh
        assert m.reject([Point(bad, "B:0")]) == [True]
        assert m.n_rejected == 7 and m.n_passed_streak == 2

    def test_vectorised_and_scalar_queries_agree(self):
        m = FailureModel(self.strategy)
        X = _uniform(m, 60)
        for i, x in enumerate(X):
            (m.add_failure if i % 3 == 0 else m.add_success)(x)
        Q = _uniform(m, 25, seed=4)
        p = m.p_fail(Q)
        assert p.shape == (25,)
        for q, pv in zip(Q, p):
            assert m.p_fail(q) == pytest.approx(pv)
        assert np.all((p >= 0.0) & (p <= 1.0))


# ---------------------------------------------------------------------------
# Integration: the proposal filter in the main loop
# ---------------------------------------------------------------------------


def _spec(name, heur, fm_kwargs=None, heur_kwargs=None):
    from panobbgo.benchmark import StrategySpec
    from panobbgo.strategies import StrategyRoundRobin

    return StrategySpec(
        name=name,
        strategy_class=StrategyRoundRobin,
        heuristics=[(heur, dict(heur_kwargs or {}))],
        analyzers=[(FailureModel, dict(fm_kwargs))] if fm_kwargs is not None else [],
        seed_name="same",
    )


def _run(spec, problem, budget, q=4, before_start=None):
    from panobbgo.virtual_clock import VirtualSpec

    strategy = spec.create_strategy(problem, seed=7, max_eval=budget)
    strategy.config.sync_evaluation = True
    if q:
        VirtualSpec(workers=q, duration="lognormal", sigma=0.5, policy="async").apply(strategy)
    if before_start is not None:
        before_start(strategy)
    strategy.start()
    return strategy


def _xs(strategy):
    return np.asarray(strategy.results.results["x"].to_numpy(), dtype=float)


@pytest.mark.parametrize("heur_name", ["CMAES", "Random"])
def test_filter_is_bit_identical_without_failures(heur_name):
    """No failure, empty model: the run is the same point for point, and no event is published."""
    from panobbgo import heuristics
    from panobbgo.lib.families import Family

    heur = getattr(heuristics, heur_name)
    p = Family("ellipsoid", dim=2, seed=5)
    a = _run(_spec("a", heur), p, 80)
    b = _run(_spec("b", heur, {"filter": True}), p, 80)
    assert len(a.results) == len(b.results) == 80
    np.testing.assert_array_equal(_xs(a), _xs(b))
    assert getattr(b, "n_predicted_failures", 0) == 0
    assert b.failure_model is not None and b.failure_model.n_fail == 0


def test_filter_rejects_in_a_crash_region_and_spends_the_budget():
    from panobbgo.heuristics import Random
    from panobbgo.lib.families import Family, FailureRegion

    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("halfspace", share=0.4, mode="crash"))
    base = _run(_spec("a", Random), p, 150)
    filt = _run(_spec("b", Random, {"filter": True}), p, 150)
    fm = filt.failure_model
    assert filt.n_predicted_failures > 0 and fm.n_rejected == filt.n_predicted_failures
    # the budget is spent in full either way; rejected candidates cost nothing
    assert base._dispatched == filt._dispatched == 150
    n_fail_base = base.n_finished - len(base.results)
    n_fail_filt = filt.n_finished - len(filt.results)
    assert n_fail_filt < n_fail_base
    # the model learns only from evaluated points: its failures are the real ones
    assert fm.n_fail == n_fail_filt


def test_rejected_points_reach_the_heuristic_as_failures():
    """CMA-ES under the filter: every rejected offspring is answered, no generation waits forever."""
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.families import Family, FailureRegion

    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("halfspace", share=0.45, mode="crash", boundary_gap=0.0))
    rejected = []

    def spy(strategy):
        orig = strategy._filter_poisoned

        def wrapped(model, points):
            kept, n = orig(model, points)
            rejected.extend(p.who for p in points if p not in kept)
            return kept, n

        strategy._filter_poisoned = wrapped

    s = _run(_spec("c", CMAES, {"filter": True}, {"failure_aware": True}), p, 200, before_start=spy)
    assert s._dispatched == 200
    assert len(rejected) == s.n_predicted_failures > 0
    cma = s.heuristic("CMAES")
    # every rejected offspring was answered (popped from the pending set) ...
    assert not set(rejected) & set(cma._pending)
    # ... and what is still pending at the end are the last generations' in-flight offspring
    assert all(info["gen"] >= cma._gen - 2 for info in cma._pending.values())


def test_timeouts_train_the_model_too():
    from panobbgo.heuristics import Random
    from panobbgo.lib.families import Family, FailureRegion

    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("ball", share=0.3, mode="timeout"))
    s = _run(_spec("t", Random, {"filter": True}), p, 120)
    fm = s.failure_model
    assert fm.kinds.get("timeout", 0) == s.n_timed_out > 0


def test_n_failed_is_recorded_per_run():
    from panobbgo.harness_families import make_failure_battery, run_family_harness
    from panobbgo.harness_ioh import make_ioh_strategies, run_record_to_dict

    inst = [x for x in make_failure_battery(dims=(2,), n_instances=1) if "crash" in x[0]][:1]
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    run = run_family_harness(spec, inst, budget_multiplier=20, base_seed=3, progress=False).runs[0]
    assert run.n_failed is not None and 0 < run.n_failed < run.n_evals
    assert run_record_to_dict(run)["n_failed"] == run.n_failed


# ---------------------------------------------------------------------------
# Arm-specific handling
# ---------------------------------------------------------------------------


class CMAESFailureAwareTests(PanobbgoTestCase):
    def test_failed_offspring_get_no_positive_weight(self):
        """More than λ − μ failures: failure_aware recombines only the finite offspring."""
        from panobbgo.heuristics import CMAES

        outs = {}
        for aware in (False, True):
            self.setUp()
            self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
            h = CMAES(self.strategy, failure_aware=aware, active=False)
            h.on_start()
            n = self.problem.dim
            lam, mu = h._lam, h._mu
            m0 = h._m
            assert m0 is not None
            rng = np.random.default_rng(0)
            entries = []
            for i in range(lam):
                y = rng.standard_normal(n)
                pen = float(i) if i < 1 else float("inf")  # one finite offspring, the rest failed
                entries.append({"penalty": pen, "x": m0 + h._sigma * y, "x_eval": m0 + h._sigma * y, "y": y})
            assert lam - mu < lam - 1  # more failures than λ − μ
            h._update(entries, n_offspring=lam)
            outs[aware] = (np.array(h._m, copy=True), entries[0]["x"])
        m_aware, x_best = outs[True]
        np.testing.assert_allclose(m_aware, x_best)  # the one finite offspring is the new mean
        assert not np.allclose(outs[False][0], outs[False][1])


class TRQFailureAwareTests(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        self.strategy.failure_model = None

    def test_default_ignores_the_model(self):
        from panobbgo.heuristics import TrustRegionQuadratic

        h = TrustRegionQuadratic(self.strategy)
        assert not h._poisoned(np.full(self.problem.dim, 0.5))

    def test_a_failed_start_centre_is_replaced(self):
        from panobbgo.heuristics import TrustRegionQuadratic

        for aware in (False, True):
            h = TrustRegionQuadratic(self.strategy, failure_aware=aware)
            first = h.produce(1)[0]
            h.on_failed_evaluations([first])
            nxt = h.produce(1)[0]
            if aware:
                assert not np.allclose(nxt.x, first.x)
                assert h.n_restarts == 1
            else:
                # the old behaviour: the unevaluated centre is proposed again
                assert np.allclose(nxt.x, first.x)

    def test_failed_points_are_not_proposed_again(self):
        from panobbgo.heuristics import TrustRegionQuadratic

        h = TrustRegionQuadratic(self.strategy, failure_aware=True)
        pts = h.produce(1)
        h.on_new_results([Result(pts[0], 1.0)])  # the centre is fine
        geo = h.produce(2)
        h.on_failed_evaluations(geo)  # both coordinate points crash
        again = h.produce(4)
        for g in geo:
            assert not any(np.allclose(g.x, a.x) for a in again)

    def test_a_poisoned_step_shrinks_the_radius(self):
        from panobbgo.heuristics import TrustRegionQuadratic

        class Everywhere:
            def in_poison(self, x):
                return True

        h = TrustRegionQuadratic(self.strategy, failure_aware=True)
        x_opt = np.array([0.7, -0.3])

        def f(x):
            return float(np.sum((np.asarray(x) - x_opt) ** 2 * np.array([3.0, 1.0])))

        for _ in range(40):  # until the model has made a step
            pts = h.produce(1)
            h.on_new_results([Result(p, f(p.x)) for p in pts])
            if h.n_steps >= 2 and not h._need_geometry:
                break
        assert h.n_steps >= 2
        r0, shrinks = h.radius, h.n_shrink
        self.strategy.failure_model = Everywhere()
        steps = h.n_steps
        pts = h.produce(1)
        assert h.n_poison_skips > 0
        assert h.n_steps == steps  # the poisoned steps were not proposed
        assert h.radius < r0 and h.n_shrink == shrinks + 1  # shrunk once for this model state
        h.produce(1)
        assert h.n_shrink == shrinks + 1  # not again without new data

    def test_the_step_after_a_poisoned_shrink_is_in_the_models_units(self):
        """After the shrink the retried step lies in the smaller region and its ``pred`` is the model's value there.

        Review of #388: the shrink changed ``self.radius`` mid-proposal and the
        retried step read the model (fitted in ``s = (u - c) / r0`` units) in
        the new units: the step had the right length but was the wrong point
        -- the full-radius minimiser compressed to half its length, not the
        minimiser in the half-radius region -- and it carried the full-radius
        ``pred``.
        """
        from panobbgo.heuristics import TrustRegionQuadratic

        h = TrustRegionQuadratic(self.strategy, failure_aware=True)
        x_opt = np.array([4.0, -3.0])  # far from the centre: the full step goes to the trust-region edge

        def f(x):
            return float(np.sum((np.asarray(x) - x_opt) ** 2 * np.array([3.0, 1.0])))

        for _ in range(40):
            pts = h.produce(1)
            h.on_new_results([Result(p, f(p.x)) for p in pts])
            if h.n_steps >= 2 and not h._need_geometry:
                break
        c, f_c = h._center()
        r0 = h.radius
        fitted = {}
        fit = h._fit

        def spy(*a):
            out = fit(*a)
            fitted["model"] = out[0]
            return out

        h._fit = spy

        class Ring:
            """Poison beyond 0.75 of the current radius (inf-norm, normalised) from the centre."""

            def in_poison(self, x):
                return float(np.max(np.abs(h._to_u(x) - c))) > 0.75 * r0

        self.strategy.failure_model = Ring()
        steps = h.n_steps
        pts = h.produce(1)
        assert h.radius == pytest.approx(0.5 * r0) and h.n_poison_skips >= 1
        assert h.n_steps == steps + 1  # the retried step was proposed
        info = h._pending[pts[0].who]
        assert info["kind"] == "step" and info["primary"] and info["radius"] == h.radius
        u = info["u"]
        assert float(np.max(np.abs(u - c))) <= h.radius * (1 + 1e-9)  # inside the shrunk region
        g, H, scale, _ = fitted["model"]
        sv = (u - c) / r0  # the model's own units
        assert info["pred"] == pytest.approx(-float(g @ sv + 0.5 * sv @ H @ sv) * scale, rel=1e-9)


def test_virtual_clock_retry_does_not_advance():
    """A pass whose candidates were all rejected asks again at the same instant."""
    from panobbgo.heuristics import Random
    from panobbgo.lib.families import Family

    s = _spec("v", Random).create_strategy(Family("sphere", dim=2, seed=1), seed=1, max_eval=20)
    s.config.sync_evaluation = True
    from panobbgo.virtual_clock import VirtualSpec

    VirtualSpec(workers=2, duration="lognormal", sigma=0.5, policy="async").apply(s)
    s.initialize() if hasattr(s, "initialize") else None
    clock = s._virtual_clock
    clock.configure()
    one = [Point(np.zeros(2), "Random")]
    clock.step(one)  # one worker busy, one free: no advance, time 0
    assert clock.now == 0.0 and clock.busy == 1
    clock.step([], retry=True)
    assert clock.now == 0.0  # retry: still at the same instant
    clock.step([])
    assert clock.now > 0.0  # nothing proposed, nothing rejected: wait for the completion


@pytest.mark.parametrize("heur_name", ["CMAES", "TrustRegionQuadratic"])
def test_failure_aware_is_bit_identical_without_failures(heur_name):
    """``failure_aware=True`` on a failure-free problem: the same run point for point as the default."""
    from panobbgo import heuristics
    from panobbgo.lib.families import Family

    heur = getattr(heuristics, heur_name)
    p = Family("rosenbrock", dim=3, seed=4)
    a = _run(_spec("a", heur), p, 150)
    b = _run(_spec("b", heur, heur_kwargs={"failure_aware": True}), p, 150)
    assert len(a.results) == len(b.results) == 150
    np.testing.assert_array_equal(_xs(a), _xs(b))


def test_rejections_cost_no_virtual_time(monkeypatch):
    """A pass with rejections and a free worker does not advance the virtual clock."""
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.families import Family, FailureRegion
    from panobbgo.virtual_clock import VirtualClock

    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("halfspace", share=0.45, mode="crash", boundary_gap=0.0))
    seen = {"retry": 0, "advanced": 0}
    orig = VirtualClock.step

    def step(clock, points, retry=False):
        t0, free = clock.now, clock.free - len(points)
        room = clock.strategy._budget_room()
        orig(clock, points, retry=retry)
        if retry and free > 0 and (room is None or room > len(points)):
            seen["retry"] += 1
            seen["advanced"] += clock.now != t0

    monkeypatch.setattr(VirtualClock, "step", step)
    s = _run(_spec("r", CMAES, {"filter": True}, {"failure_aware": True}), p, 200)
    assert s.n_predicted_failures > 0 and seen["retry"] > 0
    assert seen["advanced"] == 0


def test_a_pass_that_only_rejects_is_progress():
    """A pass that dispatches nothing but rejects candidates must not end the run (``progressed``).

    The model rejects every candidate of the first passes, and ``_alive``
    answers ``False`` during them, so the liveness predicate cannot rescue
    the loop: only counting rejections as progress keeps the run going.
    """
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.families import Family
    from panobbgo.strategies import StrategyRoundRobin

    class RejectFirst(FailureModel):
        calls = 0

        def reject(self, points):
            RejectFirst.calls += 1
            if RejectFirst.calls <= 3:
                self.n_rejected += len(points)
                return [True] * len(points)
            return [False] * len(points)

    def dead_while_rejecting(strategy):
        alive = strategy._alive
        strategy._alive = lambda: alive() if not 1 <= RejectFirst.calls <= 3 else False

    spec = StrategySpec(
        name="p",
        strategy_class=StrategyRoundRobin,
        heuristics=[(CMAES, {})],  # re-emits on a failure: the run can go on
        analyzers=[(RejectFirst, {"filter": True})],
    )
    s = _run(spec, Family("sphere", dim=2, seed=1), 40, q=0, before_start=dead_while_rejecting)
    assert RejectFirst.calls > 3 and s.n_predicted_failures > 0
    assert s._dispatched == 40


@pytest.mark.parametrize("q", [0, 1, 4])
def test_a_heuristic_without_a_failure_hook_is_not_starved(q):
    """``Random`` refills only on results: rejections must not drain its queue and end the run early.

    The model marks the whole box (the guard switched off); the streak rule
    lets every 11th candidate through.  With a queue of two and no refill on
    a rejection, the run would stop after the first two rejections.
    """
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import Random
    from panobbgo.lib.families import Family
    from panobbgo.strategies import StrategyRoundRobin

    class Everywhere(FailureModel):
        @property
        def armed(self):
            return True

        def in_poison(self, x):
            X = np.asarray(x, dtype=float)
            return True if X.ndim == 1 else np.ones(X.shape[0], dtype=bool)

    spec = StrategySpec(
        name="s",
        strategy_class=StrategyRoundRobin,
        heuristics=[(Random, {"cap": 2})],
        analyzers=[(Everywhere, {"filter": True, "max_rejects": 10})],
    )
    s = _run(spec, Family("sphere", dim=2, seed=1), 30, q=q)
    assert s._dispatched == 30
    fm = s.failure_model
    assert fm.n_rejected == s.n_predicted_failures >= 10 * 29
    assert fm.n_passed_streak == 30


def test_relay_guards_each_heuristic():
    """One heuristic's exception or StopHeuristic does not keep the others from their predicted failures."""
    import logging
    from types import SimpleNamespace

    from panobbgo.core import StopHeuristic, _PredictedFailureRelay

    got = []

    class H:
        def __init__(self, name, exc=None):
            self.name, self.exc, self.calls = name, exc, 0

        def on_failed_evaluations(self, points):
            self.calls += 1
            got.append((self.name, len(points)))
            if self.exc is not None:
                raise self.exc

    stop, boom, ok = H("stop", StopHeuristic("done")), H("boom", RuntimeError("x")), H("ok")
    nohook = SimpleNamespace(name="nohook")
    relay = _PredictedFailureRelay(
        SimpleNamespace(heuristics=[stop, boom, nohook, ok], logger=logging.getLogger("test"))
    )
    pts = [Point(np.zeros(2), "ok:1")]
    relay.on_predicted_failures(pts)
    relay.on_predicted_failures(pts)
    assert ok.calls == 2 and boom.calls == 2  # an exception is logged, the heuristic keeps getting them
    assert stop.calls == 1  # StopHeuristic ends the relay to that heuristic only


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_guard_disarms_before_the_share_exceeds_max_share(seed):
    """A failure half-space of half the box: the amortised guard disarms before the exact share passes it.

    Review of #388: a share cached between recomputes let the guard stay
    armed while the probe share was 0.535 -- new *successes* raise the share
    too (they shrink the bandwidths), and the old cache was refreshed only on
    new failures.  Here 240 uniform points (about 130 failures, past the exact-recompute range) are followed by successes only (on
    the good side ``x_0 < t``, as a search that avoids the zone would add
    them).  After every batch the exact share is computed afresh; the guard
    must be disarmed whenever it exceeds ``max_share``.
    """
    from panobbgo.lib.families import Family
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Family("sphere", dim=2, seed=1), parse_args=False, seed=1)
    m = FailureModel(s)
    box = np.array(m.problem.box[:, :], dtype=float)
    t = box[0, 1] - 0.5 * (box[0, 1] - box[0, 0])
    rng = np.random.default_rng(seed)
    worst = top = 0.0
    for step in range(200):
        hi = box[:, 1] if step < 60 else np.array([t, box[1, 1]])
        for x in rng.uniform(box[:, 0], hi, size=(4, 2)):
            (m.add_failure if x[0] > t else m.add_success)(x)
        armed = m.armed
        with m._lock:
            exact = float(np.mean(m._p_u(m._probes) >= m.threshold))
        top = max(top, exact)
        if armed:
            worst = max(worst, exact)
    assert top > m.max_share  # the marked share does get past the guard
    assert worst <= m.max_share, worst
