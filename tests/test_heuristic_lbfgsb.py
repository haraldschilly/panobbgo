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

"""Tests for the multi-start L-BFGS-B quasi-Newton heuristic."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from panobbgo.heuristics.lbfgsb import _X0_REQUEST, LBFGSB
from panobbgo.utils import PanobbgoTestCase


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class LBFGSBConstructionTests(PanobbgoTestCase):
    def test_default_construction(self):
        h = LBFGSB(self.strategy)
        assert h.max_starts is None
        assert h.maxfun is None
        assert h.epsilon is None
        assert h.seed is None
        assert h.name == "LBFGSB"
        assert h.cap == 1

    def test_custom_construction(self):
        h = LBFGSB(self.strategy, max_starts=5, maxfun=200, epsilon=1e-6, seed=7, name="LBFGSB_custom")
        assert h.max_starts == 5
        assert h.maxfun == 200
        assert h.epsilon == 1e-6
        assert h.seed == 7
        assert h.name == "LBFGSB_custom"

    def test_warm_start_defaults_off(self):
        h = LBFGSB(self.strategy)
        assert h.warm_start is False
        assert h.warm_start_sigma == 0.1
        assert h._best_x is None

    def test_warm_start_construction(self):
        h = LBFGSB(self.strategy, warm_start=True, warm_start_sigma=0.25)
        assert h.warm_start is True
        assert h.warm_start_sigma == 0.25

    def test_invalid_warm_start_type(self):
        with pytest.raises(ValueError, match="warm_start must be a bool"):
            LBFGSB(self.strategy, warm_start=1)  # type: ignore[arg-type]

    def test_invalid_warm_start_sigma_negative(self):
        with pytest.raises(ValueError, match="warm_start_sigma"):
            LBFGSB(self.strategy, warm_start_sigma=-0.1)

    def test_invalid_warm_start_sigma_inf(self):
        with pytest.raises(ValueError, match="warm_start_sigma"):
            LBFGSB(self.strategy, warm_start_sigma=float("inf"))

    def test_warm_start_sigma_zero_allowed(self):
        # 0.0 = polish the exact incumbent every restart; degenerate but valid.
        h = LBFGSB(self.strategy, warm_start=True, warm_start_sigma=0.0)
        assert h.warm_start_sigma == 0.0

    def test_invalid_max_starts_zero(self):
        with pytest.raises(ValueError, match="max_starts"):
            LBFGSB(self.strategy, max_starts=0)

    def test_invalid_max_starts_negative(self):
        with pytest.raises(ValueError, match="max_starts"):
            LBFGSB(self.strategy, max_starts=-2)

    def test_invalid_max_starts_type(self):
        with pytest.raises(ValueError, match="max_starts"):
            LBFGSB(self.strategy, max_starts=2.5)  # type: ignore[arg-type]

    def test_invalid_max_starts_bool_rejected(self):
        # bool is an int subclass — explicitly rejected to avoid True==1 surprises.
        with pytest.raises(ValueError, match="max_starts"):
            LBFGSB(self.strategy, max_starts=True)  # type: ignore[arg-type]

    def test_invalid_maxfun_zero(self):
        with pytest.raises(ValueError, match="maxfun"):
            LBFGSB(self.strategy, maxfun=0)

    def test_invalid_maxfun_type(self):
        with pytest.raises(ValueError, match="maxfun"):
            LBFGSB(self.strategy, maxfun=10.0)  # type: ignore[arg-type]

    def test_invalid_epsilon_zero(self):
        with pytest.raises(ValueError, match="epsilon"):
            LBFGSB(self.strategy, epsilon=0.0)

    def test_invalid_epsilon_negative(self):
        with pytest.raises(ValueError, match="epsilon"):
            LBFGSB(self.strategy, epsilon=-1e-6)

    def test_invalid_epsilon_inf(self):
        with pytest.raises(ValueError, match="epsilon"):
            LBFGSB(self.strategy, epsilon=float("inf"))


# ----------------------------------------------------------------------
# Subprocess lifecycle
# ----------------------------------------------------------------------


class LBFGSBLifecycleTests(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_start_spawns_process(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        try:
            assert h.lbfgsb is not None
            assert h.lbfgsb.is_alive()
        finally:
            h.__stop__()
        assert not h.lbfgsb.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_stop_force_kill(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        with mock.patch.object(h.lbfgsb, "is_alive", side_effect=[True, True, True, False]):
            with mock.patch.object(h.lbfgsb, "kill") as mock_kill:
                h.__stop__()
                mock_kill.assert_called_once()


# ----------------------------------------------------------------------
# Pipe wiring
# ----------------------------------------------------------------------


class LBFGSBPipeTests(PanobbgoTestCase):
    def test_on_new_results_routes_penalty_value(self):
        h = LBFGSB(self.strategy)
        h.p1 = mock.MagicMock()

        class _R:
            def __init__(self, who, fx):
                self.who = who
                self.fx = fx

        results = [_R("LBFGSB", 5.0), _R("Other", 3.0)]
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        h.on_new_results(results)
        h.p1.send.assert_called_once_with(5.0)

    def test_on_new_results_ignores_foreign_who(self):
        h = LBFGSB(self.strategy)
        h.p1 = mock.MagicMock()

        class _R:
            def __init__(self, who, fx):
                self.who = who
                self.fx = fx

        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        h.on_new_results([_R("OtherHeuristic", 3.14)])
        h.p1.send.assert_not_called()

    def test_on_start_exits_on_pipe_closed(self):
        h = LBFGSB(self.strategy)
        h.out1 = mock.MagicMock()
        h.out1.poll.return_value = False
        h.p1 = mock.MagicMock()
        h.p1.poll.side_effect = EOFError()
        h._stopped = False
        h.on_start()  # must exit silently

    def test_on_start_logs_output(self):
        h = LBFGSB(self.strategy)
        h.out1 = mock.MagicMock()
        h.out1.poll.return_value = True
        h.out1.recv.return_value = "status payload"
        h.p1 = mock.MagicMock()
        h.p1.poll.side_effect = EOFError()
        h._stopped = False
        h.logger = mock.MagicMock()
        h.on_start()
        h.logger.info.assert_called_with("status payload")

    def test_on_start_emits_requested_point(self):
        h = LBFGSB(self.strategy)
        h.out1 = mock.MagicMock()
        h.out1.poll.return_value = False
        h.p1 = mock.MagicMock()
        # First poll yields a point; the emit handler then flips _stopped so
        # the loop terminates after exactly one iteration.
        h.p1.poll.return_value = True
        h.p1.recv.return_value = np.array([0.1, 0.2])

        emitted = []

        def fake_emit(x):
            emitted.append(np.asarray(x))
            h._stopped = True

        h._stopped = False
        with mock.patch.object(h, "emit", side_effect=fake_emit):
            h.on_start()
        assert len(emitted) == 1
        np.testing.assert_allclose(emitted[0], [0.1, 0.2])


# ----------------------------------------------------------------------
# Warm-started restarts (memetic polish of the best incumbent)
# ----------------------------------------------------------------------


class _FakeBest:
    def __init__(self, x):
        self.x = None if x is None else np.asarray(x, dtype=float)


class LBFGSBWarmStartTests(PanobbgoTestCase):
    def test_on_new_best_tracks_incumbent(self):
        h = LBFGSB(self.strategy, warm_start=True)
        assert h._best_x is None
        h.on_new_best(_FakeBest([0.3, -0.4]))
        np.testing.assert_allclose(h._best_x, [0.3, -0.4])
        # A later, better incumbent overwrites the stored one.
        h.on_new_best(_FakeBest([0.1, 0.1]))
        np.testing.assert_allclose(h._best_x, [0.1, 0.1])

    def test_on_new_best_ignores_none(self):
        h = LBFGSB(self.strategy, warm_start=True)
        h.on_new_best(None)
        assert h._best_x is None
        h.on_new_best(_FakeBest(None))
        assert h._best_x is None

    def test_warm_start_x0_uniform_before_first_best(self):
        # With no incumbent yet, the warm-start draw is uniform over the box.
        h = LBFGSB(self.strategy, warm_start=True, seed=0)
        x0 = h._warm_start_x0()
        bounds = h._box_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        assert np.all(x0 >= lo) and np.all(x0 <= hi)

    def test_warm_start_x0_perturbs_incumbent(self):
        h = LBFGSB(self.strategy, warm_start=True, warm_start_sigma=0.05, seed=3)
        bounds = h._box_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        h.on_new_best(_FakeBest(center))
        draws = np.array([h._warm_start_x0() for _ in range(200)])
        # Every draw stays inside the box.
        assert np.all(draws >= lo) and np.all(draws <= hi)
        # Draws cluster around the incumbent, not uniformly over the box:
        # mean close to centre, spread far below the box range.
        ranges = hi - lo
        assert np.all(np.abs(draws.mean(axis=0) - center) < 0.1 * ranges)
        assert np.all(draws.std(axis=0) < 0.5 * ranges)
        # ... and the spread is roughly the requested sigma·range (not tiny).
        assert np.all(draws.std(axis=0) > 0.01 * ranges)

    def test_warm_start_x0_sigma_zero_returns_exact_incumbent(self):
        h = LBFGSB(self.strategy, warm_start=True, warm_start_sigma=0.0, seed=1)
        bounds = h._box_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = (lo + hi) / 2.0
        h.on_new_best(_FakeBest(center))
        np.testing.assert_allclose(h._warm_start_x0(), center)

    def test_warm_start_x0_clips_incumbent_perturbation_into_box(self):
        # An incumbent on the box edge + large sigma must still clip inside.
        h = LBFGSB(self.strategy, warm_start=True, warm_start_sigma=5.0, seed=2)
        bounds = h._box_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        h.on_new_best(_FakeBest(hi))
        for _ in range(20):
            x0 = h._warm_start_x0()
            assert np.all(x0 >= lo - 1e-12) and np.all(x0 <= hi + 1e-12)

    def test_on_start_answers_x0_request_inline(self):
        # An ``_X0_REQUEST`` sentinel from the worker is answered on the pipe
        # (never emitted as a search point).
        h = LBFGSB(self.strategy, warm_start=True, seed=0)
        h.out1 = mock.MagicMock()
        h.out1.poll.return_value = False
        h.p1 = mock.MagicMock()
        h.p1.poll.return_value = True

        sent = []
        h.p1.send.side_effect = lambda payload: sent.append(payload)
        # First recv yields the sentinel; the send handler then stops the loop.
        h.p1.recv.return_value = _X0_REQUEST

        def stop_after_send(payload):
            sent.append(payload)
            h._stopped = True

        h.p1.send.side_effect = stop_after_send
        h._stopped = False
        with mock.patch.object(h, "emit") as mock_emit:
            h.on_start()
        mock_emit.assert_not_called()
        assert len(sent) == 1
        # The answer is a valid in-box point (uniform draw since no incumbent).
        bounds = h._box_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0 = np.asarray(sent[0], dtype=float)
        assert np.all(x0 >= lo) and np.all(x0 <= hi)


class _WarmFakePipe:
    """Fake pipe that also answers the worker's ``_X0_REQUEST`` sentinels.

    ``send`` records requests; when the most recent send was the x0 sentinel,
    the next ``recv`` returns the configured warm-start point instead of an
    objective value.
    """

    def __init__(self, func, warm_x0, max_evals=10_000):
        self.func = func
        self.warm_x0 = np.asarray(warm_x0, dtype=float)
        self.max_evals = max_evals
        self.evals = 0
        self.last = None
        self.last_was_request = False
        self.sent_points = []
        self.x0_requests = 0

    def send(self, payload):
        if isinstance(payload, str) and payload == _X0_REQUEST:
            self.last_was_request = True
            self.x0_requests += 1
        else:
            self.last = np.asarray(payload, dtype=float)
            self.last_was_request = False
            self.sent_points.append(self.last.copy())

    def recv(self):
        if self.last_was_request:
            self.last_was_request = False
            return self.warm_x0.copy()
        self.evals += 1
        if self.evals > self.max_evals:
            raise EOFError
        return float(self.func(self.last))


class LBFGSBWarmStartWorkerTests(PanobbgoTestCase):
    def test_worker_requests_x0_for_warm_restarts(self):
        # With warm_start=True and >1 start, the worker must issue exactly
        # (max_starts - 1) x0 requests (the first start uses x0_first).
        warm_x0 = np.array([0.3, 0.3])
        pipe = _WarmFakePipe(_quadratic, warm_x0)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=4,
            maxfun=200,
            epsilon=None,
            warm_start=True,
        )
        assert pipe.x0_requests == 3
        output.send.assert_called_once_with({"done": 4})

    def test_worker_no_x0_request_when_warm_start_off(self):
        pipe = _WarmFakePipe(_quadratic, np.array([0.3, 0.3]))
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=4,
            maxfun=200,
            epsilon=None,
            warm_start=False,
        )
        assert pipe.x0_requests == 0

    def test_worker_warm_restart_descends_from_requested_x0(self):
        # The second descent should begin at the parent-supplied warm x0.
        warm_x0 = np.array([0.25, -0.25])
        pipe = _WarmFakePipe(_quadratic, warm_x0)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.array([0.9, 0.9]),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=2,
            maxfun=200,
            epsilon=None,
            warm_start=True,
        )
        # First evaluated point is x0_first; the first point after the single
        # x0 request is the warm x0.
        np.testing.assert_allclose(pipe.sent_points[0], [0.9, 0.9])
        assert pipe.x0_requests == 1
        # Locate the first evaluated point drawn from the warm restart: it is
        # the very next distinct start point at (or near) warm_x0.
        assert any(np.allclose(p, warm_x0) for p in pipe.sent_points)

    def test_worker_warm_x0_request_clean_exit_on_closed_pipe(self):
        # A closed pipe mid-x0-request raises SystemExit inside the worker,
        # which returns cleanly without a "done" payload.
        class _ClosingPipe(_WarmFakePipe):
            def recv(self):
                if self.last_was_request:
                    raise EOFError
                return super().recv()

        pipe = _ClosingPipe(_quadratic, np.array([0.3, 0.3]), max_evals=3)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=None,
            maxfun=50,
            epsilon=None,
            warm_start=True,
        )
        output.send.assert_not_called()


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class LBFGSBRestartTests(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_relaunches_subprocess(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        old_proc = h.lbfgsb
        try:
            h.on_restart(center=np.array([0.5, -0.5]), reason="test")
            assert h.lbfgsb is not None
            assert h.lbfgsb is not old_proc
            assert h.lbfgsb.is_alive()
        finally:
            h.__stop__()
        assert not old_proc.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_center_none_uses_box_centre(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        try:
            h.on_restart(center=None, reason="test")
            assert h.lbfgsb is not None
            assert h.lbfgsb.is_alive()
        finally:
            h.__stop__()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_out_of_box_center_is_clipped(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        try:
            h.on_restart(center=np.array([1000.0, -1000.0]), reason="test")
            assert h.lbfgsb is not None
            assert h.lbfgsb.is_alive()
        finally:
            h.__stop__()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_when_stopped_noop(self, _mock_setup):
        h = LBFGSB(self.strategy)
        h.__start__()
        h.__stop__()
        h._stopped = True
        h.on_restart(center=np.array([0.0, 0.0]), reason="test")
        assert not h.lbfgsb.is_alive()


# ----------------------------------------------------------------------
# Worker: multi-start behaviour (driven through a fake pipe)
# ----------------------------------------------------------------------


class _FakePipe:
    """A fake bidirectional pipe that answers the worker's ``f(x)`` calls.

    ``send`` records the most-recently requested point; ``recv`` evaluates
    the user objective at that point.  After ``max_calls`` evaluations
    ``recv`` raises ``EOFError`` to emulate the parent closing the pipe.
    """

    def __init__(self, func, max_calls=10_000):
        self.func = func
        self.max_calls = max_calls
        self.calls = 0
        self.last = None
        self.sent = []

    def send(self, x):
        self.last = np.asarray(x, dtype=float)
        self.sent.append(self.last.copy())

    def recv(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise EOFError
        return float(self.func(self.last))


def _quadratic(x):
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


_BOUNDS = [(-1.0, 1.0), (-1.0, 1.0)]
_LB = np.array([-1.0, -1.0])
_UB = np.array([1.0, 1.0])


class LBFGSBWorkerTests(PanobbgoTestCase):
    def test_worker_completes_all_starts(self):
        pipe = _FakePipe(_quadratic)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=3,
            maxfun=200,
            epsilon=None,
        )
        output.send.assert_called_once_with({"done": 3})

    def test_worker_first_start_is_box_centre(self):
        pipe = _FakePipe(_quadratic)
        output = mock.MagicMock()
        x0_first = np.array([0.123, -0.456])
        LBFGSB.worker(
            pipe,
            output,
            x0_first,
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=1,
            maxfun=200,
            epsilon=None,
        )
        # The very first point the solver evaluates is x0_first.
        np.testing.assert_allclose(pipe.sent[0], x0_first)

    def test_worker_clean_exit_on_closed_pipe(self):
        # recv raises EOFError almost immediately -> SystemExit inside f ->
        # worker returns without sending a "done" payload and without raising.
        pipe = _FakePipe(_quadratic, max_calls=1)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=None,
            maxfun=200,
            epsilon=None,
        )
        output.send.assert_not_called()

    def test_worker_restarts_are_seed_reproducible(self):
        out_a, out_b = mock.MagicMock(), mock.MagicMock()
        pipe_a = _FakePipe(_quadratic)
        pipe_b = _FakePipe(_quadratic)
        common = dict(seed=42, max_starts=3, maxfun=200, epsilon=None)
        LBFGSB.worker(pipe_a, out_a, np.zeros(2), _BOUNDS, _LB, _UB, **common)
        LBFGSB.worker(pipe_b, out_b, np.zeros(2), _BOUNDS, _LB, _UB, **common)
        assert len(pipe_a.sent) == len(pipe_b.sent)
        for xa, xb in zip(pipe_a.sent, pipe_b.sent):
            np.testing.assert_allclose(xa, xb)

    def test_worker_minimizes_quadratic(self):
        # The best point evaluated across the descents should approach the
        # quadratic's minimiser at (0.3, 0.3).
        pipe = _FakePipe(_quadratic)
        output = mock.MagicMock()
        LBFGSB.worker(
            pipe,
            output,
            np.zeros(2),
            _BOUNDS,
            _LB,
            _UB,
            seed=0,
            max_starts=1,
            maxfun=500,
            epsilon=None,
        )
        best = min(_quadratic(x) for x in pipe.sent)
        assert best < 1e-6


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class LBFGSBRegistrationTests(PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        from panobbgo.heuristics import LBFGSB as Reg

        assert Reg is LBFGSB

    def test_in_structural_catalog_candidate_pool(self):
        from panobbgo.self_improve import default_structural_catalog

        cat = default_structural_catalog()
        candidate_classes: list = []
        for rule in cat.rules:
            if getattr(rule, "op", None) == "add_heuristic":
                for cls, _kwargs in getattr(rule, "candidate_classes", ()):
                    candidate_classes.append(cls)
        assert LBFGSB in candidate_classes

    def test_kwarg_catalog_has_max_starts_rule(self):
        """``default_catalog`` ships an ``LBFGSB.max_starts`` integer_add rule.

        The rule lets the autonomous loop tune the multi-start restart
        budget on specs that opt into a concrete ``max_starts`` value;
        the heuristic's ``None`` default (= unlimited until budget) is
        filtered out by :func:`_find_targets`'s ``None``-skip predicate.
        """
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "LBFGSB" and r.param_name == "max_starts"
        ]
        assert len(rules) == 1, "expected exactly one LBFGSB.max_starts rule"
        rule = rules[0]
        assert rule.kind == "integer_add"
        assert rule.bounds == (1, 50)
        # Both halves of the perturbation cone should be reachable.
        assert any(d < 0 for d in rule.delta_choices)
        assert any(d > 0 for d in rule.delta_choices)

    def test_max_starts_rule_skips_none_sentinel(self):
        """The ``LBFGSB.max_starts`` rule must skip specs that ship
        ``max_starts=None`` — i.e. the heuristic-internal auto-default
        (unlimited until budget) — so the loop cannot try to perturb
        the sentinel value.

        Without the ``None``-skip in :func:`_find_targets` the
        integer_add rule would attempt ``int(None) + delta`` and crash
        when applied to the structural catalog's bare LBFGSB candidate.
        """
        import numpy as np

        from panobbgo.benchmark import StrategySpec
        from panobbgo.self_improve import MutationCatalog, MutationRule
        from panobbgo.strategies.rewarding import StrategyRewarding

        rule = MutationRule(
            strategy_pattern="",
            class_name="LBFGSB",
            param_name="max_starts",
            kind="integer_add",
            bounds=(1, 50),
            delta_choices=(-5, -2, -1, 1, 2, 5),
            probability=1.0,
        )

        spec_none = StrategySpec(
            name="StratNone",
            strategy_class=StrategyRewarding,
            heuristics=[(LBFGSB, {"max_starts": None})],
        )
        spec_int = StrategySpec(
            name="StratInt",
            strategy_class=StrategyRewarding,
            heuristics=[(LBFGSB, {"max_starts": 10})],
        )
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)

        # Only the int-valued spec should ever be mutated.
        for _ in range(30):
            prop = cat.sample(rng, [spec_none, spec_int])
            assert prop is not None
            assert prop.strategy_name == "StratInt"
            assert prop.old_value == 10
            assert 1 <= int(prop.new_value) <= 50


# ----------------------------------------------------------------------
# End-to-end smoke: a dedicated L-BFGS-B descent cracks Rosenbrock,
# the valley problem on which every Panobbgo *strategy* scores 0.0.
# ----------------------------------------------------------------------


class LBFGSBRosenbrockSmokeTests(PanobbgoTestCase):
    def test_scipy_lbfgsb_cracks_rosenbrock_5d(self):
        from scipy.optimize import fmin_l_bfgs_b

        def rosen(x):
            x = np.asarray(x, dtype=float)
            return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

        bounds = [(-2.048, 2.048)] * 5
        x_opt, f_opt, info = fmin_l_bfgs_b(rosen, np.zeros(5), bounds=bounds, approx_grad=True)
        # Tolerance on the standard battery for Rosenbrock_5D is 1.0; a single
        # descent from the box centre comfortably reaches the optimum basin.
        assert f_opt < 1.0
        assert info["funcalls"] < 5000
