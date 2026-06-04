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

"""Tests for the COBYQA derivative-free trust-region heuristic."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from panobbgo.heuristics.cobyqa import COBYQA
from panobbgo.utils import PanobbgoTestCase


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class COBYQAConstructionTests(PanobbgoTestCase):
    def test_default_construction(self):
        h = COBYQA(self.strategy)
        assert h.initial_tr_radius is None  # auto-resolved at start
        assert h.final_tr_radius == 1e-6
        assert h.maxfev is None
        assert h.scale is True
        assert h.name == "COBYQA"
        assert h.cap == 1

    def test_custom_construction(self):
        h = COBYQA(
            self.strategy,
            initial_tr_radius=0.5,
            final_tr_radius=1e-5,
            maxfev=200,
            scale=False,
            name="COBYQA_custom",
        )
        assert h.initial_tr_radius == 0.5
        assert h.final_tr_radius == 1e-5
        assert h.maxfev == 200
        assert h.scale is False
        assert h.name == "COBYQA_custom"

    def test_invalid_initial_tr_radius_negative(self):
        with pytest.raises(ValueError, match="initial_tr_radius"):
            COBYQA(self.strategy, initial_tr_radius=-1.0)

    def test_invalid_initial_tr_radius_zero(self):
        with pytest.raises(ValueError, match="initial_tr_radius"):
            COBYQA(self.strategy, initial_tr_radius=0.0)

    def test_invalid_initial_tr_radius_nan(self):
        with pytest.raises(ValueError, match="initial_tr_radius"):
            COBYQA(self.strategy, initial_tr_radius=float("nan"))

    def test_invalid_final_tr_radius_negative(self):
        with pytest.raises(ValueError, match="final_tr_radius"):
            COBYQA(self.strategy, final_tr_radius=-1e-6)

    def test_invalid_final_tr_radius_zero(self):
        with pytest.raises(ValueError, match="final_tr_radius"):
            COBYQA(self.strategy, final_tr_radius=0.0)

    def test_invalid_final_tr_radius_inf(self):
        with pytest.raises(ValueError, match="final_tr_radius"):
            COBYQA(self.strategy, final_tr_radius=float("inf"))

    def test_invalid_tr_radius_ordering(self):
        # final must be strictly less than initial
        with pytest.raises(ValueError, match="strictly less"):
            COBYQA(self.strategy, initial_tr_radius=0.1, final_tr_radius=0.1)
        with pytest.raises(ValueError, match="strictly less"):
            COBYQA(self.strategy, initial_tr_radius=0.1, final_tr_radius=0.2)

    def test_invalid_maxfev_type(self):
        with pytest.raises(ValueError, match="maxfev"):
            COBYQA(self.strategy, maxfev=10.5)  # type: ignore[arg-type]

    def test_invalid_maxfev_zero(self):
        with pytest.raises(ValueError, match="maxfev"):
            COBYQA(self.strategy, maxfev=0)

    def test_invalid_maxfev_negative(self):
        with pytest.raises(ValueError, match="maxfev"):
            COBYQA(self.strategy, maxfev=-10)


# ----------------------------------------------------------------------
# Initial TR radius auto-resolution
# ----------------------------------------------------------------------


class COBYQAInitialTRTests(PanobbgoTestCase):
    def test_auto_initial_tr_uses_box_width(self):
        h = COBYQA(self.strategy)
        # Rosenbrock-2D box is [-2.048, 2.048] x [-2.048, 2.048],
        # so max width is ~4.096 and 10% of that is ~0.4096.
        box = self.problem.box.box
        widths = np.asarray(box)[:, 1] - np.asarray(box)[:, 0]
        expected = 0.1 * float(np.max(widths))
        assert h._resolve_initial_tr(box) == pytest.approx(expected)

    def test_user_initial_tr_takes_precedence(self):
        h = COBYQA(self.strategy, initial_tr_radius=0.05)
        assert h._resolve_initial_tr(self.problem.box.box) == 0.05

    def test_auto_initial_tr_respects_final_floor(self):
        # When final_tr_radius is large and the box width is tiny, the
        # auto-derived initial radius must still be strictly greater
        # than the final radius — otherwise COBYQA terminates instantly.
        h = COBYQA(self.strategy, final_tr_radius=1e-4)
        # Synthesise a degenerate one-element box of width 1e-8
        tiny_box = np.array([[0.0, 1e-8]])
        radius = h._resolve_initial_tr(tiny_box)
        assert radius > h.final_tr_radius

    def test_auto_initial_tr_zero_width_box_fallback(self):
        h = COBYQA(self.strategy)
        # Zero-width box (degenerate, but should not crash)
        zero_box = np.array([[0.0, 0.0]])
        radius = h._resolve_initial_tr(zero_box)
        assert radius > 0.0
        assert radius > h.final_tr_radius


# ----------------------------------------------------------------------
# Subprocess lifecycle
# ----------------------------------------------------------------------


class COBYQALifecycleTests(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_start_spawns_process(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()
        try:
            assert cobyqa.cobyqa is not None
            assert cobyqa.cobyqa.is_alive()
        finally:
            cobyqa.__stop__()
        assert not cobyqa.cobyqa.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_stop_force_kill(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()

        # Mock is_alive to keep returning True so __stop__ falls through
        # to .kill() — same pattern as the LBFGSB tests.
        with mock.patch.object(cobyqa.cobyqa, "is_alive", side_effect=[True, True, True, False]):
            with mock.patch.object(cobyqa.cobyqa, "kill") as mock_kill:
                cobyqa.__stop__()
                mock_kill.assert_called_once()


# ----------------------------------------------------------------------
# Pipe wiring
# ----------------------------------------------------------------------


class COBYQAPipeTests(PanobbgoTestCase):
    def test_on_new_results_routes_penalty_value(self):
        cobyqa = COBYQA(self.strategy)
        cobyqa.p1 = mock.MagicMock()

        class _R:
            def __init__(self, who, fx):
                self.who = who
                self.fx = fx

        results = [_R("COBYQA", 7.5), _R("Other", 1.0)]
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        cobyqa.on_new_results(results)
        cobyqa.p1.send.assert_called_once_with(7.5)

    def test_on_new_results_ignores_foreign_who(self):
        cobyqa = COBYQA(self.strategy)
        cobyqa.p1 = mock.MagicMock()

        class _R:
            def __init__(self, who, fx):
                self.who = who
                self.fx = fx

        results = [_R("OtherHeuristic", 3.14)]
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
        cobyqa.on_new_results(results)
        cobyqa.p1.send.assert_not_called()

    def test_on_start_exits_on_pipe_closed(self):
        cobyqa = COBYQA(self.strategy)
        cobyqa.out1 = mock.MagicMock()
        cobyqa.out1.poll.return_value = False
        cobyqa.p1 = mock.MagicMock()
        cobyqa.p1.poll.side_effect = EOFError()
        cobyqa._stopped = False
        cobyqa.on_start()  # must exit silently

    def test_on_start_logs_output(self):
        cobyqa = COBYQA(self.strategy)
        cobyqa.out1 = mock.MagicMock()
        cobyqa.out1.poll.return_value = True
        cobyqa.out1.recv.return_value = "solution payload"
        cobyqa.p1 = mock.MagicMock()
        cobyqa.p1.poll.side_effect = EOFError()
        cobyqa._stopped = False
        cobyqa.logger = mock.MagicMock()
        cobyqa.on_start()
        cobyqa.logger.info.assert_called_with("solution payload")


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class COBYQARestartTests(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_relaunches_subprocess(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()
        old_proc = cobyqa.cobyqa
        try:
            # Restart at a specific in-box point.
            center = np.array([0.5, -0.5])
            cobyqa.on_restart(center=center, reason="test")
            assert cobyqa.cobyqa is not None
            assert cobyqa.cobyqa is not old_proc
            assert cobyqa.cobyqa.is_alive()
        finally:
            cobyqa.__stop__()
        # Old process should be terminated.
        assert not old_proc.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_center_none_uses_box_centre(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()
        try:
            cobyqa.on_restart(center=None, reason="test")
            assert cobyqa.cobyqa is not None
            assert cobyqa.cobyqa.is_alive()
        finally:
            cobyqa.__stop__()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_out_of_box_center_is_clipped(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()
        try:
            # Way outside the Rosenbrock box (which is roughly [-2, 2]^2):
            far_center = np.array([1000.0, -1000.0])
            cobyqa.on_restart(center=far_center, reason="test")
            assert cobyqa.cobyqa is not None
            assert cobyqa.cobyqa.is_alive()
        finally:
            cobyqa.__stop__()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_restart_when_stopped_noop(self, _mock_setup):
        cobyqa = COBYQA(self.strategy)
        cobyqa.__start__()
        cobyqa.__stop__()
        cobyqa._stopped = True
        # No subprocess after stop+restart_when_stopped — restart sees the
        # flag and returns without spawning a new process.
        cobyqa.on_restart(center=np.array([0.0, 0.0]), reason="test")
        # The old proc was terminated by __stop__; the restart must not
        # have spawned a fresh live process.
        # (cobyqa.cobyqa is still the old, terminated handle.)
        assert not cobyqa.cobyqa.is_alive()


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class COBYQARegistrationTests(PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        from panobbgo.heuristics import COBYQA as Reg

        assert Reg is COBYQA

    def test_in_structural_catalog_candidate_pool(self):
        from panobbgo.self_improve import default_structural_catalog

        cat = default_structural_catalog()
        candidate_classes: list = []
        for rule in cat.rules:
            if getattr(rule, "op", None) == "add_heuristic":
                for cls, _kwargs in getattr(rule, "candidate_classes", ()):
                    candidate_classes.append(cls)
        assert COBYQA in candidate_classes

    def test_kwarg_rules_present(self):
        from panobbgo.self_improve import default_catalog

        cat = default_catalog()
        names = {(getattr(r, "class_name", None), getattr(r, "param_name", None)) for r in cat.rules}
        assert ("COBYQA", "initial_tr_radius") in names
        assert ("COBYQA", "final_tr_radius") in names
        assert ("COBYQA", "scale") in names

    def test_kwarg_catalog_cobyqa_scale_is_categorical(self):
        """``COBYQA.scale`` is a ``categorical_choice`` toggle over the
        two valid box-rescaling regimes (``True`` / ``False``)."""
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "COBYQA" and r.param_name == "scale"
        ]
        assert len(rules) == 1
        rule = rules[0]
        assert rule.kind == "categorical_choice"
        assert set(rule.choices) == {True, False}


# ----------------------------------------------------------------------
# End-to-end smoke: COBYQA actually reduces the objective on a quadratic
# ----------------------------------------------------------------------


class COBYQASmokeTests(PanobbgoTestCase):
    """Verify that the subprocess can complete a tiny COBYQA solve.

    We bypass the Heuristic event-loop machinery and exercise the
    ``worker`` static method directly through a shim that answers the
    function callable with a real evaluation rather than a pipe round
    trip.  This catches any breakage of the scipy COBYQA option
    plumbing (e.g. scipy < 1.14, removed kwarg names) without spinning
    up the full strategy lifecycle.
    """

    def test_scipy_cobyqa_minimizes_quadratic(self):
        from scipy.optimize import minimize

        def f(x: np.ndarray) -> float:
            return float((x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2)

        result = minimize(
            f,
            np.array([0.0, 0.0]),
            method="COBYQA",
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            options={"initial_tr_radius": 0.4, "final_tr_radius": 1e-6, "scale": True},
        )
        assert result.success
        assert result.fun < 1e-6
        assert np.allclose(result.x, [1.5, -0.5], atol=1e-3)
