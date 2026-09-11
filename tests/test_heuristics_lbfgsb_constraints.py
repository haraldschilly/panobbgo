# -*- coding: utf8 -*-
import unittest
from unittest.mock import Mock
from panobbgo.heuristics.lbfgsb import LBFGSB
from panobbgo.lib import Result, Point
import numpy as np


class TestLBFGSBConstraints(unittest.TestCase):
    def setUp(self):
        self.strategy = Mock()
        self.strategy.config = Mock()
        self.strategy.config.get_logger.return_value = Mock()
        self.strategy.config.capacity = 10
        self.strategy.problem = Mock()

        # Mock constraint handler
        self.strategy.constraint_handler = Mock()
        self.strategy.constraint_handler.get_penalty_value.return_value = 123.45

        # Instantiate LBFGSB
        self.heuristic = LBFGSB(self.strategy)

        # Mock pipe
        self.heuristic.p1 = Mock()

    def test_on_new_results_uses_penalty(self):
        """The value handed to the solver is the *penalty*, not the raw fx.

        Since the pull bridge, ``on_new_results`` runs on the event-bus
        thread and only stores the value; ``produce`` puts it on the pipe.
        The contract under test is unchanged: it is
        ``constraint_handler.get_penalty_value(result)`` that reaches the
        solver, never ``result.fx``.
        """
        point = Point(np.array([1.0, 2.0]), "LBFGSB")  # Must match heuristic name
        # Force heuristic name to match what we put in point
        self.heuristic._name = "LBFGSB"

        result = Result(point, 100.0, cv_vec=np.array([1.0]))

        # Verify setup
        assert result.who == "LBFGSB"
        assert result.fx == 100.0

        # A value is only collected while an evaluation of ours is in flight.
        self.heuristic._outstanding = True
        self.heuristic.on_new_results([result])

        self.strategy.constraint_handler.get_penalty_value.assert_called_with(result)
        # The penalty (123.45), not result.fx (100.0).
        assert self.heuristic._fx_inbox.get_nowait() == 123.45

    def test_on_new_results_ignore_other_heuristics(self):
        """Test that results from other heuristics are ignored."""
        point = Point(np.array([1.0, 2.0]), "OtherHeuristic")
        result = Result(point, 100.0)

        self.heuristic._outstanding = True
        self.heuristic.on_new_results([result])

        assert self.heuristic._fx_inbox.empty()
        self.heuristic.p1.send.assert_not_called()
