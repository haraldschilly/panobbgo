# -*- coding: utf8 -*-
import unittest
from unittest.mock import Mock, MagicMock
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
        """Test that on_new_results sends penalty value to pipe, not just fx."""

        # Create a result
        point = Point(np.array([1.0, 2.0]), "LBFGSB") # Must match heuristic name
        # Force heuristic name to match what we put in point
        self.heuristic._name = "LBFGSB"

        result = Result(point, 100.0, cv_vec=np.array([1.0]))

        # Verify setup
        assert result.who == "LBFGSB"
        assert result.fx == 100.0

        # Call method
        self.heuristic.on_new_results([result])

        # Check interactions
        # 1. get_penalty_value should be called with result
        self.strategy.constraint_handler.get_penalty_value.assert_called_with(result)

        # 2. pipe.send should be called with the return value of get_penalty_value (123.45)
        # NOT with result.fx (100.0)
        self.heuristic.p1.send.assert_called_with(123.45)

    def test_on_new_results_ignore_other_heuristics(self):
        """Test that results from other heuristics are ignored."""
        point = Point(np.array([1.0, 2.0]), "OtherHeuristic")
        result = Result(point, 100.0)

        self.heuristic.on_new_results([result])

        self.heuristic.p1.send.assert_not_called()
