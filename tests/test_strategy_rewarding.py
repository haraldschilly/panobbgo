# -*- coding: utf8 -*-
import pytest
from unittest import mock
from panobbgo.utils import PanobbgoTestCase
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib import Result, Point
from panobbgo.core import Heuristic
import numpy as np


class DummyHeur(Heuristic):
    def __init__(self, strategy, name):
        super().__init__(strategy, name=name)

    @property
    def active(self):
        return True

    def get_points(self, limit):
        return [Point(np.array([1.0, 2.0]), self.name)]


class TestStrategyRewarding(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_execute_empty_heurs(self, mock_setup):
        strategy = StrategyRewarding(self.problem, parse_args=False)

        # Override the property locally for this test
        with mock.patch.object(StrategyRewarding, "heuristics", new_callable=mock.PropertyMock) as mock_heuristics:
            mock_heuristics.return_value = []
            points = strategy.execute()

        assert points == []

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_execute_with_heurs(self, mock_setup):
        strategy = StrategyRewarding(self.problem, parse_args=False)

        # We need mock evaluator property outstanding
        mock_evaluators = mock.MagicMock()
        mock_evaluators.outstanding = []
        mock_evaluators.__len__.return_value = 1

        # Add a mock heuristic
        h1 = DummyHeur(strategy, "h1")
        h1.performance = 1.0

        # Force add heuristic so it shows up in strategy.heuristics
        strategy.add_heuristic(h1)

        # Override evaluators for this test to bypass dask/processes
        with mock.patch.object(StrategyRewarding, "evaluators", new_callable=mock.PropertyMock) as mock_prop:
            mock_prop.return_value = mock_evaluators
            points = strategy.execute()

        assert len(points) > 0

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_discount(self, mock_setup):
        strategy = StrategyRewarding(self.problem, parse_args=False)
        h = DummyHeur(strategy, "h1")
        h.performance = 1.0

        # default discount
        strategy.discount(h)
        assert h.performance == 0.95

        # Explicit discount
        h.performance = 1.0
        strategy.discount(h, discount=0.9)
        assert h.performance == 0.9

        # Bad discount fallback
        h.performance = 1.0
        strategy.discount(h, discount="bad")
        assert h.performance == 0.95
