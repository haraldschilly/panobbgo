# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from panobbgo.utils import PanobbgoTestCase
from panobbgo.core import StrategyBase, Results, EventBus, Event, Module, StopHeuristic, Heuristic
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock

class TestCoreStrategyBase(PanobbgoTestCase):
    def test_validate_config(self):
        strategy = StrategyBase(self.problem, parse_args=False)
        strategy.config.max_eval = -1
        errors = strategy._validate_config()
        assert any("max_eval must be positive" in e for e in errors)

        strategy.config.max_eval = 200000
        errors = strategy._validate_config()
        assert any("seems unreasonably high" in e for e in errors)

        strategy.config.max_eval = "abc"
        errors = strategy._validate_config()
        assert any("must be a valid integer" in e for e in errors)

        strategy.config.discount = 1.5
        errors = strategy._validate_config()
        assert any("discount must be between 0 and 1" in e for e in errors)

        strategy.config.discount = "abc"
        errors = strategy._validate_config()
        assert any("discount must be a valid float" in e for e in errors)

        strategy.config.smooth = -0.5
        errors = strategy._validate_config()
        assert any("smooth must be non-negative" in e for e in errors)

        strategy.config.smooth = "abc"
        errors = strategy._validate_config()
        assert any("smooth must be a valid float" in e for e in errors)

        strategy.config.evaluation_method = "invalid"
        errors = strategy._validate_config()
        assert any("evaluation_method must be one of" in e for e in errors)

        # Restore configuration state to not affect other tests
        strategy.config.max_eval = 1000
        strategy.config.discount = 0.95
        strategy.config.smooth = 0.5
        strategy.config.evaluation_method = "threaded"

class TestCoreResults(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_load_from_storage(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        strategy.config.storage_backend = "sqlite"
        strategy.config.storage_uri = ":memory:"

        results = Results(strategy)

        # mock storage loading
        mock_backend = mock.MagicMock()
        mock_backend.load.return_value = [Result(Point(np.zeros(2), "test"), 0.0)]
        results.backend = mock_backend

        count = results.load_from_storage()
        assert count == 1

        mock_backend.load.return_value = None
        count = results.load_from_storage()
        assert count == 0

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_get_history(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        results = Results(strategy)

        # Empty
        hist = results.get_history()
        assert len(hist['x']) == 0

        # Add data
        res1 = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, cv_vec=np.array([0.1]))
        res2 = Result(Point(np.array([3.0, 4.0]), "test"), 2.0, cv_vec=np.array([0.0]))
        results.add_results([res1, res2])

        hist = results.get_history()
        assert len(hist['x']) == 2
        assert np.array_equal(hist['x'][0], [1.0, 2.0])
        assert np.array_equal(hist['cv_vec'][0], [0.1])

        hist = results.get_history(n=1)
        assert len(hist['x']) == 1
        assert np.array_equal(hist['x'][0], [3.0, 4.0])

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_flush_buffer_edge_cases(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        results = Results(strategy)

        # Empty buffer flush
        results._buffer = []
        results._flush_buffer()
        assert results._results_df is None

        # Buffer without cv_vec
        res = Result(Point(np.array([1.0]), "test"), 5.0, cv_vec=None)
        results._buffer = [res]
        results._flush_buffer()
        assert results._results_df is not None

class TestCoreEventBus(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_publish_no_subscribers(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        eb = EventBus(strategy.config)

        # Should not crash
        eb.publish("unknown_event")
        eb.publish("unknown_event", event=Event())
