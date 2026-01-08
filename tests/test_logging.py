"""
Basic tests for the logging infrastructure.

.. codeauthor:: Panobbgo Development Team
"""

import pytest
from unittest.mock import patch
import sys
from io import StringIO

from panobbgo.logging import PanobbgoLogger, ComponentLogger
from panobbgo.logging.progress import ProgressReporter, ProgressContext
from panobbgo.lib.lib import Result, Point
import numpy as np


class TestComponentLogger:
    """Test ComponentLogger functionality."""

    def test_initial_state(self):
        """Test initial logger state."""
        parent = PanobbgoLogger()
        logger = ComponentLogger('test', parent)

        assert not logger.enabled
        assert logger.level == 30  # WARNING
        assert logger.name == 'test'

    def test_enable_disable(self):
        """Test enabling and disabling logger."""
        parent = PanobbgoLogger()
        logger = ComponentLogger('test', parent)

        logger.enabled = True
        assert logger.enabled

        logger.enabled = False
        assert not logger.enabled

    def test_level_setting(self):
        """Test setting log levels."""
        parent = PanobbgoLogger()
        logger = ComponentLogger('test', parent)

        logger.level = 'DEBUG'
        assert logger.level == 10

        logger.level = 'INFO'
        assert logger.level == 20

        logger.level = 'WARNING'
        assert logger.level == 30

        logger.level = 40  # ERROR
        assert logger.level == 40

    def test_logging_when_disabled(self):
        """Test that logging doesn't output when disabled."""
        parent = PanobbgoLogger()
        logger = ComponentLogger('test', parent)

        # Should not log anything since disabled
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            logger.info("This should not appear")
            # Since disabled, no output expected
            assert mock_stderr.getvalue() == ""


class TestPanobbgoLogger:
    """Test PanobbgoLogger functionality."""

    def test_get_logger(self):
        """Test getting component loggers."""
        main_logger = PanobbgoLogger()

        logger1 = main_logger.get_logger('strategy')
        logger2 = main_logger.get_logger('strategy')

        assert logger1 is logger2  # Same instance
        assert logger1.name == 'strategy'

    def test_enable_component(self):
        """Test enabling components."""
        main_logger = PanobbgoLogger()

        main_logger.enable_component('strategy', 'DEBUG')

        strategy_logger = main_logger.get_logger('strategy')
        assert strategy_logger.enabled
        assert strategy_logger.level == 10  # DEBUG

    def test_quiet_mode(self):
        """Test quiet mode disables everything."""
        main_logger = PanobbgoLogger()

        # Enable some components first
        main_logger.enable_component('strategy')
        main_logger.enable_component('results')

        # Set quiet mode
        main_logger.set_quiet_mode()

        # Check components are disabled
        assert not main_logger.get_logger('strategy').enabled
        assert not main_logger.get_logger('results').enabled
        assert not main_logger.progress_reporter.enabled
        assert not main_logger.progress_reporter.status_enabled

    def test_verbose_mode(self):
        """Test verbose mode enables common components."""
        main_logger = PanobbgoLogger()

        main_logger.set_verbose_mode()

        # Check common components are enabled
        assert main_logger.get_logger('strategy').enabled
        assert main_logger.get_logger('results').enabled
        assert main_logger.get_logger('splitter').enabled
        assert main_logger.progress_reporter.enabled
        assert main_logger.progress_reporter.status_enabled


class TestProgressReporter:
    """Test ProgressReporter functionality."""

    def test_initial_state(self):
        """Test initial progress reporter state."""
        reporter = ProgressReporter()

        assert reporter.enabled
        assert reporter.use_symbols
        assert reporter.status_enabled
        assert reporter.evaluation_count == 0

    def test_get_progress_symbol(self):
        """Test progress symbol selection."""
        reporter = ProgressReporter()
        result = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, None, None, False)

        # Normal evaluation
        context = ProgressContext()
        symbol = reporter.get_progress_symbol(result, context)
        assert symbol == '.'  # normal

        # Major improvement
        context = ProgressContext(is_global_best=True)
        symbol = reporter.get_progress_symbol(result, context)
        assert symbol == '🎉'  # major improvement

        # Failed evaluation
        context = ProgressContext(evaluation_failed=True)
        symbol = reporter.get_progress_symbol(result, context)
        assert symbol == '❌'  # failed

    def test_plain_symbols(self):
        """Test plain text symbols."""
        reporter = ProgressReporter()
        reporter.use_symbols = False

        result = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, None, None, False)
        context = ProgressContext(is_global_best=True)

        symbol = reporter.get_progress_symbol(result, context)
        assert symbol == '!'  # plain major improvement

    @patch('builtins.print')
    def test_report_evaluation(self, mock_print):
        """Test evaluation reporting."""
        reporter = ProgressReporter()
        result = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, None, None, False)
        context = ProgressContext()

        reporter.report_evaluation(result, context)

        # Should print the symbol
        mock_print.assert_called_with('.', end='', flush=True)
        assert reporter.evaluation_count == 1

    @patch('builtins.print')
    def test_status_update(self, mock_print):
        """Test status line updates."""
        reporter = ProgressReporter()
        reporter.status_line_printed = True  # Simulate that status line exists

        reporter.update_status(
            budget_pct=50.0,
            eta_seconds=300,
            convergence=75.0,
            best_value=1.23,
            current_evals=250,
            max_evals=500
        )

        # Should have called various cursor operations and status printing
        assert mock_print.called
        # Check that the status line contains the expected content
        call_args_str = str(mock_print.call_args_list)
        assert "Evals: 50% (250/500)" in call_args_str
        assert "ETA: 5m 0s" in call_args_str
        assert "Best: 1.2300" in call_args_str