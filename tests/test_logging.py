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
from panobbgo.lib import Result, Point
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

    def test_report_evaluation(self):
        """Test evaluation reporting."""
        mock_stdout = StringIO()

        with patch('sys.stdout', mock_stdout):
            reporter = ProgressReporter()
            result = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, None, None, False)
            context = ProgressContext()

            reporter.report_evaluation(result, context)

            # Should have written the symbol
            output = mock_stdout.getvalue()
            assert '.' in output
            assert reporter.evaluation_count == 1

    def test_status_update(self):
        """Test status line updates."""
        mock_stdout = StringIO()

        with patch('sys.stdout', mock_stdout):
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

            # Check that the status text contains the expected content
            # (In fallback mode, status is stored but not printed until 40-char wrap or finalize)
            status = reporter.status_text
            assert "Evals: 50% (250/500)" in status
            assert "ETA: 5m 0s" in status
            assert "Best: 1.2300" in status


class TestProgressReporterIntegration:
    """Integration tests for the complete progress reporting workflow."""

    def test_ansi_mode_output_sequence(self):
        """Test complete output sequence in ANSI mode with cursor positioning."""
        # Mock stdout to capture raw output including ANSI codes
        mock_stdout = StringIO()

        # Patch both stdout and isatty
        with patch('sys.stdout', mock_stdout):
            # Manually set isatty to return True since we're mocking stdout
            mock_stdout.isatty = lambda: True

            # Create reporter
            reporter = ProgressReporter()
            assert reporter.supports_ansi  # Should detect TTY

            result1 = Result(Point(np.array([1.0, 2.0]), "test1"), 5.0, None, None, False)
            result2 = Result(Point(np.array([1.5, 2.5]), "test2"), 4.0, None, None, False)
            result3 = Result(Point(np.array([2.0, 3.0]), "test3"), 3.0, None, None, False)

            # Report some evaluations
            reporter.report_evaluation(result1, ProgressContext())
            reporter.report_evaluation(result2, ProgressContext(is_improvement=True))

            # Update status first time (creates status line)
            reporter.update_status(
                budget_pct=50.0,
                eta_seconds=10,
                convergence=50.0,
                best_value=0.123,
                current_evals=2,
                max_evals=4
            )

            # Report another evaluation (cursor should return to progress line)
            reporter.report_evaluation(result3, ProgressContext())

            # Update status second time (should use clear code)
            reporter.update_status(
                budget_pct=75.0,
                eta_seconds=5,
                convergence=75.0,
                best_value=0.123,
                current_evals=3,
                max_evals=4
            )

            # Get the captured output
            output = mock_stdout.getvalue()

            # Verify progress symbols are present in the progress_text attribute
            # (Rich uses Live display, so raw output contains ANSI control codes)
            assert '.' in reporter.progress_line  # normal evaluation
            assert '⭐' in reporter.progress_line  # improvement

            # Verify ANSI escape sequences are present (Rich uses its own codes)
            assert '\x1b[' in output  # Contains ANSI escape sequences

            # Verify status line content in the status_text attribute
            assert 'Evals: 50% (2/4)' in reporter.status_text or 'Evals: 75% (3/4)' in reporter.status_text
            assert 'ETA:' in reporter.status_text
            assert 'Best: 0.1230' in reporter.status_text
            assert 'Convergence:' in reporter.status_text

    def test_fallback_mode_output(self):
        """Test output in fallback mode (no ANSI support)."""
        mock_stdout = StringIO()

        with patch('sys.stdout', mock_stdout):
            # Make isatty return False to trigger fallback mode
            mock_stdout.isatty = lambda: False

            reporter = ProgressReporter()
            assert not reporter.supports_ansi  # Should detect non-TTY

            result1 = Result(Point(np.array([1.0, 2.0]), "test1"), 5.0, None, None, False)
            result2 = Result(Point(np.array([1.5, 2.5]), "test2"), 4.0, None, None, False)

            # Report evaluations
            reporter.report_evaluation(result1, ProgressContext())
            reporter.report_evaluation(result2, ProgressContext(is_improvement=True))

            # Update status
            reporter.update_status(
                budget_pct=50.0,
                eta_seconds=10,
                convergence=50.0,
                best_value=0.123,
                current_evals=2,
                max_evals=4
            )

            # In fallback mode, status is stored but not printed until 40-char wrap or finalize
            # So let's finalize to get the status output
            reporter.finalize()

            output = mock_stdout.getvalue()

            # Verify progress symbols
            assert '.' in output
            assert '⭐' in output

            # Verify NO ANSI escape sequences in fallback mode
            assert '\x1b7' not in output
            assert '\x1b8' not in output

            # Verify status content is present after finalize
            assert 'Evals: 50% (2/4)' in output
            assert 'ETA: 10s' in output

    def test_complete_demo_workflow(self):
        """Test the complete demo workflow similar to logging_demo.py."""
        mock_stdout = StringIO()

        with patch('sys.stdout', mock_stdout):
            mock_stdout.isatty = lambda: True

            # Create logger
            logger = PanobbgoLogger()
            logger.enable_progress_reporting(symbols=True)
            logger.enable_status_line()

            # Simulate a series of evaluations like the demo
            evaluations = [
                ProgressContext(),  # normal
                ProgressContext(),  # normal
                ProgressContext(is_improvement=True),  # improvement
                ProgressContext(),  # normal
                ProgressContext(is_significant_improvement=True),  # significant
                ProgressContext(new_region_created=True),  # learning
                ProgressContext(),  # normal
                ProgressContext(is_global_best=True),  # major
            ]

            for i, context in enumerate(evaluations):
                x = [0.1 * i, 0.2 * i]
                result = Result(Point(x, f"eval_{i}"), float(i) * 0.1, None, None, False)
                logger.progress_reporter.report_evaluation(result, context)

                # Update status every 4 evaluations
                if (i + 1) % 4 == 0:
                    logger.progress_reporter.update_status(
                        budget_pct=((i + 1) / len(evaluations)) * 100,
                        eta_seconds=len(evaluations) - i - 1,
                        convergence=min(100, ((i + 1) / len(evaluations)) * 100),
                        best_value=0.01,
                        current_evals=i + 1,
                        max_evals=len(evaluations)
                    )

            output = mock_stdout.getvalue()
            progress = logger.progress_reporter.progress_line

            # Verify all expected symbols appear in progress line
            assert '.' in progress  # normal (appears multiple times)
            assert '⭐' in progress  # improvement
            assert '🎊' in progress  # significant improvement
            assert '🆕' in progress  # learning
            assert '🎉' in progress  # major improvement

            # Verify status was updated (check the reporter's status_text)
            status = logger.progress_reporter.status_text
            assert 'Evals:' in status
            assert 'ETA:' in status
            assert 'Convergence:' in status
            assert 'Best: 0.0100' in status

            # Verify ANSI codes are present (Rich uses its own control sequences)
            assert '\x1b[' in output  # Contains ANSI escape sequences