#!/usr/bin/env python3
"""
Panobbgo Logging System Demo
Shows the new logging display in action

Run with: uv run python logging_demo.py

Note: The core functionality demonstrated here is thoroughly tested in
tests/test_logging.py, including:
- ANSI mode output with cursor positioning
- Fallback mode for non-TTY environments
- Complete workflow with multiple evaluation types
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from panobbgo.logging import PanobbgoLogger
from panobbgo.lib.lib import Result, Point
from panobbgo.logging.progress import ProgressContext
import time


def demo_logging():
    """Demonstrate the logging system display."""
    print("🎯 Panobbgo Logging System Demo")
    print("=" * 50)
    print()

    # Create logger with progress enabled
    logger = PanobbgoLogger()
    logger.enable_progress_reporting(symbols=True)
    logger.enable_status_line()

    # Show terminal mode
    mode = "ANSI (cursor positioning)" if logger.progress_reporter.supports_ansi else "Simple (no ANSI)"
    print(f"Terminal mode: {mode}")
    print()
    print("Watch the progress line grow and status line update below:")
    print()

    # Simulate evaluations with different outcomes
    evaluations = [
        ("normal", ProgressContext()),
        ("normal", ProgressContext()),
        ("improvement", ProgressContext(is_improvement=True)),
        ("normal", ProgressContext()),
        ("significant", ProgressContext(is_significant_improvement=True)),
        ("learning", ProgressContext(new_region_created=True)),
        ("normal", ProgressContext()),
        ("major", ProgressContext(is_global_best=True)),
        ("normal", ProgressContext()),
        ("warning", ProgressContext(has_warnings=True)),
        ("normal", ProgressContext()),
        ("improvement", ProgressContext(is_improvement=True)),
    ]

    for i, (eval_type, context) in enumerate(evaluations):
        # Create fake evaluation result
        x = [0.1 * (i % 5), 0.1 * ((i + 1) % 5)]
        result = Result(Point(x, f"eval_{i}"), float(i) * 0.05, None, None, False)

        # Report evaluation (prints progress symbol)
        logger.progress_reporter.report_evaluation(result, context)

        # Update status every 4 evaluations
        if (i + 1) % 4 == 0:
            progress_pct = ((i + 1) / len(evaluations)) * 100

            logger.progress_reporter.update_status(
                budget_pct=progress_pct,
                eta_seconds=max(0, len(evaluations) - i - 1),
                convergence=min(100, progress_pct),
                best_value=0.01,
                current_evals=i + 1,
                max_evals=len(evaluations)
            )

        time.sleep(0.3)  # Visual delay

    # Finalize the progress display
    logger.progress_reporter.finalize()

    print()
    print("✅ Demo complete!")
    print()
    print("You saw:")
    print("• Progress symbols accumulating on one line")
    print("• Status line updating on the line below")
    print("• Format: [ Evals: XX% (N/MAX) | ETA: Xs | Convergence: XX% | Best: X.XXX ]")


if __name__ == "__main__":
    demo_logging()