"""
Progress Reporting and Error Handling
======================================

Real-time progress reporting with symbols and status line updates.

.. codeauthor:: Panobbgo Development Team
"""

import sys
import time
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass

from panobbgo.lib.lib import Result


class CursorManager:
    """Helper class for managing terminal cursor position."""

    @staticmethod
    def save_position():
        """Save current cursor position."""
        print("\033[s", end="", flush=True)

    @staticmethod
    def restore_position():
        """Restore saved cursor position."""
        print("\033[u", end="", flush=True)

    @staticmethod
    def move_to_line(line_offset: int):
        """Move cursor to a specific line relative to current position."""
        if line_offset > 0:
            print(f"\033[{line_offset}B", end="", flush=True)
        elif line_offset < 0:
            print(f"\033[{-line_offset}A", end="", flush=True)

    @staticmethod
    def move_to_column(col: int):
        """Move cursor to a specific column."""
        print(f"\033[{col}G", end="", flush=True)

    @staticmethod
    def clear_line():
        """Clear the current line."""
        print("\033[2K", end="", flush=True)

    @staticmethod
    def clear_to_end_of_line():
        """Clear from cursor to end of line."""
        print("\033[K", end="", flush=True)


@dataclass
class ProgressContext:
    """Context information for progress symbol selection."""
    is_global_best: bool = False
    is_significant_improvement: bool = False
    is_improvement: bool = False
    new_region_created: bool = False
    analyzer_learned: bool = False
    evaluation_failed: bool = False
    has_warnings: bool = False


class ProgressReporter:
    """
    Handles real-time progress reporting during optimization.

    Provides character-by-character progress updates and status line.
    """

    def __init__(self):
        self.enabled = True
        self.use_symbols = True
        self.status_enabled = True
        self.update_frequency = 5  # Update status every N evaluations

        self.progress_line = ""
        self.status_line = ""
        self.evaluation_count = 0
        self.start_time = time.time()
        self.last_status_update = 0

        # Terminal state tracking - we keep cursor on progress line always
        self.status_line_printed = False

        # Check if we can use ANSI codes (need a real TTY)
        self.supports_ansi = sys.stdout.isatty()

        # Symbol sets
        self.symbols = {
            'major_improvement': '🎉',
            'significant_improvement': '🎊',
            'improvement': '⭐',
            'new_learning': '🆕',
            'normal': '.',
            'warning': '⚠️',
            'failed': '❌',
            'fatal': '💀'
        }

        self.plain_symbols = {
            'major_improvement': '!',
            'significant_improvement': '+',
            'improvement': '*',
            'new_learning': 'L',
            'normal': '.',
            'warning': 'W',
            'failed': 'X',
            'fatal': 'F'
        }

    def _get_terminal_width(self) -> int:
        """Get terminal width, defaulting to 80."""
        try:
            return shutil.get_terminal_size().columns
        except (OSError, AttributeError):
            return 80

    def _get_symbol_set(self) -> Dict[str, str]:
        """Get the appropriate symbol set."""
        return self.symbols if self.use_symbols else self.plain_symbols

    def get_progress_symbol(self, result: Result, context: ProgressContext) -> str:
        """
        Determine progress symbol based on evaluation outcome.

        Args:
            result: Evaluation result
            context: Additional context information

        Returns:
            Progress symbol character
        """
        symbols = self._get_symbol_set()

        if context.evaluation_failed or (result.fx is None and not result.failed):
            return symbols['failed']
        elif context.has_warnings:
            return symbols['warning']
        elif context.is_global_best:
            return symbols['major_improvement']
        elif context.is_significant_improvement:
            return symbols['significant_improvement']
        elif context.is_improvement:
            return symbols['improvement']
        elif context.new_region_created or context.analyzer_learned:
            return symbols['new_learning']
        else:
            return symbols['normal']

    def report_evaluation(self, result: Result, context: Optional[ProgressContext] = None):
        """
        Report a single evaluation result.

        Args:
            result: Evaluation result
            context: Additional context (optional)
        """
        if not self.enabled:
            return

        if context is None:
            context = ProgressContext()

        # Get progress symbol
        symbol = self.get_progress_symbol(result, context)

        # Add to progress line
        self.progress_line += symbol
        self.evaluation_count += 1

        # Print symbol
        self._print_progress_symbol(symbol)

    def _print_progress_symbol(self, symbol: str):
        """
        Print a progress symbol on the progress line.

        Args:
            symbol: The symbol to print

        Strategy:
        - Progress line is always the "current" line where the cursor lives
        - When status exists, it's on the line below
        - To add a symbol: just write it (cursor is already at end of progress line)
        """
        # Update our internal state
        self.progress_line += symbol

        if self.supports_ansi:
            # Simple: just write the symbol (cursor is at end of progress line)
            sys.stdout.write(symbol)
            sys.stdout.flush()
        else:
            # Fallback: just print symbols inline
            sys.stdout.write(symbol)
            sys.stdout.flush()

    def _print_status_line(self):
        """Print the status line below the progress line.

        Strategy:
        - Cursor is currently at the end of the progress line
        - We need to go down one line, update the status, and return to progress line
        - Use save/restore cursor position or explicit positioning
        """
        if self.supports_ansi:
            # Save current cursor position (end of progress line)
            sys.stdout.write("\x1b7")  # Save cursor (ESC 7)

            # Go to next line and clear it
            sys.stdout.write("\n\r\x1b[K")

            # Print status line
            sys.stdout.write(self.status_line)

            # Restore cursor position (back to end of progress line)
            sys.stdout.write("\x1b8")  # Restore cursor (ESC 8)

            sys.stdout.flush()
            self.status_line_printed = True
        else:
            # Fallback: print status on new lines
            sys.stdout.write("\n" + self.status_line + "\n")
            sys.stdout.flush()
            self.status_line_printed = True



    def update_status(self, budget_pct: float, eta_seconds: int,
                     convergence: float, best_value: float,
                     current_evals: int, max_evals: int,
                     extra_fields: Optional[Dict[str, Any]] = None):
        """
        Update status line with complete information.

        Args:
            budget_pct: Percentage of budget used (0-100)
            eta_seconds: Estimated seconds remaining
            convergence: Convergence percentage (0-100)
            best_value: Current best function value
            current_evals: Number of evaluations completed
            max_evals: Total evaluation budget
            extra_fields: Additional strategy-specific fields
        """
        # Build status line
        parts = []

        # Evaluations progress
        evals_str = f"Evals: {budget_pct:.0f}%"
        if current_evals is not None and max_evals is not None:
            evals_str += f" ({current_evals}/{max_evals})"
        parts.append(evals_str)

        # ETA
        eta_str = self._format_eta(eta_seconds)
        parts.append(f"ETA: {eta_str}")

        # Convergence
        parts.append(f"Convergence: {convergence:.0f}%")

        # Best value
        parts.append(f"Best: {self._format_value(best_value)}")

        # Extra fields from strategy
        if extra_fields:
            for key, value in extra_fields.items():
                if isinstance(value, float):
                    parts.append(f"{key}: {value:.1f}")
                else:
                    parts.append(f"{key}: {value}")

        # Create status line
        self.status_line = "  |  ".join(parts)
        if self.status_line:
            self.status_line = f"[ {self.status_line} ]"

        # Store max_evals for periodic updates
        self._max_evals = max_evals

        # Position cursor and print status line
        self._print_status_line()

    def _format_eta(self, seconds: int) -> str:
        """Format ETA in human-readable form."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def _format_value(self, value: float) -> str:
        """Format numerical value appropriately."""
        if abs(value) < 1e-4 or abs(value) > 1e6:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"

    def finalize(self):
        """Finalize progress reporting (move to clean state)."""
        # Move past the status line to a clean new line
        if self.status_line_printed:
            sys.stdout.write("\n")  # Move past status line
        sys.stdout.write("\n")  # Extra line for clean separation
        sys.stdout.flush()

    def reset(self):
        """Reset progress state."""
        self.progress_line = ""
        self.status_line = ""
        self.evaluation_count = 0
        self.start_time = time.time()
        self.last_status_update = 0
        self.status_line_printed = False


class ErrorReporter:
    """
    Handles error reporting and logging.
    """

    def __init__(self):
        self.error_count = 0
        self.warning_count = 0

    def report_error(self, component: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Report an error.

        Args:
            component: Component where error occurred
            error: Exception object
            context: Additional context information
        """
        self.error_count += 1
        error_msg = f"ERROR in {component}: {error}"
        if context:
            error_msg += f" (context: {context})"

        print(error_msg, file=sys.stderr)

    def report_warning(self, component: str, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Report a warning.

        Args:
            component: Component where warning occurred
            message: Warning message
            context: Additional context information
        """
        self.warning_count += 1
        warning_msg = f"WARNING in {component}: {message}"
        if context:
            warning_msg += f" (context: {context})"

        print(warning_msg, file=sys.stderr)

    def get_summary(self) -> Dict[str, int]:
        """Get error/warning summary."""
        return {
            'errors': self.error_count,
            'warnings': self.warning_count
        }