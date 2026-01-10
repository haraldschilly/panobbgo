"""
Progress Reporting and Error Handling
======================================

Real-time progress reporting with symbols and status line updates.

Uses the Rich library for robust terminal handling.

.. codeauthor:: Panobbgo Development Team
"""

import sys
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from rich.live import Live
from rich.console import Console
from rich.text import Text
from rich.table import Table

from panobbgo.lib.lib import Result


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

    Uses Rich library for robust terminal handling with proper emoji support
    and automatic cursor management.
    """

    def __init__(self):
        self.enabled = True
        self.use_symbols = True
        self.status_enabled = True
        self.update_frequency = 5  # Update status every N evaluations

        self.evaluation_count = 0
        self.start_time = time.time()
        self.last_status_update = 0

        # Rich components
        self.console = Console(file=sys.stdout, force_terminal=None)
        self.progress_text = Text()  # Accumulates progress symbols
        self.status_text = ""
        self.live = None  # Live display (started on first use)

        # Check if terminal supports rich features
        self.supports_ansi = self.console.is_terminal

        # Fallback mode tracking
        self._fallback_line_position = 0  # Position on current line (for 40-char wrap)

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

    def _get_symbol_set(self) -> Dict[str, str]:
        """Get the appropriate symbol set."""
        return self.symbols if self.use_symbols else self.plain_symbols

    def _print_fallback_status(self):
        """Print status line in fallback mode (non-ANSI)."""
        if self.status_text and self.status_enabled:
            sys.stdout.write("\n" + self.status_text + "\n")
            sys.stdout.flush()
        else:
            # Just newline if no status yet
            sys.stdout.write("\n")
            sys.stdout.flush()

    # Backward compatibility properties
    @property
    def progress_line(self) -> str:
        """Get progress line as string (for backward compatibility)."""
        return self.progress_text.plain

    @property
    def status_line(self) -> str:
        """Get status line as string (for backward compatibility)."""
        return self.status_text

    @property
    def status_line_printed(self) -> bool:
        """Check if status line has been printed (for backward compatibility)."""
        return self.live is not None and len(self.status_text) > 0

    @status_line_printed.setter
    def status_line_printed(self, value: bool):
        """Set status line printed flag (for backward compatibility with tests)."""
        # This is a no-op for Rich implementation, but needed for test compatibility
        pass

    def _get_display_renderable(self):
        """Create the display renderable (progress + status)."""
        # Create a simple grid table with progress on top, status below
        table = Table.grid(padding=0)
        table.add_column()
        table.add_row(self.progress_text)
        if self.status_enabled and self.status_text:
            table.add_row(self.status_text)
        return table

    def _ensure_live_started(self):
        """Start the Rich Live display if not already started."""
        if self.live is None and self.enabled and self.supports_ansi:
            self.live = Live(
                self._get_display_renderable(),
                console=self.console,
                refresh_per_second=4,
                screen=False,  # Don't use alternate screen - keep in terminal history
                auto_refresh=True
            )
            self.live.start()

    def _update_display(self):
        """Update the Rich Live display."""
        if self.live is not None:
            self.live.update(self._get_display_renderable())
        elif not self.supports_ansi:
            # Fallback mode: print directly (no live updating)
            # This just accumulates output in terminal
            pass  # Output is handled in report_evaluation fallback

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

        # Check for failure (assuming Result has a failed attribute, if not fallback to fx check)
        failed = getattr(result, 'failed', False)
        if context.evaluation_failed or (result.fx is None and not failed):
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

        # Add symbol to progress text
        self.progress_text.append(symbol)
        self.evaluation_count += 1

        if self.supports_ansi:
            # ANSI mode: use Rich Live
            self._ensure_live_started()
            self._update_display()
        else:
            # Fallback mode: print directly with 40-char line wrapping
            sys.stdout.write(symbol)
            sys.stdout.flush()
            self._fallback_line_position += 1

            # Every 40 characters, print newline and status if available
            if self._fallback_line_position >= 40:
                self._print_fallback_status()
                self._fallback_line_position = 0



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
        if not self.status_enabled:
            return

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
        self.status_text = "  |  ".join(parts)
        if self.status_text:
            self.status_text = f"[ {self.status_text} ]"

        # Store max_evals for periodic updates
        self._max_evals = max_evals

        if self.supports_ansi:
            # ANSI mode: use Rich Live
            self._ensure_live_started()
            self._update_display()
        else:
            # Fallback mode: status is stored, will be printed at next 40-char wrap or finalize
            pass

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
        if value is None:
            return "None"
        if abs(value) < 1e-4 or abs(value) > 1e6:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"

    def finalize(self):
        """Finalize progress reporting and commit to terminal."""
        if self.live is not None:
            # ANSI mode: Stop the live display (this commits the current display to terminal)
            self.live.stop()
            self.live = None
            # Add final newline for clean separation from next output
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # Fallback mode: Print final status if any progress was made
            if self._fallback_line_position > 0 or self.status_text:
                # Print newline if we're mid-line
                if self._fallback_line_position > 0:
                    sys.stdout.write("\n")
                # Print final status
                if self.status_text and self.status_enabled:
                    sys.stdout.write(self.status_text + "\n")
                sys.stdout.write("\n")  # Final separation newline
                sys.stdout.flush()

    def reset(self):
        """Reset progress state."""
        # Stop live display if running
        if self.live is not None:
            self.live.stop()
            self.live = None

        # Reset state
        self.progress_text = Text()
        self.status_text = ""
        self.evaluation_count = 0
        self.start_time = time.time()
        self.last_status_update = 0
        self._fallback_line_position = 0


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