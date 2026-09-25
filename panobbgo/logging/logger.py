"""
Panobbgo Logging System
========================

Per-strategy progress reporting and the ``panobbgo`` logger's default handler.

.. codeauthor:: Panobbgo Development Team
"""

import logging
import sys
from typing import Dict, Optional, Any

from .progress import ProgressReporter


class PanobbgoLogger:
    """
    Per-strategy logging front end: owns the :class:`ProgressReporter`.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.progress_reporter = ProgressReporter()

        # Set up default logging
        self._setup_default_logging()

        # Load configuration
        self._load_config()

    def _setup_default_logging(self):
        """Give the ``panobbgo`` logger a stderr handler at WARNING, once per process.

        Every strategy builds a :class:`PanobbgoLogger`, so this must be
        idempotent and must not undo the caller's own setup: it does nothing
        when the ``panobbgo`` logger already has a handler (ours or the
        user's), and it only sets the level when none was configured.
        """
        root_logger = logging.getLogger("panobbgo")
        if root_logger.handlers:
            return
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(handler)
        if root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.WARNING)  # Default to quiet

    def _load_config(self):
        """Load logging configuration."""
        # Default configuration.  Progress and status defaults follow
        # stdout TTY detection so CI / piped logs stay quiet without
        # having to wire a config knob.  Pass an explicit value in the
        # config dict to override.
        _is_tty = sys.stdout.isatty()
        defaults = {
            "progress_enabled": _is_tty,
            "progress_symbols": True,
            "status_line_enabled": _is_tty,
            "status_update_frequency": 5,
        }

        # Merge with provided config
        self.config = {**defaults, **self.config}

        # Apply configuration
        self.progress_reporter.enabled = self.config["progress_enabled"]
        self.progress_reporter.use_symbols = self.config["progress_symbols"]
        self.progress_reporter.status_enabled = self.config["status_line_enabled"]
        self.progress_reporter.update_frequency = self.config["status_update_frequency"]

    def enable_progress_reporting(self, symbols: bool = True):
        """Enable progress reporting with optional emoji symbols."""
        self.progress_reporter.enabled = True
        self.progress_reporter.use_symbols = symbols

    def disable_progress_reporting(self):
        """Disable progress reporting."""
        self.progress_reporter.enabled = False

    def enable_status_line(self):
        """Enable status line updates."""
        self.progress_reporter.status_enabled = True

    def disable_status_line(self):
        """Disable status line updates."""
        self.progress_reporter.status_enabled = False
