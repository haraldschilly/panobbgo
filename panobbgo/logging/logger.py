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
        idempotent and must not undo the caller's own setup.  It does nothing
        when the ``panobbgo`` logger **or the root logger** already has a
        handler: an application that configured logging (e.g.
        ``logging.basicConfig(level=logging.INFO)``) before building a
        strategy keeps full control — no second stderr handler that would
        print every warning twice, and no WARNING level on ``panobbgo`` that
        would swallow its INFO records.  Only an unconfigured process gets
        the fallback handler, and the level is set only when none was.
        Logging configured *after* the first strategy was built still sees
        the fallback handler; remove it with
        ``logging.getLogger("panobbgo").handlers.clear()``.
        """
        lib_logger = logging.getLogger("panobbgo")
        if lib_logger.handlers or logging.getLogger().handlers:
            return
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        lib_logger.addHandler(handler)
        if lib_logger.level == logging.NOTSET:
            lib_logger.setLevel(logging.WARNING)  # Default to quiet

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
        }

        # Merge with provided config
        self.config = {**defaults, **self.config}

        # Apply configuration
        self.progress_reporter.enabled = self.config["progress_enabled"]
        self.progress_reporter.use_symbols = self.config["progress_symbols"]
        self.progress_reporter.status_enabled = self.config["status_line_enabled"]
        # ("status_update_frequency" was accepted here but never read; the
        # status line is refreshed at most every 0.1 s by the strategy.)

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
