"""
Panobbgo Logging System
========================

Centralized logging infrastructure with component-based control.

.. codeauthor:: Panobbgo Development Team
"""

import logging
import sys
from typing import Dict, Optional, Any, List, Union

from .progress import ProgressReporter, ErrorReporter


class ComponentLogger:
    """
    Logger for individual components (Strategy, Heuristic, Analyzer, etc.).

    Provides component-specific enable/disable and level control.
    """

    def __init__(self, name: str, parent_logger: 'PanobbgoLogger'):
        self.name = name
        self.parent = parent_logger
        self._enabled = False
        self._level = logging.WARNING
        self._logger = logging.getLogger(f'panobbgo.{name}')

    @property
    def enabled(self) -> bool:
        """Whether logging is enabled for this component."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def level(self) -> int:
        """Current logging level."""
        return self._level

    @level.setter
    def level(self, value: Union[str, int]):
        if isinstance(value, str):
            self._level = getattr(logging, value.upper())
        elif isinstance(value, int):
            self._level = value
        else:
            raise ValueError(f"Invalid log level: {value}")

    def _should_log(self, level: int) -> bool:
        """Check if message should be logged."""
        return self._enabled and level >= self._level

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        if self._should_log(logging.DEBUG):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        if self._should_log(logging.INFO):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message (always shown)."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message (always shown)."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message (always shown)."""
        self._logger.critical(msg, *args, **kwargs)


class PanobbgoLogger:
    """
    Centralized logger for the panobbgo framework.

    Manages component loggers, progress reporting, and error handling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.component_loggers: Dict[str, ComponentLogger] = {}

        # Initialize reporters
        self.progress_reporter = ProgressReporter()
        self.error_reporter = ErrorReporter()

        # Set up default logging
        self._setup_default_logging()

        # Load configuration
        self._load_config()

    def _setup_default_logging(self):
        """Set up basic Python logging configuration."""
        # Remove any existing handlers
        root_logger = logging.getLogger('panobbgo')
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)  # Default to quiet

    def _load_config(self):
        """Load logging configuration."""
        # Default configuration
        defaults = {
            'default_level': 'WARNING',
            'enabled_components': [],
            'progress_enabled': True,
            'progress_symbols': True,
            'status_line_enabled': True,
            'status_update_frequency': 5,
            'always_show_errors': True,
            'always_show_warnings': True,
        }

        # Merge with provided config
        self.config = {**defaults, **self.config}

        # Apply configuration
        self.progress_reporter.enabled = self.config['progress_enabled']
        self.progress_reporter.use_symbols = self.config['progress_symbols']
        self.progress_reporter.status_enabled = self.config['status_line_enabled']
        self.progress_reporter.update_frequency = self.config['status_update_frequency']

        # Enable specified components
        for component in self.config['enabled_components']:
            level = self.config.get(f'{component}_level', 'INFO')
            self.enable_component(component, level)

    def get_logger(self, component_name: str) -> ComponentLogger:
        """
        Get or create a logger for a specific component.

        Args:
            component_name: Name of the component (e.g., 'strategy', 'splitter')

        Returns:
            ComponentLogger instance
        """
        if component_name not in self.component_loggers:
            self.component_loggers[component_name] = ComponentLogger(
                component_name, self
            )
        return self.component_loggers[component_name]

    def enable_component(self, component: str, level: str = 'INFO'):
        """
        Enable logging for a specific component.

        Args:
            component: Component name
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        logger = self.get_logger(component)
        logger.enabled = True
        logger.level = level

    def disable_component(self, component: str):
        """
        Disable logging for a specific component.

        Args:
            component: Component name
        """
        logger = self.get_logger(component)
        logger.enabled = False

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

    def set_quiet_mode(self):
        """Set quiet mode (errors and warnings only, no progress)."""
        self.disable_progress_reporting()
        self.disable_status_line()
        # Disable all component logging
        for logger in self.component_loggers.values():
            logger.enabled = False

    def set_verbose_mode(self, components: Optional["List[str]"] = None):
        """
        Set verbose mode for specified components.

        Args:
            components: List of component names to enable. If None, enables common ones.
        """
        if components is None:
            components = ['strategy', 'results', 'splitter']

        for component in components:
            self.enable_component(component, 'INFO')

        self.enable_progress_reporting()
        self.enable_status_line()