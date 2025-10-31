"""Clean logging utilities for data generation pipeline."""

import logging
import sys
from typing import Any

__all__ = ["PipelineLogger"]


class PipelineLogger:
    """Simple logger with clean output."""

    def __init__(self, name: str = ""):
        self.logger = logging.getLogger(name)
        self.use_colors = sys.stdout.isatty()

    def _color(self, text: str, code: str) -> str:
        """Apply ANSI color if terminal supports it."""
        return f"{code}{text}\033[0m" if self.use_colors else text

    def section(self, title: str):
        """Print a major section header."""
        line = "━" * 60
        blue_bold = "\033[1;34m"
        colored_line = self._color(line, blue_bold)
        colored_title = self._color(f"  {title}", blue_bold)
        self.logger.info("\n%s", colored_line)
        self.logger.info("%s", colored_title)
        self.logger.info("%s", colored_line)

    def subsection(self, title: str):
        """Print a subsection header."""
        bold = "\033[1m"
        self.logger.info("\n%s", self._color(f"▸ {title}", bold))

    def config(self, config_dict: dict[str, Any]):
        """Print configuration in aligned format."""
        if not config_dict:
            return
        dim = "\033[2m"
        max_key_len = max(len(str(k)) for k in config_dict)
        for key, value in config_dict.items():
            key_str = str(key).ljust(max_key_len)
            colored_key = self._color(key_str, dim)
            self.logger.info("  %s │ %s", colored_key, value)

    def info(self, message: str):
        """Print info message."""
        self.logger.info("  %s", message)

    def success(self, message: str):
        """Print success message."""
        green = "\033[92m"
        self.logger.info("  %s %s", self._color("✓", green), message)

    def warning(self, message: str):
        """Print warning message."""
        yellow = "\033[93m"
        self.logger.warning("  %s %s", self._color("⚠", yellow), message)

    def error(self, message: str):
        """Print error message."""
        red = "\033[91m"
        self.logger.error("  %s %s", self._color("✗", red), message)

    def debug(self, message: str):
        """Print debug message (dimmed)."""
        dim = "\033[2m"
        self.logger.debug("%s", self._color(f"  {message}", dim))
