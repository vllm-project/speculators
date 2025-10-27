"""Clean logging utilities for data generation pipeline."""

import logging
from typing import Dict, Any


class PipelineLogger:
    """Logger with clean formatting for pipeline stages."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def section(self, title: str):
        """Log a major section header."""
        self.logger.info(f"\n{'─' * 80}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{'─' * 80}")

    def subsection(self, title: str):
        """Log a subsection header."""
        self.logger.info(f"\n── {title}")

    def config(self, config_dict: Dict[str, Any]):
        """Log configuration in a clean format."""
        self.logger.info("Configuration:")
        max_key_len = max(len(str(k)) for k in config_dict.keys())
        for key, value in config_dict.items():
            self.logger.info(f"  {str(key).ljust(max_key_len)} : {value}")

    def info(self, message: str):
        """Standard info logging (passthrough)."""
        self.logger.info(message)

    def debug(self, message: str):
        """Standard debug logging (passthrough)."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Standard warning logging (passthrough)."""
        self.logger.warning(message)
