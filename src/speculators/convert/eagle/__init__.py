"""
Eagle checkpoint conversion utilities.
"""

from speculators.convert.eagle.cli import app as eagle_cli_app
from speculators.convert.eagle.eagle_converter import EagleConverter

__all__ = ["EagleConverter", "eagle_cli_app"]
