"""FastMTP speculator model."""

from .checkpoint import SKIP_KEYS, remap_key, remap_keys, update_weight_index
from .config import FastMTPConfig
from .core import FastMTPSpeculator

__all__ = [
    "SKIP_KEYS",
    "FastMTPConfig",
    "FastMTPSpeculator",
    "remap_key",
    "remap_keys",
    "update_weight_index",
]
