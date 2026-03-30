"""FastMTP speculator model."""

from .checkpoint import SKIP_KEYS, filter_mtp_keys, update_weight_index
from .config import FastMTPConfig
from .core import FastMTPSpeculator

__all__ = [
    "SKIP_KEYS",
    "FastMTPConfig",
    "FastMTPSpeculator",
    "filter_mtp_keys",
    "update_weight_index",
]
