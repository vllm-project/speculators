"""Config-file-first configuration for ``scripts/train.py``.

One public type, :class:`~.schema.TrainConfig`, split across ``schema`` (fields),
``resolution`` (CLI + precedence), and ``artifacts`` (reproducibility I/O).
"""

from .schema import TrainConfig

__all__ = ["TrainConfig"]
