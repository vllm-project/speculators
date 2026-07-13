"""Config-file-first configuration for ``scripts/train.py``.

The subsystem exposes essentially one public type, :class:`TrainConfig` (ADR
0004): it parses itself, layers its sources (``flag > --set > YAML > default``),
validates itself, reports its own provenance, and serializes itself.
``scripts/train.py`` collapses to ``main(TrainConfig.resolve())``.

The package separates the schema (the "what" -- :mod:`.schema`) from the
mechanism (:mod:`.cli`, :mod:`.sources`, :mod:`.provenance`, :mod:`.adapter`,
:mod:`.artifacts`), so the whole tunable surface reads in one place and no single
module is a god-object. A handful of schema-derived helpers are re-exported here
for callers (and tests) that generate or introspect the CLI surface directly.
"""

from .cli import DECODER_SHAPING_FLAGS, add_config_cli_arguments, decoder_shaping_flags
from .errors import ConfigError
from .schema import CONFIG_DESTS, TrainConfig

__all__ = [
    "CONFIG_DESTS",
    "DECODER_SHAPING_FLAGS",
    "ConfigError",
    "TrainConfig",
    "add_config_cli_arguments",
    "decoder_shaping_flags",
]
