"""The phase-1 anti-corruption seam between the typed config and the model layer.

The config subsystem resolves CLI + ``--set`` + YAML into typed pydantic groups.
The five ``SpeculatorModel`` classes still consume a flat ``vars(args)``-shaped
dict via ``**kwargs`` (ADR 0002). This module is the single, documented, one-way
adapter between the two: :func:`flatten` turns the typed config into that flat
dict; :func:`from_flat` recovers the typed view. Both preserve the schema's
declaration order so the flat dict is byte-identical to the pre-refactor parser's
``vars(args)`` (guarded by the golden equivalence test).

The named exit is phase 2 (a separate PR): push the typed groups into the model
layer and delete this module. Do not "simplify" the round-trip away meanwhile.
"""

from typing import TYPE_CHECKING, Any

from .schema import _GROUPS, _ROOT_FIELDS, CONFIG_DESTS, nest_flat

if TYPE_CHECKING:
    from .schema import TrainConfig


def flatten(cfg: "TrainConfig") -> dict[str, Any]:
    """Flatten a resolved config back into the ``vars(args)``-shaped dict."""
    flat: dict[str, Any] = {}
    for field in _ROOT_FIELDS:
        flat[field] = getattr(cfg, field)
    for gname in _GROUPS:
        group = getattr(cfg, gname)
        for fname in type(group).model_fields:
            flat[fname] = getattr(group, fname)
    return flat


def from_flat(cls: type["TrainConfig"], flat: dict[str, Any]) -> "TrainConfig":
    """Rebuild a resolved :class:`TrainConfig` from a flat working-dict.

    Used to recover the grouped config from the flattened namespace the training
    script threads around. Non-config keys (e.g. ``config``, ``dump_config``) are
    dropped; the resolution validators are idempotent on already-resolved values.
    """
    known = {k: v for k, v in flat.items() if k in CONFIG_DESTS}
    return cls(_layers={"flag": nest_flat(known)})  # type: ignore[call-arg]
