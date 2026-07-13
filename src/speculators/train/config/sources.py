"""Source layering: how the four layers become one validated config.

This module owns the *mechanism* of :meth:`TrainConfig.from_sources`:

* loading the ``--config`` YAML and unwrapping its ``train:`` stage block
  (:func:`_load_yaml`, :func:`_unwrap_stage`) -- lenient on a legacy bare mapping,
  forward-compatible with sibling pipeline stages (``prepare_data:`` etc.);
* deciding which YAML leaves are recognised vs unknown (:func:`_yaml_leaf_dests`),
  warning on the latter;
* feeding the layers to pydantic-settings in precedence order and computing the
  matching typed :class:`~.provenance.Provenance` in the same pass;
* the post-resolution warning for an algorithm block that does not apply to the
  chosen ``speculator_type`` (:func:`warn_mismatched_algo_blocks`).

The draft-init conflict check and ``--set`` parsing live in :mod:`.cli` (they are
flag-centric); this module calls into them so the whole core is assembled here.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .provenance import Provenance
from .provenance import compute as compute_provenance
from .schema import _GROUPS, _ROOT_FIELDS, nest_flat

if TYPE_CHECKING:
    from .schema import TrainConfig


def build_from_sources(
    cls: type["TrainConfig"],
    *,
    cli: dict[str, Any],
    overrides: list[str],
    config_path: str | None,
    argv: list[str] | None,
) -> "TrainConfig":
    """Assemble, validate, and attach provenance to a :class:`TrainConfig`.

    The pure core behind :meth:`TrainConfig.from_sources`; see that method for the
    parameter contract. Raises :class:`~.errors.ConfigError` / pydantic
    ``ValidationError`` on bad input, never ``SystemExit``.
    """
    # Local import breaks the schema -> (this) -> cli import cycle at module load.
    from .cli import parse_set_overrides, validate_draft_init  # noqa: PLC0415

    set_values = parse_set_overrides(overrides)

    yaml_nested: dict[str, Any] = {}
    yaml_known: set[str] = set()
    if config_path:
        yaml_nested = _unwrap_stage(_load_yaml(config_path))
        yaml_known, unknown = _yaml_leaf_dests(yaml_nested)
        if unknown:
            warnings.warn(
                f"--config '{config_path}' has unrecognised keys (ignored): "
                f"{', '.join(sorted(unknown))}",
                stacklevel=2,
            )

    # Layers fed to pydantic-settings highest-precedence first: flag > --set >
    # YAML. Field defaults are pydantic's implicit lowest layer. The private
    # ``_layers`` kwarg is consumed in settings_customise_sources.
    cfg = cls(  # type: ignore[call-arg]
        _layers={
            "flag": nest_flat(cli),
            "set": nest_flat(set_values),
            "yaml": yaml_nested,
        }
    )

    # Provenance walks the SAME layers in the SAME order, so the winning layer per
    # dest cannot drift from the winning value.
    cfg._provenance = compute_provenance(
        flag=set(cli), overrides=set(set_values), yaml=yaml_known
    )
    cfg._argv = argv

    # Cross-field checks that need provenance (a conflict expressed only in YAML
    # or via --set is caught identically to one on the CLI).
    validate_draft_init(cfg)
    warn_mismatched_algo_blocks(cfg)
    return cfg


def _load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"--config file '{path}' must contain a top-level mapping.")
    return data


def _unwrap_stage(data: dict[str, Any]) -> dict[str, Any]:
    """Return the trainer config mapping, unwrapping the ``train:`` stage block.

    The canonical file is stage-shaped -- everything nests under ``train:`` so a
    future ``prepare_data:`` / ``launch_vllm:`` stage extends the file instead of
    replacing it (ADR 0001). Sibling stage keys are forward-compatible and ignored
    here. Loading stays lenient: a legacy bare top-level mapping (no ``train:``
    key, since no config group is named ``train``) is accepted unchanged.
    """
    stage = data.get("train")
    if isinstance(stage, dict):
        return stage
    return data


def _yaml_leaf_dests(yaml_dict: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return ``(known, unknown)`` flat dests present in a parsed YAML mapping.

    Group blocks contribute their leaf keys; root scalars contribute themselves.
    Anything not recognised is reported as ``unknown`` for a non-fatal warning.
    """
    known: set[str] = set()
    unknown: set[str] = set()
    for key, value in yaml_dict.items():
        if key in _GROUPS and isinstance(value, dict):
            # Validate each leaf against ITS group's fields, not the global dest
            # set: a real field placed under the wrong block is a mistake, so it
            # should surface as unrecognised (and stay out of the provenance set)
            # rather than being silently accepted because the name exists in some
            # other group.
            group_fields = _GROUPS[key].model_fields
            for leaf in value:
                (known if leaf in group_fields else unknown).add(leaf)
        elif key in _ROOT_FIELDS:
            known.add(key)
        else:
            unknown.add(key)
    return known, unknown


# Algorithm block -> the speculator_type(s) for which it applies.
_ALGO_BLOCK_TYPES: dict[str, set[str]] = {
    "dflash": {"dflash", "dspark"},
    "dspark": {"dspark"},
    "peagle": {"peagle"},
    "mtp": {"mtp"},
}


def warn_mismatched_algo_blocks(cfg: "TrainConfig") -> None:
    """Non-fatal heads-up when an inapplicable algo block holds a non-default value.

    Preserves the previous lenient behaviour (mismatched knobs are ignored, not
    rejected) while nudging the user about a likely mistake. We key off
    "differs from the coded default" rather than "present in the sources" so that
    a full ``--dump-config`` (which emits every block at its defaults) round-trips
    cleanly without spurious warnings.
    """
    for block, types_ in _ALGO_BLOCK_TYPES.items():
        if cfg.speculator_type in types_:
            continue
        group = getattr(cfg, block)
        defaults = _GROUPS[block]()  # algo blocks are fully defaulted
        touched = sorted(
            fname
            for fname in type(group).model_fields
            if getattr(group, fname) != getattr(defaults, fname)
        )
        if touched:
            warnings.warn(
                f"speculator_type='{cfg.speculator_type}' does not use the "
                f"'{block}' block; these non-default settings are ignored: "
                f"{', '.join(touched)}",
                stacklevel=3,
            )


# Provenance is re-exported for callers that want the value type without reaching
# into the provenance module directly.
__all__ = ["Provenance", "build_from_sources", "warn_mismatched_algo_blocks"]
