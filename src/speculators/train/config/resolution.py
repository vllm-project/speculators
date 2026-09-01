"""Turns the schema fields into an argparse CLI and applies precedence.

Precedence is ``flag > yaml > default`` via pydantic-settings source ordering;
``resolve`` turns any config failure into a clean ``SystemExit(2)``.
"""

import argparse
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from hs_connectors import HiddenStatesBackend
from speculators.train.config.schema import (
    _GROUPS,
    _ROOT_FIELDS,
    CONFIG_DESTS,
    TrainConfig,
    nest_flat,
)

__all__ = [
    "DECODER_SHAPING_FLAGS",
    "ConfigError",
    "add_config_cli_arguments",
    "build_from_sources",
    "build_parser",
    "resolve",
]


class ConfigError(ValueError):
    """A user-facing configuration error (e.g. a draft-init conflict).

    Subclasses ``ValueError`` so the pure core can raise it for a unit test to
    catch, while the :func:`resolve` boundary renders it as a clean
    ``SystemExit(2)`` instead of a traceback.
    """


def _dest_to_flag(dest: str) -> str:
    """The argparse flag string for a config dest (``muon_lr`` -> ``--muon-lr``)."""
    return "--" + dest.replace("_", "-")


# Decoder-shaping dests -- the flags that synthesize the draft decoder and so
# conflict with --from-pretrained / --draft-config (each of which fully defines
# the draft). Derived from the schema so the set stays in one place.
# dest -> CLI flag; the flag strings are used only for human-readable ConfigError
# messages.
DECODER_SHAPING_FLAGS: dict[str, str] = {
    dest: _dest_to_flag(dest)
    for dest in (
        "num_layers",
        "draft_arch",
        "draft_hidden_act",
        "sliding_window",
        "full_attention_indices",
    )
}


def _required_flags() -> dict[str, str]:
    """The dests marked ``cli_required`` in the schema, mapped to their flags.

    Scans the root scalars and every group for the ``cli_required`` marker so the
    required contract stays schema-driven -- adding another required field needs no
    change here. Enforced post-build (see :func:`build_from_sources`), not by
    argparse, so a value supplied only in ``--config`` still satisfies it.
    """
    flags: dict[str, str] = {}
    fields = dict(TrainConfig.model_fields)
    for gmodel in _GROUPS.values():
        fields.update(gmodel.model_fields)
    for dest, field in fields.items():
        extra = field.json_schema_extra
        if isinstance(extra, dict) and extra.get("cli_required"):
            flags[dest] = _dest_to_flag(dest)
    return flags


# Flat dests the schema marks required (``cli_required``), mapped to their flags.
REQUIRED_FLAGS: dict[str, str] = _required_flags()


# Algorithm group -> the speculator types that consume it. A group set to a
# non-default value under a speculator_type absent from its set is ignored, and
# warns. DFlash2 and DSpark share the DFlash backbone group; their exclusive
# architecture knobs remain in their own groups. eagle3 uses no group.
_ALGORITHM_GROUP_USERS: dict[str, frozenset[str]] = {
    "dflash": frozenset({"dflash", "dflash2", "dspark"}),
    "dflash2": frozenset({"dflash2"}),
    "dspark": frozenset({"dspark"}),
    "peagle": frozenset({"peagle"}),
    "mtp": frozenset({"mtp"}),
}


# Marker key (schema ``_CLI_CHOICES``) -> the live registry supplying a field's
# argparse choices. Binding the registry here -- not in the schema -- keeps the schema
# free of runtime backend objects.
_CLI_CHOICE_REGISTRIES: dict[str, Any] = {
    "hidden_states_backends": HiddenStatesBackend.registry,
}


# ---------------------------------------------------------------------------
# Backend train-arg auto-registration
# ---------------------------------------------------------------------------
# Each HiddenStatesBackend's add_train_args is introspected once at import time
# so the backend-contributed flags appear in the real parser and survive
# resolution's CONFIG_DESTS filter. Backend args bypass pydantic entirely --
# they are carried on TrainConfig._backend_args and merged into flatten().


def _collect_backend_args() -> tuple[dict[str, list[argparse.Action]], frozenset[str]]:
    """Introspect every registered backend's ``add_train_args``.

    Returns ``(per_backend, all_dests)`` where *per_backend* maps backend name
    to its argparse actions and *all_dests* is the union of their dests.
    Raises at import time on dest collisions (cross-backend or with the schema).
    """
    per_backend: dict[str, list[argparse.Action]] = {}
    all_dests: set[str] = set()
    seen_dests: dict[str, str] = {}

    for name, backend_cls in sorted(HiddenStatesBackend.registry.items()):
        scratch = argparse.ArgumentParser(add_help=False)
        backend_cls.add_train_args(scratch)
        actions = [
            a
            for a in scratch._actions  # noqa: SLF001
            if a.dest not in ("help", argparse.SUPPRESS)
        ]
        per_backend[name] = actions
        for action in actions:
            dest = action.dest
            if dest in seen_dests:
                raise RuntimeError(
                    f"Backend '{name}' and '{seen_dests[dest]}' both register "
                    f"train-arg dest '{dest}'"
                )
            if dest in CONFIG_DESTS:
                raise RuntimeError(
                    f"Backend '{name}' train-arg dest '{dest}' collides with "
                    f"an existing schema field in CONFIG_DESTS"
                )
            seen_dests[dest] = name
            all_dests.add(dest)

    return per_backend, frozenset(all_dests)


_BACKEND_ACTIONS, _BACKEND_DESTS = _collect_backend_args()


def _backfill_backend_defaults(selected_backend: str, merged: dict[str, Any]) -> None:
    """Fill in defaults for the selected backend's unset args.

    Backend flags use ``default=SUPPRESS`` in the real parser so only
    user-provided values appear, but ``from_train_args`` expects every arg to
    exist on the namespace. This restores the backend's own defaults for any
    dest the user did not provide.
    """
    actions = _BACKEND_ACTIONS.get(selected_backend, [])
    for action in actions:
        if action.dest not in merged and action.default is not argparse.SUPPRESS:
            merged[action.dest] = action.default


def _annotation_spec(annotation: Any) -> tuple[Any, bool, list[Any] | None]:
    """Reduce a field annotation to ``(base_type, is_list, choices)``.

    Strips an optional ``| None``; maps ``list[T]`` to a ``nargs`` list and
    ``Literal[...]`` to ``choices``. ``base_type`` is the argparse ``type=``
    callable; for a ``Literal`` it is the element type so parsed values and
    ``choices`` compare like-for-like. A genuine multi-type union (e.g.
    ``int | str``) is rejected rather than silently truncated.
    """
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) != 1:
            raise TypeError(
                f"CLI generation supports only single-type optionals; got "
                f"{annotation!r} with {len(non_none)} non-None arms."
            )
        annotation = non_none[0]
        origin = get_origin(annotation)
    if origin is list:
        (inner,) = get_args(annotation)
        return inner, True, None
    if origin is Literal:
        choices = list(get_args(annotation))
        elem_types = {type(choice) for choice in choices}
        base = elem_types.pop() if len(elem_types) == 1 else str
        return base, False, choices
    return annotation, False, None


def _add_field_argument(
    parser: argparse._ArgumentGroup, name: str, field: FieldInfo
) -> None:
    """Add one argparse flag derived from a pydantic field.

    Defaults are ``SUPPRESS``ed so the parsed namespace holds only the flags the
    user actually passed -- that set is the ``flag`` layer of the precedence walk.
    """
    base, is_list, choices = _annotation_spec(field.annotation)
    flag = _dest_to_flag(name)
    kwargs: dict[str, Any] = {"dest": name, "default": argparse.SUPPRESS}
    if field.description:
        kwargs["help"] = field.description
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}

    # A str field may draw its choices from a dynamic registry named indirectly by a
    # _CLI_CHOICES marker, so the choice set is resolved here (where the registry is
    # known) with no field-name literal.
    choices_key = extra.get("cli_choices")
    if isinstance(choices_key, str):
        choices = list(_CLI_CHOICE_REGISTRIES[choices_key])

    if base is bool and not is_list:
        # True/None-default bools, or those explicitly tagged, render as
        # --x/--no-x; a plain False-default bool is a simple store_true flag.
        # store_true can only set true, so it can't flip a YAML-set true back to
        # false -- a bool needing CLI false-override must be tagged _CLI_BOOL_OPTIONAL.
        optional = (
            extra.get("cli_bool") == "optional"
            or field.default is None
            or field.default is True
        )
        action = argparse.BooleanOptionalAction if optional else "store_true"
        parser.add_argument(flag, action=action, **kwargs)
        return

    if is_list:
        kwargs["nargs"] = "+"
    kwargs["type"] = base
    if choices:
        kwargs["choices"] = choices
    parser.add_argument(flag, **kwargs)


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Append backend-contributed flags to *parser*, deduplicating by dest."""
    backend_group = parser.add_argument_group("hidden-states backends")
    seen_flags: set[str] = set()
    for _bname, actions in sorted(_BACKEND_ACTIONS.items()):
        for action in actions:
            flag = _dest_to_flag(action.dest)
            if flag in seen_flags:
                continue
            seen_flags.add(flag)
            kwargs: dict[str, Any] = {
                "dest": action.dest,
                "default": argparse.SUPPRESS,
            }
            if action.help:
                kwargs["help"] = action.help
            if action.type is not None:
                kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
            backend_group.add_argument(flag, **kwargs)


def add_config_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every config field as a flag, grouped by concern for ``--help``.

    This is the whole tunable surface: generated from the schema, so a new field
    becomes a new flag with the right type, choices, bool style, and help.
    Backend-contributed flags are appended via auto-registration from each
    :class:`HiddenStatesBackend`'s ``add_train_args``.
    """
    general = parser.add_argument_group("general")
    for name in _ROOT_FIELDS:
        _add_field_argument(general, name, TrainConfig.model_fields[name])
    for gname, gmodel in _GROUPS.items():
        group = parser.add_argument_group(gname)
        for name, field in gmodel.model_fields.items():
            _add_field_argument(group, name, field)

    if _BACKEND_ACTIONS:
        _add_backend_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    """The ``train.py`` parser: ``--config`` plus the schema-generated flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        default=None,
        metavar="PATH",
        help="YAML config file (stage-shaped: trainer keys under a top-level "
        "'train:' key; a bare mapping is also accepted). CLI flags override it.",
    )
    parser.add_argument(
        "--dump-config",
        dest="dump_config",
        action="store_true",
        help="Print the fully-resolved config as a valid run.yaml to stdout and "
        "exit, so a working invocation can be scaffolded into a shareable config.",
    )
    add_config_cli_arguments(parser)
    return parser


def _validate_draft_init(cfg: TrainConfig, provided: set[str]) -> None:
    """Enforce the draft-init contract; raise :class:`ConfigError` on conflict.

    The draft model may be defined in exactly one way: ``--from-pretrained`` (a
    complete checkpoint), ``--draft-config`` (a decoder config, rest built from
    the other flags), or the decoder-shaping flags (synthesize everything).
    ``--from-pretrained`` takes precedence over all others; ``--draft-config`` is
    incompatible with the decoder-shaping flags; and MTP-from-scratch reuses the
    verifier's own decoder, so ``--draft-config`` and the shaping flags do not
    apply to it.

    ``provided`` is the set of dests won by a non-default layer, so a shaping flag
    passed at its default value still counts as a conflict -- preserving the
    pre-refactor behavior.
    """
    shaping = [flag for dest, flag in DECODER_SHAPING_FLAGS.items() if dest in provided]
    with_draft_config = shaping + (["--draft-config"] if cfg.draft.draft_config else [])
    if cfg.draft.from_pretrained:
        _reject_conflicts(
            with_draft_config,
            "--from-pretrained loads a complete draft model and takes precedence "
            "over all other model-definition options, so these conflict with it",
        )
        return
    if cfg.speculator_type == "mtp":
        _reject_conflicts(
            with_draft_config,
            "--speculator-type mtp reuses the verifier's decoder config, so these "
            "options do not apply",
        )
        return
    if cfg.draft.draft_config:
        _reject_conflicts(
            shaping,
            "--draft-config defines the draft decoder, so these flags conflict with it",
        )


def _reject_conflicts(conflicting: list[str], reason: str) -> None:
    if conflicting:
        raise ConfigError(f"{reason} (remove them): {', '.join(conflicting)}")


def _warn_mismatched_algorithm_blocks(cfg: TrainConfig, provided: set[str]) -> None:
    """Warn when an algorithm group is set under a speculator type that ignores it.

    Every algorithm group and its flags stay present in the schema for every
    speculator type (back-compat requires the flags to always exist), so a
    mismatched block is not a parse error -- it is silently ignored by the model
    layer. Surface that so a misplaced recipe is visible, without rejecting it:
    the pre-refactor lenient behavior.

    Whether a group was "set" is read from the same in-memory winning-layer record
    (``provided`` = won by flag or yaml) as the draft-init check, so a mismatched
    block supplied entirely via YAML warns identically to one on the CLI.
    """
    for group, users in _ALGORITHM_GROUP_USERS.items():
        if cfg.speculator_type in users:
            continue
        ignored = sorted(
            _dest_to_flag(dest)
            for dest in _GROUPS[group].model_fields
            if dest in provided
        )
        if ignored:
            warnings.warn(
                f"--speculator-type {cfg.speculator_type} does not use the '{group}' "
                f"algorithm group; these settings are ignored: {', '.join(ignored)}",
                stacklevel=2,
            )


def build_from_sources(
    cls: type[TrainConfig],
    *,
    cli: dict[str, Any],
    config_path: str | None,
    argv: list[str],
    backend_cli: dict[str, Any] | None = None,
) -> TrainConfig:
    """Layer the sources into a validated config, recording each value's origin.

    The pure core behind :meth:`TrainConfig.from_sources`: it loads the optional
    stage-shaped ``config_path`` YAML, feeds the ``flag`` and ``yaml`` layers to
    pydantic-settings in precedence order (so the merge yields
    ``flag > yaml > default``), records per dest which layer won, and enforces the
    cross-field draft-init contract. Backend-contributed args bypass pydantic and
    are carried on ``cfg._backend_args``. Raises :class:`ConfigError` /
    ``ValidationError`` on bad input; never exits the process.
    """
    yaml_nested: dict[str, Any] = {}
    yaml_dests: set[str] = set()
    if config_path is not None:
        yaml_nested = _unwrap_stage(_load_config_file(config_path))
        yaml_dests, unknown = _partition_yaml_keys(yaml_nested)
        if unknown:
            warnings.warn(
                f"--config '{config_path}' has unrecognised keys (ignored): "
                f"{', '.join(sorted(unknown))}",
                stacklevel=2,
            )

    cfg = cls(_layers={"flag": nest_flat(cli), "yaml": yaml_nested})  # type: ignore[call-arg]
    cfg._provenance = {
        dest: ("flag" if dest in cli else "yaml" if dest in yaml_dests else "default")
        for dest in CONFIG_DESTS
    }
    cfg._argv = list(argv)

    # Backend-contributed args: merge YAML (backend: block) and CLI with
    # flag > yaml precedence, then backfill defaults for the selected backend.
    yaml_backend = {}
    if isinstance(yaml_nested.get("backend"), dict):
        yaml_backend = {
            k: v for k, v in yaml_nested["backend"].items() if k in _BACKEND_DESTS
        }
    merged_backend = dict(yaml_backend)
    if backend_cli:
        merged_backend.update(backend_cli)
    _backfill_backend_defaults(cfg.data.hidden_states_backend, merged_backend)
    cfg._backend_args = merged_backend

    for dest in _BACKEND_DESTS:
        if backend_cli and dest in backend_cli:
            cfg._provenance[dest] = "flag"
        elif dest in yaml_backend:
            cfg._provenance[dest] = "yaml"

    provided = {dest for dest, layer in cfg._provenance.items() if layer != "default"}
    _warn_mismatched_algorithm_blocks(cfg, provided)
    _validate_draft_init(cfg, provided)
    _validate_required(cfg)
    return cfg


def _validate_required(cfg: TrainConfig) -> None:
    """Enforce the ``cli_required`` contract once every layer has been merged.

    argparse can't do this: the value may arrive from ``--config`` rather than a
    flag, and requiring it at parse time would reject a ``run.yaml`` that supplies
    it. So the schema keeps an empty placeholder default and we check the resolved
    value here, naming the flag the user can set (or provide in ``--config``).
    """
    for dest, flag in REQUIRED_FLAGS.items():
        if cfg.provenance[dest] == "default":
            raise ConfigError(
                f"missing required value: set {flag} or provide it in --config"
            )


def _load_config_file(path: str) -> dict[str, Any]:
    """Parse a ``--config`` YAML file into a top-level mapping.

    Raises :class:`ConfigError` -- not a traceback -- for the mistakes a user
    actually makes: an unreadable path, unparseable YAML, or a top-level document
    that is not a mapping. The :func:`resolve` boundary renders that as exit 2.
    """
    try:
        text = Path(path).read_text()
        data = yaml.safe_load(text)
    except OSError as exc:
        raise ConfigError(
            f"--config '{path}' could not be read: {exc.strerror or exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"--config '{path}' is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"--config '{path}' must contain a top-level mapping.")
    return data


def _unwrap_stage(data: dict[str, Any]) -> dict[str, Any]:
    """Return the trainer config, unwrapping the ``train:`` stage block if present.

    The canonical file is stage-shaped -- trainer keys nest under ``train:`` so a
    future ``prepare_data:`` / ``launch_vllm:`` stage extends the file rather than
    replacing it. Sibling stage keys are ignored: a file authored for
    the whole pipeline still trains today using only its ``train:`` block. Loading
    stays lenient -- a bare top-level mapping (no ``train:`` key, since no config
    group is named ``train``) is accepted unchanged for back-compat.
    """
    stage = data.get("train")
    return stage if isinstance(stage, dict) else data


def _partition_yaml_keys(yaml_nested: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Split a parsed YAML mapping's leaves into ``(known, unknown)`` flat dests.

    A group block contributes its leaf keys; a root scalar contributes itself.
    Each group leaf is checked against *its own* group's fields, not the global
    dest set, so a real field placed under the wrong block reads as unknown (and
    stays out of provenance) rather than being accepted because the name exists in
    some other group. Unknown keys are warned about and ignored.
    """
    known: set[str] = set()
    unknown: set[str] = set()
    for key, value in yaml_nested.items():
        if key in _GROUPS and isinstance(value, dict):
            group_fields = _GROUPS[key].model_fields
            for leaf in value:
                (known if leaf in group_fields else unknown).add(leaf)
        elif key == "backend" and isinstance(value, dict):
            for leaf in value:
                (known if leaf in _BACKEND_DESTS else unknown).add(leaf)
        elif key in _ROOT_FIELDS:
            known.add(key)
        else:
            unknown.add(key)
    return known, unknown


def _format_config_error(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as a concise, flag-oriented message."""
    lines = []
    for err in exc.errors():
        label = ".".join(str(part) for part in err["loc"]) or "<config>"
        message = err["msg"].removeprefix("Value error, ")
        lines.append(f"{label}: {message}")
    return "invalid configuration:\n  " + "\n  ".join(lines)


def resolve(cls: type[TrainConfig], argv: list[str] | None) -> TrainConfig:
    """Parse argv, layer the sources, validate; exit(2) cleanly on any error.

    The only function that touches ``sys.argv`` or raises ``SystemExit``. Any
    configuration error -- a missing required value (which names its flag, e.g.
    ``--verifier-name-or-path``), a draft-init conflict, a bad value, or a broken
    config file -- surfaces through ``parser.error`` as ``SystemExit(2)`` with no
    traceback.
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    config_path = namespace.config
    ns_vars = vars(namespace)
    cli = {dest: value for dest, value in ns_vars.items() if dest in CONFIG_DESTS}
    backend_cli = {
        dest: value for dest, value in ns_vars.items() if dest in _BACKEND_DESTS
    }
    # The command recorded for provenance: the live argv on the real launch path,
    # or a reconstruction when resolve() is driven with an explicit argv.
    full_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    try:
        cfg = cls.from_sources(
            cli=cli,
            config_path=config_path,
            argv=full_argv,
            backend_cli=backend_cli,
        )
    except ValidationError as exc:
        parser.error(_format_config_error(exc))
    except ConfigError as exc:
        parser.error(str(exc))
    # --dump-config turns this working invocation into a shareable run.yaml: print
    # the resolved config and exit cleanly (exit 0) before any training happens.
    if namespace.dump_config:
        sys.stdout.write(cfg.dump_yaml())
        raise SystemExit(0)
    return cfg
