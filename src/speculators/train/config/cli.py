"""The CLI mechanism: schema -> argparse, ``--set`` parsing, the resolve boundary.

Everything that touches the command line lives here so the schema stays pure:

* :func:`add_config_cli_arguments` / :func:`build_parser` -- generate the whole
  ~80-flag surface from the schema (one typed field == one flag with the right
  type, choices, bool style, and help), plus the run-mode flags ``--config``,
  ``--dump-config``, and the generic ``--set`` escape hatch.
* :func:`parse_set_overrides` -- turn ``--set dotted.key=value`` into flat
  ``{dest: value}``, parsing the value as a YAML scalar and rejecting unknown or
  misplaced keys.
* :func:`decoder_shaping_flags` / :func:`validate_draft_init` -- the draft-init
  conflict contract, driven by typed :attr:`~.schema.TrainConfig.provenance` so a
  conflict expressed via YAML or ``--set`` is caught identically to one on the CLI.
* :func:`required_flags` / :func:`format_config_error` -- name the flag to set
  when a config error would otherwise show an opaque group path.
* :func:`resolve` -- the impure boundary behind :meth:`TrainConfig.resolve`: parse
  argv, delegate to the pure core, handle ``--dump-config``, turn any config
  error into a clean ``SystemExit(2)``.
"""

import argparse
import sys
import types
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from .errors import ConfigError
from .schema import _GROUPS, _ROOT_FIELDS, CONFIG_DESTS, DraftArgs, TrainConfig

# --------------------------------------------------------------------------- #
# Schema-driven argparse CLI generation
# --------------------------------------------------------------------------- #


def _dest_to_flag(dest: str) -> str:
    """The argparse flag string for a config dest (``muon_lr`` -> ``--muon-lr``)."""
    return "--" + dest.replace("_", "-")


def _annotation_spec(annotation: Any) -> tuple[Any, bool, list[Any] | None]:
    """Reduce a field annotation to ``(base_type, is_list, choices)``.

    Strips an optional ``| None``; recognises ``list[T]`` (-> nargs) and
    ``Literal[...]`` (-> choices). ``base_type`` is the argparse ``type=`` callable;
    for a ``Literal`` it is the element type (so ``choices`` and parsed values are
    compared like-for-like -- a string ``type`` against integer ``choices`` would
    reject every value).

    Raises ``TypeError`` on annotations the generator cannot represent faithfully
    (a genuine multi-type union such as ``int | str``), rather than silently
    keeping only the first arm -- this keeps "add a typed field -> get a correct
    flag" honest for future field kinds.
    """
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
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
        elem_types = {type(c) for c in choices}
        base = elem_types.pop() if len(elem_types) == 1 else str
        return base, False, choices
    return annotation, False, None


def _add_field_argument(
    parser: argparse._ActionsContainer, name: str, field: FieldInfo
) -> None:
    """Add one argparse flag derived from a pydantic field.

    Defaults are ``SUPPRESS``ed so the parsed namespace contains only the flags
    the user actually passed -- that set is the CLI provenance.
    """
    base, is_list, choices = _annotation_spec(field.annotation)
    flag = _dest_to_flag(name)
    kwargs: dict[str, Any] = {"dest": name, "default": argparse.SUPPRESS}
    if field.description:
        kwargs["help"] = field.description

    if base is bool and not is_list:
        extra = (
            field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        )
        # True/None-default bools, or those explicitly tagged, get --x/--no-x;
        # a plain False-default bool becomes a simple store_true flag.
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


def add_config_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register every config field as an argparse flag, grouped for ``--help``.

    This is the whole tunable CLI surface: it is generated from the pydantic
    schema, so a new field automatically becomes a new flag with the right type,
    choices, bool style, and help text.
    """
    general = parser.add_argument_group("general")
    for name in _ROOT_FIELDS:
        _add_field_argument(general, name, TrainConfig.model_fields[name])
    for gname, gmodel in _GROUPS.items():
        group = parser.add_argument_group(gname)
        for name, field in gmodel.model_fields.items():
            _add_field_argument(group, name, field)


def build_parser() -> argparse.ArgumentParser:
    """The full ``train.py`` parser: run-mode flags + the generated tunable flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file supplying any of the arguments below. "
        "Values nest under a top-level 'train:' key (a bare mapping is also "
        "accepted). CLI flags and --set override the file, which overrides the "
        "defaults. Optional: with no --config the behaviour is a pure-CLI run.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        default=False,
        help="Print the fully-resolved config as YAML to stdout and exit. The "
        "output is a valid --config file; use it to scaffold a run.yaml.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key by dotted path, e.g. "
        "'--set optimizer.muon_ns_steps=7'. Repeatable. The value is parsed as a "
        "YAML scalar (so 1e-4, true, null, and [2,18,33] type correctly). A "
        "'train.' prefix is accepted. Precedence sits below CLI flags and above "
        "the --config file.",
    )
    # Every tunable flag is generated from the pydantic config schema: one typed
    # field there == one CLI flag here, with the right type/choices/bool/help.
    add_config_cli_arguments(parser)
    return parser


# --------------------------------------------------------------------------- #
# --set escape hatch
# --------------------------------------------------------------------------- #


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``--set`` items into a flat ``{dest: value}`` override map.

    Each item is ``dotted.key=value``. A leading ``train.`` stage prefix is
    stripped. The value is parsed as a YAML scalar so ``1e-4``, ``true``,
    ``null``, and inline lists like ``[2,18,33]`` type correctly (pydantic coerces
    the leftover string cases at the field). An unknown key, or a group-prefixed
    key whose group does not own that dest, is a :class:`~.errors.ConfigError`.
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"--set '{item}' is not of the form KEY=VALUE (e.g. "
                "'--set optimizer.lr=1e-4')."
            )
        raw_key, _, raw_value = item.partition("=")
        key = raw_key.strip()
        if key.startswith("train."):
            key = key[len("train.") :]
        dest = _resolve_set_key(key, item)
        out[dest] = yaml.safe_load(raw_value)
    return out


def _resolve_set_key(key: str, original: str) -> str:
    """Resolve a (train.-stripped) ``--set`` key to its flat dest, or raise.

    Accepts a bare dest (``lr``, ``seed`` -- unambiguous since dests are unique
    across groups) or a ``group.key`` path; anything deeper is rejected.
    """
    if "." not in key:
        if key in CONFIG_DESTS:
            return key
        raise ConfigError(f"--set: unknown config key '{original}'.")
    group, _, dest = key.partition(".")
    if "." in dest:
        raise ConfigError(
            f"--set: key '{key}' is too deeply nested (expected KEY or GROUP.KEY)."
        )
    if group not in _GROUPS:
        raise ConfigError(
            f"--set: unknown group '{group}' in '{original}' "
            f"(known groups: {', '.join(_GROUPS)})."
        )
    if dest not in _GROUPS[group].model_fields:
        raise ConfigError(
            f"--set: group '{group}' has no key '{dest}' (in '{original}')."
        )
    return dest


# --------------------------------------------------------------------------- #
# Draft-init conflict contract (provenance-driven)
# --------------------------------------------------------------------------- #


def decoder_shaping_flags() -> dict[str, str]:
    """Ordered ``{dest: flag}`` for the draft decoder-shaping fields.

    Sourced from the schema (fields tagged ``_DECODER_SHAPING``) in declaration
    order, so adding or renaming a shaping knob needs no second edit here. Consumed
    by :func:`validate_draft_init` to enforce that these flags do not co-occur
    with --from-pretrained / --draft-config.
    """
    flags: dict[str, str] = {}
    for name, field in DraftArgs.model_fields.items():
        extra = field.json_schema_extra
        if isinstance(extra, dict) and extra.get("decoder_shaping"):
            flags[name] = _dest_to_flag(name)
    return flags


# CLI flags that synthesize the draft decoder shape. They conflict with both
# --from-pretrained and --draft-config, each of which fully defines the draft.
# Derived from the schema (fields tagged _DECODER_SHAPING) so the set stays in one
# place; adding a shaping knob is still a single field edit in the schema.
DECODER_SHAPING_FLAGS: dict[str, str] = decoder_shaping_flags()


def validate_draft_init(cfg: TrainConfig) -> None:
    """Enforce the draft-init contract; raise :class:`~.errors.ConfigError` on conflict.

    The draft model may be defined in exactly one way:

    * ``--from-pretrained`` -- load a complete speculator checkpoint (or a
      config-only directory); or
    * ``--draft-config`` -- load just the decoder config and build the rest of
      the speculator from the other CLI args; or
    * the decoder-shaping flags (``--num-layers`` etc.) -- synthesize everything.

    ``--from-pretrained`` takes precedence over all other model-definition
    options: it is mutually exclusive with ``--draft-config`` and with the
    decoder-shaping flags, since those values come from the checkpoint.
    ``--draft-config`` is likewise incompatible with the decoder-shaping flags.
    MTP from scratch (``--speculator-type mtp`` without ``--from-pretrained``)
    reuses the verifier's own decoder config, so ``--draft-config`` and the
    decoder-shaping flags do not apply and are rejected.

    "Explicitly provided" is read from typed :attr:`~.schema.TrainConfig.provenance`
    (won by flag, ``--set``, or YAML), so a conflict expressed entirely in the
    YAML file or via ``--set`` is caught identically to one on the CLI. A flag
    passed at its default value still counts; only the
    :data:`DECODER_SHAPING_FLAGS` subset is checked here.
    """
    provided = cfg.provenance.provided()
    shaping = [flag for dest, flag in DECODER_SHAPING_FLAGS.items() if dest in provided]
    draft_config = cfg.draft.draft_config
    # --from-pretrained and mtp-from-scratch both fully define the draft elsewhere,
    # so both the shaping flags AND --draft-config conflict with them.
    with_draft_config = shaping + (["--draft-config"] if draft_config else [])

    if cfg.draft.from_pretrained:
        _reject_conflicts(
            with_draft_config,
            "--from-pretrained loads a complete draft model and takes precedence "
            "over all other model-definition options, so these conflict with it",
        )
        return
    if cfg.speculator_type == "mtp":
        # MTP-from-scratch reuses the verifier's own decoder config and extracts the
        # native MTP head weights, so these do not apply.
        _reject_conflicts(
            with_draft_config,
            "--speculator-type mtp reuses the verifier's decoder config, so these "
            "options do not apply",
        )
        return
    if draft_config:
        _reject_conflicts(
            shaping,
            "--draft-config defines the draft decoder, so these flags conflict with it",
        )


def _reject_conflicts(conflicting: list[str], reason: str) -> None:
    """Raise a uniform draft-init :class:`ConfigError` if any flag conflicts."""
    if conflicting:
        raise ConfigError(f"{reason} (remove them): {', '.join(conflicting)}")


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #


def required_flags(loc: tuple[Any, ...]) -> list[str]:
    """CLI flag(s) that satisfy a failed-validation ``loc`` from a ValidationError.

    Maps a pydantic error location back to actionable flags so a config error can
    name the flag to set rather than an opaque field/group path: a leaf dest maps
    to its own flag; a bare group name (the whole required group is absent) maps
    to that group's required flags. Returns ``[]`` when nothing maps.
    """
    if not loc:
        return []
    tail = str(loc[-1])
    if tail in CONFIG_DESTS:
        return [_dest_to_flag(tail)]
    if tail in _GROUPS:
        return [
            _dest_to_flag(name)
            for name, field in _GROUPS[tail].model_fields.items()
            if field.is_required()
        ]
    return []


def format_config_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a concise CLI message."""
    lines = []
    for err in exc.errors():
        loc = err["loc"]
        label = ".".join(str(p) for p in loc) or "<config>"
        # Drop pydantic's "Value error, " prefix for a cleaner CLI message.
        msg = err["msg"].removeprefix("Value error, ")
        # For a genuinely-missing field, point at the flag to set rather than an
        # opaque group path (e.g. "verifier: Field required" -> name
        # --verifier-name-or-path).
        flags = required_flags(loc) if err.get("type") == "missing" else []
        hint = f" (set {' or '.join(flags)})" if flags else ""
        lines.append(f"{label}: {msg}{hint}")
    return "invalid configuration:\n  " + "\n  ".join(lines)


# --------------------------------------------------------------------------- #
# The impure resolve boundary
# --------------------------------------------------------------------------- #


def resolve(cls: type[TrainConfig], argv: list[str] | None) -> TrainConfig:
    """Parse argv, layer sources, validate; handle ``--dump-config`` and errors.

    The whole point of the split is that this is the *only* function that touches
    ``sys.argv`` or raises ``SystemExit``; the layering/validation is the pure
    :func:`~.sources.build_from_sources` core it delegates to.
    """
    parser = build_parser()
    ns = parser.parse_args(argv)

    # Every generated flag defaults to argparse.SUPPRESS, so the namespace holds
    # only the flags the user actually passed -- that set IS the CLI provenance.
    cli_values = {dest: v for dest, v in vars(ns).items() if dest in CONFIG_DESTS}

    # Surface any configuration failure as a clean CLI error (exit 2), never a
    # traceback: pydantic validation, a --set/draft-init ConfigError, or a broken
    # --config file (missing/unreadable/malformed/non-mapping YAML).
    # The full command (program + args) recorded in train_command.txt: the live
    # sys.argv on the real launch path, or a reconstruction when resolve() is
    # driven with an explicit argv (tests / a programmatic caller).
    full_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    try:
        cfg = cls.from_sources(
            cli=cli_values,
            overrides=ns.set_overrides,
            config_path=ns.config,
            argv=full_argv,
        )
    except ValidationError as exc:
        parser.error(format_config_error(exc))
    except ConfigError as exc:
        # ConfigError IS-A ValueError, so it must be caught before the generic
        # ValueError net below (which would wrongly prepend the --config path).
        parser.error(str(exc))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        parser.error(f"--config '{ns.config}': {exc}")

    if ns.dump_config:
        # Emit AFTER validation (incl. the draft-init contract) so the scaffold is
        # guaranteed to be a loadable config, never one that fails on re-read.
        sys.stdout.write(cfg.dump_yaml())
        parser.exit()

    return cfg
