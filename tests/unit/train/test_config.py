"""Tests for the config-file layer: YAML loading, CLI>YAML>default precedence,
provenance-aware validation, --dump-config round-trip, and error fidelity."""

import argparse
import sys
from unittest import mock

import pytest
import yaml

from scripts.train import parse_args
from speculators.train.config import (
    _GROUPS,
    _ROOT_FIELDS,
    add_config_cli_arguments,
    build_train_config,
    config_from_flat,
    dump_config_yaml,
    flatten_config,
)


def _write_yaml(tmp_path, data) -> str:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _parse(argv_tail):
    with mock.patch.object(sys, "argv", ["train.py", *argv_tail]):
        return parse_args()


# --------------------------------------------------------------------------- #
# YAML loading + precedence
# --------------------------------------------------------------------------- #


def test_yaml_supplies_values(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "from-yaml"},
            "optimizer": {"lr": 0.001},
            "trainer": {"epochs": 7},
        },
    )
    args = _parse(["--config", cfg_path])
    assert args.verifier_name_or_path == "from-yaml"
    assert args.lr == 0.001
    assert args.epochs == 7


def test_cli_overrides_yaml(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "from-yaml"},
            "optimizer": {"lr": 0.001, "weight_decay": 0.5},
        },
    )
    args = _parse(["--config", cfg_path, "--lr", "0.02"])
    assert args.lr == 0.02  # CLI wins
    assert args.weight_decay == 0.5  # untouched sibling still from YAML (deep merge)


def test_yaml_partial_group_keeps_defaults(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "optimizer": {"lr": 0.003},
        },
    )
    args = _parse(["--config", cfg_path])
    assert args.lr == 0.003
    assert args.weight_decay == 0.01  # coded default
    assert args.muon_momentum == 0.95  # coded default (untouched sibling)


def test_muon_lr_defaults_to_ten_times_lr():
    # Unset muon_lr resolves to 10 * lr (matches the post-#746 behavior).
    args = _parse(["--verifier-name-or-path", "v", "--lr", "3e-4"])
    assert args.muon_lr == pytest.approx(3e-3)
    # Explicit muon_lr wins over the derived default.
    args = _parse(["--verifier-name-or-path", "v", "--lr", "3e-4", "--muon-lr", "0.05"])
    assert args.muon_lr == 0.05


def test_default_optimizer_is_muon():
    args = _parse(["--verifier-name-or-path", "v"])
    assert args.optimizer == "muon"


def test_no_config_matches_pure_cli():
    args = _parse(["--verifier-name-or-path", "v", "--lr", "0.05"])
    assert args.lr == 0.05
    assert args.epochs == 20


# --------------------------------------------------------------------------- #
# Provenance-aware validation
# --------------------------------------------------------------------------- #


def test_draft_init_conflict_expressed_entirely_in_yaml_is_caught(tmp_path):
    # from_pretrained + a decoder-shaping flag conflict -- both only in YAML.
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "draft": {"from_pretrained": "/ckpt", "num_layers": 4},
        },
    )
    with pytest.raises(SystemExit):
        _parse(["--config", cfg_path])


def test_build_train_config_provenance_union():
    cfg, provided = build_train_config(
        {"verifier_name_or_path": "v", "lr": 0.01}, config_path=None
    )
    assert "lr" in provided
    assert "verifier_name_or_path" in provided
    assert "weight_decay" not in provided  # defaulted, not provided


def test_mismatched_algo_block_warns(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "speculator_type": "eagle3",
            "peagle": {"num_depths": 4},
        },
    )
    with pytest.warns(UserWarning, match="does not use the 'peagle' block"):
        _parse(["--config", cfg_path])


def test_unknown_yaml_key_warns_and_is_ignored(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "optimizer": {"lr": 0.01, "not_a_real_field": 1},
        },
    )
    with pytest.warns(UserWarning, match="unrecognised keys"):
        args = _parse(["--config", cfg_path])
    assert args.lr == 0.01


def test_misplaced_yaml_key_is_unrecognised_not_provided(tmp_path):
    # num_layers is a real 'draft' field; placing it under 'optimizer' is a
    # mistake. It must be reported as unrecognised (leaf validated against its
    # own group, not the global dest set) and must NOT enter the provenance set,
    # which would otherwise risk a spurious draft-init conflict.
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "optimizer": {"num_layers": 4},
        },
    )
    with pytest.warns(UserWarning, match="unrecognised keys"):
        cfg, provided = build_train_config({}, cfg_path)
    assert "num_layers" not in provided
    assert cfg.draft.num_layers == 1  # misplaced value ignored; keeps its default


# --------------------------------------------------------------------------- #
# --dump-config round-trip
# --------------------------------------------------------------------------- #


def test_dump_config_is_a_fixed_point():
    cfg, _ = build_train_config(
        {"verifier_name_or_path": "v", "speculator_type": "dflash", "lr": 0.02},
        config_path=None,
    )
    dumped = dump_config_yaml(cfg)
    reloaded = config_from_flat(flatten_config(cfg))
    assert dump_config_yaml(reloaded) == dumped  # dump -> load -> dump is stable


def test_dump_config_flag_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["--verifier-name-or-path", "v", "--dump-config"])
    assert exc.value.code in (0, None)
    out = capsys.readouterr().out
    parsed = yaml.safe_load(out)
    assert parsed["verifier"]["verifier_name_or_path"] == "v"


def test_dumped_config_round_trips_through_cli(tmp_path):
    cfg, _ = build_train_config(
        {"verifier_name_or_path": "v", "speculator_type": "peagle", "num_depths": 5},
        config_path=None,
    )
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(dump_config_yaml(cfg))
    args = _parse(["--config", str(cfg_path)])
    assert args.speculator_type == "peagle"
    assert args.num_depths == 5


# --------------------------------------------------------------------------- #
# Error fidelity
# --------------------------------------------------------------------------- #


def test_missing_verifier_errors_cleanly(capsys):
    with pytest.raises(SystemExit):
        _parse(["--lr", "0.01"])
    # The message points at the flag to set, not just the group path.
    assert "--verifier-name-or-path" in capsys.readouterr().err


def test_dump_config_validates_before_emitting(tmp_path):
    # --dump-config must run the draft-init contract first, so it never emits a
    # scaffold that would fail to load; a conflict exits 2 instead of printing.
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "draft": {"from_pretrained": "/ckpt", "num_layers": 4},
        },
    )
    with pytest.raises(SystemExit) as exc:
        _parse(["--config", cfg_path, "--dump-config"])
    assert exc.value.code == 2


def test_invalid_checkpoint_freq_errors():
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--checkpoint-freq", "-1"])


def test_invalid_dtype_errors():
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--hidden-states-dtype", "not_a_dtype"])


def test_invalid_checkpoint_freq_non_integer_over_one_errors():
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--checkpoint-freq", "2.5"])


# --------------------------------------------------------------------------- #
# Reentrant YAML-path source (no global state)
# --------------------------------------------------------------------------- #


def test_yaml_path_is_reentrant(tmp_path):
    a = _write_yaml_named(
        tmp_path,
        "a.yaml",
        {"verifier": {"verifier_name_or_path": "v"}, "optimizer": {"lr": 0.001}},
    )
    b = _write_yaml_named(
        tmp_path,
        "b.yaml",
        {"verifier": {"verifier_name_or_path": "v"}, "optimizer": {"lr": 0.999}},
    )
    ca, _ = build_train_config({}, a)
    cb, _ = build_train_config({}, b)
    ca2, _ = build_train_config({}, a)
    assert (ca.optimizer.lr, cb.optimizer.lr, ca2.optimizer.lr) == (0.001, 0.999, 0.001)
    # config_path=None loads no file and never leaks a private kwarg
    cn, _ = build_train_config({"verifier_name_or_path": "v"}, None)
    assert cn.optimizer.lr == 1e-4
    assert "_yaml_file" not in cn.model_dump()


def _write_yaml_named(tmp_path, name, data) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


# --------------------------------------------------------------------------- #
# CLI list replaces (does not merge) the YAML list
# --------------------------------------------------------------------------- #


def test_cli_list_replaces_yaml_list(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "speculator_type": "dflash",
            "draft": {"full_attention_indices": [7, 8, 9]},
        },
    )
    args = _parse(["--config", cfg_path, "--full-attention-indices", "0", "2"])
    assert args.full_attention_indices == [0, 2]  # full replace, not merge


def test_yaml_list_used_when_cli_absent(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        {
            "verifier": {"verifier_name_or_path": "v"},
            "speculator_type": "dflash",
            "draft": {"full_attention_indices": [7, 8, 9]},
        },
    )
    args = _parse(["--config", cfg_path])
    assert args.full_attention_indices == [7, 8, 9]


# --------------------------------------------------------------------------- #
# Schema-driven flag generation
# --------------------------------------------------------------------------- #


def _generated_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_config_cli_arguments(p)
    return p


def test_generator_flag_styles():
    p = _generated_parser()
    by_dest = {a.dest: a for a in p._actions}
    # store_true (no --no- variant)
    assert by_dest["fc_norm"].option_strings == ["--fc-norm"]
    # BooleanOptionalAction (has --no- variant): None-default, False-default tagged,
    # and plain True-default bools all get the --no- form.
    assert "--no-norm-output" in by_dest["norm_output"].option_strings
    assert "--no-embed-requires-grad" in by_dest["embed_requires_grad"].option_strings
    assert "--no-norm-before-residual" in by_dest["norm_before_residual"].option_strings
    assert (
        "--no-confidence-head-with-markov"
        in by_dest["confidence_head_with_markov"].option_strings
    )
    # choices from Literal (element type preserved, so type is the base callable)
    assert by_dest["optimizer"].choices == ["adamw", "muon"]
    assert by_dest["scheduler_type"].choices == ["linear", "cosine", "none"]
    assert by_dest["draft_attn_impl"].choices == [
        "simple_flex_attention",
        "sdpa",
        "eager",
    ]
    # nargs list
    assert by_dest["target_layer_ids"].nargs == "+"
    assert by_dest["target_layer_ids"].type is int


def test_generator_covers_every_field_exactly_once():
    p = _generated_parser()
    generated = {a.dest for a in p._actions if a.dest != "help"}
    expected = set(_ROOT_FIELDS)
    for model in _GROUPS.values():
        expected |= set(model.model_fields)
    assert generated == expected


def test_no_generated_flag_name_collision():
    # Guards against a future optional-bool (e.g. resume_from_checkpoint) whose
    # auto --no- variant would collide with an existing store_true flag.
    p = _generated_parser()
    seen: dict[str, str] = {}
    for action in p._actions:
        for opt in action.option_strings:
            assert opt not in seen, (
                f"flag {opt} generated by both {seen[opt]} and {action.dest}"
            )
            seen[opt] = action.dest


# --------------------------------------------------------------------------- #
# Cross-field validation (ported from the old parser tail) + error fidelity
# --------------------------------------------------------------------------- #


def test_dpace_requires_ce_loss():
    # D-PACE weighting requires CE loss; default loss_fn (kl_div) must be rejected.
    with pytest.raises(SystemExit):
        _parse(
            [
                "--verifier-name-or-path",
                "v",
                "--speculator-type",
                "dflash",
                "--per-position-loss-weight",
                "dpace",
            ]
        )


def test_dpace_alpha_out_of_range():
    with pytest.raises(SystemExit):
        _parse(
            [
                "--verifier-name-or-path",
                "v",
                "--speculator-type",
                "dflash",
                "--per-position-loss-weight",
                "dpace",
                "--loss-fn",
                "ce",
                "--dpace-alpha",
                "1.5",
            ]
        )


def test_dpace_not_enforced_on_default_run():
    # The dpace guard must NOT fire on a default (fixed-exp-decay) run.
    args = _parse(["--verifier-name-or-path", "v", "--speculator-type", "dflash"])
    assert args.per_position_loss_weight == "fixed-exp-decay"


def test_invalid_loss_fn_errors():
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--loss-fn", "not_a_loss"])


# --------------------------------------------------------------------------- #
# --config file edge cases (new surface): clean errors, never tracebacks
# --------------------------------------------------------------------------- #


def test_missing_config_file_is_a_clean_error():
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--config", "/no/such/file.yaml"])


def test_non_mapping_config_is_a_clean_error(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("- a\n- b\n")  # top-level sequence, not a mapping
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--config", str(path)])


def test_malformed_config_is_a_clean_error(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("optimizer: {lr: 0.1\n")  # unterminated flow mapping
    with pytest.raises(SystemExit):
        _parse(["--verifier-name-or-path", "v", "--config", str(path)])


def test_empty_config_is_all_defaults(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("")  # empty file parses to {} -> pure defaults
    args = _parse(["--verifier-name-or-path", "v", "--config", str(path)])
    assert args.verifier_name_or_path == "v"
    assert args.epochs == 20  # the coded default
