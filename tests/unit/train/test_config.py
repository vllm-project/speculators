"""Tests for the config subsystem, exercised at its public seams.

The primary seam is ``TrainConfig.from_sources(...)`` -- the pure, argv-free,
exception-raising core: layering/precedence (``flag > --set > YAML > default``),
the ``--set`` escape hatch, the ``train:`` stage wrapper, typed provenance,
validation/errors, and the ``dump_yaml`` -> ``--config`` round-trip all live here.
``TrainConfig.resolve(argv)`` is tested only for its process-boundary behaviour
(clean ``SystemExit`` on error, ``--dump-config`` print+exit), and ``cfg.save()``
for the reproducibility artifacts. See ``test_golden_cli_equivalence.py`` for the
``cfg.flatten()`` seam.
"""

import argparse

import pytest
import yaml
from pydantic import ValidationError

from speculators.train.config import ConfigError, TrainConfig, add_config_cli_arguments
from speculators.train.config.schema import _GROUPS, _ROOT_FIELDS


def _write(tmp_path, data, name="run.yaml") -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _staged(**train) -> dict:
    """Wrap a trainer config mapping in the canonical ``train:`` stage block."""
    return {"train": train}


# --------------------------------------------------------------------------- #
# Layering / precedence: flag > --set > YAML > default
# --------------------------------------------------------------------------- #


def test_default_when_no_source_sets_a_key():
    cfg = TrainConfig.from_sources(cli={"verifier_name_or_path": "v"})
    assert cfg.optimizer.lr == 1e-4  # coded default
    assert cfg.trainer.epochs == 20


def test_yaml_supplies_values(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "from-yaml"},
            optimizer={"lr": 0.001},
            trainer={"epochs": 7},
        ),
    )
    cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert cfg.verifier.verifier_name_or_path == "from-yaml"
    assert cfg.optimizer.lr == 0.001
    assert cfg.trainer.epochs == 7


def test_full_precedence_ladder(tmp_path):
    # lr is set at every layer; the flag must win. weight_decay only in YAML;
    # muon_ns_steps only via --set; muon_momentum nowhere -> default.
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            optimizer={"lr": 0.001, "weight_decay": 0.5},
        ),
    )
    cfg = TrainConfig.from_sources(
        cli={"lr": 0.02},
        overrides=["optimizer.lr=0.5", "optimizer.muon_ns_steps=7"],
        config_path=cfg_path,
    )
    assert cfg.optimizer.lr == 0.02  # flag beats --set beats YAML
    assert cfg.optimizer.muon_ns_steps == 7  # --set
    assert cfg.optimizer.weight_decay == 0.5  # YAML (untouched sibling)
    assert cfg.optimizer.muon_momentum == 0.95  # coded default


def test_set_beats_yaml_flag_beats_set(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(verifier={"verifier_name_or_path": "v"}, loss={"ttt_steps": 1}),
    )
    # --set overrides the YAML value...
    cfg = TrainConfig.from_sources(overrides=["loss.ttt_steps=5"], config_path=cfg_path)
    assert cfg.loss.ttt_steps == 5
    # ...and a flag overrides the --set value.
    cfg = TrainConfig.from_sources(
        cli={"ttt_steps": 9}, overrides=["loss.ttt_steps=5"], config_path=cfg_path
    )
    assert cfg.loss.ttt_steps == 9


def test_muon_lr_defaults_to_ten_times_lr():
    cfg = TrainConfig.from_sources(cli={"verifier_name_or_path": "v", "lr": 3e-4})
    assert cfg.optimizer.muon_lr == pytest.approx(3e-3)
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v", "lr": 3e-4, "muon_lr": 0.05}
    )
    assert cfg.optimizer.muon_lr == 0.05


def test_cli_list_replaces_yaml_list(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            speculator_type="dflash",
            draft={"full_attention_indices": [7, 8, 9]},
        ),
    )
    cfg = TrainConfig.from_sources(
        cli={"full_attention_indices": [0, 2]}, config_path=cfg_path
    )
    assert cfg.draft.full_attention_indices == [0, 2]  # full replace, not merge
    # YAML list survives when no higher layer sets it.
    cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert cfg.draft.full_attention_indices == [7, 8, 9]


# --------------------------------------------------------------------------- #
# --set escape hatch
# --------------------------------------------------------------------------- #


def test_set_parses_yaml_scalars():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v"},
        overrides=[
            "optimizer.lr=1e-4",  # float (pydantic coerces the YAML string)
            "draft.target_layer_ids=[2,18,33]",  # inline list
            "trainer.save_best=true",  # bool
            "logging.run_name=null",  # explicit null -> None
        ],
    )
    assert cfg.optimizer.lr == pytest.approx(1e-4)
    assert cfg.draft.target_layer_ids == [2, 18, 33]
    assert cfg.trainer.save_best is True


def test_set_accepts_train_prefix():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v"},
        overrides=["train.optimizer.muon_ns_steps=7"],
    )
    assert cfg.optimizer.muon_ns_steps == 7


def test_set_accepts_bare_root_scalar():
    cfg = TrainConfig.from_sources(
        overrides=["verifier.verifier_name_or_path=v", "seed=1234"]
    )
    assert cfg.seed == 1234


def test_set_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v"}, overrides=["not_a_key=1"]
        )


def test_set_misplaced_group_key_rejected():
    # num_layers is a real 'draft' field; addressing it under 'optimizer' must be
    # a hard error (a group-prefixed key whose group does not own the dest).
    with pytest.raises(ConfigError, match="has no key 'num_layers'"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v"}, overrides=["optimizer.num_layers=4"]
        )


def test_set_without_equals_rejected():
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v"}, overrides=["optimizer.lr"]
        )


# --------------------------------------------------------------------------- #
# train: stage wrapper
# --------------------------------------------------------------------------- #


def test_stage_wrapped_config_is_used_and_siblings_ignored(tmp_path):
    # A file authored for the future pipeline (train: alongside other stages)
    # still configures training today; sibling stage blocks are ignored, not
    # flagged as unknown keys.
    cfg_path = _write(
        tmp_path,
        {
            "prepare_data": {"some_key": 1},
            "launch_vllm": {"other": 2},
            "train": {"verifier": {"verifier_name_or_path": "staged"}, "seed": 7},
        },
    )
    cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert cfg.verifier.verifier_name_or_path == "staged"
    assert cfg.seed == 7


def test_legacy_bare_mapping_still_accepted(tmp_path):
    cfg_path = _write(
        tmp_path,
        {"verifier": {"verifier_name_or_path": "legacy"}, "optimizer": {"lr": 0.007}},
    )
    cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert cfg.verifier.verifier_name_or_path == "legacy"
    assert cfg.optimizer.lr == 0.007


def test_dump_yaml_emits_stage_wrapped_form():
    cfg = TrainConfig.from_sources(cli={"verifier_name_or_path": "v"})
    dumped = yaml.safe_load(cfg.dump_yaml())
    assert set(dumped) == {"train"}
    assert dumped["train"]["verifier"]["verifier_name_or_path"] == "v"


def test_unknown_yaml_key_warns_and_is_ignored(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            optimizer={"lr": 0.01, "not_a_real_field": 1},
        ),
    )
    with pytest.warns(UserWarning, match="unrecognised keys"):
        cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert cfg.optimizer.lr == 0.01


def test_misplaced_yaml_key_is_unrecognised_not_provided(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(verifier={"verifier_name_or_path": "v"}, optimizer={"num_layers": 4}),
    )
    with pytest.warns(UserWarning, match="unrecognised keys"):
        cfg = TrainConfig.from_sources(config_path=cfg_path)
    assert "num_layers" not in cfg.provenance.provided()
    assert cfg.draft.num_layers == 1  # misplaced value ignored; keeps its default


# --------------------------------------------------------------------------- #
# Provenance (typed cfg.provenance)
# --------------------------------------------------------------------------- #


def test_provenance_reports_winning_layer_per_key(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            optimizer={"lr": 0.001, "weight_decay": 0.3},
        ),
    )
    cfg = TrainConfig.from_sources(
        cli={"lr": 0.02}, overrides=["optimizer.muon_ns_steps=7"], config_path=cfg_path
    )
    prov = cfg.provenance
    assert prov.winner["lr"] == "flag"
    assert prov.winner["muon_ns_steps"] == "set"
    assert prov.winner["weight_decay"] == "yaml"
    assert prov.winner["muon_momentum"] == "default"
    # The full contributor trail is recorded in precedence order.
    assert prov.trail["lr"] == ("flag", "yaml")
    assert prov.provided() >= {"lr", "muon_ns_steps", "weight_decay"}
    assert "muon_momentum" not in prov.provided()


def test_provenance_all_default_for_off_argv_config():
    cfg = TrainConfig.from_flat(
        {"verifier_name_or_path": "v", "lr": 0.02, "speculator_type": "dflash"}
    )
    assert all(layer == "default" for layer in cfg.provenance.winner.values())
    assert cfg.provenance.provided() == set()


# --------------------------------------------------------------------------- #
# dump_yaml -> --config round-trip (the reproducibility contract)
# --------------------------------------------------------------------------- #


def test_dump_yaml_round_trips_through_config(tmp_path):
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v", "speculator_type": "peagle"},
        overrides=["peagle.num_depths=5"],
    )
    dumped = cfg.dump_yaml()
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(dumped)
    reloaded = TrainConfig.from_sources(config_path=str(cfg_path))
    assert reloaded.speculator_type == "peagle"
    assert reloaded.peagle.num_depths == 5
    assert reloaded.dump_yaml() == dumped  # dump -> load -> dump is a fixed point


# --------------------------------------------------------------------------- #
# Validation / errors (raised, not exited)
# --------------------------------------------------------------------------- #


def test_mismatched_algo_block_warns(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            speculator_type="eagle3",
            peagle={"num_depths": 4},
        ),
    )
    with pytest.warns(UserWarning, match="does not use the 'peagle' block"):
        TrainConfig.from_sources(config_path=cfg_path)


def test_draft_init_conflict_in_yaml_rejected(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            draft={"from_pretrained": "/ckpt", "num_layers": 4},
        ),
    )
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(config_path=cfg_path)


def test_draft_init_conflict_via_set_rejected():
    with pytest.raises(ConfigError, match="from-pretrained"):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v"},
            overrides=["draft.from_pretrained=/ckpt", "draft.num_layers=4"],
        )


def test_missing_required_verifier_raises_validation_error():
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(cli={"lr": 0.01})


def test_dpace_requires_ce_loss():
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v", "speculator_type": "dflash"},
            overrides=["dflash.per_position_loss_weight=dpace"],
        )


def test_invalid_checkpoint_freq_raises():
    with pytest.raises(ValidationError):
        TrainConfig.from_sources(
            cli={"verifier_name_or_path": "v", "checkpoint_freq": -1.0}
        )


# --------------------------------------------------------------------------- #
# resolve(argv): process-boundary behaviour (clean exits, --dump-config)
# --------------------------------------------------------------------------- #


def test_resolve_missing_verifier_names_the_flag(capsys):
    with pytest.raises(SystemExit):
        TrainConfig.resolve(["--lr", "0.01"])
    assert "--verifier-name-or-path" in capsys.readouterr().err


def test_resolve_broken_config_exits_two(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(
            ["--verifier-name-or-path", "v", "--config", "/no/such.yaml"]
        )
    assert exc.value.code == 2


def test_resolve_non_mapping_config_exits_cleanly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(SystemExit):
        TrainConfig.resolve(["--verifier-name-or-path", "v", "--config", str(path)])


def test_resolve_bad_set_key_exits_two(capsys):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--verifier-name-or-path", "v", "--set", "nope=1"])
    assert exc.value.code == 2
    assert "--set" in capsys.readouterr().err


def test_resolve_dump_config_prints_valid_config_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--verifier-name-or-path", "v", "--dump-config"])
    assert exc.value.code in (0, None)
    parsed = yaml.safe_load(capsys.readouterr().out)
    assert parsed["train"]["verifier"]["verifier_name_or_path"] == "v"


def test_resolve_dump_config_validates_before_emitting(tmp_path):
    cfg_path = _write(
        tmp_path,
        _staged(
            verifier={"verifier_name_or_path": "v"},
            draft={"from_pretrained": "/ckpt", "num_layers": 4},
        ),
    )
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--config", cfg_path, "--dump-config"])
    assert exc.value.code == 2


def test_resolve_set_on_cli(tmp_path):
    cfg = TrainConfig.resolve(
        ["--verifier-name-or-path", "v", "--set", "optimizer.muon_ns_steps=9"]
    )
    assert cfg.optimizer.muon_ns_steps == 9
    assert cfg.provenance.winner["muon_ns_steps"] == "set"


# --------------------------------------------------------------------------- #
# Reproducibility artifacts (cfg.save)
# --------------------------------------------------------------------------- #


def test_save_writes_run_yaml_command_and_provenance_sidecar(tmp_path):
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v", "lr": 0.02},
        overrides=["optimizer.muon_ns_steps=7"],
    )
    cfg.save(str(tmp_path))
    assert (tmp_path / "run.yaml").exists()
    assert (tmp_path / "train_command.txt").exists()
    sidecar_text = (tmp_path / "run.provenance.yaml").read_text()

    # run.yaml is clean (no provenance inlined) and re-runnable.
    reloaded = TrainConfig.from_sources(config_path=str(tmp_path / "run.yaml"))
    assert reloaded.optimizer.lr == 0.02
    assert reloaded.optimizer.muon_ns_steps == 7

    # The sidecar records the winning layer + trail per key; it is NOT a --config
    # input (it carries no top-level 'train:' config block, just the audit map).
    sidecar = yaml.safe_load(sidecar_text)
    assert sidecar["lr"]["winner"] == "flag"
    assert sidecar["muon_ns_steps"]["winner"] == "set"
    assert sidecar["muon_momentum"]["winner"] == "default"
    assert sidecar["muon_momentum"]["trail"] == ["default"]


def test_save_records_the_resolved_argv_in_the_manifest(tmp_path):
    # train_command.txt records the exact argv the config was resolved from, not
    # whatever sys.argv happens to be (reproducibility, user story 12 / 29).
    argv = ["scripts/train.py", "--verifier-name-or-path", "v", "--lr", "2e-4"]
    cfg = TrainConfig.from_sources(cli={"verifier_name_or_path": "v"}, argv=argv)
    cfg.save(str(tmp_path))
    manifest = (tmp_path / "train_command.txt").read_text()
    assert "scripts/train.py --verifier-name-or-path v --lr 2e-4" in manifest


# --------------------------------------------------------------------------- #
# flatten <-> from_flat (the phase-1 adapter seam)
# --------------------------------------------------------------------------- #


def test_flatten_from_flat_round_trip():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "v", "speculator_type": "dspark", "lr": 0.03}
    )
    flat = cfg.flatten()
    rebuilt = TrainConfig.from_flat(flat)
    assert rebuilt.flatten() == flat


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
    p = _generated_parser()
    seen: dict[str, str] = {}
    for action in p._actions:
        for opt in action.option_strings:
            assert opt not in seen, (
                f"flag {opt} generated by both {seen[opt]} and {action.dest}"
            )
            seen[opt] = action.dest
