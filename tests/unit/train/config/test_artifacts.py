"""Serialization / reproducibility seam tests.

The reproducibility contract is the round-trip at the ``from_sources`` seam:
``dump_yaml()`` -> re-load via ``config_path`` -> an identical resolved config.
Also asserted here: the emitted YAML is ``train:``-wrapped and clean (no
provenance inlined, re-loadable), ``cfg.save(dir)`` writes both artifacts, and
``--dump-config`` prints a valid config and exits cleanly at the ``resolve``
boundary. No golden ``vars(args)`` snapshot.
"""

import warnings

import pytest
import yaml

from speculators.train.config import TrainConfig


def _resolved(**cli):
    # Silence the mismatched-algorithm-block warnings that fire when a cross-type
    # recipe is resolved; they are exercised in test_resolution.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not use the.*algorithm group.*",
            category=UserWarning,
        )
        return TrainConfig.from_sources(cli=cli, config_path=None, argv=["train.py"])


def _reload(tmp_path, cfg):
    path = tmp_path / "run.yaml"
    path.write_text(cfg.dump_yaml())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not use the.*algorithm group.*",
            category=UserWarning,
        )
        return TrainConfig.from_sources(
            cli={}, config_path=str(path), argv=["train.py"]
        )


# --------------------------------------------------------------------------- #
# dump_yaml: the round-trip reproducibility contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cli",
    [
        {"verifier_name_or_path": "m"},
        {"verifier_name_or_path": "m", "lr": 0.9, "num_layers": 4},
        # --from-pretrained is the case a full materialized dump could not survive:
        # persisting the derived decoder-shaping defaults would self-reject on reload.
        {"verifier_name_or_path": "m", "from_pretrained": "ckpt"},
        {"verifier_name_or_path": "m", "speculator_type": "dspark", "markov_rank": 128},
        # Non-scalar recipe values: a nargs list and a JSON-dict loss spec (both
        # appear verbatim in examples/train/*.sh) must survive the YAML round-trip.
        {"verifier_name_or_path": "m", "target_layer_ids": [2, 5, 9]},
        {"verifier_name_or_path": "m", "loss_fn": '{"ce": 0.1, "tv": 0.9}'},
    ],
)
def test_dump_yaml_round_trips_to_identical_resolved_config(tmp_path, cli):
    cfg = _resolved(**cli)
    assert cfg.flatten() == _reload(tmp_path, cfg).flatten()


def test_dump_yaml_is_byte_stable_across_reload(tmp_path):
    # Emission follows the pinned flatten() order, not the provenance dict order
    # (which varies with flag-vs-yaml source), so dumping a reloaded config yields
    # byte-identical YAML -- the run.yaml artifact is stable, not just equivalent.
    cfg = _resolved(verifier_name_or_path="m", lr=0.9, num_layers=4, epochs=7)
    first = cfg.dump_yaml()
    assert _reload(tmp_path, cfg).dump_yaml() == first


def test_dump_yaml_is_stage_wrapped_and_clean(tmp_path):
    cfg = _resolved(verifier_name_or_path="m", lr=0.9)
    doc = yaml.safe_load(cfg.dump_yaml())
    # Canonical stage shape: a single top-level ``train:`` block.
    assert set(doc) == {"train"}
    # Clean: only supplied values, no provenance annotations, no default noise.
    assert doc["train"] == {
        "verifier": {"verifier_name_or_path": "m"},
        "optimizer": {"lr": 0.9},
    }


def test_saved_run_yaml_reloads_to_identical_config(tmp_path):
    # The file written next to the checkpoints re-loads (via the ``config_path``
    # seam) to the same resolved config.
    cfg = _resolved(verifier_name_or_path="m", num_layers=6, lr=0.3)
    cfg.save(str(tmp_path))
    reran = TrainConfig.from_sources(
        cli={}, config_path=str(tmp_path / "run.yaml"), argv=["train.py"]
    )
    assert reran.flatten() == cfg.flatten()


# --------------------------------------------------------------------------- #
# save: the reproducibility artifacts next to the checkpoints
# --------------------------------------------------------------------------- #


def test_save_writes_run_yaml_and_train_command(tmp_path):
    cfg = _resolved(verifier_name_or_path="m")
    cfg.save(str(tmp_path))
    assert (tmp_path / "run.yaml").exists()
    manifest = (tmp_path / "train_command.txt").read_text()
    # The manifest records the argv this config was resolved from + the environment.
    assert "train.py" in manifest
    assert "# Git SHA:" in manifest
    assert "# World size:" in manifest
    for pkg in ("speculators", "transformers", "torch"):
        assert f"# {pkg}:" in manifest


def test_no_provenance_sidecar_is_written(tmp_path):
    _resolved(verifier_name_or_path="m").save(str(tmp_path))
    assert not (tmp_path / "run.provenance.yaml").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"run.yaml", "train_command.txt"}


def test_dump_yaml_handles_from_flat_config_without_provenance(tmp_path):
    # A config rebuilt via from_flat carries no layer provenance; dump_yaml/save
    # must still work (the old code KeyError'd on the empty provenance dict),
    # falling back to "differs from defaults" to pick the emitted keys.
    resolved = _resolved(
        verifier_name_or_path="m",
        speculator_type="eagle3",
        save_path="/tmp/ckpt",
        epochs=7,
        lr=0.3,
    )
    rebuilt = TrainConfig.from_flat(resolved.flatten())
    # Precondition that triggered the old KeyError.
    assert rebuilt.provenance == {}

    doc = rebuilt.dump_yaml()
    assert doc
    assert "train:" in doc
    # The customized (differs-from-default) values survive into the emitted YAML.
    assert "save_path" in doc
    assert "/tmp/ckpt" in doc

    rebuilt.save(str(tmp_path))
    assert (tmp_path / "run.yaml").exists()


# --------------------------------------------------------------------------- #
# --dump-config: the config-out surface at the resolve boundary
# --------------------------------------------------------------------------- #


def test_dump_config_prints_valid_config_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(
            ["--verifier-name-or-path", "m", "--lr", "0.5", "--dump-config"]
        )
    assert exc.value.code == 0
    printed = capsys.readouterr().out
    doc = yaml.safe_load(printed)
    assert doc["train"]["optimizer"]["lr"] == 0.5
    assert doc["train"]["verifier"]["verifier_name_or_path"] == "m"


def test_dump_config_output_is_a_reloadable_config(tmp_path, capsys):
    with pytest.raises(SystemExit):
        TrainConfig.resolve(
            ["--verifier-name-or-path", "m", "--num-layers", "7", "--dump-config"]
        )
    path = tmp_path / "run.yaml"
    path.write_text(capsys.readouterr().out)
    reran = TrainConfig.from_sources(cli={}, config_path=str(path), argv=["train.py"])
    assert reran.draft.num_layers == 7
    assert reran.verifier.verifier_name_or_path == "m"


def test_dump_config_includes_config_file_values(tmp_path, capsys):
    # A value supplied only in --config surfaces in the --dump-config output.
    config = tmp_path / "run.yaml"
    config.write_text("train:\n  optimizer:\n    lr: 0.2\n")
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(
            ["--verifier-name-or-path", "m", "--config", str(config), "--dump-config"]
        )
    assert exc.value.code == 0
    doc = yaml.safe_load(capsys.readouterr().out)
    assert doc["train"]["optimizer"]["lr"] == 0.2
