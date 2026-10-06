"""DFlash wire-format compatibility and checkpoint round trips."""

import json
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest
import torch
from transformers import Qwen3Config

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.convert.dflash.converter import DFlashConverter
from speculators.convert.entrypoints import maybe_convert_external_checkpoint
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash2 import DFlash2SpeculatorConfig
from tests.unit.convert.test_dflash_converter import _tiny_dflash_config


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_save_matches_schema_and_config_round_trip(tmp_path, sample_from_anchor):
    config = _tiny_dflash_config()
    config.sample_from_anchor = sample_from_anchor
    config.sliding_window_non_causal = True
    config.aux_hidden_state_layer_ids = [1, 3]
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    schema_path = (
        Path(__file__).parents[2] / "fixtures/open_spec_config/dflash.schema.json"
    )
    jsonschema.validate(saved, json.loads(schema_path.read_text()))
    assert "speculators_model_type" not in saved
    assert "transformer_layer_config" not in saved
    assert saved["architectures"] == ["DflashDraftModel"]
    assert saved["speculative_config"]["speculative_tokens"] == (
        4 if sample_from_anchor else 3
    )
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, DFlashSpeculatorConfig)
    assert loaded.block_size == 4
    assert loaded.aux_hidden_state_layer_ids == [1, 3]
    assert loaded.sample_from_anchor == sample_from_anchor
    assert loaded.sliding_window_non_causal is True
    assert loaded.mask_token_id == 1
    assert loaded.draft_vocab_size == 32
    assert loaded.transformer_layer_config.hidden_size == 16
    assert json.loads(json.dumps(loaded.to_dict())) == saved


def test_legacy_speculators_config_still_loads_and_saves_new_format(tmp_path):
    config = _tiny_dflash_config()
    legacy = super(DFlashSpeculatorConfig, config).to_dict()
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert loaded.aux_hidden_state_layer_ids == config.aux_hidden_state_layer_ids
    loaded.save_pretrained(tmp_path)
    assert (
        json.loads((tmp_path / "config.json").read_text())["open_spec_config_version"]
        == "0.0.0"
    )


@pytest.mark.parametrize(("start_idx", "expected"), [(0, [2, 4]), (1, [1, 3])])
def test_external_open_spec_config_layer_indexing(start_idx, expected):
    wire = _tiny_dflash_config().to_dict()
    wire.pop("speculators_metadata")
    wire["speculative_config"]["target_layer_ids"] = [1, 3]
    wire["speculative_config"]["target_layer_start_idx"] = start_idx
    loaded = SpeculatorModelConfig.from_dict(wire)
    assert loaded.aux_hidden_state_layer_ids == expected
    with patch("speculators.convert.entrypoints.convert_model") as convert:
        assert (
            maybe_convert_external_checkpoint("checkpoint", config_dict=wire)
            == "checkpoint"
        )
        convert.assert_not_called()


@pytest.mark.parametrize(
    "updates",
    [
        {"speculative_tokens": 0},
        {"mask_token_id": -1},
        {"target_layer_ids": [1, 1]},
        {"target_layer_ids": [-1]},
        {"target_layer_start_idx": 2},
        {"unexpected": True},
        {"verifier": None},
        {"draft_vocab_size": 0},
    ],
)
def test_invalid_open_spec_settings_rejected(updates):
    wire = _tiny_dflash_config().to_dict()
    wire["speculative_config"].update(updates)
    with pytest.raises(ValueError):
        SpeculatorModelConfig.from_dict(wire)


@pytest.mark.parametrize(
    "updates",
    [
        {"open_spec_config_version": "1.0.0"},
        {"architectures": ["DFlashDraftModel"]},
    ],
)
def test_invalid_open_spec_header_rejected(updates):
    wire = _tiny_dflash_config().to_dict()
    wire.update(updates)
    with pytest.raises(ValueError):
        SpeculatorModelConfig.from_dict(wire)


def test_optional_open_spec_fields_can_be_omitted():
    wire = _tiny_dflash_config().to_dict()
    wire.pop("speculators_metadata")
    for key in (
        "verifier",
        "draft_vocab_size",
        "training_framework",
        "training_framework_version",
    ):
        wire["speculative_config"].pop(key)
    loaded = SpeculatorModelConfig.from_dict(wire)
    assert loaded.draft_vocab_size == loaded.target_vocab_size
    assert loaded.speculators_config.verifier.name_or_path is None


@patch("speculators.convert.dflash.converter.PretrainedConfig.get_config_dict")
def test_converter_accepts_open_spec_config(mock_config):
    wire = _tiny_dflash_config().to_dict()
    mock_config.return_value = ({"architectures": ["Qwen3ForCausalLM"]}, {})
    loaded = DFlashConverter()._build_config(wire, "new/verifier", None)
    assert loaded.speculators_config.verifier.name_or_path == "new/verifier"
    assert loaded.aux_hidden_state_layer_ids == [0]
    assert loaded.block_size == 4


def test_variant_keeps_its_format():
    native = super(DFlashSpeculatorConfig, _tiny_dflash_config()).to_dict()
    native["speculators_model_type"] = "dflash2"
    config = DFlash2SpeculatorConfig(**native)
    assert config.to_dict()["speculators_model_type"] == "dflash2"
    assert "open_spec_config_version" not in config.to_dict()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("draft_vocab_size", [16, 32])
def test_model_checkpoint_round_trip_without_conversion(
    tmp_path, legacy, draft_vocab_size
):
    config = _tiny_dflash_config()
    verifier_path = tmp_path / "verifier"
    Qwen3Config(hidden_size=16).save_pretrained(verifier_path)
    config.speculators_config.verifier.name_or_path = str(verifier_path)
    config.draft_vocab_size = draft_vocab_size
    model = DFlashDraftModel(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.fc.weight.fill_(0.125)
    model.save_pretrained(tmp_path)
    if not legacy:
        wire = json.loads((tmp_path / "config.json").read_text())
        wire["speculative_config"].pop("verifier")
        (tmp_path / "config.json").write_text(json.dumps(wire))
    if legacy:
        (tmp_path / "config.json").write_text(
            json.dumps(super(DFlashSpeculatorConfig, model.config).to_dict())
        )
    with (
        patch.object(DFlashDraftModel, "load_verifier_weights"),
        patch("speculators.convert.entrypoints.convert_model") as convert,
    ):
        loaded = SpeculatorModel.from_pretrained(
            tmp_path, dtype="auto", verifier=str(verifier_path)
        )
    assert isinstance(loaded, DFlashDraftModel)
    torch.testing.assert_close(loaded.fc.weight, model.fc.weight)
    assert loaded.config.draft_vocab_size == draft_vocab_size
    convert.assert_not_called()
