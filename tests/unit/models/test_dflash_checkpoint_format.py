"""Legacy serving names and native module paths share one weight mapping."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash.checkpoint import export_state_dict, import_state_dict
from speculators.train import checkpointer as checkpointer_module
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from tests.unit.models.test_dflash_family_contracts import RECIPES, make_contract_model


@pytest.fixture(autouse=True)
def cpu_checkpoints(monkeypatch):
    monkeypatch.setattr(checkpointer_module, "get_current_device", lambda: "cpu")


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("checkpoint_format", ["legacy", "native"])
@pytest.mark.parametrize("sharded", [False, True])
def test_pretrained_roundtrip(
    recipe, checkpoint_format, sharded, tmp_path, monkeypatch
):
    model = make_contract_model(recipe, False)
    before = model.config.to_dict()
    state = model.state_dict()
    frozen = {name: state[name] for name in model._keys_to_ignore_on_save}

    def restore_verifier(loaded):
        with torch.no_grad():
            for name, tensor in frozen.items():
                loaded.get_parameter(name).copy_(tensor)

    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", restore_verifier)
    model.save_pretrained(
        tmp_path,
        state_dict=state,
        checkpoint_format=checkpoint_format,
        max_shard_size="1KB" if sharded else "1GB",
    )
    assert model.config.to_dict() == before
    assert "backbone.fc.weight" in state  # supplied dictionaries are not consumed
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["checkpoint_format_version"] == (
        1 if checkpoint_format == "native" else None
    )
    expected_key = (
        "backbone.fc.weight" if checkpoint_format == "native" else "fc.weight"
    )
    if sharded:
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        assert expected_key in index["weight_map"]
    else:
        assert expected_key in load_file(tmp_path / "model.safetensors")
    loaded = SpeculatorModel.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is type(model)
    for name, tensor in state.items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)
    # The selected native/legacy format persists across another default save.
    loaded.save_pretrained(tmp_path / "again")
    assert expected_key in load_file(tmp_path / "again" / "model.safetensors")


@pytest.mark.parametrize("recipe", RECIPES)
def test_named_recipes_default_to_legacy_export(recipe, tmp_path):
    model = make_contract_model(recipe, False)
    model.save_pretrained(tmp_path)
    exported = load_file(tmp_path / "model.safetensors")
    assert "fc.weight" in exported
    assert not any(key.startswith("backbone.") for key in exported)
    if recipe == "dflash2":
        assert "layers.0.attention_conv.base_kernel" in exported


def test_import_rejects_collisions_and_mixed_layouts():
    tensor = torch.ones(16)
    with pytest.raises(ValueError, match="collide"):
        import_state_dict({"norm.weight": tensor, "backbone.norm.weight": tensor})
    with pytest.raises(ValueError, match="mixes"):
        import_state_dict(
            {"norm.weight": tensor, "backbone.hidden_norm.weight": tensor}
        )


def test_import_shares_storage_and_preserves_module_metadata():
    model = make_contract_model("dflash", False)
    state = model.state_dict()
    legacy = export_state_dict(state, native=False)
    imported = import_state_dict(legacy)
    assert imported["backbone.fc.weight"] is state["backbone.fc.weight"]
    with_metadata = import_state_dict(state)
    assert with_metadata._metadata == state._metadata


def test_pretrained_rejects_missing_body_weights(tmp_path):
    model = make_contract_model("dflash", False)
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    del state["fc.weight"]
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing learned weights.*backbone.fc"):
        DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True)


def test_custom_pretrained_key_mapping_is_validated(tmp_path, monkeypatch):
    model = make_contract_model("dflash", False)
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    state["projection.weight"] = state.pop("fc.weight")
    save_file(state, tmp_path / "model.safetensors")
    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", lambda self: None)
    loaded = DFlashDraftModel.from_pretrained(
        tmp_path,
        local_files_only=True,
        key_mapping={r"^projection\.": "backbone.fc."},
    )
    torch.testing.assert_close(loaded.backbone.fc.weight, model.backbone.fc.weight)
    with pytest.raises(ValueError, match="missing learned weights"):
        DFlashDraftModel.from_pretrained(
            tmp_path,
            local_files_only=True,
            key_mapping={r"^projection\.": "unknown."},
        )


def test_pretrained_rejects_collision_across_shards(tmp_path):
    model = make_contract_model("dflash", False)
    model.save_pretrained(tmp_path, max_shard_size="1KB")
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    save_file(
        {"backbone.fc.weight": model.backbone.fc.weight}, tmp_path / "extra.safetensors"
    )
    index["weight_map"]["backbone.fc.weight"] = "extra.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="collide"):
        DFlashDraftModel.from_pretrained(tmp_path, local_files_only=True)


def test_unknown_format_version_is_rejected():
    with pytest.raises(ValueError):
        DFlashSpeculatorConfig(checkpoint_format_version=2)  # type: ignore[arg-type]


@pytest.mark.parametrize("distributed", [False, True])
def test_checkpointer_maps_legacy_weights(distributed, tmp_path, monkeypatch):
    source = make_contract_model("dflash", False)
    optimizer = torch.optim.AdamW(source.parameters())
    SingleGPUCheckpointer(tmp_path).save_checkpoint(
        source, optimizer, epoch=0, float_dtype=torch.float32
    )
    target = make_contract_model("dflash", False)
    with torch.no_grad():
        target.backbone.fc.weight.zero_()
    if distributed:

        def load(model, state, **kwargs):
            assert "backbone.fc.weight" in state
            assert "fc.weight" not in state
            model.load_state_dict(state, strict=False)

        monkeypatch.setattr(checkpointer_module, "set_model_state_dict", load)
        monkeypatch.setattr(checkpointer_module.dist, "barrier", lambda: None)
        checkpointer = DistributedCheckpointer(tmp_path)
    else:
        checkpointer = SingleGPUCheckpointer(tmp_path)  # type: ignore[assignment]
    checkpointer.load_model_state_dict(target)
    torch.testing.assert_close(target.backbone.fc.weight, source.backbone.fc.weight)
