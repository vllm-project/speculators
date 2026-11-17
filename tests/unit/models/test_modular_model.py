"""Model recipes compose backbones and corrections without pairing subclasses."""

import json
from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file

from speculators.model import SpeculatorModel
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer
from speculators.models.modular import (
    BlockRecipe,
    ModularDraftModel,
    ModularSpeculatorConfig,
)
from tests.unit.models.test_dflash_family_contracts import (
    RECIPES,
    REFERENCE_PATH,
    assert_report_close,
    contract_report,
    make_contract_model,
)


def modular_config(
    backbone="dflash2", correction="markov", *, confidence=False, sample=False
):
    source = make_contract_model("dflash", sample)
    excluded = {"speculators_model_type", "architectures", "checkpoint_format_version"}
    values = {
        name: deepcopy(getattr(source.config, name))
        for name in DFlashSpeculatorConfig.model_fields
        if name not in excluded
    }
    values["speculators_config"].algorithm = "modular"
    return ModularSpeculatorConfig(
        **values,
        recipe=BlockRecipe.model_validate(
            {
                "backbone": {"kind": backbone, "conv_group_size": 4},
                "correction": {"kind": correction, "rank": 4, "selector_top_k": 4},
                "confidence": {
                    "enabled": confidence,
                    "with_correction_features": confidence and correction == "markov",
                },
            }
        ),
    )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("sample", [False, True])
def test_named_presets_match_legacy_numerical_contract(recipe, sample):
    correction = {
        "dflash": "identity",
        "dflash2": "selector",
        "dspark": "markov",
        "xpress": "xpress",
    }[recipe]
    config = modular_config(
        "dflash2" if recipe == "dflash2" else "dflash",
        correction,
        confidence=recipe == "dspark",
        sample=sample,
    )
    model = ModularDraftModel(config)
    model.load_state_dict(make_contract_model(recipe, sample).state_dict())
    reference = json.loads(REFERENCE_PATH.read_text())
    assert_report_close(
        contract_report(recipe, sample, model=model),
        reference["recipes"][f"{recipe}:{sample}"],
    )


@pytest.mark.parametrize("backbone", ["dflash", "dflash2"])
@pytest.mark.parametrize("correction", ["identity", "markov", "selector", "xpress"])
def test_every_block_pairing_trains(backbone, correction):
    config = modular_config(backbone, correction, confidence=correction == "markov")
    model = ModularDraftModel(config)
    SpeculatorModel.verify_training_compatible(model)
    if backbone == "dflash2":
        layer = model.get_decoder_layers()[0]
        assert isinstance(layer, Qwen3DFlash2DecoderLayer)
        conv = layer.attention_conv
        assert torch.all(conv.base_kernel[:, 0] == 1)
        assert torch.all(conv.base_kernel[:, 1:] == 0)
    with torch.no_grad():
        for parameter in (
            model.embed_tokens.weight,
            model.lm_head.weight,
            model.verifier_lm_head.weight,
        ):
            parameter.normal_()
        model.verifier_norm.weight.fill_(1)
    train, val = model.get_trainer_kwargs(
        model_config=config,
        loss_fn="kl_div",
        loss_implementation="eager",
        max_anchors=2,
    )
    assert train.keys() == val.keys()
    _, loss, _ = model(
        torch.randn(1, 16, 32),
        torch.arange(16).view(1, -1),
        torch.ones(1, 16),
        torch.randn(1, 16, 16),
        torch.zeros(1, 16, dtype=torch.long),
        **train,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name


def test_mixed_native_checkpoint_roundtrip_and_export_rejection(tmp_path, monkeypatch):
    model = ModularDraftModel(modular_config(confidence=True))
    frozen = {
        name: model.get_parameter(name).detach().clone()
        for name in model._keys_to_ignore_on_save
    }

    def restore(loaded):
        with torch.no_grad():
            for name, tensor in frozen.items():
                loaded.get_parameter(name).copy_(tensor)

    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", restore)
    model.save_pretrained(tmp_path / "native", max_shard_size="1KB")
    loaded = SpeculatorModel.from_pretrained(tmp_path / "native", local_files_only=True)
    assert isinstance(loaded, ModularDraftModel)
    assert loaded.config.recipe == model.config.recipe
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, equal_nan=True)
    with pytest.raises(ValueError, match="no legacy serving export"):
        model.save_pretrained(tmp_path / "unsupported", checkpoint_format="legacy")
    assert not (tmp_path / "unsupported").exists()


def test_mixed_objective_compiles_with_dynamic_validity():
    model = ModularDraftModel(modular_config(confidence=True))
    # Identity convolution initialization makes the remaining shared weights
    # directly reusable from the fixed DSpark reference fixture.
    model.load_state_dict(
        make_contract_model("dspark", False).state_dict(), strict=False
    )
    compiled = torch.compile(model, backend="eager")
    kwargs, _ = model.get_trainer_kwargs(
        model_config=model.config,
        loss_fn="kl_div",
        loss_implementation="eager",
        max_anchors=2,
    )
    hidden = torch.randn(1, 12, 32)
    tokens = torch.arange(12).view(1, -1)
    verifier = torch.randn(1, 12, 16)
    documents = torch.zeros(1, 12, dtype=torch.long)
    for mask in (torch.ones(1, 12), torch.zeros(1, 12)):
        torch.manual_seed(42)
        _, expected, _ = model(hidden, tokens, mask, verifier, documents, **kwargs)
        torch.manual_seed(42)
        _, actual, _ = compiled(hidden, tokens, mask, verifier, documents, **kwargs)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
@pytest.mark.parametrize("confidence", ["disabled", "hidden", "correction"])
def test_mixed_markov_modes_and_confidence_features(head_type, confidence):
    values = modular_config().to_dict()
    values["recipe"]["correction"]["markov_head_type"] = head_type
    values["recipe"]["confidence"] = {
        "enabled": confidence != "disabled",
        "with_correction_features": confidence == "correction",
    }
    config = ModularSpeculatorConfig(**values)
    # Hugging Face intentionally omits the attention implementation on save.
    config.transformer_layer_config._attn_implementation = "eager"
    model = ModularDraftModel(config)
    with torch.no_grad():
        for parameter in (
            model.embed_tokens.weight,
            model.lm_head.weight,
            model.verifier_lm_head.weight,
        ):
            parameter.normal_()
        model.verifier_norm.weight.fill_(1)
    kwargs, _ = model.get_trainer_kwargs(
        model_config=model.config,
        loss_fn="kl_div",
        loss_implementation="eager",
        max_anchors=2,
    )
    _, loss, _ = model(
        torch.randn(1, 12, 32),
        torch.arange(12).view(1, -1),
        torch.ones(1, 12),
        torch.randn(1, 12, 16),
        torch.zeros(1, 12, dtype=torch.long),
        **kwargs,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_known_recipe_exports_real_legacy_configuration(
    recipe, dtype, tmp_path, monkeypatch
):
    correction = {
        "dflash": "identity",
        "dflash2": "selector",
        "dspark": "markov",
        "xpress": "xpress",
    }[recipe]
    model = ModularDraftModel(
        modular_config(
            "dflash2" if recipe == "dflash2" else "dflash",
            correction,
            confidence=recipe == "dspark",
        )
    )
    model.load_state_dict(make_contract_model(recipe, False).state_dict())
    model.to(dtype)
    before = model.config.to_dict()
    model.save_pretrained(tmp_path, checkpoint_format="legacy")
    assert model.config.to_dict() == before
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["speculators_model_type"] == recipe
    assert config["speculators_config"]["algorithm"] == recipe
    assert "recipe" not in config
    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", lambda self: None)
    loaded = SpeculatorModel.from_pretrained(
        tmp_path, local_files_only=True, dtype=dtype
    )
    assert type(loaded) is RECIPES[recipe][0]
    assert isinstance(loaded, DFlashDraftModel)
    saved = load_file(tmp_path / "model.safetensors")
    assert saved["fc.weight"].dtype == dtype
    assert config["dtype"] == str(dtype).removeprefix("torch.")
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(loaded.backbone.state_dict()[name], tensor)


@pytest.mark.parametrize(
    "recipe",
    [
        {"correction": {"kind": "markov"}, "objective": "xpress"},
        {"correction": {"kind": "selector"}, "confidence": {"enabled": True}},
        {"confidence": {"enabled": True, "with_correction_features": True}},
        {"correction": {"kind": "identity", "rank": 0}},
        {"correction": {"kind": "unknown"}},
        {"correction": {"kind": "identity", "typo": 1}},
    ],
)
def test_incompatible_recipes_fail_before_construction(recipe):
    with pytest.raises(ValueError):
        BlockRecipe.model_validate(recipe)


@pytest.mark.parametrize(
    "change", ["groups", "kernel", "vocab", "top_k", "targets", "hidden"]
)
def test_incompatible_dimensions_are_rejected(change):
    values = modular_config(correction="selector").to_dict()
    if change == "groups":
        values["recipe"]["backbone"]["conv_group_size"] = 3
    elif change == "kernel":
        values["recipe"]["backbone"]["conv_kernel_size"] = 5
    elif change == "vocab":
        values["draft_vocab_size"] = 16
    elif change == "top_k":
        values["recipe"]["correction"]["selector_top_k"] = 33
    elif change == "hidden":
        values["target_hidden_size"] = 8
    else:
        values["aux_hidden_state_layer_ids"] = []
    with pytest.raises(ValueError):
        ModularSpeculatorConfig(**values)
