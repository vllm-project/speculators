"""Training configuration preserves component choices and objective settings."""

from pathlib import Path

import pytest

from speculators.models.dflash import DFlashDraftModel
from speculators.models.dflash2.model_definitions import Qwen3DFlash2DecoderLayer
from speculators.models.modular import ModularDraftModel
from speculators.train.cli import _build_from_config_only
from speculators.train.config import TrainConfig
from speculators.train.config.resolution import ConfigError
from tests.unit.models.test_modular_model import modular_config

EXAMPLE = (
    Path(__file__).resolve().parents[4] / "examples/train/modular_dflash2_markov.yaml"
)


def test_example_recipe_resolves_and_roundtrips(tmp_path):
    config = TrainConfig.from_sources(cli={}, config_path=str(EXAMPLE), argv=["train"])
    assert config.speculator_type == "modular"
    assert config.draft.composition_config is not None
    assert config.draft.composition_config.endswith("dflash2_markov.yaml")
    assert config.dflash.sample_from_anchor is True
    assert config.draft.num_layers == 2
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    restored = TrainConfig.from_sources(cli={}, config_path=str(path), argv=["train"])
    assert restored.flatten() == config.flatten()


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculator_type": "dflash", "composition_config": "recipe.yaml"},
        {"from_pretrained": "saved", "composition_config": "recipe.yaml"},
        {"markov_rank": 12},
        {"conv_group_size": 4},
        {"from_pretrained": "saved", "sample_from_anchor": True},
        {"from_pretrained": "saved", "block_size": 4},
    ],
)
def test_ambiguous_architecture_overrides_are_rejected(overrides):
    with pytest.raises(ConfigError):
        TrainConfig.from_sources(
            cli={
                "verifier_name_or_path": "local",
                "speculator_type": "modular",
                **overrides,
            },
            argv=["train"],
        )


def test_from_training_args_reads_component_file(tmp_path, monkeypatch):
    config = modular_config()
    verifier = tmp_path / "verifier"
    config.transformer_layer_config.save_pretrained(verifier)
    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", lambda self: None)
    recipe = EXAMPLE.parent / "recipes/dflash2_markov.yaml"
    model = ModularDraftModel.from_training_args(
        config.transformer_layer_config,
        composition_config=str(recipe),
        verifier_name_or_path=str(verifier),
        target_layer_ids=[0, 1],
        draft_vocab_size=32,
        block_size=4,
        mask_token_id=0,
    )
    assert isinstance(model.get_decoder_layers()[0], Qwen3DFlash2DecoderLayer)
    assert model.config.recipe.correction.kind == "markov"
    assert model.confidence_head is not None
    kwargs, _ = model.get_trainer_kwargs(
        model_config=model.config,
        loss_fn="kl_div",
        loss_implementation="eager",
        confidence_head_alpha=0.25,
    )
    assert kwargs["confidence_head_alpha"] == 0.25


def test_config_only_initialization_preserves_recipe(tmp_path, monkeypatch):
    config = modular_config(confidence=True)
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(DFlashDraftModel, "load_verifier_weights", lambda self: None)
    model = _build_from_config_only(
        ModularDraftModel, str(tmp_path), None, None, draft_attn_impl="eager"
    )
    assert isinstance(model, ModularDraftModel)
    assert model.config.recipe == config.recipe
    assert model.config.checkpoint_format_version == 1
