"""Regression tests for variable Eagle-family auxiliary-layer widths."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _transformer_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        _attn_implementation="eager",
    )


def _speculators_config(algorithm: str) -> SpeculatorsConfig:
    return SpeculatorsConfig(
        algorithm=algorithm,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path=None,
            architectures=["LlamaForCausalLM"],
        ),
    )


def _eagle3_config(layer_ids: list[int] | None) -> Eagle3SpeculatorConfig:
    return Eagle3SpeculatorConfig(
        transformer_layer_config=_transformer_config(),
        draft_vocab_size=64,
        norm_before_residual=False,
        embed_requires_grad=False,
        eagle_aux_hidden_state_layer_ids=layer_ids,
        speculators_config=_speculators_config("eagle3"),
    )


def _peagle_config(layer_ids: list[int] | None) -> PEagleSpeculatorConfig:
    return PEagleSpeculatorConfig(
        transformer_layer_config=_transformer_config(),
        draft_vocab_size=64,
        norm_before_residual=False,
        embed_requires_grad=True,
        eagle_aux_hidden_state_layer_ids=layer_ids,
        mask_token_id=0,
        speculators_config=_speculators_config("peagle"),
    )


@pytest.mark.parametrize("layer_ids", [[1], [1, 2], [1, 2, 3]])
def test_eagle3_fc_width_tracks_auxiliary_layer_count(layer_ids: list[int]):
    """Eagle3 projection width is dynamic rather than fixed at three layers."""
    config = _eagle3_config(layer_ids)

    model = Eagle3DraftModel(config)

    assert model.fc.in_features == len(layer_ids) * model.hidden_size


def test_eagle3_none_aux_layers_preserve_legacy_three_layer_default():
    """None remains the backward-compatible signal for the three-layer default."""
    config = _eagle3_config(None)

    model = Eagle3DraftModel(config)

    assert config.eagle_aux_hidden_state_layer_ids is None
    assert model.fc.in_features == 3 * model.hidden_size


def test_weight_bearing_legacy_checkpoint_keeps_matching_three_layer_width(tmp_path):
    """Normal legacy checkpoints resolve IDs without changing their saved FC width."""
    config = _eagle3_config(None)
    config.speculators_config.verifier.name_or_path = "unused-verifier"
    model = Eagle3DraftModel(config)
    model.save_pretrained(tmp_path)

    with (
        patch.object(Eagle3DraftModel, "load_verifier_weights"),
        patch(
            "speculators.models.utils.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(num_hidden_layers=36),
        ) as load_config,
    ):
        loaded = Eagle3DraftModel.from_pretrained(
            tmp_path,
            verifier="must-not-override-saved-verifier",
        )

    assert loaded.target_layer_ids == [2, 18, 33]
    assert loaded.fc.in_features == 3 * loaded.hidden_size
    assert loaded.config.speculators_config.verifier.name_or_path == "unused-verifier"
    load_config.assert_called_once_with("unused-verifier")


@pytest.mark.parametrize(
    ("model_class", "config_factory", "saved_verifier_path"),
    [
        (Eagle3DraftModel, _eagle3_config, None),
        (Eagle3DraftModel, _eagle3_config, ""),
        (PEagleDraftModel, _peagle_config, None),
        (PEagleDraftModel, _peagle_config, ""),
    ],
    ids=["eagle3-none", "eagle3-blank", "peagle-none", "peagle-blank"],
)
def test_weight_bearing_legacy_checkpoint_uses_missing_verifier_fallback(
    tmp_path,
    model_class,
    config_factory,
    saved_verifier_path,
):
    """The training CLI's verifier repairs omitted metadata without overriding it."""
    config = config_factory(None)
    config.speculators_config.verifier.name_or_path = saved_verifier_path
    model = model_class(config)
    model.save_pretrained(tmp_path)

    with (
        patch.object(model_class, "load_verifier_weights") as load_weights,
        patch(
            "speculators.models.utils.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(num_hidden_layers=36),
        ) as load_config,
    ):
        loaded = model_class.from_pretrained(
            tmp_path,
            verifier="trusted-verifier-fallback",
        )

    assert loaded.config.speculators_config.verifier.name_or_path == (
        "trusted-verifier-fallback"
    )
    assert loaded.target_layer_ids == [2, 18, 33]
    assert loaded.fc.in_features == 3 * loaded.hidden_size
    load_config.assert_called_once_with("trusted-verifier-fallback")
    assert load_weights.call_count == 2


@pytest.mark.parametrize(
    ("model_class", "config_factory"),
    [
        (Eagle3DraftModel, _eagle3_config),
        (PEagleDraftModel, _peagle_config),
    ],
    ids=["eagle3", "peagle"],
)
def test_weight_bearing_legacy_checkpoint_rejects_tiny_verifier_width_change(
    tmp_path,
    model_class,
    config_factory,
):
    """A legacy three-input FC cannot be relabelled as a one-input tiny model."""
    config = config_factory(None)
    config.speculators_config.verifier.name_or_path = "unused-tiny-verifier"
    model = model_class(config)
    model.save_pretrained(tmp_path)

    with (
        patch.object(model_class, "load_verifier_weights"),
        patch(
            "speculators.models.utils.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(num_hidden_layers=2),
        ),
        pytest.raises(ValueError, match="Cannot safely load.*legacy Eagle-family"),
    ):
        model_class.from_pretrained(tmp_path)


def test_eagle3_config_from_pretrained_rejects_explicit_empty_aux_layers(tmp_path):
    """Config-only loading must reject an explicit empty auxiliary-layer list."""
    config = _eagle3_config([1])
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["eagle_aux_hidden_state_layer_ids"] = []
    config_path.write_text(json.dumps(config_dict))

    with pytest.raises(ValueError, match="must be None.*or contain at least one"):
        Eagle3SpeculatorConfig.from_pretrained(tmp_path)


def test_eagle3_constructor_rejects_empty_aux_layers_after_config_mutation():
    """The model constructor defends against configs mutated after validation."""
    config = _eagle3_config([1])
    object.__setattr__(config, "eagle_aux_hidden_state_layer_ids", [])

    with pytest.raises(ValueError, match="must be None.*or contain at least one"):
        Eagle3DraftModel(config)


@pytest.mark.parametrize(
    ("layer_ids", "message"),
    [
        ([0], "positive layer IDs"),
        ([-1, 2], "positive layer IDs"),
        ([1, 1], "duplicates"),
    ],
)
def test_eagle3_config_rejects_invalid_auxiliary_layer_ids(layer_ids, message):
    with pytest.raises(ValueError, match=message):
        _eagle3_config(layer_ids)


@pytest.mark.parametrize("layer_ids", [[1], [1, 2], [1, 2, 3]])
def test_peagle_width_tracks_auxiliary_layer_count(layer_ids: list[int]):
    """P-Eagle projection and mask widths follow the configured aux layers."""
    config = _peagle_config(layer_ids)

    model = PEagleDraftModel(config)
    expected_width = len(layer_ids) * model.hidden_size

    assert model.fc.in_features == expected_width
    assert model.mask_hidden.shape[-1] == expected_width
