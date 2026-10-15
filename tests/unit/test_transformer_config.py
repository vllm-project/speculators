"""Decoder configuration compatibility across speculator families."""

import pytest
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from speculators import (
    DFlash2SpeculatorConfig,
    DFlashSpeculatorConfig,
    DSparkSpeculatorConfig,
    Eagle3SpeculatorConfig,
    MTPSpeculatorConfig,
    PEagleSpeculatorConfig,
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    XPressSpeculatorConfig,
)
from speculators.proposals import GreedyTokenProposalConfig


@pytest.fixture(
    params=[
        (DFlashSpeculatorConfig, Qwen3Config),
        (DFlash2SpeculatorConfig, Qwen3Config),
        (DSparkSpeculatorConfig, Qwen3Config),
        (Eagle3SpeculatorConfig, Qwen3Config),
        (MTPSpeculatorConfig, Qwen2Config),
        (PEagleSpeculatorConfig, Qwen3Config),
        (XPressSpeculatorConfig, Qwen3Config),
    ],
    ids=lambda pair: pair[0].__name__,
)
def config_types(request):
    return request.param


def test_decoder_default_and_untagged_dictionary(config_types):
    config_class, decoder_class = config_types
    assert type(config_class().transformer_layer_config) is decoder_class

    config = config_class(transformer_layer_config={"hidden_size": 128})
    assert type(config.transformer_layer_config) is decoder_class
    assert config.transformer_layer_config.hidden_size == 128


def test_decoder_explicit_model_type(config_types):
    config_class, _ = config_types
    config = config_class(
        transformer_layer_config={"model_type": "llama", "hidden_size": 128}
    )
    assert type(config.transformer_layer_config) is LlamaConfig
    assert config.transformer_layer_config.hidden_size == 128


def test_decoder_instance_is_preserved(config_types):
    config_class, _ = config_types
    decoder = LlamaConfig(hidden_size=128)
    assert (
        config_class(transformer_layer_config=decoder).transformer_layer_config
        is decoder
    )


def test_decoder_unknown_model_type_is_rejected(config_types):
    config_class, _ = config_types
    with pytest.raises(ValueError):
        config_class(transformer_layer_config={"model_type": "unknown_decoder"})


@pytest.mark.parametrize("decoder_type", [Qwen2Config, Qwen3Config, LlamaConfig])
def test_decoder_checkpoint_roundtrip(config_types, decoder_type, tmp_path):
    config_class, _ = config_types
    decoder = decoder_type(hidden_size=128)
    config = config_class(
        transformer_layer_config=decoder,
        speculators_config=SpeculatorsConfig(
            algorithm=config_class.model_fields["speculators_model_type"].default,
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=[]),
        ),
    )
    model_class = SpeculatorModel.registered_model_class_from_config(config)

    assert config.model_dump()["transformer_layer_config"] == decoder.to_diff_dict()
    for data in (config.to_dict(), config.to_diff_dict()):
        restored = SpeculatorModelConfig.from_dict(data)
        assert type(restored) is config_class
        assert type(restored.transformer_layer_config) is decoder_type
        assert restored.transformer_layer_config.to_dict() == decoder.to_dict()
        assert (
            SpeculatorModel.registered_model_class_from_config(restored) is model_class
        )

    config.save_pretrained(tmp_path)
    restored = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert type(restored) is config_class
    assert restored.transformer_layer_config.to_dict() == decoder.to_dict()
