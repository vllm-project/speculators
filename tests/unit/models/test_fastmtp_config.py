import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.models import FastMTPSpeculatorConfig
from speculators.proposals import GreedyTokenProposalConfig


@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["LlamaForCausalLM"],
    )


@pytest.fixture
def sample_token_proposal_config():
    return GreedyTokenProposalConfig(
        speculative_tokens=5,
        verifier_accept_k=1,
        accept_tolerance=0.0,
    )


@pytest.fixture
def sample_speculators_config(sample_token_proposal_config, sample_verifier_config):
    return SpeculatorsConfig(
        algorithm="fastmtp",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="greedy",
        verifier=sample_verifier_config,
    )


@pytest.fixture
def sample_llama_config():
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    )


@pytest.fixture
def fastmtp_config_dict():
    return {
        "speculators_model_type": "fastmtp",
        "architectures": ["FastMTPDraftModel"],
        "transformer_layer_config": {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 2048,
        },
        "draft_vocab_size": 32000,
        "num_speculative_steps": 3,
        "speculators_config": {
            "algorithm": "fastmtp",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 5,
                    "verifier_accept_k": 1,
                    "accept_tolerance": 0.0,
                }
            ],
            "default_proposal_method": "greedy",
            "verifier": {
                "name_or_path": "test/verifier",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 768,
                "intermediate_size": 3072,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
        },
    }


@pytest.mark.smoke
def test_fastmtp_config_default_initialization():
    config = FastMTPSpeculatorConfig()

    assert config.speculators_model_type == "fastmtp"
    assert config.architectures == ["FastMTPDraftModel"]
    assert isinstance(config.transformer_layer_config, LlamaConfig)
    assert config.draft_vocab_size == 32000
    assert config.num_speculative_steps == 3
    assert config.model_type == "speculator_model"
    assert config.speculators_config is None


@pytest.mark.smoke
def test_fastmtp_config_custom_initialization(
    sample_speculators_config, sample_llama_config
):
    config = FastMTPSpeculatorConfig(
        transformer_layer_config=sample_llama_config,
        draft_vocab_size=16000,
        num_speculative_steps=5,
        speculators_config=sample_speculators_config,
    )

    assert config.speculators_model_type == "fastmtp"
    assert config.transformer_layer_config == sample_llama_config
    assert config.draft_vocab_size == 16000
    assert config.num_speculative_steps == 5
    assert config.speculators_config == sample_speculators_config


@pytest.mark.smoke
def test_fastmtp_config_with_qwen3():
    qwen_config = Qwen3Config(
        vocab_size=151680,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
    )

    config = FastMTPSpeculatorConfig(
        transformer_layer_config=qwen_config,
        draft_vocab_size=151680,
        num_speculative_steps=3,
    )

    assert isinstance(config.transformer_layer_config, Qwen3Config)
    assert config.transformer_layer_config.vocab_size == 151680
    assert config.draft_vocab_size == 151680
    assert config.num_speculative_steps == 3


@pytest.mark.smoke
def test_fastmtp_config_invalid_model_type():
    with pytest.raises(ValidationError, match="speculators_model_type"):
        FastMTPSpeculatorConfig(speculators_model_type="invalid")  # type: ignore[arg-type]


@pytest.mark.smoke
def test_fastmtp_config_invalid_architectures():
    with pytest.raises(ValidationError, match="architectures"):
        FastMTPSpeculatorConfig(architectures="not_a_list")  # type: ignore[arg-type]


@pytest.mark.smoke
def test_fastmtp_config_auto_registry():
    registered_classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in registered_classes]

    assert "FastMTPSpeculatorConfig" in class_names
    assert SpeculatorModelConfig.registry is not None
    assert "fastmtp" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["fastmtp"] == FastMTPSpeculatorConfig


@pytest.mark.smoke
def test_fastmtp_config_marshalling(sample_speculators_config):
    original_config = FastMTPSpeculatorConfig(
        draft_vocab_size=16000,
        num_speculative_steps=5,
        speculators_config=sample_speculators_config,
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "fastmtp"
    assert config_dict["draft_vocab_size"] == 16000
    assert config_dict["num_speculative_steps"] == 5

    recreated_base = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(recreated_base, FastMTPSpeculatorConfig)
    assert recreated_base.draft_vocab_size == 16000
    assert recreated_base.num_speculative_steps == 5

    recreated_derived = FastMTPSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, FastMTPSpeculatorConfig)
    assert recreated_derived.draft_vocab_size == 16000
    assert recreated_derived.num_speculative_steps == 5


@pytest.mark.smoke
def test_fastmtp_config_base_initialization(sample_speculators_config):
    original_config = FastMTPSpeculatorConfig(
        draft_vocab_size=16000,
        num_speculative_steps=3,
        speculators_config=sample_speculators_config,
    )

    config_dict = original_config.model_dump()
    recreated_config = SpeculatorModelConfig.model_validate(config_dict)

    assert isinstance(recreated_config, FastMTPSpeculatorConfig)
    assert recreated_config.speculators_model_type == "fastmtp"
    assert recreated_config.draft_vocab_size == 16000
    assert recreated_config.num_speculative_steps == 3
    assert recreated_config.speculators_config == sample_speculators_config


@pytest.mark.smoke
def test_fastmtp_config_backwards_compatibility(fastmtp_config_dict):
    config_derived = FastMTPSpeculatorConfig.model_validate(fastmtp_config_dict)
    assert isinstance(config_derived, FastMTPSpeculatorConfig)
    assert config_derived.speculators_model_type == "fastmtp"
    assert config_derived.draft_vocab_size == 32000
    assert config_derived.num_speculative_steps == 3
    assert config_derived.speculators_config.algorithm == "fastmtp"

    config_base = SpeculatorModelConfig.model_validate(fastmtp_config_dict)
    assert isinstance(config_base, FastMTPSpeculatorConfig)
    assert config_base.speculators_model_type == "fastmtp"


@pytest.mark.smoke
def test_fastmtp_config_dict_marshalling(fastmtp_config_dict):
    original_config = FastMTPSpeculatorConfig.model_validate(fastmtp_config_dict)

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "fastmtp"
    assert config_dict["num_speculative_steps"] == 3

    recreated_base = SpeculatorModelConfig.from_dict(config_dict)
    assert isinstance(recreated_base, FastMTPSpeculatorConfig)
    assert recreated_base.num_speculative_steps == 3
    assert recreated_base.draft_vocab_size == 32000

    recreated_derived = FastMTPSpeculatorConfig.model_validate(config_dict)
    assert isinstance(recreated_derived, FastMTPSpeculatorConfig)
    assert recreated_derived.num_speculative_steps == 3


@pytest.mark.smoke
def test_fastmtp_config_from_pretrained_local_marshalling(fastmtp_config_dict):
    original_config = FastMTPSpeculatorConfig.model_validate(fastmtp_config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        original_config.save_pretrained(temp_path)

        config_file = temp_path / "config.json"
        assert config_file.exists()

        loaded_base = SpeculatorModelConfig.from_pretrained(temp_path)
        assert isinstance(loaded_base, FastMTPSpeculatorConfig)
        assert loaded_base.speculators_model_type == "fastmtp"
        assert loaded_base.num_speculative_steps == 3
        assert loaded_base.draft_vocab_size == 32000

        loaded_derived = FastMTPSpeculatorConfig.from_pretrained(temp_path)
        assert isinstance(loaded_derived, FastMTPSpeculatorConfig)
        assert loaded_derived.speculators_model_type == "fastmtp"
        assert loaded_derived.num_speculative_steps == 3
        assert loaded_derived.draft_vocab_size == 32000


@pytest.mark.smoke
def test_fastmtp_config_target_vocab_size():
    config = FastMTPSpeculatorConfig(
        transformer_layer_config=LlamaConfig(vocab_size=128256),
        draft_vocab_size=32000,
    )

    assert config.target_vocab_size == 128256
    assert config.draft_vocab_size == 32000
