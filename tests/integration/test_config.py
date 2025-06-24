"""
Unit tests for the config module in the Speculators library.
"""

import tempfile

import pytest
from transformers import PretrainedConfig

from speculators import (
    SpeculatorModelConfig,
    VerifierConfig,
)

# ===== VerifierConfig Tests =====


@pytest.mark.smoke
def test_verifier_config_from_verifier_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pretrained_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path="RedHatAI/Llama-3.1-8B-Instruct",
            cache_dir=tmp_dir,
        )

    config = VerifierConfig.from_verifier_config(
        pretrained_config, name_or_path="RedHatAI/Llama-3.1-8B-Instruct"
    )
    assert config.name_or_path == "RedHatAI/Llama-3.1-8B-Instruct"
    assert config.architectures == ["LlamaForCausalLM"]
    assert config.hidden_size == 4096
    assert config.intermediate_size == 14336
    assert config.vocab_size == 128256
    assert config.max_position_embeddings == 131072
    assert config.bos_token_id == 128000
    assert config.eos_token_id == [128001, 128008, 128009]


# ===== SpeculatorModelConfig Tests =====

@pytest.mark.xfail(reason="swap stub with real model id once pushed")
@pytest.mark.smoke
def test_speculator_model_config_from_pretrained():
    # swap for real config once implemented
    with pytest.raises(Exception) as exc_info:
        SpeculatorModelConfig.from_pretrained("test/model")

    assert "from_pretrained is not implemented yet" in str(exc_info.value)


@pytest.mark.regression
def test_speculator_model_config_pretrained_methods():
    # Implement saving once real config is available
    assert True
