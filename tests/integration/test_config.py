"""
Unit tests for the config module in the Speculators library.
"""

import tempfile

import pytest
from transformers import PretrainedConfig

from speculators import (
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
# Note: SpeculatorModelConfig is an abstract base class that uses a registry pattern.
# Concrete implementations like EagleSpeculatorConfig are tested in
# test_config_loading.py
# The from_pretrained functionality is tested there with real model configs.
