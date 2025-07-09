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

    config = VerifierConfig.from_pretrained(
        pretrained_config, name_or_path="RedHatAI/Llama-3.1-8B-Instruct"
    )
    assert config.name_or_path == "RedHatAI/Llama-3.1-8B-Instruct"
    assert config.architectures == ["LlamaForCausalLM"]


# ===== SpeculatorModelConfig Tests =====


@pytest.mark.smoke
def test_speculator_model_config_from_pretrained():
    # Implement loading once real config is available
    assert True


@pytest.mark.regression
def test_speculator_model_config_pretrained_methods():
    # Implement saving once real config is available
    assert True
