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
        config_dict, _ = PretrainedConfig.get_config_dict(
            "RedHatAI/Llama-3.1-8B-Instruct",
            cache_dir=tmp_dir,
        )

    # transformers v5 validates rope_scaling strictly and requires rope_theta inside
    # rope_scaling for rope_type=llama3. RedHatAI/Llama-3.1-8B-Instruct's Hub config
    # predates that requirement. Drop rope_scaling before constructing PretrainedConfig
    # — VerifierConfig.from_config only reads architectures, not rope parameters.
    config_dict.pop("rope_scaling", None)
    pretrained_config = PretrainedConfig(**config_dict)

    config = VerifierConfig.from_config(
        pretrained_config, name_or_path="RedHatAI/Llama-3.1-8B-Instruct"
    )
    assert config.name_or_path == "RedHatAI/Llama-3.1-8B-Instruct"
    assert config.architectures == ["LlamaForCausalLM"]


# ===== SpeculatorModelConfig Tests =====


@pytest.mark.smoke
@pytest.mark.skip("Test not implemented")
def test_speculator_model_config_from_pretrained():
    # Implement loading once real config is available
    assert True


@pytest.mark.regression
@pytest.mark.skip("Test not implemented")
def test_speculator_model_config_pretrained_methods():
    # Implement saving once real config is available
    assert True
