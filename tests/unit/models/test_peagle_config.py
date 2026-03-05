"""Unit tests for P-EAGLE configuration."""

import pytest
from pydantic import ValidationError
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.config import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.peagle.config import PEagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


def _make_speculators_config():
    """Helper to create a minimal SpeculatorsConfig."""
    return SpeculatorsConfig(
        algorithm="peagle",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=8)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path="test-model",
            architectures=["LlamaForCausalLM"],
        ),
    )


class TestPEagleSpeculatorConfig:
    """Tests for PEagleSpeculatorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PEagleSpeculatorConfig(
            speculators_config=_make_speculators_config(),
        )
        assert config.speculators_model_type == "peagle"
        assert config.para_num == 8
        assert config.down_sample_ratio == 0.5
        assert config.down_sample_ratio_min == 0.0
        assert config.ptd_token_id == 0
        assert config.loss_type == "cross_entropy"
        assert config.architectures == ["PEagleSpeculator"]

    def test_custom_values(self):
        """Test with custom configuration values."""
        config = PEagleSpeculatorConfig(
            speculators_config=_make_speculators_config(),
            para_num=4,
            down_sample_ratio=0.7,
            down_sample_ratio_min=0.1,
            ptd_token_id=128255,
            loss_type="kl_div",
        )
        assert config.para_num == 4
        assert config.down_sample_ratio == 0.7
        assert config.down_sample_ratio_min == 0.1
        assert config.ptd_token_id == 128255
        assert config.loss_type == "kl_div"

    def test_inherits_eagle3_fields(self):
        """Config should inherit EAGLE-3 fields."""
        llama_config = LlamaConfig(
            hidden_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
        )
        config = PEagleSpeculatorConfig(
            speculators_config=_make_speculators_config(),
            transformer_layer_config=llama_config,
            draft_vocab_size=1000,
            norm_before_residual=True,
        )
        assert config.draft_vocab_size == 1000
        assert config.norm_before_residual is True
        assert config.transformer_layer_config.hidden_size == 256

    def test_invalid_para_num(self):
        """para_num must be >= 1."""
        with pytest.raises(ValidationError):
            PEagleSpeculatorConfig(
                speculators_config=_make_speculators_config(),
                para_num=0,
            )

    def test_invalid_down_sample_ratio(self):
        """down_sample_ratio must be in (0, 1)."""
        with pytest.raises(ValidationError):
            PEagleSpeculatorConfig(
                speculators_config=_make_speculators_config(),
                down_sample_ratio=0.0,
            )
        with pytest.raises(ValidationError):
            PEagleSpeculatorConfig(
                speculators_config=_make_speculators_config(),
                down_sample_ratio=1.0,
            )

    def test_invalid_down_sample_ratio_min(self):
        """down_sample_ratio_min must be in [0, 1]."""
        with pytest.raises(ValidationError):
            PEagleSpeculatorConfig(
                speculators_config=_make_speculators_config(),
                down_sample_ratio_min=-0.1,
            )

    def test_invalid_loss_type(self):
        """loss_type must be 'kl_div' or 'cross_entropy'."""
        with pytest.raises(ValidationError):
            PEagleSpeculatorConfig(
                speculators_config=_make_speculators_config(),
                loss_type="mse",
            )

    def test_registry_registration(self):
        """PEagleSpeculatorConfig should be registered as 'peagle'."""
        SpeculatorModelConfig.auto_populate_registry()
        assert "peagle" in SpeculatorModelConfig.registry

    def test_serialization(self):
        """Config should serialize and deserialize correctly."""
        config = PEagleSpeculatorConfig(
            speculators_config=_make_speculators_config(),
            para_num=6,
            down_sample_ratio=0.6,
        )
        config_dict = config.to_dict()
        assert config_dict["para_num"] == 6
        assert config_dict["down_sample_ratio"] == 0.6
        assert config_dict["speculators_model_type"] == "peagle"
