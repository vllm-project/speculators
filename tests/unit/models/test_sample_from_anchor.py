"""Tests for sample_from_anchor behavior in DFlash and DSpark models."""

import pytest
from transformers import AutoConfig

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark import DSparkSpeculatorConfig


@pytest.fixture
def local_verifier(tmp_path):
    """Create a minimal local verifier config without a Hub dependency."""
    verifier_path = tmp_path / "verifier"
    verifier_config = AutoConfig.for_model("qwen2", num_hidden_layers=24)
    verifier_config.save_pretrained(verifier_path)
    return verifier_config, str(verifier_path)


class TestSampleFromAnchorDFlash:
    """Tests for DFlash sample_from_anchor behavior."""

    def test_default_is_false(self):
        """DFlash should default to sample_from_anchor=False."""
        config = DFlashSpeculatorConfig(draft_vocab_size=128, block_size=4)
        assert not config.sample_from_anchor

    def test_can_set_to_true(self):
        """DFlash can be configured with sample_from_anchor=True."""
        config = DFlashSpeculatorConfig(
            draft_vocab_size=128, block_size=4, sample_from_anchor=True
        )
        assert config.sample_from_anchor


class TestSampleFromAnchorDSpark:
    """Tests for DSpark sample_from_anchor behavior."""

    def test_default_is_true(self):
        """DSpark should default to sample_from_anchor=True."""
        config = DSparkSpeculatorConfig(draft_vocab_size=128, block_size=4)
        assert config.sample_from_anchor

    def test_can_override_to_false(self):
        """DSpark can be configured with sample_from_anchor=False."""
        config = DSparkSpeculatorConfig(
            draft_vocab_size=128, block_size=4, sample_from_anchor=False
        )
        assert not config.sample_from_anchor


class TestSpeculativeTokensCalculation:
    """Test that speculative_tokens is calculated correctly."""

    def test_false_produces_block_size_minus_one(self, local_verifier):
        """sample_from_anchor=False should produce block_size - 1 tokens."""
        verifier_config, verifier_path = local_verifier
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dflash",
            verifier_config=verifier_config,
            verifier_name_or_path=verifier_path,
            draft_vocab_size=128,
            block_size=8,
            sample_from_anchor=False,
            target_layer_ids=[1],
        )
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 7

    def test_true_produces_block_size(self, local_verifier):
        """sample_from_anchor=True should produce block_size tokens."""
        verifier_config, verifier_path = local_verifier
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dflash",
            verifier_config=verifier_config,
            verifier_name_or_path=verifier_path,
            draft_vocab_size=128,
            block_size=8,
            sample_from_anchor=True,
            target_layer_ids=[1],
        )
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 8

    def test_dspark_defaults_to_true(self, local_verifier):
        """DSpark algorithm should default to sample_from_anchor=True."""
        verifier_config, verifier_path = local_verifier
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dspark",
            verifier_config=verifier_config,
            verifier_name_or_path=verifier_path,
            draft_vocab_size=128,
            block_size=8,
            target_layer_ids=[1],
        )
        assert kwargs["sample_from_anchor"]
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 8
