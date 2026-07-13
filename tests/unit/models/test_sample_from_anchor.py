"""Tests for sample_from_anchor behavior in DFlash and DSpark models."""

from transformers import AutoConfig

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark import DSparkSpeculatorConfig


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

    def test_false_produces_block_size_minus_one(self):
        """sample_from_anchor=False should produce block_size - 1 tokens."""
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dflash",
            verifier_config=AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
            verifier_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            draft_vocab_size=128,
            block_size=8,
            sample_from_anchor=False,
            target_layer_ids=[0],
        )
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 7

    def test_true_produces_block_size(self):
        """sample_from_anchor=True should produce block_size tokens."""
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dflash",
            verifier_config=AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
            verifier_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            draft_vocab_size=128,
            block_size=8,
            sample_from_anchor=True,
            target_layer_ids=[0],
        )
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 8

    def test_dspark_defaults_to_true(self):
        """DSpark algorithm should default to sample_from_anchor=True."""
        kwargs = DFlashDraftModel._build_base_config_kwargs(
            algorithm="dspark",
            verifier_config=AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
            verifier_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            draft_vocab_size=128,
            block_size=8,
            target_layer_ids=[0],
        )
        assert kwargs["sample_from_anchor"]
        assert kwargs["speculators_config"].proposal_methods[0].speculative_tokens == 8
