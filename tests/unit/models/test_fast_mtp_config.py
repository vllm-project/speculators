"""Unit tests for FastMTPConfig."""

import pytest
from pydantic import ValidationError
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp import FastMTPConfig
from speculators.proposals import GreedyTokenProposalConfig

_NO_VERIFIER = VerifierConfig(name_or_path=None, architectures=[])


@pytest.fixture
def tiny_tc():
    return Qwen2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=64,
    )


@pytest.fixture
def fast_mtp_config(tiny_tc):
    return FastMTPConfig(
        transformer_layer_config=tiny_tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=_NO_VERIFIER,
        ),
    )


@pytest.mark.smoke
def test_vocab_size_derived_from_transformer_config(fast_mtp_config):
    assert fast_mtp_config.vocab_size == 256


@pytest.mark.smoke
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_num_speculative_steps_derived_from_proposal_config(tiny_tc, n):
    config = FastMTPConfig(
        transformer_layer_config=tiny_tc,
        speculators_config=SpeculatorsConfig(
            algorithm="mtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=n)],
            default_proposal_method="greedy",
            verifier=_NO_VERIFIER,
        ),
    )
    assert config.num_speculative_steps == n


@pytest.mark.smoke
def test_num_nextn_predict_layers_not_one_rejected(tiny_tc):
    with pytest.raises(ValidationError, match="num_nextn_predict_layers"):
        FastMTPConfig(
            transformer_layer_config=tiny_tc,
            num_nextn_predict_layers=2,
        )


@pytest.mark.smoke
def test_num_nextn_predict_layers_default(fast_mtp_config):
    assert fast_mtp_config.num_nextn_predict_layers == 1


@pytest.mark.smoke
def test_fast_mtp_config_registered():
    assert SpeculatorModelConfig.registry is not None
    assert "mtp" in SpeculatorModelConfig.registry
    assert SpeculatorModelConfig.registry["mtp"] is FastMTPConfig


@pytest.mark.smoke
def test_fast_mtp_config_round_trip(fast_mtp_config):
    config_dict = fast_mtp_config.model_dump()
    reloaded = SpeculatorModelConfig.model_validate(config_dict)
    assert isinstance(reloaded, FastMTPConfig)
    assert reloaded.speculators_model_type == "mtp"
    assert reloaded.vocab_size == fast_mtp_config.vocab_size
    assert reloaded.hidden_size == fast_mtp_config.hidden_size
