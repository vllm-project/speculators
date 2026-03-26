"""Unit tests for FastMTPConfig."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from transformers import AutoConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.fast_mtp.config import FastMTPConfig
from speculators.proposals import GreedyTokenProposalConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qwen3_next_config():
    """Minimal Qwen3-Next config for testing."""
    return AutoConfig.for_model("qwen3_next")


@pytest.fixture
def speculators_config():
    return SpeculatorsConfig(
        algorithm="mtp",
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
            architectures=["Qwen3NextForCausalLM"],
        ),
    )


@pytest.fixture
def mtp_config(qwen3_next_config, speculators_config):
    return FastMTPConfig(
        transformer_layer_config=qwen3_next_config,
        speculators_config=speculators_config,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_fast_mtp_config_registered_as_mtp() -> None:
    """FastMTPConfig must declare 'mtp' as its speculators_model_type."""
    assert FastMTPConfig.model_fields["speculators_model_type"].default == "mtp"


# ---------------------------------------------------------------------------
# Architecture fields — derived from transformer_layer_config
# ---------------------------------------------------------------------------


def test_fast_mtp_config_hidden_size_from_transformer_config(mtp_config) -> None:
    assert mtp_config.hidden_size == mtp_config.transformer_layer_config.hidden_size


def test_fast_mtp_config_vocab_size_from_transformer_config(mtp_config) -> None:
    assert mtp_config.vocab_size == mtp_config.transformer_layer_config.vocab_size


def test_fast_mtp_config_num_nextn_predict_layers_is_one(mtp_config) -> None:
    assert mtp_config.num_nextn_predict_layers == 1


def test_fast_mtp_config_num_nextn_predict_layers_not_one_raises() -> None:
    with pytest.raises(ValidationError, match="FastMTP currently only supports 1"):
        FastMTPConfig(num_nextn_predict_layers=2)


# ---------------------------------------------------------------------------
# num_speculative_steps property
# ---------------------------------------------------------------------------


def test_fast_mtp_config_num_speculative_steps(mtp_config) -> None:
    assert mtp_config.num_speculative_steps == 3


def test_fast_mtp_config_num_speculative_steps_no_config_raises(
    qwen3_next_config,
) -> None:
    config = FastMTPConfig(transformer_layer_config=qwen3_next_config)
    with pytest.raises(ValueError, match="speculators_config"):
        _ = config.num_speculative_steps


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_fast_mtp_config_roundtrip_json(mtp_config, tmp_path: Path) -> None:
    """Config survives save_pretrained → from_pretrained with all fields intact."""
    save_dir = tmp_path / "mtp_config"
    mtp_config.save_pretrained(str(save_dir))
    loaded = FastMTPConfig.from_pretrained(str(save_dir))
    assert loaded.speculators_model_type == "mtp"
    assert loaded.num_nextn_predict_layers == 1


def test_fast_mtp_config_transformer_layer_config_survives_roundtrip(
    mtp_config, tmp_path: Path
) -> None:
    """transformer_layer_config survives JSON round-trip with model_type preserved."""
    save_dir = tmp_path / "mtp_config"
    mtp_config.save_pretrained(str(save_dir))
    loaded = FastMTPConfig.from_pretrained(str(save_dir))
    assert loaded.transformer_layer_config.model_type == "qwen3_next"
    assert loaded.hidden_size == mtp_config.hidden_size
    assert loaded.vocab_size == mtp_config.vocab_size


# ---------------------------------------------------------------------------
# Validator: transformer_layer_config dict must include model_type
# ---------------------------------------------------------------------------


def test_fast_mtp_config_dict_without_model_type_raises() -> None:
    with pytest.raises(ValueError, match="model_type"):
        FastMTPConfig(transformer_layer_config={"hidden_size": 128})


def test_fast_mtp_config_dict_with_model_type_accepted() -> None:
    config = FastMTPConfig(transformer_layer_config={"model_type": "qwen3_next"})
    assert config.transformer_layer_config.model_type == "qwen3_next"
