"""
Unit tests for the config module in the Speculators library.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from transformers import PretrainedConfig

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    TokenProposalConfig,
    VerifierConfig,
)

# ===== TokenProposalConfig Tests =====


@pytest.mark.smoke
def test_token_proposal_config_initialization():
    config = TokenProposalConfig(proposal_type="test_proposal")
    assert config.proposal_type == "test_proposal"


@pytest.mark.smoke
def test_token_proposal_config_invalid_initialization():
    with pytest.raises(ValidationError) as exc_info:
        TokenProposalConfig()  # type: ignore[call-arg]

    assert "proposal_type" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


@pytest.mark.sanity
def test_token_proposal_config_marshalling():
    original_config = TokenProposalConfig(proposal_type="test_proposal")

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "test_proposal"

    recreated_config = TokenProposalConfig.model_validate(config_dict)
    assert recreated_config.proposal_type == original_config.proposal_type


# ===== VerifierConfig Tests =====


@pytest.fixture
def mock_pretrained_config():
    config = MagicMock(spec=PretrainedConfig)
    config.name_or_path = "test/verifier"
    config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "intermediate_size": 3072,
        "vocab_size": 50000,
        "max_position_embeddings": 512,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    return config


@pytest.mark.smoke
def test_verifier_config_initialization():
    config = VerifierConfig(
        name_or_path="test/verifier",
        architectures=["TestModel"],
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=50000,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
    )

    assert config.name_or_path == "test/verifier"
    assert config.architectures == ["TestModel"]
    assert config.hidden_size == 768
    assert config.intermediate_size == 3072
    assert config.vocab_size == 50000
    assert config.max_position_embeddings == 512
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2


@pytest.mark.smoke
def test_verifier_config_from_verifier_config(mock_pretrained_config):
    config = VerifierConfig.from_verifier_config(mock_pretrained_config)

    assert config.name_or_path == "test/verifier"
    assert config.architectures == ["TestModel"]
    assert config.hidden_size == 768
    assert config.intermediate_size == 3072
    assert config.vocab_size == 50000
    assert config.max_position_embeddings == 512
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2


@pytest.mark.smoke
def test_verifier_config_invalid_initialization():
    with pytest.raises(ValidationError) as exc_info:
        VerifierConfig()  # type: ignore[call-arg]

    error_str = str(exc_info.value)
    assert "name_or_path" in error_str
    assert "architectures" in error_str
    assert "hidden_size" in error_str
    assert "intermediate_size" in error_str
    assert "vocab_size" in error_str
    assert "max_position_embeddings" in error_str
    assert "bos_token_id" in error_str
    assert "eos_token_id" in error_str


@pytest.mark.sanity
def test_verifier_config_marshalling():
    original_config = VerifierConfig(
        name_or_path="test/verifier",
        architectures=["TestModel"],
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=50000,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["name_or_path"] == "test/verifier"
    assert config_dict["architectures"] == ["TestModel"]
    assert config_dict["hidden_size"] == 768

    recreated_config = VerifierConfig.model_validate(config_dict)
    assert recreated_config.name_or_path == original_config.name_or_path
    assert recreated_config.architectures == original_config.architectures
    assert recreated_config.hidden_size == original_config.hidden_size


# ===== SpeculatorsConfig Tests =====


@pytest.fixture
def sample_token_proposal_config():
    return TokenProposalConfig(proposal_type="test_proposal")


@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["TestModel"],
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=50000,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.mark.smoke
def test_speculators_config_initialization(
    sample_token_proposal_config, sample_verifier_config
):
    config = SpeculatorsConfig(
        algorithm="test_algorithm",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="test_proposal",
        verifier=sample_verifier_config,
    )

    assert config.algorithm == "test_algorithm"
    assert len(config.proposal_methods) == 1
    assert config.proposal_methods[0].proposal_type == "test_proposal"
    assert config.default_proposal_method == "test_proposal"
    assert config.verifier.name_or_path == "test/verifier"


@pytest.mark.smoke
def test_speculators_config_invalid_initialization(
    sample_token_proposal_config, sample_verifier_config
):
    with pytest.raises(ValidationError) as exc_info:
        SpeculatorsConfig()  # type: ignore[call-arg]

    error_str = str(exc_info.value)
    assert "algorithm" in error_str
    assert "proposal_methods" in error_str
    assert "default_proposal_method" in error_str
    assert "verifier" in error_str


@pytest.mark.sanity
def test_speculators_config_marshalling(
    sample_token_proposal_config, sample_verifier_config
):
    original_config = SpeculatorsConfig(
        algorithm="test_algorithm",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="test_proposal",
        verifier=sample_verifier_config,
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["algorithm"] == "test_algorithm"
    assert len(config_dict["proposal_methods"]) == 1
    assert config_dict["proposal_methods"][0]["proposal_type"] == "test_proposal"
    assert config_dict["default_proposal_method"] == "test_proposal"

    recreated_config = SpeculatorsConfig.model_validate(config_dict)
    assert recreated_config.algorithm == original_config.algorithm
    assert (
        recreated_config.proposal_methods[0].proposal_type
        == original_config.proposal_methods[0].proposal_type
    )
    assert (
        recreated_config.default_proposal_method
        == original_config.default_proposal_method
    )
    assert (
        recreated_config.verifier.name_or_path == original_config.verifier.name_or_path
    )


# ===== SpeculatorModelConfig Tests =====


@pytest.fixture
def sample_speculators_config(sample_token_proposal_config, sample_verifier_config):
    return SpeculatorsConfig(
        algorithm="test_algorithm",
        proposal_methods=[sample_token_proposal_config],
        default_proposal_method="test_proposal",
        verifier=sample_verifier_config,
    )


@pytest.mark.smoke
def test_speculator_model_config_initialization(sample_speculators_config):
    config = SpeculatorModelConfig(
        speculators_model_type="test_model",
        speculators_config=sample_speculators_config,
    )

    assert config.speculators_model_type == "test_model"
    assert config.speculators_config.algorithm == "test_algorithm"
    assert config.speculators_version is not None

    # Check that PretrainedConfig attributes are accessible
    assert hasattr(config, "to_dict")
    assert hasattr(config, "to_diff_dict")
    assert hasattr(config, "to_json_string")
    assert hasattr(config, "to_json_file")
    assert hasattr(config, "save_pretrained")


@pytest.mark.smoke
def test_speculator_model_config_invalid_initialization(sample_speculators_config):
    with pytest.raises(ValidationError) as exc_info:
        SpeculatorModelConfig()  # type: ignore[call-arg]

    error_str = str(exc_info.value)
    assert "speculators_model_type" in error_str
    assert "speculators_config" in error_str


@pytest.mark.sanity
def test_speculator_model_config_marshalling(sample_speculators_config):
    original_config = SpeculatorModelConfig(
        speculators_model_type="test_model",
        speculators_config=sample_speculators_config,
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "test_model"
    assert config_dict["speculators_config"]["algorithm"] == "test_algorithm"

    recreated_config = SpeculatorModelConfig.model_validate(config_dict)
    assert (
        recreated_config.speculators_model_type
        == original_config.speculators_model_type
    )
    assert (
        recreated_config.speculators_config.algorithm
        == original_config.speculators_config.algorithm
    )


@pytest.mark.smoke
def test_speculator_model_config_from_pretrained():
    with pytest.raises(NotImplementedError) as exc_info:
        SpeculatorModelConfig.from_pretrained("test/model")

    assert "from_pretrained is not implemented yet" in str(exc_info.value)


@pytest.mark.regression
def test_speculator_model_config_pretrained_methods(sample_speculators_config):
    config = SpeculatorModelConfig(
        speculators_model_type="test_model",
        speculators_config=sample_speculators_config,
    )

    # Test to_dict and to_diff_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "speculators_model_type" in config_dict
    assert "speculators_config" in config_dict

    diff_dict = config.to_diff_dict()
    assert isinstance(diff_dict, dict)
    assert "speculators_model_type" in diff_dict
    # Test to_json_string
    json_string = config.to_json_string()
    assert isinstance(json_string, str)
    parsed_json = json.loads(json_string)
    assert parsed_json["speculators_model_type"] == "test_model"

    # Test to_json_file and save_pretrained
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_path = tmp_path / "config.json"
        config.to_json_file(json_path)
        assert json_path.exists()

        save_dir = tmp_path / "save_dir"
        config.save_pretrained(save_dir)
        assert (save_dir / "config.json").exists()
        # Load the saved file and verify contents
        with (save_dir / "config.json").open() as file:
            saved_dict = json.load(file)

        assert saved_dict["speculators_model_type"] == "test_model"
        assert saved_dict["speculators_config"]["algorithm"] == "test_algorithm"
