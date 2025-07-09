"""
Unit tests for the config module in the Speculators library.
"""

import json
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from transformers import PretrainedConfig

from speculators import (
    SpeculatorModelConfig,
    SpeculatorsConfig,
    TokenProposalConfig,
    VerifierConfig,
    reload_and_populate_configs,
)

# ===== TokenProposalConfig Tests =====


@TokenProposalConfig.register("test_proposal")
class TokenProposalConfigTest(TokenProposalConfig):
    proposal_type: Literal["test_proposal"] = "test_proposal"
    test_field: int = 123


# Ensure the schemas are reloaded to include the test proposal type
reload_and_populate_configs()


@pytest.mark.smoke
def test_token_proposal_config_initialization():
    config: TokenProposalConfigTest = TokenProposalConfig(  # type: ignore[assignment]
        proposal_type="test_proposal"
    )
    assert config.proposal_type == "test_proposal"
    assert config.test_field == 123


@pytest.mark.smoke
def test_token_proposal_config_subclass_initialization():
    config = TokenProposalConfigTest()
    assert config.proposal_type == "test_proposal"
    assert config.test_field == 123


@pytest.mark.smoke
def test_token_proposal_config_invalid_initialization():
    with pytest.raises(ValidationError) as exc_info:
        TokenProposalConfig()  # type: ignore[call-arg]

    assert "proposal_type" in str(exc_info.value)


@pytest.mark.smoke
def test_token_proposal_config_auto_registry():
    classes = TokenProposalConfig.registered_classes()
    class_names = [cls.__name__ for cls in classes]
    assert len(class_names) > 0
    assert "DynamicTreeTokenProposalConfig" in class_names
    assert "GreedyTokenProposalConfig" in class_names
    assert "SamplingTokenProposalConfig" in class_names
    assert "StaticTreeTokenProposalConfig" in class_names


@pytest.mark.sanity
def test_token_proposal_config_marshalling():
    original_config = TokenProposalConfigTest()

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["proposal_type"] == "test_proposal"
    assert config_dict["test_field"] == 123

    recreated_config: TokenProposalConfigTest = (
        TokenProposalConfig.model_validate(config_dict)  # type: ignore[assignment]
    )
    assert recreated_config.proposal_type == original_config.proposal_type
    assert recreated_config.test_field == original_config.test_field


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
    )

    assert config.name_or_path == "test/verifier"
    assert config.architectures == ["TestModel"]


@pytest.mark.smoke
def test_verifier_config_from_verifier_config(mock_pretrained_config):
    config = VerifierConfig.from_pretrained(mock_pretrained_config)

    assert config.name_or_path == "test/verifier"
    assert config.architectures == ["TestModel"]


@pytest.mark.smoke
def test_verifier_config_invalid_initialization():
    with pytest.raises(ValidationError) as exc_info:
        VerifierConfig()  # type: ignore[call-arg]

    error_str = str(exc_info.value)
    assert "name_or_path" in error_str
    assert "architectures" in error_str


@pytest.mark.sanity
def test_verifier_config_marshalling():
    original_config = VerifierConfig(
        name_or_path="test/verifier",
        architectures=["TestModel"],
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["name_or_path"] == "test/verifier"
    assert config_dict["architectures"] == ["TestModel"]

    recreated_config = VerifierConfig.model_validate(config_dict)
    assert recreated_config.name_or_path == original_config.name_or_path
    assert recreated_config.architectures == original_config.architectures


# ===== SpeculatorsConfig Tests =====


@pytest.fixture
def sample_token_proposal_config():
    return TokenProposalConfigTest()


@pytest.fixture
def sample_verifier_config():
    return VerifierConfig(
        name_or_path="test/verifier",
        architectures=["TestModel"],
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


@SpeculatorModelConfig.register("test_model")
class SpeculatorModelConfigTest(SpeculatorModelConfig):
    speculators_model_type: Literal["test_model"] = "test_model"
    test_field: int = 456


# Ensure the schemas are reloaded to include the test proposal type
reload_and_populate_configs()


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
def test_speculator_model_config_auto_registry():
    classes = SpeculatorModelConfig.registered_classes()
    class_names = [cls.__name__ for cls in classes]
    assert len(class_names) > 0
    assert "EagleSpeculatorConfig" in class_names
    assert "IndependentSpeculatorConfig" in class_names
    assert "MLPSpeculatorConfig" in class_names


@pytest.mark.smoke
def test_speculator_model_config_marshalling(sample_speculators_config):
    original_config = SpeculatorModelConfigTest(
        speculators_model_type="test_model",
        speculators_config=sample_speculators_config,
        test_field=678,
    )

    config_dict = original_config.model_dump()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "test_model"
    assert config_dict["speculators_config"]["algorithm"] == "test_algorithm"
    assert config_dict["test_field"] == 678

    recreated_config = SpeculatorModelConfig.model_validate(config_dict)
    assert (
        recreated_config.speculators_model_type
        == original_config.speculators_model_type
    )
    assert (
        recreated_config.speculators_config.algorithm
        == original_config.speculators_config.algorithm
    )
    assert recreated_config.test_field == original_config.test_field


@pytest.mark.smoke
def test_speculator_model_config_dict_marshaling(sample_speculators_config):
    config: SpeculatorModelConfigTest = SpeculatorModelConfigTest(
        speculators_model_type="test_model",
        speculators_config=sample_speculators_config,
        test_field=678,
    )

    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["speculators_model_type"] == "test_model"
    assert config_dict["speculators_config"]["algorithm"] == "test_algorithm"
    assert config_dict["test_field"] == 678

    config_diff_dict = config.to_diff_dict()
    assert isinstance(config_diff_dict, dict)
    assert config_diff_dict["speculators_model_type"] == "test_model"
    assert config_diff_dict["speculators_config"]["algorithm"] == "test_algorithm"
    assert config_diff_dict["test_field"] == 678

    reload_config = SpeculatorModelConfig.from_dict(config_dict)
    assert reload_config.speculators_model_type == "test_model"
    assert reload_config.speculators_config.algorithm == "test_algorithm"
    assert reload_config.test_field == 678

    reload_diff_config = SpeculatorModelConfig.from_dict(config_diff_dict)
    assert reload_diff_config.speculators_model_type == "test_model"
    assert reload_diff_config.speculators_config.algorithm == "test_algorithm"
    assert reload_diff_config.test_field == 678


@pytest.mark.sanity
def test_speculator_model_config_from_dict_invalid(sample_speculators_config):
    with pytest.raises(ValueError) as exc_info:
        SpeculatorModelConfig.from_dict({})

    assert (
        "The config dictionary must contain the 'speculators_model_type' field"
        in str(exc_info.value)
    )

    with pytest.raises(ValueError) as exc_info:
        SpeculatorModelConfig.from_dict(
            {
                "speculators_config": sample_speculators_config.model_dump(),
                "test_field": 678,
            }
        )

    assert (
        "The config dictionary must contain the 'speculators_model_type' field "
        in str(exc_info.value)
    )


@pytest.mark.smoke
def test_speculator_model_config_from_pretrained_local_marshalling(
    sample_speculators_config,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "config.json"
        config: SpeculatorModelConfigTest = SpeculatorModelConfigTest(
            speculators_model_type="test_model",
            speculators_config=sample_speculators_config,
            test_field=678,
        )
        config.save_pretrained(tmp_path)
        assert tmp_path.exists()

        reloaded_config = SpeculatorModelConfig.from_pretrained(tmp_path)
        assert reloaded_config.speculators_model_type == "test_model"
        assert reloaded_config.speculators_config.algorithm == "test_algorithm"
        assert reloaded_config.test_field == 678


@pytest.mark.smoke
def test_speculator_model_config_from_pretrained_hf_hub(sample_speculators_config):
    config_data = {
        "speculators_model_type": "test_model",
        "speculators_config": sample_speculators_config.model_dump(),
        "test_field": 678,
    }

    with patch.object(SpeculatorModelConfig, "get_config_dict") as mock_get_config_dict:
        mock_get_config_dict.return_value = (config_data, {})
        config = SpeculatorModelConfig.from_pretrained("test/fake-model-hub-name")

        mock_get_config_dict.assert_called_once_with(
            "test/fake-model-hub-name",
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
            revision="main",
        )

        # Verify the config was loaded correctly
        assert config.speculators_model_type == "test_model"
        assert config.speculators_config.algorithm == "test_algorithm"
        assert config.test_field == 678


@pytest.mark.smoke
def test_speculator_model_config_from_pretrained_conversion(sample_speculators_config):
    # conversion not implemented yet, ensure it raises NotImplementedError
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "config.json"
        config_data = {
            "speculators_config": sample_speculators_config.model_dump(),
            "test_field": 678,
        }
        with tmp_path.open("w") as f:
            json.dump(config_data, f)

        with pytest.raises(NotImplementedError) as exc_info:
            SpeculatorModelConfig.from_pretrained(tmp_path, convert_to_speculator=True)

    assert "Loading a non-speculator model config is not supported yet" in str(
        exc_info.value
    )
