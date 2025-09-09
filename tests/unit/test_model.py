"""
Unit tests for the model module in the Speculators library.
"""

from __future__ import annotations

import os
import tempfile
from typing import Literal
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators import (
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_and_populate_models,
)
from speculators.proposals import GreedyTokenProposalConfig


def create_mock_verifier():
    """Create a properly mocked verifier model with config attribute."""
    config_dict = {
        "architectures": ["TestModel"],
        "vocab_size": 1000,
        "model_type": "test_model",
        "name_or_path": "test/verifier/path",
    }

    mock_config = MagicMock(spec=PretrainedConfig)
    mock_config.architectures = ["TestModel"]
    mock_config.vocab_size = 1000
    mock_config.model_type = "test_model"
    mock_config.name_or_path = "test/verifier/path"
    mock_config.to_dict.return_value = config_dict

    mock_verifier = MagicMock(spec=PreTrainedModel)
    mock_verifier.config = mock_config
    mock_verifier.name_or_path = "test/verifier/path"
    return mock_verifier


class SpeculatorModelTestConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["test_model_config"] = "test_model_config"
    test_param: int = 123


class SpeculatorTestModel(SpeculatorModel):
    config_class = SpeculatorModelTestConfig  # type: ignore[misc]

    def __init__(
        self,
        config: SpeculatorModelTestConfig,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        verifier_attachment_mode: Literal["detached", "full", "train_only"] | None = (
            None
        ),
        **kwargs,
    ):
        # Initialize without verifier first to avoid weight initialization issues
        super().__init__(
            config,
            verifier=None,
            verifier_attachment_mode="detached",
            **kwargs,
        )
        self.test_module = nn.Linear(10, 10)
        self.post_init()  # type: ignore[attr-defined]

        # Now attach verifier if provided
        if verifier is not None and verifier_attachment_mode != "detached":
            self.attach_verifier(
                verifier, mode=verifier_attachment_mode, add_to_config=False
            )

    def forward(self, *args, **kwargs):
        # Simple implementation for testing
        return {"logits": torch.randn(1, 10, 1000)}


class TestSpeculatorModel:
    @pytest.fixture(
        params=[
            {"test_param": 456},
            {"test_param": 123},
        ],
        ids=["custom_param", "default_param"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for SpeculatorModel."""
        constructor_args = request.param
        config = SpeculatorModelTestConfig(
            **constructor_args,
            speculators_config=SpeculatorsConfig(
                algorithm="test_algorithm",
                proposal_methods=[GreedyTokenProposalConfig()],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None,
                    architectures=["TestModel"],
                ),
            ),
        )
        instance = SpeculatorTestModel(config)
        return instance, constructor_args

    @pytest.fixture
    def test_config(self):
        return SpeculatorModelTestConfig(
            test_param=456,
            speculators_config=SpeculatorsConfig(
                algorithm="test_algorithm",
                proposal_methods=[GreedyTokenProposalConfig()],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None,
                    architectures=["TestModel"],
                ),
            ),
        )

    def setup_method(self):
        self._original_model_registry = (
            SpeculatorModel.registry.copy()  # type: ignore[misc]
            if SpeculatorModel.registry  # type: ignore[misc]
            else {}
        )
        SpeculatorModel.register_decorator(SpeculatorTestModel, "test_model")
        self._original_config_registry = (
            SpeculatorModelConfig.registry.copy()  # type: ignore[misc]
            if SpeculatorModelConfig.registry  # type: ignore[misc]
            else {}
        )
        SpeculatorModelConfig.register_decorator(
            SpeculatorModelTestConfig, "test_model_config"
        )

    def teardown_method(self):
        SpeculatorModel.registry = self._original_model_registry  # type: ignore[misc]
        SpeculatorModelConfig.registry = self._original_config_registry  # type: ignore[misc]
        SpeculatorModelConfig.reload_schema()

    @pytest.mark.smoke
    def test_class_attributes(self):
        assert SpeculatorModel.auto_package == "speculators.models"
        assert SpeculatorModel.registry_auto_discovery is True
        assert SpeculatorModel.config_class == SpeculatorModelConfig
        assert SpeculatorModel.base_model_prefix == "model"
        assert SpeculatorModel.main_input_name == "input_ids"

    @pytest.mark.smoke
    def test_registry_contains_test_model(self):
        assert SpeculatorModel.registry is not None  # type: ignore[misc]
        assert "test_model" in SpeculatorModel.registry  # type: ignore[misc]
        assert SpeculatorModel.registry["test_model"] == SpeculatorTestModel  # type: ignore[misc]

    @pytest.mark.smoke
    def test_registered_model_class_from_config(
        self,
        test_config,
    ):
        model_class = SpeculatorModel.registered_model_class_from_config(test_config)
        assert model_class == SpeculatorTestModel

    @pytest.mark.sanity
    def test_registered_model_class_from_config_invalid(self):
        with pytest.raises(
            TypeError,
            match="Expected config to be an instance of SpeculatorModelConfig",
        ):
            SpeculatorModel.registered_model_class_from_config("invalid_config")  # type: ignore[arg-type]

        config = SpeculatorModelConfig(
            speculators_model_type="test_model_config",
            test_param=456,
            speculators_config=SpeculatorsConfig(
                algorithm="test_algorithm",
                proposal_methods=[],
                default_proposal_method="test_proposal",
                verifier=VerifierConfig(
                    name_or_path="test/verifier",
                    architectures=["TestModel"],
                ),
            ),
        )

        with pytest.raises(
            TypeError,
            match="Received a SpeculatorModelConfig instance but expected a subclass",
        ):
            SpeculatorModel.registered_model_class_from_config(config)

        class UnregisteredConfig(SpeculatorModelConfig):
            speculators_model_type: Literal["unregistered"] = "unregistered"

        config = UnregisteredConfig(
            speculators_config=SpeculatorsConfig(
                algorithm="test_algorithm",
                proposal_methods=[],
                default_proposal_method="test_proposal",
                verifier=VerifierConfig(
                    name_or_path="test/verifier",
                    architectures=["TestModel"],
                ),
            )
        )

        with pytest.raises(
            ValueError, match="No registered model class found for config type"
        ):
            SpeculatorModel.registered_model_class_from_config(config)

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test SpeculatorModel initialization with valid instances."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, SpeculatorTestModel)
        assert instance.config.test_param == constructor_args["test_param"]
        assert instance.verifier is None
        assert instance.verifier_attachment_mode == "detached"
        assert instance.test_module is not None

    @pytest.mark.smoke
    def test_initialization_without_verifier(self, test_config):
        model = SpeculatorTestModel(test_config)
        assert model.config == test_config
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"

    @pytest.mark.smoke
    def test_initialization_with_verifier(self, test_config):
        verifier = create_mock_verifier()
        model = SpeculatorTestModel(test_config, verifier=verifier)
        assert model.config == test_config
        assert model.verifier == verifier
        assert model.verifier_attachment_mode == "full"

    @pytest.mark.smoke
    def test_initialization_with_verifier_path(self, test_config, monkeypatch):
        mock_model = create_mock_verifier()
        mock_from_pretrained = MagicMock(return_value=mock_model)
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained
        )

        verifier_path = "path/to/verifier/model"
        model = SpeculatorTestModel(test_config, verifier=verifier_path)

        mock_from_pretrained.assert_called_once_with(verifier_path)
        assert model.config == test_config
        assert model.verifier is mock_model
        assert model.verifier_attachment_mode == "full"

    @pytest.mark.smoke
    def test_initialization_with_verifier_train_only(
        self,
        test_config,
    ):
        verifier = create_mock_verifier()
        model = SpeculatorTestModel(
            test_config,
            verifier=verifier,
            verifier_attachment_mode="train_only",
        )
        assert model.config == test_config
        assert model.verifier is None
        assert model.verifier_attachment_mode == "train_only"

    @pytest.mark.smoke
    def test_initialization_with_verifier_detached(
        self,
        test_config,
    ):
        verifier = create_mock_verifier()
        model = SpeculatorTestModel(
            test_config,
            verifier=verifier,
            verifier_attachment_mode="detached",
        )
        assert model.config == test_config
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"

    @pytest.mark.sanity
    def test_initialization_invalid(
        self,
        test_config,
    ):
        # No config
        with pytest.raises(
            ValueError, match="Config must be provided to initialize a SpeculatorModel"
        ):
            SpeculatorModel(config=None, verifier=None, verifier_attachment_mode=None)  # type: ignore[arg-type]

        # Invalid config type
        with pytest.raises(
            TypeError,
            match="Expected config to be an instance of SpeculatorModelConfig",
        ):
            SpeculatorModel(
                config="invalid_config",  # type: ignore[arg-type]
                verifier=None,
                verifier_attachment_mode=None,  # type: ignore[arg-type]
            )

        # Invalid verifier type
        with pytest.raises(
            TypeError, match="Expected verifier to be a PreTrainedModel, a string path,"
        ):
            SpeculatorModel(
                config=test_config,
                verifier=123,  # type: ignore[arg-type]
                verifier_attachment_mode=None,
            )

        # Invalid verifier attachment mode
        with pytest.raises(ValueError, match="Invalid verifier_attachment_mode: "):
            SpeculatorModel(
                config=test_config,
                verifier=MagicMock(spec=PreTrainedModel),
                verifier_attachment_mode="invalid_mode",  # type: ignore[arg-type]
            )

    @pytest.mark.smoke
    def test_from_pretrained_config(self, test_config):
        state_dict = SpeculatorTestModel(test_config).state_dict()  # type: ignore[attr-defined]
        model = SpeculatorModel.from_pretrained(
            None, config=test_config, state_dict=state_dict
        )
        assert isinstance(model, SpeculatorTestModel)
        assert model.test_module is not None
        assert (
            model.test_module.weight.abs().sum() > 0
        )  # Ensure weights are initialized
        assert isinstance(model.config, SpeculatorModelTestConfig)
        assert model.config.speculators_model_type == "test_model_config"
        assert model.config.test_param == 456

    @pytest.mark.smoke
    def test_from_pretrained_local_marshalling(
        self,
        test_config,
    ):
        original_model = SpeculatorTestModel(test_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model to a local directory
            original_model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

            # Load the model from the local directory
            loaded_model = SpeculatorModel.from_pretrained(tmpdir)

            assert isinstance(loaded_model, SpeculatorTestModel)
            assert loaded_model.test_module is not None
            assert (
                pytest.approx(
                    (
                        loaded_model.test_module.weight
                        - original_model.test_module.weight
                    )
                    .detach()
                    .abs()
                    .sum()
                )
                == 0
            )
            assert isinstance(loaded_model.config, SpeculatorModelTestConfig)
            assert loaded_model.config.speculators_model_type == "test_model_config"
            assert loaded_model.config.test_param == 456

    @pytest.mark.smoke
    def test_from_pretrained_verifier(
        self,
        test_config,
    ):
        state_dict = SpeculatorTestModel(test_config).state_dict()  # type: ignore[attr-defined]
        verifier = create_mock_verifier()
        model = SpeculatorModel.from_pretrained(
            None,
            config=test_config,
            verifier=verifier,
            state_dict=state_dict,
        )
        assert isinstance(model, SpeculatorTestModel)
        assert model.verifier == verifier
        assert model.verifier_attachment_mode == "full"
        assert isinstance(model.config, SpeculatorModelTestConfig)
        assert model.config.speculators_model_type == "test_model_config"
        assert model.config.test_param == 456

    @pytest.mark.smoke
    def test_from_pretrained_verifier_train_only(self, test_config):
        state_dict = SpeculatorTestModel(test_config).state_dict()  # type: ignore[attr-defined]
        verifier = create_mock_verifier()
        model = SpeculatorModel.from_pretrained(
            None,
            config=test_config,
            verifier=verifier,
            verifier_attachment_mode="train_only",
            state_dict=state_dict,
        )
        assert isinstance(model, SpeculatorTestModel)
        assert model.verifier is None
        assert model.verifier_attachment_mode == "train_only"
        assert isinstance(model.config, SpeculatorModelTestConfig)
        assert model.config.speculators_model_type == "test_model_config"
        assert model.config.test_param == 456

    @pytest.mark.smoke
    def test_from_pretrained_verifier_detached(self, test_config):
        state_dict = SpeculatorTestModel(test_config).state_dict()  # type: ignore[attr-defined]
        verifier = MagicMock(spec=PreTrainedModel)
        model = SpeculatorModel.from_pretrained(
            None,
            config=test_config,
            verifier=verifier,
            verifier_attachment_mode="detached",
            state_dict=state_dict,
        )
        assert isinstance(model, SpeculatorTestModel)
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"
        assert isinstance(model.config, SpeculatorModelTestConfig)
        assert model.config.speculators_model_type == "test_model_config"
        assert model.config.test_param == 456

    @pytest.mark.sanity
    def test_from_pretrained_invalid(self, test_config):
        with pytest.raises(
            ValueError,
            match="Either `config` or `pretrained_model_name_or_path` must be provided",
        ):
            SpeculatorModel.from_pretrained(None)

        with pytest.raises(
            ValueError,
            match=(
                "Either `pretrained_model_name_or_path` or `state_dict` must be "
                "provided"
            ),
        ):
            SpeculatorModel.from_pretrained(None, config=test_config)

        with pytest.raises(
            TypeError,
            match="Expected config to be an instance of SpeculatorModelConfig",
        ):
            SpeculatorModel.from_pretrained("test/path", config="invalid_config")

        with pytest.raises(
            OSError, match="Can't load the model for 'path/does/not/exist'."
        ):
            SpeculatorModel.from_pretrained("path/does/not/exist", config=test_config)

    @pytest.mark.smoke
    def test_forward_concrete(
        self,
        test_config,
    ):
        model = SpeculatorTestModel(test_config, verifier=None)
        result = model.forward()

        assert "logits" in result
        assert result["logits"].shape == (1, 10, 1000)

    @pytest.mark.smoke
    def test_forward_abstract(self, test_config):
        model = SpeculatorModel(
            test_config, verifier=None, verifier_attachment_mode=None
        )

        with pytest.raises(
            NotImplementedError,
            match="The forward method is only supported on concrete",
        ):
            model.forward()

    @pytest.mark.smoke
    def test_attachment_lifecycle(self, test_config):
        model = SpeculatorTestModel(config=test_config)
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"

        # Attach a verifier
        verifier = create_mock_verifier()
        model.attach_verifier(verifier)
        assert model.verifier == verifier
        assert model.verifier_attachment_mode == "full"

        # Ensure attachment before detaching raises an error
        with pytest.raises(
            RuntimeError,
            match=(
                "Cannot attach a verifier when the speculator is not in detached mode."
            ),
        ):
            model.attach_verifier(verifier)

        # Detach the verifier
        model.detach_verifier()
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"

        # Ensure detaching again raises an error
        with pytest.raises(
            RuntimeError,
            match="Verifier is already detached, cannot be called again until",
        ):
            model.detach_verifier()

        # Attach train_only verifier
        model.attach_verifier(verifier, mode="train_only")
        assert model.verifier is None
        assert model.verifier_attachment_mode == "train_only"

        # Detach again
        model.detach_verifier()
        assert model.verifier is None
        assert model.verifier_attachment_mode == "detached"

        # Attach different verifier
        new_verifier = create_mock_verifier()
        model.attach_verifier(new_verifier, mode="full")
        assert model.verifier == new_verifier
        assert model.verifier_attachment_mode == "full"
        assert model.verifier != verifier

    @pytest.mark.sanity
    def test_attach_verifier_invalid(
        self,
        test_config,
    ):
        model = SpeculatorTestModel(config=test_config)

        # Invalid verifier type
        with pytest.raises(
            TypeError, match="Expected verifier to be a PreTrainedModel, a string path,"
        ):
            model.attach_verifier(123)  # type: ignore[arg-type]

        # Invalid attachment mode
        with pytest.raises(
            ValueError, match="Invalid verifier_attachment_mode: invalid_mode"
        ):
            model.attach_verifier(verifier=None, mode="invalid_mode")  # type: ignore[arg-type]

        # Attaching when not in detached mode
        model.verifier_attachment_mode = "full"
        with pytest.raises(
            RuntimeError,
            match=(
                "Cannot attach a verifier when the speculator is not in detached mode."
            ),
        ):
            model.attach_verifier(create_mock_verifier())

    @pytest.mark.smoke
    def test_resolve_verifier(self, test_config):
        model = SpeculatorTestModel(config=test_config)

        # Test with PreTrainedModel instance
        verifier = create_mock_verifier()
        resolved = model.resolve_verifier(verifier)
        assert resolved is verifier

        # Test with invalid verifier type
        with pytest.raises(
            TypeError, match="Expected verifier to be a PreTrainedModel, a string path,"
        ):
            model.resolve_verifier(123)  # type: ignore[arg-type]

        # Test with None
        with pytest.raises(
            ValueError, match="Verifier must be provided as a path, identifier"
        ):
            model.resolve_verifier(None)  # type: ignore[arg-type]

    @pytest.mark.smoke
    def test_state_dict(self, test_config):
        verifier = create_mock_verifier()
        model = SpeculatorTestModel(test_config, verifier=verifier)

        # Get state dict - should exclude verifier parameters
        state = model.state_dict()
        assert isinstance(state, dict)
        assert "test_module.weight" in state
        assert "test_module.bias" in state
        # Verifier parameters should not be in state dict
        assert not any("verifier" in key for key in state)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances):
        """Test SpeculatorModel serialization and deserialization."""
        instance, constructor_args = valid_instances

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model
            instance.save_pretrained(tmpdir)  # type: ignore[attr-defined]

            # Load the model back
            loaded_model = SpeculatorModel.from_pretrained(tmpdir)

            assert isinstance(loaded_model, SpeculatorTestModel)
            assert loaded_model.config.test_param == constructor_args["test_param"]
            assert loaded_model.config.speculators_model_type == "test_model_config"


def test_reload_and_populate_models():
    """Test module-level function for reloading and populating models."""
    # This should not raise any errors
    reload_and_populate_models()
    # Verify that registry is populated
    assert SpeculatorModel.registry is not None  # type: ignore[misc]
    assert len(SpeculatorModel.registry) > 0  # type: ignore[misc]


class TestReloadAndPopulateModels:
    """Test suite for reload_and_populate_models function."""

    @pytest.mark.smoke
    def test_invocation(self):
        """Test reload_and_populate_models function execution."""
        # This should not raise any errors
        reload_and_populate_models()
        # Verify that registry is populated
        assert SpeculatorModel.registry is not None  # type: ignore[misc]
        assert len(SpeculatorModel.registry) > 0  # type: ignore[misc]
