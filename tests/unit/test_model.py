"""
Unit tests for the model module in the Speculators library.
"""

import os
import tempfile
from typing import Literal, Optional, Union
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from speculators import (
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_and_populate_configs,
    reload_and_populate_models,
)
from speculators.proposals import GreedyTokenProposalConfig

# ===== Test Helper Classes =====


@SpeculatorModelConfig.register("test_speculator_model")
class SpeculatorModelTestConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["test_speculator_model"] = "test_speculator_model"
    test_param: int = 123


@SpeculatorModel.register("test_speculator")
class SpeculatorTestModel(SpeculatorModel):
    config_class = SpeculatorModelTestConfig

    def __init__(
        self,
        config: SpeculatorModelTestConfig,
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
        verifier_attachment_mode: Optional[
            Literal["detached", "full", "train_only"]
        ] = None,
        **kwargs,
    ):
        super().__init__(
            config,
            verifier=verifier,
            verifier_attachment_mode=verifier_attachment_mode,
            **kwargs,
        )
        self.test_module = nn.Linear(10, 10)
        self.post_init()  # type: ignore[attr-defined]

    def forward(self, *args, **kwargs):
        # Simple implementation for testing
        return {"logits": torch.randn(1, 10, 1000)}


# Reload registries to include test classes
reload_and_populate_configs()
reload_and_populate_models()


@pytest.fixture
def speculator_model_test_config():
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


# ===== SpeculatorModel Class Attributes Tests =====


@pytest.mark.smoke
def test_speculator_model_class_attributes():
    assert SpeculatorModel.auto_package == "speculators.models"
    assert SpeculatorModel.registry_auto_discovery is True
    assert SpeculatorModel.config_class == SpeculatorModelConfig
    assert SpeculatorModel.base_model_prefix == "model"
    assert SpeculatorModel.main_input_name == "input_ids"


# ===== SpeculatorModel Registry Tests =====


@pytest.mark.smoke
def test_speculator_model_registry_contains_test_model():
    assert SpeculatorModel.registry is not None
    assert "test_speculator" in SpeculatorModel.registry
    assert SpeculatorModel.registry["test_speculator"] == SpeculatorTestModel


@pytest.mark.smoke
def test_speculator_model_registered_model_class_from_config(
    speculator_model_test_config,
):
    model_class = SpeculatorModel.registered_model_class_from_config(
        speculator_model_test_config
    )
    assert model_class == SpeculatorTestModel


@pytest.mark.sanity
def test_speculator_model_registered_model_class_from_config_invalid():
    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel.registered_model_class_from_config("invalid_config")  # type: ignore[arg-type]

    config = SpeculatorModelConfig(
        speculators_model_type="test_speculator_model",
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


# # ===== SpeculatorModel Initialization Tests =====


@pytest.mark.smoke
def test_speculator_model_initialization_without_verifier(speculator_model_test_config):
    model = SpeculatorTestModel(speculator_model_test_config)
    assert model.config == speculator_model_test_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"


@pytest.mark.smoke
def test_speculator_model_initialization_with_verifier(speculator_model_test_config):
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorTestModel(speculator_model_test_config, verifier=verifier)
    assert model.config == speculator_model_test_config
    assert model.verifier == verifier
    assert model.verifier_attachment_mode == "full"


@pytest.mark.smoke
def test_speculator_model_initialization_with_verifier_path(
    speculator_model_test_config, monkeypatch
):
    mock_model = MagicMock(spec=PreTrainedModel)
    mock_from_pretrained = MagicMock(return_value=mock_model)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained
    )

    verifier_path = "path/to/verifier/model"
    model = SpeculatorTestModel(speculator_model_test_config, verifier=verifier_path)

    mock_from_pretrained.assert_called_once_with(verifier_path)
    assert model.config == speculator_model_test_config
    assert model.verifier is mock_model
    assert model.verifier_attachment_mode == "full"


@pytest.mark.smoke
def test_speculator_model_initialization_with_verifier_train_only(
    speculator_model_test_config,
):
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorTestModel(
        speculator_model_test_config,
        verifier=verifier,
        verifier_attachment_mode="train_only",
    )
    assert model.config == speculator_model_test_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "train_only"


@pytest.mark.smoke
def test_speculator_model_initialization_with_verifier_detached(
    speculator_model_test_config,
):
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorTestModel(
        speculator_model_test_config,
        verifier=verifier,
        verifier_attachment_mode="detached",
    )
    assert model.config == speculator_model_test_config
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"


@pytest.mark.sanity
def test_speculator_model_initialization_invalid(
    speculator_model_test_config,
):
    # No config
    with pytest.raises(
        ValueError, match="Config must be provided to initialize a SpeculatorModel"
    ):
        SpeculatorModel(config=None, verifier=None, verifier_attachment_mode=None)  # type: ignore[arg-type]

    # Invalid config type
    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
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
            config=speculator_model_test_config,
            verifier=123,  # type: ignore[arg-type]
            verifier_attachment_mode=None,
        )

    # Invalid verifier attachment mode
    with pytest.raises(ValueError, match="Invalid verifier_attachment_mode: "):
        SpeculatorModel(
            config=speculator_model_test_config,
            verifier=MagicMock(spec=PreTrainedModel),
            verifier_attachment_mode="invalid_mode",  # type: ignore[arg-type]
        )


# ===== SpeculatorModel from_pretrained Tests =====


@pytest.mark.smoke
def test_speculator_model_from_pretrained_config(speculator_model_test_config):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    model = SpeculatorModel.from_pretrained(
        None, config=speculator_model_test_config, state_dict=state_dict
    )
    assert isinstance(model, SpeculatorTestModel)
    assert model.test_module is not None
    assert model.test_module.weight.abs().sum() > 0  # Ensure weights are initialized
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.smoke
def test_speculator_model_from_pretrained_local_marshalling(
    speculator_model_test_config,
):
    original_model = SpeculatorTestModel(speculator_model_test_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the model to a local directory
        original_model.save_pretrained(tmpdir)  # type: ignore[attr-defined]

        # Load the model from the local directory
        loaded_model = SpeculatorModel.from_pretrained(tmpdir)

        assert isinstance(loaded_model, SpeculatorTestModel)
        assert loaded_model.test_module is not None
        assert (
            pytest.approx(
                (loaded_model.test_module.weight - original_model.test_module.weight)
                .detach()
                .abs()
                .sum()
            )
            == 0
        )
        assert isinstance(loaded_model.config, SpeculatorModelTestConfig)
        assert loaded_model.config.speculators_model_type == "test_speculator_model"
        assert loaded_model.config.test_param == 456


@pytest.mark.smoke
def test_speculator_model_from_pretrained_verifier(
    speculator_model_test_config,
):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorModel.from_pretrained(
        None,
        config=speculator_model_test_config,
        verifier=verifier,
        state_dict=state_dict,
    )
    assert isinstance(model, SpeculatorTestModel)
    assert model.verifier == verifier
    assert model.verifier_attachment_mode == "full"
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.smoke
def test_speculator_model_from_pretrained_verifier_train_only(
    speculator_model_test_config,
):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorModel.from_pretrained(
        None,
        config=speculator_model_test_config,
        verifier=verifier,
        verifier_attachment_mode="train_only",
        state_dict=state_dict,
    )
    assert isinstance(model, SpeculatorTestModel)
    assert model.verifier is None
    assert model.verifier_attachment_mode == "train_only"
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.smoke
def test_speculator_model_from_pretrained_verifier_detached(
    speculator_model_test_config,
):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    verifier = MagicMock(spec=PreTrainedModel)
    model = SpeculatorModel.from_pretrained(
        None,
        config=speculator_model_test_config,
        verifier=verifier,
        verifier_attachment_mode="detached",
        state_dict=state_dict,
    )
    assert isinstance(model, SpeculatorTestModel)
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"
    assert isinstance(model.config, SpeculatorModelTestConfig)
    assert model.config.speculators_model_type == "test_speculator_model"
    assert model.config.test_param == 456


@pytest.mark.sanity
def test_speculator_model_from_pretrained_invalid(speculator_model_test_config):
    with pytest.raises(
        ValueError,
        match="Either `config` or `pretrained_model_name_or_path` must be provided",
    ):
        SpeculatorModel.from_pretrained(None)

    with pytest.raises(
        ValueError,
        match="Either `pretrained_model_name_or_path` or `state_dict` must be provided",
    ):
        SpeculatorModel.from_pretrained(None, config=speculator_model_test_config)

    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel.from_pretrained("test/path", config="invalid_config")

    with pytest.raises(
        OSError, match="Can't load the model for 'path/does/not/exist'."
    ):
        SpeculatorModel.from_pretrained(
            "path/does/not/exist", config=speculator_model_test_config
        )


# # ===== SpeculatorModel Forward Method Tests =====


@pytest.mark.smoke
def test_speculator_model_forward_concrete(
    speculator_model_test_config,
):
    model = SpeculatorTestModel(speculator_model_test_config, verifier=None)
    result = model.forward()

    assert "logits" in result
    assert result["logits"].shape == (1, 10, 1000)


@pytest.mark.smoke
def test_speculator_model_forward_abstract(speculator_model_test_config):
    model = SpeculatorModel(
        speculator_model_test_config, verifier=None, verifier_attachment_mode=None
    )

    with pytest.raises(
        NotImplementedError, match="The forward method is only supported on concrete"
    ):
        model.forward()


# ===== SpeculatorModel Verifier Management Tests =====


@pytest.mark.smoke
def test_speculator_model_attachment_lifecycle(speculator_model_test_config):
    model = SpeculatorTestModel(config=speculator_model_test_config)
    assert model.verifier is None
    assert model.verifier_attachment_mode == "detached"

    # Attach a verifier
    verifier = MagicMock(spec=PreTrainedModel)
    model.attach_verifier(verifier)
    assert model.verifier == verifier
    assert model.verifier_attachment_mode == "full"

    # Ensure attachment before detaching raises an error
    with pytest.raises(
        RuntimeError,
        match="Cannot attach a verifier when the speculator is not in detached mode.",
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
    new_verifier = MagicMock(spec=PreTrainedModel)
    model.attach_verifier(new_verifier, mode="full")
    assert model.verifier == new_verifier
    assert model.verifier_attachment_mode == "full"
    assert model.verifier != verifier


@pytest.mark.sanity
def test_speculator_model_attach_verifier_invalid(
    speculator_model_test_config,
):
    model = SpeculatorTestModel(config=speculator_model_test_config)

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
        match="Cannot attach a verifier when the speculator is not in detached mode.",
    ):
        model.attach_verifier(MagicMock(spec=PreTrainedModel))
