"""
Unit tests for the model module in the Speculators library.
"""

import tempfile
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from torch import nn

from speculators import (
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
    reload_schemas,
)
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.mtp.core import MTPDraftModel
from speculators.models.peagle.core import PEagleDraftModel
from speculators.proposals import GreedyTokenProposalConfig

# ===== Test Helper Classes =====


@SpeculatorModelConfig.register("test_speculator_model")
class SpeculatorModelTestConfig(SpeculatorModelConfig):
    speculators_model_type: Literal["test_speculator_model"] = "test_speculator_model"
    test_param: int = 123


@SpeculatorModel.register("test_speculator")
class SpeculatorTestModel(SpeculatorModel):
    config_class = SpeculatorModelTestConfig  # type: ignore[misc]

    def __init__(self, config: SpeculatorModelTestConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.test_module = nn.Linear(10, 10)
        self.post_init()  # type: ignore[attr-defined]

    def forward(self, *args, **kwargs):
        # Simple implementation for testing
        return {"logits": torch.randn(1, 10, 1000)}

    @classmethod
    def from_training_args(cls, verifier_config, **kwargs):
        """Create model from training arguments."""
        config = SpeculatorModelTestConfig(
            speculators_config=SpeculatorsConfig(
                algorithm="test_speculator",
                proposal_methods=[GreedyTokenProposalConfig()],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs.get("verifier_name_or_path")
                ),
            )
        )
        return cls(config=config)

    @staticmethod
    def get_trainer_kwargs(**kwargs):
        """Get training and validation kwargs."""
        return {}, {}


# Reload registries to include test classes
reload_schemas()


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
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
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
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
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
def test_speculator_model_initialization(speculator_model_test_config):
    model = SpeculatorTestModel(speculator_model_test_config)
    assert model.config == speculator_model_test_config


@pytest.mark.sanity
def test_speculator_model_initialization_invalid():
    # No config
    with pytest.raises(
        ValueError, match="Config must be provided to initialize a SpeculatorModel"
    ):
        SpeculatorModel(config=None)  # type: ignore[abstract, arg-type]

    # Invalid config type
    with pytest.raises(
        TypeError, match="Expected config to be an instance of SpeculatorModelConfig"
    ):
        SpeculatorModel(  # type: ignore[abstract]
            config="invalid_config",  # type: ignore[arg-type]
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
def test_speculator_model_from_pretrained(
    speculator_model_test_config,
):
    state_dict = SpeculatorTestModel(speculator_model_test_config).state_dict()  # type: ignore[attr-defined]
    model = SpeculatorModel.from_pretrained(
        None, config=speculator_model_test_config, state_dict=state_dict
    )
    assert isinstance(model, SpeculatorTestModel)
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

    with pytest.raises(OSError, match="'path/does/not/exist'."):
        SpeculatorModel.from_pretrained(
            "path/does/not/exist", config=speculator_model_test_config
        )


def test_from_pretrained_forwards_hub_options_to_external_conversion():
    hub_options = {
        "cache_dir": "/tmp/cache",
        "force_download": True,
        "local_files_only": True,
        "token": "token",
        "revision": "test-revision",
    }
    with (
        patch(
            "speculators.model.PretrainedConfig.get_config_dict",
            return_value=({"dflash_config": {}}, {}),
        ) as get_config,
        patch(
            "speculators.convert.entrypoints.maybe_convert_external_checkpoint",
            side_effect=RuntimeError("conversion sentinel"),
        ) as convert,
        pytest.raises(RuntimeError, match="conversion sentinel"),
    ):
        SpeculatorModel.from_pretrained(
            "external/checkpoint", verifier="verifier", **hub_options
        )

    get_config.assert_called_once_with("external/checkpoint", **hub_options)
    convert.assert_called_once_with(
        "external/checkpoint",
        verifier="verifier",
        config_dict={"dflash_config": {}},
        **hub_options,
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
    model = SpeculatorModel(  # type: ignore[abstract]
        speculator_model_test_config, verifier=None, verifier_attachment_mode=None
    )

    with pytest.raises(
        NotImplementedError, match="The forward method is only supported on concrete"
    ):
        model.forward()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_class",
    [Eagle3DraftModel, DFlashDraftModel, PEagleDraftModel, MTPDraftModel],
)
def test_save_ignore_keys_are_ignored_on_load_missing(model_class):
    """Weights excluded from saved checkpoints (e.g. verifier_lm_head, which is
    reloaded from the verifier via load_verifier_weights) must also be ignored when
    missing on load. Otherwise loading an initialized/trained checkpoint flags the
    absent key as missing.
    """
    save_ignore = set(getattr(model_class, "_keys_to_ignore_on_save", None) or [])
    load_missing_ignore = set(
        getattr(model_class, "_keys_to_ignore_on_load_missing", None) or []
    )

    not_ignored_on_load = save_ignore - load_missing_ignore
    assert not not_ignored_on_load, (
        f"{model_class.__name__} excludes {sorted(not_ignored_on_load)} from saved "
        "checkpoints but does not list them in _keys_to_ignore_on_load_missing; "
        "loading a checkpoint will raise on the absent key(s)."
    )
