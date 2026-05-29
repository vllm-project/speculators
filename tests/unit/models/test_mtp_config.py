"""Unit tests for MTPConfig."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from speculators import SpeculatorModelConfig, SpeculatorsConfig, VerifierConfig
from speculators.models.mtp import MTPConfig
from speculators.models.mtp.model_definitions import mtp_model_classes
from speculators.proposals import GreedyTokenProposalConfig
from tests.conftest import requires_transformers_version

# ===== Fixtures =====


@pytest.fixture
def speculators_config():
    return SpeculatorsConfig(
        algorithm="mtp",
        proposal_methods=[
            GreedyTokenProposalConfig(
                speculative_tokens=3,
                verifier_accept_k=1,
                accept_tolerance=0.0,
            )
        ],
        default_proposal_method="greedy",
        verifier=VerifierConfig(
            name_or_path="Qwen/Qwen3.5-0.8B",
            architectures=["Qwen3_5ForCausalLM"],
        ),
    )


@pytest.fixture
def mtp_config(qwen3_5_pretrained_config, speculators_config):
    return MTPConfig(
        transformer_layer_config=qwen3_5_pretrained_config,
        speculators_config=speculators_config,
    )


# ===== Construction and derived properties =====


class TestConstruction:
    def test_default_initialization(self):
        config = MTPConfig()
        assert config.speculators_model_type == "mtp"
        assert config.architectures == ["MTPDraftModel"]
        assert config.num_nextn_predict_layers == 1
        assert config.model_type == "speculator_model"

    def test_derived_hidden_size(self, mtp_config, qwen3_5_pretrained_config):
        assert mtp_config.hidden_size == qwen3_5_pretrained_config.hidden_size

    def test_derived_vocab_size(self, mtp_config, qwen3_5_pretrained_config):
        assert mtp_config.vocab_size == qwen3_5_pretrained_config.vocab_size

    def test_derived_num_speculative_steps(self, mtp_config):
        assert mtp_config.num_speculative_steps == 3

    def test_transformer_layer_config_preserved(
        self, mtp_config, qwen3_5_pretrained_config
    ):
        assert (
            mtp_config.transformer_layer_config.hidden_size
            == qwen3_5_pretrained_config.hidden_size
        )
        assert (
            mtp_config.transformer_layer_config.num_attention_heads
            == qwen3_5_pretrained_config.num_attention_heads
        )


# ===== Validation =====


class TestValidation:
    def test_num_nextn_predict_layers_rejects_zero(self):
        with pytest.raises(ValidationError, match="1 layer"):
            MTPConfig(num_nextn_predict_layers=0)

    def test_num_nextn_predict_layers_rejects_two(self):
        with pytest.raises(ValidationError, match="1 layer"):
            MTPConfig(num_nextn_predict_layers=2)

    def test_num_nextn_predict_layers_accepts_one(self):
        config = MTPConfig(num_nextn_predict_layers=1)
        assert config.num_nextn_predict_layers == 1

    def test_invalid_speculators_model_type(self):
        with pytest.raises(ValidationError, match="speculators_model_type"):
            MTPConfig(speculators_model_type="invalid")  # type: ignore[arg-type]


# ===== Round-trip serialization =====


class TestSerialization:
    def test_to_dict_roundtrip(self, mtp_config):
        config_dict = mtp_config.to_dict()
        recreated = MTPConfig(**config_dict)

        assert recreated.speculators_model_type == "mtp"
        assert recreated.hidden_size == mtp_config.hidden_size
        assert recreated.vocab_size == mtp_config.vocab_size
        assert recreated.num_nextn_predict_layers == mtp_config.num_nextn_predict_layers
        assert recreated.architectures == mtp_config.architectures

    def test_model_dump_roundtrip(self, mtp_config):
        dumped = mtp_config.model_dump()
        recreated = MTPConfig.model_validate(dumped)

        assert isinstance(recreated, MTPConfig)
        assert recreated.speculators_model_type == "mtp"
        assert recreated.num_nextn_predict_layers == mtp_config.num_nextn_predict_layers

    def test_model_validate_via_base_class(self, mtp_config):
        dumped = mtp_config.model_dump()
        recreated = SpeculatorModelConfig.model_validate(dumped)

        assert isinstance(recreated, MTPConfig)
        assert recreated.speculators_model_type == "mtp"

    def test_from_dict(self, mtp_config):
        config_dict = mtp_config.to_dict()
        recreated = SpeculatorModelConfig.from_dict(config_dict)

        assert isinstance(recreated, MTPConfig)
        assert recreated.hidden_size == mtp_config.hidden_size

    def test_save_and_load_pretrained(self, mtp_config):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mtp_config.save_pretrained(tmp_path)

            assert (tmp_path / "config.json").exists()

            loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
            assert isinstance(loaded, MTPConfig)
            assert loaded.speculators_model_type == "mtp"
            assert loaded.hidden_size == mtp_config.hidden_size
            assert loaded.vocab_size == mtp_config.vocab_size
            assert (
                loaded.num_nextn_predict_layers == mtp_config.num_nextn_predict_layers
            )

    def test_transformer_config_serialized_as_dict(self, mtp_config):
        dumped = mtp_config.model_dump()
        assert isinstance(dumped["transformer_layer_config"], dict)


# ===== Registration =====


class TestRegistration:
    def test_registered_as_mtp(self):
        assert SpeculatorModelConfig.registry is not None
        assert "mtp" in SpeculatorModelConfig.registry
        assert SpeculatorModelConfig.registry["mtp"] == MTPConfig

    def test_discoverable_in_registered_classes(self):
        registered = SpeculatorModelConfig.registered_classes()
        class_names = [cls.__name__ for cls in registered]
        assert "MTPConfig" in class_names


# ===== Multi-architecture parametrization =====

_ALL_MODEL_TYPES = [
    pytest.param("qwen3", id="qwen3"),
    pytest.param(
        "qwen3_next",
        id="qwen3_next",
        marks=requires_transformers_version("4.57.0"),
    ),
    pytest.param(
        "qwen3_5_text",
        id="qwen3_5_text",
        marks=requires_transformers_version("5.2.0"),
    ),
    pytest.param(
        "qwen3_5_moe_text",
        id="qwen3_5_moe_text",
        marks=requires_transformers_version("5.2.0"),
    ),
]


@pytest.mark.parametrize("model_type", _ALL_MODEL_TYPES)
class TestModelTypes:
    def test_model_type_available(self, model_type):
        assert model_type in mtp_model_classes

    def test_config_construction_with_model_type(
        self, model_type, qwen3_5_pretrained_config, speculators_config
    ):
        config = MTPConfig(
            transformer_layer_config=qwen3_5_pretrained_config,
            speculators_config=speculators_config,
        )
        assert config.speculators_model_type == "mtp"
        assert config.hidden_size == qwen3_5_pretrained_config.hidden_size
