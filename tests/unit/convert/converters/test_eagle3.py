"""
Unit tests for the Eagle3 converter module in the Speculators library.
"""

from __future__ import annotations

import contextlib
import pkgutil
from abc import ABC
from pathlib import Path
from typing import Generic, Literal

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from speculators.convert import converters
from speculators.convert.converters import (
    Eagle3SpeculatorConverter,
    SpeculatorConverter,
)
from speculators.models.eagle3 import Eagle3Speculator
from speculators.utils import RegistryMixin
from tests.unit.mock import MockPretrainedTransformersFactory, PretrainedBundle


def mock_instance_types() -> list[
    dict[
        Literal["model_type", "config_type", "verifier_type"],
        Literal["hf_id", "path", "instance", "dict", "none"],
    ]
]:
    instance_types = []

    for model_type in ["hf_id", "path", "instance"]:
        for config_type, verifier_type in [
            ("hf_id", "hf_id"),
            ("path", "path"),
            ("instance", "instance"),
            ("dict", "none"),
        ]:
            instance_types.append(
                {
                    "model_type": model_type,
                    "config_type": config_type,
                    "verifier_type": verifier_type,
                }
            )

    return instance_types


def mock_instance_ids() -> list[str]:
    return [
        f"model-{types['model_type']}_config-{types['config_type']}_verifier-{types['verifier_type']}"
        for types in mock_instance_types()
    ]


def create_instances(  # noqa: C901, PLR0912
    instance_types: dict[
        Literal["model_type", "config_type", "verifier_type"],
        Literal["hf_id", "path", "instance", "dict", "none"],
    ],
    speculator: PretrainedBundle,
    verifier: PretrainedBundle,
) -> tuple[
    str | Path | PreTrainedModel,
    str | Path | PretrainedConfig | dict,
    str | Path | PreTrainedModel | None,
]:
    model_type = instance_types["model_type"]
    config_type = instance_types["config_type"]
    verifier_type = instance_types["verifier_type"]

    if model_type == "hf_id":
        model = speculator.name_or_path
    elif model_type == "path":
        model = speculator.local_dir
    elif model_type == "instance":
        model = speculator.model
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    if config_type == "hf_id":
        config = speculator.name_or_path
    elif config_type == "path":
        config = speculator.local_dir
    elif config_type == "instance":
        config = speculator.config
    elif config_type == "dict":
        config = speculator.config.to_dict()
    else:
        raise ValueError(f"Invalid config_type: {config_type}")

    if verifier_type == "hf_id":
        verifier = verifier.name_or_path
    elif verifier_type == "path":
        verifier = verifier.local_dir
    elif verifier_type == "instance":
        verifier = verifier.model
    elif verifier_type == "none":
        verifier = None
    else:
        raise ValueError(f"Invalid verifier_type: {verifier_type}")

    return model, config, verifier


class TestEagle3SpeculatorConverter:
    """Test class for Eagle3SpeculatorConverter functionality."""

    @pytest.fixture(
        params=mock_instance_types(),
        ids=mock_instance_ids(),
    )
    def valid_instances(
        self,
        request,
        mock_safeai_eagle3: tuple[
            PretrainedBundle, PretrainedBundle, MockPretrainedTransformersFactory
        ],
    ):
        """Mock Eagle3 speculator and verifier instances with various input types."""
        speculator, verifier, mock_pretrained_factory = mock_safeai_eagle3
        # Mock the required utilities and functions for all converter modules
        # This is needed because resolve_converter with algorithm="auto" will
        # call is_supported on ALL registered converters
        for _, module_name, _ in pkgutil.iter_modules(converters.__path__):
            with contextlib.suppress(AttributeError):
                mock_pretrained_factory.patch_transformers_utils(
                    f"speculators.convert.converters.{module_name}"
                )

        instance_types = request.param
        model, config, verifier = create_instances(instance_types, speculator, verifier)

        norm_before_residual = hash(str(instance_types)) % 2 == 0
        remove_embed_tokens = verifier is not None and (
            hash(instance_types["verifier_type"] + instance_types["model_type"]) % 2
            == 0
        )
        if remove_embed_tokens:
            speculator.model.state_dict().pop("embed_tokens.weight")

        return model, config, verifier, norm_before_residual, mock_safeai_eagle3

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Eagle3SpeculatorConverter inheritance and type relationships."""

        assert issubclass(Eagle3SpeculatorConverter, SpeculatorConverter)
        assert issubclass(Eagle3SpeculatorConverter, ABC)
        assert issubclass(Eagle3SpeculatorConverter, Generic)
        assert issubclass(Eagle3SpeculatorConverter, RegistryMixin)

        # Test class variables
        assert hasattr(Eagle3SpeculatorConverter, "parameter_mappings")
        assert isinstance(Eagle3SpeculatorConverter.parameter_mappings, dict)
        assert Eagle3SpeculatorConverter.parameter_mappings == {
            "midlayer.": "layers.0."
        }
        assert hasattr(Eagle3SpeculatorConverter, "keep_parameters")
        assert isinstance(Eagle3SpeculatorConverter.keep_parameters, list)
        assert Eagle3SpeculatorConverter.keep_parameters == [
            "d2t",
            "t2d",
            "embed_tokens.",
            "fc.",
            "lm_head.",
            "norm.",
        ]

        # Test method signatures
        assert hasattr(Eagle3SpeculatorConverter, "is_supported")
        assert callable(Eagle3SpeculatorConverter.is_supported)
        assert hasattr(Eagle3SpeculatorConverter, "is_supported_param")
        assert callable(Eagle3SpeculatorConverter.is_supported_param)
        assert hasattr(Eagle3SpeculatorConverter, "__init__")
        assert callable(Eagle3SpeculatorConverter)
        assert hasattr(Eagle3SpeculatorConverter, "convert_config_state_dict")
        assert hasattr(Eagle3SpeculatorConverter, "validate")

    @pytest.mark.smoke
    def test_registry_registration(self):
        """Test that Eagle3SpeculatorConverter is properly registered."""
        assert SpeculatorConverter.registry is not None
        assert "eagle3" in SpeculatorConverter.registry
        assert SpeculatorConverter.registry["eagle3"] is Eagle3SpeculatorConverter

    @pytest.mark.smoke
    def test_resolve(
        self,
        valid_instances: tuple[
            str | Path | PreTrainedModel,
            str | Path | PretrainedConfig | dict,
            str | Path | PreTrainedModel | None,
            bool,
            tuple[PretrainedBundle, PretrainedBundle],
        ],
    ):
        """Test resolve_converter and is_supported for valid Eagle3 checkpoints."""
        model, config, verifier, norm_before_residual, _ = valid_instances

        # Is supported
        result = Eagle3SpeculatorConverter.is_supported(
            model=model,
            config=config,
            verifier=verifier,
            norm_before_residual=norm_before_residual,
        )
        assert result is True

        # Resolve by algorithm name
        alg_class = SpeculatorConverter.resolve_converter(
            algorithm="eagle3",
            model=model,
            config=config,
            verifier=verifier,
            norm_before_residual=norm_before_residual,
        )
        assert alg_class is Eagle3SpeculatorConverter

        # Resolve by auto
        auto_class = SpeculatorConverter.resolve_converter(
            algorithm="auto",
            model=model,
            config=config,
            verifier=verifier,
            norm_before_residual=norm_before_residual,
        )
        assert auto_class is Eagle3SpeculatorConverter

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("param_name", "expected_supported", "expected_mapped_name"),
        [
            # Parameter mappings (midlayer -> layers.0)
            (
                "midlayer.self_attn.q_proj.weight",
                True,
                "layers.0.self_attn.q_proj.weight",
            ),
            ("midlayer.mlp.gate_proj.weight", True, "layers.0.mlp.gate_proj.weight"),
            (
                "midlayer.input_layernorm.weight",
                True,
                "layers.0.input_layernorm.weight",
            ),
            # Keep parameters (direct mapping)
            ("d2t", True, "d2t"),
            ("t2d", True, "t2d"),
            ("embed_tokens.weight", True, "embed_tokens.weight"),
            ("fc.weight", True, "fc.weight"),
            ("fc.bias", True, "fc.bias"),
            ("lm_head.weight", True, "lm_head.weight"),
            ("norm.weight", True, "norm.weight"),
            # Model prefix handling
            (
                "model.midlayer.self_attn.q_proj.weight",
                True,
                "layers.0.self_attn.q_proj.weight",
            ),
            ("model.embed_tokens.weight", True, "embed_tokens.weight"),
            ("model.fc.weight", True, "fc.weight"),
            # Unsupported parameters
            ("unsupported_param", False, "unsupported_param"),
            ("random.weight", False, "random.weight"),
            ("transformer.layer.0.weight", False, "transformer.layer.0.weight"),
        ],
        ids=[
            "midlayer_attn",
            "midlayer_mlp",
            "midlayer_norm",
            "d2t",
            "t2d",
            "embed_tokens",
            "fc_weight",
            "fc_bias",
            "lm_head",
            "norm",
            "model_midlayer",
            "model_embed",
            "model_fc",
            "unsupported",
            "random",
            "transformer",
        ],
    )
    def test_is_supported_param(
        self, param_name, expected_supported, expected_mapped_name
    ):
        """Test parameter name support and mapping logic."""
        is_supported, mapped_name = Eagle3SpeculatorConverter.is_supported_param(
            param_name
        )
        assert is_supported == expected_supported
        assert mapped_name == expected_mapped_name

    @pytest.mark.smoke
    def test_initialization(
        self,
        valid_instances: tuple[
            str | Path | PreTrainedModel,
            str | Path | PretrainedConfig | dict,
            str | Path | PreTrainedModel | None,
            bool,
            tuple[PretrainedBundle, PretrainedBundle],
        ],
    ):
        model, config, verifier, norm_before_residual, _ = valid_instances
        converter = Eagle3SpeculatorConverter(
            model=model,
            config=config,
            verifier=verifier,
            norm_before_residual=norm_before_residual,
        )

        assert converter.model is model
        assert converter.config is config
        assert converter.verifier is verifier
        assert converter.norm_before_residual is norm_before_residual

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test Eagle3SpeculatorConverter initialization without required fields."""
        with pytest.raises(TypeError):
            Eagle3SpeculatorConverter()  # type: ignore[call-arg]

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "validate_device",
        [
            None,
            "cpu" if not torch.cuda.is_available() else "cuda",
        ],
        ids=["no_validation", "device_validation"],
    )
    def test_call_lifecycle(
        self,
        valid_instances: tuple[
            str | Path | PreTrainedModel,
            str | Path | PretrainedConfig | dict,
            str | Path | PreTrainedModel | None,
            bool,
            tuple[PretrainedBundle, PretrainedBundle],
        ],
        validate_device: Literal["cpu", "cuda"] | None,
        tmp_path: Path,
    ):
        model, config, verifier, norm_before_residual, _ = valid_instances
        converter = Eagle3SpeculatorConverter(
            model=model,
            config=config,
            verifier=verifier,
            norm_before_residual=norm_before_residual,
        )
        converted = converter(output_path=tmp_path, validate_device=validate_device)

        assert isinstance(converted, Eagle3Speculator)
        assert converted.config is not None
        assert hasattr(converted, "embed_tokens")
        assert hasattr(converted, "layers")
        assert hasattr(converted, "norm")
        assert hasattr(converted, "lm_head")

        # Test that model was saved if output_path provided
        config_path = tmp_path / "config.json"
        assert config_path.exists()

        # Check for various possible model file formats
        possible_model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        ]
        model_file_exists = any(
            (tmp_path / fname).exists() for fname in possible_model_files
        )
        assert model_file_exists, (
            f"No model file found. Files in {tmp_path}: {list(tmp_path.iterdir())}"
        )

        # Test internal state consistency
        assert converter.model is model
        assert converter.config is config
        assert converter.verifier is verifier
        assert converter.norm_before_residual is norm_before_residual
