"""
Unit tests for the base converter module in the Speculators library.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor, device, nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.convert import SpeculatorConverter
from speculators.convert.converters.base import ConfigT, ModelT
from speculators.utils import RegistryMixin

__all__ = ["ConfigT", "ModelT", "SpeculatorConverter"]


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock(spec=PretrainedConfig)
    return model


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=PretrainedConfig)
    config.to_dict.return_value = {"model_type": "test_model"}
    return config


@pytest.fixture
def mock_verifier():
    """Mock verifier for testing."""
    verifier = MagicMock(spec=PreTrainedModel)
    verifier.config = MagicMock(spec=PretrainedConfig)
    return verifier


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_config_type():
    """Test that ConfigT is configured correctly as a TypeVar."""
    assert isinstance(ConfigT, type(TypeVar("test")))
    assert ConfigT.__name__ == "ConfigT"
    assert ConfigT.__bound__ is SpeculatorModelConfig
    assert ConfigT.__constraints__ == ()


def test_model_type():
    """Test that ModelT is configured correctly as a TypeVar."""
    assert isinstance(ModelT, type(TypeVar("test")))
    assert ModelT.__name__ == "ModelT"
    assert ModelT.__bound__ is SpeculatorModel
    assert ModelT.__constraints__ == ()


class MockSpeculatorConverterImpl(SpeculatorConverter):
    """Test implementation of SpeculatorConverter for unit testing."""

    @classmethod
    def is_supported(
        cls,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        **kwargs,
    ) -> bool:
        """Test implementation that always returns True."""
        return True

    def convert_config_state_dict(
        self,
    ) -> tuple[SpeculatorModelConfig, dict[str, Tensor]]:
        """Test implementation that returns mock config and state dict."""
        mock_config = MagicMock(spec=SpeculatorModelConfig)
        mock_state_dict = {"test_param": torch.tensor([1.0, 2.0, 3.0])}

        return mock_config, mock_state_dict

    def validate(self, model: SpeculatorModel, device: str | device | int):
        """Test implementation that does nothing."""


class MockSpeculatorConverterUnsupported(SpeculatorConverter):
    """Test implementation that is never supported."""

    @classmethod
    def is_supported(
        cls,
        model: str | Path | PreTrainedModel | nn.Module,
        config: str | Path | PretrainedConfig | dict,
        verifier: str | os.PathLike | PreTrainedModel | None = None,
        **kwargs,
    ) -> bool:
        """Test implementation that always returns False."""
        return False

    def convert_config_state_dict(
        self,
    ) -> tuple[SpeculatorModelConfig, dict[str, Tensor]]:
        """Test implementation that returns mock config and state dict."""
        mock_config = MagicMock(spec=SpeculatorModelConfig)
        mock_state_dict = {"test_param": torch.tensor([1.0, 2.0, 3.0])}
        return mock_config, mock_state_dict

    def validate(self, model: SpeculatorModel, device: str | device | int):
        """Test implementation that does nothing."""


class TestSpeculatorConverter:
    """Test class for SpeculatorConverter functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Store the original registry and clear it for this test
        self._original_registry = SpeculatorConverter.registry  # type: ignore[misc]
        SpeculatorConverter.registry = None  # type: ignore[misc]

    def teardown_method(self):
        """Clean up after each test method."""
        # Restore the original registry
        SpeculatorConverter.registry = self._original_registry  # type: ignore[misc]

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test SpeculatorConverter inheritance and type relationships."""
        assert issubclass(SpeculatorConverter, ABC)
        assert issubclass(SpeculatorConverter, Generic)  # type: ignore[arg-type]
        assert issubclass(SpeculatorConverter, RegistryMixin)

        # Test class methods
        assert hasattr(SpeculatorConverter, "resolve_converter")
        assert callable(SpeculatorConverter.resolve_converter)
        assert hasattr(SpeculatorConverter, "is_supported")

        # Test instance methods
        assert hasattr(SpeculatorConverter, "__init__")
        assert callable(SpeculatorConverter)
        assert hasattr(SpeculatorConverter, "save")
        assert hasattr(SpeculatorConverter, "convert_config_state_dict")
        assert hasattr(SpeculatorConverter, "validate")
        assert callable(SpeculatorConverter)

        # Test abstract methods can be called on concrete implementations
        mock_converter = MockSpeculatorConverterImpl(
            model=MagicMock(), config=MagicMock(), verifier=None
        )

        # Test is_supported method signature
        assert (
            MockSpeculatorConverterImpl.is_supported(
                model="test", config={}, verifier=None
            )
            is True
        )

        # Test convert_config_state_dict method signature
        config, state_dict = mock_converter.convert_config_state_dict()
        assert config is not None
        assert isinstance(state_dict, dict)

        # Test validate method signature
        mock_model = MagicMock(spec=SpeculatorModel)
        mock_converter.validate(mock_model, "cpu")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "verifier",
        [None, "mock_verifier"],
    )
    def test_initialization(self, mock_model, mock_config, mock_verifier, verifier):
        """Test SpeculatorConverter initialization."""
        actual_verifier = mock_verifier if verifier == "mock_verifier" else verifier
        instance = MockSpeculatorConverterImpl(mock_model, mock_config, actual_verifier)
        assert isinstance(instance, SpeculatorConverter)
        assert instance.model is mock_model
        assert instance.config is mock_config
        assert instance.verifier is actual_verifier

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("model", "config"),
        [
            (None, "valid_config"),
            ("valid_model", None),
            ("", "valid_config"),
            ("valid_model", ""),
        ],
    )
    def test_invalid_initialization_values(
        self, mock_model, mock_config, model, config
    ):
        """Test SpeculatorConverter with invalid field values."""
        actual_model = mock_model if model == "valid_model" else model
        actual_config = mock_config if config == "valid_config" else config

        with pytest.raises(ValueError) as exc_info:
            MockSpeculatorConverterImpl(actual_model, actual_config, None)

        assert "Model and config paths must be provided" in str(exc_info.value)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self, mock_config):
        """Test SpeculatorConverter initialization without required model."""
        with pytest.raises(TypeError):
            MockSpeculatorConverterImpl(config=mock_config, verifier=None)  # type: ignore[call-arg]

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("output_path", "validate_device", "should_save", "should_validate"),
        [
            (None, None, False, False),
            ("output", None, True, False),
            (None, "cuda", False, True),
            ("output", "cuda", True, True),
        ],
    )
    def test_call_invocation(
        self,
        mock_model,
        mock_config,
        temp_directory,
        output_path,
        validate_device,
        should_save,
        should_validate,
    ):
        """Test SpeculatorConverter call with various parameter combinations."""
        converter = MockSpeculatorConverterImpl(mock_model, mock_config, None)

        if output_path:
            output_path = Path(temp_directory) / output_path

        with patch.object(SpeculatorModel, "from_pretrained") as mock_from_pretrained:
            mock_speculator = MagicMock(spec=SpeculatorModel)
            mock_speculator.save_pretrained = MagicMock()
            mock_from_pretrained.return_value = mock_speculator

            with patch.object(converter, "validate") as mock_validate:
                result = converter(
                    output_path=output_path, validate_device=validate_device
                )

                assert result is mock_speculator
                mock_from_pretrained.assert_called_once()

                if should_save:
                    mock_speculator.save_pretrained.assert_called_once_with(output_path)
                else:
                    mock_speculator.save_pretrained.assert_not_called()

                if should_validate:
                    mock_validate.assert_called_once_with(
                        mock_speculator, validate_device
                    )
                else:
                    mock_validate.assert_not_called()

    @pytest.mark.sanity
    def test_call_with_none_output_and_device(self, mock_model, mock_config):
        """Test calling converter with None values for optional parameters."""
        converter = MockSpeculatorConverterImpl(mock_model, mock_config, None)

        with patch.object(SpeculatorModel, "from_pretrained") as mock_from_pretrained:
            mock_speculator = MagicMock(spec=SpeculatorModel)
            mock_from_pretrained.return_value = mock_speculator

            result = converter(output_path=None, validate_device=None)
            assert result is mock_speculator

    @pytest.mark.smoke
    @pytest.mark.parametrize("path_type", ["Path", "str"])
    def test_save(self, mock_model, mock_config, temp_directory, path_type):
        """Test SpeculatorConverter save method with different path types."""
        converter = MockSpeculatorConverterImpl(mock_model, mock_config, None)
        mock_speculator = MagicMock(spec=SpeculatorModel)
        mock_speculator.save_pretrained = MagicMock()
        output_path: Path | str

        if path_type == "Path":
            output_path = Path(temp_directory) / "output"
        else:
            output_path = str(Path(temp_directory) / "output")

        converter.save(mock_speculator, output_path)
        mock_speculator.save_pretrained.assert_called_once_with(output_path)

    @pytest.mark.smoke
    def test_registration(self):
        SpeculatorConverter.register(["test1", "test1_alt"])(
            MockSpeculatorConverterImpl
        )
        SpeculatorConverter.register("test2")(MockSpeculatorConverterUnsupported)

        assert SpeculatorConverter.registry is not None  # type: ignore[misc]
        assert "test1" in SpeculatorConverter.registry  # type: ignore[misc]
        assert "test1_alt" in SpeculatorConverter.registry  # type: ignore[misc]
        assert SpeculatorConverter.registry["test1"] is MockSpeculatorConverterImpl  # type: ignore[misc]
        assert SpeculatorConverter.registry["test1_alt"] is MockSpeculatorConverterImpl  # type: ignore[misc]
        assert "test2" in SpeculatorConverter.registry  # type: ignore[misc]
        assert (
            SpeculatorConverter.registry["test2"] is MockSpeculatorConverterUnsupported  # type: ignore[misc]
        )

        registered = SpeculatorConverter.registered_objects()

        assert isinstance(registered, tuple)
        assert len(registered) == 3
        assert MockSpeculatorConverterImpl in registered
        assert MockSpeculatorConverterUnsupported in registered

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "algorithm",
        ["test_algo", "test_algo_2", "auto"],
    )
    def test_resolve(self, mock_model, mock_config, mock_verifier, algorithm):
        """Test resolve_converter with specific algorithms and auto detection."""
        # Register test converters
        SpeculatorConverter.register("test_algo")(MockSpeculatorConverterImpl)
        SpeculatorConverter.register("test_algo_2")(MockSpeculatorConverterUnsupported)

        expected_cls = (
            MockSpeculatorConverterImpl
            if algorithm in ("test_algo", "auto")
            else MockSpeculatorConverterUnsupported
        )

        # Test both minimal and full argument scenarios
        test_scenarios = [
            {"model": mock_model, "config": mock_config},
            {
                "model": mock_model,
                "config": mock_config,
                "verifier": mock_verifier,
                "custom_arg": "test_value",
            },
        ]

        for kwargs in test_scenarios:
            if algorithm == "auto":
                with patch.object(
                    MockSpeculatorConverterImpl, "is_supported", return_value=True
                ) as mock_is_supported:
                    converter_cls = SpeculatorConverter.resolve_converter(
                        algorithm=algorithm, **kwargs
                    )
                    assert converter_cls is expected_cls
                    assert mock_is_supported.call_count >= 1
            else:
                converter_cls = SpeculatorConverter.resolve_converter(
                    algorithm=algorithm, **kwargs
                )
                assert converter_cls is expected_cls

    @pytest.mark.sanity
    def test_resolve_failures(self, mock_model, mock_config):
        """Test resolve_converter failure scenarios."""
        # Test with no registry
        with pytest.raises(ValueError) as exc_info:
            SpeculatorConverter.resolve_converter("test", mock_model, mock_config)
        assert "No converters registered" in str(exc_info.value)

        # Register test converters for remaining tests
        SpeculatorConverter.register("test_algo")(MockSpeculatorConverterUnsupported)

        # Test unknown algorithm
        with pytest.raises(ValueError) as exc_info:
            SpeculatorConverter.resolve_converter("unknown", mock_model, mock_config)
        assert "Algorithm 'unknown' is not registered" in str(exc_info.value)
        assert "Available algorithms: test_algo" in str(exc_info.value)

        # Test auto with no supported converters
        with patch.object(
            MockSpeculatorConverterUnsupported, "is_supported", return_value=False
        ):
            with pytest.raises(ValueError) as exc_info:
                SpeculatorConverter.resolve_converter("auto", mock_model, mock_config)
            assert "No supported converter found" in str(exc_info.value)
            assert "Available algorithms: test_algo" in str(exc_info.value)
