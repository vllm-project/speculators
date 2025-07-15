"""
Unit tests for the base converter module in the Speculators library.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor, device, nn
from transformers import PretrainedConfig, PreTrainedModel

from speculators import SpeculatorModel, SpeculatorModelConfig
from speculators.convert import SpeculatorConverter

# ===== Test Fixtures =====


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
def mock_speculator_model():
    """Mock speculator model for testing."""
    model = MagicMock(spec=SpeculatorModel)
    model.save_pretrained = MagicMock()
    return model


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# ===== Test Converter Implementation =====


class TestSpeculatorConverter(SpeculatorConverter):
    """Test implementation of SpeculatorConverter for unit testing."""

    @classmethod
    def is_supported(
        cls,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
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

    def validate(self, model: SpeculatorModel, device: Union[str, device, int]):
        """Test implementation that does nothing."""


class TestSpeculatorConverterUnsupported(SpeculatorConverter):
    """Test implementation that is never supported."""

    @classmethod
    def is_supported(
        cls,
        model: Union[Path, PreTrainedModel, nn.Module],
        config: Union[Path, PretrainedConfig, dict],
        verifier: Optional[Union[str, os.PathLike, PreTrainedModel]] = None,
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

    def validate(self, model: SpeculatorModel, device: Union[str, device, int]):
        """Test implementation that does nothing."""


# ===== SpeculatorConverter Base Class Tests =====


class TestSpeculatorConverterBase:
    """Test class for SpeculatorConverter base functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear the registry before each test
        SpeculatorConverter.registry = None

    @pytest.mark.smoke
    def test_class_attributes(self):
        """Test that SpeculatorConverter has the expected class attributes."""
        expected_attributes = [
            "resolve_converter",
            "is_supported",
            "__init__",
            "save",
            "convert_config_state_dict",
            "validate",
        ]

        for attr in expected_attributes:
            assert hasattr(SpeculatorConverter, attr), f"Missing attribute: {attr}"

        assert callable(SpeculatorConverter)

    @pytest.mark.smoke
    def test_initialization(self, mock_model, mock_config, mock_verifier):
        """Test successful initialization of SpeculatorConverter."""
        converter = TestSpeculatorConverter(mock_model, mock_config, mock_verifier)

        assert converter.model is mock_model
        assert converter.config is mock_config
        assert converter.verifier is mock_verifier

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model", "config"),
        [
            (None, "valid_config"),
            ("valid_model", None),
            ("", "valid_config"),
            ("valid_model", ""),
        ],
    )
    def test_initialization_invalid(self, mock_model, mock_config, model, config):
        """Test initialization fails with invalid inputs."""
        # Use actual mock objects for valid placeholders
        actual_model = mock_model if model == "valid_model" else model
        actual_config = mock_config if config == "valid_config" else config

        with pytest.raises(ValueError) as exc_info:
            TestSpeculatorConverter(actual_model, actual_config, None)

        assert "Model and config paths must be provided" in str(exc_info.value)

    @pytest.mark.smoke
    def test_resolve_converter_no_registry(self, mock_model, mock_config):
        """Test resolve_converter fails when no registry exists."""
        with pytest.raises(ValueError) as exc_info:
            SpeculatorConverter.resolve_converter("test", mock_model, mock_config)

        assert "No converters registered" in str(exc_info.value)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("algorithm", "expected_converter"),
        [
            ("test_algo", "TestSpeculatorConverter"),
            ("test_algo_2", "TestSpeculatorConverter"),
        ],
    )
    def test_resolve_converter_algorithm(
        self, mock_model, mock_config, algorithm, expected_converter
    ):
        """Test resolve_converter with specific algorithm names."""
        # Register test converters
        SpeculatorConverter.register("test_algo")(TestSpeculatorConverter)
        SpeculatorConverter.register("TEST_ALGO_2")(TestSpeculatorConverter)

        converter_cls = SpeculatorConverter.resolve_converter(
            algorithm, mock_model, mock_config
        )

        assert converter_cls.__name__ == expected_converter

    @pytest.mark.sanity
    def test_resolve_converter_auto_success(self, mock_model, mock_config):
        """Test resolve_converter with auto detection finds supported converter."""
        SpeculatorConverter.register("test_algo")(TestSpeculatorConverter)

        converter_cls = SpeculatorConverter.resolve_converter(
            "auto", mock_model, mock_config
        )

        assert converter_cls is TestSpeculatorConverter

    @pytest.mark.smoke
    def test_resolve_converter_invalid_algorithm(self, mock_model, mock_config):
        """Test resolve_converter fails with unregistered algorithm."""
        SpeculatorConverter.register("test_algo")(TestSpeculatorConverter)

        with pytest.raises(ValueError) as exc_info:
            SpeculatorConverter.resolve_converter("unknown", mock_model, mock_config)

        assert "Algorithm 'unknown' is not registered" in str(exc_info.value)
        assert "Available algorithms: test_algo" in str(exc_info.value)

    @pytest.mark.sanity
    def test_resolve_converter_auto_failure(self, mock_model, mock_config):
        """Test auto detection fails when no supported converter."""
        SpeculatorConverter.register("test_algo")(TestSpeculatorConverterUnsupported)

        with pytest.raises(ValueError) as exc_info:
            SpeculatorConverter.resolve_converter("auto", mock_model, mock_config)

        assert "No supported converter found" in str(exc_info.value)
        assert "Available algorithms: test_algo" in str(exc_info.value)

    @pytest.mark.sanity
    def test_resolve_converter_with_verifier_and_kwargs(
        self, mock_model, mock_config, mock_verifier
    ):
        """Test resolve_converter passes verifier and kwargs to is_supported."""
        SpeculatorConverter.register("test_algo")(TestSpeculatorConverter)

        with patch.object(
            TestSpeculatorConverter, "is_supported", return_value=True
        ) as mock_is_supported:
            converter_cls = SpeculatorConverter.resolve_converter(
                "auto", mock_model, mock_config, mock_verifier, custom_arg="test_value"
            )

            assert converter_cls is TestSpeculatorConverter
            mock_is_supported.assert_called_once_with(
                mock_model, mock_config, mock_verifier, custom_arg="test_value"
            )

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
    def test_converter_call_combinations(
        self,
        mock_model,
        mock_config,
        temp_directory,
        output_path,
        validate_device,
        should_save,
        should_validate,
    ):
        """Test converter call with various parameter combinations."""
        converter = TestSpeculatorConverter(mock_model, mock_config, None)

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
    def test_converter_call_complete_workflow(
        self, mock_model, mock_config, mock_verifier, temp_directory
    ):
        """Test complete converter workflow with all options."""
        converter = TestSpeculatorConverter(mock_model, mock_config, mock_verifier)
        output_path = Path(temp_directory) / "output"

        with patch.object(SpeculatorModel, "from_pretrained") as mock_from_pretrained:
            mock_speculator = MagicMock(spec=SpeculatorModel)
            mock_speculator.save_pretrained = MagicMock()
            mock_from_pretrained.return_value = mock_speculator

            with patch.object(converter, "validate") as mock_validate:
                result = converter(output_path=output_path, validate_device="cuda")

                assert result is mock_speculator
                mock_from_pretrained.assert_called_once_with(
                    pretrained_model_name_or_path=None,
                    config=mock_from_pretrained.call_args[1]["config"],
                    state_dict=mock_from_pretrained.call_args[1]["state_dict"],
                    verifier=mock_verifier,
                    verifier_attachment_mode="full",
                )
                mock_speculator.save_pretrained.assert_called_once_with(output_path)
                mock_validate.assert_called_once_with(mock_speculator, "cuda")

    @pytest.mark.smoke
    @pytest.mark.parametrize("path_type", ["Path", "str"])
    def test_save_method(self, mock_model, mock_config, temp_directory, path_type):
        """Test save method with different path types."""
        converter = TestSpeculatorConverter(mock_model, mock_config, None)
        mock_speculator = MagicMock(spec=SpeculatorModel)
        mock_speculator.save_pretrained = MagicMock()

        if path_type == "Path":
            output_path = Path(temp_directory) / "output"
        else:
            output_path = str(Path(temp_directory) / "output")  # type: ignore[assignment]

        converter.save(mock_speculator, output_path)

        mock_speculator.save_pretrained.assert_called_once_with(output_path)

    @pytest.mark.regression
    def test_registry_multiple_names(self):
        """Test registering converter with multiple names."""
        SpeculatorConverter.register(["test1", "test2"])(TestSpeculatorConverter)

        assert SpeculatorConverter.registry is not None
        assert "test1" in SpeculatorConverter.registry
        assert "test2" in SpeculatorConverter.registry
        assert SpeculatorConverter.registry["test1"] is TestSpeculatorConverter
        assert SpeculatorConverter.registry["test2"] is TestSpeculatorConverter

    @pytest.mark.regression
    def test_registered_classes_method(self):
        """Test registered_classes method returns correct converters."""
        SpeculatorConverter.register("test1")(TestSpeculatorConverter)
        SpeculatorConverter.register("test2")(TestSpeculatorConverterUnsupported)

        registered = SpeculatorConverter.registered_classes()

        assert isinstance(registered, tuple)
        assert len(registered) == 2
        assert TestSpeculatorConverter in registered
        assert TestSpeculatorConverterUnsupported in registered

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("error_stage", "exception_type", "error_message"),
        [
            ("convert", ValueError, "Test error in conversion"),
            ("validate", RuntimeError, "Test error in validation"),
            ("save", OSError, "Test save error"),
        ],
    )
    def test_error_propagation(
        self, mock_model, mock_config, error_stage, exception_type, error_message
    ):
        """Test that errors in different stages propagate through __call__."""

        class ErrorConverter(SpeculatorConverter):
            @classmethod
            def is_supported(cls, model, config, verifier=None, **kwargs):
                return True

            def convert_config_state_dict(self):
                if error_stage == "convert":
                    raise exception_type(error_message)
                mock_config = MagicMock(spec=SpeculatorModelConfig)
                mock_state_dict = {"test_param": torch.tensor([1.0])}
                return mock_config, mock_state_dict

            def validate(self, model, device):
                if error_stage == "validate":
                    raise exception_type(error_message)

        converter = ErrorConverter(mock_model, mock_config, None)

        with patch.object(SpeculatorModel, "from_pretrained") as mock_from_pretrained:
            mock_speculator = MagicMock(spec=SpeculatorModel)

            if error_stage == "save":
                mock_speculator.save_pretrained = MagicMock(
                    side_effect=exception_type(error_message)
                )
            else:
                mock_speculator.save_pretrained = MagicMock()

            mock_from_pretrained.return_value = mock_speculator

            # Test error propagation for different stages
            if error_stage == "validate":
                with pytest.raises(exception_type) as exc_info:
                    converter(validate_device="cuda")
            elif error_stage == "save":
                with pytest.raises(exception_type) as exc_info:
                    converter(output_path="/tmp/test_output")
            else:
                with pytest.raises(exception_type) as exc_info:
                    converter()

            assert error_message in str(exc_info.value)
