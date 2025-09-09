"""
Unit tests for the transformers_utils module in the Speculators library.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from speculators.utils import (
    check_download_model_checkpoint,
    check_download_model_config,
    download_model_checkpoint_from_hub,
    load_model_checkpoint_config_dict,
    load_model_checkpoint_index_weight_files,
    load_model_checkpoint_state_dict,
    load_model_checkpoint_weight_files,
    load_model_config,
)


@pytest.fixture
def mock_pretrained_config():
    """Mock PretrainedConfig for testing."""
    config = MagicMock(spec=PretrainedConfig)
    config.name_or_path = "test/model"
    config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    return config


@pytest.fixture
def mock_pretrained_model():
    """Mock PreTrainedModel for testing."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock(spec=PretrainedConfig)
    model.config.to_dict.return_value = {
        "architectures": ["TestModel"],
        "hidden_size": 768,
        "vocab_size": 50000,
        "model_type": "test_model",
    }
    model.state_dict.return_value = {
        "embedding.weight": torch.randn(50000, 768),
        "layer.0.weight": torch.randn(768, 768),
    }
    return model


@pytest.fixture
def mock_nn_module():
    """Mock nn.Module for testing."""
    module = MagicMock(spec=torch.nn.Module)
    module.state_dict.return_value = {
        "weight": torch.randn(10, 5),
        "bias": torch.randn(10),
    }
    return module


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory with mock checkpoint files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create config.json
        config_data = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        config_file = checkpoint_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Create weight files
        weight_file = checkpoint_path / "pytorch_model.bin"
        torch.save({"weight": torch.randn(10, 5)}, weight_file)

        safetensors_file = checkpoint_path / "model.safetensors"
        safetensors_file.touch()

        yield checkpoint_path


@pytest.fixture
def temp_index_checkpoint_dir():
    """Create a temporary directory with indexed checkpoint files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir)

        # Create config.json
        config_data = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        config_file = checkpoint_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Create index file
        index_data = {
            "weight_map": {
                "embedding.weight": "pytorch_model-00001-of-00002.bin",
                "layer.0.weight": "pytorch_model-00002-of-00002.bin",
            }
        }
        index_file = checkpoint_path / "pytorch_model.bin.index.json"
        index_file.write_text(json.dumps(index_data))

        # Create weight files referenced in index
        weight_file_1 = checkpoint_path / "pytorch_model-00001-of-00002.bin"
        torch.save({"embedding.weight": torch.randn(50000, 768)}, weight_file_1)

        weight_file_2 = checkpoint_path / "pytorch_model-00002-of-00002.bin"
        torch.save({"layer.0.weight": torch.randn(768, 768)}, weight_file_2)

        yield checkpoint_path


class TestDownloadModelCheckpointFromHub:
    """Test suite for download_model_checkpoint_from_hub function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "model_id",
            "cache_dir",
            "force_download",
            "local_files_only",
            "token",
            "revision",
            "kwargs",
            "expected_call",
        ),
        [
            (
                "test/model",
                None,
                False,
                False,
                None,
                None,
                {},
                {
                    "repo_id": "test/model",
                    "cache_dir": None,
                    "force_download": False,
                    "local_files_only": False,
                    "token": None,
                    "revision": None,
                    "allow_patterns": [
                        "*.json",
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                    ],
                },
            ),
            (
                "test/model",
                "/cache",
                True,
                True,
                "test_token",
                "v1.0",
                {"custom_param": "custom_value"},
                {
                    "repo_id": "test/model",
                    "cache_dir": "/cache",
                    "force_download": True,
                    "local_files_only": True,
                    "token": "test_token",
                    "revision": "v1.0",
                    "custom_param": "custom_value",
                    "allow_patterns": [
                        "*.json",
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                    ],
                },
            ),
        ],
        ids=["default_parameters", "custom_parameters"],
    )
    @patch("speculators.utils.transformers_utils.snapshot_download")
    def test_invocation(
        self,
        mock_snapshot_download,
        model_id,
        cache_dir,
        force_download,
        local_files_only,
        token,
        revision,
        kwargs,
        expected_call,
    ):
        """Test successful download of model checkpoint from HuggingFace Hub."""
        mock_snapshot_download.return_value = "/path/to/downloaded/model"

        result = download_model_checkpoint_from_hub(
            model_id=model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

        assert result == Path("/path/to/downloaded/model")
        mock_snapshot_download.assert_called_once_with(**expected_call)

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.snapshot_download")
    def test_invalid_invocation(self, mock_snapshot_download):
        """Test handling of download failure."""
        mock_snapshot_download.side_effect = Exception("Download failed")

        with pytest.raises(FileNotFoundError) as exc_info:
            download_model_checkpoint_from_hub("test/model")

        assert "Checkpoint not found: test/model" in str(exc_info.value)


class TestCheckDownloadModelCheckpoint:
    """Test suite for check_download_model_checkpoint function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model_input", "expected_type"),
        [
            ("pretrained_model", "PreTrainedModel"),
            ("nn_module", "nn.Module"),
        ],
        ids=["pretrained_model", "nn_module"],
    )
    def test_invocation_with_model_instances(
        self,
        model_input,
        expected_type,
        mock_pretrained_model,
        mock_nn_module,
    ):
        """Test with model instances (PreTrainedModel and nn.Module)."""
        if model_input == "pretrained_model":
            result = check_download_model_checkpoint(mock_pretrained_model)
            assert result is mock_pretrained_model
        else:
            result = check_download_model_checkpoint(mock_nn_module)
            assert result is mock_nn_module

    @pytest.mark.smoke
    def test_invocation_with_local_path(self, temp_checkpoint_dir):
        """Test with existing local checkpoint directory."""
        result = check_download_model_checkpoint(temp_checkpoint_dir)
        assert result == temp_checkpoint_dir.resolve()

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("model_input", "expected_error", "error_message"),
        [
            (123, TypeError, "Expected model to be a string or Path"),
            ("temp_file_path", ValueError, "Expected a directory for checkpoint"),
        ],
        ids=["invalid_type", "not_directory"],
    )
    def test_invalid_invocation(self, model_input, expected_error, error_message):
        """Test with invalid input types and paths."""
        if model_input == "temp_file_path":
            with tempfile.NamedTemporaryFile() as temp_file:
                with pytest.raises(expected_error) as exc_info:
                    check_download_model_checkpoint(temp_file.name)
                assert error_message in str(exc_info.value)
        else:
            with pytest.raises(expected_error) as exc_info:
                check_download_model_checkpoint(model_input)
            assert error_message in str(exc_info.value)

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.download_model_checkpoint_from_hub")
    def test_download_from_hub(self, mock_download):
        """Test download from hub when local path doesn't exist."""
        mock_download.return_value = Path("/downloaded/model")

        result = check_download_model_checkpoint(
            "nonexistent/path",
            cache_dir="/cache",
            force_download=True,
            token="test_token",
        )

        assert result == Path("/downloaded/model")
        mock_download.assert_called_once_with(
            model_id="nonexistent/path",
            cache_dir="/cache",
            force_download=True,
            local_files_only=False,
            token="test_token",
            revision=None,
        )


class TestCheckDownloadModelConfig:
    """Test suite for check_download_model_config function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("config_input", "expected_result"),
        [
            ("pretrained_config", "pretrained_config"),
            ("pretrained_model", "config_from_model"),
            ("dict_config", "dict_config"),
        ],
        ids=["pretrained_config", "pretrained_model", "dict_config"],
    )
    def test_invocation_with_config_instances(
        self,
        config_input,
        expected_result,
        mock_pretrained_config,
        mock_pretrained_model,
    ):
        """Test with various config instance types."""
        if config_input == "pretrained_config":
            result = check_download_model_config(mock_pretrained_config)
            assert result is mock_pretrained_config
        elif config_input == "pretrained_model":
            result = check_download_model_config(mock_pretrained_model)
            assert result is mock_pretrained_model.config
        else:
            config_dict = {"model_type": "test", "hidden_size": 768}
            result = check_download_model_config(config_dict)
            assert result is config_dict

    @pytest.mark.smoke
    def test_invocation_with_local_file(self, temp_checkpoint_dir):
        """Test with existing local config file."""
        config_path = temp_checkpoint_dir / "config.json"
        result = check_download_model_config(config_path)
        assert result == config_path.resolve()

    @pytest.mark.smoke
    def test_invocation_with_local_dir(self, temp_checkpoint_dir):
        """Test with existing local checkpoint directory."""
        result = check_download_model_config(temp_checkpoint_dir)
        assert result == (temp_checkpoint_dir / "config.json").resolve()

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("config_input", "expected_error", "error_message"),
        [
            (
                123,
                TypeError,
                "Expected config to be a string, Path, or PreTrainedModel",
            ),
            (
                "missing_config",
                OSError,
                "Can't load the configuration",
            ),
        ],
        ids=["invalid_type", "missing_config"],
    )
    def test_invalid_invocation(self, config_input, expected_error, error_message):
        """Test with invalid input types and missing configs."""
        if config_input == "missing_config":
            with tempfile.TemporaryDirectory() as temp_dir:
                missing_config_path = Path(temp_dir) / "missing_dir"
                with pytest.raises(expected_error) as exc_info:
                    check_download_model_config(missing_config_path)
                assert error_message in str(exc_info.value)
        else:
            with pytest.raises(expected_error) as exc_info:
                check_download_model_config(config_input)
            assert error_message in str(exc_info.value)

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.AutoConfig")
    def test_download_from_hub(self, mock_auto_config):
        """Test download from hub when local path doesn't exist."""
        mock_config = MagicMock(spec=PretrainedConfig)
        mock_auto_config.from_pretrained.return_value = mock_config

        result = check_download_model_config(
            "nonexistent/path",
            cache_dir="/cache",
            force_download=True,
            token="test_token",
        )

        assert result is mock_config
        mock_auto_config.from_pretrained.assert_called_once_with(
            "nonexistent/path",
            cache_dir="/cache",
            force_download=True,
            local_files_only=False,
            token="test_token",
            revision=None,
        )


class TestLoadModelConfig:
    """Test suite for load_model_config function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model_input", "expected_result"),
        [
            ("pretrained_config", "pretrained_config"),
            ("pretrained_model", "config_from_model"),
        ],
        ids=["pretrained_config", "pretrained_model"],
    )
    def test_invocation_with_instances(
        self,
        model_input,
        expected_result,
        mock_pretrained_config,
        mock_pretrained_model,
    ):
        """Test with PretrainedConfig and PreTrainedModel instances."""
        if model_input == "pretrained_config":
            result = load_model_config(mock_pretrained_config)
            assert result is mock_pretrained_config
        else:
            result = load_model_config(mock_pretrained_model)
            assert result is mock_pretrained_model.config

    @pytest.mark.smoke
    @patch("speculators.utils.transformers_utils.AutoConfig")
    def test_invocation_from_path(self, mock_auto_config):
        """Test loading config from path."""
        mock_config = MagicMock(spec=PretrainedConfig)
        mock_auto_config.from_pretrained.return_value = mock_config

        result = load_model_config(
            "test/model",
            cache_dir="/cache",
            force_download=True,
            token="test_token",
        )

        assert result is mock_config
        mock_auto_config.from_pretrained.assert_called_once_with(
            "test/model",
            cache_dir="/cache",
            force_download=True,
            local_files_only=False,
            token="test_token",
            revision=None,
        )

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.AutoConfig")
    def test_invalid_invocation(self, mock_auto_config):
        """Test with invalid input types and missing configs."""
        with pytest.raises(TypeError) as type_exc:
            load_model_config(123)  # type: ignore[arg-type]
        assert "Expected model to be a string or Path" in str(type_exc.value)

        mock_auto_config.from_pretrained.side_effect = ValueError("Config not found")
        with pytest.raises(FileNotFoundError) as file_exc:
            load_model_config("test/model")
        assert "Config not found for model: test/model" in str(file_exc.value)


class TestLoadModelCheckpointConfigDict:
    """Test suite for load_model_checkpoint_config_dict function."""

    @pytest.fixture(
        params=[
            {"model_type": "test", "hidden_size": 768},
            {"architectures": ["TestModel"], "vocab_size": 50000},
        ],
        ids=["basic_config", "extended_config"],
    )
    def config_dict_instances(self, request):
        """Fixture providing test config dictionaries."""
        return request.param

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("config_input", "expected_result"),
        [
            ("dict_config", "dict_config"),
            ("pretrained_model", "config_from_model"),
            ("pretrained_config", "config_from_config"),
        ],
        ids=["dict_config", "pretrained_model", "pretrained_config"],
    )
    def test_invocation_with_instances(
        self,
        config_input,
        expected_result,
        config_dict_instances,
        mock_pretrained_model,
        mock_pretrained_config,
    ):
        """Test with various config instance types."""
        if config_input == "dict_config":
            result = load_model_checkpoint_config_dict(config_dict_instances)
            assert result is config_dict_instances
        elif config_input == "pretrained_model":
            result = load_model_checkpoint_config_dict(mock_pretrained_model)
            assert result == mock_pretrained_model.config.to_dict.return_value
            mock_pretrained_model.config.to_dict.assert_called_once()
        else:
            result = load_model_checkpoint_config_dict(mock_pretrained_config)
            assert result == mock_pretrained_config.to_dict.return_value
            mock_pretrained_config.to_dict.assert_called_once()

    @pytest.mark.smoke
    def test_invocation_with_file(self, temp_checkpoint_dir):
        """Test with config file path."""
        config_path = temp_checkpoint_dir / "config.json"
        result = load_model_checkpoint_config_dict(config_path)

        expected_config = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        assert result == expected_config

    @pytest.mark.smoke
    def test_invocation_with_dir(self, temp_checkpoint_dir):
        """Test with checkpoint directory."""
        result = load_model_checkpoint_config_dict(temp_checkpoint_dir)

        expected_config = {
            "architectures": ["TestModel"],
            "hidden_size": 768,
            "vocab_size": 50000,
            "model_type": "test_model",
        }
        assert result == expected_config

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("config_input", "expected_error", "error_message"),
        [
            (
                123,
                TypeError,
                "Expected config to be a string, Path, PreTrainedModel, "
                "or PretrainedConfig",
            ),
            (
                "missing_config",
                FileNotFoundError,
                "No config.json found",
            ),
        ],
        ids=["invalid_type", "missing_config"],
    )
    def test_invalid_invocation(self, config_input, expected_error, error_message):
        """Test with invalid input types and missing configs."""
        if config_input == "missing_config":
            with tempfile.TemporaryDirectory() as temp_dir:
                missing_config_path = Path(temp_dir) / "config.json"
                with pytest.raises(expected_error) as exc_info:
                    load_model_checkpoint_config_dict(missing_config_path)
                assert error_message in str(exc_info.value)
        else:
            with pytest.raises(expected_error) as exc_info:
                load_model_checkpoint_config_dict(config_input)
            assert error_message in str(exc_info.value)


class TestLoadModelCheckpointIndexWeightFiles:
    """Test suite for load_model_checkpoint_index_weight_files function."""

    @pytest.mark.smoke
    def test_invocation_with_directory(self, temp_index_checkpoint_dir):
        """Test with directory containing index files."""
        result = load_model_checkpoint_index_weight_files(temp_index_checkpoint_dir)

        assert len(result) == 2
        assert all(isinstance(file, Path) for file in result)
        assert all(file.exists() for file in result)

    @pytest.mark.smoke
    def test_invocation_no_index_files(self):
        """Test with directory containing no index files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_model_checkpoint_index_weight_files(temp_dir)
            assert result == []

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("path_input", "expected_error", "error_message"),
        [
            (123, TypeError, "Expected path to be a string or Path"),
            (
                "/nonexistent/path",
                FileNotFoundError,
                "Model checkpoint path does not exist",
            ),
        ],
        ids=["invalid_type", "nonexistent_path"],
    )
    def test_invalid_invocation(self, path_input, expected_error, error_message):
        """Test with invalid input types and paths."""
        with pytest.raises(expected_error) as exc_info:
            load_model_checkpoint_index_weight_files(path_input)
        assert error_message in str(exc_info.value)

    @pytest.mark.sanity
    def test_invalid_index_content(self):
        """Test with invalid index file content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create invalid index file
            invalid_index_data = {"metadata": {"total_size": 1000}}
            index_file = checkpoint_path / "pytorch_model.bin.index.json"
            index_file.write_text(json.dumps(invalid_index_data))

            with pytest.raises(ValueError) as exc_info:
                load_model_checkpoint_index_weight_files(checkpoint_path)
            assert "does not contain a weight_map" in str(exc_info.value)

    @pytest.mark.sanity
    def test_missing_weight_file(self):
        """Test with index file referencing non-existent weight file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create index file with non-existent weight file
            index_data = {
                "weight_map": {
                    "embedding.weight": "missing_file.bin",
                }
            }
            index_file = checkpoint_path / "pytorch_model.bin.index.json"
            index_file.write_text(json.dumps(index_data))

            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_checkpoint_index_weight_files(checkpoint_path)
            assert "Weight file for" in str(exc_info.value)
            assert "does not exist" in str(exc_info.value)


class TestLoadModelCheckpointWeightFiles:
    """Test suite for load_model_checkpoint_weight_files function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("file_type", "expected_extension"),
        [
            ("bin", ".bin"),
            ("safetensors", ".safetensors"),
        ],
        ids=["bin_file", "safetensors_file"],
    )
    def test_invocation_with_single_file(
        self,
        file_type,
        expected_extension,
        temp_checkpoint_dir,
    ):
        """Test with single weight files."""
        if file_type == "bin":
            target_file = temp_checkpoint_dir / "pytorch_model.bin"
        else:  # safetensors
            target_file = temp_checkpoint_dir / "model.safetensors"

        result = load_model_checkpoint_weight_files(target_file)
        assert len(result) == 1
        assert result[0] == target_file

    @pytest.mark.smoke
    def test_invocation_with_directory_bin_only(self):
        """Test with directory containing only .bin files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create only bin files (no safetensors)
            bin_file = checkpoint_path / "pytorch_model.bin"
            torch.save({"weight": torch.randn(10, 5)}, bin_file)

            result = load_model_checkpoint_weight_files(checkpoint_path)

            assert len(result) == 1
            assert result[0].suffix == ".bin"

    @pytest.mark.smoke
    def test_invocation_with_directory_safetensors_preferred(self, temp_checkpoint_dir):
        """Test with directory containing both file types (prefers safetensors)."""
        result = load_model_checkpoint_weight_files(temp_checkpoint_dir)

        # Should return .safetensors files first as they are preferred
        assert len(result) == 1
        assert result[0].suffix == ".safetensors"

    @pytest.mark.smoke
    def test_invocation_with_directory_safetensors_only(self):
        """Test with directory containing only .safetensors files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create only safetensors files
            safetensors_file = checkpoint_path / "model.safetensors"
            safetensors_file.touch()

            result = load_model_checkpoint_weight_files(checkpoint_path)

            assert len(result) == 1
            assert result[0].suffix == ".safetensors"

    @pytest.mark.smoke
    def test_invocation_with_index_files(self, temp_index_checkpoint_dir):
        """Test with directory containing index files."""
        result = load_model_checkpoint_weight_files(temp_index_checkpoint_dir)

        assert len(result) == 2
        assert all(file.suffix == ".bin" for file in result)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("path_input", "expected_error", "error_message"),
        [
            (123, TypeError, "Expected path to be a string or Path"),
            (
                "/nonexistent/path",
                FileNotFoundError,
                "Model checkpoint path does not exist",
            ),
        ],
        ids=["invalid_type", "nonexistent_path"],
    )
    def test_invalid_invocation(self, path_input, expected_error, error_message):
        """Test with invalid input types and paths."""
        with pytest.raises(expected_error) as exc_info:
            load_model_checkpoint_weight_files(path_input)
        assert error_message in str(exc_info.value)

    @pytest.mark.sanity
    def test_no_valid_weight_files(self):
        """Test with directory containing no valid weight files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create a non-weight file
            other_file = checkpoint_path / "README.md"
            other_file.write_text("This is a readme")

            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_checkpoint_weight_files(checkpoint_path)
            assert "No valid weight files found" in str(exc_info.value)


class TestLoadModelCheckpointStateDict:
    """Test suite for load_model_checkpoint_state_dict function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("model_input", "expected_type"),
        [
            ("pretrained_model", "PreTrainedModel"),
            ("nn_module", "nn.Module"),
        ],
        ids=["pretrained_model", "nn_module"],
    )
    def test_invocation_with_model_instances(
        self,
        model_input,
        expected_type,
        mock_pretrained_model,
        mock_nn_module,
    ):
        """Test with model instances (PreTrainedModel and nn.Module)."""
        if model_input == "pretrained_model":
            result = load_model_checkpoint_state_dict(mock_pretrained_model)
            assert result == mock_pretrained_model.state_dict.return_value
            mock_pretrained_model.state_dict.assert_called_once()
        else:
            result = load_model_checkpoint_state_dict(mock_nn_module)
            assert result == mock_nn_module.state_dict.return_value
            mock_nn_module.state_dict.assert_called_once()

    @pytest.mark.smoke
    @patch("speculators.utils.transformers_utils.torch.load")
    def test_invocation_with_bin_file(self, mock_torch_load, temp_checkpoint_dir):
        """Test with .bin file."""
        bin_file = temp_checkpoint_dir / "pytorch_model.bin"
        expected_state_dict = {
            "embedding.weight": torch.randn(50000, 768),
            "layer.0.weight": torch.randn(768, 768),
        }
        mock_torch_load.return_value = expected_state_dict

        result = load_model_checkpoint_state_dict(bin_file)

        assert len(result) == 2
        assert "embedding.weight" in result
        assert "layer.0.weight" in result
        mock_torch_load.assert_called_once_with(bin_file, map_location="cpu")

    @pytest.mark.smoke
    @patch("speculators.utils.transformers_utils.safe_open")
    def test_invocation_with_safetensors_file(
        self,
        mock_safe_open,
        temp_checkpoint_dir,
    ):
        """Test with .safetensors file."""
        safetensors_file = temp_checkpoint_dir / "model.safetensors"

        # Mock the safe_open context manager
        mock_safetensors_file = MagicMock()
        mock_safetensors_file.keys.return_value = ["embedding.weight", "layer.0.weight"]
        mock_safetensors_file.get_tensor.side_effect = lambda key: torch.randn(768, 768)
        mock_safe_open.return_value.__enter__.return_value = mock_safetensors_file

        result = load_model_checkpoint_state_dict(safetensors_file)

        assert len(result) == 2
        assert "embedding.weight" in result
        assert "layer.0.weight" in result
        mock_safe_open.assert_called_once_with(
            safetensors_file, framework="pt", device="cpu"
        )

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.torch.load")
    def test_invocation_with_index_files(
        self,
        mock_torch_load,
        temp_index_checkpoint_dir,
    ):
        """Test with directory containing index files."""
        mock_torch_load.side_effect = [
            {"embedding.weight": torch.randn(50000, 768)},
            {"layer.0.weight": torch.randn(768, 768)},
        ]

        result = load_model_checkpoint_state_dict(temp_index_checkpoint_dir)

        assert len(result) == 2
        assert "embedding.weight" in result
        assert "layer.0.weight" in result
        assert mock_torch_load.call_count == 2

    @pytest.mark.sanity
    def test_invalid_invocation_unsupported_file_type(self):
        """Test with unsupported file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create an unsupported file type
            unsupported_file = checkpoint_path / "model.txt"
            unsupported_file.write_text("This is not a weight file")

            with pytest.raises(FileNotFoundError) as exc_info:
                load_model_checkpoint_state_dict(unsupported_file)

            assert "No valid weight files found" in str(exc_info.value)

    @pytest.mark.sanity
    @patch("speculators.utils.transformers_utils.torch.load")
    @patch("speculators.utils.transformers_utils.safe_open")
    def test_invocation_mixed_file_types(
        self,
        mock_safe_open,
        mock_torch_load,
    ):
        """Test with directory containing both .bin and .safetensors files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir)

            # Create both file types
            bin_file = checkpoint_path / "pytorch_model.bin"
            bin_file.touch()
            safetensors_file = checkpoint_path / "model.safetensors"
            safetensors_file.touch()

            # Mock torch.load
            mock_torch_load.return_value = {"bin_weight": torch.randn(10, 10)}

            # Mock safe_open
            mock_safetensors_file = MagicMock()
            mock_safetensors_file.keys.return_value = ["safetensors_weight"]
            mock_safetensors_file.get_tensor.return_value = torch.randn(20, 20)
            mock_safe_open.return_value.__enter__.return_value = mock_safetensors_file

            result = load_model_checkpoint_state_dict(checkpoint_path)

            # Should prefer safetensors files
            assert len(result) == 1
            assert "safetensors_weight" in result
            mock_safe_open.assert_called_once()
            mock_torch_load.assert_not_called()
