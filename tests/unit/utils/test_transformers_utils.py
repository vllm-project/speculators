"""
Unit tests for the transformers_utils module in the Speculators library.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockType
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
from tests.unit.mock import MockPretrainedTransformersFactory, PretrainedBundle


class TestDownloadModelCheckpointFromHub:
    """Test suite for download_model_checkpoint_from_hub function."""

    @pytest.fixture
    def valid_instances(
        self,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[PretrainedBundle, MockType]:
        # Register default mock model
        factory, patched = mock_pretrained_factory

        return factory.register(), patched[
            "speculators.utils.transformers_utils.snapshot_download"
        ]

    @pytest.mark.smoke
    def test_invocation(self, valid_instances: tuple[PretrainedBundle, MockType]):
        """Test successful download of model checkpoint from HuggingFace Hub."""
        bundle, patched = valid_instances
        path = download_model_checkpoint_from_hub(
            model_id=bundle.name_or_path,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
            kwarg="kwarg_value",
        )
        assert path == bundle.local_dir.resolve()
        patched.assert_called_once_with(
            repo_id=bundle.name_or_path,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
            kwarg="kwarg_value",
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
        )

    @pytest.mark.sanity
    def test_invalid_invocation(
        self, valid_instances: tuple[PretrainedBundle, MockType]
    ):
        """Test handling of download failure."""
        _, patched = valid_instances
        patched.side_effect = Exception("Download failed")

        with pytest.raises(FileNotFoundError) as exc_info:
            download_model_checkpoint_from_hub("test/model")

        assert "Failed to download checkpoint for" in str(exc_info.value)
        patched.assert_called_once_with(
            repo_id="test/model",
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
            revision=None,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
        )


class TestCheckDownloadModelCheckpoint:
    """Test suite for check_download_model_checkpoint function."""

    @pytest.fixture(
        params=["pretrained", "module", "local_path", "hf_id"],
        ids=["pretrained", "module", "local_path", "hf_id"],
    )
    def valid_instances(
        self,
        request,
        tmp_path: Path,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[
        PreTrainedModel | torch.nn.Module | str | Path,
        PretrainedBundle,
        MockType,
    ]:
        """Fixture to provide valid instances for different input types."""
        factory, patched = mock_pretrained_factory
        bundle = factory.register()

        model_type = cast("str", request.param)
        model: PreTrainedModel | torch.nn.Module | str | Path
        if model_type == "pretrained":
            model = bundle.model
        elif model_type == "module":
            model = torch.nn.Module()
        elif model_type == "local_path":
            model = bundle.local_dir
        elif model_type == "hf_id":
            model = bundle.name_or_path
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return (
            model,
            bundle,
            patched["speculators.utils.transformers_utils.snapshot_download"],
        )

    @pytest.mark.smoke
    def test_invocation(
        self,
        valid_instances: tuple[
            PreTrainedModel | torch.nn.Module | str | Path,
            PretrainedBundle,
            MockType,
        ],
    ):
        """Test with model instances (PreTrainedModel and nn.Module)."""
        model, bundle, patched = valid_instances

        result = check_download_model_checkpoint(
            model,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
        )

        if isinstance(model, (PreTrainedModel, torch.nn.Module)):
            assert result is model
            patched.assert_not_called()
        elif isinstance(model, Path):
            assert result == model.resolve()
            patched.assert_not_called()
        elif isinstance(model, str):
            assert result == bundle.local_dir.resolve()
            patched.assert_called_once_with(
                repo_id=bundle.name_or_path,
                cache_dir="/tmp/cache",
                force_download=True,
                local_files_only=True,
                token="asdf",
                revision="main",
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
            )
        else:
            raise ValueError("Invalid model type in test.")

    @pytest.mark.sanity
    def test_invalid_invocation(self):
        """Test with invalid input types and paths."""
        with pytest.raises(TypeError) as exc_info:
            check_download_model_checkpoint(123)  # type: ignore[arg-type]
        assert "Expected model to be a string or Path" in str(exc_info.value)


class TestCheckDownloadModelConfig:
    """Test suite for check_download_model_config function."""

    @pytest.fixture(
        params=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
        ids=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[
        PretrainedConfig | PreTrainedModel | dict | str | Path,
        PretrainedBundle,
        MockType,
    ]:
        """Fixture to provide valid instances for different input types."""
        factory, patched = mock_pretrained_factory
        bundle = factory.register()

        config_type = cast("str", request.param)
        config: PretrainedConfig | PreTrainedModel | dict | str | Path
        if config_type == "pretrained_config":
            config = bundle.config
        elif config_type == "pretrained_model":
            config = bundle.model
        elif config_type == "dict":
            config = bundle.config.to_dict()
        elif config_type == "local_file":
            config = bundle.local_dir / "config.json"
        elif config_type == "local_dir":
            config = bundle.local_dir
        elif config_type == "hf_id":
            config = bundle.name_or_path
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        return (
            config,
            bundle,
            patched["speculators.utils.transformers_utils.snapshot_download"],
        )

    @pytest.mark.smoke
    def test_invocation(
        self,
        valid_instances: tuple[
            PretrainedConfig | PreTrainedModel | dict | str | Path,
            PretrainedBundle,
            MockType,
        ],
    ):
        """Test with various config instance types."""
        config, bundle, patched = valid_instances

        result = check_download_model_config(
            config,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
            kwarg="kwarg_value",
        )

        expected: PretrainedConfig | Path | dict
        if not isinstance(config, str):
            if isinstance(config, PreTrainedModel):
                expected = config.config
            elif isinstance(config, Path):
                expected = (
                    config.resolve() if config.is_file() else config / "config.json"
                )
            else:
                expected = config

            assert result == expected
            patched.assert_not_called()
        else:
            assert result == bundle.local_dir.resolve() / "config.json"
            patched.assert_called_once_with(
                str(config),
                allow_patterns=["config.json"],
                cache_dir="/tmp/cache",
                force_download=True,
                local_files_only=True,
                token="asdf",
                revision="main",
                kwarg="kwarg_value",
            )

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
                FileNotFoundError,
                "Failed to download config for",
            ),
        ],
        ids=["invalid_type", "missing_config"],
    )
    def test_invalid_invocation(
        self, config_input, expected_error, error_message, tmp_path: Path
    ):
        """Test with invalid input types and missing configs."""
        with pytest.raises(expected_error) as exc_info:
            check_download_model_config(
                config_input if config_input != "missing_config" else tmp_path
            )
        assert error_message in str(exc_info.value)


class TestLoadModelConfig:
    """Test suite for load_model_config function."""

    @pytest.fixture(
        params=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
        ids=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[
        PretrainedConfig | PreTrainedModel | dict | str | Path,
        PretrainedBundle,
        MagicMock,
        MagicMock,
    ]:
        """Fixture providing valid model instances for testing."""
        factory, patched = mock_pretrained_factory
        bundle = factory.register()

        config_type = cast("str", request.param)
        config: PretrainedConfig | PreTrainedModel | dict | str | Path
        if config_type == "pretrained_config":
            config = bundle.config
        elif config_type == "pretrained_model":
            config = bundle.model
        elif config_type == "dict":
            config = bundle.config.to_dict()
        elif config_type == "local_file":
            config = bundle.local_dir / "config.json"
        elif config_type == "local_dir":
            config = bundle.local_dir
        elif config_type == "hf_id":
            config = bundle.name_or_path
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        return (
            config,
            bundle,
            patched["speculators.utils.transformers_utils.snapshot_download"],
            patched["AutoConfig.from_pretrained"],
        )

    @pytest.mark.smoke
    def test_invocation(
        self,
        valid_instances: tuple[
            PretrainedConfig | PreTrainedModel | dict | str | Path,
            PretrainedBundle,
            MagicMock,
            MagicMock,
        ],
    ):
        """Test with PretrainedConfig and PreTrainedModel instances."""
        config, bundle, patched_snapshot, patched_auto = valid_instances

        result = load_model_config(
            config,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
            kwarg="kwarg_value",
        )
        assert isinstance(result, PretrainedConfig)
        if isinstance(config, dict):
            assert result == PretrainedConfig.from_dict(config)
        else:
            assert result == bundle.config

        if not isinstance(config, str):
            patched_snapshot.assert_not_called()

        if not isinstance(config, (str, Path)):
            patched_auto.assert_not_called()
        else:
            patched_auto.assert_called_once_with(
                str(config),
                cache_dir="/tmp/cache",
                force_download=True,
                local_files_only=True,
                token="asdf",
                revision="main",
                kwarg="kwarg_value",
            )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("config_input", "expected_error", "error_message"),
        [
            (
                123,
                TypeError,
                "Expected config to be a string,",
            ),
            (
                "missing_config",
                FileNotFoundError,
                "Failed to download config for",
            ),
        ],
        ids=["invalid_type", "missing_config"],
    )
    def test_invalid_invocation(
        self, config_input, expected_error, error_message, tmp_path: Path
    ):
        """Test with invalid input types and missing configs."""
        with pytest.raises(expected_error) as exc_info:
            load_model_config(
                config_input if config_input != "missing_config" else tmp_path
            )
        assert error_message in str(exc_info.value)


class TestLoadModelCheckpointConfigDict:
    """Test suite for load_model_checkpoint_config_dict function."""

    @pytest.fixture(
        params=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
        ids=[
            "pretrained_config",
            "pretrained_model",
            "dict",
            "local_file",
            "local_dir",
            "hf_id",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[
        PretrainedConfig | PreTrainedModel | dict | str | Path,
        PretrainedBundle,
        MagicMock,
    ]:
        """Fixture providing valid model instances for testing."""
        factory, patched = mock_pretrained_factory
        bundle = factory.register()

        config_type = cast("str", request.param)
        config: PretrainedConfig | PreTrainedModel | dict | str | Path
        if config_type == "pretrained_config":
            config = bundle.config
        elif config_type == "pretrained_model":
            config = bundle.model
        elif config_type == "dict":
            config = bundle.config.to_dict()
        elif config_type == "local_file":
            config = bundle.local_dir / "config.json"
        elif config_type == "local_dir":
            config = bundle.local_dir
        elif config_type == "hf_id":
            config = bundle.name_or_path
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        return (
            config,
            bundle,
            patched["speculators.utils.transformers_utils.snapshot_download"],
        )

    @pytest.mark.smoke
    def test_invocation(
        self,
        valid_instances: tuple[
            PretrainedConfig | PreTrainedModel | dict | str | Path,
            PretrainedBundle,
            MagicMock,
        ],
    ):
        """Test with various config instance types."""
        config, bundle, patched_snapshot = valid_instances

        result = load_model_checkpoint_config_dict(
            config,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
            kwarg="kwarg_value",
        )
        assert isinstance(result, dict)
        assert result == bundle.config.to_dict()

        if not isinstance(config, str):
            patched_snapshot.assert_not_called()
        else:
            patched_snapshot.assert_called_once_with(
                str(config),
                allow_patterns=["config.json"],
                cache_dir="/tmp/cache",
                force_download=True,
                local_files_only=True,
                token="asdf",
                revision="main",
                kwarg="kwarg_value",
            )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("config_input", "expected_error", "error_message"),
        [
            (
                123,
                TypeError,
                "Expected config to be a string,",
            ),
            (
                "missing_config",
                FileNotFoundError,
                "Failed to download config for",
            ),
        ],
        ids=["invalid_type", "missing_config"],
    )
    def test_invalid_invocation(
        self, config_input, expected_error, error_message, tmp_path: Path
    ):
        """Test with invalid input types and missing configs."""
        with pytest.raises(expected_error) as exc_info:
            load_model_checkpoint_config_dict(
                config_input if config_input != "missing_config" else tmp_path
            )
        assert error_message in str(exc_info.value)


class TestLoadModelCheckpointIndexWeightFiles:
    """Test suite for load_model_checkpoint_index_weight_files function."""

    @pytest.fixture(
        params=[
            "index_file_single",
            "index_file_multiple",
            "index_dir_single",
            "index_dir_multiple",
            "no_index",
        ],
        ids=[
            "index_file_single",
            "index_file_multiple",
            "index_dir_single",
            "index_dir_multiple",
            "no_index",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[Path, list[Path]]:
        """Fixture providing valid index file instances for testing."""
        factory, _ = mock_pretrained_factory
        bundle = factory.register()

        index_type = cast("str", request.param)
        state_dict = bundle.model.state_dict()

        if index_type == "index_file_single":
            return bundle.create_index_single_file(state_dict)

        if index_type == "index_file_multiple":
            return bundle.create_index_multi_file(state_dict)

        if index_type == "index_dir_single":
            _, weight_files = bundle.create_index_single_file(state_dict)
            return bundle.local_dir, weight_files

        if index_type == "index_dir_multiple":
            _, weight_files = bundle.create_index_multi_file(state_dict)
            return bundle.local_dir, weight_files

        if index_type == "no_index":
            bundle.create_bin_file(state_dict)
            return bundle.local_dir, []

        raise ValueError(f"Unknown index type: {index_type}")

    @pytest.mark.smoke
    def test_invocation(self, valid_instances: tuple[Path, list[Path]]):
        """Test with directory containing index files."""
        path, expected_files = valid_instances

        result = load_model_checkpoint_index_weight_files(path)
        assert len(result) == len(expected_files)
        assert all(isinstance(file, Path) for file in result)
        assert all(file.exists() for file in result)
        assert {file.resolve() for file in result} == {
            file.resolve() for file in expected_files
        }

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
            assert "does not contain a valid weight_map" in str(exc_info.value)

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


class TestLoadModelCheckpointWeightFiles:
    """Test suite for load_model_checkpoint_weight_files function."""

    @pytest.fixture(
        params=[
            "index_file_single",
            "index_file_multiple",
            "index_dir_single",
            "index_dir_multiple",
            "bin_file",
            "safetensors_file",
            "bin_dir",
            "safetensors_dir",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[Path, list[Path]]:
        """Fixture providing valid weight file instances for testing."""
        factory, _ = mock_pretrained_factory
        bundle = factory.register()

        instance_type = cast("str", request.param)
        state_dict = bundle.model.state_dict()

        if instance_type in ("index_file_single", "index_file_multiple"):
            return (
                bundle.create_index_single_file(state_dict)
                if instance_type == "index_file_single"
                else bundle.create_index_multi_file(state_dict)
            )

        if instance_type == "index_dir_single":
            _, weight_files = bundle.create_index_single_file(state_dict)
            return bundle.local_dir, weight_files

        if instance_type == "index_dir_multiple":
            _, weight_files = bundle.create_index_multi_file(state_dict)
            return bundle.local_dir, weight_files

        if instance_type == "bin_file":
            weight_file = bundle.create_bin_file(state_dict)
            return weight_file, [weight_file]

        if instance_type == "safetensors_file":
            weight_file = bundle.create_safetensors_file(state_dict)
            return weight_file, [weight_file]

        if instance_type in ("bin_dir", "safetensors_dir"):
            type_: Literal["bin", "safetensors"] = (
                "bin" if instance_type == "bin_dir" else "safetensors"
            )
            weight_files, _ = bundle.create_files(
                state_dict,
                weight_file_names=[
                    f"pytorch_model_00001.{type_}",
                    f"pytorch_model_00002.{type_}",
                ],
                type_=type_,
            )
            return bundle.local_dir, weight_files

        raise ValueError(f"Unknown index type: {instance_type}")

    @pytest.mark.smoke
    def test_invocation(self, valid_instances: tuple[Path, list[Path]]):
        """Test with various valid weight file instances."""
        path, expected_files = valid_instances

        result = load_model_checkpoint_weight_files(path)
        assert len(result) == len(expected_files)
        assert all(isinstance(file, Path) for file in result)
        assert all(file.exists() for file in result)
        assert {file.resolve() for file in result} == {
            file.resolve() for file in expected_files
        }

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
    def test_no_valid_weight_files(self, tmp_path: Path):
        """Test with directory containing no valid weight files."""
        other_file = tmp_path / "README.md"
        other_file.write_text("This is a readme")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_model_checkpoint_weight_files(tmp_path)
        assert "No valid weight files found" in str(exc_info.value)


class TestLoadModelCheckpointStateDict:
    """Test suite for load_model_checkpoint_state_dict function."""

    @pytest.fixture(
        params=[
            "hf_id",
            "local_dir",
            "pretrained_model",
            "bin_file",
            "safetensors_file",
            "index_file_single",
            "index_file_multiple",
            "bin_dir",
            "safetensors_dir",
        ],
        ids=[
            "hf_id",
            "local_dir",
            "pretrained_model",
            "bin_file",
            "safetensors_file",
            "index_file_single",
            "index_file_multiple",
            "bin_dir",
            "safetensors_dir",
        ],
    )
    def valid_instances(
        self,
        request,
        mock_pretrained_factory: tuple[
            MockPretrainedTransformersFactory, dict[str, MockType]
        ],
    ) -> tuple[
        PreTrainedModel | torch.nn.Module | str | Path,
        dict[str, torch.Tensor],
        MockType,
    ]:
        """Fixture providing valid instances for testing."""
        factory, patched = mock_pretrained_factory
        bundle = factory.register()

        instance_type = cast("str", request.param)
        state_dict = bundle.model.state_dict()
        snapshot_patch = patched[
            "speculators.utils.transformers_utils.snapshot_download"
        ]

        if instance_type in ("hf_id", "local_dir"):
            bundle.create_bin_file(state_dict)
            return (
                bundle.name_or_path if instance_type == "hf_id" else bundle.local_dir,
                state_dict,
                snapshot_patch,
            )

        if instance_type == "pretrained_model":
            return bundle.model, state_dict, snapshot_patch

        if instance_type in ("bin_file", "safetensors_file"):
            weight_file = (
                bundle.create_bin_file(state_dict)
                if instance_type == "bin_file"
                else bundle.create_safetensors_file(state_dict)
            )
            return weight_file, state_dict, snapshot_patch

        if instance_type in ("index_file_single", "index_file_multiple"):
            bundle.create_index_single_file(
                state_dict
            ) if instance_type == "index_file_single" else (
                bundle.create_index_multi_file(state_dict)
            )
            return bundle.local_dir, state_dict, snapshot_patch

        if instance_type in ("bin_dir", "safetensors_dir"):
            type_: Literal["bin", "safetensors"] = (
                "bin" if instance_type == "bin_dir" else "safetensors"
            )
            weight_files, _ = bundle.create_files(
                state_dict,
                weight_file_names=[
                    f"pytorch_model_00001.{type_}",
                    f"pytorch_model_00002.{type_}",
                ],
                type_=type_,
            )
            return bundle.local_dir, state_dict, snapshot_patch

        raise ValueError(f"Unknown instance type: {instance_type}")

    @pytest.mark.smoke
    def test_invocation(
        self,
        valid_instances: tuple[
            PreTrainedModel | torch.nn.Module | str | Path,
            dict[str, torch.Tensor],
            MockType,
        ],
    ):
        """Test load_model_checkpoint_state_dict with various valid instances."""
        model, expected_state_dict, patched_snapshot = valid_instances

        result = load_model_checkpoint_state_dict(
            model,
            cache_dir="/tmp/cache",
            force_download=True,
            local_files_only=True,
            token="asdf",
            revision="main",
        )
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result)
        assert all(isinstance(value, torch.Tensor) for value in result.values())
        assert len(result) == len(expected_state_dict)

        if isinstance(model, str):
            patched_snapshot.assert_called_once_with(
                repo_id=model,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.index.json"],
                cache_dir="/tmp/cache",
                force_download=True,
                local_files_only=True,
                token="asdf",
                revision="main",
            )

    @pytest.mark.sanity
    def test_invalid_invocation_unsupported_file_type(self, tmp_path: Path):
        """Test with unsupported file type."""
        # Create an unsupported file type
        unsupported_file = tmp_path / "model.txt"
        unsupported_file.write_text("This is not a weight file")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_model_checkpoint_state_dict(unsupported_file)

        assert "No valid weight files found" in str(exc_info.value)

    @pytest.mark.sanity
    def test_no_valid_weight_files(self, tmp_path: Path):
        """Test with directory containing no valid weight files."""
        other_file = tmp_path / "README.md"
        other_file.write_text("This is a readme")

        with pytest.raises(FileNotFoundError) as exc_info:
            load_model_checkpoint_state_dict(tmp_path)

        assert "No valid weight files found" in str(exc_info.value)
