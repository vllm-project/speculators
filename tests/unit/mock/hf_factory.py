from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture, MockType
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

__all__ = [
    "MockPretrainedTransformersFactory",
    "PretrainedBundle",
    "mock_llama3_2m_config_dict",
    "mock_llama3_2m_state_dict",
]


def mock_llama3_2m_config_dict() -> dict[str, Any]:
    vocab_size = 4008
    hidden_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = hidden_size // num_attention_heads
    intermediate_size = int(3.5 * hidden_size)
    num_hidden_layers = 4

    return {
        "_name_or_path": "meta-llama/Llama-3.2-2m-Instruct",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": vocab_size - 256,  # 128000 for 8b
        "eos_token_id": [
            vocab_size - 255,  # 128001 for 8b
            vocab_size - 248,  # 128008 for 8b
            vocab_size - 247,  # 128009 for 8b
        ],
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": 131072,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_key_value_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.46.0",
        "use_cache": True,
        "vocab_size": vocab_size,
    }


def mock_llama3_2m_state_dict() -> dict[str, torch.Tensor]:
    config_dict = mock_llama3_2m_config_dict()
    config = LlamaConfig(**config_dict)
    model = LlamaForCausalLM(config)

    return model.state_dict()


@dataclass
class PretrainedBundle:
    name_or_path: str
    local_dir: Path
    aliases: list[str]
    model: PreTrainedModel
    config: PretrainedConfig

    @property
    def sources(self) -> list[Any]:
        return [
            self.name_or_path,
            self.local_dir,
            str(self.local_dir),
            self.local_dir / "config.json",
            str(self.local_dir / "config.json"),
            self.model,
            self.config,
        ] + self.aliases

    def create_bin_file(
        self, state_dict: dict, weight_file_name: str | None = None
    ) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        weight_file_name = weight_file_name or "pytorch_model.bin"
        torch.save(state_dict, self.local_dir / weight_file_name)

        return self.local_dir / weight_file_name

    def create_safetensors_file(
        self, state_dict: dict, weight_file_name: str | None = None
    ) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        weight_file_name = weight_file_name or "pytorch_model.safetensors"
        save_file(state_dict, self.local_dir / weight_file_name)

        return self.local_dir / weight_file_name

    def create_files(
        self,
        state_dict: dict,
        weight_file_names: list[str],
        type_: Literal["bin", "safetensors"] = "bin",
    ) -> tuple[list[Path], list[dict[str, torch.Tensor]]]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if not weight_file_names or len(weight_file_names) < 1:
            raise ValueError("At least one weight file name must be provided")
        if len(weight_file_names) > len(state_dict):
            raise ValueError("More weight file names provided than state dict keys")

        state_dict_keys = list(state_dict.keys())
        tensors_per_file = len(state_dict) // len(weight_file_names)
        weight_files = []
        state_dicts = []

        for index, weight_file_name in enumerate(weight_file_names):
            start_index = index * tensors_per_file
            end_index = (
                (index + 1) * tensors_per_file
                if index < len(weight_file_names) - 1
                else len(state_dict)
            )
            file_state_dict = {
                key: state_dict[key] for key in state_dict_keys[start_index:end_index]
            }
            state_dicts.append(file_state_dict)
            weight_files.append(
                self.create_safetensors_file(file_state_dict, weight_file_name)
                if type_ == "safetensors"
                else self.create_bin_file(file_state_dict, weight_file_name)
            )

        return weight_files, state_dicts

    def create_index_single_file(
        self, state_dict: dict, type_: Literal["bin", "safetensors"] = "bin"
    ) -> tuple[Path, list[Path]]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        weight_file: Path = (
            self.create_safetensors_file(state_dict)
            if type_ == "safetensors"
            else self.create_bin_file(state_dict)
        )
        index_file = self.local_dir / f"model.{type_}.index.json"
        index_file.write_text(
            json.dumps({"weight_map": dict.fromkeys(state_dict, weight_file.name)})
        )

        return index_file, [weight_file]

    def create_index_multi_file(
        self,
        state_dict: dict,
        num_files: int = 2,
        type_: Literal["bin", "safetensors"] = "bin",
    ) -> tuple[Path, list[Path]]:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        file_names = []
        for file_index in range(num_files):
            file_names.append(
                f"pytorch_model-{file_index + 1:05d}-of-{num_files:05d}.{type_}"
            )
        weight_files, state_dicts = self.create_files(state_dict, file_names, type_)
        weight_map = {}
        for file_name, state_dict_part in zip(file_names, state_dicts):
            for key in state_dict_part:
                weight_map[key] = file_name

        index_file = self.local_dir / f"model.{type_}.index.json"
        index_file.write_text(json.dumps({"weight_map": weight_map}))

        return index_file, weight_files


class MockPretrainedTransformersFactory:
    def __init__(self, temp_dir: Path, mocker: MockerFixture):
        self.temp_dir = temp_dir
        self.mocker = mocker
        self.registry: dict[Any, PretrainedBundle] = {}

    def register(
        self,
        name_or_path: str | None = None,
        aliases: list[str] | None = None,
        config_dict: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> PretrainedBundle:
        config = self.create_config(config_dict)
        model = self.create_model(config, state_dict)
        if name_or_path is None:
            name_or_path = model.name_or_path or ""
        local_dir = self.temp_dir / f"models--{name_or_path.replace('/', '--')}"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text(
            json.dumps(config.to_dict(), default=str)
        )
        bundle = PretrainedBundle(
            name_or_path=name_or_path,
            local_dir=local_dir,
            aliases=aliases or [],
            model=model,
            config=config,
        )

        for source in bundle.sources:
            self.registry[source] = bundle

        return bundle

    def create_config(
        self, config_dict: dict[str, Any] | None = None
    ) -> PretrainedConfig:
        if config_dict is None:
            config_dict = mock_llama3_2m_config_dict()

        config = MagicMock(spec=PretrainedConfig)
        config.to_dict = MagicMock(return_value=config_dict)

        return config

    def create_model(
        self,
        config: PretrainedConfig,
        state_dict: dict[str, Any] | None = None,
        name_or_path: str | None = None,
    ) -> PreTrainedModel:
        if state_dict is None:
            state_dict = mock_llama3_2m_state_dict()
        config_dict = config.to_dict()
        model = MagicMock(spec=PreTrainedModel)
        model.config = config
        model.name_or_path = name_or_path or config_dict.get(
            "_name_or_path", config_dict.get("name_or_path")
        )
        model.state_dict = MagicMock(return_value=state_dict)

        # Add required module-like attributes for PyTorch loading
        model._modules = {}
        model._parameters = {}
        model._buffers = {}
        model._non_persistent_buffers_set = set()
        model._backward_hooks = {}
        model._forward_hooks = {}
        model._forward_pre_hooks = {}
        model._state_dict_hooks = {}
        model._load_state_dict_pre_hooks = {}
        model._load_state_dict_post_hooks = {}

        return model

    def resolve(self, source: Any) -> PretrainedBundle:
        if isinstance(source, dict):
            source = source.get("_name_or_path", source.get("name_or_path"))

        if source not in self.registry and str(source) not in self.registry:
            raise ValueError(f"Unregistered source: {source}")

        if source in self.registry:
            return self.registry[source]

        if str(source) in self.registry:
            return self.registry[str(source)]

        raise ValueError(f"Unregistered source: {source}")

    def patch_transformers(self) -> dict[str, Any]:
        def mock_autoconfig_from_pretrained(
            pretrained_model_name_or_path, return_unused_kwargs=False, **kwargs
        ):
            config = self.resolve(pretrained_model_name_or_path).config
            if return_unused_kwargs:
                return config, kwargs
            return config

        return {
            "AutoConfig.from_pretrained": self.mocker.patch.object(
                AutoConfig,
                "from_pretrained",
                side_effect=mock_autoconfig_from_pretrained,
            ),
            "AutoModelForCausalLM.from_pretrained": self.mocker.patch.object(
                AutoModelForCausalLM,
                "from_pretrained",
                side_effect=lambda pretrained_model_name_or_path, **_: self.resolve(
                    pretrained_model_name_or_path
                ).model,
            ),
            "speculators.utils.transformers_utils.snapshot_download": self.mocker.patch(
                "speculators.utils.transformers_utils.snapshot_download",
                side_effect=lambda repo_id, **_: str(self.resolve(repo_id).local_dir),
            ),
        }


@pytest.fixture
def mock_pretrained_factory(
    tmp_path: Path, mocker: MockerFixture
) -> tuple[MockPretrainedTransformersFactory, dict[str, MockType]]:
    factory = MockPretrainedTransformersFactory(tmp_path, mocker)
    patched = factory.patch_transformers()

    return factory, patched
