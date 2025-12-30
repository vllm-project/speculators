"""Configuration generator for EAGLE data generation pipeline.

Provides type-safe configuration generation with reproducibility tracking
and schema documentation.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from transformers import AutoConfig

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.utils.util import get_device_name

if TYPE_CHECKING:
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )

__all__ = ["DataGenerationConfig", "PackageVersions"]

log = PipelineLogger(__name__)


def _get_gpu_info() -> str:
    """Get GPU information string.

    :return: GPU model and count or NPU model and count,
             or "CPU only" if no GPU/NPU available
    """
    device_count = torch.accelerator.device_count()
    device_name = get_device_name(0)
    if device_name == "NO ACCELERATOR":
        return "CPU ONLY"
    else:
        return device_name if device_count == 1 else f"{device_count}x {device_name}"


@dataclass
class PackageVersions:
    """Package versions for full reproducibility of data generation."""

    torch: str
    vllm: str
    transformers: str
    speculators: str

    @classmethod
    def from_environment(cls) -> PackageVersions:
        """Detect package versions from current environment.

        :return: PackageVersions with all detected versions
        """
        from importlib.metadata import version  # noqa: PLC0415

        import transformers  # noqa: PLC0415
        import vllm  # noqa: PLC0415

        return cls(
            torch=torch.__version__,
            vllm=vllm.__version__,
            transformers=transformers.__version__,
            speculators=version("speculators"),
        )


@dataclass
class ReproducibilityInfo:
    """Information needed to reproduce the data generation run."""

    command: str
    package_versions: PackageVersions
    gpu: str = field(default_factory=_get_gpu_info)


@dataclass
class ModelConfig:
    """Model configuration for the target model."""

    target_model_path: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    hidden_size: int


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    train_data_path: str
    seq_length: int
    max_samples: int | None
    num_samples: int
    seed: int
    chat_template_note: str = "Uses tokenizer's built-in chat template"


@dataclass
class HiddenStatesConfig:
    """Configuration for which hidden states to extract."""

    layer_ids: list[int]
    description: str = "Layers selected for EAGLE3 fusion and target logits"


@dataclass
class GenerationConfig:
    """Runtime generation parameters."""

    cache_dir: str


@dataclass
class FormatConfig:
    """Output format specification for generated data files."""

    file_pattern: str
    data_format_version: int
    schema: dict[str, dict[str, Any]]

    @classmethod
    def create_default(cls, num_layers: int, hidden_size: int) -> FormatConfig:
        """Create default format config with schema documentation.

        :param num_layers: Number of hidden state layers being saved
        :param hidden_size: Dimension of each hidden state tensor
        :return: FormatConfig with complete schema information
        """
        return cls(
            file_pattern="data_{idx}.pt",
            data_format_version=1,
            schema={
                "input_ids": {
                    "dtype": "torch.long",
                    "shape": "[seq_len]",
                    "description": "Tokenized input sequence",
                },
                "hidden_states": {
                    "dtype": "list[torch.bfloat16]",
                    "shape": f"list of [seq_len, {hidden_size}]",
                    "num_tensors": num_layers,
                    "description": f"Hidden states from {num_layers} layers",
                },
                "loss_mask": {
                    "dtype": "torch.long",
                    "shape": "[seq_len]",
                    "description": "1 for assistant tokens to train on, 0 elsewhere",
                },
            },
        )


@dataclass
class DataGenerationConfig:
    """Complete configuration for EAGLE data generation run.

    Saved alongside generated data for full reproducibility.
    """

    VERSION: ClassVar[str] = "2.0"

    version: str
    generated_at: str
    speculators_version: str
    reproducibility: ReproducibilityInfo
    model: ModelConfig
    data: DataConfig
    hidden_states: HiddenStatesConfig
    generation: GenerationConfig
    format: FormatConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Handles Path objects by converting them to strings.

        :return: Dictionary representation of the config
        """

        def serialize_value(obj: Any) -> Any:
            """Recursively convert Path objects to strings."""
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize_value(item) for item in obj]
            return obj

        config_dict = asdict(self)
        return serialize_value(config_dict)

    @classmethod
    def from_generator(
        cls,
        generator: VllmHiddenStatesGenerator,
        train_data_path: str,
        seq_length: int,
        cache_dir: str,
        num_samples: int,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> DataGenerationConfig:
        """Create config from an initialized VllmHiddenStatesGenerator.

        :param generator: Initialized VllmHiddenStatesGenerator instance
        :param train_data_path: Path or HF dataset name used for training data
        :param seq_length: Maximum sequence length used in preprocessing
        :param cache_dir: Directory where preprocessed data is cached
        :param num_samples: Total number of samples generated
        :param max_samples: Maximum samples to process (None = all)
        :param seed: Random seed used
        :return: Complete DataGenerationConfig ready to save as JSON
        """
        log.subsection("Generating configuration metadata")

        package_versions = PackageVersions.from_environment()
        log.info(
            f"Packages: torch={package_versions.torch}, vllm={package_versions.vllm}"
        )

        hidden_size = _get_hidden_size_from_model(generator.model_path)
        log.info(f"Hidden size: {hidden_size}")
        log.info(f"GPU: {_get_gpu_info()}")

        config = cls(
            version=cls.VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            speculators_version=package_versions.speculators,
            reproducibility=ReproducibilityInfo(
                command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
                package_versions=package_versions,
            ),
            model=ModelConfig(
                target_model_path=generator.model_path,
                tensor_parallel_size=generator.tensor_parallel_size,
                gpu_memory_utilization=generator.vllm_config.cache_config.gpu_memory_utilization,
                hidden_size=hidden_size,
            ),
            data=DataConfig(
                train_data_path=train_data_path,
                seq_length=seq_length,
                max_samples=max_samples,
                num_samples=num_samples,
                seed=seed,
            ),
            hidden_states=HiddenStatesConfig(layer_ids=generator.layer_ids),
            generation=GenerationConfig(cache_dir=cache_dir),
            format=FormatConfig.create_default(
                num_layers=len(generator.layer_ids), hidden_size=hidden_size
            ),
        )

        log.success("Configuration generated")
        return config


def _get_hidden_size_from_model(model_path: str) -> int:
    """Extract hidden size from model config.

    :param model_path: HuggingFace model ID or local path
    :return: Hidden state dimension
    :raises ValueError: If hidden size cannot be determined
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if hidden_size := getattr(config, "hidden_size", None):
        return hidden_size

    if text_config := getattr(config, "text_config", None):
        if hidden_size := getattr(text_config, "hidden_size", None):
            return hidden_size

    raise ValueError(
        f"Could not determine hidden size for {model_path}. "
        f"Expected 'hidden_size' or 'text_config.hidden_size' attribute"
    )
