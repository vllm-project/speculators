"""Configuration generator for EAGLE data generation pipeline.

Provides type-safe configuration generation with reproducibility tracking
and schema documentation.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig

from speculators.data_generation.logging_utils import PipelineLogger

if TYPE_CHECKING:
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )

__all__ = ["DataGenerationConfig", "PackageVersions"]

log = PipelineLogger(__name__)

CONFIG_VERSION = "2.0"


def _get_gpu_info() -> str:
    """Get GPU information string.

    :return: GPU model and count, or "CPU only" if no GPU available
    """
    if not torch.cuda.is_available():
        return "CPU only"

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    return gpu_name if gpu_count == 1 else f"{gpu_count}x {gpu_name}"


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

        try:
            import vllm  # noqa: PLC0415

            vllm_version = vllm.__version__
        except (ImportError, AttributeError):
            vllm_version = "unknown"

        return cls(
            torch=torch.__version__,
            vllm=vllm_version,
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
    max_model_len: int
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

    batch_size: int
    cache_dir: str


@dataclass
class FormatConfig:
    """Output format specification for generated data files."""

    file_pattern: str
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

        :return: Dictionary representation of the config
        """
        return asdict(self)

    @classmethod
    def from_generator(
        cls,
        generator: VllmHiddenStatesGenerator,
        train_data_path: str,
        seq_length: int,
        batch_size: int,
        cache_dir: str,
        num_samples: int,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> DataGenerationConfig:
        """Create config from an initialized VllmHiddenStatesGenerator.

        :param generator: Initialized VllmHiddenStatesGenerator instance
        :param train_data_path: Path or HF dataset name used for training data
        :param seq_length: Maximum sequence length used in preprocessing
        :param batch_size: Batch size used during generation
        :param cache_dir: Directory where preprocessed data is cached
        :param num_samples: Total number of samples generated
        :param max_samples: Maximum samples to process (None = all)
        :param seed: Random seed used
        :return: Complete DataGenerationConfig ready to save as JSON
        """
        return generate_config(
            target_model_path=generator.model_path,
            train_data_path=train_data_path,
            seq_length=seq_length,
            layer_ids=generator.layer_ids,
            tensor_parallel_size=generator.tensor_parallel_size,
            max_model_len=generator.vllm_config.model_config.max_model_len,
            gpu_memory_utilization=generator.vllm_config.cache_config.gpu_memory_utilization,
            batch_size=batch_size,
            cache_dir=cache_dir,
            num_samples=num_samples,
            max_samples=max_samples,
            seed=seed,
        )


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


def generate_config(
    target_model_path: str,
    train_data_path: str,
    seq_length: int,
    layer_ids: list[int],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    cache_dir: str,
    num_samples: int,
    max_samples: int | None = None,
    seed: int = 0,
) -> DataGenerationConfig:
    """Generate complete data generation configuration with full metadata.

    :param target_model_path: HuggingFace model ID or local path
    :param train_data_path: Path or HF dataset name for training data
    :param seq_length: Maximum sequence length for tokenization
    :param layer_ids: Transformer layer indices to extract hidden states from
    :param tensor_parallel_size: Number of GPUs for tensor parallelism
    :param max_model_len: Maximum sequence length the model supports
    :param gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
    :param batch_size: Number of samples to process in parallel
    :param cache_dir: Directory for cached preprocessed data
    :param num_samples: Total number of samples generated
    :param max_samples: Maximum samples to process (None = all)
    :param seed: Random seed for reproducibility
    :return: Complete configuration with package versions and GPU info
    """
    log.subsection("Generating configuration metadata")

    package_versions = PackageVersions.from_environment()
    log.info(f"Packages: torch={package_versions.torch}, vllm={package_versions.vllm}")

    hidden_size = _get_hidden_size_from_model(target_model_path)
    log.info(f"Hidden size: {hidden_size}")
    log.info(f"GPU: {_get_gpu_info()}")

    config = DataGenerationConfig(
        version=CONFIG_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        speculators_version=package_versions.speculators,
        reproducibility=ReproducibilityInfo(
            command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            package_versions=package_versions,
        ),
        model=ModelConfig(
            target_model_path=target_model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            hidden_size=hidden_size,
        ),
        data=DataConfig(
            train_data_path=train_data_path,
            seq_length=seq_length,
            max_samples=max_samples,
            num_samples=num_samples,
            seed=seed,
        ),
        hidden_states=HiddenStatesConfig(layer_ids=layer_ids),
        generation=GenerationConfig(batch_size=batch_size, cache_dir=cache_dir),
        format=FormatConfig.create_default(
            num_layers=len(layer_ids), hidden_size=hidden_size
        ),
    )

    log.success("Configuration generated")
    return config
