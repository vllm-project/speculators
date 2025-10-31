"""Configuration generator for EAGLE data generation pipeline.

Provides type-safe configuration generation with reproducibility tracking,
example generation, and schema documentation.
"""

import hashlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from speculators.data_generation.logging_utils import PipelineLogger

log = PipelineLogger(__name__)


@dataclass
class PackageVersions:
    """Package versions for reproducibility."""

    torch: str
    vllm: str
    transformers: str
    speculators: str | None = None

    @classmethod
    def from_environment(cls) -> "PackageVersions":
        """Detect package versions from current environment."""
        import transformers  # noqa: PLC0415

        try:
            import vllm  # noqa: PLC0415

            vllm_version = vllm.__version__
        except (ImportError, AttributeError):
            vllm_version = "unknown"

        try:
            from importlib.metadata import version  # noqa: PLC0415

            speculators_version = version("speculators")
        except Exception:  # noqa: BLE001
            speculators_version = None

        return cls(
            torch=torch.__version__,
            vllm=vllm_version,
            transformers=transformers.__version__,
            speculators=speculators_version,
        )


@dataclass
class ReproducibilityInfo:
    """Information for reproducing the data generation."""

    command: str
    cache_key: str
    gpu: str
    packages: dict[str, str]

    @classmethod
    def create(
        cls, command: str, cache_key: str, package_versions: PackageVersions
    ) -> "ReproducibilityInfo":
        """Create reproducibility info with detected GPU info."""
        return cls(
            command=command,
            cache_key=cache_key,
            gpu=_get_gpu_info(),
            packages=asdict(package_versions),
        )


@dataclass
class ModelConfig:
    """Model configuration."""

    target_model_path: str
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    hidden_size: int


@dataclass
class DataConfig:
    """Data configuration."""

    train_data_path: str
    chat_template: str
    seq_length: int
    max_samples: int | None
    num_samples: int
    seed: int


@dataclass
class HiddenStatesConfig:
    """Hidden states configuration."""

    layer_ids: list[int]
    description: str = "3 layers for EAGLE3 fusion, last layer for target logits"


@dataclass
class GenerationConfig:
    """Generation parameters."""

    batch_size: int
    cache_dir: str


@dataclass
class FormatConfig:
    """Output format specification."""

    file_pattern: str
    schema: dict[str, dict[str, Any]]

    @classmethod
    def create_default(cls, num_layers: int, hidden_size: int) -> "FormatConfig":
        """Create default format config."""
        return cls(
            file_pattern="data_{idx}.pt",
            schema={
                "input_ids": {
                    "dtype": "torch.long",
                    "shape": "[seq_len]",
                },
                "hidden_states": {
                    "dtype": "list[torch.bfloat16]",
                    "shape": f"list of [seq_len, {hidden_size}]",
                    "num_tensors": num_layers,
                },
                "loss_mask": {
                    "dtype": "torch.long",
                    "shape": "[seq_len]",
                },
            },
        )


@dataclass
class ExampleData:
    """Example input/output for documentation."""

    prompt_token_ids: list[int]
    prompt_str: str
    output_token_ids: list[int]
    output_str: str


@dataclass
class DataGenerationConfig:
    """Complete data generation configuration."""

    version: str
    generated_at: str
    speculators_version: str | None
    reproducibility: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
    hidden_states: dict[str, Any]
    generation: dict[str, Any]
    format: dict[str, Any]
    example_prompt_token_ids: list[int]
    example_prompt_str: str
    example_output_token_ids: list[int]
    example_output_str: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _get_gpu_info() -> str:
    """Get GPU information string."""
    if not torch.cuda.is_available():
        return "CPU only"

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    return gpu_name if gpu_count == 1 else f"{gpu_count}x {gpu_name}"


def _get_hidden_size_from_model(model_path: str) -> int:
    """Extract hidden size from model config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "d_model"):
        return config.d_model
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size

    raise ValueError(f"Could not determine hidden size for {model_path}")


def generate_example_data(
    tokenizer: PreTrainedTokenizer,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
) -> ExampleData:
    """Generate example data from actual token IDs."""
    return ExampleData(
        prompt_token_ids=prompt_token_ids,
        prompt_str=tokenizer.decode(prompt_token_ids, skip_special_tokens=False),
        output_token_ids=output_token_ids,
        output_str=tokenizer.decode(output_token_ids, skip_special_tokens=False),
    )


def generate_config(
    target_model_path: str,
    train_data_path: str,
    chat_template: str,
    seq_length: int,
    layer_ids: list[int],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    batch_size: int,
    cache_dir: str,
    num_samples: int,
    example_prompt_token_ids: list[int],
    example_output_token_ids: list[int],
    max_samples: int | None = None,
    seed: int = 0,
    tokenizer: PreTrainedTokenizer | None = None,
) -> DataGenerationConfig:
    """Generate complete data generation configuration."""
    log.subsection("Generating configuration metadata")

    if tokenizer is None:
        log.info(f"Loading tokenizer from {target_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            target_model_path, trust_remote_code=True
        )

    package_versions = PackageVersions.from_environment()
    log.info(f"Packages: torch={package_versions.torch}, vllm={package_versions.vllm}")

    hidden_size = _get_hidden_size_from_model(target_model_path)
    log.info(f"Hidden size: {hidden_size}")
    log.info(f"GPU: {_get_gpu_info()}")

    log.info("Generating example data")
    example = generate_example_data(
        tokenizer, example_prompt_token_ids, example_output_token_ids
    )

    reproducibility_info = ReproducibilityInfo.create(
        command=" ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        cache_key=hashlib.md5(
            f"{target_model_path}_{chat_template}_{seq_length}_{train_data_path}".encode()
        ).hexdigest(),
        package_versions=package_versions,
    )

    model_config = ModelConfig(
        target_model_path=target_model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        hidden_size=hidden_size,
    )

    data_config = DataConfig(
        train_data_path=train_data_path,
        chat_template=chat_template,
        seq_length=seq_length,
        max_samples=max_samples,
        num_samples=num_samples,
        seed=seed,
    )

    hidden_states_config = HiddenStatesConfig(layer_ids=layer_ids)
    generation_config = GenerationConfig(batch_size=batch_size, cache_dir=cache_dir)
    format_config = FormatConfig.create_default(
        num_layers=len(layer_ids), hidden_size=hidden_size
    )

    config = DataGenerationConfig(
        version="2.0",
        generated_at=datetime.now().isoformat(),
        speculators_version=package_versions.speculators,
        reproducibility=asdict(reproducibility_info),
        model=asdict(model_config),
        data=asdict(data_config),
        hidden_states=asdict(hidden_states_config),
        generation=asdict(generation_config),
        format=asdict(format_config),
        example_prompt_token_ids=example.prompt_token_ids,
        example_prompt_str=example.prompt_str,
        example_output_token_ids=example.output_token_ids,
        example_output_str=example.output_str,
    )

    log.success("Configuration generated")
    return config


def extract_config_from_generator(
    generator: Any,
    train_data_path: str,
    chat_template: str,
    seq_length: int,
    batch_size: int,
    cache_dir: str,
    num_samples: int,
    example_prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_samples: int | None = None,
    seed: int = 0,
) -> DataGenerationConfig:
    """Extract configuration from VllmHiddenStatesGenerator instance."""
    log.info("Generating example from vLLM generator")
    example_prompt_token_ids = generator.tokenizer.encode(
        example_prompt, add_special_tokens=True
    )

    outputs = generator.llm.generate(
        prompt_token_ids=[example_prompt_token_ids],
        sampling_params=generator.sampling_params,
    )
    example_output_token_ids = outputs[0].outputs[0].token_ids

    return generate_config(
        target_model_path=generator.model_path,
        train_data_path=train_data_path,
        chat_template=chat_template,
        seq_length=seq_length,
        layer_ids=generator.layer_ids,
        tensor_parallel_size=generator.tensor_parallel_size,
        max_model_len=generator.vllm_config.model_config.max_model_len,
        gpu_memory_utilization=generator.vllm_config.cache_config.gpu_memory_utilization,
        batch_size=batch_size,
        cache_dir=cache_dir,
        num_samples=num_samples,
        example_prompt_token_ids=example_prompt_token_ids,
        example_output_token_ids=example_output_token_ids,
        max_samples=max_samples,
        seed=seed,
        tokenizer=generator.tokenizer,
    )
