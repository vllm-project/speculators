"""
Prepare data for speculator training

Accepted inputs contain responses produced by the target model, either as
natural-language conversations or as speculator-format ``input_ids`` and
``loss_mask`` rows. For natural-language input this command:

1. Uses the target model's vLLM endpoint to render each conversation
2. Derives a loss mask from each assistant-turn boundary
3. Records token frequency statistics

Rendering converts an existing on-policy conversation into speculator format.
It does not generate responses or make an arbitrary conversation on-policy.

The output of this command is:
1. Processed dataset ready for online training or offline datagen in output_dir
2. Token frequency statistics file at token_freq_path

Preprocessing will be skipped if the dataset already exists at the output directory.
Token frequencies are saved in the output directory by default.

Usage::

    speculators prepare-data \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --data ./on_policy_conversations.jsonl \\
        --render-endpoint http://localhost:8000 \\
        --output ./training_data \\
        --max-samples 5000
"""

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import (
    default_preprocessing_workers,
    load_and_preprocess_dataset,
)

log = PipelineLogger(__name__)


PREPARE_DATA_OVERWRITE_ALLOWED_FILES = {
    "dataset_info.json",
    "state.json",
    "token_freq.pt",
}


def assert_safe_to_overwrite(output: Path, token_freq_path: Path) -> None:
    """Refuse to ``--overwrite`` a directory holding non-artifact files.

    Guards against pointing ``--output`` at a directory with unrelated user files
    and wiping it: only prepare-data's own outputs (``.arrow`` shards, dataset
    metadata, and the token frequency file) may be deleted.
    """
    unexpected_paths = []
    resolved_token_freq_path = token_freq_path.resolve()
    for path in output.iterdir():
        if path.is_file() and (
            path.suffix == ".arrow"
            or path.name in PREPARE_DATA_OVERWRITE_ALLOWED_FILES
            or path.resolve() == resolved_token_freq_path
        ):
            continue
        unexpected_paths.append(path)

    if unexpected_paths:
        formatted_paths = ", ".join(str(path) for path in unexpected_paths)
        raise ValueError(
            "--overwrite would delete files that do not look like prepare-data "
            f"artifacts: {formatted_paths}. Remove them manually or choose a "
            "different --output directory."
        )


def prepare_data(
    model: Annotated[
        str,
        typer.Option(help="HuggingFace model ID or local path for target model"),
    ],
    data: Annotated[
        list[str],
        typer.Option("--data", help="Path to training data (repeatable)"),
    ],
    output: Annotated[
        str,
        typer.Option(help="Directory to save output dataset"),
    ] = "./output",
    seq_length: Annotated[
        int,
        typer.Option(help="Maximum sequence length for preprocessing and model"),
    ] = 8192,
    max_samples: Annotated[
        int | None,
        typer.Option(help="Maximum number of samples to process"),
    ] = None,
    token_freq_path: Annotated[
        str | None,
        typer.Option(
            help="Path to save token frequency distribution",
        ),
    ] = None,
    render_endpoint: Annotated[
        str | None,
        typer.Option(
            help=(
                "Base URL of a running vLLM server (e.g. http://localhost:8000). "
                "Required unless every --data input already contains input_ids "
                "and loss_mask."
            ),
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(help="Random seed"),
    ] = 0,
    num_preprocessing_workers: Annotated[
        int | None,
        typer.Option(
            help=(
                "Number of CPU processes for dataset preprocessing. Each one "
                "blocks on a single render call at a time, so this is also the "
                "render concurrency. Defaults to a shared render CPU budget using "
                "75% of available CPUs, with a maximum of 128."
            ),
        ),
    ] = None,
    minimum_valid_tokens: Annotated[
        int | None,
        typer.Option(
            help=(
                "Drop samples whose loss mask contains fewer than this many "
                "trainable tokens."
            ),
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Forcibly rerun. Deletes existing content in output dir",
        ),
    ] = False,
    allow_empty_output: Annotated[
        bool,
        typer.Option(
            "--allow-empty-output",
            help=(
                "Allow writing an empty preprocessed dataset. By default raises "
                "when normalization or filtering removes every sample."
            ),
        ),
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code",
            help=(
                "Allow executing code from HF Hub when loading the target "
                "model's processor."
            ),
        ),
    ] = False,
) -> None:
    """Preprocess a dataset for speculator training.

    Tokenizes each sample, produces loss/assistant masks, and records token
    frequency statistics. Output is a HuggingFace dataset ready for online
    training or offline data generation.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log.section("Preparing data")
    log.config(
        {
            "Target Model": model,
            "Dataset": data,
            "Output Dir": output,
        }
    )

    output_path = Path(output)
    resolved_token_freq_path = (
        output_path / "token_freq.pt"
        if token_freq_path is None
        else Path(token_freq_path)
    )

    if output_path.exists():
        if not overwrite and list(output_path.glob("*.arrow")):
            log.warning(
                "Dataset files already exist in output directory, skipping "
                "preprocessing. To overwrite existing files use --overwrite."
            )
            raise typer.Exit
        if overwrite:
            assert_safe_to_overwrite(output_path, resolved_token_freq_path)
            log.warning(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True)
    else:
        output_path.mkdir(parents=True)

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=model,
        train_data_paths=data,
        seq_length=seq_length,
        build_dataset_num_proc=(
            num_preprocessing_workers
            if num_preprocessing_workers is not None
            else default_preprocessing_workers()
        ),
        seed=seed,
        max_samples=max_samples,
        token_freq_path=resolved_token_freq_path,
        render_endpoint=render_endpoint,
        minimum_valid_tokens=minimum_valid_tokens,
        allow_empty_output=allow_empty_output,
        trust_remote_code=trust_remote_code,
    )

    log.info("Done preparing data")
    log.section(f"Writing dataset to {output}")
    dataset.save_to_disk(output)
