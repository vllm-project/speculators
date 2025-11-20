"""
Combined EAGLE3 Data Generation and Training Pipeline

This script is a convenience wrapper around the following scripts:
  1. scripts/data_generation_offline.py
  2. scripts/build_vocab_mapping.py
  3. scripts/train.py

It can be used to run the full pipeline in one command. It also ensures each script is
run with the correct arguments and dependencies.

Prerequisites:
  - python 3.10+
  - uv (`pip install uv`)

Usage:
    Update arguments below. Then run:
    python scripts/gen_and_train.py

    Note: You can call the script with environment variables (like
    `CUDA_VISIBLE_DEVICES` and `HF_HOME`) to control the behavior of the scripts.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple
import enum

import psutil

from speculators.data_generation.vocab_mapping import (
    combine_token_frequency_distributions,
)


class _NS(enum.Enum):
    """Class containing a sentinel value used to indicate unset arguments."""

    # https://github.com/python/typing/issues/236#issuecomment-227180301
    value = 0


_NOTSET = _NS.value  # sentinel value


# Output structure:
# output_path/
#   gen/
#     <dataset1_name>/
#       data_config.json
#       data_0.pt
#       data_1.pt
#       ...
#     <dataset2_name>/
#       data_config.json
#       data_0.pt
#       data_1.pt
#       ...
#     ...
#   vocab_mapping/
#     token_freq_<dataset1_name>.pt
#     token_freq_<dataset2_name>.pt
#     ...
#     token_freq_combined.pt
#     d2t.npy
#     t2d.npy
#   checkpoints/
#     <run_name>/
#       0/
#         config.json
#         eagle3.py
#         generation_config.json
#         model.safetensors
#         optimizer_state_dict.pt
#         scheduler_state_dict.pt
#       1/
#         config.json
#         eagle3.py
#         generation_config.json
#         model.safetensors
#         optimizer_state_dict.pt
#         scheduler_state_dict.pt
#       ...
#   logs/
#     <run_name>/


class DataGenArgs(NamedTuple):
    """Arguments for data generation."""

    train_data_path: str
    """The path to the training data. Can be one of ["sharegpt", "ultrachat"] or a huggingface dataset path or a local JSON/JSONL file."""
    dataset_name: str | None = None
    """The name of the dataset to generate data for. Used exclusively for logging and output path generation. If None and train_data_path is sharegpt or ultrachat, the dataset name will be inferred from the train_data_path."""
    max_model_len: int | _NS = _NOTSET
    seq_length: int | _NS = _NOTSET
    max_samples: int | _NS = _NOTSET
    tensor_parallel_size: int | _NS = _NOTSET
    gpu_memory_utilization: float | _NS = _NOTSET
    hf_cache_dir: str | _NS = _NOTSET
    layer_ids: list[int] | _NS = _NOTSET
    batch_size: int | _NS = _NOTSET
    seed: int | _NS = _NOTSET
    start_idx: int | _NS = _NOTSET
    num_preprocessing_workers: int | _NS = _NOTSET


class VocabMappingArgs(NamedTuple):
    draft_vocab_size: int
    target_vocab_size: int


class TrainArgs(NamedTuple):
    run_name: str
    logger: str | _NS = _NOTSET
    lr: float | _NS = _NOTSET
    total_seq_len: int | _NS = _NOTSET
    ttt_steps: int | _NS = _NOTSET
    epochs: int | _NS = _NOTSET
    no_resume_from_checkpoint: bool | _NS = _NOTSET
    num_layers: int | _NS = _NOTSET
    ttt_step_loss_decay: float | _NS = _NOTSET


### END OF SCRIPT ARGUMENTS ###


def prepare_args(args: dict[str, Any]) -> list[str]:
    args_list = []
    for key, value in args.items():
        if value is _NOTSET:
            continue
        # Convert snake_case to kebab-case for command line arguments.
        dashed_key = key.replace("_", "-")
        args_list.append(f"--{dashed_key}")
        args_list.append(str(value))
    return args_list


def print_block(title: str, content: str):
    title = f" {title} "
    term_width, _terminal_height = shutil.get_terminal_size((80, 20))
    print(
        "\n",
        "#" * ((term_width - len(title)) // 2),
        title,
        "#" * ((term_width - len(title) + 1) // 2),
        "\n",
        sep="",
    )
    print(content)
    print("\n", "#" * term_width, "\n", sep="")


def run_script(
    script_name: str,
    script_args: list[str],
    requires: list[str],
    python_alt: str | None = None,
):
    command = [
        "uv",
        "run",
        "--no-sync",
        "--no-dev",
        "--no-default-groups",
        "--isolated",
    ]
    for package in requires[:1]:
        command.append("--with-editable")
        command.append(package)
    for package in requires[1:]:
        command.append("--with")
        command.append(package)

    if python_alt:
        command.extend(python_alt.split())

    script_path = (Path(__file__).parent / script_name).absolute()
    command.append(str(script_path))
    command.extend(script_args)

    print_block(f"RUNNING {script_name}", " ".join(command))

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)  # noqa: S603
        process.wait()
    except KeyboardInterrupt:
        # Clean up subprocesses
        print(
            f"Received KeyboardInterrupt. Terminating process {process.pid} "
            "and its children."
        )
        end_time = time.perf_counter()
        print_block(
            f"CANCELLED {script_name}",
            f"Time taken: {end_time - start_time:.2f} seconds",
        )

        for child in psutil.Process(process.pid).children(recursive=True):
            child.terminate()
        process.terminate()

        for _ in range(10):
            remaining_children = list(
                psutil.Process(process.pid).children(recursive=True)
            )
            if not remaining_children:
                break
            time.sleep(1)
        else:
            print(f"Failed to terminate all children of process {process.pid}.")
            print("Retrying...")
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()  # escalate to SIGKILL
            process.kill()  # escalate to SIGKILL

        sys.exit(1)

    end_time = time.perf_counter()
    print_block(
        f"COMPLETED {script_name}",
        (
            f"Time taken: {end_time - start_time:.2f} seconds. "
            f"Exit code: {process.returncode}"
        ),
    )

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def run_e2e(
    verifier_name_or_path: str,
    output_path: str,
    data_gen_args: DataGenArgs | list[DataGenArgs],
    vocab_mapping_args: VocabMappingArgs,
    train_args: TrainArgs,
):
    """Run the full pipeline in one command."""
    output_path = Path(output_path)

    # Data Generation
    if isinstance(data_gen_args, DataGenArgs):
        data_gen_args = [data_gen_args]

    token_freq_paths = []
    num_datasets = len(data_gen_args)

    for dga_obj in data_gen_args:
        dga_dict = dga_obj._asdict()
        dga_dict["target-model-path"] = verifier_name_or_path

        dataset_name = dga_dict["dataset_name"]
        if dataset_name is None:
            if dga_dict["train_data_path"] in ["sharegpt", "ultrachat"]:
                dataset_name = dga_dict["train_data_path"]
            else:
                raise ValueError(
                    f"Dataset name is required for {dga_dict['train_data_path']}"
                )
        del dga_dict["dataset_name"] # Remove name so it isn't passed as argument to data_generation_offline.py

        token_freq_path = (
            output_path / "vocab_mapping" / f"token_freq_{dataset_name}.pt"
        )
        dga_dict["token-freq-path"] = str(token_freq_path)
        token_freq_paths.append(token_freq_path)
        dga_dict["output-dir"] = str(output_path / "gen" / dataset_name)

        dga_list = prepare_args(dga_dict)
        run_script("data_generation_offline.py", dga_list, [".[datagen]"])

    # Combine token frequency files from all datasets into a single file.
    if num_datasets > 1:
        combined_token_freq_path = (
            output_path / "vocab_mapping" / "token_freq_combined.pt"
        )
        combine_token_frequency_distributions(
            token_freq_paths, combined_token_freq_path
        )
    else:
        combined_token_freq_path = token_freq_paths[0]

    # Vocab Mapping
    vma_dict = vocab_mapping_args._asdict()
    vma_dict["token-freq-path"] = str(combined_token_freq_path)
    vma_dict["output-path"] = str(output_path / "vocab_mapping")
    vma_list = prepare_args(vma_dict)
    run_script("build_vocab_mapping.py", vma_list, [".[datagen]"])

    # Training
    ta_dict = train_args._asdict()
    ta_dict["verifier-name-or-path"] = verifier_name_or_path
    ta_dict["data-path"] = str(output_path / "gen")
    ta_dict["save-path"] = str(output_path / "checkpoints" / train_args.run_name)
    ta_dict["data-format-version"] = 1
    ta_dict["log-dir"] = str(output_path / "logs" / train_args.run_name)
    ta_dict["d2t-path"] = str(output_path / "vocab_mapping" / "d2t.npy")
    ta_dict["t2d-path"] = str(output_path / "vocab_mapping" / "t2d.npy")

    ta_list = prepare_args(ta_dict)

    # Get additional packages to install if loggers are specified.
    packages = ["."]
    loggers = ta_dict["logger"]
    if loggers and loggers is not _NS:
        if isinstance(loggers, str):
            loggers = loggers.split(",")
        loggers = [logger.strip() for logger in loggers]
        packages.extend(loggers)

    run_script(
        "train.py",
        ta_list,
        packages,
        python_alt="torchrun --standalone --nproc_per_node=gpu",
    )
