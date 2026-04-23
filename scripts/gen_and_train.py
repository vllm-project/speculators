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

import enum
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import torch

from speculators.train.vocab_mapping import (
    combine_token_frequency_distributions,
)
from speculators.utils.util import is_npu_available


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
#     0/
#       config.json
#       eagle3.py
#       generation_config.json
#       model.safetensors
#       optimizer_state_dict.pt
#       scheduler_state_dict.pt
#     1/
#       config.json
#       eagle3.py
#       generation_config.json
#       model.safetensors
#       optimizer_state_dict.pt
#       scheduler_state_dict.pt
#     ...
#   logs/


class DataGenArgs(NamedTuple):
    """Arguments for data generation."""

    train_data_path: str
    """The path to the training data. Can be one of ["sharegpt", "ultrachat"] or a
 huggingface dataset path or a local JSON/JSONL file."""
    dataset_name: str | None = None
    """The name of the dataset to generate data for. Used exclusively for logging and
 output path generation. If None and train_data_path is sharegpt or ultrachat, the
 dataset name will be inferred from the train_data_path."""
    turn_dropout: bool = False
    multimodal: bool = False
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
    draft_arch: str | _NS = _NOTSET
    draft_intermediate_size: int | _NS = _NOTSET
    draft_vocab_size: int | _NS = _NOTSET
    speculator_type: str | _NS = _NOTSET
    target_layer_ids: list[int] | _NS = _NOTSET
    mask_token_id: int | _NS = _NOTSET
    block_size: int | _NS = _NOTSET
    max_anchors: int | _NS = _NOTSET
    ttt_step_loss_decay: float | _NS = _NOTSET
    use_off_policy_tokens: bool | _NS = _NOTSET
    scheduler_type: str | _NS = _NOTSET
    scheduler_warmup_steps: int | _NS = _NOTSET
    scheduler_total_steps: int | _NS = _NOTSET
    scheduler_num_cosine_cycles: float | _NS = _NOTSET
    norm_before_fc: bool | _NS = _NOTSET


### END OF SCRIPT ARGUMENTS ###


def prepare_args(args: dict[str, Any]) -> list[str]:
    args_list = []
    for key, value in args.items():
        if value is _NOTSET:
            continue
        # Convert snake_case to kebab-case for command line arguments.
        dashed_key = key.replace("_", "-")
        # Handle boolean flags (action="store_true")
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{dashed_key}")
            # If False, don't add the flag at all
        elif isinstance(value, (list, tuple)):
            if value:
                args_list.append(f"--{dashed_key}")
                args_list.extend(str(item) for item in value)
        else:
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
    python_alt: str = "python",
    use_uv: bool = True,
):
    command = []
    if use_uv:
        command = [
            "uv",
            "run",
            "--no-sync",
            "--no-dev",
            "--no-default-groups",
            "--isolated",
        ]
        for i, package in enumerate(requires):
            command.append("--with-editable" if i == 0 else "--with")
            command.append(package)

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


def _infer_dataset_name(dga_dict: dict[str, Any]) -> str:
    dataset_name = dga_dict["dataset_name"]
    if dataset_name is not None:
        return dataset_name
    if dga_dict["train_data_path"] in ["sharegpt", "ultrachat", "llava-instruct"]:
        return dga_dict["train_data_path"]
    raise ValueError(f"Dataset name is required for {dga_dict['train_data_path']}")



def run_e2e(
    verifier_name_or_path: str,
    output_path: str,
    data_gen_args: DataGenArgs | list[DataGenArgs],
    vocab_mapping_args: VocabMappingArgs | None,
    train_args: TrainArgs,
):
    """Run the full pipeline in one command."""
    output_path = Path(output_path)
    (output_path / "gen").mkdir(parents=True, exist_ok=True)
    (output_path / "vocab_mapping").mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)

    # Data Generation
    if isinstance(data_gen_args, DataGenArgs):
        data_gen_args = [data_gen_args]

    token_freq_paths = []
    num_datasets = len(data_gen_args)
    uses_multimodal = any(dga.multimodal for dga in data_gen_args)
    if uses_multimodal and num_datasets != 1:
        raise ValueError(
            "Multimodal E2E currently supports a single dataset per run. "
            "Pass one DataGenArgs with train_data_path='llava-instruct' or another "
            "single multimodal dataset."
        )

    train_data_path = output_path / "gen"

    # When LOCAL_DATAGEN_ENV is set (or running on NPU), reuse the currently
    # activated Python environment for all data-generation sub-steps instead of
    # spawning an isolated `uv` venv. This avoids re-resolving/re-installing
    # heavy extras such as `vllm` from `.[datagen]` on every run, and keeps
    # data-gen aligned with the same vLLM build that serves hidden states.
    local_datagen_env = is_npu_available() or bool(
        os.environ.get("LOCAL_DATAGEN_ENV", "")
    )

    for dga_obj in data_gen_args:
        dga_dict = dga_obj._asdict()
        multimodal = bool(dga_dict.pop("multimodal", False))

        dataset_name = _infer_dataset_name(dga_dict)
        del dga_dict["dataset_name"]

        token_freq_path = (
            output_path / "vocab_mapping" / f"token_freq_{dataset_name}.pt"
        )
        token_freq_paths.append(token_freq_path)

        dataset_output_dir = output_path / "gen" / dataset_name

        if multimodal:
            prepare_args_dict = {
                "model": verifier_name_or_path,
                "data": dga_dict["train_data_path"],
                "seq_length": dga_dict["seq_length"],
                "max_samples": dga_dict["max_samples"],
                "token_freq_path": str(token_freq_path),
                "turn_dropout": dga_dict["turn_dropout"],
                "output": str(dataset_output_dir),
                "seed": dga_dict["seed"],
                "num_preprocessing_workers": dga_dict["num_preprocessing_workers"],
                "multimodal": True,
            }
            run_script(
                "prepare_data.py",
                prepare_args(prepare_args_dict),
                [".[datagen]"],
                use_uv=not local_datagen_env,
            )

            offline2_args = {
                "model": verifier_name_or_path,
                "preprocessed_data": str(dataset_output_dir),
                "output": str(dataset_output_dir / "hidden_states"),
                "layer_ids": dga_dict["layer_ids"],
                "max_samples": dga_dict["max_samples"],
                "start_idx": dga_dict["start_idx"],
            }
            run_script(
                "data_generation_offline2.py",
                prepare_args(offline2_args),
                [".[datagen]"],
                use_uv=not local_datagen_env,
            )
            train_data_path = dataset_output_dir
        else:
            dga_dict["target-model-path"] = verifier_name_or_path
            dga_dict["token-freq-path"] = str(token_freq_path)
            dga_dict["output-dir"] = str(dataset_output_dir)
            dga_list = prepare_args(dga_dict)
            run_script(
                "data_generation_offline.py",
                dga_list,
                [".[datagen]"],
                use_uv=not local_datagen_env,
            )

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

    # Vocab Mapping (optional)
    ta_dict = {
        **train_args._asdict(),
        "verifier-name-or-path": verifier_name_or_path,
        "data-path": str(train_data_path),
        "save-path": str(output_path / "checkpoints"),
        "log-dir": str(output_path / "logs"),
    }
    if uses_multimodal:
        ta_dict["multimodal"] = True
        # Be explicit about where train.py should look for hidden-states even
        # though `ArrowDataset` would default to `<data-path>/hidden_states`.
        # Relying on the default silently desynchronizes from
        # `data_generation_offline2.py --output` if that default ever changes.
        ta_dict["hidden-states-path"] = str(train_data_path / "hidden_states")
    if vocab_mapping_args is not None:
        vma_dict = vocab_mapping_args._asdict()
        vma_dict["token-freq-path"] = str(combined_token_freq_path)
        vma_dict["output-path"] = str(output_path / "vocab_mapping")
        vma_list = prepare_args(vma_dict)
        run_script(
            "build_vocab_mapping.py",
            vma_list,
            [".[datagen]"],
            use_uv=not local_datagen_env,
        )
        ta_dict["d2t-path"] = str(output_path / "vocab_mapping" / "d2t.npy")
        ta_dict["t2d-path"] = str(output_path / "vocab_mapping" / "t2d.npy")

    ta_list = prepare_args(ta_dict)
    if not uses_multimodal:
        ta_list.append("--legacy-data")

    # Get additional packages to install if loggers are specified.
    packages = ["."]
    loggers = ta_dict["logger"]
    if loggers and loggers is not _NOTSET:
        if isinstance(loggers, str):
            loggers = loggers.split(",")
        loggers = [logger.strip() for logger in loggers]
        packages.extend(loggers)
    device_count = torch.accelerator.device_count()

    # LOCAL_TRAIN_ENV implies LOCAL_DATAGEN_ENV semantics for the training step;
    # additionally honor LOCAL_DATAGEN_ENV so users that opt into the local env
    # for data-gen can keep the same env for training without setting both.
    local_train_env = (
        local_datagen_env or bool(os.environ.get("LOCAL_TRAIN_ENV", ""))
    )

    run_script(
        "train.py",
        ta_list,
        packages,
        python_alt=f"torchrun --standalone --nproc_per_node={device_count}",
        use_uv=not local_train_env,
    )
