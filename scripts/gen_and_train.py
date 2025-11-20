#!/usr/bin/env python3
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
from typing import Any

import psutil

_NOT_SET = object()  # Sentinel value for optional arguments that are not set

### SCRIPT ARGUMENTS ###

# Shared
OUTPUT_PATH = "./output"
VERIFIER_MODEL_PATH = "Qwen/Qwen3-8B"
DATA_PATH = f"{OUTPUT_PATH}/gen"
TOKEN_FREQ_PATH = f"{OUTPUT_PATH}/token_freq.pt"
SAVE_PATH = f"{OUTPUT_PATH}/checkpoints"
RUN_NAME = "qwen3_8b_sharegpt"
LOGGERS = "trackio"
TOTAL_SEQ_LEN = 8192
DRAFT_VOCAB_SIZE = 32_000
TARGET_VOCAB_SIZE = 151936
MAX_SAMPLES = 60_000

# Data Generation
data_gen_args = {
    "target-model-path": VERIFIER_MODEL_PATH,
    "train-data-path": "sharegpt",
    "output-dir": DATA_PATH,
    "token-freq-path": TOKEN_FREQ_PATH,
    "max-model-len": TOTAL_SEQ_LEN,
    "seq-length": TOTAL_SEQ_LEN,
    "max-samples": MAX_SAMPLES,
    "tensor-parallel-size": _NOT_SET,
    "gpu-memory-utilization": _NOT_SET,
    "hf-cache-dir": _NOT_SET,
    "layer-ids": _NOT_SET,
    "batch-size": _NOT_SET,
    "seed": _NOT_SET,
    "start-idx": _NOT_SET,
    "num-preprocessing-workers": _NOT_SET,
}

# Vocab Mapping
vocab_mapping_args = {
    "token-freq-path": TOKEN_FREQ_PATH,
    "draft-vocab-size": DRAFT_VOCAB_SIZE,
    "target-vocab-size": TARGET_VOCAB_SIZE,
    "output-path": OUTPUT_PATH,
}

# Training
train_args = {
    "verifier-name-or-path": VERIFIER_MODEL_PATH,
    "data-path": DATA_PATH,
    "save-path": SAVE_PATH,
    "logger": LOGGERS,
    "lr": 3e-5,
    "total-seq-len": TOTAL_SEQ_LEN,
    "data-format-version": 1,
    "run-name": RUN_NAME,
    "d2t-path": f"{OUTPUT_PATH}/d2t.npy",
    "t2d-path": f"{OUTPUT_PATH}/t2d.npy",
    "ttt-steps": 3,
    "epochs": _NOT_SET,
    "no-resume-from-checkpoint": _NOT_SET,
    "log-dir": _NOT_SET,
    "num-layers": _NOT_SET,
    "ttt-step-loss-decay": _NOT_SET,
}

### END OF SCRIPT ARGUMENTS ###


def prepare_args(args: dict[str, Any]) -> list[str]:
    args_list = []
    for key, value in args.items():
        if value is _NOT_SET:
            continue
        args_list.append(f"--{key}")
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


def main():
    data_gen_args_list = prepare_args(data_gen_args)
    run_script("data_generation_offline.py", data_gen_args_list, [".[datagen]"])

    vocab_mapping_args_list = prepare_args(vocab_mapping_args)
    run_script("build_vocab_mapping.py", vocab_mapping_args_list, [".[datagen]"])

    train_args_list = prepare_args(train_args)

    packages = ["."]
    loggers = train_args["logger"]
    if loggers and loggers is not _NOT_SET:
        if isinstance(loggers, str):
            loggers = loggers.split(",")
        loggers = [logger.strip() for logger in loggers]
        packages.extend(loggers)
    run_script(
        "train.py",
        train_args_list,
        packages,
        python_alt="torchrun --standalone --nproc_per_node=gpu",
    )


if __name__ == "__main__":
    main()
