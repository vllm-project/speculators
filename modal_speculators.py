# ---
# cmd: ["modal", "run", "--detach", "modal_speculators.py"]
# ---
"""
Modal script for speculators (EAGLE3) online training.

Spins up a single node with multiple GPUs, launches vLLM on a subset
for hidden-state extraction, and trains a speculator on the remaining GPUs.

The speculators training scripts (prepare_data.py, launch_vllm.py, train.py)
are standalone files in the speculators repo — they are NOT exposed via the
`speculators` CLI entry point. This script clones the repo at build time and
invokes the scripts directly.

Usage:
    # Prepare data, then train (full pipeline)
    modal run --detach modal_speculators.py

    # Train only (data already prepared)
    modal run --detach modal_speculators.py --skip-data-prep

    # Custom config
    modal run --detach modal_speculators.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vllm-gpus 4 \
        --epochs 10

GPU type and count are set via the GPU_TYPE and GPU_COUNT constants at the
top of the script, since Modal requires these at decoration time.

See TrainingConfig below for all configurable parameters.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Persistent volume for datasets, hidden states, and checkpoints
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("speculators-training", create_if_missing=True)
VOLUME_MOUNT = Path("/vol")

# Path where the speculators repo is cloned inside the container
SPECULATORS_REPO = Path("/opt/speculators")
# Pin to a specific release tag for reproducible builds.
SPECULATORS_VERSION = "v0.5.0"

# ---------------------------------------------------------------------------
# Container image — clone the speculators repo for its standalone scripts,
# and install uv for creating isolated venvs at runtime.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"git clone --depth 1 --branch {SPECULATORS_VERSION}"
        f" https://github.com/vllm-project/speculators.git {SPECULATORS_REPO}",
    )
    .env({"PATH": "/root/.local/bin:$PATH"})
)

app = modal.App("speculators-training", image=image)


# ---------------------------------------------------------------------------
# GPU configuration — must be set before running the script, since Modal
# requires GPU specs at decoration time (not configurable via CLI args).
# ---------------------------------------------------------------------------
GPU_TYPE = "H100"
GPU_COUNT = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TrainingConfig:
    # Model
    model: str = "Qwen/Qwen3-8B"
    speculator_type: str = "eagle3"
    draft_arch: str = "llama"
    num_layers: int = 1
    draft_vocab_size: int = 32000

    # GPU layout — GPU_TYPE and GPU_COUNT are set above as module constants.
    vllm_gpus: int = 2  # remainder used for training

    # Data preparation
    dataset: str = "sharegpt"
    max_samples: int = 5000
    seq_length: int = 8192

    # Training hyperparameters
    epochs: int = 5
    lr: float = 1e-4
    total_seq_len: int = 8192
    seed: int = 42
    on_missing: str = "generate"
    on_generate: str = "delete"

    # vLLM install
    vllm_nightly: bool = False  # install nightly instead of stable release

    # vLLM server
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.9
    vllm_data_parallel_size: Optional[int] = None
    vllm_tensor_parallel_size: Optional[int] = None
    vllm_max_model_len: Optional[int] = None

    # Checkpoint / resume
    no_resume_from_checkpoint: bool = False
    checkpoint_freq: int = 1
    save_best: bool = True

    # Logging
    logger: str = ""  # "trackio", "wandb", "tensorboard", or ""
    run_name: Optional[str] = None

    # Extra CLI flags (appended verbatim)
    extra_vllm_args: str = ""
    extra_train_args: str = ""

    # Paths (relative to volume mount)
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    @property
    def train_gpus(self) -> int:
        return GPU_COUNT - self.vllm_gpus

    @property
    def data_path(self) -> Path:
        return VOLUME_MOUNT / self.data_dir

    @property
    def save_path(self) -> Path:
        return VOLUME_MOUNT / self.checkpoint_dir


# ---------------------------------------------------------------------------
# Helper: create a uv venv and install packages
# ---------------------------------------------------------------------------
def _create_venv(
    name: str,
    packages: list[str],
    torch_backend: str = "cu129",
) -> str:
    """Create a uv virtual environment and return the path to its python."""
    venv_dir = f"/tmp/{name}_venv"
    subprocess.run(["uv", "venv", venv_dir, "--python", "3.12"], check=True)
    python = f"{venv_dir}/bin/python"
    subprocess.run(
        ["uv", "pip", "install", f"--torch-backend={torch_backend}",
         "--python", python, *packages],
        check=True,
    )
    return venv_dir



# ---------------------------------------------------------------------------
# Helper: wait for vLLM to be ready
# ---------------------------------------------------------------------------
def _wait_for_vllm(port: int, timeout: int = 600) -> None:
    """Poll the vLLM health endpoint until it responds."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            print(f"[modal] vLLM is ready on port {port}")
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(5)
    raise TimeoutError(f"vLLM did not start within {timeout}s")


# ---------------------------------------------------------------------------
# Stage 1: Data preparation
# ---------------------------------------------------------------------------
def prepare_data(cfg: TrainingConfig, speculators_venv: str) -> None:
    if (cfg.data_path / "dataset_info.json").exists():
        print("[modal] Data already prepared, skipping.")
        return

    print(f"[modal] Preparing data with model={cfg.model}, dataset={cfg.dataset}")
    script = SPECULATORS_REPO / "scripts" / "prepare_data.py"
    cmd = [
        f"{speculators_venv}/bin/python", str(script),
        "--model", cfg.model,
        "--data", cfg.dataset,
        "--output", str(cfg.data_path),
        "--seq-length", str(cfg.seq_length),
    ]
    if cfg.max_samples:
        cmd += ["--max-samples", str(cfg.max_samples)]
    subprocess.run(cmd, check=True)
    print("[modal] Data preparation complete.")


# ---------------------------------------------------------------------------
# Stage 2: Launch vLLM server (background process)
# ---------------------------------------------------------------------------
def launch_vllm(cfg: TrainingConfig, vllm_venv: str) -> subprocess.Popen:
    vllm_gpu_ids = ",".join(str(i) for i in range(cfg.vllm_gpus))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = vllm_gpu_ids

    # launch_vllm.py wraps `vllm serve` with hidden-state extraction flags.
    script = SPECULATORS_REPO / "scripts" / "launch_vllm.py"
    cmd = [
        f"{vllm_venv}/bin/python", str(script),
        cfg.model,
        "--",
        "--port", str(cfg.vllm_port),
        "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization),
    ]
    if cfg.vllm_data_parallel_size is not None:
        cmd += ["--data-parallel-size", str(cfg.vllm_data_parallel_size)]
    if cfg.vllm_tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(cfg.vllm_tensor_parallel_size)]
    if cfg.vllm_max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.vllm_max_model_len)]
    if cfg.extra_vllm_args:
        cmd += cfg.extra_vllm_args.split()

    print(f"[modal] Launching vLLM on GPUs {vllm_gpu_ids}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


# ---------------------------------------------------------------------------
# Stage 3: Training
# ---------------------------------------------------------------------------
def run_training(cfg: TrainingConfig, speculators_venv: str) -> None:
    train_gpu_start = cfg.vllm_gpus
    train_gpu_ids = ",".join(
        str(i) for i in range(train_gpu_start, train_gpu_start + cfg.train_gpus)
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = train_gpu_ids

    script = str(SPECULATORS_REPO / "scripts" / "train.py")

    if cfg.train_gpus > 1:
        # Multi-GPU training via torchrun + FSDP
        cmd = [
            f"{speculators_venv}/bin/torchrun",
            "--standalone",
            "--nproc_per_node", str(cfg.train_gpus),
            script,
        ]
    else:
        cmd = [f"{speculators_venv}/bin/python", script]

    cmd += [
        "--verifier-name-or-path", cfg.model,
        "--speculator-type", cfg.speculator_type,
        "--num-layers", str(cfg.num_layers),
        "--data-path", str(cfg.data_path),
        "--vllm-endpoint", f"http://localhost:{cfg.vllm_port}/v1",
        "--save-path", str(cfg.save_path),
        "--epochs", str(cfg.epochs),
        "--lr", str(cfg.lr),
        "--total-seq-len", str(cfg.total_seq_len),
        "--on-missing", cfg.on_missing,
        "--on-generate", cfg.on_generate,
        "--seed", str(cfg.seed),
        "--checkpoint-freq", str(cfg.checkpoint_freq),
    ]
    if cfg.draft_vocab_size is not None:
        cmd += ["--draft-vocab-size", str(cfg.draft_vocab_size)]
    if cfg.draft_arch:
        cmd += ["--draft-arch", cfg.draft_arch]
    if cfg.no_resume_from_checkpoint:
        cmd.append("--no-resume-from-checkpoint")
    if cfg.save_best:
        cmd.append("--save-best")
    if cfg.logger:
        cmd += ["--logger", cfg.logger]
    if cfg.run_name:
        cmd += ["--run-name", cfg.run_name]
    if cfg.extra_train_args:
        cmd += cfg.extra_train_args.split()

    print(f"[modal] Starting training on GPUs {train_gpu_ids}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    print("[modal] Training complete.")


# ---------------------------------------------------------------------------
# Modal function: full pipeline
# ---------------------------------------------------------------------------
@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=86400,  # 24 hours
)
def train_speculators(cfg_dict: dict, skip_data_prep: bool = False) -> None:
    cfg = TrainingConfig(**cfg_dict)

    print(f"[modal] Setting up environments...")
    print(f"[modal] GPU layout: {cfg.vllm_gpus} for vLLM, {cfg.train_gpus} for training")

    # Create isolated venvs.
    if cfg.vllm_nightly:
        vllm_venv = _create_venv("vllm", [
            "vllm",
            "--extra-index-url", "https://wheels.vllm.ai/nightly/cu130",
        ])
    else:
        vllm_venv = _create_venv("vllm", ["vllm==0.18.1"])
    speculators_venv = _create_venv("speculators", ["speculators>=0.5.0"])

    # Stage 1: Data preparation (runs on CPU, uses speculators venv)
    if not skip_data_prep:
        prepare_data(cfg, speculators_venv)
        volume.commit()

    # Stage 2: Launch vLLM server in background
    vllm_proc = launch_vllm(cfg, vllm_venv)
    try:
        _wait_for_vllm(cfg.vllm_port)

        # Stage 3: Train
        run_training(cfg, speculators_venv)
    finally:
        # Shut down vLLM
        print("[modal] Shutting down vLLM server...")
        vllm_proc.send_signal(signal.SIGTERM)
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()

    # Persist results
    volume.commit()
    print("[modal] All artifacts saved to volume 'speculators-training'.")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    # Model
    model: str = "Qwen/Qwen3-8B",
    speculator_type: str = "eagle3",
    draft_arch: str = "llama",
    num_layers: int = 1,
    draft_vocab_size: int = 32000,
    # GPU layout
    vllm_gpus: int = 2,
    # Data
    dataset: str = "sharegpt",
    max_samples: int = 5000,
    seq_length: int = 8192,
    skip_data_prep: bool = False,
    # Training
    epochs: int = 5,
    lr: float = 1e-4,
    total_seq_len: int = 8192,
    on_missing: str = "generate",
    on_generate: str = "delete",
    seed: int = 42,
    # vLLM
    vllm_nightly: bool = False,
    vllm_port: int = 8000,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_data_parallel_size: Optional[int] = None,
    vllm_tensor_parallel_size: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    # Checkpoint
    no_resume_from_checkpoint: bool = False,
    checkpoint_freq: int = 1,
    save_best: bool = True,
    # Logging
    logger: str = "",
    run_name: Optional[str] = None,
    # Extra
    extra_vllm_args: str = "",
    extra_train_args: str = "",
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
):
    cfg = TrainingConfig(
        model=model,
        speculator_type=speculator_type,
        draft_arch=draft_arch,
        num_layers=num_layers,
        draft_vocab_size=draft_vocab_size,
        vllm_gpus=vllm_gpus,
        vllm_nightly=vllm_nightly,
        dataset=dataset,
        max_samples=max_samples,
        seq_length=seq_length,
        epochs=epochs,
        lr=lr,
        total_seq_len=total_seq_len,
        on_missing=on_missing,
        on_generate=on_generate,
        seed=seed,
        vllm_port=vllm_port,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_data_parallel_size=vllm_data_parallel_size,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_model_len=vllm_max_model_len,
        no_resume_from_checkpoint=no_resume_from_checkpoint,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
        logger=logger,
        run_name=run_name,
        extra_vllm_args=extra_vllm_args,
        extra_train_args=extra_train_args,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"[modal] Launching speculators training")
    print(f"  Model:       {cfg.model}")
    print(f"  GPUs:        {GPU_COUNT}x {GPU_TYPE} ({cfg.vllm_gpus} vLLM + {cfg.train_gpus} training)")
    print(f"  Epochs:      {cfg.epochs}")
    print(f"  LR:          {cfg.lr}")

    train_speculators.remote(dataclasses.asdict(cfg), skip_data_prep=skip_data_prep)
