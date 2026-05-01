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

    # Use an HF dataset with JSONL files
    modal run --detach modal_speculators.py \
        --hf-dataset inference-optimization/speculators-qwen3-30b-a3b-instruct \
        --hf-dataset-files "*.jsonl"

    # Use custom speculators/vllm branches
    modal run --detach modal_speculators.py \
        --speculators-branch my-feature-branch \
        --vllm-branch main \
        --vllm-repo vllm-project/vllm

    # Enable W&B logging (requires 'wandb' Modal secret)
    modal run --detach modal_speculators.py --wandb

GPU type and count are set via the GPU_TYPE and GPU_COUNT constants at the
top of the script, since Modal requires these at decoration time.

See TrainingConfig below for all configurable parameters.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
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
# Optional Modal secrets — set env vars to attach them to the function.
#   modal secret create wandb WANDB_API_KEY=<key>
#   modal secret create huggingface HF_TOKEN=<token>
# ---------------------------------------------------------------------------
WANDB_ENABLED = os.getenv("WANDB_ENABLED", "0") == "1"
HF_SECRET_ENABLED = os.getenv("HF_SECRET_ENABLED", "0") == "1"


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
    target_layer_ids: str = ""  # space-separated layer IDs, e.g. "2 14 25"

    # GPU layout — GPU_TYPE and GPU_COUNT are set above as module constants.
    vllm_gpus: int = 2  # remainder used for training

    # Data preparation
    dataset: str = "sharegpt"
    max_samples: int = 5000
    seq_length: int = 8192

    # HuggingFace dataset support — download JSONL files from an HF dataset
    # repo and pass them to prepare_data.py via --data <path> for each file.
    hf_dataset: str = ""  # e.g. "inference-optimization/speculators-qwen3-30b-a3b-instruct"
    hf_dataset_files: str = "*.jsonl"  # glob pattern for files to download
    hf_dataset_revision: str = "main"

    # Training hyperparameters
    epochs: int = 5
    lr: float = 1e-4
    total_seq_len: int = 8192
    seed: int = 42
    on_missing: str = "generate"
    on_generate: str = "delete"

    # vLLM install
    vllm_nightly: bool = False  # install nightly instead of stable release
    vllm_version: str = "0.19"  # stable version when not using nightly
    vllm_branch: str = ""  # install from a git branch instead of PyPI
    vllm_repo: str = "vllm-project/vllm"  # GitHub org/repo for branch installs

    # Speculators install
    speculators_branch: str = ""  # install from a git branch instead of PyPI

    # vLLM server
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_data_parallel_size: Optional[int] = None  # defaults to vllm_gpus // tp
    vllm_max_model_len: Optional[int] = None

    # Checkpoint / resume
    no_resume_from_checkpoint: bool = False
    checkpoint_freq: int = 1
    save_best: bool = True

    # Logging
    logger: str = ""  # "trackio", "wandb", "tensorboard", or ""
    run_name: Optional[str] = None
    wandb: bool = False  # enable W&B logging (requires Modal 'wandb' secret)

    # Extra CLI flags (appended verbatim)
    extra_vllm_args: str = ""
    extra_train_args: str = ""

    @property
    def run_id(self) -> str:
        """Deterministic short hash of config fields that define a unique run.
        Same config always produces the same run_id, enabling checkpoint resume."""
        key_fields = {
            "model": self.model,
            "speculator_type": self.speculator_type,
            "draft_arch": self.draft_arch,
            "num_layers": self.num_layers,
            "draft_vocab_size": self.draft_vocab_size,
            "dataset": self.dataset,
            "hf_dataset": self.hf_dataset,
            "seq_length": self.seq_length,
            "max_samples": self.max_samples,
            "total_seq_len": self.total_seq_len,
            "lr": self.lr,
            "seed": self.seed,
        }
        h = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode())
        return h.hexdigest()[:12]

    @property
    def train_gpus(self) -> int:
        return GPU_COUNT - self.vllm_gpus

    @property
    def data_path(self) -> Path:
        return VOLUME_MOUNT / self.run_id / "data"

    @property
    def save_path(self) -> Path:
        return VOLUME_MOUNT / self.run_id / "checkpoints"


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
# Helper: resolve speculators install spec
# ---------------------------------------------------------------------------
def _speculators_install_spec(cfg: TrainingConfig) -> str:
    """Return the pip install specifier for speculators."""
    if cfg.speculators_branch:
        return (
            f"speculators @ git+https://github.com/vllm-project/speculators.git"
            f"@{cfg.speculators_branch}"
        )
    return "speculators>=0.5.0"


# ---------------------------------------------------------------------------
# Helper: resolve vllm install packages
# ---------------------------------------------------------------------------
def _vllm_install_packages(cfg: TrainingConfig) -> list[str]:
    """Return the list of pip install args for vllm."""
    if cfg.vllm_nightly:
        return [
            "vllm",
            "--extra-index-url", "https://wheels.vllm.ai/nightly/cu130",
        ]
    if cfg.vllm_branch:
        return [
            f"vllm @ git+https://github.com/{cfg.vllm_repo}.git"
            f"@{cfg.vllm_branch}"
        ]
    return [f"vllm=={cfg.vllm_version}"]


# ---------------------------------------------------------------------------
# Helper: clone speculators repo for scripts (respects branch override)
# ---------------------------------------------------------------------------
def _ensure_speculators_scripts(cfg: TrainingConfig) -> None:
    """Re-clone the speculators repo if a custom branch is requested."""
    if not cfg.speculators_branch:
        return  # use the version cloned at image build time
    import shutil
    if SPECULATORS_REPO.exists():
        shutil.rmtree(SPECULATORS_REPO)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", cfg.speculators_branch,
         "https://github.com/vllm-project/speculators.git",
         str(SPECULATORS_REPO)],
        check=True,
    )
    print(f"[modal] Cloned speculators scripts from branch: {cfg.speculators_branch}")


# ---------------------------------------------------------------------------
# Helper: download HF dataset files
# ---------------------------------------------------------------------------
def _download_hf_dataset(cfg: TrainingConfig) -> list[str]:
    """Download JSONL files from an HF dataset repo. Returns local file paths."""
    import fnmatch
    import urllib.request

    download_dir = Path("/tmp/hf_dataset")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Use the HF Hub API to list files
    api_url = (
        f"https://huggingface.co/api/datasets/{cfg.hf_dataset}"
        f"/tree/{cfg.hf_dataset_revision}"
    )
    req = urllib.request.Request(api_url)
    # Pass HF token if available
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")

    with urllib.request.urlopen(req) as resp:
        files_meta = json.loads(resp.read())

    # Filter files matching the glob pattern
    matched = [
        f["rfilename"] for f in files_meta
        if f.get("type") == "file"
        and fnmatch.fnmatch(f["rfilename"], cfg.hf_dataset_files)
    ]
    if not matched:
        raise ValueError(
            f"No files matching '{cfg.hf_dataset_files}' found in "
            f"dataset {cfg.hf_dataset}"
        )

    print(f"[modal] Downloading {len(matched)} files from {cfg.hf_dataset}:")
    local_paths = []
    for fname in matched:
        url = (
            f"https://huggingface.co/datasets/{cfg.hf_dataset}"
            f"/resolve/{cfg.hf_dataset_revision}/{fname}"
        )
        local_path = download_dir / fname
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            print(f"  [cached] {fname}")
        else:
            print(f"  [downloading] {fname}")
            req = urllib.request.Request(url)
            if hf_token:
                req.add_header("Authorization", f"Bearer {hf_token}")
            urllib.request.urlretrieve(url, local_path)
        local_paths.append(str(local_path))

    return local_paths


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

    script = SPECULATORS_REPO / "scripts" / "prepare_data.py"
    cmd = [
        f"{speculators_venv}/bin/python", str(script),
        "--model", cfg.model,
        "--output", str(cfg.data_path),
        "--seq-length", str(cfg.seq_length),
    ]

    # Determine data source(s)
    if cfg.hf_dataset:
        print(f"[modal] Downloading HF dataset: {cfg.hf_dataset}")
        local_files = _download_hf_dataset(cfg)
        for f in local_files:
            cmd += ["--data", f]
        print(f"[modal] Preparing data from {len(local_files)} files")
    else:
        cmd += ["--data", cfg.dataset]
        print(f"[modal] Preparing data with dataset={cfg.dataset}")

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
    env["VLLM_USE_DEEP_GEMM"] = "0"

    # launch_vllm.py wraps `vllm serve` with hidden-state extraction flags.
    script = SPECULATORS_REPO / "scripts" / "launch_vllm.py"
    cmd = [
        f"{vllm_venv}/bin/python", str(script),
        cfg.model,
    ]
    if cfg.target_layer_ids:
        cmd += ["--target-layer-ids"] + cfg.target_layer_ids.split()
    cmd += [
        "--",
        "--port", str(cfg.vllm_port),
        "--gpu-memory-utilization", str(cfg.vllm_gpu_memory_utilization),
        "--disable-uvicorn-access-log",
    ]
    # Compute TP and DP to use all vLLM GPUs.
    tp = cfg.vllm_tensor_parallel_size
    if cfg.vllm_data_parallel_size is not None:
        dp = cfg.vllm_data_parallel_size
    else:
        if cfg.vllm_gpus % tp != 0:
            raise ValueError(
                f"vllm_gpus ({cfg.vllm_gpus}) must be divisible by "
                f"vllm_tensor_parallel_size ({tp})"
            )
        dp = cfg.vllm_gpus // tp
    if tp * dp != cfg.vllm_gpus:
        raise ValueError(
            f"tp ({tp}) * dp ({dp}) = {tp * dp} does not match "
            f"vllm_gpus ({cfg.vllm_gpus})"
        )
    cmd += ["--tensor-parallel-size", str(tp)]
    if dp > 1:
        cmd += ["--data-parallel-size", str(dp)]
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
        "--log-freq", "100",
    ]
    if cfg.draft_vocab_size is not None:
        cmd += ["--draft-vocab-size", str(cfg.draft_vocab_size)]
    if cfg.draft_arch:
        cmd += ["--draft-arch", cfg.draft_arch]
    if cfg.target_layer_ids:
        cmd += ["--target-layer-ids"] + cfg.target_layer_ids.split()
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
    secrets=[
        s for s in [
            modal.Secret.from_name("wandb") if WANDB_ENABLED else None,
            modal.Secret.from_name("huggingface") if HF_SECRET_ENABLED else None,
        ] if s is not None
    ],
)
def train_speculators(cfg_dict: dict, skip_data_prep: bool = False) -> None:
    cfg = TrainingConfig(**cfg_dict)

    print(f"[modal] Run ID:  {cfg.run_id}")
    print(f"[modal] Model:   {cfg.model}")
    print(f"[modal] Data:    {cfg.data_path}")
    print(f"[modal] Ckpts:   {cfg.save_path}")
    print(f"[modal] GPU layout: {cfg.vllm_gpus} for vLLM, {cfg.train_gpus} for training")

    # If a custom speculators branch was requested, re-clone the repo
    _ensure_speculators_scripts(cfg)

    print(f"[modal] Setting up environments...")

    # Create isolated venvs.
    vllm_packages = _vllm_install_packages(cfg)
    vllm_venv = _create_venv("vllm", vllm_packages)

    speculators_spec = _speculators_install_spec(cfg)
    speculators_packages = [speculators_spec]
    # Install logger backends — they aren't speculators dependencies
    logger_packages = {
        "wandb": "wandb",
        "trackio": "trackio",
        "tensorboard": "tensorboard",
    }
    for logger_name in cfg.logger.split(",") if cfg.logger else []:
        pkg = logger_packages.get(logger_name.strip())
        if pkg:
            speculators_packages.append(pkg)
    if cfg.wandb and "wandb" not in speculators_packages:
        speculators_packages.append("wandb")
    speculators_venv = _create_venv("speculators", speculators_packages)

    # Configure W&B if enabled
    if cfg.wandb:
        if not os.environ.get("WANDB_API_KEY"):
            print(
                "[modal] WARNING: --wandb enabled but WANDB_API_KEY not found. "
                "Create a Modal secret: modal secret create wandb WANDB_API_KEY=<key>"
            )

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
    target_layer_ids: str = "",
    # GPU layout
    vllm_gpus: int = 2,
    # Data
    dataset: str = "sharegpt",
    max_samples: int = 5000,
    seq_length: int = 8192,
    skip_data_prep: bool = False,
    # HF dataset
    hf_dataset: str = "",
    hf_dataset_files: str = "*.jsonl",
    hf_dataset_revision: str = "main",
    # Training
    epochs: int = 5,
    lr: float = 1e-4,
    total_seq_len: int = 8192,
    on_missing: str = "generate",
    on_generate: str = "delete",
    seed: int = 42,
    # vLLM
    vllm_nightly: bool = False,
    vllm_version: str = "0.19",
    vllm_branch: str = "",
    vllm_repo: str = "vllm-project/vllm",
    vllm_port: int = 8000,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_tensor_parallel_size: int = 1,
    vllm_data_parallel_size: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    # Speculators
    speculators_branch: str = "",
    # Checkpoint
    no_resume_from_checkpoint: bool = False,
    checkpoint_freq: int = 1,
    save_best: bool = True,
    # Logging
    logger: str = "",
    run_name: Optional[str] = None,
    wandb: bool = False,
    # Extra
    extra_vllm_args: str = "",
    extra_train_args: str = "",
):
    cfg = TrainingConfig(
        model=model,
        speculator_type=speculator_type,
        draft_arch=draft_arch,
        num_layers=num_layers,
        draft_vocab_size=draft_vocab_size,
        target_layer_ids=target_layer_ids,
        vllm_gpus=vllm_gpus,
        vllm_nightly=vllm_nightly,
        vllm_version=vllm_version,
        vllm_branch=vllm_branch,
        vllm_repo=vllm_repo,
        dataset=dataset,
        max_samples=max_samples,
        seq_length=seq_length,
        hf_dataset=hf_dataset,
        hf_dataset_files=hf_dataset_files,
        hf_dataset_revision=hf_dataset_revision,
        epochs=epochs,
        lr=lr,
        total_seq_len=total_seq_len,
        on_missing=on_missing,
        on_generate=on_generate,
        seed=seed,
        speculators_branch=speculators_branch,
        vllm_port=vllm_port,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_data_parallel_size=vllm_data_parallel_size,
        vllm_max_model_len=vllm_max_model_len,
        no_resume_from_checkpoint=no_resume_from_checkpoint,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
        logger=logger,
        run_name=run_name,
        wandb=wandb,
        extra_vllm_args=extra_vllm_args,
        extra_train_args=extra_train_args,
    )

    # Auto-set logger when --wandb is used
    if cfg.wandb and not cfg.logger:
        cfg.logger = "wandb"

    print(f"[modal] Launching speculators training")
    print(f"  Run ID:      {cfg.run_id}")
    print(f"  Model:       {cfg.model}")
    print(f"  GPUs:        {GPU_COUNT}x {GPU_TYPE} ({cfg.vllm_gpus} vLLM + {cfg.train_gpus} training)")
    print(f"  Epochs:      {cfg.epochs}")
    print(f"  LR:          {cfg.lr}")
    if cfg.hf_dataset:
        print(f"  HF Dataset:  {cfg.hf_dataset} ({cfg.hf_dataset_files})")
    if cfg.speculators_branch:
        print(f"  Speculators: branch={cfg.speculators_branch}")
    if cfg.vllm_branch:
        print(f"  vLLM:        branch={cfg.vllm_branch} (repo={cfg.vllm_repo})")
    if cfg.wandb:
        print(f"  W&B:         enabled")

    train_speculators.remote(dataclasses.asdict(cfg), skip_data_prep=skip_data_prep)
