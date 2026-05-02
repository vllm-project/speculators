import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from textwrap import indent

from loguru import logger
from PIL import Image

from speculators.data_generation.preprocessing import load_raw_dataset

__all__ = [
    "SCRIPTS_DIR",
    "VLLM_PYTHON",
    "launch_vllm_server",
    "launch_vllm_server_context",
    "run_data_generation_offline",
    "run_prepare_data",
    "run_training",
    "run_vllm_engine",
    "stop_vllm_server",
    "wait_for_server",
]

VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def wait_for_server(
    port: int,
    timeout: float = 180.0,
    poll_interval: float = 2.0,
    process: subprocess.Popen | None = None,
):
    """Poll vLLM server health endpoint until ready or timeout.

    If *process* is provided, checks whether it has exited between polls
    so that startup failures are reported immediately instead of waiting
    for the full timeout.
    """

    logger.info("Waiting for server")
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited with code {process.returncode} "
                "before becoming ready"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"vLLM server on port {port} not ready after {timeout}s")


def launch_vllm_server(
    model: str,
    port: int,
    hidden_states_path: str,
    *,
    max_model_len: int = 513,
    gpu_memory_utilization: float = 0.5,
    target_layer_ids: list[int] | None = None,
    enforce_eager: bool = True,
    allowed_local_media_path: str | None = None,
) -> subprocess.Popen:
    """Launch a vLLM server configured for hidden-state extraction.

    Returns the server subprocess. Caller is responsible for stopping it
    via stop_vllm_server().
    """
    cmd = [
        VLLM_PYTHON,
        str(SCRIPTS_DIR / "launch_vllm.py"),
        model,
        "--hidden-states-path",
        str(hidden_states_path),
    ]
    if target_layer_ids is not None:
        cmd += ["--target-layer-ids"] + [str(lid) for lid in target_layer_ids]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if allowed_local_media_path:
        cmd += ["--allowed-local-media-path", allowed_local_media_path]
    cmd += [
        "--",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--disable-uvicorn-access-log",
    ]
    logger.info("Starting vLLM server: {}", " ".join(cmd))

    process = subprocess.Popen(cmd)  # noqa: S603

    try:
        wait_for_server(port, process=process)
        logger.info("vLLM server ready on port {}", port)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    return process


def stop_vllm_server(process: subprocess.Popen):
    """Gracefully stop a vLLM server subprocess."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if process.returncode not in (0, -15):  # -15 = SIGTERM (expected)
        logger.error("vLLM server exited with code {}", process.returncode)
    logger.info("vLLM server stopped (exit code {})", process.returncode)


@contextmanager
def launch_vllm_server_context(*args, **kwargs):
    process = launch_vllm_server(*args, **kwargs)
    try:
        yield
    finally:
        stop_vllm_server(process)


def setup_dummy_sharegpt4v_coco(coco_dir: Path):
    """Enable ShareGPT4V to be used without downloading the actual COCO dataset."""
    coco_dir.mkdir(parents=True, exist_ok=True)

    dummy_image = Image.new("RGB", (256, 256))
    dummy_image_path = coco_dir / "dummy.png"
    dummy_image.save(dummy_image_path)

    raw_dataset, normalize_fn = load_raw_dataset("sharegpt4v_coco")

    # Use symlinks to avoid copying the image
    for raw_path in raw_dataset["image"]:
        image_path = coco_dir / raw_path.removeprefix("coco/")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.symlink_to(dummy_image_path)


def run_prepare_data(
    model: str,
    data: str,
    data_path: Path,
    max_samples: int = 50,
    seq_length: int = 512,
    seed: int = 0,
    timeout: float | None = None,
):
    """Tokenize data using prepare_data.py."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "prepare_data.py"),
        "--model",
        model,
        "--data",
        data,
        "--output",
        str(data_path),
        "--max-samples",
        str(max_samples),
        "--seed",
        str(seed),
        "--seq-length",
        str(seq_length),
    ]
    logger.info("Preparing data: {}", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd, check=False, timeout=timeout
    )
    assert result.returncode == 0, "prepare_data.py failed"


def run_data_generation_offline(
    data_path: Path,
    hidden_states_path: Path | None = None,
    port: int = 8321,
    max_samples: int = 50,
    concurrency: int = 4,
    validate_outputs: bool = True,
    timeout: float | None = None,
    fail_on_error: bool = True,
):
    datagen_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "data_generation_offline.py"),
        "--preprocessed-data",
        str(data_path),
        "--endpoint",
        f"http://localhost:{port}/v1",
        "--max-samples",
        str(max_samples),
        "--concurrency",
        str(concurrency),
    ]
    if validate_outputs:
        datagen_cmd.append("--validate-outputs")
    if fail_on_error:
        datagen_cmd.append("--fail-on-error")

    if hidden_states_path is not None:
        datagen_cmd += ["--output", str(hidden_states_path)]

    logger.info("Generating hidden states offline: {}", " ".join(datagen_cmd))
    result = subprocess.run(  # noqa: S603
        datagen_cmd,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, (
        f"data_generation_offline.py failed:\n{result.stderr}"
    )


def run_training(
    model: str,
    data_path: Path,
    save_path: Path,
    seq_length: int = 512,
    port: int = 8321,
    draft_vocab_size: int = 8192,
    epochs: int = 1,
    lr: float = 3e-4,
    online: bool = True,
    hidden_states_path: Path | None = None,
    timeout: float | None = None,
    speculator_type: str = "eagle3",
    extra_train_args: list[str] | None = None,
    target_layer_ids: list[int] | None = None,
    num_layers: int | None = None,
    log_freq: int = 1,
):
    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train.py"),
        "--verifier-name-or-path",
        model,
        "--data-path",
        str(data_path),
        "--vllm-endpoint",
        f"http://localhost:{port}/v1",
        "--save-path",
        str(save_path),
        "--draft-vocab-size",
        str(draft_vocab_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--total-seq-len",
        str(seq_length),
        "--speculator-type",
        speculator_type,
        "--log-freq",
        str(log_freq),
    ]
    if online:
        train_cmd += [
            "--on-missing",
            "generate",
            "--on-generate",
            "delete",
        ]
    else:
        train_cmd += [
            "--on-missing",
            "raise",
        ]
    if hidden_states_path is not None:
        train_cmd += ["--hidden-states-path", str(hidden_states_path)]
    if target_layer_ids is not None:
        train_cmd += ["--target-layer-ids"] + [str(lid) for lid in target_layer_ids]
    if num_layers is not None:
        train_cmd += ["--num-layers", str(num_layers)]
    if extra_train_args:
        train_cmd += extra_train_args

    logger.info("Running training: {}", " ".join(train_cmd))
    result = subprocess.run(  # noqa: S603
        train_cmd,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"


def run_vllm_engine(
    model_path: str,
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
    disable_compile_cache: bool = False,
    max_tokens: int = 50,
    ignore_eos: bool = True,
    acceptance_thresholds: Iterable[float] | None = None,
    timeout: float | None = None,
):
    VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
    logger.info("vLLM Python executable: {}", VLLM_PYTHON)

    run_vllm_file = str(Path(__file__).with_name("run_vllm.py"))
    results_file = str(tmp_path / "results.json")

    command = [
        VLLM_PYTHON,
        run_vllm_file,
        "--sampling-params-args",
        json.dumps(
            {
                "temperature": 0,
                "top_p": 0.9,
                "max_tokens": max_tokens,
                "ignore_eos": ignore_eos,
            }
        ),
        "--llm-args",
        json.dumps(
            {
                "model": model_path,
                "max_model_len": 1024,
                "gpu_memory_utilization": 0.8,
            }
        ),
        "--prompts",
        json.dumps(prompts),
        "--results-file",
        results_file,
    ]
    logger.info("run_vllm.py command:\n    {}", command)

    # Set environment variables for subprocess
    env = os.environ.copy()
    if disable_compile_cache:
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        logger.info("Disabling vLLM compile cache for this test")

    result = subprocess.run(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
        timeout=timeout,
    )
    logger.info("run_vllm.py output:\n{}", indent(result.stdout, "    "))

    returncode = result.returncode
    assert returncode == 0, (
        f"run_vllm.py command exited with non-zero return code: {returncode}"
    )

    with Path(results_file).open(encoding="utf-8") as f:
        results_dict = json.load(f)

    outputs_token_ids = results_dict["outputs"]
    metrics_dict = results_dict["metrics"]
    logger.info("outputs_token_ids: {}", outputs_token_ids)
    logger.info("metrics_dict: {}", metrics_dict)

    for output_token_ids in outputs_token_ids:
        # If max_tokens is 100 or less, make sure the output length is max_tokens
        assert max_tokens > 100 or len(output_token_ids) == max_tokens
        assert all(isinstance(token, int) for token in output_token_ids)

    if acceptance_thresholds is not None:
        for i, thresholdi in enumerate(acceptance_thresholds):
            assert f"acceptance_at_token_{i}" in metrics_dict, (
                f"Acceptance at token {i} is not in metrics_dict"
            )
            acci = metrics_dict[f"acceptance_at_token_{i}"]
            assert acci >= thresholdi, (
                f"Acceptance {acci} at token {i} is less than threshold {thresholdi}"
            )
