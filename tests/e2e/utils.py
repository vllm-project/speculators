import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from textwrap import indent

from loguru import logger

__all__ = [
    "SCRIPTS_DIR",
    "VLLM_PYTHON",
    "launch_vllm_server",
    "prepare_data",
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
    max_model_len: int = 513,
    gpu_memory_utilization: float = 0.5,
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
        hidden_states_path,
        "--",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    logger.info("Starting vLLM server: {}", " ".join(cmd))

    process = subprocess.Popen(cmd)  # noqa: S603

    try:
        wait_for_server(port, process=process)
        logger.info("vLLM server ready on port {}", port)
    except Exception:
        process.terminate()
        process.wait(timeout=30)
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
    logger.info("vLLM server stopped")


def prepare_data(
    model: str, data_path: Path, max_samples: int = 50, seq_length: int = 512
):
    """Tokenize ShareGPT data using prepare_data.py."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "prepare_data.py"),
        "--model",
        model,
        "--data",
        "sharegpt",
        "--output",
        str(data_path),
        "--max-samples",
        str(max_samples),
        "--seq-length",
        str(seq_length),
    ]
    logger.info("Preparing data: {}", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd, stderr=subprocess.PIPE, text=True, check=False
    )
    assert result.returncode == 0, f"prepare_data.py failed:\n{result.stderr}"


def run_vllm_engine(
    model_path: str,
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
    disable_compile_cache: bool = False,
    max_tokens: int = 50,
    ignore_eos: bool = True,
    acceptance_thresholds: Iterable[float] | None = None,
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
