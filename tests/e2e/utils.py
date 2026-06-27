import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import wraps
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
    "purge_newfiles",
    "run_data_generation_offline",
    "run_gemma4_kv_extraction",
    "run_prepare_data",
    "run_stitch_mtp",
    "run_training",
    "run_vllm_engine",
    "stop_vllm_server",
    "wait_for_server",
]

GEMMA4_KV_KEYS = (
    "kv_last_local_k",
    "kv_last_local_v",
    "kv_last_global_k",
    "kv_last_global_v",
)


def purge_newfiles(fn: Callable[..., Path]):
    """Decorator that turns a Path-returning function into a context manager.

    On exit, deletes top-level files in the resolved directory whose mtime is
    newer than when the wrapped function returned.  Does not recurse into
    subdirectories.  This prevents generated artifacts (e.g. ``d2t.npy``,
    ``t2d.npy`` potentially cached by ``train.py``) from persisting in
    shared directories (such as the HF snapshot cache) between test runs.
    """

    @wraps(fn)
    @contextmanager
    def wrapper(*args, **kwargs):
        path = fn(*args, **kwargs)
        cutoff = time.time()
        try:
            yield path
        finally:
            if path.is_dir():
                for f in path.iterdir():
                    if f.is_file() and f.stat().st_mtime > cutoff:
                        f.unlink()
                        logger.info("Purged generated artifact: {}", f.name)

    return wrapper


VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def wait_for_server(
    port: int,
    timeout: float = 600.0,
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
    enforce_eager: bool = False,
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
    if allowed_local_media_path is not None:
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

        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.symlink_to(dummy_image_path)


def run_prepare_data(
    model: str,
    data: str,
    data_path: Path,
    max_samples: int = 50,
    seq_length: int = 512,
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
        datagen_cmd, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout
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
        train_cmd, stderr=subprocess.PIPE, text=True, check=False, timeout=timeout
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"


def run_stitch_mtp(
    finetuned_checkpoint: Path,
    verifier_path: str,
    output_path: Path,
    timeout: float | None = None,
):
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "stitch_mtp.py"),
        str(finetuned_checkpoint),
        verifier_path,
        "--output-path",
        str(output_path),
    ]
    logger.info("Stitching MTP weights: {}", " ".join(cmd))
    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )
    assert result.returncode == 0, f"stitch_mtp.py failed:\n{result.stderr}"


def run_vllm_engine(
    model_path: str,
    tmp_path: Path,
    prompts: list[list[dict[str, str]]],
    max_model_len: int = 1024,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = False,
    allowed_local_media_path: str | None = None,
    speculative_config: dict | None = None,
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

    sampling_params_dict = {
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
    }

    llm_args_dict = {
        "model": model_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if allowed_local_media_path is not None:
        llm_args_dict["allowed_local_media_path"] = allowed_local_media_path
    if speculative_config is not None:
        llm_args_dict["speculative_config"] = speculative_config

    command = [
        VLLM_PYTHON,
        run_vllm_file,
        "--sampling-params-args",
        json.dumps(sampling_params_dict),
        "--llm-args",
        json.dumps(llm_args_dict),
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


def run_gemma4_kv_extraction(
    model: str,
    tmp_path: Path,
    tensor_parallel_size: int = 1,
    aux_layer_ids: Iterable[int] = (2, 21, 39, 42),
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.4,
    timeout: float | None = None,
):
    """Drive Gemma4KVConnector extraction via run_gemma4_kv_extraction.py."""
    logger.info("vLLM Python executable: {}", VLLM_PYTHON)

    driver_file = str(Path(__file__).with_name("run_gemma4_kv_extraction.py"))
    shared_storage = tmp_path / "kv_out"
    shared_storage.mkdir(exist_ok=True)
    results_file = tmp_path / "results.json"
    aux_layer_ids = list(aux_layer_ids)

    llm_args_dict = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "speculative_config": {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": aux_layer_ids}
            },
        },
        "kv_transfer_config": {
            "kv_connector": "Gemma4KVConnector",
            "kv_connector_module_path": (
                "speculators.data_generation.gemma4_kv_connector"
            ),
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(shared_storage)},
        },
        "max_model_len": max_model_len,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "gpu_memory_utilization": gpu_memory_utilization,
        "load_format": "dummy",
    }

    # Short, and a long prompt past the 512 sliding window, in one batch.
    prompts = [
        "Hello world",
        "Test prompt with several tokens",
        " ".join(["word"] * 600),
    ]

    command = [
        VLLM_PYTHON,
        driver_file,
        "--llm-args",
        json.dumps(llm_args_dict),
        "--prompts",
        json.dumps(prompts),
        "--results-file",
        str(results_file),
    ]
    logger.info("run_gemma4_kv_extraction.py command:\n    {}", command)

    env = os.environ.copy()
    # Avoid a flashinfer JIT-build failure.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    result = subprocess.run(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
        timeout=timeout,
    )
    logger.info(
        "run_gemma4_kv_extraction.py output:\n{}", indent(result.stdout, "    ")
    )
    assert result.returncode == 0, (
        f"run_gemma4_kv_extraction.py exited with non-zero return code: "
        f"{result.returncode}"
    )

    with results_file.open(encoding="utf-8") as f:
        results = json.load(f)

    assert results["num_outputs"] == len(prompts)
    assert len(results["per_request"]) == len(prompts)
    expected_heads = {
        "kv_last_local_k": results["local_total_heads"],
        "kv_last_local_v": results["local_total_heads"],
        "kv_last_global_k": results["global_total_heads"],
        "kv_last_global_v": results["global_total_heads"],
    }
    hidden_size = results["hidden_size"]
    for req in results["per_request"]:
        path = req.get("hidden_states_path")
        assert path is not None, "request produced no hidden_states_path"
        n = req["num_tokens"]
        assert req["token_ids_aligned"], "saved token_ids do not match prompt tokens"
        assert req["hidden_states_shape"] == [n, len(aux_layer_ids), hidden_size]
        for key in GEMMA4_KV_KEYS:
            kv = req["kv"][key]
            assert kv["present"], f"missing {key}"
            shape = kv["shape"]
            assert len(shape) == 3, f"{key} expected 3-D, got {shape}"
            assert shape[0] == n, f"{key} rows {shape[0]} != n_tokens {n}"
            # full (unsharded) head count even under TP
            assert shape[1] == expected_heads[key], (
                f"{key} head count {shape[1]} != unsharded {expected_heads[key]} "
                f"(tp={tensor_parallel_size})"
            )
            assert kv["finite"], f"{key} has non-finite values"
        # local (sliding) and global (full) caches use different head dims.
        local_shape = req["kv"]["kv_last_local_k"]["shape"]
        global_shape = req["kv"]["kv_last_global_k"]["shape"]
        assert local_shape[1:] != global_shape[1:]

    return results
