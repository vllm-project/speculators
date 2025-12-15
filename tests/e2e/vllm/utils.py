import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from textwrap import indent

from loguru import logger

__all__ = ["run_vllm_engine"]


def run_vllm_engine(
    model_path: str,
    tmp_path: Path,
    prompts: list[str],
    disable_compile_cache: bool = False,
    max_tokens: int = 20,
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
        json.dumps({"temperature": 0.8, "top_p": 0.95, "max_tokens": max_tokens}),
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
        for i, threshold in enumerate(acceptance_thresholds):
            assert f"acceptance_at_token_{i}" in metrics_dict, (
                f"Acceptance at token {i} is not in metrics_dict"
            )
            assert metrics_dict[f"acceptance_at_token_{i}"] >= threshold, (
                f"Acceptance at token {i} is less than threshold {threshold}"
            )
