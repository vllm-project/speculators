import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent

from loguru import logger

__all__ = ["run_vllm_engine"]


def run_vllm_engine(
    model_path: str,
    tmp_path: Path,
    prompts: list[str],
    disable_compile_cache: bool = False,
):
    VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
    logger.info("vLLM Python executable: {}", VLLM_PYTHON)

    run_vllm_file = str(Path(__file__).with_name("run_vllm.py"))
    results_file = str(tmp_path / "outputs_token_ids.json")

    command = [
        VLLM_PYTHON,
        run_vllm_file,
        "--sampling-params-args",
        json.dumps({"temperature": 0.8, "top_p": 0.95, "max_tokens": 20}),
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
        outputs_token_ids = json.load(f)
    logger.info("outputs_token_ids: {}", outputs_token_ids)

    for output_token_ids in outputs_token_ids:
        assert len(output_token_ids) == 20
        assert all(isinstance(token, int) for token in output_token_ids)
