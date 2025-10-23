import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent

import pytest
from loguru import logger

from speculators.convert.eagle.eagle3_converter import Eagle3Converter


class TestEagle3vLLM:
    def setup_method(self):
        self.prompts = [
            "The capital of France is",
            "The president of the US is",
            "My name is",
        ]

    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir(exist_ok=True)
        monkeypatch.setenv("HF_HOME", str(cache_dir))
        return cache_dir

    def _run_vllm_engine(self, model_path: str, tmp_path: Path):
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
            json.dumps(self.prompts),
            "--results-file",
            results_file,
        ]
        logger.info("run_vllm.py command:\n    {}", command)

        result = subprocess.run(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
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

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_info",
        [
            {
                "unconverted_model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            },
            {
                "unconverted_model": "nm-testing/Speculator-Qwen3-8B-Eagle3",
                "base_model": "Qwen/Qwen3-8B",
                "norm_before_residual": True,
            },
        ],
    )
    def test_convert_run_vllm_engine_eagle3(self, model_info, temp_cache_dir, tmp_path):
        unconverted_model = model_info.get("unconverted_model")
        base_model = model_info.get("base_model")
        norm_before_residual = model_info.get("norm_before_residual", False)
        converted_path = tmp_path / unconverted_model.split("/")[-1]
        converter = Eagle3Converter()

        converter.convert(
            input_path=unconverted_model,
            output_path=converted_path,
            base_model=base_model,
            cache_dir=temp_cache_dir,
            norm_before_residual=norm_before_residual,
        )
        self._run_vllm_engine(model_path=str(converted_path), tmp_path=tmp_path)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_path",
        [
            "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
            "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized",
        ],
    )
    def test_vllm_engine_eagle3(self, model_path, tmp_path):
        self._run_vllm_engine(model_path=model_path, tmp_path=tmp_path)
