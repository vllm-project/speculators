import pytest
from loguru import logger

from speculators.convert.eagle.eagle3_converter import Eagle3Converter


@pytest.mark.importorskip("vllm", reason="vLLM is not installed")
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
        monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_dir))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_dir))
        return cache_dir

    def _run_vllm_engine(self, model_path):
        from vllm import LLM, SamplingParams  # type: ignore

        sampling_params = SamplingParams(temperature=0.80, top_p=0.95, max_tokens=20)
        llm = LLM(model=model_path, max_model_len=1024, gpu_memory_utilization=0.8)
        return llm.generate(self.prompts, sampling_params)

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
        output = self._run_vllm_engine(model_path=str(converted_path))
        logger.info(output)
        assert output

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_path",
        [
            "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
            "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized",
        ],
    )
    def test_vllm_engine_eagle3(self, model_path):
        output = self._run_vllm_engine(model_path=model_path)
        logger.info(output)
        assert output
