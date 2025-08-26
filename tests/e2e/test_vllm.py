import pytest
import torch
from loguru import logger

try:
    from vllm import LLM, SamplingParams  # type: ignore

    vllm_installed = True
except ImportError:
    vllm_installed = False
    logger.warning("vllm is not installed. This test will be skipped")


@pytest.mark.skipif(not vllm_installed, reason="vLLM is not installed, skipping test")
class TestEagle3vLLM:
    def setup_method(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        sampling_params = SamplingParams(temperature=0.80, top_p=0.95)
        llm = LLM(model=model_path, gpu_memory_utilization=0.9)
        return llm.generate(self.prompts, sampling_params)

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
