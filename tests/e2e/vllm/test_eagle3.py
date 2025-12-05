import pytest

from speculators.convert.eagle.eagle3_converter import Eagle3Converter
from tests.e2e.vllm.utils import run_vllm_engine


class TestEagle3vLLM:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_info",
        [
            pytest.param(
                {
                    "unconverted_model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                },
                id="llama3-8b",
            ),
            pytest.param(
                {
                    "unconverted_model": "nm-testing/Speculator-Qwen3-8B-Eagle3",
                    "base_model": "Qwen/Qwen3-8B",
                    "norm_before_residual": True,
                },
                id="qwen3-8b",
            ),
            pytest.param(
                {
                    "unconverted_model": (
                        "nm-testing/random-weights-llama3.1.8b-2layer-eagle3-unconverted"
                    ),
                    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "norm_before_residual": True,
                    "disable_compile_cache": True,
                },
                id="llama3-2layer",
            ),
        ],
    )
    def test_convert_run_vllm_engine_eagle3(
        self, model_info, temp_cache_dir, prompts, tmp_path
    ):
        unconverted_model = model_info.get("unconverted_model")
        base_model = model_info.get("base_model")
        norm_before_residual = model_info.get("norm_before_residual", False)
        disable_compile_cache = model_info.get("disable_compile_cache", False)
        converted_path = tmp_path / unconverted_model.split("/")[-1]
        converter = Eagle3Converter()

        convert_kwargs = {
            "input_path": unconverted_model,
            "output_path": converted_path,
            "base_model": base_model,
            "cache_dir": temp_cache_dir,
            "norm_before_residual": norm_before_residual,
        }

        # including eagle_aux_hidden_state_layer_ids
        #  if specified
        if "eagle_aux_hidden_state_layer_ids" in model_info:
            convert_kwargs["eagle_aux_hidden_state_layer_ids"] = model_info[
                "eagle_aux_hidden_state_layer_ids"
            ]

        converter.convert(**convert_kwargs)
        run_vllm_engine(
            model_path=str(converted_path),
            tmp_path=tmp_path,
            disable_compile_cache=disable_compile_cache,
            prompts=prompts,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "model_path",
        [
            pytest.param(
                "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
                id="llama3-converted-quantized",
            ),
            pytest.param(
                "RedHatAI/Qwen3-8B-speculator.eagle3",
                id="qwen3-converted-quantized",
            ),
        ],
    )
    def test_vllm_engine_eagle3(self, model_path, prompts, tmp_path):
        run_vllm_engine(model_path=model_path, tmp_path=tmp_path, prompts=prompts)
