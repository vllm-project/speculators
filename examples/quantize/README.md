# Quantizing a trained speculator

[`ptq_rtn_drafter.py`](ptq_rtn_drafter.py) quantizes the weights of a trained speculator (draft model) with [compressed-tensors](https://github.com/vllm-project/compressed-tensors) and saves it in compressed-tensors format. vLLM reads the drafter's own `quantization_config` and serves it with no extra flags.

It is calibration-free round-to-nearest (min/max scales): no data, no GPTQ/AWQ. Quantizing the 2B-parameter DSpark drafter takes 30 seconds to 3 minutes on one A10 GPU, mostly reading and writing weights.

```bash
pip install "compressed-tensors>=0.20.1"

python examples/quantize/ptq_rtn_drafter.py \
    --model RedHatAI/Qwen3.8-27B-speculator.dspark \
    --scheme FP8_DYNAMIC \
    --output Qwen3.8-27B-speculator.dspark-FP8_DYNAMIC

vllm serve Qwen/Qwen3.8-27B \
    --speculative-config '{"model": "Qwen3.8-27B-speculator.dspark-FP8_DYNAMIC", "num_speculative_tokens": 7, "method": "dspark"}'
```

`--scheme` takes any compressed-tensors preset name. Tested:

| Scheme        | Weights                                           | Activations            | Size (DSpark, Qwen3.8-27B) |
| ------------- | ------------------------------------------------- | ---------------------- | -------------------------- |
| (bf16)        |                                                   |                        | 4.0 GB                     |
| `FP8_DYNAMIC` | FP8 E4M3, per channel                             | FP8, dynamic per token | 2.1 GB                     |
| `W4A16`       | INT4 symmetric, group 128                         | bf16                   | 1.2 GB                     |
| `NVFP4A16`    | FP4 E2M1, FP8 group-16 scales × FP32 global scale | bf16                   | 1.3 GB                     |
| `MXFP4A16`    | FP4 E2M1, E8M0 group-32 scales                    | bf16                   | 1.2 GB                     |

## What is quantized

- **Quantized**: every 2-D `*_proj` and `fc` weight, i.e. all attention and MLP projections of the draft layers plus the input fusion layer (1.86B of the DSpark drafter's 1.99B parameters).
- **Kept in the original dtype**: norms, `embed_tokens`, `lm_head`, and the DSpark `markov_head` / `confidence_head`.

The drafter is not a `transformers.PreTrainedModel`, so the script rebuilds a plain `nn.Module` skeleton from the checkpoint's tensor names instead of instantiating the speculators model class. The output keeps the original parameter names, which is what vLLM's speculator loaders expect. Two details matter for vLLM:

- The `ignore` list uses regexes (`re:.*markov_head.*`, ...). vLLM prefixes the drafter's module names with `model.`, so exact names from the checkpoint would not match.
- vLLM fuses `q/k/v_proj` and `gate/up_proj` into one layer and keeps one NVFP4 weight global scale per fused layer. The script gives each fused group a shared global scale (the minimum of its members, as llm-compressor does); otherwise vLLM silently mis-scales all but one member.

Besides the weights and `config.json`, the output directory holds `ptq_rtn_stats.json`, with the per-layer-type relative weight error (FP8 2.7%, INT4 11%, NVFP4 9.5%, MXFP4 11% for the DSpark drafter), and `quantize_command.txt`, with the command line and package versions.

## Results: DSpark drafter for Qwen3.8-27B

Acceptance length (AL) on the Spec-Bench subsets of [RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks) (80 prompts each), bf16 `Qwen/Qwen3.8-27B` verifier, vLLM 0.33.0, 1× H20, 7 speculative tokens, greedy decoding, thinking enabled, `max_tokens` 4096, measured with `scripts/evaluate/evaluate.py throughput`.

| Subset              | bf16 | FP8_DYNAMIC | W4A16 | NVFP4A16 | MXFP4A16 |
| ------------------- | ---: | ----------: | ----: | -------: | -------: |
| question (MT-Bench) | 3.73 |        3.71 |  3.79 |     3.78 |     3.68 |
| translation         | 4.44 |        4.41 |  4.34 |     4.37 |     4.45 |
| summarization       | 3.83 |        3.86 |  3.80 |     3.84 |     3.79 |
| qa                  | 4.16 |        4.04 |  4.07 |     4.08 |     4.08 |
| math_reasoning      | 5.36 |        5.41 |  5.35 |     5.37 |     5.42 |
| rag                 | 4.59 |        4.65 |  4.64 |     4.61 |     4.64 |
| **mean**            | 4.35 |        4.35 |  4.33 |     4.34 |     4.34 |

Paired per-prompt difference in output tokens per verify step against bf16 (n = 480 prompts, 95% CI): FP8 −0.017 ± 0.030, W4A16 −0.021 ± 0.034, NVFP4A16 −0.005 ± 0.033, MXFP4A16 −0.026 ± 0.028. Every interval includes zero, and sampled decoding (temperature 1.0, top-p 0.95, top-k 20) gives the same picture. Round-to-nearest weight quantization down to 4 bits costs no measurable acceptance on this drafter.

On Hopper the gain is memory rather than speed: the drafter's linear layers take about 1 ms of an 18.6 ms DSpark step at one request, and with closed-loop load from 1 to 64 concurrent requests the FP8 drafter stayed within about ±2% of bf16 in output tokens per second per user, while W4A16 (Machete kernels) was neutral up to 4 requests and 1.3–4.6% slower from 8 (noise floor between two bf16 servers: ±0.5%). The freed memory goes to the KV cache (+2.8% for FP8, +4.1% for W4A16). H20 has no FP4 tensor cores, so vLLM runs the FP4 checkpoints weight-only; they were not tested on Blackwell.

Quantized checkpoints produced with this script:

- [TeenSpirit/Qwen3.8-27B-speculator.dspark-FP8_DYNAMIC](https://huggingface.co/TeenSpirit/Qwen3.8-27B-speculator.dspark-FP8_DYNAMIC)
- [TeenSpirit/Qwen3.8-27B-speculator.dspark-W4A16](https://huggingface.co/TeenSpirit/Qwen3.8-27B-speculator.dspark-W4A16)
- [TeenSpirit/Qwen3.8-27B-speculator.dspark-NVFP4A16](https://huggingface.co/TeenSpirit/Qwen3.8-27B-speculator.dspark-NVFP4A16)
- [TeenSpirit/Qwen3.8-27B-speculator.dspark-MXFP4A16](https://huggingface.co/TeenSpirit/Qwen3.8-27B-speculator.dspark-MXFP4A16)
