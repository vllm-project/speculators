# Getting Started

## What is Speculative Decoding?

Speculative decoding is a technique that speeds up LLM inference without changing the output. A small, fast **draft model** (the "speculator") proposes several tokens at once, and the larger **target model** verifies them in a single forward pass. Accepted tokens are guaranteed to come from the same distribution as what the target model would have produced on its own, so the speedup is lossless.

## Serve a Speculator with vLLM

The fastest way to try speculative decoding is to serve a pre-trained speculator with [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/):

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

vLLM reads from the model's config, loads both the speculator and the target model, and enables speculative decoding automatically. See the [Serve in vLLM](tutorials/serve_vllm.md) tutorial for more options.

## Pre-trained Speculator Models

Browse the full collection of ready-to-use speculator models on Hugging Face:

**[RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)**

For a list of models that have been trained and validated end-to-end by our team, see the [supported models table](../index.md#supported-models).

## Supported Algorithms

Speculators supports multiple speculative decoding algorithms:

- **[Eagle-3](algorithms/eagle3.md)**
- **[DFlash](algorithms/dflash.md)**

For help choosing between them, see the [Algorithm Decision Guide](algorithms/decision_guide.md).

## Train Your Own Speculator

If a pre-trained speculator isn't available for your target model, you can train one with Speculators. Training requires internal hidden states from the target model, which are extracted by serving the target model with vLLM. The library supports both online and offline training modes:

- **Online training** -- Hidden states are generated on-the-fly during training. Easier to get started, lower disk usage.
- **Offline training** -- Hidden states are pre-generated and cached.

### Tutorials

- [Train Eagle-3 Online](tutorials/train_eagle3_online.md) -- Recommended starting point
- [Train Eagle-3 Offline](tutorials/train_eagle3_offline.md)
- [Train DFlash Online](tutorials/train_dflash_online.md) -- Starting point for DFlash
- [Evaluating Model Performance](tutorials/evaluating_performance.md) -- Benchmark your trained speculator
- [Response Regeneration](tutorials/response_regeneration.md) -- Improve training data quality

See all tutorials in the [Tutorials](tutorials/index.md) section.
