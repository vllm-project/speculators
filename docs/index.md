<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px; text-align: left;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" height="64" />
  </picture>
</div>

Speculators is a unified library for building, training and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster draft model (i.e "the speculator") to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. The speculator intelligently drafts multiple tokens ahead of time, and the base model verifies them in a single forward pass. This approach boosts performance without sacrificing output quality, as every accepted token is guaranteed to match what the main model would have generated on its own.

Speculators standardizes this process by providing a productionized end-to-end framework to train draft models with reusable formats and tools. Trained models can seamlessly run in vLLM, enabling the deployment of speculative decoding in production-grade inference servers.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
    <img alt="Speculators user flow diagram" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
  </picture>
</p>

## Key Features

- **Offline Training Data Generation using vLLM:** Enable the generation of hidden states using vLLM. Data samples are saved to disk and can be used for draft model training.
- **Draft Model Training Support:** E2E training support of single and multi-layer draft models. Training is supported for MoE, non-MoE and Vision Language mdoels.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories into a standard speculators format for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.

## Utility Scripts
- [Evaluate your trained model using vLLM and GuideLLM](https://github.com/vllm-project/speculators/tree/main/examples/evaluate/eval-guidellm)
- [Regenerate responses to enhance your training data](https://github.com/vllm-project/speculators/tree/main/scripts/response_regeneration)

## Quick Start

To try out a speculative decoding model you can get started by running a pre-made one with vLLM. After [installing vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/), run:

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

(Or choose another model from the [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models) collection.)

Behind the scenes, this is reading the model from Hugging Face, parsing the `speculators_config` and setting up both the speculator and verifier models to run together.

To create a speculative decoding model for a different verifier model there are two approaches you can choose:

1. Train a new speculative decoding model ([instructions](train.md))([examples](examples/data_generation_and_training.md)).
2. Convert an existing model from a third-party library to the Speculators format for easy deployment with vLLM ([instructions](convert.md)) ([examples](examples/convert.md)).
