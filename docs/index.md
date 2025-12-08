# Speculators

<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px; text-align: left;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-model-icon-blue.png">
    <img alt="Speculators Logo" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-model-icon-blue.png" width="120">
  </picture>

  <h3 style="margin: 0; text-align: left;">
    Unified Library for Speculative Decoding: Build and Deploy Faster LLM Inference
  </h3>
</div>

**Speculators** is a unified library for building and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster speculator model to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. Speculators standardizes this process with reusable formats and tools, enabling easier integration and deployment of speculative decoding in production-grade inference servers.

## Key Features

- **Unified Speculative Decoding Toolkit:** Simplifies the development and representation of speculative decoding algorithms, supporting both research and production use cases for LLMs.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.
- **Research-to-Production Pipeline:** Bridge the gap between experimental speculative decoding research and production deployments.

## Quick Start

To try out a speculative decoding model you can get started by running a pre-made one with vLLM. After [installing vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/), run:
```bash
VLLM_USE_V1=1 vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```
(Or choose another model from the [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models) collection.)

Behind the scenes, this is reading the model from Hugging Face, parsing the `speculators_config` and setting up both the speculator and verifier models to run together.

To create a speculative decoding model for a different verifier model there are two approaches you can choose:
1. Train a new speculative decoding model ([instructions](train/index.md))([examples](examples/data_generation_and_training.md)).
2. Convert an existing model from a third-party library to the Speculators format for easy deployment with vLLM ([instructions](convert/index.md)) ([examples](examples/convert.md)).
