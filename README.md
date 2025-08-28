<div align="center">

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-black.svg" height="64" />
  </picture>

[![License](https://img.shields.io/github/license/neuralmagic/speculators.svg)](https://github.com/neuralmagic/speculators/blob/main/LICENSE) [![Python Versions](https://img.shields.io/badge/Python-3.9--3.13-orange)](https://pypi.python.org/pypi/speculators)

</div>

## Overview

**Speculators** is a unified library for building, evaluating, and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster speculator model to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. Speculators standardizes this process with reusable formats and tools, enabling easier integration and deployment of speculative decoding in production-grade inference servers.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
    <img alt="Speculators user flow diagram" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
  </picture>
</p>

### Key Features

- **Unified Speculative Decoding Toolkit:** Simplifies the development, evaluation, and representation of speculative decoding algorithms, supporting both research and production use cases for LLMs.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.

## Getting Started

### Installation

#### Prerequisites

Before installing, ensure you have the following:

- **Operating System:** Linux or macOS
- **Python:** 3.9 or higher
- **Package Manager:** pip (recommended) or conda

#### Install from PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install speculators
```

#### Install from Source

For the latest development version or to contribute to the project:

```bash
git clone https://github.com/neuralmagic/speculators.git
cd speculators

pip install -e .
```

For development with additional tools:

```bash
pip install -e .[dev]
```

#### Verify Installation

You can verify your installation by checking the version:

```bash
speculators --version
```

Or by importing the package in Python:

```python
import speculators
print(speculators.__version__)
```

## Resources

Here you can find links to our research implementations. These provide prototype code for immediate enablement and experimentation, with plans for productization into the main package soon.

- [eagle3](https://github.com/neuralmagic/speculators/tree/main/research/eagle3): This implementation trains models similar to the EAGLE 3 architecture, specifically utilizing the Train Time Test method.

- [hass](https://github.com/neuralmagic/speculators/tree/main/research/hass): This implementation trains models that are a variation on the EAGLE 1 architecture using the [HASS](https://github.com/HArmonizedSS/HASS) method.

## vLLM Inference

Once in the speculators format, you can serve the speculator using vLLM:

```bash
VLLM_USE_V1=1 vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

Served models can then be benchmarked using [GuideLLM](https://github.com/vllm-project/guidellm). Below, we show sample benchmark results where we compare our speculator with its dense counterpart. We also additionally compare [quantization](https://github.com/vllm-project/llm-compressor) to explore additional performance improvements by swapping the dense verifier, `Qwen/Qwen3-8B` with the quantized FP8 model, [RedHatAI/Qwen3-8B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic) in the `speculator_config`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/qwen_quant_benchmark.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/qwen_quant_benchmark.png" width=180%>
  </picture>
</p>

### Supported Models

The following models are currently supported or are planned to be supported in the short term.

| Model (Verifier)                 | Training via speculators  | Deployment in vLLM | Pretrained Checkpoints |
|----------------------------------|---------------------------|-------------------|-------------------------|
| Llama-3.1-8B-Instruct            | EAGLE-3 ✅ \| HASS ✅     | ✅                | ✅ [EAGLE-3](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
| Llama-3.1-70B-Instruct           | EAGLE-3 ⏳                | ✅                | ✅ [EAGLE-3](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-70B)                     |
| DeepSeek-R1-Distill-LLama-8B     | EAGLE-3 ❌                | ✅                | ✅ [EAGLE-3](https://huggingface.co/yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B)
| Qwen3-8B                         | EAGLE-3 ✅                | ✅                | ❌
| Qwen3-14B                        | EAGLE-3 ❌                | ✅                | ❌
| Qwen3-32B                        | EAGLE-3 ❌                | ✅                | ❌
| Qwen3-30B-A3B                    | EAGLE-3 ❌                | ❌                | ❌
| Qwen3-235B-A22B                  | EAGLE-3 ❌                | ❌                | ✅ [EAGLE-3](https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3)
| Llama-4-Scout-17B-16E-Instruct   | EAGLE-3 ❌                | ❌                | ❌ 
| Llama-4-Maverick-17B-128E-Eagle3 | EAGLE-3 ❌                | ❌                | ✅ [EAGLE-3](https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3)
| DeepSeek-R1                      | EAGLE-3 ❌                | ❌                | ✅ [HASS](https://huggingface.co/HArmonizedSS/HASS-DeepSeek-R1)

✅ = Supported, ⏳ = In Progress, ❌ = Not Yet Supported


## License

Speculators is licensed under the [Apache License 2.0](https://github.com/neuralmagic/speculators/blob/main/LICENSE).

## Cite

If you find Speculators helpful in your research or projects, please consider citing it:

```bibtex
@misc{speculators2025,
  title={Speculators: A Unified Library for Speculative Decoding Algorithms in LLM Serving},
  author={Red Hat},
  year={2025},
  howpublished={\url{https://github.com/neuralmagic/speculators}},
}
```
