<div align="center">

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-black.svg" height="64" />
  </picture>

[![License](https://img.shields.io/github/license/neuralmagic/speculators.svg)](https://github.com/neuralmagic/speculators/blob/main/LICENSE) [![Python Versions](https://img.shields.io/badge/Python-3.10--3.13-orange)](https://pypi.python.org/pypi/speculators) [![docs](https://img.shields.io/badge/docs-Speculators-blue)](https://docs.vllm.ai/projects/speculators/en/latest/) [![PyPI](https://img.shields.io/pypi/v/speculators.svg)](https://pypi.org/project/speculators/) [![tests](https://github.com/vllm-project/speculators/actions/workflows/main.yml/badge.svg)](https://github.com/vllm-project/speculators/actions/workflows/main.yml)

</div>

## Overview

**Speculators** is a unified library for building and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster draft model (i.e "the speculators") to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. The speculator intelligently drafts multiple tokens ahead of time, and the base model verifies them in a single forward pass. This approach boosts performance without sacrificing output quality, as every accepted token is guaranteed to match what the main model would have generated on its own. Speculators standardizes this process with reusable formats and tools, enabling easier integration and deployment of speculative decoding in production-grade inference servers.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
    <img alt="Speculators user flow diagram" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
  </picture>
</p>

## Key Features

- **Unified Speculative Decoding Toolkit:** Simplifies the development and representation of speculative decoding algorithms, supporting both research and production use cases for LLMs.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories into a standard speculators format for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.
- **Coming Soon:** The ability to train speculators directly through the speculators repository

## Supported Models

The following models are currently supported or are planned to be supported in the short term.

<table>
<thead>
<tr>
<th>Verifier Architecture</th>
<th>Verifier Size</th>
<th>Training via Speculators</th>
<th>Deployment in vLLM</th>
<th>Conversion of External Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Llama</td>
<td>8B-Instruct</td>
<td><a href="https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td><a href="https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B">EAGLE-3</a> ✅</td>
</tr>
<tr>
<td>70B-Instruct</td>
<td><a href="https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td><a href="https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B">EAGLE-3</a> ✅</td>
</tr>
<tr>
<td>DeepSeek-R1-Distill-LLama-8B</td>
<td>EAGLE-3 ❌</td>
<td>✅</td>
<td><a href="https://huggingface.co/yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B">EAGLE-3</a> ✅</td>
</tr>
<tr>
<td rowspan="3">Qwen3</td>
<td>8B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr>
<td>14B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-14B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr>
<td>32B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr>
<td>Qwen3 MoE</td>
<td>235B-A22B</td>
<td>EAGLE-3 ⏳</td>
<td>⏳</td>
<td><a href="https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3">EAGLE-3</a> ⏳</td>
</tr>
<tr>
<td>Llama-4</td>
<td>Maverick-17B-128E-Eagle3</td>
<td>EAGLE-3 ❌</td>
<td>✅</td>
<td><a href="https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3">EAGLE-3</a> ✅</td>
</tr>
<tr>
<td rowspan="2">gpt-oss</td>
<td>20b</td>
<td><a href="https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
<td>❌</td>
</tr>
<tr>
<td>120b</td>
<td>EAGLE-3 ❌</td>
<td>⏳</td>
<td><a href="https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-v2">EAGLE-3</a> ⏳</td>
</tr>
<tr>
<td>Qwen3-VL</td>
<td>235B-A22B</td>
<td>EAGLE-3 ⏳</td>
<td>⏳</td>
<td>❌</td>
</tr>
</tbody>
</table>

✅ = Supported, ⏳ = In Progress, ❌ = Not Yet Supported

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

## Getting Started

### Installation

#### Prerequisites

Before installing, ensure you have the following:

- **Operating System:** Linux or macOS
- **Python:** 3.10 or higher
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
pip install -e ".[dev]"
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

- eagle3: This implementation trains models similar to the EAGLE 3 architecture, specifically utilizing the Train Time Test method.

- hass: This implementation trains models that are a variation on the EAGLE 1 architecture using the [HASS](https://github.com/HArmonizedSS/HASS) method.

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
