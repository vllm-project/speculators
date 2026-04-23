# Speculators Documentation

<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px; text-align: left;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" height="64" />
  </picture>
</div>

## Overview

Speculators is a unified library for building, training and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster draft model (i.e "the speculator") to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality.

The speculator intelligently drafts multiple tokens ahead of time, and the base model verifies them in a single forward pass. This approach boosts performance without sacrificing output quality, as every accepted token is guaranteed to come from the same distribution as using the main model on its own.

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
- **Draft Model Training Support:** E2E training support of single and multi-layer draft models. Training is supported for MoE, non-MoE, and Vision Language models.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories into a standard speculators format for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.

!!! tip
    Read more about Speculators features in this [vLLM blog post](https://blog.vllm.ai/2025/12/13/speculators-v030.html).

## Quick Start

To try out a speculative decoding model, you can get started by running a pre-trained one with vLLM. After [installing vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/), run:

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

(Or choose another model from the [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models) collection.)

Behind the scenes, this reads the model from Hugging Face, parses the `speculators_config`, and sets up both the speculator and verifier models to run together.

To create a speculative decoding model for a different verifier model, there are two approaches you can choose:

1. **Train a new speculative decoding model** - See [Getting Started](user_guide/getting_started.md) and [Tutorials](user_guide/tutorials/index.md).
2. **Convert an existing model** from a third-party library to the Speculators format for easy deployment with vLLM - See [Features](user_guide/features.md).

## Supported Models

The following table summarizes the models that have been trained end-to-end by our team:

<table>
<thead>
<tr>
<th>Verifier Architecture</th>
<th>Verifier Size</th>
<th>Training Support</th>
<th>vLLM Deployment Support</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Llama</td>
<td>8B-Instruct</td>
<td><a href="https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>70B-Instruct</td>
<td><a href="https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3">Qwen3</td>
<td>8B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>14B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-14B-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>32B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td rowspan="2">gpt-oss</td>
<td>20b</td>
<td><a href="https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>120b</td>
<td><a href="https://huggingface.co/RedHatAI/gpt-oss-120b-speculator.eagle3">
      Eagle-3
    </a> ✅</td>
<td>✅</td>
</tr>
<tr>
  <td rowspan="3">Qwen3 MoE</td>
  <td>30B-Instruct</td>
  <td><a href="https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3">
      Eagle-3
    </a> ✅</td>
  <td>✅</td>
</tr>
<tr>
  <td>235B-Instruct</td>
  <td>
    <a href="https://huggingface.co/RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3">
      Eagle-3
    </a> ✅
  </td>
  <td>✅</td>
</tr>
<tr>
  <td>235B</td>
  <td><a href="https://huggingface.co/RedHatAI/Qwen3-235B-A22B-speculator.eagle3">
      Eagle-3
    </a> ✅</td>
  <td>✅</td>
</tr>
<td>Qwen3-VL</td>
<td>235B-A22B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-VL-235B-A22B-Instruct-speculator.eagle3">
      Eagle-3
    </a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>Mistral 3 Large</td>
<td>675B-Instruct</td>
<td>Eagle-3 ⏳</td>
<td>⏳</td>
</tr>
<tr>
<td>Gemma 4</td>
<td>31B-it</td>
<td><a href="https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3">Eagle-3</a> ✅<br/><a href="https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dflash">DFlash</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>Gemma 4 MoE</td>
<td>26B-A4B-it</td>
<td><a href="https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3">Eagle-3</a> ✅</td>
<td>✅</td>
</tr>
</tbody>
</table>

✅ = Supported, ⏳ = In Progress, ❌ = Not Yet Supported

## Installation

### Prerequisites

Before installing, ensure you have the following:

- **Operating System:** Linux or macOS
- **Python:** 3.10 or higher
- **Package Manager:** pip (recommended) or conda

### Install from PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install speculators
```

### Install from Source

For the latest development version or to contribute to the project:

```bash
git clone https://github.com/vllm-project/speculators.git
cd speculators

pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

## Community & Support

💬 Join us on the [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) and share your questions, thoughts, or ideas in:

- `#speculators`
- `#feat-spec-decode`

🎥 Watch our Office Hours presentation: [Video](https://www.youtube.com/live/2ISAr_JVGLs) | [Slides](https://docs.google.com/presentation/d/1s4eAb7v-rdZt8smyULBJWGXjJXrgFTZWnwqYa2-h1l4/edit?slide=id.g3365e070742_6_0#slide=id.g3365e070742_6_0)

For more community resources, see the [Community page](community/index.md).

## Next Steps

- [User Guide](user_guide/index.md) - Learn how to use Speculators
- [Getting Started](user_guide/getting_started.md) - Quick start guide for training models
- [Tutorials](user_guide/tutorials/index.md) - Step-by-step walkthroughs
- [API Reference](api/index.md) - Python API documentation
- [CLI Reference](cli/index.md) - Command-line tools documentation

## License

Speculators is licensed under the [Apache License 2.0](https://github.com/vllm-project/speculators/blob/main/LICENSE).

## Citation

If you find Speculators helpful in your research or projects, please consider citing it:

```bibtex
@misc{speculators2025,
  title={Speculators: A Unified Library for Speculative Decoding Algorithms in LLM Serving},
  author={Red Hat},
  year={2025},
  howpublished={\url{https://github.com/vllm-project/speculators}},
}
```
