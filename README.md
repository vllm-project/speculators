<div align="center">

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-white.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" />
    <img alt="Speculators logo" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-logo-black.svg" height="64" />
  </picture>

[![License](https://img.shields.io/github/license/vllm-project/speculators.svg)](https://github.com/vllm-project/speculators/blob/main/LICENSE) [![Python Versions](https://img.shields.io/badge/Python-3.10--3.13-orange)](https://pypi.org/project/speculators/) [![docs](https://img.shields.io/badge/docs-Speculators-blue)](https://docs.vllm.ai/projects/speculators/en/latest/) [![PyPI](https://img.shields.io/pypi/v/speculators.svg)](https://pypi.org/project/speculators/) [![tests](https://github.com/vllm-project/speculators/actions/workflows/main.yml/badge.svg)](https://github.com/vllm-project/speculators/actions/workflows/main.yml)

</div>

## Overview

Speculators is a unified library for building, training and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster draft model (i.e "the speculator") to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. The speculator intelligently drafts multiple tokens ahead of time, and the base model verifies them in a single forward pass. This approach boosts performance without sacrificing output quality, as every accepted token is guaranteed to match what the main model would have generated on its own.

Speculators standardizes this process by providing a productionized end-to-end framework to train draft models with reusable formats and tools. Trained models can seamlessly run in vLLM, enabling the deployment of speculative decoding in production-grade inference servers.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
    <img alt="Speculators user flow diagram" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/branding/speculators-user-flow-light.svg" />
  </picture>
</p>

______________________________________________________________________

💬 Join us on the [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) and share your questions, thoughts, or ideas in:

- `#speculators`
- `#feat-spec-decode`

🎥 Watch our Office Hours presentation: [Video](https://www.youtube.com/live/2ISAr_JVGLs) | [Slides](https://docs.google.com/presentation/d/1s4eAb7v-rdZt8smyULBJWGXjJXrgFTZWnwqYa2-h1l4/edit?slide=id.g3365e070742_6_0#slide=id.g3365e070742_6_0)

______________________________________________________________________

## Key Features

- **Offline Training Data Generation using vLLM:** Enable the generation of hidden states using vLLM. Data samples are saved to disk and can be used for draft model training.
- **Draft Model Training Support:** E2E training support of single and multi-layer draft models. Training is supported for both non-MoE and MoE models. VL Training is coming soon.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories into a standard speculators format for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.

> [!TIP]
> Read more about Speculators features in this [vLLM blog post](https://blog.vllm.ai/2025/12/13/speculators-v030.html).

## Supported Models

The following table summarizes the models that have been trained end-to-end by our team as well as others in the roadmap:

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
<td><a href="https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>70B-Instruct</td>
<td><a href="https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3">Qwen3</td>
<td>8B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>14B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-14B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>32B</td>
<td><a href="https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td rowspan="2">gpt-oss</td>
<td>20b</td>
<td><a href="https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3">EAGLE-3</a> ✅</td>
<td>✅</td>
</tr>
<tr>
<td>120b</td>
<td>EAGLE-3 ✅</td>
<td>✅</td>
</tr>
<tr>
  <td rowspan="3">Qwen3 MoE</td>
  <td>30B-Instruct</td>
  <td><a href="https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3">
      EAGLE-3
    </a> ✅</td>
  <td>✅</td>
</tr>
<tr>
  <td>235B-Instruct</td>
  <td>
    <a href="https://huggingface.co/RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3">
      EAGLE-3
    </a> ✅
  </td>
  <td>✅</td>
</tr>
<tr>
  <td>235B</td>
  <td><a href="https://huggingface.co/RedHatAI/Qwen3-235B-A22B-speculator.eagle3">
      EAGLE-3
    </a> ✅</td>
  <td>✅</td>
</tr>
<td>Qwen3-VL</td>
<td>235B-A22B</td>
<td>EAGLE-3 ⏳</td>
<td>⏳</td>
</tr>
<tr>
<td>Mistral 3 Large</td>
<td>675B-Instruct</td>
<td>EAGLE-3 ⏳</td>
<td>⏳</td>
</tr>
</tbody>
</table>

✅ = Supported, ⏳ = In Progress, ❌ = Not Yet Supported

## Examples

End-To-End Training Examples:

- [Train Llama3 Draft Model](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/llama3_8b_sharegpt_5k.py)
- [Train Qwen3 (Non-MoE) Draft Model](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/qwen3_8b_sharegpt_ultrachat.py)
- [Train GPT-OSS Draft Model](https://github.com/vllm-project/speculators/blob/main/examples/data_generation_and_training/gpt_oss_20b_ultrachat_5k.py)

## vLLM Inference

Models trained through Speculators can run seamlessly in vLLM using a simple `vllm serve <speculator_model>` command. This will run the model in vLLM using default arguments, defined in the `speculator_config` of the model's config.json.

```bash
VLLM_USE_V1=1 vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

Served models can then be benchmarked using [GuideLLM](https://github.com/vllm-project/guidellm). Below, we show sample benchmark results where we compare our speculator with its dense counterpart. We also additionally compare [quantization](https://github.com/vllm-project/llm-compressor) to explore additional performance improvements by swapping the dense verifier, `Qwen/Qwen3-8B` with the quantized FP8 model, [RedHatAI/Qwen3-8B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic) in the `speculator_config`.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/qwen_quant_benchmark.png">
    <img alt="GuideLLM Logo" src="https://raw.githubusercontent.com/vllm-project/speculators/main/docs/assets/qwen_quant_benchmark.png" width=180%>
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
git clone https://github.com/vllm-project/speculators.git
cd speculators

pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

To enable the generation of data (i.e hidden states) from vLLM for speculator training:

```bash
pip install -e ".[datagen]"
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

## License

Speculators is licensed under the [Apache License 2.0](https://github.com/vllm-project/speculators/blob/main/LICENSE).

## Cite

If you find Speculators helpful in your research or projects, please consider citing it:

```bibtex
@misc{speculators2025,
  title={Speculators: A Unified Library for Speculative Decoding Algorithms in LLM Serving},
  author={Red Hat},
  year={2025},
  howpublished={\url{https://github.com/vllm-project/speculators}},
}
```
