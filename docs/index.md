# Home

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-white.svg">
    <img alt="Speculators Logo" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-logo-black.svg" width=55%>
  </picture>
</p>

<h3 align="center">
Unified Library for Speculative Decoding: Build, Evaluate, and Deploy Faster LLM Inference
</h3>

**Speculators** is a unified library for building, evaluating, and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster speculator model to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. Speculators standardizes this process with reusable formats and tools, enabling easier integration and deployment of speculative decoding in production-grade inference servers.

## Key Features

- **Unified Speculative Decoding Toolkit:** Simplifies the development, evaluation, and representation of speculative decoding algorithms, supporting both research and production use cases for LLMs.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.
- **Research-to-Production Pipeline:** Bridge the gap between experimental speculative decoding research and production deployments.

## Key Sections

<div class="grid cards" markdown>

<!-- - :material-rocket-launch:{ .lg .middle } Getting Started

    ---

    Install Speculators, convert your first speculative model, and deploy it with vLLM for faster inference.

    [:octicons-arrow-right-24: Getting started](./getting-started/) -->

<!-- - :material-book-open-variant:{ .lg .middle } Guides

    ---

    Detailed guides covering model conversion, format specifications, and integration with inference frameworks.

    [:octicons-arrow-right-24: Guides](./guides/) -->

- :material-console-line:{ .lg .middle } Entrypoints

  ---

  CLI tools, APIs, and end-to-end examples to get started with speculative decoding and dive into advanced implementations.

  [:octicons-arrow-right-24: Entrypoints](./entrypoints/)

- :material-api:{ .lg .middle } API Reference

    ---

    Complete reference documentation for the Speculators API to integrate speculative decoding into your workflow.

    [:octicons-arrow-right-24: API Reference](./api/)

</div>

## Quick Start
Install Speculators from PyPI:

```bash
pip install speculators
```

Or install directly from source:

```bash
pip install git+https://github.com/neuralmagic/speculators.git
```

Convert a speculative model and serve with vLLM:

```bash
# Convert a research model to Speculators format
speculators convert --help

# Serve with vLLM
VLLM_USE_V1=1 vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

## Research Implementations

Speculators includes prototype implementations of cutting-edge speculative decoding research:

- **[EAGLE 3](./research/eagle3/)**: Train Time Test method implementation for advanced token speculation
- **[HASS](./research/hass/)**: EAGLE 1 architecture variation using the HASS training method