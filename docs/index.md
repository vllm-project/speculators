# Speculators

<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px; text-align: left;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-model-icon-blue.png">
    <img alt="Speculators Logo" src="https://raw.githubusercontent.com/neuralmagic/speculators/main/docs/assets/branding/speculators-model-icon-blue.png" width="120">
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

## Key Sections

<div class="grid cards" markdown>

- :material-handshake:{ .lg .middle } Code of Conduct

    ---

    Our community guidelines ensure that participation in the Speculators project is a positive, inclusive, and respectful experience for everyone.

    [:octicons-arrow-right-24: Code of Conduct](developer/code-of-conduct.md)

- :material-source-pull:{ .lg .middle } Contributing Guide

    ---

    Learn how to effectively contribute to Speculators, including reporting bugs, suggesting features, improving documentation, and submitting code.

    [:octicons-arrow-right-24: Contributing Guide](developer/contributing.md)

- :material-source-pull:{ .lg .middle } Entrypoints

    ---

    CLI tools, APIs, and end-to-end examples to get started with speculative decoding and dive into advanced implementations.

    [:octicons-arrow-right-24: Entrypoints](entrypoints/index.md)

- :material-tools:{ .lg .middle } Development Guide

    ---

    Detailed instructions for setting up your development environment, implementing changes, and adhering to the project's coding standards and best practices.

    [:octicons-arrow-right-24: Development Guide](developer/index.md/)

- :material-api:{ .lg .middle } API Reference

    ---

    Complete reference documentation for the Speculators API to integrate speculative decoding into your workflow.

    [:octicons-arrow-right-24: API Reference](reference/speculators/)

- :material-palette:{ .lg .middle } Branding Guidelines

    ---

    Visual identity and branding guidelines for using Speculators logos, colors, and brand assets in your projects and communications.

    [:octicons-arrow-right-24: Branding Guidelines](developer/branding.md/)
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

- **EAGLE 3**: Train Time Test method implementation for advanced token speculation
- **HASS**: EAGLE 1 architecture variation using the HASS training method
