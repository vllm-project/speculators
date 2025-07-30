# Speculators

<h3 align="center">
A Unified Library for Speculative Decoding Algorithms for LLMs
</h3>

[![License](https://img.shields.io/github/license/neuralmagic/speculators.svg)](https://github.com/neuralmagic/speculators/blob/main/LICENSE) [![Python Versions](https://img.shields.io/badge/Python-3.9--3.13-orange)](https://pypi.python.org/pypi/speculators)

## Overview

**Speculators** is a unified library for creating, representing, and storing speculative decoding algorithms for large language model (LLM) serving, such as in vLLM. Speculative decoding is a lossless method that significantly improves the inference latency of LLM deployments. It achieves this by using a smaller, faster "draft" model to predict the tokens the larger, more powerful "base" model will generate, thereby accelerating the overall decoding process without sacrificing output quality. Speculators provides a standard format and tools to facilitate the productization of these decoding algorithms for inference servers.

### Key Features

- **Speculative Decoding**: Speculators is a unified library that simplifies the creation and representation of speculative decoding algorithms, aiding both research and productization efforts for LLMs.
- **Standardized Format**: We offer a universal, Hugging Face-compatible format designed to support speculative decoding algorithms. This also includes tools for converting algorithms from other research repositories into our standard.
- **vLLM Integration**: Speculators is designed for seamless and robust integration with inference servers, with **vLLM** being the primary and desired pathway for deploying speculative decoding models into production.

## Getting Started

### Installation

Before installing, ensure you have the following prerequisites:

- OS: Linux or MacOS
- Python: 3.9 or higher

Install Speculators directly from source using pip::

```bash
pip install git+https://github.com/neuralmagic/speculators.git
```

## Resources

Here you can find links to our research implementations. These provide prototype code for immediate enablement and experimentation, with plans for productization into the main package soon.

- [eagle3](https://github.com/neuralmagic/speculators/tree/main/research/eagle3): This implementation in Speculators trains models similar to the Eagle 3 architecture, specifically utilizing the Train Time Test method.

- [hass](https://github.com/neuralmagic/speculators/tree/main/research/hass): This implementation in Speculators trains models that are a variation on the Eagle 1 architecture using the [HASS](https://github.com/HArmonizedSS/HASS) method.

### License

Speculators is licensed under the [Apache License 2.0](https://github.com/neuralmagic/speculators/blob/main/LICENSE).

### Cite

If you find Speculators helpful in your research or projects, please consider citing it:

```bibtex
@misc{speculators2025,
  title={Speculators: A Unified Library for Speculative Decoding Algorithms in LLM Serving},
  author={Red Hat},
  year={2025},
  howpublished={\url{https://github.com/neuralmagic/speculators}},
}
```
