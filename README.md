<div align="center">

# Speculators

[![License](https://img.shields.io/github/license/neuralmagic/speculators.svg)](https://github.com/neuralmagic/speculators/blob/main/LICENSE) [![Python Versions](https://img.shields.io/badge/Python-3.9--3.13-orange)](https://pypi.python.org/pypi/speculators)

</div>

## Overview

**Speculators** is a unified library for building, evaluating, and storing speculative decoding algorithms for large language model (LLM) inference, including in frameworks like vLLM. Speculative decoding is a lossless technique that speeds up LLM inference by using a smaller, faster speculator model to propose tokens, which are then verified by the larger base model, reducing latency without compromising output quality. Speculators standardizes this process with reusable formats and tools, enabling easier integration and deployment of speculative decoding in production-grade inference servers.

### Key Features

- **Speculative Decoding**: Speculators is a unified library that simplifies the creation and representation of speculative decoding algorithms, aiding both research and productization efforts for LLMs.
- **Standardized, Extensible Format:** Provides a Hugging Face-compatible format for defining speculative models, with tools to convert from external research repositories for easy adoption.
- **Seamless vLLM Integration:** Built for direct deployment into vLLM, enabling low-latency, production-grade inference with minimal overhead.

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

- [eagle3](https://github.com/neuralmagic/speculators/tree/main/research/eagle3): This implementation trains models similar to the EAGLE 3 architecture, specifically utilizing the Train Time Test method.

- [hass](https://github.com/neuralmagic/speculators/tree/main/research/hass): This implementation trains models that are a variation on the Eagle 1 architecture using the [HASS](https://github.com/HArmonizedSS/HASS) method.

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
