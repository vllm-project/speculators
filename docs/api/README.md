# API Reference

This section contains the auto-generated Python API documentation for Speculators.

## Overview

The Speculators Python API provides programmatic access to:

- **Model Configuration**: Configure speculator models with `SpeculatorModelConfig`
- **Model Classes**: Access to EAGLE-3, DFlash, and other algorithm implementations
- **Data Processing**: Tools for preparing and processing training data
- **Conversion Utilities**: Convert models from external formats
- **Training Components**: Custom trainers and data loaders

## Using the API

The API is automatically generated from the source code docstrings. Navigate through the sidebar to explore:

- **Core Models**: EAGLE-3, DFlash model implementations
- **Configuration**: Model configuration classes
- **Data Generation**: Hidden states extraction and data preparation
- **Conversion**: Model format conversion utilities

## Quick Start

```python
from speculators import SpeculatorModelConfig
from speculators.models.eagle3 import Eagle3DraftModel

# Load a pre-trained speculator
config = SpeculatorModelConfig.from_pretrained("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3")
model = Eagle3DraftModel.from_pretrained("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3")
```

## Further Reading

- [User Guide](../user_guide/README.md) - High-level usage documentation
- [CLI Reference](../cli/README.md) - Command-line tool documentation
- [Tutorials](../user_guide/tutorials/README.md) - Step-by-step guides
