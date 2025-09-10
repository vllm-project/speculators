# Entrypoints (APIs, CLIs)

## What's Covered

In this document, you will learn:

1. How to interact with the Speculators toolkit through standardized CLI and Python API entry points
2. How to load, save, convert, and use speculators for inference tasks
3. The general design principles behind Speculators' entry points

Before reading this document, you should be familiar with:

- Basic Python programming concepts
- Command-line interface usage
- Basic understanding of LLMs and the Hugging Face transformers/datasets libraries
- Basic understanding of what speculative decoding is and its benefits

## Overview

Speculators is designed with user experience as a top priority. We provide standardized entry points through CLI and Python APIs to facilitate this. These entry points offer a consistent interface for interacting with the Speculators toolkit, regardless of the underlying speculative decoding algorithm.

The standardized entrypoints are a key component of Speculators' architecture, focusing on three core principles:

1. **Simplicity**: Users can leverage speculative decoding without understanding the underlying architecture or algorithmic details.
2. **Consistency**: CLI and Python APIs provide the same functionality with a similar interface pattern.
3. **Extensibility**: The entrypoints are designed to seamlessly support future enhancements and algorithms.

These entry points enable simple and straightforward training, testing, and saving for speculative decoding implementations, making speculative decoding more accessible to both researchers and practitioners.

## Core Entrypoints

The core entry points for Speculators are the CLI and Python APIs. These entry points provide a consistent interface for users to interact with the toolkit, regardless of the underlying algorithm, enabling simple training, testing, and saving for speculative decoding implementations. The CLI entry point is designed for users who prefer command-line interfaces. At the same time, the Python API is tailored for those who want to integrate Speculators into their Python applications or scripts. Both entry points are built on the same underlying components and configurations, ensuring a consistent experience across different usage scenarios.

All CLI entry points are nested under the `speculators` command and have equivalent Python API entry points nested under the `speculators` namespace. The supporting documentation for the CLI and Python API contain further details on the exact functionality, arguments, and options for the entry points. Additionally, to see what commands are available in the CLI, users can run:

```bash
speculators --help
```

Below is a summary of the core user scenarios and their corresponding entry points.

### Loading/Instantiating a Speculator

Users can load or instantiate a new instance of a speculator using the standardized entry points through the Python API directly or automatically through the later CLI commands, which include this functionality.

*Loading* from a pretrained speculator checkpoint:

```python
from pathlib import Path
from speculators import Speculator

hf_id = "HF_ORG/SPECULATOR_NAME"
local_path_str = "path/to/speculators/directory"
local_path = Path(local_path_str)
speculator = Speculator.from_pretrained(
 hf_id or local_path_str or local_path
)
```

*Instantiating* a new speculator from the algorithm's class:

```python
from speculators.algorithms import SpecDecSpeculator

speculator = SpecDecSpeculator(
 ...  # fill in required arguments
)
```

### Saving a Speculator

Users can save a speculator to a local directory or upload it to the Hugging Face Hub using the standardized entry points through the Python API directly or automatically through the later CLI commands, which include this functionality.

*Saving* a speculator to a local directory:

```python
from pathlib import Path
from speculators import Speculator

speculator = Speculator.from_pretrained(
    "HF_ORG/SPECULATOR_NAME"
)
local_path_str = "path/to/save/directory"
local_path = Path(local_path_str)
speculator.save_pretrained(
 local_path_str or local_path
)
```

*Saving* a speculator to the Hugging Face Hub:

```python
from pathlib import Path
from speculators import Speculator

speculator = Speculator.from_pretrained(
    "HF_ORG/SPECULATOR_NAME"
)
local_path = Path("path/to/save/directory")
speculator.save_pretrained(
 local_path,
    push_to_hub=True,
    repo_id="HF_ORG/SPECULATOR_NAME",
    private=False,
    token=None,
)
```

### Converting a Speculator

Users can automatically convert a speculative decoding model to an equivalent Speculators format for supported repos/research libraries (Eagle [v1-v3], HASS) using the standardized entry points through the CLI and Python API.

*Converting* a speculator from a supported repo format:

```python
from pathlib import Path
from speculators import Speculator

path_str = "path/to/specdecode/model/directory"
path = Path(path_str)
hf_id = "HF_ORG/SPECDECODE_MODEL_NAME"
speculator = Speculator.from_pretrained(
    source=path_str or path or hf_id,
)
speculator.save_pretrained(
    "path/to/save/directory",
)
```

Or using the equivalent CLI command:

```bash
speculators convert \
 --source path/to/specdecode/model/directory \
    --output path/to/save/directory
```

### Using a Speculator

Users can use a speculator to generate text using the standardized entry points through both the CLI and Python API for general, transformer-based inference, enabling quick validation and testing of the speculator.

*Sample Inference* using a speculator:

```python
from pathlib import Path
from speculators import Speculator
from transformers import AutoTokenizer

speculator = Speculator.from_pretrained(
    "HF_ORG/SPECULATOR_NAME"
)
tokenizer = AutoTokenizer.from_pretrained(
 speculator.config.verifier.model
)
input_ids = tokenizer(
    "The quick brown fox jumps over the lazy dog",
    return_tensors="pt",
).input_ids
output = speculator.generate(input_ids=input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Evaluating / Benchmarking a Speculator

Coming soon through integrations with external libraries and standardized entry points through the CLI and Python API. This will enable users to evaluate the performance of their speculators on various tasks and datasets, providing insights into their effectiveness and efficiency.

### Training a Speculator

Coming soon through integrations with external libraries and standardized entry points through the CLI and Python API. This will enable users to train their speculators on custom datasets, allowing for fine-tuning and optimizing the models for specific tasks or domains.

## Related Resources

Related Docs:

- [Model Formats Overview](./model_formats.md) - This doc provides detailed information about saving and loading speculators.
- [Architecture Overview](./architecture.md) - This doc provides detailed information about the Speculators architecture that powers the entry points.

Related External Resources:

- [Speculative Decoding Overview](https://arxiv.org/abs/2401.07851) - A general-purpose, survey paper on speculative decoding, its benefits, and its applications.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Documentation for the Transformers library, which Speculators integrates with
