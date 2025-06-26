# Eagle Checkpoint Conversion Guide

This guide explains how to convert EAGLE 1, EAGLE 2, and HASS checkpoints to the standardized speculators format.

## Overview

The speculators library provides a unified interface for speculative decoding models. To use existing Eagle/HASS checkpoints, they must first be converted to the speculators format.

## Supported Checkpoints

We support converting the following checkpoint types:

- **EAGLE 1**: Original Eagle architecture
- **EAGLE 2**: Updated Eagle architecture (same structure as EAGLE 1)
- **HASS**: Hardware-Aware Speculative Sampling variant

## Quick Start

```bash
# Install speculators
pip install speculators

# Convert a standard Eagle checkpoint
speculators convert eagle yuhuili/EAGLE-LLaMA3.1-Instruct-8B ./converted/eagle meta-llama/Llama-3.1-8B-Instruct

# Convert with extra layernorms enabled
speculators convert eagle nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT ./converted/eagle-layernorms meta-llama/Llama-3.1-8B-Instruct --layernorms
```

## Command Line Interface

### Basic Usage

```bash
speculators convert eagle <input_path> <output_path> <base_model> [OPTIONS]
```

### Arguments

- `input_path`: Path to Eagle/HASS checkpoint (local path or HuggingFace model ID)
- `output_path`: Directory where the converted checkpoint will be saved
- `base_model`: Base model name/path (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

### Options

- `--layernorms`: Enable extra layernorms (configurable feature for improved training stability)
- `--fusion-bias`: Enable fusion bias (automatically detected if checkpoint contains `fc.bias`)
- `--validate/--no-validate`: Validate the converted checkpoint (default: validate)
  - When enabled, validation performs:
    - Model loading test using `EagleSpeculator.from_pretrained()`
    - Forward pass test with dummy inputs
    - Ensures the checkpoint is properly formatted and functional

## Examples

### Converting Standard Eagle Checkpoint

```bash
speculators convert eagle \
    yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
    ./converted/eagle-llama3.1-8b \
    meta-llama/Llama-3.1-8B-Instruct
```

Output:

```
2025-06-26 02:03:32.123 | INFO     | Converting Eagle checkpoint: yuhuili/EAGLE-LLaMA3.1-Instruct-8B
2025-06-26 02:03:32.456 | INFO     | Loaded 10 weights
2025-06-26 02:03:33.789 | SUCCESS  | Saved to: converted/eagle-llama3.1-8b
2025-06-26 02:03:34.012 | INFO     | Validating converted checkpoint...
2025-06-26 02:03:34.345 | SUCCESS  | Model loaded successfully
2025-06-26 02:03:34.678 | SUCCESS  | Forward pass successful
```

### Converting with Extra Layernorms

Extra layernorms are a configurable feature that can improve training stability. They add normalization after embeddings and before the language model head.

```bash
speculators convert eagle \
    nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT \
    ./converted/eagle-with-layernorms \
    meta-llama/Llama-3.1-8B-Instruct \
    --layernorms
```

### Converting Local Checkpoint

```bash
speculators convert eagle \
    /path/to/local/checkpoint \
    ./converted/local-eagle \
    meta-llama/Llama-3.1-8B \
    --fusion-bias
```

## Python API

### Basic Conversion

```python
from speculators.convert.eagle import EagleConverter

converter = EagleConverter()
converter.convert(
    input_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    output_path="./converted/eagle",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    validate=True
)
```

### Custom Configuration

```python
# Convert with specific features
converter.convert(
    input_path="path/to/checkpoint",
    output_path="./converted/custom",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    layernorms=True,      # Enable extra layernorms
    fusion_bias=False,    # Disable fusion bias
    validate=True         # Validate after conversion
)
```

### Loading Converted Models

```python
from speculators.models.eagle import EagleSpeculator

# Load converted checkpoint
model = EagleSpeculator.from_pretrained("./converted/eagle")

# Execute forward pass with dummy inputs
import torch

batch_size = 1
seq_length = 10
hidden_size = model.config.transformer_layer_config.hidden_size

input_ids = torch.randint(0, 1000, (batch_size, seq_length))
hidden_states = torch.randn(batch_size, seq_length, hidden_size)

with torch.no_grad():
    output = model(input_ids=input_ids, hidden_states=hidden_states)
    logits = output.logits  # Shape: (batch_size, seq_length, vocab_size)
```

## Understanding the Conversion Process

### 1. Checkpoint Analysis

The converter first analyzes the input checkpoint to:

- Detect checkpoint format (safetensors, PyTorch, or sharded)
- Identify architectural features (fusion bias, extra layernorms)
- Extract model configuration

### 2. Configuration Building

Creates a `EagleSpeculatorConfig` with:

- **Transformer layer config**: Single LlamaDecoderLayer configuration
- **Speculators config**: Algorithm settings and verifier information
- **Feature flags**: `layernorms` and `fusion_bias` settings

### 3. Weight Processing

- Maps weight names if needed (e.g., for layernorm variants)
- Skips unnecessary weights (e.g., `hidden_layernorm.weight`)
- Preserves all other weights unchanged

### 4. Saving

- Saves configuration as `config.json`
- Saves weights in safetensors format as `model.safetensors`

### 5. Validation (if enabled)

- Loads the model using `EagleSpeculator.from_pretrained()`
- Performs a forward pass with random inputs
- Confirms the checkpoint is properly formatted and functional

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**

   - Verify the HuggingFace model ID is correct
   - Check you have access to private repositories
   - Ensure local paths exist

2. **"Sharded checkpoints not yet supported"**

   - The converter currently only supports single-file checkpoints
   - Try downloading and merging shards manually first

3. **"Missing or incorrect speculators_model_type"**

   - This means you're trying to load an unconverted checkpoint
   - Run the conversion process first

4. **Validation failures**

   - Check the base model matches the checkpoint architecture
   - Verify feature flags match the checkpoint type
   - Review the error message for specific issues

### Debug Logging

The converter uses loguru for detailed logging:

```python
from loguru import logger

# Enable debug logging
logger.add(lambda msg: print(msg), level="DEBUG")

# Now run conversion with detailed output
converter = EagleConverter()
converter.convert(...)
```

## Architecture Details

### Eagle Model Structure

```
Input IDs + Hidden States
         ↓
    Embedding Layer
         ↓
  [Post-Embedding LayerNorm]  # Only if layernorms=True
         ↓
    Fusion Layer (fc)
         ↓
  Single Transformer Layer
         ↓
  [Pre-LM Head LayerNorm]     # Only if layernorms=True
         ↓
      LM Head
         ↓
      Logits
```

### Key Components

1. **Fusion Layer**: Combines token embeddings with verifier hidden states

   - Input: Concatenated embeddings and hidden states
   - Output: Fused representation
   - Bias: Optional (controlled by `fusion_bias`)

2. **Transformer Layer**: Single LlamaDecoderLayer

   - Attention mechanism with RoPE embeddings
   - Feed-forward network
   - RMS normalization

3. **Extra LayerNorms** (when enabled):

   - Post-embedding normalization
   - Pre-LM head normalization
   - Improves training stability

## Advanced Usage

### Batch Conversion

```python
checkpoints = [
    ("yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "./converted/eagle1", False, False),
    ("path/to/eagle2", "./converted/eagle2", False, False),
    ("path/to/hass", "./converted/hass", True, False),
    # (input_path, output_path, layernorms, fusion_bias)
]

converter = EagleConverter()
for input_path, output_path, layernorms, fusion_bias in checkpoints:
    converter.convert(
        input_path=input_path,
        output_path=output_path,
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        layernorms=layernorms,
        fusion_bias=fusion_bias
    )
```

### Feature Detection

The converter can automatically detect certain features:

```python
# Fusion bias is automatically detected if checkpoint contains fc.bias
converter.convert(
    input_path="path/to/hass/checkpoint",  # Contains fc.bias
    output_path="./converted/hass-auto",
    base_model="meta-llama/Llama-3.1-8B",
    # fusion_bias will be automatically set to True
)

# Layernorms are automatically detected if checkpoint contains layernorm weights
converter.convert(
    input_path="path/to/layernorm/checkpoint",  # Contains embed_layernorm.weight
    output_path="./converted/layernorm-auto",
    base_model="meta-llama/Llama-3.1-8B",
    # layernorms will be automatically set to True
)
```

## Contributing

To add support for new checkpoint types:

1. Update `LAYERNORM_MAPPINGS` in `eagle_converter.py` for weight name mappings
2. Add detection logic in the `convert` method
3. Update this documentation with examples

## References

- [EAGLE Paper](https://arxiv.org/abs/2401.15077)
- [Speculators Documentation](https://github.com/foundation-model-stack/speculators)
- [HuggingFace Model Hub](https://huggingface.co/models)
