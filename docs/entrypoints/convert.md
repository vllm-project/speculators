# Convert

## What's Covered

In this document, you will learn:

1. How to convert models trained using external speculative decoding repositories into the Speculators format.
2. How to use the `convert`/`convert_model` entry point from both the CLI and Python APIs respectively.
3. Step-by-step examples of converting Eagle and HASS models using various options.

Before reading this document, you should be familiar with:

- Speculators CLI and Python entry point conventions (see [Entry Points](./index.md)).
- The basic structure of the speculative decoding repository you are converting from.
- The base model associated with your speculator.

## Overview

The `convert` entry point enables users to transform speculative decoding models—trained using formats like Eagle or HASS—into the standardized Speculators format. This conversion allows seamless integration into the broader Speculators ecosystem, including vLLM inference deployment.

Supported conversion formats:

- Eagle v1/v2/v3: https://github.com/SafeAILab/EAGLE
- HASS: https://github.com/HArmonizedSS/HASS

Conversion is available via both CLI and Python APIs.

## Usage

### CLI

```bash
speculators convert MODEL \
  --algorithm ALGORITHM \
  --algorithm-kwargs KWARGS \
  --verifier VERIFIER \
  --output-path OUTPUT_PATH \
  --validate-device DEVICE
```

#### Positional Arguments:

- `MODEL`: Speculator model source (local path, Hugging Face ID, or URL).

#### Required Arguments:

- `--algorithm`: The conversion algorithm to use for conversion: `eagle`, `eagle3`
- `--verifier`: Base model (local path, Hugging Face ID, or URL) used to validate and complete the speculator.

#### Optional Arguments:

- `--output-path`: Directory to save the converted model (default: `converted`).
- `--validate-device`: Device to run post-conversion validation (default: `cpu`).
- `--algorithm-kwargs`: Additional parameters for the conversion algorithm, passed as a JSON string of key-value pairs. Common parameters include:
  - `layernorms`: Add layer normalization layers (Eagle v1/v2, HASS)
  - `fusion_bias`: Add bias to fused fully connected layers (Eagle v1/v2, HASS)
  - `norm_before_residual`: Normalize before residual block (Eagle v3)

### Python API Usage

```python
from speculators.convert import convert_model

convert_model(
    model=MODEL,
    algorithm=ALGORITHM,
    verifier=VERIFIER,
    output_path=OUTPUT_PATH,
    validate_device=DEVICE,
    **algorithm_kwargs
)
```

#### Required Parameters:

- `model`: Source model to convert (local path, Hugging Face ID, or URL).
- `algorithm`: Conversion algorithm to use (e.g., `eagle`, `eagle3`).
- `verifier`: Base model to attach for verification (local path, Hugging Face ID, or URL).

#### Optional Parameters:

- `output_path`: Directory to save the converted model (default: `converted`).
- `validate_device`: Device to run post-conversion validation (default: `cpu`).
- `algorithm_kwargs`: Additional parameters for the conversion algorithm, passed as a JSON string of key-value pairs. Common parameters include:
  - `layernorms`: Add layer normalization layers (Eagle v1/v2, HASS)
  - `fusion_bias`: Add bias to fused fully connected layers (Eagle v1/v2, HASS)
  - `norm_before_residual`: Normalize before residual block (Eagle v3)

## Examples

Coming soon! This section will provide detailed examples of converting models using the CLI and Python API, including various configurations and options.

## Additional Notes

- For conversion to succeed, ensure the `verifier` model matches the architecture expected by the speculator model.
- If running into issues during conversion, try running with `-validate` to trigger shape and structure checks.
- Converted models can be loaded and served directly using tools like [vLLM](https://github.com/vllm-project/vllm).
