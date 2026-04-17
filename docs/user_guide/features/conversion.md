# Model Conversion

The conversion feature allows you to convert speculative decoding models from third-party research repositories into the Speculators format for easy deployment with vLLM. This enables using existing trained models without retraining from scratch.

## Overview

Model conversion transforms external checkpoints into the standardized Speculators format by:

1. **Extending config.json** - Adds `speculators_config` with proper algorithm configuration
2. **Updating weights** - Remaps model weights to match Speculators architecture
3. **Fixing embeddings** - Ensures embedding layers are compatible with vocabulary mappings
4. **Enabling vLLM compatibility** - Makes models directly runnable with `vllm serve`

After conversion, models can be served immediately with vLLM using default speculative decoding parameters.

## Supported Algorithms

### EAGLE (v1/v2)

Convert models from the original [EAGLE research repository](https://github.com/SafeAILab/EAGLE):

- **EAGLE v1** - Original EAGLE speculative decoding
- **EAGLE v2** - Improved EAGLE with enhanced training
- **HASS** - Harmonized EAGLE variant from [HASS repository](https://github.com/HArmonizedSS/HASS)

### EAGLE-3

Convert models from [EAGLE v3 research repository](https://github.com/SafeAILab/EAGLE):

- **EAGLE-3** - Latest EAGLE version with vocabulary mapping support
- Enhanced for larger models and cross-tokenizer scenarios

## Quick Start

### Converting EAGLE-3 Models

```python
from speculators.convert import convert_model

convert_model(
    model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
    output_path="./converted_model",
)
```

After conversion:

```bash
vllm serve ./converted_model
```

### Converting EAGLE (v1/v2) Models

```python
from speculators.convert import convert_model

convert_model(
    model="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle",
    output_path="./converted_model",
)
```

## Conversion API

### `convert_model()`

Main entry point for model conversion.

**Parameters:**

- `model` (str, required) - Source model to convert
  - HuggingFace model ID (e.g., `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`)
  - Local path to checkpoint directory

- `verifier` (str, required) - Target/verifier model
  - HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
  - Local path to verifier model
  - This model is referenced in the `speculators_config`

- `algorithm` (str, required) - Conversion algorithm
  - `"eagle"` - For EAGLE v1/v2/HASS models
  - `"eagle3"` - For EAGLE-3 models

- `output_path` (str, optional) - Output directory
  - Default: `"converted"`
  - Converted model saved to this directory

- `validate_device` (str, optional) - Device for validation
  - Default: `None` (no validation)
  - Example: `"cuda:0"` to validate on GPU 0
  - Runs a test forward pass to verify conversion

**EAGLE-specific kwargs:**

- `layernorms` (bool) - Include layer normalization layers
  - Default: Auto-detected from checkpoint
  - Set explicitly if auto-detection fails

- `fusion_bias` (bool) - Include fusion bias parameters
  - Default: Auto-detected from checkpoint
  - Set explicitly if auto-detection fails

**EAGLE-3 specific kwargs:**

- `norm_before_residual` (bool) - Normalization before residual connection
  - Default: `True`
  - Set to `False` for models without this feature

- `eagle_aux_hidden_state_layer_ids` (list[int]) - Auxiliary hidden state layers
  - Default: Auto-selected based on model depth
  - Example: `[1, 23, 44]` for custom layer selection

## Conversion Process

### Step 1: Load Source Model

The converter loads the source model checkpoint:

```python
# From HuggingFace Hub
model = load_from_hub("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")

# From local directory
model = load_from_local("./eagle_checkpoint")
```

### Step 2: Detect Configuration

Auto-detects model architecture and features:

- Number of layers
- Hidden size
- Vocabulary size
- Special features (layernorms, fusion bias, etc.)

### Step 3: Remap Weights

Converts weight naming from source format to Speculators format:

```python
# Example weight remapping for EAGLE-3
{
    "fc.weight": "input_norm.weight",
    "embed_layernorm.weight": "draft_norm.weight",
    "lm_head_layernorm.weight": "lm_head_norm.weight",
    ...
}
```

### Step 4: Generate Speculators Config

Creates `speculators_config` for vLLM integration:

```json
{
  "speculators_config": {
    "algorithm": "eagle3",
    "proposal_methods": [
      {
        "proposal_type": "greedy",
        "speculative_tokens": 5
      }
    ],
    "verifier": {
      "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
      "architectures": ["LlamaForCausalLM"]
    }
  }
}
```

### Step 5: Save Converted Model

Saves the converted model in Speculators format:

```
converted_model/
├── config.json              # Extended with speculators_config
├── model.safetensors       # Remapped weights
├── generation_config.json  # Generation parameters
└── (optional) d2t.pt, t2d.pt  # Vocabulary mappings
```

## Examples

### Example 1: Basic EAGLE-3 Conversion

```python
from speculators.convert import convert_model

convert_model(
    model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
    output_path="./llama31_8b_eagle3",
)

# Serve with vLLM
# vllm serve ./llama31_8b_eagle3
```

### Example 2: EAGLE with Custom Features

```python
convert_model(
    model="./my_eagle_checkpoint",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle",
    output_path="./converted",
    layernorms=True,      # Enable layer norms
    fusion_bias=True,     # Enable fusion bias
)
```

### Example 3: EAGLE-3 with Custom Layers

```python
convert_model(
    model="./eagle3_checkpoint",
    verifier="Qwen/Qwen3-8B",
    algorithm="eagle3",
    output_path="./qwen3_eagle3",
    norm_before_residual=True,
    eagle_aux_hidden_state_layer_ids=[2, 14, 26, 30],  # Custom layers
)
```

### Example 4: Conversion with Validation

```python
convert_model(
    model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
    output_path="./converted",
    validate_device="cuda:0",  # Validate on GPU 0
)
```

Validation performs a test forward pass to ensure the conversion was successful.

## Command-Line Conversion

For scripted conversions, use Python scripts:

**eagle3_convert.py:**
```python
#!/usr/bin/env python3
from speculators.convert import convert_model

convert_model(
    model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
    output_path="./converted_model",
)

print("Conversion complete! Run with:")
print("vllm serve ./converted_model")
```

Run with:
```bash
python eagle3_convert.py
```

See `examples/convert/` for more conversion scripts.

## Using Converted Models

### Serving with vLLM

```bash
# Basic serving
vllm serve ./converted_model

# With custom port
vllm serve ./converted_model --port 8000

# With tensor parallelism
vllm serve ./converted_model --tensor-parallel-size 4

# With custom speculative tokens (overrides config)
vllm serve ./converted_model --speculative-tokens 7
```

The `speculators_config` in the model's config.json provides default parameters for vLLM.

### Fine-tuning Converted Models

Converted models can be further fine-tuned:

```bash
python scripts/train.py \
  --from-pretrained ./converted_model \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./fine_tuned \
  --epochs 5
```

### Uploading to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./converted_model",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

## Conversion vs Training

**When to convert:**
- You have an existing trained model from a research repository
- You want to quickly test speculative decoding
- The existing model quality is sufficient

**When to train from scratch:**
- You need a speculator for a new target model
- You want to customize training data or hyperparameters
- You need specific vocabulary mappings
- Better quality is needed for your use case

## Troubleshooting

### Weight Mismatch Errors

If conversion fails with weight mismatch:

```python
# Try with explicit feature flags for EAGLE
convert_model(
    model="./checkpoint",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle",
    layernorms=True,     # or False
    fusion_bias=True,    # or False
)
```

### Vocabulary Size Mismatch

If the draft model has a different vocabulary size:

```python
# Ensure vocabulary mappings are included
# For EAGLE-3, these should be auto-detected
convert_model(
    model="./checkpoint",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
)
# Check for d2t.pt and t2d.pt in output
```

### vLLM Compatibility Issues

If converted model doesn't work with vLLM:

1. **Verify conversion completed successfully**
   ```python
   import json
   with open("./converted_model/config.json") as f:
       config = json.load(f)
   assert "speculators_config" in config
   ```

2. **Check vLLM version**
   ```bash
   pip show vllm
   # Ensure vLLM version supports speculators
   ```

3. **Validate model architecture**
   ```python
   convert_model(
       ...,
       validate_device="cuda:0",  # Run validation
   )
   ```

### Incorrect Layer IDs

If EAGLE-3 uses wrong hidden state layers:

```python
# Manually specify layer IDs
convert_model(
    model="./checkpoint",
    verifier="meta-llama/Llama-3.1-8B-Instruct",
    algorithm="eagle3",
    eagle_aux_hidden_state_layer_ids=[2, 16, 29, 31],  # Match training
)
```

## Best Practices

1. **Always specify verifier** - Ensure verifier matches the original training target
2. **Validate on conversion** - Use `validate_device` for important conversions
3. **Test with vLLM** - Verify serving works before deploying
4. **Document source** - Note which research repository and version the model came from
5. **Version control** - Keep both source and converted models for reproducibility

## Supported Research Repositories

| Repository | Algorithm | Converter | Notes |
|-----------|-----------|-----------|-------|
| [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | EAGLE v1/v2 | `eagle` | Original EAGLE implementation |
| [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | EAGLE v3 | `eagle3` | Latest with vocab mapping |
| [HArmonizedSS/HASS](https://github.com/HArmonizedSS/HASS) | HASS | `eagle` | EAGLE variant |

## See Also

- [Train Eagle3 Tutorial](../tutorials/train_eagle3_online.md) - Train models from scratch
- [Convert Eagle3 Tutorial](../tutorials/convert_eagle3.md) - Step-by-step conversion guide
- [Serve in vLLM Tutorial](../tutorials/serve_vllm.md) - Deploy converted models
- [API Reference](../../api/README.md) - Full API documentation
