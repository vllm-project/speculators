# Prepare Data

Data preparation is the first critical step in training a speculator model. This feature tokenizes your dataset, applies chat templates, generates loss masks, and records token frequency statistics needed for vocabulary mapping.

## Overview

The `prepare_data.py` script transforms raw conversational datasets into a format ready for speculator training. It handles:

- **Tokenization:** Converts text to token IDs using the target model's tokenizer
- **Chat template application:** Formats conversations according to the model's expected format
- **Loss mask generation:** Marks which tokens should contribute to training loss (assistant responses only)
- **Token frequency tracking:** Records token usage statistics for vocabulary mapping
- **Quality filtering:** Optionally filters samples based on minimum valid tokens

## Quick Start

Basic usage:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000
```

This command:

1. Loads the ShareGPT dataset from HuggingFace
2. Tokenizes using Llama 3.1's tokenizer and chat template
3. Generates loss masks for assistant responses
4. Saves processed dataset and token frequencies to `./training_data`

## Supported Datasets

### Built-in Datasets

Speculators includes pre-configured loaders for popular datasets:

- **`sharegpt`** - ShareGPT Vicuna unfiltered dataset

  - Path: `Aeala/ShareGPT_Vicuna_unfiltered`
  - High-quality conversational data

- **`ultrachat`** - UltraChat 200K dataset

  - Path: `HuggingFaceH4/ultrachat_200k`
  - Large-scale multi-turn conversations

### Custom Datasets

You can also use:

- **HuggingFace datasets:** Pass any HF dataset path (e.g., `allenai/tulu-v2-sft-mixture`)
- **Local files:** Provide paths to local JSONL files

Multiple datasets can be combined:

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --data ultrachat \
  --data ./my_custom_data.jsonl \
  --output ./training_data
```

## Output Structure

The script produces the following files in the output directory:

```
training_data/
├── data-00000-of-00002.arrow    # Processed dataset (Arrow format)
├── data-00001-of-00002.arrow
├── dataset_info.json             # Dataset metadata
├── state.json                    # Dataset state
└── token_freq.pt                 # Token frequency distribution
```

### Dataset Columns

Each processed sample contains:

- `input_ids` - Tokenized conversation
- `labels` - Same as input_ids (for compatibility)
- `loss_mask` - Binary mask (1 for assistant tokens, 0 for user/system)
- `assistant_mask` - Alias for loss_mask

### Token Frequency File

The `token_freq.pt` file contains a dictionary mapping token IDs to their occurrence counts. This is used for vocabulary mapping to select the most frequent tokens for the draft model's reduced vocabulary.

## Key Arguments

### Required Arguments

- `--model` - HuggingFace model ID or local path

  - Example: `meta-llama/Llama-3.1-8B-Instruct`
  - Used for tokenizer and chat template

- `--data` - Dataset path(s), can be specified multiple times

  - Built-in: `sharegpt`, `ultrachat`
  - HF dataset: `allenai/tulu-v2-sft-mixture`
  - Local file: `./my_data.jsonl`

- `--output` - Directory to save output files

### Optional Arguments

- `--max-samples` - Limit number of samples to process

  - Default: Process all samples
  - Useful for testing or creating smaller datasets

- `--seq-length` - Maximum sequence length

  - Default: 8192
  - Sequences longer than this are truncated

- `--token-freq-path` - Custom path for token frequency file

  - Default: `{output}/token_freq.pt`

- `--assistant-pattern` - Custom regex for identifying assistant responses

  - Default: Auto-detected from chat template
  - Advanced use only

- `--turn-dropout` - Enable turn dropout augmentation

  - Randomly keeps first N consecutive turns per conversation
  - Increases dataset diversity

- `--minimum-valid-tokens` - Filter samples with too few trainable tokens

  - Example: `--minimum-valid-tokens 10`
  - Removes samples where assistant responses are too short

- `--num-preprocessing-workers` - CPU processes for parallel processing

  - Default: 8
  - Increase for faster processing on machines with many cores

- `--seed` - Random seed for reproducibility

  - Default: 0
  - Must match seed used in training

- `--overwrite` - Force reprocess existing data

  - Default: Skip if output already exists

## Data Preprocessing Pipeline

### Step 1: Load Raw Dataset

```python
# Built-in datasets
dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

# Custom JSONL
dataset = load_dataset("json", data_files="./my_data.jsonl")
```

### Step 2: Normalize Conversations

Conversations are normalized to a standard format:

```python
[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm doing well, thanks!"}
]
```

Different dataset formats (ShareGPT, UltraChat, etc.) are automatically converted.

### Step 3: Apply Chat Template

The tokenizer's chat template is applied:

```python
# Example Llama 3.1 format
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>...
```

### Step 4: Tokenize

Text is converted to token IDs:

```python
input_ids = tokenizer.encode(formatted_text)
```

### Step 5: Generate Loss Mask

The loss mask marks which tokens belong to assistant responses:

```python
loss_mask = [
    0,  # <|begin_of_text|>
    0,  # user header
    0,  # user content
    1,  # assistant header (trainable)
    1,  # assistant content (trainable)
    ...
]
```

Only tokens with `loss_mask=1` contribute to the training loss.

### Step 6: Record Token Frequencies

Token frequencies are accumulated across all samples for vocabulary mapping.

## Advanced Features

### Turn Dropout

Turn dropout randomly truncates conversations to create data augmentation:

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --turn-dropout \
  --output ./training_data
```

This creates multiple training examples from a single conversation by keeping only the first N turns (where N is randomly selected).

Benefits:

- Increases effective dataset size
- Improves model robustness
- Helps with shorter conversations

### Custom Chat Templates

If your model has a custom chat template:

```bash
# The tokenizer's chat template is automatically used
python scripts/prepare_data.py \
  --model ./my_custom_model \  # Local model with custom template
  --data sharegpt \
  --output ./training_data
```

### Minimum Valid Tokens Filter

Filter out low-quality samples:

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --minimum-valid-tokens 20 \  # Require at least 20 assistant tokens
  --output ./training_data
```

This removes samples where the assistant response is too short to be useful for training.

## Performance Tips

### Parallel Processing

Use more workers for faster processing:

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data ultrachat \
  --num-preprocessing-workers 32 \  # Use 32 CPU cores
  --output ./training_data
```

### Memory Management

For very large datasets:

1. Process in batches using `--max-samples`
2. Use multiple prepare_data.py runs with different subsets
3. Combine token frequency files later

## Incremental Processing

The script automatically skips reprocessing if output exists:

```bash
# First run - processes data
python scripts/prepare_data.py --model Qwen/Qwen3-8B --data sharegpt --output ./data

# Second run - skips processing (output already exists)
python scripts/prepare_data.py --model Qwen/Qwen3-8B --data sharegpt --output ./data

# Force reprocessing
python scripts/prepare_data.py --model Qwen/Qwen3-8B --data sharegpt --output ./data --overwrite
```

## Next Steps

After preparing your data:

1. **Generate Hidden States** - See [Offline Hidden States Generation](offline_hidden_states.md) or use online generation
2. **Build Vocabulary Mapping** - Automatically created during training based on token_freq.pt
3. **Train Model** - See [Training Feature](training.md)

## Examples

### Example 1: Small Test Dataset

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --max-samples 1000 \
  --output ./test_data
```

### Example 2: Multi-Dataset Training

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --data ultrachat \
  --max-samples 50000 \
  --turn-dropout \
  --minimum-valid-tokens 10 \
  --output ./production_data
```

### Example 3: Custom Dataset

```bash
# Your JSONL file should have format:
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

python scripts/prepare_data.py \
  --model ./my_model \
  --data ./my_conversations.jsonl \
  --output ./custom_data
```

## Troubleshooting

### Chat Template Not Found

If your model doesn't have a chat template:

```
Error: No chat template found for model
```

Solution: Add a chat template to your tokenizer or use a model with a defined template.

### Out of Memory

If preprocessing runs out of memory:

```bash
# Reduce workers
python scripts/prepare_data.py --num-preprocessing-workers 4 ...

# Process in smaller batches
python scripts/prepare_data.py --max-samples 10000 ...
```

### Incorrect Loss Mask

If the loss mask isn't capturing assistant responses correctly:

1. Check the chat template format
2. Verify the dataset format matches expectations
3. Use `--assistant-pattern` to customize detection (advanced)

## See Also

- [Offline Hidden States Generation](offline_hidden_states.md) - Generate hidden states from prepared data
- [Training Feature](training.md) - Train speculators with prepared data
- [Getting Started Guide](../getting_started.md) - Complete training workflow
