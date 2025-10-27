# EAGLE Data Generation Pipeline

This module provides a complete pipeline for generating training data for EAGLE-style speculative decoding models.

## Overview

The pipeline consists of two main stages:

1. **Preprocessing**: Tokenize conversational data and create loss masks
2. **Hidden State Extraction**: Use vLLM to extract intermediate layer hidden states from a target model

## Quick Start

### Basic Usage

Generate training data from ShareGPT using Llama 3.1 8B:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-8B \
    --train-data-path sharegpt \
    --chat-template llama3 \
    --output-dir ./training_data \
    --max-samples 5000
```

### Advanced Usage

With custom settings and multi-GPU:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-70B \
    --train-data-path ./my_data.jsonl \
    --chat-template llama3 \
    --seq-length 4096 \
    --cache-dir ./cache \
    --output-dir ./training_data \
    --layer-ids 2 28 54 \
    --tensor-parallel-size 4 \
    --batch-size 16 \
    --max-samples 10000
```

## Architecture

### Core Components

#### 1. VllmHiddenStatesGenerator

Extracts hidden states from intermediate layers during prefill using vLLM's efficient engine.

```python
from speculators.data_generation import VllmHiddenStatesGenerator

generator = VllmHiddenStatesGenerator(
    model_path="meta-llama/Llama-3.1-8B",
    layer_ids=[2, 14, 24],  # or None for auto-select
    tensor_parallel_size=1,
)

token_ids = [[1, 234, 567, 890]]  # batch of sequences
results = generator.generate(token_ids=token_ids)
```

**Features:**

- Auto-selects layers using EAGLE3 pattern if not specified
- Supports multi-GPU tensor parallelism
- Prefill-only mode (no decode overhead)
- Validates layer indices at initialization

#### 2. Preprocessing Pipeline

Tokenizes conversations and creates loss masks to identify trainable tokens.

```python
from speculators.data_generation.preprocessing import load_and_preprocess_dataset

dataset, tokenizer = load_and_preprocess_dataset(
    target_model_path="meta-llama/Llama-3.1-8B",
    train_data_path="sharegpt",
    chat_template="llama3",
    seq_length=2048,
    cache_dir="./cache",
    max_samples=1000,
)
```

**Supports:**

- ShareGPT format datasets
- HuggingFace datasets (sharegpt, ultrachat)
- Local JSON/JSONL files
- Multiple chat templates (llama3, qwen2, vicuna, chatml, mistral)

#### 3. Custom Worker Extension

vLLM worker extension that captures hidden states during model forward pass.

**Features:**

- Minimal overhead - only captures target layers
- TP rank 0 only (prevents duplicate captures)
- Automatic batching across sequences

### Configuration

#### Chat Templates

Supported templates in `configs.py`:

- `llama3` - Llama 3/3.1 format
- `qwen2` - Qwen 2 format
- `vicuna` - Vicuna format
- `chatml` - ChatML format
- `mistral` - Mistral Instruct format

#### Dataset Configs

Built-in datasets in `configs.py`:

- `sharegpt` - ShareGPT Vicuna unfiltered
- `ultrachat` - HuggingFace UltraChat 200k

Add custom datasets by extending `DATASET_CONFIGS`.

## Output Format

Each training sample is saved as a `.pt` file containing:

```python
{
    'input_ids': torch.Tensor,      # [seq_len]
    'hidden_state': torch.Tensor,   # [seq_len, hidden_dim * num_layers]
    'loss_mask': torch.Tensor,      # [seq_len] - 1 for trainable tokens
}
```

## Performance Optimization

### Memory Usage

The pipeline has a TODO to optimize KV cache allocation for prefill-only workloads:

```python
# TODO at vllm_hidden_states_generator.py:133
# Currently allocating based on available memory, but we only need minimal cache
# since we abort after prefill. Could reduce to: min_blocks = (max_num_batched_tokens // block_size) + 1
```

### Batch Size Tuning

- **Small models (7-8B)**: `--batch-size 16-32`
- **Medium models (13-30B)**: `--batch-size 8-16`
- **Large models (70B+)**: `--batch-size 4-8`

Adjust based on GPU memory and sequence length.

### Caching

Preprocessing is automatically cached. Cache key includes:
- Model path
- Chat template
- Sequence length
- Dataset path
- Max samples (if specified)

## Module Structure

```
data_generation/
├── __init__.py                          # Exports VllmHiddenStatesGenerator
├── vllm_hidden_states_generator.py      # Main hidden states extraction
├── custom_worker.py                     # vLLM worker extension
├── preprocessing.py                     # Dataset preprocessing
├── configs.py                           # Chat templates & dataset configs
├── vocab_mapping.py                     # Vocabulary mapping utilities
└── logging_utils.py                     # Clean logging utilities
```

## API Reference

### VllmHiddenStatesGenerator

```python
class VllmHiddenStatesGenerator:
    def __init__(
        self,
        model_path: str,
        layer_ids: List[int] = None,           # Auto-select if None
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
    )

    def generate(
        self,
        token_ids: Union[List[int], List[List[int]], torch.Tensor]
    ) -> List[Dict]
```

### Preprocessing Functions

```python
def load_and_preprocess_dataset(
    target_model_path: str,
    train_data_path: str,
    chat_template: str,
    seq_length: int,
    cache_dir: str,
    build_dataset_num_proc: int = 8,
    seed: int = 0,
    max_samples: Optional[int] = None,
) -> Tuple[HFDataset, PreTrainedTokenizer]

def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    max_length: int = 2048,
    num_proc: int = 8,
) -> HFDataset
```

## Troubleshooting

### Common Issues

**Issue**: Out of memory during hidden state extraction

- Reduce `--batch-size`
- Reduce `--seq-length`
- Increase `--tensor-parallel-size`

**Issue**: Layer index out of bounds

- Check model's actual number of layers
- Auto-selection uses: `[2, num_layers // 2, num_layers - 3]`

**Issue**: No assistant response spans found

- Verify chat template matches your data format
- Check that conversations have assistant responses

**Issue**: Cache invalidation

- Delete cache directory if changing preprocessing parameters
- Ensure `--seed` matches between runs for reproducibility

## Development

### Adding a New Chat Template

Edit `configs.py`:

```python
CHAT_TEMPLATES["my_template"] = ChatTemplate(
    name="my_template",
    system_prompt="...",
    user_header="...",
    assistant_header="...",
    end_of_turn_token="...",
    bos_token="...",
    eos_token="...",
)
```

### Adding a New Dataset

Edit `configs.py`:

```python
DATASET_CONFIGS["my_dataset"] = DatasetConfig(
    name="my_dataset",
    hf_path="username/dataset-name",
    split="train",
    normalize_fn=_my_normalize_fn,  # Optional
)
```
