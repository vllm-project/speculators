# Offline Hidden States Generation

Offline hidden states generation pre-generates and caches all hidden states needed for training before the training process begins. This approach enables faster, more reproducible training runs with reduced dependency on vLLM during training.

## Overview

Hidden states are intermediate representations from the target model that capture its internal knowledge. The speculator learns to predict the next token using these hidden states as context. Offline generation creates these hidden states in advance and saves them to disk.

**Benefits of Offline Generation:**

- **Faster training iterations:** No vLLM inference overhead during training
- **Reproducible:** Same hidden states used across multiple training runs
- **Resource separation:** Training and inference can run on different machines
- **Disk-based caching:** Hidden states persist between training sessions
- **Better for large-scale training:** More efficient for datasets used multiple times

**Trade-offs:**

- **Disk space:** Requires significant storage (several GB per thousand samples)
- **Upfront time:** Initial generation takes time
- **Less flexible:** Changing target layers requires regeneration

## How It Works

The offline generation pipeline:

1. **Prepare data** - Tokenize and preprocess dataset with `prepare_data.py`
2. **Launch vLLM** - Start vLLM server configured for hidden states extraction
3. **Generate hidden states** - Run `data_generation_offline2.py` to extract and save hidden states
4. **Train offline** - Use cached hidden states for training

## Quick Start

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000

# Step 2: Launch vLLM server
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct \
  -- --port 8000

# Step 3: Generate hidden states
python scripts/data_generation_offline2.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-samples 5000 \
  --concurrency 32

# Step 4: Train with cached hidden states
torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \  # Fail if any hidden states are missing
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

## Data Generation Script

The `data_generation_offline2.py` script is the modern, recommended system for offline hidden states generation.

### Key Features

- **Async worker pool:** Configurable concurrency (default: 32 active requests)
- **Automatic resume:** Skips already-processed samples
- **Robust error handling:** Exponential backoff retries on failures
- **Validation:** Optionally validates generated safetensors files
- **Progress tracking:** Real-time progress with detailed statistics

### Basic Usage

```bash
python scripts/data_generation_offline2.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preprocessed-data ./training_data \
  --output ./hidden_states
```

### Key Arguments

**Data Arguments:**

- `--model` - Target model name or path (for verification)
- `--preprocessed-data` - Path to preprocessed dataset from `prepare_data.py`
- `--output` - Directory to save hidden states files
- `--max-samples` - Limit number of samples to process (default: all)

**vLLM Connection:**

- `--endpoint` - vLLM endpoint URL (default: `http://localhost:8000/v1`)
- `--request-timeout` - Timeout per request in seconds (default: 15)
- `--max-retries` - Maximum retry attempts on failure (default: 3)

**Performance:**

- `--concurrency` - Number of concurrent requests (default: 32)
  - Higher values = faster generation but more memory usage
  - Tune based on GPU memory and vLLM configuration

**Hidden States Configuration:**

- `--layer-ids` - Specific layers to extract (default: auto-select)
  - Format: `--layer-ids 2 16 29 31`
  - Auto-selected based on model depth if not specified

**Validation:**

- `--validate-outputs` - Verify safetensors file integrity
- `--max-consecutive-errors` - Abort after N consecutive failures (default: 10)

## Output Structure

Hidden states are saved as individual safetensors files:

```
hidden_states/
├── hs_0.safetensors       # Hidden states for sample 0
├── hs_1.safetensors       # Hidden states for sample 1
├── hs_2.safetensors
├── ...
└── hs_4999.safetensors
```

Each file contains:

- `hidden_states` - Tensor of shape `[seq_len, num_layers, hidden_size]`
- `prompt_token_ids` - Token IDs used for verification

## Resume Capability

If generation is interrupted, simply rerun the same command:

```bash
# First run - processes samples 0-1000, then crashes
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states

# Second run - automatically skips 0-1000, continues from 1001
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states
```

The script automatically detects existing `hs_*.safetensors` files and skips them.

## Performance Optimization

### Concurrency Tuning

The `--concurrency` parameter controls parallelism:

```bash
# Low concurrency (conservative, safe)
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --concurrency 16

# High concurrency (fast, requires more memory)
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --concurrency 64
```

**Recommendations:**

- **Small models (8B):** 32-64 concurrent requests
- **Medium models (70B):** 16-32 concurrent requests
- **Large models (400B+):** 8-16 concurrent requests

### vLLM Configuration

Launch vLLM with optimal settings for hidden states extraction:

```bash
# Data parallelism for faster generation (4 GPUs)
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 4 --port 8000

# Tensor parallelism for large models (8 GPUs)
python scripts/launch_vllm.py meta-llama/Llama-3.3-70B-Instruct \
  -- --tensor-parallel-size 8 --port 8000

# Combined: DP=2, TP=4 (8 GPUs total)
python scripts/launch_vllm.py meta-llama/Llama-3.3-70B-Instruct \
  -- --data-parallel-size 2 --tensor-parallel-size 4 --port 8000
```

### Validation Trade-offs

File validation ensures integrity but adds overhead:

```bash
# Fast mode - no validation
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states

# Safe mode - validate all outputs (slower)
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --validate-outputs
```

Use validation for production runs, skip for development.

## Target Layer Selection

Hidden states are extracted from specific layers of the target model:

### Auto-Selection (Default)

If `--layer-ids` is not specified, layers are automatically selected:

```python
# For a model with 32 layers, selects:
[2, 16, 29, 31]
```

This provides a good distribution across the model's depth.

### Manual Selection

Specify custom layers:

```bash
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --layer-ids 5 15 25 30  # Custom layer selection
```

**Note:** The same `--layer-ids` must be used when launching vLLM and during training.

## Error Handling

The script includes robust error handling:

### Automatic Retries

Failed requests are automatically retried with exponential backoff:

```
Attempt 1: immediate
Attempt 2: wait 2 seconds
Attempt 3: wait 4 seconds
...
```

### Consecutive Error Limit

Generation aborts if too many consecutive errors occur:

```bash
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-consecutive-errors 20  # Abort after 20 consecutive failures
```

This prevents wasting time on systematically failing setups.

### Common Error Types

- **Connection errors:** vLLM server not reachable → check endpoint
- **Timeout errors:** Requests taking too long → increase `--request-timeout`
- **Invalid responses:** Malformed vLLM output → check vLLM logs
- **Out of memory:** vLLM GPU OOM → reduce concurrency or increase vLLM GPUs

## Training Integration

### Using Offline Hidden States

In training, specify the hidden states path:

```bash
torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \  # Point to cached hidden states
  --on-missing raise \  # Fail if any are missing
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

### On-Missing Behavior

Control what happens when hidden states are missing:

- `--on-missing raise` - **Fail immediately** (recommended for offline training)
- `--on-missing generate` - Generate on-demand via vLLM
- `--on-missing skip` - Skip the sample
- `--on-missing warn` - Skip with warning

## Hybrid Training (Online + Offline)

For hybrid approaches, cache on first epoch:

```bash
torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \  # Generate if missing
  --on-generate cache \    # Cache generated hidden states
  --save-path ./checkpoints \
  --epochs 10
```

First epoch: Generates and caches hidden states Subsequent epochs: Uses cached hidden states

## Disk Space Requirements

Estimate disk space needed:

```
Size per sample ≈ seq_length × num_layers × hidden_size × 2 bytes (bfloat16)

Example (Llama 3.1 8B):
- seq_length: 8192
- num_layers: 4 (extracted layers)
- hidden_size: 4096
- Bytes per sample: 8192 × 4 × 4096 × 2 ≈ 268 MB

For 10K samples: ~2.6 TB
For 1K samples: ~260 GB
```

**Recommendations:**

- Use fast storage (NVMe SSD preferred)
- Consider compression for long-term storage
- Clean up hidden states after successful training if disk space is limited

## Best Practices

1. **Test with small dataset first:** Use `--max-samples 100` to verify setup
2. **Monitor progress:** Watch the progress bar and error messages
3. **Validate periodically:** Use `--validate-outputs` for important runs
4. **Keep vLLM logs:** Helps debug generation issues
5. **Document layer IDs:** Record which layers you used for reproducibility
6. **Separate storage:** Use fast storage for hidden states, cheaper storage for models

## Troubleshooting

### Slow Generation

If generation is slower than expected:

```bash
# Increase concurrency
python scripts/data_generation_offline2.py --concurrency 64 ...

# Check vLLM is using multiple GPUs
# In vLLM launch: --data-parallel-size 4
```

### Connection Timeouts

If requests timeout frequently:

```bash
# Increase timeout
python scripts/data_generation_offline2.py \
  --request-timeout 30 \  # Increase from default 15s
  --max-retries 5 \       # More retry attempts
  ...
```

### Validation Failures

If `--validate-outputs` reports corrupt files:

1. Check disk space is not full
2. Verify file system supports large files
3. Check for hardware issues
4. Regenerate failed samples

### vLLM Not Extracting Hidden States

Ensure vLLM is launched with hidden states support:

```bash
# Correct
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct

# Incorrect (plain vLLM serve)
vllm serve meta-llama/Llama-3.1-8B-Instruct  # May not have hidden states endpoint
```

## See Also

- [Prepare Data](prepare_data.md) - Prerequisite for offline generation
- [Training Feature](training.md) - Using offline hidden states in training
- [Train Eagle3 Offline Tutorial](../tutorials/train_eagle3_offline.md) - Complete walkthrough
