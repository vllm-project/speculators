# Training

The training feature provides a robust, production-ready framework for training speculator models with support for both single-GPU and distributed multi-GPU training using PyTorch FSDP (Fully Sharded Data Parallel).

## Overview

Speculators uses a custom training pipeline optimized for speculative decoding models:

- **Distributed Training:** FSDP support for efficient multi-GPU training
- **Flexible Data Loading:** Supports both online and offline hidden states generation
- **Advanced Scheduling:** Linear, cosine, and constant learning rate schedules with warmup
- **Checkpointing:** Automatic checkpoint saving, resumption, and best model tracking
- **Data Augmentation:** Noise injection to improve model robustness
- **Metric Logging:** Integration with TensorBoard, Weights & Biases, and TrackIO

## Quick Start

### Single-GPU Training

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Multi-GPU Training (FSDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

## Training Modes

### Online Training

Hidden states generated on-demand during training:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \  # vLLM server required
  --on-missing generate \                      # Generate if missing
  --on-generate delete \                       # Don't cache (pure online)
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**Pros:** Lower disk usage, no pre-generation step **Cons:** Requires vLLM server, slower per epoch

### Offline Training

Uses pre-generated hidden states from disk:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \  # Pre-generated hidden states
  --on-missing raise \                    # Fail if any missing
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**Pros:** Faster training, reproducible, no vLLM dependency during training **Cons:** Requires disk space, pre-generation time

### Hybrid Training

Cache hidden states on first epoch, reuse thereafter:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \  # Generate if missing
  --on-generate cache \    # Cache for future epochs
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

## Core Arguments

### Model Configuration

- `--verifier-name-or-path` - Target model (required)

  - HuggingFace model ID or local path
  - Example: `meta-llama/Llama-3.1-8B-Instruct`

- `--speculator-type` - Algorithm to train (default: `eagle3`)

  - Options: `eagle3`, `dflash`

- `--from-pretrained` - Continue training from existing speculator

  - Path to previously trained speculator model

- `--num-layers` - Number of draft model layers (default: 1)

  - Typical: 1-4 layers
  - More layers = better quality but slower inference

- `--draft-vocab-size` - Draft vocabulary size (default: auto)

  - Typical: 32000 for Eagle3, 64000 for DFlash
  - Smaller vocab = faster inference

- `--draft-arch` - Draft model architecture (default: `llama`)

  - Options: `llama`, `qwen3`
  - Note: Only `llama` currently supported in vLLM

### Data Configuration

- `--data-path` - Path to preprocessed dataset (required)

  - Output from `prepare_data.py`

- `--hidden-states-path` - Path to cached hidden states

  - Default: `{data_path}/hidden_states`

- `--vllm-endpoint` - vLLM server URL for online generation

  - Default: `http://localhost:8000/v1`

- `--on-missing` - Behavior when hidden states missing

  - `generate` - Generate via vLLM (online training)
  - `raise` - Fail immediately (offline training)
  - `skip` - Skip the sample
  - `warn` - Skip with warning

- `--on-generate` - What to do with newly generated hidden states

  - `delete` - Don't cache (pure online, save disk)
  - `cache` - Save for future epochs (hybrid mode)

- `--total-seq-len` - Maximum sequence length (default: 8192)

  - Must match value used in `prepare_data.py`

### Training Hyperparameters

- `--epochs` - Number of training epochs (default: 20)

- `--lr` - Learning rate (default: 1e-4)

  - Typical range: 1e-5 to 5e-4
  - Larger models often need lower LR

- `--scheduler-type` - LR scheduler type (default: `linear`)

  - `linear` - Linear decay from LR to 0
  - `cosine` - Cosine decay with cycles
  - `none` - Constant learning rate

- `--scheduler-warmup-steps` - Warmup steps (default: auto)

  - Auto: 10% of total steps

- `--scheduler-total-steps` - Total LR schedule steps (default: auto)

  - Auto: epochs × batches_per_epoch

- `--scheduler-num-cosine-cycles` - Cycles for cosine schedule (default: 0.5)

### Model-Specific Arguments (Eagle3)

- `--norm-before-residual` - RMSNorm before residual (default: True)

  - Use default for most models

- `--norm-before-fc` - RMSNorm before FC layer (default: False)

  - Set to `True` for gpt-oss models

- `--embed-requires-grad` - Train embedding layer (default: False)

  - Usually False for faster convergence

- `--target-layer-ids` - Custom target layer IDs

  - Default: Auto-selected based on model depth
  - Example: `--target-layer-ids 2 16 29 31`

### Model-Specific Arguments (DFlash)

- `--block-size` - Block size for DFlash (default: 8)

- `--max-anchors` - Maximum anchor positions (default: 256)

### Data Loading & Augmentation

- `--num-workers` - Dataloader worker processes (default: 12)

- `--prefetch-factor` - Batches to prefetch per worker (default: 4)

- `--noise-std` - Noise standard deviation for augmentation (default: 0.05)

  - Adds uniform noise to hidden states
  - Improves robustness

### Checkpointing

- `--save-path` - Directory for checkpoints (default: `./checkpoints`)

- `--checkpoint-freq` - Save checkpoint every N epochs (default: 1)

- `--save-best` - Save "best" symlink to lowest validation loss checkpoint

  - Useful for selecting optimal model

- `--no-resume-from-checkpoint` - Start fresh, ignore existing checkpoints

  - Default: Resume from latest checkpoint if found

### Logging

- `--logger` - Logging backends (comma-separated)

  - Options: `tensorboard`, `wandb`, `trackio`
  - Example: `--logger tensorboard,wandb`

- `--log-dir` - Log directory (default: `./logs`)

- `--run-name` - Experiment name for logging

### Advanced

- `--seed` - Random seed for reproducibility (default: 42)

- `--hidden-states-dtype` - Data type for hidden states (default: `bfloat16`)

  - Options: `bfloat16`, `float16`, `float32`

- `--deterministic-cuda` - Enable deterministic CUDA operations

  - May reduce performance

- `--use-off-policy-tokens` - Use off-policy tokens for training

  - Required when training on regenerated data

- `--mask-token-id` - Token ID for masking (default: auto-detect)

- `--request-timeout` - Timeout for vLLM requests (default: 15s)

- `--max-retries` - Max retry attempts for vLLM (default: 3)

## Training Pipeline

### 1. Data Loading

The training pipeline uses `ArrowDataset` which:

- Loads preprocessed dataset from disk
- Handles hidden states retrieval (online or offline)
- Applies noise augmentation
- Packs multiple samples into batches with `MultipackDistributedBatchSamplerV2`

### 2. Model Initialization

The draft model is initialized based on `--speculator-type`:

```python
# Eagle3 model initialization
model = Eagle3SpeculatorModel.from_training_args(
    verifier_config=verifier_config,
    draft_vocab_size=32000,
    num_layers=1,
    ...
)
```

### 3. Optimizer Setup

Uses AdamW optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
)
```

### 4. Learning Rate Scheduling

Supports multiple schedule types:

**Linear Decay:**

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

**Cosine with Warmup:**

```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.5,
)
```

### 5. Training Loop

Each epoch:

1. **Forward pass** - Process batch through draft model
2. **Compute loss** - KL divergence between draft and target logits
3. **Backward pass** - Compute gradients
4. **Optimizer step** - Update weights
5. **Scheduler step** - Update learning rate
6. **Log metrics** - Record loss, accuracy, learning rate

### 6. Validation

After each epoch:

1. Run validation loop (no gradient computation)
2. Compute validation loss and accuracy
3. Save checkpoint
4. Update "best" checkpoint if validation loss improved

### 7. Checkpointing

Checkpoints saved include:

```
checkpoints/
├── 0/                          # Epoch 0
│   ├── config.json            # Model configuration
│   ├── model.safetensors     # Model weights
│   ├── generation_config.json # Generation parameters
│   ├── optimizer_state_dict.pt # Optimizer state
│   └── scheduler_state_dict.pt # Scheduler state
├── 1/                         # Epoch 1
│   └── ...
├── best -> 5/                 # Symlink to best checkpoint (if --save-best)
└── latest -> 1/               # Symlink to latest
```

## Distributed Training (FSDP)

Fully Sharded Data Parallel (FSDP) enables efficient multi-GPU training:

### Launching FSDP Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node 4 \  # Number of GPUs
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000
```

### FSDP Features

- **Model sharding** - Each GPU holds a fraction of model parameters
- **Gradient reduction** - Gradients synchronized across GPUs
- **Automatic data distribution** - `DistributedBatchSampler` handles data splits
- **Checkpoint consolidation** - Distributed checkpoints merged on save

### Multi-Node Training

For training across multiple machines:

```bash
# On each node:
torchrun \
  --nproc_per_node 8 \
  --nnodes 4 \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  scripts/train.py \
  ...
```

## Vocabulary Mapping

Draft models use a reduced vocabulary for faster inference. Vocabulary mappings are automatically created during training if not provided:

### Auto-Generation

If `--draft-vocab-size` is set but no mappings exist:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --draft-vocab-size 32000 \  # Triggers auto-generation
  ...
```

The trainer will:

1. Load `token_freq.pt` from data directory
2. Select top 32K most frequent tokens
3. Generate `t2d.npy` and `d2t.npy` mappings
4. Cache mappings in data directory

### Manual Mappings

Provide pre-built mappings:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --t2d-path ./mappings/t2d.npy \
  --d2t-path ./mappings/d2t.npy \
  ...
```

## Metrics & Logging

### Built-in Metrics

Tracked automatically:

- **Loss** - KL divergence between draft and target logits
- **Accuracy** - Per-step token prediction accuracy
- **Learning Rate** - Current LR value
- **Epoch** - Current epoch number
- **Global Step** - Total training steps

### Logging Backends

**TensorBoard:**

```bash
python scripts/train.py --logger tensorboard --log-dir ./logs ...

# View logs
tensorboard --logdir ./logs
```

**Weights & Biases:**

```bash
python scripts/train.py --logger wandb --run-name my_experiment ...
```

**TrackIO:**

```bash
python scripts/train.py --logger trackio ...
```

**Multiple Loggers:**

```bash
python scripts/train.py --logger tensorboard,wandb ...
```

## Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
# First run - trains epochs 0-5
python scripts/train.py --epochs 10 --save-path ./checkpoints ...
# (interrupted after epoch 5)

# Second run - resumes from epoch 6
python scripts/train.py --epochs 10 --save-path ./checkpoints ...
```

To start fresh despite existing checkpoints:

```bash
python scripts/train.py --no-resume-from-checkpoint --save-path ./checkpoints ...
```

## Best Practices

1. **Start with small dataset** - Verify pipeline with 1K samples before scaling
2. **Use validation split** - Monitor validation loss to detect overfitting
3. **Tune learning rate** - Start with 1e-4, adjust based on loss curves
4. **Monitor GPU memory** - Adjust `--total-seq-len` or batch size if OOM
5. **Save checkpoints frequently** - Use `--checkpoint-freq 1` for safety
6. **Enable distributed training** - Use FSDP for faster training when possible
7. **Log experiments** - Use `--logger` to track and compare runs
8. **Use offline generation for production** - Pre-generate hidden states for reproducibility

## Troubleshooting

### Out of Memory

```bash
# Reduce sequence length
python scripts/train.py --total-seq-len 4096 ...

# Use fewer workers
python scripts/train.py --num-workers 4 ...

# Enable gradient checkpointing (if supported)
```

### Slow Training

```bash
# Increase number of workers
python scripts/train.py --num-workers 16 --prefetch-factor 8 ...

# Use offline hidden states
python scripts/train.py --hidden-states-path ./hs --on-missing raise ...

# Enable FSDP
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  scripts/train.py ...
```

### Loss Not Decreasing

- **Check learning rate** - Try 3e-5 or 1e-5
- **Verify data quality** - Ensure dataset is properly preprocessed
- **Check target layers** - Use default auto-selected layers
- **Increase epochs** - Some models need more training time

### vLLM Connection Errors (Online Training)

```bash
# Verify vLLM is running
curl http://localhost:8000/v1/models

# Check endpoint matches
python scripts/train.py --vllm-endpoint http://localhost:8000/v1 ...

# Increase timeout
python scripts/train.py --request-timeout 30 ...
```

## See Also

- [Prepare Data](prepare_data.md) - Prerequisite for training
- [Offline Hidden States](offline_hidden_states.md) - Pre-generate hidden states
- [Train Eagle3 Online Tutorial](../tutorials/train_eagle3_online.md) - Complete walkthrough
- [Train Eagle3 Offline Tutorial](../tutorials/train_eagle3_offline.md) - Offline training guide
- [CLI Reference](../../cli/index.md) - Complete argument reference
