# train.py

Trains speculator models using either online or offline hidden states. Supports single-GPU and multi-GPU distributed training with PyTorch FSDP.

## Basic Usage

**Single-GPU:**
```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**Multi-GPU (FSDP):**
```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

## Arguments

### Model Arguments

- **`--verifier-name-or-path`** (str, required)
  HuggingFace model ID or local path for the verifier/target model.

- **`--speculator-type`** (str, default: `"eagle3"`)
  Type of speculator model to train. Options: `eagle3`, `dflash`

- **`--from-pretrained`** (str, default: `""`)
  Path to a pretrained draft model to finetune.

- **`--num-layers`** (int, default: `1`)
  Number of transformer layers in the draft model.

- **`--draft-arch`** (str, default: `"llama"`)
  Architecture for draft decoder layers. Options: `llama`, `qwen3`
  **Note:** Only `llama` is currently supported in vLLM for inference.

### Data Arguments

- **`--data-path`** (str, default: `"./data"`)
  Path to the processed training data directory.

- **`--hidden-states-path`** (str, default: `{data-path}/hidden_states`)
  Path where cached hidden states files are stored.

- **`--vllm-endpoint`** (str, default: `"http://localhost:8000/v1"`)
  vLLM endpoint address for generating hidden states on-demand (online training).

- **`--on-missing`** (choice: `generate`|`skip`|`warn`|`raise`, default: `"generate"`)
  Behavior when cached hidden states are missing:
  - `generate`: Generate hidden states on-demand using vLLM endpoint
  - `skip`: Skip the sample silently
  - `warn`: Skip the sample with a warning
  - `raise`: Raise an error

- **`--on-generate`** (choice: `cache`|`delete`, default: `"delete"`)
  Behavior after generating new hidden states (only applies if `--on-missing=generate`):
  - `delete`: Delete hidden states after loading (pure online training)
  - `cache`: Store hidden states for reuse in future epochs (hybrid training)

- **`--request-timeout`** (float, default: `180.0`)
  Timeout in seconds for each individual vLLM request.

- **`--max-retries`** (int, default: `3`)
  Maximum number of retry attempts per vLLM request on failure.

- **`--legacy-data`** (flag)
  **DEPRECATED.** Use the old data format which stores hidden states alongside token_ids.

- **`--total-seq-len`** (int, default: `8192`)
  Maximum total sequence length for training batches.

### Vocabulary Mapping Arguments

- **`--draft-vocab-size`** (int, default: `None`)
  Vocabulary size for the draft model. If not specified, uses full verifier vocabulary.

- **`--token-freq-path`** (str, default: `{data-path}/token_freq.pt`)
  Path to token frequency distribution file.

- **`--d2t-path`** (str, default: `None`)
  Path to draft-to-target vocabulary mapping file (`.npy`). Must be provided with `--t2d-path`.

- **`--t2d-path`** (str, default: `None`)
  Path to target-to-draft vocabulary mapping file (`.npy`). Must be provided with `--d2t-path`.

- **`--mask-token-id`** (int, default: auto-detect)
  Token ID to use as mask token. Auto-detected if not provided.

- **`--target-layer-ids`** (int list, default: auto-select)
  Space-separated list of layer IDs used for hidden states.
  Default: `[2, num_layers//2, num_layers-3, num_layers]`
  **Must match the values used in vLLM if custom layers were specified.**

### Training Arguments

- **`--save-path`** (str, default: `"./checkpoints"`)
  Directory to save model checkpoints.

- **`--epochs`** (int, default: `20`)
  Number of training epochs.

- **`--lr`** (float, default: `1e-4`)
  Learning rate.

- **`--no-resume-from-checkpoint`** (flag)
  Disable automatic checkpoint resumption.

- **`--logger`** (str, default: `""`)
  Metric logging backend(s). Options: `trackio`, `wandb`, `tensorboard`
  Can specify multiple comma-separated: `--logger tensorboard,wandb`

- **`--log-dir`** (str, default: `"./logs"`)
  Directory to save training logs.

- **`--run-name`** (str, default: `None`)
  Name for the training run (used by logging backends).

- **`--seed`** (int, default: `42`)
  Random seed for reproducibility.

- **`--hidden-states-dtype`** (str, default: `"bfloat16"`)
  Data type for model weights and hidden states. Options: `float32`, `float16`, `bfloat16`

- **`--deterministic-cuda`** (flag)
  Enable deterministic CUDA operations. May impact performance.

### Model Hyperparameters

- **`--use-off-policy-tokens`** (flag)
  Use off-policy tokens during training (required for regenerated data).

- **`--norm-before-residual` / `--no-norm-before-residual`** (flag, default: `True`)
  Toggle normalization before residual connections.

- **`--embed-requires-grad` / `--no-embed-requires-grad`** (flag, default: `False`)
  Whether to train embedding layer weights.

- **`--norm-before-fc`** (flag)
  Use RMSNorm before FC layer in Eagle3 draft path (e.g., for gpt-oss models).

- **`--ttt-steps`** (int, default: `3`)
  Number of test-time training steps (for models that support it).

- **`--ttt-step-loss-decay`** (float, default: `1.0`)
  Loss decay factor for test-time training steps.

### DFlash-Specific Arguments

- **`--block-size`** (int, default: `8`)
  Block size for DFlash model.

- **`--max-anchors`** (int, default: `256`)
  Maximum anchor positions for DFlash training.

### Dataloader Arguments

- **`--num-workers`** (int, default: `12`)
  Number of dataloader worker processes.

- **`--prefetch-factor`** (int, default: `4`)
  Number of batches to prefetch per worker.

- **`--noise-std`** (float, default: `0.05`)
  Standard deviation for noise augmentation on hidden states.

### Checkpoint Arguments

- **`--checkpoint-freq`** (int, default: `1`)
  Save a checkpoint every N epochs. Must be ≥ 1.

- **`--save-best`** (flag)
  Save a symbolic link to the checkpoint with the lowest validation loss.

### Learning Rate Scheduler Arguments

- **`--scheduler-type`** (str, default: `"linear"`)
  Type of learning rate scheduler. Options: `linear`, `cosine`, `constant`

- **`--scheduler-warmup-steps`** (int, default: `None`)
  Number of warmup steps for the scheduler.

- **`--scheduler-total-steps`** (int, default: `None`)
  Total number of training steps for the scheduler.

- **`--scheduler-num-cosine-cycles`** (float, default: `0.5`)
  Number of cosine cycles for cosine scheduler.

## Examples

### Online Training

```bash
# First, start vLLM server
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --port 8000

# Then train with on-demand hidden states generation
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Offline Training

```bash
# Train using pre-generated hidden states
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Hybrid Training (Cache on First Epoch)

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate cache \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

### Multi-GPU Training with WandB Logging

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-70B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 20 \
  --lr 1e-4 \
  --logger wandb \
  --run-name eagle3-llama-70b \
  --scheduler-type cosine \
  --scheduler-warmup-steps 100 \
  --checkpoint-freq 2 \
  --save-best
```

### Fine-tuning a Pretrained Model

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --from-pretrained ./pretrained_speculator \
  --data-path ./new_training_data \
  --hidden-states-path ./hidden_states \
  --save-path ./finetuned_checkpoints \
  --epochs 5 \
  --lr 5e-6
```

## See Also

- [CLI Reference Overview](README.md)
- [Previous Step: Prepare Data](prepare_data.md)
- [Previous Step: Launch vLLM](launch_vllm.md) (for online training)
- [Previous Step: Generate Hidden States](data_generation_offline.md) (for offline training)
- [Training Feature Guide](../user_guide/features/training.md)
- [Online Training Tutorial](../user_guide/tutorials/train_eagle3_online.md)
- [Offline Training Tutorial](../user_guide/tutorials/train_eagle3_offline.md)
