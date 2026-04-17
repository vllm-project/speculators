# CLI Reference

This page provides a comprehensive reference for all command-line interface (CLI) tools available in Speculators.

## Overview

Speculators provides four main CLI scripts for different stages of the speculative decoding workflow:

| Script | Purpose |
|--------|---------|
| [`prepare_data.py`](#prepare-data) | Preprocess and tokenize datasets for training |
| [`data_generation_offline.py`](#data-generation-offline) | Generate hidden states offline using vLLM |
| [`launch_vllm.py`](#launch-vllm) | Launch vLLM server configured for hidden states extraction |
| [`train.py`](#train) | Train speculator models with online or offline hidden states |

---

## prepare_data.py

Prepares data for speculator training by:
1. Applying chat template and tokenizing each sample
2. Producing a loss/assistant mask for each sample
3. Recording token frequency statistics

The output is a processed dataset ready for online training or offline hidden states generation.

### Basic Usage

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000
```

### Arguments

#### Model Arguments

- **`--model`** (str, required)
  HuggingFace model ID or local path for the target model.

  Example: `meta-llama/Llama-3.1-8B-Instruct`

#### Data Arguments

- **`--data`** (str, required, repeatable)
  Path to training data. Can be a HuggingFace dataset name or local path. Use multiple times to specify multiple datasets.

  Example: `--data sharegpt --data ./custom_data.jsonl`

- **`--seq-length`** (int, default: `8192`)
  Maximum sequence length for preprocessing and model.

- **`--max-samples`** (int, default: `None`)
  Maximum number of samples to process. If `None`, processes all samples.

- **`--token-freq-path`** (str, default: `{output}/token_freq.pt`)
  Path to save token frequency distribution. Defaults to `token_freq.pt` in the output directory.

- **`--assistant-pattern`** (str, default: `None`)
  Custom regex pattern for matching assistant responses. If not provided, auto-detected from chat template.

- **`--turn-dropout`** (flag)
  Enable turn dropout: randomly keeps first N consecutive turns per conversation for data augmentation.

- **`--minimum-valid-tokens`** (int, default: `None`)
  Drop samples whose loss mask contains fewer than this many trainable tokens.

#### Output Arguments

- **`--output`** (str, required)
  Directory to save the processed dataset.

- **`--overwrite`** (flag)
  Forcibly rerun preprocessing and delete existing content in output directory.

#### Processing Arguments

- **`--seed`** (int, default: `0`)
  Random seed for reproducibility. Must match the seed used in other scripts.

- **`--num-preprocessing-workers`** (int, default: `8`)
  Number of CPU processes for dataset preprocessing.

### Example

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --data ./custom_conversations.jsonl \
  --output ./prepared_data \
  --seq-length 4096 \
  --max-samples 10000 \
  --turn-dropout \
  --num-preprocessing-workers 16
```

---

## data_generation_offline.py

Generates training data for speculator models by extracting hidden states from the target model using vLLM. The output is saved as individual `.pt` files for offline training.

### Basic Usage

```bash
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./training_data \
  --max-samples 5000
```

### Arguments

#### Model Arguments

- **`--target-model-path`** (str, required)
  HuggingFace model ID or local path for the target model.

- **`--tensor-parallel-size`** (int, default: GPU count)
  Tensor parallel size for the target model. Defaults to the number of available GPUs.

- **`--gpu-memory-utilization`** (float, default: `0.8`)
  Target GPU memory utilization (0.0 to 1.0).

#### Data Arguments

- **`--train-data-path`** (str, required, repeatable)
  Path to training data (same as used in preprocessing). Can be specified multiple times.

- **`--seq-length`** (int, default: `2048`)
  Maximum sequence length for preprocessing and model.

- **`--max-samples`** (int, default: `None`)
  Maximum number of samples to process. If `None`, processes all samples.

- **`--token-freq-path`** (str, default: `./token_freq.pt`)
  Path to save token frequency distribution.

- **`--hf-cache-dir`** (str, default: `None`)
  Directory for HuggingFace datasets cache. If not specified, uses `HF_DATASETS_CACHE` environment variable or default location.

- **`--assistant-pattern`** (str, default: `None`)
  Custom regex pattern for matching assistant responses. If not provided, auto-detected from chat template.

- **`--turn-dropout`** (flag)
  Enable turn dropout for data augmentation.

- **`--minimum-valid-tokens`** (int, default: `None`)
  Drop samples whose loss mask contains fewer than this many trainable tokens.

#### Output Arguments

- **`--output-dir`** (str, required)
  Directory to save `.pt` files containing hidden states.

#### Hidden States Generation Arguments

- **`--layer-ids`** (int list, default: auto-select)
  List of layer IDs from which to capture hidden states.
  Default: `[2, num_layers//2, num_layers-3, num_layers]`

  Example: `--layer-ids 2 16 30 32`

- **`--batch-size`** (int, default: `8`)
  Batch size for hidden states generation.

#### Processing Arguments

- **`--seed`** (int, default: `0`)
  Random seed for reproducibility.

- **`--start-idx`** (int, default: `0`)
  Starting index for output files. Useful for resuming interrupted runs.

- **`--num-preprocessing-workers`** (int, default: `8`)
  Number of CPU processes for dataset preprocessing.

### Example

```bash
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-70B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./hidden_states \
  --tensor-parallel-size 4 \
  --batch-size 16 \
  --layer-ids 2 20 40 80 \
  --max-samples 50000 \
  --seq-length 4096
```

---

## launch_vllm.py

Launches a vLLM server configured for hidden states extraction, used for online training or on-demand hidden states generation.

### Basic Usage

```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --hidden-states-path /tmp/hidden_states \
  -- --port 8000 --tensor-parallel-size 2
```

### Arguments

#### Positional Arguments

- **`model`** (str, required)
  Model name or path to extract hidden states from.

#### Speculators Arguments

- **`--hidden-states-path`** (str, default: `/tmp/hidden_states`)
  The directory to save hidden states to.

- **`--target-layer-ids`** (int list, default: auto-select)
  Space-separated list of integer layer IDs from which to capture hidden states.
  Default: `[2, num_layers//2, num_layers-3, num_layers]`

  **Important:** If set, you must also pass the same value to the training script using `--target-layer-ids`.

- **`--include-last-layer` / `--no-include-last-layer`** (flag, default: `True`)
  For DFlash models, append the last layer (`num_hidden_layers`) to `target_layer_ids` for verifier hidden states extraction.

- **`--dry-run`** (flag)
  Print the command that would be executed without running it.

#### vLLM Arguments

All arguments after `--` are passed directly to vLLM. Common vLLM arguments include:

- `--port`: Server port (default: `8000`)
- `--host`: Server host (default: `0.0.0.0`)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--gpu-memory-utilization`: GPU memory utilization (0.0 to 1.0)
- `--max-model-len`: Maximum model context length
- `--trust-remote-code`: Allow custom model code execution

See [vLLM CLI documentation](https://docs.vllm.ai/en/latest/cli/) for full list of options.

### Examples

**Basic Launch:**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --port 8000
```

**Multi-GPU with Custom Layers:**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-70B-Instruct \
  --hidden-states-path /data/hidden_states \
  --target-layer-ids 5 20 40 80 \
  -- --tensor-parallel-size 4 --port 8000
```

**Dry Run (Preview Command):**
```bash
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --dry-run \
  -- --port 8000
```

---

## train.py

Trains speculator models using either online or offline hidden states. Supports single-GPU and multi-GPU distributed training with PyTorch FSDP.

### Basic Usage

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

### Arguments

#### Model Arguments

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

#### Data Arguments

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

#### Vocabulary Mapping Arguments

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

#### Training Arguments

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

#### Model Hyperparameters

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

#### DFlash-Specific Arguments

- **`--block-size`** (int, default: `8`)
  Block size for DFlash model.

- **`--max-anchors`** (int, default: `256`)
  Maximum anchor positions for DFlash training.

#### Dataloader Arguments

- **`--num-workers`** (int, default: `12`)
  Number of dataloader worker processes.

- **`--prefetch-factor`** (int, default: `4`)
  Number of batches to prefetch per worker.

- **`--noise-std`** (float, default: `0.05`)
  Standard deviation for noise augmentation on hidden states.

#### Checkpoint Arguments

- **`--checkpoint-freq`** (int, default: `1`)
  Save a checkpoint every N epochs. Must be ≥ 1.

- **`--save-best`** (flag)
  Save a symbolic link to the checkpoint with the lowest validation loss.

#### Learning Rate Scheduler Arguments

- **`--scheduler-type`** (str, default: `"linear"`)
  Type of learning rate scheduler. Options: `linear`, `cosine`, `constant`

- **`--scheduler-warmup-steps`** (int, default: `None`)
  Number of warmup steps for the scheduler.

- **`--scheduler-total-steps`** (int, default: `None`)
  Total number of training steps for the scheduler.

- **`--scheduler-num-cosine-cycles`** (float, default: `0.5`)
  Number of cosine cycles for cosine scheduler.

### Examples

**Online Training:**
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

**Offline Training:**
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

**Hybrid Training (Cache on First Epoch):**
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

**Multi-GPU Training with WandB Logging:**
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

**Fine-tuning a Pretrained Model:**
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

---

## Common Workflows

### Full Training Pipeline (Offline)

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./prepared_data \
  --max-samples 10000

# Step 2: Generate hidden states offline
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./hidden_states \
  --tensor-parallel-size 2 \
  --batch-size 16

# Step 3: Train the speculator
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./prepared_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

### Full Training Pipeline (Online)

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./prepared_data \
  --max-samples 10000

# Step 2: Launch vLLM server
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --hidden-states-path /tmp/hidden_states \
  -- --port 8000 --tensor-parallel-size 2

# Step 3: Train with online hidden states generation
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./prepared_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

---

## See Also

- [Getting Started Guide](../user_guide/getting_started.md)
- [Training Tutorial](../user_guide/tutorials/train_eagle3_online.md)
- [Features: Training](../user_guide/features/training.md)
- [Features: Data Preparation](../user_guide/features/prepare_data.md)
- [vLLM CLI Reference](https://docs.vllm.ai/en/latest/cli/)
