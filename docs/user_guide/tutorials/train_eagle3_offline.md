# Train EAGLE-3 Model Offline

This tutorial walks you through training an EAGLE-3 speculator model using **offline training**, where hidden states are pre-generated and cached before training begins. This example uses `meta-llama/Llama-3.1-8B-Instruct` as the target model, but the process is the same for other models.

For a ready-to-run version of this tutorial, see [`examples/train/eagle3_llama3_8b_sharegpt_offline_5k.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/eagle3_llama3_8b_sharegpt_offline_5k.sh).

## Overview

**Time required:** ~10 mins on 2x H100 GPUs (including data generation)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- Sufficient disk space for hidden states (~1.6 TB for 50k samples for Llama-3.1-8B, avg seq len 1024)

## Step 0: Setup Your Environment

Create two virtual environments (recommended to keep separate so dependencies don't conflict):

```bash
# Speculators venv (for data prep and training)
uv venv speculators_venv
source speculators_venv/bin/activate
uv pip install "speculators>=0.5.0"
```

```bash
# vLLM venv (for serving the target model)
uv venv vllm_venv
source vllm_venv/bin/activate
uv pip install "vllm>=0.18"
```

Note: if you are using an experiment tracker (e.g. trackio, wandb, tensorboard), install it in the speculators venv manually.

## Step 1: Prepare Your Data

First, preprocess your training dataset:

```bash
# in speculators venv
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data ultrachat \
  --output ./output \
  --max-samples 5000 \
  --seq-length 8192
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (Built-in support for `sharegpt`, `ultrachat`. Otherwise provide a custom path to a jsonl file). Can be supplied multiple times to combine multiple datasets.
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing/getting started)
- `--seq-length` - Maximum sequence length

**Expected output:**

```
output/
├── data-*.arrow files
├── dataset_info.json
├── state.json
└── token_freq.pt
```

**Time:** ~5 minutes for 50K samples

**Note:** This step is used to setup the dataset that will be used to train your model and is the same for both online and offline training. It's important that any data configuration choices are made at this stage. For example, limiting the data sample length, filtering out samples with limited assistant response tokens, handling multi-turn conversation responses, etc. For more information please see the [prepare_data.py cli reference](/cli/prepare_data.md).

## Step 2: Launch vLLM Server

Next launch vLLM configured for hidden states extraction:

```bash
# in vLLM venv
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 2 --port 8000
```

**The `--` separator:** Anything after `--` is passed directly to vLLM. Common options:

- `--data-parallel-size 4` - Use 4 data parallel instances
- `--tensor-parallel-size 2` - Group GPUs in pairs for tensor parallelism
- `--port 8000` - Specify port (default: 8000)
- `--gpu-memory-utilization 0.9` - GPU memory to use

**Wait for server to start:**

```
INFO:     Started server process [2140110]
INFO:     Waiting for application startup.
INFO:     Application startup complete
```

**Note:** This stage is also when you must decide which layer ids to extract from vLLM. For eagle3, if you don't pass in `--target-layer-ids`, this script will use sensible default values. For more information on usage, please see the [launch_vllm.py cli reference](/cli/launch_vllm.md).

## Step 3: Generate Hidden States Offline

Use `data_generation_offline.py` to pre-generate all hidden states:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output \
  --endpoint http://localhost:8000/v1 \
  --output ./output/hidden_states \
  --max-samples 5000 \
  --concurrency 32 \
  --validate-outputs
```

**Key parameters:**

- `--preprocessed-data` - Path to prepared data from Step 1
- `--endpoint` - vLLM server URL
- `--output` - Where to save hidden states
- `--max-samples` - Number of samples to generate
- `--concurrency` - Parallel requests to vLLM
- `--validate-outputs` - Verify file integrity (recommended for production)

**Expected output:**

```
output/hidden_states/
├── hs_0.safetensors
├── hs_1.safetensors
├── hs_2.safetensors
├── ...
└── hs_4999.safetensors
```

### Optimizing Generation Speed

**Increase concurrency:**

```bash
--concurrency 64
```

**Use more GPUs:**

```bash
# 8 GPUs with DP=8
python scripts/launch_vllm.py model -- --data-parallel-size 8
```

**Skip validation:**

Validation loads every single generated output file and confirms that the token ids match the sent request and the hidden states length matches expectation. This is generally not required but is a good sanity check. Turn this off to skip loading generated samples during data gen.

```bash
# Omit --validate-outputs argument
```

### Resuming Interrupted Generation

If generation is interrupted, simply ensure the vllm server is launched and rerun the same command:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output \
  --endpoint http://localhost:8000/v1 \
  --output ./output/hidden_states \
  --max-samples 5000 \
  --concurrency 32
```

The script automatically detects existing `hs_*.safetensors` files and skips them, continuing where you left off.

**Note:** For more information on usage, please see the [data_generation_offline.py cli reference](/cli/data_generation_offline.md).

## Step 4: Stop vLLM Server

After hidden states generation is complete, stop the vLLM server:

```bash
# Press Ctrl+C in the vLLM terminal
```

You don't need vLLM running during offline training.

## Step 5: Train with Cached Hidden States

### Single-GPU Training

```bash
# in speculators venv
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./output \
  --hidden-states-path ./output/hidden_states \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing raise
```

### Multi-GPU Training (FSDP)

```bash
# in speculators venv
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node 2 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./output \
  --hidden-states-path ./output/hidden_states \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing raise
```

**Key parameters:**

- `--hidden-states-path` - Points to cached hidden states
- `--on-missing raise` - Fail if any hidden states are missing (recommended). Alternatives are `skip` and `warn` which both skip the missing sample, with the latter raising a warning.
- `--draft-vocab-size 32000` - Reduced vocabulary size to use
- `--epochs 5` - Number of training epochs
- `--lr 1e-4` - Learning rate
- `--total-seq-len 8192` - Maximum sequence length

**Note:** There are a lot of configuration options available at this stage. We've attempted to set sensible defaults but please see the [train.py cli reference](/cli/train.md) to see all available options.

## Step 6: Inspect Checkpoints

After training, your checkpoints directory contains:

```
checkpoints/
├── 0/                          # Epoch 0
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   ├── optimizer_state_dict.pt
│   └── scheduler_state_dict.pt
├── 1/                          # Epoch 1
├── ...
├── 4/                          # Epoch 4 (final)
└── checkpoint_best -> 4/       # Symlink to lowest val loss checkpoint
```

Each checkpoint is a complete, self-contained speculator model ready for deployment in vLLM. The checkpoints also contain optimizer and learning rate scheduler states for resume training.

## Step 7: Test Your Model

### Quick Test with vLLM

Stop the training vLLM server (Ctrl+C), then serve your speculator:

```bash
# in vllm venv
vllm serve ./checkpoints/checkpoint_best --port 8000
```

### Chat with the served model

While the model in served, in a separate window run:

```bash
# in vllm venv
vllm chat --url http://localhost:8000/v1
```

### Verify Speculative Decoding

Check vLLM logs for speculative decoding metrics

## Estimating Disk Space Requirements

```python
# For Llama-3.1-8B:
# avg_seq_len × num_layers × hidden_size × dtype_bytes
# 8192 × 4 × 4096 × 2 = ~268 MB per sample

# For 50K samples: ~13 TB
# For 10K samples: ~2.6 TB
# For 1K samples: ~260 GB
```

## Common Issues & Solutions

### Issue: Out of Memory (Training)

**Error:**

```
torch.cuda.OutOfMemoryError
```

**Solutions:**

```bash
# Reduce sequence length
python scripts/train.py --total-seq-len 4096 ...

# Consider different model configurations
# e.g. less draft layers
python scripts/train.py --num-layers 1
```

### Issue: Training Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**

1. **Lower learning rate:**

   ```bash
   --lr 1e-5  # Try lower LR
   ```

2. **Check data quality:**

   ```bash
   # Verify preprocessing succeeded
   ls -lh ./output/
   ```

3. **Increase training time:**

   ```bash
   --epochs 20  # Train longer
   ```

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)
3. **Fine-tune further** - Use `--from-pretrained ./checkpoints/latest` to continue training
4. **Upload to HuggingFace** - Share your model with the community
