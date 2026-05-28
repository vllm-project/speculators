# Train P-EAGLE Model Offline

This tutorial walks you through training a P-EAGLE (Parallel EAGLE) speculator model using **offline training**, where hidden states are pre-generated and cached before training begins. This example uses `Qwen/Qwen3-8B` as the target model, but the process is the same for other models. This tutorial follows the same structure as [Train Eagle-3 Offline](train_eagle3_offline.md).

For a ready-to-run version of this tutorial, see [`examples/train/peagle_qwen3_8b_sharegpt_online_5k.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/peagle_qwen3_8b_sharegpt_online_5k.sh).

## Overview

**Time required:** ~50 mins on 4x H100 GPUs (including data generation)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- Sufficient disk space for hidden states

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
  --model Qwen/Qwen3-8B \
  --data sharegpt \
  --output ./output/peagle_qwen3_8b_sharegpt \
  --max-samples 5000 \
  --seq-length 4096
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (built-in support for `sharegpt`, `ultrachat`, or a custom path to a jsonl file)
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing/getting started)
- `--seq-length` - Maximum sequence length

**Expected output:**

```
output/peagle_qwen3_8b_sharegpt/
├── data-00000-of-00002.arrow    #  ⎤
├── data-00001-of-00002.arrow    #  | Processed dataset on disk
├── dataset_info.json            #  |
├── state.json                   #  ⎦
└── token_freq.pt                # Token frequencies for vocab mapping
```

**Time:** ~30 seconds for 5K samples

**Note:** This step is used to setup the dataset that will be used to train your model and is the same for both online and offline training. For more information please see the [prepare_data.py cli reference](/cli/prepare_data.md).

## Step 2: Launch vLLM Server

During training, the drafter model takes internal hidden states from the verifier model as input. We use vLLM to serve the verifier and extract these hidden states. The `launch_vllm.py` script is a lightweight wrapper that sets up the right CLI arguments for vLLM to enable hidden state extraction.

```bash
# in vLLM venv
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --hidden-states-path ./output/peagle_qwen3_8b_sharegpt/hidden_states \
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

**Note:** For more information on usage, please see the [launch_vllm.py cli reference](/cli/launch_vllm.md).

## Step 3: Generate Hidden States Offline

Use `data_generation_offline.py` to pre-generate all hidden states:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output/peagle_qwen3_8b_sharegpt \
  --endpoint http://localhost:8000/v1 \
  --output ./output/peagle_qwen3_8b_sharegpt/hidden_states \
  --max-samples 5000 \
  --concurrency 32 \
  --validate-outputs
```

**Key parameters:**

- `--preprocessed-data` - Path to prepared data from Step 1
- `--endpoint` - vLLM server URL
- `--output` - Where to save hidden states
- `--max-samples` - Number of samples to generate
- `--concurrency` - Parallel requests to vLLM during data generation
- `--validate-outputs` - Verify file integrity (recommended)

**Expected output:**

```
output/peagle_qwen3_8b_sharegpt/hidden_states/
├── hs_0.safetensors             #  ⎤
├── hs_1.safetensors             #  |
├── hs_2.safetensors             #  | Cached hidden states (one per sample)
├── ...                          #  |
└── hs_4999.safetensors          #  ⎦
```

### Optimizing Generation Speed

**Increase concurrency:**

```bash
--concurrency 64  # Controls the number of concurrent requests to vllm as well as concurrent write to disk operations.
```

**Use more GPUs:**

```bash
# 8 GPUs with DP=8
python scripts/launch_vllm.py model -- --data-parallel-size 8
```

### Resuming Interrupted Generation

If generation is interrupted, simply ensure the vllm server is launched and rerun the same command:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output/peagle_qwen3_8b_sharegpt \
  --endpoint http://localhost:8000/v1 \
  --output ./output/peagle_qwen3_8b_sharegpt/hidden_states \
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
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/peagle_qwen3_8b_sharegpt \
  --hidden-states-path ./output/peagle_qwen3_8b_sharegpt/hidden_states \
  --save-path ./output/peagle_qwen3_8b_sharegpt/checkpoints \
  --speculator-type peagle \
  --num-layers 4 \
  --num-depths 4 \
  --down-sample-ratio 0.7 \
  --down-sample-ratio-min 0.2 \
  --no-norm-before-residual \
  --scheduler-type cosine \
  --epochs 5 \
  --lr 6e-4 \
  --total-seq-len 4096 \
  --on-missing raise
```

### Multi-GPU Training (FSDP)

```bash
# in speculators venv
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node 2 \
  scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/peagle_qwen3_8b_sharegpt \
  --hidden-states-path ./output/peagle_qwen3_8b_sharegpt/hidden_states \
  --save-path ./output/peagle_qwen3_8b_sharegpt/checkpoints \
  --speculator-type peagle \
  --num-layers 4 \
  --num-depths 4 \
  --down-sample-ratio 0.7 \
  --down-sample-ratio-min 0.2 \
  --no-norm-before-residual \
  --scheduler-type cosine \
  --epochs 5 \
  --lr 6e-4 \
  --total-seq-len 4096 \
  --on-missing raise
```

**Key P-EAGLE-specific parameters:**

- `--speculator-type peagle` - Use the P-EAGLE algorithm
- `--num-depths 4` - Number of parallel prediction depths (multi-token prediction)
- `--down-sample-ratio 0.7` - Initial down-sampling ratio for COD sampling
- `--down-sample-ratio-min 0.2` - Minimum down-sampling ratio for COD sampling
- `--no-norm-before-residual` - Disable normalization before residual connections
- `--scheduler-type cosine` - Use cosine learning rate schedule

**Note:** There are a lot of configuration options available at this stage. We've attempted to set sensible defaults but please see the [train.py cli reference](/cli/train.md) to see all available options.

## Step 6: Inspect Checkpoints

After training, your checkpoints directory contains:

```
checkpoints/
├── 0/                          # Epoch 0
│   ├── config.json             #   Model architecture config
│   ├── model.safetensors       #   Model weights
│   ├── optimizer_state_dict.pt #   ⎤ Training state for
│   └── scheduler_state_dict.pt #   ⎦ resuming training
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

While the model is served, in a separate window run:

```bash
# in vllm venv
vllm chat --url http://localhost:8000/v1
```

### Verify Speculative Decoding

Check vLLM logs for speculative decoding metrics.

## Expected Results

With 5K ShareGPT samples and 5 epochs on Qwen3-8B (SpecBench, 80 prompts, 256 output tokens):

| Metric            | Value  |
| ----------------- | ------ |
| Acceptance rate   | 13.35% |
| Acceptance length | 1.53   |

Per-position acceptance:

| Position | Acceptance |
| -------- | ---------- |
| 0        | 40.84%     |
| 1        | 10.84%     |
| 2        | 1.58%      |
| 3        | 0.15%      |

> **Note:** With just 5K samples, model performance will be limited. This is intended as a sanity check to verify that the pipeline is working and the model is learning. For production quality, train on significantly more data.

## Common Issues & Solutions

### Issue: Out of Memory (Training)

**Error:**

```
torch.cuda.OutOfMemoryError
```

**Solutions:**

```bash
# Reduce sequence length
python scripts/train.py --total-seq-len 2048 ...

# Consider different model configurations
# e.g. fewer draft layers or depths
python scripts/train.py --num-layers 2 --num-depths 2
```

### Issue: Training Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**

1. **Lower learning rate:**

   ```bash
   --lr 1e-4  # Try lower LR
   ```

2. **Check data quality:**

   ```bash
   # Verify preprocessing succeeded
   ls -lh ./output/peagle_qwen3_8b_sharegpt/
   ```

3. **Increase training time:**

   ```bash
   --epochs 20  # Train longer
   ```

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)
3. **Fine-tune further** - Use `--from-pretrained ./checkpoints/checkpoint_best` to continue training
4. **Upload to HuggingFace** - Share your model with the community
