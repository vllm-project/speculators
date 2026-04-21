# Train EAGLE-3 Model Online

This tutorial walks you through training an EAGLE-3 speculator model using **online training**, where hidden states are generated on-demand during the training process. This example uses `Qwen/Qwen3-8B` as the target model, but the process is the same for other models.

For a ready-to-run version of this tutorial, see [`examples/train/eagle3_qwen3_8b_sharegpt_online_5k.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/eagle3_qwen3_8b_sharegpt_online_5k.sh).

## Overview

**Time required:** ~30 mins (including training)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)

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
  --output ./output \
  --max-samples 5000 \
  --seq-length 8192
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (Built-in support for `sharegpt`, `ultrachat`. Otherwise provide a custom path to a jsonl file)
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing/getting started)
- `--seq-length` - Maximum sequence length

**Expected output:**

```
output/
├── data-00000-of-00002.arrow
├── data-00001-of-00002.arrow
├── dataset_info.json
├── state.json
└── token_freq.pt
```

**Time:** ~1-2 minutes for 5K samples

**Note:** This step is used to setup the dataset that will be used to train your model and is the same for both online and offline training. It's important that any data configuration choices are made at this stage. For example, limiting the data sample length, filtering out samples with limited assistant response tokens, handling multi-turn conversation responses, etc. For more information please see the [prepare_data.py cli reference](/cli/prepare_data.md).

## Step 2: Launch vLLM Server

Next launch vLLM configured for hidden states extraction:

```bash
# in vLLM venv

# Single GPU
python scripts/launch_vllm.py Qwen/Qwen3-8B

# Multiple GPUs with data parallelism (recommended)
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py \
  Qwen/Qwen3-8B -- --data-parallel-size 2 --port 8000
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

## Step 3: Start Training

Wait for vLLM to finish launching.

In a **separate terminal** on the same node, start the training process:

### Single-GPU Training

```bash
# in speculators venv
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing generate \
  --on-generate delete
```

### Multi-GPU Training (FSDP)

```bash
# in speculators venv
CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --standalone \
  --nproc_per_node 2 \
  scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing generate \
  --on-generate delete
```

**Key parameters:**

- `--vllm-endpoint` - vLLM server URL (localhost endpoint where vLLM is served)
- `--draft-vocab-size 32000` - Reduced vocabulary size to use
- `--epochs 5` - Number of training epochs
- `--lr 1e-4` - Learning rate
- `--total-seq-len 8192` - Maximum sequence length for training
- `--on-missing generate` - Generate hidden states on-the-fly if not cached
- `--on-generate delete` - Delete generated hidden states after use (saves disk space)

**Note:** There are a lot of configuration options available at this stage. We've attempted to set sensible defaults but please see the [train.py cli reference](/cli/train.md) to see all available options.

## Step 4: Inspect Checkpoints

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
├── 9/                          # Epoch 9 (final)
└── checkpoint_best -> 9/                # Symlink to lowest val loss checkpoint
```

Each checkpoint is a complete, self-contained speculator model ready for deployment in vLLM. The checkpoints also contain optimizer and learning rate scheduler states for resume training.

## Step 5: Test Your Model

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

### Issue: Out of Memory (vLLM)

**Error:**

```
vLLM: CUDA out of memory
```

**Solutions:**

```bash
# Increase GPU memory utilization
python scripts/launch_vllm.py model -- --gpu-memory-utilization 0.95

# Reduce max model length
python scripts/launch_vllm.py model -- --max-model-len 4096

# Use more GPUs
python scripts/launch_vllm.py model -- --tensor-parallel-size 2
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
   ls -lh ./training_data/
   ```

3. **Increase training time:**

   ```bash
   --epochs 20  # Train longer
   ```

### Issue: Inconsistent training utilization

**Symptoms:** Training logs are bursty, gpu utilization/power draw is inconsistent for training process.

**Solutions:**

1. **Redistribute GPUs between training and datagen processes:** If possible, increase the number of gpus assigned to vLLM datagen (i.e. the `launch_vllm.py` step) and reduce the number used during training. Inconsistent training performance typically means the training process is stuck waiting for new data samples to be generated.

2. **Consider switching to offline training:** Offline training pre-generates the hidden states for all samples and caches them on disk. Then during training the samples can just be loaded directly. If gpu resources are limited, this can be a better approach as it allows you to assign all gpus to data gen and then all to training.

## Advanced: Hybrid Training

Start with online, cache for later epochs:

```bash
# in speculators venv
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output \
  --hidden-states-path ./output/hidden_states \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate cache \
  --save-path ./output/checkpoints \
  --draft-vocab-size 32000 \
  --epochs 5 \
  --lr 1e-4 \
  --total-seq-len 8192
```

**First epoch:** Generates and caches hidden states to `./hidden_states/` **Subsequent epochs:** Uses cached hidden states.

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)
3. **Fine-tune further** - Use `--from-pretrained ./checkpoints/latest` to continue training
4. **Upload to HuggingFace** - Share your model with the community
