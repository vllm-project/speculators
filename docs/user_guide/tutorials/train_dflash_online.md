# Train DFlash Model Online

This tutorial walks you through training a DFlash speculator model using **online training**, where hidden states are generated on-demand during the training process. This example uses `Qwen/Qwen3-8B` as the target model, but the process is the same for other models. This tutorial follows the same structure as [Train Eagle-3 Online](train_eagle3_online.md). The key differences are:

- **Step 2 — vLLM launch:** DFlash requires explicitly passing `--target-layer-ids` to select which intermediate layers to extract hidden states from. Eagle-3 uses sensible defaults automatically.
- **Step 3 — Training:** DFlash introduces several additional parameters: `--speculator-type dflash`, `--block-size`, `--max-anchors`, and typically uses more draft layers (`--num-layers 5` vs 1 for Eagle-3).


For a ready-to-run version of this tutorial, see [`examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh).

## Overview

**Time required:** ~25 mins on 4x H100 GPUs (2 for vLLM, 2 for training)

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
  --output ./output/dflash_qwen3_8b_sharegpt \
  --max-samples 5000 \
  --seq-length 8192
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (built-in support for `sharegpt`, `ultrachat`, or a custom path to a jsonl file)
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing/getting started)
- `--seq-length` - Maximum sequence length

**Expected output:**

```
output/dflash_qwen3_8b_sharegpt/
├── data-00000-of-00002.arrow    #  ⎤
├── data-00001-of-00002.arrow    #  | Processed dataset on disk
├── dataset_info.json            #  |
├── state.json                   #  ⎦
└── token_freq.pt                # Token frequencies for vocab mapping
```

**Time:** ~14 seconds for 5K samples

**Note:** This step is used to setup the dataset that will be used to train your model and is the same for both online and offline training. For more information please see the [prepare_data.py cli reference](/cli/prepare_data.md).

## Step 2: Launch vLLM Server

During training, the drafter model takes internal hidden states from the verifier model as input. We use vLLM to serve the verifier and extract these hidden states. The `launch_vllm.py` script is a lightweight wrapper that sets up the right CLI arguments for vLLM to enable hidden state extraction.

For DFlash, you must explicitly specify which target layers to extract hidden states from using `--target-layer-ids`:

```bash
# in vLLM venv
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py Qwen/Qwen3-8B \
  --target-layer-ids 2 18 33 \
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

**Note:** The `--target-layer-ids` must match what you pass to the training step. These specify which intermediate layers of the target model provide hidden states to the DFlash drafter. For more information on usage, please see the [launch_vllm.py cli reference](/cli/launch_vllm.md).

## Step 3: Start Training

Wait for vLLM to finish launching.

In a **separate terminal** on the same node, start the training process:

### Single-GPU Training

```bash
# in speculators venv
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./output/dflash_qwen3_8b_sharegpt \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dflash_qwen3_8b_sharegpt/checkpoints \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 8192 \
  --target-layer-ids 2 18 33 \
  --epochs 5 \
  --lr 3e-4 \
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
  --data-path ./output/dflash_qwen3_8b_sharegpt \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./output/dflash_qwen3_8b_sharegpt/checkpoints \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --draft-vocab-size 8192 \
  --target-layer-ids 2 18 33 \
  --epochs 5 \
  --lr 3e-4 \
  --total-seq-len 8192 \
  --on-missing generate \
  --on-generate delete
```

**Key DFlash-specific parameters:**

- `--speculator-type dflash` - Use the DFlash algorithm
- `--block-size 8` - Number of tokens predicted per block
- `--max-anchors 3072` - Maximum anchor positions during training
- `--num-layers 5` - Number of draft transformer layers
- `--draft-vocab-size 8192` - Reduced vocabulary size
- `--target-layer-ids 2 18 33` - Target model layers to extract hidden states from (must match vLLM config)
- `--on-missing generate` - Generate hidden states on-the-fly if not cached
- `--on-generate delete` - Delete generated hidden states after use (saves disk space)

**Note:** There are a lot of configuration options available at this stage. We've attempted to set sensible defaults but please see the [train.py cli reference](/cli/train.md) to see all available options.

## Step 4: Inspect Checkpoints

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
└── checkpoint_best -> 4/       # Symlink to lowest val loss checkpoint
```

Each checkpoint is a complete, self-contained speculator model ready for deployment in vLLM.

## Step 5: Test Your Model

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

With 5K ShareGPT samples and 5 epochs on Qwen3-8B (MT-Bench, 80 prompts, 2048 max output tokens):

| Metric            | Value        |
| ----------------- | ------------ |
| Acceptance rate   | 5.90%        |
| Acceptance length | 1.47         |
| Output throughput | 129.41 tok/s |

> **Note:** With just 5K samples, model performance will be limited. This is intended as a sanity check to verify that the pipeline is working and the model is learning. For production quality, train on significantly more data.

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)
3. **Fine-tune further** - Use `--from-pretrained ./checkpoints/latest` to continue training
4. **Upload to HuggingFace** - Share your model with the community
