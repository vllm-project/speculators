# Train MTP Model Offline

This tutorial walks you through finetuning an MTP (Multi-Token Prediction) speculator model using **offline training**, where hidden states are pre-generated and cached before training begins. This example uses `Qwen/Qwen3-Next-80B-A3B-Instruct` as the target model, but the process is the same for any model with native MTP support (e.g. Qwen3.5).

Unlike Eagle-3, DFlash, or P-EAGLE which train draft models from scratch, MTP finetuning starts from the model's **native MTP head** -- you convert it to speculators format, finetune on domain-specific data, and stitch the improved weights back into the verifier checkpoint.

For a ready-to-run version of this tutorial, see [`examples/train/mtp_qwen3next_gsm8k_offline.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/mtp_qwen3next_gsm8k_offline.sh).

## Overview

**Time required:** Varies by model size. ~30 mins for a small model on 2x H100 GPUs.

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- A model with native MTP support (e.g. `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-0.8B`)
- Sufficient disk space for hidden states

## Step 0: Setup Your Environment

Create two virtual environments (recommended to keep separate so dependencies don't conflict):

```bash
# Speculators venv (for data prep and training)
uv venv speculators_venv
source speculators_venv/bin/activate
uv pip install "speculators>=0.6.0"
```

```bash
# vLLM venv (for serving the target model)
uv venv vllm_venv
source vllm_venv/bin/activate
uv pip install "vllm>=0.22.0"
```

Note: if you are using an experiment tracker (e.g. trackio, wandb, tensorboard), install it in the speculators venv manually.

## Step 1: Convert Native MTP Head

Extract the model's native MTP layers into speculators format. This is unique to MTP -- other algorithms train from scratch.

```bash
# in speculators venv
python -c "
from speculators.convert import convert_model
convert_model(
    model='Qwen/Qwen3-Next-80B-A3B-Instruct',
    verifier='Qwen/Qwen3-Next-80B-A3B-Instruct',
    algorithm='mtp',
    output_path='./output/converted_mtp',
    num_speculative_steps=3,
)
"
```

**Parameters explained:**

- `model` - The source model containing native MTP layers (same as verifier for native MTP)
- `verifier` - The verifier model to attach
- `algorithm` - Must be `"mtp"` for MTP conversion
- `output_path` - Where to save the converted speculators checkpoint
- `num_speculative_steps` - Number of tokens to predict per step (default: 3)

**Expected output:**

```
output/converted_mtp/
├── config.json             # MTPSpeculatorConfig
└── model.safetensors       # Extracted MTP layer weights + embed_tokens + lm_head
```

## Step 2: Prepare Your Data

Preprocess your training dataset:

```bash
# in speculators venv
python scripts/prepare_data.py \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --data sharegpt \
  --output ./output/mtp_qwen3next \
  --max-samples 5000 \
  --seq-length 8192
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (built-in support for `sharegpt`, `ultrachat`, `gsm8k`, or a custom path to a jsonl file)
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing/getting started)
- `--seq-length` - Maximum sequence length

**Expected output:**

```
output/mtp_qwen3next/
├── data-00000-of-00002.arrow    #  ⎤
├── data-00001-of-00002.arrow    #  | Processed dataset on disk
├── dataset_info.json            #  |
├── state.json                   #  ⎦
└── token_freq.pt                # Token frequencies
```

**Note:** This step is the same for all speculator types. For more information please see the [prepare_data.py cli reference](/cli/prepare_data.md).

## Step 3: Launch vLLM Server

Start vLLM to serve the verifier for hidden state extraction:

```bash
# in vLLM venv
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
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

## Step 4: Generate Hidden States Offline

Use `data_generation_offline.py` to pre-generate all hidden states:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output/mtp_qwen3next \
  --endpoint http://localhost:8000/v1 \
  --output ./output/mtp_qwen3next/hidden_states \
  --max-samples 5000 \
  --concurrency 32 \
  --validate-outputs
```

**Key parameters:**

- `--preprocessed-data` - Path to prepared data from Step 2
- `--endpoint` - vLLM server URL
- `--output` - Where to save hidden states
- `--max-samples` - Number of samples to generate
- `--concurrency` - Parallel requests to vLLM during data generation
- `--validate-outputs` - Verify file integrity (recommended)

**Note:** For more information on usage, please see the [data_generation_offline.py cli reference](/cli/data_generation_offline.md).

## Step 5: Stop vLLM Server

After hidden states generation is complete, stop the vLLM server:

```bash
# Press Ctrl+C in the vLLM terminal
```

You don't need vLLM running during offline training.

## Step 6: Train with Cached Hidden States

### Single-GPU Training

```bash
# in speculators venv
python scripts/train.py \
  --verifier-name-or-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --data-path ./output/mtp_qwen3next \
  --hidden-states-path ./output/mtp_qwen3next/hidden_states \
  --save-path ./output/mtp_qwen3next/checkpoints \
  --speculator-type mtp \
  --from-pretrained ./output/converted_mtp \
  --step-weight-beta 0.6 \
  --epochs 3 \
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
  --verifier-name-or-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --data-path ./output/mtp_qwen3next \
  --hidden-states-path ./output/mtp_qwen3next/hidden_states \
  --save-path ./output/mtp_qwen3next/checkpoints \
  --speculator-type mtp \
  --from-pretrained ./output/converted_mtp \
  --step-weight-beta 0.6 \
  --epochs 3 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing raise
```

**Key MTP-specific parameters:**

- `--speculator-type mtp` - Use the MTP algorithm
- `--from-pretrained ./output/converted_mtp` - Path to the converted MTP checkpoint from Step 1
- `--step-weight-beta 0.6` - Exponential decay factor for per-step loss weights (default: 0.6)
- `--on-missing raise` - Raise an error if hidden states are missing (offline mode)

**Note:** MTP does not require `--draft-vocab-size` -- it uses the full verifier vocabulary automatically. The number of speculative steps is read from the converted checkpoint's config, so `--num-speculative-steps` is also not needed when using `--from-pretrained`.

**Note:** There are a lot of configuration options available at this stage. We've attempted to set sensible defaults but please see the [train.py cli reference](/cli/train.md) to see all available options.

## Step 7: Stitch Finetuned Weights

After training, stitch the finetuned MTP weights back into the original verifier checkpoint. This produces a self-contained checkpoint deployable on vLLM with native MTP speculative decoding.

```bash
# in speculators venv
python scripts/stitch_mtp.py \
  ./output/mtp_qwen3next/checkpoints/checkpoint_best \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
  --output-path ./output/stitched
```

**Parameters explained:**

- First argument: path to the finetuned MTP checkpoint
- Second argument: verifier model (HuggingFace ID or local path)
- `--output-path` - Where to save the stitched checkpoint (defaults to `{verifier-name}-stitched`)

## Step 8: Test Your Model

### Quick Test with vLLM

Serve the stitched checkpoint with MTP speculative decoding enabled:

```bash
# in vllm venv
vllm serve ./output/stitched \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --no-enable-chunked-prefill \
  --port 8000
```

See [vLLM Recipes](https://recipes.vllm.ai/) for more deployment options and configurations.

### Chat with the served model

While the model is served, in a separate window run:

```bash
# in vllm venv
vllm chat --url http://localhost:8000/v1
```

### Verify Speculative Decoding

Check vLLM logs for speculative decoding metrics.

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - See [vLLM Recipes](https://recipes.vllm.ai/) for deployment commands
3. **Fine-tune further** - Use `--from-pretrained ./output/mtp_qwen3next/checkpoints/checkpoint_best` to continue training
4. **Upload to HuggingFace** - Share your model with the community
