# Finetune MTP Head (Offline)

This tutorial walks you through finetuning the native MTP (Multi-Token Prediction) head from a Qwen3-Next checkpoint on domain-specific data using the [FastMTP](https://arxiv.org/abs/2509.18362) methodology. The finetuned head is stitched back into the verifier checkpoint for deployment on vLLM.

This example uses `Qwen/Qwen3-Next-80B-A3B-Instruct` with GSM8K as the reference dataset. The same process works for Qwen3.5 and Qwen3.5-MoE models.

For a ready-to-run version of this tutorial, see [`examples/train/mtp_qwen3next_gsm8k_offline.sh`](https://github.com/vllm-project/speculators/blob/main/examples/train/mtp_qwen3next_gsm8k_offline.sh).

## Overview

The MTP finetuning pipeline has 7 stages. Only the MTP head (~100M-400M params) is loaded during training -- the full 80B+ verifier is never loaded.

```
Response Regen → Data Prep → Hidden States → Convert → Finetune → Stitch → Deploy
     (1)           (2)          (3)           (4)        (5)        (6)      (7)
```

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s) (8x H100 recommended for the 80B verifier)
- Sufficient disk space for hidden states

## Step 0: Setup Your Environment

Create two virtual environments to keep dependencies separate:

```bash
# Speculators venv (for data prep, conversion, training, stitching)
uv venv speculators_venv
source speculators_venv/bin/activate
uv pip install "speculators>=0.5.0"
```

```bash
# vLLM venv (for serving the verifier and deployment)
uv venv vllm_venv
source vllm_venv/bin/activate
uv pip install "vllm>=0.18"
```

## Step 1: Response Regeneration

Generate domain-specific training responses using your verifier model. This step produces a JSONL file with conversations that the model generated on your target domain.

```bash
# in vLLM venv
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --dataset gsm8k \
  --tp-size 8 \
  --limit 10000
```

This starts a vLLM server, generates responses, and saves them to a JSONL file. For more details, see the [Response Regeneration tutorial](response_regeneration.md).

If you already have a dataset in the right format (JSONL with conversations), you can skip this step and use your data directly.

## Step 2: Data Preparation

Tokenize the dataset with the model's chat template and create loss masks:

```bash
# in speculators venv
python scripts/prepare_data.py \
  --model "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --data gsm8k \
  --output ./output \
  --max-samples 10000 \
  --seq-length 8192
```

**Parameters:**

- `--model` -- The verifier model (used for tokenizer and chat template)
- `--data` -- Dataset name (`gsm8k`, `sharegpt`, `ultrachat`) or path to custom JSONL
- `--output` -- Where to save preprocessed data
- `--max-samples` -- Limit the number of samples (optional)
- `--seq-length` -- Maximum sequence length

**Expected output:**

```
output/
├── data-*.arrow            # Processed dataset
├── dataset_info.json
├── state.json
└── token_freq.pt           # Token frequencies
```

## Step 3: Hidden States Extraction

Extract the verifier's hidden states for each training sample. For MTP, you only need the **last hidden layer** -- this is the hidden state that the MTP head receives as input during inference.

First, launch the vLLM server configured for hidden state extraction:

```bash
# in vLLM venv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/launch_vllm.py \
  "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --target-layer-ids 64 \
  -- --tensor-parallel-size 8 --port 8000
```

!!! note "Why only the last layer?" Unlike Eagle-3, which concatenates hidden states from multiple intermediate layers, MTP takes a single hidden state vector from the verifier's final layer. Pass only the last layer ID to `--target-layer-ids` (e.g., `64` for Qwen3-Next-80B which has 64 layers). This also reduces the disk space needed for cached hidden states.

Then generate the hidden states:

```bash
# in speculators venv
python scripts/data_generation_offline.py \
  --preprocessed-data ./output \
  --endpoint http://localhost:8000/v1 \
  --output ./output/hidden_states \
  --max-samples 10000 \
  --concurrency 32 \
  --validate-outputs
```

After generation completes, stop the vLLM server (Ctrl+C) to free GPU memory for training.

**Expected output:**

```
output/hidden_states/
├── hs_0.safetensors
├── hs_1.safetensors
├── ...
└── hs_9999.safetensors
```

## Step 4: Model Conversion

Extract the MTP head from the native verifier checkpoint into a standalone Speculators model. This is done via the Python API:

```python
from speculators.convert.mtp import MTPConverter

converter = MTPConverter()
converter.convert(
    input_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
    output_path="./output/mtp_head",
    base_model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    num_speculative_steps=3,
)
```

**Parameters:**

- `input_path` -- HF model ID or local path to the verifier checkpoint
- `output_path` -- Where to save the extracted MTP head
- `base_model` -- HF model ID of the verifier (used for config metadata)
- `num_speculative_steps` -- Number of speculative tokens (default: 3)

The converter extracts only the MTP layer, `embed_tokens`, and `lm_head` from the checkpoint. For MoE models (Qwen3-Next, Qwen3.5-MoE), per-expert weights are automatically fused into packed tensors.

**Expected output:**

```
output/mtp_head/
├── config.json             # MTPConfig with speculators_config
└── model.safetensors       # MTP weights
```

## Step 5: Finetuning

Train the MTP head using the same `train.py` script used for Eagle-3, with `--speculator-type mtp`:

### Single-GPU Training

```bash
# in speculators venv
python scripts/train.py \
  --speculator-type mtp \
  --verifier-name-or-path "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --data-path ./output \
  --hidden-states-path ./output/hidden_states \
  --save-path ./output/checkpoints \
  --num-speculative-steps 3 \
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
  --speculator-type mtp \
  --verifier-name-or-path "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --data-path ./output \
  --hidden-states-path ./output/hidden_states \
  --save-path ./output/checkpoints \
  --num-speculative-steps 3 \
  --step-weight-beta 0.6 \
  --epochs 3 \
  --lr 1e-4 \
  --total-seq-len 8192 \
  --on-missing raise
```

**MTP-specific parameters:**

- `--speculator-type mtp` -- Selects MTP training mode
- `--num-speculative-steps` -- Number of speculative tokens to predict (default: 3)
- `--step-weight-beta` -- Exponential decay factor for step weights (default: 0.6). With beta=0.6 and 3 steps, the weights are `[0.51, 0.31, 0.18]`, giving earlier steps more weight

**Key differences from Eagle-3 training:**

- No `--draft-vocab-size` needed -- MTP uses the full verifier vocabulary
- No vocabulary mapping -- `embed_tokens` and `lm_head` are shared with the verifier (frozen during training)
- The verifier config is loaded directly from the HF model (no draft architecture selection)

**Expected output:**

```
output/checkpoints/
├── 0/                      # Epoch 0
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer_state_dict.pt
│   └── scheduler_state_dict.pt
├── 1/                      # Epoch 1
├── 2/                      # Epoch 2
└── checkpoint_best -> 0/   # Symlink to lowest val loss
```

## Step 6: Stitching

Reintegrate the finetuned MTP weights back into the full verifier checkpoint. This produces a self-contained checkpoint that vLLM can load directly.

```python
from speculators.convert.mtp import MTPStitcher

stitcher = MTPStitcher()
stitcher.stitch(
    finetuned_checkpoint="./output/checkpoints/checkpoint_best",
    verifier_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
    output_path="./output/stitched_model",
)
```

**Parameters:**

- `finetuned_checkpoint` -- Path to the best training checkpoint
- `verifier_path` -- Path to the original verifier checkpoint
- `output_path` -- Where to save the complete stitched checkpoint

The stitcher:

1. Loads only the trainable MTP weights (skipping frozen `embed_tokens`/`lm_head`)
2. Remaps keys from Speculators format (`mtp_layers.0.*`) back to native format (`mtp.*`)
3. Unfuses MoE expert weights (for MoE models)
4. Copies the full verifier checkpoint and replaces the MTP weight shards

The output is a complete, self-contained checkpoint that can be uploaded to HF Hub.

## Step 7: Deployment

Serve the stitched checkpoint on vLLM with MTP-based speculative decoding:

```bash
# in vLLM venv (8x H100 example)
vllm serve ./output/stitched_model \
  --tensor-parallel-size 8 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --no-enable-chunked-prefill
```

!!! warning "vLLM API changes frequently" The speculative decoding CLI flags in vLLM change across versions. Refer to [https://recipes.vllm.ai/](https://recipes.vllm.ai/) for the latest correct command for your vLLM version.

**Speculative config parameters:**

- `method` -- Use `qwen3_next_mtp` for Qwen3-Next models
- `num_speculative_tokens` -- Number of tokens to speculate per step (must be \<= `num_speculative_steps` used during training)

### Verify It Works

```bash
# in vLLM venv
vllm chat --url http://localhost:8000/v1
```

Check the vLLM server logs for speculative decoding metrics (acceptance rate, draft token counts).

## Next Steps

After finetuning your MTP head:

1. **Evaluate performance** -- See [Evaluating Performance](evaluating_performance.md)
2. **Upload to HF Hub** -- The stitched checkpoint is directly uploadable: `huggingface-cli upload <repo-id> ./output/stitched_model`
3. **Try different domains** -- Finetune on code, math, or other domain data to improve acceptance rates for your workload
