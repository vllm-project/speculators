# Response Regeneration

This tutorial walks you through regenerating assistant responses in an existing dataset using a target model served by vLLM. The resulting dataset pairs the original user prompts with freshly generated responses (on-policy data), and is the recommended starting point for speculator training: the drafter learns to predict what the target model actually generates, not what the dataset's original authors wrote. Training directly on the dataset's original responses (off-policy) is a cheaper fallback, since it skips a full target-model pass over the data, but costs acceptance length at inference time.

## Overview

**Time required:** ~10 mins on 2x H100 GPUs (for 1K samples)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- `vllm` installed (`uv pip install "vllm>=0.19.1"`)

## Step 1: Run the Pipeline

The simplest way to regenerate responses is using the `run_all.sh` script, which handles starting a vLLM server, running the regeneration, and stopping the server.

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dataset magpie \
  --limit 1000
```

This will:

1. Start a vLLM server with the specified model
2. Extract prompts from the dataset and generate new responses
3. Save results to a JSONL file (e.g., `magpie_Llama-3.3-70B-Instruct.jsonl`)
4. Stop the server

### Multi-GPU Configurations

For larger models, use data parallelism and/or tensor parallelism:

```bash
# Llama 3.3 70B on 8 GPUs (4 data-parallel replicas with TP=2)
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 4 --tp-size 2 \
  --dataset magpie

# Select specific GPUs
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --gpus 0,1,2,4 --tp-size 4 \
  --dataset magpie
```

## Step 2: Verify the Output

The output is a JSONL file with one pre-tokenized row per target generation. `loss_mask` is `0` over the prompt the target conditioned on and `1` over the tokens it generated, so training needs no further masking:

```json
{
  "id": "conv-abc_gen0",
  "primary_id": "conv-abc",
  "input_ids": [151644, 872, ...],
  "loss_mask": [0, 0, ..., 1, 1],
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "is_tool_call": false,
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```

Each assistant turn produces at least one row — and more when the target calls a tool, since every call is its own generation — so expect more lines than input conversations. `conversations` is a review-only twin of `input_ids`; training drops it.

Check that the output looks correct:

```bash
# Count completed rows
wc -l magpie_Llama-3.3-70B-Instruct.jsonl

# Inspect first row
head -1 magpie_Llama-3.3-70B-Instruct.jsonl | python -m json.tool
```

## Step 3: Use the Data for Training

The output JSONL can be passed directly to `prepare_data.py` for speculator training:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data ./magpie_Llama-3.3-70B-Instruct.jsonl \
  --output ./output \
  --seq-length 8192
```

## Advanced: Manual Control

If you prefer to manage the vLLM server yourself (e.g., to reuse a server across multiple runs), you can run the regeneration script directly:

```bash
# 1. Start vLLM server
vllm serve "meta-llama/Llama-3.3-70B-Instruct" \
  --data-parallel-size 4 --tensor-parallel-size 2 \
  --port 8000

# 2. Run regeneration (model auto-detected from server)
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --limit 1000

# 3. Stop server when done (Ctrl+C)
```

### Resuming Interrupted Processing

If processing is interrupted, use the `--resume` flag to skip already-processed rows:

```bash
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --outfile magpie_Llama-3.3-70B-Instruct.jsonl \
  --resume
```

### Keeping the Server Running

Use `--keep-server` with `run_all.sh` to leave the vLLM server running after processing, useful when running multiple regeneration jobs:

```bash
# First run - start server and keep it
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --dataset magpie --keep-server

# Second run - use the already-running server directly
python scripts/response_regeneration/script.py --dataset ultrachat
```

## Next Steps

After regenerating your dataset:

1. **Train a speculator** - See [Train Eagle-3 Online](train_eagle3_online.md) or [Train Eagle-3 Offline](train_eagle3_offline.md).
2. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
3. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)

For the full list of arguments for both scripts, see the [response_regeneration CLI reference](/cli/response_regeneration.md).
