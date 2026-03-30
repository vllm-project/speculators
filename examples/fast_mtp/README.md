# FastMTP Finetuning Guide — Qwen3-Next

FastMTP finetunes the Multi-Token Prediction (MTP) head that Qwen3-Next ships with, using a lightweight teacher-forcing loss on captured verifier hidden states. The finetuned weights are stitched back into the original model so vLLM can load the result without any extra configuration.

On GSM8K, one epoch of finetuning raises mean accepted tokens from **2.01 → 2.46 (+22.5%)**.

All commands are run from the repo root.

______________________________________________________________________

## Prerequisites

```bash
# Install the speculators library with data-generation dependencies
pip install -e '.[datagen]'
```

GPU requirements by step:

| Step              | Minimum         | Notes                        |
| ----------------- | --------------- | ---------------------------- |
| 0 — Convert       | CPU only        | Extracts ~300 MB of weights  |
| 1 — Responses     | 4–8× A100 80 GB | vLLM with tensor parallelism |
| 2 — Hidden states | 4–8× A100 80 GB | Prefill-only, same TP setup  |
| 3 — Finetune      | 1–4× A100 80 GB | Single-GPU or FSDP           |
| 4 — Stitch        | CPU only        | File copies + index update   |
| 5 — Deploy        | 4× A100 80 GB   | vLLM speculative decoding    |

______________________________________________________________________

## Overview

```
Step 0  Convert checkpoint    extract MTP weights → speculators format (one-time)
Step 1  Generate responses    regenerate GSM8K answers with Qwen3-Next via vLLM
Step 2  Capture hidden states  prefill-only pass → .pt files with last-layer activations
Step 3  Finetune              train the MTP head on the captured dataset
Step 4  Stitch weights        merge finetuned weights back into the verifier
Step 5  Deploy                load the stitched model in vLLM
```

Paths used throughout:

```bash
VERIFIER=Qwen/Qwen3-Next-80B-A3B-Instruct
SPECULATOR=Qwen3-Next-80B-A3B-Instruct_mtp_speculator
RESPONSES=/mnt/data/datasets/gsm8k_Qwen3-Next-80B-A3B-Instruct.jsonl
HIDDEN_STATES=/mnt/data/datasets/qwen3next-gsm8k-hidden-states
FINETUNED=output/qwen3next_gsm8k_finetuned
STITCHED=output/qwen3next_gsm8k_stitched
```

______________________________________________________________________

## Step 0 — Convert checkpoint (one-time)

Extracts the MTP layer weights from the Qwen3-Next checkpoint and wraps them in a speculators config. The extracted weights keep their original key names, so no remapping is needed at any later stage.

**From HuggingFace (downloads automatically):**

```bash
python examples/fast_mtp/convert_checkpoint.py \
    --output-dir $SPECULATOR
```

**From a local snapshot:**

```bash
python examples/fast_mtp/convert_checkpoint.py \
    --model /path/to/Qwen3-Next-80B-A3B-Instruct \
    --output-dir $SPECULATOR
```

Use `--cache-dir` to control where HuggingFace stores the download.

Output:

```
$SPECULATOR/
    config.json          # FastMTPConfig with full transformer_layer_config
    model.safetensors    # MTP layer weights only (~300 MB)
```

______________________________________________________________________

## Step 1 — Generate responses

Serves Qwen3-Next via vLLM and regenerates answers for every GSM8K train question. Skip this step if you already have a JSONL file.

```bash
cd scripts/response_regeneration && \
./run_all.sh \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --dataset gsm8k \
    --gpus 0,1,2,3 \
    --tp-size 4 \
    --max-tokens 2048 \
    --outfile $RESPONSES \
    2>&1 | tee local/logs/gsm8k_step1_regeneration.log
```

Output: `$RESPONSES` — JSONL in ShareGPT conversations format (7 473 samples for GSM8K train).

______________________________________________________________________

## Step 2 — Capture hidden states

Tokenizes the JSONL, builds per-token loss masks, then runs a prefill-only vLLM pass to capture the last verifier hidden layer for every token position.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/fast_mtp/generate_dataset.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --data-path $RESPONSES \
    --output-dir $HIDDEN_STATES \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    2>&1 | tee local/logs/gsm8k_step2_hidden_states.log
```

Output: `$HIDDEN_STATES/` — one `data_N.pt` file per sample plus a `sample_lengths.json` index. Each `.pt` file contains:

```python
{
    "input_ids":     Tensor[seq_len],          # long
    "hidden_states": Tensor[seq_len, 2048],    # float32, last verifier layer
    "loss_mask":     Tensor[seq_len],          # long, 1 = assistant tokens only
}
```

______________________________________________________________________

## Step 3 — Finetune

### Single GPU

```bash
python examples/fast_mtp/finetune.py \
    --speculator-path $SPECULATOR \
    --data-dir $HIDDEN_STATES \
    --output-dir $FINETUNED \
    --max-len 4096 \
    --lr 5e-5 \
    --num-epochs 3 \
    --batch-size 64 \
    --train-ratio 0.9 \
    --scheduler-type cosine \
    --save-best \
    2>&1 | tee local/logs/gsm8k_step3_finetune.log
```

### Multi-GPU (torchrun)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node=4 \
    examples/fast_mtp/finetune.py \
    --speculator-path $SPECULATOR \
    --data-dir $HIDDEN_STATES \
    --output-dir $FINETUNED \
    --max-len 4096 \
    --lr 5e-5 \
    --num-epochs 3 \
    --batch-size 16 \
    --train-ratio 0.9 \
    --scheduler-type cosine \
    --save-best \
    2>&1 | tee local/logs/gsm8k_step3_finetune.log
```

Key arguments:

| Argument                   | Default          | Notes                                    |
| -------------------------- | ---------------- | ---------------------------------------- |
| `--lr`                     | `5e-5`           | Learning rate                            |
| `--num-epochs`             | `3`              | Epochs to train                          |
| `--batch-size`             | `1`              | Per-GPU batch size                       |
| `--train-ratio`            | `0.9`            | Train / val split                        |
| `--scheduler-type`         | `cosine`         | `cosine`, `linear`, or `none`            |
| `--scheduler-warmup-steps` | `None`           | Steps for LR warmup                      |
| `--step-weights`           | `0.51 0.31 0.18` | Per-step MTP loss weights (β=0.6 decay)  |
| `--save-best`              | off              | Keep only the lowest-val-loss checkpoint |
| `--checkpoint-freq`        | `1`              | Save every N epochs                      |

With `--save-best`, the best checkpoint is written to:

```
$FINETUNED/best/model.safetensors
```

Without it, each epoch saves to:

```
$FINETUNED/epoch_N/model.safetensors
```

______________________________________________________________________

## Step 4 — Stitch weights

Writes the finetuned MTP weights (already in vLLM-compatible `mtp.*` key format) into a new shard, copies all original verifier shards into the output directory, and updates `model.safetensors.index.json` to route `mtp.*` keys to the new shard.

```bash
python examples/fast_mtp/stitch_weights.py \
    --checkpoint $FINETUNED/best/model.safetensors \
    --verifier /path/to/Qwen3-Next-80B-A3B-Instruct \
    --output-dir $STITCHED
```

Output layout:

```
$STITCHED/
    model.safetensors.index.json      # updated: mtp.* → new shard
    mtp_finetuned.safetensors         # finetuned MTP weights only (~300 MB)
    config.json                       # verifier config (unchanged)
    model-00001-of-00041.safetensors  # full copies of all verifier shards
    ...
```

The output is a self-contained model directory — upload directly to HuggingFace:

```bash
huggingface-cli upload <your-org>/Qwen3-Next-80B-A3B-Instruct-FastMTP $STITCHED
```

______________________________________________________________________

## Step 5 — Deploy with vLLM

The stitched directory is a standard Qwen3-Next model. vLLM picks up the finetuned MTP head automatically via the `num_nextn_predict_layers` field in `config.json`.

> **Required vLLM flags:** `--tokenizer-mode auto` and `--no-enable-chunked-prefill`

### Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="output/qwen3next_gsm8k_stitched",
    tensor_parallel_size=4,
    tokenizer_mode="auto",
    enable_chunked_prefill=False,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 3,
    },
)

outputs = llm.generate(
    ["What is 25 times 48?"],
    SamplingParams(temperature=0.0, max_tokens=512),
)
print(outputs[0].outputs[0].text)
```

### Online server

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve output/qwen3next_gsm8k_stitched \
    --tensor-parallel-size 4 \
    --tokenizer-mode auto \
    --no-enable-chunked-prefill \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

______________________________________________________________________

## Adapting to a different dataset

Replace Step 1 with any ShareGPT-format JSONL (each record has a `conversations` list of `{"from": "human"/"gpt", "value": "..."}` objects). Pass the path to `--data-path` in Step 2. Everything downstream is dataset-agnostic.
