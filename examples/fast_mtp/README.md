# FastMTP End-to-End Guide

FastMTP finetunes the MTP (Multi-Token Prediction) head already present in
Qwen3-Next using a lightweight teacher-forcing loss, then stitches the updated
weights back into the verifier so vLLM can load them without any extra
configuration.

All commands are run from the repo root unless noted otherwise.

---

## Overview

```
Step 0  Convert checkpoint   extract MTP weights from Qwen3-Next → speculators format
Step 1  Generate responses   regenerate GSM8K answers with Qwen3-Next via vLLM
Step 2  Capture hidden states  prefill-only pass to get last-layer hidden states
Step 3  Finetune             train the MTP head on the captured dataset
Step 4  Stitch weights       merge finetuned MTP weights back into the verifier
Step 5  Deploy               load the stitched model in vLLM
```

Paths used throughout:

```
VERIFIER=Qwen/Qwen3-Next-80B-A3B-Instruct
SPECULATOR=Qwen3-Next-80B-A3B-Instruct_mtp_speculator
RESPONSES=/mnt/data/rahul-tuli/datasets/gsm8k_Qwen3-Next-80B-A3B-Instruct.jsonl
HIDDEN_STATES=/mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states
FINETUNED=output/qwen3next_gsm8k_finetuned
STITCHED=output/qwen3next_gsm8k_stitched
```

---

## Step 0 — Convert checkpoint (one-time)

Extracts the MTP head from Qwen3-Next-80B-A3B-Instruct and wraps it in the
speculators config format. Only needs to be done once; the result is already
committed at `Qwen3-Next-80B-A3B-Instruct_mtp_speculator/`.

```bash
python local/scripts/convert_checkpoints.py \
    --output-dir Qwen3-Next-80B-A3B-Instruct_mtp_speculator
```

The output directory contains:
- `config.json` — `FastMTPConfig` with full `transformer_layer_config`
- `model.safetensors` — extracted MTP layer weights

---

## Step 1 — Generate responses

Serves the verifier via vLLM and regenerates answers for every GSM8K train
question. Skip if you already have the JSONL.

```bash
cd scripts/response_regeneration && \
./run_all.sh \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --dataset gsm8k \
    --gpus 0,1,2,3 \
    --tp-size 4 \
    --max-tokens 2048 \
    --outfile /mnt/data/rahul-tuli/datasets/gsm8k_Qwen3-Next-80B-A3B-Instruct.jsonl \
    2>&1 | tee local/logs/gsm8k_step1_regeneration.log
```

Output: `$RESPONSES` — JSONL in ShareGPT conversations format
(7 473 samples for GSM8K train).

---

## Step 2 — Capture hidden states

Tokenizes the JSONL, builds loss masks, then runs a prefill-only vLLM pass to
capture the last verifier hidden layer for every token.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/fast_mtp/generate_dataset.py \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --data-path /mnt/data/rahul-tuli/datasets/gsm8k_Qwen3-Next-80B-A3B-Instruct.jsonl \
    --output-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    2>&1 | tee local/logs/gsm8k_step2_hidden_states.log
```

Output: `$HIDDEN_STATES/` — one `data_N.pt` per sample plus a
`sample_lengths.json` index. Each `.pt` file contains:

```python
{
    "input_ids":     Tensor[seq_len],       # long
    "hidden_states": Tensor[seq_len, 2048], # float32, last verifier layer
    "loss_mask":     Tensor[seq_len],       # long, 1 = assistant tokens only
}
```

A symlink is created at `local/dataset/qwen3next-gsm8k-hidden-states`.

---

## Step 3 — Finetune

### Single GPU

```bash
python examples/fast_mtp/04_finetune.py \
    --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \
    --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \
    --output-dir output/qwen3next_gsm8k_finetuned \
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
    examples/fast_mtp/04_finetune.py \
    --speculator-path Qwen3-Next-80B-A3B-Instruct_mtp_speculator \
    --data-dir /mnt/data/rahul-tuli/datasets/qwen3next-gsm8k-hidden-states \
    --output-dir output/qwen3next_gsm8k_finetuned \
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

| Argument | Default | Notes |
|---|---|---|
| `--lr` | `5e-5` | Learning rate |
| `--num-epochs` | `3` | Epochs to train |
| `--batch-size` | `1` | Per-GPU batch size |
| `--train-ratio` | `0.9` | Train/val split |
| `--scheduler-type` | `cosine` | LR schedule (`cosine`, `linear`, `none`) |
| `--warmup-steps` | auto | Steps for LR warmup |
| `--step-weights` | `0.51 0.31 0.18` | Per-step MTP loss weights |
| `--save-best` | off | Keep only the best val-loss checkpoint |
| `--checkpoint-freq` | `1` | Save every N epochs |

With `--save-best` the best checkpoint is at:

```
output/qwen3next_gsm8k_finetuned/best/model.safetensors
```

Without it, each epoch saves to:

```
output/qwen3next_gsm8k_finetuned/epoch_N/model.safetensors
```

---

## Step 4 — Stitch weights

Remaps trained MTP keys from speculators namespace (`mtp_layers.0.*`) to
Qwen3-Next native namespace (`model.mtp_layers.0.*`), copies all original
verifier shards into the output directory, and writes a single new shard plus
an updated `model.safetensors.index.json`. The result is a self-contained
directory that can be uploaded directly to HuggingFace.

```bash
python examples/fast_mtp/stitch_weights.py \
    --checkpoint output/qwen3next_gsm8k_finetuned/best/model.safetensors \
    --verifier Qwen/Qwen3-Next-80B-A3B-Instruct \
    --output-dir output/qwen3next_gsm8k_stitched
```

Output layout:

```
output/qwen3next_gsm8k_stitched/
    model.safetensors.index.json      # updated to point MTP keys at new shard
    mtp_finetuned.safetensors         # finetuned MTP weights only (~300 MB)
    config.json                       # verifier config (copied, unchanged)
    model-00001-of-00041.safetensors  # full copies of all verifier shards
    ...
```

The output is a self-contained model directory — upload it directly to HuggingFace:

```bash
huggingface-cli upload <your-org>/Qwen3-Next-80B-A3B-Instruct-FastMTP \
    output/qwen3next_gsm8k_stitched
```

---

## Step 5 — Deploy with vLLM

The stitched directory is a standard Qwen3-Next model directory. vLLM loads
it as-is and picks up the finetuned MTP head automatically via the
`num_nextn_predict_layers` field in `config.json`.

### Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="output/qwen3next_gsm8k_stitched",
    tensor_parallel_size=4,
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
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

---

## Resuming / re-running individual steps

- **Step 1** (response generation): pass `--resume` to `run_all.sh` to skip
  rows already written to the JSONL.
- **Step 2** (hidden states): re-running overwrites existing `.pt` files.
  The `token_freq.pt` and `sample_lengths.json` are regenerated each run.
- **Step 3** (finetune): pass `--checkpoint-freq N` to checkpoint every N
  epochs; `--save-best` keeps only the lowest-val-loss checkpoint.

## Adapting to a different dataset

Replace Step 1 with any ShareGPT-format JSONL (fields: `conversations` list
of `{"from": "human"/"gpt", "value": "..."}`). Pass the path to
`--data-path` in Step 2. Everything downstream is dataset-agnostic.
