# H-Spec mamba_attn_hybrid drafter for Qwen3.8-27B — reproduction

Trains an H-Spec (mamba_attn_hybrid) draft model for the Qwen/Qwen3.8-27B
verifier (qwen3_5 hybrid arch, 64 layers) with online hidden-state + KV
extraction via a vLLM server, then serves it for speculative decoding at
draft depth 5.

Design (differs from the Qwen3-4B run in speculators PR #1267 only as noted):

- **Taps / KV-shared layers**: the verifier's full-attention layers
  `3, 19, 35, 51, 63` (all five are tapped and KV-shared → draft depth 5).
- **Block pattern**: `mamba mlp (mamba attention mlp) x 5` — 26 sublayers,
  5 attention sublayers = number of tapped layers.
- **Draft depth 5** = one drafted token per attention sublayer
  (`--block-size 6` = 1 anchor + 5 draft positions; serve with
  `num_speculative_tokens: 5`).
- **Draft vocab 64000** (with t2d/d2t token-frequency mappings): the full
  248320 vocab OOMs on 80 GiB H100s (fp32 logits + optimizer state).
- **max-anchors 1024**.
- 8xH100: GPU 0 = extraction server, GPUs 1-7 = one training rank each.

## Environment

- vLLM 0.31.0 with the two patches from the `hspec_qwen_27B` branch of
  https://github.com/MeganEFlynn/vllm (see that branch's README). Without
  them, any prefix-cache hit makes the connector's saved kv_cache rows NaN.
- This speculators branch (`hspec_qwen_27B`), editable-installed
  (`pip install -e .`), including `hs_connectors` (the
  `KVCacheHiddenStatesConnector`).
- `mamba-ssm` (triton ssd path) and `flashinfer` JIT need `ninja` on PATH —
  activate the env so its `bin/` is on PATH before launching vLLM.
- `HF_TOKEN` must be exported at runtime (never written into files/scripts).

## 1. Draft config

`/tmp/hspec-27b/draft_config.json` (explicit qwen3-shaped geometry; the
qwen3_5 arch's gated q_proj doesn't fit the heuristic head derivation, and
`num_hidden_layers` must equal the attention-layer count = 5):

```json
{
  "model_type": "qwen3",
  "hidden_size": 5120,
  "num_hidden_layers": 5,
  "num_attention_heads": 24,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "intermediate_size": 17408,
  "vocab_size": 248320,
  "rms_norm_eps": 1e-06,
  "max_position_embeddings": 262144,
  "attention_bias": false,
  "tie_word_embeddings": false
}
```

### RoPE (critical for KV sharing)

The drafter reads the target's **already-rotated** K prefix from the shared
KV cache, so its Q/K rotation must be the target's effective
text/full-attention rope. Qwen3.8-27B uses
`head_dim=256`, `partial_rotary_factor=0.25`, `rope_theta=1e7` — i.e. only
the first 64 channels of each head rotate (neox pairing `j ↔ j+32` inside
that prefix; channels 64:256 and all V are unrotated), with attention scale
`1/16`. The draft config JSON above deliberately **omits** rope fields:
`from_training_args` reconciles the draft's rope against the actual verifier
config on fresh construction (both the `--draft-config` and automatic
decoder-config paths) and logs the resolved settings:

```
mamba_attn_hybrid rope resolved from verifier 'Qwen/Qwen3.8-27B': head_dim=256
rotary_dim=64 rope_theta=10000000.0 partial_rotary_factor=0.25 (explicit draft override: False)
```

The saved checkpoint's `transformer_layer_config` therefore carries:

```json
{
  "head_dim": 256,
  "rope_parameters": {
    "rope_type": "default",
    "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25
  }
}
```

which vLLM's `get_rope` reconstructs into the identical partial rotation
(the fork's `get_rope` honors `partial_rotary_factor`; MRoPE metadata is
normalized away since scalar text positions make an MRotaryEmbedding
equivalent to ordinary partial RoPE). Explicit rope overrides in the draft
config JSON that are incompatible with the shared target K are rejected at
construction, and `--from-pretrained` checkpoints with incompatible rope
geometry (e.g. checkpoints trained before this fix, which used the stock
full-head `Qwen3RotaryEmbedding` with theta 1e4) fail with a clear
diagnostic instead of being silently re-interpreted. Regression tests:
`tests/unit/models/test_mamba_attn_hybrid_rotary.py`.

## 2. Extraction server (GPU 0, port 8000, TP=1)

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/tmp/hspec-27b/hf   # no PYTORCH_CUDA_ALLOC_CONF=expandable_segments here
vllm serve Qwen/Qwen3.8-27B \
    --port 8000 \
    --max-model-len 8196 \
    --attention-backend FLASHINFER \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --speculative-config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [3, 19, 35, 51, 63, 64]}}}' \
    --kv-transfer-config '{"kv_connector": "KVCacheHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_module_path": "hs_connectors.kv_cache_hidden_states_connector", "kv_connector_extra_config": {"shared_storage_path": "/tmp/hspec-27b/hidden_states", "kv_cache_layer_ids": [3, 19, 35, 51, 63]}}'
```

## 3. Per-rank GPU pinning launcher

`rank_launch.py` — torchrun runs this with python; it pins each rank to one
physical GPU (LOCAL_RANK+1) so ranks don't hold CUDA contexts on each other's
devices (~522 MiB per sibling rank, which OOMs this model):

```python
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = str(int(os.environ.get("LOCAL_RANK", "0")) + 1)

# multiprocessing spawn re-runs this file as __mp_main__ in DataLoader
# workers; only torchrun's direct invocation (run as __main__) should boot
# the trainer, otherwise each worker would re-init the process group.
if __name__ == "__main__":
    sys.argv[0] = "-m speculators.train"
    runpy.run_module("speculators.train", run_name="__main__")
```

## 4. Prepare data + train

```bash
export PATH=/path/to/env/bin:$PATH
export HF_HOME=/tmp/hspec-27b/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7   # 0 = extraction server

speculators prepare-data \
    --model Qwen/Qwen3.8-27B \
    --data /tmp/hspec-27b/data/train_50k.jsonl \
    --output /tmp/hspec-27b/training_data_50k \
    --seq-length 8192 \
    --token-freq-path /tmp/hspec-27b/training_data_50k/token_freq.pt \
    --no-skip-token-freq
```

(`--token-freq-path`/`--no-skip-token-freq` are required for
`--draft-vocab-size 64000` to take effect; without them training silently
falls back to the full verifier vocab and OOMs.)

```bash
torchrun --standalone --nproc_per_node 7 /tmp/hspec-27b/rank_launch.py \
    --speculator-type mamba_attn_hybrid \
    --verifier-name-or-path Qwen/Qwen3.8-27B \
    --draft-config /tmp/hspec-27b/draft_config.json \
    --data-path /tmp/hspec-27b/training_data_50k \
    --hidden-states-path /tmp/hspec-27b/hidden_states \
    --hidden-states-backend file_kv \
    --attn-kv-layer-ids 3 19 35 51 63 \
    --vllm-endpoint http://localhost:8000/v1 --on-missing generate \
    --block-pattern mamba mlp mamba attention mlp mamba attention mlp mamba attention mlp mamba attention mlp mamba attention mlp \
    --block-size 6 \
    --target-layer-ids 3 19 35 51 63 \
    --mamba-num-heads 44 --draft-vocab-size 64000 --max-anchors 1024 \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --optimizer adamw --lr 0.0006 --weight-decay 0.01 \
    --scheduler-type cosine --scheduler-warmup-ratio 0.01 \
    --epochs 1 --checkpoint-freq 0.06 --gradient-checkpointing \
    --save-path /tmp/hspec-27b/checkpoints_50k
```

Notes:

- Hidden states + KV are generated online: any sample missing from
  `hidden_states/` is generated through the extraction server
  (`--on-missing generate`). For 50k samples this is the long pole.
- `--gradient-checkpointing` is required to fit (79 → ~69 GiB/rank); the
  hybrid sublayers inherit `GradientCheckpointingLayer` on this branch.
- A 500-sample smoke run first (`data/smoke_500.jsonl`, separate
  save-path) to validate the train → serve pipeline is recommended.

## 5. Serve the trained drafter

```bash
export CUDA_VISIBLE_DEVICES=1   # any free GPU
vllm serve Qwen/Qwen3.8-27B \
    --speculative-config '{"model": "/tmp/hspec-27b/checkpoints_50k/<final>", "method": "mamba_attn_hybrid", "num_speculative_tokens": 5}' \
    --port 8108 --max-model-len 8196 \
    --gpu-memory-utilization 0.90 \
    --attention-backend FLASHINFER \
    --max-num-seqs 16 --max-num-batched-tokens 4096
```

Acceptance metrics appear in the server log (`SpecDecoding metrics: Mean
acceptance length ...`). A 500-sample smoke checkpoint accepted ~1%
(draft depth 5, 17 steps); the 50k run reached train/accept_rate ≈ 0.36
by ~75% of the epoch.

## Patches carried on this branch

- `hs_connectors/.../kv_cache_hidden_states_connector.py`: linear slot
  mapping (`block_ids[t // tokens_per_block] * tokens_per_block + t %
  tokens_per_block`) — scheduler blocks (800 tokens) map to 25 consecutive
  kernel pages, so token t's physical slot is linear in the block id.
- `models/mamba_attn_hybrid/{core,model_definitions}.py`:
  `supports_gradient_checkpointing` + sublayers inherit
  `GradientCheckpointingLayer`.
- `models/mamba_attn_hybrid` init-from-verifier handling for the qwen3_5
  arch (gated q_proj skipped, o_proj copied, gemma-style q/k-norm copied as
  `1+w`).
- `train/{distributed,dataloader,trainer,recovery}.py`: device indices use
  `local_rank % torch.accelerator.device_count()` so per-rank
  `CUDA_VISIBLE_DEVICES` pinning works.
