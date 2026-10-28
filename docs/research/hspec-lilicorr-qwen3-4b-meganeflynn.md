---
title: "H-Spec and LiLiCorr speculators for Qwen3-4B: best results and reproduction"
tags:
  - HSpec
  - LiLiCorr
  - DFlash
  - Qwen3
  - mamba_attn_hybrid
---

<div class="entry-byline" markdown>
**Posted on 2026-10-28** · work tracked on the `megan-hippocampus` and
`hspec-lilicorr-integration` branches
</div>

# H-Spec and LiLiCorr speculators for Qwen3-4B

## Summary

Three drafters trained against Qwen/Qwen3-4B on a 100k-sample regenerated-conversation dataset. Under greedy serving with 7 speculative tokens, **standalone LiLiCorr gives the best acceptance (mean acceptance length 3.16, greedy match 83/100)**; H-Spec (`mamba_attn_hybrid`) is second (2.62, 80/100); the H-Spec+LiLiCorr combination sits between them on acceptance (2.91, 79/100). Two training scales are reported: a short 1-epoch/128-anchor chain run with full end-to-end serving evaluation, and longer multi-epoch runs with validation-only metrics.

## Setup

- Target model: `Qwen/Qwen3-4B`, snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, bfloat16
- Drafters: trained checkpoints listed below (all `speculators` format)
- Algorithm: DFlash-family speculative decoding, 7 speculative tokens, greedy (temperature 0), eager execution, TP=1
- Training data: `inference-optimization/Qwen3-8B-Regenerated-Collection` snapshot `65d219d6b40bb27c45afe16665147a1d3fa21069`; 100k rendered rows (99k train / 1k val, 256 held out), frozen split manifest at `/data/fast/integration/phase100k_data/frozen_split_manifest.json`
- Evaluation: 100 fixed held-out prompts, ≤128 new tokens, temperature 0, sequential greedy decode compared token-for-token against plain Qwen3-4B
- Serving: vLLM 0.31.0 (integration fork, branch `hspec-lilicorr-integration`), `--enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.90`
- Software: speculators 0.10.0a7, PyTorch 2.13.0+cu130, transformers 5.17.0, compressed-tensors 0.17.0
- Hardware: 4× A100 80 GB (producer on GPUs 0–1 TP=2, training on GPUs 2–3 DDP, eval on one GPU)

## Best result per method

Metrics: **MAL** = vLLM "Mean acceptance length" over the 100-prompt eval (includes the bonus token); **greedy match** = prompts whose full 128-token greedy output is token-identical to plain Qwen3-4B; **draft acceptance** = accepted/drafted tokens. Serving numbers are from the 1-epoch/128-anchor chain runs (fully evaluated end to end); training-validation numbers are from the longer runs.

| Method                                  | MAL                               | Draft acceptance | Greedy match | Best val EAL (long run)                                                   |
| --------------------------------------- | --------------------------------- | ---------------- | ------------ | ------------------------------------------------------------------------- |
| H-Spec (`mamba_attn_hybrid`)            | 2.62 (median 2.64, peak 3.18)     | 24.1%            | 80/100       | 2.63 (5 epochs, 3072 anchors)                                             |
| LiLiCorr (standalone)                   | **3.16** (median 3.03, peak 4.03) | **29.7%**        | **83/100**   | 4.47 top-1 / 8.02 oracle (6 epochs, 1024 anchors)                         |
| H-Spec + LiLiCorr (`--enable-lilicorr`) | 2.91 (median 3.00, peak 3.23)     | 27.6%            | 79/100       | 1.38 top-1 (6 epochs, 256 anchors — invalid comparison, see Observations) |

Checkpoint SHA-256 (serving-evaluated checkpoints, drafter `model.safetensors`):

- H-Spec: `c825a0f49705c904af8e1601e684f8f9b859db76363b5876b05b034e9f59d335`
- LiLiCorr: `c9af3892696d0e1b8e74a70eebd1f3859fcafc41477879ec3bbdd3bdc1fa5dcb`
- H-Spec + LiLiCorr: `737812ec4320b9bb8dafb6d8327813e00308a76fd58fbf9ac4d65fd8047bde2a`

## Reproduction

All commands below were run on the `hspec-lilicorr-integration` branch of speculators (vLLM fork at the matching integration head). The eval and provenance artifacts live in `/data/fast/integration/eval` and `/data/fast/integration/phase100k_eval`.

### Data preparation (once)

```bash
# Sample 105k conversations and render with Qwen3-4B (thinking on)
python sample_collection_100k.py          # -> sampled_collection.jsonl + manifest
python freeze_split_100k.py               # -> 100k train / 1k val / 256 heldout, frozen manifest
# Render + prepare train/val/heldout splits, then start the hidden-state producers
```

### Training (the evaluated 1-epoch chain run)

Start the verifier+hidden-state producer on GPUs 0–1 (H-Spec legs use the NKV producer):

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm_nkv.py Qwen/Qwen3-4B \
  --hidden-states-path $OUT/transfer-hybrid \
  --attn-kv-layer-ids 17 25 34 --latent-fusion-layer-ids 1 9 17 25 34 -- \
  --tensor-parallel-size 2 --dtype bfloat16 --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --gpu-memory-utilization 0.80 --no-enable-prefix-caching --no-enable-chunked-prefill
```

Then train on GPUs 2–3. H-Spec (full command in the checkpoint's `train_command.txt`):

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 -m speculators.train \
  --verifier-name-or-path Qwen/Qwen3-4B \
  --data-path /data/fast/integration/phase100k_data/prepared_100000 \
  --train-data-ratio 0.99 --seed 42 --optimizer adamw --optimizer-implementation fused \
  --lr 1e-4 --hidden-states-dtype bfloat16 --mask-token-id 151669 --sliding-window 2048 \
  --num-workers 2 --prefetch-factor 2 --checkpoint-freq 1 --log-freq 10 \
  --hidden-states-backend file --hidden-states-path $OUT/transfer-hybrid \
  --on-missing generate --vllm-endpoint http://127.0.0.1:8000/v1 \
  --save-path $OUT/run_hspec/checkpoints --log-dir $OUT/run_hspec/logs --run-name hspec \
  --speculator-type mamba_attn_hybrid \
  --block-pattern mamba mlp mamba attention mlp mamba attention mlp mamba attention mlp \
  --attn-kv-layer-ids 17 25 34 --latent-fusion-layer-ids 1 9 17 25 34 \
  --mamba-seed-mode shared --mamba-num-heads 44 --mamba-head-dim 64 --mamba-d-state 4 \
  --mamba-n-groups 4 --mamba-conv-kernel 4 --trainable-q-proj --trainable-kv-proj \
  --use-qk-norm --sliding-window-indices 0 1 2 --no-sliding-window-non-causal \
  --parallel-drafting-loss-weight dflash --epochs 1 --total-seq-len 2048 \
  --max-anchors 128 --block-size 8 --dflash-decay-gamma 4
```

LiLiCorr (standalone) — swap the producer to the plain hidden-state one (`scripts/launch_vllm.py train Qwen/Qwen3-4B --target-layer-ids 2 10 18 26 34 --hidden-states-path $OUT/transfer-lilicorr -- ...`) and train with:

```bash
... --save-path $OUT/run_lilicorr/checkpoints --run-name lilicorr \
  --speculator-type lilicorr --num-layers 5 --target-layer-ids 2 10 18 26 34 \
  --no-sample-from-anchor --sliding-window-non-causal \
  --per-position-loss-weight fixed-exp-decay --dflash-decay-gamma 7 \
  --loss-fn ce --epochs 1 --total-seq-len 2048 --max-anchors 128 --block-size 8
```

Combined — same as H-Spec plus `--enable-lilicorr --markov-rank 0 --no-enable-confidence-head --no-confidence-head-with-markov`, LiLiCorr-style schedule (`--epochs 6 --total-seq-len 3120 --block-size 16 --scheduler-type linear --scheduler-warmup-ratio 0.04`).

The exact full commands, `run.yaml`, and `speculators.patch` for every run are saved in each checkpoint directory (e.g. `/data/fast/integration/phase100k_data/run_hspec/checkpoints/train_command.txt`).

### Long-run (best val) training settings

- **H-Spec, 5 epochs**: the command above but `--lr 6e-4 --epochs 5 --total-seq-len 8192 --max-anchors 3072 --block-size 8 --scheduler-type cosine --scheduler-num-cosine-cycles 0.5 --scheduler-warmup-ratio 0.01 --weight-decay 0.01 --loss-fn '{"ce":0.1,"tv":0.9}' --markov-rank 256 --markov-head-type vanilla --draft-vocab-size 32000 --enable-confidence-head --confidence-head-with-markov --confidence-head-alpha 1.0`. Result: val EAL 2.634, full-block acc 0.666 (`/data/fast/integration/hlili_100k/hspec/checkpoints/checkpoint_best`).
- **LiLiCorr, 6 epochs**: `--lr 6e-4 --epochs 6 --total-seq-len 3120 --block-size 16 --scheduler-type linear --scheduler-warmup-ratio 0.04 --max-anchors 1024 --dflash-decay-gamma 7 --weight-decay 0 --loss-fn ce` (3072/2048/1536 anchors all OOMed on 80 GB A100s; 1024 is the largest that fits). Result: val top-1 EAL 4.471, oracle EAL 8.018, full-block acc 0.350 (`/data/fast/integration/hlili_100k/lilicorr/checkpoints/checkpoint_best`).
- **Combined, 6 epochs @ 256 anchors** (accidental — a duplicated `--max-anchors` left 256 in force, 4× fewer than LiLiCorr standalone; do not compare directly): val top-1 EAL 1.379 (`/data/fast/integration/hlili_100k/combined/checkpoints/checkpoint_best`). A clean 1024-anchor rerun with `--gradient-checkpointing` was SIGTERM-interrupted at step ~18k of epoch 1/6 (train lilicorr_eal 2.51 at that point); resumable checkpoint at `/data/fast/integration/hlili_100k_a1024/combined/checkpoints/0` (rename to an epoch dir to auto-resume). The memory fixes needed for that run (chunked+checkpointed TV loss and candidate log-probs, gradient-checkpointable hybrid sublayers) are verified numerically identical but were **not yet committed** at the time of writing.

### Serving evaluation

```bash
python /data/fast/integration/eval_serving.py plain  --port 8123 --gpu 2
python /data/fast/integration/eval_serving.py serve hspec_100k \
  /data/fast/integration/phase100k_data/run_hspec/checkpoints/epoch0_end --port 8123 --gpu 2
python /data/fast/integration/eval_serving.py batch hspec_100k
# likewise lilicorr_100k, hspec_lilicorr_100k; then `compare`
```

The underlying vLLM launch (also recorded per run in `provenance_*/vllm_command.txt`):

```bash
python -m vllm.entrypoints.cli.main serve \
  /data/fast/hspec/hf/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --spec-model $CKPT_DIR --spec-tokens 7 --port 8123 \
  --gpu-memory-utilization 0.90 --enforce-eager --max-model-len 4096
```

## Results

Same target, workload, hardware, and serving settings across rows (1-epoch chain runs, 100 held-out prompts, ≤128 new tokens, greedy, TP=1, eager, 7 spec tokens):

| Configuration     | Acceptance rate (%) | Mean accepted draft tokens per step (MAL) | Output tokens/s (median) | Greedy match        |
| ----------------- | ------------------- | ----------------------------------------- | ------------------------ | ------------------- |
| Target only       | N/A                 | N/A                                       | ~45 (plain decode)       | 100/100 (reference) |
| H-Spec            | 24.1                | 2.62                                      | 82.9                     | 80/100              |
| LiLiCorr          | 29.7                | 3.16                                      | 101.4                    | 83/100              |
| H-Spec + LiLiCorr | 27.6                | 2.91                                      | 83.0                     | 79/100              |

Per-position acceptance (H-Spec): 0.76, 0.47, 0.26, 0.12, 0.05, 0.02, 0.01 — LiLiCorr: 0.74, 0.51, 0.38, 0.28, 0.19, 0.14, 0.08 — Combined: 0.74, 0.50, 0.31, 0.19, 0.10, 0.06, 0.04.

Long-run training validation (best epoch):

| Method            | Epochs / anchors  | val EAL                       | val full-block acc |
| ----------------- | ----------------- | ----------------------------- | ------------------ |
| H-Spec            | 5 / 3072          | 2.634                         | 0.666              |
| LiLiCorr          | 6 / 1024          | 4.471 (top-1), 8.018 (oracle) | 0.350              |
| H-Spec + LiLiCorr | 6 / 256 (invalid) | 1.379 (top-1)                 | 0.217              |

## Observations

- **LiLiCorr wins on acceptance at every position ≥1** and is the only method whose acceptance stays above 0.14 at position 5, giving it the longest useful draft prefix.
- The combined model's acceptance lies between the two standalone methods; it did not beat LiLiCorr alone in any serving metric at this scale.
- Greedy mismatches are dominated by verifier numerics, not drafting bugs: plain Qwen3-4B alone flips 24/100 prompts between sequential and batched execution, and all divergence points sit at top1–top2 logit gaps ≤ 0.25.
- A 5k-sample run of the same pipeline gave ~8% draft acceptance and MAL ≈ 1.5; the 20× data increase moved H-Spec to 24% acceptance / MAL 2.68 — data scale mattered more than any hyperparameter change tried.
- LiLiCorr anchor memory is the binding constraint: >1024 anchors OOMs on 80 GB A100s without gradient checkpointing; the memory-fix patch (chunked/checkpointed TV loss and candidate log-probs) removes the 9.3 GiB and 8.7 GiB fp32 spikes with zero numerical difference, but remains uncommitted.
- Not tested: CUDA graphs, stochastic sampling, TP>1 serving, concurrency >32, and any draft-vocab reduction for H-Spec on this dataset (no `token_freq.pt` present, so the long H-Spec run used full vocabulary).
