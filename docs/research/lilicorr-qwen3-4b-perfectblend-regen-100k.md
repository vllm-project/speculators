# LiLiCorr Speculator for Qwen3-4B on Regenerated PerfectBlend (100k, 1 Epoch)

**Date:** 2026-11-02
**Code:** `speculators` @ `74ffda8` (branch `lilicorr-qwen38-fresh-2ep-argmax`), trainer `0.9.0a72`, vLLM `0.32.0` (lilicorr fork)
**Verifier:** `Qwen/Qwen3-4B` (36 layers, hidden 2560, vocab 151,936)

## Summary

Trained a 5-layer LiLiCorr draft on 100k conversations uniformly sampled from the
perfectblend subsets of `inference-optimization/Qwen3-8B-Regenerated-Collection`
(Qwen3-8B-regenerated corpus; 4B is the verifier, not the corpus generator).
One epoch, muon hybrid optimizer with default weight decay. Served acceptance on
HumanEval: **4.29 tokens/request at temperature 1, 4.74 greedy** with 15
speculative tokens — higher per-position acceptance than the earlier
Qwen3.8-27B 100k arm at every matched position.

## Data

- Source: the 8 perfectblend subsets of `Qwen3-8B-Regenerated-Collection`
  (metamathqa 394,996 / ultrainteract 288,579 / ultrachat 207,864 / orca_math
  200,035 / evol_codealpaca 111,100 / autoif 61,492 / ultrafeedback 61,135 /
  lmsys_arena 57,362; pool 1,382,563 conversations, ShareGPT schema).
- Sample: 100,000 conversations, uniform over pool, seed 42. Per-subset share
  7.2–7.5% (proportions preserved).
- Schema note: `autoif` ships integer `id`s while the other subsets use strings,
  which breaks `datasets` JSON inference; the sample was normalized to
  `{"id": str, "conversations": [...], "source": <file>}`.
- `prepare-data` (render via 4B vLLM server, seq-length 3072): 99,969 rows,
  mean 2,165 tokens (p50 2,179, ~10%+ at the 3,072 cap from long thinking
  traces), mean 2,037 trainable tokens/row, 0 zero-supervision rows.

## Recipe

Paper recipe for 36-layer Qwen3 (`examples/train/lilicorr_qwen3_8b_online.sh`):

| Setting | Value |
|---|---|
| Draft layers / taps | 5, verifier layers `[2, 10, 18, 26, 34]` |
| Mask token | 151669 (unused Qwen3 slot) |
| Block size | 16 → 15 speculative tokens |
| Seq len / anchors | 3072 / 512, `sample_from_anchor: false` |
| Loss | CE, `fixed-exp-decay` gamma 7.0, `loss_chunk_anchors: 128` |
| Optimizer | muon hybrid, lr 6e-4, **default weight decay** (AdamW group 0.01 / Muon group 0.1) |
| Scheduler | cosine, 4% warmup |
| Epochs / steps | 1 / 16,272 (0.757 s/step, 3h25m wall) |
| Parallelism | FSDP shard + gradient checkpointing, 5×H100, 1 packed 3072-seq/GPU/step |
| Draft attn | sdpa |
| Extraction | vLLM 4B server (GPU 2, port 8001), aux layers (2,10,18,26,34,36), hidden states at `/tmp/hidden_states_qwen3_4b`, `on_missing: generate` |

## Training Results

Trajectory (every ~2,500 steps, train-batch means):

| step | lilicorr_eal | oracle_eal | lilicorr_ce |
|---|---|---|---|
| 500 | 1.68 | 2.75 | 1.90 |
| 2,500 | 3.59 | 6.08 | 1.03 |
| 4,500 | 4.74 | 7.45 | 0.74 |
| 9,000 | 5.18 | 8.56 | 0.72 |
| 12,500 | 5.43 | 8.77 | 0.68 |
| 16,000 | 6.25 | 9.27 | 0.62 |

Final window (last 8 steps): **eal 5.29 / top1_eal 4.64 / oracle_eal 8.43 / ce 0.69**.

## Served Acceptance

Served with the lilicorr vLLM fork: Qwen3-4B verifier + trained draft
(`checkpoints/0`), data-parallel 8 on H100s, 15 speculative tokens, greedy draft
sampling, `max_num_seqs 4`, eager, FA2. Bench via
`scripts/evaluate/evaluate.py throughput` (GuideLLM, concurrency 128).

| Bench | AL (temp 1) |
|---|---|
| HumanEval | **4.29** (113,280 drafts, 372,802 accepted) |
| HumanEval (greedy, temp 0) | **4.74** (102,766 drafts, 383,968 accepted) |
| math_reasoning | 5.40 |
| writing | 3.39 |
| question | 3.32 |
| rag | 3.31 |
| translation | 3.29 |
| tool_call | 3.18 |
| qa | 2.99 |
| summarization | 2.56 |

Position curve (HumanEval, temp 1): 0.81 / 0.62 / 0.47 / 0.35 / 0.27 / 0.20 /
0.15 / 0.11 / 0.09 / 0.07 / 0.05 / 0.04 / 0.03 / 0.02 / 0.02.

### Comparison to the Qwen3.8-27B 100k arm

27B arm (block 8 → 7 spec tokens): served AL 2.93 (temp 1) / 3.23 (greedy),
pos-0 0.655. This 4B run: pos-0 0.806 (temp 1), higher acceptance at every
matched position, plus 15 positions instead of 7. Not an apples-to-apples arm —
different verifier, block size, and corpus generator — but the fresh 4B
speculator is clearly healthy. Reference: the prior 4B lineage run (more data,
older recipe) finished at greedy AL 6.41.

## Artifacts

- Experiment dir: `/tmp/lilicorr_check/qwen3-4b-regen/`
  - `sample_100k.py`, `full_pool.jsonl` (1,382,563 convs), `data100k/`
  - `full.yaml`, `run_100k.sh`, `run_full.sh`
  - `run_100k_1ep_muon/` — checkpoints (quarterly + best + final `0/`), train.log, trackio logs
  - `eval/` — serve command, vLLM log, `humaneval_final/`, `humaneval_greedy/`, `fullbench/` (acceptance.csv each)

## Reproduction

```bash
# sample (uniform 100k, seed 42)
python3 sample_100k.py                     # -> sample100k.jsonl

# prepare data (needs 4B extraction server on :8001)
speculators prepare-data --model /tmp/models/Qwen3-4B \
  --data sample100k.jsonl --render-endpoint http://localhost:8001 \
  --output data100k --seq-length 3072

# train (5 GPUs, extraction server on GPU 2)
bash run_100k.sh                           # or: run_100k.sh smoke

# serve for acceptance (8 GPUs)
python -m vllm.entrypoints.cli.main serve /tmp/models/Qwen3-4B \
  --port 8002 --data-parallel-size 8 \
  --speculative-config '{"method": "dflash", "model": "<ckpt>/checkpoints/0", "num_speculative_tokens": 15, "draft_sample_method": "greedy"}' \
  --max-model-len 8192 --max-num-seqs 4 --enforce-eager \
  --gpu-memory-utilization 0.9 --per-request-spec-decode-metrics summary

# acceptance numbers
python scripts/evaluate/evaluate.py --target http://localhost:8002/v1 \
  throughput --subsets HumanEval --output-dir <dir>
```

## Follow-on

Full-pool 1-epoch run (all 1,382,563 conversations, same recipe) launched as
`qwen3-4b-lilicorr-regen-full-1ep-muon` — see `run_full_1ep_muon/`.
