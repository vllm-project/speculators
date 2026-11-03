# LiLiCorr Speculator for Qwen3-4B on Regenerated PerfectBlend (Full Pool, 1 Epoch)

**Date started:** 2026-11-02 21:08
**Status:** STOPPED at 38% (step 85,082 / 225,504) — externally SIGTERM'd on
2026-11-03 14:48; last durable checkpoint is the quarter epoch (step 56,376).
Run halted per operator decision; see "Interruption" below.
**Code:** `speculators` @ `74ffda8` (branch `lilicorr-qwen38-fresh-2ep-argmax`), trainer `0.9.0a72`, vLLM `0.32.0` (lilicorr fork)
**Verifier:** `Qwen/Qwen3-4B`

Companion run to [the 100k experiment](lilicorr-qwen3-4b-perfectblend-regen-100k.md):
same recipe, scaled from a 100k uniform sample to the full perfectblend pool to
measure what 13.8× more (non-repeating) data buys.

## Data

- All 1,382,563 conversations from the 8 perfectblend subsets of
  `inference-optimization/Qwen3-8B-Regenerated-Collection`, schema-normalized
  (`id` → string; `autoif` ships integer ids) into `full_pool.jsonl`.
- `prepare-data` (render via 4B extraction server, seq-length 3072):
  **1,382,090 rows** (473 conversations dropped — no assistant turn with
  context or fully truncated), mean ~2,165 tokens/row → ~3.0B training tokens.
- The 100k sample's hidden-state cache is reused (same extraction server and
  storage path); overlapping rows are cache hits.

## Recipe

Identical to the 100k run (see `full.yaml`, `run_full.sh`): taps
`[2,10,18,26,34]`, mask 151669, block 16 → 15 spec tokens, seq 3072 / 512
anchors, CE + fixed-exp-decay gamma 7.0, muon hybrid lr 6e-4 with default
weight decay, cosine 4% warmup, FSDP + gradient checkpointing on 5×H100
(GPUs 3–7), 4B extraction server on GPU 2 (`on_missing: generate`),
`checkpoint_freq: 0.25`. GPU reservation extended to 72 h (through Nov 5).

## Progress

Pace: 0.75–0.77 s/step steady. Train-batch metrics (single-step samples every
5k steps):

| step | lilicorr_eal | oracle_eal | lilicorr_ce |
|---|---|---|---|
| 5k | 3.60 | 6.23 | 1.00 |
| 10k | 4.08 | 6.48 | 0.84 |
| 20k | 5.19 | 8.08 | 0.74 |
| 35k | 5.94 | 9.53 | 0.62 |
| 60k | 5.87 | 9.17 | 0.62 |
| 85k (38%) | 5.6 | 8.4–9.9 | 0.67 |

Latest window (step ~85,080): **eal 5.6 / top1_eal 4.8 / oracle_eal 9.0 /
ce 0.67** (per-batch noise ±1).

### Comparison to the 100k run

At the same fraction of its (13.8× shorter) schedule, the 100k run sat at
~EAL 4.7–5.2 (30% point); this run is at ~5.6 at 38% with no data repetition,
and has already passed the 100k run's final train EAL (5.29). Caveat: the
100k sample was uniform over the same pool, so the full run is seeing the
same distribution — the gap is fresh-data diversity plus a longer LR schedule.

## Interruption (2026-11-03 14:48)

- The trainer received an external SIGTERM ("Received 15 death signal") at
  step 85,082; the 4B extraction server died in the same event. Root cause:
  the operator's client session (VS Code window) was closed, tearing down the
  agent process tree both processes belonged to. No OOM, NCCL fault, or
  storage issue preceded it — tmpfs was at 53% with 360G free.
- The graceful-shutdown checkpoint attempt timed out, so the last durable
  checkpoint remains `epoch0_step56376` (25%; the `0/` dir mirrors it).
  ~28.7k steps (~6 h) of progress past it were lost.
- The GPU reservation was subsequently released; the run was stopped per
  operator decision and has not been relaunched.

### Resuming later

Everything needed is durable under `/tmp/lilicorr_check/qwen3-4b-regen/`
(`datafull/`, hidden-state cache, checkpoints). To resume from step 56,376:
relaunch the extraction server (GPU 2, port 8001 — command in
`vllm_command.txt`), then rerun `run_full.sh` with
`no_resume_from_checkpoint: false` (the current `full.yaml` sets it `true`).
Consider launching under `tmux`/`setsid` so an external process-group signal
cannot take the run down with it. Remaining cost from the checkpoint:
~169k steps ≈ 36 h at the observed 0.76 s/step.

## Artifacts

- Experiment dir: `/tmp/lilicorr_check/qwen3-4b-regen/`
  - `full_pool.jsonl`, `datafull/`, `run_full.sh`
  - `run_full_1ep_muon/` — checkpoints, train.log, trackio logs
  - 100k run + eval artifacts: `run_100k_1ep_muon/`, `eval/`
