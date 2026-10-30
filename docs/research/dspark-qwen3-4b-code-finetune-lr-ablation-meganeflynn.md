---
title: "Fine-tuning a DSpark Qwen3-4B speculator on code data: learning-rate ablation"
tags:
  - DSpark
  - Qwen3
  - OpenCodeInstruct
---

<div class="entry-byline" markdown>
[![MeganEFlynn](https://github.com/MeganEFlynn.png?size=64){ .entry-avatar }](https://github.com/MeganEFlynn)
**[@MeganEFlynn](https://github.com/MeganEFlynn)** posted on **2026-10-30**
</div>

# Fine-tuning a DSpark Qwen3-4B speculator on code data: learning-rate ablation

> All four learning-rate arms completed training and evaluation.

We fine-tuned the published DSpark speculator
[`MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark`](https://huggingface.co/MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark)
(trained on the tulu3 subset of `Qwen3-8B-Regenerated-Collection`) on a newly
regenerated code-heavy dataset — 114k instruction/response pairs from
`nvidia/OpenCodeInstruct` regenerated with `Qwen/Qwen3-8B` — for 1 epoch,
ablating the fine-tuning learning rate over {3e-4, 1.5e-4, 7.5e-5, 3e-5}
(Muon LR scaled proportionally at 5x). HumanEval acceptance length shows a
clean monotonic trend: 5.11 at lr 3e-4 (below the 5.41 tulu3 baseline),
recovering through 5.35 at 1.5e-4, and **beating the baseline at 7.5e-5
(5.53) and 3e-5 (5.60, +3.7%)**. Fine-tuning on code data helps, but only
at learning rates well below the from-scratch recipe value (6e-4); the
best value tested was 20x smaller.

## Setup

- Target model: `Qwen/Qwen3-4B` (bf16, served with vLLM 0.31.0)
- Draft model: `MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark`
  (DSpark, 5 draft layers, verifier hidden-state layers 1/9/17/25/33,
  Markov head rank 256, confidence head, block size 16), loaded via
  `--from-pretrained` for fine-tuning
- Algorithm: DSpark, 16 speculative tokens, greedy proposals
- Training data: `nvidia/OpenCodeInstruct` regenerated with
  `Qwen/Qwen3-8B` (chat completions, max_tokens 8192, thinking traces kept),
  114,056 rows → 0.56B tokens, mean sequence length 4,907, multipacked at
  8,192. Uploaded as
  `inference-optimization/Qwen3-8B-Regenerated-Collection/opencode_instruct_train_Qwen3-8B.jsonl`
- Training: 1 epoch (~11,400 steps), Muon optimizer with proportional
  AdamW group (Muon LR = 5x main LR), cosine schedule, 4% warmup, seed 42,
  loss `0.1*CE + 0.9*TV` (fused), fixed-exponential per-position decay
  (gamma 7.0), FSDP across 6 GPUs, hidden states rendered on the fly by a
  separate vLLM server (DP-2, GPUs 0-1) with the file transfer backend
- Evaluation: `speculator_benchmarks` HumanEval subset via
  `scripts/evaluate/evaluate.py throughput`, 128 max completion tokens,
  both greedy (temperature 0) and the Qwen3-4B recommended sampling
  (temperature 0.6 / top_p 0.95 / top_k 20); ~4,159 draft iterations per run
- Serving: vLLM 0.31.0, single GPU (eval), `--max-model-len 4096`,
  launched with `scripts/launch_vllm.py eval ... --spec-tokens 16
  --spec-method dspark` and `--provenance-dir` to capture artifacts
- Software: speculators @ `1c800f2b053bcd538fc11bf49ec843d4dfa81a95`
  (the exact commit the base checkpoint was trained with), vLLM 0.31.0
  (`db9527a46873454610df6dbedf79a36d6bf1a7f6`), torch 2.13.0+cu130,
  transformers 5.17.0
- Hardware: 8x NVIDIA A100-SXM4-80GB (6 training + 2 rendering; eval on 1)

## Reproduction

Fine-tune command (per LR; `--muon-lr` is 5x `--lr`):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node 6 \
  -m speculators.train \
  --verifier-name-or-path Qwen/Qwen3-4B \
  --speculator-type dspark \
  --from-pretrained <tulu3-b16 checkpoint> \
  --data-path <prepared OpenCodeInstruct data> \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path runs/lr_<LR>/checkpoints \
  --hidden-states-backend file \
  --hidden-states-path hidden_states \
  --block-size 16 --max-anchors 1024 \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --dflash-decay-gamma 7.0 \
  --loss-implementation fused \
  --per-position-loss-weight fixed-exp-decay \
  --optimizer muon --muon-lr <5x LR> --lr <LR> --weight-decay 0.0 \
  --scheduler-type cosine --scheduler-warmup-ratio 0.04 \
  --epochs 1 --total-seq-len 8192 --seed 42 \
  --fsdp-shard --on-missing generate --on-generate delete \
  --checkpoint-freq 1 --no-resume-from-checkpoint --num-workers 2
```

Render server (provides verifier hidden states during training):

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/launch_vllm.py train Qwen/Qwen3-4B \
  --target-layer-ids 1 9 17 25 33 \
  --hidden-states-path hidden_states \
  --provenance-dir provenance \
  -- --port 8000 --max-model-len 8194 --enforce-eager --data-parallel-size 2
```

Evaluation (per checkpoint, recommended sampling shown; temperature 0
variant swaps the gen-kwargs):

```bash
python scripts/launch_vllm.py eval Qwen/Qwen3-4B \
  --spec-model runs/lr_<LR>/checkpoints/0 \
  --spec-tokens 16 --spec-method dspark \
  --provenance-dir humaneval_results/lr_<LR> \
  -- --port 8010 --max-model-len 4096

cd scripts/evaluate && python evaluate.py \
  --target http://localhost:8010/v1 \
  --output-dir humaneval_results/lr_<LR> \
  throughput --subsets HumanEval \
  --gen-kwargs '{"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                 "max_completion_tokens": 128}'
```

Per-checkpoint provenance artifacts (`eval_command.txt`, `vllm_command.txt`,
`vllm.patch`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`) are
captured by `--provenance-dir` for every run.

## Results

HumanEval acceptance length (`1 + accepted speculative tokens / draft
iterations`, bonus token included). Baseline numbers are from the published
checkpoint's model card, measured with the same benchmark suite, vLLM
version, and serving settings.

| Configuration | Acceptance length, temp 0 | Acceptance length, recommended sampling (0.6/0.95/20) |
|---------------|--------------------------:|------------------------------------------------------:|
| tulu3 baseline (published card) | 5.380 | 5.406 |
| + 1 epoch OpenCodeInstruct @ lr 3e-4 | 5.132 | 5.114 |
| + 1 epoch OpenCodeInstruct @ lr 1.5e-4 | N/A | 5.348 |
| + 1 epoch OpenCodeInstruct @ lr 7.5e-5 | N/A | 5.534 |
| + 1 epoch OpenCodeInstruct @ lr 3e-5 | N/A | 5.604 |

Throughput/speedup vs. target-only decoding was not measured (acceptance
evaluation only).

## Observations

- **Negative result at 3e-4:** one epoch of code fine-tuning at the
  from-scratch LR degraded HumanEval acceptance by ~5% under both sampling
  settings. Fine-tuning an already-trained speculator likely needs a
  substantially smaller LR than training from scratch; this motivated
  dropping the planned 6e-4 arm and adding 3e-5 at the low end.
- **Monotonic LR trend:** HumanEval acceptance length improved steadily as
  the LR decreased (5.11 → 5.35 → 5.53 → 5.60). The lowest LR tested
  (3e-5, 1/20th of the from-scratch recipe) gave the best result, +3.7%
  over the tulu3 baseline. The curve had not flattened at the bottom of
  our range, so even lower LRs (or fewer steps) may improve further.
- In-domain validation acceptance lengths were nearly flat across LRs
  (6.96 / 7.07 / 6.96 for 1.5e-4 / 7.5e-5 / 3e-5), while HumanEval
  acceptance separated cleanly — the held-out OpenCodeInstruct split is
  too in-distribution to rank fine-tunes; the out-of-distribution
  benchmark was the discriminating signal.
- Next steps: extend the sweep below 3e-5 (e.g., 1.5e-5), try
  sub-epoch training at 3e-5, and blend tulu3 + OpenCodeInstruct to
  avoid regressing non-code domains.
- The regenerated OpenCodeInstruct data is code-domain but was generated by
  Qwen3-8B (thinking mode), while the verifier is Qwen3-4B; the draft sees
  a response distribution shift relative to its tulu3 training data in
  both style and domain.
- Ops notes: training with the default 12 dataloader workers per rank
  exhausted the container's ~4k process/thread budget (vLLM render workers
  + 6 FSDP ranks + harness); `--num-workers 2` resolved it. A stale Ray
  cluster hoarding ~900 threads compounded this.
- The 3e-4 run's post-training validation hung on thread exhaustion after
  the checkpoint was written; the checkpoint itself is complete (weights,
  optimizer, scheduler states all saved before validation begins).
- Not tested: throughput/speedup, acceptance on the other 8 benchmark
  subsets, more than 1 epoch, non-proportional Muon/AdamW LR ratios, and
  data-mixture effects (e.g., blending tulu3 + OpenCodeInstruct).
