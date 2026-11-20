---
title: "DSpark data ablation on Qwen3-4B: the two subset runs tie at matched compute; the full mix leads by ~1.7%"
tags:
  - DSpark
  - Qwen3
  - Data curation
  - Speculative decoding
---

<div class="entry-byline" markdown>
[![MeganEFlynn](https://github.com/MeganEFlynn.png?size=64){ .entry-avatar }](https://github.com/MeganEFlynn)
**[@MeganEFlynn](https://github.com/MeganEFlynn)** posted on **2026-11-02**
</div>

# DSpark data ablation on Qwen3-4B: the two subset runs tie at matched compute; the full mix leads by ~1.7%

## Summary

We trained three DSpark speculators for `Qwen/Qwen3-4B` on different slices of the same regenerated corpus at approximately matched compute (228k–259k optimizer steps): the full 13-file mixture, a tulu3-only subset (~26% of the mixture), and an 8-file "perfect blend" (~49% of the mixture). At matched compute the two subset runs tie within noise of each other (4.610 vs 4.611 acceptance length at temperature 0) and both land ~1.7% below the full-mix baseline (4.689). Per-subset acceptance tracks the training mix (math and code subsets improve when math/code-heavy sources are included), but the aggregate acceptance length is insensitive to subset choice at this scale.

## Setup

- Target model: `Qwen/Qwen3-4B`, BF16
- Draft model: DSpark, 5 layers, verifier hidden states from layers 1, 9, 17, 25, 33, Markov head rank 256 (vanilla), confidence head with Markov features, 615M parameters
- Algorithm: DSpark, 16 speculative tokens, greedy proposals, block size 16
- Training data (all from `inference-optimization/Qwen3-8B-Regenerated-Collection`, Qwen3-8B regenerated responses):
  - **full mix**: all 13 files, 1 epoch, 244,909 steps (AdamW, gamma 4.0, world size 4, speculators `ae96c74`)
  - **tulu3-only**: `tulu3_Qwen3-8B.jsonl` only (935,683 prepared rows, 3.27B tokens), 4 epochs, 259,361 steps
  - **perfect blend**: 8 files (metamathqa, ultrainteract, orca_math, ultrachat, evol_codealpaca, autoif, lmsys_arena, ultrafeedback; 1,382,556 prepared rows, ~4.2B tokens), ~2.7 epochs (max-steps capped), 252,775 steps
  - The tulu3 and perfect-blend runs share an identical recipe: Muon (Muon LR 3e-3, LR 6e-4), cosine schedule with 4% warmup, loss `0.1·CE + 0.9·TV` (fused), fixed exponential position decay gamma 7.0, seed 42, seq len 8,192, max anchors 1,024, 6 ranks (FSDP), speculators `1c800f2` (0.10.0a0), vLLM 0.31.0
- Evaluation: speculator_benchmarks acceptance eval, 9 subsets (HumanEval, math_reasoning, qa, question, rag, summarization, tool_call, translation, writing), max generation length 128, up to ~200-400 requests per category, ~26k draft iterations per run; temperature 0 and Qwen3-4B default sampling (0.6 / 0.95 / 20)
- Serving: vLLM 0.31.0, one server, `--spec-tokens 16 --spec-method dspark`, `scripts/launch_vllm.py eval`
- Software (tulu3-only and perfect-blend runs): speculators `1c800f2b053bcd538fc11bf49ec843d4dfa81a95`, transformers 5.17.0, torch 2.13.0+cu130
- Hardware: 8×A100-80GB per run (render/extraction server on GPUs 0-1 DP-2, training on GPUs 2-7); ~2.5 days wall per run

Published checkpoints:

- full mix: [`MeganEFlynn/Qwen3-4B-speculator-b16.dspark`](https://huggingface.co/MeganEFlynn/Qwen3-4B-speculator-b16.dspark)
- tulu3-only: [`MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark`](https://huggingface.co/MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark)
- perfect blend: [`MeganEFlynn/Qwen3-4B-speculator-perfectblend-b16.dspark`](https://huggingface.co/MeganEFlynn/Qwen3-4B-speculator-perfectblend-b16.dspark)

## Reproduction

Each published repo carries `train_command.txt` (the exact invocation), `config.json`/`config.py`, `training_state.json`, `val_metrics.json` with per-epoch history, both `acceptance_*.csv` files, and a full `reproduce_*.sh` script (clone speculators at the pinned commit, download and prepare the data, launch the render server, train, evaluate). The perfect-blend and tulu3 repos were trained from speculators `1c800f2b053bcd538fc11bf49ec843d4dfa81a95`.

Serving command for evaluation (all three checkpoints):

```bash
vllm serve Qwen/Qwen3-4B \
  --spec-model <checkpoint> \
  --spec-tokens 16 --spec-method dspark
```

## Results

Acceptance length = `1 + total accepted speculative tokens / total draft iterations` (the verifier bonus token is included). The target-only baseline row is N/A by construction: acceptance metrics are undefined without a drafter; no throughput/speedup comparison was collected in this series.

### Temperature 0 (greedy)

| Configuration                              | Weighted overall | HumanEval |  Math |        QA |  Question |       RAG | Summ. |      Tool | Transl. | Writing |
| ------------------------------------------ | ---------------: | --------: | ----: | --------: | --------: | --------: | ----: | --------: | ------: | ------: |
| Full mix, 1 epoch (244,909 steps)          |        **4.689** |     5.525 | 7.404 |     3.683 |     4.551 |     4.798 | 3.677 |     4.186 |   5.923 |   4.515 |
| Perfect blend, ~2.7 epochs (252,775 steps) |            4.610 | **5.831** | 7.696 |     3.528 |     4.430 |     4.669 | 3.665 |     4.100 |   5.093 |   4.454 |
| Tulu3 only, 4 epochs (259,361 steps)       |            4.611 |     5.380 | 7.110 | **3.612** | **4.419** | **4.694** | 3.615 | **4.163** |   6.164 |   4.352 |

### Qwen3-4B default sampling (0.6 / 0.95 / 20)

| Configuration                              | Weighted overall | HumanEval |      Math |        QA |  Question |       RAG | Summ. |      Tool | Transl. | Writing |
| ------------------------------------------ | ---------------: | --------: | --------: | --------: | --------: | --------: | ----: | --------: | ------: | ------: |
| Full mix, 1 epoch (244,909 steps)          |        **4.594** |     5.576 |     7.122 |     3.492 |     4.364 |     4.545 | 3.676 |     4.186 |   5.656 |   4.419 |
| Perfect blend, ~2.7 epochs (252,775 steps) |            4.487 | **5.691** | **7.323** |     3.475 |     4.197 |     4.509 | 3.646 |     4.061 |   4.737 |   4.331 |
| Tulu3 only, 4 epochs (259,361 steps)       |            4.501 |     5.406 |     7.004 | **3.509** | **4.200** | **4.374** | 3.574 | **4.082** |   6.013 |   4.227 |

## Observations

- **Compute-matched subsets tie.** The tulu3-only and perfect-blend runs land within 0.001 acceptance length of each other at temperature 0 (4.611 vs 4.610) and within 0.3% at default sampling, despite completely different data: general instruction (tulu3) vs a math/code-heavy 8-file mix. At ~250k steps, which slices of this corpus you pick matters much less than training long enough.
- **The full mixture still wins, but only by ~1.7%.** Cutting the corpus to 26–49% of the mixture (by token volume) and re-spending the same step budget on more epochs recovers most — not all — of the full-mix acceptance length.
- **Per-subset results do track the training data.** The perfect blend (with metamathqa + orca_math ≈ 595k rows and evol_codealpaca) is the best of the three on HumanEval at both temperatures (5.83 greedy) and on math under sampling (7.32); tulu3 leads on qa, question, rag, and translation. Domain-targeted data curation shows up per-domain but nets out in the aggregate.
- **Training-data val metrics are misleading across mixtures.** The tulu3-only run's validation acceptance length (7.15 on its own tulu3 split) reads higher than the perfect blend's (7.10 on its blend split), yet their benchmark results tie — validation splits drawn from the training mixture are not comparable across runs.
- **Recommended sampling costs ~2.0-2.7%** acceptance length vs greedy across all three runs (full mix −2.0%, tulu3 −2.4%, perfect blend −2.7%); code (HumanEval) is nearly flat, qa/rag drop most.
- Not tested: throughput/speedup at serving concurrency (acceptance length only), smaller step budgets (whether subset choice matters more at 50-100k steps), and the magpie/nemotron subsets (a fourth run was staged but paused).
