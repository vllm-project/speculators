---
title: Dataset ablation template
---

# Dataset ablation template

Copy the block below into `docs/research/dataset-ablations/<short-title>-<username>.md` and fill it in.

The rules are on the [category page](index.md) — in particular, match compute across arms and say what you matched on. The conventions that apply to every entry are in the [hub overview](../index.md).

```markdown
---
title: "Short description of what you tried (@USERNAME)"
tags:
  - DSpark
  - Qwen3
  - Data curation
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# Short description of what you tried (@USERNAME)

## Summary

One or two sentences on the headline result. Say whether arms were matched on
optimizer steps or on tokens.

## Setup

- Target model: `org/target-model-id`, revision, precision
- Draft model: algorithm, architecture summary, parameter count
- Algorithm: number of speculative tokens, block size, proposal method
- Training data, per arm: the exact source files by name, prepared row count,
  token count, epochs, and steps
- The shared recipe, identical across arms: optimizer and all learning rates,
  schedule and warmup, loss weighting, position decay, seed, sequence length,
  max anchors, world size / parallelism
- Evaluation: dataset and split, sample count, prompt/output lengths, temperature
- Serving: vLLM version, parallelism, launch command
- Software: Speculators commit SHA, PyTorch and CUDA versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Include commands and configuration files or link to reproducibility artifacts.
For training, include `train_command.txt` and `speculators.patch`. For
evaluation, include `eval_command.txt`. For vLLM serving, include
`vllm_command.txt`, `vllm.patch`, `checkpoint_sha256.txt`, and
`drafter_checkpoint_sha256.txt` when generated.

## Results

Every arm, with the full-mixture or production baseline as a reference row.
Per-subset acceptance alongside the aggregate — the aggregate is often flat
while individual subsets move.

| Arm | Rows / tokens | Epochs | Steps | Acceptance length | Change vs. baseline |
|-----|---------------|--------|-------|-------------------|---------------------|
| baseline (full mixture) | | | | | — |
| | | | | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included. Measured at <sampling settings>.

State what counts as noise at this scale. If two arms differ by less than that,
say they tie rather than ranking them.

## Observations

What surprised you, what failed, what you would try next, and what you did not
test.
```
