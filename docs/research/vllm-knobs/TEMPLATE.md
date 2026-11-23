---
title: vLLM knobs template
---

# vLLM knobs template

Copy the block below into `docs/research/vllm-knobs/<short-title>-<username>.md` and fill it in.

The rules are on the [category page](index.md) — in particular, pin the vLLM commit, vary one thing, and give per-concurrency numbers rather than a single headline. The conventions that apply to every entry are in the [hub overview](../index.md).

```markdown
---
title: "Short description of what you tried (@USERNAME)"
tags:
  - DSpark
  - vLLM
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# Short description of what you tried (@USERNAME)

## Summary

One or two sentences on the headline result.

## Setup

- Target model: `org/target-model-id`, revision, precision
- Draft model: `org/draft-model-id` or checkpoint, revision, precision
- Algorithm: number of speculative tokens, algorithm-specific settings
- Knob under test, and its values. Everything held fixed across arms.
- Evaluation: dataset and split, sample count, prompt/output lengths, temperature
- Serving: vLLM version and commit SHA, tensor/pipeline parallelism, concurrency,
  attention backend, prefix caching, launch command for every arm
- Software: Speculators commit, PyTorch and CUDA versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Include commands and configuration files or link to reproducibility artifacts.
For vLLM serving, include `vllm_command.txt`, `vllm.patch`,
`checkpoint_sha256.txt`, and `drafter_checkpoint_sha256.txt` when generated.

## Results

Compare the same target model, workload, hardware, and serving settings. Define
metrics and units; mark measurements you did not collect as N/A. Give one row
per concurrency rather than a single aggregate.

| Configuration | Concurrency | Acceptance length | Output tokens/s | Mean ITL (ms) | Speedup vs. baseline |
|---------------|-------------|-------------------|-----------------|---------------|----------------------|
| Target only | | N/A (by construction) | | | 1.00x |
| Baseline (knob off) | | | | | |
| Knob on | | | | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included.

State the steady-state measurement window: which requests are counted, and when
the window opens and closes.

## Observations

What surprised you, what failed, what you would try next. State the concurrency
levels, draft budgets, and model sizes you did **not** test.
```
