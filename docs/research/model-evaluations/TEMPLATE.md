---
title: Model evaluation template
---

# Model evaluation template

Copy the block below into `docs/research/model-evaluations/<Family>/<exact-HF-model-name>/<short-title>-<username>.md` and fill it in.

The section order here is deliberate and differs from other categories: the **At a glance** tables come first, Setup and Reproduction go at the bottom. Keep it. The rules are on the [category page](index.md), and the conventions that apply to every entry are in the [hub overview](../index.md).

The title starts with the Hugging Face identifier of the verifier — or `[Unpublished]` if the checkpoint is not on the Hub — and ends with the GitHub handle of the person who ran the work.

```markdown
---
title: "org/target-model-id — short description of what you measured (@USERNAME)"
tags:
  - DSpark
  - Qwen3
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# org/target-model-id — short description of what you measured (@USERNAME)

## At a glance

| | |
|---|---|
| **Model** | [org/drafter-id](https://huggingface.co/org/drafter-id) |
| **Verifier** | [org/target-model-id](https://huggingface.co/org/target-model-id), revision `abc1234` |
| **Evaluation settings** | suite, N subsets, temperature / top-p, prompts per subset |
| **Block size** | K (`num_speculative_tokens`; drafter trained depth K) |

| Dataset | Acceptance length | Acceptance length (greedy) |
|---------|-------------------|----------------------------|
| subset-1 | | |
| subset-2 | | |
| **Aggregate** | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included.

## Summary

One or two sentences on the headline result.

## Results

Throughput, interactivity, and concurrency behaviour. Compare the same target
model, workload, hardware, and serving settings. Define metrics and units; mark
measurements you did not collect as N/A.

| Configuration | Acceptance length | Output tokens/s | Mean ITL (ms) | Speedup vs. baseline |
|---------------|-------------------|-----------------|---------------|----------------------|
| Target only | N/A (by construction) | | | 1.00x |
| Target + drafter | | | | |

State the concurrency range and the steady-state measurement window.

## Observations

What surprised you, what failed, what you would try next, and what you did not
test. If only greedy was measured, say so here and note that it is an upper
bound on sampled acceptance.

## Setup

- Verifier: `org/target-model-id`, revision, precision
- Drafter: `org/drafter-id` or checkpoint, revision, precision, bundled or third-party
- Drafter trained depth, and depth served at
- Algorithm and algorithm-specific settings
- Evaluation: dataset and split, sample count, prompt/output lengths, temperature
- Serving: vLLM version, tensor/pipeline parallelism, concurrency, launch command
- Software: Speculators commit, PyTorch and CUDA versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Exact commands and configuration files, or links to reproducibility artifacts.
For evaluation, include `eval_command.txt`. For vLLM serving, include
`vllm_command.txt`, `vllm.patch`, `checkpoint_sha256.txt`, and
`drafter_checkpoint_sha256.txt` when generated.

Raw per-position acceptance data and other large tables go here, not above.
```
