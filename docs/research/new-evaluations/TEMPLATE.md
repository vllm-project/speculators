---
title: New evaluation template
---

# New evaluation template

Copy the block below into `docs/research/new-evaluations/<short-title>-<username>.md` and fill it in.

The object of study is the **evaluation**, not the model. Until the evaluation is validated, its numbers are evidence about the evaluation. Do not write model conclusions from a first run of an untrusted evaluation.

The rules are on the [category page](index.md); the conventions that apply to every entry are in the [hub overview](../index.md).

```markdown
---
title: "<evaluation name> — what you assessed (@USERNAME)"
tags:
  - Evaluation
  - Qwen3
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# <evaluation name> — what you assessed (@USERNAME)

## At a glance

| | |
|---|---|
| **Evaluation** | [name](link), revision or version |
| **What it measures** | one line |
| **Run on** | models and configurations used as vehicles |
| **Compared against** | the evaluations already in use |
| **Discriminates known-different configs?** | yes / no / partially — name the pair |
| **Run-to-run spread** | measured variance, or `N/A` |
| **Cost** | prompts, wall-clock, GPU-hours, judge model if any |
| **Recommendation** | adopt / adopt for X / do not adopt — required |

## Summary

One or two sentences on whether this evaluation is worth using, and why.

## Results

### Known-ordering check

State the expected ordering **before** the measured one.

| Configuration | Expected rank | New evaluation | Incumbent evaluation |
|---------------|---------------|----------------|----------------------|
| target only | | | |
| target + drafter | | | |

### Agreement with the incumbent

Where the two agree, where they disagree, and which you believe.

### Cost

Measured, not estimated.

## Observations

Whether the evaluation is usable as-is, what would have to change, and what you
did not test. If run-to-run spread was not measured, say the evaluation cannot
yet rank anything close together.

## Setup

- Evaluation: name, version or commit, dataset and split, prompt count
- Models used as vehicles: ids, revisions, precision
- Serving: vLLM version, parallelism, launch command
- Judge model and settings, if any
- Software: Speculators commit, PyTorch and CUDA versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Exact commands and configuration files, or links to reproducibility artifacts.
Include `eval_command.txt` and, where a server was launched,
`vllm_command.txt` and the other `--provenance-dir` artifacts.
```
