---
title: Method implementation template
---

# Method implementation template

Copy the block below into `docs/research/method-implementations/<short-title>-<username>.md` and fill it in.

Keep "it runs", "it is correct" and "it is fast" as separate claims, and say which rung this report reached. Stopping at correctness is a complete report. An untuned first measurement is a smoke test, not a verdict on the method.

The rules are on the [category page](index.md); the conventions that apply to every entry are in the [hub overview](../index.md).

```markdown
---
title: "<method or mechanism> — what you implemented, and where (@USERNAME)"
tags:
  - DSpark
  - vLLM
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# <method or mechanism> — what you implemented, and where (@USERNAME)

## At a glance

| | |
|---|---|
| **Method** | name, with a link to the paper or reference implementation |
| **Implemented in** | Speculators / vLLM / both, with commit or branch |
| **Landed as** | PR links, and merged / open / not submitted |
| **Scope** | architectures, precisions and parallelism settings covered |
| **Not supported** | paths that do not work, and whether they are guarded or fail silently |
| **Correctness check** | what it was compared against and the result — or `none` |
| **Status** | prototype / in review / landed / abandoned |
| **Smoke-test numbers** | first measurements, labelled untuned — or `N/A` |

## Summary

One or two sentences: what was implemented, and which of runs / correct / fast
it reached.

## Results

### Does it run?

Name the paths exercised: architectures, batch shapes, prefill/decode mixes,
CUDA graph modes, parallelism settings.

### Is it correct?

The reference, the comparison, and the result. State sample counts for any
task-based check.

| Check | Reference | Result |
|-------|-----------|--------|
| | | |

### Smoke-test performance

Only if measured. Label it untuned and do not rank the method on it.

| Configuration | Acceptance length | Output tokens/s |
|---------------|-------------------|-----------------|
| baseline | | |
| this implementation | | |

## Observations

What was harder than expected, what is still unsupported, what you did not
test, and what the next person should know before building on it.

## Setup

- Models used to exercise the implementation: ids, revisions, precision
- Speculators and vLLM commits or branches
- Serving configuration and launch command
- Software: PyTorch, CUDA, transformers versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Exact commands, branches, and patches. Include `vllm_command.txt` and the other
`--provenance-dir` artifacts where a server was launched, and link the PRs.
```
