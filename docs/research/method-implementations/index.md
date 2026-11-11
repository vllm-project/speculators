# Method implementations

Bringing up a speculative decoding **method** — an algorithm, drafter architecture, verification policy, sampling path, loss, or kernel — in Speculators, vLLM, or both. Porting from a paper, adding a new drafter type, or implementing a mechanism such as adaptive verification or a new selector.

The deliverable is **does it work, is it correct, and what landed** — not what the best configuration is.

Start from [`TEMPLATE.md`](TEMPLATE.md). The conventions that apply to every entry are in the [hub overview](../index.md).

## Belongs here

- A new drafter architecture or speculative algorithm.
- A mechanism inside an existing method: a verification policy, a sampling path, a loss, a kernel.
- Making an existing method work on a model family or precision it did not support.
- Reproducing a published method and checking it against the paper.

The first speedup number from a new implementation belongs here as a smoke test. A considered performance comparison belongs in one of the other categories, citing this entry.

## Does not belong here

| If you are…                                | File under                                                                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| tuning a method that already works         | [hyperparameter tuning](../hyperparameter-tuning/index.md) or [finetuning experiments](../finetuning-experiments/index.md) |
| measuring a trained target + drafter pair  | [model evaluations](../model-evaluations/index.md)                                                                         |
| changing a serving flag                    | [vLLM knobs](../vllm-knobs/index.md)                                                                                       |
| assessing a benchmark rather than a method | [new evaluations](../new-evaluations/index.md)                                                                             |

## What we have learned

No entries yet.

## Keep "runs", "correct" and "fast" apart

This is the rule this category exists to enforce. An implementation report answers three questions in order, and they are not the same question:

1. **Does it run?** It loads and serves without crashing, on the paths you exercised. Name the paths — architectures, batch shapes, prefill/decode mixes, CUDA graph modes, parallelism settings. "It works" without saying what was exercised means nothing.
2. **Is it correct?** It matches a reference: the paper's numbers, an existing implementation, output equivalence against a non-speculative baseline, or unit tests. Plausible-looking output is not correctness evidence. **Speculative decoding that silently degrades quality still produces fluent text.**
3. **Is it fast?** Only meaningful once the first two hold, and an untuned first run is not an answer.

Most implementation reports stop after 1 and 2, and that is fine. Say which rung you reached. Stopping at 2 is a complete report; claiming 3 from an untuned first run is not.

**Never let a first measurement become a verdict on the method.** An unoptimised implementation that is slower than the baseline tells you about the implementation, not the algorithm. Label smoke-test numbers as smoke tests, in the table and in the prose.

## Required evidence

- **Correctness check, or an explicit statement that there is none.** Name the reference and what matched. If quality was checked by a task benchmark, give the sample count — a 100-question check is a small check and should say so.
- **Scope**: what is implemented and what is not. A method that works for one architecture, one precision, or one parallelism setting must say so in At a glance, not in a closing caveat.
- **Unsupported paths and guards**: what fails, how it fails, and whether the failure is guarded or silent. A silent wrong answer is a finding in its own right.
- **Where it landed**: PR links, and whether merged, open, or abandoned. If local-only and unmerged, say that.

## Title

```yaml
title: "DFlash2 selector miss policy — Speculators implementation (@USERNAME)"
```

Form: `<method or mechanism> — <what you implemented, and where> (@handle)`. Name the method first. Keep the finding out of it, as everywhere else.

## Page layout

1. `## At a glance`
2. `## Summary`
3. `## Results`
4. `## Observations`
5. `## Setup`
6. `## Reproduction`

### `## At a glance`

```markdown
| | |
|---|---|
| **Method** | name, with a link to the paper or reference implementation |
| **Implemented in** | Speculators / vLLM / both, with commit or branch |
| **Landed as** | PR links, and merged / open / not submitted |
| **Scope** | architectures, precisions and parallelism settings covered |
| **Not supported** | the paths that do not work, and whether they are guarded |
| **Correctness check** | what it was compared against, and the result — or `none` |
| **Status** | prototype / in review / landed / abandoned |
| **Smoke-test numbers** | first measurements if any, labelled as untuned — or `N/A` |
```

`Correctness check: none` is an acceptable and honest value. An empty or omitted row is not.

## Negative results

An implementation that did not work, or that worked and was not worth keeping, is worth writing up — it stops the next person repeating it. Say what blocked it and how far you got.
