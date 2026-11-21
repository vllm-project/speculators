# New evaluations

Trying out an **evaluation** — a benchmark, dataset, metric, harness, or scoring method — to decide whether it is worth using.

The object of study is the evaluation, not the model. Models are the vehicle you run it on.

Start from [`TEMPLATE.md`](TEMPLATE.md). The conventions that apply to every entry are in the [hub overview](../index.md).

## Belongs here

- Standing up a benchmark that has not been used before and seeing whether it tells you anything.
- A new acceptance or throughput metric, or a different way of computing one already reported.
- A new harness, judge model, or scoring method.
- Re-examining an evaluation already relied on — whether it still discriminates, how noisy it is, whether it has saturated.

The test: **would the report still matter if you swapped the model out?** If yes, it belongs here. If the conclusion is about a model's quality, it is a [model evaluation](../model-evaluations/index.md). If it is about a training knob, it is [hyperparameter tuning](../hyperparameter-tuning/index.md) or a [finetuning experiment](../finetuning-experiments/index.md).

A report that stands up a new benchmark *and* uses it to rank drafters is two reports. Validate the evaluation here; rank the models elsewhere, citing this entry.

## What we have learned

No entries yet.

## What a report here has to answer

A new evaluation is worth adopting only if it separates things worth telling apart. At minimum:

- What it measures, and what gap it fills against what is already run.
- **Does it discriminate?** Run it on configurations whose relative ordering is already known — target-only vs. drafter, or a drafter known to be better than another. An evaluation that cannot separate known-different configurations is not usable yet, and saying so is a complete and useful result.
- How it compares with the evaluations already in use, and **where it disagrees**. Disagreement is the interesting part.
- What it costs: prompts, wall-clock, GPU-hours, whether it needs a judge model.
- How noisy it is. An evaluation whose run-to-run spread exceeds the differences you want to rank cannot rank them.
- A recommendation: adopt, adopt for a specific case, or do not adopt.

## The trap specific to this category

**Until an evaluation is validated, its numbers are evidence about the evaluation, not about the models.**

An unvalidated benchmark that ranks drafter A above drafter B tells you the benchmark produced an ordering. It does not tell you A is better. Do not write model conclusions out of a first run of an untrusted evaluation, and do not let a ranking leak into the Summary as though it were a finding about the models.

If the new evaluation disagrees with one already trusted, say which you believe and why. "They disagree" is not a conclusion; neither is silently preferring the newer one.

## Title

```yaml
title: "<evaluation name> — <what you assessed> (@USERNAME)"
```

Name the evaluation first. Link it — paper, repo, or dataset page — on first mention.

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
| **Evaluation** | [name](link), revision or version |
| **What it measures** | one line |
| **Run on** | the models and configurations used as vehicles |
| **Compared against** | the evaluations already in use |
| **Discriminates known-different configs?** | yes / no / partially, with the pair used |
| **Run-to-run spread** | measured variance, or `N/A` if not collected |
| **Cost** | prompts, wall-clock, GPU-hours, judge model if any |
| **Recommendation** | adopt / adopt for X / do not adopt — required |
```

- **Recommendation is required.** "Inconclusive, needs N more runs" is a valid recommendation; silence is not.
- **Run-to-run spread** is `N/A` only if you genuinely did not measure it, and Observations then has to say the evaluation cannot yet rank anything close together.
- **Discriminates** needs the actual pair named. "Yes" without saying what it separated is unverifiable.

## Required in Results

- The known-ordering check, with the expected ordering stated *before* the measured one.
- Side-by-side comparison with the incumbent evaluation on the same configurations.
- Per-subset or per-slice results where the evaluation has them — an evaluation that only works on part of its own suite is worth knowing about.
- Cost measured, not estimated.

## Negative results

"We tried this benchmark and it does not separate our configurations" is a good report and saves everyone else the attempt. File it with the same care as a positive one, and make the Summary say it plainly rather than burying it under what was tried.
