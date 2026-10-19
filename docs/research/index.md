# Open Research Hub

A community log of speculative decoding experiments. Share findings from training a draft model, trying an algorithm, or evaluating a new target and drafter combination so others can build on your work.

Entries are experimental reports, not official benchmarks. Describe what you ran, the hardware you used, and what you measured. Negative results are welcome.

## Finding entries

Search for an algorithm or model family using the site search. Each entry carries tags so related experiments can be found together.

## How to submit

Copy the template below into `docs/research/<short-title>-<username>.md`, fill it in, and open a pull request. For example, `dspark-qwen3-shanjiaz.md`. New entries appear automatically in this section; no navigation edits are needed.

Start the filename with the algorithm and end it with your GitHub handle to group related entries and avoid filename collisions.

```markdown
---
title: "Short description of what you tried"
tags:
  - DSpark
  - Qwen3
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# Short description of what you tried

## Summary

One or two sentences on the headline result.

## Setup

- Target model: `org/target-model-id`, revision, precision
- Draft model: `org/draft-model-id` or checkpoint, revision, precision
- Algorithm: DSpark, number of speculative tokens, algorithm-specific settings
- Training (if applicable): dataset, sequence length, training steps, configuration
- Evaluation: dataset and split, sample count, prompt/output lengths, temperature
- Serving: vLLM version, tensor/pipeline parallelism, concurrency, launch command
- Software: Speculators commit, PyTorch and CUDA versions
- Hardware: GPU model, count, memory per GPU

## Reproduction

Include commands and configuration files or link to reproducibility artifacts.
For training, include `train_command.txt` and `speculators.patch`.
For evaluation, include `eval_command.txt`.
For vLLM serving, include `vllm_command.txt`, `vllm.patch`,
`checkpoint_sha256.txt`, and `drafter_checkpoint_sha256.txt` when generated.

## Results

Compare the same target model, workload, hardware, and serving settings.
Define metrics and units; mark measurements you did not collect as N/A.

| Configuration | Acceptance rate (%) | Mean accepted draft tokens per step | Output tokens/s | Speedup vs. baseline |
|---------------|---------------------|-------------------------------------|-----------------|----------------------|
| Target only | N/A | N/A | | 1.00x |
| Target + drafter | | | | |

## Observations

What surprised you, what failed, what you would try next, and what you did not test.
```

Replace `USERNAME` with your GitHub handle and `YYYY-MM-DD` with the experiment date. Results depend on software and model revisions, so the run date is more useful than the pull request merge date.

See [Evaluating Model Performance](../user_guide/tutorials/evaluating_performance.md) for evaluation commands and [Throughput vs. Interactivity](../user_guide/tutorials/throughput_interactivity.md) for comparisons across concurrency levels. Pass `--provenance-dir` when using `scripts/launch_vllm.py` to capture the serving artifacts.

### Crediting more than one author

Add one avatar per author, in credit order, and list their handles below:

```markdown
<div class="entry-byline" markdown>
[![ALICE](https://github.com/ALICE.png?size=64){ .entry-avatar }](https://github.com/ALICE)
[![BOB](https://github.com/BOB.png?size=64){ .entry-avatar }](https://github.com/BOB)
**[@ALICE](https://github.com/ALICE)** and **[@BOB](https://github.com/BOB)**
posted on **YYYY-MM-DD**
</div>
```

Credit contributors whose results or analysis you incorporated, with their permission. Link to referenced work in the body. For later contributions, add an updated date alongside the original experiment date.

## Conventions

- Use block-style YAML lists for `tags`, as in the template. Inline lists can be interpreted as Markdown links by the documentation link checker.
- Choose algorithms and model families that readers would search for, such as Eagle3, DFlash, DSpark, MTP, or Qwen3.
- State hardware, software revisions, datasets, and training and serving settings needed to reproduce the result.
- Define acceptance metrics explicitly, including whether the bonus token is counted in a reported acceptance length.
- Report a target-only baseline for speedup, and distinguish total throughput from per-user decode speed. Use comparable workloads and concurrency.
- Include failed experiments and limitations; do not present an untested hypothesis as a measured result.
