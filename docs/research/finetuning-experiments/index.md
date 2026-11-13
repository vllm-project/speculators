# Finetuning experiments

Training that starts from an existing speculator checkpoint — anything loading `--from-pretrained`. Adapting a drafter to a new domain or corpus, and the recipe choices made while doing it: fine-tuning learning rate, epochs, schedule.

Start from [`TEMPLATE.md`](TEMPLATE.md) in this directory, not a neighbouring one. Entries lead with an **At a glance** block, followed by the same six-step **Reproduction** sequence in every report so finetunes can be diffed against each other.

The conventions that apply to every entry are in the [hub overview](../index.md).

## Belongs here

- **Base model training** — training a speculator from scratch — is [hyperparameter tuning](../hyperparameter-tuning/index.md), including sweeps over learning rate, block size, and loss functions. The same knob swept in the two phases lands on different values, so the two categories are kept apart deliberately.
- If the corpus changed and the recipe did not, it is a [dataset ablation](../dataset-ablations/index.md).
- If nothing was trained, it is not this category.

## What we have learned

- [DSpark Qwen3-4B speculator fine-tuned on regenerated OpenCodeInstruct, LR swept 3e-4 … 7.5e-6](dspark-qwen3-4b-code-finetune-lr-ablation-meganeflynn.md) — the 1-epoch curve peaked at 3e-5 (5.60 HumanEval acceptance length vs a 5.41 tulu3 baseline, quoted from the published model card rather than re-measured); the best arm overall was 3 epochs at 7.5e-5 at 5.646, +4.4%. Fine-tuning at the from-scratch LR of 3e-4 clearly regressed, losing about 5% — the largest effect in the study. Single runs per arm with no variance reported, so the closer rankings near the optimum are suggestive rather than established. The 1-epoch curve had not flattened at the bottom of the range tested.

## Page layout

1. `## At a glance` — the tables below. First thing after the byline.
2. `## Reproduction` — the standard six-step block, same steps in the same order in every report.
3. `## Summary` — one or two sentences on the headline result.
4. `## Results` — the full comparison across arms.
5. `## Observations` — what surprised you, what failed, what you did not test.
6. `## Setup` — full recipe and pinning for every arm.

This differs from [model evaluations](../model-evaluations/index.md), which puts Reproduction at the bottom. Here it sits directly under At a glance so that re-running a finetune is the first thing the page supports.

### `## At a glance`

Two tables, no prose between them.

```markdown
## At a glance

| | |
|---|---|
| **Tuned model** | [MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark](https://huggingface.co/MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark) |
| **Verifier** | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| **Training dataset** | [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), regenerated with `Qwen/Qwen3-8B`, 114,056 rows / 0.56B tokens |
| **Evaluation dataset** | HumanEval, temperature 0.6 / top-p 0.95 / top-k 20 |
| **What varied** | fine-tuning learning rate, 3e-4 … 7.5e-6 |

| Evaluation dataset | Acceptance length before | After | Change |
|--------------------|--------------------------|-------|--------|
| HumanEval | 5.41 | 5.646 | +4.4% |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included.
```

- **Tuned model** — the checkpoint that received gradient updates, with its Hugging Face identifier, linked. Not the verifier.
- **Training dataset** — Hugging Face dataset id where one exists, linked, with row and token counts. Note regeneration here; full composition of a mixture goes in Setup.
- **Evaluation dataset** — what the before/after numbers were measured on, with the sampling settings. This is usually *not* the training dataset.
- **Before / After** — acceptance length of the starting checkpoint and of the tuned checkpoint, on the same evaluation dataset at the same settings. "Before" should be your own measurement of the un-tuned checkpoint; if it comes from a model card instead, say so in that cell.
- One row per evaluation dataset if you measured several. Report the best arm in this table and say which arm it is; the full sweep goes in Results.

Do not put a tuned-checkpoint number next to a before-number measured at different sampling settings or on a different subset. That comparison is the entire point of the table.

### `## Reproduction`

Every report carries the same six steps, in this order, even when a step was skipped — in that case keep the heading and say why it did not apply. Adjust flags to the algorithm; do not drop steps or reorder them.

1. **Environment** — speculators commit, vLLM version, torch / CUDA / transformers, GPU model and count.
2. **Data preparation** — `speculators prepare-data`, plus `speculators regenerate-responses` where the corpus was regenerated. Paste the exact invocation.
3. **Render server** — `scripts/launch_vllm.py train <verifier>`, which serves verifier hidden states during training. Starts before the training run.
4. **Fine-tune** — the `speculators.train` command, with the varied parameter marked as a placeholder rather than hard-coded to one arm.
5. **Evaluation server** — `scripts/launch_vllm.py eval <verifier> --spec-model <checkpoint>`.
6. **Evaluate** — `scripts/evaluate/evaluate.py`, with the gen-kwargs shown. If both greedy and recommended sampling were run, show one and name the swap.

Pass `--provenance-dir` on every `launch_vllm.py` invocation and close the section by listing the artifacts captured.

The point of fixing the steps is that reports become diffable: a reader comparing two finetunes should be able to read the same step in both and see only the intended difference.

## Title

The title names **the model that was tuned and the dataset it was tuned on**, then what varied, then the author handle:

```yaml
title: "MeganEFlynn/Qwen3-4B-speculator-tulu3-b16.dspark on nvidia/OpenCodeInstruct — learning-rate ablation (@MeganEFlynn)"
```

Form: `<tuned-model> on <dataset> — <what varied> (@handle)`.

- **Tuned model** is the checkpoint that actually received gradient updates — the speculator — not the verifier it drafts for. Naming the base model here is the common mistake: fine-tuning `Qwen3-4B-speculator-tulu3-b16.dspark` is not fine-tuning `Qwen/Qwen3-4B`.
- **Training from scratch** does not belong in this category at all — it is base model training, so file it under [hyperparameter tuning](../hyperparameter-tuning/index.md) and use that directory's title form.
- **Dataset** is the Hugging Face dataset id where one exists (`nvidia/OpenCodeInstruct`), otherwise the corpus name. Use `[Unpublished]` in place of the id for a corpus that is not published, and say in Setup where it lives.
- **Mixtures** use the mixture's common name rather than listing sources in the title. The full composition — every file, with row and token counts — goes in Setup.
- **Regenerated corpora** are named by their source dataset in the title; record the regeneration model and settings in Setup. `nvidia/OpenCodeInstruct` regenerated with `Qwen/Qwen3-8B` is still `nvidia/OpenCodeInstruct` in the title.

Link both the tuned model and the dataset on first mention in the body, using the full id as the link text.

Filenames keep the algorithm-first convention so related work groups together, and should still carry the model and dataset: `dspark-qwen3-4b-code-finetune-lr-ablation-meganeflynn.md`.

## Hold data fixed, and show that you did

- Name the exact corpus and its row and token counts, identically for every arm.
- If an arm used different data for any reason, that is a confound. State it in the Summary, not only in Observations.

## Required in Setup

The full recipe, per arm, with the varied parameter called out explicitly:

- Optimizer and all learning rates (including a separate Muon LR where applicable), schedule, warmup fraction.
- Loss: the exact weighting, e.g. `0.1·CE + 0.9·TV`, whether fused, and how per-position loss is normalized.
- Position decay type and gamma.
- Seed, sequence length, max anchors, world size and parallelism strategy.
- Speculators commit SHA and version. These results are sensitive to it; a recipe without a commit is not reproducible.

Say which single parameter the ablation varies. If more than one differs between arms, the comparison is not clean and the entry must say so.

## Required in Results

- Compare at matched steps. If arms ran to different step counts, give both the matched-step comparison and the final numbers, and label which is which.
- Per-subset acceptance plus aggregate, and per-position acceptance rate where the change targets position weighting.
- The unmodified recipe as a baseline arm.

## Failures

Configurations that diverged, collapsed, or underperformed stay in the entry. A learning-rate ablation that only reports the winning LR is not an ablation — the shape of the curve is the result.
