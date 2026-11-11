---
title: Finetuning experiment template
---

# Finetuning experiment template

Copy the block below into `docs/research/finetuning-experiments/<short-title>-<username>.md` and fill it in.

The section order is deliberate and differs from other categories: **At a glance**, then the six-step **Reproduction** block, then the narrative. Keep all six reproduction steps even if one did not apply — say why instead of deleting the heading, because reports are meant to be diffable against each other.

The rules are on the [category page](index.md); the conventions that apply to every entry are in the [hub overview](../index.md).

````markdown
---
title: "org/tuned-model-id on org/dataset-id — what varied (@USERNAME)"
tags:
  - DSpark
  - Qwen3
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# org/tuned-model-id on org/dataset-id — what varied (@USERNAME)

## At a glance

| | |
|---|---|
| **Tuned model** | [org/tuned-model-id](https://huggingface.co/org/tuned-model-id) |
| **Verifier** | [org/verifier-id](https://huggingface.co/org/verifier-id) |
| **Training dataset** | [org/dataset-id](https://huggingface.co/datasets/org/dataset-id), N rows / N tokens |
| **Evaluation dataset** | subset, temperature / top-p / top-k |
| **What varied** | the one parameter this ablation sweeps, and its range |

| Evaluation dataset | Acceptance length before | After | Change |
|--------------------|--------------------------|-------|--------|
| subset-1 | | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included. "Before" is the un-tuned starting checkpoint
measured on the same dataset at the same settings.

## Reproduction

### 1. Environment

Speculators commit, vLLM version, torch / CUDA / transformers versions, GPU
model and count.

### 2. Data preparation

```bash
# speculators regenerate-responses ...   (only if the corpus was regenerated)
# speculators prepare-data ...
```

### 3. Render server

Serves verifier hidden states during training. Start before the training run.

```bash
python scripts/launch_vllm.py train <verifier> \
  --target-layer-ids ... \
  --hidden-states-path hidden_states \
  --provenance-dir provenance \
  -- --port 8000 --max-model-len ... --enforce-eager
```

### 4. Fine-tune

Keep the swept parameter as a placeholder rather than hard-coding one arm.

```bash
python -m torch.distributed.run --standalone --nproc_per_node <N> \
  -m speculators.train \
  --verifier-name-or-path <verifier> \
  --speculator-type <algorithm> \
  --from-pretrained <starting checkpoint> \
  --data-path <prepared data> \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path runs/<arm>/checkpoints \
  --hidden-states-backend file --hidden-states-path hidden_states \
  ... \
  --seed 42
```

### 5. Evaluation server

```bash
python scripts/launch_vllm.py eval <verifier> \
  --spec-model runs/<arm>/checkpoints/0 \
  --spec-tokens <K> --spec-method <algorithm> \
  --provenance-dir results/<arm> \
  -- --port 8010 --max-model-len ...
```

### 6. Evaluate

```bash
cd scripts/evaluate && python evaluate.py \
  --target http://localhost:8010/v1 \
  --output-dir results/<arm> \
  throughput --subsets <subset> \
  --gen-kwargs '{"temperature": ..., "top_p": ..., "max_completion_tokens": ...}'
```

Provenance artifacts captured per run: `eval_command.txt`, `vllm_command.txt`,
`vllm.patch`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`,
`train_command.txt`, `speculators.patch`.

## Summary

One or two sentences on the headline result.

## Results

Every arm, including the ones that underperformed. Compare at matched steps; if
arms ran to different step counts, give both and label which is which.

| Arm | Steps | Acceptance length | Change vs. before |
|-----|-------|-------------------|-------------------|
| before (un-tuned) | — | | 1.00x |
| | | | |

## Observations

What surprised you, what failed, what you would try next, and what you did not
test. If only greedy was measured, say so and note it is an upper bound on
sampled acceptance.

## Setup

Full recipe per arm, with the varied parameter called out:

- Optimizer and all learning rates, schedule, warmup fraction
- Loss weighting, whether fused, per-position normalization
- Position decay type and gamma
- Seed, sequence length, max anchors, world size and parallelism
- Training corpus: every file, with row and token counts
- Speculators commit SHA and version
````
