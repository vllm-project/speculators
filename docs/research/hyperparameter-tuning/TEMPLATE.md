---
title: Hyperparameter sweep template
---

# Hyperparameter sweep template

For **base model training** — training a speculator from scratch. If the run loads `--from-pretrained` it is a fine-tune: use the [finetuning experiment template](../finetuning-experiments/TEMPLATE.md) instead.

Copy the block below into `docs/research/hyperparameter-tuning/<sub-directory>/<short-title>-<username>.md` and fill it in. File under [`learning-rates/`](learning-rates/index.md), [`block-sizes/`](block-sizes/index.md), [`loss-functions/`](loss-functions/index.md), or a new sub-directory you create — see the [category page](index.md) for how to add one.

Keep all six reproduction steps even if one did not apply — say why instead of deleting the heading, because reports are meant to be diffable against each other.

````markdown
---
title: "org/verifier-id speculator — <knob> sweep <range> on <dataset> (@USERNAME)"
tags:
  - DFlash2
  - Qwen3
---

<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>

# org/verifier-id speculator — <knob> sweep <range> on <dataset> (@USERNAME)

## At a glance

| | |
|---|---|
| **Verifier** | [org/verifier-id](https://huggingface.co/org/verifier-id) |
| **Speculator** | algorithm, trained from scratch, architecture summary |
| **Training dataset** | [org/dataset-id](https://huggingface.co/datasets/org/dataset-id), N rows / N tokens |
| **Evaluation dataset** | subset, temperature / top-p / top-k |
| **Knob swept** | the parameter and its range |
| **Values tried** | N, at N steps |
| **Recommended value** | the deliverable — required |
| **Other knobs held at** | the configuration this sweep was run at |

| Value | Acceptance length | Change vs. baseline |
|-------|-------------------|---------------------|
| baseline | | — |
| | | |
| **recommended** | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included. Measured on <dataset> at <sampling settings>.

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

### 4. Train

From scratch — no `--from-pretrained`. Keep the swept parameter as a
placeholder rather than hard-coding one arm.

```bash
python -m torch.distributed.run --standalone --nproc_per_node <N> \
  -m speculators.train \
  --verifier-name-or-path <verifier> \
  --speculator-type <algorithm> \
  --data-path <prepared data> \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path runs/<arm>/checkpoints \
  --hidden-states-backend file --hidden-states-path hidden_states \
  --block-size <K> --max-anchors ... \
  --loss-fn '{...}' \
  --optimizer ... --lr <LR> \
  --scheduler-type cosine --scheduler-warmup-ratio ... \
  --epochs ... --total-seq-len ... --seed 42
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

Provenance artifacts captured per run: `train_command.txt`,
`speculators.patch`, `eval_command.txt`, `vllm_command.txt`, `vllm.patch`,
`checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`.

## Summary

One or two sentences: the recommended value and what it buys.

## Results

Every value tried, in sweep order, including the ones that regressed. Compare
at matched steps.

| Value | Steps | Acceptance length | Per-subset breakdown | Change vs. baseline |
|-------|-------|-------------------|----------------------|---------------------|
| baseline | | | | — |
| | | | | |

State the noise floor at this scale. If two values differ by less than it, say
they tie rather than ranking them.

## Observations

Whether the curve is a peak or a plateau edge. Whether the optimum sits at the
edge of the range tried — if so, the real optimum may be outside it. What you
did not test, and which other knobs this recommendation is conditional on.

## Setup

- Verifier: `org/verifier-id`, revision, precision
- Speculator architecture and algorithm settings
- Full recipe, identical across arms except the swept knob: optimizer and
  learning rates, schedule, warmup, loss weighting, position decay, seed,
  sequence length, max anchors, world size and parallelism
- Training corpus: every file, with row and token counts
- Speculators commit SHA and version
- Hardware: GPU model, count, memory per GPU
````
