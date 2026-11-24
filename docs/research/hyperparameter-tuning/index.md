# Hyperparameter tuning

Recipe sweeps for **base model training** — training a speculator from scratch. One knob varied over a range, everything else held fixed, with a recommended value as the deliverable.

The test is mechanical: if the training command loads `--from-pretrained`, it is a fine-tune and belongs in [finetuning experiments](../finetuning-experiments/index.md), even when the swept knob is identical. A base-training LR sweep and a fine-tuning LR sweep answer different questions and land on different values.

Start from [`TEMPLATE.md`](TEMPLATE.md). The conventions that apply to every entry are in the [hub overview](../index.md).

A sweep with two varied parameters is not a sweep. Split it, or state in the Summary that it confounds two variables and which one the conclusion is about.

## What we have learned

- [DFlash2 on Qwen3.8-27B: per-position loss weight, selector miss policy, loss normalization](loss-functions/dflash2-selector-and-loss-normalization-ablation-shubhra.md) — no run-to-run variance was collected, so all three results are descriptive rather than statistically separated. `dpace` gave a higher mean than `ce+tv, fixed-exp-decay` for DFlash2 at 5k and 25k steps, and a lower one for DSpark at both. `strict` beat `replace` on every subset at 5k steps — the clearest of the three signals — narrowing to mixed by 25k; it landed as a training option in Speculators PR #1217. The two loss normalizations differed by 0.01 at 20k steps, a tie.

## Sub-directories

| Directory                                    | Knob                                                                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------- |
| [`learning-rates/`](learning-rates/index.md) | LR, Muon LR, LR ratios, schedule shape, warmup fraction                               |
| [`block-sizes/`](block-sizes/index.md)       | block size, draft depth, max anchors                                                  |
| [`loss-functions/`](loss-functions/index.md) | loss composition and weighting, normalization, per-position weighting, position decay |

### Adding a sub-directory

If a sweep does not fit the three above, **add a new sub-directory rather than forcing it into the closest one**. Mis-filing is worse than a thin directory: a reader scanning `loss-functions/` for loss results should not have to guess that an optimizer sweep landed there.

To add one:

1. Create the directory with a kebab-case name for the knob, pluralized — `optimizers/`, `selectors/`, `sequence-lengths/`, `warmup-schedules/`.
2. Add an `index.md` saying what belongs in it, following the pattern of the existing three.
3. Add a row to the table above.

Nothing else needs updating. Do not add a sub-directory speculatively for a sweep nobody has run.

## Boundary with the other categories

| If you are…                                                              | File under                                                   |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| sweeping a base-training knob to pick its value, data fixed              | here                                                         |
| training from an existing checkpoint — anything with `--from-pretrained` | [finetuning experiments](../finetuning-experiments/index.md) |
| changing the training corpus                                             | [dataset ablations](../dataset-ablations/index.md)           |
| changing a serving flag against a fixed checkpoint                       | [vLLM knobs](../vllm-knobs/index.md)                         |
| measuring a target + drafter pair as served                              | [model evaluations](../model-evaluations/index.md)           |

Two cases that come up repeatedly:

- **The same knob, two phases.** A learning-rate sweep during base training is here; a learning-rate sweep during fine-tuning is a [finetuning experiment](../finetuning-experiments/index.md). They produce different answers — the fine-tuning optimum sits well below the from-scratch value — so keeping them apart is the point, not bookkeeping.
- **Block size.** Training-time block size is here. Block size as a *serving* knob against an already-trained drafter is a [vLLM knob](../vllm-knobs/index.md). Are you choosing a value to train with, or measuring what an existing checkpoint does under different serving settings?

## Page layout

Same order as [finetuning experiments](../finetuning-experiments/index.md), since these are training runs:

1. `## At a glance`
2. `## Reproduction` — the same six steps, in the same order
3. `## Summary`
4. `## Results`
5. `## Observations`
6. `## Setup`

### `## At a glance`

```markdown
| | |
|---|---|
| **Verifier** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Speculator** | DFlash2, trained from scratch, 5 draft layers |
| **Training dataset** | open-perfectblend mix, N rows / N tokens |
| **Evaluation dataset** | `RedHatAI/speculator_benchmarks`, temperature / top-p |
| **Knob swept** | training block size, 4 … 16 |
| **Values tried** | 4, at 5k and 25k steps |
| **Recommended value** | 8 |

| Value | Acceptance length (5k steps) | Acceptance length (25k steps) |
|-------|------------------------------|-------------------------------|
| 4 | | |
| **8** | | |
| 16 | | |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included. Measured on <dataset> at <sampling settings>.
```

- **Recommended value** is required. A sweep without a recommendation is unfinished — if the curve is flat or non-monotonic, say that and recommend accordingly ("flat between X and Y; use X, it trains faster").
- The sweep table carries **every value tried**, including the ones that regressed, in sweep order rather than ranked. The shape of the curve is the result; a table of only the winners hides whether the optimum is a peak or a plateau edge.
- Bold the recommended row.
- If the best value sits at the edge of the range you tried, say so explicitly in Observations — the real optimum may be outside it.

### `## Reproduction`

Identical six-step sequence to [finetuning experiments](../finetuning-experiments/index.md): environment, data preparation, render server, train, evaluation server, evaluate. Keep every step even when one did not apply, and keep the swept parameter as a placeholder rather than hard-coding one arm.

## Title

```yaml
title: "Qwen/Qwen3.8-27B speculator — block size sweep 4…16 (@USERNAME)"
```

Form: `<verifier> speculator — <knob> sweep <range> (@handle)`. Add `on <dataset>` only when the dataset is what distinguishes this entry from a neighbouring one.

Base training has no tuned checkpoint to name, so the title leads with the **verifier** the speculator is being trained for, followed by the word "speculator". Name the knob and the range; "hyperparameter sweep" alone tells a reader nothing. Link the verifier and dataset on first mention in the body.

## Required in Results

- Every arm, at matched steps. If arms ran to different step counts, give both and label which is which.
- An unswept baseline row measured the same way.
- Per-subset acceptance alongside any aggregate, and per-position acceptance where the knob targets position weighting.
- The noise floor at this scale. If two values differ by less than it, say they tie rather than ranking them.

## Interactions

Hyperparameters interact — a learning rate tuned at one block size is not optimal at another. State the values of the *other* knobs the sweep was run at, and do not present a recommendation as general when it was measured at one configuration. If you only swept at one setting, that is a limitation for Observations, not a caveat to omit.
