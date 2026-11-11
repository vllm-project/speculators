# Model evaluations

Measuring an existing target + drafter pair as served: acceptance length, acceptance rate, throughput, interactivity. No training involved.

Start from [`TEMPLATE.md`](TEMPLATE.md) in this directory, not a neighbouring one — model evaluations lead with an **At a glance** table block and push Setup and Reproduction to the bottom.

Always include a target-only baseline, and state whether the bonus token is counted in any reported acceptance length. The shared conventions in the [hub overview](../index.md) apply on top of everything here.

## Belongs here

Measuring a target + drafter pair as served. If a checkpoint was trained or fine-tuned for the experiment, it is a [finetuning experiment](../finetuning-experiments/index.md) or a [dataset ablation](../dataset-ablations/index.md) even when the report is all evaluation numbers.

A pure serving-configuration sweep against a fixed checkpoint belongs in [vLLM knobs](../vllm-knobs/index.md). The test: if the conclusion is about the *model*, it is here; if it is about a *flag*, it is there.

If the conclusion is about the **evaluation** — whether a benchmark is worth using, whether it discriminates, how noisy it is — it is a [new evaluation](../new-evaluations/index.md). Here, the evaluation is a trusted instrument; there, it is the thing under test.

## What we have learned

- [Mistral-Large-4-1T-A52B-NVFP4 with its bundled DSpark drafter](Mistral/Mistral-Large-4-1T-A52B-NVFP4/dspark-mistral-large-4-nvfp4-acceptance-meganeflynn.md) — at depth 3 on 4 × B300, acceptance length ran 2.62–3.29 across nine benchmark subsets at temperature 1.0 / top-p 0.95, best on math_reasoning and worst on qa. Greedy decoding, measured on two subsets, came out about 0.17 higher. Throughput was 1.9–2.4× target-only across 1–32 streams. The drafter's trained depth is exactly 3 — requesting 8 speculative tokens stops the server starting.

## Page layout

A reader should get the headline facts without scrolling, and never have to wade through commit SHAs to find an acceptance number. Reproduction detail is kept in full — it just lives at the bottom.

1. `## At a glance` — the two tables below. First thing after the byline.
2. `## Summary` — one or two sentences on the headline result.
3. `## Results` — the fuller numbers: throughput, interactivity, concurrency sweeps, per-sampling-mode breakdowns.
4. `## Observations` — what surprised you, what failed, what you did not test.
5. `## Setup` — full pinning: model and drafter revisions, software versions, hardware.
6. `## Reproduction` — exact commands, provenance artifacts, raw per-position data, appendices.

This deliberately differs from the order some other categories use. Nothing is dropped; Setup and Reproduction move to the end.

Do not push narrative detail down into Reproduction to make the top look tidy. Only commands, pins, raw tables, and artifacts go there.

### `## At a glance`

Two tables, in this order. Keep them scannable — no prose between them.

```markdown
## At a glance

| | |
|---|---|
| **Model** | [org/drafter-id](https://huggingface.co/org/drafter-id) |
| **Verifier** | [org/target-id](https://huggingface.co/org/target-id), revision `abc1234` |
| **Evaluation settings** | `speculator_benchmarks`, 9 subsets, temperature 1.0 / top-p 0.95, N prompts per subset |
| **Block size** | 3 (`num_speculative_tokens`; drafter trained depth 3) |

| Dataset | Acceptance length | Acceptance length (greedy) |
|---------|-------------------|----------------------------|
| humaneval | 3.12 | 3.33 |
| math_reasoning | 3.29 | 3.48 |
| qa | 2.62 | N/A |
| **Aggregate** | **2.98** | N/A |

Acceptance length = `1 + accepted speculative tokens / draft iterations`; the
verifier bonus token is included.
```

Row and column definitions:

- **Model** — the speculator / drafter checkpoint under evaluation, linked. Say if it is the one bundled with the target rather than a third-party checkpoint.
- **Verifier** — the target model it drafts for, linked, with its revision SHA. This is the identifier that leads the title.
- **Evaluation settings** — benchmark suite, number of subsets, temperature and top-p, prompts per subset. Enough that a reader knows what the acceptance numbers are conditioned on.
- **Block size** — the speculative block size served at, with the parameter it maps to in parentheses, since the name differs by algorithm (`num_speculative_tokens`, draft depth, GDN block). If the drafter's trained depth differs from what you served, give both.
- **Acceptance length per dataset** — one row per evaluation subset, plus an aggregate row in bold. One column per sampling mode measured. `N/A` where a mode was not run; never leave a mode out silently.

Add rows to the first table for anything else constant across the run (hardware, precision, tensor-parallel size) if it is needed to read the numbers. Do not add anything that varies per dataset — that belongs in the second table or in Results.

The acceptance-length definition goes immediately under the second table, every time. A bare number is not interpretable without it.

## Greedy-only submissions

Acceptance is systematically higher under greedy decoding than under realistic sampling — in the Mistral Large 4 entry above, greedy raised acceptance length by about 0.17. A greedy-only report therefore reads as a better result than users will see in production.

If you can, run the evaluation at the model's recommended sampling settings before filing. Adding a sampling column later means re-running, not editing.

If that is not feasible, file it anyway — it is a real measurement and belongs here. But then:

- The `Evaluation settings` row says greedy / temperature 0 explicitly.
- The acceptance table has no second column implying a mode that was not measured.
- Observations states that acceptance at recommended sampling was not measured, and that greedy numbers are an upper bound on what sampled decoding will show.

## Title and model link

The title must begin with the Hugging Face identifier of the verifier (target model), in `org/model` form, followed by an em dash, the description, and the author handle in parentheses:

```yaml
title: "Qwen/Qwen3-4B — DSpark drafter acceptance (@MeganEFlynn)"
```

Identifier leads, handle trails. Keep the middle short and descriptive — the measured numbers belong in At a glance, not the title.

The `#` heading at the top of the body matches the title. The identifier is the full `org/model` string as it appears on the Hub, exact casing — not a prose name like "Qwen3 4B".

Always link the model. On first mention in the body, and in the At a glance table:

```markdown
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
```

### Checkpoints not on Hugging Face

Use `[Unpublished]` in place of the identifier, followed by the checkpoint name:

```yaml
title: "[Unpublished] dspark-qwen3-4b-codefinetune-run7 — acceptance (@MeganEFlynn)"
```

There is no link to add in this case. Say in the At a glance table and in Setup where the checkpoint actually lives — a path, bucket, or run id — so a reader knows what they would need to reproduce it. Publishing the checkpoint first, where you can, makes the entry far more useful.

## Placement in the tree

```
model-evaluations/
  <Family>/
    <exact-HF-model-name>/
      <short-title>-<username>.md
```

- Family directories are capitalized — `Qwen`, `Mistral`, `Gemma`. For a model whose family is not its Hugging Face org (`mistralai/Mistral-Large-4-...` lives under `Mistral/`, `google/gemma-4-31B-it` under `Gemma/`), the family directory is the family name and the org is recorded in the entry's Setup section.
- Model directories use the exact Hugging Face model name, case included — `Qwen3-4B`, not `qwen3-4b` — so a path maps directly onto a model id. This differs from the kebab-case used for category directories, deliberately: these are identifiers, not labels.
- A quantized variant is its own directory when the quantization is in the model name. `Mistral-Large-4-1T-A52B-NVFP4` and a BF16 sibling are different checkpoints with different acceptance behaviour; do not file them together.

Create a new family or model directory as needed. Nothing else has to be updated.

## Required in Setup

- Target **and** drafter revision SHAs. A model id alone is not enough; these models are re-uploaded.
- Precision for both, and for mixed-precision checkpoints say which parts are in which format.
- Whether the drafter is the one bundled with the target or a third-party checkpoint.
- The drafter's trained depth, and the depth you served at. Requesting more speculative tokens than a drafter was trained for can fail outright — if you hit a limit, record it in Observations.

## Required in Results

- A target-only baseline row, from the same hardware and serving settings.
- Acceptance reported per evaluation subset, not only as a single aggregate. Aggregates hide that code and math behave differently.
- Sampling settings on every acceptance number. Report greedy and recommended-sampling separately if both were measured; do not average them.
- Throughput and interactivity across the concurrency range tested, with the range stated.

If you are comparing against numbers from a published model card rather than your own baseline run, say so explicitly and state that the benchmark suite, vLLM version, and serving settings match.
