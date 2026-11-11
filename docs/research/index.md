# Open Research Hub

A community log of speculative decoding experiments. Share findings from training a draft model, trying an algorithm, or evaluating a new target and drafter combination so others can build on your work.

Entries are experimental reports, not official benchmarks. Describe what you ran, the hardware you used, and what you measured. Negative results are welcome.

## Finding entries

Entries are grouped by topic. Each category has its own page describing what belongs in it, the rules specific to it, and a template to start from. You can also search for an algorithm or model family using the site search — each entry carries tags so related experiments can be found together.

| Category                                                  | Scope                                                                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Model evaluations](model-evaluations/index.md)           | Measuring an existing target + drafter pair: acceptance, throughput, interactivity. No training.                     |
| [New evaluations](new-evaluations/index.md)               | Assessing an evaluation itself — a benchmark, metric, harness or judge — to decide whether to adopt it.              |
| [Method implementations](method-implementations/index.md) | Bringing up an algorithm or drafter architecture in Speculators or vLLM: does it run, is it correct, what landed.    |
| [Dataset ablations](dataset-ablations/index.md)           | Varying training data — mixtures, subsets, curation, regeneration — at matched compute.                              |
| [Hyperparameter tuning](hyperparameter-tuning/index.md)   | Base model training sweeps — learning rates, block sizes, loss functions — to pick a value.                          |
| [Finetuning experiments](finetuning-experiments/index.md) | Training from an existing checkpoint (`--from-pretrained`): domain adaptation and the recipe used to do it.          |
| [vLLM knobs](vllm-knobs/index.md)                         | Serving-side configuration: adaptive verification, CUDA graph capture sizes, draft budget, parallelism, concurrency. |

### Choosing a category

Pick the category matching your **independent variable** — what you changed, not what you measured. A data ablation evaluated on throughput is still a dataset ablation.

If a report genuinely changes two things at once, that is usually a sign it should be two entries. If it cannot be split, file it under the variable the Summary leads with and say in the text that it spans both.

Two boundaries come up repeatedly:

- **Training phase, not knob.** From-scratch base training is [hyperparameter tuning](hyperparameter-tuning/index.md); anything loading `--from-pretrained` is a [finetuning experiment](finetuning-experiments/index.md). The same knob swept in each phase lands on a different value.
- **Model or flag.** If the conclusion is about the *model*, it is a [model evaluation](model-evaluations/index.md); if it is about a *flag*, it is a [vLLM knob](vllm-knobs/index.md).

Directory names are kebab-case to keep shell and `mkdocs` paths free of spaces. The table above gives the display title for each.

## How to submit

Copy the `TEMPLATE.md` from the category you are filing under into that category as `<short-title>-<username>.md`, fill it in, and open a pull request. New entries appear automatically in the navigation; no navigation edits are needed.

Start the filename with the algorithm and end it with your GitHub handle to group related entries and avoid filename collisions. Do not put spaces in filenames.

Keep the section order your category's template gives. Several categories lead with an **At a glance** table block and order the rest differently from their neighbours — check the category page before you start, and use the template in that directory rather than a neighbouring one. You may add sections at the end; do not drop or reorder the ones the template gives you.

When you add an entry, also add a line for it to the **What we have learned** section of that category's page.

### Titles

Keep titles short and descriptive. Name the subject and what was done — not what was found.

- **The finding goes in the Summary and the At a glance block, never the title.** `"DSpark data ablation on Qwen3-4B: the two subset runs tie at matched compute"` is a conclusion. `"DSpark data ablation on Qwen3-4B"` is a title.
- Aim for roughly 60–80 characters before the handle. A Hugging Face identifier that a category requires can push past that; prose should not.
- A result in the title ages badly. It has to be rewritten whenever the numbers are revised, and it invites readers to quote the headline without the caveats attached to it.
- Name the knob, dataset or model that makes the entry findable. Drop qualifiers a reader does not need to tell this entry from its neighbours.

Every entry's title ends with the GitHub handle of the person who did the work, in parentheses — the person who **ran** the work, not whoever wrote it up:

```yaml
title: "DSpark data ablation on Qwen3-4B (@MeganEFlynn)"
```

Multiple authors go in credit order: `(@alice and @bob)`, or `(@alice, @bob and @carol)` for three or more. Use the handle's real casing, as the byline does — `@MeganEFlynn`, not `@meganeflynn`. Filenames stay lowercase; that difference is intentional. The `#` heading at the top of the body matches the title, handle included.

Three categories prescribe what the rest of the title says; the handle still trails it in all three:

- [Model evaluations](model-evaluations/index.md) — the verifier's Hugging Face identifier leads.
- [Finetuning experiments](finetuning-experiments/index.md) — the tuned model and the dataset it was tuned on lead.
- [Hyperparameter tuning](hyperparameter-tuning/index.md) — the verifier, the knob swept, and its range lead.

### Frontmatter and byline

```yaml
---
title: "Short description of what you tried (@USERNAME)"
tags:
  - DSpark
  - Qwen3
---
```

```markdown
<div class="entry-byline" markdown>
[![USERNAME](https://github.com/USERNAME.png?size=64){ .entry-avatar }](https://github.com/USERNAME)
**[@USERNAME](https://github.com/USERNAME)** posted on **YYYY-MM-DD**
</div>
```

Use block-style YAML lists for `tags`. An inline list (`tags: [DSpark, Qwen3]`) is parsed as a Markdown link by the documentation link checker and will fail CI. Choose algorithms and model families that readers would search for, such as Eagle3, DFlash, DSpark, MTP, or Qwen3.

The byline date is the date the **experiment** ran, not the date you wrote it up and not the pull request merge date. Results depend on software and model revisions, so the run date is the useful one.

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

## Writing a good entry

These conventions apply to every category.

### Do not publish a number you did not measure

This hub exists to record what was actually run. A plausible number is worse than no number, because it is indistinguishable from a real one once published.

- Every number in an entry should be traceable to a run you made or an artifact you can point at.
- Do not estimate, interpolate, or read a figure back off a chart.
- Unmeasured cells are `N/A`, with a sentence saying why. An empty cell is acceptable; an invented one is not.
- Do not state a hypothesis as a result. Mark reasoning as reasoning: "likely", "consistent with", "not tested".
- If a required field has no supporting data, write `N/A`. Never fill a cell just to complete a template. A correct report in the wrong format is a formatting problem, fixed in minutes; a well-formatted report with a wrong number is a corrupted record that other people build on, and it may never be caught.

### Be conservative with conclusions

Recording what happened is the job. Explaining *why* is a bonus. Conclusions published here feed into other people's model choices, training recipes and serving configs long after the run itself is forgotten, so claiming more than the data supports is a defect.

**Default to the weakest claim that fits the data.** If you are unsure which tier a claim belongs in, it belongs in the lower one.

| Tier                    | When                                                                                                                                                      | How to write it                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Measured**            | It is in the data                                                                                                                                         | State it plainly: "acceptance length rose from 5.41 to 5.65." |
| **Well supported**      | The data clearly implies it: there is a baseline, the effect exceeds the noise floor, no obvious confound                                                 | State it, with the supporting evidence in the same place      |
| **Partially supported** | Consistent with the data but not established: one run, no variance collected, a confound, a mechanism inferred from a profile, an effect inside the noise | **Mark it explicitly**, in the same sentence as the claim     |

An unlabelled claim reads as measured. Hedges belong next to the claim, not in a caveats section at the bottom that readers skip: "consistent with", "on a single run", "suggests, though we did not test", "not established".

Traps that come up repeatedly:

- **One run is not a trend.** Without run-to-run variance you cannot rank things that differ by a little. Say they are indistinguishable.
- **A difference smaller than the noise floor is a tie.** If the noise floor was not measured, say so, and do not rank.
- **A mechanism from a profile explains a share, not the whole.** Quantify the share. "GDN recurrence accounts for 47% of the target-time reduction" is a finding; "the speedup is explained by GDN recurrence" overstates it.
- **Arms on different hardware, GPUs or days are confounded.** Note it where the comparison is made, not only in limitations.
- **Do not generalize past what was tested** — other models, depths, concurrencies, precisions, batch sizes.
- **Do not stack weak signals into a strong claim.** Three suggestive results are three suggestive results.

Avoid "proves", "confirms", "demonstrates that", "clearly shows", "dramatically", and "significantly" unless significance was actually tested. Prefer "measured", "consistent with", "did not separate", "we did not test", "appears to".

A report whose honest conclusion is "this did not tell us much" is a good report. Say that instead of reaching.

### Acceptance length is the default metric

Unless an entry states otherwise, **acceptance length (AL)** is the convention for how much a drafter buys you:

```
AL = 1 + (accepted draft tokens / draft iterations)
```

the mean number of tokens emitted per verification step, **including the verifier's single always-emitted token**. A drafter served at depth K has a ceiling of K+1.

- Carry this one-line definition inline in your entry, where the number first appears.
- **Accepted is not verified.** The number of positions verified per step is usually larger than the number accepted, and under adaptive verification it varies step to step. Never compute acceptance length from verified positions, query length, or draft budget.
- The bonus token is included by default. If you exclude it, say so — it is a whole token of difference and silently shifts every comparison by 1.
- Always give the sampling settings next to an acceptance number. Acceptance moves with temperature, so a bare number is not interpretable.
- Do not compare acceptance length across different depths or block sizes without noting that the ceilings differ. AL 3.3 at depth 3 and AL 3.3 at depth 15 are very different results.

Report a target-only baseline for speedup. Acceptance metrics are undefined for the baseline without a drafter — write `N/A (by construction)` rather than leaving it ambiguous. Keep total throughput and per-user decode speed (TPOT/ITL) as separate numbers, never presenting one as the other, and define the steady-state measurement window: which requests are counted, and when the window opens and closes.

Acceptance is systematically higher under greedy decoding than under realistic sampling, so a greedy-only report reads better than what users will see in production. If you only measured greedy, say so in Observations and note that the numbers are an upper bound on sampled acceptance. Never present greedy numbers without the word "greedy" or "temperature 0" next to them.

### Pinning

Setup must be specific enough to re-run:

- Model and drafter: full id plus the revision SHA, and precision.
- Speculators: version and commit SHA.
- vLLM: version, and commit SHA where it matters.
- torch / CUDA / transformers versions.
- Hardware: GPU model, count, memory per GPU.

"Latest" is not a version. If a checkpoint was trained at a specific speculators commit, pin that commit, not the one you happen to have checked out.

### Provenance

| Stage        | Artifacts                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------ |
| Training     | `train_command.txt`, `speculators.patch`                                                   |
| Eval         | `eval_command.txt`                                                                         |
| vLLM serving | `vllm_command.txt`, `vllm.patch`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt` |

`scripts/launch_vllm.py` writes its artifacts **only** when `--provenance-dir` is passed. If a run was made without it, record that the artifacts are unavailable rather than omitting the question.

### Linking models

Link every model and drafter to its Hugging Face page on first mention in the body, using the full `org/model` identifier as the link text. For checkpoints not on the Hub, give the identifier you do have and say where it lives.

### Figures

Put figures in `docs/assets/research/` and reference them relatively. Give every figure alt text that states what it shows, not just its filename.

### Negative results

Failed and inconclusive runs belong here. Record what you tried, what broke, and what you did not test. Do not quietly drop a configuration that underperformed — the absence is itself misleading.

## Section summaries

Each category page carries a **What we have learned** section: a short, plain-language line or two per report, so a reader can scan the section and know what came out of it without opening anything.

When writing one:

- Lead with the result, not the method. One to three sentences.
- Plain language. A reader who does not work on speculative decoding should get the gist; a reader who does should get the number.
- Give the number and the condition it holds under. "+4.4% acceptance length on HumanEval" is useful; "improved acceptance" is not.
- Link the entry.
- **Never oversell.** If the entry hedges, the summary hedges at least as much. A summary must never be more confident than the report it summarizes. Where an entry says its differences are descriptive rather than statistically significant, carry that into the summary.
- **"We did not learn anything new" is a valid summary, and a useful one.** Ties, flat curves and null results get written as exactly that. A section whose honest summary is "three reports, no clear signal yet" is more use than one padded with invented significance.
- Do not aggregate across reports into a claim no single report makes.

**A new entry is not finished until its category's section has a line for it.** Revising an entry counts too, whenever the change moves its conclusions or numbers — a summary describing superseded results is worse than no summary. Moving an entry between categories means two changes: remove the line from the old section and add one to the new.

## See also

- [Evaluating Model Performance](../user_guide/tutorials/evaluating_performance.md) for evaluation commands.
- [Throughput vs. Interactivity](../user_guide/tutorials/throughput_interactivity.md) for comparisons across concurrency levels.
