# Dataset ablations

Varying the training corpus — mixtures, subsets, curation, filtering, response regeneration — while holding the training recipe fixed.

If the learning rate, loss, or optimizer also changed, it is a [finetuning experiment](../finetuning-experiments/index.md), or it should be split into two entries.

Start from [`TEMPLATE.md`](TEMPLATE.md). The conventions that apply to every entry are in the [hub overview](../index.md).

## What we have learned

- [DSpark on Qwen3-4B, three corpus slices at matched compute](dspark-qwen3-4b-data-ablation-meganeflynn.md) — tulu3-only and an 8-file "perfect blend" came out at 4.610 vs 4.611 acceptance length (temperature 0), which the entry calls a tie within noise; both sit ~1.7% below the full 13-file mixture at 4.689. Run-to-run variance was not reported, so the gap to the full mix is not established as exceeding noise. Per-subset acceptance tracked the training mix while the aggregate did not move with subset choice at this scale. No clear win for curation here.

## Match compute, and say what you matched on

This is the rule this category exists to enforce. A mixture that looks better because it trained longer is not a better mixture.

- State explicitly whether you matched on optimizer steps or on tokens. They are not equivalent.
- Give the step count for every arm, not a nominal target. If a run was capped by `max-steps` mid-epoch, say so and give the resulting epoch count as a fraction.
- Confirm in the text that the recipe was identical across arms, and name the one or two things that necessarily differed (epoch count follows from corpus size at matched steps).

## Required in Setup

For each arm:

- The exact source files, by name, not a prose description of the mixture.
- Prepared row count and token count.
- Epochs, and steps.
- The shared recipe in full: optimizer and LRs, schedule and warmup, loss weighting, position decay, seed, sequence length, max anchors, world size / parallelism.
- Speculators commit SHA. Note it if arms were trained at different commits — that is a confound and must be stated, not buried.

## Required in Results

- Per-subset acceptance alongside the aggregate. The interesting signal in a data ablation is usually which subsets move, and the aggregate is often flat.
- A statement of what counts as noise at this scale. If two arms differ by less than that, write that they tie — do not rank them.
- The full-mixture or production baseline as a reference row.

Do not describe a sub-1% aggregate difference as an improvement without a reason to believe it exceeds run-to-run variance.
