# Block sizes

Base-training sweeps over block size, draft depth, and max anchors.

Block size interacts with both training cost and serving behaviour: a larger block trains a drafter that can propose more tokens, but acceptance per position falls off. Report per-position acceptance alongside acceptance length, not just the aggregate.

Serving-time block size against an already-trained drafter belongs in [vLLM knobs](../../vllm-knobs/index.md). The question here is what to train with.

Start from the [hyperparameter sweep template](../TEMPLATE.md); the rules are on the [category page](../index.md).

## What we have learned

No entries yet.
