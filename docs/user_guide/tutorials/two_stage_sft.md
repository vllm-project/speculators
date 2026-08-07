# Two-Stage SFT

Two-stage SFT continues a trained speculator on a second dataset. It uses the existing training objective and pipeline; the only material differences are the training data and the checkpoint used to initialize the speculator.

This is a useful baseline when testing whether a more targeted data distribution improves the draft before introducing a new loss or architecture.

## Workflow

Start with a completed first-stage run, then:

1. Regenerate responses for a new prompt set with the same target model.
2. Prepare the regenerated JSONL with `prepare_data.py`.
3. Train with `--from-pretrained` pointing to the first-stage checkpoint and a new `--save-path` for the second-stage checkpoints.
4. Compare both checkpoints on the same held-out evaluation set.

This is the same pipeline described in [Train a Speculator](train.md). The [DFlash two-stage example](https://github.com/vllm-project/speculators/blob/main/examples/train/dflash_qwen3_8b_two_stage_sft.sh) contains the complete runnable commands, including local prompt ingestion with `--dataset-file`.

## What stays fixed

For a controlled baseline, keep the target model, hidden-state layer IDs, loss, and generation and optimization settings fixed. If stage one used a reduced vocabulary, reuse its `d2t.npy` and `t2d.npy` mappings rather than deriving new mappings from the second-stage data.

`--from-pretrained` restores the draft weights and architecture. A new `--save-path` gives stage two a fresh optimizer and scheduler while preserving the stage-one checkpoint for comparison.

## What to expect

Stage two should primarily affect prompts similar to its new training distribution. Improvements there may trade off against general performance due to specialization or forgetting, so evaluate both in-domain and general held-out prompts. A gain over the untouched stage-one checkpoint is the baseline that a more specialized post-training method should beat.

See [Response Regeneration](response_regeneration.md) for dataset formats and generation details.
