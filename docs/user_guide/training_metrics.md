# Training Metrics

`reference_acc_at_pos_i` measures how often all draft tokens through position *i* match the stored continuation. Positions start at 0: position 0 covers the first token, position 1 covers the first two. For example, `[match, mismatch, match]` succeeds at position 0 and fails at positions 1 and 2.

A start is counted only when every prediction through position *i* is available and all corresponding reference tokens are supervised within the start's non-padding document. Logs include matching (`_sum`) and eligible (`_total`) counts; validation keys add `_epoch`. A zero total means no observations. Checkpoint selection still uses validation loss.

## Example curves

These runs use Qwen3-8B on `tutorial_regen` for three epochs. Each panel shows the fraction of eligible starts with a matching prefix, smoothed over five logged steps. Higher is better. Labels show each curve’s final smoothed value.

![Training agreement for the first token, first two tokens, and first three tokens across five drafters.](../assets/training_metrics_prefix_agreement.png)

Sampling and training conditioning differ between drafters, so controlled comparisons need matched starts and settings. These curves measure agreement during training; serving acceptance and speed require separate evaluation.

## Agreement with serving

We evaluated the same five checkpoints with vLLM on `RedHatAI/speculator_benchmarks`. The validation columns below report reference-prefix agreement on `tutorial_regen`.

`acceptance_at_pos_i` and `reference_acc_at_pos_i` both measure prefixes through zero-based position *i*. Serving counts accepted prefixes over all drafts; validation counts reference matches over eligible starts.

| drafter | validation 0 | serve 0 | validation 1 | serve 1 | validation 2 | serve 2 |
| ------- | ------------ | ------- | ------------ | ------- | ------------ | ------- |
| eagle3  | 0.558        | 0.466   | 0.299        | 0.203   | 0.161        | 0.085   |
| peagle  | 0.639        | 0.565   | 0.237        | 0.165   | 0.073        | 0.030   |
| dflash  | 0.605        | 0.533   | 0.310        | 0.225   | 0.153        | 0.081   |
| dflash2 | 0.579        | 0.482   | 0.296        | 0.195   | 0.148        | 0.072   |
| dspark  | 0.617        | 0.536   | 0.335        | 0.245   | 0.177        | 0.102   |

Across these five checkpoints, validation prefix agreement and serving acceptance produce the same ranking at all three positions. This supports using the metric as an ordering signal in this experiment. Absolute rates differ across the datasets and evaluation settings. These serving runs do not measure training overhead.
