# Training Metrics

`reference_prefix_acc_i` measures how often the first *i* draft tokens all match the stored continuation. For example, `[match, mismatch, match]` succeeds at position 1 and fails at positions 2 and 3.

A start is counted only when all *i* predictions are available and all *i* reference tokens are supervised within the start's non-padding document. Logs include matching (`_sum`) and eligible (`_total`) counts; validation keys add `_epoch`. A zero total means no observations. Checkpoint selection still uses validation loss.

## Example curves

These runs use Qwen3-8B on `tutorial_regen` for three epochs. Each panel shows the fraction of eligible starts with a matching prefix, smoothed over five logged steps. Higher is better. Labels show each curve’s final smoothed value.

![Training agreement for the first token, first two tokens, and first three tokens across five drafters.](../assets/training_metrics_prefix_agreement.png)

Sampling and training conditioning differ between drafters, so controlled comparisons need matched starts and settings. These curves measure agreement during training; serving acceptance and speed require separate evaluation.
