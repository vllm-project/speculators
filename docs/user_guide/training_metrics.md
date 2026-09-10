# Training Metrics

`reference_acc_at_pos_i` measures how often all draft tokens through position *i* match the stored continuation. Positions start at 0: position 0 covers the first token, position 1 covers the first two. For example, `[match, mismatch, match]` succeeds at position 0 and fails at positions 1 and 2.

Training metrics pool all batches since the previous log across ranks. Any remaining batches are logged at epoch end or the step limit. Validation metrics are pooled across all validation batches and ranks.

A start is counted only when every prediction through position *i* is available and all corresponding reference tokens are supervised within the start's non-padding document. Logs include matching (`_sum`) and eligible (`_total`) counts; validation keys add `_epoch`. A zero total means no observations. Checkpoint selection still uses validation loss.

The shared metrics are reported alongside existing drafter-specific metrics. Those diagnostics retain their current definitions; reference agreement is not a numerical replacement for them.

Comparing drafters requires matched settings. Serving acceptance and speed need separate evaluation.
