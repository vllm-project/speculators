# Loss functions

Base-training sweeps over loss composition and weighting (CE/TV mixes), loss normalization, per-position loss weighting, and position decay.

State the exact loss expression, whether it was fused, and how per-position loss was normalized — these are easy to describe loosely and impossible to reproduce from a loose description.

Where the change targets position weighting, report per-position acceptance rate, not only acceptance length. The aggregate can be flat while the position profile moves substantially.

Start from the [hyperparameter sweep template](../TEMPLATE.md); the rules are on the [category page](../index.md).

## What we have learned

- [DFlash2 on Qwen3.8-27B: per-position loss weight, selector miss policy, loss normalization](dflash2-selector-and-loss-normalization-ablation-shubhra.md) — no run-to-run variance was collected, so all three results are descriptive rather than statistically separated. `dpace` gave a higher mean than `ce+tv, fixed-exp-decay` for DFlash2 at 5k and 25k steps, and a lower one for DSpark at both. `strict` beat `replace` on every subset at 5k steps — the clearest of the three signals — narrowing to mixed by 25k; it landed as a training option in Speculators PR #1217. The two loss normalizations differed by 0.01 at 20k steps, a tie.
