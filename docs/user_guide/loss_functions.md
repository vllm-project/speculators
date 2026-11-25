# Loss Functions

The training loss controls what the draft model is optimized for. Select it with `--loss-fn` in [`speculators train`](../cli/train.md), which accepts either a single loss name or a JSON dict for a weighted combination. It applies to Eagle-3, P-EAGLE, DFlash, and DSpark. MTP uses its own multi-step cross-entropy loss and ignores this flag.

The fused Triton implementation is the default. Use `--loss-implementation eager` for compatibility or numerical validation; eager losses may OOM with DFlash or DSpark. DFlash can also use `--loss-implementation fused_linear`, which never materializes the draft logits (see [Memory-Efficient DFlash Loss](#memory-efficient-dflash-loss)).

```bash
speculators train ... --loss-fn kl_div
```

## Available Losses

`p` is the target distribution, `q` the draft distribution, and `alpha = sum_v min(p_v, q_v)` the distributional overlap, which equals the acceptance rate of speculative decoding.

| Name        | Objective                              | Notes                                                                          |
| ----------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| `kl_div`    | Forward KL, target to draft            | Default. Mass-covering: penalizes the draft for missing target mass.           |
| `rkl`       | Reverse KL, draft to target            | Mode-seeking: the draft concentrates on the target's dominant modes.           |
| `jsd`       | Jensen-Shannon divergence              | Symmetric, bounded by `log 2`. Balances forward and reverse KL.                |
| `ce`        | Cross-entropy against `argmax p`       | Hard labels from the target. Required by `--per-position-loss-weight dpace`.   |
| `tv`        | Total variation, `1 - alpha`           | Optimizes the acceptance rate directly. Gradients vanish when overlap is low.  |
| `nla`       | Negative log-acceptance, `-log(alpha)` | TV's target with a `1 / alpha` gradient boost, so it trains from a cold start. |
| `lk_hybrid` | Adaptive KL/TV blend                   | `lambda * KL + (1 - lambda) * TV` with `lambda = exp(-3 * alpha)`, detached.   |

All losses use the selected implementation; there is no automatic fallback.

## Choosing a Loss

- Start with `kl_div`. It is the default and the best-tested option.
- To optimize acceptance rate directly, use `lk_hybrid` or `nla`. Plain `tv` has the same objective but weak gradients early in training.
- Use `ce` when you want hard-label supervision or D-PACE position weighting.

## Weighted Combinations

Pass a JSON dict to train on a weighted sum of several losses. Weights are used as provided and are not normalized. Each term is also logged separately as `{name}_loss`, before its weight is applied.

```bash
speculators train ... --loss-fn '{"ce": 0.1, "tv": 0.9}'
```

## Memory-Efficient DFlash Loss

`--loss-implementation fused_linear` computes the loss from DFlash's final hidden states and its frozen `lm_head` in row chunks, so no `[N, V]` tensor exists (`N` loss rows, `V` vocabulary): not the draft logits, not their gradient and not the target logits, which each chunk recomputes from the verifier hidden states. It supports every loss above with the same per-row semantics, including D-PACE and weighted combinations. With up to two differentiated loss terms (every single loss, `lk_hybrid`, or two weighted losses such as `{"ce": 0.1, "tv": 0.9}`) the gradient is contracted in the forward pass, so backward runs no GEMM; with more terms, backward projects each chunk again instead. At 16k rows and a 152k vocabulary (bf16, KL, MI350P) the loss head's peak memory drops from 13.9 GB to 2.2 GB at about the same forward + backward time as `fused`.

`--fused-linear-logits` (default `input`) keeps the logit tiles in the training dtype, like materialized logits. `float32` takes them straight from the GEMM accumulator: more accurate gradients, notably for TV-based losses, at 7-10% more time. `float16` casts the bf16 operands to fp16 for 8x finer tiles at bf16 tile size, for up to 4% more time and one fp16 copy of the head; operands and logits must stay within fp16's range (below 65504), else use `input` or `float32`.

With the full vocabulary, DFlash's `lm_head` is the verifier head, so with a distribution loss (KL, TV, ...) the draft logits are formed as the target logits plus `(h - h_verifier) @ W^T`. Their rounding then scales with the draft-target difference instead of the logits: at V = 152k and acceptance 0.88-0.98, the KL / TV gradient error vs fp64 drops from 16-87% to 2-9% with `input` tiles, and with `float16` tiles to about 0.4% for KL and for TV from acceptance 0.95 (TV: 8% at 0.88, where its `dp <= tp` indicator flips near ties); `float32` tiles reach 0.2-0.5%.

Chunks are at most 2048 rows (the fastest size for both GEMMs on MI355X), smaller when their scratch would exceed `SPECULATORS_FUSED_LINEAR_BUDGET_MIB` (default 4096): the budget can only lower the chunk size. Only DFlash supports `fused_linear`; other models raise an error.

## References

- Lin, "Divergence measures based on the Shannon entropy" (1991) -- Jensen-Shannon divergence
- Samarin et al., ["LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"](https://arxiv.org/abs/2602.23881) -- `nla` and `lk_hybrid`
