# Loss Functions

The training loss controls what the draft model is optimized for. Select it with `--loss-fn` in [`train.py`](../cli/train.md), which accepts either a single loss name or a JSON dict for a weighted combination. It applies to Eagle-3, P-EAGLE, DFlash, and DSpark. MTP uses its own multi-step cross-entropy loss and ignores this flag.

```bash
python scripts/train.py ... --loss-fn kl_div
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

`tv` and `nla` use a fused Triton kernel on CUDA/ROCm and fall back to eager PyTorch elsewhere.

## Choosing a Loss

- Start with `kl_div`. It is the default and the best-tested option.
- To optimize acceptance rate directly, use `lk_hybrid` or `nla`. Plain `tv` has the same objective but weak gradients early in training.
- Use `ce` when you want hard-label supervision or D-PACE position weighting.

## Weighted Combinations

Pass a JSON dict to train on a weighted sum of several losses. Each term is logged separately as `{name}_loss`.

```bash
python scripts/train.py ... --loss-fn '{"ce": 0.1, "tv": 0.9}'
```

## References

- Lin, "Divergence measures based on the Shannon entropy" (1991) -- Jensen-Shannon divergence
- Samarin et al., ["LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"](https://arxiv.org/abs/2602.23881) -- `nla` and `lk_hybrid`
