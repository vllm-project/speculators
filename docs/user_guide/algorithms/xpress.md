# XPress

XPress extends [DFlash](dflash.md) with a causal refiner over the same block-parallel draft backbone. Pure block-parallel drafting decodes every position from its own marginal, so token *k* is chosen without seeing what the drafter picked at *k-1*; the verifier scores conditionally, and blocks that are individually plausible but jointly improbable are rejected early. The refiner reads each position's hidden state, a block-global summary and the previous token, mixes across the block through a lower-triangular (causal) mixer, and emits a bias on the base logits. Because the mixer is causal rather than a local smoother, the block is resolved by Jacobi iteration instead of a left-to-right loop, so the serial depth at inference is a constant K passes rather than the block size. The draft model subclasses DFlash, so the backbone, data pipeline and verifier support are unchanged.

## How It Works

### Causal Refiner Head

The head projects the per-position hidden state, a block-mean summary and an embedding of the previous block token into a rank-`r` bottleneck, mixes across block positions with a per-channel lower-triangular matrix, applies a SwiGLU MLP, and reads out to the draft vocabulary as a logit bias. Everything except the two vocabulary projections lives in the bottleneck, so the head is small relative to the backbone.

### Jacobi Training

Training runs the refiner for several free-running rounds, each conditioned on the previous round's argmax with stop-grad between rounds. This is the exact recurrence inference performs, so the head is trained on the input distribution it will actually see rather than only on teacher-forced tokens. A base-logits term anchors the co-trained backbone so its standalone drafting quality does not erode.

## Key Parameters

| Parameter              | Default | Description                                                                        |
| ---------------------- | ------- | ---------------------------------------------------------------------------------- |
| `--xpress-rank`        | 256     | Low-rank dimension `r` of the refiner bottleneck                                   |
| `--xpress-mlp-ratio`   | 2       | Refiner MLP expansion ratio (hidden = `r * ratio`)                                 |
| `--num-jacobi-passes`  | 6       | Inference-time refine passes K, stored in the exported config                      |
| `--consistency-passes` | 3       | Free-running Jacobi rounds used during training                                    |
| `--consistency-weight` | 0.3     | Weight of the free-running consistency term                                        |
| `--base-anchor-weight` | 0.6     | Weight of the base-logits term that anchors the co-trained backbone                |
| `--decayed-loss-norm`  | off     | Normalize position-decayed losses by the decayed weight sum (a true weighted mean) |

All DFlash parameters (`--block-size`, `--max-anchors`, `--num-layers`, ...) apply unchanged.

## Pretrained Models

To train your own, see `examples/train/xpress_qwen3_8b_regen_online.sh`.

## Research & Citation

XPress is based on research from UIUC: [arXiv Paper](https://arxiv.org/abs/2608.02438)

## See Also

- [DFlash](dflash.md) -- The base algorithm XPress extends
- [DSpark](dspark.md) -- A different head over the same backbone, conditioning on the previous token only
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide
