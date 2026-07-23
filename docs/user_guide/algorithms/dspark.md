# DSpark

DSpark extends [DFlash](dflash.md) with two heads on top of the same block-parallel draft backbone: a low-rank Markov head that biases each draft position by the token before it, and a confidence head that predicts per-position acceptance probability. Pure block-parallel drafting has no dependency between tokens inside a block, so acceptance decays toward the end of the block -- the Markov head restores that dependency, and the confidence signal indicates how far a block is worth verifying. The draft model subclasses DFlash, so the architecture and training pipeline are otherwise unchanged, and it can be paired with any supported verifier. Serving uses vLLM's own `dspark` method (`"method": "dspark"` in `--speculative-config`).

## How It Works

### Markov Head

The head adds a low-rank logit bias `B = W1 @ W2` to the DFlash logits: `W1` embeds the previous block token (verifier vocabulary) into `markov_rank` dimensions and `W2` projects to the draft vocabulary. Three variants are available:

- **`vanilla` (default)**: bias from the previous token alone
- **`gated`**: bias gated by the backbone hidden state
- **`rnn`**: recurrent state carried across positions within the block

Setting `--markov-rank 0` disables the head, leaving pure DFlash drafting. It must be paired with `--no-confidence-head-with-markov`, since that option requires a Markov head.

### Confidence Head

A linear head predicts each position's acceptance probability from the backbone hidden state, concatenated with the Markov previous-token embedding when `--confidence-head-with-markov` is set. It is trained with a BCE term weighted by `--confidence-head-alpha`.

### Sample From Anchor

DSpark defaults to `sample_from_anchor: True` -- the anchor and all mask positions predict future tokens, producing `block_size` speculative tokens. See [DFlash](dflash.md#sample-from-anchor) for details.

## Key Parameters

| Parameter                       | Default   | Description                                                       |
| ------------------------------- | --------- | ----------------------------------------------------------------- |
| `--markov-rank`                 | 256       | Low-rank dimension of the Markov logit bias (0 disables the head) |
| `--markov-head-type`            | `vanilla` | Sequential head variant: `vanilla`, `gated`, or `rnn`             |
| `--enable-confidence-head`      | enabled   | Attach the per-position acceptance head                           |
| `--confidence-head-with-markov` | enabled   | Feed the Markov previous-token embedding into the confidence head |
| `--confidence-head-alpha`       | 1.0       | Weight of the confidence-head BCE term                            |

All DFlash parameters (`--block-size`, `--max-anchors`, `--num-layers`, ...) apply unchanged.

## Pretrained Models

Pretrained DSpark speculator models are available on HuggingFace from the [RedHatAI speculator models collection](https://huggingface.co/collections/RedHatAI/speculator-models):

| Verifier                | Speculator                                                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `zai-org/GLM-5.2-FP8`   | [`RedHatAI/GLM-5.2-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark)               |
| `google/gemma-4-31B-it` | [`RedHatAI/gemma-4-31B-it-speculator.dspark`](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dspark) |

To train your own, see `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`.

## Research & Citation

DSpark is based on research from DeepSeek: [arXiv Paper](https://arxiv.org/abs/2607.05147)

```bibtex
@article{cheng2026dspark,
  title={DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation},
  author={Cheng, Xin and Yu, Xingkai and Shao, Chenze and Li, Jiashi and Xiong, Yunfan and Qian, Yi and Zhu, Jiaqi and Ma, Shirong and Zhang, Xiaokang and Ye, Jiasheng and others},
  journal={arXiv preprint arXiv:2607.05147},
  year={2026}
}
```

## See Also

- [DFlash](dflash.md) -- The base algorithm DSpark extends
- [Train DFlash Tutorial](../tutorials/train_dflash_online.md) -- Step-by-step training guide; the DSpark pipeline is the same plus the flags above
