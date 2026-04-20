# DFlash

DFlash is a block-based speculative decoding algorithm that predicts all draft tokens in a single forward pass, rather than autoregressively like EAGLE-3.

## Overview

DFlash uses anchor-based speculation to predict multiple tokens ahead in blocks. It uses Qwen3-style transformer layers for the draft model, but can be paired with any supported verifier model -- the draft architecture is independent of the verifier.

**Key characteristics:**

- **Single forward pass:** All draft tokens are generated at once, rather than one at a time
- **Block-based prediction:** Tokens are predicted in blocks from anchor points
- **Anchor point mechanism:** Positions in the sequence are selected as anchors, and a block of tokens is predicted from each
- **Lossless:** Output comes from the same distribution as the target model

## How It Works

### Architecture

DFlash consists of:

1. **Token embeddings:** Embedding of the current token
2. **Hidden states:** Internal representations from the target model at specific layers
3. **Qwen3-style decoder layers:** Process combined embeddings and hidden states
4. **LM head:** Projects to vocabulary logits

### Block-Based Prediction

Unlike EAGLE-3's autoregressive drafting, DFlash predicts in blocks:

**EAGLE-3 (autoregressive):**

```
Token 1 → Token 2 → Token 3 → Token 4 → Token 5
```

**DFlash (block-based):**

```
Anchor 1 → [Block: Tokens 1-8]
Anchor 2 → [Block: Tokens 9-16]
```

### Anchor Point Mechanism

1. **Select anchors:** Choose positions in the sequence
2. **Predict from anchors:** Generate a block of tokens from each anchor in a single forward pass
3. **Verify blocks:** Target model verifies the predicted blocks
4. **Accept valid tokens:** Use the longest valid prefix

## Configuration

```python
DFlashSpeculatorConfig(
    draft_vocab_size=64000,
    block_size=8,
    max_anchors=256,
    target_layer_ids=[...],
    transformer_layer_config=...,
)
```

Key parameters:

- **`block_size`** -- Tokens predicted per anchor (default: 8). Smaller blocks give finer control; larger blocks reduce overhead if predictions align well.
- **`max_anchors`** -- Maximum anchor positions during training (default: 256).
- **`draft_vocab_size`** -- Typically 64K for DFlash.

## Training

```bash
python scripts/train.py \
  --speculator-type dflash \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path ./training_data \
  --draft-vocab-size 64000 \
  --block-size 8 \
  --max-anchors 256 \
  --epochs 10 \
  --lr 3e-5
```

The training data pipeline is the same as EAGLE-3 -- see [Train DFlash Online](../tutorials/train_dflash_online.md) for a step-by-step guide.

## Current Support

DFlash is newer than EAGLE-3 and support is improving rapidly:

- **Speculators:** Training fully functional
- **vLLM:** Supported (Qwen3-style draft layers)
- **Verifier models:** Any supported architecture

## Research & Citation

DFlash is based on research from Z Lab: [DFlash Project Page](https://z-lab.ai/projects/dflash/)

```bibtex
@article{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal={arXiv preprint arXiv:2602.06036},
  year={2026}
}
```

## See Also

- [Train DFlash Tutorial](../tutorials/train_dflash_online.md) -- Step-by-step training guide
