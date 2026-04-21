# EAGLE-3

EAGLE-3 is a speculative decoding algorithm that uses a lightweight draft model to autoregressively predict multiple tokens ahead, which are then verified by the target model in a single forward pass.

## Overview

EAGLE-3 uses Llama-style transformer layers for the draft model, but can be paired with any supported verifier model -- the draft architecture is independent of the verifier.

**Key characteristics:**

- **Autoregressive drafting:** Predicts draft tokens one at a time, sequentially
- **Cross-tokenizer support:** Draft model can use a smaller vocabulary than the target
- **Lossless:** Output comes from the same distribution as the target model

## How It Works

### Architecture

EAGLE-3 consists of:

1. **Token embeddings:** Embedding of the current token
2. **Hidden states:** Internal representations from the target model at selected layers
3. **FC projection:** Combines embeddings with hidden states
4. **Llama-style decoder layers:** 1-4 transformer decoder layers (default: 1)
5. **LM head:** Projects to vocabulary logits

### Inference Process

1. Target model generates the first token autoregressively
2. EAGLE-3 drafts the next K tokens using hidden states from the target model
3. Target model verifies all K draft tokens in one forward pass
4. The longest correct prefix is accepted
5. Repeat from the last accepted token

### Training

EAGLE-3 is trained to minimize KL divergence between its draft logits and the target model's logits, learning to predict tokens the target model is likely to generate.

## Configuration

```python
Eagle3SpeculatorConfig(
    draft_vocab_size=32000,
    norm_before_residual=True,
    target_layer_ids=[2, 16, 29, 31],
    transformer_layer_config=...,
)
```

Key parameters:

- **`draft_vocab_size`** -- Typically 32K. Smaller vocabularies give faster inference; larger ones give better token coverage.
- **`target_layer_ids`** -- Which layers to extract hidden states from. Default selects early, middle, and late layers (e.g., `[2, N//2, N-3, N-1]` for an N-layer model).
- **`num_layers`** -- Number of draft decoder layers (default: 1). More layers improve quality at the cost of speed.

## Training

```bash
python scripts/train.py \
  --speculator-type eagle3 \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5
```

See the step-by-step tutorials for detailed instructions:

- [Train EAGLE-3 Online](../tutorials/train_eagle3_online.md)
- [Train EAGLE-3 Offline](../tutorials/train_eagle3_offline.md)

## Current Support

EAGLE-3 is the more established algorithm with mature support:

- **Speculators:** Fully supported
- **vLLM:** Fully optimized (Llama-style draft layers)
- **Verifier models:** Any supported architecture

## Research & Citation

EAGLE-3 is based on research from SafeAI Lab: [EAGLE Repository](https://github.com/SafeAILab/EAGLE)

```bibtex
@article{li2024eagle,
  title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty},
  author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2401.15077},
  year={2024}
}
```
