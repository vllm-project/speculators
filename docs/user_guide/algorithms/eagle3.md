# EAGLE-3

EAGLE-3 is the primary speculative decoding algorithm supported by Speculators. It provides excellent performance across a wide range of language models including dense, MoE, and vision-language models.

## Overview

EAGLE-3 (Extrapolation Algorithm for Greater Language-model Efficiency, version 3) is a speculative decoding algorithm that uses a small, fast draft model to predict multiple tokens ahead, which are then verified by the larger target model in a single forward pass.

**Key Benefits:**

- **Lossless acceleration:** Output quality identical to the target model
- **2-3x speedup:** Typical speedup on inference workloads
- **Broad compatibility:** Works with Llama, Qwen3, gpt-oss, MoE, and VLM models
- **Cross-tokenizer support:** Draft model can use a different (smaller) vocabulary than the target
- **Production-ready:** Fully integrated with vLLM for production deployment

## How It Works

### Architecture

EAGLE-3 consists of a lightweight neural network that learns to predict the next token using:

1. **Token embeddings:** Embedding of the current token
2. **Hidden states:** Internal representations from the target model at specific layers
3. **Draft decoder layers:** 1-4 transformer decoder layers
4. **LM head:** Final projection to vocabulary logits

```
┌─────────────────────────────────────────────────┐
│              Target Model (frozen)              │
│                                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐       ┌─────┐      │
│  │Layer│→ │Layer│→ │Layer│  ...  │Layer│      │
│  │  0  │  │  2  │  │ 16  │       │ 31  │      │
│  └──┬──┘  └──┬──┘  └──┬──┘       └──┬──┘      │
└─────┼────────┼────────┼─────────────┼──────────┘
      │        │        │             │
      │   Hidden States │             │
      │        │        │             │
┌─────▼────────▼────────▼─────────────▼──────────┐
│             EAGLE-3 Draft Model                 │
│                                                 │
│  ┌──────────┐                                  │
│  │Embeddings│◄── Current Token                 │
│  └─────┬────┘                                  │
│        │                                        │
│  ┌─────▼─────────────────┐                    │
│  │FC (combine with HS)   │                    │
│  └─────┬─────────────────┘                    │
│        │                                        │
│  ┌─────▼──────┐  ┌──────────┐  ┌──────────┐  │
│  │Decoder     │→ │Decoder   │→ │Decoder   │  │
│  │Layer 0     │  │Layer 1   │  │Layer 2   │  │
│  └─────┬──────┘  └──────┬───┘  └─────┬────┘  │
│        └─────────────────┴────────────┘        │
│                         │                       │
│                   ┌─────▼──────┐               │
│                   │  LM Head   │               │
│                   └─────┬──────┘               │
│                         │                       │
│                    Draft Logits                │
└─────────────────────────────────────────────────┘
```

### Inference Process

1. **Initial prediction:** Target model generates first token autoregressively
2. **Draft speculation:** EAGLE-3 predicts next K tokens (e.g., K=5) using hidden states
3. **Verification:** Target model verifies all K draft tokens in one forward pass
4. **Acceptance:** Accept longest prefix of correct predictions
5. **Repeat:** Continue from last accepted token

This reduces multiple sequential target model calls to fewer parallel verifications.

### Training Process

EAGLE-3 is trained to minimize KL divergence between its draft logits and the target model's logits:

```python
loss = KL_divergence(draft_logits, target_logits)
```

The model learns to:
- Predict tokens that the target model is likely to generate
- Use hidden states effectively to capture target model's knowledge
- Balance precision (correct predictions) with recall (bold predictions)

## Architecture Details

### Input Processing

EAGLE-3 processes inputs by:

1. **Embedding current token** using target model's embedding layer (frozen)
2. **Extracting hidden states** from target model at 4 key layers (default: [2, N/2, N-3, N])
3. **Concatenating embeddings with hidden states** to form combined input
4. **FC projection** from 2×hidden_size to hidden_size

### Decoder Layers

The draft model uses 1-4 transformer decoder layers (default: 1):

- **Self-attention:** Causal attention over draft tokens
- **Feed-forward network:** Standard transformer FFN
- **Layer normalization:** RMSNorm for stability
- **Residual connections:** With optional normalization before residual

### Vocabulary Mapping

EAGLE-3 supports cross-tokenizer scenarios where draft and target have different vocabularies:

**t2d (target-to-draft) mapping:**
- Boolean mask indicating which target tokens exist in draft vocabulary
- Shape: `[target_vocab_size]`
- Used during inference to check if draft token is valid

**d2t (draft-to-target) mapping:**
- Index mapping from draft tokens to target tokens
- Shape: `[draft_vocab_size]`
- Maps draft model outputs to target vocabulary

This enables using a smaller draft vocabulary (e.g., 32K) compared to target (e.g., 128K) for faster inference.

### Hidden State Selection

Hidden states are extracted from 4 layers of the target model:

**Default selection for N-layer model:**
```python
[2, N//2, N-3, N-1]
```

**Example for 32-layer model:**
```python
[2, 16, 29, 31]  # Early, middle, late, final layers
```

This provides a good distribution of information from different model depths.

## Configuration

### Model Config

Key EAGLE-3 configuration parameters:

```python
Eagle3SpeculatorConfig(
    draft_vocab_size=32000,            # Draft vocabulary size
    norm_before_residual=True,         # Normalize before residual
    norm_before_fc=False,              # Normalize before FC (for gpt-oss)
    target_layer_ids=[2, 16, 29, 31],  # Hidden state layers
    transformer_layer_config=...,      # Draft decoder config
)
```

### Proposal Config

EAGLE-3 uses greedy token proposal by default:

```python
GreedyTokenProposalConfig(
    speculative_tokens=5,      # Number of draft tokens
    verifier_accept_k=1,       # Top-k acceptance tolerance
    accept_tolerance=0.0,      # Log-likelihood tolerance
)
```

## Training EAGLE-3

### Data Requirements

- **Conversational dataset:** ShareGPT, UltraChat, or custom data
- **Dataset size:** Minimum 5K samples for basic training, 50K+ for production
- **Preprocessing:** Tokenized with chat templates applied

### Training Configuration

Typical hyperparameters:

```bash
python scripts/train.py \
  --speculator-type eagle3 \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --num-layers 1 \                # Start with 1 layer
  --draft-vocab-size 32000 \      # 32K vocab for 128K target
  --lr 3e-5 \                     # Learning rate
  --epochs 10 \                   # Typical: 10-20 epochs
  --total-seq-len 8192 \          # Match data preprocessing
  --norm-before-residual \        # Recommended for most models
  --scheduler-type linear         # Linear LR decay
```

### Training Time

Example timing on 2x H100 GPUs (5K samples):

- **Data generation:** ~14 minutes
- **Vocab mapping:** ~6 seconds
- **Training (10 epochs):** ~21 minutes
- **Total:** ~35 minutes

Scaling to larger datasets:
- 50K samples: ~3-4 hours
- 500K samples: ~30-40 hours

### Model-Specific Settings

**Llama models:**
```bash
--norm-before-residual \
--norm-before-fc false
```

**gpt-oss models:**
```bash
--norm-before-residual \
--norm-before-fc  # Enable for gpt-oss
```

**Qwen3 models:**
```bash
--norm-before-residual \
--draft-vocab-size 32000
```

## Performance

### Acceptance Metrics

Typical performance on MT-Bench (5K training samples):

```
First token accuracy:  0.40
Second token accuracy: 0.13
Third token accuracy:  0.04
Average acceptance:    1.57 tokens
```

With more training data (50K+ samples):

```
First token accuracy:  0.60-0.70
Second token accuracy: 0.30-0.40
Third token accuracy:  0.15-0.25
Average acceptance:    2.0-2.5 tokens
```

### Speedup

Typical end-to-end speedup on vLLM:

- **Small models (8B):** 1.8-2.2x
- **Medium models (70B):** 2.0-2.5x
- **Large models (400B+):** 2.2-3.0x

Speedup varies based on:
- Batch size (lower batch = higher speedup)
- Prompt length (longer prompts = higher speedup)
- Hardware (faster GPUs benefit more)

## Advantages & Limitations

### Advantages

✅ **Broad model support** - Works with Llama, Qwen3, gpt-oss, MoE, VLM
✅ **Cross-tokenizer** - Draft can use smaller vocabulary
✅ **Flexible architecture** - Configurable layers and hidden states
✅ **Production-ready** - Full vLLM integration
✅ **Lossless** - No quality degradation
✅ **Easy to train** - Converges in 10-20 epochs

### Limitations

⚠️ **Training data dependency** - Needs quality conversational data
⚠️ **Layer compatibility** - Must use same layers for generation and training
⚠️ **Memory overhead** - Requires loading draft model alongside target
⚠️ **Sequential constraints** - Speedup limited by acceptance rate
⚠️ **Batch size impact** - Lower speedup at large batch sizes

## Best Practices

1. **Start with 1 layer** - Single layer often provides best speed/quality trade-off
2. **Use 32K draft vocab** - Good balance for most target models
3. **Train on diverse data** - Mix datasets (ShareGPT + UltraChat) for robustness
4. **Match target layers** - Use default layer selection unless you have specific requirements
5. **Enable noise augmentation** - Use default `--noise-std 0.05` for better generalization
6. **Validate on target workload** - Test on representative inference tasks

## Advanced Topics

### Multi-Layer Draft Models

Using 2-4 draft layers can improve quality at the cost of speed:

```bash
--num-layers 2  # Better quality, slightly slower
```

Trade-off:
- 1 layer: Fastest, good quality
- 2 layers: Better quality, ~10-15% slower
- 3-4 layers: Best quality, ~20-30% slower

### Custom Layer Selection

Manually specify hidden state layers:

```bash
--target-layer-ids 5 15 25 30  # Custom layer IDs
```

Use when:
- Default layers don't work well
- Experimenting with layer importance
- Matching external research configurations

### Vocabulary Size Tuning

Experiment with draft vocabulary size:

```bash
--draft-vocab-size 16000  # Smaller vocab, faster inference
--draft-vocab-size 64000  # Larger vocab, better coverage
```

Guidelines:
- 8K-16K: Very fast, may miss rare tokens
- 32K: Recommended default
- 64K+: Better coverage, slower inference

## See Also

- [Train EAGLE-3 Online Tutorial](../tutorials/train_eagle3_online.md) - Step-by-step training guide
- [Train EAGLE-3 Offline Tutorial](../tutorials/train_eagle3_offline.md) - Offline training workflow
- [Convert EAGLE-3 Tutorial](../tutorials/convert_eagle3.md) - Convert existing models
- [Algorithm Decision Guide](decision_guide.md) - Choosing between algorithms
- [EAGLE Research Paper](https://arxiv.org/abs/2401.15077) - Original research

## Research & Citation

EAGLE-3 is based on research from SafeAI Lab:

**EAGLE Repository:** https://github.com/SafeAILab/EAGLE

**Citation:**
```bibtex
@article{li2024eagle,
  title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty},
  author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2401.15077},
  year={2024}
}
```

Speculators extends EAGLE-3 with:
- Production-ready vLLM integration
- Simplified training pipeline
- Vocabulary mapping support
- Multi-model compatibility
- Distributed training
