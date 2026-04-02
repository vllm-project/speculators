# EAGLE-3

EAGLE-3 is the primary speculative decoding algorithm in Speculators. It uses a lightweight decoder with test-time training (TTT) steps to autoregressively generate multiple draft tokens, achieving high acceptance rates with minimal overhead.

## Overview

EAGLE-3 builds on the EAGLE family of speculative decoding methods. It uses a compact transformer decoder that predicts the next token from concatenated hidden states extracted from the verifier model. During inference, it performs multiple test-time training (TTT) steps to generate a sequence of draft tokens, which are then verified in a single forward pass of the full verifier model.

**Key features:**

- **High acceptance rates**: Achieves strong token acceptance across multiple prediction steps
- **Lightweight architecture**: Single decoder layer with a linear projection, keeping parameter count low
- **Flexible vocabulary mapping**: Supports full or reduced vocabulary via `d2t`/`t2d` mappings
- **Broad verifier support**: Compatible with Llama, Qwen3, MoE models, and Vision-Language models

## Architecture

### Model Components

The EAGLE-3 draft model consists of:

1. **Linear projection** (`fc`): Projects concatenated hidden states $[h_1, h_2, h_3]$ of shape $3 \times \text{hidden\_size}$ to $\text{hidden\_size}$
2. **Transformer decoder layer(s)**: One or more decoder layers matching the verifier architecture
3. **Rotary embeddings**: Position encoding using RoPE with $2\times$ hidden size
4. **LM head**: Draft-vocabulary-sized output projection for token prediction
5. **Verifier components**: Frozen copies of verifier embedding, LM head, and final norm (used for target computation during training)

### Forward Pass

The forward pass implements autoregressive TTT steps:

```
For each TTT step k (0 to ttt_steps-1):
    1. Embed the input token IDs using verifier embeddings
    2. Concatenate embeddings with hidden states → shape [1, seq_len, 2 * hidden_size]
    3. Apply rotary position embeddings
    4. Pass through decoder layer(s) with cached KV states
    5. Apply normalization and LM head to produce logits
    6. Compute loss against verifier targets (KL divergence)
    7. Select predicted tokens via argmax for next TTT step
    8. Map draft token IDs back to verifier vocabulary space via d2t
```

### Vocabulary Mapping

EAGLE-3 supports optional vocabulary reduction:

- **`t2d` (target-to-draft)**: Boolean mask of shape `[verifier_vocab_size]` indicating which verifier tokens are included in the draft vocabulary
- **`d2t` (draft-to-target)**: Offset tensor of shape `[draft_vocab_size]` mapping draft token indices back to verifier token indices

When vocabulary mapping is not provided, the full verifier vocabulary is used.

## Configuration

EAGLE-3 models use the `Eagle3SpeculatorConfig` configuration class:

```python
from speculators.models.eagle3 import Eagle3SpeculatorConfig

config = Eagle3SpeculatorConfig(
    transformer_layer_config=verifier_config,
    draft_vocab_size=32000,
    norm_before_residual=True,
    embed_requires_grad=False,
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transformer_layer_config` | `PretrainedConfig` | required | Decoder layer configuration (derived from verifier) |
| `draft_vocab_size` | `int` | required | Size of the draft vocabulary |
| `norm_before_residual` | `bool` | `True` | Apply normalization before residual connection |
| `embed_requires_grad` | `bool` | `False` | Whether to train the embedding layer |

## Training

### Data Generation

Generate training data using the verifier model:

```bash
python scripts/data_generation_offline.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --train-data-path sharegpt \
    --output-dir ./training_data \
    --max-samples 5000
```

### Vocabulary Mapping

Build vocabulary mappings from the generated data:

```bash
python scripts/build_vocab_mapping.py \
    --data-path ./training_data \
    --output-dir ./training_data
```

### Training Command

```bash
torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py \
    --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./training_data \
    --save-path ./checkpoints/eagle3 \
    --epochs 10 \
    --lr 1e-4 \
    --d2t-path ./training_data/d2t.npy \
    --t2d-path ./training_data/t2d.npy \
    --ttt-steps 3
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ttt-steps` | `3` | Number of test-time training steps |
| `--ttt-step-loss-decay` | `1.0` | Loss decay factor per TTT step |
| `--num-layers` | `1` | Number of decoder layers |
| `--norm-before-residual` | `True` | Toggle normalization before residual |
| `--embed-requires-grad` | `False` | Train embedding layer weights |
| `--use-off-policy-tokens` | `False` | Use off-policy tokens (for regenerated data) |

### Loss Function

EAGLE-3 uses KL divergence loss between draft and verifier logits:

$$
\mathcal{L} = \sum_{k=0}^{K-1} \gamma^k \cdot D_{KL}(\text{softmax}(\hat{y}_k) \| \text{softmax}(y_k))
$$

Where:

- $K$ is the number of TTT steps
- $\gamma$ is the step loss decay factor
- $\hat{y}_k$ are the draft logits
- $y_k$ are the verifier target logits

## Inference

### vLLM Deployment

Trained EAGLE-3 models can be served directly with vLLM:

```bash
vllm serve RedHatAI/Qwen3-8B-speculator.eagle3
```

### Pre-trained Models

Pre-trained EAGLE-3 models are available on HuggingFace:

| Model | Verifier | Link |
|-------|----------|------|
| Llama 3.1 8B | meta-llama/Llama-3.1-8B-Instruct | [RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3) |
| Llama 3.3 70B | meta-llama/Llama-3.3-70B-Instruct | [RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3](https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3) |
| Qwen3 8B | Qwen/Qwen3-8B | [RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3) |
| Qwen3 14B | Qwen/Qwen3-14B | [RedHatAI/Qwen3-14B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-14B-speculator.eagle3) |
| Qwen3 32B | Qwen/Qwen3-32B | [RedHatAI/Qwen3-32B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3) |

## References

- **EAGLE Paper**: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- **EAGLE-2 Paper**: [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
