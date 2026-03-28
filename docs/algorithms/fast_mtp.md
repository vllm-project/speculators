# FastMTP (Multi-Token Prediction)

FastMTP is a speculative decoding algorithm that extends the multi-token prediction (MTP) approach for accelerated LLM inference. It uses a series of dedicated MTP layers that share verifier hidden states and predict multiple tokens ahead in parallel.

## Overview

FastMTP is based on the Multi-Token Prediction approach from DeepSeek-V3 / R1, adapted for speculative decoding. Unlike EAGLE-3's sequential test-time training (TTT) steps, FastMTP uses independent prediction layers — one per speculative step — that each predict the next token from shared hidden state representations.

**Key advantages:**

- **Simpler architecture**: Each speculative step uses an independent MTP layer, avoiding sequential dependencies during inference
- **Efficient training**: Weighted cross-entropy loss across speculative steps with configurable step weights
- **Flexible depth**: Configurable number of speculative steps (typically 3)

**Paper**: [FastMTP: Harnessing Fast Multi-Token Prediction for Speculative Decoding (arXiv:2509.18362)](https://arxiv.org/abs/2509.18362)

## Architecture

### MTP Layer Structure

Each MTP layer in FastMTP consists of:

1. **Input Projection** (`input_proj`): A linear layer that projects the concatenated hidden state into the model's hidden dimension
2. **Self-Attention**: A transformer self-attention mechanism (based on verifier architecture)
3. **MLP**: A feed-forward network matching the verifier's MLP architecture
4. **Layer Normalization**: Five RMSNorm layers for input, post-attention, post-MLP, and shared hidden state processing

### Forward Pass

The FastMTP forward pass processes multiple speculative steps in a single pass:

```
For each speculative step k (0 to num_speculative_steps-1):
    1. Concatenate the shared hidden state with token embeddings
    2. Project through input_proj to model dimension
    3. Apply self-attention with causal mask
    4. Apply MLP feed-forward
    5. Predict logits via LM head
    6. Compute weighted cross-entropy loss against target tokens
```

### Key Differences from EAGLE-3

| Feature | EAGLE-3 | FastMTP |
|---------|---------|---------|
| Prediction approach | Sequential TTT steps | Independent MTP layers |
| Hidden state usage | Concatenation of 3 hidden states | Shared verifier hidden state |
| Loss function | KL divergence | Weighted cross-entropy |
| Token generation | Autoregressive within TTT | Independent per layer |

## Configuration

FastMTP models use the `FastMTPConfig` configuration class:

```python
from speculators.models.fast_mtp import FastMTPConfig

config = FastMTPConfig(
    transformer_config=verifier_transformer_config,
    num_speculative_steps=3,           # Number of MTP prediction layers
    mtp_loss_step_weights=[0.51, 0.31, 0.18],  # Per-step loss weights
    hidden_size=4096,                  # Must match verifier hidden size
    vocab_size=128256,                 # Must match verifier vocab size
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transformer_config` | `PretrainedConfig` | required | Verifier model transformer configuration |
| `num_speculative_steps` | `int` | `3` | Number of speculative prediction steps |
| `mtp_loss_step_weights` | `list[float]` | `[0.51, 0.31, 0.18]` | Per-step cross-entropy loss weights |
| `hidden_size` | `int` | required | Hidden dimension (must match verifier) |
| `vocab_size` | `int` | required | Vocabulary size (must match verifier) |

## Training

### Data Generation

FastMTP training uses the same data generation pipeline as EAGLE-3. Generate training data using:

```bash
python scripts/data_generation_offline.py \
    --target-model-path <verifier_model> \
    --train-data-path <dataset> \
    --output-dir ./training_data \
    --max-samples 5000
```

### Training Command

!!! note "Status"
    FastMTP training support is currently in development. Check the [speculators repository](https://github.com/vllm-project/speculators) for the latest updates.

Once available, training will follow the standard training script pattern:

```bash
torchrun --nnodes=1 --nproc_per_node=<num_gpus> scripts/train.py \
    --speculator-type mtp \
    --verifier-name-or-path <verifier_model> \
    --data-path ./training_data \
    --save-path ./checkpoints/fastmtp \
    --epochs 10 \
    --lr 1e-4
```

### Loss Function

FastMTP uses weighted cross-entropy loss across speculative steps:

$$
\mathcal{L} = \sum_{k=0}^{K-1} w_k \cdot \text{CE}(\hat{y}_k, y_{k+1})
$$

Where:

- $K$ is the number of speculative steps
- $w_k$ is the weight for step $k$ (default: `[0.51, 0.31, 0.18]`)
- $\hat{y}_k$ are the predicted logits at step $k$
- $y_{k+1}$ are the target token labels shifted by $k+1$ positions

The decreasing weights reflect decreasing prediction accuracy at farther steps.

## Inference with vLLM

!!! note "Status"
    FastMTP inference support in vLLM is under development. Check the [vLLM documentation](https://docs.vllm.ai/) for deployment instructions once available.

Once supported, FastMTP models can be served using vLLM:

```bash
vllm serve <path_or_hub_id_to_fastmtp_model>
```

## References

- **FastMTP Paper**: [FastMTP: Harnessing Fast Multi-Token Prediction for Speculative Decoding](https://arxiv.org/abs/2509.18362)
- **DeepSeek-V3**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Original multi-token prediction architecture
- **Reference Implementation**: [Tencent-BAC/FastMTP](https://github.com/Tencent-BAC/FastMTP)
