# DFlash

DFlash is a block-based speculative decoding algorithm optimized for generating tokens in parallel blocks. It provides an alternative to EAGLE-3 for specific workloads where block-aligned prediction patterns emerge.

## Overview

DFlash (Draft Flash) uses anchor-based speculation to predict multiple tokens ahead in blocks rather than sequentially. This approach can be beneficial for certain types of content generation where patterns align with block boundaries.

**Key Features:**

- **Block-based prediction:** Predicts tokens in blocks (e.g., 8 tokens at a time)
- **Anchor point sampling:** Uses anchor positions for efficient attention
- **Qwen3-specific:** Currently optimized for Qwen3 model family
- **Larger draft vocabulary:** Typically uses 64K vocab vs 32K for EAGLE-3

**Current Status:**

⚠️ DFlash is a newer algorithm with more limited support than EAGLE-3:

- **Supported models:** Qwen3 (other models coming soon)
- **vLLM support:** Available but less mature than EAGLE-3
- **Training infrastructure:** Fully functional
- **Production usage:** Recommended for experimentation, not yet production-critical workloads

## How It Works

### Architecture

DFlash predicts tokens using a block-based approach:

1. **Anchor selection:** Select anchor positions in the sequence
2. **Block prediction:** Predict blocks of tokens from each anchor
3. **Attention masking:** Use specialized attention masks for block predictions
4. **Verification:** Target model verifies predicted blocks

```
┌─────────────────────────────────────────────────┐
│              Target Model (frozen)              │
│                                                 │
│  Hidden States from Selected Layers             │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│             DFlash Draft Model                   │
│                                                  │
│  ┌──────────────────────────────────────┐       │
│  │ Select Anchor Points                 │       │
│  └──────────────┬───────────────────────┘       │
│                 │                                │
│  ┌──────────────▼───────────────────────┐       │
│  │ Block-based Attention                │       │
│  │ (via anchor block masks)             │       │
│  └──────────────┬───────────────────────┘       │
│                 │                                │
│  ┌──────────────▼───────────────────────┐       │
│  │ Qwen3-specific Decoder Layers        │       │
│  └──────────────┬───────────────────────┘       │
│                 │                                │
│  ┌──────────────▼───────────────────────┐       │
│  │ Predict Multiple Blocks              │       │
│  │ (block_size tokens per anchor)       │       │
│  └──────────────┬───────────────────────┘       │
│                 │                                │
│            Draft Logits                         │
└─────────────────────────────────────────────────┘
```

### Block-Based Prediction

Unlike EAGLE-3's sequential prediction, DFlash predicts in blocks:

**EAGLE-3 (sequential):**

```
Token 1 → Token 2 → Token 3 → Token 4 → Token 5
```

**DFlash (block-based):**

```
Anchor 1 → [Block: Tokens 1-8]
Anchor 2 → [Block: Tokens 9-16]
Anchor 3 → [Block: Tokens 17-24]
```

This can be more efficient when predictions naturally align with block boundaries.

### Anchor Point Mechanism

Anchor points are positions where the model has high confidence:

1. **Select anchors:** Choose positions based on confidence or fixed intervals
2. **Predict from anchors:** Generate a block of tokens from each anchor
3. **Verify blocks:** Target model verifies each block
4. **Accept valid blocks:** Use longest valid prefix

## Architecture Details

### Model Components

**DFlash Config:**

```python
DFlashSpeculatorConfig(
    draft_vocab_size=64000,           # Larger vocab than EAGLE-3
    block_size=8,                     # Tokens per block
    max_anchors=256,                  # Max anchor positions for training
    target_layer_ids=[...],           # Hidden state layers
    transformer_layer_config=...,     # Qwen3-specific decoder
)
```

### Attention Mechanism

DFlash uses specialized attention masks:

- **Anchor block masks:** Prevent attention across block boundaries
- **Causal masking:** Maintain autoregressive property within blocks
- **Document masking:** Prevent cross-document attention in batches

### Qwen3 Decoder Layers

DFlash currently uses Qwen3-specific decoder layers:

- `Qwen3DFlashDecoderLayer` - Custom decoder with DFlash attention
- Rotary embeddings
- MLP architectures matching Qwen3

## Configuration

### Block Size

The `block_size` parameter controls prediction granularity:

```bash
--block-size 8   # Default: 8 tokens per block
--block-size 16  # Larger blocks (fewer anchors)
--block-size 4   # Smaller blocks (more anchors)
```

**Trade-offs:**

- Smaller blocks: More anchors, finer control, potentially higher overhead
- Larger blocks: Fewer anchors, coarser predictions, faster if blocks align well

### Max Anchors

During training, limit the number of anchor positions:

```bash
--max-anchors 256  # Default
--max-anchors 512  # Allow more anchors (higher memory)
```

### Draft Vocabulary

DFlash typically uses a larger vocabulary than EAGLE-3:

```bash
--draft-vocab-size 64000  # Default for DFlash
```

This provides better token coverage at the cost of slightly slower inference.

## Training DFlash

### Basic Training Command

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

### Training Requirements

- **Supported models:** Qwen3 (8B, 14B, 32B, MoE variants)
- **Dataset:** Same as EAGLE-3 (ShareGPT, UltraChat, etc.)
- **Minimum samples:** 5K for testing, 50K+ for production
- **Hardware:** Similar to EAGLE-3 training

### DFlash-Specific Hyperparameters

```bash
--block-size 8         # Block prediction size
--max-anchors 256      # Max anchors during training
--draft-vocab-size 64000  # Larger vocab for better coverage
```

## Performance

### When DFlash Excels

DFlash can outperform EAGLE-3 in scenarios with:

✅ **Structured output** - JSON, code, or formatted text ✅ **Repetitive patterns** - Lists, tables, or template-based content ✅ **Fixed-length elements** - Tokens that naturally align with block sizes ✅ **Qwen3 models** - Optimized for Qwen3 architecture

### When to Use EAGLE-3 Instead

Prefer EAGLE-3 for:

⚠️ **General conversation** - Free-form dialogue ⚠️ **Creative writing** - Unpredictable token sequences ⚠️ **Non-Qwen3 models** - Llama, gpt-oss, etc. ⚠️ **Production deployment** - More mature and tested

### Typical Metrics

Performance varies significantly based on workload:

**Structured output (JSON generation):**

```
Average acceptance: 2.5-3.5 tokens
Speedup: 2.5-3.5x
```

**General conversation:**

```
Average acceptance: 1.5-2.0 tokens
Speedup: 1.5-2.0x
```

## Advantages & Limitations

### Advantages

✅ **Block-based efficiency** - Can predict multiple tokens in parallel ✅ **Larger vocabulary** - Better token coverage with 64K vocab ✅ **Structured content** - Excels at formatted outputs ✅ **Anchor flexibility** - Adaptive anchor selection

### Limitations

⚠️ **Limited model support** - Currently Qwen3 only ⚠️ **Less mature** - Newer algorithm with less production testing ⚠️ **Workload dependent** - Performance varies more than EAGLE-3 ⚠️ **Memory overhead** - Anchor tracking adds memory cost ⚠️ **vLLM integration** - Less optimized than EAGLE-3 in vLLM

## Best Practices

1. **Start with EAGLE-3** - Use DFlash only if you have specific requirements
2. **Test on your workload** - DFlash performance varies by use case
3. **Use default block size** - Start with 8 tokens per block
4. **Match target model** - Currently requires Qwen3
5. **Benchmark carefully** - Compare with EAGLE-3 on your specific tasks
6. **Monitor memory** - Anchor tracking uses more memory than EAGLE-3

## Use Cases

### Good Fit

**JSON API responses:**

```python
# DFlash can predict structured blocks efficiently
{
  "id": 123,           # Block 1
  "name": "...",       # Block 2
  "description": "..." # Block 3
}
```

**Code generation:**

```python
# Block-aligned function definitions
def function_name():    # Block 1
    # Implementation    # Block 2
    return result       # Block 3
```

**Table generation:**

```markdown
| Column 1 | Column 2 | Column 3 |  # Block 1
| -------- | -------- | -------- |  # Block 2
| Value 1  | Value 2  | Value 3  |  # Block 3
```

### Poor Fit

**Free-form dialogue:**

```
User: How are you?
Assistant: I'm doing well, thank you for asking! How can I help you today?
# Unpredictable lengths - better suited for EAGLE-3
```

**Creative writing:**

```
Once upon a time, in a faraway land, there lived a brave knight who...
# Unpredictable narrative flow - better suited for EAGLE-3
```

## Development Status

### Current Support

✅ **Training:** Fully functional ✅ **Qwen3 models:** 8B, 14B, 32B, MoE variants ✅ **vLLM deployment:** Available ✅ **Offline generation:** Supported

### In Progress

⏳ **Additional architectures:** Llama, gpt-oss support ⏳ **Optimized vLLM kernels:** Performance improvements ⏳ **Adaptive anchors:** Dynamic anchor selection ⏳ **Production validation:** Large-scale testing

### Future Work

🔮 **Multi-model support:** Expand beyond Qwen3 🔮 **Hybrid approaches:** Combine DFlash and EAGLE-3 🔮 **Auto-tuning:** Automatic block size selection

## Comparison with EAGLE-3

| Feature                 | EAGLE-3                         | DFlash           |
| ----------------------- | ------------------------------- | ---------------- |
| **Model Support**       | Llama, Qwen3, gpt-oss, MoE, VLM | Qwen3 only       |
| **Prediction Style**    | Sequential                      | Block-based      |
| **Draft Vocab Size**    | 32K typical                     | 64K typical      |
| **Maturity**            | Production-ready                | Experimental     |
| **General Performance** | 2-3x speedup                    | 1.5-3.5x speedup |
| **Structured Output**   | Good                            | Excellent        |
| **Free-form Text**      | Excellent                       | Good             |
| **Memory Overhead**     | Lower                           | Higher (anchors) |
| **Training Complexity** | Standard                        | Slightly higher  |
| **vLLM Integration**    | Mature                          | Developing       |

## Migration from EAGLE-3

If you have an EAGLE-3 model and want to try DFlash:

```bash
# Train DFlash with same data
python scripts/train.py \
  --speculator-type dflash \  # Change algorithm
  --verifier-name-or-path Qwen/Qwen3-8B \  # Qwen3 only
  --data-path ./same_training_data \  # Reuse data
  --draft-vocab-size 64000 \  # Larger vocab
  --block-size 8 \
  --epochs 10
```

Compare performance on your specific workload.

## Troubleshooting

### Poor Performance

If DFlash underperforms:

1. **Check workload type** - May not suit free-form text
2. **Adjust block size** - Try 4, 8, 16
3. **Compare with EAGLE-3** - Benchmark both algorithms
4. **Verify model compatibility** - Ensure using Qwen3

### Memory Issues

If training runs out of memory:

```bash
# Reduce max anchors
--max-anchors 128

# Reduce sequence length
--total-seq-len 4096

# Use fewer workers
--num-workers 4
```

### Training Not Converging

```bash
# Adjust learning rate
--lr 1e-5  # Lower LR

# Increase epochs
--epochs 20

# Check data quality
# Ensure dataset is properly preprocessed
```

## See Also

- [EAGLE-3 Algorithm](eagle3.md) - Alternative algorithm (recommended for most users)
- [Algorithm Decision Guide](decision_guide.md) - Choosing between algorithms
- [Train DFlash Tutorial](../tutorials/train_dflash_online.md) - Step-by-step training guide
- [Training Feature](../features/training.md) - General training documentation

## Research & Future Directions

DFlash is an active research area with ongoing development:

- **Adaptive block sizing:** Dynamically adjust block size based on content
- **Hybrid prediction:** Combine sequential and block-based approaches
- **Multi-architecture support:** Extend beyond Qwen3
- **Improved anchor selection:** Smarter anchor point identification

Contributions and feedback welcome!

## Summary

**Use DFlash when:**

- Working with Qwen3 models
- Generating structured output (JSON, code, tables)
- Willing to experiment and benchmark

**Use EAGLE-3 when:**

- Need broad model support
- Generating free-form text
- Want production-ready stability
- Working with non-Qwen3 models

For most users, **EAGLE-3 is the recommended starting point**. Try DFlash if you have specific requirements that align with its strengths.
