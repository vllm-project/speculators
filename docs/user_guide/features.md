# Features

Speculators is built to make training and deploying speculative decoding models fast, scalable, and easy to integrate into existing workflows. Here are some of the key features.

## Distributed Training with FSDP

Speculators supports multi-GPU training via PyTorch Fully Sharded Data Parallel (FSDP). Model parameters are sharded per-layer with a mixed precision policy — bfloat16 for parameters, float32 for gradient reductions — and distributed checkpointing handles save/restore across ranks automatically.

## Draft Vocabulary Support

Draft models use a reduced vocabulary for faster inference. Speculators automatically builds vocabulary mappings (`t2d` and `d2t` tensors) from token frequency statistics collected during data preparation, selecting the most frequent tokens for the draft vocab. Pre-built mappings can also be provided manually.

## Multi-Backend Metric Logging

Training metrics can be logged to TensorBoard, Weights & Biases, and TrackIO — individually or simultaneously, so that you can use your preferred experiment tracking tool.

## Automatic Chat Template Detection

During data preparation, Speculators automatically detects assistant response boundaries to build loss masks. It first tries HuggingFace's native `assistant_tokens_mask` support, then falls back to regex-based pattern detection — including stripping `<think>` blocks from reasoning models. No manual template configuration is needed for most models.

## Performant Flex Attention

Both [EAGLE-3](algorithms/eagle3.md) and [DFlash](algorithms/dflash.md) models use PyTorch's `flex_attention` with `BlockMask` for efficient, structured attention patterns (causal, document-aware, and anchor-based). The EAGLE-3 forward pass is wrapped with `torch.compile` for additional runtime optimization.

## Efficient Sequence Packing

The multipack batch sampler uses an LPT (Longest Processing Time First) bin-packing algorithm to pack variable-length sequences into batches, maximizing GPU utilization while respecting per-device token limits. This avoids the wasted compute from naive padding.

## Checkpoint Resume

Training automatically resumes from the latest checkpoint, restoring model weights, optimizer state, and scheduler state. The checkpointer tracks the best validation loss and maintains a symlink to the best checkpoint for easy model selection.

## Model Conversion

Speculators can convert pre-trained models from third-party repositories (EAGLE v1/v2/v3, HASS) into Speculators format for direct deployment with vLLM.

______________________________________________________________________

For hands-on guides covering the full workflow, see the [Tutorials](tutorials/index.md).
