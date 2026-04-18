# Algorithms

Speculators supports multiple speculative decoding algorithms, each with different strengths and trade-offs. This section provides an overview of the available algorithms and guidance on choosing the right one for your use case.

## Overview

Speculative decoding algorithms use a smaller "draft" model to predict multiple tokens ahead, which are then verified by the larger target model in parallel. Different algorithms employ different strategies for generating these draft tokens.

## Supported Algorithms

### [EAGLE-3](eagle3.md)

**EAGLE** (Extrapolation Algorithm for Greater Language-model Efficiency) is a state-of-the-art speculative decoding algorithm that uses auto-regression combined with feature extrapolation from the target model's hidden states.

**Key characteristics:**

- Lightweight draft model (typically 1 layer)
- Uses hidden states from multiple layers of the target model
- Excellent for general-purpose acceleration
- Well-tested across various model architectures

**Best for:** Most production use cases, especially when you want a balance of speed improvement and ease of deployment.

### [DFlash](dflash.md)

**DFlash** (Decoding with Flash attention) is a block-based speculative decoding algorithm optimized for efficient parallel verification.

**Key characteristics:**

- Block-based token generation
- Optimized for flash attention
- Configurable block sizes for different workloads

**Best for:** Workloads where batch processing and parallel verification are critical.

### [Decision Guide](decision_guide.md)

Not sure which algorithm to choose? The decision guide helps you select the right algorithm based on your:

- Model architecture (Llama, Qwen, GPT-OSS, etc.)
- Use case (chat, completion, code generation)
- Performance requirements (latency vs throughput)
- Hardware constraints (GPU memory, compute budget)

## Adding New Algorithms

Interested in implementing a new speculative decoding algorithm? See the [Developer Guide](../../developer/add_algorithm.md) for instructions on adding custom algorithms to Speculators.
