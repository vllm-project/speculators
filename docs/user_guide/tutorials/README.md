# Tutorials

Step-by-step tutorials for training, evaluating, and deploying speculative decoding models with Speculators.

## Overview

These tutorials guide you through complete workflows, from data preparation to serving trained models in production. Each tutorial includes practical examples, expected outputs, and troubleshooting tips.

## Training Tutorials

### [Train EAGLE-3 Model Online](train_eagle3_online.md)

Learn how to train an EAGLE-3 speculator using online training, where hidden states are generated on-demand during training.

**What you'll learn:**

- Prepare training data
- Launch vLLM for hidden states extraction
- Train with online generation
- Validate and serve your model

**Time required:** ~1-2 hours | **Difficulty:** Beginner

### [Train EAGLE-3 Model Offline](train_eagle3_offline.md)

Learn how to train an EAGLE-3 speculator using offline training with pre-generated hidden states.

**What you'll learn:**

- Generate hidden states in advance
- Train using cached hidden states
- Optimize for faster training iterations

**Time required:** ~2-3 hours | **Difficulty:** Intermediate

### [Train DFlash Model Online](train_dflash_online.md)

Learn how to train a DFlash speculator model with block-based token generation.

**What you'll learn:**

- Configure DFlash-specific parameters
- Train with block-based generation
- Optimize block sizes for your use case

**Time required:** ~1-2 hours | **Difficulty:** Intermediate

## Data & Evaluation Tutorials

### [Response Regeneration](response_regeneration.md)

Generate diverse training data by regenerating responses with the target model.

**What you'll learn:**

- Set up response regeneration
- Control diversity and temperature
- Integrate regenerated data into training

**Time required:** ~30 minutes | **Difficulty:** Beginner

### [Evaluating Model Performance](evaluating_performance.md)

Benchmark and evaluate your trained speculator models.

**What you'll learn:**

- Measure speedup and acceptance rates
- Use evaluation frameworks
- Compare different models and configurations

**Time required:** ~1 hour | **Difficulty:** Intermediate

## Deployment Tutorials

### [Convert EAGLE-3 Model](convert_eagle3.md)

Convert models from third-party EAGLE repositories to the Speculators format.

**What you'll learn:**

- Use conversion tools
- Validate converted models
- Deploy converted models in vLLM

**Time required:** ~30 minutes | **Difficulty:** Beginner

### [Serve in vLLM](serve_vllm.md)

Deploy your trained speculator models in vLLM for production inference.

**What you'll learn:**

- Configure vLLM for speculative decoding
- Serve models via OpenAI-compatible API
- Monitor and optimize serving performance

**Time required:** ~30 minutes | **Difficulty:** Beginner

## Prerequisites

Most tutorials assume you have:

- Python 3.10+
- CUDA-capable GPU(s)
- `speculators` installed with appropriate extras
- Basic familiarity with PyTorch and command-line tools

## Getting Help

If you encounter issues while following a tutorial:

1. Check the [Troubleshooting Guide](../troubleshooting.md)
2. Search [GitHub Issues](https://github.com/vllm-project/speculators/issues)
3. Ask in `#speculators` on the [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack)
