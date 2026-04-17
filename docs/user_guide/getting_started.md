# Getting Started

This guide walks you through the process of training your first speculator model using the Speculators library.

## Overview

Training a speculator involves the following high-level steps:

1. **Prepare Data** - Tokenize and preprocess your training dataset
2. **Generate Hidden States** - Extract hidden states from the target model using vLLM
3. **Build Vocabulary Mapping** - Create mappings between draft and target vocabularies
4. **Train Model** - Train the speculator using the prepared data

Speculators supports two training modes:

- **Online Training** - Hidden states are generated on-demand during training
- **Offline Training** - Hidden states are pre-generated and cached to disk before training

## Training Modes

### Online Training (Recommended for Development)

Online training generates hidden states on-demand during the training process. This approach:

- **Pros:** Lower disk usage, easier to get started, no pre-generation step
- **Cons:** Requires vLLM server running alongside training, may be slower per epoch
- **Best for:** Iterative development, smaller datasets, quick experiments

**Quick Start:**

```bash
# 1. Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000

# 2. Launch vLLM server for hidden states generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 4 --port 8000

# 3. Train (in a separate terminal)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

See [Train Eagle3 Online Tutorial](tutorials/train_eagle3_online.md) for detailed instructions.

### Offline Training (Recommended for Production)

Offline training pre-generates all hidden states before training begins. This approach:

- **Pros:** Faster training iterations, reproducible, better for large-scale training
- **Cons:** Requires significant disk space, longer initial setup
- **Best for:** Production training runs, large datasets, repeated experiments

**Quick Start:**

```bash
# 1. Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000

# 2. Launch vLLM server
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct

# 3. Generate hidden states offline
python scripts/data_generation_offline2.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-samples 5000

# 4. Train with cached hidden states
torchrun --standalone --nproc_per_node 4 scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

See [Train Eagle3 Offline Tutorial](tutorials/train_eagle3_offline.md) for detailed instructions.

## End-to-End Pipeline

For a complete automated workflow, use the `gen_and_train.py` script which handles all steps:

```python
from scripts.gen_and_train import DataGenArgs, VocabMappingArgs, TrainArgs, run_e2e

# Configure data generation
data_gen_args = DataGenArgs(
    train_data_path="sharegpt",
    max_samples=5000,
    seq_length=8192,
)

# Configure vocabulary mapping
vocab_mapping_args = VocabMappingArgs(
    draft_vocab_size=32000,
    target_vocab_size=128256,  # From target model config
)

# Configure training
train_args = TrainArgs(
    logger="tensorboard",
    lr=3e-5,
    total_seq_len=8192,
    run_name="my_first_speculator",
    epochs=10,
)

# Run complete pipeline
run_e2e(
    verifier_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    output_path="./output",
    data_gen_args=data_gen_args,
    vocab_mapping_args=vocab_mapping_args,
    train_args=train_args,
)
```

See examples in [`examples/data_generation_and_training/`](https://github.com/vllm-project/speculators/tree/main/examples/data_generation_and_training).

## Supported Algorithms

Speculators currently supports the following speculative decoding algorithms:

### EAGLE-3

EAGLE-3 is the primary algorithm, offering excellent performance across a wide range of models:

- **Supported Models:** Llama, Qwen3, Qwen3-MoE, Qwen3-VL, gpt-oss, and more
- **Draft Vocab Size:** Typically 32K tokens
- **Training Time:** ~35 minutes for 5K samples on 2x H100 GPUs
- **Use Cases:** General-purpose speculative decoding

See [EAGLE-3 Algorithm](algorithms/eagle3.md) and [EAGLE-3 Training Tutorial](tutorials/train_eagle3_online.md).

### DFlash

DFlash is a newer algorithm optimized for specific workloads:

- **Supported Models:** Qwen3
- **Draft Vocab Size:** Typically 64K tokens
- **Block-based Prediction:** Predicts blocks of tokens simultaneously
- **Use Cases:** Specialized workloads with block-aligned patterns

See [DFlash Algorithm](algorithms/dflash.md) and [DFlash Training Tutorial](tutorials/train_dflash_online.md).

For help choosing between algorithms, see the [Algorithm Decision Guide](algorithms/decision_guide.md).

## Supported Model Types

Speculators supports training for the following model architectures:

- **Dense Models:** Llama, Qwen3, gpt-oss
- **Mixture-of-Experts (MoE):** Qwen3-MoE
- **Vision-Language Models:** Qwen3-VL

For a complete list of tested models, see the [supported models table](../index.md#supported-models).

## Key Concepts

### Target Model vs Speculator

- **Target Model (Verifier):** The large, high-quality model you want to accelerate (e.g., Llama-3.1-70B)
- **Speculator (Draft Model):** A small, fast model that proposes tokens for the target model to verify

### Hidden States

Hidden states are intermediate representations from the target model used to train the speculator. They capture the model's internal knowledge at specific layers.

### Vocabulary Mapping

Speculators use a smaller vocabulary than the target model (e.g., 32K vs 128K tokens) for faster inference. The vocabulary mapping ensures:

- `t2d` (target-to-draft): Boolean mask indicating which target tokens are in the draft vocabulary
- `d2t` (draft-to-target): Index mapping from draft tokens to target tokens

### Online vs Offline Generation

- **Online:** vLLM generates hidden states during training (lower disk usage)
- **Offline:** Hidden states pre-generated and cached (faster training, reproducible)

## Next Steps

- **Tutorials:** Follow step-by-step guides in the [Tutorials](tutorials/README.md) section
- **Features:** Learn about specific capabilities in the [Features](features/README.md) section
- **Algorithms:** Understand different speculative decoding approaches in [Algorithms](algorithms/README.md)
- **CLI Reference:** Explore all command-line options in [CLI Reference](../cli/README.md)
- **Troubleshooting:** Find solutions to common issues in [Troubleshooting](troubleshooting.md)

## Additional Resources

- [Response Regeneration](features/response_regeneration.md) - Improve training data quality
- [Evaluating Model Performance](tutorials/evaluating_performance.md) - Benchmark your trained speculator
- [Serving in vLLM](tutorials/serve_vllm.md) - Deploy your model for inference
- [Converting Existing Models](tutorials/convert_eagle3.md) - Convert third-party models to Speculators format
