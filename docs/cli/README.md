# CLI Reference

This page provides a comprehensive reference for all command-line interface (CLI) tools available in Speculators.

## Overview

Speculators provides four main CLI scripts for different stages of the speculative decoding workflow:

| Script | Purpose | Reference |
|--------|---------|-----------|
| `prepare_data.py` | Preprocess and tokenize datasets for training | [→ Details](prepare_data.md) |
| `data_generation_offline.py` | Generate hidden states offline using vLLM | [→ Details](data_generation_offline.md) |
| `launch_vllm.py` | Launch vLLM server configured for hidden states extraction | [→ Details](launch_vllm.md) |
| `train.py` | Train speculator models with online or offline hidden states | [→ Details](train.md) |

## Common Workflows

### Full Training Pipeline (Offline)

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./prepared_data \
  --max-samples 10000

# Step 2: Generate hidden states offline
python scripts/data_generation_offline.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --train-data-path sharegpt \
  --output-dir ./hidden_states \
  --tensor-parallel-size 2 \
  --batch-size 16

# Step 3: Train the speculator
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./prepared_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

### Full Training Pipeline (Online)

```bash
# Step 1: Prepare data
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./prepared_data \
  --max-samples 10000

# Step 2: Launch vLLM server
python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  --hidden-states-path /tmp/hidden_states \
  -- --port 8000 --tensor-parallel-size 2

# Step 3: Train with online hidden states generation
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./prepared_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \
  --on-generate delete \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

## See Also

- [Getting Started Guide](../user_guide/getting_started.md)
- [Training Tutorial](../user_guide/tutorials/train_eagle3_online.md)
- [Features: Training](../user_guide/features/training.md)
- [Features: Data Preparation](../user_guide/features/prepare_data.md)
- [vLLM CLI Reference](https://docs.vllm.ai/en/latest/cli/)
