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
python scripts/prepare_data.py ...
# Step 2: Launch vLLM server
python scripts/launch_vllm.py ...
# Step 3: Generate hidden states offline
python scripts/data_generation_offline.py ...
# Step 4: Stop vLLM server
# Step 5: Train the speculator
python scripts/train.py ...
```

### Full Training Pipeline (Online)

```bash
# Step 1: Prepare data
python scripts/prepare_data.py ...
# Step 2: Launch vLLM server
python scripts/launch_vllm.py ...
# Step 3: Train with online hidden states generation
python scripts/train.py ...
```
