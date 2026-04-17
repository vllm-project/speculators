# Train EAGLE-3 Model Offline

This tutorial walks you through training an EAGLE-3 speculator model using **offline training**, where hidden states are pre-generated and cached before training begins.

## Overview

Offline training is ideal for:
- Production training runs
- Large datasets (50K+ samples)
- Repeated experimentation (multiple training runs on same data)
- Reproducible results
- Faster training iterations

**What you'll learn:**
- How to prepare and preprocess training data
- How to generate hidden states offline
- How to train using cached hidden states
- How to optimize for production workflows

**Time required:** ~2-4 hours (including data generation)

**Prerequisites:**
- Python 3.10+
- CUDA-capable GPU(s)
- `speculators` installed with `[datagen]` extras
- `vllm` installed
- Sufficient disk space for hidden states (~260 GB per 1K samples for Llama-3.1-8B)

## Step 1: Prepare Your Data

Preprocess your training dataset:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --data ultrachat \
  --output ./training_data \
  --max-samples 50000
```

**For production training:**
```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --data ultrachat \
  --output ./production_data \
  --turn-dropout \                    # Data augmentation
  --minimum-valid-tokens 10           # Quality filter
```

**Expected output:**
```
training_data/
├── data-*.arrow files (preprocessed dataset)
├── dataset_info.json
├── state.json
└── token_freq.pt
```

**Time:** ~5-15 minutes for 50K samples

## Step 2: Launch vLLM Server

Launch vLLM configured for hidden states extraction:

```bash
# For 8B model - use data parallelism
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 4 --port 8000

# For 70B model - use tensor parallelism
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/launch_vllm.py \
  meta-llama/Llama-3.3-70B-Instruct \
  -- --tensor-parallel-size 8 --port 8000
```

**Verify server is ready:**
```bash
curl http://localhost:8000/v1/models
```

## Step 3: Generate Hidden States Offline

Use `data_generation_offline2.py` to pre-generate all hidden states:

```bash
python scripts/data_generation_offline2.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-samples 50000 \
  --concurrency 32 \
  --validate-outputs
```

**Key parameters:**
- `--preprocessed-data` - Path to prepared data from Step 1
- `--output` - Where to save hidden states
- `--concurrency 32` - Number of parallel requests (tune based on GPU memory)
- `--validate-outputs` - Verify file integrity (recommended for production)

**Expected output:**
```
Processing samples: 100%|███████████| 50000/50000 [45:23<00:00, 18.4it/s]
Generated 50000 hidden state files
Validation: 50000/50000 files OK
```

**Output structure:**
```
hidden_states/
├── hs_0.safetensors
├── hs_1.safetensors
├── hs_2.safetensors
├── ...
└── hs_49999.safetensors
```

**Generation time:**
- 8B model, 50K samples, 4 GPUs (DP): ~45-60 minutes
- 70B model, 50K samples, 8 GPUs (TP): ~90-120 minutes

### Optimizing Generation Speed

**Increase concurrency:**
```bash
--concurrency 64  # Higher throughput, needs more GPU memory
```

**Use more GPUs:**
```bash
# 8 GPUs with DP=8
python scripts/launch_vllm.py model -- --data-parallel-size 8
```

**Skip validation (faster, less safe):**
```bash
# Omit --validate-outputs for faster generation
```

### Resuming Interrupted Generation

If generation is interrupted, simply rerun the same command:

```bash
python scripts/data_generation_offline2.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-samples 50000 \
  --concurrency 32
```

The script automatically detects existing `hs_*.safetensors` files and skips them.

## Step 4: Stop vLLM Server

After hidden states generation is complete, stop the vLLM server:

```bash
# Press Ctrl+C in the vLLM terminal
```

You don't need vLLM running during offline training.

## Step 5: Train with Cached Hidden States

### Single-GPU Training

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5 \
  --logger tensorboard \
  --run-name llama31_8b_offline
```

### Multi-GPU Training (FSDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \
  --on-missing raise \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5 \
  --logger tensorboard \
  --run-name llama31_8b_offline
```

**Critical parameters:**
- `--hidden-states-path` - Points to cached hidden states
- `--on-missing raise` - Fail if any hidden states are missing (recommended)

**Training speed:**
Offline training is **significantly faster** than online:
- 50K samples, 10 epochs, 4 GPUs: ~2-3 hours (vs 8-10 hours online)
- Each epoch: ~12-18 minutes

**Expected output:**
```
Epoch 1/10: 100%|███████████| 1562/1562 [12:34<00:00,  2.07it/s]
Train Loss: 2.1234 | Val Loss: 2.0567 | Accuracy: 0.42
Saved checkpoint to ./checkpoints/0

Epoch 2/10: 100%|███████████| 1562/1562 [12:28<00:00,  2.09it/s]
Train Loss: 1.9845 | Val Loss: 1.9123 | Accuracy: 0.48
Saved checkpoint to ./checkpoints/1
...
```

## Step 6: Monitor Training

### TensorBoard

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 to monitor:
- Loss curves
- Accuracy metrics
- Learning rate schedule

### Key Metrics to Watch

- **Train loss decreasing** - Model is learning
- **Val loss decreasing** - Generalizing well
- **Accuracy increasing** - Better token predictions
- **No overfitting** - Val loss not increasing while train loss decreases

## Step 7: Checkpoint Management

After training completes:

```
checkpoints/
├── 0/ ... 9/          # All epochs
├── best -> 5/         # Best validation loss (if --save-best)
└── latest -> 9/       # Most recent
```

### Select Best Checkpoint

**Option A: Lowest validation loss**
```bash
ls -la checkpoints/best
```

**Option B: Manually evaluate**
Test each checkpoint on your validation set.

## Step 8: Validate Your Model

### Quick Sanity Check

```bash
# Serve the trained model
vllm serve ./checkpoints/best

# In another terminal, test it
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='./checkpoints/best',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=50
)
print(response.choices[0].message.content)
"
```

### Full Evaluation

See [Evaluating Performance Tutorial](evaluating_performance.md) for comprehensive benchmarking.

## Advantages of Offline Training

### 1. Faster Training Iterations

**Offline:**
```
Pre-generate once: 60 min
Train (10 epochs): 120 min
Total: 180 min
```

**Online:**
```
Train (10 epochs): 600 min
Total: 600 min
```

For multiple experiments: Offline wins significantly.

### 2. Reproducibility

Hidden states are identical across runs:
- Same random seed → same results
- Easy to debug
- Fair comparisons between hyperparameters

### 3. Resource Separation

- **Generation:** Use all GPUs for vLLM
- **Training:** Use all GPUs for FSDP
- No resource contention

### 4. Disk-Based Caching

- Generate once, train many times
- Experiment with hyperparameters
- No need to regenerate hidden states

## Production Workflow

For production-scale training:

```bash
#!/bin/bash
set -e

# Configuration
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="./production_data"
HS_DIR="./hidden_states"
CHECKPOINT_DIR="./checkpoints"
MAX_SAMPLES=100000

# Step 1: Prepare high-quality data
echo "=== Preparing Data ==="
python scripts/prepare_data.py \
  --model $MODEL \
  --data sharegpt \
  --data ultrachat \
  --output $DATA_DIR \
  --max-samples $MAX_SAMPLES \
  --turn-dropout \
  --minimum-valid-tokens 20 \
  --num-preprocessing-workers 32

# Step 2: Generate hidden states
echo "=== Generating Hidden States ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/launch_vllm.py \
  $MODEL \
  -- --data-parallel-size 8 --port 8000 &

VLLM_PID=$!
sleep 90  # Wait for vLLM startup

python scripts/data_generation_offline2.py \
  --model $MODEL \
  --preprocessed-data $DATA_DIR \
  --output $HS_DIR \
  --max-samples $MAX_SAMPLES \
  --concurrency 64 \
  --validate-outputs

kill $VLLM_PID

# Step 3: Train with best practices
echo "=== Training ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train.py \
  --verifier-name-or-path $MODEL \
  --data-path $DATA_DIR \
  --hidden-states-path $HS_DIR \
  --on-missing raise \
  --save-path $CHECKPOINT_DIR \
  --draft-vocab-size 32000 \
  --num-layers 1 \
  --epochs 15 \
  --lr 3e-5 \
  --scheduler-type cosine \
  --scheduler-num-cosine-cycles 0.5 \
  --checkpoint-freq 1 \
  --save-best \
  --logger wandb \
  --run-name production_llama31_8b \
  --noise-std 0.05

echo "=== Training Complete ==="
echo "Best checkpoint: $CHECKPOINT_DIR/best"
```

## Disk Space Management

### Estimating Space Requirements

```python
# For Llama-3.1-8B:
# seq_len × num_layers × hidden_size × dtype_bytes
# 8192 × 4 × 4096 × 2 = ~268 MB per sample

# For 50K samples: ~13 TB
# For 10K samples: ~2.6 TB
# For 1K samples: ~260 GB
```

### Optimizing Disk Usage

**Use fast storage:**
```bash
# Store hidden states on NVMe SSD
--output /fast_storage/hidden_states
```

**Compress after training:**
```bash
# After successful training, compress hidden states
tar -czf hidden_states.tar.gz ./hidden_states/
# Can delete originals if needed
```

**Clean up intermediate files:**
```bash
# Remove hidden states after training if disk-constrained
rm -rf ./hidden_states/
# Keep preprocessed data and checkpoints
```

## Advanced: Parallel Experimentation

With offline hidden states, run multiple experiments in parallel:

```bash
# Experiment 1: Single layer, LR=3e-5
torchrun --nproc_per_node 4 scripts/train.py \
  --hidden-states-path ./hs \
  --save-path ./exp1 \
  --num-layers 1 --lr 3e-5 &

# Experiment 2: Two layers, LR=1e-5
torchrun --nproc_per_node 4 scripts/train.py \
  --hidden-states-path ./hs \
  --save-path ./exp2 \
  --num-layers 2 --lr 1e-5 &

# Both use same cached hidden states
```

## Common Issues & Solutions

### Issue: Missing Hidden States

**Error:**
```
FileNotFoundError: hs_1234.safetensors not found
```

**Solutions:**
```bash
# Check which files are missing
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --max-samples 50000  # Rerun to fill gaps
```

### Issue: Corrupted Hidden States Files

**Error:**
```
SafetensorsError: Invalid file format
```

**Solution:**
```bash
# Regenerate with validation
python scripts/data_generation_offline2.py \
  --preprocessed-data ./training_data \
  --output ./hidden_states \
  --validate-outputs  # Validates all files
```

Delete corrupted files and regenerate them.

### Issue: Disk Full During Generation

**Error:**
```
OSError: No space left on device
```

**Solutions:**
1. **Generate in batches:**
   ```bash
   # First 10K
   --max-samples 10000 --output ./hs_batch1

   # Next 10K (adjust dataset)
   --max-samples 20000 --output ./hs_batch2
   # etc.
   ```

2. **Use larger storage:**
   ```bash
   --output /large_disk/hidden_states
   ```

3. **Clean up old experiments:**
   ```bash
   rm -rf ./old_hidden_states/
   ```

## Comparison: Online vs Offline

| Aspect | Online | Offline |
|--------|--------|---------|
| **Initial setup** | Faster ✅ | Slower (generation time) |
| **Training speed** | Slower | Much faster ✅ |
| **Disk usage** | Minimal ✅ | High (TB scale) |
| **Reproducibility** | Lower | Perfect ✅ |
| **Multiple experiments** | Slow | Fast ✅ |
| **Resource efficiency** | vLLM + Training concurrently | Separate phases ✅ |
| **Best for** | Development | Production ✅ |

## Next Steps

After offline training:

1. **Benchmark performance** - [Evaluating Performance](evaluating_performance.md)
2. **Deploy to production** - [Serve in vLLM](serve_vllm.md)
3. **Experiment with hyperparameters** - Reuse cached hidden states
4. **Train larger models** - Scale to 70B, 405B models
5. **Share your model** - Upload to HuggingFace Hub

## Summary

You've learned how to:
- ✅ Prepare production-quality training data
- ✅ Pre-generate hidden states offline
- ✅ Train with cached hidden states for maximum speed
- ✅ Manage disk space efficiently
- ✅ Run reproducible experiments
- ✅ Optimize for production workflows

**Offline training** is the recommended approach for production deployments and large-scale training. The initial time investment in generating hidden states pays off through faster iterations and perfect reproducibility.

## See Also

- [Train EAGLE-3 Online](train_eagle3_online.md) - Online training guide
- [Offline Hidden States Generation](../features/offline_hidden_states.md) - Detailed generation docs
- [Training Feature](../features/training.md) - Complete training reference
- [Evaluating Performance](evaluating_performance.md) - Benchmark your model
- [EAGLE-3 Algorithm](../algorithms/eagle3.md) - Algorithm details
