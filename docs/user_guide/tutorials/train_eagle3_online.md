# Train EAGLE-3 Model Online

This tutorial walks you through training an EAGLE-3 speculator model using **online training**, where hidden states are generated on-demand during the training process.

## Overview

Online training is ideal for:

- Quick experimentation and iteration
- Smaller datasets (< 50K samples)
- Limited disk space
- Development and prototyping

**What you'll learn:**

- How to prepare training data
- How to launch vLLM for hidden states extraction
- How to train an EAGLE-3 model with online generation
- How to validate and serve your trained model

**Time required:** ~1-2 hours (including training)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- `speculators` installed with `[datagen]` extras
- `vllm` installed

## Step 1: Prepare Your Data

First, preprocess your training dataset:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000
```

**Parameters explained:**

- `--model` - The target model you want to accelerate
- `--data` - Dataset to use (`sharegpt`, `ultrachat`, or custom path)
- `--output` - Where to save preprocessed data
- `--max-samples` - Limit samples (optional, good for testing)

**Expected output:**

```
training_data/
├── data-00000-of-00002.arrow
├── data-00001-of-00002.arrow
├── dataset_info.json
├── state.json
└── token_freq.pt
```

**Time:** ~2-5 minutes for 5K samples

## Step 2: Launch vLLM Server

In a **separate terminal**, launch vLLM configured for hidden states extraction:

```bash
# Single GPU
python scripts/launch_vllm.py meta-llama/Llama-3.1-8B-Instruct

# Multiple GPUs with data parallelism (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 4 --port 8000
```

**The `--` separator:** Anything after `--` is passed directly to vLLM. Common options:

- `--data-parallel-size 4` - Use 4 GPUs for data parallelism
- `--tensor-parallel-size 2` - Use 2 GPUs for tensor parallelism
- `--port 8000` - Specify port (default: 8000)
- `--gpu-memory-utilization 0.9` - GPU memory to use

**Wait for server to start:**

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify server is running:**

```bash
curl http://localhost:8000/v1/models
```

You should see the model listed.

## Step 3: Start Training

In your **original terminal** (not the vLLM terminal), start training:

### Single-GPU Training

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5 \
  --logger tensorboard \
  --run-name llama31_8b_online
```

### Multi-GPU Training (FSDP)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5 \
  --logger tensorboard \
  --run-name llama31_8b_online
```

**Key parameters:**

- `--vllm-endpoint` - vLLM server URL (enables online generation)
- `--draft-vocab-size 32000` - Reduced vocabulary size
- `--epochs 10` - Number of training epochs
- `--lr 3e-5` - Learning rate
- `--logger tensorboard` - Enable TensorBoard logging

**Expected training output:**

```
Epoch 1/10: 100%|███████████| 156/156 [03:21<00:00,  1.29s/it]
Train Loss: 2.3456 | Val Loss: 2.2891 | LR: 3.00e-05
Saved checkpoint to ./checkpoints/0

Epoch 2/10: 100%|███████████| 156/156 [03:18<00:00,  1.27s/it]
Train Loss: 2.1234 | Val Loss: 2.0987 | LR: 2.70e-05
Saved checkpoint to ./checkpoints/1
...
```

**Training time:**

- 5K samples, 10 epochs, 1 GPU: ~30-40 minutes
- 5K samples, 10 epochs, 4 GPUs: ~15-20 minutes

## Step 4: Monitor Training

### TensorBoard

If you used `--logger tensorboard`:

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 to view:

- Training and validation loss curves
- Learning rate schedule
- Token prediction accuracy
- Per-epoch metrics

### Weights & Biases

For W&B logging:

```bash
python scripts/train.py \
  --logger wandb \
  --run-name llama31_8b_online \
  ...
```

### Console Output

Watch for:

- **Decreasing loss** - Model is learning
- **Validation loss** - Should decrease without overfitting
- **Learning rate** - Should decay according to schedule
- **Checkpoint saves** - Confirm checkpoints are being written

## Step 5: Inspect Checkpoints

After training, your checkpoints directory contains:

```
checkpoints/
├── 0/                          # Epoch 0
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   ├── optimizer_state_dict.pt
│   └── scheduler_state_dict.pt
├── 1/                          # Epoch 1
├── ...
├── 9/                          # Epoch 9 (final)
└── latest -> 9/                # Symlink to latest
```

Each checkpoint is a complete, self-contained speculator model ready for deployment.

## Step 6: Select Best Checkpoint

### Option A: Use Latest Checkpoint

```bash
cd checkpoints/latest
```

### Option B: Use Best Validation Loss

If you used `--save-best`:

```bash
cd checkpoints/best
```

### Option C: Manually Evaluate

Test each epoch's checkpoint on a validation set and choose the best performer.

## Step 7: Test Your Model

### Quick Test with vLLM

Stop the training vLLM server (Ctrl+C), then serve your speculator:

```bash
vllm serve ./checkpoints/latest
```

### Send a Test Request

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="./checkpoints/latest",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Verify Speculative Decoding

Check vLLM logs for speculative decoding metrics:

```
Average acceptance rate: 1.8 tokens
Speculative tokens proposed: 5
```

## Step 8: Benchmark Performance

Use [GuideLLM](https://github.com/vllm-project/guidellm) to benchmark:

```bash
# Install GuideLLM
pip install guidellm

# Benchmark your speculator
guidellm \
  --target http://localhost:8000/v1 \
  --model ./checkpoints/latest \
  --data-type emulated \
  --data "prompt_tokens=512,generated_tokens=256" \
  --request-rate 1.0
```

Compare against the base model to measure speedup.

See [Evaluating Performance Tutorial](evaluating_performance.md) for detailed benchmarking.

## Common Issues & Solutions

### Issue: vLLM Connection Timeout

**Error:**

```
TimeoutError: Request to vLLM timed out
```

**Solution:**

```bash
# Increase timeout in training
python scripts/train.py \
  --request-timeout 30 \  # Increase from default 15s
  --max-retries 5 \       # More retry attempts
  ...
```

### Issue: Out of Memory (Training)

**Error:**

```
torch.cuda.OutOfMemoryError
```

**Solutions:**

```bash
# Reduce sequence length
python scripts/train.py --total-seq-len 4096 ...

# Reduce batch size (via sequence packing)
# Use fewer dataloader workers
python scripts/train.py --num-workers 4 ...

# Use fewer GPUs for training (give more to vLLM)
```

### Issue: Out of Memory (vLLM)

**Error:**

```
vLLM: CUDA out of memory
```

**Solutions:**

```bash
# Reduce GPU memory utilization
python scripts/launch_vllm.py model -- --gpu-memory-utilization 0.7

# Reduce max model length
python scripts/launch_vllm.py model -- --max-model-len 4096

# Use more GPUs
python scripts/launch_vllm.py model -- --tensor-parallel-size 2
```

### Issue: Training Loss Not Decreasing

**Symptoms:** Loss plateaus or increases

**Solutions:**

1. **Lower learning rate:**

   ```bash
   --lr 1e-5  # Try lower LR
   ```

2. **Check data quality:**

   ```bash
   # Verify preprocessing succeeded
   ls -lh ./training_data/
   ```

3. **Increase training time:**

   ```bash
   --epochs 20  # Train longer
   ```

4. **Verify vLLM is working:**

   ```bash
   curl http://localhost:8000/v1/models
   ```

### Issue: Slow Training

**Symptoms:** Very slow iteration speed

**Solutions:**

1. **Increase vLLM concurrency (implicitly via batch size):** vLLM will handle more requests in parallel

2. **Use more vLLM GPUs:**

   ```bash
   --data-parallel-size 4  # More vLLM replicas
   ```

3. **Reduce dataloader workers:**

   ```bash
   --num-workers 8  # Instead of default 12
   ```

## Advanced: Hybrid Training

Start with online, cache for later epochs:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --hidden-states-path ./hidden_states \  # Where to cache
  --vllm-endpoint http://localhost:8000/v1 \
  --on-missing generate \     # Generate if missing
  --on-generate cache \       # Cache generated states
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10
```

**First epoch:** Generates and caches hidden states to `./hidden_states/` **Subsequent epochs:** Uses cached hidden states (faster)

## Next Steps

After training your model:

1. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
2. **Try offline training** - See [Train EAGLE-3 Offline](train_eagle3_offline.md) for faster iterations
3. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)
4. **Fine-tune further** - Use `--from-pretrained ./checkpoints/latest` to continue training
5. **Upload to HuggingFace** - Share your model with the community

## Complete Example Script

Here's a complete script for reference:

```bash
#!/bin/bash

# Step 1: Prepare data
echo "Preparing data..."
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data sharegpt \
  --output ./training_data \
  --max-samples 5000

# Step 2: Launch vLLM (in background)
echo "Launching vLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_vllm.py \
  meta-llama/Llama-3.1-8B-Instruct \
  -- --data-parallel-size 4 --port 8000 &

VLLM_PID=$!
sleep 60  # Wait for vLLM to start

# Step 3: Train
echo "Starting training..."
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node 4 \
  scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --data-path ./training_data \
  --vllm-endpoint http://localhost:8000/v1 \
  --save-path ./checkpoints \
  --draft-vocab-size 32000 \
  --epochs 10 \
  --lr 3e-5 \
  --logger tensorboard \
  --run-name llama31_8b_online

# Step 4: Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID

echo "Training complete! Checkpoints saved to ./checkpoints/"
```

## Summary

You've learned how to:

- ✅ Prepare training data
- ✅ Launch vLLM for online hidden states generation
- ✅ Train an EAGLE-3 model with online generation
- ✅ Monitor training progress
- ✅ Test and validate your trained model
- ✅ Troubleshoot common issues

**Online training** is perfect for quick iteration and development. For production training with large datasets, consider [offline training](train_eagle3_offline.md) for faster, more reproducible results.

## See Also

- [Train EAGLE-3 Offline](train_eagle3_offline.md) - Offline training guide
- [EAGLE-3 Algorithm](../algorithms/eagle3.md) - Algorithm details
- [Training Feature](../features/training.md) - Complete training reference
- [Evaluating Performance](evaluating_performance.md) - Benchmark your model
- [Serve in vLLM](serve_vllm.md) - Deploy to production
