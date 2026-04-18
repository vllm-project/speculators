# Troubleshooting

This guide covers common issues and their solutions when working with Speculators.

## Data Preparation Issues

### Chat Template Not Found

**Error:**

```
Error: No chat template found for model
```

**Cause:** Model's tokenizer doesn't include a chat template.

**Solution:**

```bash
# Use a different model with chat template support
# Or add custom template to your tokenizer
```

### Preprocessing Out of Memory

**Error:**

```
OutOfMemoryError during dataset preprocessing
```

**Solutions:**

```bash
# Reduce worker count
--num-preprocessing-workers 4

# Process in smaller batches
--max-samples 10000  # Process 10K at a time
```

### Token Frequency File Not Generated

**Issue:** `token_freq.pt` missing from output directory.

**Solution:** Verify the dataset was processed successfully:

```bash
ls -lh ./output_dir/
# Should see data-*.arrow files and token_freq.pt
```

## Hidden States Generation Issues

### vLLM Connection Timeout

**Error:**

```
TimeoutError: Request to vLLM timed out after 15s
```

**Solutions:**

```bash
# Increase timeout
python scripts/data_generation_offline2.py \
  --request-timeout 30 \
  --max-retries 5

# Check vLLM is running
curl http://localhost:8000/v1/models
```

### vLLM Out of Memory

**Error:**

```
CUDA out of memory during hidden states generation
```

**Solutions:**

```bash
# Reduce concurrency
--concurrency 16  # Instead of 32

# Use more GPUs
python scripts/launch_vllm.py model -- --data-parallel-size 4

# Reduce max model length
python scripts/launch_vllm.py model -- --max-model-len 4096
```

### Slow Hidden States Generation

**Issue:** Generation taking much longer than expected.

**Solutions:**

1. **Increase concurrency:**

   ```bash
   --concurrency 64
   ```

2. **Use more GPUs:**

   ```bash
   --data-parallel-size 8
   ```

3. **Skip validation for speed (risky):**

   ```bash
   # Omit --validate-outputs
   ```

## Training Issues

### Training Loss Not Decreasing

**Symptoms:**

- Loss plateaus or increases
- Validation loss doesn't improve

**Solutions:**

1. **Lower learning rate:**

   ```bash
   --lr 1e-5  # Instead of 3e-5
   ```

2. **Check data quality:**

   ```bash
   # Verify preprocessed data exists
   ls -lh ./training_data/

   # Check for adequate samples
   ```

3. **Train longer:**

   ```bash
   --epochs 20  # More epochs
   ```

4. **Verify hidden states:**

   ```bash
   # Check hidden states are being loaded
   # Look for "Loaded hidden states" in logs
   ```

### CUDA Out of Memory During Training

**Error:**

```
torch.cuda.OutOfMemoryError
```

**Solutions:**

1. **Reduce sequence length:**

   ```bash
   --total-seq-len 4096  # Instead of 8192
   ```

2. **Use fewer dataloader workers:**

   ```bash
   --num-workers 4  # Instead of 12
   ```

3. **Use gradient accumulation (if implemented):**

   ```bash
   --gradient-accumulation-steps 4
   ```

4. **Use more GPUs with FSDP:**

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/train.py ...
   ```

### FSDP Training Hangs

**Issue:** Distributed training freezes or doesn't start.

**Solutions:**

1. **Check all GPUs are visible:**

   ```bash
   echo $CUDA_VISIBLE_DEVICES
   nvidia-smi
   ```

2. **Verify `nproc_per_node` matches GPU count:**

   ```bash
   # If you have 4 GPUs:
   torchrun --nproc_per_node 4 ...
   ```

3. **Check for networking issues:**

   ```bash
   # Ensure localhost networking works
   ping localhost
   ```

### Checkpoint Not Saving

**Issue:** No checkpoints appear in save directory.

**Solutions:**

1. **Check save path permissions:**

   ```bash
   ls -la ./checkpoints/
   mkdir -p ./checkpoints
   ```

2. **Verify training is completing epochs:**

   ```bash
   # Look for "Epoch X/Y complete" in logs
   ```

3. **Check disk space:**

   ```bash
   df -h
   ```

## vLLM Serving Issues

### Speculative Decoding Not Enabled

**Issue:** vLLM logs don't mention speculative decoding.

**Solutions:**

1. **Verify speculators_config exists:**

   ```bash
   cat ./checkpoints/best/config.json | grep speculators_config
   ```

2. **Check vocab mappings present:**

   ```bash
   ls -la ./checkpoints/best/*.{pt,npy}
   # Should see d2t.pt and t2d.pt (or .npy)
   ```

3. **Ensure vLLM version supports speculators:**

   ```bash
   pip show vllm | grep Version
   # Should be 0.12.1 or greater
   ```

### Low Acceptance Rates in Production

**Issue:** Average acceptance < 1.5 tokens.

**Causes & Solutions:**

1. **Different data distribution:**

   - Retrain on production-like data

2. **Wrong target layers:**

   ```bash
   # Verify layers match training config
   cat config.json | grep target_layer_ids
   ```

3. **Model not loaded correctly:**

   ```bash
   # Check vLLM logs for loading errors
   tail -f vllm.log | grep ERROR
   ```

### vLLM Crashes on Startup

**Error:**

```
RuntimeError: CUDA error / Segmentation fault
```

**Solutions:**

1. **Check GPU compatibility:**

   ```bash
   nvidia-smi
   # Verify CUDA version
   ```

2. **Update vLLM:**

   ```bash
   pip install --upgrade vllm
   ```

3. **Reduce memory usage:**

   ```bash
   --gpu-memory-utilization 0.8  # Instead of 0.95
   ```

## Installation Issues

### Incompatible Dependencies

**Error:**

```
ERROR: package-A X.Y conflicts with package-B Z.W
```

**Solutions:**

1. **Use fresh virtual environment:**

   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install speculators
   ```

2. **Install specific extras:**

   ```bash
   pip install speculators[datagen]
   ```

3. **Check Python version:**

   ```bash
   python --version
   # Should be 3.10 or higher
   ```

### Missing CUDA

**Error:**

```
RuntimeError: CUDA not available
```

**Solutions:**

1. **Verify CUDA installation:**

   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Install CUDA-enabled PyTorch:**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

## Model Quality Issues

### Poor Inference Quality

**Issue:** Speculator outputs seem incorrect.

**Check:**

1. Speculative decoding is **lossless** - outputs should be identical to base model
2. If outputs differ, speculative decoding may not be enabled

**Verify:**

```bash
# Run same prompt with and without speculator
# Outputs should be identical (with same random seed)
```

### Inconsistent Results

**Issue:** Different results across runs.

**Expected:** Results vary due to sampling unless using greedy decoding or fixed seed.

**For reproducibility:**

```python
response = client.chat.completions.create(
    model="model",
    messages=[...],
    temperature=0,  # Greedy
    seed=42        # Fixed seed
)
```

## Performance Issues

### Training Slower Than Expected

**Solutions:**

1. **Use offline generation:**

   - Pre-generate hidden states for faster training

2. **Increase vLLM GPUs:**

   ```bash
   --data-parallel-size 8
   ```

3. **Use FSDP:**

   ```bash
   torchrun --nproc_per_node 4 scripts/train.py ...
   ```

4. **Reduce dataloader workers if I/O bound:**

   ```bash
   --num-workers 4
   ```

### Inference Slower Than Baseline

**Issue:** Speculator slower than base model alone.

**Causes:**

1. **Batch size too large:**

   - Speculative decoding works best at low batch sizes
   - Try `--max-num-seqs 16`

2. **Acceptance rate too low:**

   - If avg acceptance < 1.2, overhead may exceed benefit
   - Retrain or use fewer speculative tokens

3. **GPU memory bottleneck:**

   - Draft model competing for resources
   - Use tensor parallelism to distribute load

## Debugging Tips

### Enable Verbose Logging

**Training:**

```bash
export SPECULATORS_LOG_LEVEL=DEBUG
python scripts/train.py ...
```

**vLLM:**

```bash
vllm serve model --log-level debug
```

### Check Model Config

```bash
# View complete model configuration
python -c "
import json
with open('./checkpoints/best/config.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

### Validate Hidden States

```python
# Check a hidden states file
from safetensors import safe_open

with safe_open("hs_0.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: {tensor.shape}, {tensor.dtype}")
```

### Test Dataloader

```python
# Verify dataloader works
from speculators.train.data import ArrowDataset

dataset = ArrowDataset(
    datapath="./training_data",
    max_len=8192,
)

sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input IDs shape: {sample['input_ids'].shape}")
```

## Getting Help

If you're still stuck:

1. **Check existing issues:** [GitHub Issues](https://github.com/vllm-project/speculators/issues)
2. **Ask on Slack:** [vLLM Community Slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack)
   - Channels: `#speculators`, `#feat-spec-decode`
3. **File a bug report:** Include error messages, logs, and reproduction steps

## Common Error Messages

| Error                                 | Likely Cause            | Solution                                  |
| ------------------------------------- | ----------------------- | ----------------------------------------- |
| `FileNotFoundError: hs_X.safetensors` | Missing hidden states   | Regenerate or use `--on-missing generate` |
| `CUDA out of memory`                  | GPU memory exceeded     | Reduce batch size / sequence length       |
| `Connection refused`                  | vLLM not running        | Start vLLM server                         |
| `No chat template`                    | Model lacks template    | Use different model                       |
| `Invalid speculators_config`          | Malformed config        | Check config.json syntax                  |
| `Checkpoint mismatch`                 | Wrong checkpoint format | Verify checkpoint is from speculators     |

## See Also

- [Getting Started](getting_started.md) - Initial setup guide
- [Training Feature](features/training.md) - Complete training documentation
- [CLI Reference](../cli/index.md) - All command-line options
- [Community](../community.md) - Get help from the community
